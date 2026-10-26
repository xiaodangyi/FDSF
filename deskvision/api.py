"""Stage-three S2 HTTP API. Task state is process-local until S3."""

from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Literal
from urllib.parse import quote
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field
from starlette.middleware import Middleware

from . import ui as core
from .artifacts import artifact_names as _artifact_names
from . import auth
from . import queueing
from .config import ROOT
from .database import get_database
from .repository import RunRepository
from .service import run_routes
from .task_service import TASKS_ROOT
from .vision_language import prepare_image
from .web_guard import WorkbenchGuard


MAX_UPLOAD_BYTES = 32 * 1024 * 1024
RUNS_ROOT = TASKS_ROOT
logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app):
    get_database().initialize()
    auth.settings()
    queueing.validate_configuration()
    if queueing.is_celery():
        _repository().recover_expired(_now())
    else:
        _repository().recover_incomplete(_now())
    yield


app = FastAPI(title='DeskVision API', version='1.0.0', lifespan=_lifespan,
              middleware=[Middleware(WorkbenchGuard)])


def _repository() -> RunRepository:
    database = get_database()
    database.initialize()
    return RunRepository(database.sessions)


class NoteUpdate(BaseModel):
    note: str = Field(max_length=1000)


class LoginCredentials(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=1, max_length=128)


@app.get('/login', response_class=HTMLResponse, include_in_schema=False)
def login_page():
    from .auth_ui import LOGIN_PAGE

    return LOGIN_PAGE


@app.post('/api/auth/login')
def create_login(payload: LoginCredentials):
    token, identity = auth.login(payload.username, payload.password)
    result = JSONResponse({'username': identity.username, 'expires_in': auth.TOKEN_SECONDS})
    _, local_http = auth.settings()
    result.set_cookie(auth.COOKIE, token, max_age=auth.TOKEN_SECONDS, httponly=True,
                      secure=not local_http, samesite='strict', path='/')
    return result


@app.get('/api/auth/me')
def whoami(identity: auth.Identity = Depends(auth.current_user)) -> dict:
    return {'username': identity.username}


@app.post('/api/auth/logout')
def logout(identity: auth.Identity = Depends(auth.current_user)) -> JSONResponse:
    auth.revoke(identity)
    response = JSONResponse({'status': 'ok'})
    response.delete_cookie(auth.COOKIE, path='/', httponly=True,
                           secure=not auth.settings()[1], samesite='strict')
    return response


def _now() -> str:
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def _snapshot(task: dict) -> dict:
    """Return public task data without server paths or uploaded filenames."""
    results = {}
    for route, report in task.get('results', {}).items():
        public = {key: report[key] for key in (
            'status', 'route', 'input', 'model', 'summary', 'items', 'message',
            'timings_ms', 'segmentation', 'usage', 'error', 'artifacts'
        ) if key in report}
        result_dir = report.get('result_dir')
        artifact_names = _artifact_names(report)
        if 'artifacts' in public:
            public['artifacts'] = {key: name for key, name in public['artifacts'].items()
                                   if name in artifact_names}
        if result_dir:
            public['artifact_urls'] = {
                name: f"/api/v1/runs/{task['run_id']}/artifacts/{route}/{quote(name, safe='/')}"
                for name in sorted(artifact_names)
            }
        results[route] = public
    return {
        'run_id': task['run_id'], 'route': task['route'], 'status': task['status'],
        'created_at': task['created_at'], 'started_at': task.get('started_at'),
        'finished_at': task.get('finished_at'),
        'input': {key: task['input'][key] for key in
                  ('sha256', 'mime_type', 'bytes', 'width', 'height')},
        'parameters': task['parameters'], 'results': results,
        'route_states': task.get('route_states', {
            route: 'queued' for route in (('yolo', 'deepseek') if task['route'] == 'both' else (task['route'],))}),
        'comparison': task.get('comparison'), 'error': task.get('error'),
        'note': task.get('note', ''),
    }


def _mark_running(run_id: str) -> None:
    _repository().mark_running(run_id, _now())


def _run_routes(task: dict, image_path: Path, output_root: Path) -> dict:
    return run_routes(task['route'], image_path, task['parameters'], output_root=output_root,
                      expected_sha256=task['input']['sha256'],
                      on_start=lambda: _mark_running(task['run_id']))


def _execute_run(run_id: str) -> None:
    from .executor import execute_run

    execute_run(run_id)


@app.get('/health')
def health() -> dict:
    """Health check does not import or load model frameworks."""
    return {'status': 'ok'}


@app.get('/', include_in_schema=False)
def home(request: Request):
    try:
        auth.current_user(request)
    except HTTPException:
        return RedirectResponse('/login', status_code=303)
    return RedirectResponse('/ui/', status_code=303)


@app.post('/api/v1/runs', status_code=202)
async def create_run(
    background_tasks: BackgroundTasks,
    identity: auth.Identity = Depends(auth.current_user),
    image: UploadFile = File(...),
    route: Literal['yolo', 'deepseek', 'both'] = Form('yolo'),
    allow_remote: bool = Form(False),
    mode: Literal['finetuned', 'baseline'] = Form('finetuned'),
    confidence: float = Form(0.25),
    iou: float = Form(0.7),
    image_size: int = Form(640),
    device: Literal['auto', 'cpu', 'cuda:0'] = Form('auto'),
    segment: bool = Form(True),
) -> dict:
    if route in ('deepseek', 'both') and not allow_remote:
        raise HTTPException(status_code=422, detail='DeepSeek 会发送原图并产生 API 用量，请显式设置 allow_remote=true。')
    if not 0 < confidence < 1 or not 0 < iou < 1:
        raise HTTPException(status_code=422, detail='confidence 和 iou 必须在 0 与 1 之间。')
    if not 32 <= image_size <= 2048 or image_size % 32:
        raise HTTPException(status_code=422, detail='image_size 必须是 32～2048 之间的 32 倍数。')
    content = await image.read(MAX_UPLOAD_BYTES + 1)
    await image.close()
    if not content:
        raise HTTPException(status_code=422, detail='上传图片不能为空。')
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail='图片不能超过 32 MiB。')

    run_id = str(uuid4())
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    image_path = run_dir / 'input.img'
    try:
        image_path.write_bytes(content)
        _, image_info = prepare_image(image_path)
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        message = '上传文件不是受支持的有效图片，请检查格式、帧数和尺寸。'
        raise HTTPException(status_code=422, detail=message) from None

    task = {
        'run_id': run_id, 'route': route, 'status': 'queued', 'created_at': _now(),
        'image_path': image_path, 'run_dir': run_dir,
        'input': image_info,
        'parameters': {'mode': mode, 'confidence': confidence, 'iou': iou,
                       'image_size': image_size, 'device': device, 'segment': segment},
        'results': {},
    }
    try:
        _repository().create(task, owner_id=identity.user_id)
    except Exception:
        logger.exception('Could not create run %s in SQLite', run_id)
        shutil.rmtree(run_dir, ignore_errors=True)
        raise HTTPException(status_code=503, detail='任务记录暂时无法保存，请检查本地数据库。') from None
    if queueing.is_celery():
        # Broker IO must not block FastAPI's event loop.
        from starlette.concurrency import run_in_threadpool
        try:
            await run_in_threadpool(queueing.enqueue, run_id)
        except Exception:
            logger.exception('Queue delivery not confirmed for %s', run_id)
            _repository().delivery_uncertain(run_id)
            raise HTTPException(503, detail={
                'run_id': run_id, 'code': 'delivery_uncertain',
                'message': '任务已保存，队列投递未确认。请按任务号查询，不要重复提交。',
            }) from None
    else:
        background_tasks.add_task(_execute_run, run_id)
    return _snapshot(task)


@app.get('/api/v1/runs')
def list_runs(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Literal['queued', 'running', 'succeeded', 'partial_failed', 'failed'] | None = None,
    identity: auth.Identity = Depends(auth.current_user),
) -> dict:
    if queueing.is_celery():
        _repository().recover_expired(_now())
    result = _repository().list(limit=limit, offset=offset, status=status,
                                owner_id=identity.user_id)
    result['items'] = [_snapshot(item) for item in result['items']]
    return result


@app.get('/api/v1/runs/{run_id}')
def get_run(run_id: str, identity: auth.Identity = Depends(auth.current_user)) -> dict:
    if queueing.is_celery():
        _repository().recover_expired(_now())
    try:
        normalized_id = str(UUID(run_id))
    except ValueError:
        raise HTTPException(status_code=404, detail='任务不存在。') from None
    task = _repository().get(normalized_id, owner_id=identity.user_id)
    if task is None:
        raise HTTPException(status_code=404, detail='任务不存在。')
    return _snapshot(task)


@app.patch('/api/v1/runs/{run_id}/note')
def update_run_note(run_id: str, payload: NoteUpdate,
                    identity: auth.Identity = Depends(auth.current_user)) -> dict:
    try:
        normalized_id = str(UUID(run_id))
    except ValueError:
        raise HTTPException(status_code=404, detail='任务不存在。') from None
    repository = _repository()
    if not repository.update_note(normalized_id, payload.note, owner_id=identity.user_id):
        raise HTTPException(status_code=404, detail='任务不存在。')
    return {'run_id': normalized_id, 'note': payload.note}


@app.delete('/api/v1/runs/{run_id}', status_code=204)
def delete_run(run_id: str, identity: auth.Identity = Depends(auth.current_user)) -> None:
    try:
        normalized_id = str(UUID(run_id))
    except ValueError:
        raise HTTPException(status_code=404, detail='任务不存在。') from None
    result = _repository().delete(normalized_id, owner_id=identity.user_id)
    if result is None:
        raise HTTPException(status_code=404, detail='任务不存在。')
    if result.get('running'):
        raise HTTPException(status_code=409, detail='排队中或运行中的任务不能删除。')
    run_dir = result['run_dir']
    if run_dir.exists():
        shutil.rmtree(run_dir)
    return None


@app.get('/api/v1/runs/{run_id}/artifacts/{route}/{artifact_path:path}')
def get_artifact(run_id: str, route: Literal['yolo', 'deepseek'], artifact_path: str,
                 identity: auth.Identity = Depends(auth.current_user)):
    try:
        normalized_id = str(UUID(run_id))
    except ValueError:
        raise HTTPException(status_code=404, detail='文件不存在。') from None
    task = _repository().get(normalized_id, owner_id=identity.user_id)
    report = task.get('results', {}).get(route) if task else None
    result_dir = report.get('result_dir') if report else None
    run_dir = task.get('run_dir') if task else None
    allowed_names = _artifact_names(report) if report else set()
    if not result_dir or not run_dir or artifact_path not in allowed_names:
        raise HTTPException(status_code=404, detail='文件不存在。')
    base = Path(result_dir).resolve()
    allowed_root = (Path(run_dir) / 'results').resolve()
    candidate = base / artifact_path
    target = candidate.resolve()
    if (not base.is_relative_to(allowed_root) or not target.is_relative_to(base)
            or not target.is_file() or any(path.is_symlink() for path in
                                          (candidate, *candidate.parents) if path != path.parent)):
        raise HTTPException(status_code=404, detail='文件不存在。')
    return FileResponse(target, filename=target.name)


@app.get('/api/v1/runs/{run_id}/export.zip')
def export_run(run_id: str, identity: auth.Identity = Depends(auth.current_user)):
    try:
        normalized_id = str(UUID(run_id))
    except ValueError:
        raise HTTPException(status_code=404, detail='任务不存在。') from None
    task = _repository().get(normalized_id, owner_id=identity.user_id)
    if task is None:
        raise HTTPException(status_code=404, detail='任务不存在。')
    buffer = BytesIO()
    with ZipFile(buffer, 'w', ZIP_DEFLATED) as archive:
        archive.writestr('report.json', json.dumps(_snapshot(task), ensure_ascii=False, indent=2))
        for route, report in task['results'].items():
            base = report.get('result_dir')
            if not base:
                continue
            folder = Path(base).resolve()
            allowed = (task['run_dir'] / 'results').resolve()
            if not folder.is_relative_to(allowed) or not folder.is_dir():
                continue
            for name in sorted(_artifact_names(report)):
                candidate = folder / name
                target = candidate.resolve()
                if (target.is_file() and target.is_relative_to(folder)
                        and not any(part.is_symlink() for part in (candidate, *candidate.parents)
                                    if part != part.parent)):
                    archive.write(target, f'{route}/{name}')
        archive.writestr('数量对比.md', task.get('comparison') or '本任务没有双路数量对比。')
    return Response(buffer.getvalue(), media_type='application/zip',
                    headers={'Content-Disposition': f'attachment; filename="deskvision-{normalized_id}.zip"'})


def create_app() -> FastAPI:
    """Mount the existing workbench on the API process, once and without inference."""
    import gradio as gr
    from .web.app import build_demo

    if any(getattr(route, 'path', None) == '/ui' for route in app.routes):
        return app
    return gr.mount_gradio_app(
        app, build_demo(), path='/ui', blocked_paths=[str(ROOT / '.local')],
        max_file_size='32mb', show_error=False,
        auth_dependency=auth.gradio_user,
    )


def main() -> None:
    import uvicorn

    os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
    os.environ.setdefault('GRADIO_TEMP_DIR', str(ROOT / 'runs/web-cache'))
    uvicorn.run(create_app(), host='127.0.0.1', port=8000, reload=False)


if __name__ == '__main__':
    main()
