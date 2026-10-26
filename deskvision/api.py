"""Stage-three S2 HTTP API. Task state is process-local until S3."""

from copy import deepcopy
from datetime import datetime
import logging
import os
from pathlib import Path
import shutil
from threading import Lock
from typing import Literal
from urllib.parse import quote
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse

from . import ui as core
from .config import ROOT
from .service import run_routes
from .vision_language import prepare_image


MAX_UPLOAD_BYTES = 32 * 1024 * 1024
RUNS_ROOT = ROOT / 'runs/stage3-api'
_tasks: dict[str, dict] = {}
_tasks_lock = Lock()
logger = logging.getLogger(__name__)

app = FastAPI(title='DeskVision API', version='1.0.0')


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
        'comparison': task.get('comparison'), 'error': task.get('error'),
    }


def _mark_running(run_id: str) -> None:
    with _tasks_lock:
        _tasks[run_id].update(status='running', started_at=_now())


def _run_routes(task: dict, image_path: Path, output_root: Path) -> dict:
    return run_routes(task['route'], image_path, task['parameters'], output_root=output_root,
                      expected_sha256=task['input']['sha256'],
                      on_start=lambda: _mark_running(task['run_id']))


def _execute_run(run_id: str) -> None:
    with _tasks_lock:
        task = _tasks.get(run_id)
        if task is None:
            return
        task['status'] = 'queued'
    try:
        with _tasks_lock:
            task = _tasks[run_id]
            image_path = Path(task['image_path'])
            output_root = Path(task['run_dir']) / 'results'
        results = _run_routes(task, image_path, output_root)
        statuses = [report.get('status') for report in results.values()]
        comparison = (core.format_count_diff(results.get('yolo'), results.get('deepseek'))
                      if len(statuses) > 1 else None)
        successful = statuses.count('ok')
        status = ('succeeded' if successful == len(statuses) else
                  'partial_failed' if successful else 'failed')
        with _tasks_lock:
            task = _tasks[run_id]
            task.update(status=status, results=results, comparison=comparison,
                        finished_at=_now())
    except Exception:
        logger.exception('Task %s failed', run_id)
        with _tasks_lock:
            task = _tasks.get(run_id)
            if task is not None:
                task.update(status='failed', error={
                    'code': 'internal_error', 'message': '任务未能完成，请检查服务端运行日志。',
                }, finished_at=_now())


@app.get('/health')
def health() -> dict:
    """Health check does not import or load model frameworks."""
    return {'status': 'ok'}


@app.get('/', include_in_schema=False)
def home():
    return RedirectResponse('/ui/')


@app.post('/api/v1/runs', status_code=202)
async def create_run(
    background_tasks: BackgroundTasks,
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
        'image_path': str(image_path), 'run_dir': str(run_dir),
        'input': image_info,
        'parameters': {'mode': mode, 'confidence': confidence, 'iou': iou,
                       'image_size': image_size, 'device': device, 'segment': segment},
        'results': {},
    }
    with _tasks_lock:
        _tasks[run_id] = task
    background_tasks.add_task(_execute_run, run_id)
    with _tasks_lock:
        return _snapshot(deepcopy(task))


@app.get('/api/v1/runs/{run_id}')
def get_run(run_id: str) -> dict:
    try:
        normalized_id = str(UUID(run_id))
    except ValueError:
        raise HTTPException(status_code=404, detail='任务不存在。') from None
    with _tasks_lock:
        task = _tasks.get(normalized_id)
        if task is None:
            raise HTTPException(status_code=404, detail='任务不存在。')
        return _snapshot(deepcopy(task))


@app.get('/api/v1/runs/{run_id}/artifacts/{route}/{artifact_path:path}')
def get_artifact(run_id: str, route: Literal['yolo', 'deepseek'], artifact_path: str):
    try:
        normalized_id = str(UUID(run_id))
    except ValueError:
        raise HTTPException(status_code=404, detail='文件不存在。') from None
    with _tasks_lock:
        task = _tasks.get(normalized_id)
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


def create_app() -> FastAPI:
    """Mount the existing workbench on the API process, once and without inference."""
    import gradio as gr
    from .web.app import build_demo

    if any(getattr(route, 'path', None) == '/ui' for route in app.routes):
        return app
    return gr.mount_gradio_app(
        app, build_demo(), path='/ui', allowed_paths=[str(core.UI_OUTPUT_ROOT)],
        blocked_paths=[str(ROOT / '.local')], max_file_size='32mb', show_error=False,
    )


def main() -> None:
    import uvicorn

    os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
    os.environ.setdefault('GRADIO_TEMP_DIR', str(ROOT / 'runs/web-cache'))
    uvicorn.run(create_app(), host='127.0.0.1', port=8000, reload=False)


def _artifact_names(report: dict) -> set[str]:
    names = set()
    if isinstance(report.get('artifacts'), dict):
        names.update(value for value in report['artifacts'].values() if isinstance(value, str))
    for instance in report.get('segmentation', {}).get('instances', []):
        names.update(instance.get(key) for key in ('mask_file', 'contours_file')
                     if isinstance(instance.get(key), str))
    return {name for name in names if name != 'result.json' and '\\' not in name
            and not Path(name).is_absolute() and ':' not in name
            and all(part not in ('', '.', '..') for part in name.split('/'))}


if __name__ == '__main__':
    main()
