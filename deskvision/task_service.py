"""Persist UI-originated runs through the same history schema as HTTP runs."""

from datetime import datetime
import shutil
from uuid import uuid4
from zoneinfo import ZoneInfo

from . import ui as core
from .config import ROOT
from .database import get_database
from .repository import RunRepository
from . import queueing


TASKS_ROOT = ROOT / 'runs/stage3-api'


def prepare_task(route: str, image_path, parameters: dict, *, owner_id: str) -> dict:
    """Copy an original and commit the queued record before publishing anything."""
    database = get_database()
    database.initialize()
    repository = RunRepository(database.sessions)
    source = core._path_value(image_path)
    content, image_info = core.prepare_image(source)
    run_id = str(uuid4())
    run_dir = TASKS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    managed_image = run_dir / 'input.img'
    persisted = False
    try:
        managed_image.write_bytes(content)
        task = {
            'run_id': run_id, 'owner_id': owner_id, 'route': route, 'status': 'queued',
            'created_at': _now(),
            'input': image_info, 'parameters': parameters,
            'image_path': managed_image, 'run_dir': run_dir,
        }
        repository.create(task, owner_id=owner_id)
        persisted = True
        return repository.get(run_id, owner_id=owner_id)
    except Exception:
        if persisted:
            repository.fail(run_id, {'code': 'internal_error',
                                     'message': '任务未能完成，请检查本机运行日志。'}, _now())
        else:
            shutil.rmtree(run_dir, ignore_errors=True)
        raise


def submit_task(route: str, image_path, parameters: dict, *, owner_id: str) -> dict:
    if not queueing.is_celery():
        raise RuntimeError('submit_task 需要 Celery 模式。')
    task = prepare_task(route, image_path, parameters, owner_id=owner_id)
    try:
        queueing.enqueue(task['run_id'])
    except Exception:
        from .executor import repository
        repository().delivery_uncertain(task['run_id'])
        # The caller can poll this saved task even if broker acknowledgement was lost.
        return repository().get(task['run_id'], owner_id=owner_id)
    return task


def execute_and_record(route: str, image_path, parameters: dict, *, owner_id: str) -> dict:
    """Compatibility helper for explicitly selected inline development mode."""
    from .executor import execute_run, repository
    if queueing.is_celery():
        raise RuntimeError('Celery 模式必须使用 submit_task，再按任务号轮询。')
    task = prepare_task(route, image_path, parameters, owner_id=owner_id)
    execute_run(task['run_id'])
    return repository().get(task['run_id'], owner_id=owner_id)


def _now() -> str:
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
