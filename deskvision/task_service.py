"""Persist UI-originated runs through the same history schema as HTTP runs."""

from datetime import datetime
import shutil
from uuid import uuid4
from zoneinfo import ZoneInfo

from . import ui as core
from .config import ROOT
from .database import get_database
from .repository import RunRepository
from .service import run_routes


TASKS_ROOT = ROOT / 'runs/stage3-api'


def execute_and_record(route: str, image_path, parameters: dict) -> dict:
    """Copy an original into managed storage, execute selected routes, and persist results."""
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
            'run_id': run_id, 'route': route, 'status': 'queued',
            'created_at': _now(),
            'input': image_info, 'parameters': parameters,
            'image_path': managed_image, 'run_dir': run_dir,
        }
        repository.create(task)
        persisted = True
        results = run_routes(
            route, managed_image, parameters, output_root=run_dir / 'results',
            expected_sha256=image_info['sha256'],
            on_start=lambda: repository.mark_running(task['run_id'], _now()),
        )
        statuses = [report.get('status') for report in results.values()]
        successful = statuses.count('ok')
        status = ('succeeded' if successful == len(statuses) else
                  'partial_failed' if successful else 'failed')
        comparison = (core.format_count_diff(results.get('yolo'), results.get('deepseek'))
                      if len(statuses) > 1 else None)
        repository.finish(run_id, status=status, results=results, comparison=comparison,
                          finished_at=_now())
        return repository.get(run_id)
    except Exception:
        if persisted:
            repository.fail(run_id, {'code': 'internal_error',
                                     'message': '任务未能完成，请检查本机运行日志。'}, _now())
        else:
            shutil.rmtree(run_dir, ignore_errors=True)
        raise


def _now() -> str:
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
