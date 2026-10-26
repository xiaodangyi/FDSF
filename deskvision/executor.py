"""Durable execution shared by independent workers and explicit inline tests."""

from datetime import datetime
import logging
from threading import Event, Thread
from uuid import uuid4
from zoneinfo import ZoneInfo

from .database import get_database
from .repository import RunRepository
from .service import run_routes


logger = logging.getLogger(__name__)


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def repository():
    database = get_database()
    database.initialize()
    return RunRepository(database.sessions)


def recover_expired():
    return repository().recover_expired(now())


def execute_run(run_id: str):
    from . import ui as core

    repo = repository()
    recover_expired()
    execution_id = str(uuid4())
    if not repo.claim(run_id, now(), execution_id):
        return
    stopped = Event()

    def heartbeat():
        while not stopped.wait(5):
            try:
                if not repo.heartbeat(run_id, execution_id):
                    return
            except Exception:
                logger.exception('Task heartbeat unavailable for %s', run_id)

    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        task = repo.get(run_id)
        results = run_routes(
            task['route'], task['image_path'], task['parameters'],
            output_root=task['run_dir'] / 'results', expected_sha256=task['input']['sha256'],
            on_route_start=lambda route: repo.route_started(run_id, execution_id, route),
            on_route_finished=lambda route, report: repo.finish(
                run_id, status='running', results={route: report}, comparison=None,
                finished_at=now(), execution_id=execution_id, route_only=True),
        )
        successful = sum(report.get('status') == 'ok' for report in results.values())
        status = ('succeeded' if successful == len(results) else
                  'partial_failed' if successful else 'failed')
        comparison = (core.format_count_diff(results.get('yolo'), results.get('deepseek'))
                      if len(results) > 1 else None)
        repo.finish(run_id, status=status, results=results, comparison=comparison,
                    finished_at=now(), execution_id=execution_id)
    except BaseException:
        logger.exception('Execution interrupted for %s', run_id)
        stopped.set()
        with repo.session_factory.begin() as session:
            from sqlalchemy import update
            from .models import RunRecord
            session.execute(update(RunRecord).where(
                RunRecord.run_id == run_id, RunRecord.status == 'running',
                RunRecord.execution_id == execution_id).values(heartbeat_at=0))
        repo.recover_expired(now())
        raise
    finally:
        stopped.set()
        thread.join(timeout=6)
