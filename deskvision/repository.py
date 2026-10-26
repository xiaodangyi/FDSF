"""Transactional task history operations backed by SQLAlchemy."""

from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, selectinload

from .artifacts import artifact_names
from .config import ROOT
from .models import ArtifactRecord, RouteResultRecord, RunRecord


class RunRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def create(self, task: dict) -> None:
        image = task['input']
        with self.session_factory.begin() as session:
            session.add(RunRecord(
                run_id=task['run_id'], route=task['route'], status=task['status'],
                created_at=task['created_at'], input_sha256=image['sha256'],
                mime_type=image['mime_type'], input_bytes=image['bytes'],
                input_width=image['width'], input_height=image['height'],
                parameters=task['parameters'], input_path=_relative(task['image_path']),
                run_dir=_relative(task['run_dir']),
            ))

    def mark_running(self, run_id: str, started_at: str) -> None:
        with self.session_factory.begin() as session:
            record = session.get(RunRecord, run_id)
            if record is not None:
                record.status = 'running'
                record.started_at = started_at

    def finish(self, run_id: str, *, status: str, results: dict,
               comparison: str | None, finished_at: str, error: dict | None = None) -> None:
        with self.session_factory.begin() as session:
            record = session.get(RunRecord, run_id)
            if record is None:
                return
            record.status = status
            record.comparison = comparison
            record.finished_at = finished_at
            record.error = error
            for route, report in results.items():
                result_dir = report.get('result_dir')
                safe_report = {key: value for key, value in report.items()
                               if key not in ('result_dir', 'visualization_path')}
                route_result = session.scalar(select(RouteResultRecord).where(
                    RouteResultRecord.run_id == run_id,
                    RouteResultRecord.route == route))
                if route_result is None:
                    route_result = RouteResultRecord(run_id=run_id, route=route,
                                                     status=report.get('status', 'failed'),
                                                     report=safe_report,
                                                     result_dir=_relative(result_dir) if result_dir else None)
                    session.add(route_result)
                    session.flush()
                else:
                    route_result.status = report.get('status', 'failed')
                    route_result.report = safe_report
                    route_result.result_dir = _relative(result_dir) if result_dir else None
                    session.execute(delete(ArtifactRecord).where(
                        ArtifactRecord.route_result_id == route_result.id))
                for name in sorted(artifact_names(report)):
                    session.add(ArtifactRecord(route_result_id=route_result.id, name=name))

    def fail(self, run_id: str, error: dict, finished_at: str) -> None:
        self.finish(run_id, status='failed', results={}, comparison=None,
                    finished_at=finished_at, error=error)

    def recover_incomplete(self, finished_at: str) -> int:
        """Mark pre-restart queue/running tasks failed; never repeat model calls."""
        with self.session_factory.begin() as session:
            records = session.scalars(select(RunRecord).where(
                RunRecord.status.in_(('queued', 'running')))).all()
            for record in records:
                record.status = 'failed'
                record.finished_at = finished_at
                record.error = {
                    'code': 'interrupted_by_restart',
                    'message': '服务在任务完成前重启；为避免重复模型调用，本任务未自动重试。',
                }
            return len(records)

    def get(self, run_id: str) -> dict | None:
        with self.session_factory() as session:
            record = session.scalar(select(RunRecord).options(
                selectinload(RunRecord.route_results).selectinload(RouteResultRecord.artifacts)
            ).where(RunRecord.run_id == run_id))
            return _serialize(record) if record else None

    def list(self, *, limit: int, offset: int, status: str | None = None) -> dict:
        with self.session_factory() as session:
            query = select(RunRecord).order_by(RunRecord.created_at.desc(), RunRecord.run_id.desc())
            count = select(func.count()).select_from(RunRecord)
            if status:
                query = query.where(RunRecord.status == status)
                count = count.where(RunRecord.status == status)
            total = session.scalar(count) or 0
            records = session.scalars(query.offset(offset).limit(limit)).all()
            return {'total': total, 'limit': limit, 'offset': offset,
                    'items': [_serialize(record) for record in records]}

    def update_note(self, run_id: str, note: str) -> bool:
        with self.session_factory.begin() as session:
            record = session.get(RunRecord, run_id)
            if record is None:
                return False
            record.note = note
            return True

    def delete(self, run_id: str) -> dict | None:
        with self.session_factory.begin() as session:
            record = session.scalar(select(RunRecord).where(
                RunRecord.run_id == run_id).with_for_update())
            if record is None:
                return None
            if record.status in ('queued', 'running'):
                return {'running': True}
            run_dir = _absolute(record.run_dir)
            session.delete(record)
        return {'run_dir': run_dir}


def _relative(path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def _absolute(path: str) -> Path:
    candidate = (ROOT / path).resolve()
    if not candidate.is_relative_to((ROOT / 'runs/stage3-api').resolve()):
        raise ValueError('Stored task path escapes the task data directory.')
    return candidate


def _serialize(record: RunRecord) -> dict:
    report = {
        'run_id': record.run_id, 'route': record.route, 'status': record.status,
        'created_at': record.created_at, 'started_at': record.started_at,
        'finished_at': record.finished_at,
        'input': {'sha256': record.input_sha256, 'mime_type': record.mime_type,
                  'bytes': record.input_bytes, 'width': record.input_width,
                  'height': record.input_height},
        'parameters': record.parameters, 'results': {}, 'comparison': record.comparison,
        'error': record.error, 'note': record.note,
        'image_path': _absolute(record.input_path), 'run_dir': _absolute(record.run_dir),
    }
    for result in record.route_results:
        item = dict(result.report)
        item['status'] = result.status
        item['result_dir'] = _absolute(result.result_dir) if result.result_dir else None
        item['artifacts'] = {artifact.name: artifact.name for artifact in result.artifacts}
        report['results'][result.route] = item
    return report
