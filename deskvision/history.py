"""History operations shared by the HTTP endpoints and Gradio workbench."""

import shutil
from uuid import UUID

from .database import get_database
from .repository import RunRepository


def _repository() -> RunRepository:
    database = get_database()
    database.initialize()
    return RunRepository(database.sessions)


def history_rows(limit: int = 100) -> tuple[list[list[str]], list[str]]:
    page = _repository().list(limit=limit, offset=0)
    rows = []
    for item in page['items']:
        route_summaries = []
        for route in ('yolo', 'deepseek'):
            report = item['results'].get(route)
            if not report or report.get('status') != 'ok':
                continue
            counts = [entry.get('count') for entry in report.get('summary', [])]
            known_counts = [count for count in counts if type(count) is int]
            if known_counts:
                route_summaries.append(
                    f"{'本地' if route == 'yolo' else '大模型'} {sum(known_counts)}")
        rows.append([item['run_id'], item['route'], item['status'], item['created_at'],
                     ' / '.join(route_summaries) or '—', item['note']])
    return rows, [row[0] for row in rows]


def history_detail(run_id: str) -> tuple[dict, str]:
    try:
        normalized = str(UUID(run_id))
    except ValueError:
        raise ValueError('任务编号无效。') from None
    item = _repository().get(normalized)
    if item is None:
        raise ValueError('任务不存在。')
    from .api import _snapshot

    return _snapshot(item), item['note']


def update_history_note(run_id: str, note: str) -> str:
    if len(note) > 1000:
        raise ValueError('备注不能超过 1000 个字符。')
    try:
        normalized = str(UUID(run_id))
    except ValueError:
        raise ValueError('任务编号无效。') from None
    if not _repository().update_note(normalized, note):
        raise ValueError('任务不存在。')
    return '备注已保存。'


def delete_history(run_id: str) -> str:
    try:
        normalized = str(UUID(run_id))
    except ValueError:
        raise ValueError('任务编号无效。') from None
    deleted = _repository().delete(normalized)
    if deleted is None:
        raise ValueError('任务不存在。')
    if deleted.get('running'):
        raise ValueError('排队中或运行中的任务不能删除。')
    if deleted['run_dir'].exists():
        shutil.rmtree(deleted['run_dir'])
    return '历史记录和本次运行文件已删除。'
