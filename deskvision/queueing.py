"""Queue boundary for S5: durable task rows first, broker delivery second."""

import os


def mode() -> str:
    value = os.environ.get('DESKVISION_QUEUE_MODE', 'celery').strip().lower()
    if value not in ('inline', 'celery'):
        raise RuntimeError('DESKVISION_QUEUE_MODE 只允许 celery 或 inline。')
    return value


def is_celery() -> bool:
    return mode() == 'celery'


def enqueue(run_id: str) -> str | None:
    """Publish a run id to Celery; return its broker task id.

    Inline mode is intentionally explicit for local regression tests. Production
    must set DESKVISION_QUEUE_MODE=celery and DESKVISION_REDIS_URL.
    """
    if not is_celery():
        return None
    with make_app() as app:
        result = app.send_task('deskvision.run', args=[run_id],
                               task_id=f'deskvision-{run_id}', retry=False)
        return result.id


def validate_configuration() -> None:
    if is_celery():
        broker = os.environ.get('DESKVISION_REDIS_URL', '')
        if not broker.startswith(('redis://', 'rediss://')):
            raise RuntimeError('Celery 模式需要 redis:// 或 rediss:// 格式的 DESKVISION_REDIS_URL。')


def make_app():
    """Construct a producer without importing the worker, web UI or model code."""
    from celery import Celery

    validate_configuration()
    broker = os.environ['DESKVISION_REDIS_URL']
    if not broker.startswith(('redis://', 'rediss://')):
        raise RuntimeError('DESKVISION_REDIS_URL 必须使用 redis:// 或 rediss://。')
    app = Celery('deskvision', broker=broker)
    app.conf.update(
        task_default_queue=os.environ.get('DESKVISION_QUEUE_NAME', 'deskvision'),
        task_acks_late=True, task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1, task_ignore_result=True,
        task_serializer='json', accept_content=['json'], result_serializer='json',
        task_time_limit=1800, task_soft_time_limit=1500,
        broker_transport_options={'visibility_timeout': 3600, 'socket_timeout': 3,
                                  'socket_connect_timeout': 3, 'retry_on_timeout': False},
        broker_connection_timeout=3, broker_connection_retry_on_startup=True,
        task_publish_retry=False,
    )
    return app
