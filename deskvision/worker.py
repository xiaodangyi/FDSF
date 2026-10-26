"""Linux worker: python -m deskvision.worker (prefork, concurrency one)."""

from .queueing import make_app


celery_app = make_app()


@celery_app.task(name='deskvision.run', max_retries=0)
def run_task(run_id: str):
    from .executor import execute_run

    execute_run(run_id)


def main():
    celery_app.worker_main(['worker', '--loglevel=INFO', '--concurrency=1', '--pool=prefork'])


if __name__ == '__main__':
    main()
