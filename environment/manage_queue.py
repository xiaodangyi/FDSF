"""Explicit recovery/redelivery CLI. Never requeues a task that has started."""

import argparse
from pathlib import Path
import sys
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deskvision.executor import recover_expired, repository
from deskvision.queueing import enqueue, is_celery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('recover-expired')
    redeliver = sub.add_parser('redeliver')
    redeliver.add_argument('run_id')
    args = parser.parse_args()
    if args.command == 'recover-expired':
        print(f'已标记 {recover_expired()} 个过期任务；没有重跑模型。')
        return
    run_id = str(UUID(args.run_id))
    task = repository().get(run_id)
    if not task or task['status'] != 'queued':
        parser.error('只允许补投尚未开始的 queued 任务；已开始或结束的任务不能重投。')
    if not is_celery():
        parser.error('补投需要 Celery 配置。')
    enqueue(run_id)
    print(f'已补投原任务号 {run_id}；原子执行锁会过滤重复消息。')


if __name__ == '__main__':
    main()
