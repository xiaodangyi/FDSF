"""Local Linux launcher; open separate terminals for redis, worker and web."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('service', choices=['redis', 'worker', 'web'])
    parser.add_argument('--port', type=int, default=8001, help='Web port only')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('S5 正式队列请在 Ubuntu 终端运行，worker 使用 Linux prefork。')
    os.chdir(ROOT)
    os.environ.setdefault('DESKVISION_QUEUE_MODE', 'celery')
    os.environ.setdefault('DESKVISION_REDIS_URL', 'redis://127.0.0.1:6380/0')
    os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
    from deskvision.queueing import is_celery, validate_configuration
    validate_configuration()
    if not is_celery():
        parser.error('S5 启动器需要 DESKVISION_QUEUE_MODE=celery；请清除旧 inline 环境变量。')
    if args.service == 'redis':
        folder = ROOT / '.local/redis'
        folder.mkdir(parents=True, exist_ok=True)
        os.execvp('redis-server', ['redis-server', '--bind', '127.0.0.1', '--port', '6380',
                                 '--protected-mode', 'yes', '--appendonly', 'yes',
                                 '--appendfsync', 'everysec', '--maxmemory-policy', 'noeviction',
                                 '--dir', str(folder)])
    if args.service == 'worker':
        os.execv(sys.executable, [sys.executable, '-m', 'deskvision.worker'])
    os.execv(sys.executable, [sys.executable, '-m', 'deskvision.web', '--port', str(args.port)])


if __name__ == '__main__':
    main()
