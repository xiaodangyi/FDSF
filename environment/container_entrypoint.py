"""Container-only entrypoints. Host launchers retain their loopback binding."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def gpu_info():
    import torch
    info = {'torch': torch.__version__, 'cuda_build': torch.version.cuda,
            'available': torch.cuda.is_available()}
    if info['available']:
        value = torch.ones((8, 8), device='cuda:0')
        info.update(device=torch.cuda.get_device_name(0),
                    arithmetic_ok=(value @ value)[0, 0].item() == 8)
    print(json.dumps(info))
    return info


def bootstrap():
    from sqlalchemy import select
    from deskvision import auth
    from deskvision.models import UserRecord

    accounts = json.loads(Path('/run/secrets/accounts').read_text(encoding='utf-8'))
    for account in accounts:
        with auth.database().sessions() as session:
            exists = session.scalar(select(UserRecord.id).where(
                UserRecord.username == account['username']))
        if not exists:
            auth.create_user(account['username'], account['password'])
    print(json.dumps({'status': 'ok', 'accounts_checked': len(accounts)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['web', 'worker', 'bootstrap', 'health-web',
                                          'health-worker', 'gpu-info'])
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == 'health-web':
        import httpx
        httpx.get('http://127.0.0.1:8000/health', timeout=3).raise_for_status()
    elif args.command == 'health-worker':
        from deskvision.queueing import make_app
        with make_app() as app:
            replies = app.control.inspect(destination=[f'deskvision@{socket.gethostname()}'],
                                          timeout=5).ping()
        if not replies:
            raise SystemExit(1)
    elif args.command == 'gpu-info':
        gpu_info()
    elif args.command == 'bootstrap':
        bootstrap()
    elif args.command == 'web':
        # Only Nginx publishes a host port; it replaces forwarded headers.
        os.execv(sys.executable, [sys.executable, '-m', 'uvicorn', 'deskvision.api:create_app',
                                 '--factory', '--host', '0.0.0.0', '--port', '8000',
                                 '--proxy-headers', '--forwarded-allow-ips=*'])
    else:
        if os.environ.get('DESKVISION_REQUIRE_CUDA') == 'true':
            # CUDA must not be initialized in the parent of a prefork worker.
            subprocess.run([sys.executable, '-c',
                            'import torch; assert torch.cuda.is_available(), "Container CUDA unavailable"'],
                           check=True)
        os.execv(sys.executable, [sys.executable, '-m', 'celery', '-A',
                                 'deskvision.worker:celery_app', 'worker', '--loglevel=INFO',
                                 '--concurrency=1', '--pool=prefork', '--hostname=deskvision@%h'])


if __name__ == '__main__':
    main()
