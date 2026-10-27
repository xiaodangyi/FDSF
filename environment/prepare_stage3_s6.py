"""Generate ignored local container settings, preserving existing host configuration."""

import json
import os
from pathlib import Path
import secrets
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.platform != 'linux':
        raise SystemExit('请在 Ubuntu 项目环境运行，以生成 Docker 可解析的 Linux 路径。')
    folder = ROOT / '.local/s6'
    folder.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / '.local/config.json').read_text(encoding='utf-8-sig'))
    sources = {'YOLO_WEIGHTS': Path(config['yolo_weights']),
               'FINETUNED_WEIGHTS': ROOT / 'runs/train/desk8-full-20261023-204546/weights/best.pt',
               'SAM2_WEIGHTS': Path(config['sam2_weights'])}
    sources = {key: (value if value.is_absolute() else ROOT / value).resolve()
               for key, value in sources.items()}
    if any(not value.is_file() for value in sources.values()):
        raise SystemExit('缺少已有模型文件。不会自动下载或替换权重。')
    auth = folder / 'auth.json'
    if not auth.exists():
        auth.write_text(json.dumps({'jwt_secret': secrets.token_urlsafe(48), 'local_http': True}),
                        encoding='utf-8')
    accounts = folder / 'accounts.json'
    if not accounts.exists():
        accounts.write_text(json.dumps([
            {'username': name, 'password': secrets.token_urlsafe(20)}
            for name in ('teacher01', 'student01')], indent=2), encoding='utf-8')
    public = folder / 'config.json'
    public.write_text(json.dumps({'yolo_weights': '/app/models/yolov8n.pt',
                                 'sam2_weights': '/app/models/sam2.1_hiera_tiny.pt'}, indent=2), encoding='utf-8')
    secret = folder / 'deepseek.env'
    if not secret.exists():
        original = ROOT / '.local/deepseek.env'
        if original.is_file():
            shutil.copyfile(original, secret)
        elif os.environ.get('DEEPSEEK_API_KEY'):
            secret.write_text('DEEPSEEK_API_KEY=' + os.environ['DEEPSEEK_API_KEY'] + '\n', encoding='utf-8')
        else:
            secret.write_text('# Fill in privately before a real cloud call.\n', encoding='utf-8')
    env = ROOT / '.env'
    if not env.exists():
        lines = ['COMPOSE_PROJECT_NAME=deskvision-s6', 'DESKVISION_IMAGE=deskvision:s6',
                 'WEB_PORT=8001', 'PRIVATE_DIR=./.local/s6']
        lines.extend(f"{key}='{value.as_posix()}'" for key, value in sources.items())
        env.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for path in (auth, accounts, secret, env):
        path.chmod(0o600)
    # File-backed Compose secrets retain host permissions; the nonroot UID needs read
    # permission. Restrict the parent directory while allowing read-only container files.
    for path in (auth, accounts, secret):
        path.chmod(0o644)
    folder.chmod(0o700)
    print(json.dumps({'status': 'prepared', 'private_directory': '.local/s6',
                      'existing_env_preserved': True, 'weights_verified': len(sources),
                      'external_calls': 0}))


if __name__ == '__main__':
    main()
