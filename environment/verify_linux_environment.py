"""只读核验 Linux；只使用标准库，不导入模型、不输出密钥。"""

import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {"exit_code": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"exit_code": None, "error": type(error).__name__}


def inspect(root):
    release = {}
    for line in Path('/etc/os-release').read_text().splitlines():
        key, _, value = line.partition('=')
        release[key] = value.strip('"')
    modules = ('torch', 'torchvision', 'ultralytics', 'gradio', 'cv2', 'sam2',
               'fastapi', 'sqlalchemy', 'redis', 'celery', 'openai')
    packages = {name: importlib.util.find_spec(name) is not None for name in modules}
    disk = shutil.disk_usage('/')
    return {
        'stage': 'S1', 'checked_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'scope': '只读环境清单；不代表实际推理、网页或 Docker 验收通过。',
        'os_release': release, 'kernel': platform.release(),
        'architecture': platform.machine(), 'logical_cpus': os.cpu_count(),
        'cpu': command(['lscpu']), 'memory': command(['free', '-b']),
        'root_disk_bytes': dict(zip(('total', 'used', 'free'), disk)),
        'mounts': command(['findmnt', '-rn', '-o', 'TARGET,FSTYPE']),
        'python': platform.python_version(), 'python_modules': packages,
        'executables': {name: shutil.which(name) for name in
                        ('python3.12', 'uv', 'conda', 'docker', 'git', 'curl', 'fc-list')},
        'gpu': command(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                        '--format=csv,noheader']),
        'fonts': command(['fc-list', ':lang=zh', 'family']),
        'project_readable': root.is_dir() and (root/'deskvision/__main__.py').is_file(),
        'private_config_present': (root/'.local/config.json').is_file(),
        'docker': command(['docker', '--version']),
    }


if __name__ == '__main__':
    project = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    print(json.dumps(inspect(project), ensure_ascii=False, indent=2))
