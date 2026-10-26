"""Run S4 authentication and ownership regression checks; retain a redacted record."""

from datetime import datetime
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    started = datetime.now(ZoneInfo('Asia/Shanghai'))
    system = platform.system().lower()
    folder = ROOT / 'runs/stage3/S4' / f'{system}-{started:%Y%m%d-%H%M%S}'
    folder.mkdir(parents=True, exist_ok=False)
    local_config = ROOT / '.local/stage3-test-runtime'
    yolo_config = local_config / 'ultralytics'
    mpl_config = local_config / 'matplotlib'
    yolo_config.mkdir(parents=True, exist_ok=True)
    mpl_config.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
               GRADIO_ANALYTICS_ENABLED='False', YOLO_CONFIG_DIR=str(yolo_config),
               MPLCONFIGDIR=str(mpl_config))
    command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']
    process = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, timeout=360, env=env)
    output = process.stdout.decode('utf-8', 'replace').replace('\r\n', '\n')
    (folder / 'unittest.txt').write_text(output, encoding='utf-8')
    match = re.search(r'^Ran (\d+) tests? in ([0-9.]+)s', output, re.MULTILINE)
    skipped = re.search(r'OK \(skipped=(\d+)\)', output)
    names = ('SQLAlchemy', 'PyJWT', 'pwdlib', 'argon2-cffi', 'fastapi', 'starlette',
             'gradio', 'python-multipart', 'uvicorn')
    summary = {
        'stage': 'S4', 'started_at': started.isoformat(),
        'finished_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'system': platform.system(), 'python_version': platform.python_version(),
        'versions': {name: version(name) for name in names},
        'command': 'python -m unittest discover -s tests -v',
        'exit_code': process.returncode, 'tests_run': int(match[1]) if match else None,
        'skipped': int(skipped[1]) if skipped else 0,
        'duration_seconds': float(match[2]) if match else None,
        'status': 'passed' if process.returncode == 0 and match else 'failed',
        'scope': 'S3 regression, S3-to-S4 migration, auth lifecycle, API ownership, '
                 'real Gradio upload/preview/queue/SSE ownership, fixed model paths and CSRF; '
                 'model runners mocked, no external API calls.',
        'raw_log': (folder / 'unittest.txt').relative_to(ROOT).as_posix(),
        'warnings': sorted(set(line[line.index('ResourceWarning:'):].strip()
                               for line in output.splitlines() if 'ResourceWarning:' in line)),
    }
    public_log = output.replace(str(ROOT), '<project>')
    public_log = public_log.replace(str(ROOT).replace('\\', '/'), '<project>')
    public_log = public_log.replace(sys.prefix, '<python-env>').replace(str(Path.home()), '<user>')
    report = ROOT / f'reports/stage3-S4-{system}-checks.json'
    report.with_suffix('.txt').write_text(public_log, encoding='utf-8')
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=True))
    if process.returncode:
        print(public_log.encode('ascii', 'backslashreplace').decode())
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
