"""S5 full offline regression with a retained, redacted report."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    started = datetime.now(timezone.utc)
    folder = ROOT / 'runs/stage3/S5' / started.strftime('%Y%m%d-%H%M%S')
    folder.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8='1', GRADIO_ANALYTICS_ENABLED='False',
               DESKVISION_QUEUE_MODE='inline',
               YOLO_CONFIG_DIR=str(ROOT / '.local/s5-tests/ultralytics'),
               MPLCONFIGDIR=str(ROOT / '.local/s5-tests/matplotlib'))
    result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                            cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=360)
    output = result.stdout.decode('utf-8', 'replace').replace('\r\n', '\n')
    (folder / 'unittest.txt').write_text(output, encoding='utf-8')
    match = re.search(r'^Ran (\d+) tests in ([\d.]+)s', output, re.MULTILINE)
    skipped = re.search(r'OK \(skipped=(\d+)\)', output)
    system = platform.system().lower()
    public = output.replace(str(ROOT), '<project>').replace(sys.prefix, '<python-env>')
    public = public.replace(str(Path.home()), '<user>')
    report = {'stage': 'S5', 'system': system, 'started_at': started.isoformat(),
              'finished_at': datetime.now(timezone.utc).isoformat(),
              'python': platform.python_version(), 'exit_code': result.returncode,
              'tests_run': int(match[1]) if match else None,
              'duration_seconds': float(match[2]) if match else None,
              'skipped': int(skipped[1]) if skipped else 0,
              'status': 'passed' if result.returncode == 0 and match else 'failed',
              'scope': 'Offline regression and S5 queue/recovery contracts. Models mocked; '
                       'actual Redis/prefork evidence is in the separate process report.',
              'external_model_calls': 0}
    path = ROOT / f'reports/stage3-S5-{system}-checks.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    path.with_suffix('.txt').write_text(public, encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    if result.returncode:
        print(public)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
