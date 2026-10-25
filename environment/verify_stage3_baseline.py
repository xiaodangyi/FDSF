"""运行迁入基线的离线检查并留真实结果，不启动训练或云端请求。"""

from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """使用当前解释器检查本仓库，不复用旧仓库测试结果。"""
    started = datetime.now(ZoneInfo('Asia/Shanghai'))
    folder = ROOT / 'runs/stage3/S0' / started.strftime('%Y%m%d-%H%M%S-%f')
    folder.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                             encoding='utf-8', errors='replace',
                             env=dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
                                      GRADIO_ANALYTICS_ENABLED='False'))
    output = process.stdout + '\n' + process.stderr
    log = folder / 'unittest.txt'
    log.write_text(output, encoding='utf-8')
    match = re.search(r'^Ran (\d+) tests? in ', output, re.MULTILINE)
    summary = {
        'stage': 'S0', 'started_at': started.isoformat(),
        'finished_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'command': 'python -m unittest discover -s tests -v',
        'python_version': sys.version.split()[0], 'exit_code': process.returncode,
        'tests_run': int(match.group(1)) if match else None,
        'status': 'passed' if process.returncode == 0 and match else 'failed',
        'log': log.relative_to(ROOT).as_posix(),
        'scope': '离线契约与界面构建，不代表模型、Linux、Docker或云端验收。',
        'warnings': sorted(set(line[line.index('ResourceWarning:'):].strip()
                               for line in output.splitlines() if 'ResourceWarning:' in line)),
    }
    target = ROOT / 'reports/stage3-S0-checks.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
