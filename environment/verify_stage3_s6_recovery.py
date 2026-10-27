"""Read-only result checks after WSL recovery; never submits inference tasks."""

from datetime import datetime, timezone
import http.cookiejar
import json
from pathlib import Path
import subprocess
from time import monotonic, sleep
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:8001'


def runtime():
    command = ['wsl', '-d', 'Ubuntu', '-u', 'root', '--', 'bash', '-lc',
               'cat /proc/sys/kernel/random/boot_id; '
               'docker inspect deskvision-s6-web-1 deskvision-s6-worker-1 '
               'deskvision-s6-redis-1 deskvision-s6-nginx-1 '
               '--format "{{json .RestartCount}} {{.State.Health.Status}}"']
    result = subprocess.run(command, capture_output=True, check=True, timeout=30)
    lines = result.stdout.decode('utf-8').splitlines()
    return {'boot_id': lines[0], 'services': lines[1:]}


def main():
    report = {'stage': 'S6', 'status': 'running',
              'started_at': datetime.now(timezone.utc).isoformat(),
              'scope': 'Windows loopback requests with a foreground WSL Compose process; '
                       'checks original paid-task data without resubmitting it.',
              'new_inference_tasks': 0, 'external_model_calls': 0}
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def get(path):
        return opener.open(BASE + path, timeout=10)

    try:
        before = runtime()
        samples = []
        start = monotonic()
        for index in range(13):
            with get('/health') as response:
                samples.append(response.status)
            print(f'Loopback sample {index + 1}/13: HTTP {samples[-1]}', flush=True)
            if index < 12:
                sleep(5)
        after = runtime()
        assert before['boot_id'] == after['boot_id'], 'WSL rebooted during observation.'
        assert all(value == '0 healthy' for value in after['services']), after['services']
        assert len(after['services']) == 4
        report.update(observation_seconds=round(monotonic() - start, 2),
                      health_samples=samples, stable_wsl_boot=True,
                      service_restart_counts=[0, 0, 0, 0], all_services_healthy=True)

        accounts = json.loads((ROOT / '.local/s6/accounts.json').read_text(encoding='utf-8'))
        account = next(item for item in accounts if item['username'] == 'teacher01')
        request = urllib.request.Request(BASE + '/api/auth/login',
                                         data=json.dumps(account).encode(),
                                         headers={'Content-Type': 'application/json', 'Origin': BASE})
        with opener.open(request, timeout=10) as response:
            assert response.status == 200
        original = json.loads((ROOT / 'reports/stage3-S6-container-checks.json').read_text(encoding='utf-8'))
        run_id = original['paid_run']['run_id']
        with get('/api/v1/runs/' + run_id) as response:
            result = json.load(response)
        assert result['status'] == 'succeeded'
        assert all(item['input']['sha256'] == original['input_sha256']
                   for item in result['results'].values())
        with get('/api/v1/runs/' + run_id + '/export.zip') as response:
            assert response.status == 200 and response.read(2) == b'PK'
        report.update(existing_dual_run_id=run_id, existing_result_and_zip_preserved=True,
                      status='passed')
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__,
                      error=str(error).replace(str(ROOT), '<project>')[:500])
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        (ROOT / 'reports/stage3-S6-recovery-checks.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'status': report['status'], 'external_model_calls': 0}), flush=True)


if __name__ == '__main__':
    main()
