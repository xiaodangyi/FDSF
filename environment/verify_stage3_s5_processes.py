"""Real Redis + Linux prefork + independent HTTP server; model functions are offline."""

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
from time import monotonic, sleep
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def wait_for(predicate, timeout=50):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except Exception:
            pass
        sleep(.25)
    raise TimeoutError('Acceptance condition was not reached.')


def free_port():
    with socket.socket() as handle:
        handle.bind(('127.0.0.1', 0))
        return handle.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser', action='store_true')
    args = parser.parse_args()
    assert sys.platform == 'linux', 'Use Ubuntu for prefork validation.'
    folder = ROOT / '.local/s5-processes' / uuid4().hex
    folder.mkdir(parents=True)
    redis_port, web_port = free_port(), 8015
    base = f'http://127.0.0.1:{web_port}'
    os.environ.update(DESKVISION_QUEUE_MODE='celery',
                      DESKVISION_REDIS_URL=f'redis://127.0.0.1:{redis_port}/0',
                      DESKVISION_QUEUE_NAME='acceptance-' + folder.name,
                      DESKVISION_DATABASE=str(folder / 'history.sqlite3'),
                      DESKVISION_JWT_SECRET=secrets.token_urlsafe(48),
                      DESKVISION_LOCAL_HTTP='true', GRADIO_ANALYTICS_ENABLED='False',
                      DESKVISION_TEST_EVENTS=str(folder / 'events.jsonl'))
    import httpx
    from PIL import Image
    from redis import Redis
    from sqlalchemy import update
    from deskvision import auth, queueing
    from deskvision.executor import now, repository
    from deskvision.models import RunRecord

    redis = Redis(host='127.0.0.1', port=redis_port, socket_timeout=1)
    report = {'stage': 'S5', 'status': 'running',
              'started_at': datetime.now(timezone.utc).isoformat(), 'checks': [],
              'external_model_calls': 0,
              'scope': 'Actual Redis AOF, independent Web and prefork worker, offline models. '
                       'Crash recovery uses an injected expired heartbeat after a real SIGKILL.'}
    processes, logs = [], []

    def spawn(name, command):
        log = (folder / f'{name}-{len(processes)}.log').open('w')
        logs.append(log)
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env=os.environ.copy(), start_new_session=True)
        processes.append(process)
        return process

    def stop(process):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)

    def start_redis():
        process = spawn('redis', ['redis-server', '--port', str(redis_port), '--bind', '127.0.0.1',
                                 '--protected-mode', 'yes', '--appendonly', 'yes',
                                 '--appendfsync', 'always', '--dir', str(folder), '--save', ''])
        wait_for(lambda: redis.ping())
        return process

    def start_web():
        process = spawn('web', [sys.executable, '-m', 'uvicorn', 'deskvision.api:create_app',
                               '--factory', '--host', '127.0.0.1', '--port', str(web_port)])
        wait_for(lambda: httpx.get(base + '/health', timeout=1).status_code == 200)
        return process

    def start_worker():
        process = spawn('worker', [sys.executable, 'environment/s5_offline_worker.py'])
        wait_for(lambda: queueing.make_app().control.ping(timeout=1), timeout=40)
        return process

    def check(name, **evidence):
        report['checks'].append({'name': name, 'status': 'passed', **evidence})
        print(name + ': passed', flush=True)

    def events():
        path = folder / 'events.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []

    client = None
    try:
        redis_process = start_redis()
        password = secrets.token_urlsafe(20)
        auth.create_user('s5_teacher', password)
        auth.create_user('s5_student', password)
        token, _ = auth.login('s5_teacher', password)
        client = httpx.Client(base_url=base, headers={'Authorization': f'Bearer {token}'}, timeout=15)
        web_process = start_web()
        image = BytesIO()
        Image.new('RGB', (32, 24), color='white').save(image, format='PNG')
        image_data = image.getvalue()

        def submit(route='yolo', publish=True):
            response = client.post('/api/v1/runs', data={'route': route, 'allow_remote': 'true'},
                                   files={'image': ('input.png', image_data, 'image/png')})
            assert response.status_code == (202 if publish else 503), response.text
            return response.json()['run_id'] if publish else response.json()['detail']['run_id']

        first, second = submit(), submit()
        assert repository().get(first)['status'] == repository().get(second)['status'] == 'queued'
        check('Two HTTP tasks persist queued with worker stopped', run_ids=[first, second])
        worker_process = start_worker()
        wait_for(lambda: repository().get(first)['status'] == 'running')
        stop(web_process)
        wait_for(lambda: repository().get(second)['status'] == 'succeeded')
        observed = events()
        end_first = next(e['time'] for e in observed if e['run_id'] == first and e['event'] == 'end')
        start_second = next(e['time'] for e in observed if e['run_id'] == second and e['event'] == 'start')
        assert end_first <= start_second
        pids = sorted({e['pid'] for e in observed})
        assert web_process.pid not in pids and os.getpid() not in pids
        check('Single prefork child executes sequentially after Web exits',
              producer_pid=web_process.pid, worker_parent_pid=worker_process.pid,
              execution_pids=pids, first_end=end_first, second_start=start_second)
        web_process = start_web()
        assert client.get('/api/v1/runs/' + first).json()['status'] == 'succeeded'
        queueing.enqueue(first)
        sleep(1)
        assert sum(e['run_id'] == first and e['event'] == 'start' for e in events()) == 1
        check('Web restart can query completed task; duplicate delivery makes no model call')

        # Create an independent task and block only its simulated remote route.
        from deskvision.task_service import prepare_task
        source = folder / 'input.png'
        source.write_bytes(image_data)
        identity = auth.resolve_token(token)
        crash = prepare_task('both', source, {}, owner_id=identity.user_id)['run_id']
        control = ROOT / '.local/s5-controls' / crash
        control.parent.mkdir(exist_ok=True)
        control.write_text('block', encoding='utf-8')
        queueing.enqueue(crash)
        wait_for(lambda: repository().get(crash)['results'].get('yolo', {}).get('status') == 'ok')
        child = next(e['pid'] for e in events() if e['run_id'] == crash and e['route'] == 'deepseek')
        os.kill(child, signal.SIGKILL)
        # Reproduce a lease expiring without adding two minutes of wall-clock delay.
        with repository().session_factory.begin() as session:
            session.execute(update(RunRecord).where(RunRecord.run_id == crash).values(heartbeat_at=0))
        result = client.get('/api/v1/runs/' + crash).json()
        assert result['status'] == 'partial_failed', result
        assert result['error']['code'] == 'remote_result_unknown'
        assert result['results']['yolo']['status'] == 'ok'
        queueing.enqueue(crash)
        sleep(1)
        assert sum(e['run_id'] == crash and e['route'] == 'deepseek' for e in events()) == 1
        check('Real child SIGKILL retains local result and remote unknown without paid retry',
              killed_child_pid=child, task_status=result['status'], error=result['error']['code'])

        timed = prepare_task('both', source, {}, owner_id=identity.user_id)['run_id']
        (control.parent / timed).write_text('block', encoding='utf-8')
        queueing.make_app().send_task('deskvision.run', args=[timed],
                                      soft_time_limit=4, time_limit=6)
        wait_for(lambda: repository().get(timed)['results'].get('yolo', {}).get('status') == 'ok')
        timed_pid = next(e['pid'] for e in events() if e['run_id'] == timed and e['route'] == 'deepseek')

        def terminated():
            try:
                os.kill(timed_pid, 0)
                return False
            except ProcessLookupError:
                return True

        wait_for(terminated, timeout=15)
        with repository().session_factory.begin() as session:
            session.execute(update(RunRecord).where(RunRecord.run_id == timed).values(heartbeat_at=0))
        result = client.get('/api/v1/runs/' + timed).json()
        assert result['status'] == 'partial_failed' and result['error']['code'] == 'remote_result_unknown'
        check('Actual Celery soft/hard time limits stop blocked child and preserve sibling',
              soft_limit_seconds=4, hard_limit_seconds=6, task_status=result['status'])
        stop(worker_process)
        worker_process = start_worker()
        third = submit()
        wait_for(lambda: repository().get(third)['status'] == 'succeeded')
        check('Worker restarts and consumes a new task', worker_parent_pid=worker_process.pid)

        stop(worker_process)
        aof_task = submit()
        stop(redis_process)
        redis_process = start_redis()
        worker_process = start_worker()
        wait_for(lambda: repository().get(aof_task)['status'] == 'succeeded')
        check('Queued message survives Redis restart through AOF')

        stop(redis_process)
        uncertain = submit(publish=False)
        result = client.get('/api/v1/runs/' + uncertain).json()
        assert result['status'] == 'queued' and result['error']['code'] == 'delivery_uncertain'
        redis_process = start_redis()
        queueing.enqueue(uncertain)
        wait_for(lambda: repository().get(uncertain)['status'] == 'succeeded')
        check('Broker outage returns 503 with saved id; same-id redelivery completes')

        if args.browser:
            ready = ROOT / '.local/s5-browser-ready.json'
            done = ROOT / '.local/s5-browser-done.json'
            if done.exists():
                done.unlink()
            ready.write_text(json.dumps({'base_url': base, 'username': 's5_teacher',
                                         'student': 's5_student', 'password': password,
                                         'image': str(source.relative_to(ROOT))}), encoding='utf-8')
            print('BROWSER_READY', flush=True)
            browser = wait_for(lambda: json.loads(done.read_text()) if done.exists() else None, timeout=180)
            assert browser['status'] == 'passed', browser
            check('Actual browser file control, queue action, timer result, owner isolation and logout')
        report['status'] = 'passed'
        report['versions'] = {'redis_server': redis.info()['redis_version']}
        from importlib.metadata import version
        report['versions'].update({name: version(name) for name in ('celery', 'redis', 'SQLAlchemy')})
        report['events'] = events()
    except Exception as error:
        report['status'] = 'failed'
        report['error_type'] = type(error).__name__
        report['error'] = str(error).replace(str(ROOT), '<project>')[:1500]
        raise
    finally:
        if client:
            client.close()
        for process in reversed(processes):
            stop(process)
        for handle in logs:
            handle.close()
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        report['private_logs'] = folder.relative_to(ROOT).as_posix()
        path = ROOT / 'reports/stage3-S5-process-checks.json'
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'status': report['status'], 'checks': len(report['checks'])}), flush=True)


if __name__ == '__main__':
    main()
