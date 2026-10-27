"""Actual Compose/GPU/persistence acceptance. Paid request requires an explicit one-use flag."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ['docker', 'compose', '-f', 'compose.yaml', '-f', 'compose.gpu.yaml']


def wait_for(predicate, timeout=240):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        try:
            value = predicate()
        except (ConnectionError, OSError):
            value = None
        if value:
            return value
        sleep(1)
    raise TimeoutError('Container acceptance condition timed out.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-one-remote', action='store_true',
                        help='Use only after explicit human authorization; never auto-retry.')
    parser.add_argument('--browser', action='store_true')
    args = parser.parse_args()
    assert sys.platform == 'linux'
    import httpx

    started = datetime.now(timezone.utc)
    folder = ROOT / '.local/s6/acceptance' / started.strftime('%Y%m%d-%H%M%S')
    folder.mkdir(parents=True, exist_ok=True)
    report = {'stage': 'S6', 'started_at': started.isoformat(), 'status': 'running',
              'checks': [], 'paid_task_submissions': 0, 'real_model_routes': [],
              'scope': 'Actual local GPU Compose stack, Nginx HTTP loopback entry, '
                       'persistent volumes, worker/broker failure; no cloud deployment.'}

    def command(name, args, timeout=240, check=True):
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        (folder / (name + '.log')).write_text(result.stdout + result.stderr, encoding='utf-8')
        if check and result.returncode:
            raise RuntimeError(f'{name} failed (exit {result.returncode}); inspect private log.')
        return result

    def compose(name, *args, timeout=240, check=True):
        return command(name, [*COMPOSE, *args], timeout, check)

    def check(name, **evidence):
        report['checks'].append({'name': name, 'status': 'passed', **evidence})
        print(name + ': passed', flush=True)

    def finished(run_id):
        response = client.get('/api/v1/runs/' + run_id)
        response.raise_for_status()
        value = response.json()
        return value if value['status'] not in ('queued', 'running') else None

    client = student = None
    try:
        compose('config', 'config', '--quiet')
        compose('redis-up', 'up', '-d', 'redis')
        compose('bootstrap', 'run', '--rm', '--no-deps', 'bootstrap')
        compose('up', 'up', '-d', '--wait', '--wait-timeout', '180', timeout=240)
        report['runtime_versions'] = {
            'docker_server': command('docker-version', ['docker', 'version', '--format',
                                                       '{{.Server.Version}}']).stdout.strip(),
            'compose': command('compose-version', ['docker', 'compose', 'version', '--short']).stdout.strip(),
            'nvidia_container_toolkit': command('toolkit-version', ['nvidia-ctk', '--version']).stdout.strip(),
        }
        packages = command('packages', ['docker', 'run', '--rm', '--entrypoint', 'python',
                                        'deskvision:s6', '-m', 'pip', 'freeze']).stdout
        (ROOT / 'reports/stage3-S6-container-packages.txt').write_text(packages, encoding='utf-8')
        accounts = json.loads((ROOT / '.local/s6/accounts.json').read_text())
        base = 'http://127.0.0.1:8001'
        client = httpx.Client(base_url=base, headers={'Origin': base}, timeout=20)
        student = httpx.Client(base_url=base, headers={'Origin': base}, timeout=20)
        anonymous = httpx.get(base + '/api/v1/runs', timeout=10)
        assert anonymous.status_code == 401
        for current, name in ((client, 'teacher01'), (student, 'student01')):
            account = next(value for value in accounts if value['username'] == name)
            response = current.post('/api/auth/login', json=account)
            assert response.status_code == 200
        check('All four services start healthy; Nginx login and unauthenticated rejection work')

        gpu = compose('gpu-info', 'run', '--rm', '--no-deps', 'worker', 'gpu-info')
        gpu_info = json.loads(gpu.stdout.strip().splitlines()[-1])
        assert gpu_info['available'] and gpu_info['arithmetic_ok']
        assert gpu_info['torch'].endswith('+cu128')
        report['gpu'] = gpu_info
        check('GPU container executes CUDA arithmetic using GPU PyTorch', **gpu_info)

        image_audit = (
            "from pathlib import Path; import json; root=Path('/app'); "
            "assert not (root/'.env').exists(); assert not (root/'.git').exists(); "
            "assert not list((root/'.local').rglob('*')); "
            "assert not list((root/'models').rglob('*.pt')); "
            "assert not list((root/'runs').rglob('*')); "
            "print(json.dumps({'private_files_baked':False}))"
        )
        command('image-audit', ['docker', 'run', '--rm', '--entrypoint', 'python',
                                'deskvision:s6', '-c', image_audit])
        ids = compose('container-ids', 'ps', '-q').stdout.splitlines()
        inspected = json.loads(command('inspect', ['docker', 'inspect', *ids]).stdout)
        for container in inspected:
            service = container['Config']['Labels']['com.docker.compose.service']
            published = container['HostConfig'].get('PortBindings') or {}
            if service != 'nginx':
                assert not published, service
            else:
                assert all(v['HostIp'] == '127.0.0.1' for values in published.values() for v in values)
            if service in ('web', 'worker'):
                assert container['Config']['User'] == '10001:10001'
                models = [m for m in container['Mounts'] if m['Destination'].startswith('/app/models/')]
                assert len(models) == 3 and all(not m['RW'] for m in models)
        check('Image excludes private data; nonroot apps use readonly models; only loopback gateway published')

        paths = json.loads((ROOT / '.local/config.json').read_text(encoding='utf-8-sig'))
        source = Path(paths['smoke_image'])
        source = source if source.is_absolute() else ROOT / source
        content = source.read_bytes()
        fingerprint = sha256(content).hexdigest()
        report['input_sha256'] = fingerprint

        def submit(route='yolo', segment=True, expected=202):
            response = client.post('/api/v1/runs', data={
                'route': route, 'allow_remote': 'true' if route == 'both' else 'false',
                'mode': 'finetuned', 'device': 'cuda:0', 'segment': str(segment).lower(),
            }, files={'image': ('demo-original.jpg', content, 'image/jpeg')})
            assert response.status_code == expected, f'HTTP {response.status_code}'
            value = response.json()
            return value['run_id'] if expected == 202 else value['detail']['run_id']

        local = submit()
        result = wait_for(lambda: finished(local))
        assert result['status'] == 'succeeded', result.get('error') or result['results']
        model = result['results']['yolo']
        assert model['model']['device'] == 'cuda:0'
        assert model['input']['sha256'] == fingerprint
        assert model['segmentation']['device'] == 'cuda:0'
        assert model['segmentation']['prompt_count'] > 0
        report['real_model_routes'].append({'run_id': local, 'route': 'yolo', 'device': 'cuda:0',
                                            'segmentation_prompts': model['segmentation']['prompt_count']})
        assert student.get('/api/v1/runs/' + local).status_code == 404
        archive = client.get('/api/v1/runs/' + local + '/export.zip')
        assert archive.status_code == 200
        with ZipFile(BytesIO(archive.content)) as package:
            assert any(name.endswith('segmented.png') for name in package.namelist())
        check('Real fine-tuned YOLO and SAM run through Nginx/Redis/worker on GPU; result and ZIP remain private',
              run_id=local, input_sha256=fingerprint, prompts=model['segmentation']['prompt_count'])

        compose('stop-worker', 'stop', '-t', '15', 'worker')
        first, second = submit(segment=False), submit(segment=False)
        assert client.get('/api/v1/runs/' + first).json()['status'] == 'queued'
        assert client.get('/api/v1/runs/' + second).json()['status'] == 'queued'
        # Recreate the broker while both messages remain unconsumed. Its AOF
        # volume must carry the queue into the replacement container.
        sleep(2)
        compose('recreate-redis-with-pending-tasks', 'up', '-d', '--force-recreate',
                '--wait', '--wait-timeout', '90', 'redis', timeout=120)
        compose('start-worker', 'start', 'worker')
        one = wait_for(lambda: finished(first))
        two = wait_for(lambda: finished(second))
        assert one['status'] == two['status'] == 'succeeded'
        assert one['finished_at'] <= two['started_at']
        check('Unconsumed messages survive Redis container recreation; worker restart executes real GPU tasks sequentially',
              first_run=first, second_run=second)

        compose('stop-redis', 'stop', 'redis')
        uncertain = submit(segment=False, expected=503)
        pending = client.get('/api/v1/runs/' + uncertain).json()
        assert pending['error']['code'] == 'delivery_uncertain'
        compose('start-redis', 'start', 'redis')
        sleep(3)
        compose('redeliver', 'exec', '-T', 'worker', 'python', 'environment/manage_queue.py',
                'redeliver', uncertain)
        assert wait_for(lambda: finished(uncertain))['status'] == 'succeeded'
        check('Redis outage returns saved run id; same-id redelivery recovers after Redis restart')

        # Stop all containers without deleting volumes, then recreate them.
        before = client.get('/api/v1/runs/' + local).json()
        compose('down-preserve-volumes', 'down', timeout=120)
        compose('recreate', 'up', '-d', '--wait', '--wait-timeout', '180', timeout=240)
        after_response = client.get('/api/v1/runs/' + local)
        assert after_response.status_code == 200  # JWT/session and signing key also survive.
        after = after_response.json()
        assert before == after
        assert client.get('/api/v1/runs/' + local + '/export.zip').status_code == 200
        check('Recreating every container preserves authenticated session, history, original result and ZIP')

        if args.browser:
            ready = ROOT / '.local/s6-browser-ready.json'
            done = ROOT / '.local/s6-browser-done.json'
            if done.exists():
                done.unlink()
            browser_input = ROOT / '.local/s6/browser-input.jpg'
            browser_input.write_bytes(content)
            ready.write_text(json.dumps({'base_url': base, 'accounts': accounts,
                                         'image': browser_input.relative_to(ROOT).as_posix()}), encoding='utf-8')
            print('BROWSER_READY', flush=True)
            browser = wait_for(lambda: json.loads(done.read_text()) if done.exists() else None, timeout=180)
            assert browser['status'] == 'passed'
            check('Actual browser file control, Gradio SSE and result polling work through Nginx')

        if args.allow_one_remote:
            used = ROOT / '.local/s6/remote-attempt-used.json'
            # Persistent exclusive marker prevents even a rerun of this script from
            # silently spending another paid request. Do not delete it automatically.
            with used.open('x', encoding='utf-8') as marker:
                json.dump({'reserved_at': datetime.now(timezone.utc).isoformat(),
                           'human_authorized_maximum': 1}, marker)
            report['paid_task_submissions'] = 1
            remote = submit(route='both')
            used.write_text(json.dumps({'run_id': remote, 'human_authorized_maximum': 1}), encoding='utf-8')
            result = wait_for(lambda: finished(remote))
            report['paid_run'] = {'run_id': remote, 'status': result['status'],
                                  'results': {route: {key: value.get(key) for key in
                                      ('status', 'input', 'summary', 'usage', 'error')}
                                      for route, value in result['results'].items()}}
            assert result['status'] == 'succeeded', 'Paid route did not succeed; budget consumed, no retry.'
            assert result['results']['yolo']['input']['sha256'] == fingerprint
            assert result['results']['deepseek']['input']['sha256'] == fingerprint
            check('One authorized real dual-route request succeeds on identical original bytes', run_id=remote)
        else:
            report['paid_run'] = {'status': 'not_executed', 'reason': 'No paid-call flag supplied.'}
        report['status'] = 'passed'
        raw = compose('final-ps', 'ps', '--format', 'json').stdout.strip()
        services = json.loads(raw) if raw.startswith('[') else [json.loads(line) for line in raw.splitlines() if line.strip()]
        report['services'] = [{'service': item.get('Service'), 'state': item.get('State'),
                               'health': item.get('Health'), 'image': item.get('Image')}
                              for item in services]
        report['image_id'] = command('image-id', ['docker', 'image', 'inspect', 'deskvision:s6',
                                                 '--format', '{{.Id}}']).stdout.strip()
    except Exception as error:
        report['status'] = 'failed'
        report['error_type'] = type(error).__name__
        report['error'] = str(error).replace(str(ROOT), '<project>')[:1500]
        compose('failure-logs', 'logs', '--no-color', '--tail', '100', check=False)
        raise
    finally:
        if client:
            client.close()
        if student:
            student.close()
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        report['private_logs'] = folder.relative_to(ROOT).as_posix()
        (ROOT / 'reports/stage3-S6-container-checks.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'status': report['status'], 'checks': len(report['checks']),
                          'paid_task_submissions': report['paid_task_submissions']}), flush=True)


if __name__ == '__main__':
    main()
