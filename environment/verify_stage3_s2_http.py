"""Explicit real local-GPU HTTP smoke check; never selects the remote route."""

import argparse
from datetime import datetime
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from time import monotonic, sleep
from zoneinfo import ZoneInfo
from zipfile import ZipFile

import httpx
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--check-ui', action='store_true', help='Also run the real local route through Gradio HTTP callbacks.')
    args = parser.parse_args()
    source = args.image.resolve()
    content = source.read_bytes()
    started = datetime.now(ZoneInfo('Asia/Shanghai'))
    folder = ROOT / 'runs/stage3/S2' / f'http-{started:%Y%m%d-%H%M%S}'
    folder.mkdir(parents=True, exist_ok=False)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    checks = {}
    summary = {'stage': 'S2', 'started_at': started.isoformat(), 'route': 'yolo',
               'external_api_requests': 0, 'sample': source.name,
               'input_sha256': sha256(content).hexdigest(), 'checks': checks}
    server = None
    try:
        with (folder / 'server.txt').open('wb') as log:
            server = subprocess.Popen(
                [sys.executable, '-m', 'uvicorn', 'deskvision.api:create_app', '--factory',
                 '--host', '127.0.0.1', '--port', str(port)], cwd=ROOT, stdout=log, stderr=log,
                env=dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
                         GRADIO_ANALYTICS_ENABLED='False', GRADIO_TEMP_DIR=str(ROOT / 'runs/web-cache')))
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=60, trust_env=False) as client:
                deadline = monotonic() + 60
                while True:
                    if server.poll() is not None:
                        raise RuntimeError('server_exited')
                    try:
                        if client.get('/health').status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if monotonic() >= deadline:
                        raise TimeoutError('server_start_timeout')
                    sleep(.5)
                checks['health_http_200'] = True
                checks['page_http_200'] = client.get('/ui/').status_code == 200
                checks['page_config_http_200'] = client.get('/ui/config').status_code == 200
                response = client.post('/api/v1/runs', data={'route': 'yolo', 'device': 'cuda:0'},
                                       files={'image': (source.name, content, 'image/png')})
                response.raise_for_status()
                checks['upload_http_202'] = response.status_code == 202
                run_id = response.json()['run_id']
                deadline = monotonic() + 180
                states = [response.json()['status']]
                while True:
                    response = client.get(f'/api/v1/runs/{run_id}')
                    response.raise_for_status()
                    task = response.json()
                    if states[-1] != task['status']:
                        states.append(task['status'])
                    if task['status'] in ('succeeded', 'failed', 'partial_failed'):
                        break
                    if monotonic() >= deadline:
                        raise TimeoutError('inference_timeout')
                    sleep(.25)
                summary['observed_states'] = states
                summary['run_id'] = run_id
                (folder / 'task.json').write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding='utf-8')
                checks['task_succeeded'] = task['status'] == 'succeeded'
                report = task['results'].get('yolo', {})
                checks['same_original_sha256'] = report.get('input', {}).get('sha256') == summary['input_sha256']
                checks['gpu_used'] = report.get('model', {}).get('device', '').startswith('cuda')
                checks['segmentation_ok'] = report.get('segmentation', {}).get('status') == 'completed'
                summary['counts'] = report.get('summary', [])
                summary['timings_ms'] = report.get('timings_ms', {})
                summary['nonempty_masks'] = report.get('segmentation', {}).get('nonempty_masks')
                downloaded = []
                for name, url in report.get('artifact_urls', {}).items():
                    download = client.get(url)
                    download.raise_for_status()
                    if name.endswith('.png'):
                        with Image.open(BytesIO(download.content)) as image:
                            image.load()
                    downloaded.append({'name': name, 'http_status': download.status_code,
                                       'bytes': len(download.content),
                                       'sha256': sha256(download.content).hexdigest()})
                summary['downloads'] = downloaded
                checks['segmented_image_downloaded'] = any(item['name'] == 'segmented.png' for item in downloaded)
                checks['internal_json_blocked'] = client.get(
                    f'/api/v1/runs/{run_id}/artifacts/yolo/result.json').status_code == 404
                checks['no_remote_route'] = set(task['results']) == {'yolo'}
                if args.check_ui:
                    from gradio_client import Client, handle_file

                    ui = Client(f'http://127.0.0.1:{port}/ui/', verbose=False,
                                download_files=folder / 'client-files')
                    # Browser load normally initializes this per-session state.
                    ui.predict(api_name='/new_session')
                    values = (handle_file(str(source)), 'finetuned', '', .25, .7, 640, 'cuda:0', True)
                    ui.predict(*values, api_name='/set_image')
                    outputs = ui.predict(None, None, False, *values, api_name='/run_yolo')
                    ui_report = outputs[2]
                    checks['ui_local_route_ok'] = ui_report.get('status') == 'ok'
                    checks['ui_same_original_sha256'] = ui_report.get('input', {}).get('sha256') == summary['input_sha256']
                    with ZipFile(outputs[6]) as bundle:
                        names = bundle.namelist()
                        checks['ui_zip_has_detection_and_masks'] = all(
                            name in names for name in ('yolo/annotated.png', 'yolo/segmented.png',
                                                      'yolo/result.json', 'yolo/masks/instance-0001.png'))
                        summary['ui_zip_files'] = names
                    summary['ui_route_timings_ms'] = ui_report.get('timings_ms', {})
                    ui.close()
    except Exception as error:
        summary['error_type'] = type(error).__name__
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        summary['finished_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        summary['status'] = 'passed' if checks and all(checks.values()) and 'error_type' not in summary else 'failed'
        summary['server_log'] = (folder / 'server.txt').relative_to(ROOT).as_posix()
        summary['scope'] = ('Real local GPU via TCP HTTP; Gradio callbacks exercised via HTTP client.'
                            if args.check_ui else 'Real local GPU via TCP HTTP; mounted UI fetched.')
        summary['scope'] += ' No browser clicks or live DeepSeek.'
        report_path = ROOT / 'reports/stage3-S2-linux-http.json'
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=True))
    return 0 if summary['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
