"""S1 Linux GPU 实测。--vlm 才会发送一张原图，且同一阶段不自动重复调用。"""

import argparse
from datetime import datetime
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def check_web(folder):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    with (folder/'web.txt').open('w', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, '-m', 'deskvision.web', '--port', str(port)],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline and child.poll() is None:
                try:
                    with urlopen(f'http://127.0.0.1:{port}/config', timeout=2) as response:
                        data = json.load(response)
                        if response.status == 200 and data.get('components'):
                            return {'status': 'passed', 'http_status': 200,
                                    'component_count': len(data['components']),
                                    'scope': '真实 Linux 服务的 HTTP 配置检查；未操作浏览器按钮。'}
                except (OSError, ValueError):
                    time.sleep(0.5)
            return {'status': 'failed', 'process_exit_code': child.poll()}
        finally:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vlm', action='store_true', help='一次真实 DeepSeek 请求，不自动重试')
    args = parser.parse_args()
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
    os.environ['YOLO_CONFIG_DIR'] = str(ROOT/'.local/ultralytics')
    os.environ['MPLCONFIGDIR'] = str(ROOT/'.local/matplotlib')
    started = datetime.now(ZoneInfo('Asia/Shanghai'))
    folder = ROOT/'runs/stage3/S1'/started.strftime('%Y%m%d-%H%M%S')
    folder.mkdir(parents=True, exist_ok=False)
    result = {'stage': 'S1', 'started_at': started.isoformat(),
              'device_requested': 'cuda:0', 'cpu_build_removed': None,
              'training_executed': False, 'deepseek_attempts_this_run': 0,
              'run_directory': folder.relative_to(ROOT).as_posix(), 'status': 'failed'}
    stage = 'imports'
    try:
        import torch
        import torchvision
        from deskvision.detector import file_sha256
        from deskvision.ui import run_yolo_route, run_vlm_route
        from deskvision.render import find_font
        from environment.verify_linux_environment import inspect

        result['environment'] = inspect(ROOT)
        result['versions'] = {name: version(name) for name in
                              ('torch', 'torchvision', 'ultralytics', 'SAM-2', 'numpy',
                               'opencv-python', 'gradio', 'openai', 'Pillow')}
        result['cpu_build_removed'] = not torch.__version__.endswith('+cpu')
        assert result['cpu_build_removed'] and torch.version.cuda == '12.8'
        stage = 'cuda_operations'
        assert torch.cuda.is_available(), 'CUDA not available'
        result['gpu'] = {'name': torch.cuda.get_device_name(0),
                         'capability': list(torch.cuda.get_device_capability(0)),
                         'torch_architectures': torch.cuda.get_arch_list()}
        a = torch.ones((128, 128), device='cuda:0')
        assert torch.allclose(a @ a, torch.full_like(a, 128))
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device='cuda:0')
        scores = torch.tensor([0.9, 0.8], device='cuda:0')
        assert torchvision.ops.nms(boxes, scores, 0.5).tolist() == [0]
        conv = torch.nn.Conv2d(3, 8, 3).cuda().eval()
        with torch.inference_mode():
            assert torch.isfinite(conv(torch.ones(1, 3, 32, 32, device='cuda:0'))).all()
        torch.cuda.synchronize()
        result['cuda_operations'] = {'matrix': 'passed', 'torchvision_nms': 'passed',
                                     'convolution': 'passed'}
        result['font_present'] = find_font(None).is_file()

        stage = 'offline_tests'
        process = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                                 cwd=ROOT, capture_output=True, text=True, timeout=180)
        (folder/'unittest.txt').write_text(process.stdout+'\n'+process.stderr, encoding='utf-8')
        import re
        match = re.search(r'^Ran (\d+) tests? in ', process.stderr, re.MULTILINE)
        result['offline_tests'] = {'exit_code': process.returncode,
                                   'tests_run': int(match.group(1)) if match else None}
        assert process.returncode == 0 and match

        stage = 'real_yolo_sam'
        resources = json.loads((ROOT/'reports/stage3-S1-resources.json').read_text())
        for item in resources['resources']:
            assert file_sha256(ROOT/item['role']) == item['sha256']
        source = ROOT/resources['sample_path']
        local = run_yolo_route(source, device='cuda:0', output_root=folder/'inference')
        result['local'] = {key: local[key] for key in
                           ('status', 'model', 'input', 'summary', 'timings_ms', 'segmentation') if key in local}
        if local.get('error'):
            result['local']['error'] = local['error']
        assert local['status'] == 'ok'
        assert local['segmentation']['nonempty_masks'] > 0
        result['gpu']['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        result['process_peak_rss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['truth'] = resources['sample_truth']

        stage = 'web_http'
        result['web'] = check_web(folder)
        assert result['web']['status'] == 'passed'

        if args.vlm:
            stage = 'real_deepseek'
            marker = ROOT/'.local/S1-deepseek-attempt.json'
            if marker.exists():
                raise RuntimeError('S1 request already attempted; no automatic repeat')
            # 先持久化调用意图；中断后不自动重新计费。
            write(marker, {'started_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                           'run_directory': folder.relative_to(ROOT).as_posix()})
            result['deepseek_attempts_this_run'] = 1
            remote = run_vlm_route(source, output_root=folder/'inference')
            result['remote'] = {key: remote[key] for key in
                                ('status', 'input', 'summary', 'timings_ms', 'usage', 'error') if key in remote}
            result['same_original_image'] = local.get('input', {}).get('sha256') == remote.get('input', {}).get('sha256')
            assert remote['status'] == 'ok' and result['same_original_image']
        result['status'] = 'passed' if args.vlm else 'local_passed_remote_not_run'
    except Exception as error:
        result['failure'] = {'stage': stage, 'type': type(error).__name__}
        # 完整异常仅留本机，防止把环境路径或第三方响应带入公开报告。
        import traceback
        (folder/'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        result['finished_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        result['scope'] = '单张历史样例的 Linux GPU 链路检查，不代表精度、并发、云端或 Docker 验收。'
        write(ROOT/'reports/stage3-S1-checks.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
