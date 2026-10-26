"""Acceptance-only worker: real Celery, offline model functions, no remote requests."""

from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import sleep, time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deskvision import ui
from deskvision.worker import celery_app


def fake(route, image_path, *, output_root, **_parameters):
    run_id = Path(image_path).parent.name
    event = {'run_id': run_id, 'route': route, 'pid': os.getpid(), 'time': time()}
    with Path(os.environ['DESKVISION_TEST_EVENTS']).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({**event, 'event': 'start'}) + '\n')
    control = ROOT / '.local/s5-controls' / run_id
    if route == 'deepseek' and control.is_file():
        # The test kills this child while the simulated remote request is pending.
        sleep(300)
    sleep(3)
    destination = Path(output_root) / route
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'annotated.png').write_bytes(Path(image_path).read_bytes())
    (destination / 'result.json').write_text('{}', encoding='utf-8')
    with Path(os.environ['DESKVISION_TEST_EVENTS']).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({**event, 'event': 'end', 'time': time()}) + '\n')
    return {'status': 'ok', 'route': route,
            'input': {'sha256': sha256(Path(image_path).read_bytes()).hexdigest()},
            'summary': [{'category_key': 'cup', 'category_name': '杯子', 'count': 1}],
            'items': [], 'result_dir': str(destination),
            'artifacts': {'result': 'result.json', 'visualization': 'annotated.png'}}


ui.run_yolo_route = lambda path, **kw: fake('yolo', path, **kw)
ui.run_vlm_route = lambda path, **kw: fake('deepseek', path, **kw)


if __name__ == '__main__':
    celery_app.worker_main(['worker', '--loglevel=INFO', '--concurrency=1', '--pool=prefork'])
