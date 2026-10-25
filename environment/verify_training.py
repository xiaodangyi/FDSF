"""一轮环境训练自检；执行前先打开可见终端，不属于正式项目训练。"""
from pathlib import Path
import json
import os
import shutil
from project_config import ROOT, load_local_paths

os.environ['YOLO_CONFIG_DIR'] = str(ROOT / 'environment' / 'ultralytics-settings')
os.environ['MPLCONFIGDIR'] = str(ROOT / 'environment' / 'matplotlib-cache')
os.environ['YOLO_OFFLINE'] = 'true'
Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)

from ultralytics import YOLO, settings
import ultralytics
import yaml

settings.update({'sync': False, 'wandb': False, 'mlflow': False, 'comet': False, 'tensorboard': False})

if __name__ == '__main__':
    paths = load_local_paths()
    print('即将进行一轮环境训练自检，请确认终端可见；这不是八类项目训练。', flush=True)
    data_root = ROOT / 'environment' / 'training-smoke-data'
    source = paths['smoke_dataset']
    shutil.copytree(source, data_root, dirs_exist_ok=True)
    config = yaml.safe_load((Path(ultralytics.__file__).parent / 'cfg/datasets/coco8.yaml').read_text(encoding='utf-8'))
    config['path'] = str(data_root)
    config.pop('download', None)
    config_path = data_root / 'data.yaml'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    model = YOLO(str(paths['yolo_weights']))
    model.train(data=str(config_path), epochs=1, batch=2, imgsz=320, device=0,
                workers=0, amp=False, plots=False, save=False, cache=False,
                project=str(ROOT / 'environment' / 'training-smoke'),
                name='check', exist_ok=True, verbose=True)
    report = {'status': 'passed', 'epochs': 1, 'training_images': 4,
              'validation_images': 4, 'purpose': '仅为环境训练自检，不代表八类项目训练'}
    (ROOT / 'environment' / 'training-verification.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print('一轮环境训练自检通过。')
