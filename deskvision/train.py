"""核验数据与训练范围，在可见终端中启动 YOLOv8 微调。"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import traceback

import torch
import yaml

from .audit_data import require_training_ready, validate_export
from .config import ROOT, local_weights
from .data import read_json, save_json, sha256

TRAIN_CONFIG = ROOT / 'configs/training.yaml'


def record_epoch_result(state, row):
    """Ultralytics 最后会再验证 best.pt；不将该回调当成额外训练轮。"""
    if any(previous['epoch'] == row['epoch'] for previous in state['epochs']):
        state['best_checkpoint_validation_metrics'] = row['validation_metrics']
        return False
    state['epochs'].append(row)
    return True


def load_training_config(config_path=TRAIN_CONFIG):
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    required = {'schema_version', 'dataset', 'model', 'epochs', 'image_size', 'batch_size',
                'workers', 'device', 'amp', 'cache', 'seed', 'save_period', 'output_dir'}
    if not isinstance(config, dict) or config.get('schema_version') not in (1, 2):
        raise ValueError('训练配置字段或版本不符合预期。')
    full_training = config['schema_version'] == 2
    if full_training:
        required |= {'patience', 'authorization', 'dataset_manifest_sha256'}
    if set(config) != required:
        raise ValueError('训练配置字段或版本不符合预期。')
    expected = {'model': 'yolov8n', 'epochs': 3, 'image_size': 640, 'batch_size': 4,
                'workers': 0, 'device': 0, 'amp': False, 'cache': False,
                'seed': 42, 'save_period': 1, 'output_dir': 'runs/train'}
    if full_training:
        expected.update(epochs=100, patience=20, save_period=10)
        if not isinstance(config['authorization'], str) or not config['authorization'].strip():
            raise ValueError('正式训练配置必须记录用户要求长训练的授权。')
        digest = config['dataset_manifest_sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('正式训练必须固定已复核数据清单的 SHA-256。')
    if any(type(config[key]) is not type(value) or config[key] != value
           for key, value in expected.items()):
        raise ValueError('参数必须符合当前配置版本的已核对训练范围。')
    if not isinstance(config['dataset'], str):
        raise ValueError('训练数据目录必须是文字路径。')
    dataset = (ROOT / config['dataset']).resolve()
    if dataset == (ROOT / 'data').resolve() or not dataset.is_relative_to((ROOT / 'data').resolve()):
        raise ValueError('训练数据版本必须位于项目 data 目录内。')
    return config


def environment_preflight(config_path=TRAIN_CONFIG):
    """在数据尚未就绪时也可单独验证环境；不读取训练图、不启动训练。"""
    config = load_training_config(config_path)
    local_config = read_json(ROOT / '.local/config.json')
    expected_python = (ROOT / local_config['ai_cuda_root'] / 'python.exe').resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError('当前解释器不是本机配置的 ai-cuda Python。')
    weights = local_weights(None)
    dataset = (ROOT / config['dataset']).resolve()
    if not torch.cuda.is_available():
        raise RuntimeError('没有可用 CUDA GPU；为避免悄悄改用 CPU，本次停止。')
    if torch.cuda.device_count() <= config['device']:
        raise RuntimeError('配置的 CUDA 设备编号不存在。')
    device = torch.cuda.get_device_properties(config['device'])
    if device.total_memory < 6 * 1024**3:
        raise RuntimeError('GPU 显存低于首轮配置要求，请先调整配置并重新检查。')
    from importlib.metadata import version
    if version('ultralytics') != '8.3.203':
        raise RuntimeError('Ultralytics 版本与已验证环境不一致。')
    # 实际执行一次 GPU 张量运算，避免只检测到驱动却不能执行内核。
    tensor = torch.ones((16, 16), device=f'cuda:{config["device"]}')
    if (tensor @ tensor).sum().item() != 4096:
        raise RuntimeError('CUDA 张量运算结果不符。')
    return config, dataset, weights, device


def training_scope(config, review):
    """正式训练的授权单独记录，不改写旧三轮试跑的复核文件。"""
    if config['schema_version'] == 2:
        return {'purpose': 'full_training', 'maximum_epochs': config['epochs'],
                'patience': config['patience'], 'authorization': config['authorization'],
                'note': '沿用已复核标签；用户追加授权长训练，样本规模和泛化限制继续保留。'}
    scope = review.get('training_scope', {})
    if config['epochs'] > scope.get('maximum_epochs', 0):
        raise ValueError('训练轮数超过原复核范围。')
    return scope


def preflight(config_path=TRAIN_CONFIG):
    require_training_ready(read_json(ROOT / 'configs/data_review.json'))
    result = environment_preflight(config_path)
    config, dataset, _, _ = result
    training_scope(config, read_json(ROOT / 'configs/data_review.json'))
    if config['schema_version'] == 2 and sha256(dataset / 'manifest.json') != config['dataset_manifest_sha256']:
        raise ValueError('正式训练的数据版本与固定清单摘要不一致。')
    validate_export(result[1])
    return result


def print_preflight(result):
    config, dataset, weights, device = result
    print('训练条件检查通过。', flush=True)
    print(f'解释器：{sys.executable}', flush=True)
    print(f'Ultralytics：8.3.203；权重：{weights}（SHA-256 {sha256(weights)}）', flush=True)
    print(f'数据集：{dataset}（数据清单 SHA-256 {sha256(dataset / "manifest.json")}）', flush=True)
    print(f'显卡：{device.name}，显存 {device.total_memory / 1024**3:.1f} GiB', flush=True)
    print(f'参数：YOLOv8n，{config["epochs"]} 轮，输入 {config["image_size"]}，批量 {config["batch_size"]}，工作进程 {config["workers"]}，AMP 关闭，数据缓存关闭。', flush=True)
    if config['schema_version'] == 2:
        print(f'本次为正式长训练，上限 {config["epochs"]} 轮；验证适应度连续 {config["patience"]} 轮未改善时早停。样本仍少，验证分数不能代替真实桌面验收。', flush=True)
    else:
        print('三轮只检查训练链路，不代表最终效果。', flush=True)
    print('独立测试集留待模型确定后再评估，避免反复看测试成绩来调参。', flush=True)


def run_training(result, config_path=TRAIN_CONFIG):
    if os.environ.get('DESKVISION_VISIBLE_TRAINING') != '1':
        raise RuntimeError('训练只能由打开可见训练终端的脚本启动。')
    config, dataset, weights, device = result
    os.environ['YOLO_OFFLINE'] = 'true'
    from ultralytics import YOLO

    prefix = 'desk8-full-' if config['schema_version'] == 2 else 'desk8-pilot-'
    run_name = prefix + datetime.now().strftime('%Y%m%d-%H%M%S')
    output = (ROOT / config['output_dir']).resolve()
    metadata_dir = ROOT / 'runs/train-metadata'
    metadata_path = metadata_dir / (run_name + '.json')
    metadata_dir.mkdir(parents=True, exist_ok=True)
    state = {
        'status': 'running',
        'run_name': run_name,
        'started_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'training_config': config,
        'training_config_path': config_path.relative_to(ROOT).as_posix(),
        'training_config_sha256': sha256(config_path),
        'dataset_manifest_sha256': sha256(dataset / 'manifest.json'),
        'dataset_yaml_sha256': sha256(dataset / 'data.yaml'),
        'data_review_sha256': sha256(ROOT / 'configs/data_review.json'),
        'training_scope': training_scope(config, read_json(ROOT / 'configs/data_review.json')),
        'generalization_limitations': read_json(ROOT / 'configs/data_review.json').get('generalization_limitations', []),
        'weights_sha256': sha256(weights),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'gpu': device.name,
        'epochs': [],
    }
    save_json(metadata_path, state)
    print(f'运行记录：{metadata_path}', flush=True)
    print('训练开始。Epoch 是完整遍历训练图像的一轮；GPU 进度行会持续刷新。', flush=True)
    def record_epoch(trainer):
        train_loss = [round(float(value), 5) for value in trainer.tloss.detach().cpu().tolist()]
        metrics = {}
        for key, value in trainer.metrics.items():
            try:
                metrics[str(key)] = round(float(value), 5)
            except (TypeError, ValueError):
                continue
        row = {'epoch': trainer.epoch + 1, 'train_loss': train_loss, 'validation_metrics': metrics}
        new_epoch = record_epoch_result(state, row)
        save_json(metadata_path, state)
        if new_epoch:
            print(f'本轮汇总：第 {row["epoch"]}/{config["epochs"]} 轮，loss={train_loss}，验证指标={metrics}', flush=True)
        else:
            print(f'最佳权重最终验证：{metrics}', flush=True)

    try:
        model = YOLO(str(weights))
        model.add_callback('on_fit_epoch_end', record_epoch)
        model.train(
            data=str(dataset / 'data.yaml'), epochs=config['epochs'], imgsz=config['image_size'],
            batch=config['batch_size'], workers=config['workers'], device=config['device'],
            amp=config['amp'], cache=config['cache'], seed=config['seed'],
            save_period=config['save_period'], project=str(output), name=run_name,
            exist_ok=False, plots=True, verbose=True,
            **({'patience': config['patience']} if config['schema_version'] == 2 else {}),
        )
    except (Exception, KeyboardInterrupt) as error:
        state['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        state['finished_at'] = datetime.now().astimezone().isoformat(timespec='seconds')
        state['error'] = f'{type(error).__name__}: {error}'
        state['traceback_tail'] = traceback.format_exc().splitlines()[-12:]
        save_json(metadata_path, state)
        raise
    run_dir = Path(model.trainer.save_dir).resolve()
    state['status'] = 'completed'
    state['finished_at'] = datetime.now().astimezone().isoformat(timespec='seconds')
    state['run_dir'] = str(run_dir)
    state['best_weights'] = str((run_dir / 'weights' / 'best.pt').resolve())
    state['last_weights'] = str((run_dir / 'weights' / 'last.pt').resolve())
    state['epochs_completed'] = len(state['epochs'])
    state['early_stopped'] = len(state['epochs']) < config['epochs']
    save_json(metadata_path, state)
    print(f'训练完成，共 {len(state["epochs"])} 轮；权重和曲线位于：{run_dir}', flush=True)
    print('现在只根据训练日志和验证集判断是否需要改数据或参数；正式测试集尚未使用。', flush=True)
    return run_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true', help='通过可见训练终端开始配置指定的训练')
    mode.add_argument('--environment-only', action='store_true', help='只检查配置、解释器、权重与 CUDA 运算，不检查数据，不训练')
    parser.add_argument('--config', default=str(TRAIN_CONFIG), help='训练配置；正式长训练使用 configs/training-full.yaml')
    parser.add_argument('--evaluate-after-training', action='store_true', help='训练成功后重新加载最佳权重做独立验证')
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.is_relative_to(ROOT / 'configs'):
        parser.error('训练配置必须位于项目 configs 目录内。')
    if args.evaluate_after_training and not args.run:
        parser.error('独立验证选项必须和 --run 一起使用。')
    if args.environment_only:
        config, dataset, weights, device = environment_preflight(config_path)
        print(f'环境检查通过：{sys.executable}；Ultralytics 8.3.203；{device.name}，{device.total_memory / 1024**3:.1f} GiB；CUDA 矩阵运算通过。', flush=True)
        print(f'本地权重：{weights}（SHA-256 {sha256(weights)}）', flush=True)
        print('数据尚未在本命令中验收，没有启动训练。', flush=True)
        return
    result = preflight(config_path)
    print_preflight(result)
    if args.run:
        run_name = run_training(result, config_path)
        if args.evaluate_after_training:
            from .evaluate import evaluate
            evaluate(run_name)
    else:
        print('这里只完成检查，没有启动训练。使用第四阶段可见终端脚本才会开始训练。', flush=True)


if __name__ == '__main__':
    main()
