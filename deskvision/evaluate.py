"""核验已完成训练的最佳权重，只在验证集导出指标，不触碰独立测试集。"""

import argparse
import csv
import os
import re

from .audit_data import validate_export
from .config import ROOT, load_categories
from .data import read_json, resolve_data_path, save_json, sha256
from .detector import TrainedDetector


def evaluate(run_name):
    if not re.fullmatch(r'desk8-(pilot|full)-[0-9]{8}-[0-9]{6}', run_name):
        raise ValueError('请选择已有的 desk8-pilot 或 desk8-full 运行标识。')
    metadata_path = ROOT / 'runs/train-metadata' / (run_name + '.json')
    state = read_json(metadata_path)
    count = len(state['epochs'])
    if (state['status'] != 'completed' or not 0 < count <= state['training_config']['epochs']
            or [row['epoch'] for row in state['epochs']] != list(range(1, count + 1))):
        raise ValueError('只能评估训练完成且逐轮记录完整的运行。')
    dataset = resolve_data_path(state['training_config']['dataset'])
    validate_export(dataset)
    if sha256(dataset / 'manifest.json') != state['dataset_manifest_sha256']:
        raise ValueError('训练时的数据清单与当前版本不一致。')
    run_dir = ROOT / 'runs/train' / run_name
    weights = run_dir / 'weights/best.pt'
    last_weights = run_dir / 'weights/last.pt'
    if not weights.is_file() or not last_weights.is_file():
        raise FileNotFoundError('训练缺少最佳或最后权重。')
    categories = load_categories()
    detector = TrainedDetector(weights, categories, 'cuda:0')
    os.environ['YOLO_OFFLINE'] = 'true'
    metrics = detector.model.val(
        data=str(dataset / 'data.yaml'), split='val', imgsz=640, batch=4,
        device=0, workers=0, half=False, conf=.001, iou=.7, max_det=300,
        plots=False, save_json=False, project=str(ROOT / 'runs/validation'),
        name=run_name, exist_ok=False, verbose=True,
    )
    with (run_dir / 'results.csv').open(encoding='utf-8', newline='') as stream:
        epochs = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    validation_records = [row for row in read_json(dataset / 'manifest.json')['records'] if row['split'] == 'val']
    report = {
        'schema_version': 1, 'run_name': run_name,
        'purpose': '长训练后的独立验证，不是最终桌面验收' if run_name.startswith('desk8-full-') else '三轮链路试跑后的验证，不是最终验收',
        'split': 'val', 'test_evaluated': False,
        'training_scope': state.get('training_scope'),
        'generalization_limitations': state.get('generalization_limitations', []),
        'training_config': state['training_config'],
        'dataset_manifest_sha256': state['dataset_manifest_sha256'],
        'best_weights': weights.relative_to(ROOT).as_posix(), 'best_weights_sha256': sha256(weights),
        'last_weights': last_weights.relative_to(ROOT).as_posix(), 'last_weights_sha256': sha256(last_weights),
        'model_names': detector.model.names,
        'epochs_from_csv': epochs,
        'training_final_validation_metrics': state.get('best_checkpoint_validation_metrics'),
        'validation': {'images': len(validation_records),
                       'instances': sum(len(row['boxes']) for row in validation_records),
                       'batch_size': 4, 'half_precision': False, 'image_size': 640,
                       'confidence_floor': .001,
                       'nms_iou': .7, 'class_agnostic_nms': False,
                       'metrics': {key: float(value) for key, value in metrics.results_dict.items()},
                       'per_class': [{key: value.item() if hasattr(value, 'item') else value
                                      for key, value in row.items()}
                                     for row in metrics.summary(decimals=6)]},
        'interpretation': '本报告采用保存权重、FP32、batch 4 的独立验证协议；训练内置最终验证指标另列，验证批处理设置不同，数值不完全一致。以 0.001 分数下限收集预测后计算阈值曲线，不能当作默认 0.25 推理阈值的计数成绩。',
    }
    destination = ROOT / 'reports' / (run_name + '-validation.json')
    save_json(destination, report)
    print('验证报告：' + str(destination), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True, help='已完成训练的运行标识，包括验证早停的长训练')
    args = parser.parse_args()
    evaluate(args.run_name)


if __name__ == '__main__':
    main()
