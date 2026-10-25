"""冻结独立测试协议，逐图运行两路并汇总；不训练、不调参、不自动重试。"""

import argparse
from collections import Counter
from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
from statistics import median
from zoneinfo import ZoneInfo

from .config import ROOT, load_categories, local_sam2_weights
from .detector import file_sha256
from .vision_language import CLASS_KEYS, PROMPT_PATH, save_record

PROTOCOL = ROOT / 'reports/stage8-protocol.json'
OUTPUT = ROOT / 'runs/acceptance/stage8-v1'
SUMMARY = ROOT / 'reports/stage8-results.json'
WEIGHTS = ROOT / 'runs/train/desk8-full-20261023-204546/weights/best.pt'
DATASET = ROOT / 'data/desk8-train-v1'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             allow_nan=False).encode('utf-8')).hexdigest()


def runtime():
    return {'python': platform.python_version(), **{
        name: version(name) for name in ('torch', 'torchvision', 'ultralytics', 'SAM-2',
                                        'opencv-python', 'Pillow', 'numpy', 'openai', 'httpx', 'gradio')}}


def freeze():
    """必须在推理前执行；已存在时拒绝覆盖，避免看过答案后改试卷。"""
    if PROTOCOL.exists() or OUTPUT.exists():
        raise ValueError('协议或运行目录已存在，拒绝覆盖；请使用 run 或 summarize。')
    from .audit_data import validate_export
    from .vlm_config import load_vlm_settings

    validate_export(DATASET)
    categories = load_categories()
    samples = []
    for row in sorted(read(DATASET / 'manifest.json')['records'], key=lambda row: row['id']):
        if row['split'] != 'test':
            continue
        counts = Counter(int(box[0]) for box in row['boxes'])
        samples.append({key: row[key] for key in (
            'id', 'source', 'source_id', 'source_url', 'attribution', 'license', 'license_url',
            'image', 'image_sha256', 'label', 'label_sha256')} | {
                'truth': {c.key: counts[c.id] for c in categories}})
    if not samples:
        raise ValueError('测试组不能为空。')
    assets = [WEIGHTS, local_sam2_weights(None), DATASET / 'manifest.json',
              ROOT / 'configs/classes.yaml', PROMPT_PATH, *sorted((ROOT / 'deskvision').glob('*.py'))]
    protocol = {
        'schema_version': 1, 'created_at': now(), 'split': 'test',
        'purpose': '全部既有留出测试图的首次固定比较，非真实桌面泛化合格证明',
        'prior_exposure': '阶段三曾查看并标注；此前未推理评估，未用于训练、阈值或提示调整。云端模型预训练数据是否包含公开图未知。',
        'samples': samples, 'classes': list(CLASS_KEYS),
        'assets': [{'path': path.relative_to(ROOT).as_posix(), 'sha256': file_sha256(path)} for path in assets],
        'versions': runtime(), 'weights': WEIGHTS.relative_to(ROOT).as_posix(),
        'local_parameters': {'confidence': .25, 'iou': .7, 'image_size': 640,
                             'device': 'cuda:0', 'segment': True, 'class_agnostic_nms': True,
                             'half_precision': False, 'max_detections': 300},
        'vlm': load_vlm_settings().public(),
        'execution': '固定 id 排序，先逐图本地，再逐图 DeepSeek；每张本地重新加载模型。无自动重试，失败与中断保留。',
        'metrics': {
            'exact': '成功且八类计数向量与真值完全一致的图数/全部测试图数；另列成功图分母。',
            'mae': '成功图中八类绝对计数误差之和/(成功图数*8)，包括真实为零的类别。',
            'normalized_absolute_error': '成功图八类绝对误差总和/这些图真实实例总数；不是准确率，可大于1。',
            'failures': '失败、未运行、中断均无零计数，不参与MAE；始终列出成功覆盖率。',
            'timing': '本地单次总耗时含加载、输入校验、推理及图像产物；云端总耗时含请求及网络。均不含UI上传、排队、下载；不换算FPS。',
            'cost': '汇总API返回token用量；无已核实价格和账单，不估算金额；本地硬件电费未测。',
        },
        'limitations': ['仅9张、29个实例，前五类各仅1张1实例，五金仅4张固定工业背景。',
                        '缺真实桌面、遮挡、密集小物、负样本；人工标签由同一助手复核，仍需人工抽审。',
                        '无人工像素掩膜，不报分割精度；DeepSeek无像素框，不计算其mAP。',
                        '云端服务可能更新，固定请求不能保证未来答案逐字相同。'],
    }
    save_record(PROTOCOL, protocol)
    print('已冻结协议：' + file_sha256(PROTOCOL), flush=True)


def validate_protocol(protocol):
    for asset in protocol['assets']:
        if file_sha256(ROOT / asset['path']) != asset['sha256']:
            raise ValueError('冻结文件已改变：' + asset['path'])
    for sample in protocol['samples']:
        for field in ('image', 'label'):
            if file_sha256(ROOT / sample[field]) != sample[field + '_sha256']:
                raise ValueError('测试输入或真值已改变：' + sample['id'])
    if runtime() != protocol['versions']:
        raise ValueError('环境版本与冻结协议不一致。')


def checked_counts(report, image_sha):
    """只有同图成功、八类齐全的有效报告才能计分；失败绝不冒充零。"""
    if report.get('status') != 'ok':
        return None
    if report.get('input', {}).get('sha256') != image_sha:
        raise ValueError('结果输入指纹与测试图不一致。')
    rows = report.get('summary', [])
    counts = {row['category_key']: row['count'] for row in rows}
    if (len(rows) != len(CLASS_KEYS) or set(counts) != set(CLASS_KEYS)
            or any(type(value) is not int or value < 0 for value in counts.values())):
        raise ValueError('成功报告缺少有效八类计数。')
    return counts


def cached(path, protocol_sha, sample):
    record = read(path)
    if record['protocol_sha256'] != protocol_sha or record['sample_id'] != sample['id']:
        raise ValueError('断点结果属于其他协议或样本。')
    if record['state'] == 'finished':
        if digest(record['report']) != record['report_sha256']:
            raise ValueError('断点报告校验失败。')
        checked_counts(record['report'], sample['image_sha256'])
    return record


def run():
    protocol = read(PROTOCOL)
    validate_protocol(protocol)
    from .ui import run_vlm_route, run_yolo_route
    from .vlm_config import load_vlm_settings

    if load_vlm_settings().public() != protocol['vlm']:
        raise ValueError('视觉服务公开参数与冻结协议不一致。')
    protocol_sha = file_sha256(PROTOCOL)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for route in ('yolo', 'deepseek'):
        for sample in protocol['samples']:
            path = OUTPUT / (sample['id'] + '-' + route + '.json')
            if path.exists():
                cached(path, protocol_sha, sample)
                print(sample['id'], route, '保留已有记录（含失败/中断，不重试）', flush=True)
                continue
            if file_sha256(ROOT / sample['image']) != sample['image_sha256']:
                raise ValueError('调用前原图校验失败。')
            entry = {'protocol_sha256': protocol_sha, 'sample_id': sample['id'],
                     'state': 'started', 'started_at': now()}
            # 在外部请求之前落盘；进程中断后不自动重复付费或挑选更好的回答。
            save_record(path, entry)
            image = ROOT / sample['image']
            # 接口只接收图片及固定推理参数，真值仅留在离线评分步骤。
            if route == 'yolo':
                report = run_yolo_route(image, weights_path=ROOT / protocol['weights'],
                                        confidence=.25, iou=.7, image_size=640,
                                        device='cuda:0', segment=True, output_root=OUTPUT / 'artifacts')
            else:
                report = run_vlm_route(image, output_root=OUTPUT / 'artifacts')
            entry.update(state='finished', finished_at=now(), report=report, report_sha256=digest(report))
            save_record(path, entry)
            print(sample['id'], route, report['status'],
                  checked_counts(report, sample['image_sha256']), flush=True)
    summarize()


def aggregate(rows):
    successful = [row for row in rows if row['counts'] is not None]
    errors = {key: sum(abs(row['counts'][key] - row['truth'][key]) for row in successful)
              for key in CLASS_KEYS}
    exact = sum(row['counts'] == row['truth'] for row in successful)
    truth_total = sum(sum(row['truth'].values()) for row in successful)
    times = [row['timings_ms']['total_wall'] for row in successful]
    return {'images': len(rows), 'successful': len(successful), 'failed_or_missing': len(rows) - len(successful),
            'exact_images': exact, 'exact_rate_all': exact / len(rows) if rows else None,
            'exact_rate_successful': exact / len(successful) if successful else None,
            'absolute_error_per_class': errors, 'total_absolute_error': sum(errors.values()),
            'mae_per_image_category': sum(errors.values()) / (len(successful) * 8) if successful else None,
            'normalized_absolute_error': sum(errors.values()) / truth_total if truth_total else None,
            'truth_instances_on_successful': truth_total,
            'total_wall_ms': {'median': median(times), 'min': min(times), 'max': max(times)} if times else None,
            'usage': {key: sum(row.get('usage', {}).get(key) or 0 for row in rows)
                      for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
            'responses_with_usage': sum(bool(row.get('usage')) for row in rows)}


def summarize():
    protocol = read(PROTOCOL)
    validate_protocol(protocol)
    protocol_sha = file_sha256(PROTOCOL)
    routes = {}
    for route in ('yolo', 'deepseek'):
        rows = []
        for sample in protocol['samples']:
            path = OUTPUT / (sample['id'] + '-' + route + '.json')
            entry = cached(path, protocol_sha, sample) if path.exists() else {'state': 'not_run'}
            report = entry.get('report', {})
            row = {'id': sample['id'], 'image_sha256': sample['image_sha256'],
                   'truth': sample['truth'], 'status': report.get('status', entry['state']),
                   'counts': checked_counts(report, sample['image_sha256']),
                   'error': report.get('error'), 'timings_ms': report.get('timings_ms', {}),
                   'usage': report.get('usage', {}), 'report_sha256': entry.get('report_sha256')}
            if route == 'yolo':
                segment = report.get('segmentation', {})
                row['segmentation'] = {key: segment.get(key) for key in
                                       ('status', 'nonempty_masks', 'empty_masks')}
            else:
                row['returned_model'] = report.get('response', {}).get('returned_model')
            rows.append(row)
        routes[route] = {'metrics': aggregate(rows), 'samples': rows}
    summary = {'schema_version': 1, 'created_at': now(), 'protocol_sha256': protocol_sha,
               'routes': routes, 'limitations': protocol['limitations'],
               'cost_note': protocol['metrics']['cost']}
    save_record(SUMMARY, summary)
    for route, result in routes.items():
        print(route, json.dumps(result['metrics'], ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'run', 'summarize'))
    args = parser.parse_args()
    {'freeze': freeze, 'run': run, 'summarize': summarize}[args.command]()


if __name__ == '__main__':
    main()
