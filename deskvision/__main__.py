"""图片检测与可选的 SAM 2 分割入口；每次保存独立结果，不覆盖原图。"""

import argparse
from datetime import datetime
from importlib import metadata
import json
from pathlib import Path
import sys
from time import perf_counter
from uuid import uuid4
from zoneinfo import ZoneInfo

from .config import ROOT, load_categories, local_sam2_weights, local_weights
from .detector import BaselineDetector, TrainedDetector, file_sha256, read_image
from .render import find_font, render_result, render_segmentation


def probability(value: str) -> float:
    """阈值端点不接收，避免无意抑制全部结果或保留无效分数。"""
    number = float(value)
    if not 0 < number < 1:
        raise argparse.ArgumentTypeError('阈值必须在 0 与 1 之间。')
    return number


def image_size(value: str) -> int:
    """明确约束尺寸，避免库静默调整后记录与运行参数不同。"""
    number = int(value)
    if not 32 <= number <= 2048 or number % 32:
        raise argparse.ArgumentTypeError('图片输入尺寸须为 32～2048 之间的 32 倍数。')
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description='使用本地 YOLOv8 检测图片，可选 SAM 2 分割，保存中文标注和八类状态。', add_help=False)
    parser._optionals.title = '可选参数'
    parser.add_argument('-h', '--help', action='help', help='显示帮助并退出')
    parser.add_argument('--image', required=True, help='输入图片路径，支持中文和空格')
    parser.add_argument('--mode', choices=['baseline', 'finetuned'], default='baseline', help='COCO 基线或八类微调权重，默认 baseline')
    parser.add_argument('--weights', help='已有 .pt 权重路径；微调模式必须显式指定')
    parser.add_argument('--output', help='输出根目录，每次创建独立子目录；默认按模式存入 runs')
    parser.add_argument('--conf', type=probability, default=0.25, help='检测分数阈值，默认 0.25')
    parser.add_argument('--iou', type=probability, default=0.7, help='跨类别重复框抑制阈值，默认 0.7')
    parser.add_argument('--imgsz', type=image_size, default=640, help='模型输入尺寸，默认 640')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda:0'], default='auto', help='计算设备，默认自动选择')
    parser.add_argument('--font', help='中文字体文件；Windows 默认查找微软雅黑')
    parser.add_argument('--segment', action='store_true', help='将检测框传给本地 SAM 2.1 tiny，另存实例掩膜和轮廓图')
    parser.add_argument('--sam2-weights', help='已有 SAM 2.1 tiny 权重；默认读取本机 sam2_weights')
    args = parser.parse_args()
    if args.sam2_weights and not args.segment:
        parser.error('--sam2-weights 必须与 --segment 一起使用。')
    run_dir = None
    report = None
    stage = 'input'
    try:
        source = Path(args.image).expanduser().resolve()
        started = perf_counter()
        image = read_image(source)
        decode_ms = (perf_counter() - started) * 1000
        if args.mode == 'finetuned' and not args.weights:
            raise ValueError('微调模式必须通过 --weights 显式选择八类权重。')
        weights = local_weights(args.weights)
        categories = load_categories()
        font_path = find_font(args.font)
        print('正在读取已有权重并执行检测；此命令不训练、不下载模型。', flush=True)
        detector_type = TrainedDetector if args.mode == 'finetuned' else BaselineDetector
        stage = 'detection'
        detector = detector_type(weights, categories, args.device)
        report = detector.predict(image, args.conf, args.iou, args.imgsz)
        timestamp = datetime.now(ZoneInfo('Asia/Shanghai'))
        report['created_at'] = timestamp.isoformat()
        report['input'] = {'file_name': source.name, 'sha256': file_sha256(source)}
        report['versions'] = {name: metadata.version(name) for name in ('torch', 'torchvision', 'ultralytics', 'opencv-python', 'Pillow')}
        report['category_config'] = {
            'sha256': file_sha256(ROOT / 'configs/classes.yaml'),
            'mapping': [{'id': item.id, 'key': item.key, 'name': item.name, 'coco_id': item.coco_id} for item in categories],
        }
        report['timings_ms']['image_decode'] = round(decode_ms, 3)
        output_root = (Path(args.output).expanduser().resolve() if args.output
                       else ROOT / 'runs' / ('segmentation' if args.segment else args.mode))
        run_dir = output_root / f"{timestamp:%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=False)
        stage = 'artifacts'
        report['artifacts'] = {'result': 'result.json'}
        render_result(image, report, font_path).save(run_dir / 'annotated.png')
        report['artifacts']['visualization'] = 'annotated.png'
        if args.segment:
            stage = 'segmentation'
            from .segmentation import Sam2Segmenter
            print('正在按原图检测框生成 SAM 2 掩膜；分类与计数沿用 YOLO。', flush=True)
            segmenter = Sam2Segmenter(local_sam2_weights(args.sam2_weights), detector.device)
            report = segmenter.segment(image, report, run_dir)
            report['versions']['SAM-2'] = report['segmentation']['package_version']
            stage = 'artifacts'
            render_segmentation(image, report, run_dir, font_path).save(run_dir / 'segmented.png')
            report['artifacts']['segmentation_visualization'] = 'segmented.png'
            info = report['segmentation']
            print(f'SAM 2：{info["nonempty_masks"]} 个非空掩膜，{info["empty_masks"]} 个空掩膜；状态 {info["status"]}。')
        stage = 'artifacts'
        (run_dir / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        for item in report['summary']:
            value = item['count'] if item['count'] is not None else item['status_text']
            print(f"{item['category_name']}：{value}")
        print(report['message'])
        print(f'结果目录：{run_dir}')
        return 0
    except Exception as error:
        print(f'处理失败（{stage}）：{error}', file=sys.stderr)
        if run_dir is not None:
            failed = dict(report or {}, status='failed', error={'stage': stage, 'type': type(error).__name__, 'message': str(error)})
            try:
                (run_dir / 'result.json').write_text(json.dumps(failed, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            except OSError:
                pass
            print(f'输出未完成，失败记录与部分产物位于：{run_dir}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
