"""八类数据准备的共同规则；本模块不加载模型，也不启动训练。"""

import hashlib
import json
import math
from pathlib import Path

from .config import ROOT

SPLITS = ('train', 'val', 'test')


def sha256(path: Path) -> str:
    """分块计算校验值，避免一次读入大文件。"""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value) -> None:
    """先写临时文件，再替换目标，避免中断留下半份清单。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def validate_box(row: list[float]) -> list[float]:
    """拒绝错误类别、非有限数、零面积与越界框；不静默裁剪标签。"""
    if len(row) != 5:
        raise ValueError('标签每行必须有五项。')
    cls, x, y, width, height = row
    if not all(math.isfinite(v) for v in row):
        raise ValueError('标签含有非有限数。')
    if cls != int(cls) or not 0 <= cls < 8:
        raise ValueError('类别编号必须是0至7的整数。')
    if width <= 0 or height <= 0:
        raise ValueError('框面积必须为正。')
    if min(x - width / 2, y - height / 2) < -1e-5 or max(x + width / 2, y + height / 2) > 1 + 1e-5:
        raise ValueError('框超出原图归一化坐标范围。')
    return [int(cls), x, y, width, height]


def parse_yolo(text: str, mapping: dict[int, int] | None = None) -> list[list[float]]:
    """空文件不自动视为有效负样本，是否可用由独立复核决定。"""
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        row = [float(v) for v in line.split()]
        if len(row) != 5 or row[0] != int(row[0]):
            raise ValueError('标签格式或源类别编号错误。')
        if mapping is not None:
            row[0] = mapping[int(row[0])]
        rows.append(validate_box(row))
    return rows


def label_text(boxes: list[list[float]]) -> str:
    return ''.join(str(int(row[0])) + ' ' + ' '.join(f'{v:.8f}' for v in row[1:]) + '\n' for row in boxes)


def iou(a, b) -> float:
    """输入为中心形式，仅用于同一照片的标签去重候选。"""
    def corners(row):
        _, x, y, w, h = row
        return x - w / 2, y - h / 2, x + w / 2, y + h / 2
    ax, ay, ar, ab = corners(a)
    bx, by, br, bb = corners(b)
    intersection = max(0, min(ar, br) - max(ax, bx)) * max(0, min(ab, bb) - max(ay, by))
    return intersection / (a[3] * a[4] + b[3] * b[4] - intersection)


def openimages_boxes(objects: list, mapping: dict) -> tuple[list, list]:
    """保留同图全部已知目标；组框隔离；杯子父子类别高重叠框合并。"""
    boxes, flags, original = [], [], []
    for obj in objects:
        mid = obj['category']
        if mid not in mapping:
            continue
        attrs = obj.get('attributes', [])
        if any('group' in value.lower() for value in attrs):
            flags.append('存在组框，不能逐件计数')
            continue
        bb = obj['bounding_box']
        left, right = float(bb['xmin']), float(bb['xmax'])
        top, bottom = float(bb['ymin']), float(bb['ymax'])
        box = validate_box([mapping[mid], (left + right) / 2, (top + bottom) / 2, right - left, bottom - top])
        # 只合并来自不同源类别的同一杯子候选，避免删除同类别真实重叠物。
        if any(box[0] == 0 and old[0] == 0 and old_mid != mid and iou(box, old) >= 0.9
               for old, old_mid in zip(boxes, original)):
            continue
        if box not in boxes:
            boxes.append(box)
            original.append(mid)
        if attrs:
            flags.extend(attrs)
    return boxes, sorted(set(flags))


def split_group(group: str) -> str:
    """同一署名者整体分组，结果稳定；不声称可以找全相似场景。"""
    value = int(hashlib.sha256(group.encode('utf-8')).hexdigest()[:8], 16) % 100
    return 'train' if value < 70 else ('val' if value < 85 else 'test')


def require_isolated(records: list[dict]) -> None:
    """阻止相同内容、已知场景组跨用途出现。"""
    seen = {}
    ids = set()
    for record in records:
        if record['id'] in ids:
            raise ValueError('图片标识重复：' + record['id'])
        ids.add(record['id'])
        if record['split'] not in SPLITS:
            raise ValueError('用途划分无效。')
        for kind in ('image_sha256', 'pixel_sha256', 'group'):
            value = record.get(kind)
            if not value:
                raise ValueError('缺少隔离依据：' + kind)
            key = (kind, value)
            if key in seen and seen[key] != record['split']:
                raise ValueError(f'发现跨用途重复或同组：{record["id"]}，{kind}')
            seen[key] = record['split']


def resolve_data_path(relative: str) -> Path:
    """清单只允许引用本项目忽略的数据目录。"""
    path = (ROOT / relative).resolve()
    if not path.is_relative_to((ROOT / 'data').resolve()):
        raise ValueError('数据路径越过项目数据目录。')
    return path
