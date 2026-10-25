"""生成标注检查图与复核数据包，所有操作都不加载训练模型。"""

import argparse
from collections import Counter
import hashlib
import json
import os
import re
import shutil

from PIL import Image, ImageDraw, ImageFont
import yaml

from .config import ROOT, load_categories
from .data import (SPLITS, label_text, parse_yolo, read_json, require_isolated,
                   resolve_data_path, save_json, sha256, validate_box)
from .prepare_data import image_details
from .render import find_font

COLORS = ('#ff3030', '#00c0ff', '#ffe030', '#ff58e8', '#30ff80', '#ff9020', '#7090ff', '#20eeee')
MANIFEST = ROOT / 'data/candidates/manifest.json'
REVIEW = ROOT / 'configs/data_review.json'


def require_training_ready(review):
    """旧式单一布尔标志不够；实际就绪还要用复核后的样本重新核对。"""
    blockers = review.get('training_blockers', [])
    if not isinstance(blockers, list) or any(not isinstance(item, str) for item in blockers):
        raise ValueError('训练阻塞项必须是文字列表。')
    reasons = []
    if review.get('training_ready') is not True:
        reasons.extend(blockers or ['尚未显式批准训练数据。'])
    if blockers and review.get('training_ready') is True:
        reasons.append('训练就绪标志与未解决问题列表冲突。')
    if reasons:
        raise ValueError('数据集尚未获准训练：' + '；'.join(reasons))


def reviewed_label_sha256(boxes):
    """按导出时相同的小数精度计算人工复核标签的内容哈希。"""
    return hashlib.sha256(label_text(boxes).encode('utf-8')).hexdigest()


def check_training_coverage(records, review=None):
    """重新计算获准样本覆盖；清单文字或手工布尔值不能代替实际标签。"""
    review = read_json(REVIEW) if review is None else review
    approved = apply_reviews(records, review['records'], review.get('source_decisions', {}))
    require_isolated(approved)
    missing = []
    for category in load_categories():
        for split in SPLITS:
            count = sum(
                any(int(box[0]) == category.id for box in record['boxes'])
                for record in approved if record['split'] == split
            )
            if count == 0:
                missing.append(f'{category.name}/{split}')
    if missing:
        raise ValueError('通过复核的样本未覆盖：' + '、'.join(missing))
    return approved


def checked_records():
    """文件完整性先于统计，缺失标签不能当作零目标。"""
    manifest = read_json(MANIFEST)
    if manifest['source_config_sha256'] != sha256(ROOT / 'configs/data_sources.json'):
        raise ValueError('来源配置已改变，请先重新准备候选集。')
    records = manifest['records']
    require_isolated(records)
    for record in records:
        for kind in ('image', 'label'):
            path = resolve_data_path(record[kind])
            if not path.is_file() or sha256(path) != record[kind + '_sha256']:
                raise ValueError('候选文件缺失或已改变：' + record['id'] + '/' + kind)
        boxes = parse_yolo(resolve_data_path(record['label']).read_text(encoding='utf-8'))
        if label_text(boxes) != label_text(record['boxes']):
            raise ValueError('清单与标签不一致：' + record['id'])
        width, height, pixels = image_details(resolve_data_path(record['image']))
        if (width, height, pixels) != (record['width'], record['height'], record['pixel_sha256']):
            raise ValueError('图片尺寸或像素校验不符：' + record['id'])
    return records


def render_tile(record, size=(480, 360)):
    """序号对应图下清单；框用颜色和编号区分，不遮住小物品。"""
    names = load_categories()
    font = ImageFont.truetype(str(find_font(None)), 17)
    with Image.open(resolve_data_path(record['image'])) as source:
        image = source.convert('RGB')
    image.thumbnail((size[0], size[1] - 65))
    tile = Image.new('RGB', size, '#18212c')
    xoff, yoff = (size[0] - image.width) // 2, 0
    tile.paste(image, (xoff, yoff))
    draw = ImageDraw.Draw(tile)
    for index, (cls, x, y, width, height) in enumerate(record['boxes'], 1):
        box = (xoff + (x-width/2)*image.width, (y-height/2)*image.height,
               xoff + (x+width/2)*image.width, (y+height/2)*image.height)
        draw.rectangle(box, outline=COLORS[int(cls)], width=2)
        draw.text((box[0], max(0, box[1]-18)), str(index), font=font, fill=COLORS[int(cls)], stroke_width=1, stroke_fill='black')
    draw.text((5, size[1]-61), record['id'], font=font, fill='white')
    counts = Counter(int(box[0]) for box in record['boxes'])
    title = ' '.join(f'{names[key].name}{count}' for key, count in sorted(counts.items()))
    draw.text((5, size[1]-36), record['split'] + ' ' + title, font=font, fill='white')
    return tile


def sheets(records, source=None, cls=None, limit=None):
    selected = [r for r in records if (source is None or r['source'] == source)
                and (cls is None or any(int(box[0]) == cls for box in r['boxes']))]
    if limit:
        selected = selected[:limit]
    folder = ROOT / 'data/review' / f'{source or "all"}_{cls if cls is not None else "all"}'
    folder.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(selected), 9):
        page = Image.new('RGB', (1440, 1080), '#18212c')
        batch = selected[offset:offset+9]
        for index, record in enumerate(batch):
            page.paste(render_tile(record), ((index % 3) * 480, (index // 3) * 360))
        path = folder / f'{offset//9+1:03d}.jpg'
        page.save(path, quality=94)
        print(path, flush=True)
    save_json(folder / 'index.json', [{'page': i//9+1, 'position': i%9+1, 'id': r['id']} for i, r in enumerate(selected)])


def apply_reviews(records, reviews, decisions=None):
    """审批绑定候选哈希；改动后的标签需要新复核，不继承旧状态。"""
    result = []
    known = {r['id'] for r in records}
    if set(reviews) - known:
        raise ValueError('复核记录引用了未知图片。')
    if decisions is None:
        decisions = read_json(REVIEW).get('source_decisions', {})
    for record in records:
        review = reviews.get(record['id'])
        if not review or review['status'] != 'approved':
            continue
        if decisions.get(record['source'], {}).get('status') == 'quarantined':
            raise ValueError('来源已隔离，不能将其中样本导出：' + record['source'])
        for kind in ('image', 'label'):
            if review.get(kind + '_sha256') != record[kind + '_sha256']:
                raise ValueError('复核依据已失效：' + record['id'])
        if review.get('all_eight_classes_checked') is not True or not review.get('note', '').strip():
            raise ValueError('未确认八类完整性或缺少复核说明：' + record['id'])
        corrected = dict(record)
        corrected['review_note'] = review['note']
        final_boxes = review.get('boxes', record['boxes'])
        if record['source'] == 'loco':
            second = review.get('second_review')
            if not isinstance(second, dict) or second.get('status') != 'approved':
                continue
            if second.get('image_sha256') != record['image_sha256']:
                raise ValueError('第二轮复核图片依据已失效：' + record['id'])
            if second.get('all_eight_classes_checked') is not True or not second.get('note', '').strip():
                raise ValueError('第二轮复核未确认八类完整性或缺少说明：' + record['id'])
            final_boxes = second.get('boxes')
            if not isinstance(final_boxes, list):
                raise ValueError('第二轮复核缺少最终框：' + record['id'])
            validated_second = [validate_box(row) for row in final_boxes]
            if second.get('label_sha256') != reviewed_label_sha256(validated_second):
                raise ValueError('第二轮复核标签内容哈希不符：' + record['id'])
            corrected['review_note'] = review['note'] + '；第二轮复核：' + second['note']
        corrected['boxes'] = [validate_box(row) for row in final_boxes]
        if not corrected['boxes'] and review.get('verified_negative') is not True:
            raise ValueError('空标签必须显式确认负样本。')
        corrected['review_status'] = 'approved'
        corrected['split'] = review.get('split', record['split'])
        if record['source'] in ('fasteners', 'loco') and corrected['split'] != record['source_split']:
            raise ValueError('不能改变五金样本源划分。')
        corrected['group'] = review.get('group', record['group'])
        result.append(corrected)
    require_isolated(result)
    return result


def summarize(records):
    rows = []
    for category in load_categories():
        row = {'id': category.id, 'name': category.name}
        for split in SPLITS:
            subset = [r for r in records if r['split'] == split]
            row[split] = {'images': sum(any(b[0] == category.id for b in r['boxes']) for r in subset),
                          'instances': sum(b[0] == category.id for r in subset for b in r['boxes']),
                          'real_images': sum(r['domain'] == 'real' and any(b[0] == category.id for b in r['boxes']) for r in subset)}
        sizes = [min(b[3]*r['width'], b[4]*r['height']) * min(640/r['width'], 640/r['height'])
                 for r in records for b in r['boxes'] if b[0] == category.id]
        row['short_side_under_16_at_640'] = sum(size < 16 for size in sizes)
        rows.append(row)
    return {'images': len(records), 'source_images': dict(Counter(r['source'] for r in records)),
            'split_images': dict(Counter(r['split'] for r in records)), 'classes': rows}


def duplicate_candidates(records):
    """差异哈希只产生待复核线索，不能证明图片独立或重复。"""
    hashes = []
    for record in records:
        with Image.open(resolve_data_path(record['image'])) as image:
            pixels = list(image.convert('L').resize((9, 8)).getdata())
        value = 0
        for y in range(8):
            for x in range(8):
                value = value << 1 | int(pixels[y*9+x] > pixels[y*9+x+1])
        hashes.append(value)
    pairs = []
    for i, a in enumerate(records):
        for j in range(i+1, len(records)):
            b = records[j]
            if a['split'] == b['split'] or a['source'] != b['source']:
                continue
            distance = (hashes[i] ^ hashes[j]).bit_count()
            if distance <= 5:
                pairs.append({'a': a['id'], 'b': b['id'], 'distance': distance})
    return pairs


def build(records, name, *, draft=False):
    if not re.fullmatch(r'[a-z][a-z0-9-]{0,39}', name):
        raise ValueError('数据版本名只能使用小写字母、数字和连字符。')
    review = read_json(REVIEW)
    if not draft:
        require_training_ready(review)
    approved = check_training_coverage(records, review)
    if not approved:
        raise ValueError('没有通过复核的样本，不能导出训练数据。')
    duplicates = [pair for pair in duplicate_candidates(approved)
                  if next(r for r in approved if r['id'] == pair['a'])['source'] == 'openimages']
    if duplicates:
        raise ValueError('训练集与验证/测试集仍有 Open Images 相似照片候选，先调整复核分组：' + str(duplicates))
    destination = ROOT / 'data' / name
    if destination.exists():
        raise FileExistsError('版本目录已经存在；检查旧结果，使用新的版本名，不能静默混入旧文件。')
    destination.mkdir(parents=True)
    for record in approved:
        split, identifier = record['split'], record['id']
        image = destination / 'images' / split / (identifier + resolve_data_path(record['image']).suffix.lower())
        label = destination / 'labels' / split / (identifier + '.txt')
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(resolve_data_path(record['image']), image)
        except OSError:
            shutil.copy2(resolve_data_path(record['image']), image)
        label.write_text(label_text(record['boxes']), encoding='utf-8')
        record['candidate_image'] = record['image']
        record['candidate_label'] = record['label']
        record['image'] = image.relative_to(ROOT).as_posix()
        record['label'] = label.relative_to(ROOT).as_posix()
        record['label_sha256'] = sha256(label)
    config = {'path': destination.resolve().as_posix(), 'train': 'images/train', 'val': 'images/val',
              'test': 'images/test', 'names': {c.id: c.key for c in load_categories()}}
    (destination / 'data.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    provenance = {
        'schema_version': 2,
        'purpose': 'review' if draft else 'training',
        'review_sha256': sha256(REVIEW),
        'source_config_sha256': sha256(ROOT / 'configs/data_sources.json'),
        'class_config_sha256': sha256(ROOT / 'configs/classes.yaml'),
        'records': approved,
    }
    provenance['dataset_sha256'] = hashlib.sha256(
        json.dumps(approved, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()
    save_json(destination / 'manifest.json', provenance)
    save_json(destination / 'summary.json', summarize(approved))
    validate_export(destination, require_ready=not draft)
    print(('复核草稿已导出（不可训练）：' if draft else '训练数据已导出：') + str(destination), flush=True)


def compare_reviewed_records(exported, approved):
    """将导出标签与当前复核结果核对，不能只相信包内自报哈希。"""
    expected = {record['id']: record for record in approved}
    if len(exported) != len(expected) or {record['id'] for record in exported} != set(expected):
        raise ValueError('导出样本集合与当前复核结果不一致。')
    for record in exported:
        current = expected[record['id']]
        fields = ('source', 'split', 'group', 'image_sha256', 'pixel_sha256', 'width', 'height')
        if any(record.get(key) != current.get(key) for key in fields):
            raise ValueError('导出样本属性与当前复核不一致：' + record['id'])
        if (record.get('candidate_image') != current['image']
                or record.get('candidate_label') != current['label']
                or label_text(record['boxes']) != label_text(current['boxes'])):
            raise ValueError('导出标签与当前复核不一致：' + record['id'])


def exported_label_files(destination):
    """YOLO 会在 labels 根目录生成三个划分缓存，它们不是实例标注。"""
    folder = destination / 'labels'
    caches = {folder / (split + '.cache') for split in SPLITS}
    return {path.resolve() for path in folder.rglob('*') if path.is_file() and path not in caches}


def validate_export(destination, *, require_ready=True):
    """从导出目录独立核对清单、文件哈希、标签解析和类别覆盖。"""
    manifest_path = destination / 'manifest.json'
    manifest = read_json(manifest_path)
    if manifest.get('schema_version') != 2 or manifest.get('purpose') not in ('review', 'training'):
        raise ValueError('旧版或未知用途数据包需要重新导出。')
    review = read_json(REVIEW)
    if require_ready:
        require_training_ready(review)
        if manifest['purpose'] != 'training':
            raise ValueError('复核草稿不可用于训练，请导出正式训练版本。')
    for key, path in (
        ('review_sha256', REVIEW),
        ('source_config_sha256', ROOT / 'configs/data_sources.json'),
        ('class_config_sha256', ROOT / 'configs/classes.yaml'),
    ):
        if manifest.get(key) != sha256(path):
            raise ValueError('数据包来源记录与当前配置不一致：' + key)
    records = manifest['records']
    require_isolated(records)
    compare_reviewed_records(records, check_training_coverage(checked_records(), review))
    listed_images, listed_labels = set(), set()
    for record in records:
        image = resolve_data_path(record['image'])
        label = resolve_data_path(record['label'])
        if image.parent.parent.parent != destination.resolve() or label.parent.parent.parent != destination.resolve():
            raise ValueError('导出清单引用了版本目录之外的文件：' + record['id'])
        if (image.parent != destination.resolve() / 'images' / record['split']
                or label.parent != destination.resolve() / 'labels' / record['split']
                or image.stem != record['id'] or label.name != record['id'] + '.txt'):
            raise ValueError('导出文件位置与清单划分不符：' + record['id'])
        if not image.is_file() or not label.is_file():
            raise ValueError('导出文件缺失：' + record['id'])
        if sha256(image) != record['image_sha256'] or sha256(label) != record['label_sha256']:
            raise ValueError('导出文件哈希不符：' + record['id'])
        with Image.open(image) as decoded:
            decoded.verify()
        rows = parse_yolo(label.read_text(encoding='utf-8'))
        if label_text(rows) != label_text(record['boxes']):
            raise ValueError('导出标签与复核框不一致：' + record['id'])
        listed_images.add(image.resolve())
        listed_labels.add(label.resolve())
    actual_images = {p.resolve() for p in (destination / 'images').rglob('*') if p.is_file()}
    actual_labels = exported_label_files(destination)
    if actual_images != listed_images or actual_labels != listed_labels:
        raise ValueError('导出目录包含未登记或缺失的图像/标签文件。')
    yaml_path = destination / 'data.yaml'
    if not yaml_path.is_file():
        raise ValueError('数据包缺少 YOLO 配置。')
    config = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
    expected_config = {
        'path': destination.resolve().as_posix(),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {c.id: c.key for c in load_categories()},
    }
    if config != expected_config:
        raise ValueError('YOLO 配置类别顺序不符。')
    missing = [(category.id, split) for category in load_categories() for split in SPLITS
               if not any(record['split'] == split and any(int(box[0]) == category.id for box in record['boxes'])
                          for record in records)]
    if missing:
        raise ValueError('导出集缺少类别划分覆盖：' + str(missing))
    expected = hashlib.sha256(json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    if manifest.get('dataset_sha256') != expected:
        raise ValueError('数据包清单自身校验失败。')
    print('独立导出核验通过：', len(records), '张图像，清单、像素文件和标签均有效。', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    view = sub.add_parser('sheets', help='生成候选图或复核图联系表')
    view.add_argument('--source', choices=['openimages', 'fasteners', 'loco'])
    view.add_argument('--class-id', type=int, choices=range(8))
    view.add_argument('--limit', type=int)
    view.add_argument('--approved', action='store_true', help='仅查看已通过的最终标签')
    export = sub.add_parser('build', help='只导出复核通过的数据，不训练')
    export.add_argument('--name', default='desk8-v1')
    export.add_argument('--draft', action='store_true', help='仅生成明确不可训练的复核草稿')
    verify = sub.add_parser('verify-export', help='独立核对已经导出的数据版本')
    verify.add_argument('--name', default='desk8-v1')
    verify.add_argument('--draft', action='store_true', help='只核验内容，不批准进入训练')
    sub.add_parser('check', help='核对候选文件并生成统计与相似图线索')
    args = parser.parse_args()
    records = checked_records()
    if args.command == 'sheets':
        if args.approved:
            records = apply_reviews(records, read_json(REVIEW)['records'])
        sheets(records, args.source, args.class_id, args.limit)
    elif args.command == 'build':
        build(records, args.name, draft=args.draft)
    elif args.command == 'verify-export':
        validate_export(resolve_data_path('data/' + args.name), require_ready=not args.draft)
    else:
        summary = summarize(records)
        summary['near_duplicate_candidates'] = duplicate_candidates(records)
        save_json(ROOT / 'data/candidates/audit.json', summary)
        review = read_json(REVIEW)
        approved = apply_reviews(records, review['records'])
        save_json(ROOT / 'reports/data-preparation.json', {
            'schema_version': 1,
            'candidate_manifest_sha256': sha256(MANIFEST),
            'review_sha256': sha256(REVIEW),
            'training_ready': review['training_ready'],
            'training_scope': review.get('training_scope'),
            'generalization_limitations': review.get('generalization_limitations', []),
            'training_blockers': review.get('training_blockers', []),
            'candidate_images': len(records),
            'source_decisions': review.get('source_decisions', {}),
            'approved': summarize(approved),
            'approved_cross_split_similarity_candidates': duplicate_candidates(approved),
        })
        print('候选文件完整性通过，统计已保存；复核状态不因此变为通过。')
        print(summary['images'], '张；跨用途相似图线索', len(summary['near_duplicate_candidates']))


if __name__ == '__main__':
    main()
