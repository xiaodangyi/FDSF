"""按固定来源准备候选数据，下载原图、标签与署名；不训练。"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import html
import re
import time
from urllib.parse import quote

from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import ROOT
from .data import (label_text, openimages_boxes, parse_yolo, read_json,
                   require_isolated, save_json, sha256, split_group)

RAW = ROOT / 'data/raw'
CANDIDATES = ROOT / 'data/candidates'


def session():
    client = requests.Session()
    client.mount('https://', HTTPAdapter(max_retries=Retry(
        total=3, backoff_factor=0.8, status_forcelist=[429, 500, 502, 503, 504])))
    return client


def download(url, path, expected_hash=None, expected_size=None):
    """已有缓存校验后复用；中断文件不当成成功结果。"""
    metadata = path.with_suffix(path.suffix + '.download.json')
    if path.is_file() and metadata.is_file():
        previous = read_json(metadata)
        digest = sha256(path)
        if (previous.get('url') == url and previous.get('sha256') == digest
                and (expected_hash is None or expected_hash == digest)
                and (expected_size is None or path.stat().st_size == expected_size)):
            return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    with session() as client, client.get(url, stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        with temporary.open('wb') as stream:
            for block in response.iter_content(256 * 1024):
                stream.write(block)
    digest = sha256(temporary)
    if expected_hash and digest != expected_hash:
        raise ValueError('下载内容哈希不符：' + str(path))
    if expected_size is not None and temporary.stat().st_size != expected_size:
        raise ValueError('下载大小不符：' + str(path))
    temporary.replace(path)
    save_json(metadata, {'url': url, 'sha256': digest, 'bytes': path.stat().st_size})
    return path


def relative(path):
    return path.relative_to(ROOT).as_posix()


def image_details(path):
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        rgb = image.convert('RGB')
        digest = hashlib.sha256(f'{width},{height}:'.encode() + rgb.tobytes()).hexdigest()
    return width, height, digest


def write_candidate(record, boxes, image):
    width, height, pixels = image_details(image)
    label = CANDIDATES / 'labels' / (record['id'] + '.txt')
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text(label_text(boxes), encoding='utf-8')
    record.update(image=relative(image), label=relative(label), width=width, height=height,
                  image_sha256=sha256(image), pixel_sha256=pixels,
                  label_sha256=sha256(label), boxes=boxes, review_status='pending')
    return record


def prepare_fasteners(config, workers):
    repo, revision = config['repository'], config['revision']
    index = RAW / 'fasteners/tree.json'
    url = f'https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true&limit=1000'
    download(url, index)
    entries = read_json(index)
    # 本次固定版本只有916项；将来若超过一页，明确失败而不是漏下载。
    if len(entries) >= 1000:
        raise ValueError('来源目录需要分页，必须先更新下载器。')
    files = {entry['path']: entry for entry in entries if entry['type'] == 'file'}
    names = [name for name in files if name.startswith(('images/', 'labels/', 'meta/'))
             or name in ('README.md', 'data.yaml')]

    def task(name):
        entry = files[name]
        return download(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(name)}',
                        RAW / 'fasteners' / name, entry.get('lfs', {}).get('oid'), entry['size'])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(task, name) for name in names]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 100 == 0 or i == len(futures):
                print(f'五金文件已校验 {i}/{len(futures)}', flush=True)
    records = []
    mapping = {int(key): value for key, value in config['class_map'].items()}
    for name in sorted(files):
        if not name.startswith('images/') or not name.endswith('.jpg'):
            continue
        _, split, filename = name.split('/')
        stem = filename[:-4]
        path = RAW / 'fasteners' / name
        source_label = RAW / 'fasteners/labels' / split / (stem + '.txt')
        boxes = parse_yolo(source_label.read_text(encoding='utf-8'), mapping)
        if not boxes:
            raise ValueError('五金样本没有标签：' + name)
        record = dict(id='qm_' + stem, source='fasteners', domain='synthetic',
                      source_id=stem, source_split=split, split=split,
                      group='qm_scene_' + stem, revision=revision,
                      source_url=f'https://huggingface.co/datasets/{repo}/blob/{revision}/{name}',
                      attribution='quailrobot / QM Synthetic Industrial Fasteners',
                      license=config['license'], license_url=config['license_url'],
                      source_label=relative(source_label), source_label_sha256=sha256(source_label),
                      flags=['合成图，尚未确认可见框符合本项目定义'])
        records.append(write_candidate(record, boxes, path))
    return records


def prepare_loco(config, workers):
    """仅下载固定列表的真实五金原图，原异常标签不参与检测转换。"""
    repo, revision = config['repository'], config['revision']

    def task(source_split, number):
        source = f'screw_bag/{source_split}/good/{number:03d}.png'
        path = RAW / 'loco' / source
        download(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{source}', path)
        split = 'val' if source_split == 'validation' else source_split
        record = dict(id=f'loco_{split}_{number:03d}', source='loco', domain='real',
                      source_id=source, source_split=split, split=split,
                      group='loco_source_' + split, revision=revision,
                      source_url=config['upstream_url'],
                      download_url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{source}',
                      attribution='MVTec Software GmbH / MVTec LOCO AD',
                      license=config['license'], license_url=config['license_url'],
                      flags=['透明袋固定工业场景，需本项目逐件新标框'])
        return write_candidate(record, [], path)

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(task, split, number) for split, numbers in config['samples'].items() for number in numbers]
        for future in as_completed(futures):
            records.append(future.result())
    print(f'真实五金已下载 {len(records)} 张，等待逐件新标框。', flush=True)
    return records


def attribution(image):
    value = image.get('footnote_bottom_right', '')
    links = re.findall(r"href=['\"](.*?)['\"]", value)
    author = html.unescape(re.sub('<[^>]+>', '', value)).strip()
    author_url = next((link for link in links if 'flickr.com/people/' in link), '')
    license_url = next((link for link in links if 'creativecommons.org/licenses/' in link), '')
    return author, author_url, license_url


def prepare_openimages(config, workers):
    mapping = config['class_map']
    combined = {}
    for mid in mapping:
        filename = mid.replace('/', '_') + '.json'
        path = RAW / 'openimages/catalog' / filename
        download(config['catalog_base'] + filename, path)
        data = read_json(path)
        print(f'官方候选目录 {mid}：{len(data)} 张', flush=True)
        for image_id, value in data.items():
            if image_id not in combined:
                combined[image_id] = value
            else:
                for obj in value.get('objects', []):
                    if obj not in combined[image_id]['objects']:
                        combined[image_id]['objects'].append(obj)
    candidates = {}
    rejected = []
    for image_id, value in combined.items():
        image = value['image']
        author, author_url, license_url = attribution(image)
        boxes, flags = openimages_boxes(value['objects'], mapping)
        reason = None
        if not boxes:
            reason = '没有可转换的独立目标框'
        elif not author_url or license_url != config['image_license_required']:
            reason = '署名或逐图许可不满足当前下载策略'
        elif float(image.get('rotation', 0)) != 0:
            reason = '非零旋转需单独核对坐标'
        elif '存在组框，不能逐件计数' in flags:
            reason = '含组框，须另行完整复核'
        if reason:
            rejected.append({'source_id': image_id, 'reason': reason})
            continue
        candidates[image_id] = (value, boxes, flags, author, author_url, license_url)
    # 固定哈希排序，不用图像增强填充稀缺类别；同作者在所有类别中共用组。
    chosen = set()
    for cls, limit in sorted(config['image_limits'].items(), key=lambda pair: int(pair[0]), reverse=True):
        ids = [key for key, value in candidates.items() if any(row[0] == int(cls) for row in value[1])]
        chosen.update(sorted(ids, key=lambda key: hashlib.sha256(('deskvision-v1:' + key).encode()).hexdigest())[:limit])

    def task(image_id):
        value, boxes, flags, author, author_url, license_url = candidates[image_id]
        url = config['image_base'] + image_id + '.jpg'
        path = RAW / 'openimages/images' / (image_id + '.jpg')
        download(url, path)
        source = RAW / 'openimages/selected' / (image_id + '.json')
        save_json(source, value)
        group = 'oi_author_' + hashlib.sha256(author_url.encode()).hexdigest()[:20]
        record = dict(id='oi_' + image_id, source='openimages', domain='real',
                      source_id=image_id, source_split=config['source_split'],
                      split=split_group(group), group=group, source_url=value['image']['url_full_res'],
                      download_url=url, attribution=author, author_url=author_url,
                      license='CC BY 2.0', license_url=license_url,
                      annotation_license=config['annotation_license'],
                      source_metadata=relative(source), source_metadata_sha256=sha256(source),
                      flags=flags + ['八类完整性与项目定义待逐图复核'])
        return write_candidate(record, boxes, path)

    records, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(task, key): key for key in sorted(chosen)}
        for i, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            try:
                records.append(future.result())
            except (requests.RequestException, ValueError, OSError) as error:
                errors.append({'source_id': key, 'error': str(error)})
            if i % 25 == 0 or i == len(futures):
                print(f'真实图片已处理 {i}/{len(futures)}，失败 {len(errors)}', flush=True)
    save_json(CANDIDATES / 'source_rejections.json', rejected)
    save_json(CANDIDATES / 'download_errors.json', errors)
    if errors:
        raise RuntimeError('下载有失败，记录在 download_errors.json；修复后重跑，不发布半份清单。')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['all', 'fasteners', 'openimages', 'loco'], default='all', help='选择数据来源')
    parser.add_argument('--workers', type=int, default=6, choices=range(1, 13), help='并发下载数')
    args = parser.parse_args()
    config_path = ROOT / 'configs/data_sources.json'
    config = read_json(config_path)
    manifest_path = CANDIDATES / 'manifest.json'
    previous = read_json(manifest_path)['records'] if manifest_path.is_file() else []
    records = previous if args.source != 'all' else []
    started = time.monotonic()
    for source in ('fasteners', 'openimages', 'loco'):
        if args.source in ('all', source):
            prepare = {'fasteners': prepare_fasteners, 'openimages': prepare_openimages, 'loco': prepare_loco}[source]
            fresh = prepare(config[source], args.workers)
            records = [record for record in records if record['source'] != source] + fresh
    records.sort(key=lambda record: record['id'])
    require_isolated(records)
    save_json(manifest_path, dict(schema_version=1, source_config_sha256=sha256(config_path), records=records))
    print(f'已保存候选清单 {len(records)} 张，耗时 {time.monotonic()-started:.1f} 秒。尚未批准进入训练。', flush=True)


if __name__ == '__main__':
    main()
