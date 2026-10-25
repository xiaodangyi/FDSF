"""离线核对并打包第八阶段证据；不推理、不调用 API、不包含本地配置。"""

from datetime import datetime
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / 'runs/delivery'
ARCHIVE = DELIVERY / '验收证据与推理资源.zip'
UI_EXPORT = 'runs/ui/exports/20261024-105835-f1b45560/双路识别结果.zip'


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def digest(path):
    result = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def local_secrets():
    # 只在内存中作精确匹配，绝不打印或写入密钥。
    values = []
    for line in (ROOT / '.local/deepseek.env').read_text(encoding='utf-8-sig').splitlines():
        name, separator, value = line.partition('=')
        if separator and name.strip() == 'DEEPSEEK_API_KEY':
            value = value.strip().strip('\"\'')
            if value:
                values.append(value.encode())
    if not values:
        raise RuntimeError('本地密钥缺失，无法完成精确泄漏扫描。')
    return values


def scan_bytes(content, name, secrets):
    if any(secret in content for secret in secrets):
        raise RuntimeError('密钥泄漏扫描未通过：' + name)
    if name.endswith('.zip'):
        with zipfile.ZipFile(BytesIO(content)) as nested:
            for entry in nested.infolist():
                scan_bytes(nested.read(entry), name + '/' + entry.filename, secrets)


def main():
    protocol = read('reports/stage8-protocol.json')
    protocol_sha = digest(ROOT / 'reports/stage8-protocol.json')
    if protocol_sha != 'b268185811c39f10a456b1bcefeb7f531a00cc9c2425fa20087c0f34a69a8da5':
        raise RuntimeError('首次固定实验协议发生变化。')
    names = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT
    ).decode('utf-8').split('\0')
    allowed_roots = {'deskvision', 'docs', 'environment', 'tests', 'configs', 'reports', 'notes'}
    allowed_suffixes = {'.py', '.md', '.json', '.yaml', '.txt', '.ps1', '.png'}
    files = {name for name in names if name and (
        name in {'README.md', '.gitignore', '.gitattributes'} or (
            Path(name).parts[0] in allowed_roots and Path(name).suffix in allowed_suffixes))}
    secrets = local_secrets()
    missing_links = []
    for name in sorted(files):
        content = (ROOT / name).read_bytes()
        scan_bytes(content, name, secrets)
        if name.endswith('.md'):
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', content.decode('utf-8')):
                target = target.strip('<>').split('#')[0]
                if not target or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target):
                    continue
                linked = (ROOT / name).parent.joinpath(unquote(target)).resolve()
                if linked == ROOT / 'reports/stage8-checks.json':
                    continue  # 本次检查通过后生成，并收进同一交付包。
                if not linked.exists():
                    missing_links.append({'file': name, 'target': target})
    if missing_links:
        raise RuntimeError('文档链接不存在：' + json.dumps(missing_links, ensure_ascii=False))

    for asset in protocol['assets']:
        if digest(ROOT / asset['path']) != asset['sha256']:
            raise RuntimeError('冻结资产改变：' + asset['path'])
        files.add(asset['path'])
    for sample in protocol['samples']:
        for field in ('image', 'label'):
            if digest(ROOT / sample[field]) != sample[field + '_sha256']:
                raise RuntimeError('冻结样本改变：' + sample['id'])
            files.add(sample[field])
    for folder in ('runs/acceptance/stage8-v1', 'runs/stage8-demo'):
        files.update(path.relative_to(ROOT).as_posix() for path in (ROOT / folder).rglob('*')
                     if path.is_file() and path.suffix in {'.json', '.txt', '.png', '.jpg', '.mp4', '.npy'})
    for kind in ('images', 'labels'):
        suffix = 'jpg' if kind == 'images' else 'txt'
        files.add(f'data/desk8-train-v1/{kind}/val/oi_38451ec391f20ab5.{suffix}')
    files.add(UI_EXPORT)
    with zipfile.ZipFile(ROOT / UI_EXPORT) as exported:
        if exported.testzip() is not None or len(exported.infolist()) != 11:
            raise RuntimeError('界面导出完整性检查失败。')
        ui_entries = exported.namelist()
        for entry in ui_entries:
            scan_bytes(exported.read(entry), entry, secrets)
        route_reports = {route: json.loads(exported.read(f'{route}/result.json'))
                         for route in ('yolo', 'deepseek')}
    expected_image = '8ef910f0fc1047daf64d7e4bb228805c158b5866b55286df44637465ae55a01c'
    if any(row['input']['sha256'] != expected_image for row in route_reports.values()):
        raise RuntimeError('导出两路不是同一演示原图。')
    subprocess.run(['git', 'diff', '--check'], cwd=ROOT, check=True)
    checks = {
        'checked_at': datetime.now().astimezone().isoformat(),
        'protocol_sha256': protocol_sha, 'frozen_assets_and_samples': 'passed',
        'offline_summary': 'passed; 已单独执行 summarize，未重新推理或调用 API',
        'unit_tests': {'count': 57, 'result': 'OK', 'exit_code': 0,
                       'source': '本阶段先前完成的 unittest discover -s tests -v',
                       'warning': 'Windows Gradio 的 ResourceWarning: unclosed event loop；未导致失败'},
        'project_environment_verification': '本阶段先前执行 verify_project.py，通过',
        'git_diff_check': 'passed', 'markdown_local_links': 'passed',
        'exact_local_key_scan': 'passed; 不输出密钥，源码与嵌套导出逐项扫描',
        'ui_export': {'path': UI_EXPORT, 'entries': ui_entries,
                      'sha256': digest(ROOT / UI_EXPORT), 'same_image_sha256': expected_image,
                      'yolo_counts': route_reports['yolo']['summary'],
                      'deepseek_counts': route_reports['deepseek']['summary'],
                      'additional_demo_usage': {'prompt_tokens': 1154, 'completion_tokens': 89, 'total_tokens': 1243}},
        'video': read('reports/stage8-demo.json'),
        'package_verification': '打包后逐项 CRC、SHA256、密钥扫描结果见 runs/delivery/package-verification.json',
        'remaining': ['个人总结待本人核对表达', '未证明真实桌面泛化', '视频为截图序列；连续录屏要求须另补']}
    save(ROOT / 'reports/stage8-checks.json', checks)
    files.add('reports/stage8-checks.json')
    if any(Path(name).is_absolute() or '..' in Path(name).parts or '.local' in Path(name).parts
           or '.git' in Path(name).parts for name in files):
        raise RuntimeError('打包范围异常。')
    manifest = []
    for name in sorted(files):
        path = ROOT / name
        scan_bytes(path.read_bytes(), name, secrets)
        manifest.append({'path': name, 'bytes': path.stat().st_size, 'sha256': digest(path)})
    DELIVERY.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for row in manifest:
            archive.write(ROOT / row['path'], row['path'])
        archive.writestr('delivery-manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        archive.writestr('交付说明.txt',
            '本地验收证据与推理资源。入口：docs/验收与复现指南.md。\n'
            '不含密钥、解释器或完整训练集；原始记录可能含原机器绝对路径。\n'
            '公开图片及权重遵守其原许可；LOCO 为 CC BY-NC-SA 4.0。\n'
            '来源见协议、data/desk8-train-v1/manifest.json 与 reports/assets/README.md。\n'
            '截图序列不是连续录屏；模型效果与个人理解仍按验收指南核对。\n')
    with zipfile.ZipFile(ARCHIVE) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('交付 ZIP CRC 检查失败。')
        for row in manifest:
            content = archive.read(row['path'])
            if sha256(content).hexdigest() != row['sha256']:
                raise RuntimeError('ZIP 文件指纹不一致：' + row['path'])
            scan_bytes(content, row['path'], secrets)
    verification = {'path': ARCHIVE.relative_to(ROOT).as_posix(), 'sha256': digest(ARCHIVE),
                    'bytes': ARCHIVE.stat().st_size, 'manifest_files': len(manifest),
                    'crc': 'passed', 'all_file_hashes': 'passed', 'exact_key_scan': 'passed',
                    'protocol_assets_and_samples_included': True,
                    'excluded': ['.local', '.git', 'Python解释器', '完整训练数据集']}
    save(DELIVERY / 'package-verification.json', verification)
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == '__main__':
    main()
