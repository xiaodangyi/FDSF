"""将实际浏览器截图制作成明确标注的截图序列演示，不运行模型或调用 API。"""

from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

from deskvision.render import find_font

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / 'runs/stage8-demo'
ASSETS = ROOT / 'reports/assets'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    protocol = read(ROOT / 'reports/stage8-protocol.json')
    credits = ['# 第八阶段图像署名', '',
               '错误图为原图上的检测与掩膜叠加，未改变模型结果；许可与来源如下。', '']
    for sample_id, name in [('oi_02a90407e6bf3ef8', 'error-screwdriver.png'),
                            ('oi_0b92046dc7f1f249', 'error-book.png')]:
        report = read(ROOT / f'runs/acceptance/stage8-v1/{sample_id}-yolo.json')['report']
        shutil.copyfile(Path(report['result_dir']) / 'segmented.png', ASSETS / name)
        row = next(r for r in protocol['samples'] if r['id'] == sample_id)
        credits.extend([f"- `{name}`：{row['attribution']}；{row['source_url']}；"
                        f"{row['license']}，{row['license_url']}。"])
    demo_row = next(r for r in read(ROOT / 'data/desk8-train-v1/manifest.json')['records']
                    if r['id'] == 'oi_38451ec391f20ab5')
    credits.extend(['', f"界面截图中的验证图：{demo_row['attribution']}；{demo_row['source_url']}；"
                    f"{demo_row['license']}，{demo_row['license_url']}。截图包含原图的框与掩膜叠加。",
                    '截图序列视频与本项目界面为自行制作；公开数据原图的权利与许可仍归原作者。'])
    (ASSETS / 'README.md').write_text('\n'.join(credits) + '\n', encoding='utf-8')
    for name in ('03-sam.png', '04-deepseek.png', '05-comparison.png'):
        shutil.copyfile(DEMO / name, ASSETS / ('ui-' + name))

    font_path = str(find_font(None))
    title_font = ImageFont.truetype(font_path, 38)
    body_font = ImageFont.truetype(font_path, 27)
    small_font = ImageFont.truetype(font_path, 20)
    pages = [
        ('01-start.png', '桌面物品双路线识别', [
            '真实界面截图序列操作演示', '非连续录屏，无配音。',
            '页面停留时长由编辑设置，', '不能代表模型实际运行耗时。',
            '演示使用验证图，不加入测试统计。']),
        ('02-input.png', '1. 上传原图与固定参数', [
            '验证图：杯子 1、瓶子 1。', '置信度阈值 0.25，启用 SAM 2。',
            '两路各自读取同一原图。', 'DeepSeek 不接收本地检测结果。']),
        ('03-sam.png', '2. YOLO 检测后由 SAM 描边', [
            '本地图中仍出现杯瓶错类。', 'YOLO：杯子 0、瓶子 2。',
            'SAM 输出 2 个非空实例掩膜。', '描边成功不会纠正类别或漏检。']),
        ('04-deepseek.png', '3. DeepSeek 独立交卷', [
            '独立结果：杯子 1、瓶子 1。', 'API 本次耗时约 2.22 秒。',
            '单次回答正确不代表总体正确。', '失败、截断、无效 JSON 均明确报错。']),
        ('05-comparison.png', '4. 对比数量并导出证据', [
            '杯子差值 +1，瓶子差值 -1。', '差异只展示，不回写任一路结果。',
            '页面可下载两路 JSON、掩膜和图片。', '差值不能替代人工真值。']),
        (None, '5. 固定测试与验收边界', [
            '9 张留出测试图，共 29 个实例。', '八类计数全对：YOLO 7/9，DeepSeek 9/9。',
            '两路运行成功率均为 9/9。', '57 项自动检查通过。',
            '真实桌面泛化、像素分割精度尚未验收。', '完整证据、限制与署名见项目报告。'])]
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('未找到现有 FFmpeg；不自动安装。')
    output = DEMO / '验收演示.mp4'
    encoder = subprocess.Popen([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1280x720', '-r', '10', '-i', '-',
        '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
        '-movflags', '+faststart', str(output)], stdin=subprocess.PIPE)
    try:
        for index, (screenshot, title, lines) in enumerate(pages):
            canvas = Image.new('RGB', (1280, 720), '#101b2d')
            draw = ImageDraw.Draw(canvas)
            draw.text((48, 42), title, font=title_font, fill='#ffffff')
            for line_index, line in enumerate(lines):
                draw.text((48, 145 + line_index * 57), line, font=body_font, fill='#dbe5f5')
            if screenshot:
                with Image.open(DEMO / screenshot) as original:
                    fitted = ImageOps.contain(original.convert('RGB'), (352, 590))
                    canvas.paste(fitted, (875 + (352 - fitted.width) // 2, 100))
            else:
                draw.rounded_rectangle((860, 165, 1230, 490), radius=24, fill='#1c3452')
                draw.text((910, 215), '计数全对图数', font=body_font, fill='white')
                draw.text((905, 295), 'YOLO     7 / 9', font=title_font, fill='#64d4df')
                draw.text((905, 375), 'DeepSeek 9 / 9', font=title_font, fill='#b4f3cf')
            draw.text((48, 655), f'截图序列演示 · 非连续录屏 · {index + 1}/6 · 署名见 reports/assets/README.md',
                      font=small_font, fill='#9cadc4')
            canvas.save(DEMO / f'video-page-{index + 1}.png')
            raw = canvas.tobytes()
            for _ in range(60):
                encoder.stdin.write(raw)
    finally:
        encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('视频编码失败。')
    capture = cv2.VideoCapture(str(output))
    decoded = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[:2] != (720, 1280):
            raise RuntimeError('视频尺寸不一致。')
        if decoded == 180:
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(DEMO / 'video-decoded-check.png')
        decoded += 1
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if decoded != 360 or fps != 10:
        raise RuntimeError('视频完整解码检查失败。')
    metadata = {'format': '真实界面截图序列演示，非连续录屏；无配音，停留时长不代表推理耗时',
                'path': output.relative_to(ROOT).as_posix(), 'codec': 'H.264/yuv420p',
                'frames_decoded': decoded, 'fps': fps, 'seconds': decoded / fps,
                'width': 1280, 'height': 720, 'sha256': sha256(output.read_bytes()).hexdigest(),
                'source_screenshots': [row[0] for row in pages if row[0]]}
    (ROOT / 'reports/stage8-demo.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == '__main__':
    main()
