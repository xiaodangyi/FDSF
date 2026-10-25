"""第七阶段 Gradio 界面：统一操作与展示，两条识别路线保持独立。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from importlib import metadata
import json
from pathlib import Path
from threading import Lock
from time import perf_counter
from uuid import uuid4
from zoneinfo import ZoneInfo
from zipfile import ZIP_DEFLATED, ZipFile

from .config import ROOT, load_categories, local_sam2_weights, local_weights
from .detector import BaselineDetector, TrainedDetector, file_sha256, read_image
from .render import find_font, render_result, render_segmentation
from .vision_language import prepare_image, recognize
from .vlm_config import VlmError, load_vlm_settings


DEFAULT_FINETUNED_WEIGHTS = ROOT / 'runs/train/desk8-full-20261023-204546/weights/best.pt'
UI_OUTPUT_ROOT = ROOT / 'runs/ui'
# Gradio 队列与锁共同避免多会话同时使用同一个 GPU。
LOCAL_MODEL_LOCK = Lock()


def _path_value(value: str | Path | object) -> Path:
    """将 Gradio filepath 或 CLI 路径转换为绝对路径，不接受空输入。"""
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str) and value.strip():
        path = Path(value)
    elif hasattr(value, 'name'):
        path = Path(str(value.name))
    else:
        raise ValueError('请先上传一张图片。')
    if not str(path).strip():
        raise ValueError('请先上传一张图片。')
    return path.expanduser().resolve()


def _new_run_dir(root: Path, route: str) -> Path:
    timestamp = datetime.now(ZoneInfo('Asia/Shanghai'))
    destination = root / route / f'{timestamp:%Y%m%d-%H%M%S}-{uuid4().hex[:8]}'
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _failed_result(route: str, error: Exception, started: float, destination: Path | None = None,
                   partial: dict | None = None, stage: str = 'input') -> dict:
    report = dict(partial or {})
    report.update({
        'schema_version': 1,
        'status': 'failed',
        'route': route,
        'timings_ms': {**report.get('timings_ms', {}), 'total_wall': round((perf_counter() - started) * 1000, 3)},
        'error': {'stage': stage, 'type': type(error).__name__, 'code': getattr(error, 'code', 'local_error'),
                  'message': str(error) if isinstance(error, VlmError) else '本地处理失败，请检查输入、权重和环境。'},
    })
    if destination is not None:
        try:
            _write_json(destination / 'result.json', report)
        except OSError:
            pass
        report['result_dir'] = str(destination)
    if destination is not None and (destination / 'annotated.png').is_file():
        report['visualization_path'] = str(destination / 'annotated.png')
    return report


def run_yolo_route(
    image_path: str | Path | object,
    *,
    mode: str = 'finetuned',
    weights_path: str | Path | None = None,
    confidence: float = 0.25,
    iou: float = 0.7,
    image_size: int = 640,
    device: str = 'auto',
    segment: bool = True,
    output_root: Path | None = None,
) -> dict:
    """运行本地 YOLO 路线；SAM 只接收本路线的 YOLO 框。"""
    started = perf_counter()
    destination = None
    report = None
    stage = 'input'
    route = 'yolov8_sam2_independent'
    try:
        source = _path_value(image_path)
        destination = _new_run_dir(output_root or UI_OUTPUT_ROOT, 'yolo')
        _, image_info = prepare_image(source)
        if mode not in ('baseline', 'finetuned'):
            raise ValueError('权重模式必须是 baseline 或 finetuned。')
        if not 0 < float(confidence) < 1 or not 0 < float(iou) < 1:
            raise ValueError('检测阈值必须在 0 与 1 之间。')
        if not 32 <= int(image_size) <= 2048 or int(image_size) % 32:
            raise ValueError('图片输入尺寸须为 32～2048 之间的 32 倍数。')
        if device not in ('auto', 'cpu', 'cuda:0'):
            raise ValueError('计算设备必须是 auto、cpu 或 cuda:0。')
        image_started = perf_counter()
        image = read_image(source)
        decode_ms = (perf_counter() - image_started) * 1000
        categories = load_categories()
        if mode == 'finetuned':
            selected_weights = weights_path or (DEFAULT_FINETUNED_WEIGHTS if DEFAULT_FINETUNED_WEIGHTS.is_file() else None)
            if selected_weights is None:
                raise ValueError('八类微调权重不存在，请选择已有的 best.pt。')
        else:
            selected_weights = weights_path or None
        weights = local_weights(str(selected_weights) if selected_weights else None)
        detector_type = TrainedDetector if mode == 'finetuned' else BaselineDetector
        stage = 'detection'
        with LOCAL_MODEL_LOCK:
            detector = detector_type(weights, categories, device)
            report = detector.predict(image, float(confidence), float(iou), int(image_size))
        timestamp = datetime.now(ZoneInfo('Asia/Shanghai'))
        report['created_at'] = timestamp.isoformat()
        report['input'] = image_info
        for item in report['summary']:
            item['category_key'] = categories[item['category_id']].key
        report['versions'] = {name: metadata.version(name)
                              for name in ('torch', 'torchvision', 'ultralytics', 'opencv-python', 'Pillow')}
        report['category_config'] = {
            'sha256': file_sha256(ROOT / 'configs/classes.yaml'),
            'mapping': [{'id': item.id, 'key': item.key, 'name': item.name, 'coco_id': item.coco_id}
                        for item in categories],
        }
        report['timings_ms']['image_decode'] = round(decode_ms, 3)
        report['artifacts'] = {'result': 'result.json'}
        stage = 'artifacts'
        font = find_font(None)
        render_result(image, report, font).save(destination / 'annotated.png')
        report['artifacts']['visualization'] = 'annotated.png'
        if segment:
            from .segmentation import Sam2Segmenter

            stage = 'segmentation'
            segmenter = Sam2Segmenter(local_sam2_weights(None), detector.device)
            with LOCAL_MODEL_LOCK:
                report = segmenter.segment(image, report, destination)
            report['versions']['SAM-2'] = report['segmentation']['package_version']
            render_segmentation(image, report, destination, font).save(destination / 'segmented.png')
            report['artifacts']['segmentation_visualization'] = 'segmented.png'
        report['timings_ms']['total_wall'] = round((perf_counter() - started) * 1000, 3)
        report['timing_note'] = '总耗时从输入校验到产物生成，含本次模型加载；不含排队、上传或下载，不代表稳定帧率。'
        _write_json(destination / 'result.json', report)
        report['result_dir'] = str(destination)
        report['visualization_path'] = str(destination / ('segmented.png' if segment else 'annotated.png'))
        return report
    except Exception as error:
        return _failed_result(route, error, started, destination, report, stage)


def run_vlm_route(image_path: str | Path | object, *, output_root: Path | None = None) -> dict:
    """运行 DeepSeek 原图路线；不读取任何 YOLO/SAM 产物。"""
    started = perf_counter()
    destination = None
    try:
        source = _path_value(image_path)
        destination = _new_run_dir(output_root or UI_OUTPUT_ROOT, 'deepseek')
        settings = load_vlm_settings()
        report = recognize(source, destination, settings)
        report['result_dir'] = str(destination)
        report['timings_ms']['ui_total_wall'] = round((perf_counter() - started) * 1000, 3)
        _write_json(destination / 'result.json', report)
        return report
    except Exception as error:
        return _failed_result('deepseek_vision_independent', error, started, destination)


def _counts(report: dict | None) -> dict[str, int | None]:
    if not report or report.get('status') != 'ok':
        return {}
    return {item['category_key']: item.get('count') for item in report.get('summary', [])}


def format_yolo_result(report: dict) -> str:
    if report.get('status') != 'ok':
        error = report.get('error', {})
        preserved = '\n\n检测结果已保留，分割未完成。' if report.get('detections') is not None else ''
        return f"### 本地识别\n\n**失败（{error.get('stage', '执行')}）**：{error.get('message', '本地处理失败。')}" + preserved
    lines = ['### YOLOv8 + SAM 2', '', f"状态：{report.get('message', '完成')}" , '', '| 类别 | 数量 |', '|---|---:|']
    lines.extend(f"| {item['category_name']} | {'尚未训练' if item['count'] is None else item['count']} |"
                 for item in report.get('summary', []))
    timings = report.get('timings_ms', {})
    lines.extend(['', f"模型推理耗时：{timings.get('prediction_wall', '—')} ms",
                  f"总耗时：{timings.get('total_wall', '—')} ms",
                  'SAM 2 只沿用 YOLO 检测框做分割，不会改分类或计数。' if 'segmentation' in report
                  else '本次只运行 YOLO 检测。'])
    return '\n'.join(lines)


def format_vlm_result(report: dict) -> str:
    if report.get('status') != 'ok':
        error = report.get('error', {})
        return f"### DeepSeek 独立识别\n\n**失败**：{error.get('message', '视觉模型处理失败。')}"
    lines = ['### DeepSeek 独立识别', '', report.get('message', ''), '', '| 类别 | 数量 |', '|---|---:|']
    lines.extend(f"| {item['category_name']} | {item['count']} |" for item in report.get('summary', []))
    if report.get('items'):
        lines.extend(['', '**逐件位置**'])
        lines.extend(f"- {item['category_name']}：{item['location_text']}" + ('（存疑）' if item['uncertain'] else '')
                     for item in report['items'])
    if report.get('notes'):
        lines.extend(['', f"说明：{report['notes']}"])
    timings = report.get('timings_ms', {})
    lines.extend(['', f"API 耗时：{timings.get('api_wall', '—')} ms",
                  f"总耗时：{timings.get('ui_total_wall', timings.get('total_wall', '—'))} ms",
                  'DeepSeek 只读取原图，不接收 YOLO/SAM 结果。'])
    return '\n'.join(lines)


def format_count_diff(yolo_report: dict | None, vlm_report: dict | None) -> str:
    if not yolo_report or not vlm_report:
        return '### 数量对比\n\n分别运行两条路线后，这里会显示八类数量差异。'
    yolo_counts, vlm_counts = _counts(yolo_report), _counts(vlm_report)
    if yolo_report.get('status') != 'ok' or vlm_report.get('status') != 'ok':
        return '### 数量对比\n\n至少一条路线失败，暂不计算差异；失败不伪造为零检测。'
    if (not yolo_report.get('input', {}).get('sha256') or
            yolo_report['input']['sha256'] != vlm_report.get('input', {}).get('sha256')):
        return '### 数量对比\n\n两路输入图片不一致，不能比较；请使用同一张原图重新运行。'
    names = {item['category_key']: item['category_name'] for item in yolo_report.get('summary', [])}
    names.update({item['category_key']: item['category_name'] for item in vlm_report.get('summary', [])})
    lines = ['### 数量对比', '', '| 类别 | YOLOv8 | DeepSeek | DeepSeek - YOLO |', '|---|---:|---:|---:|']
    for key, name in names.items():
        left, right = yolo_counts.get(key), vlm_counts.get(key)
        if left is None or right is None:
            delta = '—'
        else:
            delta = str(right - left)
        lines.append(f'| {name} | {"—" if left is None else left} | {"—" if right is None else right} | {delta} |')
    lines.extend(['', '差异只在两条独立结果生成后计算，不会回写任一路线。'])
    return '\n'.join(lines)


def export_results(yolo_report: dict | None, vlm_report: dict | None, *, output_root: Path | None = None) -> str | None:
    """只从 UI 运行目录打包公开结果，不接受任意本地文件路径。"""
    root = (output_root or UI_OUTPUT_ROOT).resolve()
    reports = {'yolo': yolo_report, 'deepseek': vlm_report}
    if not any(reports.values()):
        return None
    destination = _new_run_dir(root, 'exports')
    archive = destination / '双路识别结果.zip'
    with ZipFile(archive, 'w', ZIP_DEFLATED) as bundle:
        for route, report in reports.items():
            if not report or not report.get('result_dir'):
                continue
            folder = Path(report['result_dir']).resolve()
            if not folder.is_relative_to(root / route) or not folder.is_dir():
                continue
            for path in sorted(folder.rglob('*')):
                if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(folder):
                    bundle.write(path, f'{route}/{path.relative_to(folder).as_posix()}')
        bundle.writestr('数量对比.md', format_count_diff(yolo_report, vlm_report))
    return str(archive)


def run_yolo_ui(image_path, mode, weights_path, confidence, iou, image_size, device, segment):
    report = run_yolo_route(image_path, mode=mode, weights_path=weights_path or None,
                            confidence=confidence, iou=iou, image_size=int(image_size),
                            device=device, segment=segment)
    return (_result_images(report), format_yolo_result(report), report,
            'YOLO 路线完成。' if report.get('status') == 'ok' else 'YOLO 路线失败。')


def run_vlm_ui(image_path):
    report = run_vlm_route(image_path)
    return (format_vlm_result(report), report,
            'DeepSeek 路线完成。' if report.get('status') == 'ok' else 'DeepSeek 路线失败。')


def run_both_ui(image_path, mode, weights_path, confidence, iou, image_size, device, segment):
    """一键运行两路；两路各自读取同一原图，结果只在最后用于展示比较。"""
    with ThreadPoolExecutor(max_workers=2) as pool:
        local = pool.submit(run_yolo_route, image_path, mode=mode, weights_path=weights_path or None,
                            confidence=confidence, iou=iou, image_size=int(image_size), device=device, segment=segment)
        remote = pool.submit(run_vlm_route, image_path)
        yolo_report, vlm_report = local.result(), remote.result()
    return (
        _result_images(yolo_report), format_yolo_result(yolo_report), yolo_report,
        format_vlm_result(vlm_report), vlm_report, format_count_diff(yolo_report, vlm_report),
        '两路均完成。' if yolo_report.get('status') == vlm_report.get('status') == 'ok'
        else '两路已执行，但至少一路失败。',
    )


def _result_images(report: dict) -> list:
    folder = Path(report['result_dir']) if report.get('result_dir') else None
    if folder is None:
        return []
    return [(str(folder / name), label) for name, label in
            [('annotated.png', '检测框'), ('segmented.png', 'SAM 2 实例掩膜')]
            if (folder / name).is_file()]


def refresh_comparison(yolo_report, vlm_report):
    return format_count_diff(yolo_report, vlm_report), export_results(yolo_report, vlm_report)


def reset_input(image_path):
    """上传、替换、删除原图后清掉两路旧结果。"""
    return (image_path, [], '尚未运行本地路线。', None, '尚未运行 DeepSeek 路线。', None,
            format_count_diff(None, None), None, '原图已更新，请运行识别。' if image_path else '等待上传原图。')


def reset_yolo():
    return [], '本地参数已修改，请重新运行。', None, format_count_diff(None, None), None


def weights_for_mode(mode):
    return str(DEFAULT_FINETUNED_WEIGHTS) if mode == 'finetuned' else ''


def new_session():
    return {'image': None, 'params': None, 'revision': 0}


def invalidate_session(session, values):
    """快速记录当前输入；正在运行的旧任务可以结束，但不能覆盖新页面。"""
    session.update(image=values[0], params=tuple(values[1:]), revision=session['revision'] + 1)


def run_session_ui(route, session, previous_yolo, previous_vlm, *values):
    import gradio as gr

    def current():
        return session['image'] == values[0] and (route == 'deepseek' or session['params'] == tuple(values[1:]))

    skipped = tuple(gr.skip() for _ in range(8))
    if not current():
        return skipped
    revision = session['revision']
    if route == 'both':
        gallery, local_text, local, remote_text, remote, _, status = run_both_ui(*values)
    elif route == 'yolo':
        gallery, local_text, local, status = run_yolo_ui(*values)
        remote, remote_text = previous_vlm, gr.skip()
    else:
        remote_text, remote, status = run_vlm_ui(values[0])
        local, gallery, local_text = previous_yolo, gr.skip(), gr.skip()
    if revision != session['revision'] or not current():
        return skipped
    comparison, archive = refresh_comparison(local, remote)
    if revision != session['revision']:
        return skipped
    return (gallery, local_text, local if route != 'deepseek' else gr.skip(),
            remote_text, remote if route != 'yolo' else gr.skip(), comparison, archive, status)


def build_demo():
    """构建 Gradio Blocks；导入本模块不会加载 torch、Ultralytics 或 SAM 2。"""
    import gradio as gr

    css = '''
    .gradio-container {max-width: 1440px !important; margin: auto;}
    #masthead {padding: 22px 26px; border-radius: 14px; background: #ecf7f5; border: 1px solid #cde5df;}
    #masthead h1 {color: #163e38; margin-top: 4px;}
    #masthead p {color: #40665e;}
    #run-status {padding: 10px 16px; border-radius: 10px; background: var(--background-fill-secondary);}
    '''
    with gr.Blocks(title='桌面物品双路线识别', theme=gr.themes.Soft(primary_hue='teal'), css=css,
                   analytics_enabled=False) as demo:
        session = gr.State(value=new_session)
        gr.Markdown(
            '# 桌面物品双路线识别\n\n'
            '上传一次原图后，可分别或一键运行 YOLOv8 + SAM 2 与 DeepSeek。两路独立读取原图，'
            '只在结果生成后展示数量差异。', elem_id='masthead'
        )
        with gr.Row():
            with gr.Column():
                image = gr.File(type='filepath', file_count='single',
                                file_types=['.jpg', '.jpeg', '.png', '.webp', '.gif'], label='上传原图文件（保留原始字节）')
                preview = gr.Image(label='原图预览', interactive=False, height=300)
            with gr.Column():
                gr.Markdown('### 运行设置\n\n本地模型覆盖八类物品；结果仍需人工核对。')
                confidence = gr.Slider(0.05, 0.95, value=0.25, step=0.01, label='检测置信度阈值')
                segment = gr.Checkbox(value=True, label='使用 SAM 2 描绘实例轮廓')
                with gr.Accordion('高级设置', open=False):
                    mode = gr.Dropdown([('八类微调模型', 'finetuned'), ('COCO 预训练基线', 'baseline')],
                                       value='finetuned', label='YOLO 权重模式')
                    weights = gr.Textbox(value=str(DEFAULT_FINETUNED_WEIGHTS), label='已有权重路径（基线留空使用本机配置）')
                    iou = gr.Slider(0.05, 0.95, value=0.7, step=0.01, label='NMS IoU 阈值')
                    image_size = gr.Dropdown([320, 640, 960, 1280], value=640, label='模型输入尺寸')
                    device = gr.Dropdown(['auto', 'cuda:0', 'cpu'], value='auto', label='计算设备')
                gr.Markdown('点击 DeepSeek 或运行两路，会将当前原图发送至 DeepSeek 官方服务并产生 API 用量。')
        with gr.Row():
            yolo_button = gr.Button('运行 YOLOv8 + SAM 2', variant='primary')
            vlm_button = gr.Button('运行 DeepSeek')
            both_button = gr.Button('运行两路并比较', variant='primary')
            gr.ClearButton([image], value='清空原图')
        status = gr.Markdown('等待上传原图。', elem_id='run-status')
        with gr.Row():
            with gr.Column():
                yolo_image = gr.Gallery(label='本地检测与分割', columns=2, height=340, object_fit='contain', interactive=False)
                yolo_markdown = gr.Markdown('尚未运行 YOLO 路线。')
                with gr.Accordion('本地路线详细报告', open=False):
                    yolo_json = gr.JSON(label='YOLO 路线原始报告')
            with gr.Column():
                vlm_markdown = gr.Markdown('尚未运行 DeepSeek 路线。')
                with gr.Accordion('DeepSeek 详细报告', open=False):
                    vlm_json = gr.JSON(label='DeepSeek 路线原始报告')
        diff_markdown = gr.Markdown('### 数量对比\n\n分别运行两条路线后，这里会显示八类数量差异。')
        download = gr.File(label='下载当前结果（JSON、图片、掩膜与数量对比）', interactive=False)
        gr.Markdown('耗时包含本次模型加载或网络请求；总耗时不含排队和上传。数量差异不能替代人工真值或准确率评估。')

        yolo_inputs = [image, mode, weights, confidence, iou, image_size, device, segment]
        queue_options = {'concurrency_id': 'deskvision', 'concurrency_limit': 1}
        outputs = [yolo_image, yolo_markdown, yolo_json, vlm_markdown, vlm_json, diff_markdown, download, status]
        for button, route in [(yolo_button, 'yolo'), (vlm_button, 'deepseek'), (both_button, 'both')]:
            def run(state, local, remote, *values, selected=route):
                return run_session_ui(selected, state, local, remote, *values)

            button.click(run, inputs=[session, yolo_json, vlm_json, *yolo_inputs],
                         outputs=outputs, **queue_options)

        def update_image(state, *values):
            invalidate_session(state, values)
            return reset_input(values[0])

        def update_local(state, *values):
            invalidate_session(state, values)
            return (*reset_yolo(), '本地参数已修改，请重新运行本地路线。')

        image.change(update_image, [session, *yolo_inputs],
                     [preview, yolo_image, yolo_markdown, yolo_json, vlm_markdown, vlm_json,
                      diff_markdown, download, status], queue=False)
        mode.change(weights_for_mode, [mode], [weights], queue=False)
        for component in [mode, weights, confidence, iou, image_size, device, segment]:
            component.change(update_local, [session, *yolo_inputs],
                             [yolo_image, yolo_markdown, yolo_json, diff_markdown, download, status], queue=False)
        demo.queue(max_size=16, default_concurrency_limit=1)
    return demo


def main() -> int:
    parser = argparse.ArgumentParser(description='启动桌面物品双路线 Gradio 界面。')
    parser.add_argument('--port', type=int, default=7860, help='监听端口，默认 7860')
    args = parser.parse_args()
    build_demo().launch(server_name='127.0.0.1', server_port=args.port, share=False,
                        allowed_paths=[str(UI_OUTPUT_ROOT)], blocked_paths=[str(ROOT / '.local')],
                        max_file_size='32mb', show_error=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
