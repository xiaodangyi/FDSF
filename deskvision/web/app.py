"""工作台展示层；推理、原图隔离、导出及过期结果保护均沿用原入口。"""

from html import escape
from pathlib import Path

from deskvision import ui as core


ASSETS = Path(__file__).parent
EXAMPLE = core.ROOT / 'data/desk8-train-v1/images/val/oi_38451ec391f20ab5.jpg'


def _total(report):
    if not report or report.get('status') != 'ok':
        return '—'
    values = [item.get('count') for item in report.get('summary', [])]
    if not values or any(type(value) is not int for value in values):
        return '—'
    return str(sum(values))


def overview(local, remote):
    """只有同图且成功的结果才能产生差异数，缺失与失败均不显示零。"""
    diff = '—'
    note = '运行两路后比较'
    if local and remote and local.get('status') == remote.get('status') == 'ok':
        fingerprint = local.get('input', {}).get('sha256')
        if fingerprint and fingerprint == remote.get('input', {}).get('sha256'):
            left, right = core._counts(local), core._counts(remote)
            if left.keys() == right.keys() and left and all(
                    type(left[key]) is int and type(right[key]) is int for key in left):
                diff = str(sum(left[key] != right[key] for key in left))
                note = '数量不一致的类别'
        else:
            note = '原图不一致，无法比较'
    cards = [('本地检出', _total(local), 'YOLOv8 + SAM 2', 'local'),
             ('大模型清点', _total(remote), 'DeepSeek 独立识别', 'remote'),
             ('差异类别', diff, note, 'difference')]
    return '<div class="metrics">' + ''.join(
        f'<div class="metric {kind}"><span>{title}</span><strong>{value}'
        f'<small>{"类" if kind == "difference" else "件"}</small></strong>'
        f'<p>{subtitle}</p></div>' for title, value, subtitle, kind in cards) + '</div>'


def remote_inventory(report):
    if not report:
        return ('<div class="inventory-empty"><span class="empty-mark" aria-hidden="true">'
                '≡</span><h3>等待独立识别</h3><p>运行 DeepSeek 后，在这里查看物品清单与位置。</p>'
                '<span class="quiet-tag">仅接收原图</span></div>')
    if report.get('status') != 'ok':
        message = escape(str(report.get('error', {}).get('message', '请查看详细报告。')))
        return f'<div class="inventory-error"><h3>本次识别未完成</h3><p>{message}</p><p>失败不计为零件，请检查后重新运行。</p></div>'
    items = report.get('items', [])
    if not items:
        return '<div class="inventory-empty"><h3>本次未列出目标</h3><p>请对照原图核查漏检。</p></div>'
    rows = []
    for index, item in enumerate(items, 1):
        name = escape(str(item.get('category_name', '未命名物品')))
        location = escape(str(item.get('location_text', '未提供位置')))
        uncertain = '<span class="uncertain">存疑</span>' if item.get('uncertain') else ''
        rows.append(f'<li><span class="item-index">{index:02}</span><div><strong>{name}</strong>'
                    f'<p>{location}</p></div>{uncertain}</li>')
    return ('<div class="inventory"><div class="inventory-heading"><span>物品</span>'
            '<span>相对位置</span></div><ol>' + ''.join(rows) +
            '</ol><p class="inventory-footnote">位置为文字描述，请结合原图核对。</p></div>')


def route_badges(local, remote):
    def badge(report, title):
        state = 'waiting' if not report else ('success' if report.get('status') == 'ok' else 'failed')
        label = {'waiting': '待运行', 'success': '已完成', 'failed': '未完成'}[state]
        return f'<span class="route-badge {state}"><i aria-hidden="true"></i>{title} · {label}</span>'
    return '<div class="route-badges">' + badge(local, '本地') + badge(remote, 'DeepSeek') + '</div>'


def run_confirmed(route, allow_remote, state, local, remote, *values):
    """Keep external-image/cost consent at the user action boundary."""
    if route in ('deepseek', 'both') and not allow_remote:
        import gradio as gr

        raise gr.Error('请先勾选云端调用确认；也可以只运行本地 YOLOv8 + SAM 2。')
    return core.run_session_ui(route, state, local, remote, *values)


def build_demo():
    import gradio as gr

    theme = gr.themes.Base(primary_hue='blue', secondary_hue='teal', neutral_hue='slate',
                           font=['Microsoft YaHei', 'Segoe UI', 'sans-serif'],
                           font_mono=['Cascadia Code', 'Consolas', 'monospace']).set(
        button_primary_background_fill='#2563eb', button_primary_text_color='white',
        button_primary_background_fill_hover='#1d4ed8',
        button_primary_background_fill_dark='#2563eb', button_primary_text_color_dark='white',
        button_primary_background_fill_hover_dark='#1d4ed8',
        body_background_fill='#f5f7fb', body_background_fill_dark='#101827',
        block_radius='12px', button_large_radius='10px',
    )
    with gr.Blocks(title='DeskVision · 桌面物品识别', theme=theme,
                   css=(ASSETS / 'styles.css').read_text(encoding='utf-8'),
                   analytics_enabled=False) as demo:
        session = gr.State(value=core.new_session)
        gr.HTML('''<header class="topbar"><div class="brand"><span class="brand-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M8 3H4a1 1 0 0 0-1 1v4m13-5h4a1 1 0 0 1 1 1v4M3 16v4a1 1 0 0 0 1 1h4m8 0h4a1 1 0 0 0 1-1v-4"/>
            <rect x="7" y="7" width="10" height="10" rx="2"/></svg></span>
            <div><strong>DeskVision<span>视觉工作台</span></strong><p>桌面物品识别与清点</p></div></div>
            <div class="topbar-meta"><span class="local-tag">本机工作台</span><span>YOLOv8 / SAM 2 / DeepSeek</span></div></header>
            <section class="intro"><div><p class="eyebrow">识别 · 清点 · 核对</p><h1>一张原图，两种独立视角。</h1>
            <p>检测轮廓与物品清单并排呈现，让每一次清点都有据可查。</p></div>
            <div class="workflow" aria-label="操作顺序"><span><b>01</b>上传原图</span><i aria-hidden="true">/</i>
            <span><b>02</b>运行识别</span><i aria-hidden="true">/</i><span><b>03</b>核对结果</span></div></section>''')
        with gr.Row(elem_id='workspace', equal_height=False):
            with gr.Column(scale=4, min_width=310, elem_id='control-panel'):
                gr.Markdown('## 原图与运行\n上传一次，两路共享同一张原图。', elem_classes='section-title')
                image = gr.File(type='filepath', file_count='single',
                                file_types=['.jpg', '.jpeg', '.png', '.webp', '.gif'],
                                label='上传原图文件（保留原始字节）', elem_id='source-file')
                preview = gr.Image(label='原图预览', interactive=False, height=210,
                                   show_download_button=False, elem_id='source-preview')
                with gr.Row(elem_id='source-actions'):
                    example = gr.Button('载入杯瓶示例', size='sm', interactive=EXAMPLE.is_file())
                    gr.ClearButton([image], value='清空原图', size='sm')
                gr.Markdown('支持 JPG、PNG、WEBP、GIF · 最大 32 MB', elem_classes='helper')
                with gr.Accordion('识别设置', open=False, elem_id='settings'):
                    confidence = gr.Slider(.05, .95, value=.25, step=.01, label='检测置信度阈值')
                    segment = gr.Checkbox(value=True, label='使用 SAM 2 描绘实例轮廓')
                    mode = gr.Dropdown([('八类微调模型', 'finetuned'), ('COCO 预训练基线', 'baseline')],
                                       value='finetuned', label='YOLO 权重模式')
                    weights = gr.Textbox(value=str(core.DEFAULT_FINETUNED_WEIGHTS),
                                         label='已有权重路径（基线留空使用本机配置）')
                    iou = gr.Slider(.05, .95, value=.7, step=.01, label='NMS IoU 阈值')
                    image_size = gr.Dropdown([320, 640, 960, 1280], value=640, label='模型输入尺寸')
                    device = gr.Dropdown(['auto', 'cuda:0', 'cpu'], value='auto', label='计算设备')
                gr.Markdown('可在识别设置中调整本地模型与分割。', elem_classes='helper')
                allow_remote = gr.Checkbox(value=False,
                    label='我同意本次将原图发送至 DeepSeek，并产生 API 用量')
                both_button = gr.Button('运行两路并比较', variant='primary', interactive=False,
                                        elem_id='run-both')
                with gr.Row(elem_id='single-actions'):
                    yolo_button = gr.Button('运行 YOLOv8 + SAM 2', interactive=False)
                    vlm_button = gr.Button('运行 DeepSeek', interactive=False)
                gr.Markdown('**云端调用说明**\n点击 DeepSeek 或运行两路，会向 DeepSeek 发送原图并产生 API 用量。',
                            elem_id='cloud-note')
                status = gr.Markdown('等待上传原图。', elem_id='run-status',
                                     container=True)
            with gr.Column(scale=9, min_width=340, elem_id='results-panel'):
                with gr.Row(elem_id='result-heading'):
                    gr.Markdown('## 识别概览', elem_classes='section-title')
                    badges = gr.HTML(route_badges(None, None), elem_id='route-status')
                metrics = gr.HTML(overview(None, None), elem_id='overview')
                with gr.Tabs(elem_id='result-tabs'):
                    with gr.Tab('识别结果'):
                        with gr.Row(equal_height=True, elem_id='route-columns'):
                            with gr.Column(min_width=250, elem_classes='route-card'):
                                gr.HTML('<div class="route-title"><span class="route-label local-label">本地路线</span>'
                                        '<h2>检测与分割</h2><p>YOLOv8 + SAM 2</p></div>')
                                yolo_image = gr.Gallery(label='本地检测与分割', columns=2, height=300,
                                                       object_fit='contain', interactive=False, elem_id='local-gallery')
                                gr.Markdown('检测框定位目标，SAM 描绘实例轮廓。', elem_classes='helper')
                                with gr.Accordion('本地数量与耗时', open=False):
                                    yolo_markdown = gr.Markdown('尚未运行本地路线。')
                            with gr.Column(min_width=250, elem_classes='route-card'):
                                gr.HTML('<div class="route-title"><span class="route-label remote-label">大模型路线</span>'
                                        '<h2>独立物品清单</h2><p>DeepSeek · 仅接收原图</p></div>')
                                inventory = gr.HTML(remote_inventory(None), elem_id='remote-inventory')
                                with gr.Accordion('大模型数量与耗时', open=False):
                                    vlm_markdown = gr.Markdown('尚未运行 DeepSeek 路线。')
                    with gr.Tab('数量对比'):
                        diff_markdown = gr.Markdown(core.format_count_diff(None, None), elem_id='comparison')
                        gr.Markdown('数量一致不代表识别正确，请对照原图或人工真值核验。', elem_classes='helper')
                    with gr.Tab('详细报告'):
                        with gr.Accordion('本地路线详细报告', open=True):
                            yolo_json = gr.JSON(label='YOLO 路线原始报告')
                        with gr.Accordion('DeepSeek 详细报告', open=False):
                            vlm_json = gr.JSON(label='DeepSeek 路线原始报告')
                    with gr.Tab('历史记录') as history_tab:
                        history_table = gr.Dataframe(
                            headers=['任务编号', '路线', '状态', '创建时间', '路线摘要', '备注'],
                            datatype=['str', 'str', 'str', 'str', 'str', 'str'],
                            value=[], interactive=False, wrap=True, label='最近 100 条任务')
                        with gr.Row():
                            history_id = gr.Dropdown(label='历史任务', choices=[], interactive=True)
                            history_refresh = gr.Button('刷新历史')
                            history_load = gr.Button('查看详情')
                        history_detail = gr.JSON(label='任务详情')
                        history_note = gr.Textbox(label='备注', lines=2, max_lines=4,
                                                  max_length=1000)
                        with gr.Row():
                            history_save = gr.Button('保存备注', variant='primary')
                            history_delete = gr.Button('删除任务')
                        history_status = gr.Markdown('选择一条历史任务。')
                with gr.Group(elem_id='export-panel'):
                    gr.Markdown('### 保存本次结果\n导出图片、掩膜、JSON 与数量对比，方便复核。')
                    download = gr.File(label='下载当前结果（JSON、图片、掩膜与数量对比）', interactive=False)
                gr.Markdown('支持八类：杯子、瓶子、书、橡皮、螺丝刀、螺丝、螺母、垫圈。'
                            '\n\nSAM 沿用检测类别；两路结果互不回写。耗时包含模型加载或网络请求，请人工核对识别结论。',
                            elem_id='workspace-note')

        yolo_inputs = [image, mode, weights, confidence, iou, image_size, device, segment]
        buttons = [both_button, yolo_button, vlm_button]
        outputs = [yolo_image, yolo_markdown, yolo_json, vlm_markdown, vlm_json,
                   diff_markdown, download, status]
        decorations = [metrics, inventory, badges]

        def decorate(local, remote):
            return overview(local, remote), remote_inventory(remote), route_badges(local, remote)

        def update_image(state, *values):
            core.invalidate_session(state, values)
            return (*core.reset_input(values[0]), *decorate(None, None),
                    *(gr.update(interactive=bool(values[0])) for _ in buttons))

        def update_local(state, remote, *values):
            core.invalidate_session(state, values)
            return (*core.reset_yolo(), '本地参数已修改，请重新运行本地路线。',
                    *decorate(None, remote))

        image.change(update_image, [session, *yolo_inputs],
                     [preview, *outputs, *decorations, *buttons], queue=False,
                     show_progress='hidden', api_name='set_image')
        example.click(lambda: str(EXAMPLE), outputs=image, queue=False, show_progress='hidden')
        mode.change(core.weights_for_mode, [mode], [weights], queue=False)
        for component in yolo_inputs[1:]:
            component.change(update_local, [session, vlm_json, *yolo_inputs],
                             [yolo_image, yolo_markdown, yolo_json, diff_markdown, download, status,
                              *decorations], queue=False, show_progress='hidden')

        for button, route in [(yolo_button, 'yolo'), (vlm_button, 'deepseek'), (both_button, 'both')]:
            def run(state, local, remote, allow_cloud, *values, selected=route):
                if selected in ('deepseek', 'both') and not allow_cloud:
                    raise gr.Error('请先勾选云端调用确认；也可以只运行本地 YOLOv8 + SAM 2。')
                running = list(gr.skip() for _ in outputs)
                running[-1] = {'yolo': '正在运行本地检测与分割…', 'deepseek': '正在等待 DeepSeek 响应…',
                               'both': '正在运行两路识别，请稍候…'}[selected]
                yield (*running, *(gr.skip() for _ in decorations),
                       *(gr.update(interactive=False) for _ in buttons))
                result = run_confirmed(selected, allow_cloud, state, local, remote, *values)
                current_local = local if selected == 'deepseek' else result[2]
                current_remote = remote if selected == 'yolo' else result[4]
                stale = result[-1] == {'__type__': 'update'}
                updated = (gr.skip(),) * 3 if stale else decorate(current_local, current_remote)
                yield (*result, *updated, *(gr.update(interactive=bool(state['image'])) for _ in buttons))

            button.click(run, inputs=[session, yolo_json, vlm_json, allow_remote, *yolo_inputs],
                         outputs=[*outputs, *decorations, *buttons],
                         concurrency_id='deskvision', concurrency_limit=1,
                         show_progress='minimal', show_progress_on=[status], trigger_mode='once',
                         api_name=f'run_{route}')

        def refresh_history():
            from deskvision.history import history_rows

            rows, ids = history_rows()
            return rows, gr.update(choices=ids, value=ids[0] if ids else None), \
                '历史记录已刷新。' if ids else '还没有识别记录。'

        def load_history(run_id):
            from deskvision.history import history_detail

            if not run_id:
                raise gr.Error('请先选择一条历史任务。')
            try:
                detail, note = history_detail(run_id)
            except ValueError as error:
                raise gr.Error(str(error)) from None
            return detail, note, '已加载历史详情。'

        def save_history(run_id, note):
            from deskvision.history import update_history_note

            if not run_id:
                raise gr.Error('请先选择一条历史任务。')
            try:
                message = update_history_note(run_id, note)
            except ValueError as error:
                raise gr.Error(str(error)) from None
            rows, choices, _ = refresh_history()
            return rows, choices, message

        def remove_history(run_id):
            from deskvision.history import delete_history

            if not run_id:
                raise gr.Error('请先选择一条历史任务。')
            try:
                message = delete_history(run_id)
            except ValueError as error:
                raise gr.Error(str(error)) from None
            rows, choices, _ = refresh_history()
            return rows, choices, None, '', message

        history_outputs = [history_table, history_id, history_status]
        history_refresh.click(refresh_history, outputs=history_outputs, queue=False)
        history_tab.select(refresh_history, outputs=history_outputs, queue=False)
        history_load.click(load_history, inputs=[history_id],
                           outputs=[history_detail, history_note, history_status], queue=False)
        history_save.click(save_history, inputs=[history_id, history_note],
                           outputs=history_outputs, queue=False)
        history_delete.click(remove_history, inputs=[history_id],
                             outputs=[history_table, history_id, history_detail,
                                      history_note, history_status], queue=False)
        demo.queue(max_size=16, default_concurrency_limit=1)
    return demo
