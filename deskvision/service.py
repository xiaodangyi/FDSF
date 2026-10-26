"""Shared route execution for HTTP and Gradio; no page or task state here."""

from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
from threading import BoundedSemaphore


_execution_slot = BoundedSemaphore(1)
logger = logging.getLogger(__name__)


def run_routes(route: str, image_path, parameters: dict, *, output_root: Path | None = None,
               expected_sha256: str | None = None, on_start=None) -> dict:
    """Run independent routes on one original; preserve a successful sibling on failure."""
    from . import ui as core

    if route not in ('yolo', 'deepseek', 'both'):
        raise ValueError('未知识别路线。')

    def failed(selected, code, message):
        return {'status': 'failed', 'route': selected,
                'error': {'code': code, 'message': message}}

    def execute(selected):
        try:
            if selected == 'yolo':
                report = core.run_yolo_route(image_path, **parameters,
                                             output_root=output_root)
            else:
                report = core.run_vlm_route(image_path, output_root=output_root)
            if (report.get('status') == 'ok' and
                    report.get('input', {}).get('sha256') != fingerprint):
                return failed(selected, 'input_mismatch', '路线结果与上传原图的 SHA256 不一致，请重新核对。')
            return report
        except Exception as error:
            logger.error('Route %s raised %s', selected, type(error).__name__)
            return failed(selected, 'route_error', '该路线处理失败，请检查模型配置和本机运行日志。')

    selected_routes = ('yolo', 'deepseek') if route == 'both' else (route,)
    with _execution_slot:
        if on_start is not None:
            on_start()
        try:
            _, image_info = core.prepare_image(core._path_value(image_path))
            fingerprint = expected_sha256 or image_info['sha256']
            if fingerprint != image_info['sha256']:
                raise ValueError('原图已改变。')
        except Exception:
            return {selected: failed(selected, 'invalid_image', '原图无效或已改变，请重新上传。')
                    for selected in selected_routes}
        if route == 'both':
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {selected: pool.submit(execute, selected) for selected in selected_routes}
                return {selected: future.result() for selected, future in futures.items()}
        return {route: execute(route)}
