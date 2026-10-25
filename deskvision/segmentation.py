"""将原图检测框交给 SAM 2；保存独立实例掩膜，不重分类或修改检测数量。"""

from contextlib import nullcontext
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from .data import save_json, sha256

SAM2_CONFIG = 'configs/sam2.1/sam2.1_hiera_t.yaml'


def mask_geometry(mask: np.ndarray) -> dict:
    """像素面积独立于多边形面积；保留所有连通区域和孔洞层级。"""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return {'area_pixels': 0, 'bbox_xyxy': None, 'centroid_xy': None, 'contours': []}
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return {
        'area_pixels': int(len(xs)),
        'bbox_xyxy': [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        'centroid_xy': [float(xs.mean()), float(ys.mean())],
        'contours': [{'points_xy': contour[:, 0, :].tolist(), 'parent_index': int(hierarchy[0, i, 3])}
                     for i, contour in enumerate(contours)],
    }


def validate_prompts(image: np.ndarray, report: dict) -> None:
    """不猜测归一化或缩放关系，拒绝与原图不符的检测输入。"""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('分割输入必须是 uint8 BGR 三通道原图。')
    h, w = image.shape[:2]
    if report['image_size'] != {'width': w, 'height': h}:
        raise ValueError('检测记录尺寸与分割原图不一致。')
    seen = set()
    for detection in report['detections']:
        identity = detection['instance_id']
        if type(identity) is not int or identity <= 0 or identity in seen:
            raise ValueError('检测实例编号必须为互不重复的正整数。')
        seen.add(identity)
        box = np.asarray(detection['bbox_xyxy'], dtype=np.float32)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError('框提示必须为有限的原图 xyxy 坐标。')
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError('框提示越界或面积为零。')


class Sam2Segmenter:
    """每张图只编码一次，逐框解码，控制密集目标的显存；无检测时不加载模型。"""

    def __init__(self, weights: Path, device: str):
        self.weights = weights.resolve()
        if not self.weights.is_file() or self.weights.suffix.lower() != '.pt':
            raise FileNotFoundError('未找到本地 SAM 2.1 tiny 的 .pt 权重。')
        if device not in ('cpu', 'cuda:0'):
            raise ValueError('SAM 2 仅接受明确的 cpu 或 cuda:0 设备。')
        self.device = device
        self.predictor = None
        self.precision = 'float32'
        self.load_ms = 0.0

    def _load_predictor(self):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if self.device == 'cuda:0' and not torch.cuda.is_available():
            raise RuntimeError('指定了 CUDA 分割，但 GPU 不可用。')
        started = perf_counter()
        model = build_sam2(SAM2_CONFIG, str(self.weights), device=self.device, apply_postprocessing=False)
        self.predictor = SAM2ImagePredictor(model)
        if self.device == 'cuda:0' and torch.cuda.is_bf16_supported():
            self.precision = 'bfloat16'
        self._synchronize()
        self.load_ms = (perf_counter() - started) * 1000

    def _synchronize(self):
        if self.device == 'cuda:0':
            import torch
            torch.cuda.synchronize(self.device)

    def segment(self, image_bgr: np.ndarray, report: dict, destination: Path) -> dict:
        """返回新报告，原始检测框、分数与计数保持不变；异常交给调用者记录失败。"""
        import torch

        validate_prompts(image_bgr, report)
        result = deepcopy(report)
        h, w = image_bgr.shape[:2]
        info = {
            'status': 'skipped_no_detections', 'model': 'SAM 2.1 Hiera tiny',
            'model_config': SAM2_CONFIG, 'package_version': version('SAM-2'),
            'weights_name': self.weights.name, 'weights_sha256': sha256(self.weights),
            'device': self.device, 'precision': self.precision,
            'optional_cuda_postprocessing': False, 'multimask_output': False,
            'coordinate_system': '原图像素坐标，xyxy；掩膜框右下边界为开区间',
            'mask_encoding': '每实例独立原尺寸灰度 PNG，0=背景，255=前景；实例可重叠',
            'quality_note': 'SAM 2 的预测 IoU 质量估计，不是检测置信度，也不是实测分割准确率；不与检测分数相乘。',
            'counting_note': '数量沿用 YOLO 检测；空掩膜不扣除检测数；分割不修复漏检或错分类。',
            'prompt_count': len(report['detections']), 'nonempty_masks': 0, 'empty_masks': 0,
            'instances': [],
        }
        result['segmentation'] = info
        result['timings_ms']['sam2_load_and_transfer'] = 0.0
        result['timings_ms']['sam2_image_encoding'] = 0.0
        result['timings_ms']['sam2_mask_decoding'] = 0.0
        result['timings_ms']['sam2_artifact_write'] = 0.0
        if not report['detections']:
            return result
        if self.predictor is None:
            self._load_predictor()
            result['timings_ms']['sam2_load_and_transfer'] = round(self.load_ms, 3)
        info['precision'] = self.precision
        folder = destination / 'masks'
        folder.mkdir(parents=True, exist_ok=False)
        autocast = (torch.autocast('cuda', dtype=torch.bfloat16)
                    if self.precision == 'bfloat16' else nullcontext())
        encoding_ms = decoding_ms = write_ms = 0.0
        try:
            with torch.inference_mode(), autocast:
                self._synchronize()
                started = perf_counter()
                self.predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                self._synchronize()
                encoding_ms = (perf_counter() - started) * 1000
                for detection in report['detections']:
                    started = perf_counter()
                    masks, scores, _ = self.predictor.predict(
                        box=np.asarray(detection['bbox_xyxy'], dtype=np.float32),
                        multimask_output=False, normalize_coords=True,
                    )
                    self._synchronize()
                    decoding_ms += (perf_counter() - started) * 1000
                    masks, scores = np.asarray(masks), np.asarray(scores)
                    if masks.shape != (1, h, w) or scores.shape != (1,):
                        raise ValueError('SAM 2 输出形状与单框、单掩膜、原图尺寸不一致。')
                    if not np.isfinite(masks).all() or not np.isfinite(scores).all():
                        raise ValueError('SAM 2 返回非有限掩膜或质量分数。')
                    if not np.isin(masks, (0, 1)).all():
                        raise ValueError('SAM 2 输出必须是二值掩膜，不能把 logits 当作前景。')
                    mask = masks[0].astype(bool)
                    started = perf_counter()
                    identity = detection['instance_id']
                    stem = f'instance-{identity:04d}'
                    mask_file = folder / (stem + '.png')
                    contour_file = folder / (stem + '-contours.json')
                    geometry = mask_geometry(mask)
                    save_json(contour_file, {'instance_id': identity, 'image_size': {'width': w, 'height': h},
                                              'coordinate_system': '原图整数像素 xy；parent_index=-1 表示外层轮廓，其余指向父轮廓',
                                              'contours': geometry.pop('contours')})
                    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_file)
                    nonempty = geometry['area_pixels'] > 0
                    info['nonempty_masks' if nonempty else 'empty_masks'] += 1
                    info['instances'].append({
                        'instance_id': identity, 'category_id': detection['category_id'],
                        'status': 'ok' if nonempty else 'empty',
                        'prompt_bbox_xyxy': list(detection['bbox_xyxy']),
                        'sam2_predicted_iou': float(scores[0]),
                        'area_ratio': geometry['area_pixels'] / (h * w), **geometry,
                        'mask_file': mask_file.relative_to(destination).as_posix(),
                        'mask_sha256': sha256(mask_file),
                        'contours_file': contour_file.relative_to(destination).as_posix(),
                        'contours_sha256': sha256(contour_file),
                    })
                    write_ms += (perf_counter() - started) * 1000
        finally:
            # 释放当前图的特征，异常后也不能把旧图的嵌入用于下一张图。
            self.predictor.reset_predictor()
        info['status'] = 'completed'
        result['timings_ms'].update(sam2_image_encoding=round(encoding_ms, 3),
                                   sam2_mask_decoding=round(decoding_ms, 3),
                                   sam2_artifact_write=round(write_ms, 3))
        return result
