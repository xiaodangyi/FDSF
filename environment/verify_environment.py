"""检查依赖、GPU 算子及真实的 YOLOv8 到 SAM 2 推理，不进行训练。"""
from pathlib import Path
import importlib.metadata as metadata
import json
import os
import sys
from project_config import ROOT, load_local_paths

paths = load_local_paths()
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'environment' / 'ultralytics-settings'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'environment' / 'matplotlib-cache'))
os.environ.setdefault('YOLO_OFFLINE', 'true')

import cv2
import gradio
import numpy as np
import openai
import torch
import torchvision
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO

report = {
    'python': sys.version,
    'executable': sys.executable,
    'torch_path': torch.__file__,
    'packages': {name: metadata.version(name) for name in [
        'torch', 'torchvision', 'ultralytics', 'SAM-2', 'opencv-python',
        'numpy', 'pandas', 'Pillow', 'matplotlib', 'gradio', 'openai',
        'hydra-core', 'iopath', 'pycocotools',
    ]},
}
assert sys.version_info >= (3, 10)
assert torch.cuda.is_available(), 'CUDA 当前不可用'
report['gpu'] = torch.cuda.get_device_name(0)
report['cuda_runtime'] = torch.version.cuda
assert torch.__version__ == '2.7.1+cu128'
assert torchvision.__version__ == '0.22.1+cu128'

a = torch.arange(16, dtype=torch.float32, device='cuda').reshape(4, 4)
b = a.cpu()
assert torch.allclose((a @ a.T).cpu(), b @ b.T)
boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=torch.float32, device='cuda')
scores = torch.tensor([0.9, 0.8], device='cuda')
assert torchvision.ops.nms(boxes, scores, 0.5).cpu().tolist() == [0]
train_probe = torch.nn.Conv2d(3, 4, 3).cuda()
loss = train_probe(torch.ones(1, 3, 16, 16, device='cuda')).square().mean()
loss.backward()
assert torch.isfinite(train_probe.weight.grad).all()
report['gpu_matrix_nms_backward'] = 'passed'
del a, b, boxes, scores, train_probe, loss

image_path = paths['smoke_image']
image_bgr = cv2.imread(str(image_path))
assert image_bgr is not None, '无法读取自检图片'
rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
yolo = YOLO(str(paths['yolo_weights']))
result = yolo.predict(source=image_bgr, device=0, imgsz=640, conf=0.20, verbose=False, save=False)[0]
count = len(result.boxes)
assert count > 0, '自检图没有检出目标，无法测试 SAM 框提示'
prompt_box = result.boxes.xyxy[0].detach().cpu().numpy()
report['yolo'] = {'image': str(image_path), 'detections': count,
                  'labels': [result.names[int(c)] for c in result.boxes.cls.cpu().tolist()]}
del result, yolo
torch.cuda.empty_cache()

model = build_sam2(
    'configs/sam2.1/sam2.1_hiera_t.yaml',
    str(paths['sam2_weights']),
    device='cuda', apply_postprocessing=False,
)
predictor = SAM2ImagePredictor(model)
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    predictor.set_image(rgb)
    masks, quality, _ = predictor.predict(box=prompt_box[None, :], multimask_output=False)
assert masks.shape[-2:] == rgb.shape[:2]
assert np.isfinite(quality).all()
assert masks.any(), 'SAM 自检掩膜为空'
report['sam2'] = {'mask_shape': list(masks.shape), 'quality': quality.tolist(),
                  'foreground_pixels': int(masks.sum()), 'optional_cuda_postprocessing': False}
report['peak_gpu_allocated_gib'] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
report['status'] = 'passed'
dest = ROOT / 'environment' / 'verification.json'
dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
print('GPU 算子及检测到分割的环境自检通过；不代表八类识别效果。')
