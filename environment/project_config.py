"""读取不入库的本机路径配置，供准备检查脚本共用。"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / '.local' / 'config.json'


def load_local_paths() -> dict[str, Path]:
    """相对路径从项目根目录解释，不依赖终端所在目录。"""
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError('缺少 .local/config.json，请参考 configs/local.example.json 填写。')
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict):
        raise ValueError('本机配置必须是 JSON 对象。')
    paths = {}
    for key in ('ai_cuda_root', 'yolo_weights', 'sam2_weights', 'smoke_image', 'smoke_dataset'):
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'本机配置缺少有效路径：{key}')
        path = Path(value).expanduser()
        paths[key] = (path if path.is_absolute() else ROOT / path).resolve()
    return paths
