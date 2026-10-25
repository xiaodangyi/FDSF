"""检查第一阶段的文件、类别、依赖和本机路径；不训练、不下载。"""

from importlib import metadata
from pathlib import Path
import sys

import yaml

from project_config import ROOT, load_local_paths


def main() -> None:
    """检查准备状态；模型效果由后续独立实验评估。"""
    paths = load_local_paths()
    expected_python = paths['ai_cuda_root'] / 'python.exe'
    if sys.version_info < (3, 10) or Path(sys.executable).resolve() != expected_python.resolve():
        raise RuntimeError('请先加载 environment/activate-ai-cuda.ps1，使用配置的 ai-cuda 解释器。')
    print(f'解释器检查通过：Python {sys.version.split()[0]}')

    config = yaml.safe_load((ROOT / 'configs/classes.yaml').read_text(encoding='utf-8'))
    classes = config['classes']
    expected = ['杯子', '瓶子', '书', '橡皮', '螺丝刀', '螺丝', '螺母', '垫圈']
    if config['schema_version'] != 1 or [item['id'] for item in classes] != list(range(8)):
        raise ValueError('类别配置的版本或编号不符合首版约定。')
    if [item['name'] for item in classes] != expected or len({item['key'] for item in classes}) != 8:
        raise ValueError('类别名称或标识不符合首版约定。')
    if [item['coco_id'] for item in classes] != [41, 39, 73, None, None, None, None, None]:
        raise ValueError('预训练类别映射不符合首版约定。')
    print('八类名称、编号和 COCO 连续编号映射检查通过。')

    for name in ('README.md', 'docs/开发路线.md', 'docs/标注规范.md', 'docs/验收清单.md',
                 'docs/开发记录.md', 'docs/学习记录.md', 'environment/README.md'):
        if not (ROOT / name).is_file():
            raise FileNotFoundError(f'缺少项目文件：{name}')
    for key in ('yolo_weights', 'sam2_weights', 'smoke_image'):
        if not paths[key].is_file():
            raise FileNotFoundError(f'配置的文件不存在：{key}')
    if not paths['smoke_dataset'].is_dir():
        raise FileNotFoundError('配置的环境自检数据目录不存在：smoke_dataset')
    print('项目文档、已有权重和环境自检素材路径检查通过。')

    expected_versions = {
        'torch': '2.7.1+cu128', 'torchvision': '0.22.1+cu128',
        'ultralytics': '8.3.203', 'SAM-2': '1.0', 'opencv-python': '4.11.0.86',
        'numpy': '1.26.4', 'gradio': '5.49.1', 'openai': '2.16.0', 'PyYAML': '6.0.3',
    }
    for package, expected_version in expected_versions.items():
        actual = metadata.version(package)
        if actual != expected_version:
            raise RuntimeError(f'版本与已验证环境不同：{package}，当前 {actual}，预期 {expected_version}')
        print(f'依赖版本符合记录：{package} {actual}')
    print('第一阶段准备检查通过。本检查不代表模型推理、训练或八类识别效果通过。')


if __name__ == '__main__':
    main()
