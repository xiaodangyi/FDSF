"""独立视觉大模型命令行；正常执行会将所选原图发送至 DeepSeek。"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
from uuid import uuid4
from zoneinfo import ZoneInfo

from .config import ROOT
from .vision_language import recognize, save_record
from .vlm_config import load_vlm_settings, VlmError


def main() -> int:
    parser = argparse.ArgumentParser(description='独立调用 DeepSeek 图像理解，保存八类清单；不运行 YOLO、SAM 或训练。')
    parser.add_argument('--image', help='待发送给 DeepSeek 的原图，支持中文和空格')
    parser.add_argument('--output', help='输出根目录，默认 runs/vlm，每次建立独立子目录')
    parser.add_argument('--config', help='不含密钥的本地 JSON 配置，默认 .local/vlm.json 或内置参数')
    parser.add_argument('--check-config', action='store_true', help='仅检查本机参数和密钥是否就绪，不联网、不显示密钥')
    args = parser.parse_args()
    if not args.image and not args.check_config:
        parser.error('请提供 --image，或使用 --check-config。')
    run_dir = None
    try:
        config_path = Path(args.config).expanduser().resolve() if args.config else None
        if config_path is not None and not config_path.is_file():
            raise VlmError('invalid_config', '显式指定的视觉模型配置文件不存在。')
        settings = load_vlm_settings(config_path)
        if args.check_config:
            print(f'配置就绪：{settings.provider} / {settings.model} / {settings.base_url}；密钥已读取但不显示。本次未联网。')
            return 0
        output_root = Path(args.output).expanduser().resolve() if args.output else ROOT / 'runs/vlm'
        run_dir = output_root / (datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-') + uuid4().hex[:8])
        run_dir.mkdir(parents=True, exist_ok=False)
        print('将原图发送至 DeepSeek deepseek-flash 独立识别；不发送 YOLO/SAM 结果，不训练，不自动重试。', flush=True)
        report = recognize(Path(args.image).expanduser().resolve(), run_dir, settings)
        print(f'结果目录：{run_dir}', flush=True)
        if report['status'] != 'ok':
            print(f"识别失败（{report['error']['code']}）：{report['error']['message']}", file=sys.stderr)
            return 1
        for item in report['summary']:
            print(f"{item['category_name']}：{item['count']}")
        print(f"存疑实例：{report['uncertain_items']}；API 耗时：{report['timings_ms']['api_wall']} ms")
        print(report['message'])
        return 0
    except VlmError as error:
        print(f'配置失败（{error.code}）：{error}', file=sys.stderr)
        return 1
    except Exception:
        if run_dir is not None and run_dir.is_dir():
            try:
                save_record(run_dir / 'result.json', {'schema_version': 1, 'status': 'failed',
                            'route': 'deepseek_vision_independent',
                            'error': {'code': 'local_io', 'message': '保存结果或执行入口失败。'}})
            except OSError:
                pass
        print('执行失败，请检查环境或本地文件权限；未打印请求与凭据。', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
