"""启动新版桌面物品识别工作台，运行资源均保存在项目中。"""

import argparse
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    cache = root / '.local' / 'web-temp'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['TEMP'] = str(cache)
    os.environ['TMP'] = str(cache)
    os.environ['GRADIO_TEMP_DIR'] = str(root / 'runs' / 'web-cache')
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

    from .app import build_demo
    from deskvision.ui import UI_OUTPUT_ROOT

    parser = argparse.ArgumentParser(description='启动桌面物品识别工作台。')
    parser.add_argument('--port', type=int, default=7860, help='本机端口，默认7860')
    args = parser.parse_args()
    build_demo().launch(server_name='127.0.0.1', server_port=args.port, share=False,
                        allowed_paths=[str(UI_OUTPUT_ROOT)],
                        blocked_paths=[str(root / '.local')],
                        max_file_size='32mb', show_error=False)


if __name__ == '__main__':
    main()
