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

    from deskvision.api import create_app

    parser = argparse.ArgumentParser(description='启动桌面物品识别工作台。')
    parser.add_argument('--port', type=int, default=8000, help='本机端口，默认8000')
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(), host='127.0.0.1', port=args.port, reload=False)


if __name__ == '__main__':
    main()
