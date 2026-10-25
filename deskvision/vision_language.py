"""原图独立交给视觉大模型；不导入 YOLO、SAM 或训练模块。"""

import base64
from collections import Counter
from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import httpx
from openai import (APIConnectionError, APIStatusError, APITimeoutError,
                    AuthenticationError, BadRequestError, OpenAI, PermissionDeniedError,
                    RateLimitError)
from PIL import Image, UnidentifiedImageError

from .config import ROOT, load_categories
from .vlm_config import VlmError, VlmSettings

CLASS_KEYS = ('cup', 'bottle', 'book', 'eraser', 'screwdriver', 'screw', 'nut', 'washer')
PROMPT_PATH = ROOT / 'configs/vlm-system-prompt.txt'
PROMPT_VERSION = 'desk8-independent-v1'


def save_record(path: Path, value: dict) -> None:
    """严格 JSON，先写临时文件，避免半份结果被当成有效输出。"""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def prepare_image(source: Path) -> tuple[bytes, dict]:
    """保留原文件字节；不读取标签、检测结果，不裁剪或缩放图片。"""
    if not source.is_file():
        raise VlmError('invalid_image', '输入图片不存在。')
    if not 0 < source.stat().st_size <= 32 * 1024 * 1024:
        raise VlmError('invalid_image', '图像必须非空且不超过 32 MiB；本入口不自动上传到 Files API。')
    content = source.read_bytes()
    try:
        with Image.open(BytesIO(content)) as image:
            mime = {'JPEG': 'image/jpeg', 'PNG': 'image/png', 'WEBP': 'image/webp', 'GIF': 'image/gif'}.get(image.format)
            if not mime or getattr(image, 'n_frames', 1) != 1 or max(image.size) > 8192:
                raise VlmError('invalid_image', '仅接受单帧 JPEG、PNG、WebP、GIF，单边不超过 8192 像素。')
            width, height = image.size
            orientation = image.getexif().get(274, 1)
            image.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise VlmError('invalid_image', '图像解码失败，不能把损坏文件发送给视觉模型。') from None
    return content, {'file_name': source.name, 'sha256': sha256(content).hexdigest(),
                     'mime_type': mime, 'bytes': len(content),
                     'width': width, 'height': height, 'exif_orientation': orientation,
                     'processing': '原文件字节，不裁剪、不缩放、不叠加检测框'}


def build_request(content: bytes, mime: str, prompt: str, settings: VlmSettings) -> dict:
    """单次独立请求的全部消息；接口不接受检测记录、权重或历史对话。"""
    return {
        'model': settings.model, 'max_tokens': settings.max_tokens,
        'response_format': {'type': 'json_object'},
        'extra_body': {'thinking': {'type': 'disabled'}},
        'messages': [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': '请独立识别这张原图中的八类物品，逐件列出并核对数量，严格返回要求的 JSON。'},
                {'type': 'image_url', 'image_url': {
                    'url': f'data:{mime};base64,' + base64.b64encode(content).decode('ascii'), 'detail': 'high'}},
            ]},
        ],
    }


def request_completion(payload: dict, settings: VlmSettings):
    """不自动重试或跟随重定向；只返回安全的错误分类，不回显服务正文。"""
    try:
        with OpenAI(api_key=settings.api_key, base_url=settings.base_url,
                    max_retries=0, timeout=settings.timeout_seconds,
                    http_client=httpx.Client(timeout=settings.timeout_seconds, follow_redirects=False)) as client:
            return client.chat.completions.create(**payload)
    except APITimeoutError:
        raise VlmError('timeout', 'DeepSeek 请求超时，本次没有自动重试。') from None
    except AuthenticationError:
        raise VlmError('authentication', 'DeepSeek 认证失败，请检查本地 API Key。', 401) from None
    except PermissionDeniedError:
        raise VlmError('permission', 'DeepSeek 拒绝访问，请检查账号与模型权限。', 403) from None
    except RateLimitError:
        raise VlmError('rate_limit', 'DeepSeek 限流或额度受限，本次没有自动重试。', 429) from None
    except BadRequestError:
        raise VlmError('bad_request', 'DeepSeek 拒绝请求，请检查图像、参数与模型权限。', 400) from None
    except APIConnectionError:
        raise VlmError('connection', '无法连接 DeepSeek，请检查网络；本次没有自动重试。') from None
    except APIStatusError as error:
        raise VlmError('api_status', 'DeepSeek 返回服务错误，本次没有自动重试。', error.status_code) from None


def parse_output(text: str) -> dict:
    """JSON 模式仅保证语法；额外核对类别、逐件清单和八类计数是否一致。"""
    def no_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('重复字段')
            result[key] = value
        return result

    def no_constants(value):
        raise ValueError('非有限 JSON 常量')

    if not isinstance(text, str) or not text.strip() or len(text) > 200000:
        raise VlmError('invalid_output', '视觉模型回答为空或过长，不能当作零检测。')
    try:
        output = json.loads(text, object_pairs_hook=no_duplicate_keys, parse_constant=no_constants)
    except (ValueError, RecursionError):
        raise VlmError('invalid_output', '视觉模型回答不是符合要求的唯一字段 JSON，不自动猜测或修补。') from None
    if not isinstance(output, dict) or set(output) != {'items', 'counts', 'notes'}:
        raise VlmError('invalid_output', '视觉模型 JSON 的顶层字段不符合约定。')
    items, counts, notes = output['items'], output['counts'], output['notes']
    if not isinstance(items, list) or len(items) > 300:
        raise VlmError('invalid_output', '逐件清单必须是最多 300 项的数组。')
    if not isinstance(notes, str) or len(notes) > 2000:
        raise VlmError('invalid_output', '说明必须是最多 2000 字符的字符串。')
    if (not isinstance(counts, dict) or set(counts) != set(CLASS_KEYS)
            or any(type(value) is not int or not 0 <= value <= 300 for value in counts.values())):
        raise VlmError('invalid_output', '八类数量必须齐全，且均为 0～300 的整数。')
    for item in items:
        if (not isinstance(item, dict) or set(item) != {'category_key', 'location', 'uncertain'}
                or not isinstance(item['category_key'], str) or item['category_key'] not in CLASS_KEYS
                or not isinstance(item['location'], str) or not item['location'].strip() or len(item['location']) > 200
                or type(item['uncertain']) is not bool):
            raise VlmError('invalid_output', '逐件清单中的类别、位置或存疑标记不符合约定。')
    observed = Counter(item['category_key'] for item in items)
    if any(counts[key] != observed[key] for key in CLASS_KEYS):
        raise VlmError('inconsistent_counts', '大模型逐件清单与汇总数量不一致，不能作为有效清点结果。')
    return output


def recognize(source: Path, destination: Path, settings: VlmSettings) -> dict:
    """供命令行和未来 UI 共用；失败报告没有伪造的零计数，也不回退到 YOLO。"""
    report = {'schema_version': 1, 'status': 'failed',
              'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'route': 'deepseek_vision_independent', 'model': settings.public(),
              'independence': '只输入原图、类别定义与固定提示；无 YOLO/SAM 结果、标签或训练反馈。',
              'timings_ms': {}, 'artifacts': {'result': 'result.json'}}
    stage = 'input'
    started = perf_counter()
    try:
        content, image_info = prepare_image(source)
        report['input'] = image_info
        categories = load_categories()
        if tuple(category.key for category in categories) != CLASS_KEYS:
            raise VlmError('invalid_config', '类别配置与视觉提示版本不一致。')
        prompt = PROMPT_PATH.read_text(encoding='utf-8')
        report['prompt'] = {'version': PROMPT_VERSION, 'sha256': sha256(prompt.encode('utf-8')).hexdigest()}
        report['category_config_sha256'] = sha256((ROOT / 'configs/classes.yaml').read_bytes()).hexdigest()
        payload = build_request(content, image_info['mime_type'], prompt, settings)
        report['request_sha256'] = sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
        report['versions'] = {package: version(package) for package in ('openai', 'httpx', 'Pillow')}
        save_record(destination / 'request-summary.json', {
            'route': report['route'], 'input': image_info, 'model': settings.public(),
            'prompt': report['prompt'], 'request_sha256': report['request_sha256'],
            'message_roles': ['system', 'user'], 'image_count': 1,
            'prior_results_included': False, 'api_key_saved': False, 'base64_saved': False,
        })
        report['artifacts']['request_summary'] = 'request-summary.json'
        report['timings_ms']['input_preparation'] = round((perf_counter() - started) * 1000, 3)
        stage = 'request'
        request_started = perf_counter()
        try:
            completion = request_completion(payload, settings)
        finally:
            report['timings_ms']['api_wall'] = round((perf_counter() - request_started) * 1000, 3)
        stage = 'validation'
        if len(completion.choices) != 1:
            raise VlmError('invalid_response', 'DeepSeek 必须返回一个完整回答。')
        choice = completion.choices[0]
        report['response'] = {'id': completion.id, 'returned_model': completion.model,
                              'finish_reason': choice.finish_reason}
        if completion.usage is not None:
            report['usage'] = {key: getattr(completion.usage, key, None)
                               for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
        if choice.finish_reason != 'stop' or choice.message.refusal or choice.message.tool_calls:
            raise VlmError('incomplete_response', '模型回答被截断、拒绝或转为工具调用，不能当作识别结果。')
        text = choice.message.content
        if isinstance(text, str) and len(text) <= 200000:
            (destination / 'response.txt').write_text(text, encoding='utf-8')
            report['artifacts']['response_text'] = 'response.txt'
        parsed = parse_output(text)
        by_key = {category.key: category for category in categories}
        report['items'] = [
            {'instance_id': i, 'category_id': by_key[item['category_key']].id,
             'category_key': item['category_key'], 'category_name': by_key[item['category_key']].name,
             'location_text': item['location'], 'uncertain': item['uncertain']}
            for i, item in enumerate(parsed['items'], 1)
        ]
        report['summary'] = [
            {'category_id': category.id, 'category_key': category.key, 'category_name': category.name,
             'count': parsed['counts'][category.key], 'status': 'reported', 'status_text': '大模型独立识别，待核对'}
            for category in categories
        ]
        report['total_items'] = len(report['items'])
        report['uncertain_items'] = sum(item['uncertain'] for item in report['items'])
        report['notes'] = parsed['notes']
        report['message'] = ('未识别到八类目标，请核对漏检。' if not report['items'] else '大模型独立清单，请人工核对错类、漏检与多计。')
        report['counting_note'] = '数量只由本次大模型逐件清单汇总；无检测置信度、像素框或分割准确率。'
        report['timing_note'] = 'API 耗时含网络和服务端生成，与本地纯模型推理耗时不可直接等同。'
        report['status'] = 'ok'
    except VlmError as error:
        report['error'] = {'stage': stage, **error.as_dict()}
    except OSError:
        report['error'] = {'stage': stage, 'code': 'local_io', 'message': '读取或保存本地文件失败。'}
    except Exception:
        # 不回显任意异常的字符串，防止第三方异常带出凭据或整段图片请求。
        report['error'] = {'stage': stage, 'code': 'unexpected', 'message': '视觉路线发生异常，请检查环境与接口格式。'}
    report['timings_ms']['total_wall'] = round((perf_counter() - started) * 1000, 3)
    save_record(destination / 'result.json', report)
    return report
