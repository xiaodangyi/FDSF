"""DeepSeek 图像路线的本地配置；密钥与可入库参数分开。"""

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path

from dotenv import dotenv_values

from .config import ROOT


class VlmError(Exception):
    """仅携带可以安全显示的错误，不保存服务返回正文或请求凭据。"""

    def __init__(self, code: str, message: str, http_status: int | None = None):
        super().__init__(message)
        self.code = code
        self.http_status = http_status

    def as_dict(self) -> dict:
        return {'code': self.code, 'message': str(self), 'http_status': self.http_status}


@dataclass(frozen=True)
class VlmSettings:
    api_key: str = field(repr=False)
    provider: str = 'deepseek'
    model: str = 'deepseek-flash'
    base_url: str = 'https://api.deepseek.com'
    timeout_seconds: float = 60
    max_tokens: int = 4096

    def __post_init__(self):
        if not isinstance(self.api_key, str) or not self.api_key.strip() or not self.api_key.isascii() or any(
                char.isspace() for char in self.api_key):
            raise VlmError('missing_api_key', '请在 .local/deepseek.env 填写有效的 DEEPSEEK_API_KEY，或设置同名环境变量。')
        if self.provider != 'deepseek' or self.model != 'deepseek-flash':
            raise VlmError('unsupported_model', '本阶段仅验证 DeepSeek 官方 deepseek-flash 图像接口，不自动换模型。')
        if self.base_url not in ('https://api.deepseek.com', 'https://api.deepseek.com/v1'):
            raise VlmError('invalid_endpoint', '本阶段仅允许 DeepSeek 官方 HTTPS 地址，不能把密钥发到其他服务。')
        if (type(self.timeout_seconds) not in (float, int) or not math.isfinite(self.timeout_seconds)
                or not 1 <= self.timeout_seconds <= 120):
            raise VlmError('invalid_config', 'timeout_seconds 必须在 1～120 秒之间。')
        if type(self.max_tokens) is not int or not 512 <= self.max_tokens <= 16384:
            raise VlmError('invalid_config', 'max_tokens 必须是 512～16384 的整数。')

    def public(self) -> dict:
        """白名单导出，禁止通过 dataclasses.asdict 导出整个配置。"""
        return {'provider': self.provider, 'model': self.model, 'base_url': self.base_url,
                'timeout_seconds': self.timeout_seconds, 'max_tokens': self.max_tokens,
                'max_retries': 0, 'thinking': 'disabled', 'response_format': 'json_object',
                'image_detail': 'high'}


def load_vlm_settings(config_path: Path | None = None, secret_path: Path | None = None) -> VlmSettings:
    """默认参数可直接用；自定义公开参数放 .local/vlm.json，密钥不混入该文件。"""
    config_path = config_path or ROOT / '.local/vlm.json'
    secret_path = secret_path or ROOT / '.local/deepseek.env'
    config = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding='utf-8-sig'))
        except (OSError, ValueError):
            raise VlmError('invalid_config', '无法读取本地视觉模型 JSON 配置。') from None
        if not isinstance(config, dict) or set(config) - {
                'provider', 'model', 'base_url', 'timeout_seconds', 'max_tokens'}:
            raise VlmError('invalid_config', '视觉模型配置有未知字段；密钥请单独放入 deepseek.env。')
    key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    if not key and secret_path.is_file():
        try:
            key = (dotenv_values(secret_path, encoding='utf-8-sig', interpolate=False).get('DEEPSEEK_API_KEY') or '').strip()
        except (OSError, ValueError):
            raise VlmError('missing_api_key', '无法读取本地 DeepSeek 密钥文件。') from None
    return VlmSettings(api_key=key, **config)
