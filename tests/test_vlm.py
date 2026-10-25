"""检查独立原图请求、计数契约与真实 SDK 的失败处理，不访问网络。"""

import base64
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from PIL import Image

from deskvision.vision_language import CLASS_KEYS, build_request, parse_output, prepare_image, recognize
from deskvision.vlm_config import load_vlm_settings, VlmError, VlmSettings


def answer():
    return {'items': [{'category_key': 'cup', 'location': '左侧', 'uncertain': False}],
            'counts': {key: int(key == 'cup') for key in CLASS_KEYS}, 'notes': ''}


def completion(content=None, finish='stop', refusal=None):
    return {'id': 'mock-response-id', 'object': 'chat.completion', 'created': 1792760400,
            'model': 'deepseek-flash', 'choices': [
                {'index': 0, 'finish_reason': finish,
                 'message': {'role': 'assistant', 'content': content, 'refusal': refusal}}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 80, 'total_tokens': 180}}


class VlmContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / '原图.png'
        Image.new('RGB', (17, 13), (100, 20, 30)).save(self.source)
        self.settings = VlmSettings(api_key='test-secret-never-print')
        self.requests = []

    def run_with_transport(self, handler, name='result'):
        # 客户端在打补丁前创建，保留 SDK 的序列化和状态码处理行为。
        client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        destination = self.root / name
        destination.mkdir()
        # 只替换本模块的依赖引用，避免改动 SDK 内部的 httpx.Client 类型检查。
        with patch('deskvision.vision_language.httpx') as dependency:
            dependency.Client.return_value = client
            result = recognize(self.source, destination, self.settings)
        return result, destination

    def test_original_file_bytes_and_real_mime_are_preserved(self):
        disguised = self.root / '扩展名不同.jpg'
        disguised.write_bytes(self.source.read_bytes())
        content, info = prepare_image(disguised)
        self.assertEqual(content, self.source.read_bytes())
        self.assertEqual(info['mime_type'], 'image/png')
        self.assertEqual((info['width'], info['height']), (17, 13))
        invalid = self.root / '损坏.png'
        invalid.write_bytes(b'not an image')
        with self.assertRaises(VlmError):
            prepare_image(invalid)
        animated = self.root / '动画.gif'
        Image.new('RGB', (3, 3), 'red').save(
            animated, save_all=True, append_images=[Image.new('RGB', (3, 3), 'blue')])
        with self.assertRaises(VlmError):
            prepare_image(animated)

    def test_request_contains_one_original_image_and_no_prior_results(self):
        payload = build_request(b'original bytes', 'image/png', '固定 json 格式', self.settings)
        self.assertEqual([message['role'] for message in payload['messages']], ['system', 'user'])
        self.assertEqual(payload['messages'][0]['content'], '固定 json 格式')
        blocks = payload['messages'][1]['content']
        self.assertEqual([block['type'] for block in blocks], ['text', 'image_url'])
        self.assertEqual(base64.b64decode(blocks[1]['image_url']['url'].split(',')[1]), b'original bytes')
        self.assertNotIn(self.settings.api_key, json.dumps(payload))
        self.assertEqual(set(payload), {'model', 'max_tokens', 'response_format', 'extra_body', 'messages'})

    def test_valid_empty_and_nonempty_outputs_are_distinguished(self):
        self.assertEqual(parse_output(json.dumps(answer())), answer())
        empty = {'items': [], 'counts': dict.fromkeys(CLASS_KEYS, 0), 'notes': '未见可辨认的八类物品'}
        self.assertEqual(parse_output(json.dumps(empty)), empty)
        for text in ('', ' ', 'null', '[]', '```json\n{}\n```', '{"items":[],"items":[]}'):
            with self.subTest(text=text), self.assertRaises(VlmError):
                parse_output(text)

    def test_invalid_types_unknown_classes_and_inconsistent_counts_fail(self):
        values = []
        for bad_count in (True, -1, 1.0, '1', 301):
            value = answer()
            value['counts']['cup'] = bad_count
            values.append(value)
        value = answer()
        value['items'][0]['category_key'] = 'unknown'
        values.append(value)
        value = answer()
        value['items'][0]['uncertain'] = 'false'
        values.append(value)
        value = answer()
        value['counts']['cup'] = 2
        values.append(value)
        value = answer()
        del value['counts']['washer']
        values.append(value)
        value = answer()
        value['confidence'] = .99
        values.append(value)
        value = answer()
        value['notes'] = float('nan')
        values.append(value)
        for index, value in enumerate(values):
            with self.subTest(index=index), self.assertRaises(VlmError):
                parse_output(json.dumps(value))

    def test_real_sdk_serialization_and_saved_result_have_independent_counts(self):
        def handler(request):
            self.requests.append(request)
            self.assertEqual(str(request.url), 'https://api.deepseek.com/chat/completions')
            wire = json.loads(request.content)
            self.assertEqual(wire['response_format'], {'type': 'json_object'})
            self.assertEqual(wire['thinking'], {'type': 'disabled'})
            data_url = wire['messages'][1]['content'][1]['image_url']['url']
            self.assertEqual(base64.b64decode(data_url.split(',')[1]), self.source.read_bytes())
            self.assertEqual(len(wire['messages']), 2)
            return httpx.Response(200, json=completion(json.dumps(answer(), ensure_ascii=False)))
        report, destination = self.run_with_transport(handler)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(report['status'], 'ok')
        self.assertEqual(report['total_items'], 1)
        self.assertEqual([row['count'] for row in report['summary']], [1, 0, 0, 0, 0, 0, 0, 0])
        self.assertNotIn('detections', report)
        self.assertNotIn('bbox_xyxy', report['items'][0])
        self.assertNotIn('detection_confidence', report['items'][0])
        self.assertEqual(report['usage']['total_tokens'], 180)
        for path in destination.iterdir():
            self.assertNotIn(self.settings.api_key, path.read_text(encoding='utf-8'))
        saved = json.loads((destination / 'result.json').read_text(encoding='utf-8'))
        self.assertEqual(saved, report)

    def test_api_errors_do_not_retry_leak_credentials_or_forge_zero_counts(self):
        for status, expected in ((400, 'bad_request'), (401, 'authentication'), (402, 'api_status'),
                                 (403, 'permission'), (429, 'rate_limit'), (503, 'api_status'), (307, 'api_status')):
            calls = []
            def handler(request):
                calls.append(request)
                return httpx.Response(status, json={'error': {'message': self.settings.api_key}},
                                      headers={'Location': 'https://example.invalid/redirect'})
            with self.subTest(status=status):
                report, destination = self.run_with_transport(handler, f'failure-{status}')
                self.assertEqual(len(calls), 1)
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(report['error']['code'], expected)
                self.assertNotIn('summary', report)
                self.assertNotIn('items', report)
                self.assertNotIn(self.settings.api_key, (destination / 'result.json').read_text(encoding='utf-8'))

    def test_timeout_fails_once_and_missing_image_never_sends(self):
        calls = []
        def handler(request):
            calls.append(request)
            raise httpx.ReadTimeout('包含凭据的第三方异常不得回显', request=request)
        report, _ = self.run_with_transport(handler)
        self.assertEqual(len(calls), 1)
        self.assertEqual(report['error']['code'], 'timeout')
        destination = self.root / 'missing'
        destination.mkdir()
        with patch('deskvision.vision_language.request_completion') as request:
            report = recognize(self.root / '不存在.png', destination, self.settings)
        request.assert_not_called()
        self.assertEqual(report['error']['code'], 'invalid_image')
        self.assertNotIn('summary', report)

    def test_truncation_refusal_empty_content_and_bad_json_are_failures(self):
        values = [completion(json.dumps(answer()), finish='length'),
                  completion(None, refusal='拒绝'), completion(None), completion('not json')]
        for index, value in enumerate(values):
            with self.subTest(index=index):
                report, _ = self.run_with_transport(lambda request: httpx.Response(200, json=value), f'invalid-{index}')
                self.assertEqual(report['status'], 'failed')
                self.assertNotIn('summary', report)

    def test_local_config_separates_key_and_rejects_wrong_models_and_hosts(self):
        self.assertNotIn(self.settings.api_key, repr(self.settings))
        self.assertNotIn(self.settings.api_key, json.dumps(self.settings.public()))
        config = self.root / 'local.json'
        secret = self.root / 'deepseek.env'
        secret.write_text('DEEPSEEK_API_KEY=test-local-key\n', encoding='utf-8')
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(load_vlm_settings(config, secret).api_key, 'test-local-key')
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test-env-key'}, clear=True):
            self.assertEqual(load_vlm_settings(config, secret).api_key, 'test-env-key')
            for value in ({'api_key': 'must-not-be-in-json'}, {'base_url': 'https://example.invalid'},
                          {'model': 'deepseek-chat'}, {'timeout_seconds': True}, {'max_tokens': '4096'}):
                config.write_text(json.dumps(value), encoding='utf-8')
                with self.subTest(config=value), self.assertRaises(VlmError):
                    load_vlm_settings(config, secret)

    def test_cli_config_check_does_not_call_api_or_show_key(self):
        from deskvision.vlm import main
        output = StringIO()
        with patch('sys.argv', ['deskvision.vlm', '--check-config']), \
                patch('deskvision.vlm.load_vlm_settings', return_value=self.settings), \
                patch('deskvision.vlm.recognize') as call, redirect_stdout(output), redirect_stderr(output):
            self.assertEqual(main(), 0)
        call.assert_not_called()
        self.assertNotIn(self.settings.api_key, output.getvalue())

    def test_cli_failures_return_nonzero_without_forging_success(self):
        from deskvision.vlm import main
        output = StringIO()
        with patch('sys.argv', ['deskvision.vlm', '--image', str(self.source), '--output', str(self.root / 'cli')]), \
                patch('deskvision.vlm.load_vlm_settings', return_value=self.settings), \
                patch('deskvision.vlm.recognize', return_value={
                    'status': 'failed', 'error': {'code': 'timeout', 'message': '请求超时'}}), \
                redirect_stdout(output), redirect_stderr(output):
            self.assertEqual(main(), 1)
        with patch('sys.argv', ['deskvision.vlm', '--check-config', '--config', str(self.root / 'not-found.json')]), \
                patch('deskvision.vlm.load_vlm_settings') as load, redirect_stdout(output), redirect_stderr(output):
            self.assertEqual(main(), 1)
        load.assert_not_called()
        self.assertNotIn(self.settings.api_key, output.getvalue())


if __name__ == '__main__':
    unittest.main()
