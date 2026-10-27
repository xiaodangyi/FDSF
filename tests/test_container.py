"""Container configuration and secret-file contracts; no Docker or remote requests."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import yaml

from deskvision.config import ROOT
from deskvision.vlm_config import load_vlm_settings
from environment import prepare_stage3_s6 as prepare


class ContainerTests(unittest.TestCase):
    def test_only_loopback_gateway_publishes_a_port(self):
        config = yaml.safe_load((ROOT / 'compose.yaml').read_text(encoding='utf-8'))
        services = config['services']
        self.assertEqual({name for name, service in services.items() if service.get('ports')}, {'nginx'})
        self.assertTrue(all(item.startswith('127.0.0.1:') for item in services['nginx']['ports']))
        for name in ('web', 'worker'):
            self.assertTrue(services[name]['read_only'])
            self.assertEqual(services[name]['tmpfs'], ['/tmp:size=256m,mode=1777'])
            models = [v for v in services[name]['volumes'] if isinstance(v, dict)
                      and v['target'].startswith('/app/models/')]
            self.assertEqual(len(models), 3)
            self.assertTrue(all(v['read_only'] for v in models))
            self.assertIn('runs:/app/runs', services[name]['volumes'])
        self.assertEqual(services['web']['secrets'], ['auth'])
        self.assertIn('deepseek', services['worker']['secrets'])

    def test_gpu_access_is_confined_to_worker(self):
        overlay = yaml.safe_load((ROOT / 'compose.gpu.yaml').read_text(encoding='utf-8'))
        self.assertEqual(set(overlay['services']), {'worker'})
        gpu = overlay['services']['worker']['deploy']['resources']['reservations']['devices'][0]
        self.assertEqual(gpu['capabilities'], ['gpu'])
        self.assertEqual(gpu['count'], 1)

    def test_readonly_deepseek_secret_path_is_supported_without_leaking_key(self):
        with TemporaryDirectory() as folder:
            secret = Path(folder) / 'secret'
            secret.write_text('DEEPSEEK_API_KEY=offline-container-test-key\n', encoding='utf-8')
            config = Path(folder) / 'vlm.json'
            config.write_text('{}', encoding='utf-8')
            with patch.dict('os.environ', {'DESKVISION_DEEPSEEK_SECRET': str(secret),
                                          'DEEPSEEK_API_KEY': ''}):
                settings = load_vlm_settings(config_path=config)
            self.assertEqual(settings.api_key, 'offline-container-test-key')
            self.assertNotIn('offline-container-test-key', json.dumps(settings.public()))

    def fixture(self, root):
        private = root / '.local'
        private.mkdir()
        sources = {'yolo_weights': 'models/base.pt', 'sam2_weights': 'models/sam.pt'}
        for name in [*sources.values(), 'runs/train/desk8-full-20261023-204546/weights/best.pt']:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'offline-fixture')
        (private / 'config.json').write_text(json.dumps(sources), encoding='utf-8')
        (private / 'auth.json').write_text('host-config-preserved', encoding='utf-8')
        return private

    def test_preparation_is_idempotent_and_keeps_host_auth_and_existing_env(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            private = self.fixture(root)
            output = StringIO()
            with patch.object(prepare, 'ROOT', root), patch.object(prepare.sys, 'platform', 'linux'), \
                    redirect_stdout(output):
                prepare.main()
                first = (private / 's6/accounts.json').read_bytes()
                signing_key = (private / 's6/auth.json').read_bytes()
                (root / '.env').write_text('USER_SETTING=preserve\n', encoding='utf-8')
                prepare.main()
            self.assertEqual(first, (private / 's6/accounts.json').read_bytes())
            self.assertEqual(signing_key, (private / 's6/auth.json').read_bytes())
            self.assertEqual((root / '.env').read_text(), 'USER_SETTING=preserve\n')
            self.assertEqual((private / 'auth.json').read_text(), 'host-config-preserved')
            for account in json.loads(first):
                self.assertNotIn(account['password'], output.getvalue())


if __name__ == '__main__':
    unittest.main()
