"""验证准备检查不误启动训练，且配置不能静默改变试跑范围。"""

import copy
import unittest
from unittest.mock import patch

import yaml

from deskvision.train import (TRAIN_CONFIG, load_training_config, preflight,
                              record_epoch_result, run_training, training_scope)


class TrainingEntryTests(unittest.TestCase):
    def test_best_checkpoint_validation_does_not_add_a_training_epoch(self):
        state = {'epochs': []}
        row = {'epoch': 3, 'train_loss': [1, 2, 3], 'validation_metrics': {'map': .1}}
        self.assertTrue(record_epoch_result(state, row))
        final = dict(row, validation_metrics={'map': .2})
        self.assertFalse(record_epoch_result(state, final))
        self.assertEqual(len(state['epochs']), 1)
        self.assertEqual(state['epochs'][0]['validation_metrics'], {'map': .1})
        self.assertEqual(state['best_checkpoint_validation_metrics'], {'map': .2})

    def test_pilot_config_rejects_invalid_or_unreviewed_parameters(self):
        original = yaml.safe_load(TRAIN_CONFIG.read_text(encoding='utf-8'))
        load_training_config()
        for key, value in (('batch_size', 0), ('workers', -1), ('amp', 'false'),
                           ('epochs', 30), ('device', False), ('output_dir', '../outside'),
                           ('dataset', '../outside')):
            changed = copy.deepcopy(original)
            changed[key] = value
            with self.subTest(key=key), patch('deskvision.train.yaml.safe_load', return_value=changed):
                with self.assertRaises(ValueError):
                    load_training_config()

    def test_unready_data_stops_before_environment_or_model_loading(self):
        with patch('deskvision.train.read_json', return_value={'training_ready': False}), \
                patch('deskvision.train.environment_preflight') as environment:
            with self.assertRaisesRegex(ValueError, '尚未获准训练'):
                preflight()
            environment.assert_not_called()

    def test_training_requires_visible_terminal_marker(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, '可见训练终端'):
                run_training(None)

    def test_full_training_is_explicit_and_does_not_change_pilot_scope(self):
        original = yaml.safe_load(TRAIN_CONFIG.read_text(encoding='utf-8'))
        formal = load_training_config(TRAIN_CONFIG.with_name('training-full.yaml'))
        review = {'training_scope': {'purpose': 'three_epoch_pilot', 'maximum_epochs': 3}}
        self.assertEqual(training_scope(original, review)['maximum_epochs'], 3)
        self.assertEqual(training_scope(formal, review)['maximum_epochs'], 100)
        self.assertEqual(review['training_scope']['maximum_epochs'], 3)
        changed = dict(original, epochs=100)
        with self.assertRaisesRegex(ValueError, '超过原复核范围'):
            training_scope(changed, review)

    def test_full_training_rejects_missing_authorization_or_unbound_dataset(self):
        formal = load_training_config(TRAIN_CONFIG.with_name('training-full.yaml'))
        for key, value in (('authorization', ''), ('dataset_manifest_sha256', ''),
                           ('patience', 0), ('epochs', 1000)):
            changed = dict(formal, **{key: value})
            with self.subTest(key=key), patch('deskvision.train.yaml.safe_load', return_value=changed):
                with self.assertRaises(ValueError):
                    load_training_config()


if __name__ == '__main__':
    unittest.main()
