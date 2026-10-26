"""新版展示层的真值边界、转义与可操作状态检查，不请求真实模型。"""

import unittest
from unittest.mock import patch

from deskvision.web.app import build_demo, overview, remote_inventory, run_confirmed


def report(fingerprint='same', count=1, status='ok'):
    return {'status': status, 'input': {'sha256': fingerprint},
            'summary': [{'category_key': 'cup', 'category_name': '杯子', 'count': count}]}


class WorkbenchTests(unittest.TestCase):
    def test_cloud_action_without_consent_never_starts_inference(self):
        import gradio as gr
        with patch('deskvision.web.app.core.run_session_ui') as run:
            for route in ('both', 'deepseek'):
                with self.assertRaises(gr.Error):
                    run_confirmed(route, False, {}, None, None)
            run.assert_not_called()

    def test_missing_and_failed_results_do_not_forge_zero_counts(self):
        self.assertIn('—', overview(None, report(status='failed')))
        self.assertNotIn('>0<', overview(None, report(status='failed')))

    def test_unknown_counts_and_different_images_cannot_produce_agreement(self):
        self.assertNotIn('>0<', overview(report(count=None), report(count=None)))
        self.assertIn('原图不一致', overview(report('one'), report('two')))
        self.assertNotIn('>0<', overview(report('one'), report('two')))

    def test_diff_counts_categories_instead_of_absolute_count_error(self):
        text = overview(report(count=5), report(count=2))
        self.assertIn('>1<small>类', text)
        self.assertIn('>5<small>件', text)

    def test_remote_model_text_is_escaped_and_failure_is_visible(self):
        text = remote_inventory({'status': 'ok', 'items': [
            {'category_name': '<script>bad</script>', 'location_text': '<img onerror="bad">',
             'uncertain': True}]})
        self.assertNotIn('<script>', text)
        self.assertNotIn('<img ', text)
        self.assertIn('&lt;script&gt;', text)
        self.assertIn('存疑', text)
        self.assertIn('失败不计为零件', remote_inventory(report(status='failed')))

    def test_empty_page_disables_inference_and_exposes_three_result_tabs(self):
        demo = build_demo()
        components = demo.config['components']
        actions = [c for c in components if c['type'] == 'button' and
                   c['props'].get('value', '').startswith('运行')]
        self.assertEqual(len(actions), 3)
        self.assertTrue(all(not c['props']['interactive'] for c in actions))
        self.assertEqual([c['props']['label'] for c in components if c['type'] == 'tabitem'],
                         ['识别结果', '数量对比', '详细报告'])


if __name__ == '__main__':
    unittest.main()
