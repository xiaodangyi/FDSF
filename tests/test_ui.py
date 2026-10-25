"""第七阶段 UI 的纯逻辑、报告隔离与界面构建检查。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from deskvision.config import Category
from deskvision import ui


class FakeDetector:
    device = 'cpu'

    def __init__(self, weights, categories, device):
        self.weights = weights
        self.categories = categories
        self.device = device

    def predict(self, image, confidence, iou, image_size):
        return {
            'status': 'ok', 'image_size': {'width': 4, 'height': 3},
            'detections': [],
            'summary': [{'category_id': item.id, 'category_key': item.key,
                         'category_name': item.name, 'count': 0,
                         'status': 'supported', 'status_text': '已微调，待验收'}
                        for item in self.categories],
            'total_supported_detections': 0, 'message': '未检出八类目标，请核对漏检。',
            'model': {'route': 'yolov8_desk8_finetuned'},
            'timings_ms': {'prediction_wall': 1.0},
        }


class UiTests(unittest.TestCase):
    def test_import_does_not_load_model_frameworks(self):
        self.assertNotIn('ultralytics', sys.modules)
        self.assertNotIn('sam2', sys.modules)

    def test_route_writes_yolo_result_without_vlm(self):
        categories = [Category(0, 'cup', '杯子', 41)]
        with TemporaryDirectory() as folder:
            source = Path(folder) / 'source.jpg'
            source.write_bytes(b'input')
            output = Path(folder) / 'runs'
            # 离线检查不得依赖旧机器训练目录中的私有 best.pt。
            weights = Path(folder) / 'weights.pt'
            weights.write_bytes(b'fake-weights-for-mocked-detector')
            with patch.object(ui, 'read_image', return_value=np.zeros((3, 4, 3), dtype=np.uint8)), \
                    patch.object(ui, 'prepare_image', return_value=(b'input', {'sha256': 'original'})), \
                    patch.object(ui, 'load_categories', return_value=categories), \
                    patch.object(ui, 'local_weights', return_value=Path(folder) / 'weights.pt'), \
                    patch.object(ui, 'TrainedDetector', FakeDetector), \
                    patch.object(ui, 'file_sha256', return_value='hash'), \
                    patch.object(ui, 'find_font', return_value=Path('font.ttf')), \
                    patch.object(ui, 'render_result') as render:
                render.return_value.save = lambda path: Path(path).write_bytes(b'png')
                result = ui.run_yolo_route(source, weights_path=weights,
                                           output_root=output, segment=False)
            self.assertEqual(result['status'], 'ok')
            self.assertTrue(Path(result['result_dir'], 'result.json').is_file())
            self.assertTrue(Path(result['visualization_path']).is_file())
            self.assertNotIn('items', result)

    def test_vlm_route_receives_original_path_and_keeps_reports_separate(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / 'desk.png'
            source.write_bytes(b'input')
            report = {'status': 'ok', 'summary': [], 'items': [], 'timings_ms': {'api_wall': 2}}
            with patch.object(ui, 'load_vlm_settings', return_value=object()), \
                    patch.object(ui, 'recognize', return_value=report) as recognize:
                result = ui.run_vlm_route(source, output_root=Path(folder) / 'runs')
            self.assertEqual(result['status'], 'ok')
            self.assertEqual(recognize.call_args.args[0], source.resolve())
            self.assertNotIn('yolo', result.get('route', '').lower())

    def test_count_diff_does_not_turn_unsupported_into_zero(self):
        yolo = {'status': 'ok', 'input': {'sha256': 'same'}, 'summary': [
            {'category_key': 'cup', 'category_name': '杯子', 'count': 2},
            {'category_key': 'book', 'category_name': '书', 'count': None},
        ]}
        vlm = {'status': 'ok', 'input': {'sha256': 'same'}, 'summary': [
            {'category_key': 'cup', 'category_name': '杯子', 'count': 1},
            {'category_key': 'book', 'category_name': '书', 'count': 3},
        ]}
        text = ui.format_count_diff(yolo, vlm)
        self.assertIn('| 杯子 | 2 | 1 | -1 |', text)
        self.assertIn('| 书 | — | 3 | — |', text)

    def test_mismatched_images_and_failures_are_never_compared(self):
        first = {'status': 'ok', 'input': {'sha256': 'first'}, 'summary': []}
        second = {'status': 'ok', 'input': {'sha256': 'second'}, 'summary': []}
        self.assertIn('输入图片不一致', ui.format_count_diff(first, second))
        self.assertIn('至少一条路线失败', ui.format_count_diff(first, {'status': 'failed'}))

    def test_one_route_failure_does_not_block_the_other_route(self):
        yolo = {'status': 'failed', 'error': {'message': '模拟错误'}}
        vlm = {'status': 'ok', 'summary': [], 'items': []}
        with patch.object(ui, 'run_yolo_route', return_value=yolo) as local, \
                patch.object(ui, 'run_vlm_route', return_value=vlm) as remote:
            result = ui.run_both_ui('original.png', 'finetuned', '', .25, .7, 640, 'auto', True)
        self.assertEqual(local.call_args.args, ('original.png',))
        self.assertEqual(remote.call_args.args, ('original.png',))
        self.assertEqual(result[2], yolo)
        self.assertEqual(result[4], vlm)
        self.assertIn('至少一条路线失败', result[5])

    def test_segmentation_failure_keeps_detection_report_and_image(self):
        with TemporaryDirectory() as folder:
            destination = Path(folder)
            (destination / 'annotated.png').write_bytes(b'image')
            partial = {'detections': [], 'summary': [{'count': 2}], 'timings_ms': {'prediction_wall': 12},
                       'artifacts': {'visualization': 'annotated.png'}}
            report = ui._failed_result('yolo', RuntimeError('secret must not appear'),
                                       ui.perf_counter(), destination, partial, 'segmentation')
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['summary'], partial['summary'])
            self.assertEqual(report['timings_ms']['prediction_wall'], 12)
            self.assertTrue(Path(report['visualization_path']).is_file())
            self.assertNotIn('secret', (destination / 'result.json').read_text(encoding='utf-8'))

    def test_zip_includes_masks_and_never_reads_files_outside_ui_runs(self):
        with TemporaryDirectory() as folder:
            root = Path(folder) / 'runs'
            run = root / 'yolo' / 'one'
            (run / 'masks').mkdir(parents=True)
            (run / 'result.json').write_text('{}', encoding='utf-8')
            (run / 'masks' / 'one.png').write_bytes(b'mask')
            private = Path(folder) / '.local'
            private.mkdir()
            (private / 'deepseek.env').write_text('do-not-export', encoding='utf-8')
            archive = ui.export_results({'result_dir': str(run)}, {'result_dir': str(private)}, output_root=root)
            with ZipFile(archive) as bundle:
                self.assertIn('yolo/result.json', bundle.namelist())
                self.assertIn('yolo/masks/one.png', bundle.namelist())
                self.assertFalse(any('deepseek.env' in name for name in bundle.namelist()))

    def test_new_input_resets_both_reports_and_mode_sets_correct_weights(self):
        values = ui.reset_input('new.png')
        self.assertIsNone(values[3])
        self.assertIsNone(values[5])
        self.assertIsNone(values[7])
        self.assertEqual(ui.weights_for_mode('baseline'), '')
        self.assertEqual(ui.weights_for_mode('finetuned'), str(ui.DEFAULT_FINETUNED_WEIGHTS))

    def test_gradio_demo_builds_without_loading_models(self):
        demo = ui.build_demo()
        self.assertTrue(hasattr(demo, 'launch'))

    def test_input_changed_during_inference_discards_stale_outputs(self):
        state = ui.new_session()
        values = ('original.png', 'finetuned', '', .25, .7, 640, 'auto', True)
        ui.invalidate_session(state, values)

        def finish_late(*args):
            ui.invalidate_session(state, ('new.png', *values[1:]))
            return [], '旧结果', {'status': 'ok'}, '完成'

        with patch.object(ui, 'run_yolo_ui', side_effect=finish_late), \
                patch.object(ui, 'refresh_comparison') as refresh:
            result = ui.run_session_ui('yolo', state, None, None, *values)
        self.assertTrue(all(value == {'__type__': 'update'} for value in result))
        refresh.assert_not_called()

    def test_queued_old_parameters_do_not_start_inference(self):
        state = ui.new_session()
        values = ('original.png', 'finetuned', '', .25, .7, 640, 'auto', True)
        ui.invalidate_session(state, ('original.png', 'finetuned', '', .5, .7, 640, 'auto', True))
        with patch.object(ui, 'run_yolo_ui') as run:
            ui.run_session_ui('yolo', state, None, None, *values)
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
