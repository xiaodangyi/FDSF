"""验证计数语义、编号转换和输入边界，避免错误被包装成有效结果。"""

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from deskvision.config import load_categories
from deskvision.detector import make_report, read_image


class BaselineContractTests(unittest.TestCase):
    """这些规则会影响后续训练比较，不能用看起来合理的零值替代。"""

    def setUp(self):
        self.categories = load_categories()

    def test_empty_result_keeps_untrained_categories_unknown(self):
        report = make_report(self.categories, [], 640, 480)
        self.assertEqual(report['total_supported_detections'], 0)
        self.assertEqual([item['count'] for item in report['summary']], [0, 0, 0, None, None, None, None, None])
        self.assertEqual(report['summary'][3]['status'], 'not_trained')

    def test_finetuned_ids_are_project_ids_and_all_classes_have_counts(self):
        report = make_report(self.categories, [[1, 2, 30, 40, .9, 6]], 640, 480, class_space='desk8')
        self.assertEqual(report['detections'][0]['category_name'], '螺母')
        self.assertEqual([item['count'] for item in report['summary']], [0, 0, 0, 0, 0, 0, 1, 0])
        empty = make_report(self.categories, [], 640, 480, class_space='desk8')
        self.assertTrue(all(item['count'] == 0 for item in empty['summary']))
        with self.assertRaises(ValueError):
            make_report(self.categories, [[1, 2, 30, 40, .9, 41]], 640, 480, class_space='desk8')

    def test_source_ids_map_to_project_ids_without_losing_precision(self):
        report = make_report(self.categories, [
            [10.125, 20.5, 50.875, 80.75, 0.9, 41],
            [40, 30, 80, 100, 0.8, 41],
            [0, 0, 60, 80, 0.75, 39],
            [5, 8, 100, 150, 0.6, 73],
        ], 640, 480)
        self.assertEqual([item['count'] for item in report['summary'][:3]], [2, 1, 1])
        self.assertEqual([item['category_id'] for item in report['detections']], [0, 0, 1, 2])
        self.assertEqual(report['detections'][0]['bbox_xyxy'], [10.125, 20.5, 50.875, 80.75])

    def test_unrequested_or_invalid_model_output_is_rejected(self):
        for row in ([0, 0, 20, 20, 0.8, 0], [20, 0, 10, 20, 0.8, 41],
                    [0, 0, 20, 20, float('nan'), 41]):
            with self.subTest(row=row), self.assertRaises(ValueError):
                make_report(self.categories, [row], 640, 480)

    def test_clipped_box_stays_in_original_image_coordinates(self):
        report = make_report(self.categories, [[-1, -2, 700, 500, 0.9, 41]], 640, 480)
        self.assertEqual(report['detections'][0]['bbox_xyxy'], [0, 0, 640, 480])

    def test_unicode_path_and_invalid_image_are_distinguished(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / '桌面 测试.png'
            image = np.full((24, 32, 3), 128, dtype=np.uint8)
            cv2.imencode('.png', image)[1].tofile(path)
            np.testing.assert_array_equal(read_image(path), image)
            path.write_bytes(b'')
            with self.assertRaises(ValueError):
                read_image(path)
            with self.assertRaises(FileNotFoundError):
                read_image(Path(folder) / '不存在.png')


if __name__ == '__main__':
    unittest.main()
