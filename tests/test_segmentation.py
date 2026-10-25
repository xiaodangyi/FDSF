"""检查原图坐标、实例关联、掩膜孔洞、空结果和错误状态，不需要加载模型。"""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from deskvision.config import load_categories
from deskvision.data import read_json
from deskvision.detector import make_report
from deskvision.segmentation import Sam2Segmenter, mask_geometry


class FakePredictor:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.images = []
        self.prompts = []
        self.resets = 0

    def set_image(self, image):
        self.images.append(image.copy())

    def predict(self, **kwargs):
        self.prompts.append(kwargs)
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        return value

    def reset_predictor(self):
        self.resets += 1


class SegmentationContractTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.weights = self.root / 'tiny.pt'
        self.weights.write_bytes(b'fake weights used only with fake predictor')
        self.image = np.full((8, 10, 3), [10, 20, 30], dtype=np.uint8)
        self.report = make_report(load_categories(), [[1, 2, 8, 7, .8, 5], [0, 0, 5, 8, .7, 6]],
                                  10, 8, class_space='desk8')
        self.report['timings_ms'] = {}

    def segmenter(self, outputs):
        segmenter = Sam2Segmenter(self.weights, 'cpu')
        segmenter.predictor = FakePredictor(outputs)
        return segmenter

    def test_original_coordinates_rgb_instance_mapping_and_empty_mask(self):
        mask = np.zeros((8, 10), dtype=bool)
        mask[2:7, 1:8] = True
        engine = self.segmenter([(mask[None], np.array([.61]), None),
                                 (np.zeros((1, 8, 10)), np.array([.2]), None)])
        before = deepcopy(self.report)
        result = engine.segment(self.image, self.report, self.root / 'results')
        self.assertEqual(self.report, before)
        self.assertEqual(result['summary'], before['summary'])
        self.assertEqual(result['detections'], before['detections'])
        self.assertEqual(len(engine.predictor.images), 1)
        np.testing.assert_array_equal(engine.predictor.images[0][0, 0], [30, 20, 10])
        for prompt, detection in zip(engine.predictor.prompts, self.report['detections']):
            np.testing.assert_array_equal(prompt['box'], detection['bbox_xyxy'])
            self.assertFalse(prompt['multimask_output'])
            self.assertTrue(prompt['normalize_coords'])
        rows = result['segmentation']['instances']
        self.assertEqual([r['instance_id'] for r in rows], [1, 2])
        self.assertEqual([r['status'] for r in rows], ['ok', 'empty'])
        self.assertEqual(rows[0]['sam2_predicted_iou'], .61)
        self.assertEqual(rows[0]['bbox_xyxy'], [1, 2, 8, 7])
        self.assertEqual(rows[1]['area_pixels'], 0)
        self.assertIsNone(rows[1]['centroid_xy'])
        self.assertEqual(engine.predictor.resets, 1)
        with Image.open(self.root / 'results' / rows[0]['mask_file']) as saved:
            self.assertEqual(saved.size, (10, 8))
            np.testing.assert_array_equal(np.asarray(saved), mask.astype(np.uint8) * 255)

    def test_no_detections_skips_model_and_keeps_counts(self):
        empty = make_report(load_categories(), [], 10, 8, class_space='desk8')
        empty['timings_ms'] = {}
        engine = Sam2Segmenter(self.weights, 'cpu')
        with patch.object(engine, '_load_predictor', side_effect=AssertionError('不应加载模型')):
            result = engine.segment(self.image, empty, self.root / 'empty')
        self.assertEqual(result['segmentation']['status'], 'skipped_no_detections')
        self.assertEqual(result['segmentation']['instances'], [])
        self.assertEqual(result['summary'], empty['summary'])
        self.assertFalse((self.root / 'empty/masks').exists())

    def test_invalid_prompt_is_rejected_before_loading(self):
        for mutation in ('outside', 'duplicate', 'shape'):
            report = deepcopy(self.report)
            if mutation == 'outside':
                report['detections'][0]['bbox_xyxy'] = [0, 0, 11, 8]
            elif mutation == 'duplicate':
                report['detections'][1]['instance_id'] = 1
            else:
                report['image_size']['width'] = 9
            engine = Sam2Segmenter(self.weights, 'cpu')
            with self.subTest(mutation=mutation), patch.object(engine, '_load_predictor') as loader:
                with self.assertRaises(ValueError):
                    engine.segment(self.image, report, self.root / mutation)
                loader.assert_not_called()

    def test_bad_model_output_fails_and_releases_image_features(self):
        outputs = [(np.ones((1, 7, 10)), np.array([.8]), None),
                   (np.full((1, 8, 10), np.nan), np.array([.8]), None),
                   (np.ones((1, 8, 10)), np.array([np.nan]), None),
                   (np.full((1, 8, 10), .5), np.array([.8]), None),
                   RuntimeError('decoder failed')]
        for i, output in enumerate(outputs):
            engine = self.segmenter([output])
            with self.subTest(index=i), self.assertRaises((ValueError, RuntimeError)):
                engine.segment(self.image, self.report, self.root / f'failure-{i}')
            self.assertEqual(engine.predictor.resets, 1)

    def test_contour_hierarchy_preserves_washer_hole(self):
        mask = np.zeros((8, 10), dtype=bool)
        mask[1:7, 1:9] = True
        mask[3:5, 4:6] = False
        geometry = mask_geometry(mask)
        self.assertEqual(geometry['area_pixels'], 44)
        self.assertEqual(geometry['bbox_xyxy'], [1, 1, 9, 7])
        self.assertEqual([x['parent_index'] for x in geometry['contours']], [-1, 0])

    def test_same_model_can_encode_a_new_image_after_failure(self):
        engine = self.segmenter([RuntimeError('first image failed'),
                                 (np.ones((1, 8, 10)), np.array([.8]), None),
                                 (np.zeros((1, 8, 10)), np.array([.2]), None)])
        with self.assertRaises(RuntimeError):
            engine.segment(self.image, self.report, self.root / 'first')
        engine.segment(self.image + 1, self.report, self.root / 'second')
        self.assertEqual(len(engine.predictor.images), 2)
        self.assertEqual(engine.predictor.resets, 2)

    def test_cli_segmentation_failure_has_failed_status_and_preserves_detections(self):
        from deskvision.__main__ import main

        path = self.root / '桌面.png'
        cv2.imencode('.png', self.image)[1].tofile(path)
        with patch('sys.argv', ['deskvision', '--image', str(path), '--segment', '--output', str(self.root / 'cli')]), \
                patch('deskvision.__main__.local_weights', return_value=self.weights), \
                patch('deskvision.__main__.local_sam2_weights', return_value=self.weights), \
                patch('deskvision.__main__.BaselineDetector') as detector, \
                patch('deskvision.__main__.render_result', return_value=Image.new('RGB', (10, 8))), \
                patch('deskvision.segmentation.Sam2Segmenter') as segmenter:
            detector.return_value.predict.return_value = deepcopy(self.report)
            detector.return_value.device = 'cpu'
            segmenter.return_value.segment.side_effect = RuntimeError('SAM 2 failure')
            self.assertEqual(main(), 1)
        result = read_json(next((self.root / 'cli').glob('*/result.json')))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['error']['stage'], 'segmentation')
        self.assertEqual(result['detections'], self.report['detections'])
        self.assertNotIn('segmentation_visualization', result['artifacts'])


if __name__ == '__main__':
    unittest.main()
