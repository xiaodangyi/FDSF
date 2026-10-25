"""验证训练数据进入模型前必须满足的边界条件。"""

import unittest
from pathlib import Path
import tempfile

from deskvision.audit_data import (apply_reviews, compare_reviewed_records, exported_label_files,
                                  require_training_ready, reviewed_label_sha256)
from deskvision.data import openimages_boxes, require_isolated, validate_box


class TrainingDataContractTests(unittest.TestCase):
    def test_only_known_yolo_split_caches_are_excluded_from_label_inventory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            labels = root / 'labels'
            (labels / 'train').mkdir(parents=True)
            for name in ('train.cache', 'val.cache', 'test.cache', 'unknown.cache', 'extra.txt', 'train/extra.cache'):
                (labels / name).touch()
            self.assertEqual(exported_label_files(root), {
                (labels / 'unknown.cache').resolve(), (labels / 'extra.txt').resolve(),
                (labels / 'train/extra.cache').resolve(),
            })

    def test_invalid_classes_coordinates_and_empty_boxes_are_rejected(self):
        for row in (
            [8, 0.5, 0.5, 0.2, 0.2],
            [1, 0.5, 0.5, 0, 0.2],
            [1, float('nan'), 0.5, 0.2, 0.2],
            [1, 0.99, 0.5, 0.2, 0.2],
        ):
            with self.subTest(row=row), self.assertRaises(ValueError):
                validate_box(row)

    def test_same_mug_hierarchy_box_is_counted_once(self):
        box = {'xmin': 0.1, 'ymin': 0.2, 'xmax': 0.8, 'ymax': 0.9}
        objects = [
            {'category': '/m/02jvh9', 'bounding_box': box},
            {'category': '/m/02p5f1q', 'bounding_box': box},
        ]
        boxes, flags = openimages_boxes(objects, {'/m/02jvh9': 0, '/m/02p5f1q': 0})
        self.assertEqual(len(boxes), 1)
        self.assertEqual(flags, [])

    def test_same_group_or_image_cannot_cross_splits(self):
        base = {'image_sha256': 'a', 'pixel_sha256': 'b', 'group': 'author-1'}
        with self.assertRaisesRegex(ValueError, '跨用途'):
            require_isolated([
                dict(base, id='first', split='train'),
                dict(base, id='second', split='test', image_sha256='different', pixel_sha256='different'),
            ])
        with self.assertRaisesRegex(ValueError, '跨用途'):
            require_isolated([
                dict(base, id='first', split='train'),
                dict(base, id='second', split='test', group='author-2'),
            ])

    def test_stale_review_is_rejected(self):
        record = {'id': 'test', 'source': 'openimages', 'split': 'train', 'group': 'author',
                  'image_sha256': 'image', 'label_sha256': 'label', 'boxes': [[0, .5, .5, .2, .2]]}
        review = {'test': {'status': 'approved', 'image_sha256': 'changed', 'label_sha256': 'label',
                           'all_eight_classes_checked': True, 'note': '复核'}}
        with self.assertRaisesRegex(ValueError, '依据已失效'):
            apply_reviews([record], review)

    def test_unapproved_empty_annotation_cannot_be_used_as_negative(self):
        record = {'id': 'test', 'source': 'openimages', 'split': 'train', 'group': 'author',
                  'image_sha256': 'image', 'label_sha256': 'label', 'boxes': []}
        review = {'test': {'status': 'approved', 'image_sha256': 'image', 'label_sha256': 'label',
                           'all_eight_classes_checked': True, 'note': '空图'}}
        with self.assertRaisesRegex(ValueError, '显式确认负样本'):
            apply_reviews([record], review)

    def test_quarantined_source_cannot_be_approved(self):
        record = {'id': 'test', 'source': 'fasteners', 'split': 'train', 'group': 'scene',
                  'image_sha256': 'image', 'label_sha256': 'label', 'boxes': [[5, .5, .5, .2, .2]]}
        review = {'test': {'status': 'approved', 'image_sha256': 'image', 'label_sha256': 'label',
                           'all_eight_classes_checked': True, 'note': '不应通过'}}
        with self.assertRaisesRegex(ValueError, '来源已隔离'):
            apply_reviews([record], review)

    def test_unready_dataset_cannot_enter_training(self):
        for review in (
            {'training_ready': False, 'training_blockers': ['五金框待复核']},
            {'training_ready': False, 'training_blockers': []},
            {},
            {'training_ready': True, 'training_blockers': ['仍有问题']},
        ):
            with self.subTest(review=review), self.assertRaisesRegex(ValueError, '尚未获准训练'):
                require_training_ready(review)
        require_training_ready({'training_ready': True})

    def test_loco_needs_second_review_bound_to_image_and_final_labels(self):
        record = {'id': 'loco', 'source': 'loco', 'split': 'train', 'source_split': 'train',
                  'group': 'scene', 'image_sha256': 'image', 'pixel_sha256': 'pixels',
                  'label_sha256': 'empty', 'boxes': []}
        review = {'status': 'approved', 'image_sha256': 'image', 'label_sha256': 'empty',
                  'all_eight_classes_checked': True, 'note': '第一轮'}
        self.assertEqual(apply_reviews([record], {'loco': review}, {}), [])
        boxes = [[5, .5, .5, .2, .2]]
        second = {'status': 'approved', 'image_sha256': 'image', 'boxes': boxes,
                  'label_sha256': reviewed_label_sha256(boxes),
                  'all_eight_classes_checked': True, 'note': '已核对叠加框'}
        review['second_review'] = second
        approved = apply_reviews([record], {'loco': review}, {})
        self.assertEqual(approved[0]['boxes'], boxes)
        self.assertIn(second['note'], approved[0]['review_note'])
        second['boxes'] = [[5, .6, .5, .2, .2]]
        with self.assertRaisesRegex(ValueError, '标签内容哈希'):
            apply_reviews([record], {'loco': review}, {})
        second['image_sha256'] = 'another-image'
        with self.assertRaisesRegex(ValueError, '图片依据'):
            apply_reviews([record], {'loco': review}, {})

    def test_self_consistent_export_must_match_current_approved_boxes(self):
        current = {'id': 'one', 'source': 'openimages', 'split': 'train', 'group': 'scene',
                   'image': 'data/raw/one.jpg', 'label': 'data/candidates/one.txt',
                   'image_sha256': 'image', 'pixel_sha256': 'pixels', 'width': 100, 'height': 100,
                   'boxes': [[0, .5, .5, .2, .2]]}
        exported = dict(current, candidate_image=current['image'], candidate_label=current['label'])
        compare_reviewed_records([exported], [current])
        exported['boxes'] = [[0, .6, .5, .2, .2]]
        with self.assertRaisesRegex(ValueError, '当前复核不一致'):
            compare_reviewed_records([exported], [current])
        with self.assertRaisesRegex(ValueError, '样本集合'):
            compare_reviewed_records([], [current])


if __name__ == '__main__':
    unittest.main()
