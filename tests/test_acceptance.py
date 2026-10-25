"""验收指标和断点保护：失败不能充当正确零计数，异图和旧协议不能复用。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from deskvision.acceptance import CLASS_KEYS, aggregate, cached, checked_counts, digest, save_record


def counts(**values):
    return dict.fromkeys(CLASS_KEYS, 0) | values


class AcceptanceTests(unittest.TestCase):
    def test_failure_is_not_zero_and_denominator_is_all_images(self):
        rows = [{'counts': counts(cup=2), 'truth': counts(cup=1), 'timings_ms': {'total_wall': 10}},
                {'counts': None, 'truth': counts(), 'timings_ms': {}}]
        result = aggregate(rows)
        self.assertEqual(result['successful'], 1)
        self.assertEqual(result['failed_or_missing'], 1)
        self.assertEqual(result['total_absolute_error'], 1)
        self.assertEqual(result['mae_per_image_category'], 1 / 8)
        self.assertEqual(result['normalized_absolute_error'], 1)
        rows[0]['counts'] = counts(cup=1)
        self.assertEqual(aggregate(rows)['exact_rate_all'], .5)

    def test_all_failed_has_no_error_or_timing_estimate(self):
        result = aggregate([{'counts': None, 'truth': counts(), 'timings_ms': {}}])
        self.assertIsNone(result['mae_per_image_category'])
        self.assertIsNone(result['total_wall_ms'])
        self.assertIsNone(result['normalized_absolute_error'])

    def test_invalid_success_and_wrong_image_are_rejected(self):
        report = {'status': 'ok', 'input': {'sha256': 'image'}, 'summary': [
            {'category_key': key, 'count': value} for key, value in counts().items()]}
        self.assertEqual(checked_counts(report, 'image'), counts())
        with self.assertRaises(ValueError):
            checked_counts(report, 'different')
        report['summary'][0]['count'] = None
        with self.assertRaises(ValueError):
            checked_counts(report, 'image')
        self.assertIsNone(checked_counts({'status': 'failed'}, 'image'))

    def test_resume_checks_protocol_sample_and_report_digest(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'record.json'
            report = {'status': 'failed'}
            entry = {'protocol_sha256': 'frozen', 'sample_id': 'one', 'state': 'finished',
                     'report': report, 'report_sha256': digest(report)}
            save_record(path, entry)
            sample = {'id': 'one', 'image_sha256': 'image'}
            self.assertEqual(cached(path, 'frozen', sample)['report'], report)
            with self.assertRaises(ValueError):
                cached(path, 'other-protocol', sample)
            with self.assertRaises(ValueError):
                cached(path, 'frozen', sample | {'id': 'other'})
            entry['report']['status'] = 'changed'
            save_record(path, entry)
            with self.assertRaises(ValueError):
                cached(path, 'frozen', sample)

    def test_started_entry_is_preserved_without_fake_result(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'record.json'
            save_record(path, {'protocol_sha256': 'frozen', 'sample_id': 'one', 'state': 'started'})
            self.assertNotIn('report', cached(path, 'frozen', {'id': 'one'}))


if __name__ == '__main__':
    unittest.main()
