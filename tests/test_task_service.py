"""Persistence-boundary checks for Gradio-originated tasks."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from deskvision import task_service


class TaskServiceTests(unittest.TestCase):
    def test_failed_database_insert_removes_untracked_run_directory(self):
        with TemporaryDirectory() as folder:
            root = Path(folder) / 'tasks'
            source = Path(folder) / 'source.png'
            source.write_bytes(b'image')
            image_info = {'sha256': 'digest', 'mime_type': 'image/png',
                          'bytes': 5, 'width': 1, 'height': 1}
            with patch.object(task_service, 'TASKS_ROOT', root), \
                    patch.object(task_service.core, '_path_value', return_value=source), \
                    patch.object(task_service.core, 'prepare_image',
                                 return_value=(b'image', image_info)), \
                    patch.object(task_service.RunRepository, 'create',
                                 side_effect=RuntimeError('database unavailable')):
                with self.assertRaisesRegex(RuntimeError, 'database unavailable'):
                    task_service.execute_and_record('yolo', source, {})
            self.assertEqual(list(root.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
