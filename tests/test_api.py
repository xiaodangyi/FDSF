"""HTTP contract tests; route execution is mocked and never calls model services."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from deskvision import api
from deskvision.database import configure_database, get_database


def png_bytes() -> bytes:
    stream = BytesIO()
    Image.new('RGB', (8, 6), color='white').save(stream, format='PNG')
    return stream.getvalue()


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=api.ROOT / 'runs')
        self.runs_root = api.ROOT / 'runs/stage3-api' / Path(self.temp.name).name
        self.root_patch = patch.object(api, 'RUNS_ROOT', self.runs_root)
        self.root_patch.start()
        self.original_database = get_database().path
        self.database = configure_database(Path(self.temp.name) / 'stage3.sqlite3')
        self.database.initialize()
        self.client = TestClient(api.app)

    def tearDown(self):
        self.client.close()
        self.root_patch.stop()
        configure_database(self.original_database)
        self.temp.cleanup()

    def total_runs(self):
        return self.client.get('/api/v1/runs').json()['total']

    def uploaded(self, route='both', content=None):
        return self.client.post('/api/v1/runs', data={'route': route, 'allow_remote': 'true'}, files={
            'image': ('private-original-name.png', content or png_bytes(), 'image/png'),
        })

    def fake_report(self, image_path, output_root, route):
        result_dir = output_root / route / 'run'
        result_dir.mkdir(parents=True)
        (result_dir / 'result.json').write_text('{}', encoding='utf-8')
        (result_dir / 'annotated.png').write_bytes(b'fake image bytes')
        return {
            'status': 'ok', 'route': route,
            'input': {'sha256': sha256(image_path.read_bytes()).hexdigest()},
            'summary': [{'category_key': 'cup', 'category_name': '杯子', 'count': 1}],
            'result_dir': str(result_dir), 'artifacts': {'result': 'result.json', 'visualization': 'annotated.png'},
        }

    def test_health_does_not_run_inference(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'status': 'ok'})

    def test_both_routes_receive_the_same_original_and_result_hides_server_paths(self):
        observed = {}

        def local(image_path, **kwargs):
            observed['yolo'] = sha256(image_path.read_bytes()).hexdigest()
            return self.fake_report(image_path, kwargs['output_root'], 'yolo')

        def remote(image_path, **kwargs):
            observed['deepseek'] = sha256(image_path.read_bytes()).hexdigest()
            return self.fake_report(image_path, kwargs['output_root'], 'deepseek')

        with patch.object(api.core, 'run_yolo_route', side_effect=local), \
                patch.object(api.core, 'run_vlm_route', side_effect=remote):
            created = self.uploaded()
        self.assertEqual(created.status_code, 202)
        run_id = created.json()['run_id']
        result = self.client.get(f'/api/v1/runs/{run_id}')
        self.assertEqual(result.status_code, 200)
        body = result.json()
        self.assertEqual(body['status'], 'succeeded')
        self.assertEqual(observed['yolo'], observed['deepseek'])
        self.assertEqual(body['results']['yolo']['input']['sha256'], observed['yolo'])
        self.assertEqual(body['results']['deepseek']['input']['sha256'], observed['yolo'])
        self.assertEqual(body['input']['sha256'], observed['yolo'])
        self.assertNotIn('result_dir', result.text)
        self.assertNotIn('private-original-name.png', result.text)
        self.assertNotIn('result.json', body['results']['yolo']['artifact_urls'])
        artifact_url = body['results']['yolo']['artifact_urls']['annotated.png']
        self.assertEqual(self.client.get(artifact_url).content, b'fake image bytes')

    def test_local_route_never_calls_deepseek(self):
        with patch.object(api.core, 'run_yolo_route', side_effect=lambda path, **kw:
                          self.fake_report(path, kw['output_root'], 'yolo')) as local, \
                patch.object(api.core, 'run_vlm_route') as remote:
            created = self.uploaded(route='yolo')
        self.assertEqual(created.status_code, 202)
        self.assertEqual(self.client.get(f"/api/v1/runs/{created.json()['run_id']}").json()['status'], 'succeeded')
        local.assert_called_once()
        remote.assert_not_called()

    def test_partial_failure_preserves_the_successful_route(self):
        def local(image_path, **kwargs):
            return self.fake_report(image_path, kwargs['output_root'], 'yolo')

        with patch.object(api.core, 'run_yolo_route', side_effect=local), \
                patch.object(api.core, 'run_vlm_route', return_value={
                    'status': 'failed', 'error': {'code': 'missing_api_key', 'message': 'not configured'},
                }):
            created = self.uploaded(route='both')
        result = self.client.get(f"/api/v1/runs/{created.json()['run_id']}").json()
        self.assertEqual(result['status'], 'partial_failed')
        self.assertEqual(result['results']['yolo']['status'], 'ok')
        self.assertEqual(result['results']['deepseek']['status'], 'failed')

    def test_invalid_parameters_are_rejected(self):
        response = self.client.post('/api/v1/runs', data={'confidence': '1'}, files={
            'image': ('image.png', png_bytes(), 'image/png'),
        })
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.total_runs(), 0)

    def test_invalid_image_is_rejected_without_creating_a_task(self):
        response = self.uploaded(content=b'not an image')
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.total_runs(), 0)

    def test_upload_over_limit_is_rejected(self):
        response = self.uploaded(content=b'x' * (api.MAX_UPLOAD_BYTES + 1))
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.total_runs(), 0)

    def test_unknown_run_and_artifact_paths_are_not_served(self):
        self.assertEqual(self.client.get('/api/v1/runs/not-a-uuid').status_code, 404)
        with patch.object(api.core, 'run_yolo_route', side_effect=lambda path, **kw:
                          self.fake_report(path, kw['output_root'], 'yolo')):
            created = self.uploaded(route='yolo')
        run_id = created.json()['run_id']
        traversal = f'/api/v1/runs/{run_id}/artifacts/yolo/%2E%2E/%2E%2E/private.txt'
        self.assertEqual(self.client.get(traversal).status_code, 404)

    def test_remote_routes_require_explicit_consent(self):
        with patch.object(api.core, 'run_vlm_route') as remote:
            for route in ('deepseek', 'both'):
                response = self.client.post('/api/v1/runs', data={'route': route}, files={
                    'image': ('image.png', png_bytes(), 'image/png')})
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.total_runs(), 0)
        remote.assert_not_called()

    def test_default_route_is_local_only(self):
        with patch.object(api.core, 'run_yolo_route', side_effect=lambda path, **kw:
                          self.fake_report(path, kw['output_root'], 'yolo')), \
                patch.object(api.core, 'run_vlm_route') as remote:
            response = self.client.post('/api/v1/runs', files={
                'image': ('image.png', png_bytes(), 'image/png')})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()['route'], 'yolo')
        remote.assert_not_called()

    def test_unexpected_route_exception_keeps_successful_sibling(self):
        with patch.object(api.core, 'run_yolo_route', side_effect=lambda path, **kw:
                          self.fake_report(path, kw['output_root'], 'yolo')), \
                patch.object(api.core, 'run_vlm_route', side_effect=RuntimeError('private details')):
            created = self.uploaded()
        response = self.client.get(f"/api/v1/runs/{created.json()['run_id']}")
        self.assertEqual(response.json()['status'], 'partial_failed')
        self.assertEqual(response.json()['results']['yolo']['status'], 'ok')
        self.assertEqual(response.json()['results']['deepseek']['error']['code'], 'route_error')
        self.assertNotIn('private details', response.text)

    def test_wrong_image_hash_is_a_failure_instead_of_a_comparison(self):
        with patch.object(api.core, 'run_yolo_route', return_value={
            'status': 'ok', 'input': {'sha256': 'wrong'}, 'summary': []}):
            created = self.uploaded(route='yolo')
        body = self.client.get(f"/api/v1/runs/{created.json()['run_id']}").json()
        self.assertEqual(body['status'], 'failed')
        self.assertEqual(body['results']['yolo']['error']['code'], 'input_mismatch')

    def test_empty_detection_is_success_and_distinct_from_failure(self):
        with patch.object(api.core, 'run_yolo_route', return_value={
            'status': 'ok', 'input': {'sha256': sha256(png_bytes()).hexdigest()},
            'summary': [{'category_key': 'cup', 'count': 0}]}):
            created = self.uploaded(route='yolo')
        body = self.client.get(f"/api/v1/runs/{created.json()['run_id']}").json()
        self.assertEqual(body['status'], 'succeeded')
        self.assertEqual(body['results']['yolo']['summary'][0]['count'], 0)

    def test_internal_json_and_unlisted_files_are_not_downloadable(self):
        with patch.object(api.core, 'run_yolo_route', side_effect=lambda path, **kw:
                          self.fake_report(path, kw['output_root'], 'yolo')):
            created = self.uploaded(route='yolo')
        run_id = created.json()['run_id']
        report = api._repository().get(run_id)['results']['yolo']
        (Path(report['result_dir']) / 'private.txt').write_text('private', encoding='utf-8')
        for name in ('result.json', 'private.txt', 'missing.png'):
            self.assertEqual(self.client.get(
                f'/api/v1/runs/{run_id}/artifacts/yolo/{name}').status_code, 404)

    def test_listed_symlink_is_not_downloadable(self):
        with patch.object(api.core, 'run_yolo_route', side_effect=lambda path, **kw:
                          self.fake_report(path, kw['output_root'], 'yolo')):
            created = self.uploaded(route='yolo')
        run_id = created.json()['run_id']
        report = api._repository().get(run_id)['results']['yolo']
        base = Path(report['result_dir'])
        try:
            (base / 'annotated.png').unlink()
            (base / 'outside.png').write_bytes(b'outside')
            (base / 'annotated.png').symlink_to(base / 'outside.png')
        except OSError:
            self.skipTest('Creating symbolic links requires Windows developer mode or privileges.')
        self.assertEqual(self.client.get(
            f'/api/v1/runs/{run_id}/artifacts/yolo/annotated.png').status_code, 404)

    def test_paginated_history_note_update_and_delete(self):
        ids = []
        for _ in range(2):
            with patch.object(api.core, 'run_yolo_route', side_effect=lambda path, **kw:
                              self.fake_report(path, kw['output_root'], 'yolo')):
                ids.append(self.uploaded(route='yolo').json()['run_id'])
        page = self.client.get('/api/v1/runs', params={'limit': 1, 'offset': 0})
        self.assertEqual(page.status_code, 200)
        self.assertEqual(page.json()['total'], 2)
        self.assertEqual(len(page.json()['items']), 1)
        self.assertEqual(self.client.get('/api/v1/runs', params={'status': 'failed'}).json()['total'], 0)
        updated = self.client.patch(f'/api/v1/runs/{ids[0]}/note', json={'note': '复核完成'})
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(self.client.get(f'/api/v1/runs/{ids[0]}').json()['note'], '复核完成')
        self.assertEqual(self.client.patch(f'/api/v1/runs/{ids[0]}/note',
                                           json={'note': 'a' * 1001}).status_code, 422)
        run_dir = self.runs_root / ids[0]
        self.assertEqual(self.client.delete(f'/api/v1/runs/{ids[0]}').status_code, 204)
        self.assertFalse(run_dir.exists())
        self.assertEqual(self.client.get('/api/v1/runs').json()['total'], 1)

    def test_restart_keeps_history_results_and_artifact_index(self):
        with patch.object(api.core, 'run_yolo_route', side_effect=lambda path, **kw:
                          self.fake_report(path, kw['output_root'], 'yolo')):
            created = self.uploaded(route='yolo')
        run_id = created.json()['run_id']
        self.client.patch(f'/api/v1/runs/{run_id}/note', json={'note': '留待课堂展示'})
        database_path = self.database.path
        self.database = configure_database(database_path)
        self.database.initialize()
        response = self.client.get(f'/api/v1/runs/{run_id}')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['note'], '留待课堂展示')
        self.assertIn('annotated.png', response.json()['results']['yolo']['artifact_urls'])

    def test_queued_task_cannot_be_deleted(self):
        from uuid import uuid4

        run_id = str(uuid4())
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True)
        image_path = run_dir / 'input.img'
        image_path.write_bytes(png_bytes())
        _, input_info = api.prepare_image(image_path)
        api._repository().create({
            'run_id': run_id, 'route': 'yolo', 'status': 'queued', 'created_at': '2026-10-26T00:00:00+08:00',
            'input': input_info, 'parameters': {}, 'image_path': image_path, 'run_dir': run_dir,
        })
        self.assertEqual(self.client.delete(f'/api/v1/runs/{run_id}').status_code, 409)
        self.assertTrue(run_dir.exists())

    def test_application_restart_marks_incomplete_tasks_without_rerunning(self):
        from uuid import uuid4

        run_id = str(uuid4())
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True)
        image_path = run_dir / 'input.img'
        image_path.write_bytes(png_bytes())
        _, input_info = api.prepare_image(image_path)
        api._repository().create({
            'run_id': run_id, 'route': 'yolo', 'status': 'running', 'created_at': '2026-10-26T00:00:00+08:00',
            'input': input_info, 'parameters': {}, 'image_path': image_path, 'run_dir': run_dir,
        })
        with patch.object(api.core, 'run_yolo_route') as inference, \
                TestClient(api.create_app()):
            result = api._repository().get(run_id)
        inference.assert_not_called()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['error']['code'], 'interrupted_by_restart')

    def test_page_and_api_share_one_application_without_inference(self):
        with patch.object(api.core, 'run_yolo_route') as local, \
                patch.object(api.core, 'run_vlm_route') as remote:
            with TestClient(api.create_app()) as client:
                self.assertEqual(client.get('/health').json(), {'status': 'ok'})
                self.assertEqual(client.get('/ui/').status_code, 200)
                config = client.get('/ui/config')
                self.assertEqual(config.status_code, 200)
                self.assertTrue(any(c['type'] == 'gallery' for c in config.json()['components']))
                self.assertEqual(client.get('/').status_code, 200)
        local.assert_not_called()
        remote.assert_not_called()


if __name__ == '__main__':
    unittest.main()
