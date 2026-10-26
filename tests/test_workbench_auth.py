"""Exercise real Gradio HTTP boundaries with two users, without model inference."""

import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
from urllib.parse import quote
from uuid import uuid4

from fastapi import HTTPException
from fastapi.testclient import TestClient
import gradio as gr
from starlette.requests import Request

from deskvision import api, auth, ui
from deskvision.web.app import checked_values, export_link, owned_report
import test_api as fixtures


class WorkbenchAuthTests(TestCase):
    setUp = fixtures.ApiTests.setUp
    tearDown = fixtures.ApiTests.tearDown
    fake_report = fixtures.ApiTests.fake_report
    uploaded = fixtures.ApiTests.uploaded

    def client_two(self):
        auth.create_user('student_two', 'correct-horse-battery-staple')
        token, _ = auth.login('student_two', 'correct-horse-battery-staple')
        return TestClient(api.create_app(), client=('127.0.0.1', 50001),
                          headers={'Authorization': f'Bearer {token}'})

    def upload_file(self):
        api.create_app()
        response = self.client.post('/ui/gradio_api/upload', files={
            'files': ('private.png', fixtures.png_bytes(), 'image/png')})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()[0]

    def file_data(self, filename):
        return {'path': filename, 'orig_name': 'private.png',
                'meta': {'_type': 'gradio.FileData'}}

    def image_payload(self, filename, session_hash=None):
        return {'data': [None, self.file_data(filename), 'finetuned',
                         str(ui.DEFAULT_FINETUNED_WEIGHTS), .25, .7, 640, 'auto', True],
                'session_hash': session_hash or uuid4().hex}

    def test_uploaded_file_and_preview_are_private_to_login(self):
        filename = self.upload_file()
        url = '/ui/gradio_api/file=' + quote(filename.replace('\\', '/'), safe='/:')
        self.assertEqual(self.client.get(url).status_code, 200)
        with self.client_two() as other:
            self.assertEqual(other.get(url).status_code, 404)
            attack = other.post('/ui/gradio_api/run/set_image', json=self.image_payload(filename))
            self.assertEqual(attack.status_code, 403)
        response = self.client.post('/ui/gradio_api/run/set_image', json=self.image_payload(filename))
        self.assertEqual(response.status_code, 200, response.text)
        preview = response.json()['data'][0]
        self.assertEqual(self.client.get('/ui/gradio_api/file=' +
                         quote(preview['path'].replace('\\', '/'), safe='/:')).status_code, 200)
        self.assertEqual(response.headers['cache-control'], 'no-store')

    def test_stolen_session_hash_cannot_reuse_state_or_read_queue(self):
        filename = self.upload_file()
        payload = self.image_payload(filename)
        first = self.client.post('/ui/gradio_api/run/set_image', json=payload)
        self.assertEqual(first.status_code, 200, first.text)
        with self.client_two() as other:
            response = other.post('/ui/gradio_api/run/set_image', json={
                'data': [None, None, 'finetuned', str(ui.DEFAULT_FINETUNED_WEIGHTS),
                         .25, .7, 640, 'auto', True], 'session_hash': payload['session_hash']})
            self.assertEqual(response.status_code, 403)
            self.assertEqual(other.get('/ui/gradio_api/queue/data', params={
                'session_hash': payload['session_hash']}).status_code, 403)
            self.assertEqual(other.get('/ui/gradio_api/heartbeat/' + payload['session_hash']).status_code, 403)

    def test_queue_event_id_is_bound_to_login(self):
        api.create_app()
        config = self.client.get('/ui/config').json()
        load = next(item for item in config['dependencies'] if item.get('api_name') == 'show_account')
        # A real queued callback is safe here: it only returns the current username.
        with TestClient(api.create_app(), client=('127.0.0.1', 50000),
                        headers=self.client.headers) as active:
            response = active.post('/ui/gradio_api/queue/join', json={
                'data': [], 'fn_index': load['id'], 'session_hash': uuid4().hex})
            self.assertEqual(response.status_code, 200, response.text)
            event_id = response.json()['event_id']
            with self.client_two() as other:
                self.assertEqual(other.get('/ui/gradio_api/call/show_account/' + event_id).status_code, 403)
                self.assertEqual(other.post('/ui/gradio_api/cancel', json={
                    'event_id': event_id, 'session_hash': uuid4().hex,
                    'fn_index': load['id']}).status_code, 403)

    def test_local_files_remote_urls_and_unused_routes_are_rejected(self):
        api.create_app()
        for filename in (str(api.ROOT / '.local/auth.json'), 'https://example.org/private.png',
                         str(api.ROOT / 'README.md')):
            response = self.client.post('/ui/gradio_api/run/set_image',
                                        json=self.image_payload(filename))
            self.assertEqual(response.status_code, 403)
        for path in ('stream/guessed', 'proxy=https://example.org', 'component_server', 'reset'):
            self.assertEqual(self.client.post('/ui/gradio_api/' + path, json={}).status_code, 403)

    def test_real_queue_callback_and_sse_gallery_keep_owner_isolation(self):
        api.create_app()

        def local_report(image_path, **kwargs):
            report = self.fake_report(image_path, kwargs['output_root'], 'yolo')
            (Path(report['result_dir']) / 'annotated.png').write_bytes(fixtures.png_bytes())
            return report

        with TestClient(api.create_app(), client=('127.0.0.1', 50000),
                        headers=self.client.headers) as active:
            filename = active.post('/ui/gradio_api/upload', files={
                'files': ('queue.png', fixtures.png_bytes(), 'image/png')}).json()[0]
            payload = self.image_payload(filename)
            self.assertEqual(active.post('/ui/gradio_api/run/set_image', json=payload).status_code, 200)
            config = active.get('/ui/config').json()
            run = next(item for item in config['dependencies'] if item.get('api_name') == 'run_yolo')
            with patch.object(api.core, 'run_yolo_route', side_effect=local_report), \
                    patch.object(api.core, 'run_vlm_route') as remote:
                started = active.post('/ui/gradio_api/queue/join', json={
                    'data': [None, None, None, False, *payload['data'][1:]],
                    'fn_index': run['id'], 'session_hash': payload['session_hash']})
                self.assertEqual(started.status_code, 200, started.text)
                events = active.get('/ui/gradio_api/queue/data', params={
                    'session_hash': payload['session_hash']})
            remote.assert_not_called()
            messages = [json.loads(line[5:]) for line in events.text.splitlines()
                        if line.startswith('data:')]
            completed = next(item for item in messages if item.get('msg') == 'process_completed')
            self.assertTrue(completed['success'], completed)
            data = completed['output']['data']
            run_id = data[2]['_run_id']
            self.assertEqual(active.get(f'/api/v1/runs/{run_id}').status_code, 200)
            image_path = data[0][0]['image']['path']
            url = '/ui/gradio_api/file=' + quote(image_path.replace('\\', '/'), safe='/:')
            self.assertEqual(active.get(url).status_code, 200)
            with self.client_two() as other:
                self.assertEqual(other.get(url).status_code, 404)

    def test_logout_invalidates_uploaded_file_access(self):
        filename = self.upload_file()
        self.assertEqual(self.client.post('/api/auth/logout').status_code, 200)
        self.assertEqual(self.client.get('/ui/gradio_api/file=' +
                         quote(filename.replace('\\', '/'), safe='/:')).status_code, 401)

    def test_queued_callback_rechecks_revoked_login(self):
        headers = [(b'authorization', self.client.headers['Authorization'].encode())]
        request = Request({'type': 'http', 'headers': headers})
        wrapped = gr.Request(request=request, username=self.user_id)
        self.assertEqual(auth.callback_user(wrapped).user_id, self.user_id)
        self.client.post('/api/auth/logout')
        with self.assertRaises(HTTPException) as error:
            auth.callback_user(wrapped)
        self.assertEqual(error.exception.status_code, 401)

    def test_forged_report_is_reloaded_from_owner_scoped_database(self):
        with patch.object(api.core, 'run_yolo_route', side_effect=lambda path, **kw:
                          self.fake_report(path, kw['output_root'], 'yolo')):
            run_id = self.uploaded(route='yolo').json()['run_id']
        fake = {'_run_id': run_id, 'result_dir': str(api.ROOT / '.local'),
                'summary': [{'count': 9999}]}
        restored = owned_report(fake, 'yolo', self.user_id)
        self.assertNotEqual(restored['result_dir'], fake['result_dir'])
        self.assertEqual(restored['summary'][0]['count'], 1)
        self.assertIsNone(owned_report(fake, 'yolo', 'another-user'))
        self.assertIsNone(owned_report({'_run_id': '<script>'}, 'yolo', self.user_id))

    def test_model_paths_are_fixed_and_separate_route_downloads_are_preserved(self):
        with self.assertRaises(gr.Error):
            checked_values([None, 'finetuned', 'uploaded-malicious.pt', .25, .7, 640, 'auto', True])
        first, second = str(uuid4()), str(uuid4())
        links = export_link({'_run_id': first}, {'_run_id': second})
        self.assertIn(first, links)
        self.assertIn(second, links)
        self.assertEqual(export_link({'_run_id': first}, {'_run_id': first}).count('<a '), 1)

    def test_export_contains_sanitized_report_without_internal_paths(self):
        with patch.object(api.core, 'run_yolo_route', side_effect=lambda path, **kw:
                          self.fake_report(path, kw['output_root'], 'yolo')):
            run_id = self.uploaded(route='yolo').json()['run_id']
        from io import BytesIO
        from zipfile import ZipFile

        with ZipFile(BytesIO(self.client.get(f'/api/v1/runs/{run_id}/export.zip').content)) as bundle:
            report = json.loads(bundle.read('report.json'))
        self.assertEqual(report['run_id'], run_id)
        self.assertNotIn('result_dir', report['results']['yolo'])

    def test_cross_origin_cookie_mutation_and_remote_local_http_are_blocked(self):
        api.create_app()
        with TestClient(api.app, client=('127.0.0.1', 50000)) as browser:
            credentials = {'username': 'teacher_one', 'password': 'correct-horse-battery-staple'}
            self.assertEqual(browser.post('/api/auth/login', json=credentials,
                             headers={'Origin': 'http://evil.example'}).status_code, 403)
            self.assertEqual(browser.post('/api/auth/login', json=credentials).status_code, 200)
            self.assertEqual(browser.post('/api/auth/logout').status_code, 403)
            self.assertEqual(browser.post('/api/auth/logout',
                             headers={'Origin': 'http://testserver'}).status_code, 200)
        with TestClient(api.app, client=('192.0.2.10', 50000)) as remote:
            self.assertEqual(remote.get('/login').status_code, 403)

    def test_production_cookie_is_secure_and_all_launchers_use_loopback(self):
        with patch.object(auth, 'settings', return_value=('test-secret-for-production-32-bytes', False)):
            response = self.client.post('/api/auth/login', json={
                'username': 'teacher_one', 'password': 'correct-horse-battery-staple'})
            self.assertIn('; secure', response.headers['set-cookie'].lower())
        from deskvision.web.__main__ import main as web_main

        with patch('sys.argv', ['deskvision.web', '--port', '8002']), patch('uvicorn.run') as run:
            web_main()
        self.assertEqual(run.call_args.kwargs['host'], '127.0.0.1')
        self.assertEqual(run.call_args.kwargs['port'], 8002)
