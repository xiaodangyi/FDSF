"""Durability, duplicate-delivery, fencing and paid-call recovery contracts."""

from concurrent.futures import ThreadPoolExecutor
from time import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import update

from deskvision import api, queueing
from deskvision.executor import execute_run, now
from deskvision.models import RunRecord
import test_api as fixtures


class QueueTests(unittest.TestCase):
    setUp = fixtures.ApiTests.setUp
    tearDown = fixtures.ApiTests.tearDown
    uploaded = fixtures.ApiTests.uploaded
    fake_report = fixtures.ApiTests.fake_report

    def queued(self, route='both'):
        with patch.dict('os.environ', {'DESKVISION_QUEUE_MODE': 'celery'}), \
                patch.object(queueing, 'enqueue') as publish, \
                patch.object(api.core, 'run_yolo_route') as model:
            response = self.uploaded(route)
            self.assertEqual(response.status_code, 202, response.text)
            publish.assert_called_once_with(response.json()['run_id'])
            model.assert_not_called()
        return response.json()['run_id']

    def test_invalid_queue_mode_is_rejected(self):
        with patch.dict('os.environ', {'DESKVISION_QUEUE_MODE': 'typo'}):
            with self.assertRaises(RuntimeError):
                queueing.validate_configuration()

    def test_web_restart_preserves_queued_and_live_running_tasks(self):
        first, second = self.queued(), self.queued()
        self.assertTrue(api._repository().claim(second, now(), 'live-worker'))
        with patch.dict('os.environ', {'DESKVISION_QUEUE_MODE': 'celery',
                                     'DESKVISION_REDIS_URL': 'redis://127.0.0.1:6379/0'}):
            with TestClient(api.app, client=('127.0.0.1', 50000)):
                self.assertEqual(api._repository().get(first)['status'], 'queued')
                self.assertEqual(api._repository().get(second)['status'], 'running')

    def test_claim_is_atomic_across_simultaneous_deliveries(self):
        run_id = self.queued()
        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(lambda n: api._repository().claim(run_id, now(), str(n)), range(8)))
        self.assertEqual(results.count(True), 1)

    def test_duplicate_delivery_does_not_repeat_models(self):
        run_id = self.queued('yolo')
        with patch.object(api.core, 'run_yolo_route', side_effect=lambda path, **kw:
                          self.fake_report(path, kw['output_root'], 'yolo')) as local:
            execute_run(run_id)
            execute_run(run_id)
        self.assertEqual(local.call_count, 1)
        self.assertEqual(api._repository().get(run_id)['status'], 'succeeded')

    def test_broker_failure_retains_record_and_returns_queryable_id(self):
        with patch.dict('os.environ', {'DESKVISION_QUEUE_MODE': 'celery'}), \
                patch.object(queueing, 'enqueue', side_effect=ConnectionError('broker down')):
            response = self.uploaded('yolo')
        self.assertEqual(response.status_code, 503)
        run_id = response.json()['detail']['run_id']
        result = self.client.get('/api/v1/runs/' + run_id).json()
        self.assertEqual(result['status'], 'queued')
        self.assertEqual(result['error']['code'], 'delivery_uncertain')
        self.assertTrue(api._repository().get(run_id)['image_path'].is_file())

    def expire(self, run_id):
        with self.database.sessions.begin() as session:
            session.execute(update(RunRecord).where(RunRecord.run_id == run_id).values(
                heartbeat_at=int(time()) - 121))

    def test_crash_preserves_success_and_marks_unfinished_remote_unknown(self):
        run_id = self.queued()
        repo = api._repository()
        repo.claim(run_id, now(), 'worker-one')
        repo.route_started(run_id, 'worker-one', 'yolo')
        repo.route_started(run_id, 'worker-one', 'deepseek')
        repo.finish(run_id, status='running', results={'yolo': {'status': 'ok'}},
                    comparison=None, finished_at=now(), execution_id='worker-one', route_only=True)
        self.expire(run_id)
        self.assertEqual(repo.recover_expired(now()), 1)
        result = repo.get(run_id)
        self.assertEqual(result['status'], 'partial_failed')
        self.assertEqual(result['results']['yolo']['status'], 'ok')
        self.assertEqual(result['error']['code'], 'remote_result_unknown')
        self.assertEqual(result['results']['deepseek']['error']['code'], 'remote_result_unknown')
        with patch.object(api.core, 'run_vlm_route') as remote:
            execute_run(run_id)
        remote.assert_not_called()
        # A late success from the expired worker cannot overwrite recovery.
        repo.finish(run_id, status='succeeded', results={}, comparison=None,
                    finished_at=now(), execution_id='worker-one')
        self.assertEqual(repo.get(run_id)['status'], 'partial_failed')

    def test_live_heartbeat_and_queued_work_do_not_expire(self):
        run_id = self.queued()
        repo = api._repository()
        self.assertEqual(repo.recover_expired(now()), 0)
        repo.claim(run_id, now(), 'worker-one')
        self.expire(run_id)
        repo.heartbeat(run_id, 'worker-one')
        self.assertEqual(repo.recover_expired(now()), 0)
        self.assertFalse(repo.heartbeat(run_id, 'wrong-token'))

    def test_local_crash_is_failed_without_remote_unknown(self):
        run_id = self.queued('yolo')
        repo = api._repository()
        repo.claim(run_id, now(), 'worker-one')
        self.expire(run_id)
        repo.recover_expired(now())
        self.assertEqual(repo.get(run_id)['error']['code'], 'worker_interrupted')

    def test_remote_transport_timeout_is_unknown_and_never_retried(self):
        run_id = self.queued('deepseek')
        with patch.object(api.core, 'run_vlm_route', return_value={
            'status': 'failed', 'error': {'code': 'timeout', 'message': 'timeout'},
        }) as remote:
            execute_run(run_id)
            execute_run(run_id)
        self.assertEqual(remote.call_count, 1)
        result = api._repository().get(run_id)
        self.assertEqual(result['results']['deepseek']['error']['code'], 'remote_result_unknown')
        self.assertEqual(result['results']['deepseek']['error']['original_code'], 'timeout')

    def test_individual_route_state_is_visible_while_sibling_runs(self):
        run_id = self.queued()
        repo = api._repository()
        repo.claim(run_id, now(), 'token')
        repo.route_started(run_id, 'token', 'deepseek')
        repo.finish(run_id, status='running', results={'yolo': {'status': 'ok'}},
                    comparison=None, finished_at=now(), execution_id='token', route_only=True)
        result = self.client.get('/api/v1/runs/' + run_id).json()
        self.assertEqual(result['route_states'], {'yolo': 'ok', 'deepseek': 'running'})


if __name__ == '__main__':
    unittest.main()
