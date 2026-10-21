"""Concurrent requests must not lose steps or build duplicate shared state."""

import threading
import unittest

from backend.app import create_app


class TestConcurrentSteps(unittest.TestCase):
    def _parallel_steps(self, app, n):
        statuses, indices = [], []
        barrier = threading.Barrier(n)

        def go():
            client = app.test_client()
            barrier.wait()
            r = client.post("/api/v1/system/step", json={"workload_profile": "balanced", "num_processes": 4})
            statuses.append(r.status_code)
            if r.status_code == 200:
                indices.append(r.get_json()["step_index"])

        threads = [threading.Thread(target=go) for _ in range(n)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        return statuses, indices

    def test_cold_start_parallel_first_steps_all_recorded(self):
        # Before the start-up lock, 8 parallel first requests each built their own
        # execution manager and only 1 of the 8 steps reached the history.
        app = create_app(test_config={"TESTING": True})
        statuses, indices = self._parallel_steps(app, 8)
        self.assertEqual(statuses, [200] * 8)
        self.assertEqual(sorted(indices), list(range(8)))
        history = app.test_client().get("/api/v1/system/history?limit=50").get_json()
        self.assertEqual(history["total_steps"], 8)

    def test_warm_parallel_steps_get_unique_indices(self):
        app = create_app(test_config={"TESTING": True})
        app.test_client().post("/api/v1/system/step", json={"workload_profile": "io_bound"})
        statuses, indices = self._parallel_steps(app, 16)
        self.assertEqual(statuses, [200] * 16)
        self.assertEqual(sorted(indices), list(range(1, 17)))


class TestConcurrencyEval(unittest.TestCase):
    def test_eval_runs_and_passes_integrity(self):
        import tempfile

        from experiments import concurrency_eval

        with tempfile.TemporaryDirectory() as out:
            r = concurrency_eval.run(levels=[1, 4], n_requests=12, n_steps=10, out=out)
        self.assertTrue(r["integrity"]["passed"].all(), r["integrity"].to_string())
        self.assertEqual(int(r["load"]["server_errors_5xx"].sum()), 0)
