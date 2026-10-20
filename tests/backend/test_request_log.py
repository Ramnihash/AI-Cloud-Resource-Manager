"""Tests for the failed-request log behind the Activity page."""

import threading
import unittest

from backend.app import create_app
from backend.request_log import RequestErrorLog


class TestRequestErrorLog(unittest.TestCase):
    def test_ring_buffer_keeps_newest_and_counts_all(self):
        log = RequestErrorLog(max_entries=3)
        for i in range(5):
            log.record("GET", f"/api/x/{i}", 400, "bad" * 100, 1.234)
        recent = log.recent(10)
        self.assertEqual([e["path"] for e in recent], ["/api/x/4", "/api/x/3", "/api/x/2"])
        self.assertEqual(log.total, 5)
        self.assertEqual(len(recent[0]["error"]), 200)
        self.assertEqual(recent[0]["duration_ms"], 1.23)
        log.clear()
        self.assertEqual((log.total, log.recent(5)), (0, []))

    def test_thread_safe_counts(self):
        log = RequestErrorLog(max_entries=10_000)
        threads = [threading.Thread(target=lambda: [log.record("GET", "/api/a", 500, None, None) for _ in range(500)])
                   for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(log.total, 4000)
        self.assertEqual(len(log.recent(10_000)), 4000)


class TestRequestErrorEndpoint(unittest.TestCase):
    def setUp(self):
        self.app = create_app(test_config={"TESTING": True})
        self.client = self.app.test_client()

    def test_failed_api_requests_are_listed_newest_first(self):
        self.client.post("/api/v1/memory/page-replacement", json={})
        self.client.get("/api/v1/does-not-exist")
        self.client.get("/api/v1/system/history?limit=abc")
        body = self.client.get("/api/v1/system/errors").get_json()
        self.assertEqual(body["total_errors"], 3)
        first = body["errors"][0]
        self.assertEqual((first["method"], first["path"], first["status"]), ("GET", "/api/v1/system/history", 400))
        self.assertIn("limit", first["error"])
        self.assertEqual(body["errors"][1]["status"], 404)
        self.assertEqual(body["errors"][2]["method"], "POST")
        for entry in body["errors"]:
            self.assertIn("recorded_at", entry)
            self.assertIsInstance(entry["duration_ms"], float)

    def test_successes_dashboard_and_own_endpoint_not_recorded(self):
        self.client.get("/health")
        self.client.get("/dashboard/no-such-page")
        self.client.get("/api/v1/system/errors?limit=0")
        self.assertEqual(self.client.get("/api/v1/system/errors").get_json()["total_errors"], 0)

    def test_limit_validation(self):
        self.assertEqual(self.client.get("/api/v1/system/errors?limit=x").status_code, 400)
        self.assertEqual(self.client.get("/api/v1/system/errors?limit=201").status_code, 400)

    def test_server_error_recorded(self):
        @self.app.route("/api/v1/boom")
        def boom():
            raise RuntimeError("boom")

        self.assertEqual(self.client.get("/api/v1/boom").status_code, 500)
        entry = self.client.get("/api/v1/system/errors").get_json()["errors"][0]
        self.assertEqual((entry["status"], entry["error"]), (500, "Internal server error"))

    def test_reset_clears_log(self):
        self.client.get("/api/v1/nope")
        self.assertEqual(self.client.post("/api/v1/system/reset").status_code, 200)
        self.assertEqual(self.client.get("/api/v1/system/errors").get_json()["total_errors"], 0)
