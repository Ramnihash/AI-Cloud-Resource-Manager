"""CP7.5 Robustness, part C: concurrent users.

Usage (from the repository root):

    python -m experiments.concurrency_eval                       # local threaded server
    python -m experiments.concurrency_eval --url http://127.0.0.1:8000 --label ec2-gunicorn

Without ``--url`` the app is started in this process on a threaded local server
(one thread per connection) with the real models. With ``--url`` the requests go
to a running deployment; on the EC2 instance ``http://127.0.0.1:8000`` reaches
gunicorn directly (one worker, 4 threads). For a URL behind Basic Auth set the
environment variables AIRM_USER and AIRM_PASSWORD rather than typing the password
on the command line.

Part 1  Load: at each concurrency level, ``--requests`` requests from that many
        parallel clients with a fixed mix (60% overview, 30% system step, 10% history).
        Reports throughput, latency percentiles and errors.
Part 2  Integrity under concurrent writes: N system steps sent in parallel. Every
        step must succeed, get its own step index, and appear in the history once.
Part 3  Cold start (local only): parallel steps sent to a fresh app before any
        request has trained the models, the moment lazy initialisation can race.

Writes to ``results/``: concurrency_latency.csv, concurrency_integrity.csv (rows are
appended with the ``--label`` so runs on different targets can sit side by side).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

RESULTS_DIR = "results"
LOCAL_LEVELS = [1, 4, 8, 16, 32]
REMOTE_LEVELS = [1, 2, 4, 8]
MIX = [("GET", "/api/v1/monitor/overview", 0.6), ("POST", "/api/v1/system/step", 0.3),
       ("GET", "/api/v1/system/history?limit=20", 0.1)]
PROFILES = ["cpu_bound", "io_bound", "priority_critical", "balanced"]
SEED = 42


class Client:
    def __init__(self, base_url: str, user: str | None = None, password: str | None = None, timeout: float = 120.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Accept": "application/json"}
        if user and password:
            token = base64.b64encode(f"{user}:{password}".encode()).decode()
            self.headers["Authorization"] = f"Basic {token}"

    def call(self, method: str, path: str, body: dict | None = None) -> tuple[int, Any, float]:
        data = json.dumps(body).encode() if body is not None else None
        headers = dict(self.headers)
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status, raw = resp.status, resp.read()
        except urllib.error.HTTPError as e:
            status, raw = e.code, e.read()
        except Exception as e:  # connection refused, reset, timeout
            return -1, {"error": f"{type(e).__name__}: {e}"}, (time.perf_counter() - t0) * 1000.0
        ms = (time.perf_counter() - t0) * 1000.0
        try:
            payload = json.loads(raw) if raw else None
        except ValueError:
            payload = None
        return status, payload, ms


def step_body(rng: random.Random) -> dict:
    return {"workload_profile": rng.choice(PROFILES), "num_processes": rng.randint(4, 8)}


def run_level(client: Client, concurrency: int, n_requests: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    plan = []
    for _ in range(n_requests):
        method, path, _ = rng.choices(MIX, weights=[w for *_, w in MIX])[0]
        plan.append((method, path, step_body(rng) if method == "POST" else None))

    def one(item: tuple[str, str, dict | None]) -> dict:
        method, path, body = item
        status, _, ms = client.call(method, path, body)
        return {"endpoint": path.split("?")[0], "status": status, "latency_ms": ms}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        t0 = time.perf_counter()
        rows = list(pool.map(one, plan))
        wall = time.perf_counter() - t0
    for r in rows:
        r["concurrency"] = concurrency
        r["wall_s"] = wall
    return rows


def summarise_load(rows: list[dict], label: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    out = []
    for (level, endpoint), g in [((lvl, "all"), g) for lvl, g in frame.groupby("concurrency")] + \
            list(frame.groupby(["concurrency", "endpoint"])):
        lat = g["latency_ms"].to_numpy()
        wall = float(g["wall_s"].iloc[0])
        out.append({
            "label": label, "concurrency": int(level), "endpoint": endpoint, "requests": len(g),
            "throughput_rps": len(g) / wall if endpoint == "all" else float("nan"),
            "non_2xx": int((g["status"] // 100 != 2).sum()), "server_errors_5xx": int((g["status"] >= 500).sum()),
            "connection_errors": int((g["status"] == -1).sum()),
            "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
            "p99_ms": float(np.percentile(lat, 99)), "max_ms": float(lat.max()),
        })
    return pd.DataFrame(out).sort_values(["concurrency", "endpoint"]).reset_index(drop=True)


def integrity(client: Client, n_steps: int, concurrency: int, label: str, scenario: str,
              seed: int = SEED, fresh: bool = False) -> dict:
    """Send ``n_steps`` steps in parallel and check every one is recorded exactly once.

    ``fresh`` means the app has served no request yet; asking for the history first
    would initialise it and hide the start-up race, so the count is taken as 0.
    """
    total_before = 0
    if not fresh:
        _, before, _ = client.call("GET", "/api/v1/system/history?limit=1")
        total_before = (before or {}).get("total_steps", 0)
    rng = random.Random(seed)
    bodies = [step_body(rng) for _ in range(n_steps)]
    barrier = threading.Barrier(min(concurrency, n_steps))

    def one(body: dict) -> tuple[int, Any]:
        try:
            barrier.wait(timeout=10)
        except threading.BrokenBarrierError:
            pass
        status, payload, _ = client.call("POST", "/api/v1/system/step", body)
        return status, payload

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, bodies))
    ok = [p for s, p in results if s == 200 and isinstance(p, dict)]
    indices = [p.get("step_index") for p in ok]
    _, hist, _ = client.call("GET", f"/api/v1/system/history?limit={min(200, max(n_steps, 1))}")
    total_after = (hist or {}).get("total_steps", 0)
    hist_indices = [s.get("step_index") for s in (hist or {}).get("steps", [])][:n_steps]
    return {
        "label": label, "scenario": scenario, "steps_sent": n_steps, "concurrency": concurrency,
        "steps_ok": len(ok), "non_200": n_steps - len(ok),
        "statuses": ",".join(sorted({str(s) for s, _ in results})),
        "distinct_step_indices": len(set(indices)), "duplicate_step_indices": len(indices) - len(set(indices)),
        "history_growth": total_after - total_before,
        "history_missing": len(ok) - (total_after - total_before),
        "history_distinct_indices": len(set(hist_indices)),
        "passed": bool(len(ok) == n_steps and len(set(indices)) == n_steps and total_after - total_before == n_steps
                       and len(set(hist_indices)) == len(hist_indices)),
    }


def start_local_server():
    """Start the real app on a threaded local server and return (base_url, server)."""
    from werkzeug.serving import make_server

    from backend.app import create_app

    server = make_server("127.0.0.1", 0, create_app(), threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}", server


def append_csv(frame: pd.DataFrame, path: str) -> None:
    header = not os.path.exists(path)
    frame.to_csv(path, mode="a", header=header, index=False)


def run(url: str | None = None, label: str | None = None, levels: list[int] | None = None,
        n_requests: int = 200, n_steps: int = 100, out: str = RESULTS_DIR) -> dict[str, pd.DataFrame]:
    os.makedirs(out, exist_ok=True)
    servers = []
    integrity_rows = []
    if url is None:
        label = label or "local-threaded"
        levels = levels or LOCAL_LEVELS
        cold_url, cold_server = start_local_server()
        servers.append(cold_server)
        integrity_rows.append(integrity(Client(cold_url), 8, 8, label, "cold start, 8 parallel first steps", fresh=True))
        url, server = start_local_server()
        servers.append(server)
    else:
        label = label or url
        levels = levels or REMOTE_LEVELS
    client = Client(url, os.environ.get("AIRM_USER"), os.environ.get("AIRM_PASSWORD"))
    try:
        status, _, _ = client.call("POST", "/api/v1/system/step", {"workload_profile": "balanced"})  # warm up models
        if status != 200:
            raise RuntimeError(f"Warm-up step failed with status {status}; check the URL and credentials")
        rows = []
        for i, level in enumerate(levels):
            rows.extend(run_level(client, level, n_requests, SEED + i))
        load = summarise_load(rows, label)
        for level in (max(levels), max(4, max(levels) // 2)):
            integrity_rows.append(integrity(client, n_steps, level, label, f"{n_steps} parallel steps, concurrency {level}"))
        integ = pd.DataFrame(integrity_rows)
    finally:
        for s in servers:
            s.shutdown()
    append_csv(load, os.path.join(out, "concurrency_latency.csv"))
    append_csv(integ, os.path.join(out, "concurrency_integrity.csv"))
    return {"load": load, "integrity": integ}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Concurrent-user load and integrity test.")
    parser.add_argument("--url", help="base URL of a running deployment (default: start one locally)")
    parser.add_argument("--label", help="name for this run in the CSVs")
    parser.add_argument("--levels", help="comma-separated concurrency levels, e.g. 1,2,4,8")
    parser.add_argument("--requests", type=int, default=200, help="requests per concurrency level")
    parser.add_argument("--steps", type=int, default=100, help="parallel steps in the integrity check")
    args = parser.parse_args(argv)
    levels = [int(x) for x in args.levels.split(",")] if args.levels else None
    r = run(args.url, args.label, levels, args.requests, args.steps)
    cols = ["concurrency", "requests", "throughput_rps", "non_2xx", "server_errors_5xx", "p50_ms", "p95_ms", "p99_ms", "max_ms"]
    print(r["load"][r["load"].endpoint == "all"][cols].round(1).to_string(index=False))
    print()
    print(r["integrity"].drop(columns=["label"]).to_string(index=False))


if __name__ == "__main__":
    main()
