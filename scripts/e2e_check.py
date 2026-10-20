"""Browser end-to-end check of the five dashboard pages (CP6.2f, rows 1-4, 7, 8).

Usage (from the repository root, with the app running):

    pip install playwright
    playwright install chromium
    python -m scripts.e2e_check --base-url http://127.0.0.1:5000

Against the live site with Basic Auth:

    python -m scripts.e2e_check --base-url https://cloudproject.cv --user USER --password PASS

Rows 5 and 6 (stop and restart the server while a page is open) need a human; see
docs/E2E_CHECKLIST.md. Writes screenshots and ``e2e_results.md`` to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

PAGES = [
    ("Overview", "/dashboard/"),
    ("CPU", "/dashboard/cpu"),
    ("Memory", "/dashboard/memory"),
    ("Device", "/dashboard/device"),
    ("Activity", "/dashboard/activity"),
]


def run(base_url: str, out: str, user: str | None, password: str | None, chromium: str | None) -> list[tuple[str, bool, str]]:
    from playwright.sync_api import sync_playwright

    os.makedirs(out, exist_ok=True)
    results: list[tuple[str, bool, str]] = []
    base = base_url.rstrip("/")

    with sync_playwright() as pw:
        launch = {"executable_path": chromium} if chromium else {}
        browser = pw.chromium.launch(**launch)
        context = browser.new_context(
            http_credentials={"username": user, "password": password} if user else None,
            viewport={"width": 1440, "height": 900},
        )
        page = context.new_page()
        console_errors: list[str] = []
        bad_responses: list[str] = []
        failed_requests: list[str] = []
        page.on("console", lambda m: console_errors.append(f"{page.url}: {m.text}") if m.type == "error" else None)
        page.on("pageerror", lambda e: console_errors.append(f"{page.url}: {e}"))
        page.on("requestfailed", lambda r: failed_requests.append(f"{r.url}: {r.failure}"))
        page.on("response", lambda r: bad_responses.append(f"{r.status} {r.url}") if r.status >= 400 else None)

        def api(method: str, path: str, **kw):
            return getattr(context.request, method)(base + path, **kw)

        # Row 1: every page renders with no console errors
        render_problems = []
        for name, path in PAGES:
            console_errors.clear()
            response = page.goto(base + path, wait_until="networkidle")
            page.screenshot(path=os.path.join(out, f"{name.lower()}.png"), full_page=True)
            if response is None or response.status != 200:
                render_problems.append(f"{name}: HTTP {response.status if response else 'none'}")
            render_problems += [f"{name}: {e}" for e in console_errors]
        results.append(("1. Open all 5 routes, no console errors", not render_problems, "; ".join(render_problems) or "all 5 rendered"))

        # Row 2: Step System x3 changes the overview
        page.goto(base + "/dashboard/", wait_until="networkidle")
        before = api("get", "/api/v1/monitor/overview").json()["cpu"]["total_steps"]
        for _ in range(3):
            with page.expect_response(lambda r: r.url.endswith("/api/v1/system/step")):
                page.click("#step-system-btn")
            page.wait_for_function("!document.getElementById('step-system-btn').disabled")
        after = api("get", "/api/v1/monitor/overview").json()["cpu"]["total_steps"]
        feedback = page.inner_text("#step-feedback")
        results.append(("2. Step System x3 raises total_steps by 3", after - before == 3, f"{before} -> {after}; feedback '{feedback}'"))

        # Rows 3 and 4: one integrated step with a forced device failure shows up on Activity and Device
        step = api("post", "/api/v1/system/step", data=json.dumps({
            "workload_profile": "balanced", "num_processes": 5,
            "device_actions": [{"device_id": "disk_0", "failure_probability": 0.95, "device_status": "unhealthy"}],
        }), headers={"Content-Type": "application/json"})
        ok_step = step.status == 200
        page.goto(base + "/dashboard/activity", wait_until="networkidle")
        text = page.inner_text("body")
        modules_seen = [m for m in ("CPU", "Memory", "Device") if m in text]
        results.append(("3. Activity lists CPU, Memory and Device events", ok_step and len(modules_seen) == 3, f"step HTTP {step.status}; modules on page: {modules_seen}"))
        page.goto(base + "/dashboard/device", wait_until="networkidle")
        dev_text = page.inner_text("body").lower()
        results.append(("4. Forced device failure visible on Device page", ok_step and ("disk_0" in dev_text) and ("failed" in dev_text or "unhealthy" in dev_text), "page mentions disk_0 and a failed/unhealthy state" if "disk_0" in dev_text else "disk_0 not found on page"))

        # Row 7: malformed POST returns a 4xx JSON error
        bad = api("post", "/api/v1/cpu/action", data="{bad", headers={"Content-Type": "application/json"})
        body = None
        try:
            body = bad.json()
        except Exception:
            pass
        results.append(("7. Malformed POST gives 4xx JSON error", 400 <= bad.status < 500 and isinstance(body, dict) and "error" in body, f"HTTP {bad.status}, body {body}"))

        # Row 8: no failed requests or 4xx/5xx while browsing (the deliberate malformed call above is made outside the page)
        page_problems = failed_requests + bad_responses
        results.append(("8. No failed requests or 4xx/5xx while browsing", not page_problems, "; ".join(page_problems[:5]) or "none"))
        browser.close()
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--out", default="results/e2e")
    parser.add_argument("--user", default=os.environ.get("E2E_USER"))
    parser.add_argument("--password", default=os.environ.get("E2E_PASSWORD"))
    parser.add_argument("--chromium", default=os.environ.get("E2E_CHROMIUM"), help="path to a Chromium binary (optional)")
    args = parser.parse_args()

    results = run(args.base_url, args.out, args.user, args.password, args.chromium)
    lines = ["| Test | Result | Notes |", "|------|--------|-------|"]
    for name, ok, note in results:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {note} |")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.out, "e2e_results.md"), "w", encoding="utf-8") as handle:
        handle.write(report + "\n")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
