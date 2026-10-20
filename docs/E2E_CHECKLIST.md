# End-to-end checklist

Run against the live site (or a local server). Rows 1 to 4, 7 and 8 are automated:

```
python -m scripts.e2e_check --base-url https://cloudproject.cv --user USER --password PASS
```

Rows 5, 6 and 9 are manual. Screenshots go in `docs/screenshots/` with the names below.

| # | Check | How | Expected | Automated |
|---|-------|-----|----------|-----------|
| 1 | All 5 pages open with no console errors | Open each page with DevTools console open | Pages render, console empty | Yes |
| 2 | Step System x3 raises total steps by 3 | Click Step System three times, compare Overview step count | Count up by 3, header shows "Step N complete" | Yes |
| 3 | Activity lists CPU, Memory and Device events | Open Activity after a step | Events from all three modules | Yes |
| 4 | Forced device failure shows on Device page | Run a step with a failing device, open Device | `disk_0` shown as failed or unhealthy | Yes |
| 5 | Backend down shows a readable error | See below | Overview shows "Backend unreachable", other pages show the red banner | Manual |
| 6 | Backend back up recovers | See below | Banner disappears after the next successful request, data refreshes | Manual |
| 7 | Malformed POST gives a 4xx JSON error | The script posts a JSON array to an endpoint | HTTP 400, body `{"error": ...}` | Yes |
| 8 | No failed requests or 4xx/5xx while browsing | The script watches network | None | Yes |
| 9 | Reset asks for confirmation and restarts at step 1 | See below | Dialog appears, cancel keeps history, OK clears it | Manual |

## Row 5 and 6: backend down and back

On the server (ssh window):

1. Open the Overview in a browser. Take `01-overview.png`.
2. `sudo systemctl stop ai-resource-manager`
3. Click Step System. Expected: "Step failed: Backend unreachable. Is the server running?". Take `07-backend-down.png`.
4. Open CPU or Memory and wait a few seconds. Expected: a red banner "Backend unreachable. Showing data from <time>, which may be out of date."
5. `sudo systemctl start ai-resource-manager`, wait about 5 seconds, click Step System.
   Expected: the banner disappears and the step succeeds. Step history restarts at step 1 because state is in memory.

## Row 9: reset

1. Click Step System twice (header shows "Step 2 complete").
2. Click Reset, then Cancel. Expected: nothing changes.
3. Click Reset, then OK. Expected: the page reloads, header empty, then Step System shows "Step 1 complete".

## Screenshots

| File | Page |
|------|------|
| `01-overview.png` | Overview after about 3 steps |
| `02-cpu.png` | CPU, showing the chosen scheduler |
| `03-memory.png` | Memory |
| `04-device.png` | Device |
| `05-activity.png` | Activity with CPU, Memory and Device events |
| `06-device-failure.png` | Device with `disk_0` failed |
| `07-backend-down.png` | Overview with the unreachable error |
| `08-https-lock.png` | Address bar with the padlock |

## Known gaps

- The header status chip ("Operational" or "Degraded") updates only on the Overview page.
- Failed requests (4xx/5xx) are not recorded on the Activity page.
- Not tested: concurrent users, and browsers other than Chromium.
