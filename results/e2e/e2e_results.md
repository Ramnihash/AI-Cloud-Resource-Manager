| Test | Result | Notes |
|------|--------|-------|
| 1. Open all 5 routes, no console errors | PASS | all 5 rendered |
| 2. Step System x3 raises total_steps by 3 | PASS | 0 -> 3; feedback 'Step 3 complete' |
| 3. Activity lists CPU, Memory and Device events | PASS | step HTTP 200; modules on page: ['CPU', 'Memory', 'Device'] |
| 4. Forced device failure visible on Device page | PASS | page mentions disk_0 and a failed/unhealthy state |
| 7. Malformed POST gives 4xx JSON error | PASS | HTTP 400, body {'error': 'JSON payload must be a JSON object'} |
| 8. No failed requests or 4xx/5xx while browsing | PASS | none |
