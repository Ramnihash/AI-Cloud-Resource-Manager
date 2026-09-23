---
marp: true
theme: default
paginate: true
---

# AI-Driven Cloud Resource Manager
## Review 2

**Team Members:**
- Sinchana (Member 2) - Predictive Memory Management & ML Pipeline
- Ramnihash (Member 3) - Device Management & Predictive Maintenance
- [Member 1] - AI-Assisted CPU Scheduling

**Project Guide / Mentor:** [Insert Name]
**Institution:** [Insert Institution]

<!--
Speaker Notes:
- What to say: "Welcome to our Review 2 presentation for the AI-Driven Cloud Resource Manager. Today, we will present the architecture, our progress since Review 1, and the specific implementations we have completed. I am Sinchana, responsible for Module 2: Predictive Memory Management."
- Professor Questions: "What exactly is the scope of your project?"
-->

---

## PROJECT OBJECTIVE

**The Problem:**
- Traditional OS resource management is entirely *reactive* (e.g., evicting a page only *after* a page fault).
- Cloud environments require proactive scaling and maintenance to prevent bottlenecks.

**Our AI-Driven Solution:**
1. **PREDICT:** Learn workload patterns (Page sequences, memory growth, I/O latency).
2. **DECIDE:** Determine the best intervention based on predictions.
3. **ACT:** Proactively manage resources (Prefetching, Restarting, Rerouting).
4. **MEASURE:** Evaluate the outcome (Hit ratio, False positives).

<!--
Speaker Notes:
- What to say: "Operating systems traditionally manage resources reactively. If memory is full, it evicts a page. If a disk fails, it throws an error. Our project introduces a proactive cycle: Predict, Decide, Act, and Measure."
- Professor Questions: "Why can't traditional algorithms handle cloud workloads?" (Answer: They lack foresight. AI models can detect long-term structural trends and spatial/temporal locality better than simple heuristic caches).
-->

---

## SYSTEM ARCHITECTURE

```text
[ Input / Telemetry Sources ]
    (Synthetic Traces, psutil, SMART data)
             │
             ▼
[ OS Resource Manager Modules ]
 ┌───────────┬────────────┬─────────────┐
 │  CPU      │   Memory   │   Device    │
 │ (Module 1)│ (Module 2) │ (Module 3)  │
 └───────────┴────────────┴─────────────┘
             │
             ▼
[ ML / Intelligence Layer (PREDICT) ]
    (Regression, Markov, Transformers)
             │
             ▼
[ Decision / Action Layer (ACT) ]
    (Prefetch, Re-route, Restart, Alert)
```

<!--
Speaker Notes:
- What to say: "Our system architecture strictly decouples classical OS mechanisms from the AI models. Telemetry feeds into the Intelligence layer, which generates predictions. The Action layer consumes these predictions to send simulated control directives back to the OS."
- Important Technical Detail: The architecture explicitly separates PREDICT from ACT. This allows ML models to be trained and evaluated in isolation before allowing them to interfere with OS state.
-->

---

## IMPLEMENTATION STATUS

**Implemented:**
- **Module 2 (Sinchana):** Classical Memory Algorithms (CP1.2-1.3), Telemetry Generators (CP2.2-2.3), ML Predictors (CP3.3-3.6), Action Layers (CP4.2-4.3). Fully verified with 282 passing tests.
- **Module 3 (Ramnihash):** Transformer Device Anomaly Detector (CP3.7).

**Partially Implemented:**
- Cross-module integration.

**Remaining (Planned):**
- **Phase 5:** Backend APIs (Flask/FastAPI) to serve the predictions.
- **Phase 6:** Dashboard / UI for live visualization.

<!--
Speaker Notes:
- What to say: "Since Review 1, we have implemented the core data and intelligence layers for our respective modules. For my module, Memory Management, I have completed all implementation through Checkpoint 4.3, meaning the entire pipeline from raw data to simulated action is functional."
-->

---

## OTHER TEAM CONTRIBUTIONS

**Module 1: CPU Scheduling**
- **Functionality:** AI-Assisted Meta-Scheduler.
- **Planned ML:** Burst-time regression model & Workload classifier.
- **Action:** Dynamically selecting between FCFS, SJF, RR, etc., based on predicted workload.

**Module 3: Device Management (Ramnihash)**
- **Functionality:** Device Anomaly Detection & Predictive Maintenance.
- **Implemented ML:** Transformer-based sequence anomaly detector (CP3.7).
- **Action:** Flag unhealthy devices, reroute I/O, trigger cloud backups.

<!--
Speaker Notes:
- What to say: "While I focus on memory, the other modules follow the same paradigm. Module 1 uses regression to classify CPU workloads and dynamically switch schedulers. Module 3 uses Transformers and SMART disk telemetry to predict hardware failures and reroute I/O."
-->

---

## MODULE 2: MEMORY MANAGEMENT

**Core Responsibility (Sinchana):**
Applying the AI-driven paradigm to OS Memory.

**1. Classical Algorithms:** Provide the baseline (FIFO, LRU, Allocators).
**2. Data/Telemetry:** Simulate page traces and collect live `psutil` data.
**3. ML Prediction:** Predict next-page accesses and system RAM saturation.
**4. Action Layer:** Translate predictions into actionable OS directives (Prefetch, Restart).

<!--
Speaker Notes:
- What to say: "My primary responsibility is Module 2. I built the system from the ground up: starting with the classical algorithms like LRU, building the telemetry collectors, training the ML models, and finally wiring them into the Action Layer."
- Professor Questions: "How is your module different from standard OS memory management?" (Answer: Standard OS is purely reactive. My module uses ML to predict the future state and acts before a bottleneck or fault occurs).
-->

---

## CLASSICAL MEMORY MANAGEMENT

**Page Replacement Algorithms:**
- **FIFO (First-In, First-Out):** Evicts the oldest page.
- **LRU (Least Recently Used):** Evicts the page unused for the longest time (Implemented in $O(1)$ time using `OrderedDict`).
- **Optimal (Theoretical):** Evicts the page needed furthest in the future.

**Memory Allocators:**
- **First Fit:** Allocates the first available block that fits.
- **Best Fit:** Allocates the tightest-fitting block to reduce fragmentation.
- **Worst Fit:** Allocates the largest block to leave maximal remaining contiguous space.

<!--
Speaker Notes:
- What to say: "To measure how much AI improves the system, we need baselines. I implemented FIFO, LRU, and Optimal replacement algorithms. Specifically, LRU is implemented using an `OrderedDict` to achieve O(1) time complexity for both hits and evictions."
- Professor Questions: "Why use an `OrderedDict`?" (Answer: It combines a hash map for O(1) lookup and a doubly-linked list for O(1) reordering via `move_to_end()`).
-->

---

## MEMORY DATA / TELEMETRY PIPELINE

**Synthetic Traces (Page Reference Generator):**
- Generates sequences simulating Spatial/Temporal Locality, Loops, and Random behavior for training Page Predictors.

**Live Telemetry (`MemoryMonitor`):**
- Uses `psutil` to collect time-series snapshots.
- **System Metrics:** RAM utilization (%), Swap utilization (%).
- **Process Metrics:** RSS (Resident Set Size in bytes), VMS.

<!--
Speaker Notes:
- What to say: "Machine learning requires robust data. Because we are a simulator, I built a `PageReferenceGenerator` that mathematically simulates OS spatial and temporal locality. For system health, I built a `MemoryMonitor` that polls live `psutil` metrics to track RAM, Swap, and per-process RSS bytes."
-->

---

## ML: PAGE PREDICTION

**1. Markov Predictor (First-Order):**
- *Input:* Current Page.
- *Processing:* Counts historical $A \rightarrow B$ transitions, normalizes to probabilities.
- *Output:* Top $K$ next page candidates.

**2. Transformer Predictor (PyTorch):**
- *Input:* Sliding window of recent pages (Context length).
- *Processing:* Uses Self-Attention (`nn.TransformerEncoderLayer`) to capture deep, long-term dependencies (e.g., returning from a deep function call).
- *Output:* Top $K$ next page candidates via Cross-Entropy classification.

<!--
Speaker Notes:
- What to say: "To predict page faults, I implemented two models. A fast First-Order Markov chain that relies on immediate history, and a PyTorch-based Transformer that uses self-attention over a sliding context window to learn complex, long-term program behaviors."
- Professor Questions: "Why use a Transformer for OS pages?" (Answer: Markov chains only remember the very last page. They fail on nested loops. Transformers look at the entire context window, allowing them to understand deep sequential patterns).
-->

---

## ML: MEMORY HEALTH

**1. Memory Pressure Forecaster:**
- *Input:* Sliding window of past RAM percentages (e.g., $t_1 \dots t_5$).
- *Processing:* Time-series `LinearRegression`. Clamps output to $[0, 100]$.
- *Output:* Predicted RAM % at $t_6$.

**2. Memory Leak Detector (Zero-Shot):**
- *Input:* Sliding window of a specific PID's RSS bytes over time.
- *Processing:* Fits `LinearRegression`. Calculates the slope (bytes/step).
- *Output:* Leak detected if **Slope > Threshold** AND **$R^2$ Score > 0.75**.

<!--
Speaker Notes:
- What to say: "For system health, I built two models. The Pressure Forecaster predicts RAM saturation using time-series regression. The Leak Detector is particularly robust: it uses regression on a process's memory footprint over time. It relies on the R^2 score to differentiate a true, linear memory leak from a chaotic, benign memory spike."
- Professor Questions: "Why not use Isolation Forests for leaks?" (Answer: Isolation forests detect point outliers. A memory leak is not an outlier, it's a structural linear trend over time. Regression accurately models that rate of change).
-->

---

## CP4.2 MEMORY PAGE PREDICTION ACTION

**The Decision Layer for Pages:**
- Consumes predictions from either the Markov or Transformer models.
- Compares the predicted page against the current physical memory frames.

**Simulated Actions:**
- **PREFETCH:** If the predicted page is *not* in memory, issue a directive to load it before the fault occurs.
- **PROTECT:** If the predicted page *is* in memory, issue a directive protecting it from LRU eviction.

<!--
Speaker Notes:
- What to say: "Checkpoint 4.2 bridges the gap between intelligence and action. It takes the Transformer's prediction, checks the physical memory state, and issues a simulated directive to either PREFETCH the page or PROTECT it from the LRU replacement algorithm."
- Important Technical Detail: Emphasize that these are safe, simulated directives represented as structured data (`{"action": "PREFETCH", "reason": "..."}`), not actual OS kernel hacks.
-->

---

## CP4.3 MEMORY HEALTH ACTION

**The Decision Layer for System Health:**
- Consumes outputs from both the Leak Detector and the Pressure Forecaster.

**Priority Action Logic:**
1. If **Leak Detected** $\rightarrow$ simulated `RESTART_PROCESS` directive for the offending PID.
2. Else if **Predicted RAM > Threshold** (e.g., 90%) $\rightarrow$ `ALERT_HIGH_PRESSURE`.
3. Else $\rightarrow$ `NO_ACTION`.

<!--
Speaker Notes:
- What to say: "Checkpoint 4.3 handles system health actions using a strict priority queue. A detected memory leak is critical, resulting in a simulated process restart directive. If there is no leak but forecasted RAM is critically high, it triggers an alert. We do not actually terminate host processes; we generate safe simulation logs."
-->

---

## PREDICT → DECIDE → ACT → MEASURE

**Completing the Loop in Module 2:**

1. **PREDICT:** (Transformer, Markov, Linear Regression) identify anomalies and future states.
2. **DECIDE:** Action Layers (CP4.2, CP4.3) evaluate predictions against thresholds and current state.
3. **ACT:** Simulated directives (`PREFETCH`, `RESTART_PROCESS`) are generated.
4. **MEASURE:** `PageReplacement` algorithms and internal logs record the impact (e.g., improved Hit Ratio).

<!--
Speaker Notes:
- What to say: "This slide shows how my implementations successfully fulfill the core architectural vision of the project. We have successfully completed the loop from raw telemetry, through ML predictions, and into actionable OS directives."
-->

---

## TESTING & VALIDATION

**Rigorous Verification:**
- We rely on `unittest` to ensure classical mechanisms are never broken by AI updates.
- **Total Tests Passing:** 282 tests.
- **Coverage Highlights:**
  - $O(1)$ assertions for LRU caches.
  - Mathematical boundary tests (Forecasts clamped to $100\%$).
  - Action Priority tests (Ensuring a leak always supersedes high pressure).
- Zero regressions across CP1.x $\rightarrow$ CP4.x.

<!--
Speaker Notes:
- What to say: "Because OS components are highly interdependent, I wrote extensive unit tests. We currently have 282 passing tests in the memory module alone. This ensures that our new AI Action layers do not break the classical algorithms."
-->

---

## DEMONSTRATION

*(See Demo Console Output)*

**Highlights:**
- Classical algorithms producing Page Fault counts.
- Transformer predicting the next logical page.
- Forecaster predicting $75.0\%$ RAM.
- Leak Detector catching a synthetic memory leak ($R^2 = 1.0$).
- Action Layer converting the leak into a `RESTART_PROCESS` directive.

<!--
Speaker Notes:
- What to say: "I have written a self-contained demonstration script that sequentially executes the entire Module 2 pipeline. It shows the classical algorithms running, trains the ML models on synthetic traces, and outputs the final Action Layer decisions based on the predictions."
-->

---

## CURRENT STATUS & NEXT STEPS

**Completed (Module 2):**
- ✅ Classical replacement/allocation.
- ✅ Telemetry and Synthetic Traces.
- ✅ ML Predictors (Markov, Transformer, Regression).
- ✅ Action Decision Layers (CP4.2, CP4.3).

**Next Steps (Phase 5 & 6):**
- 🚀 Connect predictions to a Backend API (Flask/FastAPI).
- 🚀 Expose metrics to a unified Dashboard.
- 🚀 Finalize cross-module orchestration with the CPU and Device teams.

<!--
Speaker Notes:
- What to say: "To summarize, the intelligence and decision logic for memory management is complete. Moving forward into Review 3, our focus shifts to Phase 5: exposing these systems via a Backend API and integrating them into a live dashboard for final demonstration."
-->
