(function () {
    "use strict";

    var HISTORY_ENDPOINT = "/api/v1/system/history?limit=10";
    var SERIES_COUNT = 8;

    function byId(id) {
        return document.getElementById(id);
    }

    function setText(id, text, keepMuted) {
        var el = byId(id);
        if (!el) {
            return;
        }
        el.textContent = text;
        if (!keepMuted) {
            el.classList.remove("is-muted");
        }
    }

    function num(value, digits) {
        if (typeof value !== "number" || isNaN(value)) {
            return null;
        }
        return value.toFixed(digits === undefined ? 1 : digits);
    }

    function numOrDash(value, digits) {
        var out = num(value, digits);
        return out === null ? "—" : out;
    }

    function el(tag, className, text) {
        var node = document.createElement(tag);
        if (className) {
            node.className = className;
        }
        if (text !== undefined && text !== null) {
            node.textContent = text;
        }
        return node;
    }

    function clear(node) {
        while (node && node.firstChild) {
            node.removeChild(node.firstChild);
        }
    }

    function showError(message) {
        var banner = byId("cpu-error");
        if (!banner) {
            return;
        }
        banner.hidden = !message;
        banner.textContent = message || "";
    }

    function comparePid(a, b) {
        var na = Number(a);
        var nb = Number(b);
        if (!isNaN(na) && !isNaN(nb)) {
            return na - nb;
        }
        return String(a).localeCompare(String(b));
    }

    function seriesIndex(pids, pid) {
        var i = pids.indexOf(String(pid));
        return i < 0 ? 0 : i % SERIES_COUNT;
    }

    function renderDecision(step) {
        var cpu = step.cpu || {};
        var meta = cpu.meta_result || {};
        var cls = meta.classification_result || {};

        setText("cpu-scheduler", cpu.selected_scheduler || "—");
        var selection = cpu.is_fallback ? "Fallback (classifier unsure)" : "Chosen by classifier";
        setText("cpu-selection", selection);
        setText("cpu-class", cpu.workload_class || "—");
        setText("cpu-process-count", typeof cpu.process_count === "number" ? String(cpu.process_count) : "—");

        var conf = typeof cpu.confidence === "number" ? Math.round(cpu.confidence * 100) + "% confidence" : "";
        setText("cpu-confidence", conf || " ", true);
        setText("cpu-decision-status", cpu.is_fallback ? "Fallback" : "Active", true);

        var list = byId("cpu-probs");
        clear(list);
        var probs = cls.class_probabilities;
        if (probs && typeof probs === "object") {
            Object.keys(probs)
                .sort(function (a, b) { return probs[b] - probs[a]; })
                .forEach(function (name) {
                    var pct = Math.max(0, Math.min(100, probs[name] * 100));
                    var row = el("li", "prob-row");
                    row.appendChild(el("span", "prob-name", name.replace(/_/g, " ")));
                    var bar = el("span", "prob-bar");
                    var fill = el("span", "prob-fill");
                    fill.style.width = pct + "%";
                    bar.appendChild(fill);
                    row.appendChild(bar);
                    row.appendChild(el("span", "prob-pct", pct.toFixed(0) + "%"));
                    list.appendChild(row);
                });
        } else {
            list.appendChild(el("li", "prob-empty", "Not reported for this step"));
        }
    }

    function renderMetrics(metrics) {
        setText("kpi-waiting", numOrDash(metrics.avg_waiting_time));
        setText("kpi-turnaround", numOrDash(metrics.avg_turnaround_time));
        setText("kpi-response", numOrDash(metrics.avg_response_time));
        var overhead = num(metrics.scheduling_overhead_ms, 3);
        setText("kpi-overhead", overhead === null ? "—" : overhead + " ms");
    }

    function renderGantt(schedule) {
        var segments = Array.isArray(schedule.gantt_chart) ? schedule.gantt_chart : [];
        var processes = Array.isArray(schedule.processes) ? schedule.processes : [];
        var empty = byId("gantt-empty");
        var wrap = byId("gantt");
        if (!segments.length) {
            empty.hidden = false;
            wrap.hidden = true;
            return;
        }
        empty.hidden = true;
        wrap.hidden = false;

        var pids = processes.map(function (p) { return String(p.pid); });
        segments.forEach(function (s) {
            if (!s.is_idle && pids.indexOf(String(s.pid)) < 0) {
                pids.push(String(s.pid));
            }
        });
        // Colour follows the process id, not completion order.
        pids.sort(comparePid);

        var end = 0;
        segments.forEach(function (s) {
            end = Math.max(end, Number(s.end_time) || 0);
        });
        if (end <= 0) {
            end = 1;
        }

        var track = byId("gantt-track");
        clear(track);
        segments.forEach(function (s) {
            var start = Number(s.start_time) || 0;
            var dur = Number(s.duration);
            if (isNaN(dur)) {
                dur = (Number(s.end_time) || 0) - start;
            }
            var seg = el("span", "gantt-seg" + (s.is_idle ? " is-idle" : ""));
            seg.style.left = (start / end * 100) + "%";
            seg.style.width = (dur / end * 100) + "%";
            if (!s.is_idle) {
                seg.style.setProperty("--seg-color", "var(--series-" + (seriesIndex(pids, s.pid) + 1) + ")");
                seg.appendChild(el("span", "gantt-seg-label", "P" + s.pid));
            }
            var label = (s.is_idle ? "Idle" : "Process " + s.pid) + ": " + num(start) + " to " + num(start + dur) + " (" + num(dur) + " units)";
            seg.title = label;
            seg.setAttribute("aria-label", label);
            seg.tabIndex = 0;
            track.appendChild(seg);
        });

        var axis = byId("gantt-axis");
        clear(axis);
        var ticks = 5;
        for (var i = 0; i <= ticks; i++) {
            var tick = el("span", "gantt-tick", String(Math.round(end * i / ticks * 10) / 10));
            tick.style.left = (i / ticks * 100) + "%";
            axis.appendChild(tick);
        }

        var legend = byId("gantt-legend");
        clear(legend);
        pids.forEach(function (pid, idx) {
            var item = el("li", "gantt-legend-item");
            var swatch = el("span", "gantt-swatch");
            swatch.style.setProperty("--seg-color", "var(--series-" + ((idx % SERIES_COUNT) + 1) + ")");
            item.appendChild(swatch);
            item.appendChild(el("span", null, "P" + pid));
            legend.appendChild(item);
        });

        setText("gantt-meta", (schedule.algorithm_name || "Schedule") + " · " + num(end) + " time units");
    }

    function renderProcesses(schedule) {
        var rows = Array.isArray(schedule.processes) ? schedule.processes.slice() : [];
        rows.sort(function (a, b) { return comparePid(a.pid, b.pid); });
        var body = byId("proc-body");
        clear(body);
        setText("proc-meta", rows.length + (rows.length === 1 ? " process" : " processes"), true);
        rows.forEach(function (p) {
            var tr = document.createElement("tr");
            [
                ["P" + p.pid, "cell-strong"],
                [numOrDash(p.arrival_time)],
                [numOrDash(p.burst_time)],
                [numOrDash(p.predicted_burst_time)],
                [p.priority === undefined || p.priority === null ? "—" : String(p.priority)],
                [numOrDash(p.start_time)],
                [numOrDash(p.completion_time)],
                [numOrDash(p.waiting_time)],
                [numOrDash(p.turnaround_time)],
                [numOrDash(p.response_time)]
            ].forEach(function (cell) {
                tr.appendChild(el("td", cell[1] || "cell-num", cell[0]));
            });
            body.appendChild(tr);
        });
    }

    function renderDecisions(steps) {
        var body = byId("decisions-body");
        clear(body);
        setText("decisions-meta", steps.length + (steps.length === 1 ? " step" : " steps"), true);
        var list = el("ul", "step-list");
        steps.forEach(function (step) {
            var cpu = step.cpu || {};
            var metrics = ((cpu.meta_result || {}).schedule_result || {}).metrics || {};
            var row = el("li", "step-row step-row--cpu");
            row.appendChild(el("span", "step-row-module", "STEP " + (step.step_index + 1)));
            var text = el("span", "step-row-text");
            text.appendChild(el("span", "step-row-title",
                (cpu.workload_class || "Unknown") + " → " + (cpu.selected_scheduler || "?") + (cpu.is_fallback ? " (fallback)" : "")));
            var detail = [];
            if (typeof cpu.process_count === "number") {
                detail.push(cpu.process_count + " processes");
            }
            if (num(metrics.avg_waiting_time) !== null) {
                detail.push("avg waiting " + num(metrics.avg_waiting_time));
            }
            if (num(metrics.cpu_utilization) !== null) {
                detail.push("utilization " + num(metrics.cpu_utilization) + "%");
            }
            if (typeof cpu.timestamp === "number") {
                detail.push(new Date(cpu.timestamp * 1000).toLocaleTimeString());
            }
            text.appendChild(el("span", "step-row-detail", detail.join(" · ")));
            row.appendChild(text);
            list.appendChild(row);
        });
        body.appendChild(list);
    }

    function render(data) {
        var steps = data && Array.isArray(data.steps) ? data.steps : [];
        if (!steps.length) {
            return;
        }
        var latest = steps[0];
        var schedule = ((latest.cpu || {}).meta_result || {}).schedule_result || {};

        renderDecision(latest);
        renderMetrics(schedule.metrics || {});
        renderGantt(schedule);
        renderProcesses(schedule);
        renderDecisions(steps);
        setText("cpu-meta", "Step " + (latest.step_index + 1) + " of " + data.total_steps, true);
        if (latest.timestamp) {
            var updated = new Date(latest.timestamp);
            setText("last-updated-value", isNaN(updated.getTime()) ? String(latest.timestamp) : updated.toLocaleString(), true);
        }
    }

    function load() {
        fetch(HISTORY_ENDPOINT, { headers: { Accept: "application/json" } })
            .then(function (response) {
                if (!response.ok) {
                    throw new Error("History request failed with status " + response.status);
                }
                return response.json();
            })
            .then(function (data) {
                showError("");
                render(data);
            })
            .catch(function (error) {
                showError(error instanceof TypeError
                    ? "Backend unreachable. Is the server running?"
                    : error.message);
            });
    }

    function init() {
        load();
        document.addEventListener("system:stepped", load);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
