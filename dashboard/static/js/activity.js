(function () {
    "use strict";

    var HISTORY_ENDPOINT = "/api/v1/system/history?limit=50";
    var OVERVIEW_ENDPOINT = "/api/v1/monitor/overview";
    var ERRORS_ENDPOINT = "/api/v1/system/errors?limit=50";
    var FILTERS = [
        { key: "all", label: "All" },
        { key: "cpu", label: "CPU" },
        { key: "memory", label: "Memory" },
        { key: "device", label: "Device" },
        { key: "request", label: "Errors" }
    ];
    var MODULE_LABELS = { cpu: "CPU", memory: "Memory", device: "Device", request: "Request" };

    var events = [];
    var activeFilter = "all";

    function byId(id) {
        return document.getElementById(id);
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

    function setText(id, text) {
        var node = byId(id);
        if (node) {
            node.textContent = text;
        }
    }

    function formatBytes(bytes) {
        if (typeof bytes !== "number" || isNaN(bytes)) {
            return "—";
        }
        var units = ["B", "KB", "MB", "GB", "TB"];
        var exp = bytes === 0 ? 0 : Math.min(Math.floor(Math.log(Math.abs(bytes)) / Math.log(1024)), units.length - 1);
        return (bytes / Math.pow(1024, exp)).toFixed(exp === 0 ? 0 : 1) + " " + units[exp];
    }

    function formatTime(ms) {
        if (typeof ms !== "number" || isNaN(ms)) {
            return "—";
        }
        return new Date(ms).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });
    }

    function showError(message) {
        var banner = byId("act-error");
        banner.hidden = !message;
        banner.textContent = message || "";
    }

    function describeError(error) {
        return error instanceof TypeError ? "Backend unreachable. Is the server running?" : error.message;
    }

    function getJson(url) {
        return fetch(url, { headers: { Accept: "application/json" } }).then(function (response) {
            if (!response.ok) {
                throw new Error("Request failed (" + response.status + ") for " + url.split("?")[0]);
            }
            return response.json();
        });
    }

    /* ---------- build events from backend records ---------- */

    function stepEvents(step) {
        var time = new Date(step.timestamp).getTime();
        var out = [];
        var cpu = step.cpu || {};
        out.push({
            module: "cpu",
            time: time,
            action: "Scheduled workload",
            resource: (cpu.process_count !== undefined ? cpu.process_count : "?") + " processes",
            result: (cpu.selected_scheduler || "—") + (cpu.is_fallback ? " (fallback)" : ""),
            details: "Step " + step.step_index + " · " + String(cpu.workload_class || "—").replace(/_/g, " ").toLowerCase() +
                (typeof cpu.confidence === "number" ? " (" + Math.round(cpu.confidence * 100) + "% confidence)" : ""),
            fallback: !!cpu.is_fallback
        });
        var mem = step.memory || {};
        out.push({
            module: "memory",
            time: time,
            action: "Memory snapshot",
            resource: "System RAM",
            result: typeof mem.ram_percent === "number" ? mem.ram_percent.toFixed(1) + "% used" : "—",
            details: "Step " + step.step_index + " · " + formatBytes(mem.ram_available_bytes) + " available"
        });
        return out;
    }

    function deviceEvent(action) {
        var backup = action.backup_result || {};
        var recovery = action.recovery_result || {};
        var result = recovery.rerouted ? "Rerouted" : (recovery.io_blocked ? "I/O blocked" : "No recovery needed");
        return {
            module: "device",
            time: action.recorded_at ? new Date(action.recorded_at).getTime() : NaN,
            action: "Failure-risk action",
            resource: action.device_id,
            result: result,
            details: "Risk " + Math.round(action.failure_probability * 100) + "% · " + (backup.message || "") + " " + (recovery.message || "")
        };
    }

    function errorEvent(entry) {
        var status = entry.status;
        return {
            module: "request",
            time: entry.recorded_at ? new Date(entry.recorded_at).getTime() : NaN,
            action: "Request failed",
            resource: entry.method + " " + entry.path,
            result: "HTTP " + status + (status >= 500 ? " (server error)" : ""),
            details: (entry.error || "No error message") +
                (typeof entry.duration_ms === "number" ? " · " + entry.duration_ms.toFixed(1) + " ms" : ""),
            serverError: status >= 500
        };
    }

    function buildEvents(history, overview, errors) {
        var out = [];
        ((history && history.steps) || []).forEach(function (step) {
            out = out.concat(stepEvents(step));
        });
        var actions = (((overview || {}).devices || {}).recent_actions) || [];
        actions.forEach(function (a) {
            out.push(deviceEvent(a));
        });
        ((errors && errors.errors) || []).forEach(function (entry) {
            out.push(errorEvent(entry));
        });
        out.sort(function (a, b) {
            var ta = isNaN(a.time) ? -Infinity : a.time;
            var tb = isNaN(b.time) ? -Infinity : b.time;
            return tb - ta;
        });
        return out;
    }

    /* ---------- render ---------- */

    function renderKpis(history, overview, errors) {
        var cpu = events.filter(function (e) { return e.module === "cpu"; });
        var fallbacks = cpu.filter(function (e) { return e.fallback; }).length;
        var device = events.filter(function (e) { return e.module === "device"; });
        var rerouted = device.filter(function (e) { return e.result === "Rerouted"; }).length;
        var failed = events.filter(function (e) { return e.module === "request"; });
        var serverErrors = failed.filter(function (e) { return e.serverError; }).length;
        var totalErrors = (errors && typeof errors.total_errors === "number") ? errors.total_errors : failed.length;
        setText("kpi-ev-steps", String((history && history.total_steps) || 0));
        setText("kpi-ev-fallbacks", String(fallbacks));
        setText("kpi-ev-fallbacks-sub", cpu.length ? "of " + cpu.length + " scheduling decisions" : "No decisions yet");
        setText("kpi-ev-device", String(device.length));
        setText("kpi-ev-device-sub", rerouted + " rerouted");
        setText("kpi-ev-errors", String(totalErrors));
        setText("kpi-ev-errors-sub", totalErrors ? serverErrors + " server errors (5xx) in the last " + failed.length : "No failed requests");
    }

    function renderFilter() {
        var box = byId("act-filter");
        clear(box);
        FILTERS.forEach(function (f) {
            var btn = el("button", "seg-btn" + (f.key === activeFilter ? " is-active" : ""), f.label);
            btn.type = "button";
            btn.setAttribute("role", "tab");
            btn.setAttribute("aria-selected", f.key === activeFilter ? "true" : "false");
            btn.addEventListener("click", function () {
                activeFilter = f.key;
                renderFilter();
                renderLog();
            });
            box.appendChild(btn);
        });
    }

    function cell(content, className) {
        var td = el("td", className);
        if (typeof content === "string") {
            td.textContent = content;
        } else {
            td.appendChild(content);
        }
        return td;
    }

    function renderLog() {
        var body = byId("act-body");
        clear(body);
        var shown = events.filter(function (e) { return activeFilter === "all" || e.module === activeFilter; });
        setText("act-meta", shown.length + (shown.length === 1 ? " EVENT" : " EVENTS"));
        if (!shown.length) {
            var tr = el("tr");
            var td = el("td");
            td.colSpan = 6;
            var block = el("div", "state-block state-block--compact");
            block.appendChild(el("p", "state-title", "No events yet"));
            block.appendChild(el("p", "state-desc", "Press Step System, or run a device action, and the events will appear here."));
            td.appendChild(block);
            tr.appendChild(td);
            body.appendChild(tr);
            return;
        }
        shown.forEach(function (e) {
            var tr = el("tr");
            tr.appendChild(cell(formatTime(e.time), "cell-mono"));
            var tag = el("span", "mod-tag module-" + e.module);
            tag.appendChild(el("span", "module-dot"));
            tag.appendChild(document.createTextNode(MODULE_LABELS[e.module]));
            tr.appendChild(cell(tag));
            tr.appendChild(cell(e.action));
            tr.appendChild(cell(e.resource, "cell-mono"));
            tr.appendChild(cell(e.result, "cell-strong" + (e.module === "request" ? " cell-error" : "")));
            tr.appendChild(cell(e.details));
            body.appendChild(tr);
        });
    }

    function load() {
        return Promise.all([getJson(HISTORY_ENDPOINT), getJson(OVERVIEW_ENDPOINT), getJson(ERRORS_ENDPOINT)])
            .then(function (all) {
                showError("");
                events = buildEvents(all[0], all[1], all[2]);
                renderKpis(all[0], all[1], all[2]);
                renderLog();
            })
            .catch(function (error) {
                showError(describeError(error));
                setText("act-meta", "Unavailable");
            });
    }

    function init() {
        renderFilter();
        load();
        document.addEventListener("system:stepped", load);
        document.addEventListener("system:step-failed", load);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
