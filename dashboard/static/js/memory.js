(function () {
    "use strict";

    var OVERVIEW_ENDPOINT = "/api/v1/monitor/overview";
    var MONITOR_ENDPOINT = "/api/v1/memory/monitor";
    var PAGE_ENDPOINT = "/api/v1/memory/page-replacement";
    var POLICIES = ["fifo", "lru", "optimal"];
    var POLICY_LABELS = { fifo: "FIFO", lru: "LRU", optimal: "Optimal" };
    var SAMPLE_COUNT = 10;
    var SAMPLE_INTERVAL_S = 0.4;
    var SVG_NS = "http://www.w3.org/2000/svg";

    var results = {};
    var activePolicy = "fifo";
    var lastSnapshots = [];

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

    function svg(tag, attrs) {
        var node = document.createElementNS(SVG_NS, tag);
        Object.keys(attrs || {}).forEach(function (key) {
            node.setAttribute(key, attrs[key]);
        });
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
            return null;
        }
        if (bytes === 0) {
            return "0 B";
        }
        var units = ["B", "KB", "MB", "GB", "TB"];
        var exp = Math.min(Math.floor(Math.log(Math.abs(bytes)) / Math.log(1024)), units.length - 1);
        var value = bytes / Math.pow(1024, exp);
        return (exp === 0 ? String(value) : value.toFixed(1)) + " " + units[exp];
    }

    function pct(value, digits) {
        return typeof value === "number" && !isNaN(value) ? value.toFixed(digits === undefined ? 1 : digits) + "%" : "—";
    }

    function showError(message) {
        var banner = byId("mem-error");
        banner.hidden = !message;
        banner.textContent = message || "";
    }

    function describeError(error) {
        return error instanceof TypeError ? "Backend unreachable. Is the server running?" : error.message;
    }

    function postJson(url, body) {
        return fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json", Accept: "application/json" },
            body: JSON.stringify(body)
        }).then(function (response) {
            if (!response.ok) {
                return response.json().then(
                    function (b) { throw new Error((b && b.error) || "Request failed (" + response.status + ")"); },
                    function () { throw new Error("Request failed (" + response.status + ")"); }
                );
            }
            return response.json();
        });
    }

    /* ---------- system memory ---------- */

    function renderOverview(data) {
        var m = (data && data.memory) || {};
        setText("kpi-ram-pct", pct(m.ram_percent));
        setText("kpi-ram-sub", [formatBytes(m.ram_used_bytes), formatBytes(m.ram_total_bytes)].filter(Boolean).join(" / "));
        setText("kpi-ram-avail", formatBytes(m.ram_available_bytes) || "—");
        setText("kpi-ram-total", m.ram_total_bytes ? "of " + formatBytes(m.ram_total_bytes) + " total" : " ");
        setText("kpi-swap-pct", pct(m.swap_percent));
        setText("kpi-swap-sub", m.swap_total_bytes === 0 ? "No swap configured" : [formatBytes(m.swap_used_bytes), formatBytes(m.swap_total_bytes)].filter(Boolean).join(" / "));
        if (m.timestamp) {
            var when = new Date(m.timestamp);
            var label = isNaN(when.getTime()) ? String(m.timestamp) : when.toLocaleString();
            setText("mem-meta", "Updated " + label);
            setText("last-updated-value", label);
        }
    }

    function loadOverview() {
        return fetch(OVERVIEW_ENDPOINT, { headers: { Accept: "application/json" } })
            .then(function (response) {
                if (!response.ok) {
                    throw new Error("Overview request failed with status " + response.status);
                }
                return response.json();
            })
            .then(function (data) {
                showError("");
                renderOverview(data);
            })
            .catch(function (error) {
                showError(describeError(error));
            });
    }

    /* ---------- RAM trend ---------- */

    function renderTrend(snapshots) {
        lastSnapshots = snapshots;
        var host = byId("ram-chart");
        clear(host);
        var points = snapshots
            .map(function (s) { return { t: new Date(s.timestamp).getTime(), v: s.ram_percent }; })
            .filter(function (p) { return typeof p.v === "number" && !isNaN(p.t); });
        if (points.length < 2) {
            setText("ram-chart-note", "Not enough samples to draw a trend.");
            return;
        }

        var W = Math.max(host.clientWidth || 0, 320);
        var H = 220;
        var pad = { l: 48, r: 16, t: 14, b: 28 };
        var vals = points.map(function (p) { return p.v; });
        var lo = Math.min.apply(null, vals);
        var hi = Math.max.apply(null, vals);
        // Axis starts at 0 so small changes are not exaggerated.
        var yMin = 0;
        var yMax = Math.min(100, Math.max(10, Math.ceil(hi * 1.5 / 10) * 10));
        var t0 = points[0].t;
        var t1 = points[points.length - 1].t;
        var tSpan = Math.max(t1 - t0, 1);

        function x(t) { return pad.l + (t - t0) / tSpan * (W - pad.l - pad.r); }
        function y(v) { return pad.t + (1 - (v - yMin) / (yMax - yMin)) * (H - pad.t - pad.b); }

        var chart = svg("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, class: "chart-svg" });

        for (var i = 0; i <= 4; i++) {
            var gv = yMin + (yMax - yMin) * i / 4;
            chart.appendChild(svg("line", { x1: pad.l, x2: W - pad.r, y1: y(gv), y2: y(gv), class: "chart-grid" }));
            var label = svg("text", { x: pad.l - 8, y: y(gv) + 4, class: "chart-tick", "text-anchor": "end" });
            label.textContent = gv.toFixed(0) + "%";
            chart.appendChild(label);
        }

        var d = points.map(function (p, idx) { return (idx ? "L" : "M") + x(p.t).toFixed(1) + " " + y(p.v).toFixed(1); }).join(" ");
        chart.appendChild(svg("path", { d: d, class: "chart-line" }));
        points.forEach(function (p) {
            var dot = svg("circle", { cx: x(p.t), cy: y(p.v), r: 4, class: "chart-dot" });
            var title = svg("title", {});
            title.textContent = new Date(p.t).toLocaleTimeString() + ": " + p.v.toFixed(1) + "% RAM";
            dot.appendChild(title);
            chart.appendChild(dot);
        });

        var seconds = svg("text", { x: W - pad.r, y: H - 6, class: "chart-tick", "text-anchor": "end" });
        seconds.textContent = ((t1 - t0) / 1000).toFixed(1) + " s window";
        chart.appendChild(seconds);
        var first = svg("text", { x: pad.l, y: H - 6, class: "chart-tick" });
        first.textContent = new Date(t0).toLocaleTimeString();
        chart.appendChild(first);

        host.appendChild(chart);
        setText("ram-chart-note", points.length + " samples from the backend · min " + lo.toFixed(1) + "% · max " + hi.toFixed(1) + "%");
    }

    function sampleRam() {
        var btn = byId("ram-sample-btn");
        var label = btn.querySelector(".btn-label");
        btn.disabled = true;
        label.textContent = "Sampling…";
        setText("ram-chart-note", "Collecting " + SAMPLE_COUNT + " samples…");
        return postJson(MONITOR_ENDPOINT, { count: SAMPLE_COUNT, interval_s: SAMPLE_INTERVAL_S })
            .then(function (data) {
                showError("");
                renderTrend(Array.isArray(data.snapshots) ? data.snapshots : []);
            })
            .catch(function (error) {
                showError(describeError(error));
                setText("ram-chart-note", "Sampling failed.");
            })
            .then(function () {
                btn.disabled = false;
                label.textContent = "Sample again";
            });
    }

    /* ---------- page replacement ---------- */

    function parseReferences(text) {
        var parts = text.split(/[\s,]+/).filter(function (p) { return p !== ""; });
        if (!parts.length) {
            return { error: "Enter at least one page number." };
        }
        if (parts.length > 200) {
            return { error: "Use at most 200 references." };
        }
        var pages = [];
        for (var i = 0; i < parts.length; i++) {
            if (!/^\d+$/.test(parts[i])) {
                return { error: "'" + parts[i] + "' is not a whole page number." };
            }
            pages.push(parseInt(parts[i], 10));
        }
        return { pages: pages };
    }

    function faultFlags(result) {
        var refs = result.reference_string || [];
        var frames = result.frame_snapshots || [];
        return refs.map(function (page, i) {
            return i === 0 ? true : (frames[i - 1] || []).indexOf(page) < 0;
        });
    }

    function renderComparison() {
        var body = byId("pr-body");
        clear(body);
        var maxFaults = 1;
        POLICIES.forEach(function (key) {
            if (results[key]) {
                maxFaults = Math.max(maxFaults, results[key].page_faults);
            }
        });
        var best = Math.min.apply(null, POLICIES.filter(function (k) { return results[k]; }).map(function (k) { return results[k].page_faults; }));

        POLICIES.forEach(function (key, idx) {
            var r = results[key];
            if (!r) {
                return;
            }
            var tr = document.createElement("tr");
            var name = el("td", "cell-strong");
            var swatch = el("span", "policy-swatch");
            swatch.style.setProperty("--seg-color", "var(--series-" + (idx + 1) + ")");
            name.appendChild(swatch);
            name.appendChild(document.createTextNode(POLICY_LABELS[key]));
            tr.appendChild(name);
            tr.appendChild(el("td", "cell-num", String(r.page_faults)));
            tr.appendChild(el("td", "cell-num", String(r.page_hits)));
            tr.appendChild(el("td", "cell-num", pct(r.hit_ratio * 100)));
            tr.appendChild(el("td", "cell-num", pct(r.miss_ratio * 100)));
            tr.appendChild(el("td", "cell-num", typeof r.execution_time_ms === "number" ? r.execution_time_ms.toFixed(3) + " ms" : "—"));

            var barCell = el("td", "col-bar");
            var bar = el("span", "hbar");
            var fill = el("span", "hbar-fill");
            fill.style.width = (r.page_faults / maxFaults * 100) + "%";
            fill.style.setProperty("--seg-color", "var(--series-" + (idx + 1) + ")");
            bar.appendChild(fill);
            barCell.appendChild(bar);
            if (r.page_faults === best) {
                barCell.appendChild(el("span", "best-tag", "fewest"));
            }
            tr.appendChild(barCell);
            body.appendChild(tr);
        });
    }

    function renderTabs() {
        var tabs = byId("pr-tabs");
        clear(tabs);
        POLICIES.forEach(function (key) {
            if (!results[key]) {
                return;
            }
            var btn = el("button", "seg-btn" + (key === activePolicy ? " is-active" : ""), POLICY_LABELS[key]);
            btn.type = "button";
            btn.setAttribute("role", "tab");
            btn.setAttribute("aria-selected", key === activePolicy ? "true" : "false");
            btn.addEventListener("click", function () {
                activePolicy = key;
                renderTabs();
                renderFrames();
            });
            tabs.appendChild(btn);
        });
    }

    function renderFrames() {
        var grid = byId("frame-grid");
        clear(grid);
        var r = results[activePolicy];
        if (!r) {
            return;
        }
        var refs = r.reference_string || [];
        var frames = r.frame_snapshots || [];
        var flags = faultFlags(r);
        var rows = r.num_frames;

        var head = document.createElement("tr");
        head.appendChild(el("th", "frame-label", "Ref"));
        refs.forEach(function (page, i) {
            head.appendChild(el("th", "frame-ref" + (flags[i] ? " is-fault" : ""), String(page)));
        });
        grid.appendChild(head);

        for (var row = 0; row < rows; row++) {
            var tr = document.createElement("tr");
            tr.appendChild(el("th", "frame-label", "Frame " + (row + 1)));
            refs.forEach(function (page, i) {
                var cell = frames[i] && frames[i][row] !== undefined ? String(frames[i][row]) : "";
                tr.appendChild(el("td", "frame-cell" + (flags[i] && cell !== "" ? " is-fault" : ""), cell));
            });
            grid.appendChild(tr);
        }

        var status = document.createElement("tr");
        status.appendChild(el("th", "frame-label", "Result"));
        flags.forEach(function (fault) {
            status.appendChild(el("td", "frame-status" + (fault ? " is-fault" : ""), fault ? "Fault" : "Hit"));
        });
        grid.appendChild(status);
    }

    function runComparison(event) {
        event.preventDefault();
        var errorEl = byId("pr-form-error");
        var parsed = parseReferences(byId("pr-refs").value);
        var frames = parseInt(byId("pr-frames").value, 10);
        var message = parsed.error || (isNaN(frames) || frames < 1 || frames > 20 ? "Frames must be between 1 and 20." : "");
        errorEl.hidden = !message;
        errorEl.textContent = message;
        if (message) {
            return;
        }

        var btn = byId("pr-run");
        var label = btn.querySelector(".btn-label");
        btn.disabled = true;
        label.textContent = "Running…";

        Promise.all(POLICIES.map(function (key) {
            return postJson(PAGE_ENDPOINT, { algorithm: key, num_frames: frames, reference_string: parsed.pages });
        }))
            .then(function (all) {
                showError("");
                results = {};
                POLICIES.forEach(function (key, i) { results[key] = all[i]; });
                if (!results[activePolicy]) {
                    activePolicy = POLICIES[0];
                }
                byId("pr-empty").hidden = true;
                byId("pr-results").hidden = false;
                setText("pr-meta", parsed.pages.length + " references · " + frames + " frames");
                renderComparison();
                renderTabs();
                renderFrames();
            })
            .catch(function (error) {
                errorEl.hidden = false;
                errorEl.textContent = describeError(error);
            })
            .then(function () {
                btn.disabled = false;
                label.textContent = "Run comparison";
            });
    }

    function init() {
        loadOverview();
        sampleRam();
        byId("ram-sample-btn").addEventListener("click", sampleRam);
        byId("pr-form").addEventListener("submit", runComparison);
        document.addEventListener("system:stepped", loadOverview);
        window.addEventListener("resize", function () {
            if (lastSnapshots.length) {
                renderTrend(lastSnapshots);
            }
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
