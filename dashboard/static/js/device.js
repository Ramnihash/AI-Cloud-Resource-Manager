(function () {
    "use strict";

    var OVERVIEW_ENDPOINT = "/api/v1/monitor/overview";
    var ACTION_ENDPOINT = "/api/v1/device/predictive-action";
    var HEALTHY = ["ok", "healthy", "available", "normal"];

    var fleet = [];

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

    function num(value, digits, unit) {
        return typeof value === "number" && !isNaN(value) ? value.toFixed(digits) + (unit || "") : "—";
    }

    function showError(message) {
        var banner = byId("dev-error");
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

    function badge(kind, text) {
        var b = el("span", "badge badge-" + kind);
        b.appendChild(el("span", "badge-dot"));
        b.appendChild(document.createTextNode(text));
        return b;
    }

    function row(cells) {
        var tr = el("tr");
        cells.forEach(function (c) {
            var td = el("td", c.cls);
            if (c.node) {
                td.appendChild(c.node);
            } else {
                td.textContent = c.text;
            }
            tr.appendChild(td);
        });
        return tr;
    }

    function emptyRow(cols, title) {
        var tr = el("tr");
        var td = el("td");
        td.colSpan = cols;
        var block = el("div", "state-block state-block--compact");
        block.appendChild(el("p", "state-title", title));
        td.appendChild(block);
        tr.appendChild(td);
        return tr;
    }

    function isHealthy(status) {
        return HEALTHY.indexOf(String(status).toLowerCase()) !== -1;
    }

    /* ---------- fleet ---------- */

    function renderFleet(devices) {
        var body = byId("dev-body");
        clear(body);
        if (!devices.length) {
            body.appendChild(emptyRow(8, "No devices reported"));
            return;
        }
        devices.forEach(function (d) {
            var t = d.telemetry || {};
            body.appendChild(row([
                { text: d.device_id, cls: "cell-mono cell-strong" },
                { text: d.device_type },
                { node: badge(isHealthy(d.status) ? "ok" : "crit", d.status) },
                { text: num(t.temperature, 1, " °C"), cls: "cell-num" },
                { text: num(t.error_rate, 4), cls: "cell-num" },
                { text: num(t.response_latency, 1, " ms"), cls: "cell-num" },
                { text: typeof t.queue_length === "number" ? String(t.queue_length) : "—", cls: "cell-num" },
                { text: num(t.io_activity, 1), cls: "cell-num" }
            ]));
        });
    }

    function renderSelects(devices) {
        var dev = byId("act-device");
        var backup = byId("act-backup");
        var keepDev = dev.value;
        var keepBackup = backup.value;
        clear(dev);
        clear(backup);
        var none = el("option", null, "None");
        none.value = "";
        backup.appendChild(none);
        devices.forEach(function (d) {
            var o = el("option", null, d.device_id);
            o.value = d.device_id;
            dev.appendChild(o);
            var b = el("option", null, d.device_id);
            b.value = d.device_id;
            backup.appendChild(b);
        });
        if (keepDev) { dev.value = keepDev; }
        if (keepBackup) { backup.value = keepBackup; }
    }

    /* ---------- actions ---------- */

    function renderActions(actions) {
        var body = byId("dev-actions-body");
        clear(body);
        setText("dev-actions-meta", actions.length + (actions.length === 1 ? " EVENT" : " EVENTS"));
        if (!actions.length) {
            body.appendChild(emptyRow(4, "No actions recorded yet"));
            return;
        }
        actions.slice().reverse().forEach(function (a) {
            var b = a.backup_result || {};
            var r = a.recovery_result || {};
            body.appendChild(row([
                { text: a.device_id, cls: "cell-mono cell-strong" },
                { text: (a.failure_probability * 100).toFixed(0) + "%", cls: "cell-num" },
                { text: b.message || "—" },
                { text: r.message || "—" }
            ]));
        });
    }

    function renderBackup(actions) {
        var body = byId("dev-backup-body");
        clear(body);
        var triggered = actions.filter(function (a) { return a.backup_result && a.backup_result.triggered; });
        if (!triggered.length) {
            var block = el("div", "state-block state-block--compact");
            block.appendChild(el("p", "state-title", "No backups yet"));
            block.appendChild(el("p", "state-desc", "A backup starts when a failure probability reaches the threshold (0.7)."));
            body.appendChild(block);
            return;
        }
        var ok = triggered.filter(function (a) { return a.backup_result.success; });
        var last = triggered[triggered.length - 1].backup_result;
        [
            ["Backups triggered", String(triggered.length)],
            ["Succeeded", ok.length + " of " + triggered.length],
            ["Last location", last.backup_location || "—"]
        ].forEach(function (pair) {
            var r = el("div", "kv-row");
            r.appendChild(el("span", "kv-label", pair[0]));
            r.appendChild(el("span", "kv-value", pair[1]));
            body.appendChild(r);
        });
    }

    function renderKpis(devices, actions) {
        var healthy = devices.filter(function (d) { return isHealthy(d.status); }).length;
        setText("kpi-dev-total", String(devices.length));
        setText("kpi-dev-ok", String(healthy));
        setText("kpi-dev-ok-sub", (devices.length - healthy) + " unhealthy");
        var reroutes = actions.filter(function (a) { return a.recovery_result && a.recovery_result.rerouted; }).length;
        var backups = actions.filter(function (a) { return a.backup_result && a.backup_result.triggered; }).length;
        setText("kpi-dev-actions", String(actions.length));
        setText("kpi-dev-actions-sub", backups + " backups triggered");
        setText("kpi-dev-reroutes", String(reroutes));
        setText("kpi-dev-reroutes-sub", reroutes ? "Traffic moved to a backup device" : "None so far");
    }

    function render(data) {
        var d = (data && data.devices) || {};
        fleet = d.devices || [];
        var actions = d.recent_actions || [];
        renderFleet(fleet);
        renderSelects(fleet);
        renderActions(actions);
        renderBackup(actions);
        renderKpis(fleet, actions);
        setText("dev-meta", "Updated " + new Date().toLocaleTimeString());
    }

    function load() {
        return fetch(OVERVIEW_ENDPOINT, { headers: { Accept: "application/json" } })
            .then(function (response) {
                if (!response.ok) {
                    throw new Error("Overview request failed (" + response.status + ")");
                }
                return response.json();
            })
            .then(function (data) {
                showError("");
                render(data);
            })
            .catch(function (error) {
                showError(describeError(error));
                setText("dev-meta", "Unavailable");
            });
    }

    function runAction(event) {
        event.preventDefault();
        var errorEl = byId("act-error");
        var prob = parseFloat(byId("act-prob").value);
        var message = isNaN(prob) || prob < 0 || prob > 1 ? "Failure probability must be between 0 and 1." : "";
        var device = byId("act-device").value;
        var backup = byId("act-backup").value;
        if (!message && !device) {
            message = "No device available.";
        }
        if (!message && backup && backup === device) {
            message = "The backup device must differ from the failing device.";
        }
        errorEl.hidden = !message;
        errorEl.textContent = message;
        if (message) {
            return;
        }

        var btn = byId("act-run");
        var label = btn.querySelector(".btn-label");
        btn.disabled = true;
        label.textContent = "Running…";
        postJson(ACTION_ENDPOINT, {
            device_id: device,
            failure_probability: prob,
            device_status: byId("act-status").value,
            backup_device_id: backup || null
        })
            .then(load)
            .catch(function (error) {
                errorEl.hidden = false;
                errorEl.textContent = describeError(error);
            })
            .then(function () {
                btn.disabled = false;
                label.textContent = "Run action";
            });
    }

    function init() {
        load();
        byId("dev-refresh").addEventListener("click", load);
        byId("act-form").addEventListener("submit", runAction);
        document.addEventListener("system:stepped", load);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
