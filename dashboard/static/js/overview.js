(function () {
    "use strict";

    var OVERVIEW_ENDPOINT = "/api/v1/monitor/overview";
    var GAUGE_RADIUS = 34;
    var GAUGE_CIRCUMFERENCE = 2 * Math.PI * GAUGE_RADIUS;
    var CRIT_THRESHOLD = 90;

    function findContainerByLabel(labelText) {
        var labels = document.querySelectorAll(".metric-label");
        for (var i = 0; i < labels.length; i++) {
            if (labels[i].textContent.trim() === labelText) {
                return labels[i].closest(".metric") || labels[i].closest(".gauge-widget");
            }
        }
        return null;
    }

    function setMetricValue(labelText, text) {
        var container = findContainerByLabel(labelText);
        if (!container || text === null || text === undefined) {
            return;
        }
        var el = container.querySelector(".metric-value") || container.querySelector(".gauge-widget-value");
        if (!el) {
            return;
        }
        el.classList.remove("skeleton", "is-muted");
        el.textContent = text;
    }

    function setWidgetSub(labelText, text) {
        var container = findContainerByLabel(labelText);
        if (!container || text === null || text === undefined) {
            return;
        }
        var el = container.querySelector(".gauge-widget-sub");
        if (!el) {
            return;
        }
        el.textContent = text;
    }

    function setGaugeValue(labelText, percent, displayText, options) {
        var container = findContainerByLabel(labelText);
        if (!container) {
            return;
        }

        var valueEl = container.querySelector(".gauge-widget-value");
        var fill = container.querySelector(".gauge-fill");

        if (percent === null || percent === undefined || isNaN(percent)) {
            if (valueEl) {
                valueEl.classList.remove("skeleton");
                valueEl.textContent = "—";
            }
            if (fill) {
                fill.setAttribute("stroke-dasharray", "0 " + GAUGE_CIRCUMFERENCE);
            }
            if (container.classList.contains("gauge-widget")) {
                container.classList.remove("is-critical");
            }
            return;
        }

        options = options || {};
        var invert = options.invert === true;
        var critThreshold = options.critThreshold !== undefined ? options.critThreshold : CRIT_THRESHOLD;

        var clamped = Math.max(0, Math.min(100, percent));

        if (fill) {
            var filledLength = (clamped / 100) * GAUGE_CIRCUMFERENCE;
            var remaining = GAUGE_CIRCUMFERENCE - filledLength;
            fill.setAttribute("stroke-dasharray", filledLength + " " + remaining);
        }

        if (valueEl) {
            valueEl.classList.remove("skeleton");
            valueEl.textContent = displayText !== undefined ? displayText : Math.round(clamped) + "%";
        }

        var isCritical = invert ? clamped < critThreshold : clamped >= critThreshold;
        if (container.classList.contains("gauge-widget")) {
            container.classList.toggle("is-critical", isCritical);
        }
    }

    function formatBytes(bytes) {
        if (bytes === null || bytes === undefined || isNaN(bytes)) {
            return null;
        }
        if (bytes === 0) {
            return "0 B";
        }
        var units = ["B", "KB", "MB", "GB", "TB"];
        var exponent = Math.min(
            Math.floor(Math.log(Math.abs(bytes)) / Math.log(1024)),
            units.length - 1
        );
        var value = bytes / Math.pow(1024, exponent);
        var formatted = exponent === 0 ? value.toString() : value.toFixed(1);
        return formatted + " " + units[exponent];
    }

    // The header chip is shared by every page and owned by ui.js (window.ResourceHeader).
    function header(method, arg) {
        if (window.ResourceHeader && typeof window.ResourceHeader[method] === "function") {
            window.ResourceHeader[method](arg);
        }
    }

    function applyCpuData(cpu) {
        var utilPercent = null;

        if (
            cpu &&
            cpu.mean_metrics &&
            typeof cpu.mean_metrics === "object" &&
            cpu.mean_metrics.cpu_utilization !== undefined &&
            cpu.mean_metrics.cpu_utilization !== null
        ) {
            utilPercent = Number(cpu.mean_metrics.cpu_utilization);
        }

        // Scheduler utilization near 100% means no idle time, which is not a
        // fault, so the CPU ring never switches to the critical (red) state.
        setGaugeValue(
            "CPU Utilization",
            utilPercent,
            utilPercent !== null ? utilPercent.toFixed(1) + "%" : undefined,
            { critThreshold: Infinity }
        );

        if (!cpu) {
            return;
        }

        if (typeof cpu.total_steps === "number" && cpu.total_steps > 0) {
            setMetricValue("Total steps", String(cpu.total_steps));
            if (typeof cpu.total_processes === "number") {
                setMetricValue("Total processes", String(cpu.total_processes));
            }
        }

        if (cpu.scheduler_distribution && typeof cpu.scheduler_distribution === "object") {
            var schedulerNames = Object.keys(cpu.scheduler_distribution);
            if (schedulerNames.length > 0) {
                var topScheduler = null;
                var topCount = -Infinity;
                for (var i = 0; i < schedulerNames.length; i++) {
                    var name = schedulerNames[i];
                    var count = cpu.scheduler_distribution[name];
                    if (typeof count === "number" && count > topCount) {
                        topCount = count;
                        topScheduler = name;
                    }
                }
                if (topScheduler !== null) {
                    setMetricValue("Active scheduler", topScheduler);
                }
            }
        }
    }

    function applyMemoryData(memory) {
        if (!memory) {
            return;
        }

        if (
            memory.ram_used_bytes !== undefined &&
            memory.ram_total_bytes !== undefined &&
            memory.ram_percent !== undefined &&
            memory.ram_percent !== null
        ) {
            var ramUsed = formatBytes(memory.ram_used_bytes);
            var ramTotal = formatBytes(memory.ram_total_bytes);
            var ramPercent = Number(memory.ram_percent);

            setGaugeValue("RAM Usage", ramPercent, ramPercent.toFixed(1) + "%");

            if (ramUsed !== null && ramTotal !== null) {
                setWidgetSub("RAM Usage", ramUsed + " / " + ramTotal);
            }
        }

        if (memory.ram_available_bytes !== undefined && memory.ram_available_bytes !== null) {
            setMetricValue("RAM available", formatBytes(memory.ram_available_bytes));
        }

        if (
            memory.swap_used_bytes !== undefined &&
            memory.swap_total_bytes !== undefined &&
            memory.swap_percent !== undefined &&
            memory.swap_percent !== null
        ) {
            var swapUsed = formatBytes(memory.swap_used_bytes);
            var swapTotal = formatBytes(memory.swap_total_bytes);
            var swapPercent = Number(memory.swap_percent);

            setGaugeValue("Swap Usage", swapPercent, swapPercent.toFixed(1) + "%");

            if (swapUsed !== null && swapTotal !== null) {
                setWidgetSub("Swap Usage", swapUsed + " / " + swapTotal);
            }
        }
    }

    function applyDeviceData(devices) {
        if (!devices) {
            return;
        }

        if (devices.device_count !== undefined && devices.device_count !== null) {
            setMetricValue("Devices monitored", String(devices.device_count));
        }

        if (Array.isArray(devices.devices) && devices.devices.length > 0) {
            var total = devices.devices.length;
            var healthy = 0;
            for (var i = 0; i < total; i++) {
                if (devices.devices[i] && devices.devices[i].status === "ok") {
                    healthy++;
                }
            }
            var unhealthy = total - healthy;
            var healthyPercent = (healthy / total) * 100;

            setGaugeValue("Fleet Health", healthyPercent, healthy + "/" + total, {
                invert: true,
                critThreshold: 75
            });
            setWidgetSub("Fleet Health", unhealthy === 0 ? "devices OK" : unhealthy + " degraded");
            setMetricValue("Active anomalies", String(unhealthy));
        }
    }

    function applyOverviewData(data) {
        if (!data) {
            return;
        }

        header("apply", data);

        applyCpuData(data.cpu);
        applyMemoryData(data.memory);
        applyDeviceData(data.devices);
    }

    function loadOverview() {
        header("loading");

        fetch(OVERVIEW_ENDPOINT, {
            method: "GET",
            headers: {
                Accept: "application/json"
            }
        })
            .then(function (response) {
                if (!response.ok) {
                    throw new Error(
                        "Overview request failed with status " + response.status
                    );
                }
                return response.json();
            })
            .then(function (data) {
                applyOverviewData(data);
            })
            .catch(function (error) {
                header("error");
                if (window.console && console.error) {
                    console.error("Failed to load system overview:", error);
                }
            });
    }

    function setText(id, text) {
        var el = document.getElementById(id);
        if (el) {
            el.textContent = text;
        }
    }

    function formatNumber(value, digits) {
        if (typeof value !== "number" || isNaN(value)) {
            return null;
        }
        return value.toFixed(digits === undefined ? 1 : digits);
    }

    function setStage(stage, text, state) {
        var node = document.querySelector('.pipeline-node[data-stage="' + stage + '"]');
        if (!node) {
            return;
        }
        var stateEl = node.querySelector(".pipeline-node-state");
        if (stateEl && text) {
            stateEl.textContent = text;
        }
        node.classList.remove("is-complete", "is-error");
        if (state) {
            node.classList.add("is-" + state);
        }
    }

    function buildRow(module, title, detail) {
        var row = document.createElement("li");
        row.className = "step-row step-row--" + module;

        var tag = document.createElement("span");
        tag.className = "step-row-module";
        tag.textContent = module.toUpperCase();

        var text = document.createElement("span");
        text.className = "step-row-text";

        var titleEl = document.createElement("span");
        titleEl.className = "step-row-title";
        titleEl.textContent = title;
        text.appendChild(titleEl);

        if (detail) {
            var detailEl = document.createElement("span");
            detailEl.className = "step-row-detail";
            detailEl.textContent = detail;
            text.appendChild(detailEl);
        }

        row.appendChild(tag);
        row.appendChild(text);
        return row;
    }

    function renderLatestStep(record) {
        var body = document.getElementById("latest-step-body");
        if (!body || !record) {
            return;
        }

        var cpu = record.cpu || {};
        var meta = cpu.meta_result || {};
        var schedule = meta.schedule_result || {};
        var metrics = schedule.metrics || {};
        var memory = record.memory || {};
        var devices = Array.isArray(record.devices) ? record.devices : [];
        var list = document.createElement("ul");
        list.className = "step-list";

        if (cpu.workload_class) {
            var conf = formatNumber(typeof cpu.confidence === "number" ? cpu.confidence * 100 : NaN, 0);
            list.appendChild(buildRow(
                "cpu",
                "Workload classified as " + cpu.workload_class,
                conf !== null ? conf + "% confidence" : null
            ));
        }

        if (cpu.selected_scheduler) {
            list.appendChild(buildRow(
                "cpu",
                "Scheduler selected: " + cpu.selected_scheduler + (cpu.is_fallback ? " (fallback)" : ""),
                typeof cpu.process_count === "number" ? cpu.process_count + " processes scheduled" : null
            ));
        }

        var perf = [];
        if (formatNumber(metrics.avg_waiting_time) !== null) {
            perf.push("avg waiting " + formatNumber(metrics.avg_waiting_time));
        }
        if (formatNumber(metrics.avg_turnaround_time) !== null) {
            perf.push("avg turnaround " + formatNumber(metrics.avg_turnaround_time));
        }
        if (formatNumber(metrics.cpu_utilization) !== null) {
            perf.push("utilization " + formatNumber(metrics.cpu_utilization) + "%");
        }
        if (perf.length) {
            list.appendChild(buildRow("cpu", "Schedule executed", perf.join(" · ")));
        }

        if (formatNumber(memory.ram_percent) !== null) {
            list.appendChild(buildRow("memory", "Memory snapshot recorded", "RAM " + formatNumber(memory.ram_percent) + "% used"));
        }

        if (devices.length) {
            devices.forEach(function (item) {
                var id = item && (item.device_id || item.id);
                var action = item && (item.action || item.action_type || item.decision);
                list.appendChild(buildRow("device", (id || "Device") + (action ? ": " + action : ""), item && item.outcome ? String(item.outcome) : null));
            });
        } else {
            list.appendChild(buildRow("device", "No device actions in this step", null));
        }

        body.textContent = "";
        body.appendChild(list);

        var stepNo = typeof record.step_index === "number" ? record.step_index + 1 : null;
        var when = typeof cpu.timestamp === "number" ? new Date(cpu.timestamp * 1000).toLocaleTimeString() : null;
        setText("latest-step-meta", (stepNo !== null ? "Step " + stepNo : "Latest step") + (when ? " · " + when : ""));
    }

    function renderPipeline(record) {
        var cpu = record.cpu || {};
        var memory = record.memory || {};
        var metrics = ((cpu.meta_result || {}).schedule_result || {}).metrics || {};
        var conf = formatNumber(typeof cpu.confidence === "number" ? cpu.confidence * 100 : NaN, 0);
        var ram = formatNumber(memory.ram_percent);

        setStage("observe", (typeof cpu.process_count === "number" ? cpu.process_count + " processes" : "Observed") + (ram !== null ? " · RAM " + ram + "%" : ""), "complete");
        setStage("predict", cpu.workload_class ? cpu.workload_class + (conf !== null ? " · " + conf + "%" : "") : "Predicted", "complete");
        setStage("decide", cpu.selected_scheduler ? cpu.selected_scheduler + (cpu.is_fallback ? " (fallback)" : "") : "Decided", "complete");
        setStage("act", formatNumber(metrics.total_time) !== null ? "Ran " + formatNumber(metrics.total_time) + " time units" : "Executed", "complete");

        var stepNo = typeof record.step_index === "number" ? record.step_index + 1 : null;
        setText("pipeline-status-text", stepNo !== null ? "Step " + stepNo + " complete" : "Step complete");
        var status = document.getElementById("pipeline-status");
        if (status) {
            status.classList.remove("is-critical");
        }
        setText("cpu-panel-status", "Active");
    }

    function onStepFailed(event) {
        setText("pipeline-status-text", "Last step failed");
        var status = document.getElementById("pipeline-status");
        if (status) {
            status.classList.add("is-critical");
        }
        if (window.console && console.error && event && event.detail) {
            console.error("System step failed:", event.detail.message);
        }
    }

    function init() {
        header("claim");
        loadOverview();

        document.addEventListener("system:stepped", function (event) {
            renderLatestStep(event.detail);
            renderPipeline(event.detail);
            loadOverview();
        });
        document.addEventListener("system:step-failed", onStepFailed);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();