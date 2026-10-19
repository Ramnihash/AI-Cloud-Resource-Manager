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

    function formatTimestamp(timestamp) {
        if (!timestamp) {
            return null;
        }
        var date = new Date(timestamp);
        if (isNaN(date.getTime())) {
            return timestamp;
        }
        return date.toLocaleString();
    }

    function setHeroStatus(status) {
        var header = document.getElementById("hero-status");
        var el = document.getElementById("system-health-value");
        if (!el) {
            return;
        }

        var label = "Unknown";
        var state = "ok";

        if (status === "ok") {
            label = "Operational";
            state = "ok";
        } else if (status === "warn" || status === "warning" || status === "degraded") {
            label = "Degraded";
            state = "warn";
        } else if (status === "critical" || status === "error" || status === "down") {
            label = "Critical";
            state = "critical";
        } else if (status) {
            label = String(status);
            state = "ok";
        }

        el.textContent = label;

        if (header) {
            header.classList.remove("hero-status--ok", "hero-status--warn", "hero-status--critical");
            header.classList.add("hero-status--" + state);
        }
    }

    function setHeroStatusLoading() {
        var el = document.getElementById("system-health-value");
        if (!el) {
            return;
        }
        el.textContent = "Loading…";
    }

    function setHeroStatusError() {
        var header = document.getElementById("hero-status");
        var el = document.getElementById("system-health-value");
        if (!el) {
            return;
        }
        el.textContent = "Unreachable";

        if (header) {
            header.classList.remove("hero-status--ok", "hero-status--warn");
            header.classList.add("hero-status--critical");
        }
    }

    function setLastUpdated(timestamp) {
        var el = document.getElementById("last-updated-value");
        if (!el) {
            return;
        }
        var formatted = formatTimestamp(timestamp);
        el.textContent = formatted || "—";
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

        setGaugeValue(
            "CPU Utilization",
            utilPercent,
            utilPercent !== null ? utilPercent.toFixed(1) + "%" : undefined
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

        setHeroStatus(data.status);
        setLastUpdated(data.timestamp);

        applyCpuData(data.cpu);
        applyMemoryData(data.memory);
        applyDeviceData(data.devices);
    }

    function loadOverview() {
        setHeroStatusLoading();

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
                setHeroStatusError();
                if (window.console && console.error) {
                    console.error("Failed to load system overview:", error);
                }
            });
    }

    function init() {
        loadOverview();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();