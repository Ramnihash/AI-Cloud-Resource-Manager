(function () {
    "use strict";

    var STEP_ENDPOINT = "/api/v1/system/step";

    function randomInt(min, max) {
        return Math.floor(Math.random() * (max - min + 1)) + min;
    }

    function randomFloat(min, max, digits) {
        var value = Math.random() * (max - min) + min;
        var factor = Math.pow(10, digits === undefined ? 2 : digits);
        return Math.round(value * factor) / factor;
    }

    // Demo workload profiles. Feature ranges follow the CPU module's synthetic
    // training data (data/cpu/workload_classifier_data.py), so the burst
    // predictor and workload classifier receive the metadata they were trained on.
    var WORKLOAD_PROFILES = [
        {
            name: "cpu_bound",
            priority: [2, 5], burst: [15, 30],
            cpu: [0.75, 1.0], io: [0.0, 0.35],
            instructions: [40000, 100000], complexity: [2.5, 5.0]
        },
        {
            name: "io_bound",
            priority: [2, 5], burst: [2, 10],
            cpu: [0.1, 0.5], io: [0.65, 0.9],
            instructions: [5000, 40000], complexity: [1.0, 3.0]
        },
        {
            name: "priority_critical",
            priority: [1, 10], burst: [3, 15],
            cpu: [0.1, 0.6], io: [0.05, 0.45],
            instructions: [5000, 50000], complexity: [1.0, 3.5]
        },
        {
            name: "balanced",
            priority: [2, 5], burst: [4, 14],
            cpu: [0.3, 0.6], io: [0.2, 0.5],
            instructions: [10000, 40000], complexity: [1.5, 3.0]
        }
    ];
    var nextProfile = randomInt(0, WORKLOAD_PROFILES.length - 1);

    // Each step sends 4-8 processes from the next profile in turn, with
    // metadata features. The backend does all prediction and scheduling.
    function buildDemoWorkload() {
        var profile = WORKLOAD_PROFILES[nextProfile];
        nextProfile = (nextProfile + 1) % WORKLOAD_PROFILES.length;

        var count = randomInt(4, 8);
        var processes = [];
        var arrival = 0;
        for (var i = 1; i <= count; i++) {
            processes.push({
                pid: i,
                arrival_time: arrival,
                burst_time: randomInt(profile.burst[0], profile.burst[1]),
                priority: i === 1 && profile.name === "priority_critical" ? 1 : randomInt(profile.priority[0], profile.priority[1]),
                workload_type: profile.name,
                metadata: {
                    cpu_intensity: randomFloat(profile.cpu[0], profile.cpu[1]),
                    io_wait_ratio: randomFloat(profile.io[0], profile.io[1]),
                    instruction_count: randomInt(profile.instructions[0], profile.instructions[1]),
                    process_complexity: randomFloat(profile.complexity[0], profile.complexity[1], 1)
                }
            });
            arrival += randomInt(0, 3);
        }
        return { processes: processes };
    }

    function setStepFeedback(text, state) {
        var el = document.getElementById("step-feedback");
        if (!el) {
            return;
        }
        el.textContent = text || "";
        el.classList.remove("is-error", "is-ok");
        if (state) {
            el.classList.add("is-" + state);
        }
    }

    function readError(response) {
        return response.json().then(
            function (body) {
                return (body && body.error) || "Request failed (" + response.status + ")";
            },
            function () {
                return "Request failed (" + response.status + ")";
            }
        );
    }

    function initStepSystem() {
        var stepBtn = document.getElementById("step-system-btn");
        if (!stepBtn) {
            return;
        }
        var label = stepBtn.querySelector(".btn-label");

        stepBtn.addEventListener("click", function (event) {
            event.preventDefault();
            if (stepBtn.disabled) {
                return;
            }

            stepBtn.disabled = true;
            stepBtn.setAttribute("aria-busy", "true");
            if (label) {
                label.textContent = "Running…";
            }
            setStepFeedback("", null);

            fetch(STEP_ENDPOINT, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Accept: "application/json"
                },
                body: JSON.stringify(buildDemoWorkload())
            })
                .then(function (response) {
                    if (!response.ok) {
                        return readError(response).then(function (message) {
                            throw new Error(message);
                        });
                    }
                    return response.json();
                })
                .then(function (record) {
                    var stepNo = typeof record.step_index === "number" ? record.step_index + 1 : null;
                    setStepFeedback(stepNo !== null ? "Step " + stepNo + " complete" : "Step complete", "ok");
                    document.dispatchEvent(new CustomEvent("system:stepped", { detail: record }));
                })
                .catch(function (error) {
                    var message = error && error.message ? error.message : "Request failed";
                    if (error instanceof TypeError) {
                        message = "Backend unreachable. Is the server running?";
                    }
                    setStepFeedback("Step failed: " + message, "error");
                    document.dispatchEvent(new CustomEvent("system:step-failed", { detail: { message: message } }));
                })
                .then(function () {
                    stepBtn.disabled = false;
                    stepBtn.removeAttribute("aria-busy");
                    if (label) {
                        label.textContent = "Step System";
                    }
                });
        });
    }

    function initNavKeyboardAccessibility() {
        var navLinks = document.querySelectorAll(".nav-link");
        if (!navLinks.length) {
            return;
        }

        navLinks.forEach(function (link) {
            link.addEventListener("focus", function () {
                link.classList.add("is-focused");
            });

            link.addEventListener("blur", function () {
                link.classList.remove("is-focused");
            });
        });
    }

    function initSafeFormDefaults() {
        var forms = document.querySelectorAll("form[data-prevent-default]");
        if (!forms.length) {
            return;
        }

        forms.forEach(function (form) {
            form.addEventListener("submit", function (event) {
                event.preventDefault();
            });
        });
    }

    function init() {
        initStepSystem();
        initNavKeyboardAccessibility();
        initSafeFormDefaults();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();