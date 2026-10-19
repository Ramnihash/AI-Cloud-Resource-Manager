(function () {
    "use strict";

    var STEP_ENDPOINT = "/api/v1/system/step";

    function randomInt(min, max) {
        return Math.floor(Math.random() * (max - min + 1)) + min;
    }

    // Demo workload sent with each step: 4-8 processes with random
    // arrival, burst and priority. The backend does all scheduling.
    function buildDemoWorkload() {
        var count = randomInt(4, 8);
        var processes = [];
        var arrival = 0;
        for (var i = 1; i <= count; i++) {
            processes.push({
                pid: i,
                arrival_time: arrival,
                burst_time: randomInt(1, 12),
                priority: randomInt(1, 5)
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