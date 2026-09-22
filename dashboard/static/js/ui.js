(function () {
    "use strict";

    function initStepSystemPlaceholder() {
        var stepBtn = document.getElementById("step-system-btn");
        if (!stepBtn) {
            return;
        }

        stepBtn.addEventListener("click", function (event) {
            event.preventDefault();
            // Visual placeholder only for CP6.1.
            // No API calls, no system-step execution here.
            // Live behavior is added in CP6.2.
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
        initStepSystemPlaceholder();
        initNavKeyboardAccessibility();
        initSafeFormDefaults();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();