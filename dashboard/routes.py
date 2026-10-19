from flask import Blueprint, render_template

dashboard_bp = Blueprint(
    "dashboard",
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/dashboard/static",
)


@dashboard_bp.route("/dashboard/")
def overview():
    return render_template("overview.html")


@dashboard_bp.route("/dashboard/cpu")
def cpu():
    return render_template("cpu.html")


@dashboard_bp.route("/dashboard/memory")
def memory():
    return render_template("memory.html")


@dashboard_bp.route("/dashboard/device")
def device():
    return render_template("device.html")


@dashboard_bp.route("/dashboard/activity")
def activity():
    return render_template("activity.html")