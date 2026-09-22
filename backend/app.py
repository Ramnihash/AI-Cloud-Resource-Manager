"""Flask application factory for CP5.1 Common Backend Interface."""

from __future__ import annotations

from typing import Any, Optional

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

from backend.routes import api_bp
from modules.cpu.action_loop import CPUActionLoop


def create_app(
    cpu_action_loop: Optional[CPUActionLoop] = None,
    test_config: Optional[dict[str, Any]] = None,
) -> Flask:
    """Create and configure the Flask application instance.

    Args:
        cpu_action_loop: Optional injected CPUActionLoop instance.
        test_config: Optional test configuration mapping.

    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__)

    # Default application configurations
    app.config["JSON_SORT_KEYS"] = False

    if test_config:
        app.config.update(test_config)

    if cpu_action_loop is not None:
        app.config["CPU_ACTION_LOOP"] = cpu_action_loop

    # Register API blueprint
    app.register_blueprint(api_bp)

    # Register global JSON error handlers (no Python tracebacks exposed)
    @app.errorhandler(400)
    def bad_request(error):
        msg = getattr(error, "description", "Invalid request")
        if not isinstance(msg, str):
            msg = "Invalid request"
        return jsonify({"error": msg}), 400

    @app.errorhandler(404)
    def not_found(error):
        msg = getattr(error, "description", "Resource not found")
        if (
            not isinstance(msg, str)
            or "The requested URL was not found on the server" in msg
        ):
            msg = "Resource not found"
        return jsonify({"error": msg}), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({"error": "Method not allowed"}), 405

    @app.errorhandler(500)
    def internal_server_error(error):
        return jsonify({"error": "Internal server error"}), 500

    @app.errorhandler(Exception)
    def handle_unexpected_exception(error):
        if isinstance(error, HTTPException):
            return jsonify({"error": error.description or error.name}), error.code
        # Internal unexpected exception: hide traceback and return clean JSON 500
        return jsonify({"error": "Internal server error"}), 500

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=5000, debug=False)
