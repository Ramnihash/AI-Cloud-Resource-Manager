"""gunicorn settings for the AI Cloud Resource Manager.

The backend keeps its state (step history, device actions, the trained
models) in memory, so it must run as ONE worker process. Threads handle
concurrent requests inside that process.
"""

bind = "127.0.0.1:8000"
workers = 1
threads = 4
# The first request trains the CPU models, which can take a while.
timeout = 180
accesslog = "-"
errorlog = "-"
