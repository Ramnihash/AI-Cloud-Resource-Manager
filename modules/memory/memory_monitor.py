"""
Live system memory monitor for Module 2 — Predictive Memory Management.

This module provides :class:`MemoryMonitor`, which collects real-time memory
metrics from the host system using psutil.  It is a classical OS-layer
monitoring component — it observes and records; it does not forecast or act.

Public API::

    mon = MemoryMonitor()

    # Single snapshot
    snap = mon.take_snapshot()          # system-wide only
    snap = mon.take_snapshot(pid=1234)  # include per-process metrics

    # Repeated sampling (n samples, interval_s seconds apart)
    result = mon.collect(n=10, interval_s=1.0)
    result = mon.collect(n=10, interval_s=1.0, pid=os.getpid(),
                         save_path="data/memory/live_log.csv")

Metrics collected:

    RAM (psutil.virtual_memory):
        ram_total_bytes, ram_available_bytes, ram_used_bytes,
        ram_free_bytes, ram_percent

    Swap (psutil.swap_memory):
        swap_total_bytes, swap_used_bytes, swap_free_bytes, swap_percent

    Per-process (psutil.Process, optional when pid is supplied):
        proc_pid, proc_rss_bytes, proc_vms_bytes, proc_memory_percent

All snapshots include a UTC ISO-8601 timestamp.  The proc_* fields are
present in every snapshot dict but set to None when no pid is supplied,
keeping the schema uniform across all rows.

CSV output (optional, collect() only):
    When save_path is supplied, all snapshots are written as a CSV file
    using the built-in csv module.  Columns follow CSV_COLUMNS order.
    None values are written as empty strings.

Design principle (DEVELOPMENT.md §3.3, DECISIONS.md D004):
    System memory monitoring is a component of Module 2 (Predictive Memory
    Management).  AI forecasting (CP3.5) and leak detection (CP3.6) will
    consume the time-series data produced by this monitor.
"""

import csv
import datetime
import time

import psutil




CSV_COLUMNS = (
    "timestamp",
    "ram_total_bytes",
    "ram_available_bytes",
    "ram_used_bytes",
    "ram_free_bytes",
    "ram_percent",
    "swap_total_bytes",
    "swap_used_bytes",
    "swap_free_bytes",
    "swap_percent",
    "proc_pid",
    "proc_rss_bytes",
    "proc_vms_bytes",
    "proc_memory_percent",
)




class MemoryMonitor:
    """Live system memory monitor using psutil.

    Collects RAM, swap, and (optionally) per-process memory metrics.
    All state is local to each method call; the class itself holds no
    persistent state between calls.

    Methods
    -------
    take_snapshot(pid=None) -> dict
        Capture a single memory snapshot and return it immediately.
    collect(n, interval_s, pid=None, save_path=None) -> dict
        Capture *n* snapshots spaced *interval_s* seconds apart, then
        optionally save them to a CSV file.
    """

    def take_snapshot(self, pid=None) -> dict:
        """Capture a single memory snapshot.

        Parameters
        ----------
        pid : int or None, optional
            If provided, include per-process memory metrics for this PID.
            Raises ``psutil.NoSuchProcess`` if the PID does not exist.

        Returns
        -------
        dict
            Snapshot with the following keys:

            ``timestamp``           – UTC ISO-8601 string
            ``ram_total_bytes``     – total physical RAM (int)
            ``ram_available_bytes`` – available RAM (int)
            ``ram_used_bytes``      – RAM in use (int)
            ``ram_free_bytes``      – free RAM (int)
            ``ram_percent``         – % RAM used (float)
            ``swap_total_bytes``    – total swap (int)
            ``swap_used_bytes``     – swap in use (int)
            ``swap_free_bytes``     – free swap (int)
            ``swap_percent``        – % swap used (float)
            ``proc_pid``            – monitored PID (int | None)
            ``proc_rss_bytes``      – resident set size (int | None)
            ``proc_vms_bytes``      – virtual memory size (int | None)
            ``proc_memory_percent`` – process RAM share % (float | None)

        Raises
        ------
        psutil.NoSuchProcess
            If *pid* is provided but the process does not exist.
        """
        return self._snapshot(pid)

    def collect(
        self,
        n: int,
        interval_s: float,
        pid=None,
        save_path=None,
    ) -> dict:
        """Collect *n* memory snapshots spaced *interval_s* seconds apart.

        Snapshots are taken sequentially.  The monitor sleeps *interval_s*
        seconds between snapshots (but not after the last one).

        Parameters
        ----------
        n : int
            Number of snapshots to collect.  Must be >= 1.
        interval_s : float
            Seconds to sleep between consecutive snapshots.  Must be >= 0.
            Use 0 for back-to-back sampling (useful in tests).
        pid : int or None, optional
            If provided, include per-process metrics in each snapshot.
            See :meth:`take_snapshot` for details.
        save_path : str or None, optional
            If provided, write all snapshots to this path as a CSV file.
            The file is created or overwritten.  Column order follows
            :data:`CSV_COLUMNS`.  None values are written as empty strings.

        Returns
        -------
        dict
            Keys:

            ``snapshots``  – list of *n* snapshot dicts (list[dict])
            ``count``      – number of snapshots collected (int)
            ``interval_s`` – interval used (float)
            ``saved_to``   – path written to, or None (str | None)

        Raises
        ------
        ValueError
            If *n* < 1 or *interval_s* < 0.
        psutil.NoSuchProcess
            If *pid* is provided but the process does not exist.
        """
        self._validate_collect(n, interval_s)

        snapshots = []

        for i in range(n):

            snapshots.append(self._snapshot(pid))

            if i < n - 1 and interval_s > 0:
                time.sleep(interval_s)


        saved_to = None
        if save_path is not None:
            self._save_csv(snapshots, save_path)
            saved_to = save_path

        return {
            "snapshots":  snapshots,
            "count":      len(snapshots),
            "interval_s": float(interval_s),
            "saved_to":   saved_to,
        }



    @staticmethod
    def _snapshot(pid=None) -> dict:
        """Collect one system memory snapshot and return it as a dict."""
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()

        snapshot = {

            "timestamp":           datetime.datetime.utcnow().isoformat(),

            "ram_total_bytes":     vm.total,
            "ram_available_bytes": vm.available,
            "ram_used_bytes":      vm.used,
            "ram_free_bytes":      vm.free,
            "ram_percent":         float(vm.percent),

            "swap_total_bytes":    sw.total,
            "swap_used_bytes":     sw.used,
            "swap_free_bytes":     sw.free,
            "swap_percent":        float(sw.percent),

            "proc_pid":            None,
            "proc_rss_bytes":      None,
            "proc_vms_bytes":      None,
            "proc_memory_percent": None,
        }

        if pid is not None:

            proc = psutil.Process(pid)
            mi = proc.memory_info()
            snapshot["proc_pid"]            = pid
            snapshot["proc_rss_bytes"]      = mi.rss
            snapshot["proc_vms_bytes"]      = mi.vms
            snapshot["proc_memory_percent"] = float(proc.memory_percent())

        return snapshot

    @staticmethod
    def _save_csv(snapshots: list, path: str) -> None:
        """Write *snapshots* to *path* as a CSV, column order = CSV_COLUMNS."""
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for snap in snapshots:

                row = {k: ("" if snap[k] is None else snap[k]) for k in CSV_COLUMNS}
                writer.writerow(row)

    @staticmethod
    def _validate_collect(n: int, interval_s: float) -> None:
        """Raise ValueError for invalid collect() arguments."""
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if interval_s < 0:
            raise ValueError(f"interval_s must be >= 0, got {interval_s}")
