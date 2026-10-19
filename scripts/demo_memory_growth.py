
import csv
import multiprocessing as mp
import time
from pathlib import Path

from modules.memory.memory_monitor import MemoryMonitor


OUTPUT = Path("data/memory/live_logs/controlled_growth.csv")
MAX_ALLOCATION_MB = 16
CHUNK_MB = 1
INTERVAL_S = 0.5


def memory_worker(stop_event):
    allocations = []

    for _ in range(MAX_ALLOCATION_MB // CHUNK_MB):
        if stop_event.is_set():
            break

        block = bytearray(CHUNK_MB * 1024 * 1024)

        # Touch every page so the allocation becomes resident.
        for offset in range(0, len(block), 4096):
            block[offset] = 1

        allocations.append(block)
        time.sleep(INTERVAL_S)

    stop_event.wait(timeout=10)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    stop_event = mp.Event()
    worker = mp.Process(target=memory_worker, args=(stop_event,))
    worker.start()

    try:
        monitor = MemoryMonitor()

        result = monitor.collect(
            n=20,
            interval_s=INTERVAL_S,
            pid=worker.pid,
            save_path=str(OUTPUT),
        )

        rss = [s["proc_rss_bytes"] for s in result["snapshots"]]

        print("Samples:", result["count"])
        print("Initial RSS:", rss[0])
        print("Final RSS:", rss[-1])
        print("RSS increase (MB):", round((rss[-1] - rss[0]) / (1024 ** 2), 2))
        print("Saved to:", OUTPUT)

    finally:
        stop_event.set()
        worker.join(timeout=5)

        if worker.is_alive():
            worker.terminate()
            worker.join()


if __name__ == "__main__":
    mp.freeze_support()
    main()
