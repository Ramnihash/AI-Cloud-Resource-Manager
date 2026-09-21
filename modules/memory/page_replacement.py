"""
Classical page-replacement algorithms for Module 2 — Predictive Memory Management.

Algorithms implemented:
    - FIFO    (First-In First-Out)
    - LRU     (Least Recently Used)
    - Optimal (OPT / Bélády's algorithm)

Each algorithm subclasses PageReplacementAlgorithm and exposes one public
method::

    run(reference_string: list[int]) -> dict

The returned dict is described fully in PageReplacementAlgorithm.run().

Design principle (DEVELOPMENT.md §3, §21):
    Classical OS algorithms perform the actual page-replacement operations.
    AI components (CP3.3, CP3.4) will later wrap these classes to add
    predictive prefetching / page-protection — they are not part of this
    checkpoint.
"""

import time
from collections import deque, OrderedDict


class PageReplacementAlgorithm:
    """Abstract base class for classical page-replacement algorithms.

    Subclasses must override :meth:`run`.
    """

    def __init__(self, num_frames: int):
        """
        Parameters
        ----------
        num_frames : int
            Number of physical memory frames available (must be >= 1).

        Raises
        ------
        ValueError
            If *num_frames* is less than 1.
        """
        if num_frames < 1:
            raise ValueError(
                f"num_frames must be >= 1, got {num_frames}"
            )
        self.num_frames = num_frames

    def run(self, reference_string: list) -> dict:
        """Simulate the algorithm on the given page-reference string.

        Parameters
        ----------
        reference_string : list[int]
            Ordered sequence of page numbers to reference.

        Returns
        -------
        dict
            Keys:

            * ``algorithm``          – algorithm name (str)
            * ``num_frames``         – number of frames (int)
            * ``reference_string``   – copy of the input (list[int])
            * ``page_faults``        – total page faults (int)
            * ``page_hits``          – total page hits (int)
            * ``hit_ratio``          – hits / total_references (float)
            * ``miss_ratio``         – faults / total_references (float)
            * ``execution_time_ms``  – wall-clock time in ms (float)
            * ``frame_snapshots``    – list of frame states, one per
              reference access (list[list[int]])
        """
        raise NotImplementedError("Subclasses must implement run()")

    def _build_result(
        self,
        algorithm_name: str,
        reference_string: list,
        page_faults: int,
        page_hits: int,
        frame_snapshots: list,
        execution_time_ms: float,
    ) -> dict:
        """Assemble the standardised result dictionary.

        Parameters
        ----------
        algorithm_name   : str
        reference_string : list[int]
        page_faults      : int
        page_hits        : int
        frame_snapshots  : list[list[int]]
        execution_time_ms : float

        Returns
        -------
        dict  (see :meth:`run` for key descriptions)
        """
        total = len(reference_string)
        hit_ratio = page_hits / total if total > 0 else 0.0
        miss_ratio = page_faults / total if total > 0 else 0.0
        return {
            "algorithm": algorithm_name,
            "num_frames": self.num_frames,
            "reference_string": list(reference_string),
            "page_faults": page_faults,
            "page_hits": page_hits,
            "hit_ratio": hit_ratio,
            "miss_ratio": miss_ratio,
            "execution_time_ms": execution_time_ms,
            "frame_snapshots": frame_snapshots,
        }


class FIFOPageReplacement(PageReplacementAlgorithm):
    """First-In First-Out (FIFO) page-replacement algorithm.

    The page that has been in memory the longest is evicted first.

    Implementation details:
        * ``arrival_order`` (deque, no maxlen) tracks the order in which
          pages entered memory.
        * ``frames_set`` (set) provides O(1) membership testing.
        * On a page fault when frames are full, the victim is evicted
          explicitly with ``arrival_order.popleft()``; the new page is
          added with ``arrival_order.append(page)``.
        * ``frame_snapshots`` reflect the live content of ``arrival_order``
          after each access (FIFO arrival order, oldest → newest).
    """

    def run(self, reference_string: list) -> dict:
        """Simulate FIFO page replacement.

        See :meth:`PageReplacementAlgorithm.run` for parameters and return
        value specification.
        """
        start = time.perf_counter()

        frames_set = set()       # O(1) membership check for hit/fault detection
        arrival_order = deque()  # Tracks arrival order; NO maxlen — explicit eviction
        frame_snapshots = []
        page_faults = 0
        page_hits = 0

        for page in reference_string:
            if page in frames_set:
                # ── Hit: page already in memory ──────────────────────────
                page_hits += 1
            else:
                # ── Fault: page not in memory ─────────────────────────────
                page_faults += 1
                if len(frames_set) == self.num_frames:
                    # Frames are full — evict the oldest arrival (FIFO victim)
                    victim = arrival_order.popleft()
                    frames_set.remove(victim)
                # Load the new page into memory
                arrival_order.append(page)
                frames_set.add(page)

            # Snapshot: current frame contents in FIFO arrival order (oldest first)
            frame_snapshots.append(list(arrival_order))

        elapsed_ms = (time.perf_counter() - start) * 1000

        return self._build_result(
            "FIFO",
            reference_string,
            page_faults,
            page_hits,
            frame_snapshots,
            elapsed_ms,
        )


class LRUPageReplacement(PageReplacementAlgorithm):
    """Least Recently Used (LRU) page-replacement algorithm.

    The page that was accessed least recently is evicted first.

    Implementation details:
        * ``lru_cache`` (OrderedDict) maintains pages keyed by page number;
          order reflects recency — the front (``last=False``) is the LRU
          page, the back is the MRU page.
        * On a hit, ``move_to_end(page)`` refreshes the page's recency.
        * On a fault when frames are full, ``popitem(last=False)`` evicts
          the LRU page.
        * ``frame_snapshots`` list pages in LRU → MRU order.
    """

    def run(self, reference_string: list) -> dict:
        """Simulate LRU page replacement.

        See :meth:`PageReplacementAlgorithm.run` for parameters and return
        value specification.
        """
        start = time.perf_counter()

        # Keys = page numbers; order = LRU (front) → MRU (back)
        lru_cache = OrderedDict()
        frame_snapshots = []
        page_faults = 0
        page_hits = 0

        for page in reference_string:
            if page in lru_cache:
                # ── Hit: move page to the MRU end ────────────────────────
                lru_cache.move_to_end(page)
                page_hits += 1
            else:
                # ── Fault ────────────────────────────────────────────────
                page_faults += 1
                if len(lru_cache) == self.num_frames:
                    # Frames full — evict the LRU page (front of OrderedDict)
                    lru_cache.popitem(last=False)
                # Load the new page (most recently used → back of dict)
                lru_cache[page] = True

            # Snapshot: pages in LRU → MRU order
            frame_snapshots.append(list(lru_cache.keys()))

        elapsed_ms = (time.perf_counter() - start) * 1000

        return self._build_result(
            "LRU",
            reference_string,
            page_faults,
            page_hits,
            frame_snapshots,
            elapsed_ms,
        )


class OptimalPageReplacement(PageReplacementAlgorithm):
    """Optimal (OPT / Bélády's) page-replacement algorithm.

    On a fault when frames are full, the page whose next use is furthest
    in the future is evicted.  A page that will never be used again is the
    ideal eviction candidate (treated as next_use = ∞).

    This is an **offline** algorithm — it requires the complete reference
    string in advance.  It serves as a theoretical lower-bound benchmark
    for comparing FIFO and LRU performance.
    """

    def run(self, reference_string: list) -> dict:
        """Simulate Optimal page replacement.

        See :meth:`PageReplacementAlgorithm.run` for parameters and return
        value specification.
        """
        start = time.perf_counter()

        frames = []            # List of pages currently in memory (ordered by load time)
        frame_snapshots = []
        page_faults = 0
        page_hits = 0

        for i, page in enumerate(reference_string):
            if page in frames:
                # ── Hit ──────────────────────────────────────────────────
                page_hits += 1
            else:
                # ── Fault ────────────────────────────────────────────────
                page_faults += 1
                if len(frames) < self.num_frames:
                    # Frames not yet full — load without eviction
                    frames.append(page)
                else:
                    # Frames full — find and evict the optimal victim
                    victim = self._find_optimal_victim(
                        frames, reference_string, future_start=i + 1
                    )
                    frames.remove(victim)
                    frames.append(page)

            # Snapshot: current frame contents in load order
            frame_snapshots.append(list(frames))

        elapsed_ms = (time.perf_counter() - start) * 1000

        return self._build_result(
            "Optimal",
            reference_string,
            page_faults,
            page_hits,
            frame_snapshots,
            elapsed_ms,
        )

    @staticmethod
    def _find_optimal_victim(
        frames: list, reference_string: list, future_start: int
    ) -> int:
        """Return the page in *frames* whose next use is furthest in the future.

        Parameters
        ----------
        frames          : list[int]   — pages currently in memory
        reference_string: list[int]   — the full reference string
        future_start    : int         — index of the first future reference

        Returns
        -------
        int : the page to evict (the optimal victim)
        """
        future = reference_string[future_start:]
        victim = None
        max_distance = -1

        for frame_page in frames:
            try:
                distance = future.index(frame_page)
            except ValueError:
                # Page never referenced again — optimal eviction candidate
                return frame_page
            if distance > max_distance:
                max_distance = distance
                victim = frame_page

        return victim
