"""
Classical dynamic memory allocation algorithms for Module 2 — Predictive Memory Management.

Algorithms implemented:
    - First Fit  — allocate using the first block large enough to satisfy the request
    - Best Fit   — allocate using the smallest block large enough (minimises leftover)
    - Worst Fit  — allocate using the largest block large enough (maximises leftover)

Each algorithm subclasses MemoryAllocator and exposes one public method::

    allocate(blocks, requests) -> dict

Block model:
    - Each free block is represented by its currently available size (int > 0).
    - After a successful allocation, that block's available size is reduced by
      the request size; the block remains at the same index.
    - A zero-size block cannot satisfy future requests (since all request sizes
      must be > 0, a zero-size block never matches the >= size condition).

Design principle (DEVELOPMENT.md §3, §21):
    Classical OS algorithms perform the actual allocation operations.
    AI components (future checkpoints) may later wrap these classes to add
    predictive or workload-aware allocation decisions.
"""

import time


class MemoryAllocator:
    """Abstract base class for classical dynamic memory allocation algorithms.

    Subclasses must override :meth:`_find_block` and set ``_algorithm_name``.
    """

    _algorithm_name = "MemoryAllocator"

    def allocate(self, blocks: list, requests: list) -> dict:
        """Simulate memory allocation for all requests against the given blocks.

        Requests are processed in order.  Each successful allocation immediately
        reduces the chosen block's available size before the next request is
        processed, so earlier allocations affect later ones.

        Parameters
        ----------
        blocks : list[int]
            Available memory block sizes (each entry must be > 0).
            An empty list is valid — every request will fail.
        requests : list[int]
            Ordered sequence of allocation request sizes (each entry must be > 0).
            An empty list is valid — returns a result with zero allocations.

        Returns
        -------
        dict
            Keys:
            * ``algorithm``         – algorithm name (str)
            * ``initial_blocks``    – copy of the input blocks list (list[int])
            * ``requests``          – copy of the input requests list (list[int])
            * ``allocations``       – one dict per request, in order (list[dict])
            * ``total_allocated``   – count of successfully allocated requests (int)
            * ``total_failed``      – count of failed requests (int)
            * ``final_blocks``      – block sizes after all allocations (list[int])
            * ``execution_time_ms`` – wall-clock time in ms (float)

            Each entry in ``allocations``:
            * ``request_size`` – the requested size (int)
            * ``allocated``    – True if a block was found (bool)
            * ``block_index``  – index into initial_blocks, or None if failed (int|None)
            * ``block_size``   – available size of the chosen block before allocation,
                                  or None if failed (int|None)
            * ``remaining``    – available size of that block after allocation,
                                  or None if failed (int|None)

        Raises
        ------
        TypeError
            If *blocks* or *requests* is not a list.
        ValueError
            If any block size or request size is <= 0.
        """
        self._validate(blocks, requests)
        start = time.perf_counter()


        current_blocks = list(blocks)
        allocations = []

        for req_size in requests:
            idx = self._find_block(current_blocks, req_size)

            if idx is not None:

                block_size_before = current_blocks[idx]
                current_blocks[idx] -= req_size
                allocations.append({
                    "request_size": req_size,
                    "allocated":    True,
                    "block_index":  idx,
                    "block_size":   block_size_before,
                    "remaining":    current_blocks[idx],
                })
            else:

                allocations.append({
                    "request_size": req_size,
                    "allocated":    False,
                    "block_index":  None,
                    "block_size":   None,
                    "remaining":    None,
                })

        elapsed_ms = (time.perf_counter() - start) * 1000

        return self._build_result(
            blocks, requests, allocations, current_blocks, elapsed_ms
        )

    def _find_block(self, blocks: list, size: int):
        """Return the index of the block to use, or None if no block fits.

        Parameters
        ----------
        blocks : list[int]  — current available sizes (may include zeros)
        size   : int        — allocation request size (> 0)

        Returns
        -------
        int | None
        """
        raise NotImplementedError("Subclasses must implement _find_block()")

    @staticmethod
    def _validate(blocks: list, requests: list) -> None:
        """Raise descriptive exceptions for invalid inputs.

        Raises
        ------
        TypeError  : blocks or requests is not a list
        ValueError : any block size or request size is <= 0
        """
        if not isinstance(blocks, list):
            raise TypeError(
                f"blocks must be a list, got {type(blocks).__name__}"
            )
        if not isinstance(requests, list):
            raise TypeError(
                f"requests must be a list, got {type(requests).__name__}"
            )
        for i, b in enumerate(blocks):
            if b <= 0:
                raise ValueError(
                    f"All block sizes must be > 0; blocks[{i}] = {b}"
                )
        for i, r in enumerate(requests):
            if r <= 0:
                raise ValueError(
                    f"All request sizes must be > 0; requests[{i}] = {r}"
                )

    def _build_result(
        self,
        initial_blocks: list,
        requests: list,
        allocations: list,
        final_blocks: list,
        execution_time_ms: float,
    ) -> dict:
        """Assemble the standardised result dictionary."""
        total_allocated = sum(1 for a in allocations if a["allocated"])
        return {
            "algorithm":         self._algorithm_name,
            "initial_blocks":    list(initial_blocks),
            "requests":          list(requests),
            "allocations":       allocations,
            "total_allocated":   total_allocated,
            "total_failed":      len(requests) - total_allocated,
            "final_blocks":      list(final_blocks),
            "execution_time_ms": execution_time_ms,
        }


class FirstFitAllocator(MemoryAllocator):
    """First Fit memory allocation algorithm.

    Scans blocks from left to right and allocates the **first** block whose
    available size is >= the request size.  Fast but may leave large blocks
    unused at the front after many allocations.
    """

    _algorithm_name = "FirstFit"

    def _find_block(self, blocks: list, size: int):
        """Return the index of the first block with available size >= *size*."""
        for i, b in enumerate(blocks):
            if b >= size:
                return i
        return None


class BestFitAllocator(MemoryAllocator):
    """Best Fit memory allocation algorithm.

    Searches all blocks and allocates the one with the **smallest** available
    size that is still >= the request size.  Minimises leftover space in the
    chosen block.  On a tie (two blocks of equal size), the block with the
    lower index is chosen.
    """

    _algorithm_name = "BestFit"

    def _find_block(self, blocks: list, size: int):
        """Return the index of the smallest eligible block (>= *size*)."""
        best_idx = None
        best_size = None
        for i, b in enumerate(blocks):
            if b >= size:

                if best_size is None or b < best_size:
                    best_size = b
                    best_idx = i
        return best_idx


class WorstFitAllocator(MemoryAllocator):
    """Worst Fit memory allocation algorithm.

    Searches all blocks and allocates the one with the **largest** available
    size that is >= the request size.  Maximises leftover space in the chosen
    block, aiming to leave fragments large enough for future requests.  On a
    tie (two blocks of equal size), the block with the lower index is chosen.
    """

    _algorithm_name = "WorstFit"

    def _find_block(self, blocks: list, size: int):
        """Return the index of the largest eligible block (>= *size*)."""
        worst_idx = None
        worst_size = None
        for i, b in enumerate(blocks):
            if b >= size:

                if worst_size is None or b > worst_size:
                    worst_size = b
                    worst_idx = i
        return worst_idx
