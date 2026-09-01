"""Simulate block allocation and reuse in a paged KV cache.

This example stores token labels instead of key/value tensors. It focuses on the
block table that lets each request use non-contiguous physical cache blocks.
"""

from __future__ import annotations


class PagedCache:
    """Manage fixed-size cache blocks for several generation requests."""

    def __init__(self, total_blocks: int, block_size: int) -> None:
        if total_blocks <= 0 or block_size <= 0:
            raise ValueError("total_blocks and block_size must be positive")
        self.block_size = block_size
        self.blocks: list[list[str | None]] = [
            [None] * block_size for _ in range(total_blocks)
        ]
        self.free_blocks = list(range(total_blocks))
        self.block_tables: dict[str, list[int]] = {}
        self.token_counts: dict[str, int] = {}

    def _allocate_block(self) -> int:
        if not self.free_blocks:
            raise MemoryError("paged cache has no free blocks")
        block_id = self.free_blocks.pop(0)
        return block_id

    def append(self, request_id: str, token: str) -> None:
        """Append one token and allocate a block only when it is needed."""

        table = self.block_tables.setdefault(request_id, [])
        token_count = self.token_counts.setdefault(request_id, 0)

        if token_count % self.block_size == 0:
            table.append(self._allocate_block())

        logical_block = token_count // self.block_size
        offset = token_count % self.block_size
        physical_block = table[logical_block]
        self.blocks[physical_block][offset] = token
        self.token_counts[request_id] = token_count + 1

    def read(self, request_id: str) -> list[str]:
        """Rebuild a request sequence by following its logical block table."""

        if request_id not in self.block_tables:
            raise KeyError(f"unknown request: {request_id}")

        tokens: list[str] = []
        remaining = self.token_counts[request_id]
        for physical_block in self.block_tables[request_id]:
            take = min(remaining, self.block_size)
            tokens.extend(
                token
                for token in self.blocks[physical_block][:take]
                if token is not None
            )
            remaining -= take
        return tokens

    def release(self, request_id: str) -> list[int]:
        """Release a completed request and return its physical block IDs."""

        table = self.block_tables.pop(request_id)
        self.token_counts.pop(request_id)
        for block_id in table:
            self.blocks[block_id] = [None] * self.block_size
            self.free_blocks.append(block_id)
        self.free_blocks.sort()
        return table

    def internal_waste(self) -> int:
        """Count unused token slots inside blocks owned by active requests."""

        allocated_slots = sum(
            len(table) * self.block_size for table in self.block_tables.values()
        )
        used_slots = sum(self.token_counts.values())
        return allocated_slots - used_slots

    def describe(self) -> None:
        """Print request block tables and current cache utilization."""

        print("Request block tables:")
        for request_id in sorted(self.block_tables):
            table = self.block_tables[request_id]
            tokens = " ".join(self.read(request_id))
            print(f"  {request_id}: blocks={table}, tokens=[{tokens}]")
        print("Free blocks:", self.free_blocks)
        print("Unused slots inside allocated blocks:", self.internal_waste())


def append_many(cache: PagedCache, request_id: str, tokens: list[str]) -> None:
    """Append a short token sequence to one request."""

    for token in tokens:
        cache.append(request_id, token)


def main() -> None:
    """Show non-contiguous allocation and reuse after a request finishes."""

    cache = PagedCache(total_blocks=5, block_size=3)
    append_many(cache, "A", ["A1", "A2", "A3", "A4"])
    append_many(cache, "B", ["B1", "B2"])
    append_many(cache, "C", ["C1", "C2", "C3"])

    print("Before request B finishes")
    cache.describe()

    released = cache.release("B")
    print("\nReleased blocks from B:", released)

    append_many(cache, "D", ["D1", "D2", "D3", "D4"])
    print("\nAfter request D arrives")
    cache.describe()

    d_blocks = cache.block_tables["D"]
    assert released[0] in d_blocks
    assert cache.read("D") == ["D1", "D2", "D3", "D4"]
    print("D reused B's block and can span non-contiguous physical blocks:", d_blocks)


if __name__ == "__main__":
    main()
