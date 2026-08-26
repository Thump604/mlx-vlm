"""Experimental row-addressable storage for Qwen4-Exp PLE weights.

This is deliberately not wired into model loading. It establishes the bounded,
measurable backend needed to test storage hierarchy feasibility first.
"""

import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class LookupStats:
    lookups: int
    rows: int
    cache_hits: int
    cache_misses: int
    bytes_read: int
    elapsed_seconds: float


class MMapNGramEmbedding:
    MANIFEST_VERSION = 1

    def __init__(self, manifest_path, *, cache_rows=0):
        if cache_rows < 0:
            raise ValueError("cache_rows must be non-negative")
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        required = {"version", "data_file", "dtype", "row_count", "row_width"}
        missing = required - manifest.keys()
        if missing:
            raise ValueError(f"manifest missing fields: {sorted(missing)}")
        if manifest["version"] != self.MANIFEST_VERSION:
            raise ValueError(f"unsupported manifest version: {manifest['version']}")
        if manifest["dtype"] not in {"float16", "float32"}:
            raise ValueError(f"unsupported row dtype: {manifest['dtype']}")
        data_name = Path(manifest["data_file"])
        if data_name.is_absolute() or data_name.name != str(data_name):
            raise ValueError("data_file must be a filename beside the manifest")
        self.dtype = np.dtype(manifest["dtype"])
        self.row_count = int(manifest["row_count"])
        self.row_width = int(manifest["row_width"])
        if self.row_count <= 0 or self.row_width <= 0:
            raise ValueError("row_count and row_width must be positive")
        self.data_path = manifest_path.parent / data_name
        expected = self.row_count * self.row_width * self.dtype.itemsize
        try:
            actual = self.data_path.stat().st_size
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"n-gram row store not found: {self.data_path}"
            ) from exc
        if actual != expected:
            raise ValueError(
                f"row store size mismatch: expected {expected}, got {actual}"
            )
        self._table = np.memmap(
            self.data_path,
            dtype=self.dtype,
            mode="r",
            shape=(self.row_count, self.row_width),
        )
        self.cache_rows = cache_rows
        self._cache = OrderedDict()
        self._lookups = self._rows = self._hits = self._misses = self._bytes = 0
        self._elapsed = 0.0

    def lookup(self, row_ids):
        started = time.perf_counter()
        ids = np.asarray(row_ids, dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= self.row_count):
            raise IndexError("n-gram row outside mmap table")
        rows = []
        for row_id_value in ids.reshape(-1):
            row_id = int(row_id_value)
            cached = self._cache.get(row_id)
            if cached is not None:
                self._hits += 1
                self._cache.move_to_end(row_id)
                rows.append(cached)
                continue
            self._misses += 1
            row = np.array(self._table[row_id], copy=True)
            self._bytes += row.nbytes
            rows.append(row)
            if self.cache_rows:
                self._cache[row_id] = row
                while len(self._cache) > self.cache_rows:
                    self._cache.popitem(last=False)
        values = np.stack(rows).reshape((*ids.shape, self.row_width))
        self._lookups += 1
        self._rows += ids.size
        self._elapsed += time.perf_counter() - started
        return mx.array(values)

    @property
    def stats(self):
        return LookupStats(
            self._lookups,
            self._rows,
            self._hits,
            self._misses,
            self._bytes,
            self._elapsed,
        )
