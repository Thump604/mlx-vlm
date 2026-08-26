import json

import numpy as np
import pytest

from mlx_vlm.models.qwen4_exp.ple_storage import MMapNGramEmbedding


def _store(tmp_path, table, **overrides):
    data = tmp_path / "rows.bin"
    table.tofile(data)
    manifest = {
        "version": 1,
        "data_file": data.name,
        "dtype": str(table.dtype),
        "row_count": table.shape[0],
        "row_width": table.shape[1],
        **overrides,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_mmap_lookup_and_bounded_cache_telemetry(tmp_path):
    table = np.arange(64 * 8, dtype=np.float32).reshape(64, 8)
    store = MMapNGramEmbedding(_store(tmp_path, table), cache_rows=4)
    ids = np.array([[1, 7, 1], [63, 7, 2]])
    np.testing.assert_array_equal(np.asarray(store.lookup(ids)), table[ids])
    assert store.stats.rows == 6
    assert store.stats.cache_hits == 2
    assert store.stats.cache_misses == 4
    assert store.stats.bytes_read == 4 * 8 * 4


def test_mmap_store_fails_closed_for_missing_truncated_or_escaping_data(tmp_path):
    table = np.ones((4, 8), dtype=np.float16)
    path = _store(tmp_path, table)
    (tmp_path / "rows.bin").write_bytes(b"short")
    with pytest.raises(ValueError, match="size mismatch"):
        MMapNGramEmbedding(path)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "data_file": "../rows.bin",
                "dtype": "float16",
                "row_count": 4,
                "row_width": 8,
            }
        )
    )
    with pytest.raises(ValueError, match="must be a filename"):
        MMapNGramEmbedding(path)
