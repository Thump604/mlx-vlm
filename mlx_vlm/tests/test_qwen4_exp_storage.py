import json

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.models.qwen4_exp.ple_storage import (
    QuantizedMMapNGramEmbedding,
    build_quantized_ple_manifest,
    materialize_interleaved_ple_store,
    prepare_external_ple_model,
)


def _write_quantized_store(tmp_path, table, *, cache_rows=0):
    weight, scales, biases = mx.quantize(table, group_size=32, bits=4)
    scales = scales.astype(mx.bfloat16)
    biases = biases.astype(mx.bfloat16)
    mx.eval(weight, scales, biases)
    arrays = {
        "weight": np.asarray(weight),
        "scales": np.asarray(scales.view(mx.uint16)),
        "biases": np.asarray(biases.view(mx.uint16)),
    }
    offset = 0
    tensors = {}
    data_path = tmp_path / "rows.bin"
    with data_path.open("wb") as stream:
        for name, dtype in (
            ("weight", "U32"),
            ("scales", "BF16"),
            ("biases", "BF16"),
        ):
            values = arrays[name]
            stream.write(values.tobytes())
            tensors[name] = {
                "file": data_path.name,
                "offset": offset,
                "dtype": dtype,
                "shape": list(values.shape),
            }
            offset += values.nbytes
    manifest = {
        "version": 2,
        "source_root": str(tmp_path),
        "row_width": table.shape[1],
        "row_count": table.shape[0],
        "quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
        "cache_rows": cache_rows,
        "shards": [
            {
                "row_start": 0,
                "row_count": table.shape[0],
                **tensors,
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    expected = mx.dequantize(weight, scales, biases, group_size=32, bits=4)
    return path, expected


def test_quantized_mmap_lookup_matches_resident_dequantization(tmp_path):
    table = mx.arange(64 * 160).reshape(64, 160).astype(mx.float32) / 100
    path, expected = _write_quantized_store(tmp_path, table, cache_rows=4)
    store = QuantizedMMapNGramEmbedding(path)
    ids = np.array([[1, 7, 1], [63, 7, 2]])
    actual = store(ids)
    mx.eval(actual, expected)
    np.testing.assert_array_equal(
        np.asarray(actual.astype(mx.float32)),
        np.asarray(expected.astype(mx.float32))[ids],
    )
    assert store.stats.rows == 6
    assert store.stats.cache_hits == 2
    assert store.stats.cache_misses == 4
    assert store.stats.bytes_read == 4 * (20 * 4 + 5 * 2 + 5 * 2)


def test_quantized_mmap_lookup_preserves_unsorted_unique_ids(tmp_path):
    table = mx.arange(8 * 160).reshape(8, 160).astype(mx.float32) / 100
    path, expected = _write_quantized_store(tmp_path, table)
    store = QuantizedMMapNGramEmbedding(path)
    ids = np.array([7, 1, 5, 0])
    actual = store(ids)
    mx.eval(actual, expected)
    np.testing.assert_array_equal(
        np.asarray(actual.astype(mx.float32)),
        np.asarray(expected.astype(mx.float32))[ids],
    )


def test_quantized_store_fails_closed_for_truncation_and_escape(tmp_path):
    table = mx.ones((4, 160))
    path, _ = _write_quantized_store(tmp_path, table)
    (tmp_path / "rows.bin").write_bytes(b"short")
    with pytest.raises(ValueError, match="byte range exceeds"):
        QuantizedMMapNGramEmbedding(path)

    manifest = json.loads(path.read_text())
    manifest["shards"][0]["weight"]["file"] = "../rows.bin"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="must be filenames"):
        QuantizedMMapNGramEmbedding(path)


def test_prepare_external_model_indexes_existing_q4_ranges(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    table = mx.arange(8 * 160).reshape(8, 160).astype(mx.float32) / 100
    weight, scales, biases = mx.quantize(table, group_size=32, bits=4)
    scales = scales.astype(mx.bfloat16)
    biases = biases.astype(mx.bfloat16)
    mx.eval(weight, scales, biases)
    prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0"
    tensors = {
        f"{prefix}.weight": weight,
        f"{prefix}.scales": scales,
        f"{prefix}.biases": biases,
        "language_model.embed_tokens.weight": mx.ones((2, 2)),
    }
    file_name = "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(source / file_name), tensors)
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {key: file_name for key in tensors}})
    )
    (source / "config.json").write_text(
        json.dumps(
            {
                "text_config": {},
                "quantization": {
                    "bits": 4,
                    "group_size": 64,
                    "mode": "affine",
                    prefix: {"bits": 4, "group_size": 32, "mode": "affine"},
                },
            }
        )
    )
    (source / "tokenizer.json").write_text("{}")

    prepare_external_ple_model(source, target)
    range_manifest = json.loads((target / "ple-store.json").read_text())
    assert range_manifest["source_root"] == "../source"
    materialize_interleaved_ple_store(source, target / "ple-store.json")
    interleaved_manifest = json.loads((target / "ple-store.json").read_text())
    assert interleaved_manifest["source_root"] == "../source"

    assert (target / file_name).stat().st_ino == (source / file_name).stat().st_ino
    target_index = json.loads((target / "model.safetensors.index.json").read_text())
    assert list(target_index["weight_map"]) == ["language_model.embed_tokens.weight"]
    assert (target / "ple-q4.rows").stat().st_size == 8 * 100
    store = QuantizedMMapNGramEmbedding(target / "ple-store.json")
    actual = store(np.array([0, 7]))
    expected = mx.dequantize(weight, scales, biases, group_size=32, bits=4)[[0, 7]]
    mx.eval(actual, expected)
    np.testing.assert_array_equal(
        np.asarray(actual.astype(mx.float32)),
        np.asarray(expected.astype(mx.float32)),
    )


def test_build_manifest_rejects_non_q4_before_writing(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0"
    tensors = {
        f"{prefix}.weight": mx.ones((8, 20), dtype=mx.uint32),
        f"{prefix}.scales": mx.ones((8, 5), dtype=mx.bfloat16),
        f"{prefix}.biases": mx.ones((8, 5), dtype=mx.bfloat16),
    }
    file_name = "model.safetensors"
    mx.save_safetensors(str(source / file_name), tensors)
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: file_name for key in tensors}})
    )
    (source / "config.json").write_text(
        json.dumps(
            {
                "quantization": {
                    prefix: {"bits": 8, "group_size": 32, "mode": "affine"}
                }
            }
        )
    )
    manifest_path = tmp_path / "output" / "ple-store.json"

    with pytest.raises(ValueError, match="is not affine Q4/group-32"):
        build_quantized_ple_manifest(source, manifest_path)

    assert not manifest_path.exists()


def test_materialize_checks_existing_data_before_rewriting_manifest(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    table = mx.ones((8, 160))
    weight, scales, biases = mx.quantize(table, group_size=32, bits=4)
    prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0"
    tensors = {
        f"{prefix}.weight": weight,
        f"{prefix}.scales": scales.astype(mx.bfloat16),
        f"{prefix}.biases": biases.astype(mx.bfloat16),
        "language_model.embed_tokens.weight": mx.ones((2, 2)),
    }
    file_name = "model.safetensors"
    mx.save_safetensors(str(source / file_name), tensors)
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {key: file_name for key in tensors}})
    )
    (source / "config.json").write_text(
        json.dumps(
            {
                "text_config": {},
                "quantization": {
                    "bits": 4,
                    "group_size": 64,
                    "mode": "affine",
                    prefix: {"bits": 4, "group_size": 32, "mode": "affine"},
                },
            }
        )
    )
    prepare_external_ple_model(source, target)
    manifest_path = target / "ple-store.json"
    original_manifest = manifest_path.read_text()
    (target / "ple-q4.rows").write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="row store already exists"):
        materialize_interleaved_ple_store(source, manifest_path)

    assert manifest_path.read_text() == original_manifest
