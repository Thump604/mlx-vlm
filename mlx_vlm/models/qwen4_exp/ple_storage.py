"""Row-addressable storage for Qwen4-Exp PLE weights.

The production checkpoint stores the PLE table as 128 quantized embedding
shards. Loading those shards as ordinary MLX parameters wires roughly 30 GiB
of cold lookup data. This module instead maps the existing safetensors byte
ranges and materializes only the rows selected for the current token.
"""

import json
import os
import shutil
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

PLE_MARKER = ".ple.ple_embedding.ngram_embedding.shards."
_DTYPES = {"U32": np.dtype("<u4"), "BF16": np.dtype("<u2")}


@dataclass(frozen=True)
class LookupStats:
    lookups: int
    rows: int
    cache_hits: int
    cache_misses: int
    bytes_read: int
    elapsed_seconds: float


def _bfloat16_to_float32(values):
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


class QuantizedMMapNGramEmbedding:
    """Read-only affine-Q4 PLE table backed by safetensors byte ranges."""

    MANIFEST_VERSION = 2

    def __init__(self, manifest_path, *, cache_rows=None):
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        required = {
            "version",
            "source_root",
            "row_width",
            "quantization",
        }
        missing = required - manifest.keys()
        if missing:
            raise ValueError(f"manifest missing fields: {sorted(missing)}")
        if manifest["version"] != self.MANIFEST_VERSION:
            raise ValueError(f"unsupported manifest version: {manifest['version']}")
        quantization = manifest["quantization"]
        if quantization != {"bits": 4, "group_size": 32, "mode": "affine"}:
            raise ValueError(f"unsupported PLE quantization: {quantization}")

        source_root = Path(manifest["source_root"])
        if not source_root.is_absolute():
            source_root = (manifest_path.parent / source_root).resolve()
        self.row_width = int(manifest["row_width"])
        if self.row_width <= 0 or self.row_width % 32:
            raise ValueError("row_width must be positive and divisible by group_size")

        self._interleaved = None
        layout = manifest.get("layout", "safetensors_ranges")
        if layout == "interleaved_q4":
            data_name = Path(manifest["data_file"])
            if data_name.is_absolute() or data_name.name != str(data_name):
                raise ValueError("data_file must be beside the manifest")
            data_path = manifest_path.parent / data_name
            self.row_count = int(manifest["row_count"])
            row_stride = int(manifest["row_stride"])
            expected_stride = self.row_width // 2 + 2 * (self.row_width // 32) * 2
            if row_stride != expected_stride:
                raise ValueError("interleaved Q4 row stride does not match row width")
            expected_size = self.row_count * row_stride
            if data_path.stat().st_size != expected_size:
                raise ValueError(
                    "interleaved PLE size mismatch: "
                    f"expected {expected_size}, got {data_path.stat().st_size}"
                )
            self._interleaved = np.memmap(
                data_path,
                dtype=np.uint8,
                mode="r",
                shape=(self.row_count, row_stride),
            )
            self._shards = []
        elif layout == "safetensors_ranges":
            if "shards" not in manifest:
                raise ValueError("manifest missing fields: ['shards']")
            self._init_safetensors_ranges(manifest, source_root)
        else:
            raise ValueError(f"unsupported PLE storage layout: {layout}")

        configured_cache = (
            manifest.get("cache_rows", 0) if cache_rows is None else cache_rows
        )
        self.cache_rows = int(configured_cache)
        if self.cache_rows < 0:
            raise ValueError("cache_rows must be non-negative")
        self._cache = OrderedDict()
        self._lookups = 0
        self._rows = 0
        self._hits = 0
        self._misses = 0
        self._bytes = 0
        self._elapsed = 0.0

    def _init_safetensors_ranges(self, manifest, source_root):
        self._shards = []
        expected_start = 0
        for entry in manifest["shards"]:
            start = int(entry["row_start"])
            count = int(entry["row_count"])
            if start != expected_start or count <= 0:
                raise ValueError("PLE shards must be positive, contiguous, and ordered")
            arrays = {}
            for name, expected_dtype in (
                ("weight", "U32"),
                ("scales", "BF16"),
                ("biases", "BF16"),
            ):
                tensor = entry[name]
                if tensor["dtype"] != expected_dtype:
                    raise ValueError(f"unexpected {name} dtype: {tensor['dtype']}")
                file_name = Path(tensor["file"])
                if file_name.is_absolute() or file_name.name != str(file_name):
                    raise ValueError("tensor files must be filenames under source_root")
                path = source_root / file_name
                offset = int(tensor["offset"])
                shape = tuple(int(dim) for dim in tensor["shape"])
                if shape[0] != count:
                    raise ValueError(f"{name} row count does not match shard")
                dtype = _DTYPES[expected_dtype]
                byte_count = int(np.prod(shape)) * dtype.itemsize
                if offset < 0 or path.stat().st_size < offset + byte_count:
                    raise ValueError(f"{name} byte range exceeds {path}")
                arrays[name] = np.memmap(
                    path,
                    dtype=dtype,
                    mode="r",
                    offset=offset,
                    shape=shape,
                )
            if arrays["weight"].shape[1] * 8 != self.row_width:
                raise ValueError("packed weight width does not match row_width")
            groups = self.row_width // 32
            expected_aux_shape = (groups,)
            if (
                arrays["scales"].shape[1:] != expected_aux_shape
                or arrays["biases"].shape[1:] != expected_aux_shape
            ):
                raise ValueError("scale/bias shape does not match Q4 group layout")
            self._shards.append((start, start + count, arrays))
            expected_start += count
        self.row_count = expected_start

    def _read_row(self, row_id):
        cached = self._cache.get(row_id)
        if cached is not None:
            self._hits += 1
            self._cache.move_to_end(row_id)
            return cached
        if self._interleaved is not None:
            raw = np.array(self._interleaved[row_id], copy=True)
            packed_bytes = self.row_width // 2
            aux_bytes = self.row_width // 32 * 2
            row = (
                raw[:packed_bytes].view("<u4"),
                raw[packed_bytes : packed_bytes + aux_bytes].view("<u2"),
                raw[packed_bytes + aux_bytes :].view("<u2"),
            )
            self._misses += 1
            self._bytes += raw.nbytes
            if self.cache_rows:
                self._cache[row_id] = row
                while len(self._cache) > self.cache_rows:
                    self._cache.popitem(last=False)
            return row
        for start, end, arrays in self._shards:
            if start <= row_id < end:
                local = row_id - start
                row = tuple(
                    np.array(arrays[name][local], copy=True)
                    for name in ("weight", "scales", "biases")
                )
                self._misses += 1
                self._bytes += sum(value.nbytes for value in row)
                if self.cache_rows:
                    self._cache[row_id] = row
                    while len(self._cache) > self.cache_rows:
                        self._cache.popitem(last=False)
                return row
        raise IndexError("n-gram row outside mmap table")

    def _read_rows(self, row_ids):
        """Read each distinct row once, coalescing storage access by layout."""
        row_ids = np.asarray(row_ids, dtype=np.int64)
        if row_ids.ndim != 1:
            raise ValueError("row_ids must be one-dimensional")
        if not row_ids.size:
            groups = self.row_width // 32
            return (
                np.empty((0, self.row_width // 8), dtype="<u4"),
                np.empty((0, groups), dtype="<u2"),
                np.empty((0, groups), dtype="<u2"),
            )

        rows = [None] * len(row_ids)
        missing_positions = []
        missing_ids = []
        for position, row_id in enumerate(row_ids.tolist()):
            cached = self._cache.get(row_id)
            if cached is None:
                missing_positions.append(position)
                missing_ids.append(row_id)
                continue
            self._hits += 1
            self._cache.move_to_end(row_id)
            rows[position] = cached

        if missing_ids and self._interleaved is not None:
            raw = np.array(self._interleaved[np.asarray(missing_ids)], copy=True)
            packed_bytes = self.row_width // 2
            aux_bytes = self.row_width // 32 * 2
            packed = raw[:, :packed_bytes].reshape(len(missing_ids), -1).view("<u4")
            scales = raw[
                :, packed_bytes : packed_bytes + aux_bytes
            ].reshape(len(missing_ids), -1).view("<u2")
            biases = raw[:, packed_bytes + aux_bytes :].reshape(
                len(missing_ids), -1
            ).view("<u2")
            loaded = list(zip(packed, scales, biases))
            self._bytes += raw.nbytes
        elif missing_ids:
            loaded = [None] * len(missing_ids)
            missing_array = np.asarray(missing_ids)
            for start, end, arrays in self._shards:
                selected = np.flatnonzero(
                    (missing_array >= start) & (missing_array < end)
                )
                if not selected.size:
                    continue
                local = missing_array[selected] - start
                weight = np.array(arrays["weight"][local], copy=True)
                scales = np.array(arrays["scales"][local], copy=True)
                biases = np.array(arrays["biases"][local], copy=True)
                self._bytes += weight.nbytes + scales.nbytes + biases.nbytes
                for output_index, row in zip(
                    selected.tolist(), zip(weight, scales, biases)
                ):
                    loaded[output_index] = row
            if any(row is None for row in loaded):
                raise IndexError("n-gram row outside mmap table")
        else:
            loaded = []

        self._misses += len(missing_ids)
        for position, row_id, row in zip(missing_positions, missing_ids, loaded):
            rows[position] = row
            if self.cache_rows:
                self._cache[row_id] = row
                self._cache.move_to_end(row_id)
                while len(self._cache) > self.cache_rows:
                    self._cache.popitem(last=False)

        return tuple(np.stack([row[index] for row in rows]) for index in range(3))

    def __call__(self, row_ids):
        started = time.perf_counter()
        ids = np.asarray(row_ids).astype(np.int64, copy=False)
        if ids.size and (ids.min() < 0 or ids.max() >= self.row_count):
            raise IndexError("n-gram row outside mmap table")
        flat_ids = ids.reshape(-1)
        if not flat_ids.size:
            return mx.zeros((*ids.shape, self.row_width), dtype=mx.bfloat16)

        unique_ids, inverse = np.unique(flat_ids, return_inverse=True)
        packed_np, scales_np, biases_np = self._read_rows(unique_ids)
        duplicate_count = flat_ids.size - unique_ids.size
        self._hits += duplicate_count
        packed = mx.array(packed_np, dtype=mx.uint32)
        scales = mx.array(_bfloat16_to_float32(scales_np)).astype(mx.bfloat16)
        biases = mx.array(_bfloat16_to_float32(biases_np)).astype(mx.bfloat16)
        values = mx.dequantize(packed, scales, biases, group_size=32, bits=4)
        if duplicate_count:
            values = values[mx.array(inverse, dtype=mx.int64)]
        self._lookups += 1
        self._rows += ids.size
        self._elapsed += time.perf_counter() - started
        return values.reshape(*ids.shape, self.row_width)

    lookup = __call__

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


def build_quantized_ple_manifest(model_path, output_path, *, cache_rows=0):
    """Index Q4 PLE tensors in place without copying their payload bytes."""

    model_path = Path(model_path).resolve()
    output_path = Path(output_path)
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    prefixes = {key.rsplit(".", 1)[0] for key in weight_map if PLE_MARKER in key}
    if not prefixes:
        raise ValueError("checkpoint contains no Qwen4-Exp PLE shards")

    headers = {}

    def tensor_descriptor(key):
        file_name = weight_map[key]
        if file_name not in headers:
            with (model_path / file_name).open("rb") as stream:
                header_size = struct.unpack("<Q", stream.read(8))[0]
                headers[file_name] = (
                    8 + header_size,
                    json.loads(stream.read(header_size)),
                )
        payload_start, header = headers[file_name]
        info = header[key]
        return {
            "file": file_name,
            "offset": payload_start + int(info["data_offsets"][0]),
            "dtype": info["dtype"],
            "shape": info["shape"],
        }

    shards = []
    row_start = 0
    ordered_prefixes = sorted(prefixes, key=lambda value: int(value.rsplit(".", 1)[1]))
    for prefix in ordered_prefixes:
        weight = tensor_descriptor(f"{prefix}.weight")
        scales = tensor_descriptor(f"{prefix}.scales")
        biases = tensor_descriptor(f"{prefix}.biases")
        row_count = int(weight["shape"][0])
        shards.append(
            {
                "row_start": row_start,
                "row_count": row_count,
                "weight": weight,
                "scales": scales,
                "biases": biases,
            }
        )
        row_start += row_count
    manifest = {
        "version": QuantizedMMapNGramEmbedding.MANIFEST_VERSION,
        "source_root": str(model_path),
        "layout": "safetensors_ranges",
        "row_width": int(shards[0]["weight"]["shape"][1]) * 8,
        "row_count": row_start,
        "quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
        "cache_rows": int(cache_rows),
        "shards": shards,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def materialize_interleaved_ple_store(
    model_path,
    manifest_path,
    *,
    data_file="ple-q4.rows",
    cache_rows=0,
    chunk_rows=131_072,
):
    """Rewrite scattered Q4 tensors as one mmap-friendly record per row."""

    manifest_path = Path(manifest_path).resolve()
    data_name = Path(data_file)
    if data_name.is_absolute() or data_name.name != str(data_name):
        raise ValueError("data_file must be a filename beside the manifest")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    range_manifest = build_quantized_ple_manifest(
        model_path, manifest_path, cache_rows=cache_rows
    )
    row_width = int(range_manifest["row_width"])
    packed_bytes = row_width // 2
    aux_bytes = row_width // 32 * 2
    row_stride = packed_bytes + 2 * aux_bytes
    data_path = manifest_path.parent / data_name
    if data_path.exists():
        raise FileExistsError(f"PLE row store already exists: {data_path}")
    total_size = int(range_manifest["row_count"]) * row_stride
    with data_path.open("wb") as stream:
        stream.truncate(total_size)
    source_root = Path(range_manifest["source_root"])
    for shard in range_manifest["shards"]:
        arrays = {}
        for name in ("weight", "scales", "biases"):
            tensor = shard[name]
            arrays[name] = np.memmap(
                source_root / tensor["file"],
                dtype=_DTYPES[tensor["dtype"]],
                mode="r",
                offset=int(tensor["offset"]),
                shape=tuple(tensor["shape"]),
            )
        row_start = int(shard["row_start"])
        row_count = int(shard["row_count"])
        for local_start in range(0, row_count, chunk_rows):
            local_end = min(local_start + chunk_rows, row_count)
            output = np.memmap(
                data_path,
                dtype=np.uint8,
                mode="r+",
                offset=(row_start + local_start) * row_stride,
                shape=(local_end - local_start, row_stride),
            )
            output[:, :packed_bytes] = (
                np.asarray(arrays["weight"][local_start:local_end])
                .view(np.uint8)
                .reshape(local_end - local_start, packed_bytes)
            )
            output[:, packed_bytes : packed_bytes + aux_bytes] = (
                np.asarray(arrays["scales"][local_start:local_end])
                .view(np.uint8)
                .reshape(local_end - local_start, aux_bytes)
            )
            output[:, packed_bytes + aux_bytes :] = (
                np.asarray(arrays["biases"][local_start:local_end])
                .view(np.uint8)
                .reshape(local_end - local_start, aux_bytes)
            )
            output.flush()
            del output
        del arrays
    manifest = {
        "version": QuantizedMMapNGramEmbedding.MANIFEST_VERSION,
        "source_root": str(Path(model_path).resolve()),
        "layout": "interleaved_q4",
        "data_file": data_name.name,
        "row_width": row_width,
        "row_count": int(range_manifest["row_count"]),
        "row_stride": row_stride,
        "quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
        "cache_rows": int(cache_rows),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare_external_ple_model(source_path, target_path, *, cache_rows=0):
    """Create a hard-linked model view that excludes resident PLE parameters."""

    source_path = Path(source_path).resolve()
    target_path = Path(target_path).resolve()
    if target_path.exists() and any(target_path.iterdir()):
        raise FileExistsError(f"target model directory is not empty: {target_path}")
    target_path.mkdir(parents=True, exist_ok=True)
    if source_path.stat().st_dev != target_path.stat().st_dev:
        raise ValueError("source and target must share a filesystem for hard links")

    manifest_path = target_path / "ple-store.json"
    ple_manifest = build_quantized_ple_manifest(
        source_path, manifest_path, cache_rows=cache_rows
    )
    source_index = json.loads(
        (source_path / "model.safetensors.index.json").read_text()
    )
    weight_map = {
        key: file_name
        for key, file_name in source_index["weight_map"].items()
        if PLE_MARKER not in key
    }
    used_files = sorted(set(weight_map.values()))
    for file_name in used_files:
        os.link(source_path / file_name, target_path / file_name)
    target_index = dict(source_index)
    target_index["weight_map"] = weight_map
    target_index.setdefault("metadata", {})["external_ple_bytes"] = str(
        sum(
            (
                item[name]["shape"][0]
                * int(np.prod(item[name]["shape"][1:]))
                * _DTYPES[item[name]["dtype"]].itemsize
            )
            for item in ple_manifest["shards"]
            for name in ("weight", "scales", "biases")
        )
    )
    (target_path / "model.safetensors.index.json").write_text(
        json.dumps(target_index, indent=2) + "\n"
    )

    for source_file in source_path.iterdir():
        if not source_file.is_file():
            continue
        if source_file.name == "config.json" or source_file.suffix == ".safetensors":
            continue
        if source_file.name in {
            "model.safetensors.index.json",
            "MANIFEST.sha256",
            "MEMORY_CONTEXT_SWEEP.jsonl",
        }:
            continue
        shutil.copy2(source_file, target_path / source_file.name)

    config = json.loads((source_path / "config.json").read_text())
    config["text_config"]["ple_storage"] = {
        "manifest": str(manifest_path),
        "cache_rows": int(cache_rows),
    }
    quantization = config.get("quantization", {})
    config["quantization"] = {
        key: value for key, value in quantization.items() if PLE_MARKER not in key
    }
    (target_path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    provenance = {
        "format": "qwen4_exp_external_ple_v1",
        "source_model": str(source_path),
        "source_index": "model.safetensors.index.json",
        "external_ple_manifest": manifest_path.name,
        "linked_weight_files": used_files,
    }
    (target_path / "EXTERNAL_PLE.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return provenance
