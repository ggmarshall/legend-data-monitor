"""Rewrite already-produced v1 monitoring files in the current on-disk layout.

The pipeline used to write its pandas HDF outputs uncompressed and in
float64. That costs about 7x the disk it needs: ~2.6x from the dtype and the
missing compression, and another ~2.7x from slack, because overwriting an
*uncompressed* fixed-format key orphans the block it replaces and every chunk
of a run rewrites every key it touches.

The contract (schema2) files had the same two problems in their own form:
float64 histogram storage, and the slack left behind when the float64 uhi
wrote was narrowed in place.

New runs get the right layout from :data:`utils.HDF_COMPRESSION`, the float32
pivots in :mod:`save_data` and the staged writes in :mod:`contract.writer`.
This module brings existing files over without re-running the pipeline: a
16 GB run file repacks to ~2 GB in minutes, against hours to regenerate.
"""

import glob
import os

import h5py
import numpy as np
import pandas as pd

from . import utils
from .contract import writer

# Frames of plotting metadata: a handful of strings, nothing to compress.
_METADATA_SUFFIX = "_info"


def repack_pandas_hdf(path: str) -> tuple:
    """Repack one pandas HDF file in place; return ``(before, after)`` bytes.

    Keys are converted one at a time, so peak memory is one key, not one
    file. The result is written beside the original and moved into place only
    once it is complete *and* smaller -- a file already in the current layout
    is left exactly as it was.
    """
    before = os.path.getsize(path)
    tmp = path + ".repack"
    if os.path.exists(tmp):
        os.remove(tmp)

    try:
        with pd.HDFStore(path, "r") as store:
            keys = [key.lstrip("/") for key in store.keys()]
        for key in keys:
            frame = pd.read_hdf(path, key=key)
            options = {}
            if not key.endswith(_METADATA_SUFFIX):
                wide = [
                    column
                    for column in getattr(frame, "columns", [])
                    if frame[column].dtype == "float64"
                ]
                if wide:
                    frame = frame.astype({column: "float32" for column in wide})
                options = utils.HDF_COMPRESSION
            frame.to_hdf(tmp, key=key, mode="a", **options)
        after = os.path.getsize(tmp)
        if after >= before:
            os.remove(tmp)
            return before, before
        os.replace(tmp, path)
        return before, after
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def repack_run(
    generated_path: str,
    period: str,
    run: str,
    data_type: str = "phy",
    experiment: str = "l200",
) -> dict:
    """Repack every v1 pandas HDF of one run; return ``{path: (before, after)}``."""
    run_dir = os.path.join(
        generated_path, "generated/plt/hit", data_type, period, run
    )
    pattern = os.path.join(run_dir, f"{experiment}-{period}-{run}-{data_type}-*.hdf")
    results = {}
    for path in sorted(glob.glob(pattern)):
        if path.endswith("-schema2.hdf"):
            before, after = repack_contract_hdf(path)
        else:
            before, after = repack_pandas_hdf(path)
        results[path] = (before, after)
        utils.logger.info(
            "repacked %s: %.2f -> %.2f GB (%.1fx)",
            os.path.basename(path),
            before / 2**30,
            after / 2**30,
            before / max(after, 1),
        )
    return results


# Only the contract's own histogram arrays. Everything else in the file is a
# pandas frame, whose datasets carry pytables attributes (CLASS, transposed,
# ...) that recreating them would drop -- and pandas then refuses to read them.
_NARROWABLE = ("/storage/counts", "/storage/values", "/storage/variances", "/min", "/max")


def _narrow_contract_datasets(path: str) -> None:
    """Rewrite float64 histogram storage as float32 (counts as int32), in place."""
    with h5py.File(path, "r+") as f:
        wide = []
        f.visititems(
            lambda name, obj: wide.append(name)
            if isinstance(obj, h5py.Dataset)
            and obj.dtype == np.float64
            and name.startswith("hist/")
            and name.endswith(_NARROWABLE)
            else None
        )
        for name in wide:
            dataset = f[name]
            values = dataset[...]
            options = {
                "compression": dataset.compression,
                "compression_opts": dataset.compression_opts,
            }
            if (
                name.endswith("/storage/counts")
                and np.isfinite(values).all()
                and values.min() >= 0
                and values.max() < 2**31
                and np.array_equal(values, np.rint(values))
            ):
                narrowed = values.astype(np.int32)
            else:
                narrowed = values.astype(np.float32)
            parent, _, leaf = name.rpartition("/")
            group = f[parent] if parent else f
            del group[leaf]
            group.create_dataset(leaf, data=narrowed, **options)


def _compact(src: str, dst: str) -> None:
    """Copy every object into a fresh file, leaving the slack behind.

    Each histogram group's ``axes`` dataset holds object references to its own
    ``ref_axes`` children; the copy preserves paths exactly, so the references
    are re-resolved by name afterwards (see ``contract.writer`` for why not
    ``expand_refs``).
    """
    with h5py.File(src, "r") as source, h5py.File(dst, "w") as target:
        targets = writer.reference_targets(source)
        for name, value in source.attrs.items():
            target.attrs[name] = value
        for name in source:
            source.copy(name, target, name=name, expand_refs=False)
        writer.restore_references(target, targets)


def repack_contract_hdf(path: str) -> tuple:
    """Repack one contract (schema2) file in place; return ``(before, after)``."""
    before = os.path.getsize(path)
    tmp = path + ".repack"
    if os.path.exists(tmp):
        os.remove(tmp)
    try:
        _narrow_contract_datasets(path)
        _compact(path, tmp)
        after = os.path.getsize(tmp)
        if after >= before:
            os.remove(tmp)
            return before, before
        os.replace(tmp, path)
        return before, after
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
