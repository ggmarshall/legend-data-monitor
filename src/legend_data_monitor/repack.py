"""Rewrite already-produced v1 monitoring files in the current on-disk layout.

The pipeline used to write its pandas HDF outputs uncompressed and in
float64. That costs about 7x the disk it needs: ~2.6x from the dtype and the
missing compression, and another ~2.7x from slack, because overwriting an
*uncompressed* fixed-format key orphans the block it replaces and every chunk
of a run rewrites every key it touches.

New runs get the right layout from :data:`utils.HDF_COMPRESSION` and the
float32 pivots in :mod:`save_data`. This module brings existing files over
without re-running the pipeline: a 16 GB run file repacks to ~2 GB in
minutes, against hours to regenerate. The contract (schema2) files are
untouched -- they are independent of the v1 file once built.
"""

import glob
import os

import pandas as pd

from . import utils

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
            continue  # contract files carry their own layout
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
