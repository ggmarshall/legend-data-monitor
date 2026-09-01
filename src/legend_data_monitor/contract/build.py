"""Build contract-v2 files from a run's monitoring output.

Reads the per-run v1 HDF (wide frames, rawid columns), converts to the fully
binned v2 contract: detector-name axes, (time × detector) Mean-storage
histograms at 1min/10min/60min + min/max sidecars, per-detector run-mean
frames, a /detector_map, and the run manifest.

During the migration window the v2 file coexists with the v1 outputs under
the name ``...-geds-schema2.hdf``; consumers discover the exact file names
from the manifest, so the eventual rename (when the v1 writer is retired) is
transparent.
"""

import os

import pandas as pd

from .. import utils
from ..config import settings
from ..processing import binning
from . import schema, writer


def _camel(param: str) -> str:
    return "".join(word.capitalize() for word in param.split("_"))


def _param_attrs(key: str) -> dict:
    """Best-effort unit/label/limits lookup for a v1 key name."""
    body = key.lstrip("/")
    flag, _, rest = body.partition("_")
    for suffix in ("_pulser01anaDiff", "_pulser01anaRatio", "_var", "_mean"):
        rest = rest.removesuffix(suffix) if rest.endswith(suffix) else rest
    camel_to_snake = {_camel(p): p for p in (settings.PLOT_INFO or {})}
    info = (settings.PLOT_INFO or {}).get(camel_to_snake.get(rest, ""), {})
    attrs = {
        "unit": info.get("unit"),
        "label": info.get("label"),
        "event_type": flag,
    }
    try:
        keyword = "variation" if key.endswith("_var") else "absolute"
        attrs["limits"] = info["limits"]["geds"][keyword]
    except (KeyError, TypeError):
        pass
    return {k: v for k, v in attrs.items() if v is not None}


def build_contract_files(
    generated_path: str,
    period: str,
    run: str,
    metadata_path: str | None = None,
    data_type: str = "phy",
    experiment: str = "l200",
) -> str | None:
    """Produce the v2 contract file + manifest for one (period, run).

    Parameters
    ----------
    generated_path : str
        Output root (the directory containing ``generated/``).
    period, run : str
        Run to convert.
    metadata_path : str, optional
        LEGEND metadata root (``<prod>/inputs``) for the rawid->name map;
        columns stay rawid-labelled strings when not given.
    data_type : str
        Data type (``phy``, ...).

    Returns the manifest path, or None when the v1 input file is absent.
    """
    run_dir = os.path.join(generated_path, "generated/plt/hit", data_type, period, run)
    v1_file = os.path.join(run_dir, f"{experiment}-{period}-{run}-{data_type}-geds.hdf")
    if not os.path.exists(v1_file):
        utils.logger.debug("no v1 monitoring file at %s; skipping v2 build", v1_file)
        return None

    # rawid -> detector-name mapping (fall back to string rawids)
    rename = {}
    detectors = {}
    if metadata_path is not None:
        det_info = utils.build_detector_info(metadata_path)
        detectors = det_info["detectors"]
        rename = {info["daq_rawid"]: name for name, info in detectors.items()}

    v2_name = f"{experiment}-{period}-{run}-{data_type}-geds-schema2.hdf"
    v2_file = os.path.join(run_dir, v2_name)
    if os.path.exists(v2_file):
        os.remove(v2_file)

    written_keys = []
    with pd.HDFStore(v1_file, "r") as store:
        for key in sorted(store.keys()):
            if key.endswith(("_info",)):
                continue  # replaced by per-key attrs + manifest vocabulary
            frame = store[key]
            frame.columns = [rename.get(c, str(c)) for c in frame.columns]
            body = key.lstrip("/")
            flag, _, rest = body.partition("_")
            attrs = _param_attrs(key)

            if key.endswith("_mean"):
                written_keys.append(writer.write_frame(v2_file, body, frame))
                continue

            frame = writer.apply_remove_keys(frame, period, run)
            binned = binning.frame_to_binned(frame)
            written_keys.extend(
                writer.write_binned_series(v2_file, flag, rest, binned, attrs)
            )
            # 1-D distribution over all samples (cal-view replacement)
            written_keys.append(
                writer.write_distribution(
                    v2_file,
                    flag,
                    rest,
                    binning.fill_distribution(frame.to_numpy().ravel()),
                    attrs,
                )
            )

    if detectors:
        written_keys.append(writer.write_detector_map(v2_file, detectors))

    from .._version import version

    manifest_path = writer.write_manifest(
        run_dir,
        period,
        run,
        {v2_name: {"keys": sorted(written_keys), "cadences": list(schema.CADENCES)}},
        package_version=version,
        experiment=experiment,
    )
    utils.logger.info("v2 contract file written: %s", v2_file)
    return manifest_path
