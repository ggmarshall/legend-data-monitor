"""End-to-end test of contract.build on a synthetic v1 per-run file."""

import numpy as np
import pandas as pd

from legend_data_monitor.contract import build, reader


def _make_v1_file(tmp_path, period="p19", run="r001"):
    run_dir = tmp_path / "generated/plt/hit/phy" / period / run
    run_dir.mkdir(parents=True)
    path = run_dir / f"l200-{period}-{run}-phy-geds.hdf"

    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-07-01", periods=240, freq="20s", tz="UTC")
    rawids = [1104000, 1104001]
    abs_df = pd.DataFrame(
        rng.normal(6000, 20, (240, 2)), index=idx, columns=rawids
    )
    abs_df.index.name = "datetime"
    abs_df.to_hdf(path, key="IsPulser_Trapemax", mode="a")
    var_df = (abs_df / abs_df.mean() - 1) * 100
    var_df.to_hdf(path, key="IsPulser_Trapemax_var", mode="a")
    abs_df.mean().to_frame().T.to_hdf(path, key="IsPulser_Trapemax_mean", mode="a")
    info = pd.DataFrame({"Value": ["ADC", "trapEmax"]}, index=["unit", "label"])
    info.to_hdf(path, key="IsPulser_Trapemax_info", mode="a")
    return str(tmp_path), run_dir


def test_build_contract_files_from_v1(tmp_path):
    root, run_dir = _make_v1_file(tmp_path)
    manifest_path = build.build_contract_files(root, "p19", "r001")
    assert manifest_path is not None

    manifest = reader.read_manifest(str(run_dir), "p19", "r001")
    assert manifest["schema_version"] == 2
    fname = next(iter(manifest["files"]))
    assert fname.endswith("-schema2.hdf")
    keys = manifest["files"][fname]["keys"]
    assert "hist/IsPulser_Trapemax/1min" in keys
    assert "hist/IsPulser_Trapemax/60min" in keys
    assert "hist/IsPulser_Trapemax_dist" in keys
    assert "IsPulser_Trapemax_mean" in keys
    assert not any(k.endswith("_info") for k in keys)

    v2 = str(run_dir / fname)
    binned = reader.read_binned_series(v2, "IsPulser", "Trapemax", "1min")
    # without metadata the columns are stringified rawids
    assert binned.detectors == ["1104000", "1104001"]
    total = binned.hist.view()["count"].sum()
    assert total == 480  # 240 timestamps x 2 detectors

    # binned means aggregate the raw series faithfully
    frame = binned.to_frame("mean")
    assert abs(np.nanmean(frame.to_numpy()) - 6000) < 10

    # 60min rebin preserves totals
    b60 = reader.read_binned_series(v2, "IsPulser", "Trapemax", "60min")
    assert b60.hist.view()["count"].sum() == total


def test_build_contract_files_missing_input(tmp_path):
    assert build.build_contract_files(str(tmp_path), "p19", "r001") is None
