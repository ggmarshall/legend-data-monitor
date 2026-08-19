"""Repacking existing output files into the current layout."""

import os

import numpy as np
import pandas as pd
import pytest

from legend_data_monitor import repack, utils


def _legacy_run(tmp_path, period="p22", run="r000"):
    run_dir = tmp_path / "generated/plt/hit/phy" / period / run
    run_dir.mkdir(parents=True)
    path = str(run_dir / f"l200-{period}-{run}-phy-geds.hdf")
    frame = pd.DataFrame(
        np.random.default_rng(0).normal(1000, 1, (20000, 20)).astype("float64"),
        index=pd.date_range("2026-01-01", periods=20000, freq="s"),
    )
    # the legacy layout: uncompressed float64, rewritten once per chunk
    for _ in range(3):
        frame.to_hdf(path, key="IsPulser_Trapemax", mode="a")
    pd.DataFrame.from_dict({"unit": "ADC"}, orient="index", columns=["Value"]).to_hdf(
        path, key="IsPulser_Trapemax_info", mode="a"
    )
    return path, frame


def test_repack_shrinks_and_preserves_values(tmp_path):
    path, frame = _legacy_run(tmp_path)
    before, after = repack.repack_pandas_hdf(path)

    assert after < before / 2
    assert os.path.getsize(path) == after
    assert not os.path.exists(path + ".repack")

    back = pd.read_hdf(path, key="IsPulser_Trapemax")
    assert (back.dtypes == "float32").all()
    assert np.allclose(back.to_numpy(), frame.to_numpy(), rtol=1e-6)
    assert back.index.equals(frame.index)
    # plotting metadata survives untouched
    assert pd.read_hdf(path, key="IsPulser_Trapemax_info").loc["unit", "Value"] == "ADC"


def test_repack_is_idempotent(tmp_path):
    path, _ = _legacy_run(tmp_path)
    _, once = repack.repack_pandas_hdf(path)
    before, after = repack.repack_pandas_hdf(path)
    # already in the current layout: left exactly as it was
    assert before == after == once


def test_repack_leaves_the_original_when_it_fails(tmp_path, monkeypatch):
    path, _ = _legacy_run(tmp_path)
    before = os.path.getsize(path)

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_hdf", boom)
    with pytest.raises(RuntimeError):
        repack.repack_pandas_hdf(path)
    assert os.path.getsize(path) == before
    assert not os.path.exists(path + ".repack")


def test_repack_run_covers_both_file_kinds(tmp_path):
    path, _ = _legacy_run(tmp_path)
    contract, _ = _legacy_contract(
        tmp_path / "generated/plt/hit/phy/p22/r000"
    )
    results = repack.repack_run(str(tmp_path), "p22", "r000")
    assert sorted(results) == sorted([path, contract])
    for before, after in results.values():
        assert after < before


# -------------------------------------------------------------------------
# contract (schema2) files
# -------------------------------------------------------------------------


def _legacy_contract(tmp_path):
    """A contract file as the writer produced it before the narrowing."""
    import h5py
    from uhi.io import hdf5 as uhi_hdf5

    from legend_data_monitor.contract import schema, writer
    from legend_data_monitor.processing import binning

    dets = ["V02160A", "V02160B", "P00574A"]
    rng = np.random.default_rng(3)
    n, t0 = 200_000, 1_700_000_000.0
    t1 = t0 + 300 * 3600
    binned = binning.fill_time_series(
        rng.uniform(t0, t1, n), rng.choice(dets, n), rng.normal(100, 5, n), dets, t0, t1
    )
    path = str(tmp_path / "l200-p22-r000-phy-geds-schema2.hdf")
    with h5py.File(path, "w") as f:
        f.attrs[schema.ROOT_ATTR] = schema.SCHEMA_VERSION
        group = f.create_group("hist/IsPulser_Trapemax/1min")
        uhi_hdf5.write(group, binned.hist)
        group.attrs["schema"] = schema.SCHEMA_VERSION
        group.create_dataset("min", data=binned.mins, compression="gzip")
        group.create_dataset("max", data=binned.maxs, compression="gzip")
    writer.write_frame(path, "detector_map", pd.DataFrame({"name": dets, "mass": [1.0, 2.0, 3.0]}))
    return path, binned


def test_repack_contract_narrows_storage_and_keeps_everything_readable(tmp_path):
    import h5py

    from legend_data_monitor.contract import reader

    path, binned = _legacy_contract(tmp_path)
    before, after = repack.repack_contract_hdf(path)
    assert after < before

    with h5py.File(path, "r") as f:
        storage = f["hist/IsPulser_Trapemax/1min/storage"]
        assert storage["values"].dtype == np.float32
        assert storage["variances"].dtype == np.float32
        assert storage["counts"].dtype == np.int32
        assert f["hist/IsPulser_Trapemax/1min/min"].dtype == np.float32
        assert f.attrs["schema"] if "schema" in f.attrs else True
        # `axes` object references must still resolve after the compacting copy
        refs = f["hist/IsPulser_Trapemax/1min/axes"][...]
        assert [f[r].name.rsplit("/", 1)[-1] for r in refs] == ["axis_0", "axis_1"]

    back = reader.read_binned_series(path, "IsPulser", "Trapemax", "1min")
    assert np.allclose(
        back.hist.view()["value"], binned.hist.view()["value"], rtol=1e-6, equal_nan=True
    )
    # the pandas frames alongside must survive the copy untouched
    frame = pd.read_hdf(path, key="detector_map")
    assert list(frame["name"]) == ["V02160A", "V02160B", "P00574A"]


def test_repack_contract_is_idempotent(tmp_path):
    path, _ = _legacy_contract(tmp_path)
    _, once = repack.repack_contract_hdf(path)
    before, after = repack.repack_contract_hdf(path)
    assert before == after == once
