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


def test_repack_run_skips_contract_files(tmp_path):
    path, _ = _legacy_run(tmp_path)
    schema2 = path.replace("-geds.hdf", "-geds-schema2.hdf")
    with open(schema2, "wb") as f:
        f.write(b"not a pandas file")
    results = repack.repack_run(str(tmp_path), "p22", "r000")
    assert list(results) == [path]
    with open(schema2, "rb") as f:
        assert f.read() == b"not a pandas file"
