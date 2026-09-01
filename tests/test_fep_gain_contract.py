"""FEP gain stability: pure computation, contract output, optional rendering.

The figure used to be the only artifact — pickled into a shelve that nothing
reads back. These tests pin the numbers behind it and the contract frame that
now carries them.
"""

import numpy as np
import pandas as pd
import pytest

from legend_data_monitor import calibration
from legend_data_monitor.contract import reader


def _series(n_bins=4, per_bin=10, drift_per_bin=0.0, bin_size=600):
    """Synthetic FEP events: `per_bin` entries in each 10-minute bin."""
    timestamps, values = [], []
    for b in range(n_bins):
        # place entries inside bin b, away from the edges
        timestamps += list(np.linspace(b * bin_size + 1, (b + 1) * bin_size - 1, per_bin))
        values += [2614.5 + b * drift_per_bin] * per_bin
    return np.array(timestamps, dtype=float), np.array(values, dtype=float)


def test_compute_bins_and_baseline():
    timestamps, values = _series(n_bins=3, per_bin=10)
    out = calibration.compute_fep_gain_variation(timestamps, values)
    stats = out["stats"]
    assert len(stats) == 3
    assert list(stats["count"]) == [10, 10, 10]
    # flat input -> baseline is the first bin mean and there is no drift
    assert out["baseline"] == pytest.approx(2614.5)
    assert out["drift"].abs().max() == pytest.approx(0.0)


def test_compute_blanks_sparse_bins_but_keeps_the_count():
    timestamps, values = _series(n_bins=2, per_bin=3)  # below min_counts=5
    out = calibration.compute_fep_gain_variation(timestamps, values)
    assert out["stats"]["mean"].isna().all()
    assert list(out["stats"]["count"]) == [3, 3]
    # no bin qualifies -> no baseline can be defined, so no drift
    assert out["baseline"] is None and out["drift"] is None


def test_compute_drift_is_scaled_to_qbb():
    # +1 keV per bin on a 2614.5 keV peak, expressed at 2039 keV
    timestamps, values = _series(n_bins=3, per_bin=10, drift_per_bin=1.0)
    out = calibration.compute_fep_gain_variation(timestamps, values)
    expected = 1.0 / 2614.5 * 2039.0
    assert out["drift"].iloc[1] == pytest.approx(expected, rel=1e-6)
    assert out["drift"].iloc[2] == pytest.approx(2 * expected, rel=1e-6)


def test_write_fep_gain_contract_roundtrip(tmp_path):
    timestamps, values = _series(n_bins=3, per_bin=10, drift_per_bin=0.5)
    computed = calibration.compute_fep_gain_variation(timestamps, values)

    path = calibration.write_fep_gain_contract(
        str(tmp_path), "p22", "r012", {"V01234A": computed}
    )
    assert path is not None

    frame = reader.read_frame(path, "fep_gain_stab/r012")
    assert set(frame["detector"]) == {"V01234A"}
    assert len(frame) == 3
    assert list(frame.columns) == [
        "detector",
        "run",
        "time_s",
        "mean",
        "std",
        "count",
        "drift_kev",
    ]
    # values match the computation, not a re-derivation
    assert frame["mean"].tolist() == pytest.approx(computed["stats"]["mean"].tolist())
    assert frame["drift_kev"].tolist() == pytest.approx(computed["drift"].tolist())


def test_write_fep_gain_contract_skips_empty_input(tmp_path):
    assert calibration.write_fep_gain_contract(str(tmp_path), "p22", "r012", {}) is None
    assert (
        calibration.write_fep_gain_contract(
            str(tmp_path), "p22", "r012", {"V01234A": None}
        )
        is None
    )


def test_render_false_returns_the_same_numbers_without_drawing(tmp_path):
    """Rendering is optional; the returned drift must not depend on it."""
    import matplotlib.pyplot as plt

    timestamps, values = _series(n_bins=3, per_bin=10, drift_per_bin=0.5)
    chmap = {"name": "V01234A", "string": 1, "position": 2}
    kwargs = dict(
        pars={},
        chmap=chmap,
        timestamps=timestamps,
        values=values,
        output_dir=str(tmp_path),
        save_pdf=False,
        shelf=None,
    )
    before = len(plt.get_fignums())
    means_norender, computed_norender = calibration.fep_gain_variation(
        "p22", "r012", render=False, **kwargs
    )
    assert len(plt.get_fignums()) == before  # nothing was drawn

    means_render, computed_render = calibration.fep_gain_variation(
        "p22", "r012", render=True, **kwargs
    )
    assert means_render.tolist() == pytest.approx(means_norender.tolist())
    assert computed_render["stats"]["mean"].tolist() == pytest.approx(
        computed_norender["stats"]["mean"].tolist()
    )
    plt.close("all")
