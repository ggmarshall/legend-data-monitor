"""Dead time travels as contract data, not as a value hidden in a shelve.

qc_and_evt_summary_plots computes the discharge dead time; qc_average needs it
to title and flag its IsDischarge plot. That hand-off used to run through the
pickled-figure shelve, so it broke (TypeError on None) whenever the producer
had not run. These tests pin the contract hand-off and the guard.
"""

import pandas as pd
import pytest

from legend_data_monitor import monitoring
from legend_data_monitor.contract import reader


def test_period_contract_path_is_per_period_and_datatype():
    assert monitoring.period_contract_path("/out", "p22").endswith(
        "p22/l200-p22-phy-monitoring.hdf"
    )
    assert monitoring.period_contract_path("/out", "p22", "cal").endswith(
        "p22/l200-p22-cal-monitoring.hdf"
    )


def test_dead_time_roundtrip(tmp_path):
    path = monitoring.write_dead_time(str(tmp_path), "p22", "r012", 123.0, 0.456)
    frame = reader.read_frame(path, "dead_time/r012")
    assert frame["dead_time_s"].iloc[0] == pytest.approx(123.0)

    out = monitoring.read_dead_time(str(tmp_path), "p22", "r012")
    assert out == {"dead_time_s": pytest.approx(123.0), "dead_time_pct": pytest.approx(0.456)}


def test_dead_time_runs_are_independent(tmp_path):
    monitoring.write_dead_time(str(tmp_path), "p22", "r011", 10.0, 0.1)
    monitoring.write_dead_time(str(tmp_path), "p22", "r012", 20.0, 0.2)
    assert monitoring.read_dead_time(str(tmp_path), "p22", "r011")["dead_time_s"] == 10.0
    assert monitoring.read_dead_time(str(tmp_path), "p22", "r012")["dead_time_s"] == 20.0


def test_read_dead_time_returns_none_when_absent(tmp_path):
    # no file at all
    assert monitoring.read_dead_time(str(tmp_path), "p22", "r012") is None
    # file exists but this run was never written (the case that used to raise
    # TypeError downstream when qc_and_evt_summary_plots had not run)
    monitoring.write_dead_time(str(tmp_path), "p22", "r011", 10.0, 0.1)
    assert monitoring.read_dead_time(str(tmp_path), "p22", "r012") is None


def test_dead_time_frame_is_plain_and_readable(tmp_path):
    """The point of the move: reachable without unpickling a figure."""
    import h5py

    path = monitoring.write_dead_time(str(tmp_path), "p22", "r012", 1.5, 0.02)
    with h5py.File(path, "r") as f:
        assert "dead_time" in f
    assert isinstance(reader.read_frame(path, "dead_time/r012"), pd.DataFrame)


# -------------------------------------------------------------------------
# per-detector summary (the box-plot data)
# -------------------------------------------------------------------------


def _det_info():
    return {
        "detectors": {
            "V01234A": {"string": 1, "position": 2, "usability": "on"},
            "V05678B": {"string": 1, "position": 3, "usability": "ac"},
        }
    }


def _pars(fwhm=3.1):
    return {
        "V01234A": {
            "results": {"ecal": {"cuspEmax_ctc_cal": {"eres_linear": {"Qbb_fwhm_in_kev": fwhm}}}}
        }
    }


def test_detector_summary_aggregates_per_detector():
    results = {"V01234A": [1.0, 2.0, 3.0], "V05678B": [10.0, 20.0]}
    frame = monitoring.compute_detector_summary(results, _det_info(), _pars())
    row = frame[frame.ged == "V01234A"].iloc[0]
    assert row["mean"] == pytest.approx(2.0)
    assert row["min"] == pytest.approx(1.0) and row["max"] == pytest.approx(3.0)
    assert row["fwhm"] == pytest.approx(3.1)
    assert row["string"] == 1 and row["pos"] == 2 and row["usability"] == "on"
    # a detector missing from the calibration pars keeps its stats, fwhm is NaN
    other = frame[frame.ged == "V05678B"].iloc[0]
    assert other["mean"] == pytest.approx(15.0)
    assert pd.isna(other["fwhm"])


def test_detector_summary_handles_empty_and_unknown_detectors():
    results = {"V01234A": [], "V05678B": None, "NOT_IN_MAP": [1.0]}
    frame = monitoring.compute_detector_summary(results, _det_info(), _pars())
    # unknown detectors are dropped, empty ones kept with NaN stats
    assert set(frame["ged"]) == {"V01234A", "V05678B"}
    assert frame["mean"].isna().all()


def test_detector_summary_ignores_nans_in_the_values():
    results = {"V01234A": [1.0, float("nan"), 3.0]}
    frame = monitoring.compute_detector_summary(results, _det_info(), _pars())
    assert frame["mean"].iloc[0] == pytest.approx(2.0)
    assert frame["max"].iloc[0] == pytest.approx(3.0)


def test_write_detector_summary_roundtrip(tmp_path):
    frame = monitoring.compute_detector_summary(
        {"V01234A": [1.0, 2.0, 3.0]}, _det_info(), _pars()
    )
    path = monitoring.write_detector_summary(
        str(tmp_path), "p22", "r012", "FEP_gain_stab", frame
    )
    back = reader.read_frame(path, "detector_summary/FEP_gain_stab/r012")
    assert back["ged"].tolist() == ["V01234A"]
    assert back["mean"].iloc[0] == pytest.approx(2.0)


def test_write_detector_summary_skips_empty(tmp_path):
    assert (
        monitoring.write_detector_summary(
            str(tmp_path), "p22", "r012", "FEP_gain_stab", pd.DataFrame()
        )
        is None
    )


# -------------------------------------------------------------------------
# QC rates (qc_average)
# -------------------------------------------------------------------------


def _flag_frame(values, start="2026-08-01", freq="1h"):
    idx = pd.date_range(start, periods=len(values), freq=freq, tz="UTC")
    return pd.DataFrame(values, index=idx)


def test_qc_rate_is_counts_per_second_in_mhz():
    # 3 hits over a 2 h span -> 3 / 7200 s = 0.4167 mHz
    frame = _flag_frame([[1], [1], [1]])
    rates = monitoring.compute_qc_rate_mhz(frame, "p22")
    assert rates.iloc[0] == pytest.approx(3 / 7200 * 1000)


def test_qc_rate_is_per_detector_column():
    frame = _flag_frame([[1, 0], [1, 1], [0, 1]])
    rates = monitoring.compute_qc_rate_mhz(frame, "p22")
    assert len(rates) == 2
    assert rates.iloc[0] == pytest.approx(2 / 7200 * 1000)
    assert rates.iloc[1] == pytest.approx(2 / 7200 * 1000)


def test_qc_rate_returns_none_without_a_usable_span():
    assert monitoring.compute_qc_rate_mhz(pd.DataFrame(), "p22") is None
    # a single sample has no span to divide by
    assert monitoring.compute_qc_rate_mhz(_flag_frame([[1]]), "p22") is None


def test_write_qc_rates_labels_detectors_and_roundtrips(tmp_path):
    frame = _flag_frame([[1, 0], [1, 1], [0, 1]])
    frame.columns = [1104000, 1104001]
    rates = monitoring.compute_qc_rate_mhz(frame, "p22")
    detectors = {
        "V01234A": {"daq_rawid": 1104000},
        "V05678B": {"daq_rawid": 1104001},
    }
    path = monitoring.write_qc_rates(
        str(tmp_path), "p22", "r012", {"IsDischarge": rates}, detectors
    )
    back = reader.read_frame(path, "qc_average/r012")
    assert set(back["flag"]) == {"IsDischarge"}
    assert set(back["detector"]) == {"V01234A", "V05678B"}
    assert back["rate_mhz"].tolist() == pytest.approx(rates.tolist())


def test_write_qc_rates_skips_when_nothing_computed(tmp_path):
    assert (
        monitoring.write_qc_rates(str(tmp_path), "p22", "r012", {"IsDischarge": None}, {})
        is None
    )


# -------------------------------------------------------------------------
# QC rate versus time (qc_time_series)
# -------------------------------------------------------------------------


def test_qc_rate_series_resamples_to_mhz_per_cadence():
    # 2 hits in the first hour, 1 in the second -> 2/3600*1000 then 1/3600*1000
    idx = pd.to_datetime(
        ["2026-08-01T00:10", "2026-08-01T00:40", "2026-08-01T01:20"], utc=True
    )
    frame = pd.DataFrame({1104000: [1, 1, 1]}, index=idx)
    rates = monitoring.compute_qc_rate_series(frame, "p22")
    assert rates.iloc[0, 0] == pytest.approx(2 / 3600 * 1000)
    assert rates.iloc[1, 0] == pytest.approx(1 / 3600 * 1000)


def test_qc_rate_series_renames_columns_to_detectors():
    idx = pd.date_range("2026-08-01", periods=2, freq="1h", tz="UTC")
    frame = pd.DataFrame({1104000: [1, 0]}, index=idx)
    rates = monitoring.compute_qc_rate_series(
        frame, "p22", detectors={"V01234A": {"daq_rawid": 1104000}}
    )
    assert list(rates.columns) == ["V01234A"]


def test_qc_rate_series_handles_empty_input():
    assert monitoring.compute_qc_rate_series(pd.DataFrame(), "p22") is None


def test_write_qc_rate_series_roundtrip(tmp_path):
    idx = pd.date_range("2026-08-01", periods=3, freq="1h", tz="UTC")
    frame = pd.DataFrame({1104000: [1, 1, 0]}, index=idx)
    rates = monitoring.compute_qc_rate_series(
        frame, "p22", detectors={"V01234A": {"daq_rawid": 1104000}}
    )
    path = monitoring.write_qc_rate_series(
        str(tmp_path), "p22", "r012", "IsDischarge", rates
    )
    back = reader.read_frame(path, "qc_rate_series/IsDischarge/r012")
    assert list(back.columns) == ["V01234A"]
    assert back["V01234A"].tolist() == pytest.approx(rates["V01234A"].tolist())


def test_write_qc_rate_series_skips_empty(tmp_path):
    assert (
        monitoring.write_qc_rate_series(str(tmp_path), "p22", "r012", "IsDischarge", None)
        is None
    )
