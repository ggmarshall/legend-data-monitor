"""spms_noise / spms_calibration period keys from production by-products."""

import os

import pandas as pd
import pytest
import yaml

from legend_data_monitor import monitoring
from legend_data_monitor.contract import reader


def _mock_prod(tmp_path):
    par = tmp_path / "generated/par/dsp/phy/p22/r012"
    par.mkdir(parents=True)
    for key, fwhm in [("20260731T181831Z", 0.40), ("20260731T191833Z", 0.42)]:
        (par / f"l200-p22-r012-phy-{key}-par_dsp_spms.yaml").write_text(
            yaml.safe_dump({"S002": {"baseline_curr_fwhm": fwhm}, "S003": {}})
        )
    ovr = tmp_path / "inputs/dataprod/overrides/hit"
    (ovr / "lar/p19/r005").mkdir(parents=True)
    (ovr / "lar/p19/r005/l200-p19-r005-phy-lar-T%-par_hit-overwrite.yaml").write_text(
        yaml.safe_dump(
            {
                "S002": {
                    "pars": {
                        "operations": {
                            "energy_in_pe": {"parameters": {"a": 0.02, "m": 0.7}},
                            "is_valid_hit": {"parameters": {"a": 0.5}},
                        }
                    }
                }
            }
        )
    )
    (ovr / "validity.yaml").write_text(
        yaml.safe_dump(
            [
                {
                    "valid_from": "20260116T191157Z",
                    "mode": "append",
                    "apply": [
                        "lar/p19/r005/l200-p19-r005-phy-lar-T%-par_hit-overwrite.yaml"
                    ],
                }
            ]
        )
    )
    return str(tmp_path)


def test_spms_production_keys(tmp_path):
    prod = _mock_prod(tmp_path)
    out = tmp_path / "out"
    keys = monitoring.write_spms_production_keys(
        str(out), "p22", "r012", prod, start_key="20260731T181831Z"
    )
    assert keys == ["spms_noise/r012", "spms_calibration/r012"]
    path = monitoring.period_contract_path(str(out), "p22")
    noise = reader.read_frame(path, "spms_noise/r012")
    assert list(noise.columns) == ["S002", "S003"]
    assert noise["S002"].tolist() == pytest.approx([0.40, 0.42])
    assert noise["S003"].isna().all() and isinstance(noise.index, pd.DatetimeIndex)
    calib = reader.read_frame(path, "spms_calibration/r012")
    assert calib.loc["S002", "pe_m"] == 0.7 and calib.loc["S002", "threshold_a"] == 0.5
    assert calib.loc["S002", "source"].startswith("lar/p19/r005/")


def test_spms_production_keys_absent_inputs(tmp_path):
    assert monitoring.write_spms_production_keys(str(tmp_path), "p22", "r012", str(tmp_path)) == []
    assert not os.path.exists(monitoring.period_contract_path(str(tmp_path), "p22"))
