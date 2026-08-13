"""Packaged settings constants — the single source of truth for the key vocabulary.

Everything here is loaded from the YAML resources under ``settings/`` when
this module is first imported (importing ``legend_data_monitor`` itself stays
cheap). The file-contract writer exports the relevant vocabulary
(flags, parameter names, labels, units) to consumers via the manifest, so no
other package should need to duplicate these dictionaries.
"""

import importlib.resources

import yaml

pkg = importlib.resources.files("legend_data_monitor")

# load dictionary with plot info (= units, thresholds, label, ...)
with open(pkg / "settings" / "par-settings.yaml") as f:
    PLOT_INFO = yaml.load(f, Loader=yaml.CLoader)

# load dictionary with plot info for Dashboard plots
with open(pkg / "settings" / "mtg-plot-settings.yaml") as f:
    MTG_PLOT_INFO = yaml.load(f, Loader=yaml.CLoader)

# which parameter belongs to which tier
with open(pkg / "settings" / "parameter-tiers.yaml") as f:
    PARAMETER_TIERS = yaml.load(f, Loader=yaml.CLoader)

# which lh5 parameters are needed to be loaded from lh5 to calculate them
with open(pkg / "settings" / "special-parameters.yaml") as f:
    SPECIAL_PARAMETERS = yaml.load(f, Loader=yaml.CLoader)

# flag renames for evt type
with open(pkg / "settings" / "flags.yaml") as f:
    FLAGS_RENAME = yaml.load(f, Loader=yaml.CLoader)

# list of detectors that have no pulser signal in a given period
with open(pkg / "settings" / "no-pulser-dets.yaml") as f:
    NO_PULS_DETS = yaml.load(f, Loader=yaml.CLoader)

# dictionary of keys to ignore
with open(pkg / "settings" / "ignore-keys.yaml") as f:
    IGNORE_KEYS = yaml.load(f, Loader=yaml.CLoader)

# convert all to lists for convenience
for param in SPECIAL_PARAMETERS:
    if isinstance(SPECIAL_PARAMETERS[param], str):
        SPECIAL_PARAMETERS[param] = [SPECIAL_PARAMETERS[param]]

# load SC params and corresponding flags to get specific parameters from big dfs that are stored in the database
with open(pkg / "settings" / "SC-params.yaml") as f:
    SC_PARAMETERS = yaml.load(f, Loader=yaml.CLoader)

# load final calibration run for each period
with open(pkg / "settings" / "final-calibrations.yaml") as f:
    CALIB_RUNS = yaml.load(f, Loader=yaml.CLoader)

# load list of columns to load for a dataframe
COLUMNS_TO_LOAD = [
    "name",
    "location",
    "channel",
    "position",
    "cc4_id",
    "cc4_channel",
    "daq_crate",
    "daq_card",
    "HV_card",
    "HV_channel",
    "det_type",
]

# map position/location for special systems
SPECIAL_SYSTEMS = {"pulser": 0, "pulser01ana": -1, "FCbsln": -2, "muon": -3}

# periods division for SC database access
PERIOD_TO_DB = {
    "p01": "scdbL60",
    **{f"p{str(i).zfill(2)}": "scdbL140" for i in range(2, 14)},
}

# dictionary with timestamps to remove for specific channels
with open(pkg / "settings" / "remove-keys.yaml") as f:
    REMOVE_KEYS = yaml.load(f, Loader=yaml.CLoader)["remove-keys"]

# dictionary with detectors to remove
with open(pkg / "settings" / "remove-dets.yaml") as f:
    REMOVE_DETS = yaml.load(f, Loader=yaml.CLoader)["remove-dets"]
