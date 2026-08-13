"""Plot-free helpers for loading and manipulating monitoring time series."""

import os
import re

import numpy as np
import pandas as pd
import pytz

from .. import utils

IGNORE_KEYS = utils.IGNORE_KEYS


def compute_diff(
    values: np.ndarray, initial_value: float | int, scale: float | int
) -> np.ndarray:
    """
    Compute relative differences with respect to an initial value. If the initial value is zero, returns an array of nan values.

    Parameters
    ----------
    values : np.ndarray
        Array of values to compute the differences for.
    initial_value : float
        Reference value for computing relative differences.
    scale : float
        Scaling factor.
    """
    if initial_value == 0:
        return np.full_like(values, np.nan, dtype=float)

    return (values - initial_value) / initial_value * scale


def find_hdf_file(
    directory: str, include: list[str], exclude: list[str] = None
) -> str | None:
    """
    Find the original HDF monitoring file in a given directory, matching inclusion/exclusion filters.

    Parameters
    ----------
    directory : str
        Path to the folder containing the HDF monitoring files.
    include: list[str]
        List of words that the HDF monitoring file to retrieve must contain.
    exclude: list[str] = None
        List of words that the HDF monitoring file to retrieve must NOT contain.
    """
    exclude = exclude or []
    files = os.listdir(directory)
    candidates = [
        f
        for f in files
        if f.endswith(".hdf")
        and all(tag in f for tag in include)
        and not any(tag in f for tag in exclude)
    ]

    return os.path.join(directory, candidates[0]) if candidates else None


def read_if_key_exists(hdf_path: str, key: str) -> pd.DataFrame | None:
    """
    Read an HDF dataset if the key exists, otherwise return None; handle the case where the parameter is saved under either '/key' or 'key'.

    Parameters
    ----------
    hdf_path : str
        Path to the HDF file.
    key : str
        Key to inspect.
    """
    with pd.HDFStore(hdf_path, mode="r") as f:
        try:
            return f[key]
        except KeyError:
            try:
                return f["/" + key]
            except KeyError:
                return None


def get_dfs(phy_mtg_data: str, period: str, run_list: list, parameter: str):
    """
    Load and concatenate monitoring data from HDF files for a given period and list of runs.

    Parameters
    ----------
    phy_mtg_data : str
        Path to the base directory containing monitoring HDF5 files (typically ending in `/mtg/phy`).
    period : str
        Period to inspect.
    run_list : list
        List of available runs.
    parameter : str
        Parameter name used to construct the HDF key for loading specific datasets (e.g., 'TrapemaxCtcCal' looks for 'IsPulser_TrapemaxCtcCal').
    """
    # lists to accumulate dataframes, concatenated at the endo only
    geds_df_cuspEmax_abs = []
    geds_df_cuspEmax_abs_corr = []
    puls_df_cuspEmax_abs = []

    base_dir = os.path.join(phy_mtg_data, period)
    runs = os.listdir(base_dir)
    runs = [r for r in runs if re.fullmatch(r"r\d{3}", r)]

    for r in runs:
        if r not in run_list:
            continue
        run_dir = os.path.join(base_dir, r)

        # geds file
        hdf_geds = find_hdf_file(run_dir, include=["geds"], exclude=["res", "min"])
        if hdf_geds:
            geds_abs = read_if_key_exists(hdf_geds, f"IsPulser_{parameter}")
            if geds_abs is not None:
                geds_df_cuspEmax_abs.append(geds_abs)

            geds_puls_abs = read_if_key_exists(
                hdf_geds, f"IsPulser_{parameter}_pulser01anaDiff"
            )
            if geds_puls_abs is not None:
                geds_df_cuspEmax_abs_corr.append(geds_puls_abs)
        else:
            utils.logger.debug("...hdf_geds missing in %s", r)

        # pulser file
        hdf_puls = find_hdf_file(
            run_dir, include=["pulser01ana"], exclude=["res", "min"]
        )
        if hdf_puls:
            puls_abs = read_if_key_exists(hdf_puls, f"IsPulser_{parameter}")
            if puls_abs is not None:
                puls_df_cuspEmax_abs.append(puls_abs)
        else:
            utils.logger.debug("...hdf_puls missing in %s", r)

    if (
        not geds_df_cuspEmax_abs
        and not geds_df_cuspEmax_abs_corr
        and not puls_df_cuspEmax_abs
    ):
        return None, None, None
    else:
        return (
            (
                pd.concat(geds_df_cuspEmax_abs, ignore_index=False, axis=0)
                if geds_df_cuspEmax_abs
                else pd.DataFrame()
            ),
            (
                pd.concat(geds_df_cuspEmax_abs_corr, ignore_index=False, axis=0)
                if geds_df_cuspEmax_abs_corr
                else pd.DataFrame()
            ),
            (
                pd.concat(puls_df_cuspEmax_abs, ignore_index=False, axis=0)
                if puls_df_cuspEmax_abs
                else pd.DataFrame()
            ),
        )


def get_traptmax_tp0est(phy_mtg_data: str, period: str, run_list: list):
    """
    Load and concatenate trapTmax and tp0est data from HDF files for a given period and list of runs.

    Parameters
    ----------
    phy_mtg_data : str
        Path to the base directory containing monitoring HDF5 files (typically ending in `/mtg/phy`).
    period : str
        Period to inspect.
    run_list : list
        List of available runs.
    """
    geds_df_trapTmax, geds_df_tp0est = [], []
    puls_df_trapTmax, puls_df_tp0est = [], []

    base_dir = os.path.join(phy_mtg_data, period)
    for r in os.listdir(base_dir):
        if r not in run_list:
            continue
        run_dir = os.path.join(base_dir, r)

        # geds
        hdf_geds = find_hdf_file(run_dir, include=["geds"], exclude=["res", "min"])
        if hdf_geds:
            trapTmax = read_if_key_exists(hdf_geds, "IsPulser_TrapTmax")
            if trapTmax is not None:
                geds_df_trapTmax.append(trapTmax)

            tp0est = read_if_key_exists(hdf_geds, "IsPulser_Tp0Est")
            if tp0est is not None:
                geds_df_tp0est.append(tp0est)

        # pulser
        hdf_puls = find_hdf_file(
            run_dir, include=["pulser01ana"], exclude=["res", "min"]
        )
        if hdf_puls:
            trapTmax = read_if_key_exists(hdf_puls, "IsPulser_TrapTmax")
            if trapTmax is not None:
                puls_df_trapTmax.append(trapTmax)

            tp0est = read_if_key_exists(hdf_puls, "IsPulser_Tp0Est")
            if tp0est is not None:
                puls_df_tp0est.append(tp0est)

    return (
        (
            pd.concat(geds_df_trapTmax, ignore_index=False)
            if geds_df_trapTmax
            else pd.DataFrame()
        ),
        (
            pd.concat(geds_df_tp0est, ignore_index=False)
            if geds_df_tp0est
            else pd.DataFrame()
        ),
        (
            pd.concat(puls_df_trapTmax, ignore_index=False)
            if puls_df_trapTmax
            else pd.DataFrame()
        ),
        (
            pd.concat(puls_df_tp0est, ignore_index=False)
            if puls_df_tp0est
            else pd.DataFrame()
        ),
    )


def filter_series_by_ignore_keys(
    series_to_filter: pd.Series, skip_keys: dict, period: str
):
    """
    Remove data from a time-indexed pandas Series that falls within time ranges specified by start and stop timestamps for a given period.

    Parameters
    ----------
    series_to_filter : pd.Series
        The time-indexed pandas Series to be filtered.
    skip_keys : dict
        Dictionary mapping periods to sub-dictionaries containing 'start_keys' and 'stop_keys' lists with timestamp strings in the format '%Y%m%dT%H%M%S%z'.
    period : str
        The period to check for keys to ignore. If not present, the series is returned unmodified.
    """
    if period not in skip_keys:
        return series_to_filter

    start_keys = skip_keys[period]["start_keys"]
    stop_keys = skip_keys[period]["stop_keys"]

    for ki, kf in zip(start_keys, stop_keys):
        isolated_ki = pd.to_datetime(ki.replace("Z", "+0000"), format="%Y%m%dT%H%M%S%z")
        isolated_kf = pd.to_datetime(kf.replace("Z", "+0000"), format="%Y%m%dT%H%M%S%z")
        series_to_filter = series_to_filter[
            (series_to_filter.index < isolated_ki)
            | (series_to_filter.index > isolated_kf)
        ]

    return series_to_filter


def filter_by_period(series: pd.Series, period: str | list) -> pd.Series:
    """
    Return a series filtered by ignore keys for the given period(s).

    Parameters
    ----------
    series : pd.Series
        Input time series (indexed by timestamps) to filter.
    period : str or list
        Period (or list of periods) to inspect.
    """
    if isinstance(period, list):
        for p in period:
            series = filter_series_by_ignore_keys(series, IGNORE_KEYS, p)
    else:
        series = filter_series_by_ignore_keys(series, IGNORE_KEYS, period)

    return series


def compute_diff_and_rescaling(
    series: pd.Series, reference: float, escale: float, variations: bool
):
    """
    Compute relative differences (if 'variations' is True) and rescale values by 'escale'.

    Parameters
    ----------
    series : pd.Series
        Input time series of numerical values.
    reference : float
        Reference value used to compute relative differences.
    escale : float
        Scaling factor, eg 2039 keV.
    variations : bool
        If true, compute relative difference (series - reference)/reference.
    """
    if variations:
        diff = (series - reference) / reference
    else:
        diff = series.copy()

    return diff, diff * escale


def resample_series(series: pd.Series, resampling_time: str, mask: pd.Series):
    """
    Calculate mean/std for resampled time ranges to which a mask is then applied. The function already adds UTC timezones to the series.

    Parameters
    ----------
    series : pd.Series
        Input time series of numerical values.
    resampling_time : str
        Resampling frequency, eg '1h'.
    mask : pd.Series
        Boolean mask aligned to the datetime index; false values mark timestamps that should be excluded, ie set to nan value.
    """
    mean = series.resample(resampling_time).mean()
    std = series.resample(resampling_time).std()

    # add UTC timezone
    if mean.index.tz is None:
        mean = mean.tz_localize("UTC")
        std = std.tz_localize("UTC")
    # different timezone, convert to UTC
    elif mean.index.tz != pytz.UTC:
        mean = mean.tz_convert("UTC")
        std = std.tz_convert("UTC")

    # ensure mask has the same timezone as the resampled series
    if not mask.index.tz:
        mask = mask.tz_localize("UTC")

    # set to nan when the mask is False
    mask = mask.reindex(mean.index, fill_value=False)
    mean[~mask] = np.nan
    std[~mask] = np.nan

    return mean, std


def get_pulser_data(
    resampling_time: str,
    period: str | list,
    dfs: list,
    channel: str,
    escale: float,
    variations=False,
) -> dict:
    """
    Return a dictionary of geds and pulser filtered dataframes for which a time resampling is performed.

    Parameters
    ----------
    resampling_time : str
        Resampling time, eg '1HH' or '10T'.
    period : str | list
        Period or list of periods to inspect.
    dfs : list
        List of dataframes for geds and pulser events.
    channel : str
        Channel to inspect.
    escale : float
        Scaling factor used to compute relative differences in gain and calibration constant.
    variations : bool
        True if you want to retrieve % variations (default: False).
    """
    # geds
    ser_ged_cusp = dfs[0][channel].sort_index()
    ser_ged_cusp = filter_by_period(ser_ged_cusp, period)
    ser_ged_cusp = ser_ged_cusp[
        ~ser_ged_cusp.index.duplicated(keep="first")
    ]  # remove duplicates
    ser_pul_tp0est_new = pd.DataFrame()

    if ser_ged_cusp.empty:
        utils.logger.debug("...geds series is empty after filtering")
        return None

    # check if these dfs are empty or not - if not, then remove spikes
    if isinstance(dfs[6], pd.DataFrame) and not dfs[6].empty:
        ser_pul_tp0est = dfs[6][1027203].sort_index()
        ser_pul_tp0est = filter_by_period(ser_pul_tp0est, period)
        ser_pul_tp0est = ser_pul_tp0est[
            ~ser_pul_tp0est.index.duplicated(keep="first")
        ]  # remove duplicates

        low_lim = 4.8e4
        upp_lim = 5.0e4
        mask = (ser_pul_tp0est > low_lim) & (ser_pul_tp0est < upp_lim)
        ser_pul_tp0est_new = ser_pul_tp0est[mask]

        if not ser_pul_tp0est_new.empty:
            valid_idx = ser_ged_cusp.index.intersection(ser_pul_tp0est_new.index)
            ser_ged_cusp = ser_ged_cusp.reindex(valid_idx)

    # if before, potential mismatches with ser_pul_tp0est
    ser_ged_cusp = ser_ged_cusp.dropna()
    # compute average over the first 10% of elements
    n_elements = max(int(len(ser_ged_cusp) * 0.10), 1)
    ged_cusp_av = np.nanmean(ser_ged_cusp.iloc[:n_elements])
    if np.isnan(ged_cusp_av):
        utils.logger.debug("...the geds average is NaN")
        return None

    ser_ged_cuspdiff, ser_ged_cuspdiff_kev = compute_diff_and_rescaling(
        ser_ged_cusp, ged_cusp_av, escale, variations
    )

    # hour counts masking
    mask = ser_ged_cusp.resample(resampling_time).count() > 0

    # resample geds series
    ged_cusp_hr_av, ged_cusp_hr_std = resample_series(
        ser_ged_cuspdiff_kev, resampling_time, mask
    )
    ged_index = ged_cusp_hr_av.index

    # pulser series
    ser_pul_cusp = ser_pul_cuspdiff = ser_pul_cuspdiff_kev = pul_cusp_hr_av = (
        pul_cusp_hr_std
    ) = None
    ged_cusp_corr = ged_cusp_corr_kev = ged_cusp_cor_hr_av = ged_cusp_cor_hr_std = None
    # ...if pulser is available:
    if not dfs[2].empty:
        ser_pul_cusp = dfs[2][1027203].sort_index()
        ser_pul_cusp = ser_pul_cusp[
            ~ser_pul_cusp.index.duplicated(keep="first")
        ]  # remove duplicates
        ser_pul_cusp = filter_by_period(ser_pul_cusp, period)

        # pulser average and diffs
        if not ser_pul_cusp.empty:
            # check if these dfs are empty or not - if not, then remove spikes
            if isinstance(dfs[6], pd.DataFrame) and not dfs[6].empty:
                if not ser_pul_tp0est_new.empty:
                    valid_idx = ser_pul_cusp.index.intersection(
                        ser_pul_tp0est_new.index
                    )
                    ser_pul_cusp = ser_pul_cusp.reindex(valid_idx)

            # if before, potential mismatches with ser_pul_tp0est
            ser_pul_cusp = ser_pul_cusp.dropna()
            n_elements_pul = max(int(len(ser_pul_cusp) * 0.10), 1)
            pul_cusp_av = np.nanmean(ser_pul_cusp.iloc[:n_elements_pul])
            ser_pul_cuspdiff, ser_pul_cuspdiff_kev = compute_diff_and_rescaling(
                ser_pul_cusp, pul_cusp_av, escale, variations
            )

            pul_cusp_hr_av, pul_cusp_hr_std = resample_series(
                ser_pul_cuspdiff_kev, resampling_time, mask
            )
            pul_cusp_hr_av = pul_cusp_hr_av.reindex(ged_index)
            pul_cusp_hr_std = pul_cusp_hr_std.reindex(ged_index)

            # corrected GED
            common_index = ser_ged_cuspdiff.index.intersection(ser_pul_cuspdiff.index)
            ged_cusp_corr = (
                ser_ged_cuspdiff[common_index] - ser_pul_cuspdiff[common_index]
            )
            ged_cusp_corr_kev = ged_cusp_corr * escale
            ged_cusp_cor_hr_av, ged_cusp_cor_hr_std = resample_series(
                ged_cusp_corr_kev, resampling_time, mask
            )
            ged_cusp_cor_hr_av = ged_cusp_cor_hr_av.reindex(ged_index)
            ged_cusp_cor_hr_std = ged_cusp_cor_hr_std.reindex(ged_index)

    return {
        "ged": {
            "cusp": ser_ged_cusp,
            "cuspdiff": ser_ged_cuspdiff,
            "cuspdiff_kev": ser_ged_cuspdiff_kev,
            "kevdiff_av": ged_cusp_hr_av,
            "kevdiff_std": ged_cusp_hr_std,
        },
        "pul_cusp": {
            "raw": ser_pul_cusp,
            "rawdiff": ser_pul_cuspdiff,
            "kevdiff": ser_pul_cuspdiff_kev,
            "kevdiff_av": pul_cusp_hr_av,
            "kevdiff_std": pul_cusp_hr_std,
        },
        "diff": {
            "raw": None,
            "rawdiff": ged_cusp_corr,
            "kevdiff": ged_cusp_corr_kev,
            "kevdiff_av": ged_cusp_cor_hr_av,
            "kevdiff_std": ged_cusp_cor_hr_std,
        },
    }
