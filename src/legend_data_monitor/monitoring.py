import glob
import itertools
import json
import math
import os
import pickle
import shelve
from functools import partial

import awkward as ak
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from lgdo.lh5 import read_as
from matplotlib.patches import Patch

from . import errors, utils
from .contract import reader as contract_reader
from .contract import writer as contract_writer

# --- Phase 4 re-export shims: these functions moved to loading/ and processing/;
# import them here so existing ``monitoring.X`` references keep working. ---
from .loading.calib_files import (  # noqa: F401
    _first_run_key,
    _load_validity_file,
    _run_times_cache,
    add_calibration_runs,
    evaluate_fep_cal,
    extract_fep_peak,
    extract_resolution_at_q_bb,
    get_calib_data_dict,
    get_calib_pars,
    get_calibration_file,
    get_energy_key,
    get_run_start_end_times,
    get_tier_keyresult,
)
from .processing.series import (  # noqa: F401
    compute_diff,
    compute_diff_and_rescaling,
    filter_by_period,
    filter_series_by_ignore_keys,
    find_hdf_file,
    get_dfs,
    get_pulser_data,
    get_traptmax_tp0est,
    read_if_key_exists,
    resample_series,
)

# -------------------------------------------------------------------------

SMALL_SIZE = 8


def period_contract_path(output_folder: str, period: str, data_type: str = "phy") -> str:
    """Path of the period-level monitoring contract file.

    One file per (period, datatype) holding the numbers the monitoring figures
    are drawn from, so consumers no longer have to unpickle a matplotlib
    figure out of a shelve to reach them.
    """
    return os.path.join(
        output_folder, period, f"l200-{period}-{data_type}-monitoring.hdf"
    )


def write_dead_time(
    output_folder: str, period: str, run: str, dead_time_s: float, dead_time_pct: float
) -> str:
    """Record the discharge dead time of a run in the period contract file."""
    path = period_contract_path(output_folder, period)
    contract_writer.write_frame(
        path,
        f"dead_time/{run}",
        pd.DataFrame([{"run": run, "dead_time_s": dead_time_s, "dead_time_pct": dead_time_pct}]),
    )
    return path


def read_dead_time(output_folder: str, period: str, run: str) -> dict | None:
    """Dead time of a run, or None when it has not been computed yet.

    Callers must handle None: the value comes from qc_and_evt_summary_plots,
    which may not have run for this run yet.
    """
    path = period_contract_path(output_folder, period)
    if not os.path.isfile(path):
        return None
    try:
        frame = contract_reader.read_frame(path, f"dead_time/{run}")
    except (KeyError, OSError):
        return None
    if frame is None or frame.empty:
        return None
    row = frame.iloc[0]
    return {
        "dead_time_s": float(row["dead_time_s"]),
        "dead_time_pct": float(row["dead_time_pct"]),
    }


def apply_monitoring_style():
    """Apply the monitoring plot style to matplotlib's global rcParams.

    Called by the plot-generating functions; importing this module must not
    restyle the host application's matplotlib.
    """
    plt.rc("font", size=SMALL_SIZE)
    plt.rc("axes", titlesize=SMALL_SIZE)
    plt.rc("axes", labelsize=SMALL_SIZE)
    plt.rc("xtick", labelsize=SMALL_SIZE)
    plt.rc("ytick", labelsize=SMALL_SIZE)
    plt.rc("legend", fontsize=SMALL_SIZE)
    plt.rc("figure", titlesize=SMALL_SIZE)
    plt.rcParams["font.family"] = "serif"
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    plt.rc("axes", facecolor="white", edgecolor="black", axisbelow=True, grid=True)


IGNORE_KEYS = utils.IGNORE_KEYS
CALIB_RUNS = utils.CALIB_RUNS


# -------------------------------------------------------------------------
def qc_distributions(
    auto_dir_path: str,
    phy_mtg_data: str,
    output_folder: str,
    start_key: str,
    period: str,
    run: str,
    last_cycle: str,
    det_info: dict,
    save_pdf: bool,
):
    apply_monitoring_style()
    pars_to_inspect = [
        "IsValidBlSlopeClassifier",
        "IsValidTailRmsClassifier",
        "IsValidPzSlopeClassifier",
        "IsValidBlSlopeRmsClassifier",
        "IsValidBlPolyRmsClassifier",
        "IsValidBlSlopeRmsClassifier",
        "IsValidCuspeminClassifier",
        "IsValidCuspemaxClassifier",
    ]

    my_file = os.path.join(
        output_folder, f"{period}/{run}/l200-{period}-{run}-phy-geds.hdf"
    )
    str_chns = det_info["str_chns"]
    utils.logger.debug("...inspecting QC classifiers")
    if not os.path.exists(my_file):
        utils.logger.warning(f"...file not found: {my_file}. Return!")
        return

    end_folder = os.path.join(
        output_folder,
        period,
        run,
        "mtg",
    )
    os.makedirs(end_folder, exist_ok=True)
    shelve_path = os.path.join(
        end_folder,
        f"l200-{period}-{run}-phy-monitoring",
    )

    step = 0.4
    with (
        shelve.open(shelve_path, "c", protocol=pickle.HIGHEST_PROTOCOL) as shelf,
        pd.HDFStore(my_file, "r") as store,
    ):
        df_energy_IsPhysics = store["/IsPhysics_TrapemaxCtcCal"]
        df_energy_IsPhysics = filter_series_by_ignore_keys(
            df_energy_IsPhysics, utils.IGNORE_KEYS, period
        )

        for par in pars_to_inspect:

            mask = df_energy_IsPhysics > 25
            df_All = utils.load_and_filter(store, f"/All_{par}")
            df_IsPulser = utils.load_and_filter(store, f"/IsPulser_{par}")
            df_IsBsln = utils.load_and_filter(store, f"/IsBsln_{par}")
            df_IsPhysics = utils.load_and_filter(store, f"/IsPhysics_{par}", mask=mask)

            if df_All.empty:
                continue

            df_All = filter_series_by_ignore_keys(df_All, utils.IGNORE_KEYS, period)
            if not df_IsPulser.empty:
                df_IsPulser = filter_series_by_ignore_keys(
                    df_IsPulser, utils.IGNORE_KEYS, period
                )
            if not df_IsBsln.empty:
                df_IsBsln = filter_series_by_ignore_keys(
                    df_IsBsln, utils.IGNORE_KEYS, period
                )
            if not df_IsPhysics.empty:
                df_IsPhysics = filter_series_by_ignore_keys(
                    df_IsPhysics, utils.IGNORE_KEYS, period
                )

            for string, det_list in str_chns.items():
                # grid size
                n_dets = len(det_list)
                ncols = math.ceil(math.sqrt(n_dets))
                nrows = math.ceil(n_dets / ncols)

                fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows))
                axes = axes.flatten()

                for i, det in enumerate(det_list):
                    if det not in det_info["detectors"]:
                        continue
                    if not det_info["detectors"][det]["processable"]:
                        continue

                    ax = axes[i]
                    ch = det_info["detectors"][det]["daq_rawid"]
                    # not processed detectors
                    if ch not in df_All.keys():
                        continue

                    vals_all = utils.get_vals(df_All, ch)
                    vals_pulser = utils.get_vals(df_IsPulser, ch)
                    vals_bsln = utils.get_vals(df_IsBsln, ch)
                    vals_phys = utils.get_vals(df_IsPhysics, ch)

                    vals_all = vals_all[~np.isnan(vals_all)]
                    vals_pulser = vals_pulser[~np.isnan(vals_pulser)]
                    vals_bsln = vals_bsln[~np.isnan(vals_bsln)]
                    vals_phys = vals_phys[~np.isnan(vals_phys)]

                    # global bins
                    bins = np.arange(-15, 15 + step, step)

                    # percentages
                    def safe_perc(vals, lo=-5, hi=5):
                        if len(vals) == 0:
                            return np.nan
                        return 100 * np.mean((vals >= lo) & (vals <= hi))

                    perc_all = safe_perc(vals_all)
                    perc_pulser = safe_perc(vals_pulser)
                    perc_bsln = safe_perc(vals_bsln)
                    perc_phys = safe_perc(vals_phys)

                    # plotting
                    ax.hist(
                        vals_all,
                        bins=bins,
                        label=f"All events - {perc_all:.1f}%",
                        histtype="step",
                        facecolor="g",
                    )
                    ax.hist(
                        vals_pulser,
                        bins=bins,
                        label=f"TP - {perc_pulser:.1f}%",
                        histtype="step",
                        facecolor="g",
                    )
                    ax.hist(
                        vals_bsln,
                        bins=bins,
                        label=f"FT - {perc_bsln:.1f}%",
                        histtype="step",
                        facecolor="g",
                    )
                    ax.hist(
                        vals_phys,
                        bins=bins,
                        label=f"~TP, ~FT, E>25 keV - {perc_phys:.1f}%",
                        histtype="step",
                        facecolor="g",
                    )

                    ax.axvline(-5, color="k", linestyle="--")
                    ax.axvline(5, color="k", linestyle="--")
                    ax.axvspan(-15, -5, color="darkgray", alpha=0.2)
                    ax.axvspan(5, 15, color="darkgray", alpha=0.2)
                    ax.set_ylabel("Counts")
                    ax.set_xlabel("Classifiers")
                    ax.legend(
                        title=f"{det} (pos {det_info['detectors'][det]['position']})",
                        loc="upper right",
                    )
                    ax.set_yscale("log")
                    ax.grid(False)
                    ax.set_xlim(-10, 10)

                # hide any unused subplots
                for j in range(i + 1, len(axes)):
                    axes[j].axis("off")

                fig.suptitle(
                    f"{period} {run} - string {string} - {par} - last cycle: {last_cycle}"
                )
                fig.tight_layout()

                if save_pdf:
                    pdf_folder = os.path.join(
                        output_folder, f"{period}/{run}/mtg/pdf", f"st{string}"
                    )
                    os.makedirs(pdf_folder, exist_ok=True)
                    plt.savefig(
                        os.path.join(
                            pdf_folder,
                            f"{period}_{run}_string{string}_{par}.pdf",
                        ),
                        bbox_inches="tight",
                    )

                # serialize+plot in a shelve object
                shelf[f"{period}_{run}_{par}"] = pickle.dumps(fig)
                plt.close()


def mhz_to_percent(mhz, avg_total_forced_mhz):
    return (mhz / avg_total_forced_mhz) * 100


def percent_to_mhz(pct, avg_total_forced_mhz):
    return (pct / 100) * avg_total_forced_mhz


def qc_and_evt_summary_plots(
    auto_dir_path: str,
    phy_mtg_data: str,
    output_folder: str,
    start_key: str,
    period: str,
    run: str,
    last_cycle: str,
    det_info: dict,
    save_pdf: bool,
):
    apply_monitoring_style()
    utils.logger.debug("...inspecting FT failure rates")
    evt_files_phy = sorted(
        glob.glob(f"{auto_dir_path}/generated/tier/evt/phy/{period}/{run}/*.lh5")
    )

    if not evt_files_phy:
        evt_files_phy = sorted(
            glob.glob(f"{auto_dir_path}/generated/tier/pet/phy/{period}/{run}/*.lh5")
        )

    # energies  = read_as("evt/geds", evt_files_phy, 'ak', field_mask=['energy'])
    ged_pul = read_as(
        "evt/coincident", evt_files_phy, "ak", field_mask=["geds", "puls"]
    )
    forced = read_as(
        "evt/trigger", evt_files_phy, "ak", field_mask=["is_forced", "timestamp"]
    )
    is_bb = read_as(
        "evt/geds/quality",
        evt_files_phy,
        "ak",
        field_mask=["is_bb_like", "is_good_channel"],
    )
    is_dis = read_as(
        "evt/geds/quality/is_not_bb_like",
        evt_files_phy,
        "ak",
        field_mask=["is_delayed_discharge"],
    )
    is_fail = read_as(
        "evt/geds/quality/is_not_bb_like",
        evt_files_phy,
        "ak",
        field_mask=["is_empty_bits", "rawid"],
    )

    # build dataframe for FT FAILING events (vectorized: one count matrix fill
    # instead of a python loop over every event)
    mask = forced.is_forced & ~is_bb.is_bb_like & ~is_dis.is_delayed_discharge
    temp = is_fail.rawid[mask]
    n_events = len(temp)
    flat_ch = ak.to_numpy(ak.flatten(temp))
    channels = np.unique(flat_ch)
    counts = np.zeros((n_events, len(channels)))
    if flat_ch.size:
        event_idx = np.repeat(np.arange(n_events), ak.to_numpy(ak.num(temp)))
        np.add.at(counts, (event_idx, np.searchsorted(channels, flat_ch)), 1)
    y = {ch: counts[:, j] for j, ch in enumerate(channels)}
    y["timestamp"] = ak.to_numpy(forced.timestamp[mask])

    df = pd.DataFrame(y)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("timestamp", inplace=True)
    daily_cnt = df.resample("h").sum()

    # Folders
    end_folder = os.path.join(output_folder, period, run, "mtg")
    os.makedirs(end_folder, exist_ok=True)
    shelve_path = os.path.join(end_folder, f"l200-{period}-{run}-phy-monitoring")

    str_counts = {}
    color_cycle = itertools.cycle(plt.cm.tab20.colors)

    # --- all forced triggers (denominator across all strings)
    df_all = pd.DataFrame(
        {"timestamp": ak.to_numpy(forced.timestamp[forced.is_forced])}
    )
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], unit="s")
    df_all.set_index("timestamp", inplace=True)
    total_forced = df_all.resample("h").size()  # counts/hour, all strings
    avg_total_forced_mhz = (total_forced.mean() / 3600) * 1000
    on_mass = 0

    # ONE PERIOD, ALL RUNS
    with shelve.open(shelve_path, "c", protocol=pickle.HIGHEST_PROTOCOL) as shelf:
        # --- Per-string plots ---
        for string, det_list in det_info["str_chns"].items():
            fig, ax = plt.subplots(figsize=(12, 6))

            string_counts = None
            string_mass = 0

            for det in det_list:
                if not det_info["detectors"][det]["processable"]:
                    continue
                ch = det_info["detectors"][det]["daq_rawid"]
                if ch not in daily_cnt.columns:
                    continue

                mass = det_info["detectors"][det]["mass_in_kg"]
                if det_info["detectors"][det]["usability"] == "on":
                    on_mass += mass

                hourly_rate = daily_cnt[ch] / 3600 * 1000 / mass
                color = next(color_cycle)
                hourly_rate.plot(ax=ax, drawstyle="steps-mid", label=det, color=color)

                det_counts = daily_cnt[ch]

                string_counts = (
                    det_counts
                    if string_counts is None
                    else string_counts.add(det_counts, fill_value=0)
                )

                string_mass += mass

            if string_counts is not None and string_mass > 0:
                str_counts[string] = string_counts / 3600 * 1000 / string_mass
            else:
                str_counts[string] = None

            m2p = partial(mhz_to_percent, avg_total_forced_mhz=avg_total_forced_mhz)
            p2m = partial(percent_to_mhz, avg_total_forced_mhz=avg_total_forced_mhz)
            secax = ax.secondary_yaxis("right", functions=(m2p, p2m))
            secax.set_ylabel("FT failure fraction (%)")

            ax.set_ylabel("Normalized FT failure rate (mHz/kg)")
            ax.legend(
                ncol=2,
                fontsize="small",
                loc="upper left",
                title=f"Last cycle: {last_cycle}",
            )
            ax.grid(False)
            fig.suptitle(f"{period} - {run} - string {string}")
            fig.tight_layout()

            if save_pdf:
                pdf_folder = os.path.join(
                    output_folder, period, run, "mtg/pdf", f"st{string}"
                )
                os.makedirs(pdf_folder, exist_ok=True)
                plt.savefig(
                    os.path.join(
                        pdf_folder, f"{period}_{run}_string{string}_FT_failure.pdf"
                    ),
                    bbox_inches="tight",
                )

            shelf[f"{period}_{run}_string{string}_FT_failure"] = pickle.dumps(fig)
            plt.close(fig)

        # --- Combined plot of all strings ---
        fig, ax = plt.subplots(figsize=(12, 6))
        color_cycle = itertools.cycle(plt.cm.tab20.colors)
        for string, counts in str_counts.items():
            if counts is not None:
                color = next(color_cycle)
                counts.plot(
                    ax=ax, drawstyle="steps-mid", label=f"String {string}", color=color
                )

        ax.set_ylabel("Normalized FT failure rate (mHz/kg)")
        ax.set_title(f"{period} - {run} - All strings")
        ax.legend(
            ncol=2,
            fontsize="small",
            loc="upper left",
            title=f"Last cycle: {last_cycle}",
        )
        ax.grid(False)
        fig.tight_layout()

        if save_pdf:
            pdf_folder = os.path.join(output_folder, period, run, "mtg/pdf")
            os.makedirs(pdf_folder, exist_ok=True)
            plt.savefig(
                os.path.join(pdf_folder, f"{period}_{run}_all_strings_FT_failure.pdf"),
                bbox_inches="tight",
            )

        shelf[f"{period}_{run}_all_strings_FT_failure"] = pickle.dumps(fig)
        plt.close(fig)

        # --- FT survival fraction ---
        mask_forced = forced.is_forced
        mask_survived = mask_forced & is_bb.is_bb_like & ~is_dis.is_delayed_discharge
        ts_all = pd.to_datetime(forced.timestamp[mask_forced], unit="s")
        ts_survived = pd.to_datetime(forced.timestamp[mask_survived], unit="s")
        df_all = pd.DataFrame({"count": 1}, index=ts_all)
        df_survived = pd.DataFrame({"count": 1}, index=ts_survived)
        total_forced = df_all.resample("h").sum()["count"]
        surviving = df_survived.resample("h").sum()["count"]
        surviving_frac = surviving / total_forced * 100

        fig, ax = plt.subplots(figsize=(12, 6))
        surviving_frac.plot(ax=ax, drawstyle="steps-mid", color="red")
        ax.set_ylabel("FT surviving events (%)")
        ax.set_title(f"{period} - All strings combined")
        ax.grid(False)
        fig.tight_layout()

        if save_pdf:
            pdf_folder = os.path.join(output_folder, period, run, "mtg/pdf")
            os.makedirs(pdf_folder, exist_ok=True)
            plt.savefig(
                os.path.join(pdf_folder, f"{period}_{run}_all_strings_FT_SF.pdf"),
                bbox_inches="tight",
            )

        shelf[f"{period}_{run}_all_strings_FT_SF"] = pickle.dumps(fig)
        plt.close(fig)

        # --- Event rates ---
        fig, ax = plt.subplots(figsize=(10, 3.5))

        # base sample
        base = (
            ged_pul.geds
            & ~ged_pul.puls
            & ~forced.is_forced
            & ~is_dis.is_delayed_discharge
        )

        ser = pd.to_datetime(
            forced.timestamp[ged_pul.geds & ~ged_pul.puls & ~forced.is_forced], unit="s"
        )
        ser_dis = pd.to_datetime(
            forced.timestamp[
                ged_pul.geds
                & ~ged_pul.puls
                & ~forced.is_forced
                & is_dis.is_delayed_discharge
            ],
            unit="s",
        )

        ser_pass = pd.to_datetime(forced.timestamp[base & is_bb.is_bb_like], unit="s")
        ser_fail = pd.to_datetime(forced.timestamp[base & ~is_bb.is_bb_like], unit="s")

        for s, label, color in [
            (ser, "All events", "dimgrey"),
            (ser_dis, "Delayed discharges", "darkorange"),
            (ser_fail, "Failing QC", "crimson"),
            (ser_pass, "Surviving QC", "dodgerblue"),
        ]:
            if s.empty:
                continue
            freq, bin_edges = np.histogram(
                s, bins=pd.date_range(start=s.min(), end=s.max(), freq="h")
            )
            ax.stairs(freq / 3600 * 1000 / on_mass, bin_edges, label=label, color=color)

        ax.set_ylabel("Hourly rate normalized by ON mass (mHz/kg)")
        ax.legend(
            title=f"Last cycle: {last_cycle}\nON mass = {on_mass:.1f} kg",
            loc="upper right",
        )
        ax.grid(False)
        fig.tight_layout()

        if save_pdf:
            pdf_folder = os.path.join(output_folder, period, run, "mtg/pdf")
            os.makedirs(pdf_folder, exist_ok=True)
            plt.savefig(
                os.path.join(pdf_folder, f"{period}_{run}_event_rate_qc.pdf"),
                bbox_inches="tight",
            )
        shelf[f"{period}_{run}_event_rate_qc"] = pickle.dumps(fig)
        plt.close(fig)

        # --- Dead time from discharge windows ---
        mask_puls = ged_pul.puls
        mask_puls_no_dis = ged_pul.puls & ~is_dis.is_delayed_discharge

        length = len(ak.flatten(ak.where(mask_puls)))
        length_no_dis = len(ak.flatten(ak.where(mask_puls_no_dis)))

        # pulser period is assumed to be of 20 s
        livetime_total = length * 20
        livetime_no_dis = length_no_dis * 20

        dead_time_s = livetime_total - livetime_no_dis
        dead_time_pct = (
            (dead_time_s / livetime_total * 100) if livetime_total > 0 else 0.0
        )

        shelf[f"{period}_{run}_dead_time_pct"] = dead_time_pct
        shelf[f"{period}_{run}_dead_time_s"] = dead_time_s
        # the shelve is also the only carrier of these two scalars today, and
        # qc_average needs them; publish them as data so that dependency does
        # not run through a pickled-figure store
        write_dead_time(output_folder, period, run, dead_time_s, dead_time_pct)

        utils.logger.info(
            f"...dead time from discharges: {dead_time_s:.1f} s ({dead_time_pct:.4f} %)"
        )


def compute_detector_summary(results: dict, det_info: dict, pars: dict) -> pd.DataFrame:
    """Per-detector summary of a monitoring parameter (the box-plot data).

    One row per detector: the mean/std/min/max of its values over the run, its
    Qbb resolution from the calibration pars, and its position and usability
    from the channel map. No matplotlib involved, so the numbers can be
    written to the contract and re-read without a figure.
    """
    detectors = det_info["detectors"]
    rows = []
    for ged, item in results.items():
        if ged not in detectors:
            continue
        meta_info = detectors[ged]

        if item is None or len(item) == 0:
            mean = std = min_val = max_val = np.nan
        else:
            mean = np.nanmean(item)
            std = np.nanstd(item)
            min_val = np.nanmin(item)
            max_val = np.nanmax(item)
        try:
            fwhm = pars[ged]["results"]["ecal"]["cuspEmax_ctc_cal"]["eres_linear"][
                "Qbb_fwhm_in_kev"
            ]
        except (KeyError, TypeError):
            fwhm = np.nan

        rows.append(
            {
                "ged": ged,
                "string": meta_info["string"],
                "pos": meta_info["position"],
                "mean": mean,
                "std": std,
                "min": min_val,
                "max": max_val,
                "fwhm": fwhm,
                "usability": meta_info.get("usability", None),
            }
        )
    return pd.DataFrame(rows)


def write_detector_summary(
    output_folder: str,
    period: str,
    run: str,
    metric: str,
    frame: pd.DataFrame,
    data_type: str = "phy",
) -> str | None:
    """Write a per-detector summary table into the period contract file."""
    if frame is None or frame.empty:
        return None
    path = period_contract_path(output_folder, period, data_type)
    contract_writer.write_frame(path, f"detector_summary/{metric}/{run}", frame)
    return path


def box_summary_plot(
    period: str,
    run: str,
    pars: dict,
    det_info: dict,
    results: dict,
    last_cycle: str,
    info: dict,
    output_dir: str,
    data_type: str,
    save_pdf: bool,
    run_to_apply=None,
):
    """
    Box plot summary for FEP gain variations for multiple detectors.

    Parameters
    ----------
    period : str
        Period to inspect.
    run : str
        Run to inspect.
    pars : dict
        Calibration results for each detector.
    det_info : dict
        Dictionary with channel names, IDs, and mapping to string and position.
    results : dict
        Dictionary with arrays values (per detector); None if invalid.
    last_cycle : str
        Last cycle of the inspect list; format: YYYYMMDDThhmmssZ.
    info : dict
        Dictionary containing info on a parameter basis (eg label name, file title, colours, limits, ...).
    output_dir : str
        Output folder for saving plots and shelve data.
    data_type : str
        Type of data, either 'cal' or 'phy'.
    save_pdf : bool
        If True, save the summary plot as a PDF.
    run_to_apply :
        Run to apply (eg see ssc data).
    """
    apply_monitoring_style()
    utils.logger.debug("...making summary box plots for %s", info["title"])
    df_plot = compute_detector_summary(results, det_info, pars)
    write_detector_summary(
        output_dir, period, run, info["title"], df_plot, data_type=data_type
    )
    if df_plot.empty:
        raise errors.DataError(
            f"box_summary_plot: no detector results to plot for '{info['title']}' "
            "(empty or missing input data)"
        )
    # sort by string, and then position
    df = df_plot.sort_values(["string", "pos"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))

    if info["title"] in ["FEP_gain_stab", "pulser_stab", "baseln_stab"]:
        plt.axhline(0, color="gray", lw=0.5)

    if not df["fwhm"].isna().all():
        fwhm_label = (
            r"$\pm$FWHM/2"
            if info["title"] == "FEP_gain_stab"
            else r"$\pm$FWHM (threshold)"
        )
        ax.bar(
            x,
            df["fwhm"],
            bottom=-df["fwhm"] / 2,
            width=0.4,
            color="orange",
            alpha=0.2,
            label=fwhm_label,
        )

    ax.bar(
        x,
        2 * df["std"],  # total height = twice 1 std
        bottom=df["mean"] - df["std"],  # center bar on mean
        width=0.6,
        color="skyblue",
        alpha=0.7,
        label="±1σ",
    )

    ax.scatter(x, df["mean"], color="black", zorder=3, label="Mean")

    ax.errorbar(
        x,
        df["mean"],
        yerr=[df["mean"] - df["min"], df["max"] - df["mean"]],
        fmt="none",
        ecolor="#0266c9" if info["title"] != "FEP_gain_stab" else "red",
        capsize=4,
        label="Min/Max",
    )

    ax.set_xticks(x)
    xtick_labels = ax.set_xticklabels(df["ged"], rotation=90)
    for i, label in enumerate(xtick_labels):
        if df.iloc[i]["usability"] in ["off", "false", False]:
            label.set_color("red")
        if df.iloc[i]["usability"] in ["ac"]:
            label.set_color("darkorange")

    ax.axvline(-0.5, color="gray", ls="--", alpha=0.5)
    ymin, ymax = ax.get_ylim()
    label_y = ymin * (ymax / ymin) ** 0.05 if ymin > 0 else -4
    label_y = label_y if info["title"] != "baseln_spike" else 1
    unique_strings = df["string"].unique()
    for s in unique_strings:
        idx = df.index[df["string"] == s]
        left, right = idx.min(), idx.max()
        ax.axvline(right + 0.5, color="gray", ls="--", alpha=0.5)
        ax.text(left, label_y, f"String {s}", rotation=90)

    ax.set_ylabel(info["ylabel"])
    ax.set_title(f"{period} {run}")

    if info["title"] in ["baseln_stab"]:
        ax.axhline(
            -10,
            ls="--",
            color="black",
            label=r"$\pm$" + f"{info['limits'][1]}% threshold",
        )
        ax.axhline(10, ls="--", color="black")
        ax.axhspan(10, 500, color="gray", alpha=0.25)
        ax.axhspan(-10, -500, color="gray", alpha=0.25)
    if info["title"] in ["baseln_spike"]:
        ax.axhline(
            50, ls="--", color="black", label=f"{info['limits'][1]} ADC upper threshold"
        )
        ax.axhspan(50, 500, color="gray", alpha=0.25)

    if info["title"] == "FEP_gain_stab":
        ax.axhline(-2, ls="--", color="black", label=r"$\pm$2 keV threshold")
        ax.axhline(2, ls="--", color="black")
        ax.axhspan(2, 500, color="gray", alpha=0.25)
        ax.axhspan(-2, -500, color="gray", alpha=0.25)
        plt.ylim(-6, 6)
    if info["title"] == "pulser_stab":
        plt.ylim(-6, 6)
    if info["title"] in ["baseln_stab"]:
        plt.ylim(-20, 20)
    if info["title"] in ["baseln_spike"]:
        plt.ylim(0, 100)

    # Create custom legend entries for usability colors
    legend_patches = []
    handles, labels = ax.get_legend_handles_labels()
    legend_patches.append(Patch(color="red", label="Usability: off"))
    legend_patches.append(Patch(color="darkorange", label="Usability: ac"))
    all_handles = handles + legend_patches
    plot_title = f"Last cycle: {last_cycle}" if last_cycle is not None else None
    ax.legend(handles=all_handles, loc="upper right", title=plot_title)
    ax.grid(False)
    plt.tight_layout()

    if save_pdf:
        pdf_folder = os.path.join(output_dir, f"{period}/{run}/mtg/pdf")
        os.makedirs(pdf_folder, exist_ok=True)
        plt.savefig(
            os.path.join(
                pdf_folder,
                f"{period}_{run}_{info['title']}.pdf",
            ),
            bbox_inches="tight",
        )

    # serialize+plot in a shelve object
    serialized_plot = pickle.dumps(fig)
    with shelve.open(
        os.path.join(
            output_dir,
            period,
            run,
            f"mtg/l200-{period}-{run}-{data_type}-monitoring",
        ),
        "c",
        protocol=pickle.HIGHEST_PROTOCOL,
    ) as shelf:
        shelf[f"{period}_{run}_{info['title']}"] = serialized_plot

    plt.close()


def qc_average(
    auto_dir_path: str,
    output_folder: str,
    det_info: dict,
    period: str,
    run: str,
    last_cycle: str,
    save_pdf: bool,
    pars_to_inspect: list | None = None,
):
    """
    Evaluate the average rate of passing quality cuts for a given run and period across the whole array for different QC flags.

    Parameters
    ----------
    auto_dir_path : str
        Path to tmp-auto public data files (eg /data2/public/prodenv/prod-blind/tmp-auto).
    output_folder : str
        Path to generated monitoring hdf files.
    det_info : dict
        Dictionary with channel names, IDs, and mapping to string and position.
    period : str
        Period to inspect.
    run : str
        Run under inspection.
    last_cycle : str
        Last cycle of the inspect list; format: YYYYMMDDThhmmssZ.
    save_pdf : bool
        True if you want to save pdf files too; default: False.
    pars_to_inspect : list
        List of parameters (boolean flags) to inspect.
    """
    apply_monitoring_style()
    if pars_to_inspect is None:
        pars_to_inspect = [
            "IsHighlyPositivePolarityCandidate",
            "IsValidBlSlope",
            "IsValidBlSlopeRms",
            "IsValidBlPolyRms",
            "IsValidTailRms",
            "IsNotNoiseBurst",
            "IsValidCuspemin",
            "IsValidCuspemax",
            "IsValidTrapTpmax",
            "IsLowCuspemax",
            "IsDischarge",
            "IsSaturated",
        ]

    my_file = os.path.join(
        output_folder, f"{period}/{run}/l200-{period}-{run}-phy-geds.hdf"
    )
    detectors = det_info["detectors"]
    str_chns = det_info["str_chns"]
    utils.logger.debug("...inspecting QC average values")
    if not os.path.exists(my_file):
        utils.logger.warning(f"...file not found: {my_file}. Return!")
        return

    end_folder = os.path.join(
        output_folder,
        period,
        run,
        "mtg",
    )
    os.makedirs(end_folder, exist_ok=True)
    shelve_path = os.path.join(
        end_folder,
        f"l200-{period}-{run}-phy-monitoring",
    )

    usability_map_file = os.path.join(
        output_folder,
        period,
        run,
        f"l200-{period}-{run}-qcp_summary.yaml",
    )
    output = utils.load_yaml_or_default(usability_map_file, detectors)

    with (
        shelve.open(shelve_path, "c", protocol=pickle.HIGHEST_PROTOCOL) as shelf,
        pd.HDFStore(my_file, "r") as store,
    ):
        for par in pars_to_inspect:
            key = f"/IsPhysics_{par}"
            if key not in store:
                utils.logger.debug("...skipping %s (not found in HDF)", par)
                continue

            geds_df_abs = store[key]
            geds_df_abs = filter_series_by_ignore_keys(
                geds_df_abs, utils.IGNORE_KEYS, period
            )

            # time span
            time_min, time_max = geds_df_abs.index.min(), geds_df_abs.index.max()
            diff = (time_max - time_min).total_seconds()

            # rates in mHz
            rates = geds_df_abs.sum(axis=0) / diff * 1000

            fig, ax = plt.subplots(figsize=(12, 4), sharex=True)
            ax.set_title(f"period: {period} - run: {run} - passing {par}")
            dt_condition = False
            if par == "IsDischarge":
                dead_time = read_dead_time(output_folder, period, run)
                dt = (
                    dead_time["dead_time_pct"]
                    if dead_time is not None
                    else shelf.get(f"{period}_{run}_dead_time_pct", None)
                )
                if dt is None:
                    # qc_and_evt_summary_plots has not run for this run; say so
                    # rather than crashing on the format/comparison below
                    utils.logger.warning(
                        "\033[93mno dead time recorded for %s-%s; "
                        "plotting %s without it\033[0m",
                        period,
                        run,
                        par,
                    )
                    ax.set_title(
                        f"period: {period} - run: {run} - passing {par} - "
                        "tot dead time unavailable"
                    )
                else:
                    ax.set_title(
                        f"period: {period} - run: {run} - passing {par} - tot dead time {dt:.3f}%"
                    )
                    dt_condition = bool(
                        dt > utils.MTG_PLOT_INFO["tot_discharge_dead_time"]["limits"][1]
                    )

            x_labels, xs, ys = [], [], []
            string_indices = {}
            ct = -1

            for string, det_list in str_chns.items():
                indices = []

                for det_name in det_list:
                    det = detectors[det_name]
                    rawid = det["daq_rawid"]

                    ct += 1
                    x_labels.append(det_name)
                    indices.append(ct)
                    if rawid not in rates:
                        utils.logger.debug(
                            f"{det_name} ({rawid}) missing in dataframe for {par}"
                        )
                        continue

                    ys.append(rates[rawid])
                    xs.append(ct)

                    if par in ["IsDischarge", "IsSaturated"]:
                        condition = bool(
                            (rates[rawid] > utils.MTG_PLOT_INFO[par]["limits"][1]).any()
                        )  # no lower limit for rates
                        utils.update_evaluation_in_memory(
                            output,
                            det_name,
                            "phy",
                            utils.MTG_PLOT_INFO[par]["title"],
                            not condition,
                        )

                        utils.update_evaluation_in_memory(
                            output,
                            det_name,
                            "phy",
                            utils.MTG_PLOT_INFO["tot_discharge_dead_time"]["title"],
                            not dt_condition,
                        )

                string_indices[string] = indices

            ax.scatter(xs, ys, color="dodgerblue", marker="o")

            ax.set_ylabel(f"Average rate {par}=True (mHz)")
            ax.set_yscale("log")
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=90)
            ax.grid(False)

            ymin, ymax = ax.get_ylim()
            label_y = ymin * (ymax / ymin) ** 0.05 if ymin > 0 else 0.1
            for string, indices in string_indices.items():
                left, right = min(indices), max(indices)
                if string == 1:
                    ax.axvline(left - 0.5, ls="--", color="k", alpha=0.5)
                ax.axvline(right + 0.5, ls="--", color="k", alpha=0.5)
                ax.text(
                    left,
                    label_y,
                    f"String {string}",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            if par in ["IsDischarge", "IsSaturated"]:
                upper_limit = (
                    ax.get_ylim()[1]
                    if ax.get_ylim()[1] > 5
                    else utils.MTG_PLOT_INFO[par]["limits"][1] * 1.1
                )
                ax.axhspan(
                    utils.MTG_PLOT_INFO[par]["limits"][1],
                    upper_limit,
                    color="gray",
                    alpha=0.25,
                )
                ax.axhline(
                    utils.MTG_PLOT_INFO[par]["limits"][1],
                    ls="--",
                    color="black",
                    label=f"{utils.MTG_PLOT_INFO[par]['limits'][1]} mHz upper threshold",
                )

            ax.legend(title=f"Last cycle: {last_cycle}")
            plt.tight_layout()

            if par in ["IsDischarge", "IsSaturated"]:
                plot_name = f"{period}_{run}_{utils.MTG_PLOT_INFO[par]['title']}_avg"
            else:
                plot_name = f"{period}_{run}_{par}_avg"

            if save_pdf:
                pdf_dir = os.path.join(end_folder, "pdf")
                os.makedirs(pdf_dir, exist_ok=True)
                pdf_name = os.path.join(pdf_dir, f"{plot_name}.pdf")
                fig.savefig(pdf_name)

            shelf[plot_name] = pickle.dumps(fig)
            plt.close(fig)

    with open(usability_map_file, "w") as f:
        yaml.dump(output, f)


def qc_time_series(
    auto_dir_path: str,
    output_folder: str,
    det_info: dict,
    period: str,
    run: str,
    last_cycle: str,
    save_pdf: bool,
    pars_to_inspect: list | None = None,
):
    """
    Evaluate rate over time of passing quality cuts for a given run and period across the whole array for different QC flags.

    Parameters
    ----------
    auto_dir_path : str
        Path to tmp-auto public data files (eg /data2/public/prodenv/prod-blind/tmp-auto).
    output_folder : str
        Path to generated monitoring hdf files.
    det_info : dict
        Dictionary with channel names, IDs, and mapping to string and position.
    period : str
        Period to inspect.
    run : str
        Run under inspection.
    last_cycle : str
        Last cycle of the inspect list; format: YYYYMMDDThhmmssZ.
    save_pdf : bool
        True if you want to save pdf files too; default: False.
    pars_to_inspect : list
        List of parameters (boolean flags) to inspect.
    """
    apply_monitoring_style()
    if pars_to_inspect is None:
        pars_to_inspect = [
            "IsHighlyPositivePolarityCandidate",
            "IsValidBlSlope",
            "IsValidBlSlopeRms",
            "IsValidBlPolyRms",
            "IsValidTailRms",
            "IsNotNoiseBurst",
            "IsValidCuspemin",
            "IsValidCuspemax",
            "IsValidTrapTpmax",
            "IsLowCuspemax",
            "IsDischarge",
            "IsSaturated",
        ]
    my_file = os.path.join(
        output_folder, f"{period}/{run}/l200-{period}-{run}-phy-geds.hdf"
    )
    detectors = det_info["detectors"]
    str_chns = det_info["str_chns"]
    utils.logger.debug("...inspecting QC time series")
    if not os.path.exists(my_file):
        utils.logger.warning(f"...file not found: {my_file}. Return!")
        return

    end_folder = os.path.join(
        output_folder,
        period,
        run,
        "mtg",
    )
    os.makedirs(end_folder, exist_ok=True)
    shelve_path = os.path.join(
        end_folder,
        f"l200-{period}-{run}-phy-monitoring",
    )

    color_cycle = itertools.cycle(plt.cm.tab20.colors)

    with (
        shelve.open(shelve_path, "c", protocol=pickle.HIGHEST_PROTOCOL) as shelf,
        pd.HDFStore(my_file, "r") as store,
    ):

        for par in pars_to_inspect:
            key = f"/IsPhysics_{par}"
            if key not in store:
                utils.logger.debug("...skipping %s (not found in HDF)", key)
                continue

            geds_df_abs = store[key]
            geds_df_abs = filter_series_by_ignore_keys(
                geds_df_abs, utils.IGNORE_KEYS, period
            )

            for string, channel_list in str_chns.items():
                fig, ax = plt.subplots(figsize=(12, 4))

                for channel_name in channel_list:
                    det = detectors[channel_name]
                    rawid = det["daq_rawid"]
                    pos = det["position"]

                    if rawid not in geds_df_abs.columns:
                        utils.logger.debug(
                            f"{channel_name} ({rawid}) missing in dataframe for {par}"
                        )
                        continue

                    data = geds_df_abs[rawid].copy()
                    true_count = data.sum()
                    time_min, time_max = data.index.min(), data.index.max()
                    diff = (time_max - time_min).total_seconds()

                    true_rate_mHz = round(true_count / diff * 1000, 2)
                    hourly_rate = data.resample("1h").sum() / 3600 * 1000

                    color = next(color_cycle)
                    hourly_rate.plot(
                        ax=ax,
                        drawstyle="steps-mid",
                        label=f"{channel_name} - pos {pos} - {true_rate_mHz} mHz",
                        color=color,
                    )

                ax.grid(False)
                ax.set_ylabel(f"{period} {run} - 1h {par} rate (mHz)")
                fig.suptitle(f"{period} {run} - String: {string}")
                if par in ["IsDischarge", "IsSaturated"]:
                    upper_limit = (
                        ax.get_ylim()[1]
                        if ax.get_ylim()[1] > 5
                        else utils.MTG_PLOT_INFO[par]["limits"][1] * 1.1
                    )
                    ax.axhspan(
                        utils.MTG_PLOT_INFO[par]["limits"][1],
                        upper_limit,
                        color="gray",
                        alpha=0.25,
                    )
                    ax.axhline(
                        utils.MTG_PLOT_INFO[par]["limits"][1],
                        ls="--",
                        color="black",
                        label=f"{utils.MTG_PLOT_INFO[par]['limits'][1]} mHz upper threshold",
                    )
                ax.legend(title=f"Last cycle: {last_cycle}")
                plt.tight_layout()

                if par in ["IsDischarge", "IsSaturated"]:
                    plot_name = f"{period}_{run}_string{string}_{utils.MTG_PLOT_INFO[par]['title']}"  # "_rate" already in the title
                else:
                    plot_name = f"{period}_{run}_string{string}_{par}_rate"

                if save_pdf:
                    pdf_dir = os.path.join(end_folder, "pdf", f"st{string}")
                    os.makedirs(pdf_dir, exist_ok=True)
                    pdf_name = os.path.join(pdf_dir, f"{plot_name}.pdf")
                    fig.savefig(pdf_name)

                # serialize+save plot
                shelf[plot_name] = pickle.dumps(fig)
                plt.close(fig)


def build_new_files(generated_path: str, period: str, run: str, data_type="phy"):
    """
    Generate and store resampled HDF files for a given data run and extract summary info.

    This function:

      - loads the original `.hdf` file for the specified `period` and `run`
      - extracts available keys from the HDF file
      - resamples all applicable time series data into multiple time intervals (10min, 60min)
      - stores each resampled dataset into a separate HDF file
      - extracts metadata from the 'info' key and saves it as a .yaml file

    Parameters
    ----------
    generated_path : str
        Root directory where the data is stored and where new files will be written.
    period : str
        Period (e.g. 'p03') used to construct paths.
    run : str
        Run (e.g. 'r001') used to construct paths.
    data_type : str
        Data type to load; default: 'phy'.
    """
    data_file = os.path.join(
        generated_path,
        "generated/plt/hit",
        data_type,
        period,
        run,
        f"l200-{period}-{run}-{data_type}-geds.hdf",
    )

    if not os.path.exists(data_file):
        utils.logger.debug(f"File not found: {data_file}. Exit here.")
        raise errors.DataError("build_new_files failed (see log for details)")

    with h5py.File(data_file, "r") as f:
        my_keys = list(f.keys())

    info_dict = {"keys": my_keys}

    resampling_times = ["10min", "60min"]

    for idx, resample_unit in enumerate(resampling_times):
        new_file = os.path.join(
            generated_path,
            "generated/plt/hit",
            data_type,
            period,
            run,
            f"l200-{period}-{run}-{data_type}-geds-res_{resample_unit}.hdf",
        )
        # remove it if already exists so we can start again to append resampled data
        if os.path.exists(new_file):
            os.remove(new_file)

        for k in my_keys:
            if "info" in k:
                # do it once
                if idx == 0:
                    original_df = pd.read_hdf(data_file, key=k)
                    original_df = original_df.astype(str)
                    info_dict.update(
                        {
                            k: {
                                "subsystem": original_df.loc["subsystem", "Value"],
                                "unit": original_df.loc["unit", "Value"],
                                "label": original_df.loc["label", "Value"],
                                "event_type": original_df.loc["event_type", "Value"],
                                "lower_lim_var": original_df.loc[
                                    "lower_lim_var", "Value"
                                ],
                                "upper_lim_var": original_df.loc[
                                    "upper_lim_var", "Value"
                                ],
                                "lower_lim_abs": original_df.loc[
                                    "lower_lim_abs", "Value"
                                ],
                                "upper_lim_abs": original_df.loc[
                                    "upper_lim_abs", "Value"
                                ],
                            }
                        }
                    )
                continue

            original_df = pd.read_hdf(data_file, key=k)

            # mean dataframe is kept
            if "_mean" in k:
                original_df.to_hdf(new_file, key=k, mode="a")
                continue

            original_df.index = pd.to_datetime(original_df.index)
            # resample
            resampled_df = original_df.resample(resample_unit).mean()
            # substitute the original df with the resampled one
            original_df = resampled_df
            # append resampled data to the new file
            resampled_df.to_hdf(new_file, key=k, mode="a")

        if idx == 0:
            json_output = os.path.join(
                generated_path,
                "generated/plt/hit",
                data_type,
                period,
                run,
                f"l200-{period}-{run}-{data_type}-geds-info.yaml",
            )
            with open(json_output, "w") as file:
                json.dump(info_dict, file, indent=4)


def plot_time_series(
    auto_dir_path: str,
    phy_mtg_data: str,
    output_folder: str,
    data_type: str,
    period: str,
    runs: list,
    current_run: str,
    det_info: dict,
    save_pdf: bool,
    escale_val: float,
    last_checked: float | None,
    last_cycle: str,
    partition: bool,
    quadratic: bool,
    zoom: bool,
):
    """
    Generate and save time-series plots of calibration and monitoring data for germanium detectors across multiple runs.

    This function collects physics and calibration data from HDF5 monitoring files and visualizes stability over time.
    Channels with no pulser entries are automatically skipped.
    Corrections are applied to the gain if pulser data is available ('GED corrected'), otherwise uncorrected data is plotted.
    The plots are saved as pickled objects for later retrieval (eg. in the online Dashboard) and optionally as PDFs:

    - plots saved in shelve database files under ``<output_folder>/<period>/mtg/l200-<period>-phy-monitoring``;
    - if `save_pdf=True`, PDF copies saved under ``<output_folder>/<period>/mtg/pdf/st<string>/``.

    Parameters
    ----------
    auto_dir_path : str
        Path to tmp-auto public data files (eg /data2/public/prodenv/prod-blind/tmp-auto).
    phy_mtg_data : str
        Path to generated monitoring hdf files.
    output_folder : str
        Path to output folder.
    period : str
        Period to inspect.
    runs : list
        Available runs to inspect for a given period.
    current_run : str
        Run under inspection.
    det_info : dict
        Dictionary containing detector metadata.
    save_pdf : bool
        True if you want to save pdf files too; default: False.
    escale_val : float
        Energy scale at which evaluating the gain differences; default: 2039 keV (76Ge Qbb).
    last_checked : float | None
        Timestamp of the last check.
    last_cycle : str
        Last cycle of the inspect list; format: YYYYMMDDThhmmssZ.
    partition : bool
        False if not partition data; default: False.
    quadratic : bool
        True if you want to plot the quadratic resolution too; default: False.
    zoom : bool
        True to zoom over y axis; default: False.
    """
    apply_monitoring_style()
    avail_runs = []
    for entry in runs:
        new_entry = entry.replace(",", "").replace("[", "").replace("]", "")
        avail_runs.append(new_entry)
    dataset = {period: avail_runs}
    period_list = list(dataset.keys())
    xlim_idx = 1
    fit_flag = "quadratic" if quadratic is True else "linear"

    detectors = det_info["detectors"]
    str_chns = det_info["str_chns"]
    usability_map_file = os.path.join(
        output_folder,
        period,
        current_run,
        f"l200-{period}-{current_run}-qcp_summary.yaml",
    )
    output = utils.load_yaml_or_default(usability_map_file, detectors)

    # skip detectors with no pulser entries
    no_puls_dets = utils.NO_PULS_DETS
    flag_expr = " or ".join(
        f'(channel == "{channel}" and period in {periods})'
        for channel, periods in no_puls_dets.items()
    )

    # gain over period
    results = {}
    for index_i in range(len(period_list)):
        period = period_list[index_i]
        run_list = dataset[period]

        (
            geds_df_cuspEmax_abs,
            geds_df_cuspEmax_abs_corr,
            puls_df_cuspEmax_abs,
        ) = get_dfs(phy_mtg_data, period, run_list, "Trapemax")
        geds_df_trapTmax, geds_df_tp0est, puls_df_trapTmax, puls_df_tp0est = (
            get_traptmax_tp0est(phy_mtg_data, period, run_list)
        )

        if (
            geds_df_cuspEmax_abs is None
            or geds_df_cuspEmax_abs_corr is None
            # no need to exit if pulser01ana does not exits, handled it properly now
            # or puls_df_cuspEmax_abs is None
        ):
            utils.logger.debug("Dataframes are None for %s!", period)
            continue

        # check if geds df is empty; if pulser is, means we do not apply any correction
        # (and thus geds_corr is also empty - the code will handle the case)
        if (
            geds_df_cuspEmax_abs.empty
            # or geds_df_cuspEmax_abs_corr.empty
            # or puls_df_cuspEmax_abs.empty
        ):
            utils.logger.debug("Dataframes are empty for %s!", period)
            continue

        dfs = [
            geds_df_cuspEmax_abs,
            geds_df_cuspEmax_abs_corr,
            puls_df_cuspEmax_abs,
            geds_df_trapTmax,
            geds_df_tp0est,
            puls_df_trapTmax,
            puls_df_tp0est,
        ]

        end_folder = os.path.join(
            output_folder,
            period,
            "mtg",
        )
        os.makedirs(end_folder, exist_ok=True)
        shelve_path = os.path.join(end_folder, f"l200-{period}-phy-monitoring")
        utils.logger.debug(f"...inspecting gain over {period}")
        with shelve.open(shelve_path, "c", protocol=pickle.HIGHEST_PROTOCOL) as shelf:
            for plot_type in ["corr", "uncorr"]:
                for string, det_list in str_chns.items():
                    for channel_name in det_list:
                        channel = detectors[channel_name]["channel_str"]
                        rawid = detectors[channel_name]["daq_rawid"]
                        pos = detectors[channel_name]["position"]

                        resampling_time = "1h"  # if len(runs)>1 else "10T"

                        rawid = np.int64(rawid)
                        if rawid not in set(dfs[0].columns):
                            utils.logger.debug(
                                f"{channel} is not present in the dataframe!"
                            )
                            continue

                        pulser_data = get_pulser_data(
                            resampling_time,
                            period,
                            dfs,
                            rawid,
                            escale=escale_val,
                            variations=True,
                        )

                        fig, ax = plt.subplots(figsize=(12, 4))
                        pars_data = get_calib_pars(
                            auto_dir_path,
                            period,
                            run_list,
                            [channel, channel_name],
                            partition,
                            data_type,
                            escale=escale_val,
                            fit=fit_flag,
                        )

                        t0 = pars_data["run_start"]
                        if not eval(flag_expr):
                            # PULS01ANA has a signal - we can correct GEDS energies for it!
                            if (
                                pulser_data["pul_cusp"]["kevdiff_av"] is not None
                                and plot_type == "corr"
                            ):
                                pul_cusp_av = pulser_data["pul_cusp"][
                                    "kevdiff_av"
                                ].values.astype(float)
                                diff_av = pulser_data["diff"][
                                    "kevdiff_av"
                                ].values.astype(float)
                                diff_std = pulser_data["diff"][
                                    "kevdiff_std"
                                ].values.astype(float)
                                x = pulser_data["diff"]["kevdiff_av"].index.values

                                plt.fill_between(
                                    x,
                                    diff_av - diff_std,
                                    diff_av + diff_std,
                                    color="k",
                                    alpha=0.2,
                                    label=r"±1$\sigma$",
                                )
                                plt.plot(x, pul_cusp_av, "C2", label="PULS01ANA")
                                plt.plot(x, diff_av, "C4", label="GED corrected")
                            else:
                                ged_av = pulser_data["ged"]["kevdiff_av"].values.astype(
                                    float
                                )
                                ged_std = pulser_data["ged"][
                                    "kevdiff_std"
                                ].values.astype(float)
                                x = pulser_data["ged"]["kevdiff_av"].index.values

                                plt.fill_between(
                                    x,
                                    ged_av - ged_std,
                                    ged_av + ged_std,
                                    color="k",
                                    alpha=0.2,
                                    label=r"±1$\sigma$",
                                )
                                plt.plot(
                                    x,
                                    ged_av,
                                    color="dodgerblue",
                                    label="GED uncorrected",
                                )

                        plt.plot(
                            pars_data["run_start"] - pd.Timedelta(hours=5),
                            pars_data["fep_diff"],
                            "kx",
                            label="FEP gain",
                        )
                        plt.plot(
                            pars_data["run_start"] - pd.Timedelta(hours=5),
                            pars_data["cal_const_diff"],
                            "rx",
                            label="cal. const. diff",
                        )

                        for ti in pars_data["run_start"]:
                            plt.axvline(ti, color="dimgrey", ls="--")

                        for i in range(len(t0)):
                            if i == len(pars_data["run_start"]) - 1:
                                plt.plot(
                                    [t0[i], t0[i] + pd.Timedelta(days=7)],
                                    [pars_data["res"][i] / 2, pars_data["res"][i] / 2],
                                    "b-",
                                )
                                plt.plot(
                                    [t0[i], t0[i] + pd.Timedelta(days=7)],
                                    [
                                        -pars_data["res"][i] / 2,
                                        -pars_data["res"][i] / 2,
                                    ],
                                    "b-",
                                )
                                if quadratic:
                                    plt.plot(
                                        [t0[i], t0[i] + pd.Timedelta(days=7)],
                                        [
                                            pars_data["res_quad"][i] / 2,
                                            pars_data["res_quad"][i] / 2,
                                        ],
                                        color="dodgerblue",
                                        linestyle="-",
                                    )
                                    plt.plot(
                                        [t0[i], t0[i] + pd.Timedelta(days=7)],
                                        [
                                            -pars_data["res_quad"][i] / 2,
                                            -pars_data["res_quad"][i] / 2,
                                        ],
                                        color="dodgerblue",
                                        linestyle="-",
                                    )
                            else:
                                plt.plot(
                                    [t0[i], t0[i + 1]],
                                    [pars_data["res"][i] / 2, pars_data["res"][i] / 2],
                                    "b-",
                                )
                                plt.plot(
                                    [t0[i], t0[i + 1]],
                                    [
                                        -pars_data["res"][i] / 2,
                                        -pars_data["res"][i] / 2,
                                    ],
                                    "b-",
                                )
                                if quadratic:
                                    plt.plot(
                                        [t0[i], t0[i + 1]],
                                        [
                                            pars_data["res_quad"][i] / 2,
                                            pars_data["res_quad"][i] / 2,
                                        ],
                                        color="dodgerblue",
                                        linestyle="-",
                                    )
                                    plt.plot(
                                        [t0[i], t0[i + 1]],
                                        [
                                            -pars_data["res_quad"][i] / 2,
                                            -pars_data["res_quad"][i] / 2,
                                        ],
                                        color="dodgerblue",
                                        linestyle="-",
                                    )

                            if str(pars_data["res"][i] / 2 * 1.1) != "nan" and i < len(
                                pars_data["res"]
                            ) - (xlim_idx - 1):
                                plt.text(
                                    t0[i],
                                    pars_data["res"][i] / 2 * 1.1,
                                    "{:.2f}".format(pars_data["res"][i]),
                                    color="b",
                                )

                            if quadratic:
                                if str(
                                    pars_data["res_quad"][i] / 2 * 1.5
                                ) != "nan" and i < len(pars_data["res"]) - (
                                    xlim_idx - 1
                                ):
                                    plt.text(
                                        t0[i],
                                        pars_data["res_quad"][i] / 2 * 1.5,
                                        "{:.2f}".format(pars_data["res_quad"][i]),
                                        color="dodgerblue",
                                    )

                        fig.suptitle(
                            f"period: {period} - string: {string} - position: {pos} - ged: {channel_name}"
                        )
                        plt.ylabel(r"Energy diff / keV")
                        plt.plot(
                            [0, 1],
                            [0, 1],
                            "b",
                            label=r"Q$_{\beta\beta}$ $\pm$FWHM/2 lin. (threshold)",
                        )
                        if quadratic:
                            plt.plot(
                                [1, 2],
                                [1, 2],
                                "dodgerblue",
                                label=r"Q$_{\beta\beta}$ $\pm$FWHM/2 quad. (threshold)",
                            )

                        if zoom:
                            if flag_expr:
                                plt.ylim(-3, 3)
                            else:
                                bound = np.average(
                                    pulser_data["ged"]["cusp_av"].dropna()
                                )
                                plt.ylim(-2.5 * bound, 2.5 * bound)
                        max_date = pulser_data["ged"]["kevdiff_av"].index.max()
                        time_difference = max_date.tz_localize(None) - t0[
                            -xlim_idx
                        ].tz_localize(None)
                        plt.xlim(
                            t0[0] - pd.Timedelta(hours=8),
                            t0[-xlim_idx] + time_difference * 1.5,
                        )  # pd.Timedelta(days=7))# --> change me to resize the width of the last run
                        plt.legend(loc="lower left", title=f"Last cycle: {last_cycle}")
                        plt.tight_layout()

                        if save_pdf:
                            mgt_folder = os.path.join(end_folder, "pdf", f"st{string}")
                            os.makedirs(mgt_folder, exist_ok=True)

                            pdf_name = os.path.join(
                                mgt_folder,
                                f"{period}_string{string}_pos{pos}_{channel_name}_{plot_type}_gain_shift.pdf",
                            )
                            plt.savefig(pdf_name)

                        # serialize+save the plot
                        serialized_plot = pickle.dumps(plt.gcf())
                        shelf[
                            f"{period}_string{string}_pos{pos}_{channel_name}_{plot_type}_gain_shift"
                        ] = serialized_plot
                        plt.close(fig)

                        # structure of pickle files:
                        #  - p08_string1_pos1_V02160A_param
                        #  - p08_string1_pos2_V02160B_param
                        #  - ...

    # parameters (bsln, gain, ...) variations over run
    utils.logger.debug("...inspecting gain/bsln/etc time series")
    info = utils.MTG_PLOT_INFO
    last_checked = None

    for inspected_parameter in [
        "BlStd",
        "TrapemaxCtcCal",
        "Baseline",
        "Trapemax",
    ]:
        escale_par = escale_val if inspected_parameter == "TrapemaxCtcCal" else 1
        results.update({inspected_parameter: {}})

        for index_i in range(len(period_list)):
            period = period_list[index_i]

            (
                geds_df_cuspEmax_abs,
                geds_df_cuspEmax_abs_corr,
                puls_df_cuspEmax_abs,
            ) = get_dfs(phy_mtg_data, period, [current_run], inspected_parameter)
            geds_df_trapTmax, geds_df_tp0est, puls_df_trapTmax, puls_df_tp0est = (
                get_traptmax_tp0est(phy_mtg_data, period, [current_run])
            )

            if (
                geds_df_cuspEmax_abs is None
                or geds_df_cuspEmax_abs_corr is None
                # no need to exit if pulser01ana does not exits, handled it properly now
                # or puls_df_cuspEmax_abs is None
            ):
                utils.logger.debug(
                    "Dataframes are None for %s-%s!", period, current_run
                )
                continue
            if geds_df_cuspEmax_abs.empty:
                utils.logger.debug(
                    "Dataframes are empty for %s-%s!", period, current_run
                )
                continue
            dfs = [
                geds_df_cuspEmax_abs,
                geds_df_cuspEmax_abs_corr,
                puls_df_cuspEmax_abs,
                geds_df_trapTmax,
                geds_df_tp0est,
                puls_df_trapTmax,
                puls_df_tp0est,
            ]

            end_folder = os.path.join(
                output_folder,
                period,
                current_run,
                "mtg",
            )
            os.makedirs(end_folder, exist_ok=True)
            shelve_path = os.path.join(
                end_folder,
                f"l200-{period}-{current_run}-phy-monitoring",
            )
            utils.logger.debug(
                f"...inspecting {info[inspected_parameter]['title']} over {current_run}"
            )

            with shelve.open(
                shelve_path, "c", protocol=pickle.HIGHEST_PROTOCOL
            ) as shelf:
                for string, det_list in str_chns.items():
                    for channel_name in det_list:
                        channel = detectors[channel_name]["channel_str"]
                        rawid = detectors[channel_name]["daq_rawid"]
                        pos = detectors[channel_name]["position"]

                        resampling_time = "1h"
                        rawid = np.int64(rawid)
                        if rawid not in set(dfs[0].columns):
                            utils.logger.debug(
                                f"{channel} is not present in the dataframe!"
                            )
                            continue

                        pulser_data = get_pulser_data(
                            resampling_time,
                            period,
                            dfs,
                            rawid,
                            escale=escale_par,
                            variations=info[inspected_parameter]["percentage"],
                        )

                        fig, ax = plt.subplots(figsize=(12, 4))
                        pars_data = get_calib_pars(
                            auto_dir_path,
                            period,
                            [current_run],
                            [channel, channel_name],
                            partition,
                            data_type,
                            escale=escale_par,
                            fit=fit_flag,
                        )
                        threshold = (
                            [-pars_data["res"][0] / 2, pars_data["res"][0] / 2]
                            if "Trapemax" in inspected_parameter
                            else info[inspected_parameter]["limits"]
                        )

                        t0 = pars_data["run_start"]
                        if not eval(flag_expr):
                            check_kevdiff = None
                            if (
                                info[inspected_parameter]["percentage"] is True
                                and float(escale_par) == 1.0
                            ):
                                check_kevdiff = pulser_data["ged"]["kevdiff_av"] * 100
                            else:
                                check_kevdiff = pulser_data["ged"]["kevdiff_av"]
                            # check threshold and update YAML summary file
                            # (for energy, do it only for TrapemaxCtcCal and not Trapemax at the moment)
                            if inspected_parameter != "Trapemax":
                                utils.check_threshold(
                                    check_kevdiff,
                                    channel_name,
                                    last_checked,
                                    t0,
                                    threshold,
                                    info[inspected_parameter]["title"],
                                    output,
                                    period=period,
                                    run=current_run,
                                )

                            # PULS01ANA has a signal - we can correct GEDS energies for it!
                            # only in the case of energy parameters
                            if (
                                pulser_data["pul_cusp"]["kevdiff_av"] is not None
                                and inspected_parameter == "TrapemaxCtcCal"
                            ):
                                pul_cusp_av = pulser_data["pul_cusp"][
                                    "kevdiff_av"
                                ].values.astype(float)
                                diff_av = pulser_data["diff"][
                                    "kevdiff_av"
                                ].values.astype(float)
                                diff_std = pulser_data["diff"][
                                    "kevdiff_std"
                                ].values.astype(float)
                                x = pulser_data["diff"]["kevdiff_av"].index.values

                                plt.plot(x, pul_cusp_av, "C2", label="PULS01ANA")
                                plt.plot(x, diff_av, "C4", label="GED corrected")
                                plt.fill_between(
                                    x,
                                    diff_av - diff_std,
                                    diff_av + diff_std,
                                    color="k",
                                    alpha=0.2,
                                    label=r"±1$\sigma$",
                                )

                                results[inspected_parameter].update(
                                    {channel_name: pul_cusp_av.values.astype(float)}
                                )
                            # else, no correction is applied
                            else:
                                if (
                                    info[inspected_parameter]["percentage"] is True
                                    and float(escale_par) == 1.0
                                ):
                                    pulser_data["ged"]["kevdiff_av"] *= 100
                                    pulser_data["ged"]["kevdiff_std"] *= 100

                                vals_av = pulser_data["ged"][
                                    "kevdiff_av"
                                ].values.astype(float)
                                vals_std = pulser_data["ged"][
                                    "kevdiff_std"
                                ].values.astype(float)
                                x = pulser_data["ged"]["kevdiff_av"].index.values

                                plt.plot(
                                    x,
                                    vals_av,
                                    color=info[inspected_parameter]["colors"][0],
                                    label="GED uncorrected",
                                )
                                plt.fill_between(
                                    x,
                                    vals_av - vals_std,
                                    vals_av + vals_std,
                                    color="k",
                                    alpha=0.2,
                                    label=r"±1$\sigma$",
                                )

                                results[inspected_parameter].update(
                                    {
                                        channel_name: pulser_data["ged"][
                                            "kevdiff_av"
                                        ].values.astype(float)
                                    }
                                )

                        # plot resolution only for the energy parameters
                        if inspected_parameter == "TrapemaxCtcCal":
                            plt.plot(
                                [t0[0], t0[0] + pd.Timedelta(days=7)],
                                [pars_data["res"][0] / 2, pars_data["res"][0] / 2],
                                color=info[inspected_parameter]["colors"][1],
                                ls="-",
                            )
                            plt.plot(
                                [t0[0], t0[0] + pd.Timedelta(days=7)],
                                [-pars_data["res"][0] / 2, -pars_data["res"][0] / 2],
                                color=info[inspected_parameter]["colors"][1],
                                ls="-",
                            )

                            if str(pars_data["res"][0] / 2 * 1.1) != "nan" and 0 < len(
                                pars_data["res"]
                            ) - (xlim_idx - 1):
                                plt.text(
                                    t0[0],
                                    pars_data["res"][0] / 2 * 1.1,
                                    "{:.2f}".format(pars_data["res"][0]),
                                    color=info[inspected_parameter]["colors"][1],
                                )
                            plt.plot(
                                [0, 1],
                                [0, 1],
                                color=info[inspected_parameter]["colors"][1],
                                label=r"Q$_{\beta\beta}$ $\pm$FWHM/2 lin. (threshold)",
                            )
                        else:
                            if threshold[1] is not None:
                                plt.plot(
                                    [t0[0], t0[0] + pd.Timedelta(days=7)],
                                    [
                                        threshold[1],
                                        threshold[1],
                                    ],
                                    color=info[inspected_parameter]["colors"][1],
                                    ls="-",
                                    label="Threshold",
                                )
                            if threshold[0] is not None:
                                plt.plot(
                                    [t0[0], t0[0] + pd.Timedelta(days=7)],
                                    [
                                        threshold[0],
                                        threshold[0],
                                    ],
                                    color=info[inspected_parameter]["colors"][1],
                                    ls="-",
                                    label=(
                                        None
                                        if threshold[1] is not None
                                        else "Threshold"
                                    ),
                                )

                        plt.ylabel(info[inspected_parameter]["ylabel"])
                        fig.suptitle(
                            f"period: {period} - string: {string} - position: {pos} - ged: {channel_name}"
                        )

                        if zoom is True:
                            bound = np.average(
                                pulser_data["ged"]["kevdiff_std"].dropna()
                            )
                            plt.ylim(-3.5 * bound, 3.5 * bound)

                        max_date = pulser_data["ged"]["kevdiff_av"].index.max()
                        time_difference = max_date.tz_localize(None) - t0[
                            -xlim_idx
                        ].tz_localize(None)
                        plt.xlim(
                            t0[0] - pd.Timedelta(hours=0.5),
                            t0[-xlim_idx] + time_difference * 1.1,
                        )
                        plt.legend(loc="lower left", title=f"Last cycle: {last_cycle}")
                        plt.tight_layout()

                        if save_pdf:
                            mgt_folder = os.path.join(end_folder, "pdf", f"st{string}")
                            os.makedirs(mgt_folder, exist_ok=True)

                            pdf_name = os.path.join(
                                mgt_folder,
                                f"{period}_{current_run}_string{string}_pos{pos}_{channel_name}_{info[inspected_parameter]['title']}.pdf",
                            )
                            plt.savefig(pdf_name)

                        # serialize+save the plot
                        serialized_plot = pickle.dumps(plt.gcf())
                        shelf[
                            f"{period}_{current_run}_string{string}_pos{pos}_{channel_name}_{info[inspected_parameter]['title']}"
                        ] = serialized_plot
                        plt.close(fig)

    with open(usability_map_file, "w") as f:
        yaml.dump(output, f)

    return results
