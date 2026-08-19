import os

import h5py
from pandas import DataFrame, concat, read_hdf

from . import utils

# -------------------------------------------------------------------------
# Saving related functions
# -------------------------------------------------------------------------


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SHELVE OBJECTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def save_df_and_info(df: DataFrame, plot_info: dict) -> dict:
    """Return a dictionary containing a dataframe for the parameter(s) under study for a given subsystem. The plotting info are saved too."""
    columns_to_drop = [
        "name",
        "location",
        "position",
        "cc4_channel",
        "cc4_id",
        "status",
        "det_type",
        "flag_muon",
        "flag_pulser",
        "flag_fc_bsln",
        "daq_crate",
        "daq_card",
        "HV_card",
        "HV_channel",
    ]
    columns_existing = [col for col in columns_to_drop if col in df.copy().columns]

    if columns_existing:
        df = df.drop(columns=columns_existing)

    par_dict_content = {
        "df_" + plot_info["subsystem"]: df,  # saving dataframe
        "plot_info": plot_info,  # saving plotting info
    }

    return par_dict_content


def check_level0(dataframe: DataFrame) -> DataFrame:
    """Check if a dataframe contains the 'level_0' column. If so, remove it."""
    if "level_0" in dataframe.columns:
        return dataframe.drop(columns=["level_0"])
    else:
        return dataframe


def get_param_info(param: str, plot_info: dict) -> dict:
    """Subselect from 'plot_info' the plotting info for the specified parameter ```param```. This is needed for the multi-parameters case."""
    # get the *naked* parameter name and apply some if statements to avoid problems
    param = param + "_var" if "_var" not in param else param
    parameter = param.split("_var")[0]

    # but what if there is no % variation? We don't want any "_var" in our parameters!
    if (
        isinstance(plot_info["unit_label"], dict)
        and param not in plot_info["unit_label"].keys()
    ):
        if plot_info["unit_label"][parameter] != "%":
            param = parameter
    if isinstance(plot_info["unit_label"], str):
        if plot_info["unit_label"] != "%":
            param = parameter

    # re-shape the plot_info dictionary for the given parameter under study
    plot_info_param = plot_info.copy()
    plot_info_param["title"] = f"Plotting {param}"
    plot_info_param["unit"] = (
        plot_info["unit"][param]
        if isinstance(plot_info["unit"], dict)
        else plot_info["unit"]
    )
    plot_info_param["label"] = (
        plot_info["label"][param]
        if isinstance(plot_info["label"], dict)
        else plot_info["label"]
    )
    plot_info_param["unit_label"] = (
        plot_info["unit_label"][param]
        if isinstance(plot_info["unit_label"], dict)
        else plot_info["unit_label"]
    )
    plot_info_param["limits"] = (
        plot_info["limits"][param]
        if isinstance(plot_info["limits"], dict)
        else plot_info["limits"]
    )
    plot_info_param["event_type"] = (
        plot_info["event_type"][param]
        if isinstance(plot_info["event_type"], dict)
        else plot_info["event_type"]
    )
    plot_info_param["param_mean"] = parameter + "_mean"
    plot_info_param["variation"] = (
        True if plot_info_param["unit_label"] == "%" else False
    )
    plot_info_param["parameters"] = (
        param if plot_info_param["variation"] is True else parameter
    )

    # ... need to go back to the one parameter case ...
    # if "parameters" in plot_info_param.keys():
    #    plot_info_param["parameter"] = plot_info_param.pop("parameters")

    return plot_info_param


def get_param_df(parameter: str, df: DataFrame) -> DataFrame:
    """Subselect from 'df' only the dataframe columns that refer to a given parameter. The case of 'parameter' being a special parameter is carefully handled."""
    # list needed to better divide the parameters stored in the dataframe...
    keep_cols = [
        "index",
        "channel",
        "HV_card",
        "HV_channel",
        "cc4_channel",
        "cc4_id",
        "daq_card",
        "daq_crate",
        "datetime",
        "det_type",
        "flag_fc_bsln",
        "flag_muon",
        "flag_pulser",
        "location",
        "name",
        "position",
        "status",
    ]
    # build the full column list first, then take one single subset (no full-frame copies)
    param_cols = [x for x in df.columns if parameter in x]
    meta_cols = [x for x in df.columns if x in keep_cols]

    # check if the parameter belongs to a special one
    other_cols = []
    if parameter in utils.SPECIAL_PARAMETERS:
        # get the other columns to keep in the new dataframe
        # (of course, avoid to load columns if the special parameter does not request any special parameter,
        # eg event rate or exposure are not build on the basis of any other parameter)
        other_cols_to_keep = utils.SPECIAL_PARAMETERS[parameter]
        if isinstance(other_cols_to_keep, str):
            other_cols_to_keep = [other_cols_to_keep]
        if isinstance(other_cols_to_keep, list):
            other_cols = [
                col
                for col in other_cols_to_keep
                if col is not None and col in df.columns
            ]

    df_param = df[param_cols + meta_cols + other_cols].copy()

    return df_param


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HDF OBJECTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def save_hdf(
    saving: str,
    file_path: str,
    df,
    aux_ch: str,
    aux_analysis,
    aux_ratio_analysis,
    aux_diff_analysis,
    plot_info: dict,
) -> dict:
    """Save the input dataframe in an external hdf file, using a different structure (time vs channel, with values in cells). Plot info are saved too."""
    utils.logger.info("Building HDF file(s)")
    # save the final dataframe as a hdf object
    parameters = plot_info["parameters"]
    keys_to_drop = [
        "std",
        "range",
        "plot_style",
        "variation",
        "limits",
        "title",
        "parameters",
        "parameter",
        "param_mean",
        "locname",
        "time_window",
        "resampled",
        "unit_label",
    ]

    for param in parameters:
        evt_type = (
            plot_info["event_type"][param]
            if isinstance(plot_info["event_type"], dict)
            else plot_info["event_type"]
        )
        param_orig = param.removesuffix("_var")
        param_orig_camel = utils.convert_to_camel_case(param_orig, "_")

        # get dictionary with useful plotting info
        plot_info_param = get_param_info(param, plot_info)
        # drop the list, and get directly lower/upper limits (set to False if no limits are provided);
        # this helps to avoid mixing types with PyTables

        # fix the label (in general, it could contain info for aux data too - here, we want a simple version of the label)
        plot_info_param["label"] = (
            None
            if param
            in [
                "quality_cuts",
                "geds/quality/is_not_bb_like/is_delayed_discharge",
                "geds/quality/is_bb_like",
            ]
            else utils.PLOT_INFO[param_orig]["label"]
        )

        try:
            subsystem = plot_info_param.get("subsystem")
            if subsystem in utils.PLOT_INFO.get(param_orig, {}).get("limits", {}):
                limits_var = utils.PLOT_INFO[param_orig]["limits"][subsystem][
                    "variation"
                ]
            else:
                limits_var = [None, None]
        except Exception as e:
            utils.logger.error(
                "\033[91mError in determining limits_var: %s. Exit here.\033[0m", e
            )
            raise
        try:
            subsystem = plot_info_param.get("subsystem")
            if subsystem in utils.PLOT_INFO.get(param_orig, {}).get("limits", {}):
                limits_abs = utils.PLOT_INFO[param_orig]["limits"][subsystem][
                    "absolute"
                ]
            else:
                limits_abs = [None, None]
        except Exception as e:
            utils.logger.error(
                "\033[91mError in determining limits_abs: %s. Exit here.\033[0m", e
            )
            raise

        # for limits, change from 'None' to 'False' to be hdf-friendly
        plot_info_param["lower_lim_var"] = str(limits_var[0]) or False
        plot_info_param["upper_lim_var"] = str(limits_var[1]) or False
        plot_info_param["lower_lim_abs"] = str(limits_abs[0]) or False
        plot_info_param["upper_lim_abs"] = str(limits_abs[1]) or False

        # drop useless keys
        for key in keys_to_drop:
            del plot_info_param[key]

        # one-param case
        if len(parameters) == 1:
            df_to_save = df.data
            if not utils.check_empty_df(aux_analysis):
                df_aux_to_save = aux_analysis.data
            if not utils.check_empty_df(aux_ratio_analysis):
                df_aux_ratio_to_save = aux_ratio_analysis.data
            if not utils.check_empty_df(aux_diff_analysis):
                df_aux_diff_to_save = aux_diff_analysis.data
        # multi-param case (get only the df for the param of interest)
        if len(parameters) > 1:
            df_to_save = get_param_df(param_orig, df.data)
            if not utils.check_empty_df(aux_analysis):
                df_aux_to_save = get_param_df(param_orig, aux_analysis.data)
            if not utils.check_empty_df(aux_ratio_analysis):
                df_aux_ratio_to_save = get_param_df(param_orig, aux_ratio_analysis.data)
            if not utils.check_empty_df(aux_diff_analysis):
                df_aux_diff_to_save = get_param_df(param_orig, aux_diff_analysis.data)

        # still need to check overwrite/append (and existence of file!!!)
        # SOLVE THIS!!!
        # if saving == "overwrite":
        #    check_existence_and_overwrite(file_path)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PLOTTING INFO
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if param not in [
            "quality_cuts",
            "geds/quality/is_not_bb_like/is_delayed_discharge",
            "geds/quality/is_bb_like",
        ]:
            # this is constant over time, so with 'append' we simply overwrite previous content
            df_info = DataFrame.from_dict(
                plot_info_param, orient="index", columns=["Value"]
            )

            df_info.to_hdf(
                file_path,
                key=f"{utils.FLAGS_RENAME[evt_type]}_{param_orig_camel}_info",
                mode="a",
            )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PURE VALUES - AUX CHANNEL
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if not utils.check_empty_df(aux_analysis):
            # SOLVE THIS!!!
            # if saving == "overwrite":
            #    check_existence_and_overwrite(
            #        file_path.replace(plot_info_param["subsystem"], aux_ch)
            #    )

            plot_info_aux = plot_info_param.copy()
            plot_info_aux["subsystem"] = aux_ch
            # --- plotting info
            df_info_aux = DataFrame.from_dict(
                plot_info_aux, orient="index", columns=["Value"]
            )
            df_info_aux.to_hdf(
                file_path.replace(plot_info_param["subsystem"], aux_ch),
                key=f"{utils.FLAGS_RENAME[evt_type]}_{param_orig_camel}_info",
                mode="a",
            )

            # ... absolute values
            get_pivot(
                df_aux_to_save,
                param_orig,
                f"{utils.FLAGS_RENAME[evt_type]}_{param_orig_camel}",
                file_path.replace(plot_info_param["subsystem"], aux_ch),
                saving,
            )
            # ... mean values
            get_pivot(
                df_aux_to_save,
                param_orig + "_mean",
                f"{utils.FLAGS_RENAME[evt_type]}_{param_orig_camel}_mean",
                file_path.replace(plot_info_param["subsystem"], aux_ch),
                saving,
            )
            # ... % variations wrt absolute values
            get_pivot(
                df_aux_to_save,
                param_orig + "_var",
                f"{utils.FLAGS_RENAME[evt_type]}_{param_orig_camel}_var",
                file_path.replace(plot_info_param["subsystem"], aux_ch),
                saving,
            )
            utils.logger.info(
                f"... HDF file for {aux_ch} - pure AUX values - saved in: \33[4m{file_path.replace(plot_info_param['subsystem'], aux_ch)}\33[0m"
            )
            del df_aux_to_save

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PURE VALUES
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if param in [
            "quality_cuts",
            "geds/quality/is_not_bb_like/is_delayed_discharge",
            "geds/quality/is_bb_like",
        ]:
            # save each flag/classifier separately
            is_cols = [
                col
                for col in df_to_save.columns
                if (col.startswith("is_") or col.endswith("_classifier"))
                and "_mean" not in col
                and "_var" not in col
            ]
            for param_orig in is_cols:
                param_orig_camel = utils.convert_to_camel_case(param_orig, "_")
                # ... absolute values ONLY
                get_pivot(
                    df_to_save,
                    param_orig,
                    f"{utils.FLAGS_RENAME[evt_type]}_{param_orig_camel}",
                    file_path,
                    saving,
                )
            del df_to_save
        else:
            # ... absolute values
            get_pivot(
                df_to_save,
                param_orig,
                f"{utils.FLAGS_RENAME[evt_type]}_{param_orig_camel}",
                file_path,
                saving,
            )
            # ... mean values
            get_pivot(
                df_to_save,
                param_orig + "_mean",
                f"{utils.FLAGS_RENAME[evt_type]}_{param_orig_camel}_mean",
                file_path,
                saving,
            )
            # ... % variations wrt absolute values
            get_pivot(
                df_to_save,
                param_orig + "_var",
                f"{utils.FLAGS_RENAME[evt_type]}_{param_orig_camel}_var",
                file_path,
                saving,
            )
            del df_to_save

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # RATIO WRT AUX CHANNEL
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if not utils.check_empty_df(aux_ratio_analysis):
                # ... absolute values
                get_pivot(
                    df_aux_ratio_to_save,
                    param_orig,
                    f"{utils.FLAGS_RENAME[evt_type]}_{param_orig_camel}_{aux_ch}Ratio",
                    file_path,
                    saving,
                )
                # ... mean values
                get_pivot(
                    df_aux_ratio_to_save,
                    param_orig + "_mean",
                    f"{utils.FLAGS_RENAME[evt_type]}_{param_orig_camel}_{aux_ch}Ratio_mean",
                    file_path,
                    saving,
                )
                # ... % variations wrt absolute values
                get_pivot(
                    df_aux_ratio_to_save,
                    param_orig + "_var",
                    f"{utils.FLAGS_RENAME[evt_type]}_{param_orig_camel}_{aux_ch}Ratio_var",
                    file_path,
                    saving,
                )
                del df_aux_ratio_to_save

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # DIFFERENCE WRT AUX CHANNEL
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if not utils.check_empty_df(aux_diff_analysis):
                # ... absolute values
                get_pivot(
                    df_aux_diff_to_save,
                    param_orig,
                    f"{utils.FLAGS_RENAME[evt_type]}_{param_orig_camel}_{aux_ch}Diff",
                    file_path,
                    saving,
                )
                # ... mean values
                get_pivot(
                    df_aux_diff_to_save,
                    param_orig + "_mean",
                    f"{utils.FLAGS_RENAME[evt_type]}_{param_orig_camel}_{aux_ch}Diff_mean",
                    file_path,
                    saving,
                )
                # ... % variations wrt absolute values
                get_pivot(
                    df_aux_diff_to_save,
                    param_orig + "_var",
                    f"{utils.FLAGS_RENAME[evt_type]}_{param_orig_camel}_{aux_ch}Diff_var",
                    file_path,
                    saving,
                )
                del df_aux_diff_to_save

    utils.logger.info(
        f"... HDF file for {plot_info_param['subsystem']} saved in: \33[4m{file_path}\33[0m"
    )


def get_pivot(
    df: DataFrame, parameter: str, key_name: str, file_path: str, saving: str
):
    """Get pivot: datetimes (first column) vs channels (other columns)."""
    df_pivot = df.pivot(index="datetime", columns="channel", values=parameter)
    # the frame is loaded as float32, but the mean/variation arithmetic widens
    # it again; store what we mean to store rather than what pandas inferred
    if (df_pivot.dtypes == "float64").any():
        df_pivot = df_pivot.astype(
            {c: "float32" for c in df_pivot.columns if df_pivot[c].dtype == "float64"}
        )
    # just select one row for mean values (since mean is constant over time for a given channel)
    # take into consideration parameters that are named with 'mean' in it, eg "bl_mean"
    if ("_mean" in parameter and parameter.count("mean") > 1) or (
        "mean" in parameter.split("_")[-1] and "mean" not in parameter.split("_")[:-1]
    ):
        df_pivot = df_pivot.iloc[[0]]

    # append new data
    if saving == "append":
        # check if the file exists: if not, create a new one
        if not os.path.exists(file_path):
            df_pivot.to_hdf(file_path, key=key_name, mode="a", **utils.HDF_COMPRESSION)
            return
        # the file exists, but this specific key was not saved - create the new key
        saved_keys = []
        with h5py.File(file_path, "r") as file:
            saved_keys = list(file.keys())
        if os.path.exists(file_path) and key_name not in saved_keys:
            df_pivot.to_hdf(file_path, key=key_name, mode="a", **utils.HDF_COMPRESSION)
            return

        mean_pars = ["bl_mean", "pz_mean"]
        if (
            "_mean" in parameter
            and parameter.count("mean") == 1
            and parameter not in mean_pars
        ) or (parameter in mean_pars and parameter.count("mean") == 2):
            # for the mean entry, we overwrite the already existing content with the new mean value
            df_pivot.to_hdf(file_path, key=key_name, mode="a", **utils.HDF_COMPRESSION)

        if "_mean" not in parameter or (
            "_mean" in parameter
            and parameter in mean_pars
            and parameter.count("mean") == 1
        ):
            # if % variations, we have to re-calculate all of them for the new mean values
            if "_var" in parameter:
                key_name_orig = key_name.replace("_var", "")
                new_mean = read_hdf(
                    file_path, key=key_name_orig + "_mean"
                )  # gia' aggiornata (perche' la media la aggiorniamo prima delle variazioni %)
                new_var_data = read_hdf(
                    file_path, key=key_name_orig
                )  # df vecchio con TUTTI i valori assoluti (anche quelli di prima)

                # recompute % variations for all channels at once (column-aligned).
                # Overwriting the freshly-read frame in place saves a full copy
                # of the largest object in the run (a classifier key is ~0.5 GB
                # per 10 files, and this is the peak of the whole pipeline).
                channels = list(df["channel"].unique())
                new_var_data[channels] = (
                    new_var_data[channels].div(new_mean.iloc[0][channels]) - 1
                ) * 100

                # Write the combined DataFrame to the HDF5 file
                new_var_data.to_hdf(
                    file_path, key=key_name, mode="a", **utils.HDF_COMPRESSION
                )

            # otherwise, just read the existing HDF5 file
            else:
                # Read the existing HDF5 file
                existing_data = read_hdf(file_path, key=key_name)
                # Concatenate the existing data and the new data
                combined_data = concat([existing_data, df_pivot])
                # the concat already copied both inputs: holding them across
                # the write would keep three copies of the key alive at once
                del existing_data
                # Write the combined DataFrame to the HDF5 file
                combined_data.to_hdf(
                    file_path, key=key_name, mode="a", **utils.HDF_COMPRESSION
                )

    # overwrite already existing data
    else:
        df_pivot.to_hdf(file_path, key=key_name, mode="a", **utils.HDF_COMPRESSION)


def check_existence_and_overwrite(file: str):
    """Check for the existence of a file, and if it exists removes it."""
    if os.path.exists(file):
        os.remove(file)
