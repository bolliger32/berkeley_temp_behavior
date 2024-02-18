import warnings
from datetime import datetime as dt
from datetime import time, timedelta
from glob import glob
from os import listdir
from pathlib import Path

import numpy as np
import pandas as pd


# def variables
class Settings:
    def __init__(self, home_dir):
        self.base_dir = Path(home_dir)
        self.data_dir = self.base_dir / "data" / "main"
        self.raw_data_dir = self.data_dir / "raw"
        self.clean_data_dir = self.data_dir / "clean"
        self.berk = SiteSettings("berk", self.raw_data_dir, self.clean_data_dir)
        self.bus = SiteSettings("bus", self.raw_data_dir, self.clean_data_dir)
        self.module_temps_path = self.clean_data_dir / "module_level_temps.parquet"
        self.ts_dfs_path = self.clean_data_dir / "ts_dfs.pickle"
        self.fig_dir = self.base_dir / "figures"
        self.table_dir = self.base_dir / "tables"
        self.sites = ["Nairobi", "California"]
        self.txs = ["control", "treatment"]
        self.site_map = {"California": "berk", "Nairobi": "bus"}


class SiteSettings:
    def __init__(self, site, home_dir, save_dir):
        if site == "berk":
            self.loc = "xlab"
            self.sens_locs = ["partition", "RA"]
            self.sensor_swap_date = "2017-10-03"
            self.sensor_interval_early = 5
            self.sensor_interval = 1

        elif site == "bus":
            self.loc = "busara"
            self.sens_locs = ["near", "far"]
        else:
            raise NameError(site)
        self.home_dir = home_dir / self.loc
        self.env_dir = home_dir / self.loc / "environment"
        self.modules_dir = home_dir / self.loc / "modules"
        self.other_dir = self.env_dir / "other_env_vars"
        self.temps_dir = self.env_dir / "temp"
        self.timing_fpath = self.env_dir / "experiment_timing.csv"
        self.download_dates = get_download_dates(site, self.temps_dir / "indoor")
        self.download_dates_oth = get_download_dates(
            site, self.env_dir / "other_env_vars"
        )
        self.env_out_path = (
            save_dir / f"session_level_environmental_data_{self.loc}.parquet"
        )


#################################
## DATA DOWNLOAD FUNCTIONS
#################################


def get_download_dates(site, temps_dir):
    """Find all dates when data was downloaded from loggers, based on available files."""

    all_files = temps_dir.glob("[0-9]*")
    if site == "berk":
        download_dates = [dt.strptime(i.name, "%Y%m%d") for i in all_files]
    elif site == "bus":
        # only download data from dates where we have room-level sensors
        valid_dates = [a for a in all_files if a.glob("*F.*csv") != []]

        download_dates = [
            dt.strptime(i.name[:-5], "%Y%m%d")
            for i in valid_dates
            if i.name[-4:].lower() == "warm"
        ]

    download_dates.sort()

    return download_dates


def load_vals_berkeley(s):
    """Input all temperature data from Berkeley experiment into dataframe."""

    download_dates = s.berk.download_dates
    download_dates_oth = s.berk.download_dates_oth
    sensor_swap_date = s.berk.sensor_swap_date

    dfs = {"indoor": {}}

    # outdoor temp
    dfs["outdoor"] = get_outdoor_data(s.berk.temps_dir, "berk")

    # base directory for indoor temp measurements
    indoor_temp_dir = s.berk.temps_dir / "indoor"

    print("Downloading temps: {}...".format(download_dates[0]))

    # grab room temp data from both control and treatment rooms
    for gi, g in enumerate(["control", "treatment"]):
        this_df = dfs["indoor"][g] = {}

        ## download data from early in experiment when we were using different sensors
        this_df["partition"] = pd.read_excel(
            indoor_temp_dir / download_dates[0].strftime("%Y%m%d") / f"{g}_p.xls",
            sheet_name="Records",
            parse_dates=True,
            index_col=0,
        ).loc[: pd.to_datetime(sensor_swap_date), :]

        this_df["RA"] = (
            pd.read_excel(
                indoor_temp_dir / download_dates[0].strftime("%Y%m%d") / f"{g}_RA.xls",
                sheet_name="Data Table",
                parse_dates=True,
                index_col=1,
                header=21,
            )
            .iloc[:, 1]
            .loc[: pd.to_datetime(sensor_swap_date)]
        )
        this_df["RA"].name = "T"
        this_df["RA"] = pd.DataFrame(this_df["RA"])
        this_df["RA"]["RH"] = np.nan

        for loc in ["partition", "RA"]:
            this_df[loc].columns = ["T", "RH"]
            this_df[loc].index.name = "time"

        ## now download data from sensors we switched to
        for d in download_dates[1:]:
            csv_dir = indoor_temp_dir / d.strftime("%Y%m%d") / "csvs"
            print("Downloading temps: {}...".format(d))
            for loc in [("partition", "p"), ("RA", "RA")]:
                fpath = csv_dir / f"{g}_{loc[1]}.csv"
                this_df = add_file_to_dfs(
                    fpath,
                    this_df,
                    [1, 2, 3],
                    ["T", "RH"],
                    loc[0],
                    sensor_swap=sensor_swap_date,
                )

            ## add individual temp/RH
            for s_ix in range(1, 7):
                if (csv_dir / f"{g}_{s_ix}.csv").is_file():
                    fpath = csv_dir / f"{g}_{s_ix}.csv"
                    this_df = add_file_to_dfs(
                        fpath, this_df, [1, 2, 3], ["T", "RH"], str(s_ix)
                    )

            ## add operative temp
            if (csv_dir / f"{g}_ot.csv").is_file():
                fpath = csv_dir / f"{g}_ot.csv"
                this_df = add_file_to_dfs(fpath, this_df, [1, 4], ["Top"], "Top")

        ## add CO2
        for d in download_dates_oth:
            print("Downloading co2: {}...".format(d))
            csv_dir = s.berk.other_dir / d.strftime("%Y%m%d")

            # pass when the file doesn't exist (aka when
            # the CO2 sensor's batteries died
            fpath = csv_dir / f"{g}_co2.csv"
            if not fpath.is_file():
                continue

            # otherwise, parse
            this_df = add_file_to_dfs(fpath, this_df, [1, 4], ["co2"], "co2")

        this_df = drop_duplicates_and_flags(this_df)

    return dfs


def load_vals_bus(s):
    """Input all temperature data from Busara experiment into dataframe."""

    download_dates = s.bus.download_dates

    dfs = {"indoor": {}}

    dfs["outdoor"] = get_outdoor_data(s.bus.temps_dir, "bus")

    # grab room temp data from both control and treatment rooms
    for gi, gx in enumerate([("Cool", "control"), ("Warm", "treatment")]):
        ga = gx[0]
        g = gx[1]
        tx_lab = ga[0].upper()

        print(f"Downloading {g} room data...")

        this_df = dfs["indoor"][g] = {}

        ## now download data from sensors we switched to
        for dx, d in enumerate(download_dates):

            d_str = d.strftime("%Y%m%d")
            dirname = s.bus.temps_dir / "indoor" / f"{d_str}_{ga}"
            print(f"Downloading {d_str}...")

            for loc in [("far", "F"), ("near", "N")]:
                print("Downloading {} sensor...".format(loc[0]))

                # need to use glob because some files have two "."s and some have one
                fname = f"{d_str}_Temp_{tx_lab}{loc[1]}.*csv"
                files = list(dirname.glob(fname))
                fpath = files[0]

                # check for missing files
                if len(files) != 1:
                    with warnings.catch_warnings():
                        warnings.simplefilter("always")
                        warnings.warn(f"Missing file: {fname}")
                    continue
                this_df = add_file_to_dfs(
                    fpath, this_df, [1, 2, 3], ["T", "RH"], loc[0]
                )

            ## add individual data
            for s_ix in range(1, 7):
                fname = f"{d_str}_Temp_{tx_lab}{s_ix}.csv"
                this_df = add_file_to_dfs(
                    dirname / fname, this_df, [1, 2, 3], ["T", "RH"], str(s_ix)
                )

            ## add T_operative
            fname = f"{d_str}_PingPong_{ga}.csv"
            this_df = add_file_to_dfs(dirname / fname, this_df, [1, 4], ["Top"], "Top")

            ## add CO2
            fname = "{}_{}_co2.csv".format(d_str, g)
            this_df = add_file_to_dfs(dirname / fname, this_df, [1, 4], ["co2"], "co2")

        this_df = drop_duplicates_and_flags(this_df)

    return dfs


def add_file_to_dfs(
    fpath, dfs, cols_to_use, colnames, df_name, header=1, sensor_swap=None
):
    """Load data from file into a dataframe and add it to an existing dictionary of dataframes"""
    new_df = pd.read_csv(
        fpath,
        usecols=cols_to_use,
        header=header,
        index_col=0,
        date_format="mixed",
    )
    if sensor_swap:
        new_df = new_df.loc[pd.to_datetime(sensor_swap) :, :]
    new_df.columns = colnames
    new_df.index.name = "time"
    if df_name in dfs.keys():
        dfs[df_name] = pd.concat((dfs[df_name], new_df))
    else:
        dfs[df_name] = new_df

    return dfs


def drop_duplicates_and_flags(dfs):
    """After appending indoor temp/RH files together, drop duplicated measurements and logger flag rows."""
    for loc in dfs.keys():
        # drop duplicate times from multiple files
        dfs[loc] = dfs[loc][~dfs[loc].index.duplicated(keep="last")].sort_index()

        # drop data logger flag rows
        dfs[loc] = dfs[loc][dfs[loc].notnull().any(axis=1)]

    return dfs


def correct_tz(df, site):
    """Correct weird timezones in Berkeley outdoor temp data logger"""
    if site == "berk":
        tz_offset = df.index.name[-5:]
        if tz_offset[0] == "-":
            tz = "+"
        else:
            tz = "-"
        hr_adjust = int(tz_offset[1:3])
        tz = "Etc/GMT" + tz + str(hr_adjust)
        df.index = df.index.tz_localize(tz).tz_convert("America/Los_Angeles")

    return df


def get_outdoor_data(temp_dir, site):
    """Extract outdoor temp and RH data"""
    if site == "berk":
        files_od = (temp_dir / "outdoor").glob("20*.xlsx")
    elif site == "bus":
        files_od = (temp_dir / "outdoor").glob("Busara*.csv")
    else:
        raise NameError(site)

    dfs = []
    for f in files_od:
        if site == "berk":
            this_df = pd.read_excel(
                f, sheet_name=0, usecols="B:D", index_col=0, parse_dates=True, header=1
            )
        elif site == "bus":
            this_df = pd.read_csv(
                f, usecols=[0, 1, 2], index_col=0, parse_dates=True, header=2
            )

        # drop missing values that prevented conversion to float type
        if this_df.iloc[:, 0].dtype != np.float64:
            this_df = this_df[this_df.iloc[:, 0] != " "]
            this_df = this_df.astype(np.float64)

        # correct for weird timezones in berkeley datalogger
        this_df = correct_tz(this_df, site)

        this_df.columns = ["T", "RH"]
        this_df.index.name = "time"

        # convert to celsius
        this_df["T"] = (this_df["T"] - 32) * 5 / 9
        dfs.append(this_df)

    df_od = pd.concat(dfs)

    # drop duplicated measurements
    df_od = df_od[~df_od.index.duplicated(keep="last")].sort_index()

    # separate out into daily min,mean,max
    groups = df_od.groupby(df_od.index.date)
    dfs_od = {
        "all": df_od,
        "min": groups.min(),
        "mean": groups.mean(),
        "max": groups.max(),
    }

    for i in ["min", "mean", "max"]:
        # remove first and last day to ignore days where we did not get full recording
        dfs_od[i] = dfs_od[i].iloc[1:-1, :]

        # name index so that we can merge onto multiIndex'd dataframe
        dfs_od[i].index.name = "date"

    return dfs_od


#################################
## SESSION TIMING FUNCTIONS
#################################


def get_timing_df_berk(s):
    # Load csv containing start and stop times for session
    raw_timing_df = pd.read_csv(
        s.berk.timing_fpath,
        index_col=[0, 1, 2],
        date_format="%m-%d-%y",
        usecols=[
            "Date",
            "Session in day",
            "Treatment group",
            "Time entering room",
            "Time ending modules",
        ],
    )
    timing_df = correct_raw_timing_df(raw_timing_df)
    return timing_df


def get_timing_df_bus(s):
    timing_fpath_bus = s.bus.timing_fpath

    timing_df = pd.read_csv(
        timing_fpath_bus, usecols=range(5), parse_dates=["Date"], date_format="%m/%d/%y"
    )
    timing_df.loc[:, ["Date", "Session in day"]] = timing_df.loc[
        :, ["Date", "Session in day"]
    ].ffill()
    timing_df = timing_df.set_index(["Date", "Session in day", "Treatment group"])

    timing_df = correct_raw_timing_df(timing_df)

    return timing_df


def correct_raw_timing_df(timing_df):

    # Drop cancelled sessions
    timing_df = timing_df.loc[
        ~(
            (timing_df.loc[:, "Time entering room"].isin(["Cancelled", "canceled", ""]))
            | (timing_df.loc[:, "Time entering room"].isnull())
        )
    ].copy()

    # Convert to datetime objects
    timing_df["start_time"] = pd.to_datetime(
        timing_df.index.get_level_values(0).astype(str)
        + "T"
        + timing_df["Time entering room"]
    )
    timing_df["end_time"] = pd.to_datetime(
        timing_df.index.get_level_values(0).astype(str)
        + "T"
        + timing_df["Time ending modules"]
    )

    # Impute end time from other concurrent session if not recorded
    min_end_times = timing_df.groupby(level=[0, 1])["end_time"].min()
    max_start_times = timing_df.groupby(level=[0, 1])["start_time"].max()
    imputed_vals = pd.DataFrame(
        {"min_end_time": min_end_times, "max_start_time": max_start_times},
        index=min_end_times.index,
    )

    imputed_vals = pd.concat((imputed_vals, imputed_vals))
    imputed_vals["Treatment group"] = [0] * int(imputed_vals.shape[0] / 2) + [1] * int(
        imputed_vals.shape[0] / 2
    )
    imputed_vals = imputed_vals.set_index("Treatment group", append=True)

    timing_df = timing_df.join(imputed_vals)
    timing_df.loc[:, "start_time"] = timing_df.loc[:, "start_time"].where(
        cond=timing_df.loc[:, "start_time"].notnull(), other=timing_df["max_start_time"]
    )
    timing_df.loc[:, "end_time"] = timing_df.loc[:, "end_time"].where(
        cond=timing_df.loc[:, "end_time"].notnull(), other=timing_df["min_end_time"]
    )

    timing_df = timing_df[["start_time", "end_time"]]

    ## Correct for not using 24 hour clock
    for i in ["start_time", "end_time"]:
        bad_times_bool = timing_df[i].dt.time < time(8)
        bad_times = timing_df.loc[bad_times_bool, i]
        new_times = bad_times + timedelta(hours=12)
        timing_df[i] = timing_df[i].where(bad_times_bool == False, other=new_times)

    # correct potential string values for session number
    timing_df.index = timing_df.index.remove_unused_levels()
    timing_df.index = timing_df.index.set_levels(
        timing_df.index.levels[1].astype(int), level=1
    )

    timing_df = convert_to_interval_index(timing_df)

    return timing_df


def convert_to_interval_index(df):
    """Convert a dataframe w/ start_date and end_date to an interval idexed df"""
    # convert timing DF to have a interval index
    res = df.reset_index()
    idx = pd.IntervalIndex.from_arrays(res["start_time"], res["end_time"])
    res.index = idx

    return res
