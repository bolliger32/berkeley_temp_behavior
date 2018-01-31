import pandas as pd
from os.path import join
from os import listdir
from datetime import datetime as dt, time, timedelta
import numpy as np
from glob import glob

## def variables
class Settings:
    def __init__(self, home_dir):
        
        self.berk = SiteSettings('berk',home_dir)
        self.bus = SiteSettings('bus',home_dir)
        
class SiteSettings:
    def __init__(self, site, home_dir):
        if site == 'berk':
            self.sens_locs = ['partition','RA']
            self.home_dir = join(home_dir,'Xlab data','Main experiment data')
            self.sensor_swap_date = '2017-10-03'
            self.sensor_interval_early = 5
            self.sensor_interval = 1
            self.temps_dir = join(self.home_dir,'Temperature')
            
        elif site == 'bus':
            self.sens_locs = ['near','far']
            self.home_dir = join(home_dir,'Busara data','Main experiment data')
            self.temps_dir = join(self.home_dir,'Temperature')
            
            
        else:
            raise NameError(site)
        
        self.timing_fpath = join(self.home_dir,'experiment_timing.csv')
        self.other_dir = join(self.home_dir,'Other Environmental Vars')
        self.download_dates = get_download_dates(site, self.temps_dir)

#################################
## DATA DOWNLOAD FUNCTIONS
#################################

def get_download_dates(site, temps_dir):
    """Find all dates when data was downloaded from loggers, based on available files."""

    all_files = listdir(join(temps_dir,'indoor'))
    if site == 'berk':
        download_dates = [dt.strptime(i,'%Y%m%d') for i in all_files if i[0] != '.']
    elif site == 'bus':
        # only download data from dates where we have room-level sensors
        valid_dates = [a for a in all_files if glob(join(temps_dir,'indoor',a,'*F.*csv')) != []]
        
        download_dates = [dt.strptime(i[:-5],'%Y%m%d') for i in valid_dates if i[-4:].lower()=='warm']
        
    download_dates.sort()
    
    return download_dates

def load_vals_berkeley(s):
    """Input all temperature data from Berkeley experiment into dataframe."""
    
    download_dates = s.berk.download_dates
    sensor_swap_date = s.berk.sensor_swap_date
    
    dfs = {'indoor':{}}
    
    # co2
#    dfs['co2'] = load_co2_vals(download_dates,s.berk.home_dir)
    
    # outdoor temp
    dfs['outdoor'] = get_outdoor_data(s.berk.temps_dir,'berk')
    
    # base directory for indoor temp measurements
    indoor_temp_dir = join(s.berk.temps_dir,'indoor')
    
    # operative temp
#    dfs['t_op'] = load_t_op_vals(download_dates,indoor_temp_dir)

    print("Downloading {}...".format(download_dates[0]))
    
    # grab room temp data from both control and treatment rooms
    for gi,g in enumerate(['control','treatment']):
        this_df = dfs['indoor'][g] = {}
        
        ## download data from early in experiment when we were using different sensors
        this_df['partition'] = pd.read_excel(join(indoor_temp_dir,download_dates[0].strftime('%Y%m%d'),'{}_p.xls'.format(g)),
                          sheet_name='Records',parse_dates=True,index_col=0).loc[:pd.to_datetime(sensor_swap_date),:]
        
        this_df['RA'] = pd.read_excel(join(indoor_temp_dir,download_dates[0].strftime('%Y%m%d'),'{}_RA.xls'.format(g)),
                          sheet_name='Data Table',parse_dates=True,index_col=1,header=21).iloc[:,1].loc[:pd.to_datetime(sensor_swap_date)]
        this_df['RA'].name = 'T'
        this_df['RA'] = pd.DataFrame(this_df['RA'])
        this_df['RA']['RH'] = np.nan
        
        for loc in ['partition','RA']:  
            this_df[loc].columns = ['T','RH']
            this_df[loc].index.name='time'
        
        ## now download data from sensors we switched to
        for d in download_dates[1:]:
            print("Downloading {}...".format(d))
            for loc in [('partition','p'),('RA','RA')]:
                new_df = pd.read_csv(join(indoor_temp_dir,d.strftime('%Y%m%d'),'csvs',
                        '{}_{}.csv'.format(g,loc[1])),usecols=[1,2,3],header=1,
                       index_col=0,parse_dates=True).loc[pd.to_datetime(sensor_swap_date):,:]
                new_df.columns=['T','RH']
                new_df.index.name='time'
                this_df[loc[0]] = this_df[loc[0]].append(new_df)
            
        this_df = drop_duplicates_and_flags(this_df, s.berk)
        
    return dfs

def load_vals_bus(s):
    """Input all temperature data from Busara experiment into dataframe."""
    
    download_dates = s.bus.download_dates
    
    dfs = {'indoor':{}}
    
    dfs['outdoor'] = get_outdoor_data(s.bus.temps_dir,'bus')

    # grab room temp data from both control and treatment rooms
    for gi,gx in enumerate([('Cool','control'),('Warm','treatment')]):
        
        ga = gx[0]
        g = gx[1]
        tx_lab = ga[0].upper()

        this_df = dfs['indoor'][g] = {}
        
        ## now download data from sensors we switched to
        for dx,d in enumerate(download_dates):
            d_str = d.strftime('%Y%m%d')
            for loc in [('far','F'),('near','N')]:
                file = glob(join(s.bus.temps_dir,'indoor','{}_{}'.format(d_str,ga),
                        '{}_Temp_{}{}.*csv'.format(d_str,tx_lab,loc[1])))[0]
                new_df = pd.read_csv(file,usecols=[1,2,3],header=1,
                       index_col=0,parse_dates=True)
                new_df.columns=['T','RH']
                new_df.index.name='time'
                if dx == 0:
                    this_df[loc[0]] = new_df
                else:
                    this_df[loc[0]] = this_df[loc[0]].append(new_df)
        this_df = drop_duplicates_and_flags(this_df, s.bus)
        
    return dfs

def drop_duplicates_and_flags(dfs, siteSettings):
    """After appending indoor temp/RH files together, drop duplicated measurements and logger flag rows."""
    for loc in siteSettings.sens_locs:
        # drop duplicate times from multiple files
        dfs[loc] = dfs[loc][~dfs[loc].index.duplicated(keep='last')].sort_index()

        # drop data logger flag rows
        dfs[loc] = dfs[loc][dfs[loc].notnull().all(axis=1)]
        
    return dfs
    
def correct_tz(df,site):
    """Correct weird timezones in Berkeley outdoor temp data logger"""
    if site == 'berk':
        tz_offset = df.index.name[-5:]
        if tz_offset[0] == '-':
            tz = '+'
        else:
            tz = '-'
        hr_adjust = int(tz_offset[1:3])
        tz = 'Etc/GMT' + tz + str(hr_adjust)
        df.index = df.index.tz_localize(tz).tz_convert('America/Los_Angeles')
    
    return df

def get_outdoor_data(temp_dir,site):
    """Extract outdoor temp and RH data"""
    if site == 'berk':
        files_od = glob(join(temp_dir,'outdoor','20*.xlsx'))
    elif site == 'bus':
        files_od = glob(join(temp_dir,'outdoor','Busara*.csv'))
    else:
        raise NameError(site)
    df_od = pd.DataFrame()

    for f in files_od:
        if site == 'berk':
            this_df = pd.read_excel(f,sheet_name=0,usecols='B:D',index_col=0,parse_dates=True)
        elif site == 'bus':
            this_df = pd.read_csv(f,usecols=[0,1,2],index_col=0,parse_dates=True,header=1)
        
        # drop missing values that prevented conversion to float type
        if this_df.iloc[:,0].dtype != np.float64:
            this_df = this_df[this_df.iloc[:,0] != ' ']
            this_df = this_df.astype(np.float64)

        # correct for weird timezones in berkeley datalogger
        this_df = correct_tz(this_df,site)
        
        this_df.columns = ['T','RH']
        this_df.index.name = 'time'

        # convert to celsius
        this_df['T'] = (this_df['T'] - 32) * 5/9
        df_od = df_od.append(this_df)

    # drop duplicated measurements
    df_od = df_od[~df_od.index.duplicated(keep='last')].sort_index()
    
    # separate out into daily min,mean,max
    groups = df_od.groupby(df_od.index.date)
    dfs_od = {'all':df_od,
             'min': groups.min(),
             'mean': groups.mean(),
             'max': groups.max()}
    
    for i in ['min','mean','max']:
        # remove first and last day to ignore days where we did not get full recording
        dfs_od[i] = dfs_od[i].iloc[1:-1,:]
        
        # name index so that we can merge onto multiIndex'd dataframe
        dfs_od[i].index.name = 'date'
    
    return dfs_od


#################################
## SESSION TIMING FUNCTIONS
#################################

def get_timing_df_berk(s):
# Load csv containing start and stop times for session
    raw_timing_df = pd.read_csv(s.berk.timing_fpath,index_col = [0,1,2], parse_dates = True,
            usecols=['Date','Session in day','Treatment group','Time entering room','Time ending modules'])
    timing_df = correct_raw_timing_df(raw_timing_df)
    return timing_df

def get_timing_df_bus(s):
    timing_fpath_bus = s.bus.timing_fpath
    
    timing_df = pd.read_csv(timing_fpath_bus,usecols=range(5))
    timing_df.loc[:,['Date','Session in day']] = timing_df.loc[:,['Date','Session in day']].fillna(method='ffill')
    timing_df['Date'] = pd.to_datetime(timing_df['Date'])
    timing_df = timing_df.set_index(['Date','Session in day','Treatment group'])
    
    timing_df = correct_raw_timing_df(timing_df)
    
    return timing_df

def correct_raw_timing_df(timing_df):

    # Drop cancelled sessions
    timing_df = timing_df[~((timing_df['Time entering room'].isin(["Cancelled",'canceled',''])) | (timing_df['Time entering room'].isnull()))]

    # Convert to datetime objects
    timing_df.loc[:,'start_time'] = pd.to_datetime(
        timing_df.index.get_level_values(0).astype(str) + 'T' + timing_df['Time entering room'])
    timing_df.loc[:,'end_time'] = pd.to_datetime(
        timing_df.index.get_level_values(0).astype(str) + 'T' + timing_df['Time ending modules'])

    # Impute end time from other concurrent session if not recorded
    min_end_times = timing_df.groupby(level=[0,1])['end_time'].min()
    max_start_times = timing_df.groupby(level=[0,1])['start_time'].max()
    imputed_vals = pd.DataFrame({'min_end_time':min_end_times,
                   'max_start_time':max_start_times},index = min_end_times.index)

    imputed_vals = imputed_vals.append(imputed_vals)
    imputed_vals['Treatment group'] = [0] * int(imputed_vals.shape[0] / 2) + [1] * int(imputed_vals.shape[0] / 2)
    imputed_vals = imputed_vals.set_index('Treatment group',append=True)

    timing_df = timing_df.join(imputed_vals)
    timing_df.loc[:,'start_time'] = timing_df.loc[:,'start_time'].where(
        cond=timing_df.loc[:,'start_time'].notnull(), other=timing_df['max_start_time'])
    timing_df.loc[:,'end_time'] = timing_df.loc[:,'end_time'].where(
        cond=timing_df.loc[:,'end_time'].notnull(), other=timing_df['min_end_time'])

    timing_df = timing_df[['start_time','end_time']]

    ## Correct for not using 24 hour clock
    for i in ['start_time','end_time']:
        bad_times_bool = (timing_df[i].dt.time < time(8))
        bad_times = timing_df.loc[bad_times_bool,i]
        new_times = bad_times + timedelta(hours=12)
        timing_df[i] = timing_df[i].where(bad_times_bool==False,other=new_times)
        
    # correct potential string values for session number
    timing_df.index = timing_df.index.remove_unused_levels()
    timing_df.index = timing_df.index.set_levels(timing_df.index.levels[1].astype(int),level=1)
    
    timing_df = convert_to_interval_index(timing_df)
    
    return timing_df

def convert_to_interval_index(df):
    """Convert a dataframe w/ start_date and end_date to an interval idexed df"""
    # convert timing DF to have a interval index
    res = df.reset_index()
    idx = pd.IntervalIndex.from_arrays(res['start_time'],res['end_time'])
    res.index = idx
    
    return res