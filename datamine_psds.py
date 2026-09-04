#%%
from urllib.request import urlopen
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import os
import time
from urllib.error import URLError


stations = ['HM01', 'HM02', 'HM03', 'HM04', 'HM05', 'HM06', 'HM07', 'HM08',
            'HM09','HM10', 'HM11', 'HM12', 'HM13','HM14', 'HM15', 'HM16', 
            'HM17','HM18','HM19','HM20','HM21','HM22', 'HM23', 'HM24',
            'HM25', 'HM26']

stations = ['KD01', 'KD02', 'KD03', 'KD04', 'KD05', 'KD06', 'KD07', 'KD08',
            'KD09','KD10', 'KD11', 'KD12', 'KD13','KD14', 'KD15', 'KD16', 
            'KD17','KD18','KD19','KD20','KD21','KD22', 'KD23', 'KD24',
            'KD25', 'KD26']

#stations = ['HM01', 'HM02']
global_start = '2025-09-08'
global_end = '2025-11-13' #'2025-11-14'

# Pull in earthquake dataset
#df = pd.read_csv('/Users/cadequigley/repos/array_aggregator/2A_1000km_m3_lts__window_freq_map_fig.csv')
figpath = '/Users/cadequigley/Downloads/Research/deployment_array_design/datamine_paper_figures/'
df = pd.read_csv(figpath+'m2_earthquakes_hom_kod.csv')
df = pd.DataFrame(df[df['magnitude']>= 3]).reset_index() #1800

#%%
# Pull out date for each earthquake
times = [df['time_utc'][i][:10] for i in range(len(df))]

# Find frequency of events on each day
time_series = pd.Series(times)
f = time_series.value_counts().reset_index()
f.columns = ['string', 'count']
f = pd.DataFrame(f[f['count'] > 2])

# Create list of dates for full deployment
dates = pd.date_range(start=global_start, end=global_end, freq='D')
dates_str = dates.strftime('%Y-%m-%d').tolist()

# Remove dates with >2 M3+ earthquakes within 1000 km
dates_series = pd.Series(dates_str)
filtered_dates = dates_series[~dates_series.isin(f['string'])].tolist()


def fetch_psd_for_station(station):
    """Fetch all dates sequentially for a single station."""
    print(f'Starting station {station}')
    psds_list = []

    for start_time in filtered_dates:
        end_time = (pd.Timestamp(start_time) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        print(f'Grabbing data for station {station} on {start_time}')

        for attempt in range(3):
            try:
                xml_url = (
                    f"https://service.iris.edu/mustang/noise-psd/1/query?"
                    f"target=4E.{station}.*.DHZ.M&starttime={start_time}&endtime={end_time}&format=xml"
                )

                with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as tmp:
                    tmp_path = tmp.name
                    tmp.write(urlopen(xml_url).read().decode('utf-8'))

                test4 = pd.read_xml(tmp_path, xpath="/PsdRoot/Psds/Psd/value")
                os.unlink(tmp_path)

                unique = test4['freq'].unique()
                median_power, mean_power = [], []
                for cat in unique:
                    mask = test4['freq'] == cat
                    vals1 = test4.loc[mask, 'power']
                    median_power.append(np.median(vals1))
                    mean_power.append(np.mean(vals1))
                
                psds_list.append(pd.DataFrame({
                    'mean_power': mean_power,
                    'median_power': median_power,
                    'frequency': unique,
                    'time': [start_time] * len(median_power),
                    'station': [station] * len(median_power),
                }))
                break  # Success, exit retry loop
            
            
            except Exception as e:
                wait = 5 * (attempt + 1)
                if attempt < 2:
                    print(f'Attempt {attempt+1} failed for {station} on {start_time}: {e}. Retrying in {wait}s...')
                    time.sleep(wait)
                else:
                    print(f'All attempts failed for {station} on {start_time}: {e}')

    if psds_list:
        return pd.concat(psds_list, ignore_index=True)
    return None


# Parallelize only across stations
station_psds = []
with ThreadPoolExecutor(max_workers=len(stations)) as executor:
    futures = {executor.submit(fetch_psd_for_station, sta): sta for sta in stations}
    for future in as_completed(futures):
        result = future.result()
        if result is not None:
            station_psds.append(result)

# Combine all stations into single dataframe
full_df = pd.concat(station_psds, ignore_index=True)
#print(full_df)
full_df.to_csv(figpath+'hm_psds.csv')