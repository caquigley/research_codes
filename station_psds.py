# %%
from urllib.request import urlopen
from urllib.request import urlopen
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import ssl
from obspy import read
from obspy import Stream
from obspy import Trace
from obspy import UTCDateTime

stations = ['2A01', '2A02', '2A03', '2A04', '2A05', '2A06', '2A07', 
            '2A09','2A10','2A12', '2A13', '2A14', '2A15']

stations = ['3A01', '3A02', '3A03', '3A05', '3A06', '3A07', 
            '3A09','3A12', '3A14']

stations = ['POM01', 'POM02', 'POM03', 'POM04',  'POM08',  
            'POM09','POM11','POM12', 'POM14', 'POM15', 'POM16', 'POM19']

start_time = '2015-07-02'
#end_time = '2015-07-10'
end_time = '2016-03-17'
##Pull in earthquake dataset---------------
df = pd.read_csv('/Users/cadequigley/repos/array_aggregator/2A_1000km_m3_lts__window_freq_map_fig.csv') #_const_lfreq.csv')

#Pull out date for each earthquake---------------------
times = []
for i in range(len(df)):
    temp = df['time'][i][:10]
    times.append(temp)

#Find frequency of events on each day---------------
time = pd.Series(times)
f = time.value_counts()
f = f.reset_index()
f.columns = ['string', 'count']
f = pd.DataFrame(f[f['count']> 2 ])

#Create list of dates for full deployment----------------------
dates = pd.date_range(start=start_time, end=end_time, freq='D')
dates_list = dates.tolist()
dates_str = dates.strftime('%Y-%m-%d').tolist()

#Remove dates that have >2 M3+ earthquakes within 1000 km--------------
dates_series = pd.Series(dates_str)
filtered_dates = dates_series[~dates_series.isin(f['string'])].tolist()

#Pull PSDS from MUSTANG--------------
station_psds = []
for sta in range(len(stations)):
    station = stations[sta]
    psds_list = []
    print('Starting station ', station)
    for date in range(len(filtered_dates)):
        start_time = filtered_dates[date]
        end_time = (pd.Timestamp(start_time) + pd.Timedelta(days=1)).strftime('%Y-%m-%d') #pulls one day of data
        print('Grabbing data for', start_time)

        xml = open("pdf_test.xml", "w")
        xml_url = "https://service.iris.edu/mustang/noise-psd/1/query?target=9C."+station+".*.SHZ.M&starttime="+start_time+"&endtime="+end_time+"&format=xml"
        xml.write(urlopen(xml_url).read().decode('utf-8'))
        
        xml1_file = "pdf_test.xml"
        test3 = pd.read_xml(xml1_file, xpath="/PsdRoot/Psds/Psd")
        test4 = pd.read_xml(xml1_file, xpath="/PsdRoot/Psds/Psd/value")
        unique = test4['freq'].unique()

        median_power = []
        mean_power = []
        for cat in unique:
            # Boolean mask for rows matching this category
            mask = test4['freq'] == cat
            indices = test4.index[mask].tolist()

            # Pull out corresponding values in other columns
            vals1 = test4.loc[mask, 'power']
            median_power.append(np.median(vals1))
            mean_power.append(np.mean(vals1))

        #Create lists for saving--------------------
        time_list = [start_time] * len(median_power)
        station_list = [station] * len(median_power)
        data = {
            'mean_power': mean_power,
            'median_power': median_power,
            'frequency': unique,
            'time': time_list,
            'station':  station_list,
            }
        #%%
        psds = pd.DataFrame(data)
        psds_list.append(psds)

    #Save everything into single dataframe per station------------
    full_df = pd.concat(psds_list, ignore_index=True)
    station_psds.append(full_df)
#Save everything into single dataframe------------
full_df = pd.concat(station_psds, ignore_index=True)

'''
fig, ax = plt.subplots()
ax.plot(1/unique, median_power)
ax.set_xlabel('Frequency')
ax.set_ylabel('Power (dB)')
ax.grid(alpha = 0.3)
plt.xscale('log')
plt.show()
'''

# %%


# %%
