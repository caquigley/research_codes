#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from array_maps_pygmt import pygmt_array_earthquakes
from array_functions import pull_earthquakes, data_from_inventory, get_geometry
from obspy import read_inventory
from obspy import UTCDateTime

max_rad = '1000'
min_mag = '2' #'1'
array_name = 'KD'
velocity_model = 'ak135'

df = pd.read_csv('/Users/cadequigley/Downloads/Research/deployment_array_design/kodiak_mseed_completeness.csv')
df3 = pd.read_csv('/Users/cadequigley/Downloads/Research/deployment_array_design/homer_mseed_completeness.csv')
df['start_datetime'] = pd.to_datetime(df['start_mseed_d1'], utc=True)
df['end_datetime'] = pd.to_datetime(df['end_mseed_d2'], utc=True)

end_mseed_d2 = df3['end_mseed_d2'].to_numpy()
end_mseed_d2[5] = end_mseed_d2[6]
df3['end_mseed_d2'] = end_mseed_d2
df3['start_datetime'] = pd.to_datetime(df3['start_mseed_d1'], utc=True)
df3['end_datetime'] = pd.to_datetime(df3['end_mseed_d2'], utc=True)


#start = str(UTCDateTime((df['start_datetime'].min())))
#end = str(UTCDateTime((df3['end_datetime'].max())))
start = str(UTCDateTime('2025-09-08'))
end = str(UTCDateTime('2025-11-15'))

array = 'kodiak'
deployment = 'd1'
path = '/Users/cadequigley/Downloads/Research/deployment_array_design/'
inv1 = read_inventory(path + array+'_'+deployment+'_station.xml')
remove_stations = []
keep_stations = []

(lat_list, lon_list, elev_list, full_station_list,
 start_d1_list, end_d1_list, num_channels_d1_list) = data_from_inventory(inv1, remove_stations, keep_stations)

output = get_geometry(lat_list, lon_list, elev_list, return_center = True)
kd_lat = str(output[-1][1])
kd_lon = str(output[-1][0])

array = 'homer'
deployment = 'd1'
figpath = '/Users/cadequigley/Downloads/Research/deployment_array_design/datamine_paper_figures/'
inv1 = read_inventory(path + array+'_'+deployment+'_station.xml')
remove_stations = []
keep_stations = []

(lat_list, lon_list, elev_list, full_station_list,
 start_d1_list, end_d1_list, num_channels_d1_list) = data_from_inventory(inv1, remove_stations, keep_stations)

output = get_geometry(lat_list, lon_list, elev_list, return_center = True)
hm_lat = str(output[-1][1])
hm_lon = str(output[-1][0])
print(hm_lat)
print(hm_lon)

df = pull_earthquakes(kd_lat, kd_lon, max_rad, start, end, min_mag, 
                        array_name, velocity_model)
#%%
df.to_csv(figpath+'m2_earthquakes_hom_kod.csv')
# %%
df = df.sort_values(by ='magnitude', ascending=False)

array_lats = [float(kd_lat), float(hm_lat)]
array_lons = [float(kd_lon), float(hm_lon)]
array_names = [array_name]
array_names = []
earthquake_lats = df['latitude'].to_numpy()
earthquake_lons = df['longitude'].to_numpy()
earthquake_mags = df['magnitude'].to_numpy()
earthquake_depths = df['depth'].to_numpy()
    
#figpath = '/Users/cadequigley/Downloads/Research/'
pygmt_array_earthquakes(array_lats, array_lons, array_names, 
                        earthquake_lats,earthquake_lons, 
                        earthquake_mags, earthquake_depths,
                        save=False, 
                        path = figpath+'earthquake_map_reference.png')

# %%
