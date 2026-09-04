# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import obspy
from obspy.signal import spectral_estimation
from obspy import UTCDateTime
from scipy.optimize import curve_fit

from obspy.clients.fdsn import Client
from obspy import read_inventory

from array_functions import (data_from_inventory, get_geometry, fourier5,
                             pull_earthquakes, check_num_stations, 
                             stations_available_generator,
                             array_time_window, moveout_time, grab_preprocess,
                             least_trimmed_squares, triggers, fk_obspy)
from array_figures import (baz_error_spatial, 
                           slow_error_spatial)
nhnm_freq, nhnm_pow = spectral_estimation.get_nhnm()
nlnm_freq, nlnm_pow = spectral_estimation.get_nlnm()

fig, ax = plt.subplots(figsize = (6,4))

ax.axvspan(1/10, 1,  color = 'gray',alpha = 0.2)
ax.plot(nhnm_freq, nhnm_pow, color = 'blue', alpha = 0.3, linestyle = 'dashdot', label = 'NHNM')
#ax.plot(nlnm_freq, nlnm_pow, color = 'blue', alpha = 0.5)

array_2A = ['2A01', '2A02', '2A03', '2A04', '2A05', '2A06', '2A07', 
            '2A09','2A10','2A12', '2A13', '2A14', '2A15']

array_3A = ['3A01', '3A02', '3A03', '3A05', '3A06', '3A07', '3A08', 
            '3A09','3A11', '3A12', '3A14']

array_POM = ['POM01', 'POM02', 'POM03', 'POM04',  'POM08',  
            'POM09','POM11','POM12', 'POM14', 'POM15', 'POM16', 'POM19']


colors = ['darkorange', 'purple', 'black']
arrays = ['2A', 'POM', '3A']
array_list = [array_2A, array_POM, array_3A]

power_list = []
for k in range(len(arrays)):

    df = pd.read_csv('/Users/cadequigley/repos/array_aggregator/'+arrays[k]+'_psds.csv')
    stations = array_list[k]


    #Pull out for each station------------------------
    df_list = []
    for i in range(len(stations)):
        temp = pd.DataFrame(df[df['station']== stations[i] ])
        unique = temp['frequency'].unique()

        median_power = []
        mean_power = []
        for cat in unique:
                # Boolean mask for rows matching this category
            mask = temp['frequency'] == cat
            indices = temp.index[mask].tolist()

                # Pull out corresponding values in other columns
            vals1 = temp.loc[mask, 'median_power']
            median_power.append(np.median(vals1))
            mean_power.append(np.mean(vals1))

        station_list = [stations[i]] * len(median_power)
        data = {
                'mean_power': mean_power,
                'median_power': median_power,
                'frequency': unique,
                'station':  station_list,
                }
        psds = pd.DataFrame(data)
        df_list.append(psds)
    power_list.append(df_list)
    #Pull out average for all stations------------------------
    full_df = pd.concat(df_list, ignore_index=True)
    unique = temp['frequency'].unique()
    median_power = []
    mean_power = []
    for cat in unique:
                # Boolean mask for rows matching this category
        mask = full_df['frequency'] == cat
        indices = full_df.index[mask].tolist()

                # Pull out corresponding values in other columns
        vals1 = full_df.loc[mask, 'median_power']
        median_power.append(np.median(vals1))
        mean_power.append(np.mean(vals1))

    #Plot-----------------------------------


    for i in range(len(df_list)):
        temp = df_list[i]
        ax.plot(1/temp['frequency'], temp['median_power'], color = colors[k], alpha = 0.2)

    ax.plot(1/np.array(unique), median_power, color = colors[k], alpha = 1,
             linewidth = 2, label  = arrays[k])


ax.set_xlabel('period (s)')
ax.set_ylabel('power (dB)')
ax.set_ylim(-145, -90)
ax.set_xlim(0.03, 200)
ax.grid(alpha = 0.3)
ax.axvline(1, color = 'red', linestyle = '--', alpha = 0.5)
ax.axvline(1/10, color = 'red', linestyle = '--', alpha = 0.5)
plt.legend(loc="lower right")
plt.xscale('log')
#fig.savefig('/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/figure_components/noise_plot.pdf')
plt.show()

# %%


#############################################
#-------Trigger/power bar plot----------------
arrays = ['2A', '3A', 'POM']
path = '/Users/cadequigley/repos/array_aggregator/'
paths = [path+'2A_1000km_m3_lts__window_freq_map_fig.csv',
         path+'3A_2000km_m3_lts_6_window_freq_test.csv',
         path+'POM_2000km_m3_lts_6_window_freq_test.csv']

event_number = []
taup_number = []
trigger_number = []
low_pow_number = []
for i in range(len(paths)):
    df = pd.read_csv(paths[i]) #_const_lfreq.csv')

    df = pd.DataFrame(df[df['distance']<= 400])
    print(arrays[i]+' Number of events:', len(df))
    taup = pd.DataFrame(df[df['trigger_type']== 'Taup'])
    print(arrays[i]+' Number of taup events:', len(taup))
    #print('Number of triggered events:', len(df) - len(taup))
    trigger = pd.DataFrame(df[df['trigger_type']!= 'Taup'])
    print(arrays[i]+' Number of triggered events:', len(trigger))
    low_pow = pd.DataFrame(trigger[trigger['mdccm']<= 0.35])
    print(arrays[i]+' Number of low power events:', len(low_pow))

    event_number.append(len(df))
    taup_number.append(len(taup))
    trigger_number.append(len(trigger))
    low_pow_number.append(len(low_pow))

import matplotlib.pyplot as plt

arrays = ('2A', '3A', 'POM')
array_stats = {
    'STA/LTA triggered': trigger_number,
    'Missed events': taup_number,
    'Low power events': low_pow_number,
}


fig, ax = plt.subplots(layout='constrained', figsize = (6,4))
ax.set_prop_cycle(color=['gold', 'firebrick', 'darkblue'])
res = ax.grouped_bar(array_stats, tick_labels=arrays, group_spacing=1,
                     edgecolor='black',
                     #color=['#2196F3', '#FF5722', '#4CAF50'],
                     linewidth=0.8,       
                     alpha=0.8)
for container in res.bar_containers:
    ax.bar_label(container, padding=3)

# Add some text for labels, title, etc.
ax.set_ylabel('Number of events')
#ax.set_title('Penguin attributes by species')
#ax.legend(loc='upper left', ncols=3)
ax.grid(axis = 'y', alpha = 0.3)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncols=3)
ax.set_ylim(0, 550)
#plt.legend()
#fig.savefig('/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/figure_components/trigger_noise_plot.pdf')
plt.show()
# %%
#################################
#-------Spatial Plots---------------------

#Take average power for 1-10 Hz
stations = []
mean_powers = []
for i in range(len(power_list)):
    temp1 = power_list[i]
    for k in range(len(temp1)):
        temp = temp1[k]
        temp = pd.DataFrame(temp[temp['frequency']>= 1 ])
        temp = pd.DataFrame(temp[temp['frequency']<= 10 ])
        mean_power = np.mean(temp['median_power'].to_numpy())
        station = temp['station'].to_numpy()[0]
        stations.append(station)
        mean_powers.append(mean_power)
        #print(mean_power)
        #print(station)

data = {
        'mean_power': mean_powers,
        'station':  stations
            }
power_data = pd.DataFrame(data)

net = '9C'
sta = '2A*'
chan = 'SHZ'
loc = '*'
remove_stations = []
keep_stations = []

client_str = 'EARTHSCOPE'
client = Client(client_str)
        
inv = client.get_stations(network=net, station=sta, channel=chan,
                                    location=loc, 
                                    starttime=UTCDateTime('2015-10-01'),
                                    endtime=UTCDateTime('2015-10-02'), 
                                    level='response')
    
    #Pull station information out of inventory-----
(lat_list, lon_list, elev_list, station_d1_list,
start_d1_list, end_d1_list, num_channels_d1_list) = data_from_inventory(
        inv,
        remove_stations,
        keep_stations)

geometry = get_geometry(lat_list, lon_list, elev_list, return_center = False)

xpos = list(geometry[:,0])
ypos = list(geometry[:,1])

data = {
        'xpos': xpos,
        'ypos':  ypos,
        'station': station_d1_list
            }
locations = pd.DataFrame(data)

merged  = pd.merge(locations, power_data, on="station", how="outer")

fig,ax = plt.subplots(figsize = (6,6))
vmax = -122
vmin = -140

sc = ax.scatter(merged['xpos'], merged['ypos'], c = merged['mean_power'],
           cmap = 'viridis', vmin = vmin, vmax = vmax, marker = '^', linewidths = 1, s = 300,  edgecolors = 'black')
ax.grid(alpha = 0.3)
ax.set_xlabel('x position (km)')
ax.set_ylabel('y position (km)')
ax.set_aspect('equal', adjustable='box')
plt.colorbar(sc, label = 'power (dB)')
plt.show()

##########################################
#-----POM/3A comparison plots---------------
# %%

df = pd.read_csv('/Users/cadequigley/repos/array_aggregator/3A_1000km_m3_lts__fig5.csv')
#df = pd.DataFrame(df[df['distance']<= 400])
df = pd.DataFrame(df[df['trigger_type']!= 'Taup'])
df = pd.DataFrame(df[df['mdccm']>= 0.35])

save_fig = False
fig_path = 'wawa'

#color_data = df['relpow']

color_data = df['mdccm']
    
    #color_data = df['conf_int_baz']
    #color_data = df['magnitude']
color_label = 'MDCCM'
model_data = []
    
baz_error_spatial(df["backazimuth"], df["baz_error"], model_data,
            color_data, color_label, niazi=False, plot_fourier = False, 
                      plot_anisotropic = False, save=False,
            path= "/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/figure_components/3A_fig5_baz_error_spatial.pdf")   


slow_error_spatial(df["backazimuth"], df["slow_error"], model_data,
            color_data, color_label, niazi=False, save=save_fig,
            path=fig_path + "slow_error_spatial.png")

def quantile_range(array): #numpy array
    ah = np.quantile(array, 0.95)
    aw = np.quantile(array, 0.05)
    return ah-aw

baz_error = df['baz_error']
#baz_error = df['slow_error']
angles = df['backazimuth']
values = baz_error

bins = np.arange(0, 361, 10)

medians = []
counts = []
bin_centers = []
min_count = 2 #10

for i in range(len(bins)-1):

    mask = (angles >= bins[i]) & (angles < bins[i+1])

    vals = values[mask]

    counts.append(len(vals))
    bin_centers.append((bins[i] + bins[i+1]) / 2)

    if len(vals) >= min_count:
        medians.append(np.median(vals))
    else:
        medians.append(np.nan)

medians = np.array(medians)
counts = np.array(counts)
bin_centers = np.array(bin_centers)
    
    
    #drop nan values
mask = ~np.isnan(medians)
medians = medians[mask]
counts = counts[mask]
bin_centers = bin_centers[mask]
        
    #Fourier fit------------------------------
    #theta = np.deg2rad(baz)
theta = np.deg2rad(bin_centers)

#Fourier fit------------------------------
#theta = np.deg2rad(df['backazimuth'].to_numpy())

#params, _ = curve_fit(fourier5, theta, baz_error)
params, _ = curve_fit(fourier5, theta, medians)
## Smooth curve
#theta_fit = np.linspace(0, 2*np.pi, 500)
theta = np.deg2rad(df['backazimuth'])
y_fit = fourier5(theta, *params)

print('Rupture value:', fourier5(np.deg2rad(235), *params)) #232 POM, 235 3A

#Keep error between -180 to 180
baz_error_temp = baz_error - y_fit
baz_corrected = ((baz_error_temp + 180) % 360) - 180
df['baz_corrected'] = baz_corrected

baz_error_temp = df['array_baz'] + y_fit
array_baz_correct = ((baz_error_temp + 180) % 360) - 180
df['array_baz_correct'] = array_baz_correct

qrange = quantile_range(baz_corrected)
print('Quantile range:', qrange)
print('STD:', np.std(baz_corrected))
print('Mean abs error:', np.mean(np.abs(baz_corrected)))
print('Quantile range before correction:', quantile_range(baz_error))
print('STD before correction:', np.std(baz_error))
print('Mean abs error before correction:', np.mean(np.abs(baz_error)))
# %%
