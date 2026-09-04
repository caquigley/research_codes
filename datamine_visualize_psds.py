#%%
import pandas as pd
import matplotlib.pyplot as plt
from obspy.signal import spectral_estimation
from obspy import UTCDateTime
from array_functions import (data_from_inventory, get_geometry)
from obspy import read_inventory
import numpy as np

freq_low = 1 #1
freq_high = 10 #10

vmin = -155 #-155
vmax = -144 #-144

figpath = '/Users/cadequigley/Downloads/Research/deployment_array_design/datamine_paper_figures/'
#hm = pd.read_csv(figpath+'hm_psds.csv')
#kd = pd.read_csv(figpath+'kd_psds.csv')

hm_stations = ['HM01', 'HM02', 'HM03', 'HM04', 'HM05', 'HM06', 'HM07', 'HM08',
            'HM09','HM10', 'HM11', 'HM12', 'HM13','HM14', 'HM15', 'HM16', 
            'HM17','HM18','HM19','HM20','HM21','HM22', 'HM23', 'HM24',
            'HM25', 'HM26']


kd_stations = ['KD01', 'KD02', 'KD03', 'KD04', 'KD05', 'KD06', 'KD07', 'KD08',
            'KD09','KD10', 'KD11', 'KD12', 'KD13','KD14', 'KD15', 'KD16', 
            'KD17','KD18','KD19','KD20','KD21','KD22', 'KD23', 'KD24',
            'KD25', 'KD26']

bear_stations = ['HM01', 'HM02', 'HM05', 'HM05', 'HM06', 'HM22']
bear_start = ['2025-10-21', '2025-09-09', '2025-10-03', '2025-10-20',
               '2025-10-03', '2025-10-17']
bear_end = ['2025-11-13', '2025-10-12', '2025-10-12', '2025-11-09','2025-10-12',
            '2025-11-11']

#%%
#df = pd.read_csv(figpath+'hm_psds.csv')
df = pd.read_csv(figpath+'hm_psds_all_dates.csv')

df['date'] = pd.to_datetime(df['time'])

# Start by keeping every row
keep = pd.Series(True, index=df.index)

for station, start, end in zip(bear_stations, bear_start, bear_end):

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    remove = (
        (df['station'] == station) &
        (df['date'] >= start) &
        (df['date'] <= end)
    )

    keep &= ~remove

# Remove the bear-affected rows
hm = df[keep]
#%%
#kd = pd.read_csv(figpath+'kd_psds.csv')
kd = pd.read_csv(figpath+'kd_psds_all_dates.csv')
array_dfs = [hm, kd]
fig, ax = plt.subplots(nrows= 1, ncols = 2, figsize = (12,4))

colors = ['darkorange', 'purple']
arrays = ['hm', 'kd']
arrays2 = ['HM', 'KD']
array_list = [hm_stations, kd_stations]
nhnm_freq, nhnm_pow = spectral_estimation.get_nhnm()
nlnm_freq, nlnm_pow = spectral_estimation.get_nlnm()

power_list = []
for k in range(len(arrays)):

    #df = pd.read_csv('/Users/cadequigley/repos/array_aggregator/'+arrays[k]+'_psds.csv')
    #df = pd.read_csv(figpath+arrays[k]+'_psds.csv')
    df = array_dfs[k]
    
    
    
    
    
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
        ax[k].plot(1/temp['frequency'], temp['median_power'], color = colors[k], alpha = 0.2)

    ax[k].plot(1/np.array(unique), median_power, color = colors[k], alpha = 1,
             linewidth = 2, label  = arrays2[k])

    ax[k].axvspan(1/freq_high, 1/freq_low,  color = 'gray',alpha = 0.1)
    ax[k].plot(nhnm_freq, nhnm_pow, color = 'blue', alpha = 0.3, 
        linestyle = 'dashdot', label = 'NHNM')
    ax[k].plot(nlnm_freq, nlnm_pow, color = 'gray', alpha = 0.3,
               linestyle = 'dashdot', label = 'NLNM')
    ax[k].set_xlabel('period (s)')
    ax[k].set_ylabel('power (dB)')
    ax[k].set_ylim(-170, -90)
    #ax[k].set_xlim(0.03, 200)
    ax[k].set_xlim(1/122, 200)
    ax[k].grid(alpha = 0.3)
    ax[k].axvline(1/freq_low, color = 'red', linestyle = '--', alpha = 0.5)
    ax[k].axvline(1/freq_high, color = 'red', linestyle = '--', alpha = 0.5)
    ax[k].legend(loc = "lower right")
    ax[k].set_xscale('log')
#plt.legend(loc="lower right")
#plt.xscale('log')
#fig.savefig('/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/figure_components/noise_plot.pdf')
#plt.savefig(figpath+'/kd_hm_psds.png', transparent=True, dpi= 720)
plt.show()

#%%
#################################
#-------Spatial Plots---------------------
#Take average power for 1-10 Hz
stations = []
mean_powers = []
for i in range(len(power_list)):
    temp1 = power_list[i]
    for k in range(len(temp1)):
        temp = temp1[k]
        temp = pd.DataFrame(temp[temp['frequency']>= freq_low ])
        temp = pd.DataFrame(temp[temp['frequency']<= freq_high ])
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
#%%

arrays = ['kodiak', 'homer']
deployment = 'd1'
dfs = []
for i in range(len(arrays)):
    path = '/Users/cadequigley/Downloads/Research/deployment_array_design/'
    inv1 = read_inventory(path + arrays[i]+'_'+deployment+'_station.xml')
    remove_stations = []
    keep_stations = []

    (lat_list, lon_list, elev_list, full_station_list,
    start_d1_list, end_d1_list, num_channels_d1_list) = data_from_inventory(inv1, remove_stations, keep_stations)

    output = get_geometry(lat_list, lon_list, elev_list, return_center = True)
    xpos = []
    ypos = []
    for i in range(len(output)-1):
        xpos.append((output[i][0])*1000)
        ypos.append((output[i][1])*1000)

    data = {
            'xpos': xpos,
            'ypos':  ypos,
            'lat': lat_list,
            'lon': lon_list,
            'station': full_station_list
                }
    locations = pd.DataFrame(data)
    dfs.append(locations)

locations = pd.concat(dfs, ignore_index=True)
merged  = pd.merge(locations, power_data, on="station", how="outer")
#%%
fig,ax = plt.subplots(ncols = 2, nrows = 1, figsize = (12,6), constrained_layout=True)


for i in range(len(arrays2)):
    df = merged[merged['station'].str.startswith(arrays2[i])].reset_index(drop=True)





    sc = ax[i].scatter(df['xpos'], df['ypos'], c = df['mean_power'],
            cmap = 'viridis', vmin = vmin, vmax = vmax, marker = '^', 
            linewidths = 1, s = 300,  edgecolors = 'black')
    ax[i].grid(alpha = 0.3)
    ax[i].set_xlabel('x position (m)')
    ax[i].set_ylabel('y position (m)')
    ax[i].set_aspect('equal', adjustable='box')
    ax[i].set_xlim(-1300,1300)
    ax[i].set_ylim(-1300,1300)
#plt.colorbar(sc, label = 'power (dB)')
fig.colorbar(sc, ax=ax[:2], shrink=0.4, 
             location='bottom', label = 'power (dB)')
#fig.tight_layout()
#plt.savefig(figpath+'/kd_hm_psds_plan_view.png', transparent=True, dpi= 720)
plt.show()
# %%

#----------------------------------
#####HOMER MAP#####################
#---------------------------------
import pygmt
from array_maps_pygmt import basemap_cpt

df = merged[merged['station'].str.startswith('HM')].reset_index(drop=True)
station_lons = df['lon']
station_lats = df['lat']
color_data = df['mean_power']

left = -151.173
right = -151.107
bottom = 59.605
top = 59.63

# Small padding around DEM
region = [
    left - 0.005,
    right + 0.005,
    bottom - 0.005,
    top + 0.005,
]

projection = "M12c"

fig = pygmt.Figure()

pygmt.config(
    FORMAT_GEO_MAP="ddd.xx",
    MAP_FRAME_TYPE="plain",
    MAP_FRAME_PEN="1p",
)

# Elevation CPT
pygmt.makecpt(
    cmap="gray",
    series=[0, 200, 20], #[0,300,20]
    reverse=True,
)

# DEM
fig.grdimage(
    grid="/Users/cadequigley/Downloads/dem_wgs84_hm.tif",
    region=region,
    projection=projection,
    cmap=True,
    shading=True,
)

# Alaska coastline/boundary
fig.coast(
    dcw="US.AK+p0.7p",
    borders="1/1p,black",
)

# Map frame + scale
fig.basemap(
    region=region,
    projection=projection,
    frame=True,
    map_scale="jBR+w1k+o0.5c/0.5c+f+lkm",
)
pygmt.makecpt(cmap='hot', series = [vmin,vmax])
        #pygmt.makecpt(cmap="SCM/lajolla", series=[0, 360])
        
fig.plot(x= station_lons, y= station_lats, 
        size=[0.25]*26,
        style="i0.7c", pen='0.5p,#3e000d',cmap=True, fill = color_data)
fig.colorbar(frame="xaf+lMean power (dB)")
#fig.savefig(figpath+'homer_dem.png', transparent=True, dpi=720)
fig.show()


#----------------------------------
#####Kodiak MAP#####################
#---------------------------------



df = merged[merged['station'].str.startswith('KD')].reset_index(drop=True)
station_lons = df['lon']
station_lats = df['lat']
color_data = df['mean_power']

left = -152.38 #-152.3905002
right = -152.32 #-152.3085199
bottom = 57.425 #57.4161717
top = 57.453 #57.4569283

# Small padding around DEM
region = [
    left - 0.005,
    right + 0.005,
    bottom - 0.005,
    top + 0.005,
]

projection = "M12c"

fig = pygmt.Figure()

pygmt.config(
    FORMAT_GEO_MAP="ddd.xx",
    MAP_FRAME_TYPE="plain",
    MAP_FRAME_PEN="1p",
)

# Elevation CPT
pygmt.makecpt(
    cmap="gray",
    series=[0, 200, 20], #[0,300,20]
    reverse=True,
)

# DEM
fig.grdimage(
    grid="/Users/cadequigley/Downloads/dem_wgs84.tif",
    region=region,
    projection=projection,
    cmap=True,
    shading=True,
)

# Alaska coastline/boundary
fig.coast(
    dcw="US.AK+p0.7p",
    borders="1/1p,black",
)

# Map frame + scale
fig.basemap(
    region=region,
    projection=projection,
    frame=True,
    map_scale="jBR+w1k+o0.5c/0.5c+f+lkm",
)
pygmt.makecpt(cmap='hot', series = [vmin,vmax])
        #pygmt.makecpt(cmap="SCM/lajolla", series=[0, 360])
        
fig.plot(x= station_lons, y= station_lats, 
        size=[0.25]*26,
        style="i0.7c", pen='0.5p,#3e000d',cmap=True, fill = color_data)
fig.colorbar(frame="xaf+lmean power (dB)")
#fig.savefig(figpath+'kodiak_dem.png', transparent=True, dpi=720)
fig.show()






'''
###TERMINAL COMMAND TO CONVERT ARCTIC DEM TO USABLE FORMAT###
conda activate arrayseis

gdalwarp \
    -s_srs "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +datum=WGS84 +units=m" \ 
    -t_srs EPSG:4326 \
    "/Users/cadequigley/Downloads/51_06_2_1_2m_v4.1/51_06_2_1_2m_v4.1_dem.tif" \    
    "/Users/cadequigley/Downloads/dem_wgs84.tif"


gdalwarp \
    -s_srs "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +datum=WGS84 +units=m" \ 
    -t_srs EPSG:4326 \
    "/Users/cadequigley/Downloads/50_08_1_2_2m_v4.1/50_08_1_2_2m_v4.1_dem.tif" \    
    "/Users/cadequigley/Downloads/dem_wgs84_hm.tif"


# Initialize figure
fig = pygmt.Figure()

# Plot the DEM file directly with a color map and hillshading
fig.grdimage(
    grid="/Users/cadequigley/Downloads/51_06_2_1_2m_v4.1/51_06_2_1_2m_v4.1_dem.tif",
    region=[
        -152.4205002,
        -152.3085199,
        57.4131717,
        57.4589283,
    ],  # Optional: specify your bounding region [xmin, xmax, ymin, ymax]
    projection="M6i",
    cmap="geo",
    shading=True,  # Automatically calculates and applies hillshading
)

# Add a color bar
#fig.colorbar(position="jBC+w5c/0.5c+o0/1c", frame=True)

# Show the map
fig.show()
'''
# %%


# %%
