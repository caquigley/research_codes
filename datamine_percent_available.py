#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from obspy import UTCDateTime

##############################################
#__________KODIAK DATA COMPLETENESS_____________
##############################################
figpath = '/Users/cadequigley/Downloads/Research/deployment_array_design/'

df = pd.read_csv('/Users/cadequigley/Downloads/Research/deployment_array_design/kodiak_mseed_completeness.csv')
# %%
stations = df['station_name'].to_numpy()
#cpu_1 = [(0, 3), (3.5, 1), (5, 5)]

df['start_datetime'] = pd.to_datetime(df['start_mseed_d1'], utc=True)
df['end_datetime'] = pd.to_datetime(df['end_mseed_d2'], utc=True)


start_dep = df['start_datetime'].min()
end_dep = df['end_datetime'].max()
print(start_dep)
print(end_dep)

start_dep = UTCDateTime(start_dep)
end_dep = UTCDateTime(end_dep)

#df3 = pd.read_csv('/Users/cadequigley/Downloads/Research/deployment_array_design/kodiak_mseed_completeness.csv')

df5 = df.copy()
df5 = df5.fillna(0)
df5 = df5.sort_values(by = 'station_name')
#df5 = df5.dropna()
times_start = df5['start_mseed_d1'].to_numpy()
times_end = df5['end_mseed_d1'].to_numpy()
times_start2 = df5['start_mseed_d2'].to_numpy()
times_end2 = df5['end_mseed_d2'].to_numpy()
elapsed_d1 = df5['elapsed_time_d1'].to_numpy()
elapsed_d2 = df5['elapsed_time_d2'].to_numpy()
elapsed_inter = df5['missing_time_hours'].to_numpy()
stations = df5['station_name'].to_numpy()

elapsed_start = df5['missing_time_start'].to_numpy()
elapsed_end = df5['missing_time_end'].to_numpy()
#bear_time_d1 = df5['bear_removal_time_d1'].to_numpy()
#bear_time_d2 = df5['bear_removal_time_d2'].to_numpy()

#Set up start times
matplot_begin = []
matplot_end = []
matplot_start1 = []
matplot_start2 = []
matplot_start3 = []

##Duration of first and second deployment
duration1 = []
duration2 = []
##Missing time 
missing_start = []
missing_end = []
inter = []


no_data = []
for i in range(len(times_start)):
    ##Missing time start-------------
    wa = str(start_dep)
    matplot_begin.append(dt.datetime(int(wa[0:4]),int(wa[5:7]), int(wa[8:10]), int(wa[11:13]),int(wa[14:16]),int(wa[17:19])))
    missing_start.append(dt.timedelta(hours = elapsed_start[i]))
    ##Missing time end-------------
    wa = str(times_end2[i])
    matplot_end.append(dt.datetime(int(wa[0:4]),int(wa[5:7]), int(wa[8:10]), int(wa[11:13]),int(wa[14:16]),int(wa[17:19])))
    missing_end.append(dt.timedelta(hours = elapsed_end[i]))
    ##First segment-----------------
    wa = str(times_start[i])
    matplot_start1.append(dt.datetime(int(wa[0:4]),int(wa[5:7]), int(wa[8:10]), int(wa[11:13]),int(wa[14:16]),int(wa[17:19])))
    duration1.append(dt.timedelta(hours = elapsed_d1[i]*24))
    ##Between deployments-----------------
    wa = str(times_end[i])
    matplot_start2.append(dt.datetime(int(wa[0:4]),int(wa[5:7]), int(wa[8:10]), int(wa[11:13]),int(wa[14:16]),int(wa[17:19])))
    inter.append(dt.timedelta(hours = elapsed_inter[i]))
    ##Second segment-----------------
    if times_start2[i] == '0': #stations that don't have any data-----------------
        wa = str(times_end[i])
        matplot_start3.append(dt.datetime(int(wa[0:4]),int(wa[5:7]), int(wa[8:10]), int(wa[11:13]),int(wa[14:16]),int(wa[17:19])))
        duration2.append(dt.timedelta(hours = 31*24))
        no_data.append(1)
    else:
        wa = str(times_start2[i])
        matplot_start3.append(dt.datetime(int(wa[0:4]),int(wa[5:7]), int(wa[8:10]), int(wa[11:13]),int(wa[14:16]),int(wa[17:19])))
        duration2.append(dt.timedelta(hours = elapsed_d2[i]*24))
        no_data.append(0)

        
    
#fig, ax = plt.subplots(figsize = (10,6))
fig,ax = plt.subplots(figsize = (8,6))
# broken_barh(xranges, (ymin, height))
for i in range(len(stations)):
    work0 = [(matplot_begin[i],missing_start[i])]
    work1 = [(matplot_start1[i],duration1[i])]
    work2 = [(matplot_start2[i],inter[i])]
    work3 = [(matplot_start3[i], duration2[i])]
    work4 = [(matplot_end[i],missing_end[i])]
    #nonwork = 
    ax.broken_barh(work0, (-0.2+i, 0.4), color = '#E69F00')#'firebrick') #time between d1 and d2
    ax.broken_barh(work1, (-0.2+i, 0.4), color ='cornflowerblue' )#'gray' ) #working time d1
    ax.broken_barh(work2, (-0.2+i, 0.4), color = '#E69F00')#'firebrick') #time between d1 and d2
    ax.broken_barh(work3, (-0.2+i, 0.4), color = 'mediumblue', alpha = 0.8)#'royalblue', alpha = 0.8) #working time d2
    ax.broken_barh(work4, (-0.2+i, 0.4), color = '#E69F00')#'firebrick') #time between d1 and d2
    
fig.autofmt_xdate()
#ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
#ax.set_xlim(0, 10)
ax.set_xlim(pd.to_datetime("2025-09-09"),
            pd.to_datetime("2025-11-18"))

ax.set_yticks(range(len(stations)),labels=list(stations))
ax.invert_yaxis()
#plt.savefig(figpath+'/datamine_paper_figures/data_availability_kodiak_array.png', transparent=True, dpi = 720)
plt.show()

##############################################
#__________KODIAK DATA COMPLETENESS MAP_______
##############################################

#%%
from array_functions import (data_from_inventory, get_geometry)
from obspy import read_inventory
vmin = 35
vmax = 100
array = 'kodiak'
deployment = 'd2'
path = '/Users/cadequigley/Downloads/Research/deployment_array_design/'
inv1 = read_inventory(path + array+'_'+deployment+'_station.xml')
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
fig,ax = plt.subplots() #c = color_data, cmap = 'plasma_r'
sc = ax.scatter(xpos, ypos, c = df['data_completeness_all']*100, vmin = vmin, vmax= vmax,cmap = 'Oranges',
           marker = '^', linewidths = 1, s = 300,  edgecolors = 'black') #cornflowerblue
#ax.scatter(bhz_x, bhz_y, color = 'gray', marker = '^', linewidths = 1, s = 300, edgecolors= 'black', alpha= 0.5)
#ax.scatter(0,0, color = 'red', s= 100)
for i in range(len(full_station_list)):
    ax.text(xpos[i], ypos[i]+60, full_station_list[i], weight = 'semibold', horizontalalignment = 'center')
ax.set_xlabel("x position (m)")
ax.set_ylabel("y position (m)")
ax.grid(alpha = 0.1)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-1300,1300)
ax.set_ylim(-1300, 1300)
plt.colorbar(sc, label = 'data completeness (%)')
#plt.savefig(figpath+'/datamine_paper_figures/kodiak_percent_available.png', transparent=True, dpi= 720)
plt.show()
# %%
##############################################
#__________HOMER DATA COMPLETENESS _______
##############################################

df3 = pd.read_csv('/Users/cadequigley/Downloads/Research/deployment_array_design/homer_mseed_completeness.csv')
end_mseed_d2 = df3['end_mseed_d2'].to_numpy()
end_mseed_d2[5] = end_mseed_d2[6]
df3['end_mseed_d2'] = end_mseed_d2
df3['start_datetime'] = pd.to_datetime(df3['start_mseed_d1'], utc=True)
df3['end_datetime'] = pd.to_datetime(df3['end_mseed_d2'], utc=True)


start_dep = df3['start_datetime'].min()
end_dep = df3['end_datetime'].max()
print(start_dep)
print(end_dep)
start_dep = UTCDateTime(start_dep)
end_dep = UTCDateTime(end_dep)

df5 = df3.copy()
df5 = df5.fillna(0)
df5 = df5.sort_values(by = 'station_name')
#df5 = df5.dropna()
times_start = df5['start_mseed_d1'].to_numpy()
times_end = df5['end_mseed_d1'].to_numpy()
times_start2 = df5['start_mseed_d2'].to_numpy()
times_end2 = df5['end_mseed_d2'].to_numpy()
elapsed_d1 = df5['elapsed_time_d1'].to_numpy()
elapsed_d2 = df5['elapsed_time_d2'].to_numpy()
elapsed_inter = df5['missing_time_hours'].to_numpy()
stations = df5['station_name'].to_numpy()
bear_time_d1 = df5['bear_removal_time_d1'].to_numpy()
bear_time_d2 = df5['bear_removal_time_d2'].to_numpy()

elapsed_start = df5['missing_time_start'].to_numpy()
elapsed_end = df5['missing_time_end'].to_numpy()

matplot_begin = []
matplot_end = []
missing_start = []
missing_end = []

matplot_start1 = []
matplot_start2 = []
matplot_start3 = []
duration1 = []
duration2 = []
inter = []
bear_start = []
bear_duration = []
bear_elapsed1 = []
bear_start2 = []
bear_duration2 = []
bear_elapsed2 = []
no_data = []
for i in range(len(times_start)):
    ##Missing time start-------------
    
    wa = str(start_dep)
    matplot_begin.append(dt.datetime(int(wa[0:4]),int(wa[5:7]), int(wa[8:10]), int(wa[11:13]),int(wa[14:16]),int(wa[17:19])))
    missing_start.append(dt.timedelta(hours = elapsed_start[i]))
    ##Missing time end-------------
    if str(times_end2[i]) == '0':
        matplot_end.append(0)
        missing_end.append(0)

    else:
        wa = str(times_end2[i])
        matplot_end.append(dt.datetime(int(wa[0:4]),int(wa[5:7]), int(wa[8:10]), int(wa[11:13]),int(wa[14:16]),int(wa[17:19])))
        missing_end.append(dt.timedelta(hours = elapsed_end[i]))
    
    ##First segment-----------------
    wa = str(times_start[i])
    matplot_start1.append(dt.datetime(int(wa[0:4]),int(wa[5:7]), int(wa[8:10]), int(wa[11:13]),int(wa[14:16]),int(wa[17:19])))
    duration1.append(dt.timedelta(hours = elapsed_d1[i]*24))
    ##Between deployments-----------------
    wa = str(times_end[i])
    matplot_start2.append(dt.datetime(int(wa[0:4]),int(wa[5:7]), int(wa[8:10]), int(wa[11:13]),int(wa[14:16]),int(wa[17:19])))
    inter.append(dt.timedelta(hours = elapsed_inter[i]))
    ##Second segment-----------------
    if str(times_start2[i]) == '0': #stations that don't have any data-----------------
        wa = str(times_end[i])
        matplot_start3.append(dt.datetime(int(wa[0:4]),int(wa[5:7]), int(wa[8:10]), int(wa[11:13]),int(wa[14:16]),int(wa[17:19])))
        duration2.append(dt.timedelta(hours = 31*24))
        no_data.append(1)
    else:
        wa = str(times_start2[i])
        matplot_start3.append(dt.datetime(int(wa[0:4]),int(wa[5:7]), int(wa[8:10]), int(wa[11:13]),int(wa[14:16]),int(wa[17:19])))
        duration2.append(dt.timedelta(hours = elapsed_d2[i]*24))
        no_data.append(0)
    ##Bear times D1------------
    if bear_time_d1[i] == '0':
        bear_start.append(0)
        bear_duration.append(0)
        bear_elapsed1.append(0)
    else:
        wa = bear_time_d1[i]
        bear_start.append(dt.datetime(int(wa[0:4]),int(wa[5:7]), int(wa[8:10]), int(wa[11:13]),int(wa[14:16])))
        bear_elapsed = abs(UTCDateTime(wa) - UTCDateTime(times_end[i]))/(60*60*24)
        bear_duration.append(dt.timedelta(hours = bear_elapsed*24))
        bear_elapsed1.append(bear_elapsed)
    ##Bear times D2------------
    if bear_time_d2[i] == '0':
        bear_start2.append(0)
        bear_duration2.append(0)
        bear_elapsed2.append(0)
    else:
        wa = bear_time_d2[i]
        bear_start2.append(dt.datetime(int(wa[0:4]),int(wa[5:7]), int(wa[8:10]), int(wa[11:13]),int(wa[14:16])))
        bear_elapsed = abs(UTCDateTime(wa) - UTCDateTime(times_end2[i]))/(60*60*24)
        bear_duration2.append(dt.timedelta(hours = bear_elapsed*24))
        bear_elapsed2.append(bear_elapsed)
        

#fig, ax = plt.subplots(figsize = (10,6))
fig,ax = plt.subplots(figsize = (8,6))
# broken_barh(xranges, (ymin, height))
for i in range(len(stations)):
    work0 = [(matplot_begin[i],missing_start[i])]
    work1 = [(matplot_start1[i],duration1[i])]
    work2 = [(matplot_start2[i],inter[i])]
    work3 = [(matplot_start3[i], duration2[i])]
    work4 = [(matplot_end[i],missing_end[i])]
    #nonwork = 
    ax.broken_barh(work0, (-0.2+i, 0.4), color = '#E69F00') #time between d1 and d2
    ax.broken_barh(work1, (-0.2+i, 0.4), color = 'cornflowerblue' ) #working time d1
    ax.broken_barh(work2, (-0.2+i, 0.4), color = '#E69F00') #time between d1 and d2
    ax.broken_barh(work3, (-0.2+i, 0.4), color = 'mediumblue', alpha = 0.8) #working time d2
    ax.broken_barh(work4, (-0.2+i, 0.4), color = '#E69F00') #time between d1 and d2
    #Add bear removal coloring----------------------
    if bear_duration[i] == 0:
        wawa = 0
    else:
        work4 = [(bear_start[i], bear_duration[i])]
        ax.broken_barh(work4, (-0.2+i, 0.4), color = 'black') #working time d2
        
    if bear_duration2[i] == 0:
        wawa =0
    else:
        work6 = [(bear_start2[i], bear_duration2[i])]
        ax.broken_barh(work6, (-0.2+i, 0.4), color = 'black') #working time d2
    #Add no data coloring--------------------------
    if no_data[i] == 0:
        wawa = 0
    else:
        work5 = work3
        ax.broken_barh(work5, (-0.2+i, 0.4), color = 'red') #working time d2

fig.autofmt_xdate()
#ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
#ax.set_xlim(0, 10)
ax.set_xlim(pd.to_datetime("2025-09-07"),
            pd.to_datetime("2025-11-14"))

ax.set_yticks(range(len(stations)),labels=list(stations))
ax.invert_yaxis()
#plt.savefig(figpath+'/datamine_paper_figures/data_availability_homer_array.png', transparent=True, dpi = 720)
plt.show()
# %%
##############################################
#__________HOMER DATA COMPLETENESS MAP_______
##############################################

array = 'homer'
deployment = 'd1'
path = '/Users/cadequigley/Downloads/Research/deployment_array_design/'
inv1 = read_inventory(path + array+'_'+deployment+'_station.xml')
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
fig,ax = plt.subplots() #c = color_data, cmap = 'plasma_r'
sc = ax.scatter(xpos, ypos, c = df5['data_completeness_all']*100, vmin = vmin, vmax= vmax, cmap = 'Oranges',
           marker = '^', linewidths = 1, s = 300,  edgecolors = 'black') #cornflowerblue
#ax.scatter(bhz_x, bhz_y, color = 'gray', marker = '^', linewidths = 1, s = 300, edgecolors= 'black', alpha= 0.5)
#ax.scatter(0,0, color = 'red', s= 100)
for i in range(len(full_station_list)):
    ax.text(xpos[i], ypos[i]+60, full_station_list[i], weight = 'semibold', horizontalalignment = 'center')
ax.set_xlabel("x position (m)")
ax.set_ylabel("y position (m)")
ax.grid(alpha = 0.1)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-1300,1300)
ax.set_ylim(-1300, 1300)
plt.colorbar(sc, label = 'data completeness (%)')
#plt.savefig(figpath+'/datamine_paper_figures/homer_percent_available.png', transparent=True, dpi= 720)
plt.show()

# %%
