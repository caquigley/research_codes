#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from array_functions import pull_earthquakes, data_from_inventory, get_geometry
from obspy import read_inventory
from obspy import UTCDateTime
from matplotlib.transforms import blended_transform_factory
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth

net = '4E'
sta = 'HM*'
loc = '*'
chan = 'DHZ'
client = Client('EARTHSCOPE')
freq_min = 0.5#1
freq_max = 20#10

figpath = '/Users/cadequigley/Downloads/Research/deployment_array_design/datamine_paper_figures/'
array = 'homer' #homer, kodiak
deployment = 'd1'
min_dist = 0
max_dist = 250
min_mag = 4.0
max_mag = 6
plot_type = 'close' #close, far

#index = 2 # 3 Homer, 2 Kodiak
#xmin = 13 #15 HOmer, 13 Kodiak
#xmax = 23 #25 Homer, 23 Kodiak
hm_lat = 59.61711691923077
hm_lon = -151.14245023461538

#Pull earthquake of interest
df = pd.read_csv(figpath+'m2_earthquakes_hom_kod.csv')
dist_hm = []
for i in range(len(df)):
    tlat = df['latitude'].to_numpy()[i]
    tlon = df['longitude'].to_numpy()[i]
    dist, az, baz = gps2dist_azimuth(hm_lat, hm_lon, tlat, tlon)
    dist_hm.append(dist/1000)
df['hm_distance'] = dist_hm


if array == 'kodiak':
    index = 2
    xmin = 13
    xmax = 23
    normalization = 4
    df = pd.DataFrame(df[df['distance']<= max_dist ]) #1800
    df = pd.DataFrame(df[df['distance']>= min_dist ]) #1800
elif array == 'homer':
    index = 3
    xmin = 15
    xmax = 25
    normalization = 12
    df = pd.DataFrame(df[df['hm_distance']<= max_dist ]) #uncomment for hm
    df = pd.DataFrame(df[df['hm_distance']>= min_dist ]) #uncomment for hm

df = pd.DataFrame(df[df['magnitude']<= max_mag ]) #1800
df1 = pd.DataFrame(df[df['magnitude']>= min_mag ]) #1800


print('Number of events with specification:', len(df1))
df = df1.iloc[index]

eq_time = UTCDateTime(df['time_utc'])
eq_lat = df['latitude']
eq_lon = df['longitude']
event = df['event_id']
eq_mag = df['magnitude']
depth = df['depth']
if array == 'kodiak':
    eq_dist = df['distance']
elif array == 'homer':
    eq_dist = df['hm_distance']
START = eq_time
END = START +df['p_arrival'] + 30
print('Event:', event)
print('Mag:', eq_mag)
print('Depth:', depth)
#%%
#path = '/Users/cadequigley/Downloads/Research/deployment_array_design/'
#inv1 = read_inventory(path + array+'_'+deployment+'_station.xml')
remove_stations = []
keep_stations = []

inv = client.get_stations(network=net, station=sta, channel=chan,
                          location=loc, starttime=START,
                          endtime=END, level='channel')

(lat_list, lon_list, elev_list, full_station_list,
 start_d1_list, end_d1_list, num_channels_d1_list) = data_from_inventory(inv, remove_stations, keep_stations)



st = client.get_waveforms(net, sta, loc, chan, START, END, attach_response=True)

st.merge(fill_value='latest')
st.remove_sensitivity()
st.filter("bandpass", freqmin=freq_min, freqmax=freq_max, 
          corners=2, zerophase=True)
st.sort()
##%%
#Plot--------------------------------
fig, ax = plt.subplots(figsize = (10,15)) #(10,8)
if array == 'homer':
    color = 'maroon'
elif array == 'kodiak':
    color = 'navy'

trans = blended_transform_factory(ax.transAxes, ax.transData)

distance = []
for i in range(len(lat_list)):
    dist,baz,az = gps2dist_azimuth(lat_list[i], lon_list[i], eq_lat, eq_lon)
    ypos = dist/1000
    #ypos = 0.1*i
    distance.append(ypos)

for i in range(len(st)):
    tr = st[i]
    station = full_station_list[i]
    #dist,baz,az = gps2dist_azimuth(lat_list[i], lon_list[i], eq_lat, eq_lon)
    #ypos = dist/1000
    #ypos = 0.1*i
    #distance.append(ypos)
    ypos = distance[i]
    time_range = np.max(tr.times()) - np.min(tr.times())
    ax.plot(tr.times() , ypos+((-1*tr.data/(normalization*max(tr.data)))), #-1 accounts for flipping of axis later, 4 Kodiak, 12 Homer
                color = color, alpha = 0.8)
    ax.text(1.01, ypos, station, transform=trans, color = color, 
                fontweight = 'bold',fontsize = 10, ha="left", va="center")
    
#plt.axvline(x=0, color = 'red', linestyle = '--')
    
#plt.axvline(x = trigger_time, color = 'purple', linestyle = '--')
distance = np.array(distance)
ax.text(0.05, min(distance)-0.15,
            'Event: '+event+'; M'+str(eq_mag)+'; Depth: '+str(depth)+' km',
            transform = trans, fontsize = 15, fontweight = 'bold',
            color = 'black')
ax.set_xlabel('time since origin time (s)')
ax.set_ylabel('distance from earthquake (km)')
if plot_type =='far':
    ax.set_xlim(0,np.max(tr.times()))
elif plot_type == 'close':
        #ax.set_xlim(-6+time_range/2,6+time_range/2)
    #ax.set_xlim(30,50)
    ax.set_xlim(xmin,xmax)
ax.set_ylim(np.min(distance)-0.2, np.max(distance)+0.2)
ax.grid(alpha = 0.3)
ax.invert_yaxis()
#plt.savefig(figpath+'/kd_record_section.png', transparent=True, dpi= 720, 
            #bbox_inches='tight', pad_inches=0.1)
plt.show()
# %%

#Plot--------------------------------
fig, ax = plt.subplots(figsize=(10,15)) #(10,8)
if array == 'homer':
    color = 'maroon'
elif array == 'kodiak':
    color = 'navy'

trans = blended_transform_factory(ax.transAxes, ax.transData)

# Compute true distances first
distance = []
for i in range(len(lat_list)):
    dist, baz, az = gps2dist_azimuth(lat_list[i], lon_list[i], eq_lat, eq_lon)
    distance.append(dist/1000)
distance = np.array(distance)

# Get the sort order (closest to furthest)
order = np.argsort(distance)

spacing = 0.1  # evenly spaced km-equivalent between traces; adjust to taste

for rank, i in enumerate(order):
    tr = st[i]
    station = full_station_list[i]
    true_dist = distance[i]
    ypos = rank * spacing  # evenly spaced position based on rank, not true distance

    time_range = np.max(tr.times()) - np.min(tr.times())
    ax.plot(tr.times(), ypos + ((-1*tr.data/(normalization*max(tr.data)))), #4 kodiak, 12 homer
                color=color, alpha=0.8)
    ax.text(-0.06, ypos, f'{station}', transform=trans, color=color,
                fontweight='bold', fontsize=10, ha="left", va="center")

ax.text(0.01, -spacing*1.0,
            'Event: '+event+'; M'+str(eq_mag)+'; Depth: '+str(depth)+' km; Distance: '+str(round(eq_dist, 1))+' km',
            transform=trans, fontsize=15, fontweight='bold',
            color='black')
ax.set_xlabel('time since origin time (s)')
#ax.set_ylabel('station rank (closest to farthest)')
if plot_type == 'far':
    ax.set_xlim(0, np.max(tr.times()))
elif plot_type == 'close':
    ax.set_xlim(xmin, xmax)
ax.set_ylim(-spacing*1.5, (len(order)-1)*spacing + spacing*1)
ax.grid(alpha=0.3)
ax.invert_yaxis()
plt.yticks([])
plt.savefig(figpath+'/hm_record_section.png', transparent=True, dpi= 720, 
            bbox_inches='tight', pad_inches=0.1)
plt.show()
# %%
