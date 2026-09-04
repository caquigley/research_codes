#%%
import pandas as pd
import numpy as np
import obspy
import lts_array
import matplotlib.pyplot as plt

from obspy.clients.fdsn import Client
from obspy import read
import matplotlib.dates as mdates
from obspy import read_events
from obspy import read_inventory
from obspy import Stream
from obspy import UTCDateTime
from obspy.taup import TauPyModel
from obspy.core.util import AttribDict
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.signal.array_analysis import array_processing


#Station inputs--------------
net =  "9C"
sta = "2A*"
loc = "*"
chan = "SHZ"
client =  "EARTHSCOPE" #"EARTHSCOPE" #"EARTHSCOPE"
pull_local = False # True, False
mseed_path =  "./2A_earthquakes_mseeds/"
start_time = -10
end_time = 100 #180

#Array inputs----------------
processing = 'lts' #'fk', 'lts', 'ls'
freq_min = 0.5 #0.5
freq_max = 8 # 10, 20
window_length = 5 #5
window_overlap  = (window_length-0.25)/window_length

correction = 'empirical'

#Earthquake info---------------------
magnitude_min = 6.5 #5, 4
max_distance = 300 #260
min_distance = 180 #200

#FK inputs--------------
sll_x = -0.5#-1.0
slm_x = 0.5 #1.0
sll_y = -0.5 #-1.0
slm_y = 0.5 #1.0
sl_s = 0.01 #0.01
semb_thres = -1e9
vel_thres = -1e9
timestamp = "mlabday"
prewhiten = 0
win_frac = 0.25

alpha = 0.5
color_length = 15 #seconds
time_before = 2.5 #2.5 for POM/2A, 1 for 3A
baz_correction = 'single' #'single', 'function', None
single_correction = 13.5 #13.5 degrees 2A, -4.11 or 0 for 3A, 11.27 degrees for POM

#######Pull event from local dataset-------------------------
index = 0 #21, #20, 12, 400, 410, 200, 210, 230, 8 4 on 180 dataset
#df = pd.read_csv('/Users/cadequigley/repos/array_aggregator/2A_1000km_m3_lts__window_freq_map_fig.csv')
df = pd.read_csv('/Users/cadequigley/repos/array_aggregator/2A_2000km_m3_lts_6_window_freq_test.csv')
#%%
df = pd.DataFrame(df[df['distance']<= max_distance ])
df = pd.DataFrame(df[df['distance']>= min_distance ])
df = pd.DataFrame(df[df['magnitude'] >= magnitude_min])
#df = pd.DataFrame(df[df['backazimuth']> 190])
#df = pd.DataFrame(df[df['backazimuth']<270])
print(len(df), 'events greater than M', magnitude_min)

magnitude = df['magnitude'].to_numpy()[index]
distance = df['distance'].to_numpy()[index]
depth = df['depth'].to_numpy()[index]
origin_time = df['time_utc'].to_numpy()[index]
event_id = df['event_id'].to_numpy()[index]
real_backazimuth = df['backazimuth'].to_numpy()[index]
array_backazimuth = df['array_baz'].to_numpy()[index]
real_tracevel = df['trace_vel'].to_numpy()[index]
trigger_time = UTCDateTime(df['trigger_time'].to_numpy()[index])
trig_time = trigger_time - UTCDateTime(origin_time)
latitude = df['latitude'].to_numpy()[index]
longitude = df['longitude'].to_numpy()[index]
print('Trig time', trig_time)
print('Event_id:', event_id)
print('Magnitude:', magnitude)
print('Origin time:', origin_time)
print('Distance:', distance, 'km')
print('Depth:', depth, 'km')
print('Backazimuth:', real_backazimuth)
print('Array backazimuth:', array_backazimuth)
#%%
##Pull traces----------------------------
START = UTCDateTime(origin_time)+start_time
END = UTCDateTime(origin_time) + end_time
client = Client(client)

if pull_local == True:
    st = read(mseed_path+event_id+'.mseed')
    st = st.slice(START, END)
elif pull_local == False:

    st = Stream()
    try:
        st += client.get_waveforms(net, sta, loc, chan, START, END)
    except FDSNNoDataException:
        print(f"No data for station {sta}")
        
    except Exception as e:
        print(f"Error for station {sta}: {e}")


#%% Get inventory and lat/lon info
inv = client.get_stations(network=net, station=sta, channel=chan,
                            location=loc, starttime=START,
                            endtime=END, level='response') #'channel'

lat_list = []
lon_list = []
elev_list = []
staname = []
for network in inv:
    for station in network:
        for channel in station:
            lat_list.append(channel.latitude)
            lon_list.append(channel.longitude)
            staname.append(channel.code)
            elev_list.append(station.elevation)


#%%            
st.merge(fill_value='latest')
st.trim(START, END, pad='true', fill_value=0)
st.sort()
st.remove_sensitivity(inventory = inv)

st1 = st.copy()
# Filter the data
st.filter("bandpass", freqmin=freq_min, freqmax=freq_max, corners=2, zerophase=True)
st.taper(max_percentage=0.05)

# Run array processing-------------------------
if processing == 'lts' or processing == 'ls':
    if processing == 'lts':
        alpha = alpha
    elif processing == 'ls':
        alpha = 1
    (lts_vel, lts_baz, t, mdccm, stdict, sigma_tau, 
    conf_int_vel, conf_int_baz) = lts_array.ltsva(st, lat_list, lon_list, 
                                                window_length, window_overlap, 
                                                alpha)


    time_error = []
    for j in range(len(t)):
        matplotlib_time = t[j]
        x = mdates.num2date(matplotlib_time) 
        x = UTCDateTime(x)
        
        diff = x - START+start_time
            #diff = x-UTCDateTime(time1[i])-(WINDOW_LENGTH/2)
        time_error.append(diff)
    
    time_error = np.array(time_error)
    data = {

        'baz': lts_baz,
        'slow': 1/lts_vel,
        'trace_vel': lts_vel,
        'mdccm': mdccm,
        'conf_int_vel': conf_int_vel,
        'conf_int_baz': conf_int_baz,
        #'time': str(UTCDateTime(mdates.num2date(t))),
        'time': t,
        'time_since_origin': time_error
    }
    df = pd.DataFrame(data)

elif processing == 'fk':
    for l in range(len(st1)):  # Uses all stations in pd dataframe stations
        st1[l].stats.coordinates = AttribDict({'latitude': lat_list[l],
                                               'elevation': elev_list[l],
                                               'longitude': lon_list[l]})
    kwargs = dict(
        # slowness grid: X min, X max, Y min, Y max, Slow Step
        sll_x=sll_x, slm_x=slm_x, sll_y=sll_y, slm_y=slm_y, sl_s=sl_s,
        # sliding window properties
        win_len=window_length, win_frac = win_frac, #0.25#win_frac=1-WINDOW_OVERLAP,
        # frequency properties
        frqlow=freq_min, frqhigh=freq_max, prewhiten=prewhiten,
        # restrict output
        semb_thres=semb_thres, vel_thres=vel_thres, timestamp=timestamp,
        # start and end of analysis
        stime=START+1, etime = END-1
                )
    out = array_processing(st1, **kwargs)
    
    #OUTPUT FROM FK PROCESSING-------------------------------------------------
    array_out = pd.DataFrame(out, columns = ['time','relpow','abspow',
                                             'baz_obspy','array_slow'])
        

    #Convert times and baz to same scale as lts (UTC time, centered on window)-
    t = array_out['time'].to_numpy()
    baz_obspy = array_out['baz_obspy'].to_numpy()
        
    bazs = []
    time_error = []
    time_since_origin = []
    for j in range(len(t)):
        matplotlib_time = t[j]
        x = mdates.num2date(matplotlib_time) 
        x = UTCDateTime(x)
        #diff = (x-UTCDateTime(time_station))+(win_len/2)
        diff = str(x+(window_length/2)) #time centered on point
        time_error.append(diff)
        time_since_origin.append(UTCDateTime(diff) - UTCDateTime(origin_time))
        baz = baz_obspy[j]
        if baz <= 0:
            baz_correct = baz+360 #converts to all positive backazimuth
        else:
            baz_correct = baz
        bazs.append(baz_correct)
        
    time_error = np.array(time_error)

    fk_bazs = np.array(bazs)
    array_out['time'] = time_error
    array_out['baz'] = fk_bazs
    array_out['trace_vel'] = 1/array_out['array_slow'].to_numpy()
    array_out['time_since_origin'] = time_since_origin
    df = array_out
    #print(time_error)

    
# Add correction to data---------------------
a0 = -5.00919269
a1 = -16.57537395; b1 = -25.89907705
a2 = -17.23837615; b2 = -14.54117385
a3 = -3.74992031;  b3 = -7.58511478
a4 = 0.7111211;    b4 = -7.3153489
a5 = -0.41993486;  b5 = -4.77020138

def f(theta_deg):
    theta = np.deg2rad(theta_deg)
    return (a0 + a1*np.cos(1*theta) + b1*np.sin(1*theta)
               + a2*np.cos(2*theta) + b2*np.sin(2*theta)
               + a3*np.cos(3*theta) + b3*np.sin(3*theta)
               + a4*np.cos(4*theta) + b4*np.sin(4*theta)
               + a5*np.cos(5*theta) + b5*np.sin(5*theta))

def f_prime(theta_deg):
    # derivative w.r.t. theta_deg, chain rule through deg2rad
    theta = np.deg2rad(theta_deg)
    dtheta_ddeg = np.pi / 180
    dfdtheta = (-a1*1*np.sin(1*theta) + b1*1*np.cos(1*theta)
                -a2*2*np.sin(2*theta) + b2*2*np.cos(2*theta)
                -a3*3*np.sin(3*theta) + b3*3*np.cos(3*theta)
                -a4*4*np.sin(4*theta) + b4*4*np.cos(4*theta)
                -a5*5*np.sin(5*theta) + b5*5*np.cos(5*theta))
    return dfdtheta * dtheta_ddeg

def estimate_real_baz(baz_array_new, tol=1e-8, max_iter=50):
    baz_array_new = np.atleast_1d(baz_array_new).astype(float)
    x = baz_array_new.copy()  # initial guess: assume error is small, start at baz_array

    for _ in range(max_iter):
        g = x - f(x) - baz_array_new
        g_prime = 1 - f_prime(x)
        step = g / g_prime
        x = x - step
        if np.max(np.abs(step)) < tol:
            break

    return x % 360
#xvals = np.linspace(180,270, 40)
if baz_correction == 'function':
    df['baz_corrected'] = estimate_real_baz(df['baz'])
elif baz_correction == 'single':
     df['baz_corrected'] = df['baz']+single_correction
elif baz_correction == None:
     df['baz_corrected'] = df['baz']

#%%
#Plots-------------------

##########################################
#Trace-----------------
##########################################
tr = st1[0]
fig, ax = plt.subplots(figsize = (10,4))
ax.plot(tr.times()-10,tr.data,color = 'black')
ax.axvline(trig_time, color = 'blue', alpha = 0.6, linestyle = '--')
ax.axvline(distance/((distance/trig_time)*0.6), color = 'red', alpha = 0.6, linestyle = '--')
ax.grid(alpha = 0.3)
ax.set_xlabel('time since earthquake origin (seconds)')
ax.set_ylabel('velocity (m/s)')
ax.set_xlim(-5,90)
#plt.savefig('./69_rupture_trace.png', transparent=True, dpi = 720)
plt.savefig('/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/figure_components/69_rupture_trace.png', transparent=True, dpi = 720)
plt.show()

##########################################
#Backazimuth over time-----------------
##########################################

if processing == 'fk':
     color = df['relpow']
elif processing == 'ls' or processing == 'lts':
     color = df['mdccm']

#Pull out times to color
#df2 = df[(df['time_since_origin'] > 32.5) & (df['time_since_origin'] <= 50)]
df2 = df[(df['time_since_origin'] > trig_time-time_before) & (df['time_since_origin'] <= trig_time+color_length)]

fig, ax = plt.subplots(figsize = (10,4))
#ax.scatter(df['time_since_origin'], df['baz'], color = 'gray',)
#ax.scatter(df2['time_since_origin'], df2['baz'], c = df2['time_since_origin'], cmap = 'plasma')

sc = ax.scatter(df['time_since_origin'], df['baz'], c = color,cmap ='hot', linewidths = 0.2, edgecolors = 'black', s = 60)
ax.set_xlim(-5,90)
ax.set_ylim(0,360)
ax.grid(alpha = 0.3)
#ax.scatter(df2['new_time'], lts_baz + conf_int_baz, color='black', marker='_', linewidths=0.2, alpha=1)
#ax.scatter(df2['new_time'], lts_baz - conf_int_baz, color='black', marker='_', linewidths=0.2, alpha=1)
if processing == 'lts' or processing == 'ls':
    ax.scatter(df['time_since_origin'], df['baz'] + df['conf_int_baz'], color='black', marker='_', linewidths=0.2, alpha=1)
    ax.scatter(df['time_since_origin'], df['baz'] - df['conf_int_baz'], color='black', marker='_', linewidths=0.2, alpha=1)
    for e, d, c in zip(df['time_since_origin'], df['baz'], df['conf_int_baz']):
            square_y_top = d + c
            square_y_bottom = d - c
            ax.plot([e, e], [square_y_bottom, square_y_top], color='black', linestyle='--', alpha=1, linewidth = 0.2)
ax.set_xlabel("time since earthquake origin (seconds)")
ax.set_ylabel("backazimuth (degrees)")
ax.axhline(real_backazimuth, color = 'gray', alpha = 0.6, linestyle = '--')
ax.axvline(trig_time, color = 'blue', alpha = 0.6, linestyle = '--')
ax.axvline(distance/((distance/trig_time)*0.6), color = 'red', alpha = 0.6, linestyle = '--')
plt.colorbar(sc, label = 'relative power')
#plt.savefig('./69_rupture_long.png', transparent=True, dpi = 720)
plt.show()


##########################################
#Trace velocity over time-----------------
##########################################

fig, ax = plt.subplots(figsize = (10,4))
#ax.scatter(df['time_since_origin'], df['trace_vel'], color = 'gray',)
#ax.scatter(df2['time_since_origin'], df2['trace_vel'], c = df2['time_since_origin'], cmap = 'plasma')

sc = ax.scatter(df['time_since_origin'], df['trace_vel'], c = color,cmap ='hot', linewidths = 0.2, edgecolors = 'black', s = 60)
ax.set_xlim(-5,90)
ax.set_ylim(-1,15)
ax.grid(alpha = 0.3)
#ax.scatter(df2['new_time'], lts_baz + conf_int_baz, color='black', marker='_', linewidths=0.2, alpha=1)
#ax.scatter(df2['new_time'], lts_baz - conf_int_baz, color='black', marker='_', linewidths=0.2, alpha=1)
if processing == 'lts' or processing == 'ls':
    ax.scatter(df['time_since_origin'], df['trace_vel'] + df['conf_int_vel'], color='black', marker='_', linewidths=0.2, alpha=1)
    ax.scatter(df['time_since_origin'], df['trace_vel'] - df['conf_int_vel'], color='black', marker='_', linewidths=0.2, alpha=1)
    for e, d, c in zip(df['time_since_origin'], df['trace_vel'], df['conf_int_vel']):
            square_y_top = d + c
            square_y_bottom = d - c
            ax.plot([e, e], [square_y_bottom, square_y_top], color='black', linestyle='--', alpha=1, linewidth = 0.2)
ax.set_xlabel("Time since earthquake origin (seconds)")
ax.set_ylabel("Trace velocity (km/s)")
ax.axhline(real_tracevel, color = 'gray', alpha = 0.6, linestyle = '--')
ax.axvline(trig_time, color = 'blue', alpha = 0.6, linestyle = '--')
ax.axvline(distance/((distance/trig_time)*0.6), color = 'red', alpha = 0.6, linestyle = '--')
plt.colorbar(sc, label = 'relative power')
#plt.savefig('./69_rupture_long.png', transparent=True, dpi = 720)
plt.show()

##########################################
#Trace velocity zoomed in-----------------
##########################################

### TRACE VELOCITY shortened time------------------------------------------

fig, ax = plt.subplots(figsize = (10,4))


#ax.set_xlim(25,60)
#ax.set_ylim(4,11)
ax.set_xlim(trig_time - 10, trig_time + 30)
ax.set_ylim(real_tracevel-4, real_tracevel+4)
ax.grid(alpha = 0.3)
#ax.scatter(df2['new_time'], lts_baz + conf_int_baz, color='black', marker='_', linewidths=0.2, alpha=1)
#ax.scatter(df2['new_time'], lts_baz - conf_int_baz, color='black', marker='_', linewidths=0.2, alpha=1)
if processing == 'lts' or processing == 'ls':
    ax.scatter(df['time_since_origin'], df['trace_vel'] + df['conf_int_vel'], color='black', marker='_', linewidths=0.2, alpha=1)
    ax.scatter(df['time_since_origin'], df['trace_vel'] - df['conf_int_vel'], color='black', marker='_', linewidths=0.2, alpha=1)
    for e, d, c in zip(df['time_since_origin'], df['trace_vel'], df['conf_int_vel']):
            square_y_top = d + c
            square_y_bottom = d - c
            ax.plot([e, e], [square_y_bottom, square_y_top], color='black', linestyle='--', alpha=1, linewidth = 0.2)
ax.scatter(df['time_since_origin'], df['trace_vel'], color = 'gray',)
sc = ax.scatter(df2['time_since_origin'], df2['trace_vel'], c = df2['time_since_origin'], cmap = 'hot_r', linewidths = 0.2, edgecolors = 'black', s = 60) #plasma


ax.axhline(real_tracevel, color = 'gray', alpha = 0.6, linestyle = '--')
ax.axvline(trig_time, color = 'blue', alpha = 0.6, linestyle = '--')
ax.axvline(distance/((distance/trig_time)*0.6), color = 'red', alpha = 0.6, linestyle = '--')
ax.set_xlabel("time since earthquake origin (seconds)")
ax.set_ylabel("trace velocity (km/s)")
ax.set_xlim(25,60)
plt.colorbar(sc, label = 'seconds since p-arrival')

#plt.savefig('/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/figure_components/69_rupture_trace.png', transparent=True, dpi = 720)
plt.show()



##########################################
#Backazimuth zoomed in-----------------
##########################################

fig, ax = plt.subplots(figsize = (10,4))

#ax.set_xlim(25,60)
ax.set_xlim(trig_time - 10, trig_time + 30)
ax.set_ylim(real_backazimuth-30, real_backazimuth+30)
#ax.set_ylim(190,235)
ax.grid(alpha = 0.3)
#ax.scatter(df2['new_time'], lts_baz + conf_int_baz, color='black', marker='_', linewidths=0.2, alpha=1)
#ax.scatter(df2['new_time'], lts_baz - conf_int_baz, color='black', marker='_', linewidths=0.2, alpha=1)
if processing == 'lts' or processing == 'ls':
    ax.scatter(df['time_since_origin'], df['baz_corrected'] + df['conf_int_baz'], color='black', marker='_', linewidths=0.2, alpha=1)
    ax.scatter(df['time_since_origin'], df['baz_corrected'] - df['conf_int_baz'], color='black', marker='_', linewidths=0.2, alpha=1)
    for e, d, c in zip(df['time_since_origin'], df['baz_corrected'], df['conf_int_baz']):
            square_y_top = d + c
            square_y_bottom = d - c
            ax.plot([e, e], [square_y_bottom, square_y_top], color='black', linestyle='--', alpha=1, linewidth = 0.2)
ax.scatter(df['time_since_origin'], df['baz_corrected'], color = 'gray',)
sc = ax.scatter(df2['time_since_origin'], df2['baz_corrected'], c = df2['time_since_origin'], cmap = 'hot_r', linewidths = 0.2, edgecolors = 'black', s = 60)
ax.axhline(real_backazimuth, color = 'gray', alpha = 0.6, linestyle = '--')
ax.axvline(trig_time, color = 'blue', alpha = 0.6, linestyle = '--')
ax.axvline(distance/((distance/trig_time)*0.6), color = 'red', alpha = 0.6, linestyle = '--')

ax.set_xlabel("time since earthquake origin (seconds)")
ax.set_ylabel("backazimuth (degrees)")
ax.set_xlim(25,60)
ax.set_ylim(205,250)
#plt.colorbar(sc, label = 'seconds since p-arrival')

plt.savefig('/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/figure_components/69_rupture_short.png', transparent=True, dpi = 720)
plt.show()

#%%
#####################
#-----EMPIRICAL CORRECTION---------
################################
'''
a0 = -5.00919269
a1 = -16.57537395
b1 = -25.89907705
a2 = -17.23837615
b2 = -14.54117385
a3 = -3.74992031
b3 = -7.58511478
a4 = 0.7111211
b4 = -7.3153489
a5 = -0.41993486
b5 = -4.77020138

theta = df['baz'].to_numpy()
#theta = np.linspace(0,360, 360)
#theta = real_backazimuth
#theta = array_backazimuth
theta = np.deg2rad(theta)
#df['baz_corrected'] = df['baz']+ (a0 + a1*np.cos(1*theta) + b1*np.sin(1*theta)
                    #+ a2*np.cos(2*theta) + b2*np.sin(2*theta)
                    #+ a3*np.cos(3*theta) + b3*np.sin(3*theta)
                    #+ a4*np.cos(4*theta) + b4*np.sin(4*theta)
                    #+ a5*np.cos(5*theta) + b5*np.sin(5*theta))

y = (a0 + a1*np.cos(1*theta) + b1*np.sin(1*theta)
                    + a2*np.cos(2*theta) + b2*np.sin(2*theta)
                    + a3*np.cos(3*theta) + b3*np.sin(3*theta)
                    + a4*np.cos(4*theta) + b4*np.sin(4*theta)
                    + a5*np.cos(5*theta) + b5*np.sin(5*theta))
'''


#fig,ax = plt.subplots()

#ax.plot(np.rad2deg(theta),y)
#ax.axvline(real_backazimuth)
#ax.grid()
#plt.show()

#df['baz_corrected'] = df['baz'] + y
########################################
#---------PYGMT MAP--------------------
# ##################################### 
# 
####################    PLOT ONTO MAP      ###################################
#%%
from array_functions import pull_earthquakes
import pygmt


min_mag = 3
earthquakes = pull_earthquakes(str(latitude), str(longitude), '100', str(START), str(START + 60*60*24*5), str(min_mag), '2A',
                    'iasp91')
#%%
df = pd.DataFrame(df[df['time_since_origin'] >= trig_time-time_before])
df = pd.DataFrame(df[df['time_since_origin'] <= trig_time+color_length])
df = df.dropna()
df = df.iloc[::4]

array_lats = [53.6949, 53.779, 53.8566]
array_lons = [-166.7333,-166.2131,-166.4161]
sizes = [500,500,500]


def transform_degrees(degree):
    # Shift from north (0) to east (90)
    transformed_degree = (degree - 90) % 360
    return transformed_degree
baz_real_pygmt = 360 - transform_degrees(df['baz_corrected'].to_numpy()) #-12

lengths = np.ones(len(baz_real_pygmt))*8
if sta == '2A*':
     array_num = 0
elif sta == '3A*':
     array_num = 1
elif sta == 'POM*':
     array_num = 2
vec_lats = np.ones(len(baz_real_pygmt))*array_lats[array_num]
vec_lons = np.ones(len(baz_real_pygmt))*array_lons[array_num]
timing = df['time_since_origin'].to_numpy()

color = timing
data = np.column_stack([vec_lons,vec_lats, color, baz_real_pygmt,lengths])



CPT_Option = '/Users/cadequigley/Downloads/Research/AEC_BaseMap.cpt'


shallow = earthquakes[earthquakes['depth'] <= 35]
intermediate = earthquakes[(earthquakes['depth'] > 35) & (earthquakes['depth'] <= 100)]
deep = earthquakes[earthquakes['depth'] > 100]

#pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", MAP_FRAME_AXES='lrbt') # Highlevel formatting (no ticks, no labels)

pygmt.config(FORMAT_GEO_MAP="ddd.x") # Highlevel formatting (no ticks, no labels)
#Define projection and grid map resolution (for BOTH maps)


region=[-180,-160,48,60]
region_rect = "-170.5/51.5/-166/54.1r"

#rectangular designation for plotted mat
#projection = 'S210/90/8i'
projection = "M4i"
projection = "M"+str(array_lons[0])+"/"+str(array_lats[0])+"/12c"

run_topo = True
##---Begin basemap w/ only AK topography---##

if run_topo == True:
    # Load topography
    load_grid = pygmt.datasets.load_earth_relief(resolution='15s', region=region, registration=None, data_source='igpp', use_srtm=False) #15s
    
    #pyGMT basemap with topography figure
    fig = pygmt.Figure()
    #pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", MAP_FRAME_AXES='lrbt', MAP_FRAME_PEN='1p') #Formatting
    pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain",
                    MAP_FRAME_PEN='1p') #Formatting, MAP_FRAME_AXES='lrbt'
   

    fig.basemap(frame=True, region=region_rect, projection=projection,
                    map_scale="jBR+w50k+o0.5c/0.5c+f+lkm")
    #fig.coast(dcw="US.AK+p0.25p")
    
    #Define outline and color pallete of basemap
        #fig.coast( shorelines=True, water='#C6E2EE', borders="1/1p,black") #frame=[fig_title]
    dgrid = pygmt.grdgradient(grid=load_grid, radiance=[270,30])

    pygmt.makecpt(cmap=CPT_Option)  #, series=[-1.5, 0.3, 0.01])
    #fig.grdimage(grid=load_grid, shading='+a300+nt0.8', cmap=True)
    fig.grdimage(grid=load_grid, shading='+a300+nt0.8', cmap=True, #0.8
                    transparency = 50) #60
    fig.coast(water=None, borders="10/10p,black", 
                  shorelines="1/0.5p,black")
    
    #Plot mini arrays-----------------------------------------------------------------
    fig.plot(x = array_lons,
         y = array_lats,
         style = "i1c",pen = '0.5p,#3e000d', size = sizes, fill = 'cyan4')

    #Plot earthquakes------------------------------------------------------------
    fig.plot(x=intermediate['longitude'], y=intermediate['latitude'], size=0.05*(1.5**intermediate['magnitude']),
         style="cc", pen='0.5p,#3e000d', fill = 'gray40') #gold1, gray40, #EBB41E

    fig.plot(x=shallow['longitude'], y=shallow['latitude'], size=0.05*(1.5**shallow['magnitude']),
         style="cc", pen='0.5p,#3e000d', fill = 'gray66') #firebrick, gray66, #FB0006
    fig.plot(x = [longitude], y = [latitude], size = [0.05*(1.5**magnitude)],
             style = 'cc', pen = '0.5p,#3e000d', fill = 'red' )

    #fig.plot(x=deep['longitude'], y=deep['latitude'], size=0.05*(1.5**deep['magnitude']),
         #style="cc", pen='0.5p,#3e000d', fill = 'gray14') #4D0010, gray14

    ####PLOT VECTORS-------------------
    #pygmt.makecpt(cmap='plasma', series = [22,40])
    pygmt.makecpt(cmap='hot', series = [trig_time-time_before,trig_time+color_length], reverse = True )
    fig.plot(data=data, style = "v0.7c+ea", fill = "+z", cmap=True, pen = '0.5p,+z')

    

    #for i_vector in range(len(data)):
        #fig.plot(data = data[i_vector],
            #style="v0.5c+ea",
            #cmap = True,
            #zvalue = timing[i_vector],
            #fill = timing[i],
        #fill="royalblue",
            #pen="1.0p, +z")
    
    #Plot text---------------------------------------------
    #fig.text(text=["HOM", "KOD"], x=[-151.1412100+0.7, -152.3516100+0.7], y=array_lats,font = "18p,Helvetica-Bold,black")

    
    #pygmt.makecpt(cmap='hot', series = [0,220])
    
    fig.colorbar(frame="xaf+lTime since earthquake origin (s)")
    #fig.savefig('/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/figure_components/rupture_testing_3A.png', transparent=True, dpi=720)
    fig.show(dpi=720)  



# %%
