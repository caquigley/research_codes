#%%

import pandas as pd

from obspy.core import UTCDateTime

from obspy.clients.fdsn import Client

from obspy.core import UTCDateTime
from obspy import read_inventory

#Functions needed for code----------------------------
from array_functions import (data_from_inventory, get_geometry, pull_earthquakes,
                             check_num_stations, stations_available_generator,
                             array_time_window, moveout_time, grab_preprocess,
                             least_trimmed_squares, triggers, fk_obspy)
from array_figures import baz_error_spatial, slow_error_spatial
from array_maps_pygmt import pygmt_array_earthquakes, pygmt_baz_error


'''
Conducts array analysis for a set array and number of events in the vicinity
of the array. This can be used to determine how well an array is performing,
and how errors are occuring spatially.
    
Parameters:
    use_full_deployment: whether to use full time window deployment was out (True or False)
    start_d1_list: list of start times for each station
    end_d1_list: list of end times for each station
    starttime: specified starttime, will use if use_full_deployment = True
    endtime: speficied endtime, will use if use_full_deployment = True

Returns: 
    df: dataframe containing earthquake information, array output parameters,
    plots: baz error, slowness error
        
        
    '''

#from array_functions import rotate_data

###############################
#----------INPUTS---------------
###############################

#station/array inputs----------
net = '9C' #9C, 4E, 5E, UW, XG
sta = 'POM*' #'2A*', '3A*', 'POM*', S1**, UIL*, LC*, IL*
loc = '*' #0
chan = 'SHZ' #SHZ, DHZ, HHZ
client = 'IRIS' #IRIS, GEOFON, path, #if 'path', create new variable path =
starttime = '2015-06-23' #'2015-10-01' , '2011-05-11' '2025-09-07'
endtime = '2016-04-30'#'2015-10-02' , '2013-05-01', '2025-11-13'
min_stations = 10 # if you only want times with all stations, list the number of stations
remove_stations =  ['POM06', 'POM07', 'POM18']#['3A10', '3A15'] #['POM06', 'POM07', 'POM18'] #['3A10', '3A15'] # ['POM06', 'POM07', 'POM18']
keep_stations = [] 
array_name = 'POM' #2A, 3A, POM, KD, HM, S1, UIL
use_full_deployment = False #if True, searches for full deployment length in inventory and finds all events
path_to_inventory = None #if inventory object is stored locally
save_events = False #save the dataframe to CSV or not
save_stations = False #save station info
#mseed info
save_mseed = True #save mseeds
#mseed_path = '/Users/cadequigley/Downloads/Research/deployment_array_design/POM_earthquakes_mseeds/'
mseed_path = './POM_earthquakes_mseeds/'
mseed_length = 120 #seconds, centered on expected p-arrival
#Earthquake inputs----------
min_mag = '3.0' #minimum magnitude
max_rad = '400' #maximum radius from arrays
velocity_model = 'ak135' #iasp91, pavdut, scak, ak135, #fix japan_1d

#Array processing inputs---------------
processing = 'fk' #ls, fk, lts
FREQ_MIN = 0.5 #0.5 (Cade)
FREQ_MAX = 10.0 #10 (Cade)
WINDOW_LENGTH = 2.5 #seconds
#WINDOW_LENGTH = [2.5, 4.5]
WINDOW_STEP = 0.25
window_start = -1 #1 second before trigger

# STA/LTA inputs-------------------

timing = 'trigger' #'power', 'trigger', NEED TO FIX POWER
min_triggers = min_stations // 3 #minimum station triggers to associate
#min_triggers = 3
ptolerance = 5 #seconds, +/- around p-arrival
multiple_triggers = 'closest' #'closest', 'peak', 'first', which trigger to choose if multiple
no_triggers = 'max mdccm' #'max mdccm', 'taup', method to handle no triggers

#Following inputs representative of EPIC parameters
trig_freq_min = 1
trig_freq_max = 10
short_window = 0.05 # 0.05 (EPIC), 2.5 (Cade)
long_window = 5 #  5(EPIC), 30 (Cade)
on_threshold = 20 # 20 (EPIC), 2.5 (Cade)
off_theshold = 5 # 1, 5 epic

#Inputs for FK array processing---------

sll_x=-1.0 # X min, X max, Y min, Y max, Slow Step
slm_x=1.0 # X max
sll_y=-1.0 # Y min
slm_y=1.0 # Y max
sl_s=0.03 # Slow Step
semb_thres=-1e9
vel_thres=-1e9
timestamp='mlabday'
prewhiten = 0

#%%
###############################
#----------PROCESSING-----------
###############################

#Pull inventory-----------------------
#------------------------------------------------
if client == 'path':
    inv = read_inventory(path_to_inventory) #need to add something at some point about
else:
    #client = Client(client, user="caquigley@alaska.edu", password="U9sWxXLREK4FsdUX", debug = True)
    client = Client(client)
    inv = client.get_stations(network=net, station=sta, channel=chan,
                                location=loc, starttime=UTCDateTime(starttime),
                                endtime=UTCDateTime(endtime), level='response') #level = 'channel'
#Pull station information out of inventory
(lat_list, lon_list, elev_list, station_d1_list,
 start_d1_list, end_d1_list, num_channels_d1_list) = data_from_inventory(inv, 
                                                                         remove_stations, 
                                                                         keep_stations)

#Check if enough stations present to continue
check = check_num_stations(min_stations, station_d1_list)

#Save stations for later
data = {
        'station': station_d1_list,
        'lat': lat_list,
        'lon': lon_list,
        'elevation': elev_list}

station_info = pd.DataFrame(data) 


#Pull earthquakes-----------------------
#------------------------------------------------

# Get center of array--------
output = get_geometry(lat_list, lon_list, elev_list, return_center = True)
origin_lat = str(output[-1][1])
origin_lon = str(output[-1][0])
# Get expected moveout time across array--------
moveout = moveout_time(output)

#Pull earthquakes during deployment
start, end = array_time_window(use_full_deployment, start_d1_list, end_d1_list,
                               starttime, endtime)
df = pull_earthquakes(origin_lat, origin_lon, max_rad, start, end, min_mag, 
                      array_name, velocity_model)
print('Number of earthquakes >'+min_mag+' within '+max_rad+' km:', len(df))


#Create station availability lists-----------------------
#------------------------------------------------
earthquake_time = df['time_utc'].to_numpy()
earthquake_names = df['event_id'].to_numpy()

(stations_lists, 
 stations_available) = stations_available_generator(earthquake_time, 
                                                    station_d1_list, 
                                                    start_d1_list, end_d1_list)

### Drop events that don't have enough stations present--------------
bad_idx = [i for i, v in enumerate(stations_available) if v < min_stations]
keep_idx = [i for i, v in enumerate(stations_available) if v >= min_stations]

stations_available = [stations_available[i] for i in keep_idx]
stations_lists = [stations_lists[i] for i in keep_idx]
df = df.drop(index=bad_idx)
df = df.reset_index(drop = True)

print('Station lists for each earthquake created. New earthquake number:', len(df))
#%%
###Loop over all events---------------------------------

event_ids = df['event_id'].to_numpy()
eq_depths = df['depth'].to_numpy()
mag = df['magnitude'].to_numpy()
eq_lats = df['latitude'].to_numpy()
eq_lons = df['longitude'].to_numpy()
eq_time = df['time_utc'].to_numpy()
expected_parrival = df['p_arrival'].to_numpy()
eq_baz = df['backazimuth'].to_numpy()
eq_slow = df['slowness'].to_numpy()
eq_distance = df['distance'].to_numpy()
array_data_list = []

#Handles case for only single input given. If multiple inputs, it will be a list
if isinstance(FREQ_MAX, (float, int)):
    FREQ_MAX = [FREQ_MAX]
if isinstance(FREQ_MIN, (float,int)):
    FREQ_MIN = [FREQ_MIN]
if isinstance(WINDOW_LENGTH, (float,int)):
    WINDOW_LENGTH = [WINDOW_LENGTH]

#Loop through window lengths
for window in range(len(WINDOW_LENGTH)):

    window_length = WINDOW_LENGTH[window]
    WINDOW_OVERLAP = (window_length-WINDOW_STEP)/window_length #0.25s between each window
#Loop through frequencies
    for freq in range(len(FREQ_MAX)):
        freq_min = FREQ_MIN[freq]
        freq_max = FREQ_MAX[freq]


        print('Starting analysis for', window_length, 's window and '+str(freq_min)+'-'+str(freq_max), ' Hz bandpass filter')

        #%%
        #Loop through events
        for event in range(len(df)):
            try:
                print("Starting", event_ids[event], 'Ml', mag[event], eq_time[event])
                stations = stations_lists[event] #pull out stations available for each event
                eq_slow_real = eq_slow[event]
                eq_baz_real = eq_baz[event]
                event_id = event_ids[event]

                #Pull out one minute on either side of expected arrival time
                START = UTCDateTime(eq_time[event])+expected_parrival[event]- (mseed_length/2) 
                END = START + mseed_length

                ###Grab and preprocess data----------------------------
                (st, stations, sta_lats, 
                sta_lons, sta_elev) = grab_preprocess(stations, station_info, inv, 
                                                    net, loc, chan, min_stations, 
                                                    START, END, client, array_name,
                                                    event_id, mseed_path, save_mseed)
                #%%
                
                
                st1 = st.copy() #Pulling this out for FK processing

                ###Finding triggers---------------------------------
                if timing == 'trigger': #use sta/lta triggers
                    (st, trigger, peak, length, 
                    trigger_type, trigger_time, 
                    START_new, END_new)= triggers(st, short_window, long_window, 
                                                on_threshold, off_theshold, 
                                                moveout, min_triggers, 
                                                        ptolerance, START, 
                                                        window_start, 
                                                        window_length, freq_min, 
                                                        freq_max, trig_freq_min,
                                                        trig_freq_max, 
                                                        multiple_triggers,
                                                        mseed_length, no_triggers)
            
                        

                ###Array processing---------------------------------
                ##Least squares--------------------
                if processing == 'lts' or processing == 'ls':

                    array_data = least_trimmed_squares(processing, st, sta_lats, sta_lons, 
                                            window_length, WINDOW_OVERLAP,
                                            eq_baz_real, eq_slow_real)
                    
                ##Frequency wavenumber--------------------
                elif processing == 'fk': 
                    array_data = fk_obspy(st1, stations, sta_lats, sta_lons, sta_elev, 
                                        START_new, END_new, window_length, 
                                        WINDOW_OVERLAP, freq_min, freq_max, sll_x, 
                                        slm_x, sll_y, slm_y, sl_s, semb_thres, 
                                        vel_thres, timestamp, prewhiten,
                                        eq_baz_real, eq_slow_real)


            ################################################################ 
                #Save common data------------------------
                array_data['max_freq'] = freq_max
                array_data['min_freq'] = freq_min
                array_data['window_length'] = window_length
                array_data['window_start'] = window_start
                array_data['multiple_triggers'] = multiple_triggers
                array_data['no_triggers'] = no_triggers
                array_data['trigger_time'] = str(trigger_time)
                array_data['trigger_type'] = trigger_type
                array_data['sta/lta'] = peak
                array_data['trigger_length'] = length 
                array_data['num_stations'] = len(st)
                array_data['array_lat'] = origin_lat
                array_data['array_lon'] = origin_lon
                array_data['event_id'] = event_id
                array_data['velocity_model'] = velocity_model
                array_data['array_processing'] = processing
                array_data['min_triggers'] = min_triggers


            
                array_data_list.append(array_data)

                print('Events completed:', str(event+1)+'/'+str(len(df)))

            except ValueError as e:
                print(f"Skipping event {event_ids[event]}: {e}")
                continue

            except Exception as e:
                print(f"Unexpected error for event {event_ids[event]}: {e}")
                continue

#Putting data into single dataframe----------------------
array_data_comb1 = pd.concat(array_data_list, ignore_index=True)

#Combining with earthquake data-----------------------
array_data_comb = pd.merge(array_data_comb1, df, on='event_id', how='inner')

#Save to csv if specified
if save_events == True:
    array_data_comb.to_csv(array_name+'_'+max_rad+'km_m3_max_mdccm.csv')

if save_stations == True:
    station_info.to_csv(array_name+'_'+max_rad+'km_m3_stations.csv')

#Plot some figures
df = array_data_comb

drop = True #drop Taup picks, i.e. events without an STA/LTA pick

if drop ==True:
    
    temp = pd.DataFrame(df[df['trigger_type']!= 'Taup'])
    print('Number of dropped events for Taup:', len(df) - len(temp))
    df = temp

color_data = df['distance']
color_label = 'distance (km)'
model_data = []
baz_error_spatial(df['backazimuth'], df['baz_error'], model_data, color_data, color_label, niazi = True)

slow_error_spatial(df['backazimuth'], df['slow_error'], model_data, color_data, color_label, niazi = True)

#Plot map of earthquakes-----------------------------
array_lats = [53.6974, 53.779, 53.8566]  
array_lons = [-166.7343, -166.2131,-166.4161]
array_names = ['2A', '3A', 'POM']
array_names = []
earthquake_lats = df['latitude'].to_numpy()
earthquake_lons = df['longitude'].to_numpy()
earthquake_mags = df['magnitude'].to_numpy()
earthquake_depths = df['depth'].to_numpy()

pygmt_array_earthquakes(array_lats, array_lons, array_names, earthquake_lats, earthquake_lons, earthquake_mags, earthquake_depths, save=False, path = None)

#Plot baz error on map-----------------------------
baz = df['backazimuth'].to_numpy()
baz_error = df['baz_error'].to_numpy()

pygmt_baz_error(array_lats[0], array_lons[0], array_name, earthquake_lats, 
                earthquake_lons, earthquake_mags, baz, baz_error, save = False, 
                path = None)

# %%
