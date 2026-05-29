import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import datetime as dt
from scipy.optimize import least_squares

import lts_array

#Obspy dependencies-----------------------------
from obspy.clients.fdsn import Client
from obspy import read
from obspy import read_events
from obspy import read_inventory
from obspy import Stream
from obspy import UTCDateTime
from obspy.taup import TauPyModel
from obspy.core.util import AttribDict
from obspy.signal.array_analysis import array_processing
from obspy.geodetics import gps2dist_azimuth
from obspy.geodetics import kilometers2degrees
from obspy.signal.util import util_geo_km
from obspy.signal.trigger import trigger_onset, classic_sta_lta
from obspy.clients.fdsn.header import FDSNNoDataException



############################################################
#### FUNCTIONS FOR ARRAY CALCULATION ###########################
############################################################


def least_trimmed_squares(processing, st, sta_lats, sta_lons, WINDOW_LENGTH,
                  WINDOW_OVERLAP, eq_baz_real, eq_slow_real):
    
    '''
    Calculates least trimmed squares and organizes data
    
    Parameters:
        processing: 'lts', 'ls'
        st: obspy stream containing traces
        sta_lats: station lats (list or array)
        sta_lons: station lons (list or array)
        WINDOW_LENGTH: window length of array analysis window (seconds)
        WINDOW_OVERLAP: overlap between analysis windows
        trigger_time: trigger time in UTC (string)
        trigger_type: 
            'STA/LTA': single STA/LTA trigger
            'Multiple triggers': multiple triggers, chosen based on multiple triggers input
            'Taup': no STA/LTA trigger, using Taup time and larger window to search 
        peak: STA/LTA ratio of nearest peak to trigger
        length: length of STA/LTA trigger duration
        origin_lat: latitude of array center
        origin_lon: longitude of array center
        event_id: USGS earthquake identifier
        eq_baz_real: backazimuth between catalog event and array
        eq_slow_real: calculated slowness based on velocity model

    Returns: 
        pandas dataframe: 
        'array_baz': array backazimuth
        'array_slow': array slowness
        'array_vel': array trace velocity
        'mdccm': array mdccm (cross correlation power)
        'conf_int_vel': confidence interval of trace velocity
        'conf_int_baz': confidene interval of backazimuth
        'time': time of array analysis (UTC)
        'event_id': USGS event ID
        'baz_error': backazimuth error (real - array)
        'slow_error': slowness error (real - array)
        'trigger_time': trigger time
        'trigger_type': trigger type
        'sta/lta': peak of nearest peak from sta/lta for trigger
        'trigger_length': length of trigger in seconds
        'num_stations': number of stations used
        'array_lat': array center latitude
        'array_lon': array center longitude
    '''
    
    if len(st) < 4: #Can't perform lts for less than 4 stations
        processing = 'ls'

    if processing == 'lts':
        ALPHA = 0.5 #least trimmed squares
        print('Starting LTS')
    else:
        ALPHA = 1 #least squares
        print('Starting LS')
            
    (lts_vel, lts_baz, t, mdccm, stdict, sigma_tau,
        conf_int_vel, conf_int_baz) = lts_array.ltsva(st, sta_lats, 
                                                    sta_lons, 
                                                    WINDOW_LENGTH, 
                                                    WINDOW_OVERLAP, 
                                                    ALPHA)
    #print('Length of data:', len(lts_vel))

    if len(lts_baz) >1: #pulling out max cross correlation
        print('Pulling out max mdccm')
        idx = np.argmax(mdccm)
    else: #should only be one value for trigger time
        idx = 0
    data = {
        'array_baz': lts_baz[idx],
        'array_slow': 1/lts_vel[idx],
        'array_vel': lts_vel[idx],
        'mdccm': mdccm[idx],
        'conf_int_vel': conf_int_vel[idx],
        'conf_int_baz': conf_int_baz[idx],
        'time': str(UTCDateTime(mdates.num2date(t[idx]))),
        'baz_error': baz_error(eq_baz_real, lts_baz[idx]),
        'slow_error': eq_slow_real - (1/lts_vel[idx])
        }
    array_data = pd.DataFrame(data, index=[0]) #print(array_data)
    return array_data

def fk_obspy(st1, stations, sta_lats, sta_lons, sta_elev, START_new, END_new,
             WINDOW_LENGTH, WINDOW_OVERLAP, FREQ_MIN, FREQ_MAX,
             sll_x, slm_x, sll_y, slm_y, sl_s, semb_thres, vel_thres, timestamp, prewhiten,
            eq_baz, eq_slow):
    print('Starting FK')
    #Add necessary data to streams----------------
    for l in range(len(stations)):  # Uses all stations in pd dataframe stations
        st1[l].stats.coordinates = AttribDict({
            'latitude': sta_lats[l],
            'elevation': sta_elev[l],
            'longitude': sta_lons[l]})

    #Set up dictionary based on input parameters----------------------------
    kwargs = dict(
            # slowness grid: X min, X max, Y min, Y max, Slow Step
            sll_x=sll_x, slm_x=slm_x, sll_y=sll_y, slm_y=slm_y, sl_s=sl_s,
            # sliding window properties
            win_len=WINDOW_LENGTH, win_frac=WINDOW_OVERLAP,
            # frequency properties
            frqlow=FREQ_MIN, frqhigh=FREQ_MAX, prewhiten=prewhiten,
            # restrict output
            semb_thres=semb_thres, vel_thres=vel_thres, timestamp=timestamp,
            #stime=START+1, etime=END-1 #had to add and subtract 2 to avoid timing errors
            stime=START_new, etime = END_new
                )
    out = array_processing(st1, **kwargs)
    
    #OUTPUT FROM FK PROCESSING-----------------------------------------------------
    array_out = pd.DataFrame(out, columns = ['time','relpow','abspow','baz_obspy','array_slow'])
        

    #Convert times and baz to same scale as lts (UTC time, centered on window)------------
    t = array_out['time'].to_numpy()
    baz_obspy = array_out['baz_obspy'].to_numpy()
        
    bazs = []
    time_error = []
    for j in range(len(t)):
        matplotlib_time = t[j]
        x = mdates.num2date(matplotlib_time) 
        x = UTCDateTime(x)
        #diff = (x-UTCDateTime(time_station))+(win_len/2)
        diff = str(x+(WINDOW_LENGTH/2)) #time centered on point
        time_error.append(diff)
        baz = baz_obspy[j]
        if baz <= 0:
            baz_correct = baz+360 #converts to all positive backazimuth
        else:
            baz_correct = baz
        bazs.append(baz_correct)
        
    time_error = np.array(time_error)
    fk_bazs = np.array(bazs)

    #Calculate baz/slow error---------------------------------------
    fk_baz_error = baz_error(eq_baz, fk_bazs)

    trace_vel_error = (1/eq_slow)- 1/array_out['array_slow'].to_numpy() #real - array

    slowness_error = (eq_slow) - array_out['array_slow'].to_numpy()

    #Start to aggregate data----------------------------
    array_out['baz_error'] = fk_baz_error
    #array_out['centered_time'] = time_error
    array_out['time'] = time_error
    array_out['array_baz'] = fk_bazs
    array_out['slow_error'] = slowness_error
    array_out['array_vel'] = 1/array_out['array_slow'].to_numpy()

    #Pull out greatest power (should be only one value if using triggers)------------
    idx = np.argmax(array_out['relpow'].to_numpy())

    #Save data--------------------------------------
    array_data = array_out.loc[[idx]]
    array_data['conf_int_vel'] = 0 #values not returned for FK, keeping for consistency with other codes
    array_data['conf_int_baz'] = 0 #values not returned for FK, keeping for consistency with other codes
    array_data['mdccm'] = 0 #values not returned for FK, keeping for consistency with other codes
    


    return array_data


############################################################
#### HOMER/KODIAK SPECIFIC ###########################
############################################################

def stations_available_generator_hm_kd(
    earthquake_time,
    station_d1_list, start_d1_list, end_d1_list,
    station_d2_list, start_d2_list, end_d2_list,
    array_name
):

    # ---------------------------------
    # Convert times once
    # ---------------------------------
    earthquake_time = np.array([utc2datetime(str(t)) for t in earthquake_time])

    start_d1 = np.array([utc2datetime(str(t)) for t in start_d1_list])
    end_d1   = np.array([utc2datetime(str(t)) for t in end_d1_list])

    start_d2 = np.array([utc2datetime(str(t)) for t in start_d2_list])
    end_d2   = np.array([utc2datetime(str(t)) for t in end_d2_list])

    station_d1 = np.array(station_d1_list)
    station_d2 = np.array(station_d2_list)

    # ---------------------------------
    # Load bear removal data
    # ---------------------------------
    if array_name == "HM":

        bears = pd.read_csv(
            "/Users/cadequigley/Downloads/Research/deployment_array_design/homer_mseed_completeness.csv"
        )[["station_name","bear_removal_time_d1","bear_removal_time_d2"]]

        bear_d1 = dict(zip(bears.station_name, bears.bear_removal_time_d1))
        bear_d2 = dict(zip(bears.station_name, bears.bear_removal_time_d2))

        # build arrays aligned with station lists
        bear_d1_arr = np.array([bear_d1.get(sta, '0') for sta in station_d1])
        bear_d2_arr = np.array([bear_d2.get(sta, '0') for sta in station_d2])

        # apply bear removal overrides
        mask = bear_d1_arr != '0'
        end_d1[mask] = np.array([utc2datetime(str(t)) for t in bear_d1_arr[mask]])

        mask = bear_d2_arr != '0'
        end_d2[mask] = np.array([utc2datetime(str(t)) for t in bear_d2_arr[mask]])

    # ---------------------------------
    # Output containers
    # ---------------------------------
    stations_lists = []
    stations_available = []
    deployment_all = []

    # ---------------------------------
    # Loop earthquakes only
    # ---------------------------------
    for eq in earthquake_time:

        mask_d1 = (eq >= start_d1) & (eq <= end_d1)
        mask_d2 = (eq >= start_d2) & (eq <= end_d2)

        sta_d1 = station_d1[mask_d1]
        sta_d2 = station_d2[mask_d2]

        stations = np.concatenate([sta_d1, sta_d2])
        deployments = ["d1"]*len(sta_d1) + ["d2"]*len(sta_d2)

        stations_lists.append(stations.tolist())
        stations_available.append(len(stations))
        deployment_all.append(deployments)

    return stations_lists, stations_available, deployment_all

'''
def process_hm_kd_data(net, sta, loc, chan,  starttime, endtime, array_name, array, processing, 
                 FREQ_MIN, FREQ_MAX, WINDOW_LENGTH, WINDOW_OVERLAP, window_start, min_mag, max_rad,
                 short_window, long_window, on_threshold, off_theshold, client = Client('EARTHSCOPE'), remove_stations = [], keep_stations = [], 
                 gain = None, min_stations = 3, use_full_deployment = False, save = False, velocity_model = 'iasp91', 
                 timing = 'trigger', min_triggers = 1, ptolerance = 5, multiple_triggers = 'peak', no_triggers = 'taup', sll_x = -1.0,
                 slm_x = 1.0, sll_y=-1.0, slm_y = 1.0, sl_s = 0.03, semb_thres = -1e9, vel_thres = -1e9, timestamp = 'mlabday', prewhiten = 0):
    
    #Pull inventory-----------------------
    #------------------------------------------------
    deployment = 'd1'
    path = '/Users/cadequigley/Downloads/Research/deployment_array_design/'
    inv1 = read_inventory(path + array+'_'+deployment+'_station.xml')
    
    
    (lat_list, lon_list, elev_list, station_d1_list,
     start_d1_list, end_d1_list, num_channels_d1_list) = data_from_inventory(inv1, remove_stations, keep_stations)
    
    data = {
            'station': station_d1_list,
            'lat': lat_list,
            'lon': lon_list,
            'elevation': elev_list}
    
    station_info = pd.DataFrame(data)
    
    ## PULL IN DATA FOR SECOND DEPLOYMENT##########################
    deployment = 'd2'
    
    ## READ IN INVENTORY FOR D2-------------------------
    path = '/Users/cadequigley/Downloads/Research/deployment_array_design/'
    inv2 = read_inventory(path + array+'_'+deployment+'_station.xml')
    
    (lat_list_d2, lon_list_d2, elev_list_d2, station_d2_list,
     start_d2_list, end_d2_list, num_channels_d2_list) = data_from_inventory(inv2, remove_stations, keep_stations)
    
    #Pull earthquakes-----------------------
    #------------------------------------------------
    
    #### Get center of array--------
    
    output = get_geometry(lat_list, lon_list, elev_list, return_center = True)
    origin_lat = str(output[-1][1])
    origin_lon = str(output[-1][0])
    
    moveout = moveout_time(output)
    
    
    ### Pull in earthquakes--------------
    start, end = array_time_window(use_full_deployment, start_d1_list, end_d2_list,
                                   starttime, endtime)
    
    df = pull_earthquakes(origin_lat, origin_lon, max_rad, start, end, min_mag, array_name, velocity_model)
    print('Number of earthquakes >'+min_mag+' within '+max_rad+' km:', len(df))
    
    
    #Create station availability lists-----------------------
    #------------------------------------------------
    earthquake_time = df['time_utc'].to_numpy()
    earthquake_names = df['event_id'].to_numpy()
    
    stations_lists, stations_available, deployment_list = stations_available_generator_hm_kd(earthquake_time, station_d1_list, start_d1_list,
                                             end_d1_list, station_d2_list, start_d2_list, end_d2_list, array_name)
    
    #stations_lists, stations_available = stations_available_generator(earthquake_time, station_d1_list, start_d1_list, end_d1_list)
    
    ### Drop events that don't have enough stations present--------------
    bad_idx = [i for i, v in enumerate(stations_available) if v < min_stations]
    keep_idx = [i for i, v in enumerate(stations_available) if v >= min_stations]
    
    stations_available = [stations_available[i] for i in keep_idx]
    stations_lists = [stations_lists[i] for i in keep_idx]
    deployment_list = [deployment_list[i] for i in keep_idx]
    df = df.drop(index=bad_idx)
    
    print('Station lists for each earthquake created. New earthquake number:', len(df))
    
    ###Loop over all events---------------------------------
    
    event_ids = df['event_id'].to_numpy()
    eq_depths = df['depth'].to_numpy()
    eq_lats = df['latitude'].to_numpy()
    eq_lons = df['longitude'].to_numpy()
    eq_time = df['time_utc'].to_numpy()
    expected_parrival = df['p_arrival'].to_numpy()
    eq_baz = df['backazimuth'].to_numpy()
    eq_slow = df['slowness'].to_numpy()
    eq_distance = df['distance'].to_numpy()
    array_data_list = []
    misbehaving_list = []
    for event in range(len(df)):
        try:
            print("Starting", event_ids[event])
            stations = stations_lists[event] #pull out stations available for each event
            station_sub = station_info[station_info['station'].isin(stations)] #pull out specific station info
        
            sta_lats = station_sub['lat'].to_numpy()
            sta_lons = station_sub['lon'].to_numpy()
            stations = station_sub['station'].to_numpy()
            sta_elev = station_sub['elevation'].to_numpy()
            eq_baz_real = eq_baz[event]
            eq_slow_real = eq_slow[event]
            event_id = event_ids[event]
    
    
            START = UTCDateTime(eq_time[event])+expected_parrival[event]-60
            END = START +120
            try: #Try to pull event from locally
                if array_name =='HM':
                    array = 'homer'
                else:
                    array = 'kodiak'
                st = read('/Users/cadequigley/Downloads/Research/deployment_array_design/'+array+'_earthquakes_mseeds/'+event_ids[event]+".mseed")
                st = st.slice(START, END)
                #st = read('/Users/cadequigley/Downloads/Research/'+array+'_earthquakes_mseeds/'+event_ids[event]+".mseed")
            except FileNotFoundError:
                print('File not found locally, trying to pull from IRIS')
                st = Stream()
                failed_stations = []
        
                for sta in stations:
                    try:
                        st += client.get_waveforms(net, sta, loc, chan, START, END)
                    except FDSNNoDataException:
                        print(f"No data for station {sta}")
                        failed_stations.append(sta)
                    except Exception as e:
                        print(f"Error for station {sta}: {e}")
                        failed_stations.append(sta)
        
                # Remove failed stations
                if failed_stations:
                    mask = ~np.isin(stations, failed_stations)
                    stations = stations[mask]
                    sta_lats = sta_lats[mask]
                    sta_lons = sta_lons[mask]
                    sta_elev = sta_elev[mask]
                    
            if len(keep_stations) >0:
                st_subset = Stream()
                for sta in stations:
                    st_subset += st.select(station=sta)
                st = st_subset.copy()
    
            if len(st) < min_stations:
                raise ValueError("Not enough traces")
    
            #st.resample(100) #testing to see if data is resampled now
            st.merge(fill_value='latest')
            #st.trim(START, END, pad='true', fill_value=0)
            st.sort()
            #if deployment_list[event][0] == 'd1':
                #st.remove_sensitivity(inventory = inv1)
            #else:
                #st.remove_sensitivity(inventory = inv2)
            for b in range(len(st)):
                tr = st[b]
                tr.data = tr.data/gain
                
                
    
            # Filter the data
            st.filter("bandpass", freqmin=FREQ_MIN, freqmax=FREQ_MAX, corners=2, zerophase=True)
            st.taper(max_percentage=0.05)
            #print('Done pulling data')
            st1 = st.copy()
            ###Finding triggers---------------------------------
            if timing == 'trigger': #use sta/lta triggers
                (st, trigger, peak, length, 
                 trigger_type, trigger_time, 
                 START_new, END_new)= triggers(st, short_window, long_window, 
                                               on_threshold, off_theshold, 
                                               moveout, min_triggers, 
                                               ptolerance, START, 
                                               window_start, WINDOW_LENGTH, 
                                               multiple_triggers, no_triggers)
                
            ###Array processing---------------------------------
            ##Least squares--------------------
            if processing == 'lts' or processing == 'ls':
                array_data = least_trimmed_squares(processing, st, sta_lats, sta_lons, 
                                           WINDOW_LENGTH, WINDOW_OVERLAP,
                                           trigger_time, trigger_type, peak,
                                           length, origin_lat, origin_lon, 
                                           event_id, eq_baz_real, eq_slow_real)
                
                
            else: #fk analysis
                
                array_data = fk_obspy(st1, stations, sta_lats, sta_lons, sta_elev, START, START_new, END_new,
                                      WINDOW_LENGTH, WINDOW_OVERLAP, FREQ_MIN, FREQ_MAX,
                                      sll_x, slm_x, sll_y, slm_y, sl_s, semb_thres, vel_thres, timestamp, prewhiten,
                                      eq_baz_real, eq_slow_real, event_id, trigger, trigger_type, peak, length, origin_lat, origin_lon)
                
        
            array_data_list.append(array_data)
            print('Events completed:', str(event+1)+'/'+str(len(df)))
    
        except ValueError as e:
            print(f"Skipping event {event_ids[event]}: {e}")
            #traceback.print_exc()
            continue
    
        except Exception as e:
            print(f"Unexpected error for event {event_ids[event]}: {e}")
            #traceback.print_exc()
            continue
    
    #Putting data into single dataframe----------------------
    array_data_comb1 = pd.concat(array_data_list, ignore_index=True)
    
    array_data_comb = pd.merge(array_data_comb1, df, on='event_id', how='inner')
    
    
    if save == True:
        array_data_comb.to_csv(array_name+'_'+max_rad+'km_m3_fk_wawa.csv')
        
    return array_data_comb, station_info

'''
############################################################
#### FUNCTIONS FOR PREPROCESSING ###########################
############################################################



def grab_preprocess(stations, station_info, inv, 
                    net, loc, chan, min_stations, 
                    START, END, client, array, event_id, path, save_mseed = False):
    
    station_sub = station_info[station_info['station'].isin(stations)] #pull out specific station info
        
    sta_lats = station_sub['lat'].to_numpy()
    sta_lons = station_sub['lon'].to_numpy()
    stations = station_sub['station'].to_numpy()
    sta_elev = station_sub['elevation'].to_numpy()

    try: #Try to pull event from locally
        
        #st = read('/Users/cadequigley/Downloads/Research/deployment_array_design/'+array+'_earthquakes_mseeds/'+event_id+".mseed")
        st = read(path+event_id+'.mseed')
        st = st.slice(START, END)
        stations = set(tr.stats.station for tr in st)
        station_sub = station_info[station_info['station'].isin(stations)] #pull out specific station info
        
        valid_stations = set(station_sub['station'])

        # Remove traces not in station list
        st = Stream([tr for tr in st if tr.stats.station in valid_stations])
        #st = st.select(station=list(valid_stations))

        sta_lats = station_sub['lat'].to_numpy()
        sta_lons = station_sub['lon'].to_numpy()
        stations = station_sub['station'].to_numpy()
        sta_elev = station_sub['elevation'].to_numpy()
            #st = read('/Users/cadequigley/Downloads/Research/'+array+'_earthquakes_mseeds/'+event_ids[event]+".mseed")
    except FileNotFoundError:
        print('File not found locally, trying to pull from IRIS')
        
        st = Stream()
        failed_stations = []

        for sta in stations:
            try:
                st += client.get_waveforms(net, sta, loc, chan, START, END)
            except FDSNNoDataException:
                print(f"No data for station {sta}")
                failed_stations.append(sta)
            except Exception as e:
                print(f"Error for station {sta}: {e}")
                failed_stations.append(sta)

        # Remove stations that did not have data from metadata
        if failed_stations:
            mask = ~np.isin(stations, failed_stations)
            stations = stations[mask]
            sta_lats = sta_lats[mask]
            sta_lons = sta_lons[mask]
            sta_elev = sta_elev[mask]
            
        # Check to see if there are enough stations----
        if len(st) < min_stations:
            raise ValueError("Not enough traces")
        
        if save_mseed == True:
            st.sort()
            st.write(path+event_id+".mseed", format="MSEED")
            print('mseed saved')


    # Basic data preperation-----------    
    st.merge(fill_value='latest')
    st.trim(START, END, pad='true', fill_value=0)
    st.sort()
    
    st.remove_sensitivity(inventory = inv)

        # Filter the data
    #st.filter("bandpass", freqmin=FREQ_MIN, freqmax=FREQ_MAX, 
                #corners=2, zerophase=True)
        
    st.taper(max_percentage=0.05)
    
    return st, stations, sta_lats, sta_lons, sta_elev



def moveout_time(output):
    '''
    Calculates the minimum moveout time across the array based on maximum
    interstation distacne and velocity, including some wiggle room
    
    Parameters:
        output: output from get_geometry function. Contains interstation distances

    Returns: 
        moveout: expected moveout time in seconds (float)
        
        
    '''
    #### Calculate interstation distances/moveout time
    xpos = list(output[:,0])
    xpos = xpos[:-1]
    ypos = list(output[:,1])
    ypos = ypos[:-1]
    distances_temp = interstation_distances(xpos, ypos)
    moveout = (np.max(distances_temp)/3)+0.5 #t = d/v + error
    return moveout

def array_time_window(use_full_deployment, start_d1_list, end_d1_list,
                      starttime, endtime):
    '''
    Defines what dates to look for earthquakes based on active stations
    
    Parameters:
        use_full_deployment: whether to use full time window deployment was out (True or False)
        start_d1_list: list of start times for each station
        end_d1_list: list of end times for each station
        starttime: specified starttime, will use if use_full_deployment = True
        endtime: speficied endtime, will use if use_full_deployment = True

    Returns: 
        start: start time to look for earthquakes
        end: end time to look for earthquakes
        
        
    '''
    if use_full_deployment ==True:
        start = str(np.min(start_d1_list)) #time when first station online
        if str(type(end_d1_list[0])) == "<class 'NoneType'>": #deals with case where station/array is still active by taking time today
            end_temp = UTCDateTime.now()
            end = str(end_temp)
            temp = []
            for i in range(len(end_d1_list)):
                temp.append(end_temp)
            end_d1_list = temp
        else:
            end = str(np.max(end_d1_list)) #time when last station offline

    else: #use restricted time window specified at start
        start = starttime
        end = endtime
    
    return start, end


def rotate_channel(st, inv, channel): ###NEED TO FINISH-------------
    for i in range(len(st)):
        tr = st[i]
        if channel[:-1] == 'Z':
            if inv[0][i].channels[0].dip == 90:
                tr.data = -1*tr.data
        elif channel[:-1] == 'N':
            if inv[0][i].channels[0].azimuth == 180:
                tr.data = -1*tr.data
        elif channel[:-1] == 'E':
            if inv[0][i].channels[0].azimuth == 270:
                tr.data = -1*tr.data

def calculate_slowness(distance_km, depth, velocity_model):

    """
    Calculates the slowness of an event based on known information about hypocenter. This is 
    a 1D calculation using the Taup calculator (Crotwell et al.)
    
    Parameters:
        distance_km: epicentral distance to event in km
        depth: depth of event in km
        velocity_model: velocity model to use for slowness calculation ('iasp91', 'ak135', 'pavdut', 'japan_1d', '')
        
    Returns:
        slowness: expected slowness at surface (s/km)
        trace_vel: expected trace_vel at surface (km/s)
        incident_angle: incident angle of ray at surface (degrees)
        p_arrival: calculated p-arrival time (seconds after origin time)
    """
    
    mod = velocity_model #pavdut, iasp91, japan_1d, ak135, scak
    model = TauPyModel(model=mod)
    
   
    dist_deg = kilometers2degrees(distance_km)

        
    arrivals_p = model.get_travel_times(source_depth_in_km=depth,
                                distance_in_degree=dist_deg,
                                phase_list = ["P","p"])
    arr = arrivals_p[0]
    p_arrival = arr.time
    incident_angle = arr.incident_angle
    if mod == 'iasp91':
        trace_vel = 5.8/(np.sin(np.deg2rad(incident_angle))) #iasp91 surface velocity: 5.8
    elif mod == 'japan_1d':
        trace_vel = 4.8/(np.sin(np.deg2rad(incident_angle))) #japan_1D surface velocity: 5.8
    elif mod == 'ak135':
        trace_vel = 5.8/(np.sin(np.deg2rad(incident_angle)))
    elif mod == 'scak':
        trace_vel = 5.3/(np.sin(np.deg2rad(incident_angle)))
    else: #pavdut
        trace_vel = 3.05/(np.sin(np.deg2rad(incident_angle))) #pavdut surface velocity: 3.05

    slowness = 1/trace_vel

    return slowness, trace_vel, incident_angle, p_arrival

def misbehaving_stations_lts(d, threshold=4):
    """
    Returns a list of values that appear more than `threshold` times
    in the first array-like value found in the dictionary `d`.
    Ignores non-array entries like 'size'.
    """
    # Find the first array in the dictionary
    for key, val in d.items():
        if isinstance(val, (np.ndarray, list)):  # supports arrays or lists
            arr = np.array(val)  # convert to numpy array if list
            unique, counts = np.unique(arr, return_counts=True)
            return unique[counts > threshold].tolist()
    # Return empty if no array found
    return []

def data_from_inventory(inv, remove_stations, keep_stations):

    """
    Pulls pertinent information out of an inventory for arrays.
    
    Parameters:
        inv: station inventory based on station.xml format from FDSN
        remove_stations: list of station names to remove if there is a known
                          issue with the station. Example: ['2A12', '2A14']
        
    Returns:
        lat_list: list of station latitudes
        lon_list: list of station longitudes
        elev_list: list of station elevation
        station_list: list of station names
        start_list: stat times of data available
        end_list: end times of data available
        num_channels_list: number of channels with associated station            
    """
    ## PULL INFORMATION OUT OF INVENTORY-------------------------
    lat_list = []
    lon_list = []
    elev_list = []
    station_list = []
    start_list = []
    end_list = []
    num_channels_list = []

    for network in inv:
        for station in network:
            lat_list.append(station.latitude)
            lon_list.append(station.longitude)
            station_list.append(station.code)
            elev_list.append(station.elevation)
            start_list.append(station.start_date)
            if station.end_date == None:
                end_list.append(UTCDateTime.now())
            else:
                end_list.append(station.end_date)
            num_channels_list.append(station.total_number_of_channels)
            
    if len(remove_stations) > 0: 
        for k in range(len(remove_stations)):
            station = remove_stations[k]
            idx = station_list.index(station)
            del lat_list[idx]
            del lon_list[idx]
            del station_list[idx]
            del elev_list[idx]
            del start_list[idx]
            del end_list[idx]
            del num_channels_list[idx]

    if len(keep_stations) > 0:

        mask = [sta in keep_stations for sta in station_list]

        lat_list = [lat_list[i] for i in range(len(mask)) if mask[i]]
        lon_list = [lon_list[i] for i in range(len(mask)) if mask[i]]
        station_list = [station_list[i] for i in range(len(mask)) if mask[i]]
        elev_list = [elev_list[i] for i in range(len(mask)) if mask[i]]
        start_list = [start_list[i] for i in range(len(mask)) if mask[i]]
        end_list = [end_list[i] for i in range(len(mask)) if mask[i]]
        num_channels_list = [num_channels_list[i] for i in range(len(mask)) if mask[i]]

        

    return lat_list, lon_list, elev_list, station_list, start_list, end_list, num_channels_list


def check_num_stations(min_stations, station_list):
    '''
    Checks if the minimum number of stations is met.
    
    Parameters:
        min_stations: minimum stations wanted
        station_list: list of stations

    Returns: 
        ValueError if not enough stations
        
        
    '''
    num_stations = len(station_list)
    if num_stations < min_stations:
        raise ValueError("The minimum stations is greater then the number of available stations.")


def get_geometry(lat_list, lon_list, elev_list, return_center = False):

    """
    Gets the geometry of the array in terms of meters from a center point.
    
    Parameters:
        lat_list: list of station latitudes
        lon_list: list of station longitudes
        elev_list: list of station elevations
        return_center: return center of array (True of False)
        
    Returns:
        geometry of array, including center point if return_center = True.             
    """
    nstat = len(lat_list)
    center_lat = 0.
    center_lon = 0.
    center_h = 0.
    geometry = np.empty((nstat, 3))

    for i in range(nstat):
        geometry[i, 0] = lon_list[i]
        geometry[i, 1] = lat_list[i]
        geometry[i, 2] = elev_list[i]

    center_lon = geometry[:, 0].mean()
    center_lat = geometry[:, 1].mean()
    center_h = geometry[:, 2].mean()
    for i in np.arange(nstat):
        x, y = util_geo_km(center_lon, center_lat, geometry[i, 0],
                               geometry[i, 1])
        geometry[i, 0] = x
        geometry[i, 1] = y
        geometry[i, 2] -= center_h

    if return_center:
        return np.c_[geometry.T,
                     np.array((center_lon, center_lat, center_h))].T
    else:
        return geometry

def interstation_distances(xpos, ypos):
    points = np.column_stack((xpos, ypos))  # shape (N, 2)

    dx = points[:, 0][:, None] - points[:, 0][None, :]
    dy = points[:, 1][:, None] - points[:, 1][None, :]

    distances = np.sqrt(dx**2 + dy**2)
    return distances

def utc2datetime(utctime): 
    '''
    Converts string of utctime to datetime
    
    Parameters:
        utctime: time in utc (string)

    Returns: 
        datetime object
        
        
    '''
    return dt.datetime(int(utctime[0:4]),int(utctime[5:7]), int(utctime[8:10]), int(utctime[11:13]),int(utctime[14:16]),int(utctime[17:19]))
    


def baz_error(baz_real, baz_calculated):
    '''
    Calculates backazimuth error between catalog baz and array baz
    
    Parameters:
        baz_real: catalog backazimuth
        baz_calculated: array calculated backazimuth

    Returns: 
        baz_error: catalog baz - array baz
        
        
    '''
    baz_error_temp = baz_real - baz_calculated
    baz_error = ((baz_error_temp + 180) % 360) - 180
    return baz_error


############################################################
#### FUNCTIONS FOR PULLING EARTHQUAKES ###########################
############################################################
def pull_earthquakes(lat, lon, max_rad, start, end, min_mag, array_name, velocity_model):

    """
    Pulls in earthquakes from a region based on lat, lon, timing, and magnitude.
    It also returns other values of interest about the event for array processing,
    such as backazimuth, slowness, and epicentral distance to the event.
    
    Parameters:
        lat: latitude of array/station (str)
        lon: longitude of array/station (str)
        max_rad: maximum radius of earthquakes in kilometers (str)
        start: start time in UTC format (str)
        end: end time in UTC format (str)
        min_mag: minimum magnitude of earthquakes (str)
        array_name: name of array/station (str)
        velocity_model: name of velocity model (ex. 'iasp91', 'ak135')
        
    Returns:
        pandas DataFrame:
           'event_id': event id from USGS catalog
           'depth': depth of earthquake in km
           'magnitude': magnitude of earthquake
           'latitude': earthquake latitude
           'longitude': earthquake longitude
           'time_utc': origin time in UTC
           'time_ak': origin time in AK
           'distance': epicentral distance to event in km
           'backazimuth': backazimuth from array/station to earthquake
           'array': name of station/array
           'slowness': surface slowness (s/km)
           'trace_vel': surface trace velocity (km/s)
           'incident_angle': angle from vertical of first arriving wave (degrees)
           'p_arrival': arrival time of p-wave (seconds)             
    """

    ##Pull data in from FDSNWS: https://earthquake.usgs.gov/fdsnws/event/1/
    url = ('https://earthquake.usgs.gov/fdsnws/event/1/query?format=quakeml&starttime='+start+'&endtime='
           +end+'&latitude='+lat+'&longitude='+lon+'&maxradiuskm='+max_rad+'&minmagnitude='+min_mag+'')

    catalog = read_events(url)
    depths = []
    magnitudes = []
    latitudes = []
    longitudes = []
    times_utc = []
    times_ak = []
    names = []
    distances = []
    backazimuth = []
    array = []
    slowness = []
    trace_vel = []
    incident_angle = []
    p_arrival = []

    # Extract data from each event
    for event in catalog:
        # Extract depth
        depth = event.origins[0].depth / 1000  # Depth is in meters, convert to kilometers
    
        # Extract magnitude
        magnitude = event.magnitudes[0].mag
    
        # Extract latitude and longitude
        latitude = event.origins[0].latitude
        longitude = event.origins[0].longitude
    
        # Extract time
        time = event.origins[0].time

        # Extract event_id
        resource_id = event.resource_id.id
        name = resource_id.split('?')[-1]
        name = name[:-15]
        name= name[8:]

        #Calculate distance, backazimuth
        dist, baz, az = gps2dist_azimuth(float(lat), float(lon), latitude, longitude)
        dist = dist/1000 #converts m to km

        if depth < 0:
            slow = 0
            t_vel = 0
            incident = 0
            p = 0
        else:
            #Calculate slowness, trace velocity, incident angle, and arrival time
            slow, t_vel, incident, p = calculate_slowness(dist, depth, velocity_model)
        
        # Append data to lists
        depths.append(depth)
        magnitudes.append(magnitude)
        latitudes.append(latitude)
        longitudes.append(longitude)
        times_utc.append(time)
        times_ak.append(time - 60*60*8)  # conversion to AK time
        names.append(name)
        distances.append(dist)
        backazimuth.append(baz)
        array.append(array_name)
        slowness.append(slow)
        trace_vel.append(t_vel)
        incident_angle.append(incident)
        p_arrival.append(p)

    # Combine into DataFrame
    data = {
        'event_id': names,
        'depth': depths,
        'magnitude': magnitudes,
        'latitude': latitudes,
        'longitude': longitudes,
        'time_utc': times_utc,
        'time_ak': times_ak,
        'distance': distances,
        'backazimuth': backazimuth,
        'array': array,
        'slowness': slowness,
        'trace_vel': trace_vel,
        'incident_angle': incident_angle,
        'p_arrival': p_arrival,
    }

    df = pd.DataFrame(data)
    return df

def stations_available_generator(earthquake_time_list, station_d1_list, start_d1_list, end_d1_list):
    '''
    Finds which stations are available for each earthquake.
    
    Parameters:
        earthquake_time_list: list of earhtquake times (UTC)
        station_d1_list: list of all stations
        start_d1_list: list of start times for each station
        end_d1_list: list of end times for each station
        

    Returns: 
        station_lists: list of different stations available for each event
        stations_available: number of stations available for each earthquake
        
        
    '''

    def is_between(check, start, end): 
        """
        Checks if a time is between two other times. Useful for determining what stations to use.
        
        Parameters:
            check: time to test
            start: start time data
            end:
            
        Returns:
            True or False            
        """
        return start <= check <= end

    stations_lists = []
    stations_available = []
    for i in range(len(earthquake_time_list)): #setting up earthquakes to loop through
        eq_time = earthquake_time_list[i]
        eq_time = utc2datetime(str(eq_time))
        station_temp = []

        ### Check deployment for station availability----------------------------
        for k in range(len(station_d1_list)):
            start_mseed = start_d1_list[k]
            start_mseed = utc2datetime(str(start_mseed))
            end_mseed = end_d1_list[k]
            end_mseed = utc2datetime(str(end_mseed))
        
            #Find if station exists------------------------    
            x = is_between(eq_time, start_mseed, end_mseed)

            if x == True:
                station_temp.append(station_d1_list[k])
            
        stations_lists.append(station_temp)
    
        stations_available.append(len(station_temp))
        
    return stations_lists, stations_available

############################################################
#### FUNCTIONS FOR STA/LTA TRIGGERS ###########################
############################################################

def trigger_list(tr, short_window, long_window, on_threshold, off_theshold):
    '''
    Calculates triggers in a station trace using classic sta/lta
    
    Parameters:
        trace: trace for a single station

    Returns: 
        trigger_times: list of triggers (seconds since start)
        trigger_peaks: list of sta/lta peak values
        trigger_lengths: list of length of trigger
        
        
    '''
    sr = tr.stats.sampling_rate
    #cft = classic_sta_lta(tr.data, int(2.5 * sr), int(30. * sr))
    #on_of = trigger_onset(cft, 2.5, 1.0)
    cft = classic_sta_lta(tr.data, int(short_window * sr), int(long_window * sr))
    on_of = trigger_onset(cft, on_threshold, off_theshold)

    trigger_times = []
    trigger_peaks = []
    trigger_lengths = []

    for on, off in on_of:
        # time of trigger (seconds from trace start)
        trigger_time = on / sr
        trigger_times.append(trigger_time)

        # peak STA/LTA value in that window
        peak_value = np.max(cft[on:off])
        trigger_peaks.append(peak_value)

        # window duration in seconds
        window_length = (off - on) / sr
        trigger_lengths.append(window_length)

    return trigger_times, trigger_peaks, trigger_lengths


def triggers_associator(trigger_lists, peak_lists, length_lists,
                        moveout, min_stations):

    time_groups = []
    peak_groups = []
    length_groups = []

    for i in range(len(trigger_lists)):
        triggers = trigger_lists[i]
        peaks = peak_lists[i]
        lengths = length_lists[i]

        comparison_times = trigger_lists[:i] + trigger_lists[i+1:]
        comparison_peaks = peak_lists[:i] + peak_lists[i+1:]
        comparison_lengths = length_lists[:i] + length_lists[i+1:]

        for k in range(len(triggers)):

            target_time = triggers[k]
            target_peak = peaks[k]
            target_length = lengths[k]

            group_times = [target_time]
            group_peaks = [target_peak]
            group_lengths = [target_length]

            for l in range(len(comparison_times)):
                comp_times = comparison_times[l]
                comp_peaks = comparison_peaks[l]
                comp_lengths = comparison_lengths[l]

                for t in range(len(comp_times)):
                    test_time = comp_times[t]

                    if abs(target_time - test_time) < moveout:
                        group_times.append(test_time)
                        group_peaks.append(comp_peaks[t])
                        group_lengths.append(comp_lengths[t])

            group_times.sort()

            if len(group_times) > min_stations:
                time_groups.append(group_times)
                peak_groups.append(group_peaks)
                length_groups.append(group_lengths)

    # Remove duplicates
    unique = list(set(tuple(tg) for tg in time_groups))

    trigger_times = []
    trigger_peaks = []
    trigger_lengths = []

    for group in unique:

        group = list(group)

        if abs(np.max(group) - np.min(group)) < moveout:

            idx = time_groups.index(group)

            median_time = np.median(group)

            # strength metric
            cluster_peak = np.median(peak_groups[idx]) #np.sum

            # length metric (choose what you prefer)
            cluster_length = np.mean(length_groups[idx])  # recommended

            trigger_times.append(median_time)
            trigger_peaks.append(cluster_peak)
            trigger_lengths.append(cluster_length)

    sorted_idx = np.argsort(trigger_times)

    return (
        np.array(trigger_times)[sorted_idx],
        np.array(trigger_peaks)[sorted_idx],
        np.array(trigger_lengths)[sorted_idx]
    )


def triggers(st, short_window, long_window, on_threshold, off_theshold, 
             moveout, min_triggers, ptolerance,
             START, window_start, WINDOW_LENGTH, FREQ_MIN, FREQ_MAX, 
             trig_freq_min, trig_freq_max,
            multiple_triggers, mseed_length, no_triggers = None):
    '''
    Combines the different trigger functions (trigger_lists, 
    triggers_associator) into a single function.
    
    Parameters:
        st: obspy stream containing traces
        moveout: expected moveout time across the array (seconds) (float)
        min_triggers: minimum picks to associate into a trigger (int)
        ptolerance: seconds around p-pick to allow association (float)
        START: start time of stream (UTCDateTime)
        window_start: where in time (seconds) to start analysis relative to p-pick (float)
        WINDOW_LENGTH: window length of array analysis window (seconds)
        multiple_triggers: how to handle multiple triggers ('peak' or 'closest')

    Returns: 
        st: stream that is trimmed based on trigger time and other parameters
        trigger: trigger time relative to start time (seconds)
        peak: STA/LTA ratio of nearest peak to trigger
        length: length of STA/LTA trigger duration
        trigger_type: 
            'STA/LTA': single STA/LTA trigger
            'Multiple triggers': multiple triggers, chosen based on multiple triggers input
            'Taup': no STA/LTA trigger, using Taup time and larger window to search
        trigger_time: trigger time (str)
        START_new: start time of new stream
        END_new: end time of new stream
    '''
    #Filter stream according to EPIC for calculating trigger times
    st_trig = st.copy()
    st_trig.filter("bandpass", freqmin=trig_freq_min, freqmax=trig_freq_max, 
                corners=2, zerophase=True)
    
    #Filter stream according to array processing input
    st.filter("bandpass", freqmin=FREQ_MIN, freqmax=FREQ_MAX, 
                corners=2, zerophase=True)
    trigger_lists = []
    trigger_peaks = []
    trigger_lengths = []
    for s in range(len(st)):
        times, peaks, lengths = trigger_list(st[s], short_window, long_window,
                                              on_threshold, off_theshold)
        trigger_lists.append(times)
        trigger_peaks.append(peaks)
        trigger_lengths.append(lengths)


    # Associate triggers together based on expected moveout------------
            

    times, peaks, lengths = triggers_associator(trigger_lists, 
                                                trigger_peaks, 
                                                trigger_lengths, 
                                                moveout, min_triggers)
            
    times = np.array(times)
    peaks = np.array(peaks)
    lengths = np.array(lengths)
    # Create mask to find triggers around expected p-arrival---------
    mask = np.abs(times - (mseed_length/2)) <= ptolerance
    trigger_filtered = times[mask]
    peaks_filtered = peaks[mask]
    lengths_filtered = lengths[mask]
            
    if len(trigger_filtered) == 0: #no triggers around p-pick
        trigger = mseed_length/2 
        trigger_type = 'Taup'
        peak = 0
        length = 0
        #Trim stream to allow for search for cross correlation
        if no_triggers == 'max mdccm':
            print('Pulling max mdccm')
            START_new = START + trigger + window_start- 0.001 - ptolerance
            END_new = START_new + 2*ptolerance
            st = st.slice(START_new, END_new)
        elif no_triggers == 'taup':
            START_new = START + trigger + window_start- 0.001
            END_new = START_new + WINDOW_LENGTH
            st = st.slice(START_new, END_new)

    elif len(trigger_filtered) >1: #multiple triggers around p-pick
        
        trigger_type = 'Multiple triggers'

        if multiple_triggers == 'peak':
            ####CHOOSING TRIGGER WITH LARGER PEAK-
            idx = np.argmax(peaks_filtered)
            trigger = trigger_filtered[idx]
            peak = peaks_filtered[idx]
            length = lengths_filtered[idx]
        elif multiple_triggers == 'closest': 
            ###CHOOSING TRIGGER CLOSEST TO EXPECTED ARRIVAL
            idx = np.argmin(np.abs(trigger_filtered - 60))
            trigger = trigger_filtered[idx]
            peak = peaks_filtered[idx]
            length = lengths_filtered[idx]
        elif multiple_triggers == 'first':
            ###CHOOSING TRIGGER CLOSEST TO EXPECTED ARRIVAL
            idx = np.argmin(trigger_filtered)
            trigger = trigger_filtered[idx]
            peak = peaks_filtered[idx]
            length = lengths_filtered[idx]

                
                
        #Trim stream to window of interest-------------
        START_new = START + trigger + window_start- 0.001
        END_new = START_new + WINDOW_LENGTH
        st = st.slice(START_new, END_new)

    else:
        trigger = trigger_filtered[0]
        peak = peaks_filtered[0]
        length = lengths_filtered[0]
        trigger_type = 'STA/LTA trigger'
        #Trim stream to window of interest-------------
        START_new = START + trigger + window_start- 0.001
        END_new = START_new + WINDOW_LENGTH
        st = st.slice(START_new, END_new)

    print(trigger_type)

    trigger_time = str(START+trigger)

    return st, trigger, peak, length, trigger_type, trigger_time, START_new, END_new


############################################################
#### FUNCTIONS FOR 3D SNELLS LAW ###########################
############################################################

def baz_to_az(backazimuth):
    azimuth = (backazimuth + 180) % 360
    return azimuth

def plane_normal(dip, strike):
    """
    Converts dip and strike to a unit normal vector (X, Y, Z).
    
    Parameters:
        dip_deg: float — Dip angle in degrees (0 = horizontal, 90 = vertical)
        dip_dir_deg: float — Dip direction in degrees (clockwise from North)

    Returns:
        np.array([x, y, z]) — unit normal vector to the plane
    """
    dip_dir_deg = (strike + 90) % 360
    dip_rad = np.radians(dip)
    dip_dir_rad = np.radians(dip_dir_deg)

    nx = np.sin(dip_rad) * np.sin(dip_dir_rad)  # X = East
    ny = np.sin(dip_rad) * np.cos(dip_dir_rad)  # Y = North
    nz = np.cos(dip_rad)                        # Z = Up

    n = np.array([nx, ny, nz])
    normal = n / np.linalg.norm(n)  # normalize just in case
    return normal

def spherical_to_xyz(azimuth, takeoff):
    """
    Converts azimuth (0-360°, clockwise from North) and takeoff angle (0-90°, from vertical)
    to a unit 3D direction vector [x, y, z].

    Parameters:
        azimuth_deg: float — azimuth angle in degrees, clockwise from North (Y+ axis)
        takeoff_deg: float — takeoff angle in degrees, 0° = vertical up, 90° = horizontal

    Returns:
        np.array([x, y, z]) — unit direction vector
    """
    az_rad = np.radians(azimuth)
    takeoff_rad = np.radians(takeoff)

    r_xy = np.sin(takeoff_rad)       # projection in XY plane
    x = r_xy * np.sin(az_rad)
    y = r_xy * np.cos(az_rad)
    z = np.cos(takeoff_rad)          # vertical component

    vector = np.array([x, y, z])

    return vector


    
def angle_to_xy(angle_deg):
    """
    Converts a clockwise-from-north angle (degrees) to a unit XY vector.
    0° = North (positive Y), 90° = East (positive X), etc.
    """
    theta_rad = np.radians(angle_deg)
    x = np.sin(theta_rad)
    y = np.cos(theta_rad)
    return np.array([x, y])
    

def snell_3d(incident, normal, v1, v2):
    """
    Snells law in 3 dimensions.
    Args:
        v1: velocity below moho (cold lithosphere) (float)
        v2: velocity above moho (hot lithosphere) (float)
        incident: directional vector of incident ray (3 component np.array)
        normal: normal vector to dipping moho plane (strike and dip)
    Returns:
        refracted: refracted ray
    
    """
    ratio = v2/v1
    l = incident/np.linalg.norm(incident) #incident vector of ray
    n = normal/np.linalg.norm(normal) #normal vector to subduction surface
    costheta1 = np.dot(n,l)
    costheta2 = np.sqrt((1-ratio**2)*(1-costheta1**2))
    refracted = ratio*l+(ratio*costheta1 + costheta2)*n
    return refracted



def deflection_xy(incident, refracted): #analogous with baz error
    """
    Calculates the angle between the incident wave and refracted wave in the x-y plane
    Args:
        incident: incident vector (3 component np.array)
        refracted: refracted vector (3 component np.array)

    Returns:
        angle_deg: angle in degrees of vector
    """
    # Project to XY
    u = np.array([incident[0], incident[1]])  # (x, y); incident vector
    v = np.array([refracted[0], refracted[1]])  # (x, y); refracted vector

    # Normalize
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)

    # Compute signed angle using atan2
    angle_rad = np.arctan2(u[0]*v[1] - u[1]*v[0], u[0]*v[0] + u[1]*v[1]) #refracted - incident
    angle_deg = np.rad2deg(angle_rad)

    return angle_deg # converts back to incident - refracted: definition of BAZ error, *-1

def deflection_yz(incident, refracted):
    """
    Args:
    Calculates the angle between the incident wave and refracted wave in the y-z plane
    
        incident: incident vector (3 component np.array)
        refracted: refracted vector (3 component np.array)

    Returns:
        angle_deg: angle in degrees of vector
    """
    # Project to YZ
    u = np.array([incident[1], incident[2]])  # (y, z)
    v = np.array([refracted[1], refracted[2]])  # (y, z)

    # Normalize
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)

    # Compute signed angle using atan2
    angle_rad = np.arctan2(u[0]*v[1] - u[1]*v[0], u[0]*v[0] + u[1]*v[1])
    angle_deg = np.rad2deg(angle_rad)

    return angle_deg


def rotate_about_z(v, angle_deg): ##v is a vector, angle_deg is the backazimuth error rotation
    angle = np.radians(angle_deg)
    R = np.array([
        [ np.cos(angle), -np.sin(angle), 0],
        [ np.sin(angle),  np.cos(angle), 0],
        [ 0,              0,             1]
    ])
    return R @ v


def incidence_angle(v): #incidence angle from vertical 
    v = v / np.linalg.norm(v)
    return np.degrees(np.arccos(v[2]))  # z = up


def horizontal_slowness(v, velocity): #v is vector, velocity is p-wave velocity of medium
    v = v / np.linalg.norm(v)
    theta = np.arccos(v[2])
    return np.sin(theta) / velocity


def calculate_deflection(strike, dip, oceanic_vel, continental_vel, distance_list, depth_list, azimuth_list, baz_list, event_id_list):

    """
    Args:
    Calculates forward model for deflection from snell 3D functions given an orientation of a dipping plane. 

    strike: strike of plane (degrees from north)
    dip: dip of plane (degrees down from horizontal)
    oceanic_vel: velocity below interface
    continental_vel: velocity above interface
    distance_list: list of epicentral distance to events
    depth_list: list of depths of earthquakes
    azimuth_list: azimuth of earthquakes
    baz_list: backazimuth of earthquakes
    event_id_list: list of event ids, for tracking
        

    Returns:
        DataFrame:
            source_baz: backazimuth input
            model_baz_error: model backazimuth error
            source_distance: distance to event 
            model_incident_deflection: model incident angle deflection
            model_slowness_error: model backazimuth error
            event_id: event_id list
    """
    #Calculate takeoff angle from depth
    takeoff = (np.rad2deg(np.arctan(np.array(distance_list)/np.array(depth_list))))

    ### data to be saved--------------------
    event_id = np.array(event_id_list)

    ##Input data to be used------------------ 
    baz = np.array(baz_list)
    #az = baz_to_az(baz) #from functions list
    az = np.array(azimuth_list)

    ##Calculate deflection-----------------
    normal = plane_normal(dip, strike)

    deflection_backazimuth = []
    deflection_incident = []
    deflection_slow = []

    for i in range(len(baz)):
        azimuth = az[i]
        takeoff1 = takeoff[i]
        #print('Takeoff angle',takeoff1)
        incident = spherical_to_xyz(azimuth, takeoff1)
        refracted = snell_3d(incident, normal, oceanic_vel, continental_vel)

        ###BAZ ERROR--------------------------------------
        deflection_baz = deflection_xy(incident, refracted)
        deflection_backazimuth.append(deflection_baz)

        ###SLOWNESS ERROR---------------------------------

        # Undo azimuthal deflection to get vertical variation
        refracted_unrot = rotate_about_z(refracted, 0) #-deflected_baz #gives refracted vector that is rotated in the x-y plane back into plane of incident wave

        #Calculate original incident angle/final incident angle
        theta_inc = incidence_angle(incident) #incident angle of incident wave
        theta_ref = incidence_angle(refracted) #incident angle of refracted wave
        
        #Incident angle error
        incident_error_deg = theta_inc - theta_ref #change in incident angle
        deflection_incident.append(incident_error_deg)

        #Slowness error
        p_inc = horizontal_slowness(incident, oceanic_vel) # slowness of incident wave, oceanic_vel, 6.04
        p_ref = horizontal_slowness(refracted_unrot, continental_vel) # slowness of refracted wave, continental_vel, 6.04

        #delta_p = p_ref - p_inc #slowness error between 
        delta_p = p_inc - p_ref  #slowness error: incident ray - refracted ray

        deflection_slow.append(delta_p)

    temp_deflect = pd.DataFrame({
            'source_baz': np.array(baz_list),
            'model_baz_error': np.array(deflection_backazimuth),
            'source_distance':np.array(distance_list),
            'model_incident_deflection': np.array(deflection_incident),
            'model_slowness_error': np.array(deflection_slow),
            'event_id': event_id
    })
    
    #temp_deflect.to_csv('/Users/cadequigley/Downloads/Research/paper_figures/'+array+'_3Dsnell_dip_'+str(dip)+'_strike_'+str(strike)+'.csv')
    print('3D Snells forward model finished')

    return temp_deflect

############################################################
#### FUNCTIONS FOR NIAZI FIT ###########################
############################################################

def cos_model(Z_deg, a, b, phi_deg):
    Z = np.radians(Z_deg)
    phi = np.radians(phi_deg)
    return a - b * np.cos(Z - phi)


############################################################
#### FUNCTIONS FOR 3D SNELLS INVERSION ###########################
############################################################

def combined_residuals(initial_guess, baz, takeoff, baz_error, slow_error, w_baz, w_p):
    
    """
    Args:
    Calculates residuals between model and observed for different strike/dip/continental_vel/oceanic vel combinations
    
        p: list of guess values [strike, dip, v_oceanic, v_continental]; list
        baz: source baz from USGS catalog (degrees from north); numpy.array
        takeoff: takeoff angle from depth or 1D velocity model; numpy.array
        baz_obs: observed baz deflection (degrees); numpy.array
        dp_obs: slowness deflection (s/km); numpy.array
        w_baz: weight for baz
        w_p: weight for slowness
        

    Returns:
        angle_deg: angle in degrees of vector
    """
    strike, dip, v_oceanic, v_continental = initial_guess

    N = len(baz)

    baz_res = np.zeros(N)
    p_res   = np.zeros(N)

    for i in range(N):
        azimuth = baz_to_az(baz[i])
        normal = plane_normal(dip, strike)
        incident = spherical_to_xyz(azimuth, takeoff[i])
        refracted = snell_3d(incident, normal, v_oceanic, v_continental)

        # --- BAZ residual (wrapped) ---
        baz_model = deflection_xy(incident, refracted)
        #baz_res[i] = np.angle(np.exp(1j * (baz_model - baz_obs[i]))) #for radians
        baz_res[i] = np.deg2rad((baz_model - baz_error[i] + 180) % 360 - 180) #for angles


        # --- Slowness residual ---
        refracted_unrot = rotate_about_z(refracted, 0)
        p_inc = horizontal_slowness(incident, v_oceanic)
        p_ref = horizontal_slowness(refracted_unrot, v_continental)
        p_model = p_inc - p_ref

        p_res[i] = p_model - slow_error[i]
        
        return np.hstack([w_baz * baz_res, w_p * p_res])

def slab_inversion(initial_guess, bounds, source_baz, takeoff, baz_error, slow_error, weight_baz, weight_slow):
    #####INVERSION######################################

    #Initial guess---------------------
    x0 = initial_guess

    #Value bounds---------------------
    bounds = bounds

    res = least_squares(
        combined_residuals,
        x0=x0,
        bounds=bounds,
        args=(source_baz, takeoff, baz_error, slow_error, weight_baz, weight_slow),
        )

    strike_fit, dip_fit, v_oceanic_fit, v_continental_fit = res.x

    print('Best strike:', strike_fit)
    print('Best dip:', dip_fit)
    print('Best oceanic vel:', v_oceanic_fit)
    print('Best continental vel:', v_continental_fit)
    return strike_fit, dip_fit, v_oceanic_fit, v_continental_fit
