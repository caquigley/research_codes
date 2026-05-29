#%%
from obspy import UTCDateTime
import pandas as pd
from obspy.clients.fdsn import Client
from obspy import read_inventory
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml
import sys
import traceback

#Functions needed for processing----------------------------
from array_functions import (data_from_inventory, get_geometry, 
                             pull_earthquakes, check_num_stations, 
                             stations_available_generator,
                             array_time_window, moveout_time, grab_preprocess,
                             least_trimmed_squares, triggers, fk_obspy)

#Functions needed for plotting----------------------------
from array_figures import (baz_error_spatial, 
                           slow_error_spatial, 
                           time_series_density,
                           histogram)

from array_maps_pygmt import (pygmt_array_earthquakes, pygmt_baz_error, 
                             pygmt_slow_error)

#%%
'''
Conducts array analysis for a set array and number of events in the vicinity
of the array. This can be used to determine how well an array is performing,
and how errors are occuring spatially.
    
Parameters:
    Parameters are defined in the input_parameters.yaml file. For more 
    information on possible parameters, go to the Github 
    Wiki: https://github.com/caquigley/array_aggregator/wiki/Input-parameters

Returns: 
    df: dataframe containing earthquake information and array output parameters.
        See Github 
        wiki: https://github.com/caquigley/array_aggregator/wiki/Outputs

    stations: dataframe with information about stations used (see Github
        Wiki: https://github.com/caquigley/array_aggregator/wiki/Outputs)
    plots: 
        - map of earthquakes
        - baz error
        - slowness error
        - map of baz error
        - map of slowness error
        
        
    '''
def read_params(params):
    '''
    Read parameters from .yaml file

    Inputs: 
        params: input parameter (input_parameters.yaml) file read into python

    Returns:
        variables needed for computation

    '''
    #Network inputs----------
    net = params["network"]["net"] 
    sta = params["network"]["sta"]  
    loc = params["network"]["loc"] 
    chan = params["network"]["chan"] 
    client_str = params["network"]["client"]

    #Station inputs----------

    min_stations = params["stations"]["min_stations"]
    remove_stations = params["stations"]["remove_stations"]
    keep_stations = params["stations"]["keep_stations"]
    array_name = params["stations"]["array_name"]
    use_full_deployment = params["stations"]["use_full_deployment"]
    path_to_inventory = params["stations"]["path_to_inventory"]
    save_events = params["stations"]["save_events"]
    save_stations = params["stations"]["save_stations"]

    #MSEED parameters-------------------
    save_mseed = params["mseed"]["save_mseed"]
    mseed_path = params["mseed"]["mseed_path"]
    mseed_length = params["mseed"]["mseed_length"]

    #Earthquake inputs----------
    min_mag = str(params["earthquakes"]["min_mag"])
    max_rad = str(params["earthquakes"]["max_rad"])
    velocity_model = params["earthquakes"]["velocity_model"]
    starttime = params["earthquakes"]["starttime"]
    endtime = params["earthquakes"]["endtime"]

    #Array processing inputs---------------
    processing = params["array_processing"]["processing"]
    FREQ_MIN = params["array_processing"]["freq_min"]
    FREQ_MAX = params["array_processing"]["freq_max"]
    WINDOW_LENGTH = params["array_processing"]["window_length"]
    WINDOW_STEP = params["array_processing"]["window_step"]
    window_start = params["array_processing"]["window_start"]

    # STA/LTA inputs-------------------

    timing = params["trigger"]["timing"]
    min_triggers = min_stations // 3 #minimum station triggers to associate
    ptolerance = params["trigger"]["ptolerance"]
    multiple_triggers = params["trigger"]["multiple_triggers"]
    no_triggers = params["trigger"]["no_triggers"]

    #Following inputs representative of EPIC parameters
    trig_freq_min = params["trigger"]["trig_freq_min"]
    trig_freq_max = params["trigger"]["trig_freq_max"]
    short_window = params["trigger"]["short_window"]
    long_window = params["trigger"]["long_window"]
    on_threshold = params["trigger"]["on_threshold"]
    off_theshold = params["trigger"]["off_threshold"]
    analysis_type = params["trigger"]["analysis_type"]
    time_series_start = params["trigger"]["time_series_start"]
    time_series_end = params["trigger"]["time_series_end"]
    use_EPIC_triggers = params["trigger"]["use_EPIC_triggers"]
    

    #Inputs for FK array processing---------

    sll_x = params["fk"]["sll_x"]
    slm_x = params["fk"]["slm_x"]
    sll_y = params["fk"]["sll_y"]
    slm_y = params["fk"]["slm_y"]
    sl_s = params["fk"]["sl_s"]
    semb_thres = params["fk"]["semb_thres"]
    vel_thres = params["fk"]["vel_thres"]
    timestamp = params["fk"]["timestamp"]
    prewhiten = params["fk"]["prewhiten"]

    #Inputs for plots----------------
    earthquake_map = params["plots"]["earthquake_map"]
    baz_error_plot = params["plots"]["baz_error"]
    slow_error_plot = params["plots"]["slow_error"]
    baz_error_map = params["plots"]["baz_error_map"]
    slow_error_map = params["plots"]["slow_error_map"]
    plot_histogram = params["plots"]["plot_histogram"]
    save_fig = params["plots"]["save_fig"]
    fig_path = params["plots"]["fig_path"]
    drop_pow = params["plots"]["drop_pow"]
    pow_thresh = params["plots"]["pow_thresh"]


    #Handles cases for single freq_min/freq_max/window_length given
    if isinstance(FREQ_MAX, (float, int)):
        FREQ_MAX = [FREQ_MAX]
    if isinstance(FREQ_MIN, (float,int)):
        FREQ_MIN = [FREQ_MIN]
    if isinstance(WINDOW_LENGTH, (float,int)):
        WINDOW_LENGTH = [WINDOW_LENGTH]


    return (net, sta, loc, chan, client_str, min_stations, remove_stations, 
            keep_stations, array_name, use_full_deployment, path_to_inventory,
            save_events, save_stations, save_mseed, mseed_path, mseed_length,
            min_mag, max_rad, velocity_model, starttime, endtime, processing,
            FREQ_MIN, FREQ_MAX, WINDOW_LENGTH, WINDOW_STEP, window_start, 
            timing, min_triggers, ptolerance, multiple_triggers, no_triggers,
            trig_freq_min, trig_freq_max, short_window, long_window, 
            on_threshold, off_theshold, analysis_type, time_series_start,
            time_series_end, use_EPIC_triggers, sll_x, slm_x, sll_y, slm_y, sl_s, 
            semb_thres, vel_thres, timestamp, prewhiten, earthquake_map, 
            baz_error_plot, slow_error_plot, baz_error_map, slow_error_map,
            plot_histogram, save_fig, fig_path, drop_pow, pow_thresh)


def preprocess_earthquakes(lat_list, lon_list, elev_list, use_full_deployment, 
                           start_d1_list, end_d1_list, starttime, endtime, 
                           max_rad, min_mag, array_name, velocity_model,
                           min_stations):
    
    '''
    Pulls earthquakes in the vicinity of the array based on specified magnitude/
    distance range and deployment time. It then calculates the catalog baz/
    slowness. It then removes events where there are not enough stations
    operating.

    Inputs:
        lat_list: list of station latitudes
        lon_list: list of station longitudes
        elev_list: list of station elevations
        use_full_deployment: whether or not to use full deployment time
        start_d1_list: start times from station list
        end_d1_list: end times from station list
        starttime: start time for pulling earthquakes
        endtime: end time for pulling earthquakes
        max_rad: maximum radius for pulling earthquake data
        min_mag: minimum magnitude for pulling earthquake data
        array_name: name of array for saving data later
        velocity_model: name of velocity model for TauP calculations
        min_stations: minimum number of stations to run array analysis
    Returns:
        df: dataframe of earthquakes, including: origin time, magnitude,
            distance, slowness, backazimuth, event_id
        moveout: how fast the seismic wave is expected to move across the
            array
        origin_lat: latitude of center of array
        origin_lon: longitude of center of array
        stations_lists: list of stations available for each event.

    '''
    #Get center of array -----------------
    output = get_geometry(lat_list, lon_list, elev_list, return_center = True)
    origin_lat = str(output[-1][1])
    origin_lon = str(output[-1][0])

    # Get expected moveout time across array--------
    moveout = moveout_time(output)

    #Pull earthquakes during deployment------------
    start, end = array_time_window(use_full_deployment, start_d1_list, 
                                   end_d1_list, starttime, endtime)
    df = pull_earthquakes(origin_lat, origin_lon, max_rad, start, end, min_mag, 
                        array_name, velocity_model)
    print('Number of earthquakes >'+min_mag+' within '+max_rad+' km:', len(df))


    #Create station availability lists-----------------------
    #------------------------------------------------
    earthquake_time = df['time_utc'].to_numpy()

    (stations_lists, 
    stations_available) = stations_available_generator(earthquake_time, 
                                                        station_d1_list, 
                                                        start_d1_list, 
                                                        end_d1_list)

    ### Drop events that don't have enough stations present--------------
    bad_idx = [i for i, v in enumerate(stations_available) if v < min_stations]
    keep_idx = [i for i, v in enumerate(stations_available) if v >= min_stations]

    ### Drop events from dataframe without enough stations-------------
    stations_available = [stations_available[i] for i in keep_idx]
    stations_lists = [stations_lists[i] for i in keep_idx]
    df = df.drop(index=bad_idx)
    df = df.reset_index(drop = True)
    
    print('Station lists for each earthquake created. New earthquake number:', 
          len(df))

    return df, moveout, origin_lat, origin_lon, stations_lists


def process_event(event, event_ids, mag, eq_time, client_str, stations_lists, 
                  eq_slow, eq_baz,expected_parrival,mseed_length, station_info,
                  inv, net, loc, chan, min_stations, array_name, save_mseed, 
                  mseed_path, short_window, long_window, on_threshold, 
                  off_theshold, moveout, min_triggers, ptolerance, window_start, 
                  window_length, freq_min,freq_max, trig_freq_min, 
                  trig_freq_max, multiple_triggers, no_triggers, WINDOW_OVERLAP,
                  sll_x, slm_x, sll_y, slm_y, sl_s, semb_thres,vel_thres, 
                  timestamp, prewhiten, timing, velocity_model, processing, 
                  origin_lat, origin_lon, 
                  analysis_type, time_series_start, time_series_end, 
                  use_EPIC_triggers = False, epic_triggers = None,
                  pvsnr = None):
    
    '''
    Function for pulling data, identifying STA/LTA triggers, and conducting
    array analysis for each event from earthquake list. This functions acts
    as a wrapper so that each event can be split onto different cores using
    concurrent.futures.

    '''

    try:
        #Pull seismic data------------------------------
        #-----------------------------------------------
        print("Starting", event_ids[event], 'Ml', mag[event], eq_time[event])

        client = Client(client_str)
        stations = stations_lists[event]
        eq_slow_real = eq_slow[event]
        eq_baz_real = eq_baz[event]
        event_id = event_ids[event]

        START = (UTCDateTime(eq_time[event])
            + expected_parrival[event]
            - (mseed_length / 2)
            )
        END = START + mseed_length

        # Grab and preprocess data----------------
        (st, stations, sta_lats,
         sta_lons, sta_elev) = grab_preprocess(
            stations, station_info, inv,
            net, loc, chan, min_stations,
            START, END, client, array_name,
            event_id, mseed_path, save_mseed)

        st1 = st.copy()

        #Calculate STA/LTA trigger times------------------------------
        #-----------------------------------------------
        #If existing dataset of triggers exist
        if use_EPIC_triggers == True:
            print('EPIC trigger')

            try: #See if triggers exist
                trigger = abs(START - UTCDateTime(epic_triggers[event]))
                if analysis_type == 'time_series':
                    START_new = START + trigger + time_series_start
                    END_new = START + trigger + time_series_end
                if analysis_type == 'single_point':
                    #START_new = START + trigger + window_start- 0.001
                    #END_new = START_new + window_length
                    
                    if timing == 'trigger': #analysis based on timing
                        #Trim stream to window of interest-------------
                        START_new = START + trigger + window_start- 0.001
                        END_new = START_new + window_length
                        
                    elif timing == 'power': #Find max power in a tolerance window
                        #Trim stream to window of interest-------------
                        START_new = START + trigger + window_start- 0.001 - ptolerance
                        END_new = START_new + 2*ptolerance

        
                st = st.slice(START_new, END_new)
                peak = pvsnr[event]
                trigger_time = str(START+trigger)
                length = 0
                trigger_type = 'EPIC'
            except FileNotFoundError:
                raise ValueError("Earthquake not found in " \
                                "provided dataset, skipping.")
        
        #If calculating new dataset of triggers
        elif use_EPIC_triggers == False: 
            (st, trigger, peak, length,
            trigger_type, trigger_time,
            START_new, END_new) = triggers(
                    st, short_window, long_window,
                    on_threshold, off_theshold,
                    moveout, min_triggers,
                    ptolerance, START,
                    window_start,
                    window_length, freq_min,
                    freq_max, trig_freq_min,
                    trig_freq_max,
                    multiple_triggers,
                    mseed_length, timing, no_triggers,
                    analysis_type, time_series_start, time_series_end)
            
        


        # Array processing------------------------------
        #-----------------------------------------------
        if processing == 'lts' or processing == 'ls':

            (array_data, time_series, baz_error_series,
             slow_error_series) = least_trimmed_squares(
                processing, st, sta_lats, sta_lons,
                window_length, WINDOW_OVERLAP,
                eq_baz_real, eq_slow_real, analysis_type)

        elif processing == 'fk':
            
            (array_data, time_series, 
             baz_error_series, slow_error_series)= fk_obspy(
                st1, analysis_type, stations, sta_lats, sta_lons, sta_elev,
                START_new, END_new, window_length,
                WINDOW_OVERLAP, freq_min, freq_max, float(sll_x),
                float(slm_x), float(sll_y), float(slm_y), float(sl_s), 
                float(semb_thres), float(vel_thres), timestamp, prewhiten,
                eq_baz_real, eq_slow_real)
            

        #Save metadata------------------------------
        #-----------------------------------------------
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

        return array_data, time_series, baz_error_series, slow_error_series

    #Handles data that has errors raised so code continues to run.-----
    #except ValueError as e:
        #print(f"Skipping event {event_ids[event]}: {e}")
        #return None, None, None, None
    

    except ValueError as e:
        print(f"Skipping event {event_ids[event]}:")
        print(traceback.format_exc())
        return None, None, None, None

    except Exception as e:
        print(f"Unexpected error for event {event_ids[event]}: {e}")
        return None, None, None, None
    




if __name__ == "__main__":

    #############################################################
    #----------STEP 1: LOAD INPUTS FROM PARAMETER FILE-----------
    #############################################################
    with open(sys.argv[1]) as f:
        params = yaml.safe_load(f)
    
    (net, sta, loc, chan, client_str, min_stations, remove_stations, 
            keep_stations, array_name, use_full_deployment, path_to_inventory,
            save_events, save_stations, save_mseed, mseed_path, mseed_length,
            min_mag, max_rad, velocity_model, starttime, endtime, processing,
            FREQ_MIN, FREQ_MAX, WINDOW_LENGTH, WINDOW_STEP, window_start, 
            timing, min_triggers, ptolerance, multiple_triggers, no_triggers, 
            trig_freq_min, trig_freq_max, short_window, long_window, 
            on_threshold, off_theshold, analysis_type, time_series_start, 
            time_series_end, use_EPIC_triggers, sll_x, slm_x, sll_y, slm_y, sl_s, 
            semb_thres, vel_thres, timestamp, prewhiten, earthquake_map, 
            baz_error_plot, slow_error_plot, baz_error_map, slow_error_map,
            plot_histogram, 
            save_fig, fig_path, drop_pow, pow_thresh) = read_params(params)
    

    #%%
    ##################################################
    #-----STEP 2: LOAD STATION INFORMATION FROM CLIENT
    ##################################################

    #Pull inventory-----------------------
    #------------------------------------------------
    if client_str == 'path':
        inv = read_inventory(path_to_inventory) 
    else:
        client = Client(client_str)
        inv = client.get_stations(network=net, station=sta, channel=chan,
                                    location=loc, 
                                    starttime=UTCDateTime(starttime),
                                    endtime=UTCDateTime(endtime), 
                                    level='response')
    
    #Pull station information out of inventory-----
    (lat_list, lon_list, elev_list, station_d1_list,
    start_d1_list, end_d1_list, num_channels_d1_list) = data_from_inventory(
        inv,
        remove_stations,
        keep_stations)

    #Check if enough stations present to continue------
    check = check_num_stations(min_stations, station_d1_list)

    #Save stations for later-------
    data = {
            'station': station_d1_list,
            'lat': lat_list,
            'lon': lon_list,
            'elevation': elev_list}

    station_info = pd.DataFrame(data) 

    #################################################################
    #-----STEP 3: FIND EARTHQUAKES AND CALCULATE CATALOG BAZ/SLOWNESS
    #################################################################

    (df, moveout, origin_lat, 
     origin_lon, stations_lists) = preprocess_earthquakes(lat_list, 
                           lon_list, elev_list, use_full_deployment, 
                           start_d1_list, end_d1_list, starttime, endtime, 
                           max_rad, min_mag, array_name, velocity_model,
                           min_stations)
    
    #len_before = len(df)

    if use_EPIC_triggers == True:
        epic = pd.read_csv('/Users/cadequigley/Downloads/completed_runs_v2/record_sections_picks/'+array_name+'_epic_times.csv')
        epic = epic.rename(columns={'eq_name': 'event_id'})
        df = df.merge(epic, on='event_id', how='inner')
        print('Number of earthquakes with EPIC triggers:', len(df))
        epic_triggers = df['time'].to_numpy()
        pvsnr = df['pvsnr'].to_numpy()
    else:
        epic_triggers = 0
        pvsnr = 0
        
    len_before = len(df)
    #%%
    ##################################
    #-----STEP 4: LOOP OVER ALL EVENTS
    ##################################
    
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
    time_series_list = []
    baz_error_list = []
    slow_error_list = []

    #Loop through window lengths-------------
    for window in range(len(WINDOW_LENGTH)):

        window_length = WINDOW_LENGTH[window]
        WINDOW_OVERLAP = (window_length-WINDOW_STEP)/window_length 
        WINDOW_OVERLAP = WINDOW_STEP/window_length
        WINDOW_OVERLAP = 0.5
        #WINDOW_OVERLAP = 1 - WINDOW_STEP/window_length
        #print(WINDOW_OVERLAP)
        freq_min = FREQ_MIN[window]
        #Loop through frequencies-----------
        for freq in range(len(FREQ_MAX)):
            #freq_min = FREQ_MIN[freq]
            freq_max = FREQ_MAX[freq]


            print(f"Starting analysis for {window_length} s window "
                f"and {freq_min}-{freq_max} Hz bandpass filter")
            
            #Do analysis on each event-----------------------
            #------------------------------------------------

            #Splitting events onto multiple cores
            with ProcessPoolExecutor() as executor: 
                futures = [executor.submit(process_event, event, event_ids, mag,
                    eq_time, client_str, stations_lists, eq_slow, eq_baz, 
                    expected_parrival, mseed_length, station_info, inv, net, 
                    loc, chan, min_stations, array_name, save_mseed, mseed_path,
                    short_window, long_window, on_threshold, off_theshold, 
                    moveout, min_triggers, ptolerance, window_start, 
                    window_length, freq_min,freq_max, trig_freq_min,
                    trig_freq_max, multiple_triggers, no_triggers, 
                    WINDOW_OVERLAP, sll_x, slm_x, sll_y, slm_y, sl_s, 
                    semb_thres, vel_thres, timestamp, prewhiten, timing, 
                    velocity_model, processing, origin_lat, 
                    origin_lon, analysis_type, time_series_start,
                    time_series_end, use_EPIC_triggers, 
                    epic_triggers, pvsnr) for event in range(len(df))]

                for i, future in enumerate(as_completed(futures)):
                    result, series1, series2, series3 = future.result()

                    if result is not None:
                        array_data_list.append(result)
                        time_series_list.append(series1)
                        baz_error_list.append(series2)
                        slow_error_list.append(series3)

    #######################
    #-----STEP 5: SAVE DATA
    #######################
    
    #Putting data into single dataframe----------------------
    array_data_comb1 = pd.concat(array_data_list, ignore_index=True)

    #Combining with earthquake data-----------------------
    array_data_comb = pd.merge(array_data_comb1, df, on='event_id', how='inner')

    print('Number of events dropped for processing error:', len_before - len(array_data_comb))
    #Save to csv-----------------------------------------------------
    if save_events == True:
        #array_data_comb.to_csv(array_name+'_'+max_rad+'km_m'+str(int(float(min_mag)))
                               #+'_'+processing+'_'+str(window_length)
                               #+'_window_freq_test.csv')
        array_data_comb.to_csv(array_name+'_'+max_rad+'km_m'+str(int(float(min_mag)))
                               +'_'+processing+'_'
                               +'_window_freq_test2.csv')

    if save_stations == True:
        station_info.to_csv(array_name+'_'+max_rad+'km_m'
                            +str(int(float(min_mag)))+'_'+processing+'_'+str(window_length)+'stations.csv')


    #Time series analysis--------------------------------
    
    
    if analysis_type == 'time_series':
        import numpy as np
        #array_data_comb = array_data_comb.dropna()
        tlist = []
        for k in range(len(array_data_comb)):
            trig = UTCDateTime(array_data_comb['trigger_time'].to_numpy()[k])
            t = time_series_list[k]
            t_list = []
            for l in range(len(t)):
                t_list.append(trig - UTCDateTime(t[l]))
            baz = baz_error_list[k]
            tlist.append(-1*np.array(t_list))

        #Drop out taup events--------------------
        bad_idx = array_data_comb.index[array_data_comb["trigger_type"] == "Taup"].tolist()
        tlist = [arr for i, arr in enumerate(tlist) if i not in bad_idx]
        baz_error_list = [arr for i, arr in enumerate(baz_error_list) if i not in bad_idx]
        slow_error_list = [arr for i, arr in enumerate(slow_error_list) if i not in bad_idx]

        


        time_series = np.concatenate(tlist)
        baz_error = np.concatenate(baz_error_list)
        
        mask = np.isfinite(baz_error)

        time_series = time_series[mask]
        baz_error = baz_error[mask]

        #print(baz_error)
        time_series_density(time_series, baz_error, time_series_start, time_series_end, save = False)
            
    
    ###############################
    #-----STEP 6: PLOT SOME FIGURES
    ###############################

    df1 = array_data_comb
    df = df1.dropna()
    print('Number of dropped events for nans:', len(df1) - len(df))

     
    #Plot map of earthquakes-----------------------------
    #----------------------------------------------------
    array_lats = [float(origin_lat)]
    array_lons = [float(origin_lon)]
    array_names = [array_name]
    array_names = []
    earthquake_lats = df['latitude'].to_numpy()
    earthquake_lons = df['longitude'].to_numpy()
    earthquake_mags = df['magnitude'].to_numpy()
    earthquake_depths = df['depth'].to_numpy()
    
    if earthquake_map == True:
        pygmt_array_earthquakes(array_lats, array_lons, array_names, 
                                earthquake_lats,earthquake_lons, 
                                earthquake_mags, earthquake_depths,
                                save=save_fig, 
                                path = fig_path+'earthquake_map.png')

    #Plot baz_slow_error-----------------------------
    #----------------------------------------------------
    drop = True #drop Taup picks, i.e. events without an STA/LTA pick
    if drop ==True:
        temp = pd.DataFrame(df[df['trigger_type']!= 'Taup'])
        print('Number of dropped events for Taup:', len(df) - len(temp))
        df = temp

    if drop_pow == True:
        if processing == 'fk':
            temp = df[df["relpow"] >= pow_thresh]
        elif processing == 'lts' or 'ls':
            temp = df[df["mdccm"] >= pow_thresh]

        print('Number of dropped events for low power:', len(df) - len(temp))
        df = temp

    if processing == 'fk':
        color_data = df['relpow']
    else: 
        color_data = df['mdccm']
    
    #color_data = df['conf_int_baz']
    #color_data = df['magnitude']
    color_label = 'cross correlation/power'
    model_data = []
    
    if baz_error_plot == True:
        baz_error_spatial(df["backazimuth"], df["baz_error"], model_data,
            color_data, color_label, niazi=True, save=save_fig,
            path=fig_path + "baz_error_spatial.png")   

    if slow_error_plot == True:
        slow_error_spatial(df["backazimuth"], df["slow_error"], model_data,
            color_data, color_label, niazi=True, save=save_fig,
            path=fig_path + "slow_error_spatial.png")  
        
    outliers = pd.DataFrame(df[df['slow_error'] < -0.2])

    print('Number of outliers below slowness threshold: ', len(outliers))

    #Plot baz error on map-----------------------------
    #----------------------------------------------------
    baz = df['backazimuth'].to_numpy()
    baz_error = df['baz_error'].to_numpy()
    slow_error = df['slow_error'].to_numpy()
    earthquake_lats = df['latitude'].to_numpy()
    earthquake_lons = df['longitude'].to_numpy()
    earthquake_mags = df['magnitude'].to_numpy()
    earthquake_depths = df['depth'].to_numpy()

    if baz_error_map == True:

        pygmt_baz_error(array_lats[0], array_lons[0], array_name, 
                        earthquake_lats, earthquake_lons, earthquake_mags, baz,
                        baz_error, save = save_fig, 
                        path = fig_path+'baz_error_map.png')
                        
    
    #Plot slowness error on map-----------------------------
    #----------------------------------------------------
    if slow_error_map == True:
        pygmt_slow_error(array_lats[0], array_lons[0], array_name, 
                        earthquake_lats, earthquake_lons, earthquake_mags, 
                        slow_error, save = save_fig, 
                        path = fig_path+'slow_error_map.png')
        
    if plot_histogram == True:
        upper_quantile = 0.95
        lower_quantile = 0.05
        variable_name = 'backazimuth_error'
        histogram(baz_error, lower_quantile, upper_quantile, variable_name, 
              save = save_fig, path = fig_path+'backaimuth_histogram.png')
        
        variable_name = 'slowness_error'
        histogram(slow_error, lower_quantile, upper_quantile, variable_name, 
              save = save_fig, path = fig_path+'slowness_histogram.png')