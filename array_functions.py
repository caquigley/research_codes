import pandas as pd
import obspy
import matplotlib.pyplot as plt
import numpy as np
from obspy import read
from obspy import read_events
from obspy import Stream
from obspy import Trace
from obspy import UTCDateTime
from datetime import datetime, time, timezone
import datetime as dt
from scipy.optimize import least_squares
from obspy.taup import TauPyModel
from obspy.taup import taup_create
from obspy.geodetics import gps2dist_azimuth
from obspy.geodetics import kilometers2degrees
from obspy.signal.util import util_geo_km
from obspy.signal.trigger import recursive_sta_lta, trigger_onset, classic_sta_lta


############################################################
#### FUNCTIONS FOR PULLING EARTHQUAKES ###########################
############################################################
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

def data_from_inventory(inv, remove_stations):

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

        

    return lat_list, lon_list, elev_list, station_list, start_list, end_list, num_channels_list

def check_num_stations(min_stations, station_list):
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

def utc2datetime(utctime): #utc time as string
    return dt.datetime(int(utctime[0:4]),int(utctime[5:7]), int(utctime[8:10]), int(utctime[11:13]),int(utctime[14:16]),int(utctime[17:19]))
    
def is_between(check, start, end): #Returns true/false based on whether time is between two values
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


def baz_error(baz_real, baz_calculated):
    baz_error_temp = baz_real - baz_calculated
    baz_error = ((baz_error_temp + 180) % 360) - 180
    return baz_error

    

def trigger_associator(st, estimated_p_arrival):
    
    expected = estimated_p_arrival  # seconds
    tolerance = 10.0     # seconds

    trigger_list = []

    for k in range(len(st)):
        tr = st[k]
        sr = tr.stats.sampling_rate
        cft = classic_sta_lta(tr.data, int(2.5 * sr), int(30. * sr)) #classic sta/lta
        on_of = trigger_onset(cft, 2.5, 1.0) #1.7, 1.0

        if len(on_of) > 0: 
            for i in range(len(on_of)):
                triggers = on_of[:, 0]/sr #triggers
            
                mask = np.abs(triggers - expected) <= tolerance
                trigger_filtered = triggers[mask]
                if len(trigger_filtered) >0:
                    trigger_list.append(trigger_filtered[0])
            
    

    if len(trigger_list) > 5:
        trigger = np.median(trigger_list)
        trigger_type = 'trigger'
    else:
        trigger = expected
        trigger_type = 'estimated'
    return trigger, trigger_type

def rotate_channel(st):
    if stats.channel[-1] == 'Z':
        data = -data



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

### Trying some new deflections in slowness plane

### Function to rotate refracted ray back into azimuth-normal plane
def rotate_about_z(v, angle_deg): ##v is a vector, angle_deg is the backazimuth error rotation
    angle = np.radians(angle_deg)
    R = np.array([
        [ np.cos(angle), -np.sin(angle), 0],
        [ np.sin(angle),  np.cos(angle), 0],
        [ 0,              0,             1]
    ])
    return R @ v

## Function to calculate incidence angle from vector
def incidence_angle(v): #incidence angle from vertical 
    v = v / np.linalg.norm(v)
    return np.degrees(np.arccos(v[2]))  # z = up

### Definition of horizontal slowness
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
