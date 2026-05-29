import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from array_functions import calculate_slowness, calculate_deflection
from array_figures import baz_error_spatial, slow_error_spatial

from obspy.geodetics import gps2dist_azimuth
from obspy.geodetics import kilometers2degrees
from obspy.taup import TauPyModel

#Inputs-------------------------------------
df = pd.read_csv("~/Downloads/offsets_coordinates.csv")
df = df.rename(columns={'ID': 'event_id'})

array_lat = 47.93927
array_lon = -124.55857
velocity_model = 'iasp91'
strike = 340
dip = 10
oceanic_vel = 8.04
continental_vel = 6.2
save_csv = True

#Calculate slowness---------------------------
lats = df['Lat'].to_numpy()
lons = df['Lon'].to_numpy()
depth = df['Depth'].to_numpy()
event_id = df['event_id'].to_numpy()
baz_list = []
az_list = []
distance_list = []
slowness = []
tvel = []
incident_list = []
takeoff = []
azimuth = []
parrival = []

for i in range(len(df)):

    lat = lats[i]
    lon = lons[i]
    depth_temp = depth[i]


    
    dist, baz, az = gps2dist_azimuth(array_lat, array_lon, 
                                         lat, lon)
    
    dist = dist/1000
    distance_list.append(dist)
    baz_list.append(baz)
    az_list.append(az)

    slow_surf, t_vel, incident, p = calculate_slowness(dist, 
                                                  depth_temp,
                                                  velocity_model)
    
    slowness.append(slow_surf)
    tvel.append(t_vel)
    incident_list.append(incident)
    parrival.append(p)
    #Calculate angle to approach
    tmod = TauPyModel(model = velocity_model)
    dist_deg = kilometers2degrees(dist)
    arrivals = tmod.get_pierce_points(depth_temp, dist_deg, 
                                      #phase_list=["P", "p"],
                                      receiver_depth_in_km=10)

       
    incident = arrivals[0].incident_angle
    takeoff.append(incident)

#Calculate deflection-------------------------------
distance = np.array(distance_list)
takeoff = np.array(takeoff)
baz = np.array(baz_list)
az = np.array(az_list)
model = calculate_deflection(strike, dip, oceanic_vel, continental_vel, distance, depth, takeoff, az, baz, event_id)


#Save data--------------------------

array_data_comb = pd.merge(model, df, on='event_id', how='inner')
if save_csv == True:
    array_data_comb.to_csv("/Users/cadequigley/Downloads/uw_earthquakes_3d_snell.csv")
    print("CSV saved")
#Plot-------------------------------

color_label = 'distance (km)'
model_data = []

model_data_baz = model['model_baz_error'].to_numpy()
model_data_slow = model['model_slowness_error'].to_numpy()
color_data = distance
    

baz_error_spatial(baz, model_data_baz, model_data,
        color_data, color_label, niazi=False, save = True, 
        path = '/Users/cadequigley/Downloads/uw_projected_baz_error.png') 
slow_error_spatial(baz, model_data_slow, model_data,
        color_data, color_label, niazi=False, save = True, 
        path = '/Users/cadequigley/Downloads/uw_projected_slow_error.png') 