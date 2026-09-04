#%%
import numpy as np
import pandas as pd
from obspy.taup import TauPyModel
from obspy.geodetics import kilometers2degrees

from array_figures import (baz_error_spatial, 
                           slow_error_spatial, 
                           time_series_density,
                           histogram,
                           stacked_histogram)

from array_maps_pygmt import (pygmt_array_earthquakes, pygmt_baz_error, 
                             pygmt_slow_error_new)

from array_functions import baz_to_az, cos_model, fourier5
from array_functions import calculate_deflection, anisotropic_harmonic
from array_functions import slab_inversion, niazi_dip_inversion
from array_maps_pygmt import pygmt_single_event
from array_figures import baz_error_spatial, slow_error_spatial

array_name = 'KD' #'HM', "KD"
save_fig = False
fig_path = '/Users/cadequigley/Downloads/Research/deployment_array_design/datamine_paper_figures/'

pow_thresh = 0.08
drop_pow = True
drop_taup = True #drop Taup picks, i.e. events without an STA/LTA pick
processing = 'fk'

array_lats = [57.441811, 59.618433]  
array_lons = [-152.352174, -151.141416]
array_names = ['KD', 'HM']
if array_name == 'HM':
    origin_lat = array_lats[1]
    origin_lon = array_lons[1]
elif array_name == 'KD':
    origin_lat = array_lats[0]
    origin_lon = array_lons[0]

#Slab modeling-----------------
model = 'inversion' #'inversion', 'fixed'
strike = 269 #272 #277
dip = 20 #17 #29
oceanic_vel = 8.04 #8.3
continental_vel = 5.0 #4 #4, 6.2, 5.8
takeoff_type = 'surface' #'moho', 'surface', 'source'
weight_baz = 1
weight_slow = 0

df = pd.read_csv('./'+array_name+'_1000km_m2_fk_4_window_freq_test.csv')
# %%



df = df.dropna()
    #print('Number of dropped events for nans:', len(df1) - len(df))

     
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
    
    
    #Plot baz_slow_error-----------------------------
    #----------------------------------------------------

if drop_taup ==True:
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


####################################
###Forward model processing-----------
####################################


### Forward model
baz = df['backazimuth'].to_numpy()
az = baz_to_az(baz)
depth = df['depth'].to_numpy()
distance = df['distance'].to_numpy()
event_id = df['event_id'].to_numpy()

if takeoff_type == 'source':
    # Takeoff from distance/depth in flat earth--------------------------
    takeoff = np.rad2deg(np.arctan(df['distance'].to_numpy()/df['depth'].to_numpy()))
elif takeoff_type == 'surface':
    #Takeoff from expected surface arrival incident angle from Taup---------------
    takeoff = df['incident_angle'].to_numpy()
elif takeoff_type == 'moho':
    #Takeoff from expected incident angle at moho-------------------

    tmod = TauPyModel(model = 'pavdut')
    incident_angle_list = []
    for i in range(len(distance)):
        dist_deg = kilometers2degrees(distance[i])
        arrivals = tmod.get_pierce_points(depth[i], dist_deg, 
                                        phase_list=["P", "p"],
                                        receiver_depth_in_km=15) #10

        #p = arrivals[0].ray_param /6371
        #r = 6371 - 20
        #incident = np.rad2deg(np.arcsin((p * 8.2)/r))
        #print('Incident angle:', incident)
        #incident_angle_list.append(incident)
        incident = arrivals[0].incident_angle
        incident_angle_list.append(incident)
        #print('Incident angle:', incident)
        #print(arrivals)
    takeoff = np.array(incident_angle_list)
#takeoff = np.ones(len(takeoff))*40
#print(takeoff)
baz_error = df['baz_error'].to_numpy()
slow_error = df['slow_error'].to_numpy() #+ 0.025

if model == 'inversion':

    #Inputs for interface inversion---------
    initial_guess = [
        249.0,   # strike, based on trench strike
        10.0,   # dip, based on subduction zone dip
        8.04,    # oceanic velocity
        6.2     # continental velocity
        ]

    #Value bounds---------------------
    bounds = ( #strike, dip, oceanic_vel, continental_vel
        [0,   0,   8.03, 4], #lower bounds, 5.8, 4
        [360, 90,  8.04, 8] #upper bounds
            )

    #residuals = combined_residuals(initial_guess, baz, takeoff, baz_error, slow_error, weight_baz, weight_slow)
    strike_fit, dip_fit, v_oceanic_fit, v_continental_fit = slab_inversion(initial_guess, bounds, baz, takeoff, baz_error, slow_error, weight_baz, weight_slow)


    model = calculate_deflection(strike_fit, dip_fit, v_oceanic_fit, v_continental_fit, distance, depth, takeoff, az, baz, event_id)

elif model == 'fixed':
    #Fixed values
    model = calculate_deflection(strike, dip, oceanic_vel, continental_vel, distance, depth, takeoff, az, baz, event_id)
#%%
model_data_baz = model['model_baz_error'].to_numpy()
model_data_slow = model['model_slowness_error'].to_numpy()
    
#color_data = df['conf_int_baz']
#color_data = df['magnitude']
print('Number of remaining events:', len(df))
color_label = 'cross correlation/power'
#model_data_baz = []
#model_data_slow = []
    

baz_error_spatial(df["backazimuth"], df["baz_error"], model_data_baz,
            color_data, color_label, niazi=False, save=False, plot_fourier=True,
            plot_bins = False,
            path=fig_path + array_name+"_baz_error_spatial.png")   


slow_error_spatial(df["backazimuth"], df["slow_error"], model_data_slow,
            color_data, color_label, niazi=False, save=False, plot_fourier = True,
            plot_bins = False,
            path=fig_path + array_name+"_slow_error_spatial.png") 

model_data = []

baz_error_spatial(df["backazimuth"], df["baz_error"]- model_data_baz, model_data,
            color_data, color_label, niazi=False, save=False, plot_fourier=False,
            path=fig_path + array_name+"_baz_error_spatial.png") 

print('Mean absolute error after snells:', np.mean(np.abs(df["baz_error"]- model_data_baz)))
baz = df['backazimuth'].to_numpy()
baz_error = df['baz_error'].to_numpy()
slow_error = df['slow_error'].to_numpy()
earthquake_lats = df['latitude'].to_numpy()
earthquake_lons = df['longitude'].to_numpy()
earthquake_mags = df['magnitude'].to_numpy()
earthquake_depths = df['depth'].to_numpy()



pygmt_baz_error(array_lats[0], array_lons[0], array_name, 
                        earthquake_lats, earthquake_lons, earthquake_mags, baz,
                        baz_error, save = False, 
                        path = fig_path+'kd_baz_error_map.png')


#####################################################
#Double beam plot-----------------------------------
####################################################

#%%

#Pull in Homer data and drop taup and low power
hm = pd.read_csv('./HM_1000km_m2_fk_4_window_freq_test.csv')
hm = pd.DataFrame(hm[hm['trigger_type']!= 'Taup'])
hm = hm[hm["relpow"] >= pow_thresh]
kd = pd.read_csv('./KD_1000km_m2_fk_4_window_freq_test.csv')
kd = pd.DataFrame(kd[kd['trigger_type']!= 'Taup'])
kd = kd[kd["relpow"] >= pow_thresh]

#Try to remove trend from data########################

from scipy.optimize import curve_fit


def remove_fourier(angles, values, array_baz, use_bins = False):
    if use_bins == True:
        bins = np.arange(0, 361, 10)
        medians = []
        counts = []
        bin_centers = []
        min_count = 1

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
    elif use_bins == False:
        medians = values
        bin_centers = angles
        counts = np.ones(len(values))
        
        
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
    #theta = np.deg2rad(df['backazimuth'])
    #theta = angles
    y_fit = fourier5(theta, *params)

    #Keep error between -180 to 180
    baz_error_temp = values - y_fit
    baz_corrected = ((baz_error_temp + 180) % 360) - 180

    baz_error_temp = array_baz + y_fit
    array_baz_correct = ((baz_error_temp + 180) % 360) - 180
    return baz_corrected, array_baz_correct

angles = hm['backazimuth']
values = hm['baz_error']
array_baz = hm['array_baz']
hm['baz_corrected'], hm['array_baz_correct'] = remove_fourier(angles, values, array_baz, use_bins = False)

angles = kd['backazimuth']
values = kd['baz_error']
array_baz = kd['array_baz']
kd['baz_corrected'], kd['array_baz_correct'] = remove_fourier(angles, values, array_baz, use_bins = False)

print('Mean absolute error KD:', np.mean(np.abs(kd['baz_corrected'])))
print('Mean absolute error HM:', np.mean(np.abs(hm['baz_corrected'])))


#hm = pd.read_csv('./HM_1000km_m2_fk_4_window_freq_test.csv')
#kd = pd.read_csv('./KD_1000km_m2_fk_4_window_freq_test.csv')

comb = pd.merge(hm, kd, on='event_id', how='inner')

#comb = pd.DataFrame(comb[comb['magnitude_x']>= min_mag])

drop = True

if drop ==True:
    
    temp = pd.DataFrame(comb[comb['trigger_type_x']!= 'Taup'])
    temp = pd.DataFrame(temp[temp['trigger_type_y']!= 'Taup'])
    print('Number of dropped events for Taup:', len(comb) - len(temp))
    comb = temp

#%%
#combined = pd.merge()
#comb = pd.merge(hm, kd, on='event_id', how='inner')
print('Number of events with both arrays:', len(comb))
array_lats = [59.618433, 57.441811]  
array_lons = [-151.141416, -152.352174]
index = 103 #61, 44, 3
#array_lats = [59.618433] #HM
#array_lons = [-151.141416] #HM
real_bazs_array1 = comb['backazimuth_x'].to_numpy()
array1_bazs = comb['array_baz_correct_x'].to_numpy()
earthquake_mags = comb['magnitude_x'].to_numpy()
earthquake_lats = comb['latitude_x'].to_numpy()
earthquake_lons = comb['longitude_x'].to_numpy()
earthquake_depths = comb['depth_x'].to_numpy()
baz_conf = 6
real_bazs_array2 = comb['backazimuth_y'].to_numpy()
array2_bazs = comb['array_baz_correct_y'].to_numpy()

#pygmt_single_event(index, array_lat, array_lon, earthquake_lats, earthquake_lons, earthquake_mags, real_bazs, array_bazs, baz_conf, plot_real = False)

pygmt_single_event(index, array_lats, array_lons, earthquake_lats, 
                   earthquake_lons, earthquake_mags, earthquake_depths,
                     real_bazs_array1, array1_bazs, real_bazs_array2, 
                     array2_bazs, baz_conf, plot_real = False, save = False,
                    path = '/Users/cadequigley/Downloads/Research/gigls_2026/intersecting_beams_example.png')
                        
 
# %%

from array_maps_pygmt import intersect_beams
from obspy.geodetics import gps2dist_azimuth

array_lats = [59.618433, 57.441811]  
array_lons = [-151.141416, -152.352174]

real_bazs_array1 = comb['backazimuth_x'].to_numpy()
array1_bazs = comb['array_baz_correct_x'].to_numpy()
array2_bazs = comb['array_baz_correct_y'].to_numpy()

earthquake_lats = comb['latitude_x'].to_numpy()
earthquake_lons = comb['longitude_x'].to_numpy()

distance_error = []
for i in range(len(comb)):
    point1, point2 = intersect_beams(array_lats[0], array_lons[0], array1_bazs[i], array_lats[1], array_lons[1], array2_bazs[i])
    dist1, az, baz = gps2dist_azimuth(point1[0], point1[1], earthquake_lats[i], earthquake_lons[i])
    dist2, az, baz = gps2dist_azimuth(point2[0], point2[1], earthquake_lats[i], earthquake_lons[i])
    min_dist = np.min([dist1,dist2])
    distance_error.append(min_dist/1000)      
    print('Distance error from intersecting beams:', min_dist/1000, 'km')
print('Median distance error from intersecting beams:', np.median(distance_error))
# %%
