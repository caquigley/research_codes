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
                             pygmt_slow_error)

from array_functions import baz_to_az, cos_model, fourier5
from array_functions import calculate_deflection
from array_functions import slab_inversion, niazi_dip_inversion

####################################
###INPUTS---------------------------
####################################
processing = 'lts'

window_length = 6 #
array_name = '2A'

#Figures----------
earthquake_map = False
baz_error_plot = True
slow_error_plot = True
baz_error_map = False
slow_error_map = False
plot_histogram = False
save_fig = False
fig_path = '/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/figure_components/'
#array_lats = [53.6974, 53.779, 53.8566]  
#array_lons = [-166.7343, -166.2131,-166.4161]
origin_lat = 53.6974 #53.8566
origin_lon = -166.7343 #-166.4161

#Downsampling--------------
drop_taup = True #drop Taup picks, i.e. events without an STA/LTA pick
drop_pow = True
pow_thresh = 0.35 #0.35 fk, 0.5 lts
drop_conf = False
conf_thresh = 20

#Slab modeling-----------------
model = 'inversion' #'inversion', 'fixed'
strike = 269 #272 #277
dip = 20 #17 #29
oceanic_vel = 8.04 #8.3
continental_vel = 5.8 #4 #4, 6.2
takeoff_type = 'moho' #'moho', 'surface', 'source'
weight_baz = 1
weight_slow = 0



df = pd.read_csv('./'+array_name+'_2000km_m3_'+processing+'_'+str(window_length)+'_window_freq_test.csv') #400 pom , 450 2A
#df = pd.DataFrame(df[df['backazimuth']>= 210])
#df = pd.DataFrame(df[df['backazimuth']<= 80])
#df = pd.DataFrame(df[df['distance']<= 1800 ]) #1800
df = pd.DataFrame(df[df['distance']<= 1000 ]) #1800
#df1 = pd.read_csv('./'+array_name+'_5000km_m5_'+processing+'_'+str(window_length)+'_window_freq_test.csv')
#df = pd.concat([df, df1]).drop_duplicates( subset='event_id', keep='first')
#%%
print('Number of events:', len(df))
#%%
##Drop values that did not have a trigger-----------------
if drop_taup ==True:
    temp = pd.DataFrame(df[df['trigger_type']!= 'Taup'])
    print('Number of dropped events for Taup:', len(df) - len(temp))
    df = temp

##Drop values with low cross correlation-----------------
if drop_pow == True:
    if processing == 'fk':
        temp = df[df["relpow"] >= pow_thresh]
    elif processing == 'lts' or 'ls':
        temp = df[df["mdccm"] >= pow_thresh]

    print('Number of dropped events for low power:', len(df) - len(temp))
    df = temp
## Drop values with low confidence interval---------------
if drop_conf == True:
    if processing == 'fk':
        print("FK analysis doesn't have confidence intervals, can't drop")
    elif processing == "lts":
        temp = df[df['conf_int_baz'] <= conf_thresh]
        print('Number of dropped events for high confidence interval:', len(df) - len(temp))
        df = temp
####################################
### Niazi-----------
####################################
from scipy.optimize import curve_fit
slowness = df['array_slow'].to_numpy()
slow_error = df['slow_error'].to_numpy()
baz_error = df['baz_error'].to_numpy()
baz = df['backazimuth'].to_numpy()

p0 = [1.0, 10.0, 180.0]   # a, b, phi guesses

Z_data = baz
y_data = baz_error
params, cov = curve_fit(cos_model, Z_data, y_data, p0=p0)
a_fit, b_fit, phi_fit = params
#Plot niazi fit
Z_fit = np.linspace(0, 360, 500)
y_fit = cos_model(Z_fit, *params)

crust_vel = 3.05 # km/s
print('Niazi strike fit:', 180 + phi_fit)

niazi_dip = niazi_dip_inversion(baz, slowness, slow_error, 180 + phi_fit, crust_vel)



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
                                        receiver_depth_in_km=10)

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

model_data_baz = model['model_baz_error'].to_numpy()
model_data_slow = model['model_slowness_error'].to_numpy()
baz_error_spatial(df['backazimuth'], baz_error, model_data_baz, df['distance'], 'distance (km)', niazi = True, 
                  save = False, path = '/Users/cadequigley/Downloads/Research/fig3_baz_error_model.png')
slow_error_spatial(df['backazimuth'], slow_error, model_data_slow, df['distance'], 'distance (km)', niazi = False, 
                  save = False, path = '/Users/cadequigley/Downloads/Research/fig3_slow_error_model.png')

#Plot with removed trend-----------
model_data = []
baz_error_spatial(df['backazimuth'], baz_error - model_data_baz, model_data, df['distance'], 'distance (km)', niazi = True, 
                  save = False, path = '/Users/cadequigley/Downloads/Research/fig3_baz_error_model.png')

####################################
###PLOTS---------------------------
####################################

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
 #drop Taup picks, i.e. events without an STA/LTA pick
if drop_taup ==True:
    temp = pd.DataFrame(df[df['trigger_type']!= 'Taup'])
    print('Number of dropped events for Taup:', len(df) - len(temp))
    df = temp

if processing == 'fk':
    color_data = df['relpow']
else: 
    color_data = df['mdccm']
    
#color_data = df['conf_int_baz']
#color_data = df['incident_angle']
#color_data = df['slowness']
#color_data = df['array_slow']
#color_data = df['distance']
#color_data = df['magnitude']
color_label = 'cross correlation/power'
model_data = []
    
if baz_error_plot == True:
    baz_error_spatial(df["backazimuth"], df["baz_error"], model_data,
        color_data, color_label, niazi=True, save=save_fig,
        path=fig_path + "baz_error_spatial.pdf")   

if slow_error_plot == True:
    slow_error_spatial(df["backazimuth"], df["slow_error"], model_data,
        color_data, color_label, niazi=False, save=save_fig,
        path=fig_path + "slow_error_spatial.pdf")  
        
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
                    path = fig_path+'baz_error_map_1000km.png')
                        
    
#Plot slowness error on map-----------------------------
#----------------------------------------------------
if slow_error_map == True:
    pygmt_slow_error(array_lats[0], array_lons[0], array_name, 
                    earthquake_lats, earthquake_lons, earthquake_mags, 
                    slow_error, save = save_fig, 
                    path = fig_path+'slow_error_map_1000km.png')
        
if plot_histogram == True:
    upper_quantile = 0.95
    lower_quantile = 0.05
    variable_name = 'backazimuth_error'
    histogram(baz_error, lower_quantile, upper_quantile, variable_name, 
              save = save_fig, path = fig_path+'backaimuth_histogram.png')
        
    variable_name = 'slowness_error'
    histogram(slow_error, lower_quantile, upper_quantile, variable_name, 
            save = save_fig, path = fig_path+'slowness_histogram.png')
# %%
#Plot stacked histogram----------------------------------------
#--------------------------------------------------------------
theta = np.deg2rad(baz)

params, _ = curve_fit(fourier5, theta, baz_error)
print(params)
    # Smooth curve
#theta_fit = np.linspace(0, 2*np.pi, 500)


y_fit = fourier5(theta, *params)

raw_values = baz_error
correct_values2 = baz_error - y_fit
upper_quantile = 0.95
lower_quantile = 0.05
variable_name = 'backazimuth_error'

correct_values = baz_error - model_data_baz


stacked_histogram(raw_values, correct_values, lower_quantile, upper_quantile, variable_name, 
              correct_values2 = correct_values2,
              save = False, path = '/Users/cadequigley/Downloads/fig3_baz_hist.png')

print('Mean absolute error:', np.mean(np.abs(correct_values)))

#------------------------------------------------

p0 = [1.0, 10.0, 180.0]   # a, b, phi guesses

Z_data = baz
y_data = slow_error
params, cov = curve_fit(cos_model, Z_data, y_data, p0=p0)
a_fit, b_fit, phi_fit = params
#Plot niazi fit

y_fit = cos_model(Z_data, *params)


raw_values = slow_error
correct_values2 = slow_error - y_fit
upper_quantile = 0.95
lower_quantile = 0.05
variable_name = 'slowness_error'

correct_values = slow_error - model_data_slow


stacked_histogram(raw_values, correct_values, lower_quantile, upper_quantile, variable_name, 
              correct_values2 = correct_values2,
              save = False, path = '/Users/cadequigley/Downloads/fig3_slow_hist.png')