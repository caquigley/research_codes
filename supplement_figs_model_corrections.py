#%%
import numpy as np
import pandas as pd
from obspy.taup import TauPyModel
from obspy.geodetics import kilometers2degrees
from scipy.optimize import curve_fit

from array_figures import (baz_error_spatial, 
                           slow_error_spatial, 
                           time_series_density,
                           histogram,
                           stacked_histogram)

from array_maps_pygmt import (pygmt_array_earthquakes, pygmt_baz_error_new, 
                             pygmt_slow_error_new)

from array_functions import baz_to_az, cos_model, fourier5
from array_functions import calculate_deflection, anisotropic_harmonic
from array_functions import slab_inversion, niazi_dip_inversion

####################################
###INPUTS---------------------------
####################################
processing = 'lts'

window_length = 6 #
array_name = '2A'

#Figures----------
save_fig = False
fig_path = '/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/figure_components/'

#Correction types----------
plot_3D_snell = False
plot_anisotropic = False
plot_anisotropic_reduced = False
plot_niazi = False
plot_bins = True
plot_fourier = True
variable_name = 'slowness_error'
line_color = 'skyblue' #red, blue, green, skyblue

correction = 'fourier' #3D_snell #label for saving

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
strike = 246 #269 #272 #277
dip = 45 #17 #29
oceanic_vel = 8.04 #8.3
continental_vel = 6 #5, 4 #4, 6.2, 5.8
takeoff_type = 'moho' #'moho', 'surface', 'source'
moho_depth = 15 #km
velocity_model = 'pavdut'
#velocity_model = 'iasp91'
#velocity_model = 'ak135'

#Inputs for interface inversion---------
initial_guess = [
            249.0,   # strike, based on trench strike
            10.0,   # dip, based on subduction zone dip
            8.04,    # oceanic velocity
            6.2     # continental velocity
            ]

        #Value bounds---------------------
bounds = ( #strike, dip, oceanic_vel, continental_vel
            [0,   0,   8.03, 4],#4], #lower bounds, 5.8, 4
            [360, 90,  8.04, 8]#8] #upper bounds, 90
                )

sigma_baz = 4.5 #np.std(baz_error)   # or a physically motivated value, e.g. array resolution
sigma_p   = 0.1 #0.009 #np.std(slow_error)

#weight_baz = 1 / sigma_baz**2
#weight_slow   = 1 / sigma_p**2
#weight_baz = 1/sigma_baz
#weight_slow = 1/sigma_p 
#print('Weight baz:', weight_baz)
#print('Weight slow:', weight_slow)
weight_baz = 1
weight_slow = 3



df1 = pd.read_csv('./'+array_name+'_2000km_m3_'+processing+'_'+str(window_length)+'_window_freq_test.csv') #400 pom , 450 2A
df = pd.DataFrame(df1[df1['distance']<= 1000 ]) #1800
df = pd.read_csv('./2A_1000km_m3_lts__window_freq_map_fig.csv')
#%%
#df = pd.DataFrame(df[df['backazimuth']>= 210])
#df = pd.DataFrame(df[df['backazimuth']<= 80])
#df = pd.DataFrame(df[df['distance']<= 1800 ]) #1800
df = pd.DataFrame(df[df['distance']<= 1000 ]) #1800
#df1 = pd.read_csv('./'+array_name+'_5000km_m5_'+processing+'_'+str(window_length)+'_window_freq_test.csv')
#df = pd.concat([df, df1]).drop_duplicates( subset='event_id', keep='first')
#%%
print('Number of events:', len(df))
#df['baz_error'] = -1*df['baz_error']
#df['slow_error'] = -1*df['slow_error']
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
print('Remaining events:', len(df))

slowness = df['array_slow'].to_numpy()
slow_error = df['slow_error'].to_numpy()
baz_error = df['baz_error'].to_numpy()
baz = df['backazimuth'].to_numpy()



####################################
###Forward model processing-----------
####################################
if plot_3D_snell == True:
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

        tmod = TauPyModel(model = velocity_model)
        incident_angle_list = []
        for i in range(len(distance)):
            dist_deg = kilometers2degrees(distance[i])
            arrivals = tmod.get_pierce_points(depth[i], dist_deg, 
                                            phase_list=["P", "p"],
                                            receiver_depth_in_km=moho_depth) #10

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

        #residuals = combined_residuals(initial_guess, baz, takeoff, baz_error, slow_error, weight_baz, weight_slow)
        strike_fit, dip_fit, v_oceanic_fit, v_continental_fit = slab_inversion(initial_guess, bounds, baz, takeoff, baz_error, slow_error, weight_baz, weight_slow)


        model = calculate_deflection(strike_fit, dip_fit, v_oceanic_fit, v_continental_fit, distance, depth, takeoff, az, baz, event_id)

    elif model == 'fixed':
        #Fixed values
        model = calculate_deflection(strike, dip, oceanic_vel, continental_vel, distance, depth, takeoff, az, baz, event_id)

    model_data_baz = model['model_baz_error'].to_numpy()
    model_data_slow = model['model_slowness_error'].to_numpy()

elif plot_3D_snell == False:
    model_data_baz = []
    model_data_slow = []

baz_error_spatial(df['backazimuth'], baz_error, model_data_baz,
                  df['mdccm'], 'MDCCM', 
                      niazi = plot_niazi, 
                      plot_fourier = plot_fourier, 
                      plot_anisotropic = plot_anisotropic, 
                      plot_anisotropic_reduced = plot_anisotropic_reduced,
                      plot_bins = plot_bins, save = save_fig, 
                      path = fig_path+'supp_baz_error_'+correction+'.pdf')

slow_error_spatial(df['backazimuth'], slow_error, model_data_slow,
                  df['mdccm'], 'MDCCM', 
                      niazi = plot_niazi, 
                      plot_fourier = plot_fourier, 
                      plot_anisotropic = plot_anisotropic, 
                      plot_anisotropic_reduced = plot_anisotropic_reduced,
                      plot_bins = plot_bins, save = save_fig, 
                      path = fig_path+'supp_slow_error_'+correction+'.pdf')
#slow_error_spatial(df['backazimuth'], slow_error, model_data_slow, df['distance'], 'distance (km)', niazi = False, 
                  #save = False, path = '/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/figure_components/fig3_slow_error_model.pdf')
#%%

#---------------------------------------
#### Set up corrections------------
#---------------------------------------
#variable_name = 'slowness_error'
#Binned statistics
if variable_name == 'backazimuth_error':
    values = baz_error
elif variable_name == 'slowness_error':
    values = slow_error

angles = baz


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

#Drop nan values
mask = ~np.isnan(medians)
medians = medians[mask]
counts = counts[mask]
bin_centers = bin_centers[mask]

#Anisotropic correction---------------------
if plot_anisotropic == True:
    if plot_bins == True:
        params, _ = curve_fit(anisotropic_harmonic, bin_centers, medians)
    if plot_bins == False:
        params, _ = curve_fit(anisotropic_harmonic, baz, values)

    A1, A2, A3, A4, A5 = params


    fast_dir = 0.5*np.arctan(A4/A5) + np.pi/4
    print('Anisotropic fast direction:', np.rad2deg(fast_dir))
    ani_amp = np.sqrt((A4**2) +(A5**2))
    print('Anisotropic amplitude:', ani_amp)

    dip_max = np.sqrt((A2**2)+(A3**2))
    print('Dip amplitude:', dip_max)

    baz_fit = np.linspace(0, 360, 500)

    anisotropic_fit = anisotropic_harmonic(baz, *params)

    #correct_values3 = baz_error - anisotropic_fit
    model_data_baz = anisotropic_fit

#Niazi correction-----------------------
if plot_niazi == True:
    p0 = [1.0, 10.0, 180.0]   # a, b, phi guesses
    if plot_bins == True:
        Z_data = bin_centers
        y_data = medians
    elif plot_bins == False:
        Z_data = baz
        y_data = values
    params, cov = curve_fit(cos_model, Z_data, y_data, p0=p0)
    a_fit, b_fit, phi_fit = params
    #Plot niazi fit
    Z_fit = np.linspace(0, 360, 500)
    #y_fit = cos_model(Z_fit, *params)
    y_fit = cos_model(baz, *params)

    model_data_baz = y_fit
    print('Niazi strike fit:', phi_fit) # +180

#Fourier correction-----------------
if plot_fourier == True:
        if plot_bins == True:
            theta = np.deg2rad(bin_centers)
            params, _ = curve_fit(fourier5, theta, medians)
        elif plot_bins == False:
            theta = np.deg2rad(baz)
            params, _ = curve_fit(fourier5, theta, values)

        # Smooth curve
        theta_fit = np.linspace(0, 2*np.pi, 500)

        #y_fit = fourier5(theta_fit, *params)
        y_fit = fourier5(np.deg2rad(baz), *params)
        
        model_data_baz = y_fit


#Plot with removed trend-----------
model_data = []
color_data = []
if correction == '3D_snell':
    if variable_name == 'slowness_error':
        model_data_baz = model_data_slow
if variable_name == 'backazimuth_error':
    baz_error_spatial(df['backazimuth'], values - model_data_baz, model_data, color_data, 'MDCCM',  
                    save = save_fig, path = fig_path+'supp_baz_error_'+correction+'_correction.pdf')
elif variable_name == 'slowness_error':
    slow_error_spatial(df['backazimuth'], values - model_data_baz, model_data, color_data, 'MDCCM',  
                    save = save_fig, path = fig_path+'supp_slow_error_'+correction+'_correction.pdf')

#%%



lower_quantile = 0.05
upper_quantile = 0.95
raw_values = values
correct_values = values - model_data_baz
if variable_name == 'backazimuth_error':
    val = 'baz'
elif variable_name == 'slowness_error':
    val = 'slow'
stacked_histogram(raw_values, correct_values, lower_quantile, upper_quantile, 
                  variable_name, line_color = line_color,
              save = save_fig, path = fig_path+'supp_'+val+'_error_'+correction+'_histogram.pdf')
# %%


from array_functions import (snell_3d_test, spherical_to_xyz, plane_normal)

# test both versions at ratio=1 with your actual normal convention
test_incident = spherical_to_xyz(45, 20)   # pick any realistic azimuth/takeoff
test_normal = plane_normal(dip=30, strike=120)  # pick any realistic dip/strike

print(snell_3d_test(test_incident, test_normal, 6.0, 6.0, '+'))
print(snell_3d_test(test_incident, test_normal, 6.0, 6.0, '-'))
print(test_incident / np.linalg.norm(test_incident))  # what both should match
# %%