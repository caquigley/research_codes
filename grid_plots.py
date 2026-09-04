#%%
import pandas as pd
import numpy as np
from obspy import UTCDateTime
import matplotlib.pyplot as plt
from obspy import read
#from array_figures import baz_error_spatial
from matplotlib.transforms import blended_transform_factory
from array_functions import cos_model
from scipy.optimize import curve_fit
from array_functions import (get_geometry, fourier5, anisotropy_model,
                              anisotropic_harmonic, )
from array_maps_pygmt import pygmt_single_event_new

df = pd.read_csv('/Users/cadequigley/repos/array_aggregator/2A_600km_m3_lts__window_freq_test2.csv') #_const_lfreq.csv')
#df = pd.read_csv('/Users/cadequigley/repos/array_aggregator/2A_600km_m3_fk__window_freq_test_smallgrid.csv') #_const_lfreq.csv')
save_fig = False
fig_path = '/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/figure_components/'

df = pd.DataFrame(df[df['distance']<= 400 ])
#df = pd.DataFrame(df[df['window_length']== 5 ])
#df = pd.DataFrame(df[df['max_freq']!= 6 ])


drop_taup = True
drop_pow = False
pow_thresh = 0.35
processing = 'lts' #fk, lts

##Drop values with no sta/lta trigger-----------------
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
#%%

def baz_error_spatial(baz, baz_error, baz_error_model, color_data, 
                      color_data_label, niazi = True, save = False, 
                      path = None):

    fig, ax = plt.subplots(figsize = (7,4))
    
    trans = blended_transform_factory(ax.transData, ax.transAxes)

    if len(baz_error_model) > 0:
        ax.scatter(baz, baz_error, color = 'gray', edgecolors = 'black',
                   s = 100, label = 'measured')
        ax.scatter(baz, baz_error_model, color = 'skyblue', edgecolors = 'black', 
                   s = 100, label = 'modeled', alpha = 0.6)
    else:

        if len(color_data) > 0:
            sc = ax.scatter(baz, baz_error, c = color_data, cmap = 'plasma_r',
                            edgecolors = 'black', s = 100, alpha= 0.3)#vmin = 20, vmax = 40)
            fig.colorbar(sc, label = color_data_label)
        else:
            ax.scatter(baz, baz_error, color = 'gray', alpha = 1, 
                       edgecolors = 'black', s = 100, label = 'observed')

    

    if niazi == True:
        
        p0 = [1.0, 10.0, 180.0]   # a, b, phi guesses

        Z_data = baz
        y_data = baz_error
        params, cov = curve_fit(cos_model, Z_data, y_data, p0=p0)
        a_fit, b_fit, phi_fit = params
        print('Strike from Niazi plot:', 180 + phi_fit)
        #Plot niazi fit
        Z_fit = np.linspace(0, 360, 500)
        y_fit = cos_model(Z_fit, *params)
        #ax.plot(Z_fit, y_fit, color = 'red', linewidth = 2.5, 
                #label = 'Niazi fit', alpha= 0.5)
    
    #5th order polynomial---------------------
    coefficients = np.polyfit(baz, baz_error, 5)
    #print("Coefficients:", coefficients)
    polynomial = np.poly1d(coefficients)

    x_fit = np.linspace(0, 360, 1000)
    y_fit = polynomial(x_fit)
    #ax.plot(x_fit, y_fit, color = 'blue', label = 'Polynomial fit')

    #Fourier fit------------------------------
    theta = np.deg2rad(baz)

    params, _ = curve_fit(fourier5, theta, baz_error)

    # Smooth curve
    theta_fit = np.linspace(0, 2*np.pi, 500)

    y_fit = fourier5(theta_fit, *params)
    #ax.plot(np.rad2deg(theta_fit), y_fit, color = 'red', linewidth = 2.5, 
              # alpha= 0.8, label = 'Fourier fit')
    #------------------------------------------------------
    #Anisotropy model
    
    params, _ = curve_fit(anisotropy_model, theta, baz_error)
    

    # Smooth curve
    theta_fit = np.linspace(0, 2*np.pi, 500)

    y_fit = anisotropy_model(theta_fit, *params)
    #ax.plot(np.rad2deg(theta_fit), y_fit, color = 'red', label = 'Anisotropy')
    #------------------------------------------------------
    #Binned statistics
    angles = baz
    values = baz_error

    bins = np.arange(0, 361, 10)

    medians = []
    counts = []
    bin_centers = []

    min_count = 3 #30

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
    print('Medians', medians)
    print('Counts', counts)
    print('Bin centers', bin_centers)
    ax.scatter(bin_centers, medians, color = 'gray', s = 150, edgecolors='black', linewidths=1, label = 'median bins')
    
    #drop nan values
    mask = ~np.isnan(medians)
    medians = medians[mask]
    counts = counts[mask]
    bin_centers = bin_centers[mask]
    #------------------------------------------------------
    #Anisotropic harmonic
    #params, _ = curve_fit(anisotropic_harmonic, baz, baz_error)
    params, _ = curve_fit(anisotropic_harmonic, bin_centers, medians)
    A1, A2, A3, A4, A5 = params

    print('Anisotropic parameters:')
    print('A1:', A1)
    print('A2:', A2)
    print('A3:', A3)
    print('A4:', A4)
    print('A5:', A5)

    fast_dir = 0.5*np.arctan(A4/A5) + np.pi/4
    print('Anisotropic fast direction:', np.rad2deg(fast_dir))
    ani_amp = np.sqrt((A4**2) +(A5**2))
    print('Anisotropic amplitude:', ani_amp)

    dip_max = np.sqrt((A2**2)+(A3**2))
    print('Dip amplitude:', dip_max)

    baz_fit = np.linspace(0, 360, 500)

    y_fit = anisotropic_harmonic(baz_fit, *params)
    #ax.plot(baz_fit, y_fit, color = 'red', label = 'Anisotropic harmonic')
    
    #Fit fourier5 to median bins-----------------------
    params, _ = curve_fit(fourier5, np.deg2rad(bin_centers), medians)
    baz_fit = np.linspace(0, 360, 500)

    y_fit = fourier5(np.deg2rad(baz_fit), *params)
    ax.plot(baz_fit, y_fit, color = 'red', label = 'Fourier fit', linewidth = 2.5)

    #------------------------------------------------------
    ax.text(45,0.9, "NE", transform = trans, color = 'black', 
            fontweight = 'bold',fontsize = 15, ha='center')
    ax.text(135,0.9, "SE", transform = trans, color = 'black', 
            fontweight = 'bold',fontsize = 15, ha='center')
    ax.text(225,0.9, "SW", transform = trans, color = 'black', 
            fontweight = 'bold',fontsize = 15, ha='center')
    ax.text(315,0.9, "NW", transform = trans, color = 'black', 
            fontweight = 'bold',fontsize = 15, ha='center')

    ax.axvline(x=90, color = 'black', linestyle = '--')
    ax.axvline(x=180, color = 'black', linestyle = '--')
    ax.axvline(x=270, color = 'black', linestyle = '--')
    ax.axhline(y=0, color = 'red', linestyle = '--', alpha = 0.3)

    ax.grid(alpha = 0.3, zorder = 0)
    ax.set_xlabel('catalog backazimuth (degrees)')
    ax.set_ylabel('backazimuth error (degrees)')
    ax.set_xlim(0,360)
    #ax.set_ylim(-np.max(abs(baz_error)),np.max(abs(baz_error)))
    ax.set_ylim(-90,90) #-80, 80

    ax.invert_xaxis()
    plt.legend(loc = 'upper left', bbox_to_anchor=(0, 0.25))
    
    if save == True:
            fig.savefig(path, transparent=True, dpi=720)
    #
    plt.show()



# %%
'''
array = '2A'
event_ids = df['event_id'].to_numpy()
freq_min = 0.5
freq_max = 10
for event in range(len(df)):
    st = read('/Users/cadequigley/Repos/array_aggregator/'+array+'_earthquakes_mseeds/'+event_ids[event]+".mseed")
#st.taper(max_percentage=0.05)
    st.filter("bandpass", freqmin=freq_min, freqmax=freq_max, corners=2, zerophase=True)

    trigger = UTCDateTime(df['trigger_time'].to_numpy()[event])
    fig, ax = plt.subplots()
    for tr in st:
        ax.plot(tr.times(), tr.data, color = 'black', alpha = 0.3)
    ax.axvline(trigger - tr.stats.starttime, color = 'red', linestyle = '--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Counts')
    plt.show()
'''

color_data = df['mdccm']
color_label = 'median cross-correlation maximum'
baz_error = df['baz_error'].to_numpy()
model_data = []
baz_error_spatial(df['backazimuth'], baz_error, model_data, color_data, color_label, niazi = True, 
                  save = False, path = '/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/figure_components/supp_fig_stacked_models_baz_error.png')


#Try to remove trend from data########################

angles = df['backazimuth']
values = baz_error

bins = np.arange(0, 361, 10)

medians = []
counts = []
bin_centers = []
min_count = 10

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
theta = np.deg2rad(df['backazimuth'])
y_fit = fourier5(theta, *params)

#Keep error between -180 to 180
baz_error_temp = baz_error - y_fit
baz_corrected = ((baz_error_temp + 180) % 360) - 180
df['baz_corrected'] = baz_corrected

baz_error_temp = df['array_baz'] + y_fit
array_baz_correct = ((baz_error_temp + 180) % 360) - 180
df['array_baz_correct'] = array_baz_correct

baz_error_spatial(df['backazimuth'], baz_corrected, model_data, color_data, color_label, niazi = True, 
                  save = False, path = '/Users/cadequigley/Downloads/Research/fig3_baz_error_model.png')
#%%

######################
#_____SLOWNESS_____________
######################


def slow_error_spatial(baz, slow_error, slow_error_model, color_data,
                       color_data_label, niazi = True, 
                       save = False, path = None):

    fig, ax = plt.subplots(figsize = (7,4))
    
    trans = blended_transform_factory(ax.transData, ax.transAxes)


    if len(slow_error_model) > 0:
        ax.scatter(baz, slow_error, color = 'gray', edgecolors = 'black',
                   s = 100, label = 'Observed')
        ax.scatter(baz, slow_error_model, color = 'skyblue', 
                   edgecolors = 'black', s = 80, alpha = 0.6, marker = 'D',  label = '3D Snell')
        #plt.legend(loc = 'upper left', bbox_to_anchor=(0, 0.25))
    else:
        if len(color_data) > 0:
            sc = ax.scatter(baz, slow_error, c = color_data, 
                            cmap = 'cividis_r', edgecolors = 'black', s = 100,
                              vmin = 0, vmax = 1, alpha = 0.6) #'cividis_r'
            fig.colorbar(sc, label = color_data_label)
        else:
            ax.scatter(baz, slow_error, color = 'gray', alpha = 1, 
                       edgecolors = 'black', s = 100, 
                       label = array+' observed')

    
    if niazi == True:
        
        p0 = [1.0, 10.0, 180.0]   # a, b, phi guesses

        Z_data = baz
        y_data = slow_error
        params, cov = curve_fit(cos_model, Z_data, y_data, p0=p0)
        a_fit, b_fit, phi_fit = params
        #Plot niazi fit
        Z_fit = np.linspace(0, 360, 500)
        y_fit = cos_model(Z_fit, *params)
        ax.plot(Z_fit, y_fit, color = 'red', linewidth = 2.5, 
                label = 'Niazi fit', alpha= 0.5)
        
    #Binned statistics
    angles = baz
    values = slow_error

    bins = np.arange(0, 361, 10)

    medians = []
    counts = []
    bin_centers = []

    min_count = 10

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
    
    ax.scatter(bin_centers, medians, color = 'red', s = 150, edgecolors='black', linewidths=1, label = 'median bins')
    
    #drop nan values
    mask = ~np.isnan(medians)
    medians = medians[mask]
    counts = counts[mask]
    bin_centers = bin_centers[mask]
        
    #Fourier fit------------------------------
    #theta = np.deg2rad(baz)
    theta = np.deg2rad(bin_centers)

    # Fourier series function
    def fourier5(theta,
             a0,
             a1,b1):
             #a2,b2,
             #a3,b3):
             #a4,b4,
             #a5,b5):
             #a6,b6,
             #a7,b7,
             #a8,b8):

        return (
            a0
            + a1*np.cos(1*theta) + b1*np.sin(1*theta)
            #+ a2*np.cos(2*theta) + b2*np.sin(2*theta)
            #+ a3*np.cos(3*theta) + b3*np.sin(3*theta)
            #+ a4*np.cos(4*theta) + b4*np.sin(4*theta)
            #+ a5*np.cos(5*theta) + b5*np.sin(5*theta)
            #+ a6*np.cos(6*theta) + b6*np.sin(6*theta)
            #+ a7*np.cos(7*theta) + b7*np.sin(7*theta)
            #+ a8*np.cos(8*theta) + b8*np.sin(8*theta)
        )

    # Fit
    #params, _ = curve_fit(fourier5, theta, slow_error)
    params, _ = curve_fit(fourier5, theta, medians)

    # Smooth curve
    theta_fit = np.linspace(0, 2*np.pi, 500)

    y_fit = fourier5(theta_fit, *params)
    ax.plot(np.rad2deg(theta_fit), y_fit, color = 'red', linewidth = 2.5, 
               alpha= 0.8, label = 'Fourier fit')

    ax.text(45,0.9, "NE", transform = trans, color = 'black', 
            fontweight = 'bold',fontsize = 15, ha='center')
    ax.text(135,0.9, "SE", transform = trans, color = 'black', 
            fontweight = 'bold',fontsize = 15, ha='center')
    ax.text(225,0.9, "SW", transform = trans, color = 'black', 
            fontweight = 'bold',fontsize = 15, ha='center')
    ax.text(315,0.9, "NW", transform = trans, color = 'black', 
            fontweight = 'bold',fontsize = 15, ha='center')

    ax.axvline(x=90, color = 'black', linestyle = '--')
    ax.axvline(x=180, color = 'black', linestyle = '--')
    ax.axvline(x=270, color = 'black', linestyle = '--')
    ax.axhline(y=0, color = 'red', linestyle = '--', alpha = 0.3)

    ax.grid(alpha = 0.3, zorder = 0)
    ax.set_xlabel('catalog backazimuth (degrees)')
    ax.set_ylabel('slowness error (s/km)')
    ax.set_xlim(0,360)
    #ax.set_ylim(-np.max(abs(slow_error))-0.05,np.max(abs(slow_error))+0.05)
    ax.set_ylim(-0.15, 0.15) #(-0.2, 0.2)
    #ax.set_ylim(-0.5, 0.5)

    ax.invert_xaxis()
    
    models = [niazi]
    if any(models):
        plt.legend(loc = 'upper left', bbox_to_anchor=(0, 0.25))
    plt.legend(loc = 'upper left', bbox_to_anchor=(0, 0.25))
    if save == True:
        fig.savefig(path, transparent=True, dpi=720)
    
    plt.show()


color_data = df['mdccm']
color_label = 'median cross-correlation maximum'
slow_error = df['slow_error'].to_numpy()
model_data = []
slow_error_spatial(df['backazimuth'], slow_error, model_data, color_data, color_label, niazi = False, 
                  save = True, path = '/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/figure_components/supp_fig_stacked_models_slow_error.png')
#%%
def fourier2(theta,
             a0,
             a1,b1):
             

        return (
            a0
            + a1*np.cos(1*theta) + b1*np.sin(1*theta)
            
        )
#Try to remove trend from data########################
#Binned statistics
angles = df['backazimuth']
values = slow_error

bins = np.arange(0, 361, 10)

medians = []
counts = []
bin_centers = []

min_count = 10

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

params, _ = curve_fit(fourier2, theta, medians)

## Smooth curve
#theta_fit = np.linspace(0, 2*np.pi, 500)

theta = np.deg2rad(df['backazimuth'].to_numpy())

y_fit = fourier2(theta, *params)

#Keep error between -180 to 180
slow_error_temp = slow_error - y_fit
slow_corrected = slow_error_temp
df['slow_corrected'] = slow_corrected

slow_error_spatial(df['backazimuth'], slow_corrected, model_data, color_data, color_label, niazi = True, 
                  save = False, path = '/Users/cadequigley/Downloads/Research/fig3_baz_error_model.png')


########################################

#----------GRID PLOTS------------------

########################################


def quantile_range(array): #numpy array
    ah = np.quantile(array, 0.95)
    aw = np.quantile(array, 0.05)
    return ah-aw

def grid_plot(df1, plot_type, freq_list, window_list, save = False, path = None):
    
    if plot_type == 'baz':
        cmap = 'inferno_r' #'YlOrRd'#'Reds_r' #inferno_r
        vmin = 15 #20
        vmax = 75 #50
    elif plot_type == 'slow':
        cmap = 'cividis_r'
        vmin = 0.04
        vmax = 0.2

     #Blues_r
#data = [df6,df4]
    fig, ax = plt.subplots(figsize = (len(freq_list),len(window_list)))
    
    total = 0
    Z1 = []
    '''
    for i in range(len(freq_list)):
    # SET UP FK-----------------------------------------------
        x1 = pd.DataFrame(df1[df1['max_freq']== freq_list[i]])
        x1 = x1.sort_values(by='window_length')
        temp_x1 = []
        for k in range(len(window_list)):
            x2 = pd.DataFrame(x1[x1['window_length'] == window_list[k]])
            print('Window:', window_list[k], 's')
            print('Freq:', freq_list[i], 'Hz')
            print('Number of events:', len(x2))
            qrange = quantile_range(x2['baz_corrected'])
            print('Quantile range:', qrange, 'degrees')
            temp_x1.append(qrange)
            total = total + len(x2)
    '''
    quantiles = []
    quantiles2 = []
    outliers = []
    freqs = []
    wins = []
    mean_abs_error_correct = []
    mean_abs_error = []
    for i in range(len(freq_list)):
        temp_x1 = []
        temp_x2 = []
        for k in range(len(window_list)):
            df_subset = df1[(df1["window_length"] == window_list[k]) &
                        (df1["max_freq"] == freq_list[i])]
            print('Window:', window_list[k], 's')
            wins.append(window_list[k])
            print('Freq:', freq_list[i], 'Hz')
            freqs.append(freq_list[i])
            print('Number of events:', len(df_subset))
            n_nans = df_subset[plot_type+"_corrected"].isna().sum()
            print('Number of nans:', n_nans)
            qrange = quantile_range(df_subset[plot_type+'_corrected'])
            quantiles.append(qrange)
            temp_x1.append(qrange)
            print('Quantile range:', qrange, 'degrees')
            #outi = len(abs(df_subset['baz_corrected'].to_numpy()) > 50)
            if plot_type == 'baz':
                outi = (df_subset['baz_corrected'].abs() > 30).sum()
                
            elif plot_type == 'slow':
                outi  = outi = (df_subset['slow_corrected'].abs() > 0.1).sum()
                
            outliers.append(outi)
            print('Number of outliers:', outi)
            mean_error = np.mean(np.abs(df_subset[plot_type+'_corrected']))
            mean_abs_error_correct.append(mean_error)
            mean_error = np.mean(np.abs(df_subset[plot_type+'_error']))
            mean_abs_error.append(mean_error)
            print('Mean absolute error:', mean_error)
            #Fourier fit------------------------------
            theta = np.deg2rad(df_subset['backazimuth'].to_numpy())
            baz_error = df_subset['baz_error'].to_numpy()

            params, _ = curve_fit(fourier5, theta, baz_error)

            ## Smooth curve
            #theta_fit = np.linspace(0, 2*np.pi, 500)

            y_fit = fourier5(theta, *params)
            
            #Keep error between -180 to 180
            baz_error_temp = baz_error - y_fit
            baz_corrected = ((baz_error_temp + 180) % 360) - 180
            qrange = quantile_range(baz_corrected)
            temp_x2.append(qrange)
            quantiles2.append(qrange)
            


            

        #x1 = x1['quantile_range_'+y_variable+'_'+correction].to_numpy()
        Z1.append(np.array(temp_x1))
        #Z1.append(np.array(temp_x2))
    data = {
        'window_length': wins,
        'frequency': freqs,
        'quantiles': quantiles,
        'outliers': outliers,
        'mean_abs_error': mean_abs_error,
        'mean_abs_error_correct': mean_abs_error_correct
        }
    
    array_data = pd.DataFrame(data) #print(array_data)
    array_data = array_data.sort_values(by='window_length').reset_index()

    print('Total events:', total)
    Z1 = np.array(Z1)

    #SET UP PLOTTING-------------------------------------------------
    im1 = ax.imshow(Z1, cmap = cmap, vmin= vmin,vmax =vmax, origin = 'lower') #inferno
    ax.set_xticks([0,1,2,3,4,5])
    ax.set_xticklabels(['0.5', '1', '2', '3', '4', '5'])
    ax.set_yticks([0,1,2,3,4,5])
    ax.set_yticklabels(['4', '6', '8', '10', '15', '20'])
    ax.set_xlabel('Window length (s)')
    ax.set_ylabel('Max frequency (Hz)')

    plt.tight_layout()
    fig.colorbar(im1, ax=ax, orientation='vertical', label='90% Quantile Range', shrink=0.8)
    if save:
        fig.savefig(path+'_'+plot_type+'_grid_plot.pdf', transparent=True, dpi=720)

    plt.show()
    return array_data

plot_type = 'baz'
#freq_list = [20, 15, 10, 8, 6, 4]
freq_list = [4, 6, 8, 10, 15, 20]
window_list = [1.5, 2, 3, 4, 5, 6]
array_data_baz = grid_plot(df, plot_type, freq_list, window_list, save = False, path = fig_path)

plot_type = 'slow'
array_data_slow = grid_plot(df, plot_type, freq_list, window_list, save = False, path = fig_path)

print(array_data_baz)
print(array_data_slow)
######-----------------------------
###------Plotting single event------------
#########################################
# %%
df1 = df.copy()


df = pd.DataFrame(df[df['distance']<= 600 ]) #300
df = pd.DataFrame(df[df['window_length']== 6 ])
df = pd.DataFrame(df[df['max_freq']== 10 ]).reset_index(drop=True)
#df = pd.DataFrame(df[df['magnitude']>= 6.5]).reset_index(drop=True)
indices = df.index[df['magnitude'] >= 5.0].tolist()
index = indices[11]
#index = 0 #140 #16, 15 if using power
array_lats = [53.6974]  #2A
array_lons = [-166.7343] #2A

#array_lats = [59.618433] #HM
#array_lons = [-151.141416] #HM
real_bazs_array1 = df['backazimuth'].to_numpy()
array1_bazs = df['array_baz_correct'].to_numpy()
earthquake_mags = df['magnitude'].to_numpy()
earthquake_lats = df['latitude'].to_numpy()
earthquake_lons = df['longitude'].to_numpy()
earthquake_depths = df['depth'].to_numpy()
baz_conf = 4.5
real_bazs_array2 = []
array2_bazs = []

#pygmt_single_event(index, array_lat, array_lon, earthquake_lats, earthquake_lons, earthquake_mags, real_bazs, array_bazs, baz_conf, plot_real = False)

pygmt_single_event_new(index, array_lats, array_lons, earthquake_lats, earthquake_lons, earthquake_mags, earthquake_depths, real_bazs_array1, array1_bazs,
                    real_bazs_array2, array2_bazs, baz_conf, plot_real = False, save = False, path  = fig_path)

# %%

####################################
#---CONFIDENCE----------------------
####################################
df = df1.copy()
if processing == 'fk':
    var = 'relpow'
elif processing == 'lts' or processing == 'ls':
    var = 'mdccm'

window = 6
freq = 10

df = pd.DataFrame(df[df['window_length']== window]) #6
df = pd.DataFrame(df[df['max_freq']== freq])

xvals = np.linspace(0, 1.0, 21)

quantiles_conf = []
low_range = []
num_values = []
dropped_events = []
baz_std = []
slow_std = []
baz_quantiles = []
slow_quantiles = []
baz_mean_abs_err = []
slow_mean_abs_err = []
percent_events = []

for i in range(len(xvals)-1):
    df_temp = pd.DataFrame(df[df[var] >= xvals[i]]) #'mdccm'
    df_temp = pd.DataFrame(df_temp[df_temp[var] < xvals[i+1]]) #'mdccm'
    baz_correct = df_temp['baz_corrected'].to_numpy()
    slow_correct = df_temp['slow_corrected'].to_numpy()
    if len(df_temp) > 2: #10
        baz_quantiles.append(quantile_range(baz_correct))
        slow_quantiles.append(quantile_range(slow_correct))
        low_range.append(xvals[i])
        num_values.append(len(df_temp))
        baz_std.append(np.std(baz_correct))
        slow_std.append(np.std(slow_correct))
        baz_mean_abs_err.append(np.mean(abs(baz_correct)))
        slow_mean_abs_err.append(np.mean(abs(slow_correct)))
        dropped_events.append(abs(len(df) - len(df_temp)))
        percent_events.append((len(df_temp)/len(df))*100)
        

    else:
        print('Not enough earthquakes above', xvals[i])
data = {
        'baz_mean_abs_err': baz_mean_abs_err,
        'low_range': low_range}
data = pd.DataFrame(data)
data_temp = pd.DataFrame(data[data['baz_mean_abs_err'] < 10])
thresh = data_temp['low_range'].to_numpy()[0]
data_temp = pd.DataFrame(data[data['baz_mean_abs_err'] < 20])
thresh20 = data_temp['low_range'].to_numpy()[0]
fig, ax = plt.subplots(figsize = (6,4))


#ax.scatter(low_range, baz_mean_abs_err, color = 'firebrick', edgecolors = 'black',
                   #s = 100, label = 'Observed', alpha = 1.0)
low_range = np.array(low_range)
sc = ax.scatter(low_range , baz_mean_abs_err, c = num_values, cmap = 'hot_r', edgecolors = 'black',
                   s = 100, vmin = 0, vmax = 80, label = 'Observed', alpha = 1.0 )
for i in range(len(low_range)):
    ax.text(low_range[i] , baz_mean_abs_err[i] + 3, str(num_values[i]),
            ha = 'center')
    
#ax.text(0.5, 70, 'Window length: '+str(window)+' s', weight = 'semibold')
#ax.text(0.5, 60, 'Max freq: '+str(freq)+ ' Hz', weight = 'semibold')
ax.text(0.98, 0.95, 'Window length: '+str(window-1)+' s', weight='semibold',
        transform=ax.transAxes, ha='right', va='top')
ax.text(0.98, 0.88, 'Max freq: '+str(freq)+' Hz', weight='semibold',
        transform=ax.transAxes, ha='right', va='top')
ax.set_xlabel('MDCCM bins')
ax.set_ylabel('Mean absolute back azimuth error (degrees)')
ax.grid(alpha = 0.3)
#ax.axvline(0.325, color = 'red', linestyle = '--')
ax.axvline(thresh, color = 'red', linestyle = '--', alpha = 0.4)
ax.axhline(10, color = 'red', linestyle = '--', alpha = 0.4)
#ax.axvline(thresh20, color = 'blue', linestyle = '--')
#ax.axhline(20, color = 'blue', linestyle = '--')
plt.colorbar(sc, label = 'Number of events')
#ax.set_xlim(-0.05, 1)
#ax.set_xlim(0.15,0.85) #for figure plot
#ax.set_ylim(0,10)
fig.savefig('/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/figure_components/supp_fig_mdccm_bins.png', transparent=True, dpi=720)
plt.show()
print('Threshold (10):', thresh)
print('Threshold (20):', thresh20 )
print('Percent of events above threshold:', len(pd.DataFrame(df[df['mdccm']> thresh]))/len(df))
#%%
####################################
#Calculate thresholds---------------
#----------------------------------

thresh20_list = []
thresh10_list = []
percent20_list = []
percent10_list = []
windows = []
freqs = []
mean_abs_error_20_list = []
mean_abs_error_10_list = []
for l in range(len(window_list)):
    for k in range(len(freq_list)):
        window = window_list[l]
        freq = freq_list[k]
        df = df1.copy()
        df = pd.DataFrame(df[df['window_length']== window]) #6
        df = pd.DataFrame(df[df['max_freq']== freq])

        xvals = np.linspace(0, 1.0, 21)

        quantiles_conf = []
        low_range = []
        num_values = []
        dropped_events = []
        baz_std = []
        slow_std = []
        baz_quantiles = []
        slow_quantiles = []
        baz_mean_abs_err = []
        slow_mean_abs_err = []
        percent_events = []

        for i in range(len(xvals)-1):
            df_temp = pd.DataFrame(df[df[var] >= xvals[i]]) #'mdccm'
            df_temp = pd.DataFrame(df_temp[df_temp[var] < xvals[i+1]]) #'mdccm'
            baz_correct = df_temp['baz_corrected'].to_numpy()
            slow_correct = df_temp['slow_corrected'].to_numpy()
            if len(df_temp) > 2: #10
                baz_quantiles.append(quantile_range(baz_correct))
                slow_quantiles.append(quantile_range(slow_correct))
                low_range.append(xvals[i])
                num_values.append(len(df_temp))
                baz_std.append(np.std(baz_correct))
                slow_std.append(np.std(slow_correct))
                baz_mean_abs_err.append(np.mean(abs(baz_correct)))
                slow_mean_abs_err.append(np.mean(abs(slow_correct)))
                dropped_events.append(abs(len(df) - len(df_temp)))
                percent_events.append((len(df_temp)/len(df))*100)
                

            else:
                print('Not enough earthquakes above', xvals[i])
        data = {
                'baz_mean_abs_err': baz_mean_abs_err,
                'low_range': low_range}
        data = pd.DataFrame(data)
        data_temp = pd.DataFrame(data[data['baz_mean_abs_err'] < 10])
        thresh10 = data_temp['low_range'].to_numpy()[0]
        data_temp = pd.DataFrame(data[data['baz_mean_abs_err'] < 20])
        thresh20 = data_temp['low_range'].to_numpy()[0]
        percent10 = len(pd.DataFrame(df[df['mdccm']> thresh10]))/len(df)
        percent20 = len(pd.DataFrame(df[df['mdccm']> thresh20]))/len(df)
        
        
        temp_df = pd.DataFrame(df[df['mdccm']> thresh20])
        mean_abs_error_20_list.append(np.mean(np.abs(temp_df['baz_corrected'].to_numpy())))
        temp_df = pd.DataFrame(df[df['mdccm']> thresh10])
        mean_abs_error_10_list.append(np.mean(np.abs(temp_df['baz_corrected'].to_numpy())))
        thresh10_list.append(thresh10)
        thresh20_list.append(thresh20)
        percent10_list.append(percent10)
        percent20_list.append(percent20)
        windows.append(window_list[l])
        freqs.append(freq_list[k])
data = {'freq': freqs, 
        'window': windows, 
        'thresh10': thresh10_list,
        'thresh20': thresh20_list,
        'percent20': percent20_list,
        'percent10': percent10_list,
        'mean_abs_error_20': mean_abs_error_20_list,
        'mean_abs_error_10': mean_abs_error_10_list,
        }
data = pd.DataFrame(data)
data = data.sort_values(by='window')

####################################
#---MAGNITUDE BINS-----------------
####################################
#%%
df = df1.copy()
df = pd.DataFrame(df[df['window_length']== 4]) #6
df = pd.DataFrame(df[df['max_freq']== 10]) #10
xvals = np.linspace(3, 7, 9)
var = 'magnitude'

quantiles_conf = []
low_range = []
num_values = []
dropped_events = []
baz_std = []
slow_std = []
baz_quantiles = []
slow_quantiles = []
baz_mean_abs_err = []
slow_mean_abs_err = []
percent_events = []

for i in range(len(xvals)-1):
    df_temp = pd.DataFrame(df[df[var] >= xvals[i]]) #'mdccm'
    df_temp = pd.DataFrame(df_temp[df_temp[var] < xvals[i+1]]) #'mdccm'
    baz_correct = df_temp['baz_corrected'].to_numpy()
    slow_correct = df_temp['slow_corrected'].to_numpy()
    if len(df_temp) > 0: #10
        baz_quantiles.append(quantile_range(baz_correct))
        slow_quantiles.append(quantile_range(slow_correct))
        low_range.append(xvals[i])
        num_values.append(len(df_temp))
        baz_std.append(np.std(baz_correct))
        slow_std.append(np.std(slow_correct))
        baz_mean_abs_err.append(np.mean(abs(baz_correct)))
        slow_mean_abs_err.append(np.mean(abs(slow_correct)))
        dropped_events.append(abs(len(df) - len(df_temp)))
        percent_events.append((len(df_temp)/len(df))*100)
        

    else:
        print('Not enough earthquakes above', xvals[i])

fig, ax = plt.subplots(figsize = (6,4))


#ax.scatter(low_range, baz_mean_abs_err, color = 'firebrick', edgecolors = 'black',
                   #s = 100, label = 'Observed', alpha = 1.0)
sc = ax.scatter(low_range, baz_mean_abs_err, c = num_values, cmap = 'hot_r', edgecolors = 'black',
                   s = 100, label = 'Observed', alpha = 1.0 )
for i in range(len(low_range)):
    ax.text(low_range[i], baz_mean_abs_err[i] + 0.3, str(num_values[i]),
            ha = 'center')



ax.set_xlabel('Magnitude bin')
ax.set_ylabel('Mean absolute error')
ax.grid(alpha = 0.3)
#ax.axvline(0.375, color = 'red', linestyle = '--')
plt.colorbar(sc, label = 'Number of events')
#ax.set_xlim(-0.05, 1)
ax.set_ylim(0,12)

plt.show()


###############################
#---------SPATIAL ERROR--------
###############################
#%%
df = pd.read_csv('/Users/cadequigley/repos/array_aggregator/2A_600km_m3_lts__window_freq_test2.csv') #_const_lfreq.csv')
df = pd.DataFrame(df[df['window_length']== 2]) #6
df = pd.DataFrame(df[df['max_freq']== 6]) #10

baz_error_model = []
color_data_label = 'MDCCM'
color_data = df['mdccm']

from array_figures import baz_error_spatial
baz_error_spatial(df['backazimuth'], df['baz_error'], baz_error_model, color_data, 
                      color_data_label, niazi = True, plot_fourier = False, 
                      plot_anisotropic = False, plot_anisotropic_reduced = False,
                      plot_bins = False, save = False, 
                      path = None)
# %%

def grid_plot2(df1, plot_type, freq_list, window_list, variable, variable2, save = False, path = None):
    
    if plot_type == 'baz':
        cmap = 'YlGn' #'BuPu_r' #'inferno_r' #'YlOrRd'#'Reds_r' #inferno_r
        vmin = 0#50 #4 
        vmax =1 #100 #12 
    elif plot_type == 'slow':
        cmap = 'cividis_r'
        vmin = 0.04
        vmax = 0.2

    fig, ax = plt.subplots(figsize = (len(freq_list),len(window_list)))
    
    total = 0
    Z1 = []
    Z2 = []
    
    for i in range(len(freq_list)):
        for k in range(len(window_list)):
            x1 = pd.DataFrame(df1[df1['freq']== freq_list[i]])
            x1 = x1.sort_values(by='window')
            temp_x1 = x1[variable].to_numpy()
            temp_x2 = x1[variable2].to_numpy()
            if variable == 'percent10' or variable == 'percent20':
                temp_x1 = temp_x1*100
            if variable2 == 'percent10' or variable2 == 'percent20':
                temp_x2 = temp_x2*100
        Z1.append(np.array(temp_x1))
        Z2.append(np.array(temp_x2))
    
    Z1 = np.array(Z1)
    Z2 = np.array(Z2)

    #SET UP PLOTTING-------------------------------------------------
    im1 = ax.imshow(Z1, cmap = cmap, origin = 'lower', vmin= vmin,vmax =vmax) #inferno, vmin= vmin,vmax =vmax
    ax.set_xticks([0,1,2,3,4,5])
    ax.set_xticklabels(['0.5', '1', '2', '3', '4', '5'])
    ax.set_yticks([0,1,2,3,4,5])
    ax.set_yticklabels(['4', '6', '8', '10', '15', '20'])
    ax.set_xlabel('Window length (s)')
    ax.set_ylabel('Max frequency (Hz)')

    # ADD NUMBER LABELS IN EACH CELL -----------------------------------
    for row in range(Z1.shape[0]):
        for col in range(Z1.shape[1]):
            value = Z2[row, col]
            # pick a text color that stays readable against the cell color
            #norm_val = (value - vmin) / (vmax - vmin)
            text_color = 'black'#'white' #if norm_val < 0.5 else 'black'
            ax.text(col, row, f'{value:.2f}', ha='center', va='center',
                     color=text_color, fontsize=9)

    plt.tight_layout()
    fig.colorbar(im1, ax=ax, orientation='vertical', 
                 label='MDCCM threshold', shrink=0.8) #'Mean absolute error (degrees)', 'Percent events above threshold'
    if save:
        fig.savefig(path+'supp_grid_plot_mdccm_thresh.pdf', transparent=True, dpi=720)

    plt.show()


plot_type = 'baz'
#freq_list = [20, 15, 10, 8, 6, 4]
freq_list = [4, 6, 8, 10, 15, 20]
window_list = [1.5, 2, 3, 4, 5, 6]
#percent10, percent20, thresh10, thresh20
grid_plot2(data, plot_type, freq_list, window_list, 'thresh10',
            'thresh10', save = False, path = fig_path)
# %%
