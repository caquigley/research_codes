import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from array_figures import baz_error_spatial
from matplotlib.transforms import blended_transform_factory
from array_functions import cos_model
from scipy.optimize import curve_fit
from array_functions import get_geometry, fourier5, anisotropy_model, anisotropic_harmonic


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

    # 3. Create the polynomial function
    polynomial = np.poly1d(coefficients)

    # 4. Evaluate the polynomial at specific points
    x_fit = np.linspace(0, 360, 1000)
    y_fit = polynomial(x_fit)
    #ax.plot(x_fit, y_fit, color = 'blue', label = 'Polynomial fit')

    #Fourier fit------------------------------
    theta = np.deg2rad(baz)

    params, _ = curve_fit(fourier5, theta, baz_error)

    # Smooth curve
    theta_fit = np.linspace(0, 2*np.pi, 500)

    y_fit = fourier5(theta_fit, *params)
    ax.plot(np.rad2deg(theta_fit), y_fit, color = 'green', linewidth = 2.5, 
               alpha= 0.8, label = 'Fourier fit')
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

    min_count = 30

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
    ax.scatter(bin_centers, medians, color = 'red', s = 150, edgecolors='black', linewidths=1)
    
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
    ax.plot(baz_fit, y_fit, color = 'red', label = 'Anisotropic harmonic')



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

df = pd.read_csv('/Users/cadequigley/repos/array_aggregator/2A_600km_m3_lts__window_freq_test2.csv')
drop_taup = True
drop_pow = False
pow_thresh = 0.5
processing = 'lts'


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

color_data = df['mdccm']
color_label = 'median cross-correlation maximum'
baz_error = df['baz_error'].to_numpy()
model_data = []
baz_error_spatial(df['backazimuth'], baz_error, model_data, color_data, color_label, niazi = True, 
                  save = False, path = '/Users/cadequigley/Downloads/Research/fig3_baz_error_model.png')



def grid_plot(df1, plot_type, freq_list, window_list, save = False, path = None):
    
    if plot_type == 'baz':
        cmap = 'inferno_r'
        vmin = 20 #10
        vmax = 75 #50
    elif plot_type:
        cmap = 'cividis_r'
        vmin = 0.04
        vmax = 0.15

     #Blues_r
#data = [df6,df4]
    fig, ax = plt.subplots(figsize = (5,5))
    
    
    Z1 = []
    for i in range(freq_list):
    # SET UP FK-----------------------------------------------
        x1 = pd.DataFrame(df1[df1['max_freq']== freq_list[i]])
        x1 = x1.sort_values(by='window_length')
        x1 = x1['quantile_range_'+y_variable+'_'+correction].to_numpy()
        Z1.append(x1)
    '''    
    x2 = pd.DataFrame(df1[df1['max_freq']== 14])
    x2 = x2.sort_values(by='window_length')
    x2 = x2['quantile_range_'+y_variable+'_'+correction].to_numpy()
    x3 = pd.DataFrame(df1[df1['max_freq']== 10])
    x3 = x3.sort_values(by='window_length')
    x3 = x3['quantile_range_'+y_variable+'_'+correction].to_numpy()
    x4 = pd.DataFrame(df1[df1['max_freq']== 8])
    x4 = x4.sort_values(by='window_length')
    x4 = x4['quantile_range_'+y_variable+'_'+correction].to_numpy()
    x5 = pd.DataFrame(df1[df1['max_freq']== 6])
    x5 = x5.sort_values(by='window_length')
    x5 = x5['quantile_range_'+y_variable+'_'+correction].to_numpy()
    '''
    #Z1 = np.array([x5,x4,x3,x2,x1])
    
    Z1 = np.array(Z1)

    #SET UP LS PLOTTING-------------------------------------------------
    im1 = ax.imshow(Z1, cmap = cmap, vmin= vmin,vmax =vmax, origin = 'lower') #inferno
    ax.set_xticks([0,1,2,3,4])
    ax.set_xticklabels(['0.5', '1', '2', '3', '4'])
    ax.set_yticks([0,1,2,3,4])
    ax.set_yticklabels(['6', '8', '10', '14', '20'])
    ax.set_xlabel('Window length (s)')
    ax.set_ylabel('Max frequency (Hz)')

    plt.tight_layout()
    fig.colorbar(im1, ax=ax, orientation='vertical', label='90% Quantile Range')
    if save:
        fig.savefig(path, transparent=True, dpi=720)

    plt.show()