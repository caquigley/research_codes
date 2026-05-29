import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from array_functions import get_geometry



### DEFINITION OF ARRAY RESPONSE FROM CARDINAL--------------------------------------------------------------------------------------
def array_response(x, y, c_app=280, c_steps=50, freqmin=1, freqmax=2, freqsteps=50, px_0=0, py_0=0):
    '''---------------------------------------------------------------------------------------------------------
    Calculate array response on a square slowness grid for an arbitrary array of N elements

    Input:
        x (array): x-points in array
        y (array): y-points in array
        c_app (float/int): apparent velocity used to construct extent of slowness grid
        c_steps (int): define resolution of slowness grid
        freqmin (int): minimum frequency (Hz)
        freqmax (int): maximum frequency (Hz)
        freqsteps (int): frequency resolution
        px_0, py_0 (float/int): coordinates which define slowness correction

    Output:
        resp_norm[::-1] (array): response function map
        p_x (array): x-component slowness
        p_y (array): y-component slowness
        resp.max(): array gain
        RC: reponse condition
    ---------------------------------------------------------------------------------------------------------'''
    # Construct slowness square grid
    s_max = 1 / c_app 
    px = np.linspace(-s_max, s_max, c_steps)
    py = np.linspace(-s_max, s_max, c_steps)
    px, py = np.meshgrid(px, py)
    #-----------------------------------------------------------------------------------------------------------------#
    # Calculate each part
    i = 1j
    omega = 2 * np.pi * np.linspace(freqmin, freqmax, freqsteps)
    p_r_product = ((px[..., np.newaxis] + px_0) * np.array(x) + (py[..., np.newaxis] + py_0) * np.array(y))
    complex = -i * omega * p_r_product[..., np.newaxis]
    #-----------------------------------------------------------------------------------------------------------------#
    # Compile
    resp = np.sum(np.abs(np.sum(np.exp(complex), 2))**2, 2)
    resp_norm = resp / resp.max()


    # Calculate Response condition
    array_resp = resp_norm[::-1]

    mean_tmp = np.mean(array_resp)
    std_tmp = np.std(array_resp)
    RC = ((mean_tmp + std_tmp) / resp.max())
    
    
    return resp_norm[::-1], px, py, resp.max(), RC


def response_figure(array_resp, px, py, max_resp, RC, save = False, path = None):
    
    
    fig, ax = plt.subplots()
    
    resp = array_resp * max_resp

    sc = ax.pcolormesh(px, py, array_resp, cmap='hot_r', vmin = 0, vmax = 1) #all normalized individually

    ax.set_xlabel(r'$U_{x}$ [s/km]')
    ax.set_ylabel(r'$U_{y}$ [s/km]')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5,0.5)
    plt.colorbar(sc, label = 'relative power', ax = ax)
    ax.set_title('RC: '+ str(RC))
    
    if save:
        fig.savefig(path, transparent=True, dpi=720)
        
    plt.show()



def array_layout(lat_list, lon_list, elev_list, station_names,
                 station_names_sublist=None, save=False, path=None, plot = False):

    output = get_geometry(lat_list, lon_list, elev_list, return_center=True)

    station_names = np.array(station_names)

    xpos = [(output[i][0]) * 1000 for i in range(len(output)-1)]
    ypos = [(output[i][1]) * 1000 for i in range(len(output)-1)]

    xmax = np.max(np.abs(xpos))
    ymax = np.max(np.abs(ypos))
    scale = max(xmax, ymax)

    fig, ax = plt.subplots(figsize=(5,5))

    if station_names_sublist is None:

        ax.scatter(xpos, ypos,
                   color='firebrick', marker='^',
                   linewidths=1, s=300, edgecolors='black')
        
        for i in range(len(xpos)):
            ax.text(xpos[i]-100, ypos[i]+80, station_names[i])

        #return xpos,ypos

    else:

        xpos_sub = []
        ypos_sub = []
        lat_list_sub = []
        lon_list_sub = []
        elev_list_sub = []

        for sta in station_names_sublist:
            idx = np.where(station_names == sta)[0][0]
            xpos_sub.append(xpos[idx])
            ypos_sub.append(ypos[idx])
            lat_list_sub.append(lat_list[idx])
            lon_list_sub.append(lon_list[idx])
            elev_list_sub.append(elev_list[idx])

        output = get_geometry(lat_list_sub, lon_list_sub, elev_list, return_center=True)


        xpos_sub_cent = [(output[i][0]) * 1000 for i in range(len(output)-1)]
        ypos_sub_cent = [(output[i][1]) * 1000 for i in range(len(output)-1)]

        ax.scatter(xpos, ypos,
                   color='gray', marker='^',
                   linewidths=1, s=300, edgecolors='black', alpha = 0.5)

        ax.scatter(xpos_sub, ypos_sub,
                   color='firebrick', marker='^',
                   linewidths=1, s=300, edgecolors='black')

        for i in range(len(station_names_sublist)):
            ax.text(xpos_sub[i]-100, ypos_sub[i]+80, station_names_sublist[i], weight = 'bold')

        #return xpos, ypos, xpos_sub, ypos_sub, xpos_sub_cent, ypos_sub_cent

    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.grid(alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlim(-scale-(0.1*scale), scale+(0.1*scale))
    ax.set_ylim(-scale-(0.1*scale), scale+(0.1*scale))

    if save:
        fig.savefig(path, transparent=True, dpi=720)

    if plot == True:
        plt.show()
    else:
        plt.close()

    if station_names_sublist is None:
        
        return xpos, ypos

    else:
        
        return xpos, ypos, xpos_sub, ypos_sub, xpos_sub_cent, ypos_sub_cent

def subarray_layout_response(xpos, ypos, xpos_sub, ypos_sub, station_names_sublist,
                            p_x, p_y, array_resp, RC, save = False, path = None):
    fig, ax = plt.subplots(ncols = 2, nrows = 1, figsize = (8,4))

    #Plot array layout------------------
    xmax = np.max(np.abs(xpos))
    ymax = np.max(np.abs(ypos))
    scale = max(xmax, ymax)
    
    ax[0].scatter(xpos, ypos,
                color='gray', marker='^',
                linewidths=1, s=300, edgecolors='black', alpha = 0.5)

    ax[0].scatter(xpos_sub, ypos_sub,
                color='firebrick', marker='^',
                linewidths=1, s=300, edgecolors='black')

    for i in range(len(station_names_sublist)):
        ax[0].text(xpos_sub[i]-100, ypos_sub[i]+80, station_names_sublist[i], weight = 'bold')

    ax[0].set_xlabel("x position (m)")
    ax[0].set_ylabel("y position (m)")
    ax[0].grid(alpha=0.3)
    ax[0].set_aspect('equal', adjustable='box')

    ax[0].set_xlim(-scale-(0.1*scale), scale+(0.1*scale))
    ax[0].set_ylim(-scale-(0.1*scale), scale+(0.1*scale))


    #Plot reponse function------------------

    #resp = array_resp * max_resp

    sc = ax[1].pcolormesh(p_x, p_y, array_resp, cmap='hot_r', vmin = 0, vmax = 1) #all normalized individually

    ax[1].set_xlabel(r'$U_{x}$ [s/km]')
    ax[1].set_ylabel(r'$U_{y}$ [s/km]')
    ax[1].set_aspect('equal', adjustable='box')
    plt.colorbar(sc, label = 'relative power', ax = ax[1])
    ax[1].set_title('RC: '+ str(RC))
    ax[1].set_xlim(-0.5, 0.5)
    ax[1].set_ylim(-0.5,0.5)

    fig.tight_layout()
    if save == True:
        fig.savefig(path, transparent=True, dpi=720)
    
    plt.show()











    

    