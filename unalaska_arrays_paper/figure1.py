import pandas as pd
import numpy as np
import pygmt
from pygmt.params import Position, Box
import matplotlib.pyplot as plt

import tempfile
from obspy import read_events
import obspy
from scipy import stats
from datetime import datetime
import ssl
from obspy import read
from obspy import Stream
from obspy import Trace
from obspy import UTCDateTime
#from geopy.distance import geodesic
from obspy.geodetics import gps2dist_azimuth
#import geopy

import tempfile
from pyproj import Geod

from obspy import UTCDateTime
import pandas as pd
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
from obspy import read_inventory

from array_functions import get_geometry, data_from_inventory

from subarray_functions import array_layout, array_response



###################################
#---------PLOT BASEMAP------------
###################################
def pygmt_array_earthquakes(array_lats, array_lons, array_names, earthquake_lats, earthquake_lons, earthquake_mag, earthquake_depth, save=False, path = None):

    # DEFINE CPT BASED ON AEC BASEMAP
    AEC_BASEMAP_CPT = """
    # COLOR_MODEL = RGB
    -12000  76  81  88  -7000  76  81  88
    -7000  111 117 124  -6000 111 117 124
    -6000  122 129 136  -5000 122 129 136
    -5000  131 137 144  -4000 131 137 144
    -4000  139 146 153  -3000 139 146 157
    -3000  142 149 157  -2000 142 149 157
    -2000  154 161 168  -1000 154 161 168
    -1000  162 168 176   -500 162 168 176
    -500   165 172 179   -250 165 172 179
    -250   167 174 182      0 167 174 182
    0      240 240 240   9000 240 240 240
    """
    
    # Create a temporary file for the CPT
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.cpt') as tmp_cpt:
        tmp_cpt.write(AEC_BASEMAP_CPT)
        tmp_cpt_path = tmp_cpt.name  # Save path to use later


    amplitude = 0.05 #0.2

    pygmt.config(FORMAT_GEO_MAP="ddd.x") # Highlevel formatting (no ticks, no labels)

    
    #Combine data for sorting
    data = {
            'depth': earthquake_depth,
            'latitude': earthquake_lats,
            'longitude': earthquake_lons,
            'magnitude': earthquake_mag,
            'normalized_mag': earthquake_mag/np.mean(earthquake_mag)
            }
    df = pd.DataFrame(data)

    shallow = df[df['depth'] <= 35]
    intermediate = df[(df['depth'] > 35) & (df['depth'] <= 100)]
    deep = df[df['depth'] > 100]
    #shallow_sm = df[df['depth'] <= 35]
   # intermediate_sm = df[(df['depth'] > 35) & (df['depth'] <= 100)]
    #deep_sm = df[df['depth'] > 100]


    #Define projection and grid map resolution (for BOTH maps)

    sizes = list(np.ones(len(array_lats))*300)

    #Grabs larger scale map
    left = np.min(df['longitude'])
    right = np.max(df['longitude'])
    bottom = np.min(df['latitude'])
    top = np.max(df['latitude'])

    region = [left-5, right+5, bottom-5, top+5]

#region=[-170,-140,50,68]
    #region_rect = str(left-0.5)+"/"+str(bottom-0.5)+"/"+str(right+0.5)+"/"+str(top+0.5)
    region_rect = str(left-0.5)+"/"+str(bottom-0.5)+"/"+str(right+0.5)+"/"+str(top+0.5)+"r"
#region_rect = "-162/52/-142/64r"

#rectangular designation for plotted mat

    # ADD north or south hemisphere check

    
    #if north == True:
        #hemisphere = 90
    projection="M0/0/12c"
    
    #projection = f'S210/{hemisphere}/8i'

    run_topo = True
##---Begin basemap w/ only AK topography---##

    if run_topo == True:
    # Load topography
        load_grid = pygmt.datasets.load_earth_relief(resolution='15s', region=region, registration=None, data_source='igpp', use_srtm=False) #30s
    
        #pyGMT basemap with topography figure
        fig = pygmt.Figure()
        pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain",  MAP_FRAME_PEN='1p') #Formatting, MAP_FRAME_AXES='lrbt',
        #pygmt.config(FORMAT_GEO_MAP="ddd.x",  MAP_FRAME_PEN='1p') #Formatting
    
        #Add topography basemap (DEM)
        fig.basemap(frame=True, region=region_rect, projection=projection, map_scale="jBR+w200k+o0.5c/0.5c+f+lkm")
    #fig.coast(dcw="US.AK+p0.25p")
    
    #Define outline and color pallete of basemap
        #fig.coast( water='#C6E2EE', borders="10/10p,black", shorelines="1/0.5p,black") #frame=[fig_title], shorelines=True,
        dgrid = pygmt.grdgradient(grid=load_grid, radiance=[270,30])
        pygmt.makecpt(cmap=tmp_cpt_path)
    #pygmt.makecpt(cmap=CPT_Option)  #, series=[-1.5, 0.3, 0.01])
   
        fig.grdimage(grid=load_grid, shading='+a300+nt0.8', cmap=True, transparency=60) #35

        fig.coast( water=None, borders="10/10p,black", shorelines="1/0.5p,black")

        #Plot circle-------------
        fig.plot(x=array_lons[2], y=array_lats[2], size=[350], style="E-", pen="1.5p,black,-", transparency = 60)
        
        #Plot earthquakes------------------------------------------------------------

        #Plot earthquakes------------------------------------------------------------
        if len(deep)>0:
            fig.plot(x=deep['longitude'], y=deep['latitude'], size=amplitude*(1.6**deep['magnitude']),
                 style="cc", pen='0.5p,black', fill = '#4D0010') #darkbrown, gray14
        
        fig.plot(x=intermediate['longitude'], y=intermediate['latitude'], size=amplitude*(1.6**intermediate['magnitude']), #2.1
             style="cc", pen='0.5p,black', fill = 'gold1') #gold1, gray40, #EBB41E

        fig.plot(x=shallow['longitude'], y=shallow['latitude'], size=amplitude*(1.6**shallow['magnitude']),
             style="cc", pen='0.5p,black', fill = 'firebrick') #firebrick, gray66, #FB0006

        

        #Create earthquakes for size scaling----------------
        #tempx = [-171,-171,-171, -171]
        #tempy = [52.2, 52.5, 53, 54]
        #tempmag = [3,4,5,6]
        #fig.plot(x = tempx, y = tempy, size = amplitude*(1.6**np.array(tempmag)), style="cc", pen='0.5p,black', fill = 'whitesmoke')


        #Plot mini arrays-----------------------------------------------------------------
        fig.plot(x = array_lons,
             y = array_lats,
             style = "i1c",pen = '1.5p,black', size = sizes, fill = 'cyan4')
        
        #Plot text---------------------------------------------
        if len(array_names) > 0:
            fig.text(text=array_names, x=array_lons, y=np.array(array_lats)+0.2,
                     font = "18p,Helvetica-Bold,black") #fill = 'whitesmoke')

    
    
        if save == True:
            fig.savefig(path,  dpi=720) # transparent=True,
        
        fig.show(dpi=720)


#Plot figure
save_fig = False
fig_path = './figure_components/'
df = pd.read_csv('./all_earthquakes_m3_400km.csv')
earthquake_map = True

array_lats = [53.6974, 53.779, 53.8566]  
array_lons = [-166.7343, -166.2131,-166.4161]
array_names = []
earthquake_lats = df['latitude'].to_numpy()
earthquake_lons = df['longitude'].to_numpy()
earthquake_mags = df['magnitude'].to_numpy()
earthquake_depths = df['depth'].to_numpy()

if earthquake_map == True:
    pygmt_array_earthquakes(array_lats, array_lons, array_names, earthquake_lats,
                            earthquake_lons, earthquake_mags, earthquake_depths,
                            save=save_fig, path = fig_path+'earthquake_map.pdf')#path = '/Users/cadequigley/Downloads/Research/deployment_array_design/POM_eq_map_SSA.png')
    
###################################
#------PLOT ARRAY REPONSE----------
###################################



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

    ax.set_xlabel(r'$U_{x}$ (s/km)')
    ax.set_ylabel(r'$U_{y}$ (s/km)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5,0.5)
    plt.colorbar(sc, label = 'relative power', ax = ax)
    #ax.set_title('RC: '+ str(RC))
    
    if save:
        fig.savefig(path, transparent=True, dpi=720)
        
    plt.show()


client = 'EARTHSCOPE'
net = '9C'
sta = '3A**'
chan = '*'
loc = '*'
starttime = '2015-10-01'
endtime = '2015-10-02'
freq_min = 3 #0.5
freq_max = 8 #10

client = Client(client)
keep_stations = []
remove_stations = []
bhz_list = ['3A04', '3A13'] #['3A04', '3A13']#['2A08', '2A11']
bad_list = ['3A10', '3A15']#['3A10', '3A15']#['POM06', 'POM07', 'POM18']
inv = client.get_stations(network=net, station=sta, channel=chan,
                         location=loc, starttime=UTCDateTime(starttime),
                         endtime=UTCDateTime(endtime), level='response') #level = 'channel'
    
    #Pull station information out of inventory
(lat_list, lon_list, elev_list, station_d1_list,
start_d1_list, end_d1_list, num_channels_d1_list) = data_from_inventory(inv, 
                                                                            remove_stations, 
                                                                            keep_stations)


    #Save stations for later
data = {
        'station': station_d1_list,
        'lat': lat_list,
        'lon': lon_list,
        'elevation': elev_list}

station_info = pd.DataFrame(data) 

#Calculate array reponse for non bhz/bad stations
bhz_mask = np.isin(station_d1_list, bhz_list)
bad_mask = np.isin(station_d1_list, bad_list)

# stations that are neither bhz nor bad
normal_mask = ~(bhz_mask | bad_mask)

station_names_sublist = station_info['station'].to_numpy()[normal_mask]

#Get array layout-------------------
xpos, ypos, xpos_sub, ypos_sub, xpos_sub_cent, ypos_sub_cent = array_layout(lat_list, lon_list, elev_list, station_d1_list,
                 station_names_sublist, save=False, path=None, plot = False)

#Calculate response-----------------
xpos_sub1  = np.array(xpos_sub_cent)/1000
ypos_sub1  = np.array(ypos_sub_cent)/1000
print(len(xpos_sub1))
print(len(ypos_sub1))
array_resp, px, py, array_resp_max, RC = array_response(xpos_sub1, ypos_sub1, c_app=2, c_steps=300, 
                                                          freqmin=freq_min, freqmax=freq_max, freqsteps=50, 
                                                          px_0=0, py_0=0)
#Plot reponse----------------------
response_figure(array_resp, px, py, array_resp_max, RC, save = False, path = None)


###################################
#------PLOT STATION LAYOUT----------
###################################

def array_layout_plot(lat_list, lon_list, elev_list, station_names,
                 bhz_list, bad_list, save=False, path=None):

    output = get_geometry(lat_list, lon_list, elev_list, return_center=True)

    station_names = np.array(station_names)

    xpos = np.array([(output[i][0]) * 1000 for i in range(len(output)-1)])
    ypos = np.array([(output[i][1]) * 1000 for i in range(len(output)-1)])

    xmax = np.max(np.abs(xpos))
    ymax = np.max(np.abs(ypos))
    scale = max(xmax, ymax)

    fig, ax = plt.subplots(figsize=(5,5))

    # ---- Create masks ----
    bhz_mask = np.isin(station_names, bhz_list)
    bad_mask = np.isin(station_names, bad_list)

    # stations that are neither bhz nor bad
    normal_mask = ~(bhz_mask | bad_mask)

    # ---- Plot normal stations ----
    ax.scatter(xpos[normal_mask], ypos[normal_mask],
               color='steelblue', marker='^',
               linewidths=1, s=300, edgecolors='black')

    # ---- Plot BHZ stations ----
    ax.scatter(xpos[bhz_mask], ypos[bhz_mask],
               color='gray', marker='^',
               linewidths=1, s=300, edgecolors='black', alpha = 0.5)

    # ---- Plot bad stations ----
    ax.scatter(xpos[bad_mask], ypos[bad_mask],
               color='firebrick', marker='^',
               linewidths=1, s=300, edgecolors='black', alpha=0.5)

    # ---- Label stations ----
    for i in range(len(station_names)):
        ax.text(xpos[i]-100, ypos[i]+80, station_names[i], weight = 'bold')

    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.grid(alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    #ax.set_xlim(-scale-(0.1*scale), scale+(0.1*scale))
    #ax.set_ylim(-scale-(0.1*scale), scale+(0.1*scale))
    ax.set_xlim(-650, 650)
    ax.set_ylim(-650, 650)

    if save:
        fig.savefig(path, transparent=True, dpi=720, format = 'pdf')

    plt.show()

array_layout_plot(lat_list, lon_list, elev_list, station_d1_list,
                 bhz_list, bad_list, save=True, path='./figure_components/3A_layout.pdf')
