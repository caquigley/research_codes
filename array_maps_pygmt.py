import pandas as pd
import numpy as np
import pygmt
#from pygmt.params import Position, Box
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



def basemap_cpt(cpt_type):
    '''
    Define and create temporary CPTs for pygmt basemaps.

    Inputs:

    
    Output:
    '''
    
    if cpt_type == "AEC":
            
        # DEFINE CPT BASED ON AEC BASEMAP
        BASEMAP_CPT = """
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
        
    
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.cpt') as tmp_cpt:
        tmp_cpt.write(BASEMAP_CPT)
        tmp_cpt_path = tmp_cpt.name  # Save path to use later

    return tmp_cpt_path



def pygmt_array_earthquakes(array_lats, array_lons, array_names, 
                            earthquake_lats, earthquake_lons, earthquake_mag,
                            earthquake_depth, save=False, path = None, 
                            cpt_type ='AEC'):


    tmp_cpt_path = basemap_cpt(cpt_type)
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
    


    #Define projection and grid map resolution (for BOTH maps)

    sizes = list(np.ones(len(array_lats))*300)
    
    lons = df['longitude'].to_numpy() % 360

    left = np.min(lons)
    right = np.max(lons)
    bottom = np.min(df['latitude'])
    top = np.max(df['latitude'])

    region = [left-5, right+5, bottom-5, top+5]


    region_rect = str(left-0.5)+"/"+str(bottom-0.5)+"/"+str(right+0.5)+"/"+str(top+0.5)+"r"

    projection = "M"+str(array_lons[0])+"/"+str(array_lats[0])+"/12c"
    
    

    run_topo = True


    if run_topo == True:
    # Load topography
        load_grid = pygmt.datasets.load_earth_relief(resolution= '15s',#'30s',#'10m', 
                                                     region=region, 
                                                     registration=None, 
                                                     data_source='igpp', 
                                                     use_srtm=False) #30s
    
        #pyGMT basemap with topography figure
        fig = pygmt.Figure()
        pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain",  
                     MAP_FRAME_PEN='1p') #Formatting, MAP_FRAME_AXES='lrbt',
        #pygmt.config(FORMAT_GEO_MAP="ddd.x",  MAP_FRAME_PEN='1p') #Formatting
    
        #Add topography basemap (DEM)
        fig.basemap(frame=True, region=region_rect, projection=projection,
                    map_scale="jBR+w500k+o0.5c/0.5c+f+lkm")
    #fig.coast(dcw="US.AK+p0.25p")
    
    #Define outline and color pallete of basemap
        #fig.coast( water='#C6E2EE', borders="10/10p,black", shorelines="1/0.5p,black") #frame=[fig_title], shorelines=True,
        dgrid = pygmt.grdgradient(grid=load_grid, radiance=[270,30])
        pygmt.makecpt(cmap=tmp_cpt_path)
    #pygmt.makecpt(cmap=CPT_Option)  #, series=[-1.5, 0.3, 0.01])
   
        fig.grdimage(grid=load_grid, shading='+a300+nt0.8', 
                     cmap=True, transparency=60) #35

        fig.coast(water=None, borders="1/1p,black", 
                  shorelines="1/0.5p,black")

        #Plot circle-------------
        #fig.plot(x=list(array_lons), y=list(array_lats), size=[350], style="E-", pen="1.5p,black,-")
        
        #Plot earthquakes------------------------------------------------------------

        #Plot earthquakes------------------------------------------------------------
        if len(deep)>0:
            small_deep = pd.DataFrame(deep[deep['magnitude']<2.5])
            big_deep = pd.DataFrame(deep[deep['magnitude']>= 2.5])
            
            if len(small_deep >0):
                fig.plot(x=small_deep['longitude'], y=small_deep['latitude'], 
                     size=amplitude*(1.6**small_deep['magnitude']), style="cc", 
                     pen='0.5p,black', fill = '#4D0010', transparency= 80) #darkbrown, gray14
            
        if len(intermediate)>0:
            small = pd.DataFrame(intermediate[intermediate['magnitude']<2.5])
            big_intermediate = pd.DataFrame(intermediate[intermediate['magnitude']>=2.5])
            
            if len(small)> 0:
                fig.plot(x=small['longitude'], y=small['latitude'], 
                    size=amplitude*(1.6**small['magnitude']), #2.1
                    style="cc", pen='0.5p,black', fill = 'gold1', transparency=80) #gold1, gray40, #EBB41E
            
        if len(shallow)>0:
            small = pd.DataFrame(shallow[shallow['magnitude']<2.5])
            big_shallow = pd.DataFrame(shallow[shallow['magnitude']>=2.5])
            
            if len(small)> 0:
                fig.plot(x=small['longitude'], y=small['latitude'], 
                    size=amplitude*(1.6**small['magnitude']), #2.1
                    style="cc", pen='0.5p,black', fill = 'firebrick', transparency=80) #gold1, gray40, #EBB41E
            
            #fig.plot(x=shallow['longitude'], y=shallow['latitude'], 
                    #size=amplitude*(1.6**shallow['magnitude']), style="cc", 
                    #pen='0.5p,black', fill = 'firebrick') #firebrick, gray66, #FB0006

        fig.plot(x=big_deep['longitude'], y=big_deep['latitude'], 
                     size=amplitude*(1.6**big_deep['magnitude']), style="cc", 
                     pen='0.5p,black', fill = '#4D0010') #darkbrown, gray14
        
        fig.plot(x=big_intermediate['longitude'], y=big_intermediate['latitude'], 
                    size=amplitude*(1.6**big_intermediate['magnitude']), #2.1
                    style="cc", pen='0.5p,black', fill = 'gold1') #gold1, gray40, #EBB41E
        
        fig.plot(x=big_shallow['longitude'], y=big_shallow['latitude'], 
                    size=amplitude*(1.6**big_shallow['magnitude']), #2.1
                    style="cc", pen='0.5p,black', fill = 'firebrick') #gold1, gray40, #EBB41E

        #Create earthquakes for size scaling----------------
        #tempx = [-171,-171,-171, -171] #9C arrays
        #tempy = [52.2, 52.5, 53, 54] #9C arryas
        #tempx = [-145,-145,-145, -145, -145] #4E arrays
        #tempy = [55, 56, 57, 58, 59] #4E arrays
        #tempmag = [1,2,3,4,5]
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
            fig.savefig(path, transparent=False, dpi=720)
        
        fig.show(dpi=720)



def transform_degrees_pygmt(degree):
    # Shift from north (0) to east (90)
    transformed_degree = (degree - 90) % 360
    return transformed_degree



def pygmt_baz_error(array_lat, array_lon, array_name, earthquake_lats, 
                    earthquake_lons, earthquake_mags, baz, baz_error, 
                    save=False, path = None):
    
    baz_real_pygmt = 360 - transform_degrees_pygmt(baz)
    vector_direction = []
    #error = comb['baz_error'].to_numpy()
    for i in range(len(baz_real_pygmt)):
        temp = baz_error[i]
        tempbaz = baz_real_pygmt[i]
        if temp > 0:
            wa = 90+tempbaz
        else:
            wa = tempbaz-90
        vector_direction.append(wa)
    
    
    vector_direction = np.array(vector_direction)
    
    new_vec = []
    for j in range(len(vector_direction)):
        if vector_direction[j] < 0:
            wa = vector_direction[j]+360
        else:
            wa = vector_direction[j]
        new_vec.append(wa)
    new_vec = np.array(new_vec)  
  
    
    
    
    df6 = pd.DataFrame(baz_error, columns = ['baz_error'])
    df6['vec_direction'] = new_vec
    df6['lat'] = earthquake_lats
    df6['lon'] = earthquake_lons
    df6['baz'] = baz
    df6['mags'] = earthquake_mags
    pos_error = pd.DataFrame(df6[df6['baz_error']>= 0])
    neg_error = pd.DataFrame(df6[df6['baz_error']<= 0])

    NE = df6[df6['baz'] <= 80]
    S = df6[(df6['baz'] > 80) & (df6['baz'] <= 210)]
    W = df6[df6['baz'] > 210]

    tmp_cpt_path = basemap_cpt("AEC")

    pygmt.config(FORMAT_GEO_MAP="ddd.x") # Highlevel formatting (no ticks, no labels)



    #Define projection and grid map resolution (for BOTH maps)
    lons = earthquake_lons % 360

    left = np.min(lons)
    right = np.max(lons)
    bottom = np.min(earthquake_lats)
    top = np.max(earthquake_lats)
    extend_region = 5 # 8, 5
    region = [left-extend_region, right+extend_region, bottom-extend_region, top+extend_region]

    extend_rect = 0.75 #3 #0.5, 5
    region_rect = str(left-extend_rect)+"/"+str(bottom-extend_rect)+"/"+str(right+extend_rect)+"/"+str(top+extend_rect)+"r"

    projection = "M"+str(array_lon)+"/"+str(array_lat)+"/12c"

    '''
    left = np.min(earthquake_lons)
    right = np.max(earthquake_lons)
    bottom = np.min(earthquake_lats)
    top = np.max(earthquake_lats)

    region = [left-5, right+5, bottom-5, top+5]

    
    region_rect = str(left-0.5)+"/"+str(bottom-0.5)+"/"+str(right+0.5)+"/"+str(top+0.5)+"r"
    

    projection="M0/0/12c"
    '''
    amplitude = 0.2 #for plotting earthquakes
    
    #projection = f'S210/{hemisphere}/8i'

    run_topo = True
    perspective = [157.5, 30, 0]
    perspective = None
    

    if run_topo == True:
    # Load topography
        load_grid = pygmt.datasets.load_earth_relief(resolution= '30s', #'10m','30s' 
                                                     region=region, 
                                                     registration=None, 
                                                     data_source='igpp', 
                                                     use_srtm=False) #30s
    
        #pyGMT basemap with topography figure
        fig = pygmt.Figure()
        pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain",
                    MAP_FRAME_PEN='1p') #Formatting, MAP_FRAME_AXES='lrbt'
        #pygmt.config(FORMAT_GEO_MAP="ddd.x",  MAP_FRAME_PEN='1p') #Formatting
    
        #Add topography basemap (DEM)
        fig.basemap(frame=True, region=region_rect, projection=projection,
                    map_scale="jBR+w500k+o0.5c/0.5c+f+lkm", perspective=perspective)
    #fig.coast(dcw="US.AK+p0.25p")
    
    #Define outline and color pallete of basemap
        #fig.coast( shorelines=True, water='#C6E2EE', borders="1/1p,black") #frame=[fig_title]
        dgrid = pygmt.grdgradient(grid=load_grid, radiance=[270,30])
        pygmt.makecpt(cmap=tmp_cpt_path)
    #pygmt.makecpt(cmap=CPT_Option)  #, series=[-1.5, 0.3, 0.01])
   
        fig.grdimage(grid=load_grid, shading='+a300+nt0.8', cmap=True,
                    transparency = 60, perspective=perspective)
        fig.coast(water=None, borders="1/0.5p,black", 
                  shorelines="1/0.5p,black", perspective=perspective)

        pygmt.makecpt(cmap='polar', series = [-110,110]) #-80, 80
        #pygmt.makecpt(cmap="SCM/lajolla", series=[0, 360])
        
        fig.plot(x= earthquake_lons, y= earthquake_lats, 
                 size=amplitude*(1.8**(earthquake_mags/np.mean(earthquake_mags))),
                 style="cc", pen='0.5p,#3e000d',cmap=True, fill = baz_error, perspective=perspective)#fill = baz_error)
        '''
        fig.plot(x=NE['lon'], y=NE['lat'], 
                 #size=amplitude*(1.6**NE['mags']), #2.1
                 size = amplitude*(1.8**(NE['mags']/np.mean(NE['mags']))),
                 style="cc", pen='0.5p,black', fill = 'gold1') #gold1, gray40, #EBB41E

        fig.plot(x=S['lon'], y=S['lat'], 
                 #size=amplitude*(1.6**S['mags']), style="cc",
                 size = amplitude*(1.8**(S['mags']/np.mean(S['mags']))), 
                 style = "cc",
                 pen='0.5p,black', fill = 'firebrick') #firebrick, gray66, #FB0006
        fig.plot(x=W['lon'], y=W['lat'], 
                 #size=amplitude*(1.6**W['mags']), 
                 size = amplitude*(1.8**(W['mags']/np.mean(W['mags']))),
                 style="cc", 
                 pen='0.5p,black', fill = 'gray66') #firebrick, gray66, #FB0006
        '''
        fig.plot(x=pos_error['lon'],
                y=pos_error['lat'],
                direction = [pos_error['vec_direction'],0.03*pos_error['baz_error']],#0.045*pos_error['baz_error']], 0.06
                #direction = [baz_array_pygmt, [length]],
                style="v0.5c+ea",
                fill = "red3",
                #fill="royalblue",
                pen="1.0p", perspective=perspective)
                #label = "Array backazimuth abs. power")
        
        fig.plot(x=neg_error['lon'],
                y=neg_error['lat'],
                direction = [neg_error['vec_direction']-180, 0.03*neg_error['baz_error']],#0.045*neg_error['baz_error']], 0.06
                #direction = [baz_array_pygmt, [length]],
                style="v0.5c+ea",
                #fill = "cyan4",
                fill="royalblue",
                pen="1.0p", perspective=perspective)
        
        fig.plot(x = array_lon,
                 y = array_lat,
                 style = "i1c",pen = '0.5p,#3e000d', size = [600], fill = 'gold1', perspective=perspective)
        
        #Plot vector for reference length---------
        #fig.plot(x=-162,
               # y=51,
                #direction = [90, 0.06*20],
                #direction = [baz_array_pygmt, [length]],
                #style="v0.5c+ea",
                #fill = "red3",
                #fill="royalblue",
                #pen="1.0p")
        
        
    
        
        #Plot text---------------------------------------------
        #fig.text(text=array_names, x=array_lons, y=np.array(array_lats)+0.2,
                 #font = "18p,Helvetica-Bold,black") #fill = 'whitesmoke')

    
    
        fig.colorbar(frame="xaf+lBackazimuth error (degrees)", perspective=perspective)
        if save == True:
            fig.savefig(path, transparent=True, dpi=720)
        #fig.savefig('/Users/cadequigley/Downloads/Research/hom_kod_earthquakes.png', transparent=True, dpi=720)
        fig.show(dpi=720)


   

def pygmt_slow_error(array_lat, array_lon, array_name, earthquake_lats, 
                     earthquake_lons, earthquake_mags, slow_error, 
                     save = False, path = None):
    
    length = 2 #vector length
    
    df6 = pd.DataFrame(slow_error, columns = ['slow_error'])
    df6['lat'] = earthquake_lats
    df6['lon'] = earthquake_lons
    pos_error = pd.DataFrame(df6[df6['slow_error']>= 0])
    neg_error = pd.DataFrame(df6[df6['slow_error']<= 0])
    amplitude = 0.2 #for plotting earthquakes

    # DEFINE CPT BASED ON AEC BASEMAP
    tmp_cpt_path = basemap_cpt("AEC")


    pygmt.config(FORMAT_GEO_MAP="ddd.x") # Highlevel formatting (no ticks, no labels)



    #Define projection and grid map resolution (for BOTH maps)
    lons = earthquake_lons % 360

    left = np.min(lons)
    right = np.max(lons)
    bottom = np.min(earthquake_lats)
    top = np.max(earthquake_lats)

    extend_region = 5 #8, 5
    region = [left-extend_region, right+extend_region, bottom - extend_region, top + extend_region]

    extend_rect = 0.75 #3 #5 (regional), 0.5 (local)
    region_rect = str(left-extend_rect)+"/"+str(bottom-extend_rect)+"/"+str(right+extend_rect)+"/"+str(top+extend_rect)+"r"

    projection = "M"+str(array_lon)+"/"+str(array_lat)+"/12c"

    '''
    left = np.min(earthquake_lons)
    right = np.max(earthquake_lons)
    bottom = np.min(earthquake_lats)
    top = np.max(earthquake_lats)

    region = [left-5, right+5, bottom-5, top+5]

    #region=[-170,-140,50,68]
    #region_rect = str(left-0.5)+"/"+str(bottom-0.5)+"/"+str(right+0.5)+"/"+str(top+0.5)
    region_rect = str(left-0.5)+"/"+str(bottom-0.5)+"/"+str(right+0.5)+"/"+str(top+0.5)+"r"
    #region_rect = "-162/52/-142/64r"

    projection="M0/0/12c"
    '''
    #projection = f'S210/{hemisphere}/8i'

    run_topo = True
    ##---Begin basemap w/ only AK topography---##

    if run_topo == True:
    # Load topography
        load_grid = pygmt.datasets.load_earth_relief(resolution='30s', 
                                                     region=region, 
                                                     registration=None, 
                                                     data_source='igpp', 
                                                     use_srtm=False) #30s
    
        #pyGMT basemap with topography figure
        fig = pygmt.Figure()
        pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", 
                     MAP_FRAME_PEN='1p') #Formatting , MAP_FRAME_AXES='lrbt',
        #pygmt.config(FORMAT_GEO_MAP="ddd.x",  MAP_FRAME_PEN='1p') #Formatting
    
        #Add topography basemap (DEM)
        fig.basemap(frame=True, region=region_rect, projection=projection)
    #fig.coast(dcw="US.AK+p0.25p")
    
    #Define outline and color pallete of basemap
        #fig.coast( shorelines=True, water='#C6E2EE', borders="1/1p,black") #frame=[fig_title]
        dgrid = pygmt.grdgradient(grid=load_grid, radiance=[270,30])
        pygmt.makecpt(cmap=tmp_cpt_path)
    #pygmt.makecpt(cmap=CPT_Option)  #, series=[-1.5, 0.3, 0.01])
   
        fig.grdimage(grid=load_grid, shading='+a300+nt0.8', cmap=True, 
                     transparency = 60)

        fig.coast( water=None, borders="1/0.5p,black", shorelines="1/0.5p,black")

        #pygmt.makecpt(cmap='polar', series = [-50,50])
        cpt_file = './green-purple.cpt'
        pygmt.makecpt(cmap=cpt_file, series = [-0.12,0.12])#red2green
        
        fig.plot(x= earthquake_lons, y= earthquake_lats, 
                 size=amplitude*(1.8**(earthquake_mags/np.mean(earthquake_mags))),
                 style="cc", pen='0.5p,#3e000d',cmap=True, fill = slow_error)
        
                ###Plot slowness error
        fig.plot(x=pos_error['lon'],
                y=pos_error['lat'],
                direction = [90*np.ones(len(pos_error)),40*pos_error['slow_error']],
                #direction = [baz_array_pygmt, [length]],
                style="v0.5c+ea",
                fill = "purple1",
                #fill="royalblue",
                pen="1.0p")
                #label = "Array backazimuth abs. power")
        
        fig.plot(x=neg_error['lon'],
                y=neg_error['lat'],
                direction = [90*np.ones(len(neg_error)),40*neg_error['slow_error']],
                #direction = [baz_array_pygmt, [length]],
                style="v0.5c+ea",
                #fill = "cyan4",
                fill="green2",
                pen="1.0p")
        
        fig.plot(x = array_lon,
                 y = array_lat,
                 style = "i1c",pen = '0.5p,#3e000d', size = [600], fill = 'gold1')
        
        #Example reference length----
        #fig.plot(x=-162,
                #y=51,
               # direction = [90, 20*0.05],
                #direction = [baz_array_pygmt, [length]],
                #style="v0.5c+ea",
                #fill = "red3",
                #fill="royalblue",
                #pen="1.0p")
    
        
        #Plot text---------------------------------------------
        #fig.text(text=array_names, x=array_lons, y=np.array(array_lats)+0.2,
                 #font = "18p,Helvetica-Bold,black") #fill = 'whitesmoke')

    
    
        fig.colorbar(frame="xaf+lSlowness error (s/km)")
        
        if save == True:
            fig.savefig(path, transparent=True, dpi=720)
            
        fig.show(dpi=720)




def intersect_beams(lat1, lon1, baz1, lat2, lon2, baz2):
    
    geod = Geod(ellps="WGS84")
    az1 = baz1
    az2 = baz2
    
    def to_cart(lat, lon):
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
        return np.array([
            np.cos(lat)*np.cos(lon),
            np.cos(lat)*np.sin(lon),
            np.sin(lat)
        ])
    
    # Station points
    p1 = to_cart(lat1, lon1)
    p2 = to_cart(lat2, lon2)
    
    # Second points slightly along azimuth
    lon1b, lat1b, _ = geod.fwd(lon1, lat1, az1, 1000)
    lon2b, lat2b, _ = geod.fwd(lon2, lat2, az2, 1000)
    
    p1b = to_cart(lat1b, lon1b)
    p2b = to_cart(lat2b, lon2b)
    
    # Great circle normals
    n1 = np.cross(p1, p1b)
    n2 = np.cross(p2, p2b)
    
    # Intersection line
    intersection = np.cross(n1, n2)
    intersection /= np.linalg.norm(intersection)
    
    # Two antipodal solutions
    i1 = intersection
    i2 = -intersection
    
    def to_latlon(vec):
        lat = np.rad2deg(np.arcsin(vec[2]))
        lon = np.rad2deg(np.arctan2(vec[1], vec[0]))
        return lat, lon
    
    return to_latlon(i1), to_latlon(i2)




def pygmt_single_event(index, array_lats, array_lons, earthquake_lats, 
                       earthquake_lons, earthquake_mags, earthquake_depths,
                       real_bazs_array1, array1_bazs, real_bazs_array2, 
                       array2_bazs, baz_conf, plot_real = True, save = False,
                       path = None):
    
    
    earthquake_mag = earthquake_mags[index]
    earthquake_lon = earthquake_lons[index]
    earthquake_lat = earthquake_lats[index]
    earthquake_depth = earthquake_depths[index]
    #print(earthquake_depth)
    lengths = 20
    
    real_baz = real_bazs_array1[index]
    array_baz = array1_bazs[index]
    baz_real_pygmt = 360 - transform_degrees_pygmt(real_baz)
    baz_array_pygmt = 360 - transform_degrees_pygmt(array_baz)
    vec_lats = array_lats[0]
    vec_lons = array_lons[0]

    #Set up real vector---------------------
    real_vec = np.column_stack([vec_lons,vec_lats, baz_real_pygmt,lengths])
    
    #Set up array vectors----------
    array_vec = np.column_stack([vec_lons,vec_lats, baz_array_pygmt,lengths])
    array_vec_conf1 = np.column_stack([vec_lons,vec_lats, baz_array_pygmt-baz_conf,lengths])
    array_vec_conf2 = np.column_stack([vec_lons,vec_lats, baz_array_pygmt+baz_conf,lengths])
    

    if len(real_bazs_array2) > 0: 
        real_baz = real_bazs_array2[index]
        array_baz = array2_bazs[index]
        baz_real_pygmt = 360 - transform_degrees_pygmt(real_baz)
        baz_array_pygmt = 360 - transform_degrees_pygmt(array_baz)

        vec_lats = array_lats[1]
        vec_lons = array_lons[1]
    
        #Set up real vector---------------------
        real_vec2 = np.column_stack([vec_lons,vec_lats, baz_real_pygmt,lengths])
        
        #Set up array vectors----------
        array_vec2 = np.column_stack([vec_lons,vec_lats, baz_array_pygmt,lengths])
        array_vec2_conf1 = np.column_stack([vec_lons,vec_lats, baz_array_pygmt-baz_conf,lengths])
        array_vec2_conf2 = np.column_stack([vec_lons,vec_lats, baz_array_pygmt+baz_conf,lengths])

    
    
    

    #if len(array_lats) > 0: #more than one array
        #vec_lats = np.ones(len(baz_real_pygmt))*array_lats[0]
        #vec_lons = np.ones(len(baz_real_pygmt))*array_lons[0]

    if earthquake_depth > 100:
        color = '#4D0010'
    elif earthquake_depth < 35:
        color = 'firebrick'
    else:
        color = 'gold1'

    # DEFINE CPT BASED ON AEC BASEMAP
    tmp_cpt_path = basemap_cpt("AEC")

    pygmt.config(FORMAT_GEO_MAP="ddd.x") # Highlevel formatting (no ticks, no labels)

    #Define projection and grid map resolution (for BOTH maps)
    lons = earthquake_lons % 360

    left = np.min(lons)
    right = np.max(lons)
    bottom = np.min(earthquake_lats)
    top = np.max(earthquake_lats)

    region = [left-5, right+5, bottom-5, top+5]


    region_rect = str(left-0.5)+"/"+str(bottom-0.5)+"/"+str(right+0.5)+"/"+str(top+0.5)+"r"

    projection = "M"+str(array_lons[0])+"/"+str(array_lats[0])+"/12c"
    '''
    left = np.min(earthquake_lons)
    right = np.max(earthquake_lons)
    bottom = np.min(earthquake_lats)
    top = np.max(earthquake_lats)

    region = [left-5, right+5, bottom-5, top+5]

    #region=[-170,-140,50,68]
    #region_rect = str(left-0.5)+"/"+str(bottom-0.5)+"/"+str(right+0.5)+"/"+str(top+0.5)
    region_rect = str(left-0.5)+"/"+str(bottom-0.5)+"/"+str(right+0.5)+"/"+str(top+0.5)+"r"
    #region_rect = "-162/52/-142/64r"

    projection="M0/0/12c"
    '''
    amplitude = 0.3 #for plotting earthquakes
    
    #projection = f'S210/{hemisphere}/8i'

    run_topo = True
    ##---Begin basemap w/ only AK topography---##

    if run_topo == True:
    # Load topography
        load_grid = pygmt.datasets.load_earth_relief(resolution='30s', 
                                                     region=region, 
                                                     registration=None, 
                                                     data_source='igpp', 
                                                     use_srtm=False) #30s
    
        #pyGMT basemap with topography figure
        fig = pygmt.Figure()
        pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain",  
                     MAP_FRAME_PEN='1p') #Formatting, MAP_FRAME_AXES='lrbt',
       #pygmt.config(FORMAT_GEO_MAP="ddd.x",  MAP_FRAME_PEN='1p') #Formatting

        fig = pygmt.Figure()
        pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain",
                    MAP_FRAME_PEN='1p') #Formatting, MAP_FRAME_AXES='lrbt'
        #pygmt.config(FORMAT_GEO_MAP="ddd.x",  MAP_FRAME_PEN='1p') #Formatting
    
        #Add topography basemap (DEM)
        fig.basemap(frame=True, region=region_rect, projection=projection,
                    map_scale="jBR+w500k+o0.5c/0.5c+f+lkm")
    #fig.coast(dcw="US.AK+p0.25p")
    
    #Define outline and color pallete of basemap
        #fig.coast( shorelines=True, water='#C6E2EE', borders="1/1p,black") #frame=[fig_title]
        dgrid = pygmt.grdgradient(grid=load_grid, radiance=[270,30])
        pygmt.makecpt(cmap=tmp_cpt_path)
    #pygmt.makecpt(cmap=CPT_Option)  #, series=[-1.5, 0.3, 0.01])
   
        fig.grdimage(grid=load_grid, shading='+a300+nt0.8', cmap=True,
                    transparency = 100)
        fig.coast(water=None, borders="1/0.5p,black", 
                  shorelines="1/0.5p,black")
        
    
        #Add topography basemap (DEM)
        #fig.basemap(frame=True, region=region_rect, projection=projection)
    #fig.coast(dcw="US.AK+p0.25p")
    
    #Define outline and color pallete of basemap
       # fig.coast( shorelines=True, water='#C6E2EE', borders="1/1p,black") #frame=[fig_title]
        #dgrid = pygmt.grdgradient(grid=load_grid, radiance=[270,30])
        #pygmt.makecpt(cmap=tmp_cpt_path)
    #pygmt.makecpt(cmap=CPT_Option)  #, series=[-1.5, 0.3, 0.01])

    

        fig.grdimage(grid=load_grid, shading='+a300+nt0.8', cmap=True, 
                     transparency = 60)

        #pygmt.makecpt(cmap='polar', series = [-50,50])
        
        fig.plot(x= [earthquake_lon], y= [earthquake_lat], 
                 size=[amplitude*(1.8**(earthquake_mag/np.mean(earthquake_mags)))],
                 style="cc", pen='0.5p,#3e000d', fill = color)
        
        #Plot real vector-----
        if plot_real == True:
            fig.plot(data=real_vec, style = "v1.5c", 
                     fill = "black", pen = '1.2p,-')
            if len(real_bazs_array2) > 0:
                fig.plot(data=real_vec2, style = "v1.5c", 
                         fill = "black", pen = '1.2p,-')

        #Plot array vector and cone-----
        fig.plot(data=array_vec, style = "v1.5c", fill = "red", pen = '1.2p,-')
        fig.plot(data=array_vec_conf1, style = "v1.5c", 
                 fill = "red", pen = '1.2p,#CC0000')
        fig.plot(data=array_vec_conf2, style = "v1.5c", 
                 fill = "red", pen = '1.5p,#CC0000') # '#CC0000'

        #Plot array
        fig.plot(x = array_lons[0],
                 y = array_lats[0],
                 style = "i1c",pen = '0.5p,#3e000d', size = [500], 
                 fill = 'cyan4') #'#CC33CC'

        if len(real_bazs_array2) > 0:
            #Plot array vector and cone-----
            fig.plot(data=array_vec2, style = "v1.5c", 
                     fill = "red", pen = '1.2p,-')
            fig.plot(data=array_vec2_conf1, style = "v1.5c", 
                     fill = "red", pen = '1.2p,#0000FF')
            fig.plot(data=array_vec2_conf2, style = "v1.5c", 
                     fill = "red", pen = '1.5p,#0000FF')

            #Plot array
            fig.plot(x = array_lons[1],
                     y = array_lats[1],
                     style = "i1c",pen = '0.5p,#3e000d', 
                     size = [500], fill = '#0000FF') #'cyan4'

            point1, point2 = intersect_beams(array_lats[0], array_lons[0], 
                                             array1_bazs[index], array_lats[1],
                                               array_lons[1], 
                                               array2_bazs[index])

            dist1, az, baz = gps2dist_azimuth(point1[0], point1[1], 
                                              earthquake_lat, earthquake_lon)
            dist2, az, baz = gps2dist_azimuth(point2[0], point2[1], 
                                              earthquake_lat, earthquake_lon)
            min_dist = np.min([dist1,dist2])
            
            print('Distance error from intersecting beams:', min_dist/1000, 'km')

            fig.plot(x= [point1[1]], y= [point1[0]], size=[0.2],
                style="cc", pen='0.5p,#3e000d', fill = 'red')

            fig.plot(x= [point2[1]], y= [point2[0]], size=[0.2],
                style="cc", pen='0.5p,#3e000d', fill = 'red')
        
    
        
        #Plot text---------------------------------------------
        fig.text(text='M'+str(earthquake_mag)+', '+str(earthquake_depth)+' km',
                  x=(abs(left-right)/6)+left, y=top,
                 font = "15p,Helvetica-Bold,black") #fill = 'whitesmoke')

    
    
        #fig.colorbar(frame="xaf+lBackazimuth error (degrees)")
        #fig.savefig('/Users/cadequigley/Downloads/Research/hom_kod_earthquakes.png', transparent=True, dpi=720)

        if save == True:
            fig.savefig(path+'single_event.png', transparent=True, dpi=720)
        fig.show(dpi=720)



def pygmt_network_subarrays(array_lats, array_lons, earthquake_lat, 
                            earthquake_lon, earthquake_mag, earthquake_depth,
                              array_bazs, time_since_origin, element_lats, 
                              element_lons, save = False, path = None):

    '''
    array_lats: lats of subarrays
    array_lons: lon of subarrays
    earthquake_lat: lat of earthquake/event
    earthquake_lon: lon of earthquake
    earthquake_mag: mag of earthquake
    earthquake_depth: depth of earthquake
    array_bazs: bazs for each subarray
    time_since_origin: time since origin time of event (seconds)
    subarray_lists: all elements of subarray

    '''
    
    
    lengths = np.ones(len(array_lats))*1.5
    #lengths = lengths.tolist()
    sizes = np.ones(len(array_lats))*60
    sizes = sizes.tolist()
    sizes_sta = np.ones(len(element_lons))*20
    sizes_sta = sizes_sta.tolist()

    baz_array_pygmt = 360 - transform_degrees_pygmt(array_bazs)

    color = time_since_origin
    #Set up real vector---------------------
    data = np.column_stack([array_lons,array_lats,color,baz_array_pygmt,lengths])

    #data = np.column_stack([vec_lons,vec_lats, color, baz_real_pygmt,lengths])

    if earthquake_depth > 100:
        color = '#4D0010'
    elif earthquake_depth < 35:
        color = 'firebrick'
    else:
        color = 'gold1'

    # DEFINE CPT BASED ON AEC BASEMAP
    tmp_cpt_path = basemap_cpt("AEC")

    pygmt.config(FORMAT_GEO_MAP="ddd.x") # Highlevel formatting (no ticks, no labels)

    #Define projection and grid map resolution (for BOTH maps)
    if np.all(np.array(array_lons) < 0):
        left = np.min(array_lons)
        right = np.max(array_lons)
    else:
        arr = np.array(array_lons)
        positives = arr[arr > 0]

        if positives.size > 0:
            left = positives.min()

        negatives = arr[arr < 0]
        right = negatives.max()
            
    bottom = np.min(array_lats)
    top = np.max(array_lats)

    region = [left-5, right+5, bottom-5, top+5]

    #region=[-170,-140,50,68]
    #region_rect = str(left-0.5)+"/"+str(bottom-0.5)+"/"+str(right+0.5)+"/"+str(top+0.5)
    region_rect = str(left-0.5)+"/"+str(bottom-0.5)+"/"+str(right+0.5)+"/"+str(top+0.5)+"r"
    #region_rect = "-162/52/-142/64r"

    region = [148, 252, 40, 83]     #lat/lon curved DEM, needs to be oversized
    #region_rect = "178/45/248/67r"  #rectangular designation for plotted mat
    region_rect = "178/50/235/72r"
    #projection="M0/0/12c"
    projection = "M200/70/12c"
    
    #projection = 'S210/90/8i'
    
    amplitude = 0.3 #for plotting earthquakes
    
    #projection = f'S210/{hemisphere}/8i'

    run_topo = True
    ##---Begin basemap w/ only AK topography---##

    if run_topo == True:
    # Load topography
        load_grid = pygmt.datasets.load_earth_relief(resolution='02m', 
                                                     region=region, 
                                                     registration=None, 
                                                     data_source='igpp', 
                                                     use_srtm=False) #30s #30s
    
        #pyGMT basemap with topography figure
        fig = pygmt.Figure()
        pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain",  
                     MAP_FRAME_PEN='1p') #Formatting, MAP_FRAME_AXES='lrbt',
        #pygmt.config(FORMAT_GEO_MAP="ddd.x",  MAP_FRAME_PEN='1p') #Formatting
    
        #Add topography basemap (DEM)
        fig.basemap(frame=True, region=region_rect, projection=projection)
    #fig.coast(dcw="US.AK+p0.25p")
    
    #Define outline and color pallete of basemap
        fig.coast( shorelines=True, water='#C6E2EE', borders="1/1p,black") #frame=[fig_title]
        dgrid = pygmt.grdgradient(grid=load_grid, radiance=[270,30])
        pygmt.makecpt(cmap=tmp_cpt_path)
    #pygmt.makecpt(cmap=CPT_Option)  #, series=[-1.5, 0.3, 0.01])
   
        fig.grdimage(grid=load_grid, shading='+a300+nt0.8', 
                     cmap=True, transparency = 60)

        #pygmt.makecpt(cmap='polar', series = [-50,50])
        
        
        

         #Plot elements
        fig.plot(x = element_lons,
                 y = element_lats,
                 style = "t0.2c", pen = '0.5p,#3e000d', 
                 size = sizes_sta, fill = 'gray')

        #Plot vectors----------------
        pygmt.makecpt(cmap='plasma', series = [0,np.max(time_since_origin)] )
        fig.plot(data=data, style = "v0.5c+ea", 
                 fill = "+z", cmap=True, pen = '0.5p,+z')
        
        #Plot array
        fig.plot(x = array_lons,
                 y = array_lats,
                 style = "i0.5c",pen = '0.5p,#3e000d', 
                 size = sizes, cmap = True, fill = time_since_origin) 

        #sizes_sta = np.ones(len(element_lons))*0.5

        fig.plot(x= [earthquake_lon], y= [earthquake_lat], 
                 size=[amplitude*(1.8**(earthquake_mag/np.mean(earthquake_mag)))],
                 style="cc", pen='0.5p,#3e000d', fill = color)
       

        
    
        
        #Plot text---------------------------------------------
        fig.text(text='M'+str(earthquake_mag)+', '+str(earthquake_depth)+' km',
                 x=-172, y=top, #(abs(left-right)/6)+left
                 font = "15p,Helvetica-Bold,black") #fill = 'whitesmoke')

    
    
        fig.colorbar(frame="xaf+lTime since event origin (seconds)")
        #fig.savefig('/Users/cadequigley/Downloads/Research/hom_kod_earthquakes.png', transparent=True, dpi=720)
        
        if save == True:
            fig.savefig(path, transparent=True, dpi=720)
        fig.show(dpi=720)







def pygmt_baz_error_new(array_lat, array_lon, array_name, earthquake_lats, 
                    earthquake_lons, earthquake_mags, baz, baz_error, 
                    save=False, path = None):
    
    baz_real_pygmt = 360 - transform_degrees_pygmt(baz)
    vector_direction = []
    #error = comb['baz_error'].to_numpy()
    for i in range(len(baz_real_pygmt)):
        temp = baz_error[i]
        tempbaz = baz_real_pygmt[i]
        if temp > 0:
            wa = 90+tempbaz
        else:
            wa = tempbaz-90-180
        vector_direction.append(wa)
    
    
    vector_direction = np.array(vector_direction)
    
    new_vec = []
    for j in range(len(vector_direction)):
        if vector_direction[j] < 0:
            wa = vector_direction[j]+360
        else:
            wa = vector_direction[j]
        new_vec.append(wa)
    new_vec = np.array(new_vec)  
  
    
    
    
    df6 = pd.DataFrame(baz_error, columns = ['baz_error'])
    df6['vec_direction'] = new_vec
    df6['lat'] = earthquake_lats
    df6['lon'] = earthquake_lons
    df6['baz'] = baz
    df6['mags'] = earthquake_mags
    #pos_error = pd.DataFrame(df6[df6['baz_error']>= 0])
    #neg_error = pd.DataFrame(df6[df6['baz_error']<= 0])
    lengths = 0.06*df6['baz_error'].to_numpy()
    direction = df6['vec_direction'].to_numpy()
    vec_lats = df6['lat'].to_numpy()
    vec_lons = df6['lon'].to_numpy()
    color = df6['baz_error'].to_numpy()


    data = np.column_stack([vec_lons,vec_lats, color, direction,lengths])

    

    tmp_cpt_path = basemap_cpt("AEC")

    pygmt.config(FORMAT_GEO_MAP="ddd.x") # Highlevel formatting (no ticks, no labels)



    #Define projection and grid map resolution (for BOTH maps)
    lons = earthquake_lons % 360

    left = np.min(lons)
    right = np.max(lons)
    bottom = np.min(earthquake_lats)
    top = np.max(earthquake_lats)
    extend_region = 5 # 8, 5
    region = [left-extend_region, right+extend_region, bottom-extend_region, top+extend_region]

    extend_rect = 5 #0.5, 5
    region_rect = str(left-extend_rect)+"/"+str(bottom-extend_rect)+"/"+str(right+extend_rect)+"/"+str(top+extend_rect)+"r"

    projection = "M"+str(array_lon)+"/"+str(array_lat)+"/12c"

    amplitude = 0.2 #for plotting earthquakes
    
    #projection = f'S210/{hemisphere}/8i'

    run_topo = True
    

    if run_topo == True:
    # Load topography
        load_grid = pygmt.datasets.load_earth_relief(resolution= '30s', #'10m','30s' 
                                                     region=region, 
                                                     registration=None, 
                                                     data_source='igpp', 
                                                     use_srtm=False) #30s
    
        #pyGMT basemap with topography figure
        fig = pygmt.Figure()
        pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain",
                    MAP_FRAME_PEN='1p') #Formatting, MAP_FRAME_AXES='lrbt'
        #pygmt.config(FORMAT_GEO_MAP="ddd.x",  MAP_FRAME_PEN='1p') #Formatting
    
        #Add topography basemap (DEM)
        fig.basemap(frame=True, region=region_rect, projection=projection,
                    map_scale="jBR+w200k+o0.5c/0.5c+f+lkm")
    #fig.coast(dcw="US.AK+p0.25p")
    
    #Define outline and color pallete of basemap
        #fig.coast( shorelines=True, water='#C6E2EE', borders="1/1p,black") #frame=[fig_title]
        dgrid = pygmt.grdgradient(grid=load_grid, radiance=[270,30])
        pygmt.makecpt(cmap=tmp_cpt_path)
    #pygmt.makecpt(cmap=CPT_Option)  #, series=[-1.5, 0.3, 0.01])
   
        fig.grdimage(grid=load_grid, shading='+a300+nt0.8', cmap=True,
                    transparency = 60)
        fig.coast(water=None, borders="10/10p,black", 
                  shorelines="1/0.5p,black")

        pygmt.makecpt(cmap='polar', series = [-80,80])
        #pygmt.makecpt(cmap="SCM/lajolla", series=[0, 360])
        
        fig.plot(x= earthquake_lons, y= earthquake_lats, 
                 size=amplitude*(1.8**(earthquake_mags/np.mean(earthquake_mags))),
                 style="cc", pen='0.5p,#3e000d',cmap=True, fill = baz_error)#fill = baz_error)
        
        #pygmt.makecpt(cmap='plasma', series = [22,40] )
        fig.plot(data=data, style = "v0.5c+ea", fill = "+z", cmap=True, pen = '0.7p,+z') #0.5, 0.7
        
        fig.plot(x = array_lon,
                 y = array_lat,
                 style = "i1c",pen = '0.5p,#3e000d', size = [600], fill = 'gold1')
        
        #Plot vector for reference length---------
        fig.plot(x=-162,
                y=51,
                direction = [90, 0.06*20],
                #direction = [baz_real_pygmt, [length]],
                style="v0.5c+ea",
                fill = "red3",
                #fill="royalblue",
                pen="1.0p")
        
        
    
        
        #Plot text---------------------------------------------
        #fig.text(text=array_names, x=array_lons, y=np.array(array_lats)+0.2,
                 #font = "18p,Helvetica-Bold,black") #fill = 'whitesmoke')

    
    
        fig.colorbar(frame="xaf+lBackazimuth error (degrees)")
        if save == True:
            fig.savefig(path, transparent=True, dpi=720)
        #fig.savefig('/Users/cadequigley/Downloads/Research/hom_kod_earthquakes.png', transparent=True, dpi=720)
        fig.show(dpi=720)




def pygmt_slow_error_new(array_lat, array_lon, array_name, earthquake_lats, 
                     earthquake_lons, earthquake_mags, slow_error, 
                     save = False, path = None):
    
    vec_direction = []
    for i in range(len(slow_error)):
        if slow_error[i] > 0:
            vec_direction.append(90)
        else: 
            vec_direction.append(90)
    df6 = pd.DataFrame(slow_error, columns = ['slow_error'])
    df6['lat'] = earthquake_lats
    df6['lon'] = earthquake_lons
    df6['vec_direction'] = vec_direction
    lengths = 40*df6['slow_error'].to_numpy()
    direction = df6['vec_direction'].to_numpy()
    vec_lats = df6['lat'].to_numpy()
    vec_lons = df6['lon'].to_numpy()
    color = df6['slow_error'].to_numpy()


    #data = np.column_stack([vec_lons,vec_lats, direction,lengths]) #color,
    data = np.column_stack([vec_lons,vec_lats, color, direction,lengths])
    
    
    
    amplitude = 0.2 #for plotting earthquakes

    # DEFINE CPT BASED ON AEC BASEMAP
    tmp_cpt_path = basemap_cpt("AEC")


    pygmt.config(FORMAT_GEO_MAP="ddd.x") # Highlevel formatting (no ticks, no labels)



    #Define projection and grid map resolution (for BOTH maps)
    lons = earthquake_lons % 360

    left = np.min(lons)
    right = np.max(lons)
    bottom = np.min(earthquake_lats)
    top = np.max(earthquake_lats)

    extend_region = 5 #8, 5
    region = [left-extend_region, right+extend_region, bottom - extend_region, top + extend_region]

    extend_rect = 5 #5 (regional), 0.5 (local)
    region_rect = str(left-extend_rect)+"/"+str(bottom-extend_rect)+"/"+str(right+extend_rect)+"/"+str(top+extend_rect)+"r"

    projection = "M"+str(array_lon)+"/"+str(array_lat)+"/12c"

    #projection = f'S210/{hemisphere}/8i'

    run_topo = True
    ##---Begin basemap w/ only AK topography---##

    if run_topo == True:
    # Load topography
        load_grid = pygmt.datasets.load_earth_relief(resolution='30s', 
                                                     region=region, 
                                                     registration=None, 
                                                     data_source='igpp', 
                                                     use_srtm=False) #30s
    
        #pyGMT basemap with topography figure
        fig = pygmt.Figure()
        pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", 
                     MAP_FRAME_PEN='1p') #Formatting , MAP_FRAME_AXES='lrbt',
        #pygmt.config(FORMAT_GEO_MAP="ddd.x",  MAP_FRAME_PEN='1p') #Formatting
    
        #Add topography basemap (DEM)
        fig.basemap(frame=True, region=region_rect, projection=projection)
    #fig.coast(dcw="US.AK+p0.25p")
    
    #Define outline and color pallete of basemap
        #fig.coast( shorelines=True, water='#C6E2EE', borders="1/1p,black") #frame=[fig_title]
        dgrid = pygmt.grdgradient(grid=load_grid, radiance=[270,30])
        pygmt.makecpt(cmap=tmp_cpt_path)
    #pygmt.makecpt(cmap=CPT_Option)  #, series=[-1.5, 0.3, 0.01])
   
        fig.grdimage(grid=load_grid, shading='+a300+nt0.8', cmap=True, 
                     transparency = 60)
        

        fig.coast( water=None, borders="10/10p,black", shorelines="1/0.5p,black")

        #pygmt.makecpt(cmap='polar', series = [-50,50])
        cpt_file = './green-purple.cpt'
        pygmt.makecpt(cmap=cpt_file, series = [-0.07,0.07])#red2green, [-0.12,0.12]
        
        fig.plot(x= earthquake_lons, y= earthquake_lats, 
                 size=amplitude*(1.8**(earthquake_mags/np.mean(earthquake_mags))),
                 style="cc", pen='0.5p,#3e000d',cmap=True, fill = slow_error)
        
                ###Plot slowness error
        
        fig.plot(data=data, style = "v0.5c+ea", fill = "+z", cmap=True, pen = '0.7p,+z') #0.5, 0.7

        fig.plot(x = array_lon,
                 y = array_lat,
                 style = "i1c",pen = '0.5p,#3e000d', size = [600], fill = 'gold1')
        
        #Example reference length----
        #fig.plot(x=-162,
               # y=51,
                #direction = [90, 40*0.05],
                #direction = [baz_array_pygmt, [length]],
                #style="v0.5c+ea",
                #fill = "red3",
                #fill="royalblue",
                #pen="1.0p")
    
        
        #Plot text---------------------------------------------
        #fig.text(text=array_names, x=array_lons, y=np.array(array_lats)+0.2,
                 #font = "18p,Helvetica-Bold,black") #fill = 'whitesmoke')

    
    
        fig.colorbar(frame="xaf+lSlowness error (s/km)")
        
        if save == True:
            fig.savefig(path, transparent=True, dpi=720)
            
        fig.show(dpi=720)




def pygmt_single_event_new(index, array_lats, array_lons, earthquake_lats, 
                       earthquake_lons, earthquake_mags, earthquake_depths,
                       real_bazs_array1, array1_bazs, real_bazs_array2, 
                       array2_bazs, baz_conf, plot_real = True, save = False,
                       path = None):
    
    
    earthquake_mag = earthquake_mags[index]
    earthquake_lon = earthquake_lons[index]
    earthquake_lat = earthquake_lats[index]
    earthquake_depth = earthquake_depths[index]
    #print(earthquake_depth)
    length = 20
    
    real_baz = real_bazs_array1[index]
    array_baz = array1_bazs[index]
    baz_real_pygmt = 360 - transform_degrees_pygmt(real_baz)
    baz_array_pygmt = 360 - transform_degrees_pygmt(array_baz)
    vec_lats = array_lats[0]
    vec_lons = array_lons[0]

    #Set up real vector---------------------
    real_vec = np.column_stack([vec_lons,vec_lats, baz_real_pygmt,length])
    array_vec_conf1 = np.column_stack([vec_lons,vec_lats, baz_array_pygmt-baz_conf,length])
    array_vec_conf2 = np.column_stack([vec_lons,vec_lats, baz_array_pygmt+baz_conf,length])
    
    #Set up array vectors----------
    conf_ints = [10, 20, 30, -10, -20, -30]
    vector_lats = np.ones(len(conf_ints))*array_lats[0]
    vector_lons = np.ones(len(conf_ints))*array_lons[0]
    direction = baz_array_pygmt + np.array(conf_ints)
    colors = 2*np.abs(np.array(conf_ints))
    lengths = np.ones(len(conf_ints))*20
    array_vec = np.column_stack([vec_lons,vec_lats, baz_array_pygmt,length])
    data = np.column_stack([vector_lons,vector_lats, colors, direction, lengths])
    #data = np.column_stack([lons,lats, direction, lengths, color])
    
    

    if len(real_bazs_array2) > 0: 
        real_baz = real_bazs_array2[index]
        array_baz = array2_bazs[index]
        baz_real_pygmt = 360 - transform_degrees_pygmt(real_baz)
        baz_array_pygmt = 360 - transform_degrees_pygmt(array_baz)

        vec_lats = array_lats[1]
        vec_lons = array_lons[1]
    
        #Set up real vector---------------------
        real_vec2 = np.column_stack([vec_lons,vec_lats, baz_real_pygmt,lengths])
        
        #Set up array vectors----------
        array_vec2 = np.column_stack([vec_lons,vec_lats, baz_array_pygmt,lengths])
        array_vec2_conf1 = np.column_stack([vec_lons,vec_lats, baz_array_pygmt-baz_conf,lengths])
        array_vec2_conf2 = np.column_stack([vec_lons,vec_lats, baz_array_pygmt+baz_conf,lengths])

    
    
    

    #if len(array_lats) > 0: #more than one array
        #vec_lats = np.ones(len(baz_real_pygmt))*array_lats[0]
        #vec_lons = np.ones(len(baz_real_pygmt))*array_lons[0]

    if earthquake_depth > 100:
        color = '#4D0010'
    elif earthquake_depth < 35:
        color = 'firebrick'
    else:
        color = 'gold1'

    # DEFINE CPT BASED ON AEC BASEMAP
    tmp_cpt_path = basemap_cpt("AEC")

    pygmt.config(FORMAT_GEO_MAP="ddd.x") # Highlevel formatting (no ticks, no labels)

    #Define projection and grid map resolution (for BOTH maps)
    lons = earthquake_lons % 360

    left = np.min(lons)
    right = np.max(lons)
    bottom = np.min(earthquake_lats)
    top = np.max(earthquake_lats)

    #Custom points for figure plot---------
    #left = -172
    #right = -165
    #bottom = 51
    #top = 54.5

    region = [left-5, right+5, bottom-5, top+5]


    region_rect = str(left-0.5)+"/"+str(bottom-0.5)+"/"+str(right+0.5)+"/"+str(top+0.5)+"r"

    projection = "M"+str(array_lons[0])+"/"+str(array_lats[0])+"/12c"
    '''
    left = np.min(earthquake_lons)
    right = np.max(earthquake_lons)
    bottom = np.min(earthquake_lats)
    top = np.max(earthquake_lats)

    region = [left-5, right+5, bottom-5, top+5]

    #region=[-170,-140,50,68]
    #region_rect = str(left-0.5)+"/"+str(bottom-0.5)+"/"+str(right+0.5)+"/"+str(top+0.5)
    region_rect = str(left-0.5)+"/"+str(bottom-0.5)+"/"+str(right+0.5)+"/"+str(top+0.5)+"r"
    #region_rect = "-162/52/-142/64r"

    projection="M0/0/12c"
    '''
    amplitude = 0.3 #for plotting earthquakes
    
    #projection = f'S210/{hemisphere}/8i'

    run_topo = True
    ##---Begin basemap w/ only AK topography---##

    if run_topo == True:
    # Load topography
        load_grid = pygmt.datasets.load_earth_relief(resolution='30s', 
                                                     region=region, 
                                                     registration=None, 
                                                     data_source='igpp', 
                                                     use_srtm=False) #30s
    
        #pyGMT basemap with topography figure
        fig = pygmt.Figure()
        pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain",  
                     MAP_FRAME_PEN='1p') #Formatting, MAP_FRAME_AXES='lrbt',
       #pygmt.config(FORMAT_GEO_MAP="ddd.x",  MAP_FRAME_PEN='1p') #Formatting

        fig = pygmt.Figure()
        pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain",
                    MAP_FRAME_PEN='1p') #Formatting, MAP_FRAME_AXES='lrbt'
        #pygmt.config(FORMAT_GEO_MAP="ddd.x",  MAP_FRAME_PEN='1p') #Formatting
    
        #Add topography basemap (DEM)
        fig.basemap(frame=True, region=region_rect, projection=projection,
                    map_scale="jBR+w200k+o0.5c/0.5c+f+lkm")
    #fig.coast(dcw="US.AK+p0.25p")
    
    #Define outline and color pallete of basemap
        #fig.coast( shorelines=True, water='#C6E2EE', borders="1/1p,black") #frame=[fig_title]
        dgrid = pygmt.grdgradient(grid=load_grid, radiance=[270,30])
        pygmt.makecpt(cmap=tmp_cpt_path)
    #pygmt.makecpt(cmap=CPT_Option)  #, series=[-1.5, 0.3, 0.01])
   
        fig.grdimage(grid=load_grid, shading='+a300+nt0.8', cmap=True, #0.8
                    transparency = 100) #60
        fig.coast(water=None, borders="10/10p,black", 
                  shorelines="1/0.5p,black")
        
    
        #Add topography basemap (DEM)
        #fig.basemap(frame=True, region=region_rect, projection=projection)
    #fig.coast(dcw="US.AK+p0.25p")
    
    #Define outline and color pallete of basemap
       # fig.coast( shorelines=True, water='#C6E2EE', borders="1/1p,black") #frame=[fig_title]
        #dgrid = pygmt.grdgradient(grid=load_grid, radiance=[270,30])
        #pygmt.makecpt(cmap=tmp_cpt_path)
    #pygmt.makecpt(cmap=CPT_Option)  #, series=[-1.5, 0.3, 0.01])

    

        fig.grdimage(grid=load_grid, shading='+a300+nt0.8', cmap=True, 
                     transparency = 60)

        #pygmt.makecpt(cmap='polar', series = [-50,50])
        
        fig.plot(x= [earthquake_lon], y= [earthquake_lat], 
                 size=[amplitude*(1.8**(earthquake_mag/np.mean(earthquake_mags)))],
                 style="cc", pen='0.5p,#3e000d', fill = color)
        
        #Plot real vector-----
        if plot_real == True:
            fig.plot(data=real_vec, style = "v1.5c", 
                     fill = "black", pen = '1.2p,-')
            if len(real_bazs_array2) > 0:
                fig.plot(data=real_vec2, style = "v1.5c", 
                         fill = "black", pen = '1.2p,-')

        #Plot array vector and cone-----
        fig.plot(data=array_vec, style = "v1.5c", fill = "red", pen = '1.2p,-')
        
        pygmt.makecpt(cmap='inferno', series = [15,75] )

        #fig.plot(data=data, style = "v0.5c+ea", fill = "+z",  pen = '5p,+z', cmap=True ) #0.5, 0.7, "+z", fill = "+z",

        #pygmt.makecpt(cmap='inferno', series=[15,75], reverse = True)

        fig.plot(data=array_vec_conf1, style = "v1.5c", 
                     fill = "red", pen = '2p,#0000FF')
        fig.plot(data=array_vec_conf2, style = "v1.5c", 
                    fill = "red", pen = '2p,#0000FF')

        data = np.column_stack([vector_lons, vector_lats, direction, lengths, colors])

        pygmt.makecpt(cmap='inferno', series=[15, 75], reverse = True)

        for i in range(len(vector_lons)):
            row = data[i]
            fig.plot(
                data=[row[:-1]],          # lon, lat, direction, length
                style="v0.5c+ea",
                fill="+z",
                cmap=True,
                pen="3.0p,-,+z",
                zvalue=colors[i]
            )

        #Plot array
        fig.plot(x = array_lons[0],
                 y = array_lats[0],
                 style = "i1c",pen = '0.5p,#3e000d', size = [500], 
                 fill = 'cyan4') #'#CC33CC'

        #Handle multiple arrays----------------------
        if len(real_bazs_array2) > 0:
            #Plot array vector and cone-----
            fig.plot(data=array_vec2, style = "v1.5c", 
                     fill = "red", pen = '1.2p,-')
            fig.plot(data=array_vec2_conf1, style = "v1.5c", 
                     fill = "red", pen = '1.2p,#0000FF')
            fig.plot(data=array_vec2_conf2, style = "v1.5c", 
                     fill = "red", pen = '1.5p,#0000FF')

            #Plot array
            fig.plot(x = array_lons[1],
                     y = array_lats[1],
                     style = "i1c",pen = '0.5p,#3e000d', 
                     size = [500], fill = '#0000FF') #'cyan4'

            point1, point2 = intersect_beams(array_lats[0], array_lons[0], 
                                             array1_bazs[index], array_lats[1],
                                               array_lons[1], 
                                               array2_bazs[index])

            dist1, az, baz = gps2dist_azimuth(point1[0], point1[1], 
                                              earthquake_lat, earthquake_lon)
            dist2, az, baz = gps2dist_azimuth(point2[0], point2[1], 
                                              earthquake_lat, earthquake_lon)
            min_dist = np.min([dist1,dist2])
            
            print('Distance error from intersecting beams:', min_dist/1000, 'km')

            fig.plot(x= [point1[1]], y= [point1[0]], size=[0.2],
                style="cc", pen='0.5p,#3e000d', fill = 'red')

            fig.plot(x= [point2[1]], y= [point2[0]], size=[0.2],
                style="cc", pen='0.5p,#3e000d', fill = 'red')
        
    
        
        #Plot text---------------------------------------------
        fig.text(text='M'+str(earthquake_mag)+', '+str(earthquake_depth)+' km depth',
                  x=(abs(left-right)/5)+left, y=top, #6
                 font = "15p,Helvetica-Bold,black") #fill = 'whitesmoke')

    
    
        #fig.colorbar(frame="xaf+lBackazimuth error (degrees)")
        #fig.savefig('/Users/cadequigley/Downloads/Research/hom_kod_earthquakes.png', transparent=True, dpi=720)

        if save == True:
            fig.savefig(path+'single_event.png', transparent=True, dpi=720)
        fig.show(dpi=720)
