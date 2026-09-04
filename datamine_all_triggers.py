#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from array_functions import (pull_earthquakes, data_from_inventory, 
                             get_geometry, trigger_list, triggers_associator)
from obspy import read_inventory
from obspy import UTCDateTime
from matplotlib.transforms import blended_transform_factory
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth

net = '4E'
sta = 'HM*'
loc = '*'
chan = 'DHZ'
client = Client('EARTHSCOPE')

##Trigger parameters--------------------

freq_min = 1#1
freq_max = 10#10
short_window = 0.05
long_window = 5
on_threshold = 15 #20
off_threshold = 5
moveout = 4 # seconds
min_triggers = 8

eq_time = UTCDateTime('2025-10-20T16:27:54.855000Z')
START = eq_time
END = START+60*60*4

st = client.get_waveforms(net, sta, loc, chan, START, END, attach_response=True)

st.merge(fill_value='latest')
st.remove_sensitivity()
st.filter("bandpass", freqmin=freq_min, freqmax=freq_max, 
          corners=2, zerophase=True)
st.sort()
# %%
trigger_lists = []
trigger_peaks = []
trigger_lengths = []
for s in range(len(st)):
    times, peaks, lengths = trigger_list(st[s], short_window, long_window,
                                              on_threshold, off_threshold)
    trigger_lists.append(times)
    trigger_peaks.append(peaks)
    trigger_lengths.append(lengths)


    # Associate triggers together based on expected moveout------------
            

times, peaks, lengths = triggers_associator(trigger_lists, 
                                            trigger_peaks, 
                                            trigger_lengths, 
                                            moveout, min_triggers)

def cluster_times(times, threshold=4.0):
    times = np.sort(times)
    groups = []
    current_group = [times[0]]

    for t in times[1:]:
        if t - current_group[-1] <= threshold:
            current_group.append(t)
        else:
            groups.append(current_group)
            current_group = [t]
    groups.append(current_group)

    return groups

groups = cluster_times(times)
collapsed = [np.mean(g) for g in groups]


# %%
fig, ax = plt.subplots()
for i in range(len(st)):
    tr = st[i]
    ax.plot(tr.times(), tr.data + 0.00000008*i, color = 'black', alpha = 0.6)
for k in range(len(collapsed)):
    ax.axvline(collapsed[k], color = 'red', alpha = 0.8, linestyle = '--')
#ax.set_xlim(50,55)
ax.set_ylim(0, 2*1e-6)
plt.show()
# %%
