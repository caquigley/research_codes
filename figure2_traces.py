#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime, read


drop_taup = True
path = "./POM_earthquakes_mseeds/"
freq_min = 1
freq_max = 10

#df = pd.read_csv('/Users/cadequigley/repos/array_aggregator/2A_1000km_m3_lts__window_freq_map_fig.csv')
#df = pd.read_csv('/Users/cadequigley/repos/array_aggregator/3A_1000km_m3_lts__fig5.csv')
df = pd.read_csv('/Users/cadequigley/repos/array_aggregator/POM_1000km_m3_lts__fig5.csv')
df = pd.DataFrame(df[df['distance']<= 400])
print('Number of events:', len(df))

if drop_taup ==True:
    temp = pd.DataFrame(df[df['trigger_type']!= 'Taup'])
    print('Number of dropped events for Taup:', len(df) - len(temp))
    df = temp

trigger_times = df['trigger_time'].to_numpy()
event_ids = df['event_id'].to_numpy()


fig, ax = plt.subplots(figsize = (8,4))

for i in range(len(df)):
    event_id = event_ids[i]
    trigger_time = trigger_times[i]
    START = UTCDateTime(trigger_time) - 4
    END = START + 12

    st = read(path+event_id+'.mseed')
    st.filter("bandpass", freqmin=freq_min, freqmax=freq_max, 
                corners=2, zerophase=True)
    st = st.slice(START, END)
    
    tr = st[0]
    times = tr.times()-4
    data = tr.data/(np.max(abs(tr.data)))
    data = data - np.mean(data)
    ax.plot(times, data, color = 'black', alpha = 0.05)

ax.set_xlabel('time relative to p-pick (seconds)')
ax.set_ylabel('normalized counts')
ax.grid(alpha=0.3)
ax.set_xlim(-4, 8)
ax.axvline(x=0, color='red', linestyle='--')

#fig.savefig('/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/figure_components/supp_fig_POM_traces.png', transparent=True, dpi=720)
plt.show()
#%%
###############################
#Quantiles figure--------------
###############################

#fk = pd.read_csv('/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/fk_time_quantiles.csv')
#ls = pd.read_csv('/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/ls_time_quantiles.csv')
#lts = pd.read_csv('/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/lts_time_quantiles.csv')

#fk = pd.read_csv('/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/2A_fk_slow_time_quantiles.csv')
#ls = pd.read_csv('/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/2A_ls_slow_time_quantiles.csv')
#lts = pd.read_csv('/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/2A_lts_slow_time_quantiles.csv')

fk = pd.read_csv('/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/POM_fk_baz_time_quantiles.csv')
ls = pd.read_csv('/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/POM_ls_baz_time_quantiles.csv')
lts = pd.read_csv('/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/POM_lts_baz_time_quantiles.csv')

fig, ax = plt.subplots(figsize = (8,4))

#Plot FK results-------------------------------------------------------------
ax.plot(fk['time_bins'], fk['quantiles'], color ='gray', alpha = 0.5)
ax.scatter(fk['time_bins'], fk['quantiles'], color = 'skyblue',
           edgecolors = 'black', linewidths = 0.5, s = 50, label = 'FK')


#Plot LS results------------------------------------------------------------
ax.plot(ls['time_bins'], ls['quantiles'], color ='gray', alpha = 0.5)
ax.scatter(ls['time_bins'], ls['quantiles'], color = 'firebrick',marker = '^',edgecolors = 'black', 
           linewidths = 0.5, s = 80, label = 'LS')



#Plot LTS results------------------------------------------------------------------

ax.plot(lts['time_bins'], lts['quantiles'], color ='gray', alpha = 0.5)
ax.scatter(lts['time_bins'], lts['quantiles'], color = 'orange',marker = 'D',edgecolors = 'black', 
           linewidths = 0.5, s = 50, label = 'LTS')


ax.grid(alpha = 0.3)
ax.set_xlim(-4,8)
ax.set_ylim(0,350)
#ax.set_ylim(0, 0.35)
ax.set_xlabel('time relative to p-pick (seconds)')
ax.set_ylabel('90% quantile range')
#ax.set_title("2A quantiles over time")
#ax.text(0.5, 0.95, 'EPIC Pick', ha='left', va='top', transform=ax.transAxes, fontweight='bold', color='purple')
ax.axvline(x=0, color='red', linestyle='--')


#plt.legend()
plt.legend(loc='lower center', bbox_to_anchor=(0.1, 0.2), ncol=1) # ncol for multiple columns
plt.savefig('/Users/cadequigley/Downloads/Research/unalaska_arrays_paper/figure_components/supp_fig_POM_slow_quantiles.png', transparent=True, dpi = 720)
plt.show()
# %%
