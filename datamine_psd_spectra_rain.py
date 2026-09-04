#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

#------------------------
#---Plot rain data-------
array = 'HM'
station = 'HM09'
min_freq = 1
vmin = -151
vmax = -119

df = pd.read_csv('./nws_daily_precipitation.csv')
date_list = df['date'].to_list()
dates = [datetime.strptime(d, "%Y-%m-%d") for d in date_list]

# %%
fig, ax = plt.subplots(figsize = (14,6))

ax.bar(dates, df[array+'_precip_cm'])

ax.grid(alpha = 0.3)
ax.set_xlabel('Date')
ax.set_ylabel('precipitation (cm)')
fig.autofmt_xdate()
ax.set_ylim(0,3.25)

plt.show()

figpath = '/Users/cadequigley/Downloads/Research/deployment_array_design/datamine_paper_figures/'

df = pd.read_csv(figpath+'hm_psds_all_dates.csv')
df = pd.DataFrame(df[df['station']== station ])
df = pd.DataFrame(df[df['frequency']>= min_freq ])
date_list = df['time'].to_list()
dates = [datetime.strptime(d, "%Y-%m-%d") for d in date_list]
df['date'] = dates
#%%

# df has columns: date, frequency, power (in dB)

# 1. Pivot to a 2D grid: rows = frequency, columns = date
pivot = df.pivot_table(index='frequency', columns='date', values='median_power')
pivot = pivot.sort_index()  # ensure frequency is ascending

freqs = pivot.index.values          # 1D array, length = n_freq
dates = pivot.columns.values        # 1D array, length = n_time
Z = pivot.values                    # shape (n_freq, n_time)

# 2. Plot
fig, ax = plt.subplots(figsize=(14, 6))

mesh = ax.pcolormesh(
    dates, freqs, Z,
    shading='auto',      # or 'nearest'/'gouraud'; avoids the meshgrid dimension headache
    cmap='plasma', # closest built-in to that PQLX-style rainbow; see note below
    vmin=vmin, vmax=vmax
)

ax.set_yscale('log')
ax.set_ylabel('Frequency (Hz)')
ax.grid(color = 'white', alpha = 0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
fig.autofmt_xdate()

#cbar = fig.colorbar(mesh, ax=ax, label='Power [dB]')

plt.tight_layout()
plt.show()

#%%
#-------------------------
#----Combined plot--------
# ------------------------

# --- Load precip data ---
array = 'KD'
array2 = 'kd'
station = 'KD04' #HM07 River, HM04 figure
min_freq = 0.001
max_freq = 120
vmin = -160 #151
vmax = -119

precip_df = pd.read_csv('./nws_daily_precipitation.csv')
precip_df['date'] = pd.to_datetime(precip_df['date'], format="%Y-%m-%d")

# ------------------------
# --- Load PSD data -------
figpath = '/Users/cadequigley/Downloads/Research/deployment_array_design/datamine_paper_figures/'
psd_df = pd.read_csv(figpath + array2+'_psds_all_dates.csv')
psd_df = psd_df[psd_df['station'] == station]
psd_df = psd_df[psd_df['frequency'] >= min_freq]
psd_df = psd_df[psd_df['frequency'] <= max_freq]
psd_df['date'] = pd.to_datetime(psd_df['time'], format="%Y-%m-%d")

pivot = psd_df.pivot_table(index='frequency', columns='date', values='median_power')
pivot = pivot.sort_index()

# ------------------------
# --- Build a common daily date range and reindex BOTH datasets onto it ---
full_range = pd.date_range(
    start=min(psd_df['date'].min(), pivot.columns.min()),
    end=max(psd_df['date'].max(), pivot.columns.max()),
    freq='D'
)

# Reindex precip to the full range, filling missing days with 0
precip_df = precip_df.set_index('date').reindex(full_range)
precip_df.index.name = 'date'

# Reindex the PSD pivot's columns to the full range; missing days become NaN
# (NaN cells render as transparent/blank in pcolormesh, which is fine)
pivot = pivot.reindex(columns=full_range)

freqs = pivot.index.values
psd_dates = pivot.columns.values
Z = pivot.values

# ------------------------
# --- Combined figure -----
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.05)

ax_precip = fig.add_subplot(gs[0])
ax_psd = fig.add_subplot(gs[1], sharex=ax_precip)

# Top: precipitation bars
ax_precip.bar(precip_df.index, precip_df[array + '_precip_cm'], color='steelblue', width=1.0)
ax_precip.grid(alpha=0.3)
ax_precip.set_ylabel('daily precip (cm)')
ax_precip.set_ylim(0, 3.25)
plt.setp(ax_precip.get_xticklabels(), visible=False)

# Bottom: spectrogram
mesh = ax_psd.pcolormesh(
    psd_dates, freqs, Z,
    shading='auto',
    cmap='plasma',
    vmin=vmin, vmax=vmax
)
ax_psd.set_yscale('log')
ax_psd.set_ylabel('Frequency (Hz)')
ax_psd.set_xlabel('Date')
ax_psd.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

# Explicitly force both axes to the exact same limits, matching the full range
ax_psd.set_xlim(full_range[0], full_range[-1])
ax_precip.set_xlim(full_range[0], full_range[-1])
ax_psd.grid(color = 'white', alpha = 0.3)
##cbar = fig.colorbar(mesh, ax=ax_psd, pad=0.01, label='Power [dB]')

fig.canvas.draw()
pos_psd = ax_psd.get_position()
pos_precip = ax_precip.get_position()
ax_precip.set_position([pos_psd.x0, pos_precip.y0, pos_psd.width, pos_precip.height])

fig.autofmt_xdate()
#plt.savefig(figpath+'/hm04_rain_spectra.png', transparent=True, dpi= 720, 
            #bbox_inches='tight', pad_inches=0.1)
plt.show()


# %%
#-------------------------
#----Plot heliocorder-----


from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy import Stream
import matplotlib.pyplot as plt

net = '4E'
sta = 'KD04' #HM07, HM26, HM04 (figure), HM09
loc = '*'
chan = 'DHZ'
length = 24  # hours
interval = 30  # minutes, 20
global_start = UTCDateTime('2025-10-15T00') #'2025-10-05T00', '2025-10-31T00', '2025-10-15T00'

starts = [global_start]
for i in range(int(length*60/interval)):
    global_start = global_start + interval*60
    starts.append(global_start)

client = Client('EARTHSCOPE')

#colors = ['tab:black', 'tab:firebrick', 'tab:orange']  # rotates every 3rd line
colors = ['black', 'firebrick', 'orange']
colors = ['black', 'royalblue', 'teal']
fig, ax = plt.subplots(figsize=(10, 0.3*len(starts)))

datas = []
times = []
offset_step = 0.0000005  # vertical spacing between traces
yticks = []
yticklabels = []

for k in range(len(starts)):
    START = starts[k]
    END = START + interval*60
    st = client.get_waveforms(net, sta, loc, chan, START, END, attach_response=True)

    st.merge(fill_value='latest')
    st.remove_sensitivity()
    time = st[0].times()/60
    data = st[0].data
    datas.append(data)
    times.append(time)

    offset = -offset_step*k
    color = colors[k % len(colors)]
    ax.plot(time, data + offset, color=color)
    if START.minute == 0 and START.second == 0:
        yticks.append(offset)
        yticklabels.append(START.strftime('%Y-%m-%d %H:%M'))
    #yticks.append(offset)
    #yticklabels.append(START.strftime('%Y-%m-%d %H:%M'))

ax.set_xlabel('minutes')
ax.set_xlim(0, interval)
ax.set_ylim(-0.0000005*(len(starts)),0.0000005*1)
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.grid(alpha=0.3)
ax.margins(y=0.01)
#plt.savefig(figpath+'/hm04_helicorder.png', transparent=True, dpi= 720, 
            #bbox_inches='tight', pad_inches=0.1)
plt.show()
# %%


