#%%
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy import Stream
import matplotlib.pyplot as plt

figpath = '/Users/cadequigley/Downloads/Research/deployment_array_design/'

net = '4E'
sta = 'HM06'
loc = '*'
channels = ['DHZ','DHN','DHE']

#START = UTCDateTime('2025-09-09T17:36:00')
#END = UTCDateTime('2025-09-09T17:37:00')
START = UTCDateTime('2025-10-03T05:38:30')
END = UTCDateTime('2025-10-03T05:40:30')
client = Client('EARTHSCOPE')
st = Stream()
for chan in channels:
    st += client.get_waveforms(net, sta, loc, chan, START, END, attach_response = True)

st.merge(fill_value='latest')
#st.trim(START, END, pad='true', fill_value=0)
st.sort()
print(st)

print('Removing sensitivity...')
st.remove_sensitivity()
# %%
fig, ax = plt.subplots()

for i in range(len(st)):
    tr = st[i]
    ax.plot(tr.times("matplotlib"), 0.0011*i+tr.data, color = 'black', alpha = 0.8)
ax.grid(alpha = 0.3)
ax.xaxis_date()
fig.autofmt_xdate()
plt.yticks([])
#plt.savefig(figpath+'/datamine_paper_figures/hm06_bear_trace.png', transparent=True, dpi = 720)
plt.show()

# %%
