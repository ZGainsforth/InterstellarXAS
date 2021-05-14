# based on Brocksopp et al. (1999)
# Compute the orbital phase of Cyg X-1.

import numpy as np
from astropy.io import fits
import sys
import pandas as pd
import matplotlib.pyplot as plt

csv = pd.read_csv('Summary with orbit.csv', skiprows=2)
csv = csv.sort_values(by='Orbit phase at start')
print(csv)
# print(csv[['        obsid', 'Orbit phase at start']])

fig = plt.figure()
for i, r in csv.iterrows(): #[['        obsid', 'Orbit phase at start']]:
    # print(r['        obsid'], r['Orbit phase at start'])
    obsid = r['        obsid']
    phase = r['Orbit phase at start']
    print(obsid, phase)
    x = np.genfromtxt(str(obsid) + '.csv', skip_header=1)
    x = x[539:730, :]
    x[:,1] -= np.min(x[:,1])
    x[:,1] /= np.max(x[:,1])
    plt.plot(x[:,0], x[:,1], label=obsid) #, color=(2*np.abs(phase-0.5), 2*np.abs(phase-0.5), 1-2*np.abs(phase-0.5)))
plt.legend()
plt.show()
