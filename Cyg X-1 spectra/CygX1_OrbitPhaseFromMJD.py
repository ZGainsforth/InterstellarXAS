# based on Brocksopp et al. (1999)
# Compute the orbital phase of Cyg X-1.

import numpy as np
from astropy.io import fits
import sys
import pandas as pd

# if len(sys.argv) != 3:
#     print('Command format:\npython ChandraEphemeris.py filename\nFilename is the name of a CSV with start and end observation times.\nThe ephemeris times will be read from columns "start time mjd" and "end time mjd"  New columns will be made with the orbital start/end and phase.')
#     sys.exit()

# csv = pd.read_csv(sys.argv[1])
csv = pd.read_csv('Summary.csv')

# Observation start and end in MJD.
mjd1 = csv[' start time mjd']# Start
mjd2 = csv['   end time mjd'] # End
mjd = np.array([mjd1,mjd2])

# This is an example of some real values from an observation so it can be checked.
# # Ephemeris times for which you want to compute the phase.
# mjd1 = 55210.118  # start of Chandra ObsID 11044
# mjd2 = 55210.495  # stop of Chandra ObsID 11044
# mjd = np.array([mjd1,mjd2])

# orbital period with errors.
Porb = 5.599829
Porb_err = 0.000016

# zero phase.
# This is superior conjunction of the black hole (meaning the black hole is furthest from us)
# TzeroJD = 41874.707 # time of superior conjunction (Brocksopp)
# Tzero = TzeroJD-0.5 # conversion from RJD to MJD
#Tzero = 50077.99 # time of superior conjunction (bodaghee, derived from Brocksopp99)
#Tzero = 51729.949 # time of superior conjunction (Gies03)
#Tzero = 52872.288 # time of superior conjunction (Gies08)

# Choose Tzero to be the time which is closest but lesser than the start of the observation.
TzeroTimes = np.array([41874.207, 50077.99, 51729.949, 52872.288])
Tzero = np.array([TzeroTimes[np.where(t1 >= TzeroTimes)[0][-1]] for t1 in mjd1])

# The computation
ncycles = (mjd-Tzero)/Porb
ncycles_err = ncycles*Porb_err/Porb

csv['Tzero'] = Tzero
# print('Time at start of observation: %0.3f' % (mjd1))
# print('Time at end of observation: %0.3f' % (mjd2))
# print('Using TZero = %0.3f' % Tzero)

csv['mjd-T0 at start'] = ncycles[0,:]
csv['mjd-T0 at start (err)'] = ncycles_err[0,:]
csv['mjd-T0 at end'] = ncycles[1,:]
csv['mjd-T0 at end (err)'] = ncycles_err[1,:]
# print('Orbital location at start of observation: %g +/- %g' % (ncycles[0], ncycles_err[0]))
# print('Orbital location at end of observation: %g +/- %g' % (ncycles[1], ncycles_err[1]))

csv['Orbit phase at start'] = np.modf(ncycles[0,:])[0]
csv['Orbit phase at start (err)'] = ncycles_err[0,:]
csv['Orbit phase at end'] = np.modf(ncycles[1,:])[0]
csv['Orbit phase at end (err)'] = ncycles_err[1,:]
# print('Orbital phase at start of observation: %g +/- %g' % (np.modf(ncycles[0])[0], ncycles_err[0]))
# print('Orbital phase at end of observation: %g +/- %g' % (np.modf(ncycles[1])[0], ncycles_err[1]))

# print('A phase of 0 or 1 means the black hole is behind the star.')
# print('A phase of 0.5 means the black hole is closer than the star.')

HeaderNote = '# A phase of 0 or 1 means the black hole is behind the star.\n'
HeaderNote += '# A phase of 0.5 means the black hole is closer than the star.\n'

with open('Summary with orbit.csv', 'w') as f:
    f.write(HeaderNote)
    csv.to_csv(f)
