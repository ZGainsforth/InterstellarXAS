from astropy. io import fits
import matplotlib.pyplot as plt
from astroquery.simbad import Simbad
# from astroquery.esa.xmm_newton import XMMNewton

targetname = 'Serpens X-1'

Simbad.add_votable_fields('distance')
target = Simbad.query_object(targetname)

# XMMNewton.download_data('0084020401', level='PPS', extension='FTZ', name='FLUXED', filename='result0084020401.tar.gz')


print(f'Distance to {targetname} = {target["Distance_distance"][0]} {target["Distance_unit"][0]}')

x = fits.open('P0084020401RGX000FLUXED1025.ftz')

print(x.info())

plt.plot(x[1].data.field('CHANNEL'), x[1].data.field('FLUX'))
plt.xlabel(x[1].header['TUNIT1'])
plt.ylabel(x[1].header['TUNIT2'])
plt.show()
