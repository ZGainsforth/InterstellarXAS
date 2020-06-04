import streamlit as st
import time
import os, sys, shutil
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import dustmaps
from dustmaps.planck import PlanckQuery
from dustmaps.bayestar import BayestarQuery
from astroquery.esa.xmm_newton import XMMNewton
import tarfile
import yaml
import requests
from rich.traceback import install; install()

''' 
# XMMMaster database Browser 

The XMMMaster database is maintained by HEASARC and is listed with other databases here:
https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3catindex.pl#MASTER%20CATALOG

The desription of the XMMMaster database is here:
https://heasarc.gsfc.nasa.gov/W3Browse/xmm-newton/xmmmaster.html

### Configuration:

'''

# Read the config.yaml
try:
    with open(os.path.join('..', 'Config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
except Exception as e:
    st.exception(e)
config['DataDirectory'] = st.text_input('Data directory for storing large files (e.g. HEASARC database.):', config['DataDirectory'])
# Finally save the yaml back to disk in case the user made changes.
with open(os.path.join('..', 'Config.yaml'), 'w') as f:
    yaml.dump(config, f)

# Make sure the data directory exists
if not os.path.exists(config['DataDirectory']):
    os.mkdir(config['DataDirectory'])

# Make sure the data directory for Fluxed RGS spectra exists
if not os.path.exists(os.path.join(config['DataDirectory'], 'XMMNewtonFluxed')):
    os.mkdir(os.path.join(config['DataDirectory'], 'XMMNewtonFluxed'))

# If the database is not already downloaded, then we need to download it.
if not os.path.exists(os.path.join(config['DataDirectory'], 'heasarc_xmmmaster.tdat')):
    heasarc_url = 'https://heasarc.gsfc.nasa.gov/FTP/heasarc/dbase/tdat_files/heasarc_xmmmaster.tdat.gz'
    heasarc_filesize = float(requests.head(heasarc_url, stream=True).headers['Content-Length'])
    # st.write(f'Downloading {heasarc_filesize} bytes.')

    bytesdownloaded = 0
    heasarc_downloadmessage = st.info(f'Downloading HEASARC XMMMaster database: {heasarc_filesize} bytes.')
    heasarc_downloadprogress = st.progress(0)

    with requests.get(heasarc_url, stream=True) as r:
        r.raise_for_status()
        with open(os.path.join(config['DataDirectory'], 'heasarc_xmmmaster.tdat'), 'wb') as f:
            for data in r.iter_content(chunk_size=8192):
                print(heasarc_filesize, len(data), bytesdownloaded)
                bytesdownloaded += len(data)
                f.write(data)
                heasarc_downloadprogress.progress(min(100, int(100*bytesdownloaded/heasarc_filesize)))

    heasarc_downloadmessage.success('Downloaded HEASARC XMMMaster database.')

# Read the XMMMaster database.
@st.cache
def parse_xray_database():
    # Load the file into memory
    xray_database = pd.read_table(os.path.join(config['DataDirectory'], 'heasarc_xmmmaster.tdat'), 
            skiprows=89, 
            names=['obsid', 'pno', 'name', 'ra', 'dec', 'lii', 'bii', 'time', 'end_time', 'duration', 'scheduled_duration', 'estimated_exposure', 'pi_title', 'pi_fname', 'pi_lname', 'sas_version', 'pps_flag', 'pps_version', 'odf_date', 'process_date', 'distribution_date', 'public_date', 'xmm_revolution', 'process_status', 'mos1_num', 'mos1_time', 'mos1_mode', 'mos2_num', 'mos2_time', 'mos2_mode', 'pn_num', 'pn_time', 'pn_mode', 'rgs1_num', 'rgs1_time', 'rgs1_mode', 'rgs2_num', 'rgs2_time', 'rgs2_mode', 'om_num', 'om_time', 'om_mode', 'data_in_heasarc', 'subject_category', 'status', 'class'], 
            index_col=False, delimiter='|')
    # Drop any records that don't have galactic coordinates.
    xray_database.dropna(subset=['lii'], inplace=True)
    xray_database = xray_database[xray_database['data_in_heasarc'] == 'Y']
    # Also make a version of galactic coordinates ranging from +/- 180 instead of 0-360.
    xray_database['lii_180'] = xray_database['lii'].astype(float).apply(lambda x: x if x < 180 else x-360.)
    return xray_database
xray_database = parse_xray_database()

# Apply filters if the user wants them.
xray_subset = xray_database.copy()

# # Filter by observatory.
# observatory_filter = st.sidebar.multiselect('Filter observatories:', xray_database['observatory'].unique().tolist(), ['CHANDRA'])
# xray_subset = xray_subset[xray_subset.observatory.isin(observatory_filter)]

# Filter by source name.
name_filter = st.sidebar.text_input('Filter by source name:', '')
if len(name_filter) > 0:
    xray_subset.dropna(subset=['name'], inplace=True)
    xray_subset = xray_subset[xray_subset.name.str.contains(name_filter, case=False)]

# Filter by specific query.
query_filter = st.sidebar.text_input('Filter by Pandas query string:', '')
if len(query_filter) > 0:
    xray_subset = xray_subset.query(query_filter)

# Filter by minimum RGS exposure.
min_exposure_filter = st.sidebar.slider('Minimum exposure time (log10(s)):', 0., 7., 1.)
st.sidebar.text(f'log10({min_exposure_filter}) = {10**min_exposure_filter:0.0f} seconds.')
xray_subset = xray_subset[xray_subset.rgs1_time > 10**min_exposure_filter]

@st.cache
def InitializeDustmap(config):
    # Fetch the planck data if we don't have it yet.
    if not os.path.exists(os.path.join(config['DataDirectory'], 'planck')):
        print('downloading planck data')
        from dustmaps.config import config as dustconfig
        dustconfig['data_dir'] = config['DataDirectory']
        from dustmaps.planck import fetch
        fetch()
    planck = PlanckQuery()
    return planck
    # # Fetch the bayestar data if we don't have it yet.
    # if not os.path.exists(os.path.join(config['DataDirectory'], 'bayestar')):
    #     print('downloading bayestar data')
    #     from dustmaps.config import config as dustconfig
    #     dustconfig['data_dir'] = config['DataDirectory']
    #     from dustmaps.bayestar import fetch
    #     fetch()
    # bayestar = BayestarQuery()
    # return planck, bayestar

@st.cache
def GetPlanckMap(planck=None, lmin=-180,lmax=180, bmin=-90,bmax=90, lbstep=1):
    l = np.arange(lmin,lmax,lbstep)[::-1]
    b = np.arange(bmin,bmax,lbstep)
    L,B = np.meshgrid(l,b)
    c = SkyCoord(L*u.deg, B*u.deg, frame='galactic')
    dust = planck(c)
    return dust, l,b

@st.cache(suppress_st_warning=True)
def GetBayestarMap3D(bayestar=None, lmin=0,lmax=360, bmin=-90,bmax=90, lbstep=5, distmin=10, distmax=10000, diststep=1000, viewparallax=1):
    planck_msg = st.text('Computing 3D planck map.')
    l = np.arange(lmin,lmax,lbstep)
    b = np.arange(bmin,bmax,lbstep)
    l,b = np.meshgrid(l,b)
    dist = np.arange(distmin,distmax,diststep)
    dust = []
    for d in dist:
        lparallax = np.arctan(viewparallax/d)
        planck_msg.text(lparallax)
        loffset = l + lparallax
        c = SkyCoord(loffset*u.deg, b*u.deg, distance=d*u.pc, frame='galactic')
        dust.append(bayestar(c))
    return dust

planck = InitializeDustmap(config)

NumObservationsToDraw = st.sidebar.selectbox('Max observations to plot:', [0,10,100,1000,10000,100000], 5)

dust, l, b = GetPlanckMap(planck)
fig_dust = px.imshow(np.log(dust), x=l, y=b, title=f'Observations plotted on Planck dust map')
fig_dust['layout']['yaxis'].update(title='Galactic latitude', autorange = True)
fig_dust['layout']['xaxis'].update(title='Galactic longitude', autorange='reversed')
plotsubset = xray_subset.head(NumObservationsToDraw)
fig_dust.add_trace(go.Scattergl(
    x=plotsubset['lii_180'], 
    y=plotsubset['bii'], 
    text=list(f'{r[0]}, {r.obsid}, {r.name}' for r in plotsubset.itertuples()),
    mode='markers', 
    marker=dict(color='darkgreen', size=10,opacity=0.5)))
st.plotly_chart(fig_dust)

''' ### Currently selected data: '''
# Show the table with all the sources.
st.markdown(f'Currently selected {len(xray_subset)} records of {len(xray_database)} from the XMMMaster database:')
st.dataframe(xray_subset)

# And a detail for a specific record in case data gets truncated.
detail_number = st.number_input('Get detailed information on record #: ', format='%d', value=xray_subset.index[0])
st.write(xray_subset.loc[detail_number])

# Download this observation ID if we haven't already done so.
obsidnumeric = int(xray_subset.loc[detail_number]['obsid'])
obsidstr = f"{obsidnumeric:010.0f}"
if not os.path.isfile(os.path.join(config['DataDirectory'], 'XMMNewtonFluxed', obsidstr+'.tar')):
    XMMNewton.download_data(obsidstr, filename=os.path.join(config['DataDirectory'], 'XMMNewtonFluxed', obsidstr+'.tar'), verbose=True, level='PPS', name='FLUXED')
    # ''' http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno=0824720101&level=PPS&instrument=RGS1&name=FLUXED '''

    # Now we can extract the gziped fits file (ftz) containing the spectrum and place it on the disk too.
    tar = tarfile.open(os.path.join(config['DataDirectory'], 'XMMNewtonFluxed', obsidstr+'.tar'))
    for tarinfo in tar:
        # if tarinfo.name == f'{obsidstr}/pps/P{obsidstr}RGX000FLUXED1001.FTZ':
        if '.FTZ' in tarinfo.name:
            st.write(f'Downloaded spectrum {tarinfo.name}, {tarinfo.size} bytes.')
            spectruminfo = tarinfo
    spectrumfile = tar.extractfile(spectruminfo)
    content = spectrumfile.read()
    with open(os.path.join(config['DataDirectory'], 'XMMNewtonFluxed', obsidstr+'.ftz'), 'wb') as f:
        f.write(content)

# Now we can plot it.
spectrumfits = fits.open(os.path.join(config['DataDirectory'], 'XMMNewtonFluxed', obsidstr+'.ftz'))
spec = spectrumfits[1]
fig_spec = px.line(x=spec.data.field('CHANNEL'), y=spec.data.field('FLUX'))
fig_spec['layout']['xaxis'].update(title=spec.header['TUNIT1'])
fig_spec['layout']['yaxis'].update(title=spec.header['TUNIT2'])
st.plotly_chart(fig_spec)

