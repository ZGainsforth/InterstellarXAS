import streamlit as st
import time
import os, sys, shutil
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import astropy.units as u
from astropy.coordinates import SkyCoord
import dustmaps
from dustmaps.planck import PlanckQuery
from dustmaps.bayestar import BayestarQuery
import yaml
import requests

''' 
# Master X-ray database Browser 

The Master X-ray database is maintained by HEASARC and is available here:
https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3catindex.pl#MASTER%20CATALOG

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

# If the database is not already downloaded, then we need to download it.
if not os.path.exists(os.path.join(config['DataDirectory'], 'heasarc_xray.tdat')):
    heasarc_url = 'https://heasarc.gsfc.nasa.gov/FTP/heasarc/dbase/tdat_files/heasarc_xray.tdat.gz'
    heasarc_filesize = float(requests.head(heasarc_url, stream=True).headers['Content-Length'])
    # st.write(f'Downloading {heasarc_filesize} bytes.')

    bytesdownloaded = 0
    heasarc_downloadmessage = st.info(f'Downloading HEASARC Master X-ray database: {heasarc_filesize} bytes.')
    heasarc_downloadprogress = st.progress(0)

    with requests.get(heasarc_url, stream=True) as r:
        r.raise_for_status()
        with open(os.path.join(config['DataDirectory'], 'heasarc_xray.tdat'), 'wb') as f:
            for data in r.iter_content(chunk_size=8192):
                print(heasarc_filesize, len(data), bytesdownloaded)
                bytesdownloaded += len(data)
                f.write(data)
                heasarc_downloadprogress.progress(min(100, int(100*bytesdownloaded/heasarc_filesize)))

    heasarc_downloadmessage.success('Downloaded HEASARC Master X-ray database.')

# Read the Master X-ray database.
@st.cache
def parse_xray_database():
    # Load the file into memory
    xray_database = pd.read_table(os.path.join(config['DataDirectory'], 'heasarc_xray.tdat'), 
            skiprows=53, 
            names=['name', 'database_table', 'ra', 'dec', 'gal_l', 'gal_b', 'count_rate', 'count_rate_error', 'error_radius', 'flux', 'exposure', 'class', 'observatory'], 
            index_col=False, delimiter='|')
    # Drop any records that don't have a valid observatory.
    xray_database.dropna(subset=['observatory'], inplace=True)
    return xray_database
xray_database = parse_xray_database()

# Apply filters if the user wants them.
xray_subset = xray_database

# Filter by observatory.
observatory_filter = st.sidebar.multiselect('Filter observatories:', xray_database['observatory'].unique().tolist(), ['CHANDRA'])
xray_subset = xray_subset[xray_subset.observatory.isin(observatory_filter)]

# Filter by database table.
database_filter = st.sidebar.multiselect('Filter database:', xray_subset['database_table'].unique().tolist())
if len(database_filter) > 0:
    xray_subset = xray_subset[xray_subset.database_table.isin(database_filter)]

# Filter by class.
class_filter = st.sidebar.multiselect('Filter class:', xray_subset['class'].unique().tolist())
if len(class_filter) > 0:
    xray_subset = xray_subset[xray_subset['class'].isin(class_filter)]

# Filter by minimum exposure.
min_exposure_filter = st.sidebar.slider('Minimum exposure time (log10(s)):', 0., 7., 1.)
st.sidebar.text(f'log10({min_exposure_filter}) = {10**min_exposure_filter:0.0f} seconds.')
xray_subset = xray_subset[xray_subset.exposure > 10**min_exposure_filter]

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
    l = np.arange(lmin,lmax,lbstep)
    b = np.arange(bmin,bmax,lbstep)
    l,b = np.meshgrid(l,b)
    c = SkyCoord(l*u.deg, b*u.deg, frame='galactic')
    dust = planck(c)
    return dust

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

NumObservationsToDraw = st.sidebar.selectbox('Max observations to plot:', [10,100,1000,10000,100000], 1)

dust = GetPlanckMap(planck)
lmin=0; lmax=360; bmin=-90; bmax=90; lbstep=1
fig_dust = px.imshow(np.log(dust), x=np.arange(lmin,lmax,lbstep), y=np.arange(bmin,bmax,lbstep), title=f'First {NumObservationsToDraw} observations plotted on Planck dust map')
fig_dust['layout']['yaxis'].update(title='latitude', autorange = True)
print(type(fig_dust))
plotsubset = xray_subset.head(NumObservationsToDraw)
fig_dust.add_trace(go.Scattergl(
    x=plotsubset['gal_l'], 
    y=plotsubset['gal_b'], 
    text=list(f'{r[0]}, {r.name}' for r in plotsubset.itertuples()),
    mode='markers', 
    marker=dict(color='green', size=10,opacity=0.5)))
st.plotly_chart(fig_dust)

# fig2 = go.scatter(xray_subset.head(1000), x='gal_l', y='gal_b')
# st.plotly_chart(fig2)

''' ### Currently selected data: '''
# Show the table with all the sources.
st.markdown(f'First {min(10000,len(xray_subset))} records of {len(xray_subset)} from the current selection from the Master X-ray database:')
st.dataframe(xray_subset.head(10000))

# And a detail for a specific record in case data gets truncated.
detail_number = st.number_input('Get detailed information on record #: ', format='%d', value=xray_subset.index[0])
st.write(xray_subset.loc[detail_number])
