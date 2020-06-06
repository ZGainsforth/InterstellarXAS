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

''' ### Currently selected data: '''
# Show the table with all the sources.
st.markdown(f'Currently selected {len(xray_subset)} records of {len(xray_database)} from the XMMMaster database:')
st.dataframe(xray_subset)

for i, obsid in enumerate(xray_subset['obsid']):
    print(f'Record {i} of {len(xray_subset)}')
    try:
        # Download this observation ID if we haven't already done so.
        obsidnumeric = int(obsid)
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
        else:
            st.write(f'Spectrum {obsidstr} already downloaded.')
    except:
        st.write(f'Exception so skipping {obsidstr}.')
        continue

