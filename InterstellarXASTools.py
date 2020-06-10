import streamlit as st
import os, sys, shutil
import yaml

from astropy.constants import c, h, e
from astropy.io import fits
import numpy as np
import pandas as pd

def init_config():
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

    return config

# Read the XMMMaster database.
@st.cache
def load_xmm_master_database(config):
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

@st.cache
def GetOneXMMSpectrum(config, obsid):
    obsidnumeric = int(obsid)
    obsidstr = f"{obsidnumeric:010.0f}"
    spectrumfits = fits.open(os.path.join(config['DataDirectory'], 'XMMNewtonFluxed', obsidstr+'.ftz'))
    spec = spectrumfits[1]
    angstrom = np.nan_to_num(spec.data.field('CHANNEL').copy())
    eV = np.nan_to_num((h*c/(e.si*angstrom*1e-10)).value)
    flux = np.nan_to_num(spec.data.field('FLUX').copy())
    spectrumfits.close()
    #       x-axis(lambda),            x-axis(eV),          y-axis,                x-label,                x-label,  y-label
    return  angstrom.astype('float'),  eV.astype('float'),  flux.astype('float'),  spec.header['TUNIT1'],  'eV',     spec.header['TUNIT2']

def GetSpectrumPortion(E, Intensity, eVmin = 600., eVmax=800.):
    # Make sure we are dealing with an increasing x-axis.
    if E[1] < E[0]:
        E = E[::-1]
        Intensity = Intensity[::-1]
    # Get the index into the spectrum based on the energy.
    Eindex = lambda x: np.argmin(np.abs(E-x))
    # Extract only the portion of the spectrum we want
    Intensity_trim = np.nan_to_num(Intensity[Eindex(eVmin):Eindex(eVmax)])
    E_trim = E[Eindex(eVmin):Eindex(eVmax)]
    return E_trim, Intensity_trim