from copy import deepcopy
import os, sys, shutil
import yaml

import streamlit as st

from astropy.constants import c, h, e
from astropy.io import fits
import numpy as np
import pandas as pd
from urllib.error import HTTPError

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

# @st.cache
def GetOneXMMSpectrum(config, obsid):
    obsidnumeric = int(obsid)
    obsidstr = f"{obsidnumeric:010.0f}"
    spectrumfits = fits.open(os.path.join(config['DataDirectory'], 'XMMNewtonFluxed', obsidstr+'.ftz'))
    spec = spectrumfits[1]
    angstrom = np.nan_to_num(spec.data.field('CHANNEL').copy())
    eV = np.nan_to_num((h*c/(e.si*angstrom*1e-10)).value)
    flux = np.nan_to_num(spec.data.field('FLUX').copy())
    error = np.nan_to_num(spec.data.field('ERROR').copy())
    spectrumfits.close()
    #       x-axis(lambda),            x-axis(eV),          y-axis,                x-label,                x-label,  y-label
    return  angstrom.astype('float'),  eV.astype('float'),  flux.astype('float'),  error.astype('float'), spec.header['TUNIT1'],  'eV',     spec.header['TUNIT2'],      spec.header['TUNIT3']

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

def CombineXMMSpectra(config, xray_subset, N=-1):
    # Load multiple XMM spectra and add them together using records from a dataframe (xray_subset).
    # N=-1 means to combine all records -- otherwise, N = a maximum number to combine.

    CombiningMessage = st.text('Combining records...')

    # Make structures to hold the sum spectrum.  Load a spectrum of Cyg X-1 that we know is good.
    obsid = 745250601 # This is a cyg x-1 spectrum
    angstromsum, eVsum, fluxsum, errorsum, angstrom_label, eV_label, flux_label, error_label = deepcopy(GetOneXMMSpectrum(config, obsid))
    fluxsum[:] = 0
    total_observation_time = 0

    # Two special cases mean we use the whole input subset: 1) the user asks for it with N=-1 or 2) the subset only has one record anyway.
    if (N == -1) or (len(xray_subset) == 1):
        xray_iterate = xray_subset
    else:
        xray_iterate = xray_subset.iloc[:N]

    for i, r in enumerate(xray_iterate.itertuples()):
        CombiningMessage.text(f'Record {i} of {len(xray_iterate)}')
        try:
            print(r.obsid)
            # Read this observation ID and add it to the cumulative.
            obsid = int(r.obsid)
            obsidnumeric = r.obsid
            angstrom, eV, flux, error, _, _, _, _ = deepcopy(GetOneXMMSpectrum(config, obsid))
            fluxsum += np.nan_to_num(flux)
            if len(xray_iterate) == 1:
                # Only copy the errors over if this isn't actually a sum of spectra.  
                errorsum = np.nan_to_num(error)
            else:
                # We can't add errors.  So don't pretend we have them when combining spectra.  In the future I'll do an SNR algorithm.
                errorsum = np.zeros(len(errorsum)) 
            total_observation_time += np.nan_to_num(r.rgs1_time)
        except FileNotFoundError as e:
            CombiningMessage.text(f'Skipping {obsid} -- no spectrum found.')
        except HTTPError as e:
            CombiningMessage.text(f'Skipping {obsid} -- no spectrum found.')
        except OSError as E:
            CombiningMessage.text(f'Empty or corrupt FITS file: {obsid}.')

    CombiningMessage.text(f'Summed {i+1} spectra with total {total_observation_time} seconds of observation.')

    return angstromsum, eVsum, fluxsum, errorsum, angstrom_label, eV_label, flux_label, error_label, total_observation_time
