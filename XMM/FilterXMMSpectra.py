from copy import deepcopy
import os, sys, shutil
import time

import streamlit as st

from astropy.constants import c, h, e
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
# import dustmaps
# from dustmaps.planck import PlanckQuery
# from dustmaps.bayestar import BayestarQuery
from astroquery.esa.xmm_newton import XMMNewton
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from rich.traceback import install; install()
import requests
from scipy.stats import mode
import tarfile

sys.path.append(os.path.abspath('..'))
import InterstellarXASTools    

''' 
# XMM Spectrum combiner 

The spectrum combiner can be used to combine XMM RGS spectra from different sources and cluster them by location and spectral content.  This script expects to have all the spectra available locally, so please run DownloadAllXMMSpectra.py first.

### Configuration:

'''

config = InterstellarXASTools.init_config()

xray_database = InterstellarXASTools.load_xmm_master_database(config)

st.markdown(f'Currently using {len(xray_database)} records from the XMMMaster database.')


@st.cache
def GetFeaturesFeL(E, Intensity, plotting=True):
    # Make sure we are dealing with an increasing x-axis.
    if E[1] < E[0]:
        E = E[::-1]
        Intensity = Intensity[::-1]

    # Define the positions of the pre and post-edge for Fe-L
    preedge_start = 660 # eV
    preedge_end   = 700 # eV
    postedge_start = 740 # eV
    postedge_end   = 800 # eV

    # Get the index into the spectrum based on the energy.
    Eindex = lambda x: np.argmin(np.abs(E-x))

    # First we find some fluorescence lines that allow us to rule out spectra.
    features = dict()
    if np.mean(np.abs(Intensity[Eindex(651.5):Eindex(656.5)])) > 5*np.mean(np.abs(Intensity[Eindex(673.0):Eindex(683.0)])):
        features['Fluor_654'] = True
    else:
        features['Fluor_654'] = False

    # Trim the spectrum for feature finding to be only from the start of the pre-edge to the end of the post.
    Intensity = Intensity[Eindex(preedge_start):Eindex(postedge_end)]
    E = E[Eindex(preedge_start):Eindex(postedge_end)]

    features['mean'] = np.nan_to_num(np.mean(Intensity))   # Mean
    features['mode'] = np.nan_to_num(mode(Intensity)[0][0])   # Mode
    if features['mode'] == 0:
        features['mean/mode'] = 0
    else:
        features['mean/mode'] = np.nan_to_num(features['mean'] / features['mode']) # Mean_Mode
    features['stdev'] = np.nan_to_num(np.std(Intensity))
    features['stdevdiff'] = np.nan_to_num(np.std(np.diff(Intensity)))   # Standard deviation of derivative.

    # Now smooth it a bit.
    kernel = Gaussian1DKernel(1.0) # 1 eV Gaussian kernel (since the x-axis units are in eV).
    # kernel /= np.sum(kernel)
    Intensity = convolve(Intensity, kernel, boundary='extend')

    # Fit preedge
    Pre = np.polyfit(E[0:Eindex(preedge_end)], Intensity[0:Eindex(preedge_end)], 1)
    #print(Pre)
    preedge = Pre[0]*E + Pre[1]
    IntensityNew = Intensity - preedge

    if plotting==True:
        # Show it.
        plt.figure(1)
        plt.clf()
        plt.plot(E,Intensity, 'b', E, preedge, 'g')
        st.write(plt.gcf())

    # Fit postedge
    Post = np.polyfit(E[Eindex(postedge_start):], Intensity[Eindex(postedge_start):], 1)
    postedge  = Post[0]*E + Post[1]
    if plotting==True:
        # Show it.
        plt.figure(2)
        plt.clf()
        plt.plot(E,Intensity, 'b', E, preedge, 'g', E, postedge, 'r')
    # Calculate the edge jump.
    features['edgejump'] = -(Post[0]*707 + Post[1] -(Pre[0]*707 + Pre[1])) # assume the edge is at 707 eV.  And make big jumps positive.
    if plotting==True:
        plt.vlines(707, 0, -features['edgejump'], 'k')
        st.write(plt.gcf())

    features['hardedge'] =  np.mean(Intensity[Eindex(690):Eindex(700)]) - np.mean(Intensity[Eindex(707):Eindex(720)])

    # # Get height of L3 relative to edge jump.  This is the mean distance from the spectrum to the postedge line in the region of the L3 whiteline.
    # def indFromE(x): return np.argmin(np.abs(E-x))
    # features['L3'] = -np.mean(IntensityNew[indFromE(707):indFromE(716)] - postedge[indFromE(707):indFromE(716)])
    # if plotting==True:
    #     plt.fill_between(E[indFromE(707):indFromE(716)], IntensityNew[indFromE(707):indFromE(716)], postedge[indFromE(707):indFromE(716)], facecolor='brown')
    # print('L3 feature = ', features['L3'])
    # # Bin region for L3: 175-200
    # # Bin region for L2: 200-225

    if plotting==True:
        plt.pause(0.01)
        input()
    return features

@st.cache(suppress_st_warning=True)
def GenerateFeaturesForNSpectra(N=1000):
    xray_subset = xray_database.copy()
    # Get a random record, just to get the x-axis (channels)
    obsid = 745250601 # This is a cyg x-1 spectrum
    angstromsum, eVsum, fluxsum, angstrom_label, eV_label, flux_label = deepcopy(InterstellarXASTools.GetOneXMMSpectrum(config, obsid))
    # Get features and make a dataframe.
    xyz = deepcopy(GetFeaturesFeL(eVsum, fluxsum, plotting=False))
    xyz['obsid']=obsid
    df = pd.DataFrame([xyz])
    # Zero out the flux before we start adding spectra.
    fluxsum[:] = 0

    CombiningMessage = st.text('Combining records...')
    for i, obsid in enumerate(xray_subset['obsid'][:N]):
        CombiningMessage.text(f'Record {i} of {len(xray_subset)}')
        try:
        # if True:
            # Download this observation ID if we haven't already done so.
            obsidnumeric = int(obsid)
            angstrom, eV, flux, _, _, _ = deepcopy(InterstellarXASTools.GetOneXMMSpectrum(config, obsid))
            # Double check that the x-axis is the same.
            if np.all(angstromsum != angstrom):
                CombiningMessage.text(f'Could not add record {i} of {len(xray_subset)}')
                continue
            fluxsum += np.nan_to_num(flux)
            xyz = deepcopy(GetFeaturesFeL(eV, flux, plotting=False))
            xyz['obsid']=obsid
            df = df.append(xyz, ignore_index=True)
        except:
            CombiningMessage.text(f'Could not add record {i} of {len(xray_subset)}')
            pass
    return df

st.markdown(f'### Features from all data:')
df = deepcopy(GenerateFeaturesForNSpectra(N=1000))
st.write(df)
st.write(df.describe())

if st.checkbox('Show Sweetviz analysis of raw data.', False):
    import sweetviz
    sweetviz.analyze(df).show_html()

st.markdown(f'### Trim dataset and re-view:')

# Remove spectra with known strong fluorescence lines.
df_trim = deepcopy(df)
df_trim = df_trim[df_trim['Fluor_654'] == False]

# Trim based on the hardedge amplitude
hardedge_cutoff = st.number_input('Set filtering threshold for hardedge', value=1e-4, format='%g')
df_trim = df_trim[df_trim['hardedge'] > hardedge_cutoff]

# Trim based on the noise
noise_cutoff = st.number_input('Set filtering threshold for noise (stdevdiff/abs(stdev))', value=1, format='%g')
df_trim = df_trim[df_trim['stdevdiff']/np.abs(df_trim['stdev']) < noise_cutoff]


st.write(f'Filtered data contains {len(df_trim)} spectra.')
st.write(df_trim)
st.write(df_trim.describe())

if st.checkbox('Show Sweetviz analysis of filtered data.', False):
    import sweetviz
    sweetviz.analyze(df_trim).show_html()

# Plot spectra remaining.
filtered_index = st.slider('View filtered spectrum: ', 0, len(df_trim)-1, 0)
angstrom, eV, flux, angstrom_label, eV_label, flux_label = deepcopy(InterstellarXASTools.GetOneXMMSpectrum(config, int(df_trim.iloc[filtered_index]['obsid'])))
fig_spec = px.line(x=eV.astype('float'), y=flux.astype('float'))
fig_spec['layout']['xaxis'].update(title=eV_label)
fig_spec['layout']['yaxis'].update(title='Proportional to: ' + flux_label)
fig_spec.update_xaxes(range=[650,800])
st.write(f"Vieweing Fe-L edge of obsid={int(df_trim.iloc[filtered_index]['obsid'])}")
eV_trim, flux_trim = InterstellarXASTools.GetSpectrumPortion(eV, flux, 650, 800)
fig_spec.update_yaxes(range=[np.min(flux_trim), np.max(flux_trim)])
st.plotly_chart(fig_spec)
# st.write(df_trim.iloc[filtered_index]['Fluor_654'])
# st.write(df_trim.iloc[filtered_index]['test1'])
st.write(df_trim.iloc[filtered_index]['stdevdiff']/np.abs(df_trim.iloc[filtered_index]['stdev']))