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
from numba import njit, jit
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

@njit
def DetectPeak(E, Intensity, EnergyList, sigma):
    # E and Intensity are the input energies and intensity.
    # EnergyList has 6 elements: (prestart, preend, peakstart, peakend, poststart,postend)
    # Sigma gives a threshold for how many sigma are necessary in each reason to call it.
    Eindex = lambda x: np.argmin(np.abs(E-x))
    # Get the mean value and standard deviation in each of the regions: pre, peak and post
    prepeak = np.mean(np.abs(Intensity[Eindex(EnergyList[0]):Eindex(EnergyList[1])]))
    prepeakstd = np.std(np.abs(Intensity[Eindex(EnergyList[0]):Eindex(EnergyList[1])]))
    onpeak = np.mean(np.abs(Intensity[Eindex(EnergyList[2]):Eindex(EnergyList[3])]))    
    onpeakstd = np.std(np.abs(Intensity[Eindex(EnergyList[2]):Eindex(EnergyList[3])]))    
    postpeak = np.mean(np.abs(Intensity[Eindex(EnergyList[4]):Eindex(EnergyList[5])]))
    postpeakstd = np.std(np.abs(Intensity[Eindex(EnergyList[4]):Eindex(EnergyList[5])]))
    if ((onpeak - sigma*onpeakstd) > (prepeak + sigma*prepeakstd)) and ((onpeak - sigma*onpeakstd) > (postpeak + sigma*postpeakstd)):
        # print(prepeak, onpeak, postpeak)
        return(True)
    else:
        return(False)
 
@st.cache
@jit
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

    features = dict()
    # First we find some fluorescence lines that allow us to rule out spectra.

    # if np.mean(np.abs(Intensity[Eindex(651.5):Eindex(656.5)])) > 5*np.mean(np.abs(Intensity[Eindex(673.0):Eindex(683.0)])):
    #     features['Fluor_654'] = True
    # else:
    #     features['Fluor_654'] = False

    features['Fluor_654'] = DetectPeak(E, Intensity, (638.2, 644.2, 651.5, 656.5, 673.0, 683.0), 1)
    features['Fluor_685'] = DetectPeak(E, Intensity, (667.9, 669.8, 682.5, 689.3, 697.7, 706.3), 1)
    features['Fluor_755'] = DetectPeak(E, Intensity, (741.3, 749.3, 752.6, 755.9, 760.9, 771.1), 1)
    features['Fluor_771'] = DetectPeak(E, Intensity, (750.9, 765.9, 766.0, 774.6, 778.1, 785.2), 1)

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

    features['hardedge'] =  np.mean(Intensity[Eindex(700.5):Eindex(704.8)]) - np.mean(Intensity[Eindex(707.7):Eindex(715)])

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
    angstromsum, eVsum, fluxsum, error, angstrom_label, eV_label, flux_label, error_label = deepcopy(InterstellarXASTools.GetOneXMMSpectrum(config, obsid))
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
            print(obsid)
            angstrom, eV, flux, error, _, _, _, _ = deepcopy(InterstellarXASTools.GetOneXMMSpectrum(config, obsid))
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

    # We included an arbitrary spectrum in the first row to get the dataframe started.  Now get rid of it.
    df = df.loc[1:]

    return df

N = st.number_input('Cap input datasize to (-1 for all records): ', value=1000)

st.markdown(f'### Features from all data:')
df = deepcopy(GenerateFeaturesForNSpectra(N=N))
st.write(df)
st.write(df.describe())

# Write a CSV putting all this data on disk for other analyses.
df.to_csv('FeaturesVsobsid.csv')

if st.checkbox('Show Sweetviz analysis of raw data.', False):
    import sweetviz
    sweetviz.analyze(df).show_html()

st.markdown(f'### Trim dataset and re-view:')

# Remove spectra with known strong fluorescence lines.
df_trim = deepcopy(df)
df_trim = df_trim[df_trim['Fluor_654'] == False]
df_trim = df_trim[df_trim['Fluor_685'] == False]
df_trim = df_trim[df_trim['Fluor_755'] == False]
df_trim = df_trim[df_trim['Fluor_771'] == False]

# Remove spectra where the most common value is zero -- these usually have big parts chopped out.
df_trim = df_trim[df_trim['mode'] != 0]

# Trim based on the hardedge amplitude
hardedge_cutoff = st.number_input('Set filtering threshold for hardedge (higher numbers more stringent):', value=1e-4, format='%g')
df_trim = df_trim[df_trim['hardedge'] > hardedge_cutoff]

# Trim based on the noise
noise_cutoff = st.number_input('Set filtering threshold for noise (stdevdiff/abs(stdev)) (lower numbers more stringent)', value=1.0, format='%g')
df_trim = df_trim[df_trim['stdevdiff']/np.abs(df_trim['stdev']) < noise_cutoff]

# Trim based on the edge jump
edgejump_cutoff = st.number_input('Set filtering threshold for edge jump (higher numbers more stringent): ', value=0.0005, format='%g')
df_trim = df_trim[df_trim['edgejump'] > edgejump_cutoff]

# Trim based on the hardedge/edge jump ratio.  
jumpoverhard_cutoff = st.number_input('Set filtering threshold for edgejump/hardedge (ratios must be closer than a factor of, lower numbers more stringent): ', value=3.0, format='%g')
df_trim = df_trim[df_trim['edgejump']/df_trim['hardedge'] < jumpoverhard_cutoff]

# Reindex the trimmed set.
df_trim = df_trim.reset_index()


st.write(f'Filtered data contains {len(df_trim)} spectra.')
st.write(df_trim)
st.write(df_trim.describe())

if st.checkbox('Show Sweetviz analysis of filtered data.', False):
    import sweetviz
    sweetviz.analyze(df_trim).show_html()

xray_subset = df_trim.merge(xray_database, on='obsid')
st.write(xray_subset)

# Plot spectra remaining.
filtered_index = st.slider('View filtered spectrum: ', 0, len(df_trim)-1, 0)
angstrom, eV, flux, error, angstrom_label, eV_label, flux_label, error_label = deepcopy(InterstellarXASTools.GetOneXMMSpectrum(config, int(df_trim.iloc[filtered_index]['obsid'])))
fig_spec = px.line(x=eV.astype('float'), y=flux.astype('float'))
fig_spec['layout']['xaxis'].update(title=eV_label)
fig_spec['layout']['yaxis'].update(title='Proportional to: ' + flux_label)
fig_spec.update_xaxes(range=[650,800])
st.write(f"Vieweing Fe-L edge of obsid={int(df_trim.iloc[filtered_index]['obsid'])}")
eV_trim, flux_trim = InterstellarXASTools.GetSpectrumPortion(eV, flux, 650, 800)
fig_spec.update_yaxes(range=[np.min(flux_trim), np.max(flux_trim)])
st.plotly_chart(fig_spec)

plot_all_selected_sum = st.checkbox('Sum together and plot all selected data.', False)
if plot_all_selected_sum:
    angstromsum, eVsum, fluxsum, errorsum, angstrom_label, eV_label, flux_label, error_label, total_observation_time = InterstellarXASTools.CombineXMMSpectra(config, xray_subset, -1)
    fig_spec = px.line(x=eVsum.astype('float'), y=fluxsum.astype('float'))
    fig_spec['layout']['xaxis'].update(title=eV_label)
    fig_spec['layout']['yaxis'].update(title='Proportional to: ' + flux_label)
    fig_spec.update_xaxes(range=[650,800])
    eV_trim, flux_trim = InterstellarXASTools.GetSpectrumPortion(eVsum, fluxsum, 650, 800)
    fig_spec.update_yaxes(range=[np.min(flux_trim), np.max(flux_trim)])
    st.plotly_chart(fig_spec)

    if st.button('Save sum spectrum'):
        save_file_root = f'Sum_of_{len(xray_subset)}_spectra_with_{total_observation_time}_seconds'
        fig_spec.write_image(f'{save_file_root}.png', width=2048, height=1024)
        fig_spec.write_html(f'{save_file_root}.html')
        xray_subset.to_csv(f'{save_file_root}.csv')
        st.write(f'Saved to: {save_file_root} png/html/csv')
