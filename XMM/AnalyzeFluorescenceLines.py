import streamlit as st
import os, sys, shutil
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import least_squares
import sweetviz as sv

# Connect to the master database which allows us to get spectra for specific obsids.
sys.path.append(os.path.abspath('..'))
import InterstellarXASTools    
config = InterstellarXASTools.init_config()
xray_database = InterstellarXASTools.load_xmm_master_database(config)
st.markdown(f'Currently using {len(xray_database)} records from the XMMMaster database.')

Features = pd.read_csv('FeaturesVsobsid.csv')

st.write(f'There are {len(Features[Features["Fluor_654"] == True])} spectra with fluor line at 654 eV.')
st.write(f'There are {len(Features[Features["Fluor_685"] == True])} spectra with fluor line at 685 eV.')
st.write(f'There are {len(Features[Features["Fluor_755"] == True])} spectra with fluor line at 755 eV.')
st.write(f'There are {len(Features[Features["Fluor_771"] == True])} spectra with fluor line at 771 eV.')
# st.write(Features)

def IndexOfEnergy(eV, E0):
    return np.argmin(np.abs(E0-eV))

def GenerateFluorSpectrum(x, eV):
    m = x[0] # linear background slope
    b = x[1] # linear background intercept
    A = x[2] # Gaussian Amplitude
    E0 = x[3] # Gaussian centroid in eV.
    sigma = x[4] # Gaussian sigma.

    # Generate the simulated spectrum as a line and Gaussian.
    y = m*eV + b
    y += A * np.exp(-(eV-E0)**2/(2*sigma**2))
    return y

def GenerateFluorError(x, eV, flux):
    y = GenerateFluorSpectrum(x, eV)
    # Return the difference between this spectrum and the desired spectrum.
    Error = np.sum((y-flux)**2)
    # st.write(Error, x)
    return Error

def FitFluorLine(eV, flux, centroid):
    # Initial guess for a line.
    L = np.polyfit(eV, flux, deg=1)
    # Initial guess for Gaussian.
    CentroidIndex = IndexOfEnergy(eV,centroid)
    A = flux[CentroidIndex] - (L[0]*centroid + L[1])
    sigma = 1.0

    # Pack that into a vector for the least squares fitter.
    x0 = [L[0], L[1], A, centroid, sigma]

    res = least_squares(GenerateFluorError, x0, args=(eV, flux), x_scale='jac')
    y = GenerateFluorSpectrum(res.x, eV)

    # fig = go.Figure()
    # fig.add_trace(go.Line(x=eV, y=flux))
    # fig.add_trace(go.Line(x=eV, y=y))
    # st.write(fig)
    # st.write(res)

    return res.x


def ProcessOneFluorescenceLine(Name, energy, minE, maxE):
    df = Features[Features[Name]==True]
    df['Fluor_centroid'] = 0.0
    df['Fluor_amp'] = 0.0
    df['Fluor_sigma'] = 0.0
    minE, maxE = minE, maxE
    # st.write(df)
    for i, r in df.iterrows():
        angstrom, eV, flux, error, angstrom_label, eV_label, flux_label, error_label = InterstellarXASTools.GetOneXMMSpectrum(config, int(r['obsid']))
        eV_trim, flux_trim = InterstellarXASTools.GetSpectrumPortion(eV, flux, minE, maxE)
        x = FitFluorLine(eV_trim, flux_trim, energy)
        if x[2] > 0.001:
            df.at[i, 'Fluor_centroid'] = x[3]
            df.at[i, 'Fluor_amp'] = x[2]
            df.at[i, 'Fluor_sigma'] = x[4]
        else:
            df.at[i, 'Fluor_centroid'] = np.nan
            df.at[i, 'Fluor_amp'] = np.nan
            df.at[i, 'Fluor_sigma'] = np.nan

    st.write(df)
    df.to_csv(Name+'.csv')
    analysis = sv.analyze(df)
    analysis.show_html(filepath=Name+'.html')

ProcessOneFluorescenceLine('Fluor_771', 771, 760, 780)
# ProcessOneFluorescenceLine('Fluor_755', 755, 745, 765)
# ProcessOneFluorescenceLine('Fluor_685', 685, 680, 690)
# ProcessOneFluorescenceLine('Fluor_654', 654, 649, 659)

#     Fluor_654 = Features[Features['Fluor_654']==True]
#     Fluor_654['Fluor_centroid'] = 0.0
#     Fluor_654['Fluor_amp'] = 0.0
#     Fluor_654['Fluor_sigma'] = 0.0
#     minE, maxE = 649.0, 659.0
#     # st.write(Fluor_654)
#     for i, r in Fluor_654.iterrows():
#         angstrom, eV, flux, error, angstrom_label, eV_label, flux_label, error_label = InterstellarXASTools.GetOneXMMSpectrum(config, int(r['obsid']))
#         eV_trim, flux_trim = InterstellarXASTools.GetSpectrumPortion(eV, flux, minE, maxE)
#         x = FitFluorLine(eV_trim, flux_trim, 654)
#         Fluor_654.at[i, 'Fluor_centroid'] = x[3]
#         Fluor_654.at[i, 'Fluor_amp'] = x[2]
#         Fluor_654.at[i, 'Fluor_sigma'] = x[4]
#     st.write(Fluor_654)
#     analysis = sv.analyze(Fluor_654)
#     analysis.show_html(filepath='Fluor_654.html')
