import io
import numpy as np
import pandas as pd
import os, sys, shutil
import plotly.express as px
import plotly.graph_objects as go
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d
import streamlit as st
# st.set_option('deprecation.showfileUploaderEncoding', False)

def OpenSpectrumFile(SpecPath, IsData=False):
    # Strip out any lines starting with common comment symbols.
    # SpecText = (line for line in open(SpecPath) if not line[0] in ('%', '$', '#'))
    if IsData == False:
        with open(SpecPath, 'r') as f:
            SpecRawText = f.readlines()
    else:
        breakpoint()
        SpecRawText = SpecPath.split('\n')
    SpecText = (line for line in SpecRawText if not line[0] in ('%', '$', '#', ','))
    SpecText = '\n'.join(SpecText)
    S = None
    # Try reading it as a csv.
    try:
        S = np.genfromtxt(io.StringIO(SpecText), delimiter=',', skip_header=1)
    except e:
        pass
    # If that failed, it will return a 1D array of NaNs.
    if len(S.shape) == 1:
        # Open it as just two columns separated by whitespace.
        try:
            S = np.genfromtxt(io.StringIO(SpecText))
        except e:
            pass
    if BinChandraLikeXMM and 'Chandra' in SpecPath:
        # If this is a Chandra spectrum and we are supposed to bin it so it matches XMM, then do that now.
        SBin=np.zeros((len(XMMEnergyAxis),2))
        SBin[:,0] = XMMEnergyAxis
        for i in range(len(XMMEnergyAxis)):
            # if i == 800:
            #     breakpoint()
            ChandraBinStart = np.argmin((S[:,0] - XMMBinStart[i])**2)
            ChandraBinEnd = np.argmin((S[:,0] - XMMBinEnd[i])**2)
            SBin[i,1] = np.mean(S[ChandraBinStart:ChandraBinEnd,1])
            S[ChandraBinStart:ChandraBinEnd,1] = 0
        S = SBin
    assert S is not None, f'Unable to parse {SpecPath}.'
    return S

# Read in the excel file containing a record of all the spectra and their metadata.
Library = pd.read_excel(os.path.join('..', 'EdgeLibrary.xlsx'), header=1)
Library = Library.drop(['Index'], axis=1)
Library['Index'] = list(range(Library.shape[0]))
# Library.reset_index(inplace=True)
Lib = Library

# Get the energy axis for XMM which we will need for binning Chandra spectra to match XMM.
XMMEnergyAxis = np.genfromtxt(os.path.join('..', 'XMMBinning.csv'), skip_header=1)
XMMBinSize = np.diff(XMMEnergyAxis)
# Make bins which are halfway between the energies.
XMMBinStart = np.zeros(len(XMMEnergyAxis))
XMMBinEnd = np.zeros(len(XMMEnergyAxis))
XMMBinStart[1:] = XMMEnergyAxis[1:] - XMMBinSize/2
XMMBinStart[0] = XMMEnergyAxis[0] - XMMBinSize[0]/2  # Extrapolate the first bin minimum to be symmetric around its energy
XMMBinEnd[:-1] = XMMBinStart[1:]
XMMBinEnd[-1] = XMMEnergyAxis[-1] + XMMBinSize[-1]/2  # Extrapolate the last bin maximum to be symmetric around its energy

# Add a field so we can add external spectra (that aren't in the database).  NaN mean internal, and a numeric value gives the index into the list of external spectra.
Lib['External Spectrum'] = np.nan

# Allow the user to only view data from a specific telescope.
InstrumentFilter = st.sidebar.multiselect('Filter by instrument:', Lib['Instrument'].unique())
if InstrumentFilter:
    st.write(f'Viewing only instrument: {InstrumentFilter}.')
    Lib = Lib[Lib['Instrument'].isin(InstrumentFilter)]

# Allow the user to only view a specific edge(s).
EdgeFilter = st.sidebar.multiselect('Filter by edge:', Lib['Edge'].unique())
if EdgeFilter:
    st.write(f'Viewing only edges: {EdgeFilter}.')
    Lib = Lib[Lib['Edge'].isin(EdgeFilter)]

# Allow the user to only view a specific target.
TargetFilter = st.sidebar.multiselect('Filter by target:', Lib['Target Name'].unique())
if TargetFilter:
    st.write(f'Viewing only targets: {TargetFilter}.')
    Lib = Lib[Lib['Target Name'].isin(TargetFilter)]

# # Show the currently selectable spectra.
# st.write(Lib)

def AddExternalSpectrum(i=0):
    # st.write(f'Recursion {i}.')
    if st.checkbox('Add spectrum...', key=f'AddSpectrum{i}'):
        uploaded_file = st.file_uploader("Choose a 2 column file", key=f'file_uploader{i}')
        if uploaded_file is not None:
            S = OpenSpectrumFile(uploaded_file.getvalue().decode('utf-8'), IsData=True)
            SpectrumName = st.text_input('Spectrum Name:', key=f'SpectrumName{i}')
            EShift = st.number_input('Energy shift', key=f'EShift{i}')
            S[:,0] += EShift
            ExternalSpectrumList.append(S)
            LibTemp = AddExternalSpectrum(i+1)
            return LibTemp.append({'Edge':' ',
                'Spectrum Name' : SpectrumName,
                'External Spectrum' : i,
                    }, ignore_index=True)
        else:
            return Lib
    else:
        return Lib

# Now that we're done trimming the dataframe and adding external spectra, we want and index that will work for us.
Lib.reset_index(drop=True, inplace=True)
st.write('Available Spectra for plotting:')
st.write(Lib)
SelectedSpectra = st.multiselect('Select available Spectra:', Lib.index.tolist(), format_func=lambda x: f"{Lib.iloc[x]['Spectrum Name']} ({Lib.iloc[x]['Instrument']})")

st.markdown('<hr>', unsafe_allow_html=True)

AutorangeX = st.sidebar.checkbox('Autorange eV axis?', True)
AutorangeY = st.sidebar.checkbox('Autorange Y axis?', True)
BinChandraLikeXMM = st.sidebar.checkbox('Bin Chandra data to match XMM?', False)

Normalization = st.sidebar.radio('Auto normalize each spectra to:', ['No normalization', 'Max intensity'], 1) 

DefaultEshift = st.sidebar.radio('Apply default energy shift?', ['Yes', 'No'], 0) 

ExternalSpectrumList = list()
Lib = AddExternalSpectrum()

if len(SelectedSpectra) == 0:
    st.write('No spectra selected yet.  Nothing to show.')
    st.stop()

# Go through all the selected spectra and load the spectral data into an ordered dict.
# The key is the index into the Lib dataframe, and the value is a two column numpy array, eV vs intensity.
SpectrumDict = dict()
for ThisSpec in SelectedSpectra:
    if not np.isnan(Lib.iloc[ThisSpec]['External Spectrum']):
        SpectrumDict[ThisSpec] = ExternalSpectrumList[int(Lib.iloc[ThisSpec]['External Spectrum'])]
        continue
    SpecPath = os.path.join('..', 'Data', Lib.iloc[ThisSpec]['Instrument'], Lib.iloc[ThisSpec]['Spectrum Name']) + '.csv'
    S = OpenSpectrumFile(SpecPath)
    if DefaultEshift == 'Yes':
        S[:,0] += np.nan_to_num(Lib.iloc[ThisSpec]['Eshift'])
    SpectrumDict[ThisSpec] = S

MineVs = list()
MaxeVs = list()
# Loop through all the spectra.  We will bound based on the recommended range first, and if that is absent, then we use the min/max energy values in the 2 column file.
# for SName, S in SpectrumList.items():
for i in SelectedSpectra:
    # If the user selected autorange, then we find out the ranges to use for the plot.
    if AutorangeX == True:
        # The record in the dataframe contains info about the recommended range.
        SpecMetadata = Lib.iloc[i]
        # If there is a recommended value for the spectrum eV axis minimum, then use it.
        if not np.isnan(SpecMetadata['Min eV']):
            MineVs.append(SpecMetadata['Min eV'])
        else:
            # Otherwise, use the lowest energy in the spectrum.
            MineVs.append(np.min(SpectrumDict[i][:,0]))
        if not np.isnan(SpecMetadata['Max eV']):
            MaxeVs.append(SpecMetadata['Max eV'])
        else:
            MaxeVs.append(np.max(SpectrumDict[i][:,0]))
    # If no autorange, then we choose the range which includes all of all spectra.
    if AutorangeX == False:
        MineVs.append(np.min(SpectrumDict[i][:,0]))
        MaxeVs.append(np.max(SpectrumDict[i][:,0]))
MineV = np.min(MineVs)
MaxeV = np.max(MaxeVs)

# If the user wants max intensity, then we just remove the minimum and divide by the maximum in the plotting range.
def DoNormalization(S):
    if Normalization == 'Max intensity':
        # Figure out the range of the spectrum we need to consider for normalization.
        MinIndex = np.searchsorted(S[:,0], MineV)
        MaxIndex = np.searchsorted(S[:,0], MaxeV)
        # Remove the baseline (minimum intensity within the plotting range.
        S[:,1] -= np.min(S[MinIndex:MaxIndex,1])
        # Normalize by maximum intensity
        S[:,1] /= np.max(S[MinIndex:MaxIndex,1])
    return S

def ConvolveGaussian(S, Sigma):
    # In order to do all the math goodness below, we need a constant step energy axis.
    # Let's use 10000 steps, just to be wasteful.
    E = np.linspace(S[0,0], S[-1,0], 10000)
    dE = E[1] - E[0]
    SInterp = interp1d(S[:,0], S[:,1])(E)
    SigmaPx = Sigma / 2.3548 / dE
    SSmooth = gaussian_filter1d(SInterp, SigmaPx)
    S = np.stack((E, SSmooth), axis=1)
    return S

def GetSpectrumName(index):
    SpectrumName = f"{Lib.iloc[SName]['Spectrum Name']} ({Lib.iloc[SName]['Instrument']})"
    if len(SpectrumName) > 20:
        SpectrumName = SpectrumName[:20] + '...'
    return SpectrumName

def IndexOfEnergy(S, Energy):
    return np.argmin((S[:,0] - Energy)**2)

# Finally, plot the figure.
fig = go.Figure()
for SName, S in SpectrumDict.items():
    SpectrumName = GetSpectrumName(SName)
    st.write(f'{SpectrumName} spectrum tweaks:')
    Eshift = st.slider(f'Energy shift (eV): ', -30.0, 30.0, 0.0, key=SName)
    Offset = st.slider(f'Offset spectrum in y: ', -5.0, 5.0, 0.0, key=SName)
    Sigma = st.slider(f'Gaussian convolution sigma: ', 0.0, 5.0, 0.0, key=SName)
    if Sigma != 0:
        S = ConvolveGaussian(S, Sigma)
    st.markdown('<hr>', unsafe_allow_html=True)
    S = DoNormalization(S)
    fig.add_trace(go.Line(x=S[:,0]+Eshift, y=S[:,1]+Offset, name=SpectrumName))
if AutorangeX == True:
    # Autorange the x-axis.
    fig.update_layout(xaxis={'range':[MineV, MaxeV]})

if AutorangeY == True:
    # Autorange the y-axis based on the current view.
    # Get the y axis value based on the x.axis of the leftmost part of the plot range.
    MinI = S[IndexOfEnergy(S, fig.layout.xaxis.range[0]),1]
    # Get the y axis value based on the x.axis of the rightmost part of the plot range.
    MaxI = S[IndexOfEnergy(S, fig.layout.xaxis.range[1]),1]
    # And set those ranges.
    fig.update_layout(yaxis={'range':[MinI, MaxI]})

st.write(fig)
