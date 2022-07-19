import io
import numpy as np
import pandas as pd
import os, sys, shutil
import plotly.express as px
import plotly.graph_objects as go
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d
import streamlit as st
import datapane as dp
# st.set_option('deprecation.showfileUploaderEncoding', False)

def EnergyToIndex(S, Energy):
    return np.argmin((S[:,0]-Energy)**2)

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
BinChandraLikeXMM = False
DefaultEshift = False

MineV = 300
MaxeV = 1000
st.write(Lib)

rownum = st.slider('Which row:', 0, Lib.shape[0], 1)
row = Lib.iloc[rownum]
st.write(row['Spectrum Name'], row['Instrument'])
SpecPath = os.path.join('..', 'Data', row['Instrument'], row['Spectrum Name']) + '.csv'
S = OpenSpectrumFile(SpecPath)
if DefaultEshift == 'Yes':
    S[:,0] += np.nan_to_num(row['Eshift'])
# Compute the amplitude scale for plotting.  We want the max value up to 1 keV to be at the top of the plot.
CutoffEnergyIndex = EnergyToIndex(S, 1000) #np.argmin((S[:,0]-1000)**2)
YMax = np.log10(np.max(S[:CutoffEnergyIndex,1]))
YMin = np.log10(np.mean(S[:300,1]))

Report = list()
Report.append(dp.Text(f"### {row['Spectrum Name']} as seen by {row['Instrument']}"))

# Plot the overall figure.
fig = go.Figure()
fig.add_trace(go.Line(x=S[:,0], y=S[:,1], name=row['Spectrum Name']))
fig.update_xaxes(range=[MineV, MaxeV], title_text='eV')
fig.update_yaxes(type='log', title_text='Counts')
fig.update_layout(yaxis={'range':[YMin, YMax]})
st.write(fig)
Report.append(dp.Plot(fig))

def AverageWindow(S, Start,End):
    Start = EnergyToIndex(S, Start)
    End = EnergyToIndex(S, End)
    return np.mean(S[Start:End, 1])

def MinMax(S, Start,End):
    Start = EnergyToIndex(S, Start)
    End = EnergyToIndex(S, End)
    return [np.min(S[Start:End, 1]), np.max(S[Start:End, 1])]

def FitEdgeLines(S, PreStart, PreEnd, PostStart, PostEnd):
    PreStart = EnergyToIndex(S, PreStart)
    PreEnd = EnergyToIndex(S, PreEnd)
    PostStart = EnergyToIndex(S, PostStart)
    PostEnd = EnergyToIndex(S, PostEnd)
    PreLine = np.polyfit(S[PreStart:PreEnd,0], S[PreStart:PreEnd,1], 1)
    Bkg = PreLine[0]*S[:,0] + PreLine[1]
    PostLine = np.polyfit(S[PostStart:PostEnd,0], S[PostStart:PostEnd,1], 1)
    Post = PostLine[0]*S[:,0] + PostLine[1]
    return Bkg, Post

def GetAndPlotOD(S, EdgeName, EdgeEnergy, PreStart, PreEnd, PostStart, PostEnd):
    # Fit and plot the O-K Edge jump
    Bkg, Post = FitEdgeLines(S, PreStart, PreEnd, PostStart, PostEnd)
    OD = -np.log(Post[EnergyToIndex(S, EdgeEnergy)]/Bkg[EnergyToIndex(S, EdgeEnergy)])
    ODStr = f'OD of {EdgeName} edge is {OD:0.2f}'
    st.write(ODStr)
    Report.append(ODStr)
    fig = go.Figure()
    fig.add_trace(go.Line(x=S[:,0], y=S[:,1], name=row['Spectrum Name']))
    fig.add_trace(go.Line(x=S[:,0], y=Bkg, name='Linear Background'))
    fig.add_trace(go.Line(x=S[:,0], y=Post, name='Linear Post edge'))
    fig.update_xaxes(range=[PreStart, PostEnd], title_text='eV')
    fig.update_yaxes(range=MinMax(S,PreStart,PostEnd), title_text='Counts')
    fig.update_layout(title=f'{EdgeName} edge')
    st.write(fig)
    Report.append(fig)

GetAndPlotOD(S, EdgeName='O-K', EdgeEnergy=536.0, PreStart=500.0, PreEnd=520.0, PostStart=550.0, PostEnd=570.0)
GetAndPlotOD(S, EdgeName='Fe-L', EdgeEnergy=707.0, PreStart=680.0, PreEnd=702.0, PostStart=730.0, PostEnd=780.0)
GetAndPlotOD(S, EdgeName='Ne-K', EdgeEnergy=866.5, PreStart=840.0, PreEnd=863.0, PostStart=876.0, PostEnd=900.0)

dp.Report(*Report).save(path=os.path.join('Results', f"{row['Spectrum Name']} - {row['Instrument']}.html"))
