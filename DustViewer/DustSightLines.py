from astropy.coordinates import SkyCoord
import astropy.units as u
from dask.distributed import Client, wait, as_completed
from dustmaps.bayestar import BayestarQuery
from dustmaps.planck import PlanckQuery
from genfire.fileio import writeMRC
from io import StringIO
#from numba import njit, jit
import numpy as np
import os, sys, shutil
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import streamlit as st
import yaml
import WhiteLineMethods as wlm

# Read the config.yaml
try:
    with open(os.path.join('..', 'Config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
except Exception as e:
    st.exception(e)
config['DataDirectory'] = st.text_input('Data directory for storing large files (e.g. dust maps):', config['DataDirectory'])
# Finally save the yaml back to disk in case the user made changes.
with open(os.path.join('..', 'Config.yaml'), 'w') as f:
    yaml.dump(config, f)

@st.cache
def InitializeDustmap(config):
    from dustmaps.config import config as dustconfig
    dustconfig['data_dir'] = config['DataDirectory']
    # Fetch the planck data if we don't have it yet.
    if not os.path.exists(os.path.join(config['DataDirectory'], 'planck')):
        print('downloading planck data')
        from dustmaps.planck import fetch
        fetch()
    planck = PlanckQuery()
    # Fetch the bayestar data if we don't have it yet.
    if not os.path.exists(os.path.join(config['DataDirectory'], 'bayestar')):
        print('downloading bayestar data')
        from dustmaps.bayestar import fetch
        fetch()
    bayestar = BayestarQuery(version='bayestar2015')
    return planck, bayestar

@st.cache
def GetPlanckMap(planck=None, lmin=-180,lmax=180, bmin=-90,bmax=90, lbstep=1):
# def GetPlanckMap(planck=None, lmin=0,lmax=-180, bmin=-90,bmax=90, lbstep=1):
    l = np.arange(lmin,lmax,lbstep)
    b = np.arange(bmin,bmax,lbstep)
    l,b = np.meshgrid(l,b)
    c = SkyCoord(l*u.deg, b*u.deg, frame='galactic')
    dust = planck(c)
    return dust

planck, bayestar = InitializeDustmap(config)
dustmap = GetPlanckMap(planck)

# ------------------------ USER DISPLAY STUFF STARTS HERE ------------------------

st.write('Summary csv must contain a source name, distance, lii and bii column.')
CSVFileName = st.text_input('CSV file input:', os.path.join('..', 'First Survey', 'Source Summary.csv'))
st.write(CSVFileName)
DirName = os.path.dirname(CSVFileName)
# Load the summary csv and remove any entries that don't have a reliable distance.
Summary = pd.read_csv(CSVFileName)
Summary.dropna(inplace=True)
# Make sure lii coordinates are numerical.
Summary = Summary.astype({'Distance (kpc)': float, 'lii':float, 'bii':float})
# Make sure galactic coordinates go from -180 to 180 per our current convention.
mask = Summary.lii > 180
Summary.loc[mask,'lii'] = Summary.loc[mask, 'lii'] - 180.0

lmin=-180; lmax=180; bmin=-90; bmax=90; lbstep=1
fig_dust = px.imshow(np.log(dustmap), x=np.arange(lmin,lmax,lbstep), y=np.arange(bmin,bmax,lbstep), title='Observations plotted on Planck dust map')
# fig_dust.update_geos(projection_type='mollweide')
fig_dust['layout']['yaxis'].update(title='latitude', autorange = True)
fig_dust['layout']['xaxis'].update(title='longitude', autorange = 'reversed')
fig_dust.add_trace(go.Scatter(x=Summary['lii'], y=Summary['bii'], mode='markers', marker=dict(size=50/Summary['Distance (kpc)']),
    hovertext=Summary['Source Name']))
st.plotly_chart(fig_dust)

# @st.cache(suppress_st_warning=True)
def ComputeDustLine(r, lii, bii):
    # Populate the sphere one radial line at a time.
    c = SkyCoord(l=lii*u.deg, b=bii*u.deg, distance=r*u.pc, frame='galactic')
    rho = bayestar(c, mode='best')
    drho = np.diff(rho, axis=0, prepend=0)
    return(rho, drho)

# Now we'll populate the E(B-V) densities from bayestar.
Summary['E(B-V)'] = 0.0
rlist = []
rholist = []
drholist = []
rhofig = go.Figure()
drhofig = go.Figure()
for index,row in Summary.iterrows():
    r = np.linspace(0,row['Distance (kpc)']*1000,1000) # Radius units for bayestar are pc.
    rho, drho = ComputeDustLine(r, row.lii, row.bii)
    st.write(f"E(B-V) for {row['Source Name']} at distance {row['Distance (kpc)']} kpc, is {np.max(rho):0.3g}.") 
    Summary.loc[index,'E(B-V)'] = np.max(rho)
    rlist.append(rho)
    rholist.append(rho)
    drholist.append(drho)
    rhofig.add_trace(go.Scatter(x=r, y=rho, name=row['Source Name'])) #, line=dict(color='blue')))
    drhofig.add_trace(go.Line(x=r, y=drho, name=row['Source Name']))
st.write(Summary)

# Add the total column density to the dataframe.
rho = np.max(rholist, axis=1)
Summary['rho'] = rho

rhofig.update_layout(title_text='E(B-V) as a function of distance to X-ray sources.', xaxis_title='parsec', yaxis_title='E(B-V)')
rhofig.update_xaxes(type='log', range=[1.5,4])
drhofig.update_layout(title_text='d(E(B-V))/dr, with r in parsec.', xaxis_title='parsec', yaxis_title='dE(B-V)/dr')
st.write(rhofig)
st.write(drhofig)

st.markdown('----')

def LoadSpectrumFile(FileName):
    with open(FileName, 'r') as f:
        SpecText = f.read()
    SpecBody = re.search(r'#\s*eV,.*',SpecText, flags=re.DOTALL)[0]
    Spec = np.genfromtxt(StringIO(SpecBody), delimiter=',', names=['eV', 'A', 'Flux'])
    Spec.sort(order=['eV','Flux']) # Sort for increasing eV instead of increasing wavelength.
    # st.write(px.line(x=Spec['eV'], y=Spec['Flux'], labels={'x':'eV', 'y':'Flux in 1/(s cm^2 A)'}))
    return Spec

def GetFeLStats(SourceName):
    SpecName = os.path.join(DirName, SourceName+'.csv')
    Spec = LoadSpectrumFile(SpecName)
    Einterp, Sraw, SNoBkg, SPeaks, Jump, E0Amp, L3Area, L2Area = wlm.FeLStats(Spec['eV'], Spec['Flux'], PreEdgeWindowStart=670.0, PreEdgeWindowStop=700.0, PostEdgeEnergy=770.0)
    OD = -np.log((E0Amp-np.abs(Jump))/E0Amp)
    fig = go.Figure(go.Line(x=Einterp, y=Sraw, name='Spectrum'))
    fig.add_trace(go.Line(x=Einterp, y=Sraw-SNoBkg, name='Edge'))
    fig.update_layout(title_text=SourceName+':', xaxis_title='eV', yaxis_title='Flux')
    st.write(fig)
    st.write(f'OD = {OD}') 
    st.markdown('----')
    return OD

ODlist = []
for index,row in Summary.iterrows():
    ODlist.append(GetFeLStats(row['Source Name']))
Summary['OD'] = ODlist

st.write(Summary)
st.write(px.scatter(Summary, x='OD', y='rho', text='Source Name', title='rho vs OD for all sources.'))
st.write(px.scatter(Summary, x='OD', y='Distance (kpc)', text='Source Name', title='distance vs OD for all sources.'))
st.write(px.scatter(Summary, x='Distance (kpc)', y='rho', text='Source Name', title='rho vs distance for all sources.'))
st.write(px.scatter(Summary, x='OD', y='lii', text='Source Name'))
st.write(px.scatter(Summary, x='OD', y='bii', text='Source Name'))

import seaborn as sns

grid = sns.pairplot(Summary, kind='reg')
st.pyplot()

