import streamlit as st
import dustmaps.bayestar
from dustmaps.bayestar import BayestarQuery
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from dask.distributed import Client, wait, as_completed
from genfire.fileio import writeMRC
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from numba import njit, jit

@st.cache
def InitializeDustmap():
    # Initialize dustmaps using Bayestar2019
    print('Initializing Bayestar2019.')
    #dustmaps.bayestar.fetch()
    bayestar = BayestarQuery(max_samples=1)
    return bayestar

# # Function to convert our cartesian coordinates to spherical coordinates (i.e. galactic coordinates).
# def Cartesian2Spherical(x,y,z):
#     r = np.sqrt(x**2 + y**2 + z**2)
#     theta = np.arccos(z/r) * 180/np.pi 
#     phi = np.arctan2(y,x) * 180/np.pi + 180
#     theta = np.nan_to_num(theta)
#     phi = np.nan_to_num(phi)
#     # phi[phi==360] = 0
#     if(phi == 360):
#         phi = 0
#     return r, theta, phi

# Function to convert our cartesian coordinates to spherical coordinates (i.e. galactic coordinates).
@st.cache
@njit
def Spherical2Cartesian(r, phi, theta):
    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta+90)
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return x, y, z

if __name__ == "__main__":

    print('-------------------- START --------------------')

    # Parameters for 1 kpc quick map.  Units are pc.
    selectiondistance = 1000 # Distance from origin to center of one side of the box.
    selectiondistanceincrement = 100 # Voxel diameter in pc.
    phi = np.linspace(0, 360, 120)
    theta = np.linspace(-90, 90, 45) 

    # # Parameters for 1 kpc map.  Units are pc.
    # selectiondistance = 1000 # Distance from origin to center of one side of the box.
    # selectiondistanceincrement = 100 # Voxel diameter in pc.
    # phi = np.linspace(0, 360, 180)
    # theta = np.linspace(-90, 90, 90) 

    # # Parameters for 10 kpc map.  Units are pc.
    # selectiondistance = 10000 # Distance from origin to center of one side of the box.
    # selectiondistanceincrement = 80 # Voxel diameter in pc.
    # phi = np.linspace(0, 360, 180)
    # theta = np.linspace(-90, 90, 60) 

    # Make a sphere for populating values for the dustmap density.
    r = np.linspace(selectiondistanceincrement, selectiondistance, int(selectiondistance/selectiondistanceincrement))
    'r', r
    'phi', phi
    'theta', theta
    R,PHI,THETA = np.meshgrid(r, phi, theta, indexing='ij')
    RHO = np.zeros(R.shape)

    X,Y,Z = Spherical2Cartesian(R.ravel(), PHI.ravel(), THETA.ravel())
    X = np.nan_to_num(X)
    Y = np.nan_to_num(Y)
    Z = np.nan_to_num(Z)
    
    # Display the coordinates in both frames so we can compare and make sure the conversion is correct.
    dfdata = []
    for i in range(len(X)):
        dfdata.append((X[i], Y[i], Z[i], R.ravel()[i], PHI.ravel()[i], THETA.ravel()[i]))

    df = pd.DataFrame(dfdata, columns=['X', 'Y', 'Z', 'R', 'PHI', 'THETA'])

    ''' Coordinate conversion check: '''
    df

    ''' Show initial gridding for how we will extract rho from the dustmap. '''
    fig = px.scatter_3d(x=X, y=Y, z=Z, color=THETA.ravel(), size=R.ravel(), size_max=5, opacity=0.05)
    st.plotly_chart(fig)

    @st.cache(suppress_st_warning=True)
    def ComputeDustLines(r, theta,phi):
        bayestar = InitializeDustmap()

        # Populate the sphere one radial line at a time.
        st.write('Computing dust lines...')
        DustCalcProgress = st.progress(0)
        for ti, t in enumerate(theta):
            DustCalcProgress.progress(ti/len(theta))
            for pi, p in enumerate(phi):
                print(f'Computing sightline (phi, theta) = ({p:3.2f}, {t:2.2f}).', end='\r')
                c = SkyCoord(p*u.deg, t*u.deg, r*u.pc, frame='galactic')
                rho = bayestar(c, mode='best')
                RHO[:, pi, ti] = rho
        del bayestar
        return(RHO)

    RHO = ComputeDustLines(r,theta,phi)
    DRHO = np.diff(RHO, axis=0, prepend=0)

    import gc
    gc.collect()

    st.plotly_chart(px.histogram(RHO.ravel(), title='RHO histogram from radial sight lines.'))

    RHOscaled = RHO.copy()
    RHOscaled = np.nan_to_num(RHOscaled)
    RHOscaled[RHOscaled>0.5] = 0.

    DRHOscaled = DRHO.copy()
    DRHOscaled = np.nan_to_num(DRHOscaled)
    DRHOscaled[DRHOscaled>0.1] = 0.

    ''' Show dustmap rho. '''
    ColorMask = DRHOscaled>0.001
    ColorMask = ColorMask.ravel()
    # fig = px.scatter_3d(x=X[ColorMask], y=Y[ColorMask], z=Z[ColorMask], color=DRHOscaled.ravel()[ColorMask], size=R.ravel()[ColorMask], size_max=10, color_continuous_scale="Viridis", opacity=0.1)
    fig = px.scatter_3d(x=X[ColorMask], y=Y[ColorMask], z=Z[ColorMask], color=DRHOscaled.ravel()[ColorMask], size_max=10, color_continuous_scale="gray", opacity=0.1)
    fig.update_traces(marker=dict(line=dict(width=0)))
    st.plotly_chart(fig)

    # Now we are going to interpolate the radial sightlines onto a cubic grid.

    from scipy.interpolate import griddata, NearestNDInterpolator

    # New cartesian grid for output.
    newX, newY, newZ = np.mgrid[
            -selectiondistance:selectiondistance+0.1:selectiondistanceincrement, 
            -selectiondistance:selectiondistance+0.1:selectiondistanceincrement, 
            -selectiondistance:selectiondistance+0.1:selectiondistanceincrement]
    # st.write('Interpolated grid dimensions:', newX.shape)
    # st.write(newX)

    @st.cache(suppress_st_warning=True)
    def InterpolateOntoCartesian(X,Y,Z, RHO, newX, newY, newZ):
        from joblib import Parallel, delayed
        from scipy.spatial import Delaunay
        from scipy.interpolate import NearestNDInterpolator

        radialCoords = np.stack((X,Y,Z)).T
        newCoords = np.stack((newX.ravel(),newY.ravel(),newZ.ravel())).T
        newCoordsChunks = np.array_split(newCoords, indices_or_sections=100, axis=0)
    
        st.write('Preparing interpolation grid...')
        tri = Delaunay(radialCoords, ) #qhull_options='Qbb Qc Qz Q0 Q3 Q5 Q8 C0 C-0')
        interpolator = NearestNDInterpolator(tri, RHO.ravel())

        st.write('Interpolating...')
        def DoOneChunk(i, chunk):
            print(f'Chunk {i} of {len(newCoordsChunks)}')
            RHOinterpolated = interpolator(chunk)
            RHOinterpolated = np.nan_to_num(RHOinterpolated)
            # Zero out cartesian coordinates that are outside the original radius.
            dist = np.sqrt(chunk[:,0]**2 + chunk[:,1]**2 + chunk[:,2]**2)
            RHOinterpolated[dist > np.max(X)] = 0
            return RHOinterpolated

        results = Parallel(n_jobs=5, verbose=100, pre_dispatch=10)(delayed(DoOneChunk)(i, chunk) for i, chunk in enumerate(newCoordsChunks))
        return np.concatenate(results)

    from datetime import datetime, timedelta

    starttime = datetime.now()
    DRHOinterpolated = InterpolateOntoCartesian(X,Y,Z, DRHO, newX, newY, newZ)
    print(f'Interpolation time {datetime.now() - starttime}')
    st.write(DRHOinterpolated.shape)
    writeMRC('drho.mrc', DRHOinterpolated.reshape(newX.shape))

    ''' Show dustmap rho. '''
    st.write(DRHOinterpolated.shape, newX.shape)
    DRHOscaled = DRHOinterpolated.copy()
    DRHOscaled = np.nan_to_num(DRHOscaled)
    DRHOscaled[DRHOscaled>0.1] = 0.
    ColorMask = DRHOscaled>0.001
    ColorMask = ColorMask.ravel()
    fig = px.scatter_3d(x=newX.ravel()[ColorMask], y=newY.ravel()[ColorMask], z=newZ.ravel()[ColorMask], color=DRHOscaled.ravel()[ColorMask], size_max=10, color_continuous_scale="gray", opacity=0.1, title='Interpolated DRHO scaled.')
    fig.update_traces(marker=dict(line=dict(width=0)))
    st.plotly_chart(fig)

    starttime = datetime.now()
    RHOinterpolated = InterpolateOntoCartesian(X,Y,Z, RHO, newX, newY, newZ)
    print(f'Interpolation time {datetime.now() - starttime}')
    st.write(RHOinterpolated.shape)
    writeMRC('rho.mrc', RHOinterpolated.reshape(newX.shape))

    # Plot the histogram to show it isn't drastically different.
    st.plotly_chart(px.histogram(RHOinterpolated.ravel(), title='RHO histogram after interpolation to cartesian grid.'))

    print('-------------------- DONE --------------------')
