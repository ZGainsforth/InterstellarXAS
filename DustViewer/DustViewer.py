import dustmaps.bayestar
from dustmaps.bayestar import BayestarQuery
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from dask.distributed import Client, wait, as_completed

def InitializeDustmap():
    # Initialize dustmaps using Bayestar2019
    print('Initializing Bayestar2019.')
    #dustmaps.bayestar.fetch()
    bayestar = BayestarQuery(max_samples=1)
    return bayestar

# Function to convert our cartesian coordinates to spherical coordinates (i.e. galactic coordinates).
def Cartesian2Spherical(x,y,z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r) * 180/np.pi 
    phi = np.arctan2(y,x) * 180/np.pi + 180
    theta = np.nan_to_num(theta)
    phi = np.nan_to_num(phi)
    # phi[phi==360] = 0
    if(phi == 360):
        phi = 0
    return r, theta, phi

# A function for computing the density for a single pixel using Bayestar.
def GetDensity(x,y,z, bayestar):
    # def GetDensity(r, phi, theta):
    r, theta, phi = Cartesian2Spherical(x,y,z)
    c = SkyCoord(phi*u.deg, (theta-90)*u.deg, r*u.kpc, frame='galactic')
    rho = bayestar(c, mode='best')
    rho = np.nan_to_num(rho)
    return x,y,z, rho

if __name__ == "__main__":

    print('-------------------- START --------------------')

    bayestar = InitializeDustmap()

    # client = Client()
    # remote_bayestar = client.scatter(bayestar)

    # Choose a volume in which we will generate our 3D dust map.  Units are kpc.
    selectiondistance = 15 # Distance from origin to center of one side of the box.
    selectiondistanceincrement = 1 # Voxel diameter in kpc.

    # Make a box, however many kiloparsecs across. 
    v = np.linspace(-selectiondistance, selectiondistance, int(2*selectiondistance/selectiondistanceincrement+1))
    X,Y,Z = np.meshgrid(v, v, v)
    XYZ = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)

    print(f'Creating a cube {2*selectiondistance} kpc wide, with {selectiondistanceincrement} kpc voxels.')

    # # Get the spherical coordinates for each voxel.
    # r,theta,phi = Cartesian2Spherical(X,Y,Z)
    # print('r in range: ', np.min(r), np.max(r))
    # print('theta in range: ', np.min(theta), np.max(theta))
    # print('phi in range: ', np.min(phi), np.max(phi))

    NumVoxels = X.size
    print(f'Total number of voxels to compute is: {NumVoxels}')

    # Make an cube to hold the computed densities.
    rho = np.zeros(Z.shape)

    print(GetDensity(100,0,0, bayestar))
    VoxelsDone = 0
    for i in range(len(v)):
        for j in range(len(v)):
            for k in range(len(v)):
                if (VoxelsDone % 100) == 0: 
                    print(f'Completed {VoxelsDone} of {NumVoxels}, {VoxelsDone/NumVoxels*100:0.0f}%', end='\r')
                VoxelsDone += 1
                result = GetDensity(X[i,j,k], Y[i,j,k], Z[i,j,k], bayestar)
                # print(i,j,k, result)
                rho[i,j,k] = result[-1]

    # Doh, not enough RAM do load a bayestar object for each CPU core.  Can't parallelize.  *sniff*

    # # Make a list to hold the futures for the dask compute.
    # futures = []
    # for x,y,z in XYZ:
    #     futures.append(client.submit(GetDensity, x,y,z))
    # print('Queuing complete.')

    # VoxelsDone = 0
    # for f in as_completed(futures):
    #     if (VoxelsDone % 100) == 0: 
    #         print(f'Completed {VoxelsDone} of {NumVoxels}, {VoxelsDone/NumVoxels*100:0.0f}%', end='\r')
    #     VoxelsDone += 1
    #     res = f.result()
    #     rho[res[0], res[1], res[2]] = res[3]

    # client.close()
    np.save('rho.npy', rho)

    print('-------------------- DONE --------------------')
