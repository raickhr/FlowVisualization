from matplotlib.animation import FuncAnimation, writers
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from functions import *
from configurationFile import *
from readData import *

from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

if rank == 0:
    print("Running the code with {0:d} processors".format(nprocs))

    print("Reading the grid data in processor 0")
    grid = gridData()

    print("Reading the velocity data in processor 0")
    vel = velData(grid)

    print("Reading the stress data in processor 0")
    stress = stressData(grid)

else:
    grid = None
    vel = None
    stress = None

grid = comm.bcast(grid, root=0)
vel = comm.bcast(vel, root=0)
stress = comm.bcast(stress, root=0)
if rank == 0:
    print("grid, velocity and stress data shared across all processors")




MPI.Finalize()
sys.exit()
##############################
##############################

xslots = np.ones((nhistories, nslots)) * float('nan')
yslots = np.ones((nhistories, nslots)) * float('nan')

uvelSlots = np.zeros((nhistories, nslots))
vvelSlots = np.zeros((nhistories, nslots))

randXpoints = selectRandomFrmList(nPartBirth, waterULONG)
randYpoints = selectRandomFrmList(nPartBirth, waterULAT)

xslots[0, 0:nPartBirth] = randXpoints
yslots[0, 0:nPartBirth] = randYpoints

uvelSlots[0, 0:nPartBirth], vvelSlots[0, 0:nPartBirth] = getVelAtPointsRegGrid(
    randXpoints, randYpoints, ugos[0, :,
                                   :], vgos[0, :, :], GridXpoints, GridYpoints,
    landMask)



fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 12))
if cornerDefined:
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)
    fig.set_size_inches((x2-x1)* 12/(y2-y1), 12)
else:   
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)

cmap = plt.cm.get_cmap('seismic')
cmap.set_bad('k')
pmesh = ax.pcolormesh(GridXpoints, GridYpoints, taux[0, :, :],
                      cmap=cmap, vmin=-1, vmax=1)
cb = plt.colorbar(pmesh)


def getSize(uvel, vvel, mask):
    velMag = np.sqrt(uvel**2, vvel**2)
    velMag = np.ma.array(velMag, mask=mask, fill_value=0.0).filled()
    velMag = np.ma.array(velMag, mask=velMag > 1.0, fill_value=1.0).filled()
    #velMag /= 2

    return (velMag)**2 * 0.4


scatList = []
for i in range(nhistories):
    velmag = getSize(uvelSlots[i, :], vvelSlots[i, :], np.isnan(xslots[i, :]))
    scat = ax.scatter(xslots[i, :], yslots[i, :], s=velmag,
                      c='darkgreen', alpha=(1.0 - i/nhistories)**6)  # 1 - (100-abs(100-i)) * 0.01)  # c=mem_particle[0,:, :]
    scatList.append(scat)


def init():
    pmesh.set_array(taux[0, :-1, :-1].flatten())
    for i in range(nhistories):
        arr = np.stack((xslots[i, :], yslots[i, :]), axis=0).transpose()
        velmag = getSize(
            uvelSlots[i, :], vvelSlots[i, :], np.isnan(xslots[i, :]))
        scatList[i].set_offsets(arr)
        scatList[i].set_sizes(velmag)
    return scatList, pmesh


def update(i):
    print(i)
    global xslots, yslots, uvelSlots, vvelSlots, ugos, vgos, GridXpoints, \
        GridYpoints, landMask, nPartBirth, waterULAT, waterULONG, \
        highEKEULONG, highEKEULAT

    day = i // 10 + 1
    uu = ugos[day, :, :]
    vv = vgos[day, :, :]

    pltTaux = np.ma.array(taux[day, :, :], mask=landMask,
                          fill_value=float('nan')).filled()

    # randXpoints = selectRandomFrmList(nPartBirth, highEKEULONG)
    # randYpoints = selectRandomFrmList(nPartBirth, highEKEULAT)

    randXpoints = selectRandomFrmList(nPartBirth, waterULONG)
    randYpoints = selectRandomFrmList(nPartBirth, waterULAT)

    # if i%3 == 0:
    #     randXpoints = selectRandomFrmList(nPartBirth, waterULONG)
    #     randYpoints = selectRandomFrmList(nPartBirth, waterULAT)

    xslots, yslots, uvelSlots, vvelSlots = updateScatterArray(xslots, yslots,
                                                              uvelSlots, vvelSlots, uu, vv,
                                                              GridXpoints, GridYpoints,
                                                              landMask, dt,
                                                              randXpoints, randYpoints)

    pmesh.set_array(pltTaux[:-1, :-1].flatten())

    for i in range(nhistories):
        arr = np.stack((xslots[i, :], yslots[i, :]), axis=0).transpose()
        velmag = getSize(
            uvelSlots[i, :], vvelSlots[i, :], np.isnan(xslots[i, :]))
        scatList[i].set_offsets(arr)
        scatList[i].set_sizes(velmag)
    return scatList, pmesh
    # fig.clear()
    # ax.set_xlim(0, 360)
    # ax.set_ylim(-90, 90)

    # cb = fig.colorbar(pmesh)
    # shootinStarVecplot(xslots, yslots)


def getAnimation(Title):
    animation = FuncAnimation(
        fig, update, init_func=init, frames=900, interval=300)
    animation.save(Title + '_test.gif', writer='imagemagick', fps=30)


getAnimation('Kuroshio')
