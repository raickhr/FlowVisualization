from netCDF4 import Dataset
import numpy as np
from functions import *
from configurationFile import *
from readData import *
from writeParticleFile import *

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


nlocalslots = nslots//(nprocs)
nlocalPartBirth = nPartBirth//(nprocs)

totSlots = nlocalslots * (nprocs)
totPartBirth = nlocalPartBirth * (nprocs)

waterULONG, waterULAT = grid.get_waterMaskedLongLat2D()
GridXpoints, GridYpoints = grid.get_gridXpointsYpoints()
ugos, vgos = vel.get_UgosVgos_2D()
taux, tauy = stress.get_TauxTauy()
timeLen = vel.get_timeLen()
landMask = grid.get_landMask2D()

xslots = np.ones((nhistories, nlocalslots), dtype =float) * float('nan')
yslots = np.ones((nhistories, nlocalslots), dtype =float) * float('nan')

uvelSlots = np.zeros((nhistories, nlocalslots), dtype=float)
vvelSlots = np.zeros((nhistories, nlocalslots), dtype=float)

### to keeep the record of the data in each processor

totXslots = np.ones((timeLen, nhistories, nlocalslots), dtype =float) * float('nan')
totYslots = np.ones((timeLen, nhistories, nlocalslots), dtype =float) * float('nan')
totUvelSlots = np.zeros((timeLen, nhistories, nlocalslots), dtype=float)
totVvelSlots = np.zeros((timeLen, nhistories, nlocalslots), dtype=float)
totBackGround = np.shape(ugos)

### get random points within the water region of the domain

randXpoints = selectRandomFrmList(nlocalPartBirth, waterULONG)
randYpoints = selectRandomFrmList(nlocalPartBirth, waterULAT)

## the starting part of the particles are the random point created above

xslots[0, 0:nlocalPartBirth] = randXpoints
yslots[0, 0:nlocalPartBirth] = randYpoints

## from the read velocity field the velocities at the above location is obtained

uvelSlots[0, 0:nlocalPartBirth], vvelSlots[0, 0:nlocalPartBirth] = getVelAtPointsRegGrid(
    randXpoints, randYpoints, ugos[0, :, :], vgos[0, :, :], GridXpoints, GridYpoints,
    landMask)

day = 0
curr_dt = 0

## Open the file to write the particle paths

while day < timeLen:

    uu = ugos[day, :, :]
    vv = vgos[day, :, :]
    background = taux[day , :, : ]

    
    xslots, yslots, uvelSlots, vvelSlots, pltBackground = \
                          getFramDataInEachProc(xslots, yslots,
                          uvelSlots, vvelSlots,
                          nlocalPartBirth,
                          uu, vv, background,
                          GridXpoints, GridYpoints,
                          landMask, waterULONG, waterULAT)

    ## appending the array for the time series data

    totXslots[day,:,:] = xslots
    totYslots[day,:,:] = yslots
    totUvelSlots[day, :, :] = uvelSlots
    totVvelSlots[day,:,:] = vvelSlots


## close the calcuation loop

## Append data from all processors
gatheredXslot = comm.gather(totXslots,root=0)
gatheredYslot = comm.gather(totYslots, root=0)
gatheredUvelSlot = comm.gather(totUvelSlots, root=0)
gatheredVvelSlot = comm.gather(totVvelSlots, root=0)

if rank == 0:
    gatheredXslot = np.concatenate(gatheredXslot,axis=2)
    gatheredYslot = np.concatenate(gatheredYslot,axis=2)
    gatheredUvelSlot = np.concatenate(gatheredUvelSlot,axis=2)
    gatheredVvelSlot = np.concatenate(gatheredVvelSlot,axis=2)

    dimensonList = ['Time', 'history', 'PID']
    dimLenList = [None, nhistories, totSlots]

    varnames = ['Xpos', 'Ypos', 'uvel', 'vvel']
    vardimensons = [('Time', 'history', 'PID'),
                    ('Time', 'history', 'PID'),
                    ('Time', 'history', 'PID'),
                    ('Time', 'history', 'PID')]

    varvalues = [gatheredXslot, gatheredYslot, gatheredVvelSlot, gatheredVvelSlot]

    fullFileNameWithPath = outLoc +'/' +  outFile
    
    writeParticleFile(fullFileNameWithPath, dimensonList,
                      dimLenList, varnames, vardimensons, varvalues)

### write the particle file in netcdf format


### Start the frame writing loop and write frame in parallel

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





