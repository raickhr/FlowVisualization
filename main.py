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

if rank == 1:
    print("Running the code with {0:d} processors".format(nprocs))

    print("Reading the grid data in processor ", rank)
    grid = gridData()

    print("Reading the velocity data in processor ", rank)
    vel = velData(grid)

    print("Reading the stress data in processor", rank)
    stress = stressData(grid)

    waterULONG, waterULAT = grid.get_waterMaskedLongLat2D()
    GridXpoints, GridYpoints = grid.get_gridXpointsYpoints()
    ugos, vgos = vel.get_UgosVgos_2D()
    taux, tauy = stress.get_TauxTauy()
    timeLen = vel.get_timeLen()
    landMask = grid.get_landMask2D()
    ugosShape = np.shape(ugos)
else:
    timeLen = None
    ugosShape = None
    waterULONG = None
    waterULAT = None
    GridXpoints = None
    GridYpoints = None
    landMask = None

timeLen = comm.bcast(timeLen, root = 1)
ugosShape = comm.bcast(ugosShape, root = 1)
waterULONG = comm.bcast(waterULONG, root=1)
waterULAT = comm.bcast(waterULAT, root=1)
GridXpoints = comm.bcast(GridXpoints, root = 1)
GridYpoints = comm.bcast(GridYpoints, root=1)
landMask = comm.bcast(landMask, root=1)

nlocalPartBirth = nPartBirth//(nprocs)
nlocalMaxParts = nlocalPartBirth * nhistories

nlocalslots = nlocalMaxParts + nhistories

totSlots = nlocalslots * (nprocs)
totPartBirth = nlocalPartBirth * (nprocs)

xslots = np.ones((nhistories, nlocalslots), dtype =float) * float('nan')
yslots = np.ones((nhistories, nlocalslots), dtype =float) * float('nan')

uvelSlots = np.zeros((nhistories, nlocalslots), dtype=float)
vvelSlots = np.zeros((nhistories, nlocalslots), dtype=float)

### to keeep the record of the data in each processor

totXslots = np.ones((timeLen, nhistories, nlocalslots), dtype =float) * float('nan')
totYslots = np.ones((timeLen, nhistories, nlocalslots), dtype =float) * float('nan')
totUvelSlots = np.zeros((timeLen, nhistories, nlocalslots), dtype=float)
totVvelSlots = np.zeros((timeLen, nhistories, nlocalslots), dtype=float)
totBackGround = np.zeros(ugosShape, dtype =float)


### get random points within the water region of the domain each processors

randXpoints = selectRandomFrmList(nlocalPartBirth, waterULONG)
randYpoints = selectRandomFrmList(nlocalPartBirth, waterULAT)

## the starting part of the particles are the random point created above

xslots[0, 0:nlocalPartBirth] = randXpoints
yslots[0, 0:nlocalPartBirth] = randYpoints

## from the read velocity field the velocities at the above location is obtained

if rank == 1:
        uu = np.array(ugos[0, :, :], dtype =float)
        vv = np.array(vgos[0, :, :], dtype =float)
        shape = np.array(np.shape(uu))
        
else:
    shape = np.zeros((2),dtype=int)

comm.Bcast(shape, root= 1)

if rank != 1:
    uu = np.zeros(shape, dtype=float)
    vv = np.zeros(shape, dtype=float)


comm.Bcast(uu, root = 1)
comm.Bcast(vv, root = 1)

comm.Barrier()

uvelSlots[0, 0:nlocalPartBirth], vvelSlots[0, 0:nlocalPartBirth] = getVelAtPointsRegGrid(
    randXpoints, randYpoints, uu, vv, GridXpoints, GridYpoints,
    landMask, cornerDefined)

day = 0
curr_dt = 0

## Open the file to write the particle paths

while day < 2 :#timeLen:

    if rank == 1:
        uu = np.array(ugos[day, :, :], dtype=float)
        vv = np.array(vgos[day, :, :], dtype=float)
        background = np.array(taux[day , :, : ], dtype =float)

    else:
        uu = np.zeros(shape, dtype=float)
        vv = np.zeros(shape, dtype=float)
        background = np.zeros(shape, dtype=float)

    comm.Barrier()

    comm.Bcast(uu, root = 1)
    comm.Bcast(vv, root = 1)
    comm.Bcast(background, root = 1)

    comm.Barrier()
    
    xslots, yslots, uvelSlots, vvelSlots, pltBackground = \
                          getFramDataInEachProc(xslots, yslots,
                          uvelSlots, vvelSlots,
                          nlocalPartBirth,
                          uu, vv, background,
                          GridXpoints, GridYpoints,
                          landMask, waterULONG, waterULAT, dt, 
                          nlocalMaxParts, nlocalslots, nlocalPartBirth, nhistories, cornerDefined)

    ## appending the array for the time series data

    totXslots[day,:,:] = xslots
    totYslots[day,:,:] = yslots
    totUvelSlots[day, :, :] = uvelSlots
    totVvelSlots[day,:,:] = vvelSlots

    day +=1


## close the calcuation loop

## Append data from all processors

## every processor is calculating and storing in variables totXslots, totYslots, totUvelSlots, totVvelSlots

if rank == 0: ## rank is processor id

    allXposData = [totXslots]
    for i in range(1,nprocs):
        data = np.empty(np.shape(totXslots), dtype=float)
        comm.Recv(data, source=i, tag=13 + i*100)
        allXposData.append(data)
        del data
    print('Received Xpos values')
else:
    comm.Send(totXslots, dest=0, tag=13 + rank * 100)

comm.Barrier()

if rank == 0:
    allYposData = [totYslots]
    for i in range(1, nprocs):
        data = np.empty(np.shape(totYslots), dtype=float)
        comm.Recv(data, source=i, tag=13 + i*100)
        allYposData.append(data)
        del data
    print('Received Ypos values')
else:
    comm.Send(totYslots, dest=0, tag=13 + rank * 100)
comm.Barrier()

if rank == 0:
    allUvelData = [totUvelSlots]
    for i in range(1, nprocs):
        data = np.empty(np.shape(totUvelSlots), dtype=float)
        comm.Recv(data, source=i, tag=13 + i*100)
        allUvelData.append(data)
        del data
    print('Received Uvel values')
else:
    comm.Send(totUvelSlots, dest=0, tag=13 + rank * 100)
comm.Barrier()

if rank == 0:
    allVvelData = [totVvelSlots]
    for i in range(1, nprocs):
        data = np.empty(np.shape(totVvelSlots), dtype=float)
        comm.Recv(data, source=i, tag=13 + i*100)
        allVvelData.append(data)
        del data
    print('Received Vvel values')
else:
    comm.Send(totVvelSlots, dest=0, tag=13 + rank * 100)
comm.Barrier()

if rank == 0:
    gatheredXslot = np.concatenate(allXposData,axis=2)
    gatheredYslot = np.concatenate(allYposData,axis=2)
    gatheredUvelSlot = np.concatenate(allUvelData,axis=2)
    gatheredVvelSlot = np.concatenate(allVvelData,axis=2)

    del allXposData, allYposData, allUvelData, allVvelData

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





