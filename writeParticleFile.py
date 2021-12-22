from netCDF4 import Dataset
import numpy as np

def writeParticleFile(fullFileNameWithPath,dimensonList,dimLenList, varnames, vardimensions, varvalues):

    ## This function writes the netcdf file for write the particle paths

    ds = Dataset(fullFileNameWithPath,'w',format='NETCDF4_CLASSIC')

    nDims = len(dimensonList)

    ### The dimensions will be Time, the fading history length and the number of particles for the partilce paths and 
    ### latitude and longtitude for the background plot

    for i in range(nDims):
        dimName = dimensonList[i]
        dimLen = dimLenList[i]

        ds.createDimension(dimName,dimLen)

    for varNum in range(len(varnames)):
        varName = varnames[varNum]
        varValue = varvalues[varNum]
        varDimension = vardimensions[varNum]

        var = ds.createVariable[varName, float, varDimension]
        var[:] = varValue[:]


    ds.close()


# def appendFile(fullFileNameWithPath, varnames, varvalues, appendTime):
#     ds = Dataset(fullFileNameWithPath)

#     for var in varnames:
        