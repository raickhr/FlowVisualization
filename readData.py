from netCDF4 import Dataset
import numpy as np
from fileLocations import *
from configurationFile import *


class gridData:
    def __init__(self):
        ## initialize the grid variables.
        gridDS = Dataset(fldLoc + '/' + gridFile)
        ULAT = np.array(gridDS.variables[ulatVarname])
        ULONG = np.array(gridDS.variables[ulongVarname])
        KMT = np.array(gridDS.variables[kmtVarname])

        ## lat and long are 2D arrays
        ## all the columns longitude array is same
        ## all the rows of latitude array is same 
        ## so we need only a row of longitudes and a column of latitudes
        GridXpoints = 180.0 / np.pi * ULONG[0, :]
        GridYpoints = 180.0 / np.pi * ULAT[:, 0]

        ### if the boxed region is defined we need only the latitudes and longitudes within the box
        ### the bottom left corner of the grid is (y1_index, x1_index)
        ### the top right corner of the grid is (y2_index,x2_index)

        if cornerDefined:
            # the grid data is only the data inside the corner
            self.__x1_indx = min(range(len(GridXpoints)),
                          key=lambda i: abs(GridXpoints[i]-x1))
            self.__x2_indx = min(range(len(GridXpoints)),
                          key=lambda i: abs(GridXpoints[i]-x2))

            self.__y1_indx = min(range(len(GridYpoints)),
                          key=lambda i: abs(GridYpoints[i]-y1))
            self.__y2_indx = min(range(len(GridYpoints)),
                          key=lambda i: abs(GridYpoints[i]-y2))


            self.__ULAT = ULAT[self.__y1_indx:self.__y2_indx,
                             self.__x1_indx:self.__x2_indx]
            self.__ULONG = ULONG[self.__y1_indx:self.__y2_indx,
                               self.__x1_indx:self.__x2_indx]
            self.__KMT = KMT[self.__y1_indx:self.__y2_indx,
                           self.__x1_indx:self.__x2_indx]

            self.__GridXpoints = GridXpoints[self.__x1_indx:self.__x2_indx]
            self.__GridYpoints = GridYpoints[self.__y1_indx:self.__y2_indx]

        else:
            ### the grid data is the whole data
            self.__ULAT = ULAT
            self.__ULONG = ULONG
            self.__KMT = KMT
            self.__GridXpoints = GridXpoints
            self.__GridYpoints = GridYpoints

        self.__landMask = self.__KMT < 1

        self.__waterULAT = 180.0 / np.pi * self.__ULAT[~self.__landMask]
        self.__waterULONG = 180.0 / np.pi * self.__ULONG[~self.__landMask]

    def get_longlat2D(self):
        return self.__ULONG, self.__ULAT

    def get_waterMaskedLongLat2D(self):
        return self.__waterULONG, self.__waterULAT 

    def get_LongLat1D(self):
        return self.__GridXpoints, self.__GridYpoints

    def get_landMask2D(self):
        return self.__landMask

    def get_returnCornerIndices_y1x1y2x2(self):
        return self.__y1_indx, self.__x1_indx, self.__y2_indx, self.__x2_indx


class velData:
    def __init__(self, grid):
        currentDS = Dataset(fldLoc + '/' + surfaceCurrentFile)
        ugos = np.array(currentDS.variables[ugosVarname])
        vgos = np.array(currentDS.variables[vgosVarname])
        ugos = np.ma.array(ugos, mask=abs(ugos) > 10, fill_value=0.0).filled()
        vgos = np.ma.array(vgos, mask=abs(vgos) > 10, fill_value=0.0).filled()

        if cornerDefined:
            y1_indx, x1_indx, y2_indx, x2_indx = grid.get_returnCornerIndices_y1x1y2x2()
            self.__ugos = ugos[:, y1_indx:y2_indx, x1_indx:x2_indx]
            self.__vgos = vgos[:, y1_indx:y2_indx, x1_indx:x2_indx]
            
        else:
            self.__ugos = ugos
            self.__vgos = vgos

    def get_UgosVgos_2D(self):
        return self.__ugos, self.__vgos


class stressData:
    def __init__(self, grid):
        stressDS = Dataset(fldLoc + '/' + windStressFile)
        taux = np.array(stressDS.variables[tauxVarname])
        tauy = np.array(stressDS.variables[tauyVarname])

        if cornerDefined:
            y1_indx, x1_indx, y2_indx, x2_indx = grid.get_returnCornerIndices_y1x1y2x2()
            self.__taux = taux[:, y1_indx:y2_indx, x1_indx:x2_indx]
            self.__tauy = tauy[:, y1_indx:y2_indx, x1_indx:x2_indx]
        else:
            self.__taux = taux
            self.__tauy = tauy

    
    def get_TauxTauy(self):
        return self.__taux, self.__tauy
