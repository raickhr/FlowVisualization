from matplotlib.animation import FuncAnimation, writers
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from fileLocations import *
from functions import *
from configurationFile import *


gridDS = Dataset(fldLoc + '/' + gridFile)
currentDS = Dataset(fldLoc + '/' + surfaceCurrentFile)
stressDS = Dataset(fldLoc + '/' + windStressFile)

KEandEKE = Dataset(fldLoc + '/' + KEandEKEfile)

ugos = np.array(currentDS.variables[ugosVarname])
vgos = np.array(currentDS.variables[vgosVarname])

ugos = np.ma.array(ugos, mask=abs(ugos) > 10, fill_value=0.0).filled()
vgos = np.ma.array(vgos, mask=abs(vgos) > 10, fill_value=0.0).filled()

taux = np.array(stressDS.variables[tauxVarname])
tauy = np.array(stressDS.variables[tauyVarname])

ULAT = np.array(gridDS.variables[ulatVarname])
ULONG = np.array(gridDS.variables[ulongVarname])
KMT = np.array(gridDS.variables[kmtVarname])

EKE = np.array(KEandEKE.variables['avgKE'])

GridXpoints = 180.0 / np.pi * ULONG[0, :]
GridYpoints = 180.0 / np.pi * ULAT[:, 0]

if cornerDefined:
    x1_indx = min(range(len(GridXpoints)),
                  key=lambda i: abs(GridXpoints[i]-x1))
    x2_indx = min(range(len(GridXpoints)),
                  key=lambda i: abs(GridXpoints[i]-x2))

    y1_indx = min(range(len(GridYpoints)),
                  key=lambda i: abs(GridYpoints[i]-y1))
    y2_indx = min(range(len(GridYpoints)),
                  key=lambda i: abs(GridYpoints[i]-y2))

    ugos = ugos[:, y1_indx:y2_indx, x1_indx:x2_indx]
    vgos = vgos[:, y1_indx:y2_indx, x1_indx:x2_indx]
    taux = taux[:, y1_indx:y2_indx, x1_indx:x2_indx]
    tauy = tauy[:, y1_indx:y2_indx, x1_indx:x2_indx]

    ULAT = ULAT[y1_indx:y2_indx, x1_indx:x2_indx]
    ULONG = ULONG[y1_indx:y2_indx, x1_indx:x2_indx]
    KMT = KMT[y1_indx:y2_indx, x1_indx:x2_indx]

    EKE = EKE[y1_indx:y2_indx, x1_indx:x2_indx]

    GridXpoints = GridXpoints[x1_indx:x2_indx]
    GridYpoints = GridYpoints[y1_indx:y2_indx]



EKEmask = EKE > 0.07


highEKEULAT = 180.0 / np.pi * ULAT[EKEmask]
highEKEULONG = 180.0 / np.pi * ULONG[EKEmask]

landMask = KMT < 1

waterULAT = 180.0 / np.pi * ULAT[~landMask]
waterULONG = 180.0 / np.pi * ULONG[~landMask]

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
