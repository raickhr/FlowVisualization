import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from functions import *


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 12))
if cornerDefined:
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)
    fig.set_size_inches((x2-x1) * 12/(y2-y1), 12)
else:
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)

cmap = plt.cm.get_cmap('seismic')
cmap.set_bad('k')
pmesh = ax.pcolormesh(GridXpoints, GridYpoints, taux[0, :, :],
                      cmap=cmap, vmin=-1, vmax=1)
cb = plt.colorbar(pmesh)

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
        velmag = getVelMag(
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



