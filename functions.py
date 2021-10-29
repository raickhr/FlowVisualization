from matplotlib.pyplot import fill
import numpy as np
from numpy.linalg import inv
from configurationFile import *

def selectRandomFrmList(n, numList):
    ## given a list this function selects n random items from it
    randIndx = np.random.randint(0, len(numList), (n))
    randPoints = numList[randIndx]
    return randPoints

def interpolateIrreg(Q1, Q2, Q3, Q4, x, y, xArr, yArr):
    ### this function bilinearly interpolates inside a quad
    #
    #             (4)---------(3)
    #              |           |
    #              |           |
    #              |           |
    #             (1)---------(2)
    #
    # Q1, Q2, Q3 and Q4 are the function values at nodes 1,2,3 and 4
    # x, y is a co-ordinate inside the quad where the value is to be interpolated

    ## This is for irregular grid interpolation

    A = np.matrix([[1, xArr[0], yArr[0], xArr[0] * yArr[0]],
                   [1, xArr[1], yArr[1], xArr[1] * yArr[1]],
                   [1, xArr[2], yArr[2], xArr[2] * yArr[2]],
                   [1, xArr[3], yArr[3], xArr[3] * yArr[3]]], dtype =float)

    x = np.matrix([1, x, y, x*y], dtype=float )

    phi = np.matrix([[Q1],[Q2],[Q3],[Q4]], dtype =float)

    N = np.matmul(x, inv(A))

    return np.matmul(N, phi)


def interpolateReg(Q1, Q2, Q3, Q4, x, y, dx, dy, xArr, yArr):
    ### this function bilinearly interpolates inside a quad
    #
    #             (4)---------(3)
    #              |           |
    #              |           |
    #              |           |
    #             (1)---------(2)
    #
    # Q1, Q2, Q3 and Q4 are the function values at nodes 1,2,3 and 4
    # x, y is a co-ordinate inside the quad where the value is to be interpolated

    ## This for regular grid interpolation

    x1, y1 = xArr[0], yArr[0]
    x2, y2 = xArr[1], yArr[1]
    x3, y3 = xArr[2], yArr[2]
    x4, y4 = xArr[3], yArr[3] 

    # shape function
    N1 = 1/(dx*dy) * (x - x2) * (y - y4)
    N2 = -1/(dx * dy) * (x - x1) * (y - y3)
    N3 = 1/(dx * dy) * (x - x4) * (y - y2)
    N4 = -1/(dx * dy) * (x - x3) * (y - y1)

    return(N1 * Q1 + N2 * Q2 + N3 * Q3 + N4 * Q4)


def getVelAtPointsRegGrid(xpoints, ypoints, u2D, v2D, GridXpoints, GridYpoints, landMask):
    npoints = len(xpoints)

    if npoints != len(ypoints):
        print('func:getVelAtPoints -> X and Y dimension not equal ')

    dx = GridXpoints[1] - GridXpoints[0]
    dy = GridYpoints[1] - GridYpoints[0]

    Xlen = len(GridXpoints)
    Ylen = len(GridYpoints)

    uvel = np.zeros((npoints), dtype=float)
    vvel = np.zeros((npoints), dtype=float)

    for i in range(npoints):
        if np.isnan(xpoints[i]) or np.isnan(ypoints[i]):
            uvel[i] = 0.0 #float('nan')
            vvel[i] = 0.0 #float('nan')
            #print('skipping ', i, 'nan location for xpoint or ypoint')
            continue

        if xpoints[i] > 360.0:
            xpoints[i] -= 360.0

        x1 = int((xpoints[i] - GridXpoints[0])//dx)
        x2 = x1 + 1

        y1 = int((ypoints[i] - GridYpoints[0])//dy)
        y2 = y1 + 1

        if (y1 < 0) or (y2 > (Ylen - 1)):
            uvel[i] = 0.0  # float('nan')
            vvel[i] = 0.0  # float('nan')
            #print('skipping ', i, 'out of location y index', 'y1', y1)
            continue

        
        if cornerDefined:
            if (x1 < 0) or (x2 > (Xlen - 1)):
                uvel[i] = 0.0  # float('nan')
                vvel[i] = 0.0  # float('nan')
                #print('skipping ', i, 'out of location y index', 'y1', y1)
                continue 
        else:
            if (xpoints[i] < GridXpoints[0] + 360.00) and xpoints[i] > GridXpoints[Xlen-1]:
                x1 = Xlen-1
                x2 = 0


        #print('x1, x2', x1, x2)
        xArr = GridXpoints[[x1, x2, x2, x1]]
        yArr = GridYpoints[[y1, y1, y2, y2]]

        
        uvel[i] = interpolateReg(u2D[y1, x1], u2D[y1, x2], 
                                    u2D[y2, x2], u2D[y2, x1], 
                                    xpoints[i], ypoints[i], dx, dy, xArr, yArr)

        # print('uvel corners', u2D[y1, x1], u2D[y1, x2],
        #       u2D[y2, x2], u2D[y2, x1], '\interpolated u', uvel[i])

        # print('x postions', xArr, xpoints[i])

        vvel[i] = interpolateReg(v2D[y1, x1], v2D[y1, x2],
                                    v2D[y2, x2], v2D[y2, x1],
                                    xpoints[i], ypoints[i], dx, dy, xArr, yArr)

        # print('vvel corners', v2D[y1, x1], v2D[y1, x2],
        #       v2D[y2, x2], v2D[y2, x1], '\interpolated v', vvel[i])

        # print('y postions', yArr, ypoints[i])

        msk = landMask[y1, x1] + landMask[y2, x1] + \
            landMask[y2, x2] + landMask[y1, x2]

        if msk == True:
            uvel[i] = 0.0 #float('nan')
            vvel[i] = 0.0 #float('nan')

        #print(i, 'uvel, vvel', uvel[i], vvel[i])

        # velMag = np.sqrt(uvel**2 + vvel**2)

        # mskMag = velMag < 0.02

        # uvel[mskMag] = float('nan')
        # vvel[mskMag] = float('nan')

    return uvel, vvel        

def moveParticlesReg(xpoints, ypoints, uvel, vvel, dt):
    ## latitude and longitude are in degrees
    ## xpoints and ypoints are in longitude and latitude and in degrees

    earthRad = 6.371e6

    mask = np.isnan(xpoints) + np.isnan(ypoints)

    uvel = np.ma.array(uvel, mask = mask, fill_value=0.0).filled()
    vvel = np.ma.array(vvel, mask = mask, fill_value=0.0).filled()

    xpoints = np.ma.array(xpoints, mask=mask, fill_value=0.0).filled()
    ypoints = np.ma.array(ypoints, mask=mask, fill_value=0.0).filled()

    delx = uvel * dt
    dely = vvel * dt

    radiusArr = earthRad * np.cos(np.radians(ypoints))

    delLat = np.degrees(dely / earthRad)
    delLonInRad = delx / radiusArr
    delLon = np.degrees(delLonInRad)

    #print('max del_lon', np.nanmax(delLon))
    #print('max del_lat', np.nanmax(delLat))

    newXpoints = xpoints + delLon
    newYpoints = ypoints + delLat

    cyclicMask = newXpoints >= 360.00
    newXpoints[cyclicMask] -= 360.00

    cyclicMask = newXpoints < 0.00
    newXpoints[cyclicMask] += 360.00

    newXpoints = np.ma.array(newXpoints, mask = mask, fill_value= float('nan')).filled()
    newYpoints = np.ma.array(newYpoints, mask = mask, fill_value= float('nan')).filled()

    return newXpoints, newYpoints


def updateScatterArray(xslots, yslots, uvelSlots, vvelSlots, 
                       u2D, v2D, GridXpoints, GridYpoints, landMask, 
                       dt, randXpoints, randYpoints):

    historyAxis = 0
    particleAxis = 1 

    ## getting the most recent particles and their velocity
    xpoints = xslots[0, 0:nMaxParts]
    ypoints = yslots[0, 0:nMaxParts]

    uvel = uvelSlots[0, 0:nMaxParts]
    vvel = vvelSlots[0, 0:nMaxParts]

    ## find the new postions and velocities at new postion
    newXpoints, newYpoints = moveParticlesReg(xpoints, ypoints, uvel, vvel, dt)
    newUvel, newVvel = getVelAtPointsRegGrid(
        newXpoints, newYpoints, u2D, v2D, GridXpoints, GridYpoints, landMask)

    ## update the array to push the new positions and velocities
    xslots = np.roll(xslots, 1, axis=historyAxis)
    yslots = np.roll(yslots, 1, axis=historyAxis)

    uvelSlots = np.roll(uvelSlots, 1, axis=historyAxis)
    vvelSlots = np.roll(vvelSlots, 1, axis=historyAxis)

    xslots[0, 0:nMaxParts] = newXpoints
    yslots[0, 0:nMaxParts] = newYpoints

    uvelSlots[0, 0:nMaxParts] = newUvel
    vvelSlots[0, 0:nMaxParts] = newVvel

    ### These are vanishing particles. The old poisition remains but the new position is not there
    xslots[0, nMaxParts:nslots] = float('nan')
    yslots[0, nMaxParts:nslots] = float('nan')

    uvelSlots[0, nMaxParts:nslots] = 0.0
    vvelSlots[0, nMaxParts:nslots] = 0.0

    ### make space for new random points generated 
    xslots = np.roll(xslots, nPartBirth, axis=particleAxis)
    yslots = np.roll(yslots, nPartBirth, axis=particleAxis)

    uvelSlots = np.roll(uvelSlots, nPartBirth, axis=particleAxis)
    vvelSlots = np.roll(vvelSlots, nPartBirth, axis=particleAxis)
    
    xslots[0, 0:nPartBirth] = randXpoints
    yslots[0, 0:nPartBirth] = randYpoints

    uvelSlots[0, 0:nPartBirth], vvelSlots[0, 0:nPartBirth] = getVelAtPointsRegGrid(
        randXpoints, randYpoints, u2D, v2D, GridXpoints, GridYpoints, landMask)

    xslots[1:nhistories, 0:nPartBirth] = float('nan')
    yslots[1:nhistories, 0:nPartBirth] = float('nan')

    uvelSlots[1:nhistories, 0:nPartBirth] = 0.0
    vvelSlots[1:nhistories, 0:nPartBirth] = 0.0

    return xslots, yslots, uvelSlots, vvelSlots













