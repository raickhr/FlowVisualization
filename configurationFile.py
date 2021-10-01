import numpy as np
nPartBirth = 150
nhistories = 200
nMaxParts = nPartBirth * nhistories

nslots = nMaxParts + nhistories
dt = 36000

##left Bottom corners
cornerDefined = True


#gulf
# x1 = 250.0
# y1 = 20.0
# x2 = 350.0
# y2 = 60.0

#Kuroshio
x1 = 125.0
y1 = 20.0
x2 = 225.0
y2 = 60.0

## variable names
ugosVarname = 'ugos'#'UGOS'
vgosVarname = 'vgos'#'VGOS'
tauxVarname = 'eastward_stress'#'TAUX'
tauyVarname = 'northward_stress'#'TAUY'

ulatVarname = 'ULAT'
ulongVarname = 'ULONG'
kmtVarname = 'KMT'



