import numpy as np
import matplotlib.pyplot as plt
import pricingLib as lib


def chooseStrikeBound(T):
    if T < 0.1:
        return [75, 115]
    elif T < 0.3:
        return [68, 130]
    elif T < 1.0:
        return [60, 150]
    elif T < 2.0:
        return [40, 180]
    else:
        return [25, 200]

Maturity = [0.041, 0.12, 0.15, 0.19, 0.29, 0.38, 0.40, 0.62, 0.65, 0.87, 0.90, 1.36, 1.86, 2.88, 3.0]

numT = len(Maturity)


Expir = Maturity[numT-1]
m = 5000       #number of time steps
n = 5000         #number of simulation
rho = -0.9
H = 0.07
xi0 = 0.09
eta = 1.9
S0 = 100.0
r = 0
tolerence = [0.0001, 0.0001]  #tol[0] for the x tolerence, tol[1] for the y tolerence


covMat = lib.covarianceGenerator(Expir, m, rho, H)  #compute the covariance matrix (Wt, Z)
L = lib.choleskyDecom(covMat)   #Cholesky decomposition
normalMat = np.random.randn(2*m, n) 
WtZ = np.dot(L, normalMat)  #(Wt, Z) = L * G
Z = np.zeros((n, m+1))
Wt = np.zeros((n, m+1))
Z[:,0] = 0
Wt[:,0] = 0
Wt[:,1:m+1] = WtZ.T[:,0:m]
Z[:,1:m+1] = WtZ.T[:,m:2*m]


#Euler shema for simulating variance and price 
variance = lib.simuVariance(xi0, eta, Wt, Expir, H)
assetPrice = lib.simuAssetPrice(S0, Z, variance, Expir, r)

Kbound = {}
result = {}

for i, T in enumerate(Maturity):
    
    KBound = chooseStrikeBound(T)
    K = np.arange(KBound[0], KBound[1], 1)
    putValue = np.zeros(len(K))
    volImpliedPut = np.zeros(len(K))   
    index = int(T*m/Expir)
    for j, k in enumerate(K):
        term = lib.positivePart(k - assetPrice[:,index])
        putValue[j] = np.mean(term)
        volImpliedPut[j] = lib.impliedVol("put", putValue[j], S0, T, k, r, tolerence)
    
    Kbound[i] = KBound
    result[i] = volImpliedPut

    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(r'T = {}'.format(T))
    ax.set_xlabel(r'Log-Strike')
    ax.set_ylabel(r'Implied Vol.')
    ax.plot(np.log(K/S0), volImpliedPut, 'b')
    '''

f, axarr = plt.subplots(3, 5)
for i in range(15):
    row = i / 5
    col = i % 5
    K = np.arange(Kbound[i][0], Kbound[i][1], 1)
    axarr[row, col].set_title('T = {}'.format(Maturity[i]))
    axarr[row, col].set_xlabel('Log-Strike')
    axarr[row, col].set_ylabel('Implied Vol')
    axarr[row, col].plot(np.log(K/S0), result[i])

plt.show()

