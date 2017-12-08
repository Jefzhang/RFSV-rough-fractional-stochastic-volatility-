import numpy as np
import matplotlib.pyplot as plt
import simulibraries as rfsv
import math
from fbm import FBM

hurst = [0.08, 0.10, 0.14, 0.20, 0.30, 0.40, 0.50, 0.60, 0.7, 0.8]
hurstSimu = np.zeros(len(hurst))
v = 0.3
m = -5.0
X0 = -5.0
P0 = 100.0
alpha = 5e-4
Ndays = 2000
delta = 1.0/4000

output = open('simulation1.txt','w')
n = int(np.floor(Ndays/delta))

for i, H in enumerate(hurst):
    f = FBM(n=n, hurst=H, length=int(Ndays), method='daviesharte')
    fgn_sample = f.fgn()

    startTime = 8
    endTime = 16
    sampleFreInMinuts = 5

    X = rfsv.simXEuler(X0, fgn_sample, v, alpha, m, delta, Ndays)
    sigma = np.exp(X)
    P = rfsv.simPriceEuler(P0, sigma, delta, Ndays)

    logP = np.log(P)
    realizedVariance = rfsv.realizedVariance(logP, delta, Ndays, sampleFreInMinuts, startTime, endTime)
    volatility = np.sqrt(realizedVariance)
    logVola = np.log(volatility)

    Delta = np.arange(1,50)
    Q = np.array([0.5, 1, 1.5, 2, 3])
    m_delta = np.zeros((len(Q),len(Delta)))
    for r in range(len(Q)):
        q = Q[r]
        for c in range(len(Delta)):
            m_delta[r][c] = rfsv.computeM(q, Delta[c], logVola)
        #output.write('q = '+str(Q[i])+' :\n')
        #np.savetxt(output, m_delta[i,:], fmt='%.8f', delimiter='; ')
    logDelta = np.log(Delta)
    logM = np.log(m_delta)

    Beta0 = np.zeros(len(Q))
    Beta1 = np.zeros(len(Q))
    for j in range(logM.shape[0]):
        (beta0, beta1) = rfsv.linearRegression(logDelta, logM[j,:])
        Beta0[j] = beta0
        Beta1[j] = beta1
    print i , np.mean(Beta1/Q)
    hurstSimu[i] = np.mean(Beta1/Q)

output.write('Original hurst parameter :\n')
np.savetxt(output, hurst, fmt='%.2f', newline='\n')
output.write('Hurst parameter got by simulation :\n')
np.savetxt(output, hurstSimu, fmt='%.2f', newline='\n')