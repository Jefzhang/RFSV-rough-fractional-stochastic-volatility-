import numpy as np
import matplotlib.pyplot as plt
import simulibraries as rfsv
import math
from fbm import FBM

H = 0.5
v = 0.3
m = -5.0
X0 = -5.0
P0 = 10.0
alpha = 1e-4
Ndays = 2000
delta = 1.0/4000

output = open('simulation1.txt','w')


n = int(np.floor(Ndays/delta))

f = FBM(n=n, hurst=H, length=int(Ndays), method='daviesharte')
fgn_sample = f.fgn()


startTime = 8
endTime = 16
sampleFreInMinuts = 5

X = rfsv.simXEuler(X0, fgn_sample, v, alpha, m, delta, Ndays)

sigma = np.exp(X)
P = rfsv.simPriceEuler(P0, sigma, delta, Ndays)

plt.figure(0)
time = np.arange(0, Ndays+delta, delta)
plt.xlabel('t')
plt.ylabel(r'$\sigma$')
plt.plot(time, sigma)

plt.figure(1)
plt.xlabel('t')
plt.ylabel(r'$Y_t$')
plt.plot(time, P)

plt.figure(2)
plt.xlabel('t')
plt.ylabel(r'$X_t$')
plt.plot(time, X)


#P = rfsv.simPrice(P0, sigma, delta, Ndays)
'''
logP = np.log(P)
realizedVariance = rfsv.realizedVariance(logP, delta, Ndays, sampleFreInMinuts, startTime, endTime)
volatility = np.sqrt(realizedVariance)
logVola = np.log(volatility)

#output.write('Realized variance:\n')
#np.savetxt(output, realizedVariance, fmt='%.8f', delimiter=';')


Delta = np.arange(1,50)
Q = np.array([0.5, 1, 1.5, 2, 3])

output.write('m(delta)\n')
m_delta = np.zeros((len(Q),len(Delta)))
for i in range(len(Q)):
    q = Q[i]
    for j in range(len(Delta)):
        m_delta[i][j] = rfsv.computeM(q, Delta[j], logVola)
    output.write('q = '+str(Q[i])+' :\n')
    np.savetxt(output, m_delta[i,:], fmt='%.8f', delimiter='; ')

logDelta = np.log(Delta)
logM = np.log(m_delta)

Beta0 = np.zeros(len(Q))
Beta1 = np.zeros(len(Q))
plt.figure(0)
plt.xlabel(r'$log(\bigtriangleup)$')
plt.ylabel(r'$log(m(q,\bigtriangleup))$')

for i in range(logM.shape[0]):
    (beta0, beta1) = rfsv.linearRegression(logDelta, logM[i,:])
    Beta0[i] = beta0
    Beta1[i] = beta1
    plt.plot(logDelta, beta0+beta1*logDelta, 'r-')
    plt.plot(logDelta, logM[i,:],'*', label='q = '+str(Q[i]))
plt.legend(loc = 'best')

output.write('Simulation regression result:\n')
np.savetxt(output, Q, fmt='%.2f' )
np.savetxt(output, Beta1, fmt='%.5f', newline='\n')

output.close()


plt.figure(1)
h1 = np.mean(Beta1/Q)

Q = np.insert(Q, 0, 0)
Beta1 = np.insert(Beta1, 0, 0)
plt.xlabel(r'$q$')
plt.ylabel(r'$\zeta$')
plt.plot(Q, Beta1, 'r', label=r'Simulation result')
plt.plot(Q, h1*Q, 'b', label=r'y = {0:.3f} * q'.format(h1))
plt.legend(loc='best')
'''

plt.show()


