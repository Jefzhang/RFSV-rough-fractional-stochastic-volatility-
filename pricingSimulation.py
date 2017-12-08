import numpy as np
import matplotlib.pyplot as plt
import pricingLib as lib


T = 0.7
m = 2000        #number of time steps
n = 2000         #number of simulation
rho = -0.8
H = 0.07
xi0 = 0.09
eta = 1.9
S0 = 100.0
r = 0
tolerence = [0.0001, 0.00001]


covMat = lib.covarianceGenerator(T, m, rho, H)  #compute the covariance matrix (Wt, Z)
L = lib.choleskyDecom(covMat)   #Cholesky decomposition
normalMat = np.random.randn(2*m, n) 

WtZ = np.dot(L, normalMat)  #(Wt, Z) = L * G

Z = np.zeros((n, m+1))
Wt = np.zeros((n, m+1))
Z[:,0] = 0
Wt[:,0] = 0
Wt[:,1:m+1] = WtZ.T[:,0:m]
Z[:,1:m+1] = WtZ.T[:,m:2*m]


#Plot samples of Wt
'''
plt.figure(0)
plt.title(r'10 samples of $\tilde{W}_t$')
plt.xlabel(r'$(t_i)_{i=0,...,m}$')
plt.plot(Wt[0:10,:].T)


#Plot samples of Z
plt.figure(1)
plt.title(r'10 samples of $Z_t$')
plt.xlabel(r'$(t_i)_{i=0,...,m}$')
plt.plot(Z[0:10,:].T)
'''

#Euler shema for simulating variance and price 
variance = lib.simuVariance(xi0, eta, Wt, T, H)
assetPrice = lib.simuAssetPrice(S0, Z, variance, T, r)

'''
plt.figure(2)
plt.title(r'10 samples of $S_t$')
plt.xlabel(r'$(t_i)_{i=0,...,m}$')
plt.plot(assetPrice[0:10,:].T)

plt.figure(3)
plt.title(r'1 sample of valatility path')
plt.plot(np.sqrt(variance[0,:]))
'''

K = np.arange(60, 150, 0.2)
callValue = np.zeros(len(K))
putValue = np.zeros(len(K))
halfInterCall = np.zeros(len(K))
halfInterPut = np.zeros(len(K))

volImpliedCall = np.zeros(len(K))
volImpliedPut = np.zeros(len(K))

for i, k in enumerate(K):
    term1 = lib.positivePart(assetPrice[:,m] - k)
    term2 = lib.positivePart(k - assetPrice[:,m])
    callValue[i] = np.mean(term1)
    putValue[i] = np.mean(term2)
    halfInterCall[i] = 1.96 * np.sqrt(np.var(term1)) / np.sqrt(n)
    halfInterPut[i] = 1.96 * np.sqrt(np.var(term2)) / np.sqrt(n)
    volImpliedCall[i] = lib.impliedVol("call", callValue[i], S0, T, k, r, tolerence)
    volImpliedPut[i] = lib.impliedVol("put", putValue[i], S0, T, k, r, tolerence)



plt.figure(4)
#plt.title(r'Monte carlo approximation of call price, N = '+str(n))
plt.xlabel('K')
plt.plot(K, callValue, 'r', label= 'European call price')
plt.plot(K, callValue + halfInterCall, 'b--', label = 'Call confidence interval')
plt.plot(K, callValue - halfInterCall, 'b--')
plt.plot(K, putValue, 'k', label = 'European put price')
plt.plot(K, putValue + halfInterPut, 'g--', label = 'Put confidence interval')
plt.plot(K, putValue - halfInterPut, 'g--')
plt.legend(loc = 'best')

plt.figure(5)
plt.title(r'SPX Call Option Implied Volatility with T = '+str(T))
plt.xlabel(r'Log-Strike')
plt.ylabel(r'Implied Vol.')
plt.plot(np.log(K/S0), volImpliedCall, 'r', label = "European call")
plt.plot(np.log(K/S0), volImpliedPut, 'b', label="European put")
plt.legend()

plt.figure(6)
plt.xlabel('K')
plt.plot(K, S0-K*np.exp(-r*T), 'r', label=r'$S0 - K\exp(-rT)$')
plt.plot(K, callValue - putValue, 'b', label = r'$C - P$')
plt.legend()

'''
plt.figure(6)
plt.title(r'Monte carlo approximation of put price, N = '+str(n))
plt.xlabel('K')
plt.plot(K, putValue, 'r', label= 'Put price')
plt.plot(K, putValue + halfInterPut, 'b', label = 'Confidence interval')
plt.plot(K, putValue - halfInterPut, 'b')
plt.legend(loc = 'best')


plt.figure(7)
plt.title(r'SPX Put Option Implied Volatility with T = '+str(T))
plt.xlabel(r'Log-Strike')
plt.ylabel(r'Implied Vol.')
'''
plt.figure(7)
plt.xlabel(r'Log-Strike')
plt.ylabel(r'Implied Vol.')
plt.plot(np.log(K/S0), volImpliedPut, 'b', label="put")
#plt.legend()







plt.show()