import numpy as np
import matplotlib.pyplot as plt
import pricingLib as lib

T = 2.3
m = 3000        #number of time steps
n = 3000         #number of simulation
rho = -0.9
H = 0.06
xi0 = 0.09
eta = 1.9
S0 = 100.0
r = 0
tolerence = [0.0001, 0.0001]

espilonk = 0.015


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

#Euler shema for simulating variance and price 
variance = lib.simuVariance(xi0, eta, Wt, T, H)
assetPrice = lib.simuAssetPrice(S0, Z, variance, T, r)


#atmSkew = np.zeros(m)
start = 100
step = 10
atmSkew = lib.ATMSkew('put', assetPrice, espilonk, tolerence, S0, T, start, step, r)


'''
atmSkewSample = np.zeros((m-start)/step+1)
for i in range(len(atmSkewSample)):
    atmSkewSample[i] = atmSkew[start-1 + i*step]
'''
time = np.arange(start,m+1,step)*T/m
plt.xlabel(r'Time to expiry $\tau$')
#plt.ylabel(r'$\psi(\tau)$')
plt.plot(time, atmSkew, 'r-', label = r'Simulated result, $\tilde{\psi}(\tau)$')

alpha = 0.5 - H
A = np.mean(np.log(atmSkew)+alpha*np.log(time))
#[A, B] = lib.linearRegression(np.log(time), np.log(atmSkew))

regreY = np.exp(A)/time**(alpha)
#empiricalY = A/time**(alpha)
plt.plot(time, regreY, 'b-', label = r'$a/\tau^\tau$, a = {0:.3f}'.format(np.exp(A)))#+r'b = {0:.3f}'.format(-B))
#plt.plot(time, empiricalY, 'g-', label = r'{0:.2f}/$\tau^\gamma$'.format(A))
plt.legend()
plt.show()
