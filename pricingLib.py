import numpy as np
import scipy.special as ssp
from scipy.stats import norm
from Bisect import bisect
#from mpmath import *
import matplotlib.pyplot as plt

#------------------------------------
#T: maturity date
#m : number of time steps
#rho : correlation ratio
# H : fractional brownian motion parameter
#----------------------------------------
def covarianceGenerator(T, m, rho, H):
    deltaT = T / m
    cov = np.zeros((2*m, 2*m))
    for i in range(m):
        cov[i, i] = ((i+1)*deltaT) ** (2*H)
        cov[i+m, i+m] = (i+1)*deltaT
    for i in range(m):
        for j in range(i+1, m):
            cov[i, j] = covarianceW((i+1)*deltaT, (j+1)*deltaT, H)
            cov[i+m, j+m] = covarianceZ((i+1)*deltaT, (j+1)*deltaT)
    for i in range(m):
        for j in range(m, 2*m):
            cov[i, j] = covarianceWZ((i+1)*deltaT, (j+1-m)*deltaT, rho, H)
    return cov + cov.T - np.diag(cov.diagonal())

#t2>=t1 
def covarianceW(t1, t2, H):
    a = t1 ** (2*H)
    gamma = 0.5 - H
    return a * G(t2/t1, gamma)

def G(x, gamma):
    return (1-2*gamma)/(1 - gamma) * (1.0/x)**gamma * ssp.hyp2f1(gamma, 1, 2 - gamma, 1.0/x) 

def covarianceZ(t1, t2):
    return np.minimum(t1, t2)


def covarianceWZ(t1, t2, rho, H):
    D = np.sqrt(2*H) / (H + 0.5)
    a = t1**(H+0.5)
    b = (t1 - np.minimum(t1, t2)) ** (H + 0.5)
    return rho * D * (a - b)

def choleskyDecom(covMat):
    return np.linalg.cholesky(covMat)


def simuVariance(xi0, eta, W, T, H):
    m = W.shape[1] - 1
    time = T / m * np.arange(0, m+1)
    return xi0 * np.exp(eta * W - 0.5 * eta**2 * time **(2*H))

#-------------------------------
# Z : brownian motion sample path
# v : variance sample path
#-------------------------------
def simuAssetPrice(S0, Z, v, T, r):
    m = Z.shape[1] - 1
    N = Z.shape[0]
    deltaT = T / m
    volat = np.sqrt(v)
    deltaZ = Z[:,1:m+1] - Z[:,0:m]
    term1 = np.cumsum(volat[:,0:m]*deltaZ, axis=1)
    term2 = np.cumsum(deltaT * v[:,0:m], axis=1)
    term3 = r * np.arange(1, m+1) * deltaT
    S = np.zeros((N, m+1))
    S[:,0] = S0
    S[:,1:m+1] = S0 * np.exp(term1 + term3 - 0.5 * term2)
    return S


def positivePart(x):
    return np.maximum(x, 0)


def impliedVol(callput, price, S0, T, K, r, tolerence):
    def obejctFunc(x):
        return BS(callput, S0, K, T, r, x)
    if callput=="call" and (price > S0 or price < positivePart(S0 - K * np.exp(-r * T))):
        print "invalid call price"
        return float('nan')
    elif callput=="put" and (price < positivePart(K*np.exp(-r*T) - S0) or price > K * np.exp(-r * T)):
        print "invalid put price"
        print T, K, price
        return float('nan')
    start = 0.30
    result = bisect(price, obejctFunc, start, None, tolerence)
    impVol = result[-1]
    return impVol

def BS(callput, S0, K, T, r, sigma):
    def getD(s, k, v, sign):
        a = np.log(s/k)
        a = a/np.sqrt(v)
        b = np.sqrt(v)/2
        if sign == '+':
            return a+b
        elif sign == '-':
            return a-b
        else:
            pass
    def optionValueOfCall(S0, K_d, d_p, d_m):
        return S0 * norm.cdf(d_p) - K_d * norm.cdf(d_m)
    def optionValueOfPut(S0, K_d, d_p, d_m):
        return K_d * norm.cdf(-d_m) - S0 * norm.cdf(-d_p)
    K_d = K/np.exp(r*T)
    d_p = getD(S0, K_d, sigma*sigma*T, '+')
    d_m = getD(S0, K_d, sigma*sigma*T, '-')
    if callput == "call":
        return optionValueOfCall(S0, K_d, d_p, d_m)
    else:
        return optionValueOfPut(S0, K_d, d_p, d_m)


def ATMSkew(callput, assetPrice, epsilonk, tolerence, S0, T, r):
    def getPayoff(callput, assetPrice, K):
        if callput=='call':
            return positivePart(assetPrice-K)
        else:
            return positivePart(K-assetPrice)
    Kplus = S0*np.exp(epsilonk)
    Kminus = S0*np.exp(-epsilonk)
    m = assetPrice.shape[1]-1
    priceKplus = np.mean(getPayoff(callput, assetPrice[:,1:m+1], Kplus), axis=0)
    priceKminus = np.mean(getPayoff(callput, assetPrice[:,1:m+1], Kminus), axis=0)
    volaPlus = np.zeros(m)
    volaMinus = np.zeros(m)
    for i, price in enumerate(priceKplus):
        t = (i+1)*T/m
        volaPlus[i] = impliedVol(callput, price, S0, t, Kplus, r, tolerence)
    for i, price in enumerate(priceKminus):
        t = (i+1)*T/m
        volaMinus[i] = impliedVol(callput, price, S0, t, Kminus, r, tolerence)
    return np.abs((volaPlus - volaMinus)/2/epsilonk)
