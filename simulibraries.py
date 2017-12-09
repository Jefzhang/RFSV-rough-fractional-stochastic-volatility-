from fbm import FBM
import numpy as np
import numpy.random as npr
import math

def simXEuler(X0, dW, v, alpha, m, delta, Ndays): 
    N = int(math.floor(Ndays / delta))
    X = np.zeros(N+1)
    X[0] = X0
    for i in range(1,N+1):
        X[i] = X[i-1]+v*dW[i-1]+alpha*delta*(m- X[i-1])
    return X


def simX(X0, dW, v, alpha, m, delta, Ndays):
    N = int(math.floor(Ndays / delta))
    X = np.zeros(N+1)
    X[0] = X0
    time = np.arange(1, N+1)*delta
    term1 = np.exp(-alpha * time)
    term2 = (X0 - m)*term1
    term3 = v*term1*np.cumsum(np.exp(alpha*time)*dW)
    X[1:N+1] = m + term2 + term3
    return X



def simPriceEuler(P0, sigma, delta, Ndays):
    N = int(np.floor(Ndays / delta))
    P = np.zeros(N+1)
    P[0] = P0
    sqrtDelta =np.sqrt(delta)
    U = sqrtDelta* npr.randn(N)
    for i in range(1, N+1):
        P[i] = (1.0+sigma[i-1]*U[i-1])*P[i-1]
    return P

def simPrice(P0, sigma, delta, Ndays):
    N = len(sigma) - 1
    P = np.zeros(N+1)
    P[0] = P0
    a = -0.5 * delta* np.cumsum(sigma[0:N]**2)
    b = np.cumsum(sigma[0:N] * np.sqrt(delta) * npr.randn(N))
    P[1:N+1] = P0 * np.exp(a+b)
    return P

def inteVarianceOneHour(logP):
    pass


def realizedVariance(logP, delta, Ndays, sampleFreInMinuts, startTime, endTime):
    numPerDay = 1.0/delta
    varianceRealized = np.zeros(Ndays)
    (startIndex, endIndex) = indexRangeForOneDay(numPerDay, startTime, endTime)
    print('For every day, we calculate from {} to {}'.format(startIndex, endIndex))
    sampleFreInNum = sampleFrequencyInNum(numPerDay, sampleFreInMinuts)
    for i in range(Ndays):
        varianceRealized[i] = varianceForOneDay(logP[i*numPerDay:(i+1)*numPerDay], startIndex, endIndex, sampleFreInNum)
    return varianceRealized


def indexRangeForOneDay(numPerDay, startTime, endTime):
    numPerHour = numPerDay / 24
    return (int(startTime*numPerHour), int(endTime*numPerHour))


def sampleFrequencyInNum(numPerDay, sampleFreInMinuts):
    numPerMinute = numPerDay / (24*60)
    return int(sampleFreInMinuts*numPerMinute)


def varianceForOneDay(logPForOneDay, start, end, sampleFre):
    return quadricVariation(logPForOneDay[start:end+1], sampleFre)


def quadricVariation(sample, sampleFre):
    N = int(np.floor(len(sample)/sampleFre)) - 1
    variation = 0
    for i in range(N):
        term = (sample[(i+1)*sampleFre] - sample[i*sampleFre])**2
        variation += term
    return variation
    

def computeM(q, delta, logVola):
    sampleNumber = int(np.floor(float(len(logVola))/delta)) - 1
    '''
    res = 0
    for start in range(delta):
        first = start
        second  = start
        for i in range(sampleNumber):
            second = first + delta
            res = res + np.abs(logVola[second] - logVola[first])**q
            first = second
    '''
    res = np.sum(np.abs(logVola[delta:len(logVola)-1] - logVola[0:len(logVola)-delta-1])**q)
    return res/sampleNumber/delta

def linearRegression(X, Y):
    meanX = np.mean(X)
    meanY = np.mean(Y)
    beta1 = np.sum((X-meanX)*(Y-meanY))/np.sum((X-meanX)**2)
    beta0 = meanY - beta1*meanX
    return (beta0, beta1)