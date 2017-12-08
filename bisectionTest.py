import numpy as np 
from Bisect import bisect
import pricingLib as lib 

S0 = 100 
sigma = 0.04
T = 0.5
r = 0.0
K = np.arange(50, 150, 2)
k = 50

tolerence = [0.0001, 0.00001]

callA = lib.BS('call', S0, k, T, r, sigma)
callB = lib.BS('call', S0, k, T, r, 0.04)


Call = np.zeros(len(K))
VolImplied = np.zeros(len(K))
Put = np.zeros(len(K))
VolImpliedPut = np.zeros(len(K))
for i, k in enumerate(K):
    Call[i] = lib.BS("call", S0, k, T, r, sigma)
    #print(Call[i])
    VolImplied[i] = lib.impliedVol("call", Call[i], S0, T, k, r, tolerence)
    Put[i] = Call[i] - S0 + k
    #Put[i] = lib.BS("put", S0, k, T, r, sigma)
    print k
    VolImpliedPut[i] = lib.impliedVol("put", 7, S0, T, k, r, tolerence)

#print(VolImplied)
#print(VolImpliedPut)
