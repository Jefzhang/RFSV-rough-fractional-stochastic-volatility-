import csv
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import math



startDate = "20000103"
endDate = "20170926"

def getData(file, startDate, endDate):
    volatility = []#np.zeros(N)
    #count = 0
    shouldImport = False
    with open(file, 'rb') as csvfile:
        Data = csv.reader(csvfile, delimiter = ';')
        for row in Data:
            try:
                date = row[0]
                variance = float(row[1])
            except ValueError:
                continue
            else:
                print type(date)
                print date
                if date==startDate:
                    shouldImport = True
                if date==endDate:
                    shouldImport = False
                    break
                if shouldImport:
                    volatility = np.append(volatility, np.sqrt(variance))
    return volatility

def logVol(volatility):
    return np.log(volatility)

def computeM(q, delta, logVola):
    #sampleVola = np.zeros(np.floor(float(len(logSpxVola))/delta))
    sampleNumber = int(np.floor(float(len(logVola))/delta)) - 1
    res = 0
    '''
    for start in range(delta):
        first = start
        second  = start
        for i in range(sampleNumber):
            second = first + delta
            res = res + np.abs(logVola[second] - logVola[first])**q
            first = second
    '''
    res = np.sum(np.abs(logVola[delta:len(logVola)-1] - logVola[0:len(logVola)-delta-1])**q)
    return res/(len(logVola)-delta)


def linearRegression(X, Y):
    meanX = np.mean(X)
    meanY = np.mean(Y)
    beta1 = np.sum((X-meanX)*(Y-meanY))/np.sum((X-meanX)**2)
    beta0 = meanY - beta1*meanX
    return (beta0, beta1)


def distriDeltaVol(delta, logVola):
    diff = np.zeros(len(logVola) - delta)
    diff = logVola[delta:len(logVola)-1] - logVola[0:len(logVola)-delta-1]
    return diff


Delta = np.arange(1,50)
Q = np.array([0.5, 1, 1.5, 2, 3])


realVola = getData('../data/Nasdaq.csv', startDate, endDate)

print "We use data of {} days from {} to {}".format(len(realVola), startDate, endDate)

logVola = logVol(realVola)


result = np.zeros((len(Q), len(Delta)))

for i in range(len(Q)):
    q = Q[i]
    for j in range(len(Delta)):
        delta = Delta[j]
        result[i][j] = computeM(q, delta, logVola)

logResult = np.log(result)
logDelta = np.log(Delta)
Beta0 = np.zeros(len(Q))
Beta1 = np.zeros(len(Q))


plt.figure(0)
plt.xlabel(r'$log(\bigtriangleup)$')
plt.ylabel(r'$log(m(q,\bigtriangleup))$')
for i in range(result.shape[0]):
    (beta0, beta1) = linearRegression(logDelta, logResult[i,:])
    Beta0[i] = beta0
    Beta1[i] = beta1
    plt.plot(logDelta, beta0+beta1*logDelta, 'r-')
    plt.plot(logDelta, logResult[i,:],'*', label='q = '+str(Q[i]))
plt.legend(loc = 'best')



h = np.mean(Beta1 / Q)
print "The empirical H got is about {0:.4f}".format(h)

Q = np.insert(Q, 0, 0)
Beta1 = np.insert(Beta1, 0, 0)
plt.figure(1)
plt.xlabel(r'$q$')
plt.ylabel(r'$\zeta$')
plt.plot(Q, Beta1, 'b-', label='Empirical results')
plt.plot(Q, h*Q, 'r-', label=r'$\zeta = ${0:.3f}$*q$'.format(h))
plt.legend(loc='best')

plt.figure(2)
plt.plot(realVola)


'''
h = 0.129

deltaD = [1, 5, 25, 125]
diff1 = distriDeltaVol(deltaD[0], logVola)
diff5 = distriDeltaVol(deltaD[1], logVola)
diff25 = distriDeltaVol(deltaD[2], logVola)
diff125 = distriDeltaVol(deltaD[3], logVola)

plt.figure(3)
ax1 = plt.subplot(221)
ax1.set_title(r'$\bigtriangleup = 1$ day')
ax1.hist(diff1, bins=50, normed = True, color = 'w')
mean = np.mean(diff1)
var = np.var(diff1)
standM = mean
standV = var
g = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
ax1.plot(mean+np.sqrt(var)*g, norm.pdf(g)/np.sqrt(var), 'b--', linewidth=2.0)
ax1.plot(standM+np.sqrt(standV)*g, norm.pdf(g)/np.sqrt(standV), 'r-', linewidth=1.0)

ax2 = plt.subplot(222)
ax2.set_title(r'$\bigtriangleup = 5$ days')
ax2.hist(diff5, bins=50,  normed = True, color = 'w')
mean = np.mean(diff5)
var = np.var(diff5)
ax2.plot(mean+np.sqrt(var)*g, norm.pdf(g)/np.sqrt(var), 'b--', linewidth=2.0)
var2 = (float(deltaD[1])/deltaD[0])**(2*h) * standV
ax2.plot(standM+np.sqrt(var2)*g, norm.pdf(g)/np.sqrt(var2), 'r-', linewidth=1.0)

ax3 = plt.subplot(223)
ax3.set_title(r'$\bigtriangleup = 25$ days')
ax3.hist(diff25, bins=50, normed = True, color='w', range=[-2,2])
mean = np.mean(diff25)
var = np.var(diff25)
ax3.plot(mean+np.sqrt(var)*g, norm.pdf(g)/np.sqrt(var), 'b--', linewidth=2.0)
var2 = (float(deltaD[2])/deltaD[0])**(2*h) * standV
ax3.plot(standM+np.sqrt(var2)*g, norm.pdf(g)/np.sqrt(var2), 'r-', linewidth=1.0)

ax4 = plt.subplot(224)
ax4.set_title(r'$\bigtriangleup = 125$ days')
ax4.hist(diff125, bins=50,normed = True, color='w')
mean = np.mean(diff125)
var = np.var(diff125)
ax4.plot(mean+np.sqrt(var)*g, norm.pdf(g)/np.sqrt(var), 'b--', linewidth=2.0)
var2 = (float(deltaD[3])/deltaD[0])**(2*h) * standV
ax4.plot(standM+np.sqrt(var2)*g, norm.pdf(g)/np.sqrt(var2), 'r-', linewidth=1.0)

'''
plt.show()


    


 