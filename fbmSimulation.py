#Script for observing the fractional brownian motion

import numpy as np
import matplotlib.pyplot as plt
from fbm import fbm

N = 5000
H1 = 0.1
H2 = 0.5

f1 = fbm(n=N, hurst=H1, length=1, method='daviesharte')
f2 = fbm(n=N, hurst=H2, length=1, method='daviesharte')
f3 = np.cumsum(np.sqrt(1.0/N)*np.random.randn(N))

plt.figure(0)
plt.plot(f1, label='h = 0.1')
plt.legend()
plt.figure(1)
plt.plot(f2, label='h = 0.8')
plt.plot(f3, 'r')
plt.legend()

plt.show()


