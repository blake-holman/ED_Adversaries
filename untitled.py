import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib
    # %matplotlib inline    
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
mpl.rcParams['figure.dpi'] =300


def f(s, n):
    if s <= n**(1/3):
        return np.sqrt(n/s)
    else:
        return n**(1/3)

n=100
X = np.arange(1, n**(1/3+.1), .01)
Q = [f(x, n)  for x in X]
C = [np.sqrt(n) for _ in X]
# print(X, Q, C)
fig, ax = plt.subplots()
plt.xlabel('Allowable Space', fontsize=20)
plt.ylabel('Time', fontsize=20)
plt.plot(X, Q, label='Quantum', linewidth=3)
plt.plot(X, C, label='Classical', linewidth=3)

plt.legend()
ax.set_xticks([1, n**(1/3)])
ax.set_xticklabels([r'$\log N$', r'$N^{1/3}\log N$'])
ax.set_yticks([n**(1/2), n**(1/3)])
ax.set_yticklabels([r'$\sqrt{N}$', r'$N^{1/3}$'])
ax.set_ylim((n**(1/4), 1.2 * n**(1/2)))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('time.png')


n=10000
X = np.arange(1, n**(1/3)*1.5, .1)
print(X)
Q = [f(x, n) * np.min([x, n**(1/3)])   for x in X]
C = [np.sqrt(n) for _ in X]
# print(X, Q, C)
fig, ax = plt.subplots()
plt.plot(X, Q, label='Quantum', linewidth=3)
plt.plot(X, C, label='Classical', linewidth=3)
print(np.log2(n), n**(1/3))
ax.legend()
ax.set_xticks([1, n**(1/3)])
ax.set_xticklabels([r'$\log N$', r'$N^{1/3}\log N$'])
ax.set_yticks([n**(1/2), n**(2/3)])
ax.set_yticklabels([r'$\sqrt{N}\log N$', r'$N^{2/3}\log N$'])
ax.set_ylim((n**(1/2)*.8, n**(2/3)* 1.1))
# ax.set_xlim((np.log(n), X[-1]))
plt.xlabel('Allowable Space', fontsize=20)
plt.ylabel('Space-Time', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('spacetime.png')