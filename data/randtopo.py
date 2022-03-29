# for generating random topography with given spectral
# characteristics
#
# Author: Joseph H. LaCasce
# Translation to Python: André Palóczy, 4/21
import numpy as np
import matplotlib.pyplot as plt
from hdf5storage import savemat


def wavspec2d(h, nx, L, plot=False):
    """
    calculates angle-averaged spectra E(k) for
    the field h(x,y), assumed doubly periodic
    nx is the number of grid points
    """
    dk = 2*np.pi/L
    k = np.concatenate((np.arange(0, nx/2+1), np.arange(-nx/2+1, 0)))*dk # wavenumbers
    kplot = k[0:int(nx/2+1)]
    kk, ll = np.meshgrid(k, k)
    k2 = kk**2 + ll**2
    kmag = k2**0.5
    kmagv = kmag.ravel()
    nxh = int(nx/2)

    Ek = np.zeros(nxh)
    E = np.abs(np.fft.fft2(h))**2
    E = E/nx**4
    E = E.ravel()
    for j1 in range(0, nxh):
        a = np.logical_and(kplot[j1]<=kmagv, kmagv<kplot[j1+1])
        Ek[j1] = np.sum(E*a)/a.sum()

    Ek = E.sum()/Ek.sum()*Ek
    kplot = kplot[0:nxh]

    if plot:
        fig, ax = plt.subplots()
        ax.loglog(kplot, Ek, marker='o')

    return kplot, Ek


def rms(x):
    x = x - x.mean()

    return np.sqrt(np.sum(x**2)/x.size)


# ---
plt.close("all")

spectyp = "Km2"

nx = 256
L = 2
tk = 10
nxfilt = 32

dk = 2*np.pi/L
k = np.concatenate((np.arange(0, nx/2+1), np.arange(-nx/2+1, 0)))*dk # wavenumbers

filt = nx/nxfilt*dk
lowcut = 0*dk

kplot = k[0:int(nx/2+1)]
kk, ll = np.meshgrid(k, k)
k2 = kk**2 + ll**2
kmag = k2**0.5

h = np.random.randn(nx, nx)              # random topography
h = np.fft.fft2(h)

if spectyp=="exp":
    fac = np.exp(-k2**0.5/tk)              # with exp spectrum
elif spectyp=="Km2":
    fac = 1/(1 + (k2/(tk*np.pi)**2))**0.5  # with k**(-2) spectrum
elif spectyp=="Km4":
    fac = 1/(1 + (k2/(tk*np.pi)**2))       # with k**(-4) spectrum

h = h*fac

a = np.logical_and(kmag<filt*dk, kmag>lowcut*dk)
h = h*a
h[0, 0] = 0
h = np.fft.ifft2(h).real
h = h/rms(h)

dx = L/nx
x = np.arange(0, nx)
x, y = np.meshgrid(x*dx, x*dx - L/2)

fig, ax = plt.subplots()
cs = ax.contourf(x, y, h)
plt.colorbar(cs)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")

nxh = int(nx/2)
fig, ax = plt.subplots()
ax.plot(x[nxh, :], h[nxh, :])
ax.set_xlim(x[nxh, 0], x[nxh, -1])
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$h$")

kplot, Ek = wavspec2d(h, nx, L, plot=True)

plt.show()

fname = "hrand%d%stk%dfiltnx%d.mat"%(nx, spectyp, tk, nxfilt)
# savemat("../code_simulations/" + fname, dict(h=h))
