# Two layer instability
# Channel, with y=[-1,1], non-dimensional
# Bottom slope in y-direction (h=alpha*y)
# with 1D topography h=h(y)
#
# (U1-c)*(P1_yy - k^2 P1 - F1 P1 + F1 P2) + (1-U1_yy) P1 = 0
# (U2-c)*(P2_yy - k^2 P2 - F2 P2 + F2 P1) + (1+alpha-U2_yy+hy) P2 = 0
#
# Author: Joseph H. LaCasce, 1/18
# Translation to Python: André Palóczy, 12/21

import numpy as np
from numpy import matrix, array, diag, eye, block, arange, zeros, ones
from scipy.linalg import eig, norm
import matplotlib.pyplot as plt


def cheb(n):
    """
    compute Dx = differentiation matrix, x = Chebyshev grid

    Matlab function written by Joe H. LaCasce.
    Translated to Python by André Palóczy (5/2021).
    """
    if n==0:
        x = matrix([1.0])
        Dx = matrix([0.0])
    else:
        N = arange(n + 1)
        x = matrix(np.cos(np.pi*N/n)).T
        m1 = (-1)**N
        c = matrix(np.concatenate(([2], ones(n - 1), [2]))*m1).T
        X = np.tile(x, (1, n + 1))
        dX = X - X.T
        Dx  = array(c*(1/c).T)/array(dX + eye(n + 1)) # off-diagonal entries
        Dx = matrix(Dx)
        Dx  = Dx - diag(array(Dx.T.sum(axis=0)).squeeze()) # diagonal entries

    return x, Dx


def sech(x):
    return 1/np.cosh(x)


#---
plt.close("all")

#allhts = [0, 10]
allhts = [10]

# lt = 10*np.pi                     #  bump wavenumber
lt = 50*np.pi


allF1s = [25, 75, 400]
PLOT = False
N = 512                           #  no. grid points in y
h1o2 = 1/4                        #  layer depth ratio
bet = 0                           #  beta
al = 0                            #  bottom slope
kmax = 60                         #  max wavenumber
dk = 1                            #  resolution wavenumber
U0 = 1                            #  amplitude surface jet
du = 0                            #  U2=du*U1
L = 0.2                           #  width surface jet

efac = N**4/2
for F1 in allF1s:
    for ht in allhts:
        print("F1 = ", str(F1), " lt = ", str(int(lt/np.pi)), "pi, ht = ", str(ht))

        kp = matrix(np.arange(dk, kmax+dk, dk)).T
        d1 = h1o2/(1 + h1o2)
        d2 = 1/(1 + h1o2)
        F = d1*F1
        Lx = 2*np.pi

        Nm = N - 1
        y, D = cheb(N)
        D2 = D**2
        D2 = D2[1:N, 1:N]
        D = D[1:N, 1:N]
        y = y[1:N, :]
        y0 = array(y).flatten()
        dy = np.gradient(y0)[:, np.newaxis]
        F1e = F1*matrix(eye(Nm))
        F2 = h1o2*F1
        F2e = F2*matrix(eye(Nm))
        n2 = int(np.floor(N/2))
        Np = kp.size
        ci = zeros(Np)
        cr, hf, m1, m2, rat, pha = ci.copy(), ci.copy(), ci.copy(), ci.copy(), ci.copy(), ci.copy()
        ke1, ke2, pe = ci.copy(), ci.copy(), ci.copy()

        U1 = U0*sech(y0/L)**2
        U1yy = 2/L**2*U1*(3*np.tanh(y0/L)**2 - 1) # Bickley jet.
        U1, U1yy = map(matrix, (U1, U1yy))
        U1, U1yy = U1.T, U1yy.T

        U2 = du*U1
        U2yy = du*U1yy

        qs1y =  bet - U1yy + F1*(U1 - U2)
        hy = -ht*lt*np.sin(lt*y)
        qs2y =  bet - U2yy + F2*(U2 - U1) + al + hy
        qs1ye = diag(array(qs1y).flatten())
        qs2ye = diag(array(qs2y).flatten())
        hye = diag(array(hy).flatten())
        U1e = diag(array(U1).flatten())
        U2e = diag(array(U2).flatten())
        qs1ye, qs2ye, hye, U1e, U2e = map(matrix, (qs1ye, qs2ye, hye, U1e, U2e))
        U1, U2 = map(array, (U1, U2))

        for n in range(Np):
            k = array(kp[n]).flatten()[0]
            k2e = matrix(k**2*eye(Nm))

            A11 = U1e*(D2 - k2e - F1e) + qs1ye
            A12 = U1e*F1e
            A21 = U2e*F2e
            A22 = U2e*(D2 - k2e - F2e) + qs2ye
            A = block([[A11, A12], [A21, A22]])

            B11 = D2 - k2e - F1e
            B12 = F1e
            B21 = F2e
            B22 = D2 - k2e - F2e
            B = block([[B11, B12], [B21, B22]])

            c, V = eig(A, b=B, check_finite=False)
            V = matrix(V)

            j = np.where(c.imag > 0)[0]
            if len(j)>0:
                nm = c[j].imag.argmax()
                jnm = j[nm]
                ci[n] = c[jnm].imag
                cr[n] = c[jnm].real

                Vs1 = V[:Nm, jnm]
                Vs2 = V[Nm:, jnm]
                u1y = -D2*V[:Nm, :]
                u1y = u1y[:, jnm]
                u2y = -D2*V[Nm:, :]
                u2y = u2y[:, jnm]
                Vs1, Vs2, u1y, u2y = map(array, (Vs1, Vs2, u1y, u2y))

                # Calculate diagnostic quantities for each mode.
                hf[n] = np.real(F*np.sum(1j*k*(U1 - U2)*(Vs2.conj()*Vs1 - Vs2*Vs1.conj())*dy)) # Thickness flux.
                m1[n] = np.real(d1*np.sum(1j*k*U1*(Vs1.conj()*u1y - Vs1*u1y.conj())*dy))       # Upper-layer momentum flux.
                m2[n] = np.real(d2*np.sum(1j*k*U2*(Vs2.conj()*u2y - Vs2*u2y.conj())*dy))       # Lower-layer momentum flux.
                rat[n] = norm(Vs2, ord=2)/norm(Vs1, ord=2)
                pha[n] = np.arctan2(Vs1[n2].imag, Vs1[n2].real) - np.arctan2(Vs2[n2].imag, Vs2[n2].real)

                # Energies.
                u1 = -D*V[:Nm, :] # -dpsi1/dy
                u2 = -D*V[Nm:, :] # -dpsi2/dy
                u1sq = np.array(np.abs(u1[:, jnm]))**2
                u2sq = np.array(np.abs(u2[:, jnm]))**2
                v1 = 1j*k*V[:Nm, :] # dpsi1/dx
                v2 = 1j*k*V[Nm:, :] # dpsi2/dx
                v1sq = np.array(np.abs(v1[:, jnm]))**2
                v2sq = np.array(np.abs(v2[:, jnm]))**2

                ke1[n] = d1*np.sum(u1sq + v1sq)/efac
                ke2[n] = d2*np.sum(u2sq + v2sq)/efac
                pe[n] = F*(np.array(np.abs(Vs2 - Vs1))**2).sum()/efac

        kp = array(kp).flatten()
        sig = kp*ci
        nm = sig.argmax()

        if sig[nm]==0:
            msig = 0
            mk = 0
            mrat = 0
            mpha = 0
            mcr = 0
            ac = 0
            mmom = 0
            mthick = 0
        else:
            msig = sig[nm]
            mk = kp[nm]
            mrat = rat[nm]
            mpha = pha[nm]
            mcr = cr[nm]
            ac = np.sqrt(mcr**2 + ci[nm]**2)
            mmom = m1[nm]
            mthick = hf[nm]

        if ht==0:
            ttl = "Flat bottom, F1 = " + str(F1)
            figname = "momthick-F1_" + str(F1) + "_flatbottom.png"
        else:
            ttl = "1D ridge (ht = " + str(ht) + " lt = " + str(int(lt/np.pi)) + "\pi), F1 = " + str(F1)
            figname = "momthick-F1_" + str(F1) + "ht" + str(ht) + "lt" + str(int(lt/np.pi)) + ".png"

        if PLOT:
            fig, ax = plt.subplots(nrows=2, sharex=True)
            ax1, ax2 = ax
            xlab = "N=" + str(N) + ", F1/F2=" + str(F1) + "/" + str(F2) + ", al=" + str(al) + ", ht=" + str(ht) + ", L=" + str(L) + ", du=" + str(du)

            ax1.plot(kp, sig, 'b.-')
            ax1.set_ylabel("Growth rate", fontsize=11)
            ax1.set_xlabel(xlab)
            ax1r = ax1.twinx()
            ax1r.plot(kp, cr, 'r--')
            ax1r.set_ylim(0, 1)
            ax1r.set_ylabel("Phase speed", fontsize=11)
            ax1.set_title(ttl, fontsize=14)

            ax2.plot(kp, hf, 'b.-')
            ax2.set_xlabel("k", fontsize=12)
            ax2.set_ylabel("Thickness flux", fontsize=11)
            ax2r = ax2.twinx()
            ax2r.plot(kp, m1, "r.-")
            ax2.set_xlabel("k", fontsize=12)
            ax2r.set_ylabel("Momentum flux", fontsize=11)
            fig.savefig(figname, bbox_inches="tight")
            plt.close()

        npzname = figname.replace("png", "npz")
        np.savez(npzname, kp=kp, sig=sig, cr=cr, hf=hf, m1=m1, ke1=ke1, ke2=ke2, pe=pe, N=N, F1=F1, F2=F2, al=al, ht=ht, lt=lt, L=L, du=du, h1o2=h1o2, bet=bet, kmax=kmax, dk=dk, U0=U0)
