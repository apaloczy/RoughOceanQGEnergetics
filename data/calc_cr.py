# Calculate real part of phase speed from Hovmöller diagrams of the perturbation streamfunction.
import numpy as np
import matplotlib.pyplot as plt
from cmocean.cm import balance


#---
plt.close("all")

# ttyp = "rand"
ttyp = "cosi"
PLOT_HOVMOLLER = False

cmap = balance
N = 256
all_F1s = [25, 75, 400]
all_hts = range(1, 11)
head = "../simulations/"

x = np.linspace(0, 2, num=N)
for F1 in all_F1s:
    for httall in all_hts:
        print("F1 = ", F1, " ht = ", httall)
        fflat = "lin_N%d_ht0_F1%d_cosi10.npz"%(N, F1)
        ftall = "lin_N%d_ht%d_F1%d_%s10.npz"%(N, httall, F1, ttyp)
        if ttyp=="rand":
            ftall = ftall.replace("rand10", "hrand256Km2tk10filtnx32")

        d = np.load(head+fflat)
        t_flat, psi1_flat, psi2_flat = d["thovm"], d["p1hovm"], d["p2hovm"]
        d = np.load(head+ftall)
        t_tall, psi1_tall, psi2_tall = d["thovm"], d["p1hovm"], d["p2hovm"]
        t_flat -= t_flat[0]
        t_tall -= t_tall[0]
        mint = np.minimum(t_tall[-1], t_flat[-1])

        # Calculate phase speeds.
        ddt_psi1_flat, ddx_psi1_flat = np.gradient(psi1_flat, t_flat, x)
        ddt_psi2_flat, ddx_psi2_flat = np.gradient(psi2_flat, t_flat, x)
        ddt_psi1_tall, ddx_psi1_tall = np.gradient(psi1_tall, t_tall, x)
        ddt_psi2_tall, ddx_psi2_tall = np.gradient(psi2_tall, t_tall, x)
        cr_psi1_flat = ddt_psi1_flat/ddx_psi1_flat
        cr_psi2_flat = ddt_psi2_flat/ddx_psi2_flat
        cr_psi1_tall = ddt_psi1_tall/ddx_psi1_tall
        cr_psi2_tall = ddt_psi2_tall/ddx_psi2_tall
        cr_psi1_flat = np.median(np.abs(cr_psi1_flat))
        cr_psi2_flat = np.median(np.abs(cr_psi2_flat))
        cr_psi1_tall = np.median(np.abs(cr_psi1_tall))
        cr_psi2_tall = np.median(np.abs(cr_psi2_tall))

        # Normalize by amplitude at each time step to see the phase propagation more clearly.
        psi1_flat = psi1_flat/np.abs(psi1_flat).max(axis=1)[:, np.newaxis]
        psi2_flat = psi2_flat/np.abs(psi2_flat).max(axis=1)[:, np.newaxis]
        psi1_tall = psi1_tall/np.abs(psi1_tall).max(axis=1)[:, np.newaxis]
        psi2_tall = psi2_tall/np.abs(psi2_tall).max(axis=1)[:, np.newaxis]

        figname = "hovmoller_cr_F1%d_ht%d_%s_unfiltered.png"%(F1, httall, ttyp)
        if PLOT_HOVMOLLER:
            xt, yt = 0.01, 0.85
            fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
            ax1, ax2 = ax[0]
            ax3, ax4 = ax[1]
            ax1.pcolormesh(x, t_flat, psi1_flat, cmap=cmap)
            ax2.pcolormesh(x, t_flat, psi2_flat, cmap=cmap)
            ax3.pcolormesh(x, t_tall, psi1_tall, cmap=cmap)
            ax4.pcolormesh(x, t_tall, psi2_tall, cmap=cmap)
            ax4.set_ylim(top=mint)
            ax1.text(xt, yt, "$c_r = %.2f$"%cr_psi1_flat, color="w", fontweight="black", fontsize=16, transform=ax1.transAxes)
            ax2.text(xt, yt, "$c_r = %.2f$"%cr_psi2_flat, color="w", fontweight="black", fontsize=16, transform=ax2.transAxes)
            ax3.text(xt, yt, "$c_r = %.2f$"%cr_psi1_tall, color="w", fontweight="black", fontsize=16, transform=ax3.transAxes)
            ax4.text(xt, yt, "$c_r = %.2f$"%cr_psi2_tall, color="w", fontweight="black", fontsize=16, transform=ax4.transAxes)
            ax3.set_ylabel("$t$", y=1.1, fontsize=16)
            ax3.set_xlabel("$x$", x=1.1, fontsize=16)
            ax1.set_title("$\psi_1$, flat bottom", fontsize=16)
            ax2.set_title("$\psi_2$, flat bottom", fontsize=16)
            ax3.set_title("$\psi_1$, $h_t = %d$ %s"%(httall, ttyp), fontsize=16)
            ax4.set_title("$\psi_2$, $h_t = %d$ %s"%(httall, ttyp), fontsize=16)
            fig.suptitle("$F_1 = %d$, unfiltered"%F1, fontsize=16)

            fig.savefig(figname, bbox_inches="tight")
            plt.close()

        npzname = figname.strip("hovmoller_").replace("png", "npz")
        np.savez(npzname, **dict(cr1_flat=cr_psi1_flat, cr2_flat=cr_psi2_flat, cr1_tall=cr_psi1_tall, cr2_tall=cr_psi2_tall, F1=F1))
