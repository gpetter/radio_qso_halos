import matplotlib.pyplot as plt
from plottools import aesthetic
import matplotlib.cm as cm
import numpy as np
import pickle
import os
os.chdir('/home/graysonpetter/ssd/Dartmouth/radio_qso_halos/')
plotdir = '/home/graysonpetter/Dropbox/radioplots/'
def read_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

hizc = 'firebrick'

def halomass():
    #dndz = read_pickle('results/dndz/rg.pickle')
    z_centers = []
    import glob
    qsofitnames = glob.glob('results/fits/tomo/rg_qso*.pickle')
    fits = []
    for name in qsofitnames:
        thisfit = read_pickle(name)
        fits.append(thisfit)
        z_centers.append(thisfit['eff_z'])

    radmasses = [fit['Mx'] for fit in fits]
    raderrs = [fit['sigMx'] for fit in fits]
    elgfit = read_pickle('results/fits/elgfit.pickle')
    #autofit = read_pickle('results/fits/rg_auto.pickle')

    elgx = read_pickle('results/fits/tomo/rg_elg.pickle')

    lensfit = read_pickle('results/lensfits/rg.pickle')
    izrgfit = read_pickle('results/fits/auto/izrg.pickle')

    rgcolor='firebrick'

    plt.figure(figsize=(8, 7))
    plt.errorbar(z_centers, radmasses, yerr=raderrs, label=r'HzRGs $\times$ eBOSS QSOs', color=rgcolor, fmt='o')


    #plt.scatter(1.25, elgx[0], c=rgcolor, marker='*', s=100)
    plt.errorbar(elgx['eff_z'], elgx['Mx'], elgx['sigMx'], color=rgcolor, fmt='*', ms=10, label=r'HzRGs $\times$ DESI ELGs')

    #plt.fill_between(z_centers, qsomasses - qsoerrs, qsomasses + qsoerrs, alpha=0.5, color='royalblue')
    plt.text(1.5, 12.35, 'eBOSS QSOs', color='royalblue', fontsize=15)
    from halomodelpy import halo_growthrate
    zgrid, ms = halo_growthrate.evolve_halo_mass(12.5, 2.5, 0.01)
    plt.plot(zgrid, ms, c='k', ls='--', alpha=0.3)
    zgrid, ms = halo_growthrate.evolve_halo_mass(13., 2.5, 0.01)
    plt.plot(zgrid, ms, c='k', ls='--', alpha=0.3)
    zgrid, ms = halo_growthrate.evolve_halo_mass(12.5, 2.5, 0.01, wantmean=False)
    plt.plot(zgrid, ms, c='k', ls='--', alpha=0.1)
    zgrid, ms = halo_growthrate.evolve_halo_mass(13., 2.5, 0.01, wantmean=False)
    plt.plot(zgrid, ms, c='k', ls='--', alpha=0.1)

    zranges = [1, 1.5, 2, 2.5]
    for j in range(3):
        qsoauto = read_pickle('results/fits/tomo/qso_fit_%s.pickle' % j)
        plt.fill_between([zranges[j], zranges[j+1]], (qsoauto['M'] - qsoauto['sigM'])*np.ones(2), (qsoauto['M'] + qsoauto['sigM'])*np.ones(2), color='cornflowerblue', edgecolor='none')

    #plt.scatter(1.6, rgautofit['M'], c='k')
    #plt.errorbar(1.6, rgautofit['M'], yerr=rgautofit['sigM'], ecolor='k', fmt='none', label='HzRG auto')


    yup = elgfit['M'] + elgfit['sigM']
    ylo = elgfit['M'] - elgfit['sigM']
    plt.fill_between([1., 1.5], ylo*np.ones(2), yup*np.ones(2), color='cadetblue', edgecolor='none')
    plt.text(.5, ylo, 'DESI ELGs', color='cadetblue', fontsize=15)

    plt.errorbar(1.6, lensfit['M'], lensfit['sigM'], markeredgecolor=rgcolor, markerfacecolor='none', ecolor=rgcolor, fmt='o', label='HzRG CMB Lensing')

    #plt.scatter(1.6, autofit['M'])
    #plt.errorbar(1.6, autofit['M'], autofit['sigM'], ecolor='k', fmt='none')
    #plt.fill_between([1, 2.5], (autofit['M'] - autofit['sigM'])*np.ones(2), (autofit['M'] + autofit['sigM'])*np.ones(2), color=rgcolor, alpha=0.5, edgecolor='none')
    #im2 = plt.imshow(np.outer(np.ones(len(dndz[0])), dndz[1]), cmap=cm.Reds,
    #                 extent=[0.1, 4., autofit['M'] - autofit['sigM'], autofit['M'] + autofit['sigM']],
    #                 interpolation="bicubic", alpha=.4, aspect="auto")
    #plt.text(1.6, autofit['M'] + 0.1, 'HzRG auto', color=rgcolor, fontsize=15)
    plt.errorbar(izrgfit['eff_z'], izrgfit['M'], izrgfit['sigM'], color='orange', fmt='o', label='IzRGs')

    #plt.fill_between(laurentzspace, mlit[2], mlit[1], color='seagreen', alpha=0.2, edgecolor='none')
    plt.xlim(0, 2.5)
    plt.fill_between([0, 0.1], [14, 14], [15, 15], color='maroon', alpha=0.2, edgecolor='none')
    plt.text(0.15, 14.3, 'Clusters', color='maroon', alpha=0.5)

    plt.ylim(12, 14.5)

    #plt.scatter(elgzs, rgmass, c='none', edgecolors='firebrick')
    #plt.errorbar(elgzs, rgmass, yerr=rgerr, ecolor='firebrick', fmt='none')
    #plt.text(1, 12., 'DESI ELGs', color='b')


    plt.xlabel('Redshift')
    plt.legend(fontsize=15)
    plt.ylabel(r'log$_{10}(M_h / h^{-1} M_{\odot}$)')

    plt.savefig(plotdir + 'lotss_tomo.pdf')
    plt.close('all')

def cross_cfs(vertical=False):

    import glob
    qsocfnames = sorted(glob.glob('results/cfs/tomo/rg_qso*.pickle'))
    qsofitnames = sorted(glob.glob('results/fits/tomo/rg_qso*.pickle'))
    autofitnames = sorted(glob.glob('results/fits/tomo/qso_fit_*.pickle'))
    cfs = []
    fits = []
    autofits = []
    for name in qsocfnames:
        cfs.append(read_pickle(name))
    for name in qsofitnames:
        fits.append(read_pickle(name))
    for name in autofitnames:
        autofits.append(read_pickle(name))
    elgcf = read_pickle('results/cfs/tomo/rg_elgcf.pickle')
    if vertical:
        fig, axs = plt.subplots(nrows=len(cfs), ncols=1, sharex=True, figsize=(8, 7 * len(cfs)))
        plt.xscale('log')
        plt.xlim(1e-2, 1)
        plt.subplots_adjust(hspace=0)
        for j in range(len(cfs)):
            cf = cfs[j]
            fit = fits[j]
            axs[j].scatter(cf['theta'], cf['w_theta'], c='k')
            axs[j].errorbar(cf['theta'], cf['w_theta'], yerr=cf['w_err'], ecolor='k', fmt='none')
            axs[j].set_yscale('log')
            axs[j].plot(fit['modscales'], fit['dmcf'], c='k', ls='dotted')
            # axs[j].set_xscale('log')
            axs[j].set_ylim(1e-4, 1e-1)
    else:
        fig, axs = plt.subplots(nrows=1, ncols=len(cfs), sharey=True, figsize=(8 * len(cfs), 7))
        plt.yscale('log')
        plt.ylim(1e-4, 1e-1)
        axs[0].set_ylabel(r'Angular Cross-Correlation Function $w(\theta)$')
        plt.subplots_adjust(wspace=0)

        axs[0].scatter(elgcf['theta'], elgcf['w_theta'], c='cadetblue')
        axs[0].errorbar(elgcf['theta'], elgcf['w_theta'], yerr=elgcf['w_err'], ecolor='cornflowerblue', fmt='none')
        zrange = [1., 1.5, 2., 2.5]

        for j in range(len(cfs)):
            axs[j].set_xlabel(r'Separation $\theta$ [deg]')
            cf = cfs[j]
            fit = fits[j]
            autofit = autofits[j]
            axs[j].errorbar(cf['theta'], cf['w_theta'], yerr=cf['w_err'], color='firebrick', fmt='o')
            axs[j].set_xscale('log')
            axs[j].plot(fit['modscales'], fit['dmcf'], c='k', ls='dotted')
            if j == 0:
                axs[0].text(5e-2, 1.5e-3, 'Matter', rotation=-15, color='k', fontsize=15)
            axs[j].plot(fit['modscales'], fit['xfitcf'], c='firebrick', ls='dashed')
            axs[j].fill_between(fit['modscales'], (autofit['b'] - autofit['sigb']) * fit['dmcf'],
                                (autofit['b'] + autofit['sigb']) * fit['dmcf'], label=r'$b_{\mathrm{QSO}}$',
                                color='cornflowerblue', edgecolor='none')

            axs[j].text(0.05, .93, r'$%s < z < %s$' % (zrange[j], zrange[j + 1]), transform=axs[j].transAxes,
                        fontsize=20)
            # axs[j].set_xscale('log')
            axs[j].set_xlim(1e-2, 1)
        plt.legend()
    plt.savefig(plotdir + 'crosscfs.pdf')

def hzrg_lenscorr():


    hzrglens = read_pickle('results/lenscorr/hzrg.pickle')
    hzrglensfit = read_pickle('results/lensfits/hzrg.pickle')
    hzrgdndz = read_pickle('results/dndz/hzrg.pickle')
    from halomodelpy import hm_calcs

    fig, ax = plt.subplots(figsize=(8, 7))
    ellgrid = np.logspace(0.5, 3.5, 1000)
    hm = hm_calcs.halomodel(hzrgdndz)
    plt.errorbar(hzrglens['ell'], hzrglens['cl'], hzrglens['cl_err'], fmt='o', color=hizc)
    hm.set_powspec()
    plt.plot(ellgrid, hm.get_c_ell_kg(ellgrid), c='k', ls='dotted')
    hm.set_powspec(log_meff=hzrglensfit['M'])
    plt.plot(ellgrid, hm.get_c_ell_kg(ellgrid), c=hizc, ls='dashed')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(30, 2000)
    plt.xlabel(r'Multipole $\ell$')
    plt.ylabel('Cross-power $C_{\ell}^{\kappa g}$')
    plt.ylim(np.min(hzrglens['cl']) / 3., np.max(hzrglens['cl']) * 2.)
    plt.text(0.7, .93, r'HzRG Lensing', transform=ax.transAxes,
                fontsize=20)

    plt.savefig(plotdir+'hzrglensing.pdf')
    plt.close('all')

def autocorrs():
    fig, (ax, ax2) = plt.subplots(figsize=(8, 14))

def dutycycle():
    """plt.figure(figsize=(8,7))
        import lumfunc
        occs, upoccs, lowoccs = [], [], []
        for j in range(len(radmasses)):
            occs.append(lumfunc.occupation_frac(radmasses[j]-0.2, z_centers[j]))
            upoccs.append(lumfunc.occupation_frac(radmasses[j] - 0.2 + raderrs[j], z_centers[j]))
            lowoccs.append(lumfunc.occupation_frac(radmasses[j] - 0.2 - raderrs[j], z_centers[j]))
        plt.scatter(z_centers, occs, c='k')
        plt.errorbar(z_centers, occs, np.array(upoccs) - np.array(occs), ecolor='k', fmt='none')
        plt.yscale('log')
        plt.ylim(1e-3, 3)
        plt.xlabel('Redshift')
        plt.ylabel('Duty cycle')

        plt.savefig(plotdir + 'dutycycle.pdf')
        plt.close('all')"""

def energetics(zs, e, elo, ehi):
    import lumfunc
    plt.figure(figsize=(8, 7))

    thermal2kev = lumfunc.thermal_energy_mass(13.2, 2)
    bindingenergy = lumfunc.halobinding_energy(13.2, 1.5)
    plt.axhline(thermal2kev, c='k', ls='--', label=r'T=2 keV, $M_{gas} = \Omega_b/\Omega_m*M_h$')

    plt.axhline(bindingenergy, c='green', ls='dashed', label='Gas binding energy')


    plt.axhline(lumfunc.lx_energy(44, (1., 1.5)), label=r'$\Delta t \times L_x=44$')

    plt.errorbar(zs, e, yerr=[elo, ehi], color='firebrick', fmt='o')
    plt.ylabel(r'log$\langle$ Heating energy per halo $\rangle$ [erg]')
    plt.xlabel('Redshift')
    plt.legend(fontsize=10, loc='lower right')
    plt.savefig(plotdir + 'energetics.pdf')
    plt.close('all')



#cross_cfs()
hzrg_lenscorr()