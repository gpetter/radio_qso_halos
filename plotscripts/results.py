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

def halomass():
    dndz = read_pickle('results/dndz/rg.pickle')
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
    autofit = read_pickle('results/fits/rg_auto.pickle')

    elgx = read_pickle('results/fits/tomo/rg_elg.pickle')

    lensfit = read_pickle('results/lensfits/rg.pickle')

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

    plt.errorbar(1.6, lensfit['M'], lensfit['sigM'], color='green', fmt='o')

    #plt.scatter(1.6, autofit['M'])
    #plt.errorbar(1.6, autofit['M'], autofit['sigM'], ecolor='k', fmt='none')
    #plt.fill_between([1, 2.5], (autofit['M'] - autofit['sigM'])*np.ones(2), (autofit['M'] + autofit['sigM'])*np.ones(2), color=rgcolor, alpha=0.5, edgecolor='none')
    im2 = plt.imshow(np.outer(np.ones(len(dndz[0])), dndz[1]), cmap=cm.Reds,
                     extent=[0.1, 4., autofit['M'] - autofit['sigM'], autofit['M'] + autofit['sigM']],
                     interpolation="bicubic", alpha=.4, aspect="auto")
    plt.text(1.6, autofit['M'] + 0.1, 'HzRG auto', color=rgcolor, fontsize=15)

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


def autocorrs():
    fig, (ax, ax2) = plt.subplots(figsize=(8, 14))



halomass()