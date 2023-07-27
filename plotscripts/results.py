import matplotlib.pyplot as plt
from plottools import aesthetic
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pickle
import os
os.chdir('/home/graysonpetter/ssd/Dartmouth/radio_qso_halos/')
plotdir = '/home/graysonpetter/Dropbox/radioplots/'
def read_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

hizc = 'firebrick'
izc = 'orange'

def halomass(elg=False):
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


    hzrglensfit = read_pickle('results/lensfits/hzrg.pickle')
    izrgfit = read_pickle('results/fits/auto/izrg.pickle')

    rgcolor='firebrick'

    plt.figure(figsize=(8, 7))
    plt.errorbar(z_centers, radmasses, yerr=raderrs, label=r'HzRGs $\times$ eBOSS QSOs', color=rgcolor, fmt='o')

    if elg:
        elgfit = read_pickle('results/fits/elgfit.pickle')
        elgx = read_pickle('results/fits/tomo/rg_elg.pickle')
        plt.errorbar(elgx['eff_z'], elgx['Mx'], elgx['sigMx'], color=rgcolor, fmt='*', ms=10, label=r'HzRGs $\times$ DESI ELGs')
        yup = elgfit['M'] + elgfit['sigM']
        ylo = elgfit['M'] - elgfit['sigM']
        plt.fill_between([1., 1.5], ylo * np.ones(2), yup * np.ones(2), color='cadetblue', edgecolor='none')
        plt.text(.5, ylo, 'DESI ELGs', color='cadetblue', fontsize=15)

    #plt.fill_between(z_centers, qsomasses - qsoerrs, qsomasses + qsoerrs, alpha=0.5, color='royalblue')
    plt.text(1.5, 12.35, 'eBOSS QSOs', color='royalblue', fontsize=15)
    from halomodelpy import halo_growthrate
    zgrid, ms = halo_growthrate.evolve_halo_mass(12.5, 2.5, 0.01)
    plt.plot(zgrid, ms, c='k', ls='--', alpha=0.3)
    zgrid, ms = halo_growthrate.evolve_halo_mass(13., 2.5, 0.01)
    plt.plot(zgrid, ms, c='k', ls='--', alpha=0.3)
    plt.text(0.1, 13.3, 'Mean', rotation=-18, alpha=0.5)

    zgrid, ms = halo_growthrate.evolve_halo_mass(12.5, 2.5, 0.01, wantmean=False)
    plt.plot(zgrid, ms, c='k', ls='--', alpha=0.1)
    zgrid, ms = halo_growthrate.evolve_halo_mass(13., 2.5, 0.01, wantmean=False)
    plt.plot(zgrid, ms, c='k', ls='--', alpha=0.1)
    plt.text(0.1, 12.9, 'Median growth rate', rotation=-12, alpha=0.2)

    zranges = [1, 1.5, 2, 2.5]
    for j in range(3):
        qsoauto = read_pickle('results/fits/tomo/qso_fit_%s.pickle' % j)
        plt.fill_between([zranges[j], zranges[j+1]], (qsoauto['M'] - qsoauto['sigM'])*np.ones(2), (qsoauto['M'] + qsoauto['sigM'])*np.ones(2), color='cornflowerblue', edgecolor='none')

    #plt.scatter(1.6, rgautofit['M'], c='k')
    #plt.errorbar(1.6, rgautofit['M'], yerr=rgautofit['sigM'], ecolor='k', fmt='none', label='HzRG auto')




    plt.errorbar(1.6, hzrglensfit['M'], hzrglensfit['sigM'], markeredgecolor=rgcolor, markerfacecolor='none', ecolor=rgcolor, fmt='o', label='HzRG CMB Lensing')

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

def cross_cfs(vertical=False, elg=False):

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
        if elg:
            elgcf = read_pickle('results/cfs/tomo/rg_elgcf.pickle')
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
                axs[0].text(fit['modscales'][int(len(fit['modscales'])/2)],
                            fit['dmcf'][int(len(fit['modscales'])/2)]/2.,
                            'Matter', rotation=-12, color='k', fontsize=15)

                axs[0].text(fit['modscales'][int(len(fit['modscales']) / 2)],
                            autofit['b'] * fit['dmcf'][int(len(fit['modscales']) / 2)] / 1.5,
                            r'$b_{\mathrm{QSO}}$', rotation=-12, color='cornflowerblue', fontsize=15)

            axs[j].plot(fit['modscales'], fit['xfitcf'], c='firebrick', ls='dashed')
            axs[j].fill_between(fit['modscales'], (autofit['b'] - autofit['sigb']) * fit['dmcf'],
                                (autofit['b'] + autofit['sigb']) * fit['dmcf'],
                                color='cornflowerblue', edgecolor='none')

            axs[j].text(0.05, .93, r'$%s < z < %s$' % (zrange[j], zrange[j + 1]), transform=axs[j].transAxes,
                        fontsize=20)
            # axs[j].set_xscale('log')
            axs[j].set_xlim(2e-2, 9e-1)
            axs[j].get_xaxis().set_major_formatter(ScalarFormatter())

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
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 14))
    hzrgfit = read_pickle('results/fits/auto/hzrg.pickle')
    hzrgcf = read_pickle('results/cfs/auto/hzrg.pickle')

    izrgfit = read_pickle('results/fits/auto/izrg.pickle')
    izrgcf = read_pickle('results/cfs/auto/izrg.pickle')

    ax.errorbar(izrgcf['theta'], izrgcf['w_theta'], izrgcf['w_err'], fmt='o', color=izc)
    ax.plot(izrgfit['modscales'], izrgfit['dmcf'], ls='dotted', c='k')
    ax.plot(izrgfit['modscales'], izrgfit['autofitcf'], ls='dashed', c=izc)
    ax.set_yscale('log')
    ax.set_xscale('log')

    ax2.errorbar(hzrgcf['theta'], hzrgcf['w_theta'], hzrgcf['w_err'], fmt='o', color=hizc)
    ax2.plot(hzrgfit['modscales'], hzrgfit['dmcf'], ls='dotted', c='k')
    ax2.plot(hzrgfit['modscales'], hzrgfit['autofitcf'], ls='dashed', c=hizc)
    ax2.set_yscale('log')
    ax2.set_xscale('log')


    plt.subplots_adjust(hspace=0)
    plt.savefig(plotdir + 'autocorrelations.pdf')
    plt.close('all')

def dutycycle(izrgresult, zcenters, duty, loerr, hierr):
    plt.figure(figsize=(8,7))

    plt.errorbar(zcenters, duty, yerr=[loerr, hierr], color=hizc, fmt='o')
    plt.yscale('log')
    plt.ylim(1e-3, 3)
    plt.xlim(0, 2.75)

    plt.errorbar([izrgresult[0]], [izrgresult[1]], yerr=[[izrgresult[2]], [izrgresult[3]]], color=izc, fmt='o')


    plt.xlabel('Redshift')
    plt.ylabel('Duty cycle')
    plt.fill_between([0, 4], 1 * np.ones(2), 16 * np.ones(2), facecolor="none", hatch="///",
                     edgecolor="lightgrey", linewidth=0.0)
    plt.axhline(1, color='lightgrey')

    plt.savefig(plotdir + 'dutycycle.pdf')
    plt.close('all')

def energetics(zs, e, elo, ehi, qsowindinfo):
    import lumfunc
    plt.figure(figsize=(8, 7))

    thermal2kev = lumfunc.thermal_energy_mass(13., 2)
    bindingenergy = lumfunc.halobinding_energy(13., np.linspace(0., 4, 30))
    plt.plot(np.linspace(0., 4., 30), bindingenergy, c='grey', ls='dotted')
    plt.axhline(thermal2kev, c='grey', ls='--')
    plt.text(1., thermal2kev+0.15, r'$U_{\mathrm{th}}: M_h = 10^{13} \ h^{-1} M_{\odot}$,', color='grey')
    plt.text(1., thermal2kev - 0.2, r'$T$ = 2 keV, cosmic $f_{b}$', color='grey')
    plt.text(1., bindingenergy[15] + 0.05, '$U_{\mathrm{bind, gas}}$', color='grey', rotation=4)




    #maxqsowind = lumfunc.type1_windheatperhalo((1., 2.), 13., 0.005)
    #minqsowind = lumfunc.type1_windheatperhalo((1., 2.), 13., 0.0001)

    windcolor = 'green'
    for j in range(len(qsowindinfo)):
        thiswind = qsowindinfo[j]
        zrange, erange = thiswind[0], thiswind[1]

        plt.fill_between([zrange[0], zrange[1]], [erange[0], erange[0]], [erange[1], erange[1]], color='green', alpha=0.2,
                         edgecolor='none')
        #plt.axhline(erange[0], xmin=zrange[0], xmax=zrange[1], color=windcolor, alpha=0.5, ls='dotted')
        #plt.axhline(erange[1], xmin=zrange[0], xmax=zrange[1], color=windcolor, alpha=0.5, ls='dashed')


    plt.text(0.12, qsowindinfo[0][1][1] - 0.15, 'QSO winds', color=windcolor, fontsize=8)
    plt.text(0.1, qsowindinfo[0][1][1] - 0.3, '(Hopkins+07)', color=windcolor, fontsize=8)
    plt.text(0.03, qsowindinfo[0][1][1] - 0.45, r'$M_h > 10^{12.8} \ h^{-1} M_{\odot}$', color=windcolor, fontsize=8)

    plt.hlines(qsowindinfo[1][1][1], xmin=qsowindinfo[1][0][0], xmax=qsowindinfo[1][0][1], color=windcolor, ls='dashed')
    plt.hlines(qsowindinfo[1][1][0], xmin=qsowindinfo[1][0][0], xmax=qsowindinfo[1][0][1], color=windcolor, ls='dashed')
    plt.text(0.7, qsowindinfo[1][1][1]+0.1, r'$0.5\%$', color=windcolor)
    plt.text(0.6, qsowindinfo[1][1][0] - 0.2, r'$\frac{L_{\mathrm{wind}}}{L_{\mathrm{bol}}} = 0.1\%$', color=windcolor)




    #plt.axhline(lumfunc.lx_energy(44, (1., 1.5)), label=r'$\Delta t \times L_x=44$')

    plt.errorbar(zs, e, yerr=[elo, ehi], color='firebrick', fmt='o')
    plt.ylabel(r'log$\langle$ Heating energy per halo $\rangle$ [erg]')
    plt.xlabel('Redshift')
    plt.xlim(0, 2.5)
    plt.legend(fontsize=10, loc='lower right')
    plt.savefig(plotdir + 'energetics.pdf')
    plt.close('all')


#halomass()
#cross_cfs()
#hzrg_lenscorr()
#autocorrs()