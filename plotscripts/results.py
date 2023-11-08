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

hic = 'firebrick'
midc = '#ff4d00'
lowc = '#ffc100'
qso_c = 'teal'


from halomodelpy import cosmo
def z2lbt(zs):
    return cosmo.col_cosmo.lookbackTime(zs)
def lbt2z(lbs):
    return cosmo.col_cosmo.lookbackTime(lbs, inverse=True)


def halomass(elg=False ,lzrg=True):
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

    hzrgautofit = read_pickle('results/fits/auto/hzrg.pickle')

    hzrglensfit = read_pickle('results/lensfits/hzrg.pickle')
    izrgfit = read_pickle('results/fits/auto/izrg.pickle')


    fig, ax = plt.subplots(figsize=(8, 7))

    if lzrg:
        lzrgfit = read_pickle('results/fits/tomo/rg_lowz_fit.pickle')
        plt.errorbar([lzrgfit['eff_z']], [lzrgfit['Mx']], [lzrgfit['sigMx']], color=lowc, fmt='X', label=r'LzRG $\times$ BOSS')

    plt.errorbar(izrgfit['eff_z'], izrgfit['M'], izrgfit['sigM'], color=midc, fmt='o', label='IzRG auto')

    plt.errorbar(z_centers, radmasses, yerr=raderrs, label=r'HzRGs $\times$ eBOSS QSOs', color=hic, fmt='X')

    if elg:
        elgfit = read_pickle('results/fits/elgfit.pickle')
        elgx = read_pickle('results/fits/tomo/rg_elg.pickle')
        plt.errorbar(elgx['eff_z'], elgx['Mx'], elgx['sigMx'], color=hic, fmt='*', ms=10, label=r'HzRGs $\times$ DESI ELGs')
        yup = elgfit['M'] + elgfit['sigM']
        ylo = elgfit['M'] - elgfit['sigM']
        plt.fill_between([1., 1.5], ylo * np.ones(2), yup * np.ones(2), color='cadetblue', edgecolor='none')
        plt.text(.5, ylo, 'DESI ELGs', color='cadetblue', fontsize=15)

    #plt.fill_between(z_centers, qsomasses - qsoerrs, qsomasses + qsoerrs, alpha=0.5, color='royalblue')
    plt.text(1.4, 12.3, 'eBOSS QSOs', color=qso_c, fontsize=15, alpha=0.8)
    from halomodelpy import halo_growthrate
    zgrid, ms = halo_growthrate.evolve_halo_mass(12.5, 3, 0.01)
    plt.plot(zgrid, ms, c='k', ls='--', alpha=0.3)
    plt.text(0.1, ms[0] - 0.2, 'Mean', rotation=-18, alpha=0.2)
    zgrid, ms = halo_growthrate.evolve_halo_mass(13., 3, 0.01)
    plt.plot(zgrid, ms, c='k', ls='--', alpha=0.3)

    zgrid, ms = halo_growthrate.evolve_halo_mass(12.5, 3, 0.01, wantmean=False)
    plt.plot(zgrid, ms, c='k', ls='--', alpha=0.1)
    plt.text(0.1, ms[0] - 0.25, 'Median growth rate', rotation=-15, alpha=0.2)
    zgrid, ms = halo_growthrate.evolve_halo_mass(13., 3, 0.01, wantmean=False)
    plt.plot(zgrid, ms, c='k', ls='--', alpha=0.1)



    zranges = [1, 1.5, 2, 3]
    for j in range(3):
        qsoauto = read_pickle('results/fits/tomo/qso_fit_%s.pickle' % j)
        plt.fill_between([zranges[j], zranges[j+1]], (qsoauto['M'] - qsoauto['sigM'])*np.ones(2), (qsoauto['M'] + qsoauto['sigM'])*np.ones(2), color=qso_c, edgecolor='none', alpha=0.5)


    plt.errorbar(hzrgautofit['eff_z'], hzrgautofit['M'], yerr=hzrgautofit['sigM'], color=hic, fmt='o', label='HzRG auto')
    plt.errorbar(hzrglensfit['eff_z'], hzrglensfit['M'], hzrglensfit['sigM'], markeredgecolor=hic, markerfacecolor='none', ecolor=hic, fmt='o', label='HzRG CMB Lensing')

    #plt.scatter(1.6, autofit['M'])
    #plt.errorbar(1.6, autofit['M'], autofit['sigM'], ecolor='k', fmt='none')
    #plt.fill_between([1, 2.5], (autofit['M'] - autofit['sigM'])*np.ones(2), (autofit['M'] + autofit['sigM'])*np.ones(2), color=rgcolor, alpha=0.5, edgecolor='none')
    #im2 = plt.imshow(np.outer(np.ones(len(dndz[0])), dndz[1]), cmap=cm.Reds,
    #                 extent=[0.1, 4., autofit['M'] - autofit['sigM'], autofit['M'] + autofit['sigM']],
    #                 interpolation="bicubic", alpha=.4, aspect="auto")
    #plt.text(1.6, autofit['M'] + 0.1, 'HzRG auto', color=rgcolor, fontsize=15)


    #plt.fill_between(laurentzspace, mlit[2], mlit[1], color='seagreen', alpha=0.2, edgecolor='none')
    plt.xlim(0, 3.)
    plt.fill_between([0, 0.1], [14, 14], [15, 15], color='maroon', alpha=0.2, edgecolor='none')
    plt.text(0.15, 14.1, 'Clusters', color='maroon', alpha=0.5)

    plt.ylim(12.1, 14.4)

    plt.tick_params(which='both', top=False)
    secax = ax.secondary_xaxis('top', functions=(z2lbt, lbt2z))
    secax.set_xticks([3, 7, 10])
    secax.set_xlabel('Lookback time [Gyr]')


    plt.xlabel('Redshift')
    plt.legend(fontsize=15)
    plt.ylabel(r'log$(\mathrm{Halo \ mass} \  [h^{-1} M_{\odot}]$)')

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
        plt.ylim(1e-4, 9e-2)
        axs[0].set_ylabel(r'Angular Cross-Correlation Function $w(\theta)$')
        plt.subplots_adjust(wspace=0)
        if elg:
            elgcf = read_pickle('results/cfs/tomo/rg_elgcf.pickle')
            axs[0].scatter(elgcf['theta'], elgcf['w_theta'], c='cadetblue')
            axs[0].errorbar(elgcf['theta'], elgcf['w_theta'], yerr=elgcf['w_err'], ecolor='cornflowerblue', fmt='none')
        zrange = [1, 1.5, 2, 3]

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
                            r'$b_{\mathrm{QSO}}$', rotation=-12, color=qso_c, fontsize=15)

            axs[j].plot(fit['modscales'], fit['xfitcf'], c='firebrick', ls='dashed')
            axs[j].fill_between(fit['modscales'], (autofit['b'] - autofit['sigb']) * fit['dmcf'],
                                (autofit['b'] + autofit['sigb']) * fit['dmcf'],
                                color=qso_c, edgecolor='none', alpha=0.6)

            axs[j].text(0.05, .93, r'$%s < z < %s$' % (zrange[j], zrange[j + 1]), transform=axs[j].transAxes,
                        fontsize=20)
            # axs[j].set_xscale('log')
            axs[j].set_xlim(2e-2, 9e-1)
            axs[j].get_xaxis().set_major_formatter(ScalarFormatter())

    plt.savefig(plotdir + 'crosscfs.pdf')
    plt.close('all')

def cross_lzrg():
    plt.figure(figsize=(8, 7))

    autofit = read_pickle('results/fits/tomo/lowz_fit.pickle')
    xcf = read_pickle('results/cfs/tomo/rg_lowz.pickle')
    xfit = read_pickle('results/fits/tomo/rg_lowz_fit.pickle')
    plt.plot(xfit['modscales'], xfit['dmcf'], c='k', ls='dotted')
    plt.errorbar(xcf['theta'], xcf['w_theta'], yerr=xcf['w_err'], color=lowc, fmt='o')
    plt.plot(xfit['modscales'], xfit['xfitcf'], c=lowc, ls='dashed')
    plt.xscale('log')
    plt.yscale('log')

    plt.xlim(2e-2, 2)

    plt.savefig(plotdir + 'lzrg_xcf.pdf')
    plt.close('all')




def hzrg_lenscorr():


    hzrglens = read_pickle('results/lenscorr/hzrg.pickle')
    hzrglensfit = read_pickle('results/lensfits/hzrg.pickle')
    hzrgdndz = read_pickle('results/dndz/hzrg.pickle')
    from halomodelpy import hm_calcs

    fig, ax = plt.subplots(figsize=(8, 7))
    ellgrid = np.logspace(0.5, 3.5, 1000)
    hm = hm_calcs.halomodel(hzrgdndz)
    plt.errorbar(hzrglens['ell'], hzrglens['cl'], hzrglens['cl_err'], fmt='o', color=hic)
    hm.set_powspec()
    plt.plot(ellgrid, hm.get_c_ell_kg(ellgrid), c='k', ls='dotted')
    hm.set_powspec(log_meff=hzrglensfit['M'])
    plt.plot(ellgrid, hm.get_c_ell_kg(ellgrid), c=hic, ls='dashed')
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

def lenscorrs(hod=True, both=False):



    hzrglens = read_pickle('results/lenscorr/hzrg.pickle')
    hzrglensfit = read_pickle('results/lensfits/hzrg.pickle')
    hzrgdndz = read_pickle('results/dndz/hzrg.pickle')
    modells = np.logspace(1, 4, 200)
    from halomodelpy import hm_calcs, mcmc
    if both:
        fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 14), sharex=True)
    else:
        fig, ax2 = plt.subplots(figsize=(8,7))
    plt.xlim(50, 5000)
    plt.xscale('log')



    ellgrid = np.logspace(0.5, 3.8, 1000)




    hm = hm_calcs.halomodel(hzrgdndz)
    ax2.errorbar(hzrglens['ell'], hzrglens['cl'], hzrglens['cl_err'], fmt='o', color=hic)
    hm.set_powspec()
    ax2.plot(ellgrid, hm.get_c_ell_kg(ellgrid), c='k', ls='dotted')
    hm.set_powspec(log_meff=hzrglensfit['M'])
    ax2.plot(ellgrid, hm.get_c_ell_kg(ellgrid), c=hic, ls='dashed', label='Linear fit')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'Multipole $\ell$')
    ax2.set_ylabel('Cross-power $C_{\ell}^{\kappa g}$')
    #ax2.set_ylim(np.min(hzrglens['cl']) / 3., np.max(hzrglens['cl']) * 2.)
    ax2.set_ylim(9e-10, 9e-7)
    ax2.text(0.7, .93, r'HzRG Lensing', transform=ax2.transAxes,
                fontsize=20, color=hic)
    ax2.set_yscale('log')
    ax2.text(200, 3e-8, 'Matter', rotation=-35, fontsize=15)
    ax2.text(2000, 4e-9, 'Clustering HOD', rotation=-40, fontsize=12, color=hic, alpha=0.5)
    if hod:

        dndz = read_pickle('results/dndz/hzrg.pickle')
        hzrgchain = read_pickle('results/hod/hzrg.pickle')['chain']
        hm = hm_calcs.halomodel(dndz)

        newchain = hzrgchain[np.random.choice(len(hzrgchain), 100, replace=False)]
        for j in range(len(newchain)):
            paramset = list(newchain[j])
            hm.set_powspec(hodparams=mcmc.parse_params(paramset, ['M', 'sigM', 'M1', 'alpha']))
            xcorr = hm.get_c_ell_kg(ells=modells)
            ax2.plot(modells, xcorr, alpha=0.1, rasterized=True, c=hic, ls='dotted')

    ax2.legend(loc='lower left')
    if both:

        izrglens = read_pickle('results/lenscorr/izrg.pickle')
        izrglensfit = read_pickle('results/lensfits/izrg.pickle')
        izrgdndz = read_pickle('results/dndz/izrg.pickle')

        hm = hm_calcs.halomodel(izrgdndz)
        ax.errorbar(izrglens['ell'], izrglens['cl'], izrglens['cl_err'], fmt='o', color=midc)
        hm.set_powspec()
        ax.plot(ellgrid, hm.get_c_ell_kg(ellgrid), c='k', ls='dotted')
        hm.set_powspec(log_meff=izrglensfit['M'])
        ax.plot(ellgrid, hm.get_c_ell_kg(ellgrid), c=midc, ls='dashed')
        ax.set_yscale('log')
        ax.set_ylabel('Cross-power $C_{\ell}^{\kappa g}$')
        # ax.set_ylim(np.min(izrglens['cl']) / 3., np.max(izrglens['cl']) * 2.)
        ax.set_ylim(3e-9, 9e-7)
        ax.set_yscale('log')
        ax.text(0.7, .93, r'IzRG Lensing', transform=ax.transAxes,
                fontsize=20, color=midc)

        if hod:

            dndz = read_pickle('results/dndz/izrg.pickle')
            izrgchain = read_pickle('results/hod/izrg.pickle')['chain']
            hm = hm_calcs.halomodel(dndz)

            newchain = izrgchain[np.random.choice(len(izrgchain), 100, replace=False)]
            for j in range(len(newchain)):
                paramset = list(newchain[j])
                hm.set_powspec(hodparams=mcmc.parse_params(paramset, ['M', 'sigM', 'M1', 'alpha']))
                xcorr = hm.get_c_ell_kg(ells=modells)
                ax.plot(modells, xcorr, alpha=0.1, rasterized=True, c=midc, ls='dotted')

    plt.subplots_adjust(hspace=0)
    plt.savefig(plotdir+'lensing.pdf')
    plt.close('all')


def autocorrs(hod=True, showlens=True):

    from halomodelpy import cosmo

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 14), sharex=True)
    plt.xlim(5e-3, 1e0)
    plt.xlabel(r'Separation $\theta$ [deg]')
    hzrgfit = read_pickle('results/fits/auto/hzrg.pickle')
    hzrgcf = read_pickle('results/cfs/auto/hzrg.pickle')
    hzrg_refangle = cosmo.rp2angle(1., hzrgfit['eff_z'])

    izrgfit = read_pickle('results/fits/auto/izrg.pickle')
    izrgcf = read_pickle('results/cfs/auto/izrg.pickle')
    izrg_refangle = cosmo.rp2angle(1., izrgfit['eff_z'])

    linidx, nonlinidx = izrgcf['linidx'], izrgcf['nonlinidx']
    lintheta, lincf, linerr = izrgcf['theta'][linidx], izrgcf['w_theta'][linidx], izrgcf['w_err'][linidx]
    nonlintheta, nonlincf, nonlinerr = izrgcf['theta'][nonlinidx], izrgcf['w_theta'][nonlinidx], izrgcf['w_err'][nonlinidx]
    #if showlens:
        #izrglensfit = read_pickle('results/lensfits/hzrg.pickle')
        #lensb, lenssigb = izrglensfit['b'], izrglensfit['sigb']
        #ax.fill_between(izrgfit['modscales'], ((lensb - lenssigb) ** 2) * izrgfit['dmcf'],
        #                 ((lensb + lenssigb) ** 2) * izrgfit['dmcf'], color=midc, alpha=0.1, edgecolor='none', label=r'$b_{\mathrm{CMB}}$')

    ax.errorbar(lintheta, lincf, linerr, fmt='o', color=midc)
    ax.errorbar(nonlintheta, nonlincf, nonlinerr, fmt='o', markerfacecolor='none', markeredgecolor=midc, ecolor=midc)
    ax.plot(izrgfit['modscales'], izrgfit['dmcf'], ls='dotted', c='k')
    ax.text(izrgfit['modscales'][int(len(izrgfit['dmcf']) / 2)],
             izrgfit['dmcf'][int(len(izrgfit['dmcf']) / 2)] / 2., 'Matter', rotation=-12, fontsize=15)
    ax.plot(izrgfit['modscales'], izrgfit['autofitcf'], ls='dashed', c=midc)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'Autocorrelation function $w(\theta)$')
    ax.text(.65, .9, 'IzRG Clustering', transform=ax.transAxes, fontsize=20, color=midc)

    ax.axvline(izrg_refangle, color='k', alpha=0.3, ls='dotted')
    ax.text(izrg_refangle * 1.3, 4e-1, r'$r_p \sim 1$ Mpc/h', rotation=90, color='k', alpha=0.5, fontsize=15)


    if hod:
        modthetas = np.logspace(-2.5, 0., 100)
        from halomodelpy import hm_calcs, mcmc
        dndz = read_pickle('results/dndz/izrg.pickle')
        izrgchain = read_pickle('results/hod/izrg.pickle')['chain']
        hm = hm_calcs.halomodel(dndz)

        newchain = izrgchain[np.random.choice(len(izrgchain), 100, replace=False)]
        for j in range(len(newchain)):
            paramset = list(newchain[j])
            hm.set_powspec(hodparams=mcmc.parse_params(paramset, ['M', 'sigM', 'M1', 'alpha']))
            cf = hm.get_ang_cf(modthetas)
            ax.plot(modthetas, cf, alpha=0.1, rasterized=True, c=midc, ls='dotted')


        dndz = read_pickle('results/dndz/hzrg.pickle')
        hzrgchain = read_pickle('results/hod/hzrg.pickle')['chain']
        hm = hm_calcs.halomodel(dndz)

        newchain = hzrgchain[np.random.choice(len(hzrgchain), 100, replace=False)]
        for j in range(len(newchain)):
            paramset = list(newchain[j])
            hm.set_powspec(hodparams=mcmc.parse_params(paramset, ['M', 'sigM', 'M1', 'alpha']))
            cf = hm.get_ang_cf(modthetas)
            ax2.plot(modthetas, cf, alpha=0.1, rasterized=True, c=hic, ls='dotted')



    linidx, nonlinidx = hzrgcf['linidx'], hzrgcf['nonlinidx']
    lintheta, lincf, linerr = hzrgcf['theta'][linidx], hzrgcf['w_theta'][linidx], hzrgcf['w_err'][linidx]
    nonlintheta, nonlincf, nonlinerr = hzrgcf['theta'][nonlinidx], hzrgcf['w_theta'][nonlinidx], hzrgcf['w_err'][
        nonlinidx]



    ax2.errorbar(lintheta, lincf, linerr, fmt='o', color=hic)
    ax2.errorbar(nonlintheta, nonlincf, nonlinerr, fmt='o', markerfacecolor='none', markeredgecolor=hic, ecolor=hic)
    ax2.plot(hzrgfit['modscales'], hzrgfit['dmcf'], ls='dotted', c='k')
    ax2.axvline(hzrg_refangle, color='k', alpha=0.3, ls='dotted')


    if showlens:
        hzrglensfit = read_pickle('results/lensfits/hzrg.pickle')
        lensb, lenssigb = hzrglensfit['b'], hzrglensfit['sigb']
        ax2.fill_between(hzrgfit['modscales'], ((lensb-lenssigb) ** 2) * hzrgfit['dmcf'],
                         ((lensb+lenssigb) ** 2) * hzrgfit['dmcf'], color=hic, alpha=0.1, edgecolor='none',
                         label=r'$b_{\mathrm{CMB}}$')





    ax2.text(hzrgfit['modscales'][int(len(hzrgfit['dmcf'])/2)],
             hzrgfit['dmcf'][int(len(hzrgfit['dmcf'])/2)] / 2., 'Matter', rotation=-12, fontsize=15)
    ax2.plot(hzrgfit['modscales'], hzrgfit['autofitcf'], ls='dashed', c=hic, label='Linear fit')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_ylabel(r'Autocorrelation function $w(\theta)$')
    ax2.set_ylim(2e-4, 2e0)
    ax2.text(.65, .9, 'HzRG Clustering', transform=ax2.transAxes, fontsize=20, color=hic)

    ax2.legend(loc='lower left')
    plt.subplots_adjust(hspace=0)
    plt.savefig(plotdir + 'autocorrelations.pdf')
    plt.close('all')


def corners(smooth=1, fsats=True):
    import corner
    from halomodelpy import hod_model
    izrgchain = read_pickle('results/hod/izrg.pickle')['chain']
    hzrgchain = read_pickle('results/hod/hzrg.pickle')['chain']
    labels = [r'log$M_{\mathrm{min}}$', r'$\sigma_{\mathrm{log}M}$', r'log$M_1$', r'$\alpha$']
    if fsats:
        dndz = read_pickle('results/dndz/izrg.pickle')
        izrgchain = np.insert(izrgchain, 4, np.log10(hod_model.fsat_dndz(dndz, izrgchain, ['M', 'sigM', 'M1', 'alpha'])), 1)

        dndz = read_pickle('results/dndz/hzrg.pickle')
        hzrgchain = np.insert(hzrgchain, 4, np.log10(hod_model.fsat_dndz(dndz, hzrgchain, ['M', 'sigM', 'M1', 'alpha'])), 1)
        labels.append(r'log$f_{\mathrm{sat}}$')

    fig = corner.corner(izrgchain, color=midc, alpha=0.3, smooth=smooth, labels=labels,
                        plot_datapoints=False, levels=(1 - np.exp(-0.5), 1 - np.exp(-2)),
                        plot_density=False,
                        fill_contours=True,
                        )
    corner.corner(hzrgchain, fig=fig, color=hic, alpha=0.3, smooth=smooth,
                  plot_datapoints=False, levels=(1 - np.exp(-0.5), 1 - np.exp(-2)),
                  plot_density=False,
                  fill_contours=True,
                  )

    plt.savefig(plotdir + 'corner.pdf')
    plt.close('all')

def hods():
    fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 14), sharex=True)
    from halomodelpy import hod_model
    izrgchain = read_pickle('results/hod/izrg.pickle')['chain']
    hzrgchain = read_pickle('results/hod/hzrg.pickle')['chain']
    plt.xlim(12, 14.5)

    newchain = izrgchain[np.random.choice(len(izrgchain), 100, replace=False)]
    for j in range(len(newchain)):
        paramset = list(newchain[j])
        hod = hod_model.zheng_hod(paramset, ['M', 'sigM', 'M1', 'alpha'])
        ax.plot(hod['mgrid'], hod['hod'], alpha=0.2, rasterized=True, c=midc, ls='dotted')
    ax.set_ylim(1e-1, 5e1)
    ax.set_ylabel('Number per halo')
    ax.set_yscale('log')

    newchain = hzrgchain[np.random.choice(len(hzrgchain), 100, replace=False)]
    for j in range(len(newchain)):
        paramset = list(newchain[j])
        hod = hod_model.zheng_hod(paramset, ['M', 'sigM', 'M1', 'alpha'])
        ax2.plot(hod['mgrid'], hod['hod'], alpha=0.2, rasterized=True, c=hic, ls='dotted')
    ax2.set_ylim(1e-1, 5e1)
    ax2.set_ylabel('Number per halo')
    ax2.set_yscale('log')


    ax2.set_xlabel(r'log$(\mathrm{Halo \ mass} \ [h^{-1} M_{\odot}])$')
    plt.subplots_adjust(hspace=0)
    plt.savefig(plotdir + 'hods.pdf')
    plt.close('all')



def dutycycle(lzrgresult, izrgresult, hzrgresult, lensresult, zcenters, duty, loerr, hierr):
    fig, ax = plt.subplots(figsize=(8,7))

    plt.errorbar(zcenters, duty, yerr=[loerr, hierr], color=hic, fmt='X')
    plt.yscale('log')
    plt.ylim(1e-3, 3)
    plt.xlim(0, 2.75)
    plt.errorbar([lzrgresult[0]], [lzrgresult[1]], yerr=[[lzrgresult[2]], [lzrgresult[3]]], color=lowc, fmt='o')
    plt.errorbar([izrgresult[0]], [izrgresult[1]], yerr=[[izrgresult[2]], [izrgresult[3]]], color=midc, fmt='o')
    plt.errorbar([hzrgresult[0]], [hzrgresult[1]], yerr=[[hzrgresult[2]], [hzrgresult[3]]], color=hic, fmt='o')

    plt.errorbar([lensresult[0]], [lensresult[1]], yerr=[[lensresult[2]], [lensresult[3]]],
                 markeredgecolor=hic, markerfacecolor='none', ecolor=hic, fmt='o')

    laurent=True
    if laurent:
        import pandas as pd
        low = pd.read_csv('results/duty/laurent_duty_low.csv', names=['z', 'f'])
        high = pd.read_csv('results/duty/laurent_duty_high.csv', names=['z', 'f'])
        highmassfrac = 1.

        plt.fill_between(low['z'], highmassfrac*low['f'],
                         highmassfrac*high['f'], color=qso_c, alpha=0.2, edgecolor='none')
        plt.text(1.3, 0.7e-2, 'Laurent+17 QSOs', color=qso_c, alpha=0.6, fontsize=20)

    plt.xlabel('Redshift')
    plt.ylabel(r'Duty cycle $f_{\mathrm{duty}}$')
    plt.fill_between([0, 4], 1 * np.ones(2), 16 * np.ones(2), facecolor="none", hatch="///",
                     edgecolor="lightgrey", linewidth=0.0)
    plt.axhline(1, color='lightgrey')
    plt.text(2.1, 1.5, r'$f_{\mathrm{duty}} > 1$', fontsize=20)

    plt.tick_params(which='both', top=False)
    secax = ax.secondary_xaxis('top', functions=(z2lbt, lbt2z))
    secax.set_xticks([3, 7, 10])
    secax.set_xlabel('Lookback time [Gyr]')

    """def duty2lifetime(duty):
        return cosmo.col_cosmo.age(0)*duty
    def lifetime2duty(life):
        return life / cosmo.col_cosmo.age(0)

    plt.tick_params(which='both', right=False)
    thirdax = ax.secondary_yaxis('right', functions=(duty2lifetime, lifetime2duty))
    thirdax.set_ylabel(r'Lifetime $f_{\mathrm{duty}} t_H$ [Gyr]')"""



    plt.savefig(plotdir + 'dutycycle.pdf')
    plt.close('all')

def energetics(izrgresult, zs, e, elo, ehi, qsowindinfo):
    import lumfunc
    plt.figure(figsize=(8, 7))

    thermal2kev = lumfunc.thermal_energy_mass(13., 2)
    bindingenergy = lumfunc.halobinding_energy(13., np.linspace(0., 4, 30))
    plt.plot(np.linspace(0., 4., 30), bindingenergy, c='grey', ls='dotted')
    plt.axhline(thermal2kev, c='grey', ls='--')
    plt.text(0.2, thermal2kev+0.1, r'$U_{\mathrm{th}}: M_h = 10^{13} \ h^{-1} M_{\odot}$,', color='grey')
    plt.text(0.2, thermal2kev - 0.15, r'$T$ = 2 keV, cosmic $f_{b}$', color='grey')
    plt.text(0.2, bindingenergy[2] + 0.05, '$U_{\mathrm{bind, gas}}$', color='grey', rotation=4)

    plt.errorbar([izrgresult[0]], [izrgresult[1]], yerr=[[izrgresult[2]], [izrgresult[3]]], color=midc, fmt='o')

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


    plt.text(1.6, 57.7, 'QSO winds', color=windcolor, fontsize=8)
    plt.text(1.6, 57.5, '(Hopkins+07)', color=windcolor, fontsize=8)
    plt.text(1.5, 57.3, r'$M_h > 10^{12.8} \ h^{-1} M_{\odot}$', color=windcolor, fontsize=8)

    plt.hlines(qsowindinfo[1][1][1], xmin=qsowindinfo[1][0][0], xmax=qsowindinfo[1][0][1], color=windcolor, ls='dashed')
    plt.hlines(qsowindinfo[1][1][0], xmin=qsowindinfo[1][0][0], xmax=qsowindinfo[1][0][1], color=windcolor, ls='dashed')
    plt.text(0.7, qsowindinfo[1][1][1]+0.1, r'$0.5\%$', color=windcolor)
    plt.text(0.6, qsowindinfo[1][1][0] - 0.2, r'$\frac{L_{\mathrm{wind}}}{L_{\mathrm{bol}}} = 0.1\%$', color=windcolor)




    #plt.axhline(lumfunc.lx_energy(44, (1., 1.5)), label=r'$\Delta t \times L_x=44$')

    plt.errorbar(zs, e, yerr=[elo, ehi], color='firebrick', fmt='o')
    plt.ylabel(r'log$\langle$ Heating energy per halo $\rangle$ [erg]')
    plt.xlabel('Redshift')
    plt.xlim(0, 3)
    plt.ylim(56.8, 62)
    plt.legend(fontsize=10, loc='lower right')
    plt.savefig(plotdir + 'energetics.pdf')
    plt.close('all')

def avghalopower(lzrgs, izrgs, hzrgs, qsowindinfo):
    imgs = True
    import lumfunc
    fig, ax = plt.subplots(figsize=(8, 7))
    if imgs:
        import matplotlib.image as image
        im = image.imread('results/imgs/mrk231.jpg')
        ax.imshow(im, aspect='auto', extent=(1.8, 2.5, 39.5, 41.), zorder=-1)
        im = image.imread('results/imgs/HercA.jpg')
        ax.imshow(im, aspect='auto', extent=(0.25, 0.7, 43.3, 44.3), zorder=-1)

    ax.errorbar([lzrgs[0]], [lzrgs[1]], yerr=[[lzrgs[2]], [lzrgs[3]]], color=lowc, fmt='X')

    ax.text(.85, 43.7, r'$L_{150 \ \mathrm{MHz}} \gtrsim 10^{25}$ W/Hz', fontsize=15)
    ax.errorbar([izrgs[0]], [izrgs[1]], yerr=[[izrgs[2]], [izrgs[3]]], color=midc, fmt='o')

    # maxqsowind = lumfunc.type1_windheatperhalo((1., 2.), 13., 0.005)
    # minqsowind = lumfunc.type1_windheatperhalo((1., 2.), 13., 0.0001)

    ax.fill_between(qsowindinfo[0], qsowindinfo[1], qsowindinfo[2], color=qso_c, alpha=0.05, edgecolor='none')
    ax.plot(qsowindinfo[0], qsowindinfo[2], color=qso_c, alpha=0.3)
    ax.plot(qsowindinfo[0], qsowindinfo[1], color=qso_c, alpha=0.2, ls='dashed')

    z_arrow = qsowindinfo[0][0] + (qsowindinfo[0][1] - qsowindinfo[0][0])/2.
    yarrow = np.interp(z_arrow, qsowindinfo[0], qsowindinfo[2])

    plt.arrow(z_arrow, yarrow, dx=0, dy=-0.3, facecolor=qso_c, width=0.005, head_width=0.05, alpha=0.3, edgecolor='none')
    plt.text(z_arrow + 0.2, yarrow+0.15, 'QSO winds', rotation=25, color=qso_c, alpha=0.6, fontsize=15)
    plt.text(z_arrow + 0.2, yarrow - 0.15, 'Hopkins+07', rotation=25, color=qso_c, alpha=0.6, fontsize=15)

    plt.text(z_arrow - 0.2, yarrow-0.2, r'$f_{\mathrm{wind}} = 0.5\%$', rotation=45, color=qso_c, alpha=0.6)
    plt.text(z_arrow - 0.2, qsowindinfo[1][0]+0.2, r'$f_{\mathrm{wind}} = 0.1\%$', rotation=45, color=qso_c, alpha=0.6)

    s_per_year = 3.154e7
    s_per_gyr = s_per_year * 1e9
    def hz2pergyr_log(x):
        return x + np.log10(s_per_gyr)
    def pergyr2hz_log(x):
        return x - np.log10(s_per_gyr)



    plt.tick_params(which='both', right=False)
    plt.tick_params(which='both', top=False)

    secax = ax.secondary_yaxis('right', functions=(hz2pergyr_log, pergyr2hz_log))
    secax.set_ylabel('log [erg/Gyr]')

    thirdax = ax.secondary_xaxis('top', functions=(z2lbt, lbt2z))
    thirdax.set_xticks([3, 7, 10])
    thirdax.set_xlabel('Lookback time [Gyr]')

    #plt.text(1.6, 57.7, 'QSO winds', color=windcolor, fontsize=8)
    #plt.text(1.6, 57.5, '(Hopkins+07)', color=windcolor, fontsize=8)
    #plt.text(1.5, 57.3, r'$M_h > 10^{12.8} \ h^{-1} M_{\odot}$', color=windcolor, fontsize=8)


    # plt.axhline(lumfunc.lx_energy(44, (1., 1.5)), label=r'$\Delta t \times L_x=44$')
    zs, e, elo, ehi = hzrgs
    ax.errorbar(zs, e, yerr=[elo, ehi], color='firebrick', fmt='X')
    ax.set_ylabel(r'log$\langle$Kinetic power per halo [erg/s]$\rangle$')
    ax.set_xlabel('Redshift')

    ax.text(0.05, .93, r'$M_h > 10^{13} h^{-1} M_{\odot}$', transform=ax.transAxes, fontsize=18)
    ax.set_xlim(0, 3)
    ax.set_ylim(39, 45.4)
    #plt.ylim(56.8, 62)
    #ax.legend(fontsize=10, loc='lower right')



    plt.savefig(plotdir + 'halopower.pdf', dpi=300)
    plt.close('all')
