from corrfunc_helper import twoPointCFs
from halomodelpy import clustering_fit, redshift_helper, pipeline, lensing_fit, cosmo
import numpy as np

import lumfunc
import sample
from astropy.table import Table
from plotscripts import mass_v_prop
import pickle
import glob
import os
from plotscripts import results
plotdir = '/home/graysonpetter/Dropbox/radioplots/'


def retrieve_eboss_autocf():
    with open('results/cfs/ebossqso_cf.pickle', 'rb') as f:
        autocf = pickle.load(f)
    return autocf
def write_pickle(filename, pickleobj):
    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(pickleobj, f)

def read_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)



def dndz_error(cf, cat, nzbins, zrange, nboots=100):
    from scipy.signal import savgol_filter
    zs = sample.treat_dndz_pdf(cat)

    for j in range(nboots):
        bootzs = np.random.choice(zs, len(zs), replace=True)
        dndz = redshift_helper.dndz_from_z_list(bootzs, nbins=nzbins, zrange=zrange)

        smoothed = savgol_filter(dndz[1], int(nzbins/10.), polyorder=1)
        dndz = redshift_helper.norm_z_dist((dndz[0], smoothed))


def draw_hzrg_dndz(zs, nbins_mean=30, nbins_std=3, zrange=(0.1, 4.)):
    min_smooth_hzrg = 0.025
    max_smooth_hzrg = 0.075
    dndz = redshift_helper.dndz_from_z_list(zs, nbins=nbins_mean, zrange=zrange)
    bootzs = np.random.choice(zs, len(zs), replace=True)
    dndz_new = redshift_helper.dndz_from_z_list(bootzs, nbins=int(np.random.normal(nbins_mean, nbins_std)), zrange=zrange)
    smoothed = redshift_helper.spline_dndz(dndz_new, spline_k=4, smooth=np.random.uniform(min_smooth_hzrg, max_smooth_hzrg))(dndz[0])
    smoothed[np.where(smoothed < 0)] = 0.
    dndz = redshift_helper.norm_z_dist((dndz[0], smoothed))
    return dndz







def linear_cf(cf, eff_z, min_r=5., max_r=25.):
    """
    Cull a measured angular correlation function to only certain linear scales
    (e.g. 5-25 projected Mpc/h) , for fitting with linear model
    :param cf:
    :param eff_z:
    :return:
    """
    linear_thetas = cosmo.rp2angle(np.array([min_r, max_r]), z=eff_z)
    mintheta, maxtheta = linear_thetas[0], linear_thetas[1]
    linear_idxs = np.where((cf['theta_bins'] >= mintheta) & (cf['theta_bins'] <= maxtheta))[0]
    lincf = {}
    lincf['theta_bins'] = cf['theta_bins'][linear_idxs]
    lincf['theta'] = cf['theta'][linear_idxs[:-1]]
    lincf['w_theta'] = cf['w_theta'][linear_idxs[:-1]]
    lincf['w_err'] = cf['w_err'][linear_idxs[:-1]]
    cf['linidx'] = linear_idxs[:-1]
    cf['nonlinidx'] = np.where(np.logical_not(np.in1d(np.arange(len(cf['theta'])), cf['linidx'])))[0]
    return cf, lincf

def autocorr_hzrgs(rpscales, nzbins=30, zrange=(0.1, 4.)):

    old = glob.glob('results/cfs/auto/hzrg*')
    old += glob.glob('results/fits/auto/hzrg*')
    old += glob.glob('results/dndz/hzrg*')
    for name in old:
        os.remove(name)

    lotss = sample.hzrg_sample()
    rand = sample.lotss_randoms()
    dndz_lotss, zs = sample.hzrg_dndz(lotss, nzbins=nzbins, zrange=zrange)
    np.array(dndz_lotss).dump('results/dndz/hzrg.pickle')
    eff_z = np.median(zs)
    thetabins = cosmo.rp2angle(rpscales, eff_z, True)

    cf = twoPointCFs.autocorr_cat(thetabins, lotss, rand, nbootstrap=500)

    cf, lincf = linear_cf(cf, eff_z)

    cf.pop('plot')
    write_pickle('results/cfs/auto/hzrg', cf)

    fit = clustering_fit.fit_pipeline(dndz_lotss, lincf)
    fit['eff_z'] = eff_z
    fit.pop('plot')
    write_pickle('results/fits/auto/hzrg', fit)
    results.autocorrs()
    results.halomass()


def autocorr_izrgs(rpscales):
    old = glob.glob('results/cfs/auto/izrg*')
    old += glob.glob('results/fits/auto/izrg*')
    old += glob.glob('results/dndz/izrg*')
    for name in old:
        os.remove(name)


    lotss = sample.izrg_sample()
    rand = sample.lotss_randoms()
    zs = sample.treat_dndz_pdf(lotss)
    dndz_lotss = redshift_helper.dndz_from_z_list(zs, 7, zrange=(0.3, 1.2))
    np.array(dndz_lotss).dump('results/dndz/izrg.pickle')
    eff_z = np.median(zs)
    thetabins = cosmo.rp2angle(rpscales, eff_z, True)

    cf = twoPointCFs.autocorr_cat(thetabins, lotss, rand, nbootstrap=500)
    cf, lincf = linear_cf(cf, eff_z)

    cf.pop('plot')
    write_pickle('results/cfs/auto/izrg', cf)

    fit = clustering_fit.fit_pipeline(dndz_lotss, lincf)
    fit['eff_z'] = eff_z
    fit.pop('plot')
    write_pickle('results/fits/auto/izrg', fit)
    results.autocorrs()
    results.halomass()

def autocorr_lzrgs(rpscales, nzbins=15):
    old = glob.glob('results/cfs/auto/lzrg*')
    old += glob.glob('results/fits/auto/lzrg*')
    old += glob.glob('results/dndz/lzrg*')
    for name in old:
        os.remove(name)


    lotss = sample.lzrg_sample()
    rand = sample.lotss_randoms()
    zs = sample.treat_dndz_pdf(lotss)
    dndz_lotss = redshift_helper.dndz_from_z_list(zs, nzbins, zrange=(0.1, 2.))
    np.array(dndz_lotss).dump('results/dndz/lzrg.pickle')
    eff_z = np.median(zs)
    thetabins = cosmo.rp2angle(rpscales, eff_z, True)

    cf = twoPointCFs.autocorr_cat(thetabins, lotss, rand, nbootstrap=500)
    cf, lincf = linear_cf(cf, eff_z)

    cf.pop('plot')
    write_pickle('results/cfs/auto/lzrg', cf)

    fit = clustering_fit.fit_pipeline(dndz_lotss, lincf)
    fit['eff_z'] = eff_z
    fit.pop('plot')
    write_pickle('results/fits/auto/lzrg', fit)
    results.auto_lzrg()
    results.autocorrs()
    results.halomass()

def hodfit(whichsample, nwalkers=10, niter=1000, freeparam_ids=['M', 'M1', 'alpha'], inital_params=[12.7, 13.1, .8]):

    old = glob.glob('results/hod/%s*' % whichsample)
    for name in old:
        os.remove(name)

    cf = read_pickle('results/cfs/auto/%s.pickle' % whichsample)
    dndz = read_pickle('results/dndz/%s.pickle' % whichsample)

    fitmc = clustering_fit.fitmcmc(nwalkers=nwalkers, niter=niter, dndz=dndz, cf=cf,
                                   freeparam_ids=freeparam_ids, initial_params=inital_params)
    fitmc.pop('corner')
    write_pickle('results/hod/%s' % whichsample, fitmc)
    results.corners()
    results.hods()
    results.autocorrs()





def desi_elg_xcorr(rpscales):
    oldresults = glob.glob('results/cfs/tomo/rg_elg*.pickle')
    oldresults += glob.glob('results/fits/tomo/rg_elg*.pickle')

    for result in oldresults:
        os.remove(result)
    zbins = np.array([1., 1.5])
    zcenters = np.array([1.25])
    lotss = sample.hzrg_sample()
    lotss = sample.cat_in_desi_elg(lotss)
    lotssrand = sample.lotss_randoms()
    lotssrand = sample.cat_in_desi_elg(lotssrand)
    elgmass, elgmasserr = [], []

    zs = sample.treat_dndz_pdf(lotss)

    dndz_lotss = redshift_helper.dndz_from_z_list(zs, 20, zrange=(0.1, 4.))
    for j in range(len(zcenters)):
        elg, rand = sample.desi_elg(zbins[j], zbins[j+1])
        elg = sample.cat_in_lotss(elg)
        rand = sample.cat_in_lotss(rand)
        elg = sample.cat_in_goodwise(elg)
        rand = sample.cat_in_goodwise(rand)
        theta_scales = cosmo.rp2angle(rps=rpscales, z=np.median(zcenters[j]), h_unit=True)
        autocf = twoPointCFs.autocorr_cat(rpscales, elg, rand)
        autocf['wp_err'] = autocf['wp_poisson_err']
        autofit = clustering_fit.fit_pipeline(redshift_helper.dndz_from_z_list(rand['Z'], 10), autocf)
        autofit.pop('plot')
        write_pickle('results/fits/elgfit', autofit)

        elg.remove_column('CHI')
        rand.remove_column('CHI')

        xcf = twoPointCFs.crosscorr_cats(theta_scales, elg, lotss, rand, lotssrand)
        xcf.pop('plot')
        xcf['w_err'] = xcf['w_err_poisson']
        write_pickle('results/cfs/tomo/rg_elgcf', xcf)

        dndz_elg_matched = redshift_helper.dndz_from_z_list(rand['Z'], 20, zrange=(0.1, 4.))
        xfit = clustering_fit.fit_xcf(dndz_lotss, xcf, dndz_elg_matched, autocf, model='mass')
        xfit.update(clustering_fit.fit_xcf(dndz_lotss, xcf, dndz_elg_matched, autocf, model='minmass'))
        xfit['eff_z'] = redshift_helper.effective_z(dndz_lotss, dndz_elg_matched)
        write_pickle('results/fits/tomo/rg_elg', xfit)
    results.cross_cfs()
    results.halomass()





def hzrg_xcorr(rpscales, pimax=30, nzbins=30, nboots=0, zrange=(0.1, 4.)):
    oldresults = glob.glob('results/cfs/tomo/rg_qso*.pickle')
    oldresults += glob.glob('results/fits/tomo/rg_qso*.pickle')
    oldresults += glob.glob('results/dndz/hzrg.pickle')
    for result in oldresults:
        os.remove(result)
    cflist = []

    eboss_zbins = [1., 1.5, 2., 3.]

    qso, rand = sample.eboss_qso()

    lotss = sample.hzrg_sample()
    lotss = sample.cat_in_eboss(lotss)

    qso = sample.cat_in_goodclustering_area(qso)
    rand = sample.cat_in_goodclustering_area(rand)


    dndz_lotss, zs = sample.hzrg_dndz(lotss, nzbins=nzbins, zrange=zrange)
    np.array(dndz_lotss).dump('results/dndz/hzrg.pickle')

    for j in range(len(eboss_zbins)-1):
        minz, maxz = eboss_zbins[j], eboss_zbins[j+1]
        qsoz = qso[np.where((qso['Z'] > minz) & (qso['Z'] < maxz))]
        randz = rand[np.where((rand['Z'] > minz) & (rand['Z'] < maxz))]

        # convert rp scales to angular scales at median redshift of bin
        theta_scales = cosmo.rp2angle(rps=rpscales, z=np.median(randz['Z']), h_unit=True)


        autofit, autocf = pipeline.measure_and_fit_autocf(scales=rpscales, datcat=qsoz, randcat=randz, nbootstrap=500, nzbins=10, pimax=pimax)
        autofit.pop('plot')
        write_pickle('results/fits/tomo/qso_fit_%s' % j, autofit)

        dndz_qso_matched = redshift_helper.dndz_from_z_list(randz['Z'], nzbins, zrange=zrange)
        #dndz_lotss = redshift_helper.spl_interp_dndz(dndz_lotss, newzs=dndz_qso_matched[0], spline_k=5, smooth=0.03)
        dndz_lotss = redshift_helper.fill_in_coarse_dndz(dndz_lotss, dndz_qso_matched[0])



        qsoz.remove_column('CHI'), randz.remove_column('CHI')


        xcf = twoPointCFs.crosscorr_cats(scales=theta_scales, datcat1=qsoz, datcat2=lotss,
                                         randcat1=randz, nbootstrap=500, estimator='Peebles')
        xcf.pop('plot')
        write_pickle('results/cfs/tomo/rg_qso_%s' % j, xcf)

        xfit = clustering_fit.fit_xcf(dndz_lotss, xcf, dndz_qso_matched, autocf, model='mass')
        xfit.update(clustering_fit.fit_xcf(dndz_lotss, xcf, dndz_qso_matched, autocf, model='minmass'))
        xfit['eff_z'] = redshift_helper.effective_z(dndz_lotss, dndz_qso_matched)

        if nboots > 0:
            ms, m_mins = [], []
            for k in range(nboots):
                boot_dndz = draw_hzrg_dndz(zs, nbins_mean=nzbins, zrange=zrange)
                xfit_boot = clustering_fit.fit_xcf(boot_dndz, xcf, dndz_qso_matched, autocf, model='mass')
                xfit_boot.update(clustering_fit.fit_xcf(boot_dndz, xcf, dndz_qso_matched, autocf, model='minmass'))
                ms.append(xfit_boot['Mx'])
                m_mins.append(xfit_boot['Mxmin'])
            # add error in quadrature
            xfit['sigMx'] = np.sqrt(np.square(xfit['sigMx']) + np.square(np.std(ms)))
            xfit['sigMxmin'] = np.sqrt(np.square(xfit['sigMxmin']) + np.square(np.std(m_mins)))


        cflist.append(xcf)
        write_pickle('results/fits/tomo/rg_qso_fit_%s' % j, xfit)

    results.cross_cfs()
    results.halomass()






def lzrg_xcorr(rpscales, nzbins=15):
    oldresults = glob.glob('results/fits/tomo/lowz_fit.pickle')
    oldresults += glob.glob('results/cfs/tomo/rg_lowz*.pickle')
    oldresults += glob.glob('results/fits/tomo/rg_lowz*.pickle')
    for result in oldresults:
        os.remove(result)

    minz, maxz = 0.25, 0.5
    effz = minz + (maxz - minz) / 2.

    lotss = sample.lzrg_sample()
    lotss = sample.cat_in_boss(lotss)

    boss, rand = sample.boss_gals(minz=minz, maxz=maxz)

    autofit, autocf = pipeline.measure_and_fit_autocf(scales=rpscales, datcat=boss, randcat=rand, nbootstrap=500,
                                                      nzbins=10)
    autofit.pop('plot')
    write_pickle('results/fits/tomo/lowz_fit', autofit)

    boss = sample.cat_in_goodclustering_area(boss)
    rand = sample.cat_in_goodclustering_area(rand)


    # convert rp scales to angular scales at median redshift of bin
    theta_scales = cosmo.rp2angle(rps=rpscales, z=effz, h_unit=True)

    xcf = twoPointCFs.crosscorr_cats(scales=theta_scales, datcat1=boss, datcat2=lotss,
                                     randcat1=rand, nbootstrap=0, estimator='Peebles')
    xcf['w_err'] = xcf['w_err_poisson']
    xcf.pop('plot')
    write_pickle('results/cfs/tomo/rg_lowz', xcf)

    zs_lzrg = sample.redshift_dist(lotss, 3., bootesonly=False)
    dndz_lzrg = redshift_helper.dndz_from_z_list(zs_lzrg, nbins=nzbins, zrange=(0.1, 2))
    np.array(dndz_lzrg).dump('results/dndz/lzrg.pickle')

    dndz_boss = redshift_helper.dndz_from_z_list(rand['Z'], nbins=nzbins, zrange=(0.1, 2))

    xfit = clustering_fit.fit_xcf(dndz_lzrg, xcf, dndz_boss, autocf, model='mass')
    xfit.update(clustering_fit.fit_xcf(dndz_lzrg, xcf, dndz_boss, autocf, model='minmass'))
    xfit.update(clustering_fit.fit_xcf(dndz_lzrg, xcf, dndz_boss, autocf, model='bias'))
    xfit['eff_z'] = redshift_helper.effective_z(dndz_lzrg, dndz_boss)


    write_pickle('results/fits/tomo/rg_lowz_fit', xfit)
    results.cross_lzrg()
    results.halomass(lzrg=True)









def lens_hzrg(rpscales, nside=1024, nzbins=30, zrange=(0.1, 4.)):
    from lensing_Helper import measure_lensing_xcorr



    oldresults = glob.glob('results/lenscorr/hzrg*')
    oldresults += glob.glob('results/lensfits/hzrg*')
    for result in oldresults:
        os.remove(result)

    lotss = sample.hzrg_sample()

    mask = sample.rg_mask(nside)
    #mask = np.ones(hp.nside2npix(nside))
    dndz, zs = sample.hzrg_dndz(lotss, nzbins=nzbins, zrange=zrange)
    eff_z = redshift_helper.effective_lensing_z(dndz)
    ells = np.array(cosmo.rp2ell(rpscales, eff_z), dtype=int)
    ells = np.logspace(np.log10(200), np.log10(2000), 11)

    corr = measure_lensing_xcorr.measure_planck_xcorr(ells, (lotss['RA'], lotss['DEC']), nside, mask=mask, accurate=True)
    write_pickle('results/lenscorr/hzrg', corr)


    fit = lensing_fit.xcorr_fit_pipeline(dndz, corr)
    fit['eff_z'] = eff_z
    fig = fit.pop('plot')
    write_pickle('results/lensfits/hzrg', fit)

    fig.savefig(plotdir + 'hiz_rg_lensing.pdf')
    results.halomass()
    results.lenscorrs()


def lens_izrg(rpscales, nside=1024):
    from mocpy import MOC
    import healpy as hp
    import astropy.units as u
    from lensing_Helper import measure_lensing_xcorr

    oldresults = glob.glob('results/lenscorr/izrg*')
    oldresults += glob.glob('results/lensfits/izrg*')
    for result in oldresults:
        os.remove(result)


    lotss = sample.izrg_sample()


    mask = sample.rg_mask(nside)
    #mask = np.ones(hp.nside2npix(nside))
    zs = sample.treat_dndz_pdf(lotss)
    eff_z = np.median(zs)
    ells = np.array(cosmo.rp2ell(rpscales, eff_z), dtype=int)
    dndz = redshift_helper.dndz_from_z_list(zs, 7, zrange=(0.3, 1.2))

    corr = measure_lensing_xcorr.measure_planck_xcorr(ells, (lotss['RA'], lotss['DEC']), nside, mask=mask, accurate=True)
    write_pickle('results/lenscorr/izrg', corr)


    fit = lensing_fit.xcorr_fit_pipeline(dndz, corr)
    fig = fit.pop('plot')
    write_pickle('results/lensfits/izrg', fit)

    fig.savefig(plotdir + 'izrg_lensing.pdf')


def lumtrend(rpscales, nzbins=15, minz=0.5, maxz=0.8):
    lbins = [24.75, 25., 25.5, 28]


    lotss = sample.izrg_sample(lummin=24.5, minz=minz, maxz=maxz)

    rand = sample.lotss_randoms()

    duty, loerr, hierr, medlums = [], [], [], []
    for j in range(len(lbins)-1):
        lumlotss = lotss[np.where((lotss['L150'] > lbins[j]) & (lotss['L150'] <= lbins[j+1]))]
        medlums.append(np.median(lumlotss['L150']))

        zs = sample.treat_dndz_pdf(lumlotss)


        dndz_lotss = redshift_helper.dndz_from_z_list(zs, nzbins, zrange=(0.1, 2.))

        eff_z = np.median(zs)
        thetabins = cosmo.rp2angle(rpscales, eff_z, True)

        cf = twoPointCFs.autocorr_cat(thetabins, lumlotss, rand, nbootstrap=500)
        cf, lincf = linear_cf(cf, eff_z)
        fit = clustering_fit.fit_pipeline(dndz_lotss, lincf)

        duty.append(lumfunc.occupation_lumbound_zrange(fit['Mmin'], zrange=(minz, maxz), lmin=lbins[j], lmax=lbins[j+1]))
        hierr.append(lumfunc.occupation_lumbound_zrange(fit['Mmin']+fit['sigMmin'], zrange=(minz, maxz), lmin=lbins[j],
                                                    lmax=lbins[j+1]))
        loerr.append(lumfunc.occupation_lumbound_zrange(fit['Mmin']-fit['sigMmin'], zrange=(minz, maxz),
                                                             lmin=lbins[j], lmax=lbins[j+1]))
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8,7))
    plt.errorbar(medlums, duty, yerr=[loerr, hierr], fmt='o', color='k')
    plt.yscale('log')
    plt.ylim(1e-2, 2)
    plt.savefig('/home/graysonpetter/Dropbox/radioplots/lum_duty.pdf')
    plt.close('all')




def duty_cycle():
    fluxcuthi = 2.
    fluxcutmid = 5.
    fluxcutlo = 20.
    fluxmax = 1000.



    lzrgfit = read_pickle('results/fits/tomo/rg_lowz_fit.pickle')

    dutylzrg = lumfunc.occupation_fluxlim_zrange(lzrgfit['Mxmin'], (0.25, 0.5), fluxcut=fluxcutlo, fluxmax=fluxmax, lftype='lerg')
    lzrgloerr = dutylzrg - lumfunc.occupation_fluxlim_zrange(lzrgfit['Mxmin'] - lzrgfit['sigMxmin'], (0.25, 0.5),
                                                     fluxcut=fluxcutlo, fluxmax=fluxmax, lftype='lerg')
    lzrghierr = lumfunc.occupation_fluxlim_zrange(lzrgfit['Mxmin'] + lzrgfit['sigMxmin'], (0.25, 0.5),
                                          fluxcut=fluxcutlo, fluxmax=fluxmax, lftype='lerg') - dutylzrg

    lzrgstuff = [lzrgfit['eff_z'], dutylzrg, lzrgloerr, lzrghierr]


    izrgfit = read_pickle('results/fits/auto/izrg.pickle')

    dutyizrg = lumfunc.occupation_fluxlim_zrange(izrgfit['Mmin'], (0.5, 1.), fluxcut=fluxcutmid, fluxmax=fluxmax, lftype='lerg')
    izrgloerr = dutyizrg - lumfunc.occupation_fluxlim_zrange(izrgfit['Mmin']-izrgfit['sigMmin'], (0.5, 1.),
                                                     fluxcut=fluxcutmid, fluxmax=fluxmax, lftype='lerg')
    izrghierr = lumfunc.occupation_fluxlim_zrange(izrgfit['Mmin'] + izrgfit['sigMmin'], (0.5, 1.),
                                                     fluxcut=fluxcutmid, fluxmax=fluxmax, lftype='lerg') - dutyizrg

    izrgstuff = [izrgfit['eff_z'], dutyizrg, izrgloerr, izrghierr]


    hzrgfit = read_pickle('results/fits/auto/hzrg.pickle')

    dutyhzrg = lumfunc.occupation_fluxlim_zrange(hzrgfit['Mmin'], (1., 2.), fluxcut=fluxcuthi, fluxmax=fluxmax, lftype='agn')
    hzrgloerr = dutyhzrg - lumfunc.occupation_fluxlim_zrange(hzrgfit['Mmin'] - hzrgfit['sigMmin'], (1., 3.),
                                                     fluxcut=fluxcuthi, fluxmax=fluxmax, lftype='agn')
    hzrghierr = lumfunc.occupation_fluxlim_zrange(hzrgfit['Mmin'] + hzrgfit['sigMmin'], (1., 3.),
                                          fluxcut=fluxcuthi, fluxmax=fluxmax, lftype='agn') - dutyhzrg

    hzrgstuff = [hzrgfit['eff_z'], dutyhzrg, hzrgloerr, hzrghierr]



    lensfit = read_pickle('results/lensfits/hzrg.pickle')

    dutylens = lumfunc.occupation_fluxlim_zrange(lensfit['Mmin'], (1., 2.), fluxcut=fluxcuthi, fluxmax=fluxmax, lftype='agn')
    lensloerr = dutylens - lumfunc.occupation_fluxlim_zrange(lensfit['Mmin'] - lensfit['sigMmin'], (1., 3.),
                                                     fluxcut=fluxcuthi, fluxmax=fluxmax, lftype='agn')
    lenshierr = lumfunc.occupation_fluxlim_zrange(lensfit['Mmin'] + lensfit['sigMmin'], (1., 3.),
                                          fluxcut=fluxcuthi, fluxmax=fluxmax, lftype='agn') - dutylens

    lensstuff = [lensfit['eff_z'], dutylens, lensloerr, lenshierr]



    z_centers = []

    qsofitnames = glob.glob('results/fits/tomo/rg_qso*.pickle')
    fits = []
    qsozrange = [1., 1.5, 2., 3]
    for name in qsofitnames:
        thisfit = read_pickle(name)
        fits.append(thisfit)
        z_centers.append(thisfit['eff_z'])

    duty, loerr, hierr = [], [], []
    for j in range(len(fits)):
        fduty = lumfunc.occupation_fluxlim_zrange(logminmass=fits[j]['Mxmin'], zrange=(qsozrange[j], qsozrange[j+1]),
                                          fluxcut=fluxcuthi, fluxmax=1000.)
        duty.append(fduty)
        loerr.append(fduty-lumfunc.occupation_fluxlim_zrange(zrange=(qsozrange[j], qsozrange[j+1]), logminmass=fits[j]['Mxmin'] - fits[j]['sigMxmin'], fluxcut=fluxcuthi, fluxmax=fluxmax))
        hierr.append(lumfunc.occupation_fluxlim_zrange(zrange=(qsozrange[j], qsozrange[j + 1]), logminmass=fits[j]['Mxmin'] + fits[j]['sigMxmin'],
                                                 fluxcut=fluxcuthi, fluxmax=1000.)-fduty)

    results.dutycycle(lzrgstuff, izrgstuff, hzrgstuff, lensstuff, z_centers, duty, loerr, hierr)

def haloenergy():
    z_centers = []
    import glob
    fluxcuthi = 2.
    fluxcutmid = 5.
    fluxmax = 1000.

    izrgfit = read_pickle('results/fits/auto/izrg.pickle')

    e_izrg = lumfunc.energy_per_halo_zrange((0.5, 1.), izrgfit['Mmin'],
                                            fluxcut=fluxcutmid, fluxmax=fluxmax, lftype='lerg')
    izrgloerr = e_izrg - lumfunc.energy_per_halo_zrange((0.5, 1.), izrgfit['Mmin'] - izrgfit['sigMmin'],
                                                     fluxcut=fluxcutmid, fluxmax=fluxmax, lftype='lerg')
    izrghierr = lumfunc.energy_per_halo_zrange((0.5, 1.), izrgfit['Mmin'] + izrgfit['sigMmin'],
                                          fluxcut=fluxcutmid, fluxmax=fluxmax, lftype='lerg') - e_izrg

    izrgstuff = [izrgfit['eff_z'], e_izrg, izrgloerr, izrghierr]


    qsofitnames = glob.glob('results/fits/tomo/rg_qso*.pickle')
    fits = []
    qsozrange = [1., 1.5, 2., 2.5]
    for name in qsofitnames:
        thisfit = read_pickle(name)
        fits.append(thisfit)
        z_centers.append(thisfit['eff_z'])

    logminmass = 12.8
    kineticfraclow, kineticfrachi = 0.001, 0.005
    allzranges = [0.0, 0.5, 1., 1.5, 2., 3.]
    qsowindinfo = []
    for j in range(len(allzranges)-1):
        qsowindinfo.append([])
        qsowindlow = lumfunc.type1_windheatperhalo((allzranges[j], allzranges[j+1]),  logminmass=logminmass,
                                                   kinetic_frac=kineticfraclow)
        qsowindhigh = lumfunc.type1_windheatperhalo((allzranges[j], allzranges[j + 1]), logminmass=logminmass,
                                                   kinetic_frac=kineticfrachi)
        qsowindinfo[j].append([allzranges[j], allzranges[j+1]])
        qsowindinfo[j].append([qsowindlow, qsowindhigh])

    es, elo, ehi = [], [], []
    for j in range(len(fits)):
        e = lumfunc.energy_per_halo_zrange(zrange=(qsozrange[j], qsozrange[j+1]), logminmass=fits[j]['Mxmin'], fluxcut=2., fluxmax=1000.)
        es.append(e)
        elo.append(e-lumfunc.energy_per_halo_zrange(zrange=(qsozrange[j], qsozrange[j+1]), logminmass=fits[j]['Mxmin'] - fits[j]['sigMxmin'], fluxcut=2., fluxmax=1000.))
        ehi.append(lumfunc.energy_per_halo_zrange(zrange=(qsozrange[j], qsozrange[j + 1]), logminmass=fits[j]['Mxmin'] + fits[j]['sigMxmin'],
                                                 fluxcut=2., fluxmax=1000.)-e)

    results.energetics(izrgstuff, z_centers, es, elo, ehi, qsowindinfo)

def halopower(mcut=13.):
    z_centers = []
    import glob
    fluxcuthi = 2.
    fluxcutmid = 5.
    fluxcutlo = 20.
    fluxmax = 1000.

    izrgfit = read_pickle('results/fits/auto/izrg.pickle')

    e_izrg = lumfunc.energy_per_halo_zrange((0.5, 1.), logminmass_hod=izrgfit['Mmin'], logminmass_cut=mcut,
                                            fluxcut=fluxcutmid, fluxmax=fluxmax, lftype='lerg', avg_power=True)
    izrgloerr = e_izrg - lumfunc.energy_per_halo_zrange((0.5, 1.), logminmass_hod=(izrgfit['Mmin'] - izrgfit['sigMmin']),
                                                        logminmass_cut=mcut,
                                                     fluxcut=fluxcutmid, fluxmax=fluxmax, lftype='lerg', avg_power=True)
    izrghierr = lumfunc.energy_per_halo_zrange((0.5, 1.), logminmass_hod=(izrgfit['Mmin'] + izrgfit['sigMmin']),
                                               logminmass_cut=mcut,
                                          fluxcut=fluxcutmid, fluxmax=fluxmax, lftype='lerg', avg_power=True) - e_izrg
    izrgs = [izrgfit['eff_z'], e_izrg, izrgloerr, izrghierr]


    lzrgfit = read_pickle('results/fits/tomo/rg_lowz_fit.pickle')
    e_lzrg = lumfunc.energy_per_halo_zrange((0.25, 0.5), logminmass_hod=lzrgfit['Mxmin'], logminmass_cut=mcut,
                                            fluxcut=fluxcutlo, fluxmax=fluxmax, lftype='lerg', avg_power=True)
    lzrgloerr = e_lzrg - lumfunc.energy_per_halo_zrange((0.25, 0.5),
                                                        logminmass_hod=(lzrgfit['Mxmin'] - lzrgfit['sigMxmin']),
                                                        logminmass_cut=mcut,
                                                        fluxcut=fluxcutlo, fluxmax=fluxmax, lftype='lerg',
                                                        avg_power=True)
    lzrghierr = lumfunc.energy_per_halo_zrange((0.25, 0.5),
                                               logminmass_hod=(lzrgfit['Mxmin'] + lzrgfit['sigMxmin']),
                                               logminmass_cut=mcut,
                                               fluxcut=fluxcutlo, fluxmax=fluxmax, lftype='lerg',
                                               avg_power=True) - e_lzrg

    lzrgs = [lzrgfit['eff_z'], e_lzrg, lzrgloerr, lzrghierr]


    qsofitnames = sorted(glob.glob('results/fits/tomo/rg_qso*.pickle'))
    fits = []
    qsozrange = [1., 1.5, 2., 2.5]
    for name in qsofitnames:
        thisfit = read_pickle(name)
        fits.append(thisfit)
        z_centers.append(thisfit['eff_z'])


    kineticfraclow, kineticfrachi = 0.001, 0.005
    allzranges = np.linspace(0.,  3.5, 10)
    qsozs = (allzranges[:-1] + allzranges[1:]) / 2
    windlow, windhigh = [], []

    for j in range(len(allzranges)-1):

        windlow.append(lumfunc.type1_windpowerperhalo((allzranges[j], allzranges[j+1]),  logminmass=mcut,
                                                   kinetic_frac=kineticfraclow))
        windhigh.append(lumfunc.type1_windpowerperhalo((allzranges[j], allzranges[j + 1]), logminmass=mcut,
                                                   kinetic_frac=kineticfrachi))
    qsowindinfo = [qsozs, windlow, windhigh]


    es, elo, ehi = [], [], []
    for j in range(len(fits)):
        e = lumfunc.energy_per_halo_zrange(zrange=(qsozrange[j], qsozrange[j+1]),
                                           logminmass_hod=fits[j]['Mxmin'],
                                           logminmass_cut=mcut, fluxcut=2.,
                                           fluxmax=1000., avg_power=True)
        es.append(e)
        elo.append(e-lumfunc.energy_per_halo_zrange(zrange=(qsozrange[j], qsozrange[j+1]),
                                                    logminmass_hod=(fits[j]['Mxmin'] - fits[j]['sigMxmin']),
                                                    logminmass_cut=mcut, fluxcut=2., fluxmax=1000., avg_power=True))
        ehi.append(lumfunc.energy_per_halo_zrange(zrange=(qsozrange[j], qsozrange[j + 1]),
                                                  logminmass_hod=(fits[j]['Mxmin'] + fits[j]['sigMxmin']),
                                                  logminmass_cut=mcut,
                                                  fluxcut=2., fluxmax=1000., avg_power=True)-e)

    hzrgs = [z_centers, es, elo, ehi]

    results.avghalopower(lzrgs, izrgs, hzrgs, qsowindinfo)

def halopower_ratio(sigM=0.5, kineticfraclow=0.001, kineticfrachi=0.005):
    from halomodelpy import bias_tools
    z_centers = []
    import glob
    fluxcuthi = 2.
    fluxcutmid = 5.
    fluxcutlo = 20.
    fluxmax = 1000.

    mass_cut_grid = np.linspace(12., 13.5, 30)

    izrgfit = read_pickle('results/fits/auto/izrg.pickle')

    izrg_ratios = []
    for mcut in mass_cut_grid:
        dndz_izrg = read_pickle('results/dndz/izrg.pickle')
        m_char, mlo, mhi = bias_tools.avg_bias2mass_transition(dndz=dndz_izrg, b=izrgfit['b'], sigma=sigM, berr=izrgfit['sigb'])
        e_izrg = lumfunc.energy_per_halo_zrange((0.5, 1.), logminmass_hod=m_char, logminmass_cut=mcut,
                                        fluxcut=fluxcutmid, fluxmax=fluxmax, lftype='lerg', avg_power=True, sigM=sigM)

        izrgloerr = e_izrg - lumfunc.energy_per_halo_zrange((0.5, 1.),
                                                            logminmass_hod=(m_char - mlo),
                                                            logminmass_cut=mcut,
                                                            fluxcut=fluxcutmid, fluxmax=fluxmax, lftype='lerg',
                                                            avg_power=True, sigM=sigM)
        izrghierr = lumfunc.energy_per_halo_zrange((0.5, 1.), logminmass_hod=(m_char + mhi),
                                                   logminmass_cut=mcut,
                                                   fluxcut=fluxcutmid, fluxmax=fluxmax, lftype='lerg',
                                                   avg_power=True, sigM=sigM) - e_izrg
        qpowermin = lumfunc.type1_windpowerperhalo(zrange=(0.5, 1.), logminmass=mcut, kinetic_frac=kineticfraclow)
        qpowermax = lumfunc.type1_windpowerperhalo(zrange=(0.5, 1.), logminmass=mcut, kinetic_frac=kineticfrachi)

        # highest jet to wind ratio is combination of max jet power and min wind power
        max_ratio = 10 ** ((e_izrg + izrghierr) - qpowermin)
        min_ratio = 10 ** ((e_izrg - izrgloerr) - qpowermax)
        izrg_ratios.append([min_ratio, max_ratio])

    izrg_ratios = np.array(izrg_ratios)

    lzrg_ratios = []
    lzrgfit = read_pickle('results/fits/tomo/rg_lowz_fit.pickle')
    for mcut in mass_cut_grid:
        dndz_lzrg = read_pickle('results/dndz/izrg.pickle')
        m_char, mlo, mhi = bias_tools.avg_bias2mass_transition(dndz=dndz_lzrg, b=lzrgfit['bx'], sigma=sigM, berr=lzrgfit['sigbx'])
        e_lzrg = lumfunc.energy_per_halo_zrange((0.25, 0.5), logminmass_hod=m_char, logminmass_cut=mcut,
                                        fluxcut=fluxcutmid, fluxmax=fluxmax, lftype='lerg', avg_power=True, sigM=sigM)

        lzrgloerr = e_lzrg - lumfunc.energy_per_halo_zrange((0.25, 0.5),
                                                            logminmass_hod=(m_char - mlo),
                                                            logminmass_cut=mcut,
                                                            fluxcut=fluxcutlo, fluxmax=fluxmax, lftype='lerg',
                                                            avg_power=True, sigM=sigM)
        lzrghierr = lumfunc.energy_per_halo_zrange((0.25, 0.5), logminmass_hod=(m_char + mhi),
                                                   logminmass_cut=mcut,
                                                   fluxcut=fluxcutlo, fluxmax=fluxmax, lftype='lerg',
                                                   avg_power=True, sigM=sigM) - e_lzrg
        qpowermin = lumfunc.type1_windpowerperhalo(zrange=(0.25, 0.5), logminmass=mcut, kinetic_frac=kineticfraclow)
        qpowermax = lumfunc.type1_windpowerperhalo(zrange=(0.25, 0.5), logminmass=mcut, kinetic_frac=kineticfrachi)

        # highest jet to wind ratio is combination of max jet power and min wind power
        max_ratio = 10 ** ((e_lzrg + lzrghierr) - qpowermin)
        min_ratio = 10 ** ((e_lzrg - lzrgloerr) - qpowermax)
        lzrg_ratios.append([min_ratio, max_ratio])

    lzrg_ratios = np.array(lzrg_ratios)
    results.halopower_ratios(mcut_grid=mass_cut_grid, lzrgs=lzrg_ratios, izrgs=izrg_ratios)


    """qsofitnames = sorted(glob.glob('results/fits/tomo/rg_qso*.pickle'))
    fits = []
    qsozrange = [1., 1.5, 2., 2.5]
    for name in qsofitnames:
        thisfit = read_pickle(name)
        fits.append(thisfit)
        z_centers.append(thisfit['eff_z'])


    allzranges = np.linspace(0., 3.5, 10)
    qsozs = (allzranges[:-1] + allzranges[1:]) / 2
    windlow, windhigh = [], []

    for j in range(len(allzranges) - 1):
        windlow.append(lumfunc.type1_windpowerperhalo((allzranges[j], allzranges[j + 1]), logminmass=mcut,
                                                      kinetic_frac=kineticfraclow))
        windhigh.append(lumfunc.type1_windpowerperhalo((allzranges[j], allzranges[j + 1]), logminmass=mcut,
                                                       kinetic_frac=kineticfrachi))
    qsowindinfo = [qsozs, windlow, windhigh]

    es, elo, ehi = [], [], []
    for j in range(len(fits)):
        e = lumfunc.energy_per_halo_zrange(zrange=(qsozrange[j], qsozrange[j + 1]),
                                           logminmass_hod=fits[j]['Mxmin'],
                                           logminmass_cut=mcut, fluxcut=2.,
                                           fluxmax=1000., avg_power=True)
        es.append(e)
        elo.append(e - lumfunc.energy_per_halo_zrange(zrange=(qsozrange[j], qsozrange[j + 1]),
                                                      logminmass_hod=(fits[j]['Mxmin'] - fits[j]['sigMxmin']),
                                                      logminmass_cut=mcut, fluxcut=2., fluxmax=1000.,
                                                      avg_power=True))
        ehi.append(lumfunc.energy_per_halo_zrange(zrange=(qsozrange[j], qsozrange[j + 1]),
                                                  logminmass_hod=(fits[j]['Mxmin'] + fits[j]['sigMxmin']),
                                                  logminmass_cut=mcut,
                                                  fluxcut=2., fluxmax=1000., avg_power=True) - e)

    hzrgs = [z_centers, es, elo, ehi]

    results.avghalopower(lzrgs, izrgs, hzrgs, qsowindinfo)"""



def lrg_radioloud(scales, fcut=5.):
    lrg = Table.read('catalogs/masked/eBOSS_LRG.fits')
    lrgrand = Table.read('catalogs/masked/eBOSS_LRG_randoms.fits')
    lrgfit, lrgcf = pipeline.measure_and_fit_autocf(scales, lrg, lrgrand)
    lrgm, lrgmerr = lrgfit['M'], lrgfit['sigM']

    loudness, ms, merrs = [], [], []

    radiolrg, foo, newlrg, newlrgrand = sample.select_radioloud('LOTSS_DR2', f_cuts[j], lrg, lrgrand)
    #loudness.append(np.median(radiolrg['L_144'] - radiolrg['L2']))
    radiofit, radiocf = pipeline.measure_and_fit_xcf(lrgcf, scales, newlrg, radiolrg, newlrgrand)
    ms.append(radiofit['Mx']), merrs.append(radiofit['sigMx'])

    mass_v_prop.plot_halomass_v_lrg_radioloud(lrgm, lrgmerr, loudness, ms, merrs, 'LRG')