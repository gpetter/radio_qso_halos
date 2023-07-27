from corrfunc_helper import twoPointCFs
from halomodelpy import clustering_fit, redshift_helper, pipeline, hm_calcs, lensing_fit, cosmo
import numpy as np

import lumfunc
import sample
from SkyTools import catalog_utils, coordhelper
from astropy.table import Table
from plotscripts import mass_v_prop
import pickle
import glob
import os
from plotscripts import results
plotdir = '/home/graysonpetter/Dropbox/radioplots/'
def eboss_qso_autocf(scales):
    qso, rand = sample.qsocat(eboss=True, boss=False)
    fitpars, autocf = pipeline.measure_and_fit_autocf(scales, qso, rand)
    autocf.pop('plot')
    with open('results/cfs/ebossqso_cf.pickle', 'wb') as f:
        pickle.dump(autocf, f)

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



def autcorrs_by_property(scales, cat, randcat, prop, prop_bins, pimax=40., dpi=1., mubins=None, wedges=None):
    b, berr = [], []
    m, merr = [], []
    prop_meds = []
    for j in range(len(prop_bins) - 1):
        bincat = cat[np.where((cat[prop] > prop_bins[j]) & (cat[prop] <= prop_bins[j+1]))]
        prop_meds.append(np.median(bincat[prop]))
        binrand = randcat[np.where((randcat[prop] > prop_bins[j]) & (randcat[prop] <= prop_bins[j+1]))]
        fitpars, autocf = pipeline.measure_and_fit_autocf(scales, bincat, binrand,
                                                    pimax=pimax, dpi=dpi, mubins=mubins, wedges=wedges)
        b.append(fitpars['b']), berr.append(fitpars['sigb'])
        m.append(fitpars['M']), merr.append(fitpars['sigM'])
    return b, berr, m, merr, prop_meds


def clustering_v_z(scales, cat, randcat, prop_bins, pimax=40., dpi=1., mubins=None, wedges=None):
    results = autcorrs_by_property(scales, cat, randcat, 'Z', prop_bins,
                                   pimax=pimax, dpi=dpi, mubins=mubins, wedges=wedges)
    from plotscripts import halos_v_z
    halos_v_z.haloprop_z(results[4], [results[0], results[2]], [results[1], results[3]])

def z1_properties(scales, npercentiles=10):
    percentiles = np.arange(0, 101, npercentiles)
    totcat, totrand = sample.collate_z_equals_1_tracers()
    qso, qsorand = sample.qsocat(eboss=True)
    qso = catalog_utils.cut_table(qso, 'Z', np.min(totrand['Z']), np.max(totrand['Z']))
    qsorand = catalog_utils.cut_table(qsorand, 'Z', np.min(totrand['Z']), np.max(totrand['Z']))


    #qsofit, qsocf = pipeline.measure_and_fit_autocf(scales, qso, qsorand)

    totfit, totcf = pipeline.measure_and_fit_autocf(scales, totcat, totrand)

    medlum, lumMass, lumMasserr = [], [], []
    medbh, bhMass, bhMasserr = [], [], []
    for j in range(len(percentiles)-1):
        qsolum = sample.rolling_percentile_selection(qso, prop='Lbol',
                                                     minpercentile=percentiles[j],
                                                     maxpercentile=percentiles[j+1], nzbins=10)
        medlum.append(np.median(qsolum['Lbol']))
        lumfit, lumcf = pipeline.measure_and_fit_xcf(totcf, scales, totcat, qsolum, totrand)
        lumMass.append(lumfit['Mx']), lumMasserr.append(lumfit['sigMx'])


        qso = qso[np.where(qso['MBH'] > 6)]
        qsobh = sample.rolling_percentile_selection(qso, prop='MBH',
                                                     minpercentile=percentiles[j],
                                                     maxpercentile=percentiles[j+1], nzbins=10)
        medbh.append(np.median(qsobh['MBH']))
        bhfit, bhcf = pipeline.measure_and_fit_xcf(totcf, scales, totcat, qsobh, totrand)
        bhMass.append(bhfit['Mx']), bhMasserr.append(bhfit['sigMx'])

    mass_v_prop.plot_halomass_v_lum(medlum, lumMass, lumMasserr, 'z1')
    mass_v_prop.plot_halomass_v_mbh(medbh, bhMass, bhMasserr, 'z1')
    return (medlum, lumMass, lumMasserr), (medbh, bhMass, bhMasserr)

def quasarlum_zbins(scales, nzbins, boss=False, highpercentile=90, lowpercentile=10):

    qso, qsorand = sample.qsocat(eboss=True)
    qso = qso[np.where(qso['MBH'] > 6)]
    zbins = np.linspace(np.min(qsorand['Z']), np.max(qsorand['Z']), nzbins+1)
    medzs, ms, merrs, mlum, mlumerr, mbh, mbherr = [], [], [], [], [], [], []
    lowlum, lowlumerr, lowmass, lowmasserr = [], [], [], []

    for j in range(nzbins):
        zqso = qso[np.where((qso['Z'] > zbins[j]) & (qso['Z'] <= zbins[j+1]))]
        zrand = qsorand[np.where((qsorand['Z'] > zbins[j]) & (qsorand['Z'] <= zbins[j+1]))]
        zqsolum = sample.rolling_percentile_selection(zqso, 'Lbol', highpercentile, nzbins=10)
        zqsohimass = sample.rolling_percentile_selection(zqso, 'MBH', highpercentile, nzbins=10)
        autofit, autocf = pipeline.measure_and_fit_autocf(scales, zqso, zrand)
        xlumfit, xlumcf = pipeline.measure_and_fit_xcf(autocf, scales, zqso, zqsolum, zrand)
        mlum.append(xlumfit['Mx']), mlumerr.append(xlumfit['sigMx'])
        himassfit, himasscf = pipeline.measure_and_fit_xcf(autocf, scales, zqso, zqsohimass, zrand)
        mbh.append(himassfit['Mx']), mbherr.append(himassfit['sigMx'])

        zqsololum = sample.rolling_percentile_selection(zqso, 'Lbol', minpercentile=0, maxpercentile=lowpercentile, nzbins=10)
        zqsolomass = sample.rolling_percentile_selection(zqso, 'MBH', minpercentile=0, maxpercentile=lowpercentile, nzbins=10)
        xlumfit, xlumcf = pipeline.measure_and_fit_xcf(autocf, scales, zqso, zqsololum, zrand)
        lowlum.append(xlumfit['Mx']), lowlumerr.append(xlumfit['sigMx'])
        lomassfit, lomasscf = pipeline.measure_and_fit_xcf(autocf, scales, zqso, zqsolomass, zrand)
        lowmass.append(lomassfit['Mx']), lowmasserr.append(lomassfit['sigMx'])

        medzs.append(np.median(zqso['Z'])), ms.append(autofit['M']), merrs.append(autofit['sigM'])

    # now do same for BOSS
    qso, qsorand = Table.read('catalogs/masked/BOSS_QSO.fits'), Table.read('catalogs/masked/BOSS_QSO_randoms.fits')
    qso = qso[np.where(qso['MBH'] > 6)]
    zqsolum = sample.rolling_percentile_selection(qso, 'Lbol', highpercentile, nzbins=10)
    zqsohimass = sample.rolling_percentile_selection(qso, 'MBH', highpercentile, nzbins=10)

    autofit, autocf = pipeline.measure_and_fit_autocf(scales, qso, qsorand)
    xlumfit, xlumcf = pipeline.measure_and_fit_xcf(autocf, scales, qso, zqsolum, qsorand)
    mlum.append(xlumfit['Mx']), mlumerr.append(xlumfit['sigMx'])
    himassfit, himasscf = pipeline.measure_and_fit_xcf(autocf, scales, qso, zqsohimass, qsorand)
    mbh.append(himassfit['Mx']), mbherr.append(himassfit['sigMx'])
    zqsololum = sample.rolling_percentile_selection(qso, 'Lbol', minpercentile=0, maxpercentile=lowpercentile,
                                                    nzbins=10)
    zqsolomass = sample.rolling_percentile_selection(qso, 'MBH', minpercentile=0, maxpercentile=lowpercentile,
                                                     nzbins=10)

    xlumfit, xlumcf = pipeline.measure_and_fit_xcf(autocf, scales, qso, zqsololum, qsorand)
    lowlum.append(xlumfit['Mx']), lowlumerr.append(xlumfit['sigMx'])
    lomassfit, lomasscf = pipeline.measure_and_fit_xcf(autocf, scales, qso, zqsolomass, qsorand)
    lowmass.append(lomassfit['Mx']), lowmasserr.append(lomassfit['sigMx'])

    medzs.append(np.median(qso['Z'])), ms.append(autofit['M']), merrs.append(autofit['sigM'])

    mass_v_prop.plot_hilum_v_z(medzs, ms, merrs, mlum, mlumerr, lowlum, lowlumerr, highpercentile, lowpercentile, True)
    mass_v_prop.plot_hibhmass_v_z(medzs, ms, merrs, mbh, mbherr, lowmass, lowmasserr, highpercentile, lowpercentile, True)



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



def qsolum_allz(scales, npercentiles=10, boss=False):
    percentiles = np.arange(0, 101, npercentiles)
    qso, qsorand = sample.qsocat(eboss=True, boss=boss)

    totfit, totcf = pipeline.measure_and_fit_autocf(scales, qso, qsorand)

    medlum, lumMass, lumMasserr = [], [], []
    medbh, bhMass, bhMasserr = [], [], []
    for j in range(len(percentiles) - 1):
        qsolum = sample.rolling_percentile_selection(qso, prop='Lbol',
                                                     minpercentile=percentiles[j],
                                                     maxpercentile=percentiles[j + 1], nzbins=100)
        medlum.append(np.median(qsolum['Lbol']))
        lumfit, lumcf = pipeline.measure_and_fit_xcf(totcf, scales, qso, qsolum, qsorand)
        lumMass.append(lumfit['Mx']), lumMasserr.append(lumfit['sigMx'])

        qso = qso[np.where(qso['MBH'] > 6)]
        qsobh = sample.rolling_percentile_selection(qso, prop='MBH',
                                                    minpercentile=percentiles[j],
                                                    maxpercentile=percentiles[j + 1], nzbins=100)
        medbh.append(np.median(qsobh['MBH']))
        bhfit, bhcf = pipeline.measure_and_fit_xcf(totcf, scales, qso, qsobh, qsorand)
        bhMass.append(bhfit['Mx']), bhMasserr.append(bhfit['sigMx'])

    mass_v_prop.plot_halomass_v_lum(medlum, lumMass, lumMasserr, 'allz')
    mass_v_prop.plot_halomass_v_mbh(medbh, bhMass, bhMasserr, 'allz')
    return (medlum, lumMass, lumMasserr), (medbh, bhMass, bhMasserr)

def extended_jets(scales):
    qso, qsorand = sample.qsocat(eboss=True, boss=False)
    foo, foo2, qso, qsorand = sample.select_radioloud('LOTSS_DR2', 1.5, qso, qsorand)
    autocf = retrieve_eboss_autocf()
    cffit = clustering_fit.fit_pipeline(redshift_helper.dndz_from_z_list(qso['Z'], 15), autocf)
    ms, merrs, sizes = [cffit['M']], [cffit['sigM']], [2.]
    unmass, unmasserrs = [], []
    fcuts = [0.9, 10., 10000.]
    for j in range(len(fcuts)-1):

        ext = qso[np.where((qso['Resolved'] == 1) & (qso['F_144'] > fcuts[j]) & (qso['F_144'] < fcuts[j+1]))]
        unresolved = qso[np.where((qso['Resolved'] == 0) & (qso['F_144'] > fcuts[j]) & (qso['F_144'] < fcuts[j+1]))]
        sizes.append(np.median(np.log10(ext['Phys_size_144'])))
        extfit, extcf = pipeline.measure_and_fit_xcf(autocf, scales, qso, ext, qsorand)
        #unfit, uncf = pipeline.measure_and_fit_xcf(autocf, scales, qso, unresolved, qsorand)
        ms.append(extfit['Mx']), merrs.append(extfit['sigMx'])
        #unmass.append(unfit['Mx']), unmasserrs.append(unfit['sigMx'])
    print(ms, merrs)
    print(unmass, unmasserrs)

    #mass_v_prop.lofar_extended(sizes, ms, merrs)


def bal_qso(scales):
    qso, rand = sample.qsocat()
    # can't measure BAL at lower z
    qso = qso[np.where(qso['Z'] > 1.58)]
    rand = rand[np.where(rand['Z'] > 1.58)]
    foo, foo2, qso, rand = sample.select_radioloud('LOTSS_DR2', 1.5, qso, rand)
    bal = qso[np.where((qso['BAL_PROB'] > 0.8))]
    balfit, autocf, balcf = pipeline.measure_and_fit_auto_and_xcf(scales, qso, bal, rand)
    balrad = qso[np.where((qso['BAL_PROB'] > 0.8) & (qso['det_144'] == 1) & (qso['F_144'] > 1.1))]
    radfit, radcf = pipeline.measure_and_fit_xcf(autocf, scales, qso, balrad, rand)

    nonbal = qso[np.where((qso['BAL_PROB'] == 0))]
    nonbalfit, nonbalcf = pipeline.measure_and_fit_xcf(autocf, scales, qso, nonbal, rand)

    labels = ['All eBOSS', 'non BAL', 'BAL', 'BAL+RL']
    ms = [balfit['M'], nonbalfit['Mx'], balfit['Mx'], radfit['Mx']]
    merrs = [balfit['sigM'], nonbalfit['sigMx'], balfit['sigMx'], radfit['sigMx']]

    mass_v_prop.mass_bal(labels, ms, merrs)

def radio_loud_lum(scales, nlumbins):
    percentiles = np.arange(0, 101, nlumbins)
    qso, rand = sample.qsocat()
    autocf = retrieve_eboss_autocf()
    lums, mlum, mlumerr, mloud, mlouderr = [], [], [], [], []
    for j in range(len(percentiles)-1):

        qsolum = sample.rolling_percentile_selection(qso, 'Lbol', minpercentile=percentiles[j],
                                                     maxpercentile=percentiles[j+1], nzbins=30)
        lums.append(np.median(qsolum['Lbol']))
        xlumfit, xlumcf = pipeline.measure_and_fit_xcf(autocf, scales, qso, qsolum, rand)

        foo, foo2, qso, rand = sample.select_radioloud('LOTSS_DR2', 3, qso, rand)
        loud, quiet, qsolum, foo = sample.select_radioloud('LOTSS_DR2', 3, qsolum, rand)

        loudfit, loudcf = pipeline.measure_and_fit_xcf(autocf, scales, qso, loud, rand)

        mlum.append(xlumfit['Mx']), mlumerr.append(xlumfit['sigMx'])
        mloud.append(loudfit['Mx']), mlouderr.append(loudfit['sigMx'])
    mass_v_prop.radioloud_mass_v_lum(lums, mlum, mlumerr, mloud, mlouderr)


def radio_luminosity(scales):
    percentiles = np.array([10, 50, 90, 100])
    #autocf = retrieve_eboss_autocf()

    qso, rand = sample.qsocat()
    qso = qso[np.where(qso['Z'] > 1.3)]
    rand = rand[np.where(rand['Z'] > 1.3)]
    qso = qso[np.where(qso['Z'] < 1.7)]
    rand = rand[np.where(rand['Z'] < 1.7)]
    fit, autocf = pipeline.measure_and_fit_autocf(scales, qso, rand, nbootstrap=500)
    qso = qso[np.where(qso['det_144'] > -1)]
    rand = rand[np.where(rand['in_144'] == 1)]
    det = qso[np.where(qso['det_144'] == 1)]
    undet = qso[np.where(qso['det_144'] == 0)]

    lum_mass, lum_sigmass = [], []
    xlumfit, xlumcf = pipeline.measure_and_fit_xcf(autocf, scales, qso, undet, rand)
    lum_mass.append(xlumfit['Mx'])
    lum_sigmass.append(xlumfit['sigMx'])
    for j in range(len(percentiles)-1):
        qsolum = sample.rolling_percentile_selection(det, 'L_144', minpercentile=percentiles[j],
                                                     maxpercentile=percentiles[j + 1], nzbins=15)
        xlumfit, xlumcf = pipeline.measure_and_fit_xcf(autocf, scales, qso, qsolum, rand)
        lum_mass.append(xlumfit['Mx'])
        lum_sigmass.append(xlumfit['sigMx'])


    print(lum_mass)
    print(lum_sigmass)

def fluxcut(scales):
    import matplotlib.pyplot as plt
    cuts = [.5, 3., 10000.]
    autocf = retrieve_eboss_autocf()
    qso, rand = sample.qsocat()
    qso = qso[np.where(qso['det_144'] > -1)]
    rand = rand[np.where(rand['in_144'] == 1)]

    mbhs = []
    for j in range(len(cuts)):
        if j == 0:
            thisqso = qso[np.where((qso['F_144'] < cuts[0]) | (qso['det_144'] == 0))]
        else:
            thisqso = qso[np.where((qso['F_144'] > cuts[j-1]) & (qso['F_144'] <= cuts[j]) & (qso['det_144'] == 1))]
        mbhs.append(thisqso['MBH'])
    weights = sample.overlap_weights_1d(mbhs, (6, 11), 20)

    ms, errs, lums = [], [], []
    for j in range(len(cuts)):
        if j == 0:
            thisqso = qso[np.where((qso['F_144'] < cuts[0]) | (qso['det_144'] == 0))]
        else:
            thisqso = qso[np.where((qso['F_144'] > cuts[j-1]) & (qso['F_144'] <= cuts[j]) & (qso['det_144'] == 1))]
        thisqso['weight'] *= weights[j]
        fit, cf = pipeline.measure_and_fit_xcf(autocf, scales, qso, thisqso, rand)

        ms.append(fit['Mx']), errs.append(fit['sigMx']), lums.append(np.nanmedian(thisqso['L_144']))

    plt.close('all')
    plt.figure(figsize=(8,7))
    plt.scatter(lums, ms, c='k')
    plt.scatter(39.8, ms[0], c='k')
    plt.errorbar(39.8, ms[0], xuplims=True, xerr=0.1, yerr=errs[0], ecolor='k', fmt='none')
    plt.errorbar(lums[1:], ms[1:], yerr=errs[1:], ecolor='k', fmt='none')
    plt.xlabel(r'log$_{10} L_{144 \ \mathrm{MHz}}$ [erg/s]')
    plt.ylabel(r'log$_{10} (M_h / h^{-1} M_{\odot})$')
    plt.ylim(12.4, 13.)
    plt.savefig(plotdir + 'eboss_lum.pdf')
    plt.close('all')

def autocorr_hzrgs(rpscales):
    old = glob.glob('results/cfs/auto/hzrg*')
    old += glob.glob('results/fits/auto/hzrg*')
    old += glob.glob('results/dndz/hzrg*')
    for name in old:
        os.remove(name)

    lotss = sample.hzrg_sample()
    rand = sample.lotss_randoms()
    zs = sample.treat_dndz_pdf(lotss)
    dndz_lotss = redshift_helper.dndz_from_z_list(zs, 30, zrange=(0.1, 4))
    np.array(dndz_lotss).dump('results/dndz/hzrg.pickle')
    eff_z = np.median(dndz_lotss)
    thetabins = cosmo.rp2angle(rpscales, eff_z, True)

    cf = twoPointCFs.autocorr_cat(thetabins, lotss, rand, nbootstrap=500)
    cf.pop('plot')
    write_pickle('results/cfs/auto/hzrg', cf)

    fit = clustering_fit.fit_pipeline(dndz_lotss, cf)
    fit['eff_z'] = eff_z
    fit.pop('plot')
    write_pickle('results/fits/auto/hzrg', fit)
    results.autocorrs()


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
    eff_z = np.median(dndz_lotss)
    thetabins = cosmo.rp2angle(rpscales, eff_z, True)

    cf = twoPointCFs.autocorr_cat(thetabins, lotss, rand, nbootstrap=500)
    cf.pop('plot')
    write_pickle('results/cfs/auto/izrg', cf)

    fit = clustering_fit.fit_pipeline(dndz_lotss, cf)
    fit['eff_z'] = eff_z
    fit.pop('plot')
    write_pickle('results/fits/auto/izrg', fit)
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

    dndz_lotss = redshift_helper.dndz_from_z_list(zs, 30, zrange=(0.1, 4.))
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

        dndz_elg_matched = redshift_helper.dndz_from_z_list(rand['Z'], 30, zrange=(0.1, 4.))
        xfit = clustering_fit.fit_xcf(dndz_lotss, xcf, dndz_elg_matched, autocf, model='mass')
        xfit.update(clustering_fit.fit_xcf(dndz_lotss, xcf, dndz_elg_matched, autocf, model='minmass'))
        xfit['eff_z'] = redshift_helper.effective_z(dndz_lotss, dndz_elg_matched)
        write_pickle('results/fits/tomo/rg_elg', xfit)
    results.cross_cfs()
    results.halomass()





def hzrg_xcorr(rpscales):
    oldresults = glob.glob('results/cfs/tomo/rg_qso*.pickle')
    oldresults += glob.glob('results/fits/tomo/rg_qso*.pickle')
    oldresults += glob.glob('results/dndz/hzrg.pickle')
    for result in oldresults:
        os.remove(result)
    cflist = []

    eboss_zbins = np.linspace(1., 2.5, 4)

    qso, rand = sample.eboss_qso()

    lotss = sample.hzrg_sample()
    lotss = sample.cat_in_eboss(lotss)

    qso = sample.cat_in_lotss(qso)
    rand = sample.cat_in_lotss(rand)
    qso = sample.cat_in_goodwise(qso)
    rand = sample.cat_in_goodwise(rand)

    zs = sample.treat_dndz_pdf(lotss)

    dndz_lotss = redshift_helper.dndz_from_z_list(zs, 30, zrange=(0.1, 4.))
    np.array(dndz_lotss).dump('results/dndz/hzrg.pickle')

    for j in range(len(eboss_zbins)-1):
        minz, maxz = eboss_zbins[j], eboss_zbins[j+1]
        qsoz = qso[np.where((qso['Z'] > minz) & (qso['Z'] < maxz))]
        randz = rand[np.where((rand['Z'] > minz) & (rand['Z'] < maxz))]

        # convert rp scales to angular scales at median redshift of bin
        theta_scales = cosmo.rp2angle(rps=rpscales, z=np.median(randz['Z']), h_unit=True)


        autofit, autocf = pipeline.measure_and_fit_autocf(scales=rpscales, datcat=qsoz, randcat=randz, nbootstrap=500, nzbins=5)
        autofit.pop('plot')
        write_pickle('results/fits/tomo/qso_fit_%s' % j, autofit)

        dndz_qso_matched = redshift_helper.dndz_from_z_list(randz['Z'], 30, zrange=(0.1, 4.))
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
        cflist.append(xcf)
        write_pickle('results/fits/tomo/rg_qso_fit_%s' % j, xfit)

    results.cross_cfs()
    results.halomass()











def lens_hzrg(nside=1024):
    from mocpy import MOC
    import healpy as hp
    import astropy.units as u
    from lensing_Helper import measure_lensing_xcorr

    oldresults = glob.glob('results/lenscorr/hzrg*')
    oldresults += glob.glob('results/lensfits/hzrg*')
    for result in oldresults:
        os.remove(result)

    lotss = sample.hzrg_sample()

    lotssmoc = MOC.from_fits('../data/radio_cats/LOTSS_DR2/lotss_dr2_hips_moc.fits')
    mask = np.zeros(hp.nside2npix(nside))
    l, b = hp.pix2ang(nside, np.arange(len(mask)), lonlat=True)
    ra, dec = coordhelper.galactic_to_equatorial(l, b)
    idx = lotssmoc.contains(ra*u.deg, dec*u.deg)
    mask[idx] = 1.
    #mask = np.ones(hp.nside2npix(nside))

    corr = measure_lensing_xcorr.measure_planck_xcorr(np.logspace(2, 3, 10), (lotss['RA'], lotss['DEC']), nside, mask=mask, accurate=True)
    write_pickle('results/lenscorr/hzrg', corr)

    dndz = redshift_helper.dndz_from_z_list(sample.treat_dndz_pdf(lotss), 30, zrange=(0.01, 4.))
    fit = lensing_fit.xcorr_fit_pipeline(dndz, corr)
    fig = fit.pop('plot')
    write_pickle('results/lensfits/hzrg', fit)

    fig.savefig(plotdir + 'hiz_rg_lensing.pdf')


def lens_izrg(nside=1024):
    from mocpy import MOC
    import healpy as hp
    import astropy.units as u
    from lensing_Helper import measure_lensing_xcorr

    oldresults = glob.glob('results/lenscorr/izrg*')
    oldresults += glob.glob('results/lensfits/izrg*')
    for result in oldresults:
        os.remove(result)


    lotss = sample.izrg_sample()

    lotssmoc = MOC.from_fits('../data/radio_cats/LOTSS_DR2/lotss_dr2_hips_moc.fits')
    mask = np.zeros(hp.nside2npix(nside))
    l, b = hp.pix2ang(nside, np.arange(len(mask)), lonlat=True)
    ra, dec = coordhelper.galactic_to_equatorial(l, b)
    idx = lotssmoc.contains(ra*u.deg, dec*u.deg)
    mask[idx] = 1.
    #mask = np.ones(hp.nside2npix(nside))

    corr = measure_lensing_xcorr.measure_planck_xcorr(np.logspace(2, 3, 10), (lotss['RA'], lotss['DEC']), nside, mask=mask, accurate=True)
    write_pickle('results/lenscorr/izrg', corr)

    dndz = redshift_helper.dndz_from_z_list(sample.treat_dndz_pdf(lotss), 7, zrange=(0.3, 1.2))
    fit = lensing_fit.xcorr_fit_pipeline(dndz, corr)
    fig = fit.pop('plot')
    write_pickle('results/lensfits/izrg', fit)

    fig.savefig(plotdir + 'izrg_lensing.pdf')



def duty_cycle():
    fluxcuthi = 2.
    fluxcutmid = 5.
    fluxmax = 1000.

    import glob

    izrgfit = read_pickle('results/fits/auto/izrg.pickle')

    dutyizrg = lumfunc.occupation_zrange(izrgfit['Mmin'], (0.5, 1.), fluxcut=fluxcutmid, fluxmax=fluxmax, lftype='lerg')
    izrgloerr = dutyizrg - lumfunc.occupation_zrange(izrgfit['Mmin']-izrgfit['sigMmin'], (0.5, 1.),
                                                     fluxcut=fluxcutmid, fluxmax=fluxmax, lftype='lerg')
    izrghierr = lumfunc.occupation_zrange(izrgfit['Mmin'] + izrgfit['sigMmin'], (0.5, 1.),
                                                     fluxcut=fluxcutmid, fluxmax=fluxmax, lftype='lerg') - dutyizrg

    izrgstuff = [izrgfit['eff_z'], dutyizrg, izrgloerr, izrghierr]

    z_centers = []

    qsofitnames = glob.glob('results/fits/tomo/rg_qso*.pickle')
    fits = []
    qsozrange = [1., 1.5, 2., 2.5]
    for name in qsofitnames:
        thisfit = read_pickle(name)
        fits.append(thisfit)
        z_centers.append(thisfit['eff_z'])

    duty, loerr, hierr = [], [], []
    for j in range(len(fits)):
        fduty = lumfunc.occupation_zrange(logminmass=fits[j]['Mxmin'], zrange=(qsozrange[j], qsozrange[j+1]),
                                          fluxcut=fluxcuthi, fluxmax=1000.)
        duty.append(fduty)
        loerr.append(fduty-lumfunc.occupation_zrange(zrange=(qsozrange[j], qsozrange[j+1]), logminmass=fits[j]['Mxmin'] - fits[j]['sigMxmin'], fluxcut=fluxcuthi, fluxmax=fluxmax))
        hierr.append(lumfunc.occupation_zrange(zrange=(qsozrange[j], qsozrange[j + 1]), logminmass=fits[j]['Mxmin'] + fits[j]['sigMxmin'],
                                                 fluxcut=fluxcuthi, fluxmax=1000.)-fduty)

    results.dutycycle(izrgstuff, z_centers, duty, loerr, hierr)

def haloenergy():
    z_centers = []
    import glob
    qsofitnames = glob.glob('results/fits/tomo/rg_qso*.pickle')
    fits = []
    qsozrange = [1., 1.5, 2., 2.5]
    for name in qsofitnames:
        thisfit = read_pickle(name)
        fits.append(thisfit)
        z_centers.append(thisfit['eff_z'])

    logminmass = 12.8
    kineticfraclow, kineticfrachi = 0.001, 0.005
    allzranges = [0.0, 0.5, 1., 1.5, 2., 2.5]
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

    results.energetics(z_centers, es, elo, ehi, qsowindinfo)
