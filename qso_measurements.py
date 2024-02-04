"""
Deprecated codes studying clustering of optical-spectroscopic quasars by radio properties
"""
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

def desiqso(minz=None, maxz=None):
	dat = Table.read('catalogs/masked/desiQSO_edr.fits')
	rand = Table.read('catalogs/masked/desiQSO_edr_randoms.fits')
	if minz is not None:
		dat = dat[np.where(dat['Z'] > minz)]
		rand = rand[np.where(rand['Z'] > minz)]
	if maxz is not None:
		dat = dat[np.where(dat['Z'] < maxz)]
		rand = rand[np.where(rand['Z'] < maxz)]
	return dat, rand

def desi_elg(minz=None, maxz=None):
	dat = Table.read('/home/graysonpetter/ssd/Dartmouth/data/lss/desiELG_edr/desiELG_edr.fits')
	rand = Table.read('/home/graysonpetter/ssd/Dartmouth/data/lss/desiELG_edr/desiELG_edr_randoms.fits')
	if minz is not None:
		dat = dat[np.where(dat['Z'] > minz)]
		rand = rand[np.where(rand['Z'] > minz)]
	if maxz is not None:
		dat = dat[np.where(dat['Z'] < maxz)]
		rand = rand[np.where(rand['Z'] < maxz)]
	return dat, rand

def desi_lrg(minz=None, maxz=None):
	dat = Table.read('catalogs/masked/desiLRG_edr.fits')
	rand = Table.read('catalogs/masked/desiLRG_edr_randoms.fits')
	if minz is not None:
		dat = dat[np.where(dat['Z'] > minz)]
		rand = rand[np.where(rand['Z'] > minz)]
	if maxz is not None:
		dat = dat[np.where(dat['Z'] < maxz)]
		rand = rand[np.where(rand['Z'] < maxz)]
	return dat, rand

def collate_z_equals_1_tracers(elgs=False):
	if elgs:
		maxz = 1.1
	else:
		maxz = 1.
	qso = Table.read('../data/lss/eBOSS_QSO/eBOSS_QSO.fits')
	qsorand = Table.read('../data/lss/eBOSS_QSO/eBOSS_QSO_randoms.fits')
	lrg = Table.read('../data/lss/eBOSS_LRG/eBOSS_LRG.fits')
	lrgrand = Table.read('../data/lss/eBOSS_LRG/eBOSS_LRG_randoms.fits')
	qso = catalog_utils.cut_table(qso, 'Z', 0.8, maxz)
	qsorand = catalog_utils.cut_table(qsorand, 'Z', 0.8, maxz)
	lrg = catalog_utils.cut_table(lrg, 'Z', 0.8, maxz)
	lrgrand = catalog_utils.cut_table(lrgrand, 'Z', 0.8, maxz)
	totdat = vstack((qso, lrg))
	totrand = vstack((qsorand, lrgrand))
	if elgs:
		elg = Table.read('../data/lss/eBOSS_ELG/eBOSS_ELG.fits')
		elgrand = Table.read('../data/lss/eBOSS_ELG/eBOSS_ELG_randoms.fits')
		elg = catalog_utils.cut_table(elg, 'Z', 0.8, maxz)
		elgrand = catalog_utils.cut_table(elgrand, 'Z', 0.8, maxz)
		totdat = vstack((totdat, elg))
		totrand = vstack((totrand, elgrand))
	return totdat, totrand


def radio_match(qsosample, radio_names, sep, alpha_fid=-0.8):
    # qsocat = mask_sample(qsosample, radio_names)
    qsocat = Table.read('../data/lss/%s/%s.fits' % (qsosample, qsosample))

    for i, radio_name in enumerate(radio_names):
        radio_freq = freq_dict[radio_name]
        # VLASS quick look images have systematic undermeasurement of flux https://science.nrao.edu/vlass/data-access/vlass-epoch-1-quick-look-users-guide
        peak_corr, tot_corr = 1., 1.
        if radio_name == 'VLASS':
            peak_corr, tot_corr = 1. / (1. - 0.08), 1. / (1. - 0.03)
        qsocat['det_%s' % radio_freq] = -1 * np.ones(len(qsocat))
        qsocat['F_%s' % radio_freq] = np.full(len(qsocat), np.nan)
        qsocat['Ferr_%s' % radio_freq] = np.full(len(qsocat), np.nan)
        qsocat['angsize_%s' % radio_freq] = np.full(len(qsocat), np.nan)

        radiocat = Table.read(
            '../data/radio_cats/%s/%s.fits' % (radio_name, radio_name)
        )

        # set detect to zero if within footprint but not detected
        in_footprint = mask_sample((qsocat['RA'], qsocat['DEC']), [radio_name])
        qsocat['det_%s' % radio_freq][in_footprint] = 0.
        qsocat['F_%s' % radio_freq][in_footprint] = upper_limits((qsocat['RA'][in_footprint],
                                                                  qsocat['DEC'][in_footprint]),
                                                                 radio_name)

        radidx, qsoidx = coordhelper.match_coords((radiocat['RA'], radiocat['DEC']), (qsocat['RA'], qsocat['DEC']),
                                                  sep, symmetric=False)

        qsocat['det_%s' % radio_freq][qsoidx] = 1.
        qsocat['det_%s' % radio_freq][np.logical_not(in_footprint)] = -1.
        # qsocat['angsize_%s' % radio_freq][qsoidx] = radiocat['Maj'][radidx]
        qsocat['F_%s' % radio_freq][qsoidx] = radiocat['Total_flux'][radidx] * tot_corr
        qsocat['Ferr_%s' % radio_freq][qsoidx] = radiocat['E_Total_flux'][radidx] * tot_corr
        qsocat['angsize_%s' % radio_freq][qsoidx] = radiocat['Maj'][radidx]
        if radio_name.startswith('LOTSS'):
            # lofar_resolved(qsocat)
            qsocat['PeakF_%s' % freq_dict[radio_name]] = np.full(len(qsocat), np.nan)
            qsocat['PeakFerr_%s' % freq_dict[radio_name]] = np.full(len(qsocat), np.nan)
            qsocat['PeakF_%s' % radio_freq][qsoidx] = radiocat['Peak_flux'][radidx]
            qsocat['PeakFerr_%s' % radio_freq][qsoidx] = radiocat['E_Peak_flux'][radidx]
            qsocat = lotss_dr2resolved(qsocat)

        if 'Z' in qsocat.colnames:

            if radio_name.startswith('LOTSS'):
                # lofar_resolved(qsocat)
                qsocat['PeakF_%s' % freq_dict[radio_name]] = np.full(len(qsocat), np.nan)
                qsocat['PeakFerr_%s' % freq_dict[radio_name]] = np.full(len(qsocat), np.nan)
                qsocat['PeakF_%s' % radio_freq][qsoidx] = radiocat['Peak_flux'][radidx]
                qsocat['PeakFerr_%s' % radio_freq][qsoidx] = radiocat['E_Peak_flux'][radidx]
                qsocat['Phys_size_144'] = np.full(len(qsocat), np.nan)
                qsocat['Phys_size_144'][qsoidx] = physical_size(qsocat['angsize_144'][qsoidx], qsocat['Z'][qsoidx])
                # qsocat = lofar_resolved(qsocat)
                qsocat = lotss_dr2resolved(qsocat)

                # qsocat['L_144'] = np.log10(150. * 10**6 * (qsocat['F_LOTSS_DR2'] * u.mJy * 4 * np.pi * (lum_dists * (1 + qsocat['Z']) ** (-(alpha_fid + 1) / 2.)) ** 2).to(u.erg).value)
                # s_1400_lotss = qsocat['F_LOTSS_DR2'] * (1400./150.) ** alpha_fid
                # qsocat['L_1400_lotss'] = np.log10(1.4e9 * (s_1400_lotss * u.mJy * 4 * np.pi * (lum_dists * (1 + qsocat['Z']) ** (-(alpha_fid + 1) / 2.)) ** 2).to(u.erg).value)
                qsocat['L_144'] = np.log10(fluxutils.luminosity_at_rest_nu(qsocat['F_144'], alpha=alpha_fid,
                                                                           nu_obs=.144, nu_rest_want=.144,
                                                                           z=qsocat['Z'], flux_unit=u.mJy))

                qsocat['SFR_lotss'] = fluxutils.best23_lum150_2sfr(
                    np.log10(fluxutils.luminosity_at_rest_nu(qsocat['F_144'], alpha=alpha_fid,
                                                             nu_obs=.144, nu_rest_want=.144,
                                                             z=qsocat['Z'], flux_unit=u.mJy, energy=False)))
                qsocat['L_1400_lotss'] = np.log10(fluxutils.luminosity_at_rest_nu(qsocat['F_144'], alpha=alpha_fid,
                                                                                  nu_obs=.144, nu_rest_want=1.4,
                                                                                  z=qsocat['Z'], flux_unit=u.mJy))

            if radio_name == 'FIRST':
                # qsocat['L_1400_fid'] = np.log10(1.4e9 * (qsocat['F_FIRST'] * u.mJy * 4 * np.pi * (lum_dists * (1 + qsocat['Z']) ** (-(alpha_fid + 1) / 2.)) ** 2).to(u.erg).value)
                qsocat['L_1400'] = np.log10(fluxutils.luminosity_at_rest_nu(qsocat['F_1400'], alpha=alpha_fid,
                                                                            nu_obs=1.4, nu_rest_want=1.4,
                                                                            z=qsocat['Z'], flux_unit=u.mJy))
            # freq1, freq2 = freq_dict['LOTSS_DR2'], freq_dict['FIRST']
            # qsocat['alpha_%s_%s' % (freq1, freq2)] = np.log10(qsocat['F_%s' % freq2] / qsocat['F_%s' % freq1]) / np.log10(float(freq2)/float(freq1))
            if radio_name == 'VLASS':
                # qsocat['L_3000_fid'] = np.log10(3.e9 * (qsocat['F_VLASS'] * u.mJy * 4 * np.pi * (lum_dists * (1 + qsocat['Z']) ** (-(alpha_fid + 1) / 2.)) ** 2).to(u.erg).value)
                # s_1400_vlass = qsocat['F_VLASS'] * (1400./3000.) ** alpha_fid
                # qsocat['L_1400_vlass'] = np.log10(1.4e9 * (s_1400_vlass * u.mJy * 4 * np.pi * (lum_dists * (1 + qsocat['Z']) ** (-(alpha_fid + 1) / 2.)) ** 2).to(u.erg).value)
                qsocat['L_3000'] = np.log10(fluxutils.luminosity_at_rest_nu(qsocat['F_3000'], alpha=alpha_fid,
                                                                            nu_obs=3., nu_rest_want=3.,
                                                                            z=qsocat['Z'], flux_unit=u.mJy))
                qsocat['L_1400_vlass'] = np.log10(fluxutils.luminosity_at_rest_nu(qsocat['F_3000'], alpha=alpha_fid,
                                                                                  nu_obs=3., nu_rest_want=1.4,
                                                                                  z=qsocat['Z'], flux_unit=u.mJy))

        if i > 0:
            freq1, freq2 = freq_dict[radio_names[0]], freq_dict[radio_names[i]]
            qsocat['alpha_%s_%s' % (freq1, freq2)] = np.full(len(qsocat), np.nan)
            atleast1detection = np.where((qsocat['det_%s' % freq1] == 1) | (qsocat['det_%s' % freq2] == 1))
            qsocat['alpha_%s_%s' % (freq1, freq2)][atleast1detection] = \
                np.log10(qsocat['F_%s' % freq2][atleast1detection] /
                         qsocat['F_%s' % freq1][atleast1detection]) / \
                np.log10(float(freq2) / float(freq1))

    qsocat.write('catalogs/masked/%s.fits' % qsosample, overwrite=True)


def lotss_deep_match(qsosample, sep, alpha_fid=-0.8):
    lockman_ctr = (161.75, 58.083)
    lockmanmask = myhp.mask_from_pointings(([lockman_ctr[0]], [lockman_ctr[1]]), 2048, pointing_radius=2.)
    highresmask = myhp.mask_from_pointings(([lockman_ctr[0]], [lockman_ctr[1]]), 2048, pointing_side=2.)

    radiocat = Table.read('../data/radio_cats/lofar_deep/lockman_pybdsf_source.fits')
    highrescat = Table.read('../data/radio_cats/lofar_highres/lofar_highres.fits')
    highrescat = highrescat[myhp.inmask((highrescat['RA'], highrescat['DEC']), highresmask)]

    highresrmsmap = myhp.healpix_average_in_pixels(highrescat['RA'], highrescat['DEC'], nsides=256,
                                                   values=np.log10(1000. * highrescat['E_Peak_flux']))
    highresrmsmap = 10 ** highresrmsmap

    # qsocat = mask_sample(qsosample, radio_names)
    qsocat = Table.read('catalogs/masked/%s.fits' % qsosample)
    qsocat = qsocat[myhp.inmask((qsocat['RA'], qsocat['DEC']), lockmanmask)]

    qsocat['det_deep'] = np.zeros(len(qsocat))
    qsocat['det_highres'] = np.full(len(qsocat), np.nan)
    qsocat['F_ILT'] = np.full(len(qsocat), np.nan)
    qsocat['Ferr_ILT'] = np.full(len(qsocat), np.nan)
    qsocat['Peak_ILT'] = np.full(len(qsocat), np.nan)
    qsocat['Peakerr_ILT'] = np.full(len(qsocat), np.nan)
    qsocat['det_highres'][myhp.inmask((qsocat['RA'], qsocat['DEC']), highresmask)] = 0.

    # qsocat['Peak_%s' % freq_dict[radio_name]] = np.full(len(qsocat), np.nan)

    # 3 sigma limit of lockman field
    qsocat['F_144'][np.where(qsocat['det_144'] == 0)] = .075

    qsoidx, radioidx = coordhelper.match_coords((qsocat['RA'], qsocat['DEC']),
                                                (radiocat['RA'], radiocat['DEC']),
                                                max_sep=sep)
    qsocat['det_deep'][qsoidx] = 1

    qsocat['F_144'][qsoidx] = 1000. * np.array(radiocat['Total_flux'][radioidx])
    qsocat['Ferr_144'][qsoidx] = 1000. * np.array(radiocat['E_Total_flux'][radioidx])
    qsocat['L_144'] = np.log10(fluxutils.luminosity_at_rest_nu(qsocat['F_144'], alpha_fid, nu_obs=.144,
                                                               nu_rest_want=.144, z=qsocat['Z'], flux_unit=u.mJy))
    qsoidx, radioidx = coordhelper.match_coords((qsocat['RA'], qsocat['DEC']),
                                                (highrescat['RA'], highrescat['DEC']),
                                                max_sep=sep)
    qsocat['det_highres'][qsoidx] = 1
    qsocat['F_ILT'][qsoidx] = 1000. * np.array(radiocat['Total_flux'][radioidx])
    qsocat['Ferr_ILT'][qsoidx] = 1000. * np.array(radiocat['E_Total_flux'][radioidx])
    qsocat['Peak_ILT'][qsoidx] = 1000. * np.array(radiocat['Peak_flux'][radioidx])
    qsocat['Peakerr_ILT'][qsoidx] = 1000. * np.array(radiocat['E_Peak_flux'][radioidx])
    # upper limits for high resolution
    nondet = qsocat[np.where(qsocat['det_highres'] == 0)]
    qsocat['Peak_ILT'][np.where(qsocat['det_highres'] == 0)] = \
        3. * myhp.coords2mapvalues(nondet['RA'], nondet['DEC'], highresrmsmap)
    qsocat['F_ILT'][np.where(qsocat['det_highres'] == 0)] = \
        3. * myhp.coords2mapvalues(nondet['RA'], nondet['DEC'], highresrmsmap)
    radio_lum.highres_flux(qsocat)
    radio_lum.loudness_dist(qsocat)

    qsocat.write('catalogs/masked/%s_deep.fits' % qsosample, overwrite=True)


def mask_sample(coords, radio_names):
    ras, decs = coords
    if 'FIRST' in radio_names:
        """good_ngc = ((decs > -8) & (decs < 15) & (ras < 232.5) & (ras > 138.75)) | ((decs > 15) & (decs < 57.6) & (ras < 251.25) & (ras > 120))
        goodsgc = (decs > -7) & (decs < 7) & ((ras > 315) | (ras < 45))
        good_idxs = np.logical_or(good_ngc, goodsgc)
        firstrms = hp.read_map('masks/FIRST_rms.fits')
        goodrms = np.log10(firstrms[hp.ang2pix(hp.npix2nside(len(firstrms)), ras, decs, lonlat=True)]) < -0.75
        good_idxs = goodrms * good_idxs"""
        good_idxs = myhp.inmask(coords, hp.read_map('../data/radio_cats/FIRST/mask.fits'), return_bool=True)

    if 'LOTSS_DR2' in radio_names:
        moc = MOC.from_fits('../data/radio_cats/LOTSS_DR2/lotss_dr2_hips_moc.fits')
        good_idxs = moc.contains(np.array(ras) * u.deg, np.array(decs) * u.deg)
    # qsocat = qsocat[np.where(in_mocs)]
    # rmsmask = hp.read_map('masks/LOTSS_DR2_noise_mask.fits')
    # goodrms = rmsmask[hp.ang2pix(hp.npix2nside(len(rmsmask)), ras, decs, lonlat=True)]

    if 'LoLSS_DR1' in radio_names:
        moc = MOC.from_fits('../data/radio_cats/LoLSS_DR1/Moc.fits')
        good_idxs = moc.contains(np.array(ras) * u.deg, np.array(decs) * u.deg)
    # qsocat = qsocat[np.where(in_mocs)]
    # rmsmask = hp.read_map('masks/LOTSS_DR2_noise_mask.fits')
    # goodrms = rmsmask[hp.ang2pix(hp.npix2nside(len(rmsmask)), ras, decs, lonlat=True)]

    # good_idxs = good_idxs * goodrms

    if 'VLASS' in radio_names:
        vlass_mask = hp.read_map('masks/VLASS.fits')
        good_idxs = vlass_mask[hp.ang2pix(hp.npix2nside(len(vlass_mask)), ras, decs, lonlat=True)]
    if 'Apertif' in radio_names:
        apertif_mask = hp.read_map('masks/Apertif.fits')
        good_idxs = apertif_mask[hp.ang2pix(hp.npix2nside(len(apertif_mask)), ras, decs, lonlat=True)]

    return good_idxs.astype(bool)


def upper_limits(coords, radio_name, upper_lim_multiplier=3):
    ras, decs = coords
    rmsmap = hp.read_map('masks/%s_rms.fits' % radio_name)
    rmsvals = rmsmap[hp.ang2pix(hp.npix2nside(len(rmsmap)), ras, decs, lonlat=True)]
    return rmsvals * upper_lim_multiplier



def mask_randoms(sample='eBOSS', nrand2ndata=15):
	radio_names = ['LOTSS_DR2', 'FIRST', 'VLASS']
	cat = Table.read('../data/lss/%s/%s.fits' % (sample,  sample))
	rands = Table.read('../data/lss/%s/%s_randoms.fits' % (sample,  sample))[:nrand2ndata*len(cat)]

	for i, radio_name in enumerate(radio_names):
		radio_freq = freq_dict[radio_name]
		rands['in_%s' % radio_freq] = np.zeros(len(rands))
		in_footprint = mask_sample((rands['RA'], rands['DEC']), [radio_name])
		rands['in_%s' % radio_freq][in_footprint] = 1.

	rands.write('catalogs/masked/%s_randoms.fits' % sample, overwrite=True)