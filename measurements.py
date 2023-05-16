from corrfunc_helper import twoPointCFs
from halomodelpy import clustering_fit, redshift_helper, pipeline
import numpy as np
import sample
from SkyTools import catalog_utils
from astropy.table import Table
from plotscripts import mass_v_prop
import pickle

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



def lrg_radioloud(scales, f_cuts=np.array([0.5, 1.1, 5.])):
    lrg = Table.read('catalogs/masked/eBOSS_LRG.fits')
    lrgrand = Table.read('catalogs/masked/eBOSS_LRG_randoms.fits')
    lrgfit, lrgcf = pipeline.measure_and_fit_autocf(scales, lrg, lrgrand)
    lrgm, lrgmerr = lrgfit['M'], lrgfit['sigM']

    loudness, ms, merrs = [], [], []
    for j in range(len(f_cuts)):
        radiolrg, foo, newlrg, newlrgrand = sample.select_radioloud('LOTSS_DR2', f_cuts[j], lrg, lrgrand)
        loudness.append(np.median(radiolrg['L_144'] - radiolrg['L2']))
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
    fcuts = [1.1, 5.]
    for j in range(len(fcuts)):

        ext = qso[np.where((qso['angsize_144'] > 15) & (qso['F_144'] > fcuts[j]))]
        sizes.append(np.median(np.log10(ext['Phys_size_144'])))
        extfit, extcf = pipeline.measure_and_fit_xcf(autocf, scales, qso, ext, qsorand)
        ms.append(extfit['Mx']), merrs.append(extfit['sigMx'])

    mass_v_prop.lofar_extended(sizes, ms, merrs)


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
    lumbins = [41.5, 42.5, 45]
    qso, rand = sample.qsocat()
    foo, foo2, qso, rand = sample.select_radioloud('LOTSS_DR2', 1.1, qso, rand)

    brighter_mass, brighter_sigmass = [], []
    for j in range(len(lumbins)-1):
        morelum = qso[np.where((qso['det_144'] == 1) & (qso['L_144'] > lumbins[j]))]
        lumfit, autocf, lumcf = pipeline.measure_and_fit_auto_and_xcf(scales, qso, morelum, rand)
        brighter_mass.append(lumfit['Mx'])
        brighter_sigmass.append(lumfit['sigMx'])


