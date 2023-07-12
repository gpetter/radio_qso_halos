from corrfunc_helper import twoPointCFs
from halomodelpy import clustering_fit, redshift_helper, pipeline, hm_calcs, lensing_fit, cosmo
import numpy as np
import sample
from SkyTools import catalog_utils, coordhelper
from astropy.table import Table
from plotscripts import mass_v_prop
import pickle

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

def autocorr_rgs(fcut=2):
    lotss = sample.lotss_rg_sample(fcut=fcut)
    rand = sample.lotss_randoms()
    cf = twoPointCFs.autocorr_cat(np.logspace(-1.5, -0.5, 10), lotss, rand, nbootstrap=500)
    return cf

def desi_elg_xcorr(rpscales):
    zbins = np.array([1., 1.5])
    zcenters = np.array([1.25])
    lotss = sample.lotss_rg_sample()
    lotss = sample.cat_in_desi_elg(lotss)
    lotssrand = sample.lotss_randoms()
    lotssrand = sample.cat_in_desi_elg(lotssrand)
    elgmass, elgmasserr = [], []
    xmass, xerr = [], []
    zs = sample.redshift_dist(lotss, 5.)

    dndz_lotss = redshift_helper.dndz_from_z_list(zs, 15, zrange=(0.1, 4.))
    for j in range(len(zcenters)):
        elg, rand = sample.desi_elg(zbins[j], zbins[j+1])
        theta_scales = cosmo.rp2angle(rps=rpscales, z=np.median(zcenters[j]), h_unit=True)
        autocf = twoPointCFs.autocorr_cat(rpscales, elg, rand)
        autocf['wp_err'] = autocf['wp_poisson_err']
        autofit = clustering_fit.fit_pipeline(redshift_helper.dndz_from_z_list(rand['Z'], 10), autocf)
        elgmass.append(autofit['M'])
        elgmasserr.append(autofit['sigM'])

        elg.remove_column('CHI')
        rand.remove_column('CHI')

        xcf = twoPointCFs.crosscorr_cats(theta_scales, elg, lotss, rand, lotssrand)
        xcf['w_err'] = xcf['w_err_poisson']
        dndz_elg_matched = redshift_helper.dndz_from_z_list(rand['Z'], 15, zrange=(0.1, 4.))
        xfit = clustering_fit.fit_xcf(dndz_lotss, xcf, dndz_elg_matched, autocf, model='mass')
        xmass.append(xfit[0])
        xerr.append(xfit[1])


    return zcenters, np.array(elgmass), np.array(elgmasserr), np.array(xmass), np.array(xerr)



def lotss_selected_xcorr(rpscales, fcut=2):
    cflist = []
    radmasses, raderrs, qsomasses, qsoerrs = [], [], [], []


    import matplotlib.pyplot as plt

    eboss_zbins = np.linspace(1., 2.5, 4)
    z_centers = list(eboss_zbins[:-1] + (eboss_zbins[1] - eboss_zbins[0])/2)



    qso, rand = sample.eboss_qso()

    lotss = sample.lotss_rg_sample(fcut=fcut)
    lotss = sample.cat_in_eboss(lotss)


    qso = sample.cat_in_lotss(qso)
    rand = sample.cat_in_lotss(rand)
    qsomods = []
    for j in range(len(eboss_zbins)-1):
        minz, maxz = eboss_zbins[j], eboss_zbins[j+1]
        qsoz = qso[np.where((qso['Z'] > minz) & (qso['Z'] < maxz))]
        randz = rand[np.where((rand['Z'] > minz) & (rand['Z'] < maxz))]

        # convert rp scales to angular scales at median redshift of bin
        theta_scales = cosmo.rp2angle(rps=rpscales, z=np.median(randz['Z']), h_unit=True)

        #fluxthresh = fluxutils.flux_at_obsnu_from_rest_lum(10**best23_ridgeline_lum, -0.8,
        #                                                        .144, .144, minz, energy=False) / 1000.
        #print('Flux cut for z>%s is S_144=%s mJy' % (minz, round(fluxthresh, 2)))
        fluxthresh = 2

        lotss_bright = lotss[np.where(lotss['Total_flux'] > fluxthresh)]
        #lotss_bright = lotss_bright[np.where(lotss_bright['r90'] == 1)]

        zs = sample.redshift_dist(lotss_bright, 5.)

        dndz_lotss = redshift_helper.dndz_from_z_list(zs, 15, zrange=(0.1, 4.))

        autofit, autocf = pipeline.measure_and_fit_autocf(scales=rpscales, datcat=qsoz, randcat=randz, nbootstrap=500, nzbins=5)

        dndz_qso_matched = redshift_helper.dndz_from_z_list(randz['Z'], 30, zrange=(0.1, 4.))
        #dndz_lotss = redshift_helper.spl_interp_dndz(dndz_lotss, newzs=dndz_qso_matched[0], spline_k=5, smooth=0.03)
        dndz_lotss = redshift_helper.fill_in_coarse_dndz(dndz_lotss, dndz_qso_matched[0])

        qsoz.remove_column('CHI'), randz.remove_column('CHI')
        hm = hm_calcs.halomodel(dndz1=dndz_qso_matched, dndz2=dndz_lotss)
        hm.set_powspec(bias1=autofit['b'])
        qsomods.append((np.logspace(-3, 0, 100), hm.get_ang_cf(np.logspace(-3, 0, 100))))

        xcf = twoPointCFs.crosscorr_cats(scales=theta_scales, datcat1=qsoz, datcat2=lotss_bright,
                                         randcat1=randz, nbootstrap=500, estimator='Peebles')
        xfit = clustering_fit.fit_xcf(dndz_lotss, xcf, dndz_qso_matched, autocf, model='mass')
        cflist.append(xcf)
        radmasses.append(xfit[0])
        raderrs.append(xfit[1])
        qsomasses.append(autofit['M'])
        qsoerrs.append(autofit['sigM'])



    qsomasses, qsoerrs = np.array(qsomasses), np.array(qsoerrs)

    rg_autocf = autocorr_rgs(fcut=fcut)
    rgautofit = clustering_fit.fit_pipeline(dndz_lotss, rg_autocf)



    plt.figure(figsize=(8, 7))
    plt.scatter(z_centers, radmasses, c='firebrick', label=r'HzRGs $\times$ eBOSS QSOs')
    plt.errorbar(z_centers, radmasses, yerr=raderrs, ecolor='firebrick', fmt='none')
    plt.fill_between(z_centers, qsomasses - qsoerrs, qsomasses + qsoerrs, alpha=0.5, color='royalblue')
    plt.text(1.2, 12.5, 'eBOSS QSOs', color='royalblue')
    from halomodelpy import halo_growthrate
    zgrid, ms = halo_growthrate.evolve_halo_mass(12.5, 2.5, 0.01)
    plt.plot(zgrid, ms, c='k', ls='--', alpha=0.3)
    zgrid, ms = halo_growthrate.evolve_halo_mass(13., 2.5, 0.01)
    plt.plot(zgrid, ms, c='k', ls='--', alpha=0.3)


    plt.scatter(1.6, rgautofit['M'], c='k')
    plt.errorbar(1.6, rgautofit['M'], yerr=rgautofit['sigM'], ecolor='k', fmt='none', label='HzRG auto')

    elgzs, elgm, elgmerr, rgmass, rgerr = desi_elg_xcorr(rpscales)
    plt.scatter(elgzs, elgm, c='b')
    plt.errorbar(elgzs, elgm, yerr=elgmerr, ecolor='b', fmt='none')

    plt.scatter(elgzs, rgmass, c='none', edgecolors='firebrick')
    plt.errorbar(elgzs, rgmass, yerr=rgerr, ecolor='firebrick', fmt='none')
    plt.text(1, 12., 'DESI ELGs', color='b')


    plt.xlabel('Redshift')
    plt.legend()
    plt.ylabel(r'log$_{10}(M_h / h^{-1} M_{\odot}$)')

    plt.savefig(plotdir + 'lotss_tomo.pdf')
    plt.close('all')


    plt.figure(figsize=(8,7))
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
    plt.close('all')

    return cflist, qsomods







def lens_hiz(nside=1024, fcut=2.):
    from mocpy import MOC
    import healpy as hp
    import astropy.units as u
    from lensing_Helper import measure_lensing_xcorr
    import sample
    lotss = sample.lotss_rg_sample(fcut=fcut)

    lotssmoc = MOC.from_fits('../data/radio_cats/LOTSS_DR2/lotss_dr2_hips_moc.fits')
    mask = np.zeros(hp.nside2npix(nside))
    l, b = hp.pix2ang(nside, np.arange(len(mask)), lonlat=True)
    ra, dec = coordhelper.galactic_to_equatorial(l, b)
    idx = lotssmoc.contains(ra*u.deg, dec*u.deg)
    mask[idx] = 1.
    #mask = np.ones(hp.nside2npix(nside))

    corr = measure_lensing_xcorr.measure_planck_xcorr(np.logspace(2, 3, 10), (lotss['RA'], lotss['DEC']), nside, mask=mask, accurate=True)

    dndz = redshift_helper.dndz_from_z_list(sample.redshift_dist(lotss, 5), 10, zrange=(0.01, 4.))
    fit = lensing_fit.xcorr_fit_pipeline(dndz, corr)
    fig = fit.pop('plot')
    fig.savefig(plotdir + 'hiz_rg_lensing.pdf')
