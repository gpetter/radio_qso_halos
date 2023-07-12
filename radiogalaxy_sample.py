import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import astropy.units as u
import sample
from SkyTools import coordhelper
plotdir = '/home/graysonpetter/Dropbox/radioplots/'
def wisediagram():
    import matplotlib.pyplot as plt
    from SkyTools import coordhelper
    from astropy.table import vstack, hstack
    lotss = Table.read('../data/radio_cats/LOTSS_DR2/LOTSS_DR2_2mass_cw.fits')
    lotss = lotss[np.where(lotss['Total_flux'] > 2.)]
    lotss = lotss[np.where(lotss['sep_cw'] < 5.)]
    bootes = Table.read('../data/radio_cats/LoTSS_deep/classified/bootes.fits')
    bootidx, lotidx = coordhelper.match_coords((bootes['RA'], bootes['DEC']), (lotss['RA'], lotss['DEC']), 5.,
                                               symmetric=False)
    lotss = lotss[lotidx]
    bootes = bootes[bootidx]

    comb = hstack([lotss, bootes])

    lowz = comb[np.where(comb['z_best'] < 0.5)]
    star = comb[np.where(comb['Overall_class'] == 'SFG')]
    midz = comb[np.where((comb['z_best'] > 0.5) & (comb['z_best'] < 1.))]
    hiz = comb[np.where(comb['z_best'] > 1)]

    fig, ax = plt.subplots()

    plt.scatter(lowz['W2_cw'], lowz['W1_cw'] - lowz['W2_cw'], s=20, c='none', edgecolors='cornflowerblue', marker='o')

    plt.scatter(midz['W2_cw'], midz['W1_cw'] - midz['W2_cw'], s=20, c='none', edgecolors='orange', marker='o')
    plt.scatter(hiz['W2_cw'], hiz['W1_cw'] - hiz['W2_cw'], s=20, c='none', edgecolors='firebrick', marker='o')
    plt.scatter(star['W2_cw'], star['W1_cw'] - star['W2_cw'], s=10, c='blue', marker='*', edgecolors='none')
    plt.scatter(0, 0, s=100, c='b', marker='*', label="SF-only", edgecolors='none')
    plt.plot(np.linspace(10, 13.86, 5), 0.65 * np.ones(5), ls='dotted', c='grey', alpha=0.5)
    plt.plot(np.linspace(13.86, 20, 100), 0.65 * np.exp(0.153 * np.square(np.linspace(13.86, 20, 100) - 13.86)), c='grey',
             ls='dotted', alpha=0.5)

    plt.text(17.6, 1.6, r'W2 90$\%$ Completeness', rotation=90, fontsize=8)
    plt.ylim(-0.5, 2.5)
    plt.text(16.15, 1.8, 'R90 AGN', fontsize=10, rotation=73, color='grey')
    plt.text(11.5, 0.4, r'$z < 0.5$', color='cornflowerblue', fontsize=20)
    plt.text(14.2, -0.4, r'$0.5 < z < 1$', color='orange', fontsize=20)
    plt.text(18, 1., r'$z > 1$', color='firebrick', fontsize=20)
    plt.xlim(11, 19)
    plt.text(11.2, 1.35, '(17-W2)/4+0.15', rotation=-28)
    plt.plot(np.linspace(10, 20, 100), (17 - np.linspace(10, 20, 100))/4 + 0.15, c='k', ls='--', alpha=0.8)
    plt.axvline(17.5, ls='--', c='k', alpha=0.8)
    plt.xlabel('W2 [Vega mag]')
    plt.ylabel(r'W1 $-$ W2 [Vega mag]')
    plt.legend(fontsize=10, loc='lower left')
    ax.text(0.05, 0.93, r'$S_{150 \ \mathrm{MHz}} >$ 2 mJy', transform=ax.transAxes, fontsize=20)
    plt.savefig(plotdir + 'wise_diagram.pdf')
    plt.close('all')

def hostgals():
    lotz = sample.lotss_rg_sample()

    bootes = Table.read('../data/radio_cats/LoTSS_deep/classified/bootes.fits')
    bootes = bootes[np.where(bootes['z_best'] < 4)]
    bootidx, lotidx = coordhelper.match_coords((bootes['RA'], bootes['DEC']), (lotz['RA'], lotz['DEC']), 5.,
                                               symmetric=False)
    bootes = bootes[bootidx]

    bootes = bootes[np.where((bootes['SFR_cons'] > -3) & (bootes['Mass_cons'] > 0))]

    herg = bootes[np.where(bootes['Overall_class'] == "HERG")]
    lerg = bootes[np.where(bootes['Overall_class'] == "LERG")]
    sfg = bootes[np.where(bootes['Overall_class'] == "SFG")]
    agn = bootes[np.where(bootes['Overall_class'] == "RQAGN")]

    from halomodelpy import params
    pob = params.param_obj()
    cos = pob.apcosmo


    s = 15
    plt.scatter(herg['z_best'], herg['SFR_cons'] - herg['Mass_cons'], s=s, c='none', label='HERG', edgecolors='orange')
    plt.scatter(lerg['z_best'], lerg['SFR_cons'] - lerg['Mass_cons'], s=s, c='none', label='LERG',
                edgecolors='firebrick')
    plt.scatter(sfg['z_best'], sfg['SFR_cons'] - sfg['Mass_cons'], s=s, c='none', label='SFG', marker='*',
                edgecolors='cornflowerblue')
    plt.scatter(agn['z_best'], agn['SFR_cons'] - agn['Mass_cons'], s=s, c='none', label='RQAGN', marker='s',
                edgecolors='green')
    plt.plot(np.linspace(0, 4, 100), np.log10(.2 / cos.age(np.linspace(0, 4, 100)).to('yr').value), c='k', ls='--')
    plt.arrow(3, np.log10(.2 / cos.age(3).to('yr').value), 0, -.2, head_width=.1, color='k')
    plt.text(2.5, -10.6, 'Quenched', fontsize=15)
    plt.ylabel('log sSFR (yr$^{-1}$)')
    plt.legend(fontsize=10)
    plt.xlabel('Redshift')
    plt.savefig(plotdir + 'sSFR.pdf')
    plt.close('all')

def treat_dndz_pdf(bootes):
    specs = bootes[np.where(bootes['f_zbest'] == 1)]
    phots = bootes[np.where(bootes['f_zbest'] == 0)]

    singlephots = phots[np.where(phots['z2med'] == -99)]
    doublephots = phots[np.where(phots['z2med'] > 0)]
    maxpossiblez = 3.5

    speczs = list(specs['z_best'])
    specweights = list(100. * np.ones(len(specs)))

    singlezs, singleweights = [], []
    for j in range(len(singlephots)):
        thisrow = singlephots[j]
        uperr = (thisrow['z1max'] - thisrow['z1med']) / 1.3
        loerr = (thisrow['z1med'] - thisrow['z1min']) / 1.3
        weight = thisrow['z1area']

        b = np.random.normal(loc=thisrow['z1med'], scale=uperr, size=100)
        above = b[np.where(b > thisrow['z1med'])]
        c = np.random.normal(loc=thisrow['z1med'], scale=loerr, size=100)
        below = c[np.where((c > 0) & (c < thisrow['z1med']))]
        tot = np.concatenate((above, below))
        weights = weight * np.ones_like(tot)
        singlezs += list(tot)
        singleweights += list(weights)

    doublezs1, doublezs2, doubleweights1, doubleweights2 = [], [], [], []
    for j in range(len(doublephots)):
        thisrow = doublephots[j]
        uperr = (thisrow['z1max'] - thisrow['z1med']) / 1.3
        loerr = (thisrow['z1med'] - thisrow['z1min']) / 1.3
        weight = thisrow['z1area']
        if thisrow['z1med'] > maxpossiblez:
            weight = 0.

        b = np.random.normal(loc=thisrow['z1med'], scale=uperr, size=100)
        above = b[np.where(b > thisrow['z1med'])]
        c = np.random.normal(loc=thisrow['z1med'], scale=loerr, size=100)
        below = c[np.where((c > 0) & (c < thisrow['z1med']))]
        tot1 = np.concatenate((above, below))
        weights1 = weight * np.ones_like(tot1)

        uperr = (thisrow['z2max'] - thisrow['z2med']) / 1.3
        loerr = (thisrow['z2med'] - thisrow['z2min']) / 1.3
        weight = thisrow['z2area']

        if thisrow['z2med'] > maxpossiblez:
            weight = 0.

        b = np.random.normal(loc=thisrow['z2med'], scale=uperr, size=100)
        above = b[np.where(b > thisrow['z2med'])]
        c = np.random.normal(loc=thisrow['z2med'], scale=loerr, size=100)
        below = c[np.where((c > 0) & (c < thisrow['z2med']))]
        tot2 = np.concatenate((above, below))
        weights2 = weight * np.ones_like(tot2)
        doublezs1 += list(tot1)
        doublezs2 += list(tot2)
        doubleweights1 += list(weights1)
        doubleweights2 += list(weights2)

    finalzs = np.array(speczs + singlezs + doublezs1 + doublezs2)
    finalweights = np.array(specweights + singleweights + doubleweights1 + doubleweights2) / 100.
    return finalzs, finalweights


def redshift_dist(nbins=20):
    rgs = sample.lotss_rg_sample()
    zrange = (0.1, 4)

    f, (ax0, ax1) = plt.subplots(figsize=(8,9), nrows=2, ncols=1, gridspec_kw={'height_ratios': [1, 5]}, sharex=True)

    #zs = sample.redshift_dist(rgs, 5.)
    bootes = sample.match2bootes(rgs, sep=3)
    specbootes = bootes[np.where(bootes['f_zbest'] == 1)]
    #spl_dndz = sample.spline_dndz(rgs, 5., zrange=zrange, n_newzs=100, spline_k=5, smooth=0.03)

    #plt.hist(zs, density=True, histtype='step', range=zrange, bins=15)
    #plt.plot(spl_dndz[0], spl_dndz[1], c='k', ls='--')
    finalzs, finalweights = treat_dndz_pdf(bootes)
    hist, foo = np.histogram(finalzs, bins=nbins, range=zrange, weights=finalweights)
    normhist, foo = np.histogram(finalzs, bins=nbins, range=zrange, weights=finalweights, density=True)
    weight = normhist[0]/hist[0]
    ax1.hist(finalzs, range=zrange, bins=nbins, weights=weight*finalweights, histtype='step', edgecolor='k')
    ax1.hist(specbootes['z_best'], range=zrange, bins=nbins, histtype='step', hatch='////',
             weights=weight*np.ones(len(specbootes)), edgecolor='firebrick', alpha=0.3)
    ax1.text(1, 0.03, 'Spectroscopic', color='firebrick', fontsize=25)
    plt.xlabel('Redshift')
    plt.ylabel('Redshift distribution')

    zs, tomo_dndz, tomo_err = sample.tomographer_dndz()
    ax0.scatter(zs, tomo_dndz, c='k')
    ax0.errorbar(zs, tomo_dndz, yerr=tomo_err, ecolor='k', fmt='none')
    ax0.text(2.3, 0.8, r'Tomographer ($b \propto 1/D(z))$', fontsize=15)
    plt.subplots_adjust(hspace=0)
    plt.savefig(plotdir + 'rg_dndz.pdf')
    plt.close('all')

def lum_redshift(fluxcut=2):
    from SkyTools import fluxutils
    lotz = sample.lotss_rg_sample(fcut=fluxcut)

    bootes = Table.read('../data/radio_cats/LoTSS_deep/classified/bootes.fits')
    bootes = bootes[np.where(bootes['z_best'] < 4)]
    bootidx, lotidx = coordhelper.match_coords((bootes['RA'], bootes['DEC']), (lotz['RA'], lotz['DEC']), 3.,
                                               symmetric=False)
    bootes = bootes[bootidx]
    lotz = lotz[lotidx]

    zspace = np.linspace(0.1, 4, 100)
    lums = np.log10(fluxutils.luminosity_at_rest_nu(np.array(lotz['Total_flux']), -0.7, .144, .15, bootes['z_best'], flux_unit=u.mJy, energy=False))
    limlum = np.log10(fluxutils.luminosity_at_rest_nu(fluxcut*np.ones_like(zspace), -0.7, .144, .15, zspace, flux_unit=u.mJy, energy=False))

    fig, ax = plt.subplots(figsize=(8,7))
    plt.scatter(bootes['z_best'], lums, edgecolors='k', c='none')
    plt.plot(zspace, limlum, c='k', ls='--')
    plt.xlabel('Redshift')
    plt.ylabel(r'log $L_{150 \ \mathrm{MHz}}$ [W/Hz]')
    plt.ylim(23.5, 28.5)
    plt.xlim(0, 3.5)
    plt.savefig(plotdir + 'lum_z.pdf')
    plt.close('all')

#wisediagram()
#hostgals()
redshift_dist(30)
#lum_redshift()