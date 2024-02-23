import numpy as np
from astropy.table import Table, hstack, vstack
import matplotlib.pyplot as plt
import astropy.units as u
import sample
from SkyTools import coordhelper, fluxutils
import pandas as pd
import params
hic = 'firebrick'
#midc = '#ff7400'
midc = '#ff4d00'
lowc = '#ffc100'

plotdir = '/home/graysonpetter/Dropbox/radioplots/'


def wisediagram():
    """
    Plot radio sources in WISE color space to illustrate how HzRG cut works
    """

    starc = 'royalblue'
    outalpha = 0.5

    # select bright lotss sample
    lotss = Table.read('catalogs/LoTSS.fits')
    lotss = lotss[np.where((lotss['Total_flux'] > params.hzrg_fluxcut) & (lotss['Total_flux'] < params.hzrg_maxflux))]

    # match to Bootes deep field for "truth" redshifts and properties
    bootes = sample.match2bootes(lotss, params.supercede_cw_sep, stack=True)
    # keep track of entries to see which don't pass HzRG cut
    bootes['INDEX'] = np.arange(len(bootes))
    #fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(8, 14))

    # set up figure
    fig, ax = plt.subplots(figsize=(8,7))
    plt.xlim(11.2, 18.6)
    ax.set_xlabel('W2 [Vega mag]')
    ax.set_ylabel(r'W1 $-$ W2 [Vega mag]')
    ax.text(0.05, 0.93, r'$S_{150 \ \mathrm{MHz}} >$ %s mJy' % (params.hzrg_fluxcut), transform=ax.transAxes,
            fontsize=20)
    #ax.text(16.3, 1.8, 'HzRGs', fontsize=20, color=hic)
    ax.text(16.65, 1.7, '$z>1$', fontsize=20, color=hic)
    ax.text(params.w2faint + .15, 1.6, r'W2 limit', rotation=90, fontsize=12)
    ax.text(16.3, 1.9, 'R90 AGN', fontsize=10, rotation=75, color='grey')
    ax.text(11.5, 0.4, r'$z < 0.5$', color=lowc, fontsize=20)
    ax.text(14.2, -0.4, r'$0.5 < z < 1$', color=midc, fontsize=20)
    ax.legend(fontsize=15, loc='lower left')
    ax.arrow(16.72, 2.3, -0.2, 0.03, facecolor='grey', width=0.015, alpha=0.5, edgecolor='none')

    # cut
    hzrg = sample.hzrg_cut(bootes)
    not_hzrg = bootes[np.where(np.logical_not(np.in1d(bootes['INDEX'], hzrg['INDEX'])))]
    not_hzrg = sample.supercede_catwise(not_hzrg)
    lowz = hzrg[np.where((hzrg['z_best'] < 0.5) & (hzrg['z_best'] > 0))]
    star = hzrg[np.where(hzrg['Overall_class'] == 'SFG')]
    midz = hzrg[np.where((hzrg['z_best'] > 0.5) & (hzrg['z_best'] < 1.))]
    hiz = hzrg[np.where((hzrg['z_best'] > 1) & (hzrg['z_best'] < 3))]


    ax.scatter(lowz['W2'], lowz['W1'] - lowz['W2'], s=20, c='none', edgecolors=lowc,
                   marker='o')
    ax.scatter(midz['W2'], midz['W1'] - midz['W2'], s=20, c='none', edgecolors=midc, marker='o')
    ax.scatter(hiz['W2'], hiz['W1'] - hiz['W2'], s=20, c='none', edgecolors=hic, marker='o')
    ax.scatter(star['W2'], star['W1'] - star['W2'], s=15, c=starc, marker='*', edgecolors='none')
    ax.scatter(0, 0, s=100, c=starc, marker='*', label="SF-only", edgecolors='none')


    lowz = not_hzrg[np.where((not_hzrg['z_best'] < 0.5) & (not_hzrg['z_best'] > 0))]
    star = not_hzrg[np.where(not_hzrg['Overall_class'] == 'SFG')]
    midz = not_hzrg[np.where((not_hzrg['z_best'] > 0.5) & (not_hzrg['z_best'] < 1.))]
    hiz = not_hzrg[np.where((not_hzrg['z_best'] > 1) & (not_hzrg['z_best'] < 3))]

    ax.scatter(lowz['W2'], lowz['W1'] - lowz['W2'], s=20, c='none', edgecolors=lowc,
                   marker='o', alpha=outalpha)
    ax.scatter(midz['W2'], midz['W1'] - midz['W2'], s=20, c='none', edgecolors=midc, marker='o', alpha=outalpha)
    ax.scatter(hiz['W2'], hiz['W1'] - hiz['W2'], s=20, c='none', edgecolors=hic, marker='o', alpha=outalpha)
    ax.scatter(star['W2'], star['W1'] - star['W2'], s=15, c=starc, marker='*', edgecolors='none', alpha=outalpha)
    ax.scatter(0, 0, s=100, c=starc, marker='*', edgecolors='none', alpha=outalpha)





    ax.plot(np.linspace(10, 13.86, 5), 0.65 * np.ones(5), ls='dotted', c='grey', alpha=0.5)
    ax.plot(np.linspace(13.86, 20, 100), 0.65 * np.exp(0.153 * np.square(np.linspace(13.86, 20, 100) - 13.86)),
                c='grey', ls='dotted', alpha=0.5)

    ax.set_ylim(-0.5, 2.4)
    w2space = np.linspace(10, params.w2faint, 100)
    ax.plot(w2space, params.hzrg_cut_eqn(w2space), c=hic, ls='--', alpha=0.8)
    ax.axvline(params.w2faint, ls='--', c='k', alpha=0.4)




    # axs[1].text(13.1, -.45, '(W2-17)/3+0.75', rotation=40, color=midc)
    #axs[1].text(13.1, -.35, 'LzRGs', rotation=40, color=lowc, fontsize=20)
    #axs[1].text(15, -.35, 'IzRGs', rotation=0, color=midc, fontsize=20)
    #plt.subplots_adjust(hspace=0)
    plt.savefig(plotdir + 'wise_diagram.pdf')
    plt.close('all')

def hostgals():
    from halomodelpy import params as hmparams
    pob = hmparams.param_obj()
    cos = pob.apcosmo
    lotz = Table.read('catalogs/LoTSS.fits')

    bootes = Table.read('../data/radio_cats/LoTSS_deep/classified/bootes.fits')
    bootes = bootes[np.where(bootes['z_best'] < 4)]
    bootes, lotz = coordhelper.match_coords(bootes, lotz, 3., symmetric=False)
    bootes = bootes['Overall_class', 'SFR_cons', 'Mass_cons', 'z_best']
    bootes = hstack((lotz, bootes))

    bootes = bootes[np.where((bootes['SFR_cons'] > -3) & (bootes['Mass_cons'] > 0))]
    plt.figure(figsize=(8, 7))
    s = 15
    samps = ['lzrg', 'izrg', 'hzrg']
    labels = ['HERG', 'LERG', 'SFG', 'RQAGN']
    for j in range(len(samps)):
        if samps[j] == 'lzrg':
            tab = sample.lzrg_cut(bootes)
        elif samps[j] == 'izrg':
            tab = sample.izrg_cut(bootes)
        elif samps[j] == 'hzrg':
            tab = sample.hzrg_cut(bootes)
        else:
            tab = None
        herg = tab[np.where(tab['Overall_class'] == "HERG")]
        lerg = tab[np.where(tab['Overall_class'] == "LERG")]
        sfg = tab[np.where(tab['Overall_class'] == "SFG")]
        agn = tab[np.where(tab['Overall_class'] == "RQAGN")]


        if j == 0:
            label = labels
        else:
            label = [None, None, None, None]


        plt.scatter(herg['z_best'], herg['SFR_cons'] - herg['Mass_cons'], s=s, c='none', label=label[0],
                    edgecolors='goldenrod', marker='D')
        plt.scatter(lerg['z_best'], lerg['SFR_cons'] - lerg['Mass_cons'], s=s, c='none', label=label[1],
                    edgecolors='tomato')
        plt.scatter(sfg['z_best'], sfg['SFR_cons'] - sfg['Mass_cons'], s=s, c='none', label=label[2], marker='*',
                    edgecolors='cornflowerblue')
        plt.scatter(agn['z_best'], agn['SFR_cons'] - agn['Mass_cons'], s=s, c='none', label=label[3], marker='s',
                    edgecolors='darkgreen')
    plt.plot(np.linspace(0, 4, 100), np.log10(.2 / cos.age(np.linspace(0, 4, 100)).to('yr').value), c='k', ls='--')
    plt.arrow(3, np.log10(.2 / cos.age(3).to('yr').value), 0, -.2, head_width=.05, color='k')
    plt.text(2.5, -10.6, 'Quenched', fontsize=15)
    plt.ylabel('log sSFR (yr$^{-1}$)')
    plt.xlim(0, 3.5)


    plt.axvline(1.8, c='grey', ls='dotted', alpha=0.6)
    plt.text(1.3, -11.6, r'Reliable $M_{\star}$', fontsize=10, color='grey')
    plt.arrow(1.8, -11.7, -.5, 0, facecolor='none', alpha=0.6, head_width=0.05, edgecolor='grey')
    plt.legend(fontsize=10)
    plt.xlabel('Redshift')
    plt.savefig(plotdir + 'sSFR.pdf')
    plt.close('all')




def hzrg_redshift_dist(nbins=20, ndraws=100):
    rgs = sample.hzrg_sample(2)
    zrange = (0.1, 4)

    f, (ax0, ax1) = plt.subplots(figsize=(8,9), nrows=2, ncols=1, gridspec_kw={'height_ratios': [1, 5]}, sharex=True)

    #zs = sample.redshift_dist(rgs, 5.)
    bootes = sample.match2bootes(rgs, sep=3)
    specbootes = bootes[np.where(bootes['f_zbest'] == 1)]
    #spl_dndz = sample.spline_dndz(rgs, 5., zrange=zrange, n_newzs=100, spline_k=5, smooth=0.03)

    #plt.hist(zs, density=True, histtype='step', range=zrange, bins=15)
    #plt.plot(spl_dndz[0], spl_dndz[1], c='k', ls='--')
    finalzs = sample.treat_dndz_pdf(bootes, ndraws)
    hist, foo = np.histogram(finalzs, bins=nbins, range=zrange)
    normhist, foo = np.histogram(finalzs, bins=nbins, range=zrange, density=True)
    weight = normhist[0]/hist[0]
    nphot2nspec = len(bootes) / len(specbootes)
    ax1.hist(finalzs, range=zrange, bins=nbins, weights=weight*np.ones_like(finalzs), histtype='step', edgecolor='k')
    ax1.hist(specbootes['z_best'], range=zrange, bins=nbins, histtype='step', hatch='////',
             weights=ndraws*weight*np.ones(len(specbootes)), edgecolor='firebrick', alpha=0.3)
    ax1.text(1, 0.03, 'Spectroscopic', color=hic, fontsize=25)
    plt.xlabel('Redshift')
    plt.ylabel('Redshift distribution')

    zs, tomo_dndz, tomo_err = sample.tomographer_dndz()
    ax0.scatter(zs, tomo_dndz, c='k')
    ax0.errorbar(zs, tomo_dndz, yerr=tomo_err, ecolor='k', fmt='none')
    ax0.text(2.3, 0.8, r'Tomographer ($b \propto 1/D(z))$', fontsize=15)
    #ax0.hist(finalzs, range=zrange, bins=nbins, weights=weight*finalweights, histtype='step', edgecolor='k')
    #ax0.axhline(0, ls='--', c='k')

    plt.subplots_adjust(hspace=0)
    plt.savefig(plotdir + 'hzrg_dndz.pdf')
    plt.close('all')

def izrg_redshift_dist(nbins=5, ndraws=100):
    rgs = sample.izrg_sample()
    zrange = (0.3, 1.2)

    f, ax = plt.subplots(figsize=(8,7))

    bootes = sample.match2bootes(rgs, sep=3)
    specbootes = bootes[np.where(bootes['f_zbest'] == 1)]

    finalzs = sample.treat_dndz_pdf(bootes, ndraws)
    hist, foo = np.histogram(finalzs, bins=nbins, range=zrange)
    normhist, foo = np.histogram(finalzs, bins=nbins, range=zrange, density=True)
    weight = normhist[0]/hist[0]
    ax.hist(finalzs, range=zrange, bins=nbins, weights=weight*np.ones_like(finalzs), histtype='step', edgecolor='k')
    ax.hist(specbootes['z_best'], range=zrange, bins=nbins, histtype='step', hatch='////',
             weights=ndraws*weight*np.ones(len(specbootes)), edgecolor='firebrick', alpha=0.3)
    ax.text(0.5, 0.05, 'Spectroscopic', color=midc, fontsize=25)
    plt.xlabel('Redshift')
    plt.ylabel('Redshift distribution')

    plt.subplots_adjust(hspace=0)
    plt.savefig(plotdir + 'izrg_dndz.pdf')
    plt.close('all')

def both_redshift_dist(nbins=30, ndraws=100, bootesonly=False, showspec=False):
    zrange = (0.1, 4)
    norm_unity = True

    hzrgs = sample.hzrg_sample()
    hifinalzs = sample.treat_dndz_pdf(hzrgs, ndraws, bootesonly=bootesonly)


    izrgs = sample.izrg_sample()
    midfinalzs = sample.treat_dndz_pdf(izrgs, ndraws, bootesonly=bootesonly)

    lzrg = sample.lzrg_sample()
    lowzs = sample.redshift_dist(lzrg, 3., bootesonly=False)

    f, (ax0, ax1) = plt.subplots(figsize=(8,9), nrows=2, ncols=1, gridspec_kw={'height_ratios': [1, 5]}, sharex=True)
    plt.xlim(-0.1, 3.75)



    hist, foo = np.histogram(hifinalzs, bins=nbins, range=zrange)
    normhist, foo = np.histogram(hifinalzs, bins=nbins, range=zrange, density=True)
    if norm_unity:
        weight = 1 / np.max(hist)
    else:
        weight = normhist[0]/hist[0]

    if showspec:
        hibootes = sample.match2bootes(hzrgs, sep=3)
        hispecbootes = hibootes[np.where(hibootes['f_zbest'] == 1)]
        nphot2nspec = len(hibootes) / len(hispecbootes)

        midbootes = sample.match2bootes(izrgs, sep=3)
        midspecbootes = midbootes[np.where(midbootes['f_zbest'] == 1)]

        ax1.hist(hispecbootes['z_best'], range=zrange, bins=nbins, histtype='step', hatch='////',
                 weights=ndraws * weight * np.ones(len(hispecbootes)), edgecolor=hic, alpha=0.3)
        ax1.text(1.2, 0.03, 'Spectroscopic', color=hic, fontsize=25)

    ax1.hist(hifinalzs, range=zrange, bins=nbins, weights=weight*np.ones_like(hifinalzs),
             histtype='step', edgecolor=hic)



    plt.xlabel('Redshift')
    plt.ylabel('Redshift distribution')

    zs, tomo_dndz, tomo_err = sample.tomographer_dndz('hi')
    maxd = np.max(tomo_dndz)
    tomo_dndz, tomo_err = tomo_dndz / maxd, tomo_err / maxd
    #ax0.errorbar(zs, tomo_dndz, yerr=tomo_err, color=hic, fmt='o')
    ax0.fill_between(zs, tomo_dndz - tomo_err, tomo_dndz + tomo_err, color=hic, alpha=0.3, edgecolor='none')

    ax0.text(2.1, 0.8, r'Tomographer ($b \propto 1/D(z))$', fontsize=15)

    zs, tomo_dndz, tomo_err = sample.tomographer_dndz('mid')
    maxd = np.max(tomo_dndz)
    tomo_dndz, tomo_err = tomo_dndz / maxd, tomo_err / maxd
    #ax0.errorbar(zs, tomo_dndz, yerr=tomo_err, color=midc, fmt='o')
    ax0.fill_between(zs, tomo_dndz - tomo_err, tomo_dndz + tomo_err, color=midc, alpha=0.3, edgecolor='none')

    ax0.axhline(0, color='k', alpha=0.2, ls='dashed')

    zs, tomo_dndz, tomo_err = sample.tomographer_dndz('lo')
    maxd = np.max(tomo_dndz)
    tomo_dndz, tomo_err = tomo_dndz / maxd, tomo_err / maxd
    # ax0.errorbar(zs, tomo_dndz, yerr=tomo_err, color=midc, fmt='o')
    ax0.fill_between(zs, tomo_dndz - tomo_err, tomo_dndz + tomo_err, color=lowc, alpha=0.15, edgecolor='none')

    ax0.axhline(0, color='k', alpha=0.2, ls='dashed')

    #ax0.hist(finalzs, range=zrange, bins=nbins, weights=weight*finalweights, histtype='step', edgecolor='k')
    #ax0.axhline(0, ls='--', c='k')

    hist, foo = np.histogram(midfinalzs, bins=nbins, range=zrange)
    normhist, foo = np.histogram(midfinalzs, bins=nbins, range=zrange, density=True)
    normhist = normhist[np.where(normhist > 0)]
    hist = hist[np.where(hist > 0)]
    if norm_unity:
        weight = 1 / np.max(hist)
    else:
        weight = normhist[0] / hist[0] / (len(hibootes) / len(midbootes))


    ax1.hist(midfinalzs, range=zrange, bins=nbins, histtype='step',
             weights=weight * np.ones_like(midfinalzs),
             ls='dashed', alpha=0.9, edgecolor=midc)
    # ax1.text(0.1, 0.6, 'IzRGs', color='orange', fontsize=20)

    hist, foo = np.histogram(lowzs, bins=nbins, range=zrange)
    normhist, foo = np.histogram(lowzs, bins=nbins, range=zrange, density=True)
    if norm_unity:
        weight = 1 / np.max(hist)
    else:
        weight = normhist[0] / hist[0] / (len(hibootes) / (len(lowzs)/3.))

    ax1.text(0.05, 0.6, 'LzRGs', color=lowc, fontsize=20, rotation=90)
    ax1.text(0.4, 1.05, 'IzRGs', color=midc, fontsize=20)
    ax1.text(1.6, 0.9, 'HzRGs', color=hic, fontsize=20)
    ax1.set_ylim(0, 1.15)

    ax1.hist(lowzs, range=zrange, bins=nbins, histtype='step',
             weights=weight * np.ones_like(lowzs),
             ls='dotted', alpha=0.9, edgecolor=lowc)



    plt.subplots_adjust(hspace=0)
    plt.savefig(plotdir + 'rg_dndz.pdf')
    plt.close('all')

def lum_redshift(sep=3.):
    import seaborn as sns
    s=40

    lotss = Table.read('catalogs/LoTSS.fits')
    lotss = sample.match2bootes(lotss, sep, stack=True)
    lotss.rename_column('L150_1', 'L150')
    lotss["L"] = np.log10(
        fluxutils.luminosity_at_rest_nu(np.array(lotss['Total_flux']), -0.7, .144, .15,
                                        lotss['z_best'], flux_unit=u.mJy,
                                        energy=False))





    fig, (ax2, ax) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 5]}, figsize=(8, 9))
    plt.xlim(0, 2.5)
    plt.xlabel('Redshift')


    def zbin_plot(which):
        if which == 'hzrg':
            cat = sample.hzrg_cut(lotss)
            zspace = np.linspace(1., 4, 100)
            color = hic
            limlum = np.log10(
                fluxutils.luminosity_at_rest_nu(params.hzrg_fluxcut * np.ones_like(zspace),
                                                -0.7, .144, .15, zspace,
                                                flux_unit=u.mJy,
                                                energy=False))
            ax.plot(zspace, limlum, c=color, ls='--')


        elif which == 'izrg':
            cat = sample.izrg_cut(lotss)
            zspace = np.linspace(params.izrg_minzphot, params.izrg_maxzphot, 20)
            color = midc

            #limlum = np.log10(
            #    fluxutils.luminosity_at_rest_nu(izrgflux * np.ones_like(zspace), -0.7, .144, .15, zspace,
            #                                    flux_unit=u.mJy,
            #                                    energy=False))
            limlum = np.ones_like(zspace) * params.lumcut

        else:
            cat = sample.lzrg_cut(lotss)
            zspace = np.linspace(params.lzrg_minzphot, params.lzrg_maxzphot, 20)
            color = lowc
            limlum = np.ones_like(zspace) * params.lumcut

        ax.plot(zspace, limlum, c=color, ls='--')

        herg = cat[np.where(cat['Overall_class'] == "HERG")]
        lerg = cat[np.where(cat['Overall_class'] == "LERG")]
        sfg = cat[np.where(cat['Overall_class'] == "SFG")]
        agn = cat[np.where(cat['Overall_class'] == "RQAGN")]

        ax.scatter(herg['z_best'], herg['L'], s=s, c='none', label='HERG',
                   edgecolors=color, marker='o')
        ax.scatter(lerg['z_best'], lerg['L'], s=s, c='none', label='LERG',
                   edgecolors=color)
        ax.scatter(sfg['z_best'], sfg['L'], s=s, c='none', label='SFG', marker='*',
                   edgecolors=color)
        ax.scatter(agn['z_best'], agn['L'], s=s, c='none', label='RQAGN',
                   edgecolors=color, marker='s')

    zbin_plot('hzrg')
    zbin_plot('izrg')
    zbin_plot('lzrg')

    ax.set_ylim(23.6, 27.9)

    lstars = pd.read_csv('results/kondapally23_agnlf/Lstars.csv')
    ax.fill_between(lstars['z'], lstars['Lstar'] - lstars['loerr'], lstars['Lstar'] + lstars['hierr'], color='purple',
                    alpha=0.1, edgecolor='none')
    ax.text(0.9, 26.45, '$L_{\star}$', fontsize=20, color='purple', alpha=0.4)

    ax.text(0.7, 24, 'HERGs + LERGs', fontsize=15, rotation=32, color='k', alpha=0.4)

    bootes = Table.read('../data/radio_cats/LoTSS_deep/classified/bootes.fits')
    rgs = bootes[np.where((bootes['Overall_class'] == "HERG") | (bootes['Overall_class'] == "LERG"))]
    rgs = rgs[np.where((np.isfinite(rgs['z_best']) & (np.isfinite(rgs['L150']))))]

    sns.kdeplot(x=np.array(rgs['z_best'], dtype=float), y=np.array(rgs['L150'], dtype=float), levels=4, ax=ax, bw_adjust=0.7,
                color='k', alpha=0.1)

    ax.set_ylabel('log($L_{150 \ \mathrm{MHz}}$ [W/Hz])')

    totcat = Table()
    totcat = vstack((totcat, sample.hzrg_cut(lotss)))
    totcat = vstack((totcat, sample.izrg_cut(lotss)))
    allrgs = vstack((totcat, sample.lzrg_cut(lotss)))


    zbins = np.linspace(0.15, 2.75, 8)
    hergfracs = []
    lergfracs = []
    hergerrs = []
    lergerrs = []
    for j in range(len(zbins) - 1):
        inzbin = allrgs[np.where((allrgs['z_best'] > zbins[j]) & (allrgs['z_best'] <= zbins[j + 1]))]
        n_inbin = len(inzbin)
        lergfrac = len(np.where(inzbin['Overall_class'] == 'LERG')[0]) / n_inbin
        hergfrac = len(np.where(inzbin['Overall_class'] == 'HERG')[0]) / n_inbin
        hergfracs.append(hergfrac)
        lergfracs.append(lergfrac)
        hergerrs.append(np.sqrt(hergfrac * (1 - hergfrac) / len(inzbin)))
        lergerrs.append(np.sqrt(lergfrac * (1 - lergfrac) / len(inzbin)))

    zcenters = zbins[:-1] + (zbins[1] - zbins[0]) / 2.
    hergfracs, hergerrs = np.array(hergfracs), np.array(hergerrs)
    lergfracs, lergerrs = np.array(lergfracs), np.array(lergerrs)
    ax2.fill_between(zcenters, hergfracs - hergerrs, hergfracs + hergerrs, alpha=0.5, color='lightseagreen', edgecolor='none')
    ax2.fill_between(zcenters, lergfracs - lergerrs, lergfracs + lergerrs, alpha=0.5, color='green', edgecolor='none')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Fraction")
    ax2.text(0.6, 0.75, 'LERGs', color='green', fontsize=15)
    ax2.text(1.1, 0.25, 'HERGs', color='lightseagreen', fontsize=15)

    plt.subplots_adjust(hspace=0)
    plt.savefig(plotdir + 'lum_z.pdf')
    plt.close('all')


def heating_contribution(fcuthi=2., fcutmid=5.):
    fig, ax = plt.subplots(figsize=(8, 7))
    import matplotlib.ticker as mtick

    heat0 = pd.read_csv('results/kondapally_heating/z05_1_lerg.csv', names=['lum', 'heat'])
    heat1 = pd.read_csv('results/kondapally_heating/z1_15_agn.csv', names=['lum', 'heat'])
    heat2 = pd.read_csv('results/kondapally_heating/z15_2_agn.csv', names=['lum', 'heat'])
    heat3 = pd.read_csv('results/kondapally_heating/z2_25_agn.csv', names=['lum', 'heat'])

    plt.plot(heat0['lum'], heat0['heat'], c='purple')
    plt.plot(heat1['lum'], heat1['heat'], c='cornflowerblue')
    plt.plot(heat2['lum'], heat2['heat'], c='orange')
    plt.plot(heat3['lum'], heat3['heat'], c='firebrick')

    plt.axvline(np.log10(fluxutils.luminosity_at_rest_nu(fcutmid, -0.7, .144, .15, .75, flux_unit=u.mJy, energy=False)),
                color='purple', ls='dotted')
    plt.axvline(np.log10(fluxutils.luminosity_at_rest_nu(fcuthi, -0.7, .144, .15, 1.25, flux_unit=u.mJy, energy=False)),
                color='cornflowerblue', ls='dotted')
    plt.axvline(np.log10(fluxutils.luminosity_at_rest_nu(fcuthi, -0.7, .144, .15, 1.75, flux_unit=u.mJy, energy=False)),
                color='orange', ls='dotted')
    plt.axvline(np.log10(fluxutils.luminosity_at_rest_nu(fcuthi, -0.7, .144, .15, 2.25, flux_unit=u.mJy, energy=False)),
                color='firebrick', ls='dotted')

    plt.text(23, 0.3e32, r'$z=1 - 1.5$', color='cornflowerblue')
    plt.text(27.5, 1e32, r'$z=1.5 - 2$', color='orange')
    plt.text(27.2, 0.5e32, r'$z=2 - 2.5$', color='firebrick')
    plt.text(26.2, 0.3e32, r'$z=0.5 - 1$', color='Purple')

    plt.text(24.8, 0.3e32, r'$L(S_{150}, z)$', rotation=90, c='k')

    plt.ylabel(r'Specific Heating Rate [W Mpc$^{-3}$ log$L^{-1}$]')
    plt.xlabel(r'log$L_{150 \ \mathrm{MHz}}$ [W/Hz]')
    plt.savefig('/home/graysonpetter/Dropbox/radioplots/heating_func.pdf')
    plt.close('all')

def rg_fraction():
    bootes = Table.read('../data/radio_cats/LoTSS_deep/classified/bootes.fits')
    hiz = bootes[np.where((bootes['z_best'] > 1) & (bootes['z_best'] < 3))]
    tot_hzrgs = len(hiz[np.where((hiz['Overall_class'] == "LERG") | (hiz["Overall_class"] == "HERG"))])

    iz = bootes[np.where((bootes['z_best'] > .5) & (bootes['z_best'] < 1))]
    tot_izrgs = len(iz[np.where((iz['Overall_class'] == "LERG") | (iz["Overall_class"] == "HERG"))])




    bright_hzrgs = (hiz[np.where((hiz['S_150MHz'] > 0.002))])
    bright_izrgs = (iz[np.where(iz['S_150MHz'] > 0.005)])

    print('Fraction of z>1 radio galaxies selected by flux cut is: %s' % (len(bright_hzrgs) / tot_hzrgs))
    print('Fraction of 0.5<z<1 radio galaxies selected by flux cut is: %s' % (len(bright_izrgs) / tot_izrgs))

    hisfgs = bright_hzrgs[np.where(bright_hzrgs['Overall_class'] == "SFG")]
    isfgs = bright_izrgs[np.where(bright_izrgs['Overall_class'] == "SFG")]
    print('Fraction of bright HzRGs are actually SFGs: %s' % (len(hisfgs)/len(bright_hzrgs)))
    print('Fraction of bright IzRGs are actually SFGs: %s' % (len(isfgs) / len(bright_izrgs)))

def test_selections():
    hzrg = sample.hzrg_sample()
    izrg = sample.izrg_sample()
    lzrg = sample.lzrg_sample()
    rand = sample.lotss_randoms()

    hzrg.write('catalogs/rgs/hzrg.fits', overwrite=True)
    izrg.write('catalogs/rgs/izrg.fits', overwrite=True)
    lzrg.write('catalogs/rgs/lzrg.fits', overwrite=True)
    rand.write('catalogs/rgs/rand.fits', overwrite=True)

#rg_fraction()
#wisediagram(2.)
#wisediagram_both()
#both_redshift_dist()
#izrg_redshift_dist(7)
#hostgals(True)
#lum_redshift(True, True)
#heating_contribution(fcuthi=3.)
#test_selections()