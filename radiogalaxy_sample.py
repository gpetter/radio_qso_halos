import numpy as np
from astropy.table import Table, hstack, vstack
import matplotlib.pyplot as plt
import astropy.units as u
import sample
from SkyTools import coordhelper, fluxutils
import pandas as pd
hic = 'firebrick'
#midc = '#ff7400'
midc = '#ff4d00'
lowc = '#ffc100'

plotdir = '/home/graysonpetter/Dropbox/radioplots/'
def wisediagram(fcut=2.):

    lotss = Table.read('../data/radio_cats/LOTSS_DR2/LOTSS_DR2_2mass_cw.fits')
    lotss = lotss[np.where(lotss['Total_flux'] > fcut)]
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

    plt.scatter(lowz['W2_cw'], lowz['W1_cw'] - lowz['W2_cw'], s=20, c='none', edgecolors=lowc, marker='o')

    plt.scatter(midz['W2_cw'], midz['W1_cw'] - midz['W2_cw'], s=20, c='none', edgecolors=midc, marker='o')
    plt.scatter(hiz['W2_cw'], hiz['W1_cw'] - hiz['W2_cw'], s=20, c='none', edgecolors=hic, marker='o')
    plt.scatter(star['W2_cw'], star['W1_cw'] - star['W2_cw'], s=10, c='blue', marker='*', edgecolors='none')
    plt.scatter(0, 0, s=100, c='b', marker='*', label="SF-only", edgecolors='none')
    plt.plot(np.linspace(10, 13.86, 5), 0.65 * np.ones(5), ls='dotted', c='grey', alpha=0.5)
    plt.plot(np.linspace(13.86, 20, 100), 0.65 * np.exp(0.153 * np.square(np.linspace(13.86, 20, 100) - 13.86)), c='grey',
             ls='dotted', alpha=0.5)

    #plt.plot(np.linspace(10, 13.07, 5), 0.486 * np.ones(5), ls='dotted', c='grey', alpha=0.5)
    #plt.plot(np.linspace(13.07, 20, 100), 0.486 * np.exp(0.092 * np.square(np.linspace(13.07, 20, 100) - 13.07)),
    #         c='grey', ls='dotted', alpha=0.5)
    if fcut == 5.:
        plt.plot(np.linspace(12, 15.97, 10), (np.linspace(12, 15.97, 10) - 17)/3+0.75, c=midc, ls='dashed')

    plt.text(17.6, 1.6, r'W2 90$\%$ Completeness', rotation=90, fontsize=8)
    plt.ylim(-0.5, 2.4)
    plt.text(16.15, 1.8, 'R90 AGN', fontsize=10, rotation=73, color='grey')
    plt.text(11.5, 0.4, r'$z < 0.5$', color=lowc, fontsize=20)
    plt.text(14.2, -0.4, r'$0.5 < z < 1$', color=midc, fontsize=20)
    plt.text(18, 1., r'$z > 1$', color=hic, fontsize=20)
    plt.xlim(11.2, 18.5)
    plt.text(11.2, 1.35, '(17-W2)/4+0.15', rotation=-28, color=hic)
    plt.plot(np.linspace(10, 17.5, 100), (17 - np.linspace(10, 17.5, 100))/4 + 0.15, c=hic, ls='--', alpha=0.8)
    plt.axvline(17.5, ls='--', c='k', alpha=0.8)
    plt.xlabel('W2 [Vega mag]')
    plt.ylabel(r'W1 $-$ W2 [Vega mag]')
    plt.legend(fontsize=10, loc='lower left')
    ax.text(0.05, 0.93, r'$S_{150 \ \mathrm{MHz}} >$ %s mJy' % (int(fcut)), transform=ax.transAxes, fontsize=20)
    plt.savefig(plotdir + 'wise_diagram%s.pdf' % int(fcut))
    plt.close('all')

def wisediagram_both():
    fcuts = [2., 5.]

    lotss = Table.read('../data/radio_cats/LOTSS_DR2/LOTSS_DR2_2mass_cw.fits')
    lotss = lotss[np.where(lotss['Total_flux'] > 2.)]
    lotss = lotss[np.where(lotss['sep_cw'] < 7.)]
    bootes = Table.read('../data/radio_cats/LoTSS_deep/classified/bootes.fits')

    bootidx, lotidx = coordhelper.match_coords((bootes['RA'], bootes['DEC']), (lotss['RA'], lotss['DEC']), 5.,
                                               symmetric=False)
    lotss = lotss[lotidx]
    bootes = bootes[bootidx]
    comb = hstack([lotss, bootes])
    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(8, 14))
    plt.xlim(11.2, 18.6)
    axs[1].set_xlabel('W2 [Vega mag]')

    outalpha = 0.3
    axs[0].text(15, 1.8, 'HzRGs', fontsize=20, color=hic)
    axs[0].text(15.2, 1.6, '$z>1$', fontsize=20, color=hic)

    starc = 'royalblue'



    for j in range(2):
        comb_b = comb[np.where(comb['Total_flux'] > fcuts[j])]

        if j == 0:
            inbox = np.where(((comb_b['W1_cw'] - comb_b['W2_cw']) > ((17 - comb_b['W2_cw']) / 4. + 0.15)) & (
                        comb_b['W2_cw'] < 17.5))[0]
        elif j == 1:
            #inbox = np.where(((comb_b['W1_cw'] - comb_b['W2_cw']) < ((17 - comb_b['W2_cw']) / 4. + 0.15)) & (
            #            comb_b['W2_cw'] < 17.5) & (
            #                             (comb_b['W1_cw'] - comb_b['W2_cw']) < ((comb_b['W2_cw'] - 17) / 3. + 0.75)))[0]
            inbox = np.where(
                (
                                     ((comb_b['W1_cw'] - comb_b['W2_cw']) < ((17 - comb_b['W2_cw']) / 4. + 0.15)) & (
                            comb_b['W2_cw'] < 17.5) & (
                                             (comb_b['W1_cw'] - comb_b['W2_cw']) < ((comb_b['W2_cw'] - 17) / 3. + 0.75))
                )
                |
                (
                    ((comb_b['W1_cw'] - comb_b['W2_cw']) < ((17 - comb_b['W2_cw']) / 4. - 0.15))
                    &
                    ((comb_b['W1_cw'] - comb_b['W2_cw']) < ((comb_b['W2_cw'] - 17) / 3. + 1.15))
                )
                )[0]
        else:
            inbox = np.where(((comb_b['W1_cw'] - comb_b['W2_cw']) < ((17 - comb_b['W2_cw']) / 4. - 0.15)) & (
                        comb_b['W2_cw'] < 17.5) & (comb_b['W1_cw'] - comb_b['W2_cw']) > (
                                         (comb_b['W2_cw'] - 17) / 3. + 0.65) & (comb_b['W1_cw'] - comb_b['W2_cw']) < (
                                         (comb_b['W2_cw'] - 17) / 3. + 1.15))[0]
        outbox = np.logical_not(np.in1d(np.arange(len(comb_b)), inbox))

        combbox = comb_b[inbox]
        combout = comb_b[outbox]

        lowz = combbox[np.where(combbox['z_best'] < 0.5)]
        star = combbox[np.where(combbox['Overall_class'] == 'SFG')]
        midz = combbox[np.where((combbox['z_best'] > 0.5) & (combbox['z_best'] < 1.))]
        hiz = combbox[np.where(combbox['z_best'] > 1)]

        axs[j].scatter(lowz['W2_cw'], lowz['W1_cw'] - lowz['W2_cw'], s=20, c='none', edgecolors=lowc,
                       marker='o')

        axs[j].scatter(midz['W2_cw'], midz['W1_cw'] - midz['W2_cw'], s=20, c='none', edgecolors=midc, marker='o')
        axs[j].scatter(hiz['W2_cw'], hiz['W1_cw'] - hiz['W2_cw'], s=20, c='none', edgecolors=hic, marker='o')
        axs[j].scatter(star['W2_cw'], star['W1_cw'] - star['W2_cw'], s=15, c=starc, marker='*', edgecolors='none')
        axs[j].scatter(0, 0, s=100, c=starc, marker='*', label="SF-only", edgecolors='none')

        lowz = combout[np.where(combout['z_best'] < 0.5)]
        star = combout[np.where(combout['Overall_class'] == 'SFG')]
        midz = combout[np.where((combout['z_best'] > 0.5) & (combout['z_best'] < 1.))]
        hiz = combout[np.where(combout['z_best'] > 1)]

        axs[j].scatter(lowz['W2_cw'], lowz['W1_cw'] - lowz['W2_cw'], s=20, c='none', edgecolors=lowc,
                       marker='o', alpha=outalpha)

        axs[j].scatter(midz['W2_cw'], midz['W1_cw'] - midz['W2_cw'], s=20, c='none', edgecolors=midc, marker='o',
                       alpha=outalpha)
        axs[j].scatter(hiz['W2_cw'], hiz['W1_cw'] - hiz['W2_cw'], s=20, c='none', edgecolors=hic, marker='o',
                       alpha=outalpha)
        axs[j].scatter(star['W2_cw'], star['W1_cw'] - star['W2_cw'], s=15, c=starc, marker='*', edgecolors='none',
                       alpha=outalpha)

        axs[j].plot(np.linspace(10, 13.86, 5), 0.65 * np.ones(5), ls='dotted', c='grey', alpha=0.5)
        axs[j].plot(np.linspace(13.86, 20, 100), 0.65 * np.exp(0.153 * np.square(np.linspace(13.86, 20, 100) - 13.86)),
                    c='grey', ls='dotted', alpha=0.5)
        if j == 1:
            plt.plot(np.linspace(12, 15.97, 10), (np.linspace(12, 15.97, 10) - 17) / 3 + 0.75, c=midc, ls='dashed')
            plt.plot(np.linspace(12, 14.77, 10), (np.linspace(12, 14.77, 10) - 17) / 3 + 1.15, c=lowc, ls='dashed')
            plt.plot(np.linspace(14.77, 15.62, 10), (17 - np.linspace(14.77, 15.62, 10)) / 4 - 0.15, c=lowc,
                     ls='dashed')
            plt.plot(np.linspace(12, 15.62, 10), (np.linspace(12, 15.62, 10) - 17) / 3 + 0.65, c=lowc, ls='dashed')


        axs[j].set_ylim(-0.5, 2.4)
        if j == 0:
            axs[j].text(17.6, 1.6, r'W2 90$\%$ Completeness', rotation=90, fontsize=8)
            axs[j].text(16.3, 1.9, 'R90 AGN', fontsize=10, rotation=75, color='grey')
            axs[j].text(11.5, 0.4, r'$z < 0.5$', color=lowc, fontsize=20)
            axs[j].text(14.2, -0.4, r'$0.5 < z < 1$', color=midc, fontsize=20)

            axs[j].legend(fontsize=15, loc='lower left')
            axs[0].arrow(16.72, 2.3, -0.2, 0.03, facecolor='grey', width=0.015, alpha=0.5, edgecolor='none')

        # axs[j].text(11.2, 1.35, '(17-W2)/4+0.15', rotation=-28, color=hic)
        axs[j].plot(np.linspace(10, 17.5, 100), (17 - np.linspace(10, 17.5, 100)) / 4 + 0.15, c=hic, ls='--', alpha=0.8)
        axs[j].axvline(17.5, ls='--', c='k', alpha=0.4)

        axs[j].set_ylabel(r'W1 $-$ W2 [Vega mag]')

        axs[j].text(0.05, 0.93, r'$S_{150 \ \mathrm{MHz}} >$ %s mJy' % (int(fcuts[j])), transform=axs[j].transAxes,
                    fontsize=20)
    # axs[1].text(13.1, -.45, '(W2-17)/3+0.75', rotation=40, color=midc)
    axs[1].text(13.1, -.35, 'LzRGs', rotation=40, color=lowc, fontsize=20)
    axs[1].text(15, -.35, 'IzRGs', rotation=0, color=midc, fontsize=20)
    plt.subplots_adjust(hspace=0)
    plt.savefig(plotdir + 'wise_diagram.pdf')
    plt.close('all')

def hostgals(izrgs=False):
    lotz = sample.hzrg_sample()

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
    plt.figure(figsize=(8,7))

    s = 15
    plt.scatter(herg['z_best'], herg['SFR_cons'] - herg['Mass_cons'], s=s, c='none', label='HERG', edgecolors='goldenrod', marker='D')
    plt.scatter(lerg['z_best'], lerg['SFR_cons'] - lerg['Mass_cons'], s=s, c='none', label='LERG',
                edgecolors='tomato')
    plt.scatter(sfg['z_best'], sfg['SFR_cons'] - sfg['Mass_cons'], s=s, c='none', label='SFG', marker='*',
                edgecolors='cornflowerblue')
    plt.scatter(agn['z_best'], agn['SFR_cons'] - agn['Mass_cons'], s=s, c='none', label='RQAGN', marker='s',
                edgecolors='darkgreen')
    plt.plot(np.linspace(0, 4, 100), np.log10(.2 / cos.age(np.linspace(0, 4, 100)).to('yr').value), c='k', ls='--')
    plt.arrow(3, np.log10(.2 / cos.age(3).to('yr').value), 0, -.2, head_width=.05, color='k')
    plt.text(2.5, -10.6, 'Quenched', fontsize=15)
    plt.ylabel('log sSFR (yr$^{-1}$)')
    plt.xlim(0, 3.5)


    if izrgs:
        lotz = sample.izrg_sample()
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

        s = 15
        plt.scatter(herg['z_best'], herg['SFR_cons'] - herg['Mass_cons'], s=s, c='none',
                    edgecolors='goldenrod', marker='D', alpha=0.3)
        plt.scatter(lerg['z_best'], lerg['SFR_cons'] - lerg['Mass_cons'], s=s, c='none', label='IzLERGs',
                    edgecolors='tomato', alpha=0.3)
        plt.scatter(sfg['z_best'], sfg['SFR_cons'] - sfg['Mass_cons'], s=s, c='none', marker='*',
                    edgecolors='cornflowerblue', alpha=0.3)
        plt.scatter(agn['z_best'], agn['SFR_cons'] - agn['Mass_cons'], s=s, c='none', marker='s',
                    edgecolors='darkgreen', alpha=0.3)




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

def both_redshift_dist(nbins=30, ndraws=100):
    zrange = (0.1, 4)

    rgs = sample.hzrg_sample()
    hibootes = sample.match2bootes(rgs, sep=3)
    hispecbootes = hibootes[np.where(hibootes['f_zbest'] == 1)]
    hifinalzs = sample.treat_dndz_pdf(hibootes, ndraws)


    rgs = sample.izrg_sample()
    midbootes = sample.match2bootes(rgs, sep=3)
    midspecbootes = midbootes[np.where(midbootes['f_zbest'] == 1)]
    midfinalzs = sample.treat_dndz_pdf(midbootes, ndraws)

    lzrg = sample.lzrg_sample()
    lowzs = sample.redshift_dist(lzrg, 3., bootesonly=False)

    norm_unity=True


    f, (ax0, ax1) = plt.subplots(figsize=(8,9), nrows=2, ncols=1, gridspec_kw={'height_ratios': [1, 5]}, sharex=True)
    plt.xlim(-0.1, 3.75)



    hist, foo = np.histogram(hifinalzs, bins=nbins, range=zrange)
    normhist, foo = np.histogram(hifinalzs, bins=nbins, range=zrange, density=True)
    if norm_unity:
        weight = 1 / np.max(hist)
    else:
        weight = normhist[0]/hist[0]
    nphot2nspec = len(hibootes) / len(hispecbootes)
    ax1.hist(hifinalzs, range=zrange, bins=nbins, weights=weight*np.ones_like(hifinalzs), histtype='step', edgecolor=hic)
    ax1.hist(hispecbootes['z_best'], range=zrange, bins=nbins, histtype='step', hatch='////',
             weights=ndraws*weight*np.ones(len(hispecbootes)), edgecolor=hic, alpha=0.3)
    ax1.text(1.2, 0.03, 'Spectroscopic', color=hic, fontsize=25)

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

def lum_redshift(izrgs=False, lzrgs=False, hzrgflux=2., izrgflux=5., lzrgflux=20.):
    import seaborn as sns
    s=40

    lotz = sample.hzrg_sample(hzrgflux)

    bootes = Table.read('../data/radio_cats/LoTSS_deep/classified/bootes.fits')
    bootes = bootes[np.where(bootes['z_best'] < 4)]
    bootidx, lotidx = coordhelper.match_coords((bootes['RA'], bootes['DEC']), (lotz['RA'], lotz['DEC']), 3.,
                                               symmetric=False)
    bootes = bootes[bootidx]
    lotz = lotz[lotidx]
    bootes = hstack((bootes, lotz))

    herg = bootes[np.where(bootes['Overall_class'] == "HERG")]
    lerg = bootes[np.where(bootes['Overall_class'] == "LERG")]
    sfg = bootes[np.where(bootes['Overall_class'] == "SFG")]
    agn = bootes[np.where(bootes['Overall_class'] == "RQAGN")]

    zspace = np.linspace(1., 4, 100)
    herglums = np.log10(
        fluxutils.luminosity_at_rest_nu(np.array(herg['Total_flux']), -0.7, .144, .15, herg['z_best'], flux_unit=u.mJy,
                                        energy=False))
    lerglums = np.log10(
        fluxutils.luminosity_at_rest_nu(np.array(lerg['Total_flux']), -0.7, .144, .15, lerg['z_best'], flux_unit=u.mJy,
                                        energy=False))
    sfglums = np.log10(
        fluxutils.luminosity_at_rest_nu(np.array(sfg['Total_flux']), -0.7, .144, .15, sfg['z_best'], flux_unit=u.mJy,
                                        energy=False))
    agnlums = np.log10(
        fluxutils.luminosity_at_rest_nu(np.array(agn['Total_flux']), -0.7, .144, .15, agn['z_best'], flux_unit=u.mJy,
                                        energy=False))
    limlum = np.log10(
        fluxutils.luminosity_at_rest_nu(hzrgflux * np.ones_like(zspace), -0.7, .144, .15, zspace, flux_unit=u.mJy,
                                        energy=False))

    fig, (ax2, ax) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 5]}, figsize=(8, 9))
    plt.xlim(0, 2.5)
    plt.xlabel('Redshift')
    ax.plot(zspace, limlum, c=hic, ls='--')
    ax.scatter(herg['z_best'], herglums, s=s, c='none', label='HERG',
               edgecolors=hic, marker='o')
    ax.scatter(lerg['z_best'], lerglums, s=s, c='none', label='LERG',
               edgecolors=hic)
    ax.scatter(sfg['z_best'], sfglums, s=s, c='none', label='SFG', marker='*',
               edgecolors=hic)
    ax.scatter(agn['z_best'], agnlums, s=s, c='none', label='RQAGN',
               edgecolors=hic, marker='s')

    if izrgs:
        lotz = sample.izrg_sample(izrgflux)

        bootes = sample.match2bootes(lotz, 3.)
        bootes = hstack((lotz, bootes))
        bootes = bootes[np.where(bootes['z_best'] < 4)]

        herg = bootes[np.where(bootes['Overall_class'] == "HERG")]
        lerg = bootes[np.where(bootes['Overall_class'] == "LERG")]
        sfg = bootes[np.where(bootes['Overall_class'] == "SFG")]
        agn = bootes[np.where(bootes['Overall_class'] == "RQAGN")]

        zspace = np.linspace(0.1, 4, 100)
        if len(herg) > 0:
            herglums = np.log10(
                fluxutils.luminosity_at_rest_nu(np.array(herg['Total_flux']), -0.7, .144, .15, herg['z_best'],
                                                flux_unit=u.mJy, energy=False))
            ax.scatter(herg['z_best'], herglums, s=s, c='none',
                       edgecolors=midc)
        lerglums = np.log10(
            fluxutils.luminosity_at_rest_nu(np.array(lerg['Total_flux']), -0.7, .144, .15, lerg['z_best'],
                                            flux_unit=u.mJy,
                                            energy=False))
        if len(sfg) > 0:
            sfglums = np.log10(
                fluxutils.luminosity_at_rest_nu(np.array(sfg['Total_flux']), -0.7, .144, .15, sfg['z_best'],
                                                flux_unit=u.mJy,
                                                energy=False))
            ax.scatter(sfg['z_best'], sfglums, s=s, c='none', marker='*',
                       edgecolors=midc)
        if len(agn) > 0:
            agnlums = np.log10(
                fluxutils.luminosity_at_rest_nu(np.array(agn['Total_flux']), -0.7, .144, .15, agn['z_best'],
                                                flux_unit=u.mJy,
                                                energy=False))
            ax.scatter(agn['z_best'], agnlums, s=s, c='none',
                       edgecolors=midc)
        zspace = np.linspace(0.5, 1, 20)
        limlum = np.log10(
            fluxutils.luminosity_at_rest_nu(izrgflux * np.ones_like(zspace), -0.7, .144, .15, zspace, flux_unit=u.mJy,
                                            energy=False))
        ax.plot(zspace, limlum, c=midc, ls='--')

        ax.scatter(lerg['z_best'], lerglums, s=s, c='none', label='IzLERGs',
                   edgecolors=midc)

    if lzrgs:
        lotz = sample.lzrg_sample(lzrgflux)

        bootes = sample.match2bootes(lotz, 3.)
        bootes = hstack((lotz, bootes))
        bootes = bootes[np.where(bootes['z_best'] < 4)]

        herg = bootes[np.where(bootes['Overall_class'] == "HERG")]
        lerg = bootes[np.where(bootes['Overall_class'] == "LERG")]
        sfg = bootes[np.where(bootes['Overall_class'] == "SFG")]
        agn = bootes[np.where(bootes['Overall_class'] == "RQAGN")]

        if len(herg) > 0:
            herglums = np.log10(
                fluxutils.luminosity_at_rest_nu(np.array(herg['Total_flux']), -0.7, .144, .15, herg['z_best'],
                                                flux_unit=u.mJy, energy=False))
            ax.scatter(herg['z_best'], herglums, s=s, c='none',
                       edgecolors=lowc)
        lerglums = np.log10(
            fluxutils.luminosity_at_rest_nu(np.array(lerg['Total_flux']), -0.7, .144, .15, lerg['z_best'],
                                            flux_unit=u.mJy,
                                            energy=False))
        if len(sfg) > 0:
            sfglums = np.log10(
                fluxutils.luminosity_at_rest_nu(np.array(sfg['Total_flux']), -0.7, .144, .15, sfg['z_best'],
                                                flux_unit=u.mJy,
                                                energy=False))
            ax.scatter(sfg['z_best'], sfglums, s=s, c='none', marker='*',
                       edgecolors=lowc)
        if len(agn) > 0:
            agnlums = np.log10(
                fluxutils.luminosity_at_rest_nu(np.array(agn['Total_flux']), -0.7, .144, .15, agn['z_best'],
                                                flux_unit=u.mJy,
                                                energy=False))
            ax.scatter(agn['z_best'], agnlums, s=s, c='none',
                       edgecolors=lowc)
        lowzspace = np.linspace(0.25, 0.5, 30)
        lowlum = np.log10(
            fluxutils.luminosity_at_rest_nu(20. * np.ones_like(lowzspace), -0.7, .144, .15, lowzspace, flux_unit=u.mJy,
                                            energy=False))
        ax.plot(lowzspace, lowlum, c=lowc, ls='dashed')

        ax.scatter(lerg['z_best'], lerglums, s=s, c='none', label='IzLERGs',
                   edgecolors=lowc)

    ax.set_ylim(23.6, 27.9)

    lstars = pd.read_csv('results/kondapally23_agnlf/Lstars.csv')
    ax.fill_between(lstars['z'], lstars['Lstar'] - lstars['loerr'], lstars['Lstar'] + lstars['hierr'], color='purple',
                    alpha=0.1, edgecolor='none')
    ax.text(0.9, 26.45, '$L_{\star}$', fontsize=20, color='purple', alpha=0.4)

    ax.text(0.7, 24, 'HERGs + LERGs', fontsize=15, rotation=32, color='k', alpha=0.4)

    bootes = Table.read('../data/radio_cats/LoTSS_deep/classified/bootes.fits')
    rgs = bootes[np.where((bootes['Overall_class'] == "HERG") | (bootes['Overall_class'] == "LERG"))]
    rgs = rgs[np.where((np.isfinite(rgs['z_best']) & (np.isfinite(rgs['L150']))))]
    sns.kdeplot(x=np.array(rgs['z_best'], dtype=float), y=np.log10(rgs['L150']), levels=4, ax=ax, bw_adjust=0.7,
                color='k', alpha=0.1)

    ax.set_ylabel('log($L_{150 \ \mathrm{MHz}}$ [W/Hz])')

    hzrgs = sample.hzrg_sample(2.)
    izrgs = sample.izrg_sample(5.)
    lzrgs = sample.lzrg_sample(20.)
    hzrgs = sample.match2bootes(hzrgs, 3.)
    izrgs = sample.match2bootes(izrgs, 3.)
    lzrgs = sample.match2bootes(lzrgs, 3.)
    hzrgs = hzrgs[np.where(hzrgs['z_best'] > 0.8)]
    allrgs = vstack((hzrgs, izrgs, lzrgs))

    zbins = np.linspace(0.1, 3., 11)
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
    # plt.scatter(zcenters, hergfracs)
    # plt.scatter(zcenters, lergfracs)
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
lum_redshift(True, True)
#heating_contribution(fcuthi=3.)
#test_selections()