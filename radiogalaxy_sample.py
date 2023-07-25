import numpy as np
from astropy.table import Table, hstack, vstack
import matplotlib.pyplot as plt
import astropy.units as u
import sample
from SkyTools import coordhelper, fluxutils
import pandas as pd
hic = 'firebrick'
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

    plt.scatter(lowz['W2_cw'], lowz['W1_cw'] - lowz['W2_cw'], s=20, c='none', edgecolors='cornflowerblue', marker='o')

    plt.scatter(midz['W2_cw'], midz['W1_cw'] - midz['W2_cw'], s=20, c='none', edgecolors='orange', marker='o')
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
        plt.plot(np.linspace(12, 15.97, 10), (np.linspace(12, 15.97, 10) - 17)/3+0.75, c='orange', ls='dashed')

    plt.text(17.6, 1.6, r'W2 90$\%$ Completeness', rotation=90, fontsize=8)
    plt.ylim(-0.5, 2.5)
    plt.text(16.15, 1.8, 'R90 AGN', fontsize=10, rotation=73, color='grey')
    plt.text(11.5, 0.4, r'$z < 0.5$', color='cornflowerblue', fontsize=20)
    plt.text(14.2, -0.4, r'$0.5 < z < 1$', color='orange', fontsize=20)
    plt.text(18, 1., r'$z > 1$', color=hic, fontsize=20)
    plt.xlim(11, 19)
    plt.text(11.2, 1.35, '(17-W2)/4+0.15', rotation=-28, color=hic)
    plt.plot(np.linspace(10, 20, 100), (17 - np.linspace(10, 20, 100))/4 + 0.15, c=hic, ls='--', alpha=0.8)
    plt.axvline(17.5, ls='--', c='k', alpha=0.8)
    plt.xlabel('W2 [Vega mag]')
    plt.ylabel(r'W1 $-$ W2 [Vega mag]')
    plt.legend(fontsize=10, loc='lower left')
    ax.text(0.05, 0.93, r'$S_{150 \ \mathrm{MHz}} >$ %s mJy' % (int(fcut)), transform=ax.transAxes, fontsize=20)
    plt.savefig(plotdir + 'wise_diagram%s.pdf' % int(fcut))
    plt.close('all')

def wisediagram_both():

    fcuts = [2., 5.]
    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(8, 14))
    plt.xlim(11, 19)
    axs[1].set_xlabel('W2 [Vega mag]')
    for j in range(2):
        lotss = Table.read('../data/radio_cats/LOTSS_DR2/LOTSS_DR2_2mass_cw.fits')
        lotss = lotss[np.where(lotss['Total_flux'] > fcuts[j])]
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

        axs[j].scatter(lowz['W2_cw'], lowz['W1_cw'] - lowz['W2_cw'], s=20, c='none', edgecolors='cornflowerblue', marker='o')

        axs[j].scatter(midz['W2_cw'], midz['W1_cw'] - midz['W2_cw'], s=20, c='none', edgecolors='orange', marker='o')
        axs[j].scatter(hiz['W2_cw'], hiz['W1_cw'] - hiz['W2_cw'], s=20, c='none', edgecolors=hic, marker='o')
        axs[j].scatter(star['W2_cw'], star['W1_cw'] - star['W2_cw'], s=10, c='blue', marker='*', edgecolors='none')
        axs[j].scatter(0, 0, s=100, c='b', marker='*', label="SF-only", edgecolors='none')
        axs[j].plot(np.linspace(10, 13.86, 5), 0.65 * np.ones(5), ls='dotted', c='grey', alpha=0.5)
        axs[j].plot(np.linspace(13.86, 20, 100), 0.65 * np.exp(0.153 * np.square(np.linspace(13.86, 20, 100) - 13.86)),
                 c='grey', ls='dotted', alpha=0.5)
        if j == 1:
            plt.plot(np.linspace(12, 15.97, 10), (np.linspace(12, 15.97, 10) - 17) / 3 + 0.75, c='orange', ls='dashed')

        axs[j].text(17.6, 1.6, r'W2 90$\%$ Completeness', rotation=90, fontsize=8)
        axs[j].set_ylim(-0.5, 2.5)
        axs[j].text(16.15, 1.8, 'R90 AGN', fontsize=10, rotation=73, color='grey')
        axs[j].text(11.5, 0.4, r'$z < 0.5$', color='cornflowerblue', fontsize=20)
        axs[j].text(14.2, -0.4, r'$0.5 < z < 1$', color='orange', fontsize=20)
        axs[j].text(18, 1., r'$z > 1$', color=hic, fontsize=20)

        axs[j].text(11.2, 1.35, '(17-W2)/4+0.15', rotation=-28, color=hic)
        axs[j].plot(np.linspace(10, 20, 100), (17 - np.linspace(10, 20, 100)) / 4 + 0.15, c=hic, ls='--', alpha=0.8)
        axs[j].axvline(17.5, ls='--', c='k', alpha=0.8)

        axs[j].set_ylabel(r'W1 $-$ W2 [Vega mag]')
        axs[j].legend(fontsize=10, loc='lower left')
        axs[j].text(0.05, 0.93, r'$S_{150 \ \mathrm{MHz}} >$ %s mJy' % (int(fcuts[j])), transform=axs[j].transAxes, fontsize=20)
    plt.subplots_adjust(hspace=0)
    plt.savefig(plotdir + 'wise_diagram.pdf')
    plt.close('all')

def hostgals(izrgs=False):
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
        lotz = sample.midz_rg_sample()
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
    rgs = sample.lotss_rg_sample(2)
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
    ax1.text(1, 0.03, 'Spectroscopic', color='firebrick', fontsize=25)
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
    rgs = sample.midz_rg_sample()
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
    ax.text(0.5, 0.05, 'Spectroscopic', color='firebrick', fontsize=25)
    plt.xlabel('Redshift')
    plt.ylabel('Redshift distribution')

    plt.subplots_adjust(hspace=0)
    plt.savefig(plotdir + 'izrg_dndz.pdf')
    plt.close('all')

def lum_redshift(izrgs=False, hzrgflux=2., izrgflux=5.):

    lotz = sample.lotss_rg_sample(hzrgflux)
    s=40

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

    zspace = np.linspace(0.1, 4, 100)
    herglums = np.log10(fluxutils.luminosity_at_rest_nu(np.array(herg['Total_flux']), -0.7, .144, .15, herg['z_best'], flux_unit=u.mJy, energy=False))
    lerglums = np.log10(
        fluxutils.luminosity_at_rest_nu(np.array(lerg['Total_flux']), -0.7, .144, .15, lerg['z_best'], flux_unit=u.mJy,
                                        energy=False))
    sfglums = np.log10(
        fluxutils.luminosity_at_rest_nu(np.array(sfg['Total_flux']), -0.7, .144, .15, sfg['z_best'], flux_unit=u.mJy,
                                        energy=False))
    agnlums = np.log10(
        fluxutils.luminosity_at_rest_nu(np.array(agn['Total_flux']), -0.7, .144, .15, agn['z_best'], flux_unit=u.mJy,
                                        energy=False))
    limlum = np.log10(fluxutils.luminosity_at_rest_nu(hzrgflux*np.ones_like(zspace), -0.7, .144, .15, zspace, flux_unit=u.mJy, energy=False))



    fig, ax = plt.subplots(figsize=(8,7))
    plt.plot(zspace, limlum, c='k', ls='--')
    plt.scatter(herg['z_best'], herglums, s=s, c='none', label='HERG',
                edgecolors='goldenrod', marker='D')
    plt.scatter(lerg['z_best'], lerglums, s=s, c='none', label='LERG',
                edgecolors='tomato')
    plt.scatter(sfg['z_best'], sfglums, s=s, c='none', label='SFG', marker='*',
                edgecolors='cornflowerblue')
    plt.scatter(agn['z_best'], agnlums, s=s, c='none', label='RQAGN', marker='s',
                edgecolors='darkgreen')

    if izrgs:
        lotz = sample.midz_rg_sample(izrgflux)


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
            plt.scatter(herg['z_best'], herglums, s=s, c='none',
                        edgecolors='goldenrod', marker='D', alpha=0.3)
        lerglums = np.log10(
            fluxutils.luminosity_at_rest_nu(np.array(lerg['Total_flux']), -0.7, .144, .15, lerg['z_best'],
                                            flux_unit=u.mJy,
                                            energy=False))
        if len(sfg) > 0:
            sfglums = np.log10(
                fluxutils.luminosity_at_rest_nu(np.array(sfg['Total_flux']), -0.7, .144, .15, sfg['z_best'],
                                                flux_unit=u.mJy,
                                                energy=False))
            plt.scatter(sfg['z_best'], sfglums, s=s, c='none', marker='*',
                        edgecolors='cornflowerblue', alpha=0.3)
        if len(agn) > 0:
            agnlums = np.log10(
                fluxutils.luminosity_at_rest_nu(np.array(agn['Total_flux']), -0.7, .144, .15, agn['z_best'],
                                                flux_unit=u.mJy,
                                                energy=False))
            plt.scatter(agn['z_best'], agnlums, s=s, c='none', marker='s',
                        edgecolors='darkgreen', alpha=0.3)
        zspace = np.linspace(0.1, 1, 20)
        limlum = np.log10(
            fluxutils.luminosity_at_rest_nu(izrgflux * np.ones_like(zspace), -0.7, .144, .15, zspace, flux_unit=u.mJy,
                                            energy=False))
        plt.plot(zspace, limlum, c='k', ls='--')


        plt.scatter(lerg['z_best'], lerglums, s=s, c='none', label='IzLERGs',
                    edgecolors='tomato', alpha=0.3)




    #plt.scatter(bootes['z_best'], lums, edgecolors='k', c='none')

    plt.legend()
    plt.xlabel('Redshift')
    plt.ylabel(r'log $L_{150 \ \mathrm{MHz}}$ [W/Hz]')
    plt.ylim(23.5, 28.5)
    plt.xlim(0, 3.5)
    plt.savefig(plotdir + 'lum_z.pdf')
    plt.close('all')


def heating_contribution():
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

    plt.axvline(np.log10(fluxutils.luminosity_at_rest_nu(5., -0.7, .144, .15, .75, flux_unit=u.mJy, energy=False)),
                color='purple', ls='dotted')
    plt.axvline(np.log10(fluxutils.luminosity_at_rest_nu(2., -0.7, .144, .15, 1.25, flux_unit=u.mJy, energy=False)),
                color='cornflowerblue', ls='dotted')
    plt.axvline(np.log10(fluxutils.luminosity_at_rest_nu(2., -0.7, .144, .15, 1.75, flux_unit=u.mJy, energy=False)),
                color='orange', ls='dotted')
    plt.axvline(np.log10(fluxutils.luminosity_at_rest_nu(2., -0.7, .144, .15, 2.25, flux_unit=u.mJy, energy=False)),
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

#wisediagram(2.)
wisediagram_both()
#izrg_redshift_dist(7)
#hostgals(True)
#redshift_dist(30)
#lum_redshift(True)
#heating_contribution()