import pandas as pd
from SkyTools import fluxutils
import astropy.units as u
from halomodelpy import luminosityfunction, hubbleunits, redshift_helper
import numpy as np
from astropy import constants as const
from halomodelpy import cosmo
apcosmo = cosmo.apcosmo


ragn1_13 = pd.read_csv('AGN_LF_Kondapally22-main/radio-excess_AGN/LF_radio_excess_AGN_1.0_z_1.3_tab.csv')
ragn13_17 = pd.read_csv('AGN_LF_Kondapally22-main/radio-excess_AGN/LF_radio_excess_AGN_1.3_z_1.7_tab.csv')
ragn17_21 = pd.read_csv('AGN_LF_Kondapally22-main/radio-excess_AGN/LF_radio_excess_AGN_1.7_z_2.1_tab.csv')
ragn21_25 = pd.read_csv('AGN_LF_Kondapally22-main/radio-excess_AGN/LF_radio_excess_AGN_2.1_z_2.5_tab.csv')
lfs = [ragn1_13, ragn13_17, ragn17_21, ragn21_25]
#zcenters = np.array([1.15, 1.5, 1.9, 2.3])
zcenters = np.array([1.25, 1.75, 2.25])


def integrate_lf_data(l_grid, logl_tab, log_rho):
    # fill -infinity for right extrapolation, 10**-inf = 0
    logrho_interp = np.interp(l_grid, np.array(logl_tab), np.array(log_rho), right=-np.inf)
    return np.trapz(10**logrho_interp, x=l_grid)


def integrate_lf_model(lftab, lmin, lmax=29):
    lgrid = np.linspace(lmin, lmax, 100)
    sortidx = np.argsort(lftab['lum'])
    lum, rho = lftab['lum'][sortidx], lftab['rho'][sortidx]

    interp_lf = np.interp(lgrid, lum, rho, right=-np.inf)

    return hubbleunits.add_h_to_density(np.trapz(10 ** interp_lf, x=lgrid))


"""def density_brighter_than_flux(fluxcut=2.):
    l150 = np.log10(fluxutils.luminosity_at_rest_nu(fluxcut, -0.7, .144, .15, zcenters, flux_unit=u.mJy, energy=False))
    dens = []
    for j in range(len(zcenters)):
        dens.append(hubbleunits.add_h_to_density(
            integrate_lf(np.linspace(l150[j], 30, 10), lfs[j]['logL'], lfs[j]['rho_agn'])))
    return np.array(dens)"""

def interp_density_brighter_than_lum(z, lcut, lmax=29):
    """
    Interpolate luminosity functions (integrated over lum limits) in redshift bins to any redshift in between
    :param z:
    :param lcut:
    :param lmax:
    :return:
    """

    ragn05_1 = pd.read_csv('results/kondapally23_agnlf/agn_05_1.csv', names=['lum', 'rho'])
    ragn1_15 = pd.read_csv('results/kondapally23_agnlf/agn_1_15.csv', names=['lum', 'rho'])
    ragn15_2 = pd.read_csv('results/kondapally23_agnlf/agn_15_2.csv', names=['lum', 'rho'])
    ragn2_25 = pd.read_csv('results/kondapally23_agnlf/agn_2_25.csv', names=['lum', 'rho'])
    zcenters = np.array([.75, 1.25, 1.75, 2.25])
    tables = [ragn05_1, ragn1_15, ragn15_2, ragn2_25]
    #l150 = np.log10(fluxutils.luminosity_at_rest_nu(fluxcut, -0.7, .144, .15, zcenters, flux_unit=u.mJy, energy=False))
    dens = []
    for j in range(len(zcenters)):
        dens.append(integrate_lf_model(tables[j], lcut, lmax))
    dens = np.array(dens)
    return 10 ** np.interp(z, zcenters, np.log10(dens))

def lf_fluxlim_dndz(dndz, fluxcut, fluxmax=None):
    """
    Integrate the luminosity function of your flux limited sample over the redshift distirbution
    :param dndz:
    :param fluxcut:
    :param fluxmax:
    :return:
    """
    dndz = redshift_helper.norm_z_dist(dndz)
    zspace = dndz[0]
    l150s = np.log10(fluxutils.luminosity_at_rest_nu(fluxcut, -0.7, .144, .15, zspace, flux_unit=u.mJy, energy=False))
    l150max = 29 * np.ones_like(zspace)
    if fluxmax is not None:
        l150max = np.log10(fluxutils.luminosity_at_rest_nu(fluxmax, -0.7, .144, .15, zspace, flux_unit=u.mJy, energy=False))
    densities = []
    for j in range(len(zspace)):
        densities.append(interp_density_brighter_than_lum(zspace[j], l150s[j], l150max[j]))
    return np.trapz(np.array(densities)*dndz[1], x=dndz[0])



def lf_fluxlim_zrange(zrange, fluxcut, fluxmax=None):
    """
    assume a constant dndz over a given z range tuple
    :param zrange:
    :param fluxcut:
    :param fluxmax:
    :return:
    """
    dndz = redshift_helper.dndz_from_z_list(np.random.uniform(zrange[0], zrange[1], 10000), 10)

    return lf_fluxlim_dndz(dndz=dndz, fluxcut=fluxcut, fluxmax=fluxmax)

def hmf_zrange(zrange, logminmass):
    dndz = redshift_helper.dndz_from_z_list(np.random.uniform(zrange[0], zrange[1], 10000), 10)
    return luminosityfunction.int_hmf_z(dndz=dndz, logminmass=logminmass)


def occupation_dndz(logminmass, dndz, fluxcut, fluxmax=None):
    galdens = lf_fluxlim_dndz(dndz, fluxcut=fluxcut, fluxmax=fluxmax)
    halodens = luminosityfunction.int_hmf_z(dndz, logminmass=logminmass)
    return galdens / halodens

def occupation_zrange(logminmass, zrange, fluxcut, fluxmax=None):
    galdens = lf_fluxlim_zrange(zrange, fluxcut, fluxmax)
    halodens = hmf_zrange(zrange, logminmass)
    return galdens / halodens



#def interp_lf_brighter(z, fluxcut=2.):
#    dens = density_brighter_than_flux(fluxcut=fluxcut)
#    return 10**np.interp(z, zcenters, np.log10(dens))

def occupation_frac(logminmass, z):
    dens = interp_lf_brighter(z)
    halodens = luminosityfunction.int_hmf(z, logminmass)
    focc = dens/halodens
    return focc

def kinetic_lum_kondapally(flux150, z, fcav=4):
    ghz_lum = fluxutils.luminosity_at_rest_nu(flux150, -0.7, .144, 1.4, z, flux_unit=u.mJy, energy=False)
    return 7e36 * fcav * (ghz_lum / 1e25) ** (0.68)

def kinetic_lum_kondapally_l150(l150, fcav=4):
    ghzlum = fluxutils.extrap_flux(10**l150, -0.7, .15, 1.4)
    return 7e36 * fcav * (ghzlum / 1e25) ** (0.68)

"""def int_heat(heatfile, z, fluxlim, maxflux):

    Integrate Kondapally+23 heating function to get total kinetic power
    :param heatfile:
    :param z:
    :param fluxlim:
    :param maxflux:
    :return:

    sortidx = np.argsort(heatfile['lum'])
    lum, heat = heatfile['lum'][sortidx], heatfile['heat'][sortidx]
    minlum = np.log10(fluxutils.luminosity_at_rest_nu(fluxlim, -0.7, .144, .15, z, flux_unit=u.mJy, energy=False))
    maxlum = np.log10(fluxutils.luminosity_at_rest_nu(maxflux, -0.7, .144, .15, z, flux_unit=u.mJy, energy=False))
    lumgrid = np.linspace(minlum, maxlum, 100)
    interp_heat = np.interp(lumgrid, lum, heat)
    totheat = np.trapz(interp_heat, lumgrid) * 1e7  # erg/s/Mpc^3
    return totheat"""

def int_heat(heatfile, lmin, lmax=29):
    sortidx = np.argsort(heatfile['lum'])
    lum, heat = heatfile['lum'][sortidx], heatfile['heat'][sortidx]
    lumgrid = np.linspace(lmin, lmax, 100)
    interp_heat = np.interp(lumgrid, lum, heat, right=0.)
    totheat = np.trapz(interp_heat, lumgrid) * 1e7  # erg/s/Mpc^3
    return hubbleunits.add_h_to_density(totheat)

def interp_heat_above_lum(z, lmin, lmax=29):
    heat0 = pd.read_csv('results/kondapally_heating/z05_1_agn.csv', names=['lum', 'heat'])
    heat1 = pd.read_csv('results/kondapally_heating/z1_15_agn.csv', names=['lum', 'heat'])
    heat2 = pd.read_csv('results/kondapally_heating/z15_2_agn.csv', names=['lum', 'heat'])
    heat3 = pd.read_csv('results/kondapally_heating/z2_25_agn.csv', names=['lum', 'heat'])
    heatfiles = [heat0, heat1, heat2, heat3]
    zs = [0.75, 1.25, 1.75, 2.25]

    totheats = []
    for j in range(len(heatfiles)):
        totheats.append(np.log10(int_heat(heatfiles[j], lmin, lmax)))
    return 10 ** np.interp(z, zs, totheats)


def heating_fluxlim_dndz(dndz, fluxcut, fluxmax=None):
    """
    Integrate the heating function over dndz for a fluxlimited sample
    :param dndz:
    :param fluxcut:
    :param fluxmax:
    :return:
    """
    dndz = redshift_helper.norm_z_dist(dndz)
    zspace = dndz[0]
    l150s = np.log10(fluxutils.luminosity_at_rest_nu(fluxcut, -0.7, .144, .15, zspace, flux_unit=u.mJy, energy=False))
    l150max = 29 * np.ones_like(zspace)
    if fluxmax is not None:
        l150max = np.log10(fluxutils.luminosity_at_rest_nu(fluxmax, -0.7, .144, .15, zspace, flux_unit=u.mJy, energy=False))
    densities = []
    for j in range(len(zspace)):
        densities.append(interp_heat_above_lum(zspace[j], l150s[j], l150max[j]))
    return np.trapz(np.array(densities)*dndz[1], x=dndz[0])

def heating_fluxlim_zrange(zrange, fluxcut, fluxmax=None):
    """
    assume a constant dndz over a given z range tuple
    :param zrange:
    :param fluxcut:
    :param fluxmax:
    :return:
    """
    dndz = redshift_helper.dndz_from_z_list(np.random.uniform(zrange[0], zrange[1], 10000), 10)

    return heating_fluxlim_dndz(dndz=dndz, fluxcut=fluxcut, fluxmax=fluxmax)

def heat_per_halo_zrange(zrange, logminmass, fluxcut, fluxmax=None):
    heatdens = heating_fluxlim_zrange(zrange=zrange, fluxcut=fluxcut, fluxmax=fluxmax)
    halodens = hmf_zrange(zrange=zrange, logminmass=logminmass)
    return heatdens / halodens

def energy_per_halo_zrange(zrange, logminmass, fluxcut, fluxmax=None):
    dutycycle = occupation_zrange(zrange=zrange, logminmass=logminmass, fluxcut=fluxcut, fluxmax=fluxmax)
    elapsedtime = (apcosmo.age(zrange[0]) - apcosmo.age(zrange[1])).to('s').value
    heatperhalo = heat_per_halo_zrange(zrange=zrange, logminmass=logminmass, fluxcut=fluxcut, fluxmax=fluxmax)
    return np.log10(dutycycle * elapsedtime * heatperhalo)


def mgas_from_mhalo(logmh):

    mh = hubbleunits.remove_h_from_mass(10 ** logmh)
    mgas = apcosmo.Ob0 / apcosmo.Om0 * mh
    return mgas

def halobinding_energy(logmh, z):

    from colossus.halo import mass_so


    r200c = hubbleunits.remove_h_from_scale(mass_so.M_to_R(10**logmh, z, '200c'))
    mgas = mgas_from_mhalo(logmh)

    ubind = (3 * const.G * (mgas * u.solMass) ** 2 / (5 * r200c * u.kpc)).to(u.erg).value
    return np.log10(ubind)




def thermal_energy_mass(logmh, tempkev):
    mgas = mgas_from_mhalo(logmh) * u.solMass
    nparticles = mgas / const.m_p
    return np.log10((3/2 * nparticles * (tempkev * u.keV)).to('erg').value)


def lx_energy(loglx, zrange):
    tcosmic = (apcosmo.age(zrange[0]) - apcosmo.age(zrange[1])).to('s').value
    return np.log10(tcosmic * 10 ** loglx)


def windheating(zrange, kinetic_frac=0.005):
    hopkins_ebol = pd.read_csv('results/kondapally_heating/ebol_agn_hopkins.csv', names=['z', 'Ebol'])
    goodidx = np.where((hopkins_ebol['z'] > zrange[0]) & (hopkins_ebol['z'] < zrange[1]))
    ebols = hopkins_ebol['Ebol'].loc[goodidx]
    return hubbleunits.add_h_to_density(kinetic_frac*(np.max(ebols) - np.min(ebols)) * 1e7)


def windheat_per_halo(zrange, logminmass, fduty, kinetic_frac=0.005):
    heatdens = windheating(zrange, kinetic_frac)
    halodens = hmf_zrange(zrange, logminmass)
    heatperhalo = heatdens / halodens
    return np.log10(heatperhalo)

def type1_windheatperhalo(zrange, logminmass, kinetic_frac=0.005):
    # quasars live in 12.5 halos, or minimum mass 12.2
    halodens_type1 = hmf_zrange(zrange=zrange, logminmass=12.2)
    # fraction of quasars in halos more massive than M
    massive_frac = hmf_zrange(zrange=zrange, logminmass=logminmass) / halodens_type1
    # energy released in massive halos is massive_frac * energy released by all quasars
    epervolume = massive_frac * windheating(zrange=zrange, kinetic_frac=kinetic_frac)
    return np.log10(epervolume / hmf_zrange(zrange=zrange, logminmass=logminmass))


def hod_density_above_m(hodparams, fduty, eff_z, logm_min):
    from halomodelpy import hod_model
    hod = hod_model.zheng_hod(hodparams, param_ids=['M', 'sigM', 'M0', 'M1', 'alpha'])
    number_per_volume = fduty * hod['hod'] * cosmo.hmf_z(hod['mgrid'], eff_z)
    goodidx = np.where(hod['mgrid'] > logm_min)
    mgrid = 10**hod['mgrid'][goodidx]
    number_per_volume = number_per_volume[goodidx]

    dens_above_m = np.trapz(number_per_volume, x=np.log(mgrid))
    return dens_above_m







def interp_heat_z(z, fluxlim, maxflux):
    """
    Interpolate between redshift bins to estimate total heating power at any redshift z
    :param z:
    :param fluxlim:
    :param maxflux:
    :return:
    """
    heat1 = pd.read_csv('results/kondapally_heating/z1_15_agn.csv', names=['lum', 'heat'])
    heat2 = pd.read_csv('results/kondapally_heating/z15_2_agn.csv', names=['lum', 'heat'])
    heat3 = pd.read_csv('results/kondapally_heating/z2_25_agn.csv', names=['lum', 'heat'])
    heatfiles = [heat1, heat2, heat3]
    zs = [1.25, 1.75, 2.25]

    totheats = []
    for j in range(len(heatfiles)):
        totheats.append(np.log10(int_heat(heatfiles[j], zs[j], fluxlim, maxflux)))
    return 10 ** np.interp(z, zs, totheats)
