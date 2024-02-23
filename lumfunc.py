import pandas as pd
from SkyTools import fluxutils
import astropy.units as u
from halomodelpy import luminosityfunction, hubbleunits, redshift_helper
import numpy as np
from scipy.interpolate import interp1d
from astropy import constants as const
from halomodelpy import cosmo
apcosmo = cosmo.apcosmo





def parse_dndz_or_zrange(dndz_or_zrange):
    """
    pass either a redshift distribution or a set of redshift bounds, return formatted dndz
    :param dndz_or_zrange:
    :return:
    """
    # if only (min, max) redshift bounds given, assume constant dn/dz in those bounds
    if np.isscalar(dndz_or_zrange[0]):
        dndz = redshift_helper.dndz_from_z_list(np.random.uniform(dndz_or_zrange[0], dndz_or_zrange[1], 10000), 10)
    else:
        dndz = redshift_helper.norm_z_dist(dndz_or_zrange)
    return dndz


def integrate_lf_data(l_grid, logl_tab, log_rho):
    """
    Integrate a luminosity function
    :param l_grid:
    :param logl_tab:
    :param log_rho:
    :return:
    """
    # fill -infinity for right extrapolation, 10**-inf = 0
    logrho_interp = np.interp(l_grid, np.array(logl_tab), np.array(log_rho), right=-np.inf)
    return np.trapz(10**logrho_interp, x=l_grid)


def integrate_lf_model(lftab, lmin, lmax=29.):
    """
    Integrate a radio luminosity function above a threshold luminosity lmin, up to default max luminosity 10^29 W/Hz
    :param lftab:
    :param lmin:
    :param lmax:
    :return:
    """
    lgrid = np.linspace(lmin, lmax, 100)
    sortidx = np.argsort(lftab['lum'])
    lum, rho = lftab['lum'][sortidx], lftab['rho'][sortidx]

    interp_lf = np.interp(lgrid, lum, rho, right=-np.inf)

    return hubbleunits.add_h_to_density(np.trapz(10 ** interp_lf, x=lgrid))


def interp_lf_lumbounds(z, lcut, lmax=29., lftype='agn'):
    """
    Interpolate luminosity functions (integrated over lum limits) in redshift bins to any redshift in between
    :param z:
    :param lcut:
    :param lmax:
    :return:
    """
    ragn0_03 = pd.read_csv('results/kondapally23_agnlf/agn_0_03.csv', names=['lum', 'rho'])
    ragn05_1 = pd.read_csv('results/kondapally23_agnlf/%s_05_1.csv' % lftype, names=['lum', 'rho'])
    ragn1_15 = pd.read_csv('results/kondapally23_agnlf/%s_1_15.csv' % lftype, names=['lum', 'rho'])
    ragn15_2 = pd.read_csv('results/kondapally23_agnlf/%s_15_2.csv' % lftype, names=['lum', 'rho'])
    ragn2_25 = pd.read_csv('results/kondapally23_agnlf/%s_2_25.csv' % lftype, names=['lum', 'rho'])
    zcenters = np.array([0.21, .75, 1.25, 1.75, 2.25])
    tables = [ragn0_03, ragn05_1, ragn1_15, ragn15_2, ragn2_25]
    #l150 = np.log10(fluxutils.luminosity_at_rest_nu(fluxcut, -0.7, .144, .15, zcenters, flux_unit=u.mJy, energy=False))
    dens = []
    for j in range(len(zcenters)):
        dens.append(integrate_lf_model(tables[j], lcut, lmax))
    dens = np.array(dens)
    return 10 ** interp1d(zcenters, np.log10(dens), fill_value='extrapolate')(z)
    #return 10 ** np.interp(z, zcenters, np.log10(dens))


def lf_lumbounds(dndz_or_zrange, lmin, lmax=29., lftype='agn'):
    """
    Integrate luminosity function between two luminosity bounds, then integrate over redshift distribution
    :param dndz_or_zrange: either a tuple (zs, p(z)s) or a tuple (min_z, max_z) (assumes constant dndz over range)
    :param lmin: minimum luminosity bound log W/Hz
    :param lmax: maximum luminosity bound log W/Hz
    :param lftype: 'agn', or 'lerg'
    :return: number density of RGs in hubble units
    """
    dndz = parse_dndz_or_zrange(dndz_or_zrange)
    zspace = dndz[0]
    densities = []
    for j in range(len(zspace)):
        densities.append(interp_lf_lumbounds(zspace[j], lmin, lmax, lftype=lftype))
    return np.trapz(np.array(densities)*dndz[1], x=dndz[0])

def lf_fluxlim(dndz_or_zrange, fluxcut, fluxmax=None, lftype='agn'):
    """
    integrating luminosity function evolving with redshift corresponding to flux limit
    :param dndz_or_zrange: either a tuple (zs, p(z)s) or a tuple (min_z, max_z) (assumes constant dndz over range)
    :param fluxcut: minimum flux bound mJy
    :param fluxmax: maximum flux bound mJy
    :return: number density of RGs in hubble units
    """
    dndz = parse_dndz_or_zrange(dndz_or_zrange)
    zspace = dndz[0]
    l150s = np.log10(fluxutils.luminosity_at_rest_nu(fluxcut, -0.7, .144, .15, zspace, flux_unit=u.mJy, energy=False))
    l150max = 29 * np.ones_like(zspace)
    if fluxmax is not None:
        l150max = np.log10(fluxutils.luminosity_at_rest_nu(fluxmax, -0.7, .144, .15, zspace, flux_unit=u.mJy, energy=False))
    densities = []
    for j in range(len(zspace)):
        densities.append(interp_lf_lumbounds(zspace[j], l150s[j], l150max[j], lftype=lftype))
    return np.trapz(np.array(densities)*dndz[1], x=dndz[0])




def occupation(logminmass, dndz_or_zrange, f_or_l_min, f_or_l_max=None, lftype='agn', lumbound=True):
    """
    duty cycle given as number density of RGs to density of halos massive enough to host them
    using a flux or luminosity-limited sample
    :param logminmass: log minimum halo mass (hubble unit) to integrate above
    :param dndz_or_zrange: either a tuple (zs, p(z)s) or a tuple (min_z, max_z) (assumes constant dndz over range)
    :param fluxcut: minimum flux bound mJy
    :param fluxmax: maximum flux bound mJy
    :return: occupation fraction number RGs/number halos
    """
    if lumbound:
        galdens = lf_lumbounds(dndz_or_zrange=dndz_or_zrange, lmin=f_or_l_min, lmax=f_or_l_max, lftype=lftype)
    else:
        galdens = lf_fluxlim(dndz_or_zrange=dndz_or_zrange, fluxcut=f_or_l_min, fluxmax=f_or_l_max, lftype=lftype)

    dndz = parse_dndz_or_zrange(dndz_or_zrange)
    halodens = luminosityfunction.int_hmf_z(dndz, logminmass=logminmass)
    return galdens / halodens







def kinetic_lum_kondapally(flux150, z, fcav=4):
    """
    Equation 2 of Kondapally+23 adapted for measured observed frame flux
    :param flux150:
    :param z:
    :param fcav:
    :return:
    """
    # convert to rest frame L(1.4 GHz)
    ghz_lum = fluxutils.luminosity_at_rest_nu(flux150, -0.7, .144, 1.4, z, flux_unit=u.mJy, energy=False)
    return 7e36 * fcav * (ghz_lum / 1e25) ** (0.68)

def kinetic_lum_kondapally_l150(l150, fcav=4):
    """
    Equation 2 of Kondapally+23
    :param l150:
    :param fcav:
    :return:
    """
    # k correct L(150 MHz) to L(1.4 GHz)
    ghzlum = fluxutils.extrap_flux(10**l150, -0.7, .15, 1.4)
    return 7e36 * fcav * (ghzlum / 1e25) ** (0.68)


def local_heating():
    """
    Convert local luminosity function to heating function
    :return:
    """
    locallf = pd.read_csv('results/kondapally23_agnlf/agn_0_03.csv', names=['lum', 'rho'])
    heats = kinetic_lum_kondapally_l150(locallf['lum'])*10**locallf['rho']
    newdf = pd.DataFrame()
    newdf['lum'] = locallf['lum']
    newdf['heat'] = heats
    return newdf


def integrate_heat(heatfile, lmin, lmax=29.):
    """
    Integrate heating function between jet luminosity bounds
    :param heatfile:
    :param lmin:
    :param lmax:
    :return: Volume averaged heating power injected by radio galaxies between luminosity bounds, units are erg/s/(Mpc/h)^3
    """
    # ensure sorted
    sortidx = np.argsort(heatfile['lum'])
    lum, heat = heatfile['lum'][sortidx], heatfile['heat'][sortidx]
    lumgrid = np.linspace(lmin, lmax, 100)
    interp_heat = np.interp(lumgrid, lum, heat, right=0.)
    totheat = np.trapz(interp_heat, lumgrid) * 1e7  # erg/s/Mpc^3
    # add hubble units for natural comparison to Halo mass function
    return hubbleunits.add_h_to_density(totheat)


def interp_heat_lumbounds(z, lmin, lmax=29., lftype='agn'):
    """
    Integrate heating function between luminosity bounds, interpolate to any given redshift
    :param z:
    :param lmin:
    :param lmax:
    :param lftype: 'agn', or 'lerg'
    :return:
    """
    # z=0.2
    heatlocal = local_heating()
    # z=0.75
    heat0 = pd.read_csv('results/kondapally_heating/z05_1_%s.csv' % lftype, names=['lum', 'heat'])
    # z=1.25
    heat1 = pd.read_csv('results/kondapally_heating/z1_15_%s.csv' % lftype, names=['lum', 'heat'])
    # z=1.75
    heat2 = pd.read_csv('results/kondapally_heating/z15_2_%s.csv' % lftype, names=['lum', 'heat'])
    # z=2.25
    heat3 = pd.read_csv('results/kondapally_heating/z2_25_%s.csv' % lftype, names=['lum', 'heat'])
    heatfiles = [heatlocal, heat0, heat1, heat2, heat3]
    zs = [0.21, 0.75, 1.25, 1.75, 2.25]
    # interpolate to given z
    totheats = []
    for j in range(len(heatfiles)):
        totheats.append(np.log10(integrate_heat(heatfiles[j], lmin, lmax)))
    return 10 ** np.interp(z, zs, totheats)



def dens_above_cut_stepfunc_hod_z(z, hod_logminmass, logmass_integral_bound, sigM=None, logmaxmass=16.):
    from halomodelpy import bias_tools
    """
    For a step function HOD with Mmin=hod_logminmass, integrate HOD*HMF above a different mass threshold
    to get the density of galaxies in halos more massive than logmass_integral_bound
    :param z: redshift
    :param hod_logminmass:
    :param logmass_integral_bound:
    :return:
    """
    if sigM is None:
        mgrid = np.logspace(11, 15, 1000)
        hod = np.heaviside(np.log10(mgrid)-hod_logminmass, 1.)
    else:
        mgrid = bias_tools.paramobj.mass_space
        hod = bias_tools.ncen_zheng(logMmin=hod_logminmass, sigma=sigM)
    above_bound = np.where((np.log10(mgrid) >= logmass_integral_bound) & (np.log10(mgrid) <= logmaxmass))
    mgrid, hod = mgrid[above_bound], hod[above_bound]
    return hubbleunits.add_h_to_density(np.trapz(hod * cosmo.hmf_z(np.log10(mgrid), z), x=np.log(mgrid)))

def dens_above_cut_hodparams_z(z, hodparams, logmass_integral_bound, logmaxmass=16.):
    """
    Integrate HOD*HMF between mass bounds for any given Zheng HOD
    to get the density of galaxies in halos more between mass bounds
    :param z: redshift
    :param hodparams:
    :param logmass_integral_bound:
    :return:
    """
    from halomodelpy import hod_model

    hod = hod_model.zheng_hod(hodparams, ['M', 'sigM', 'M1', 'alpha'])
    mgrid = 10**hod['mgrid']
    hod = hod['hod']
    above_bound = np.where((np.log10(mgrid) >= logmass_integral_bound) & (np.log10(mgrid) <= logmaxmass))
    mgrid, hod = mgrid[above_bound], hod[above_bound]
    return hubbleunits.add_h_to_density(np.trapz(hod * cosmo.hmf_z(np.log10(mgrid), z), x=np.log(mgrid)))


def fraction_above_cut_stepfunc(dndz_or_zrange, hod_logminmass, logmass_integral_bound, sigM=None, logmaxmass=16.):
    """
    The fraction of galaxies in halos between mass intervals is integral between bounds divided by indefinite integral
    :param dndz_or_zrange:
    :param hod_logminmass:
    :param logmass_integral_bound:
    :param sigM:
    :param logmaxmass:
    :return:
    """
    dndz = parse_dndz_or_zrange(dndz_or_zrange)
    fracs = []
    for z in dndz[0]:
        fracs.append(
            (
                dens_above_cut_stepfunc_hod_z(z, hod_logminmass, logmass_integral_bound,
                                              sigM=sigM, logmaxmass=logmaxmass) /
                     dens_above_cut_stepfunc_hod_z(z, hod_logminmass, logmass_integral_bound=11.,
                                                   sigM=sigM, logmaxmass=16.)
            ))
    return np.trapz(np.array(fracs)*dndz[1], x=dndz[0])


def fraction_above_cut_hodparams(dndz_or_zrange, hodparams, logmass_integral_bound, logmaxmass=16.):
    dndz = parse_dndz_or_zrange(dndz_or_zrange)
    fracs = []
    for z in dndz[0]:
        fracs.append(
            (
                    dens_above_cut_hodparams_z(z, hodparams, logmass_integral_bound, logmaxmass=logmaxmass) /
                    dens_above_cut_hodparams_z(z, hodparams, logmass_integral_bound=11., logmaxmass=16.)
            ))
    return np.trapz(np.array(fracs)*dndz[1], x=dndz[0])



def heating_fluxlim(dndz_or_zrange, fluxcut, fluxmax=None, lftype='agn'):
    """
    Integrate the heating function over dndz for a fluxlimited sample
    :param dndz:
    :param fluxcut:
    :param fluxmax:
    :param lftype: 'agn', or 'lerg'
    :return:
    """
    dndz = parse_dndz_or_zrange(dndz_or_zrange)
    zspace = dndz[0]
    l150s = np.log10(fluxutils.luminosity_at_rest_nu(fluxcut, -0.7, .144, .15, zspace, flux_unit=u.mJy, energy=False))
    l150max = 29 * np.ones_like(zspace)
    if fluxmax is not None:
        l150max = np.log10(fluxutils.luminosity_at_rest_nu(fluxmax, -0.7, .144, .15, zspace, flux_unit=u.mJy, energy=False))
    densities = []
    for j in range(len(zspace)):
        densities.append(interp_heat_lumbounds(zspace[j], l150s[j], l150max[j], lftype=lftype))
    return np.trapz(np.array(densities)*dndz[1], x=dndz[0])

def heating_lumbounds(dndz_or_zrange, lmin, lmax=29., lftype='agn'):
    """
    Integrate the heating function over dndz for a luminosity limited sample
    :param dndz_or_zrange:
    :param lmin:
    :param lmax:
    :param lftype: 'agn', or 'lerg'
    :return:
    """
    dndz = parse_dndz_or_zrange(dndz_or_zrange)
    zspace = dndz[0]
    heatdens = []
    for j in range(len(zspace)):
        heatdens.append(interp_heat_lumbounds(z=zspace[j], lmin=lmin, lmax=lmax, lftype=lftype))
    return np.trapz(np.array(heatdens) * dndz[1], x=dndz[0])


def allheat_per_massive_halo(dndz_or_zrange, logminmass, f_or_l_min,
                             f_or_l_max=None, lftype='agn', lumbound=True, logmaxmass=16.):
    """
    total heating density divided by total halo number density between mass bounds
    volume-averaged heating power per halo
    :param dndz_or_zrange:
    :param logminmass:
    :param fluxcut:
    :param fluxmax:
    :param lftype: 'agn', or 'lerg'
    :return:
    """
    if lumbound:
        heatdens = heating_lumbounds(dndz_or_zrange=dndz_or_zrange, lmin=f_or_l_min, lmax=f_or_l_max, lftype=lftype)
    else:
        heatdens = heating_fluxlim(dndz_or_zrange=dndz_or_zrange, fluxcut=f_or_l_min, fluxmax=f_or_l_max, lftype=lftype)
    dndz = parse_dndz_or_zrange(dndz_or_zrange)
    halodens = luminosityfunction.int_hmf_z(dndz, logminmass=logminmass, logmaxmass=logmaxmass)
    return heatdens / halodens



def heat_per_massive_halo(dndz_or_zrange, hodparams, logminmass_cut, f_or_l_min, f_or_l_max=None,
                    lftype='agn', avg_power=True, lumbound=True, logmaxmass=16.):
    """
    If a fraction of luminous RGs are in halos less massive than logminmass_cut, they aren't contributing heat
    to the set of halos more massive than logminmass_cut
    Thus, the heating power deposited in massive halos is dependent on the HOD, -> the fraction of RGs in halos
    more massive than logminmass_cut
    :param dndz_or_zrange:
    :param hodparams: either array of [logM_min, sigma_M, M1, alpha] or tuple of (logM_min, sigma_M)
    :param logminmass_cut: threshold halo mass
    :param f_or_l_min: lower flux or luminosity limit
    :param f_or_l_max: upper flux or luminosity limit
    :param lftype: 'agn', or 'lerg'
    :param avg_power: True for power per halo in erg/s or False for total energy over redshift interval in erg
    :param lumbound: True for luminosity limited sample, otherwise flux-limited
    :return:
    """

    #dutycycle_cut = occupation_fluxlim(dndz_or_zrange=dndz_or_zrange, logminmass=logminmass_cut, fluxcut=fluxcut,
    #                                   fluxmax=fluxmax, lftype=lftype)

    #dutycycle_cut = 1.
    dndz = parse_dndz_or_zrange(dndz_or_zrange)
    if avg_power:
        elapsedtime = 1.
    else:
        zrange = (np.min(dndz[0]), np.max(dndz[1]))
        elapsedtime = (apcosmo.age(zrange[0]) - apcosmo.age(zrange[1])).to('s').value
    heatperhalo_cut = allheat_per_massive_halo(dndz_or_zrange=dndz_or_zrange, logminmass=logminmass_cut,
                                            f_or_l_min=f_or_l_min, f_or_l_max=f_or_l_max, lftype=lftype,
                                            lumbound=lumbound, logmaxmass=logmaxmass)
    if len(hodparams) == 2:
        frac_above_cut = fraction_above_cut_stepfunc(dndz_or_zrange=dndz_or_zrange, hod_logminmass=hodparams[0],
                                                logmass_integral_bound=logminmass_cut, sigM=hodparams[1],
                                                logmaxmass=logmaxmass)
    else:
        frac_above_cut = fraction_above_cut_hodparams(dndz_or_zrange=dndz_or_zrange, hodparams=hodparams,
                                                      logmass_integral_bound=logminmass_cut, logmaxmass=logmaxmass)
    return np.log10(frac_above_cut * elapsedtime * heatperhalo_cut)


def heat_per_massive_halo_frombias(dndz_or_zrange, bias_and_err, logminmass_cut, sigM, f_or_l_min,  f_or_l_max=None,
                    lftype='agn', avg_power=True, lumbound=True, logmaxmass=16.):
    from halomodelpy import bias_tools
    dndz = parse_dndz_or_zrange(dndz_or_zrange)
    m_char, mlo, mhi = bias_tools.avg_bias2mass_transition(dndz=dndz, b=bias_and_err[0], sigma=sigM,
                                                           berr=bias_and_err[1])
    power = heat_per_massive_halo(dndz_or_zrange=dndz, hodparams=[m_char, sigM], logminmass_cut=logminmass_cut,
                                  f_or_l_min=f_or_l_min, f_or_l_max=f_or_l_max, lftype=lftype, avg_power=avg_power,
                                    lumbound=lumbound, logmaxmass=logmaxmass)
    up_power = heat_per_massive_halo(dndz_or_zrange=dndz, hodparams=[m_char+mhi, sigM], logminmass_cut=logminmass_cut,
                                  f_or_l_min=f_or_l_min, f_or_l_max=f_or_l_max, lftype=lftype, avg_power=avg_power,
                                    lumbound=lumbound, logmaxmass=logmaxmass)
    lo_power = heat_per_massive_halo(dndz_or_zrange=dndz, hodparams=[m_char-mlo, sigM], logminmass_cut=logminmass_cut,
                                  f_or_l_min=f_or_l_min, f_or_l_max=f_or_l_max, lftype=lftype, avg_power=avg_power,
                                    lumbound=lumbound, logmaxmass=logmaxmass)
    up_err = up_power - power
    lo_err = power - lo_power
    return power, lo_err, up_err

def heat_per_massive_halo_from_hod_draws(dndz_or_zrange, hoddraws, logminmass_cut, f_or_l_min, f_or_l_max=None,
                                         lftype='agn', avg_power=True, lumbound=True, logmaxmass=16.):
    powers = []
    for j in range(len(hoddraws)):
        powers.append(heat_per_massive_halo(dndz_or_zrange=dndz_or_zrange, hodparams=hoddraws[j],
                                      logminmass_cut=logminmass_cut, f_or_l_min=f_or_l_min, f_or_l_max=f_or_l_max,
                                      lftype=lftype, avg_power=avg_power, lumbound=lumbound, logmaxmass=logmaxmass))
    power = np.median(powers)
    up_err = np.percentile(powers, 84) - power
    lo_err = power - np.percentile(powers, 16)
    return power, lo_err, up_err



def dutycycle_from_bias(dndz_or_zrange, bias_and_err, f_or_l_min, f_or_l_max=None, lftype='agn', lumbound=True):
    from halomodelpy import bias_tools
    dndz = parse_dndz_or_zrange(dndz_or_zrange)
    mmin, mmin_lo, mmin_hi = bias_tools.avg_bias2min_mass(dndz=dndz, b=bias_and_err[0], berr=bias_and_err[1])
    fduty = occupation(logminmass=mmin, dndz_or_zrange=dndz, f_or_l_min=f_or_l_min, f_or_l_max=f_or_l_max,
                       lftype=lftype, lumbound=lumbound)
    fduty_up = occupation(logminmass=mmin+mmin_hi, dndz_or_zrange=dndz, f_or_l_min=f_or_l_min, f_or_l_max=f_or_l_max,
                       lftype=lftype, lumbound=lumbound)
    fduty_lo = occupation(logminmass=mmin-mmin_lo, dndz_or_zrange=dndz, f_or_l_min=f_or_l_min, f_or_l_max=f_or_l_max,
                       lftype=lftype, lumbound=lumbound)

    up_err = fduty_up - fduty
    lo_err = fduty - fduty_lo
    return fduty, lo_err, up_err



def windheating(zrange, kinetic_frac=0.005):
    """
    Global wind-heating energy density released by AGN

    :param zrange: tuple (min redshift, max redshift)
    :param kinetic_frac: Coupling coefficient between bolometric AGN luminosity and wind power
    :return:
    """
    # cumulative bolometric energy density released by AGN as function of redshift
    # taken from Figure 3 of Kondapally+23, which is using data from Hopkins+07
    hopkins_ebol = pd.read_csv('results/kondapally_heating/ebol_agn_hopkins.csv', names=['z', 'Ebol'])
    # take entries between given redshift range
    goodidx = np.where((hopkins_ebol['z'] > zrange[0]) & (hopkins_ebol['z'] < zrange[1]))
    ebols = hopkins_ebol['Ebol'].loc[goodidx]
    # heat released in redshift interval is (wind-power coupling efficiency) * (bolometric energy released)
    return hubbleunits.add_h_to_density(kinetic_frac*(np.max(ebols) - np.min(ebols)) * 1e7)


def windheat_per_massivehalo(zrange, logminmass, kinetic_frac=0.005, logmaxmass=16.):
    """
    Total wind-heating energy in erg of radiative AGN per halo for all halos between logminmass and logmaxmass
    :param zrange: tuple (min redshift, max redshift)
    :param logminmass: minimum mass of halos above which to calculate density
    :param kinetic_frac: Coupling coefficient between bolometric AGN luminosity and wind power
    :return: heat per massive halo in erg
    """
    heatdens = windheating(zrange, kinetic_frac)
    halodens = luminosityfunction.hmf_zrange(zrange, logminmass, logmaxmass=logmaxmass)
    heatperhalo = heatdens / halodens
    return np.log10(heatperhalo)

def windpower_per_massivehalo(zrange, logminmass, kinetic_frac=0.005, logmaxmass=16.):
    """
    Total wind-heating power in erg/s over redshift interval per halo for all halos between logminmass and logmaxmass
    Just dividing total heat energy released by time elapsed ind redshift interval
    :param zrange: tuple (min redshift, max redshift)
    :param logminmass: minimum mass of halos above which to calculate density
    :param kinetic_frac: Coupling coefficient between bolometric AGN luminosity and wind power
    :return: heating power per massive halo in erg/s
    """
    tcosmic = (apcosmo.age(zrange[0]) - apcosmo.age(zrange[1])).to('s').value
    heatperhalo = 10**windheat_per_massivehalo(zrange=zrange, logminmass=logminmass, kinetic_frac=kinetic_frac,
                                               logmaxmass=logmaxmass)
    powerperhalo = heatperhalo / tcosmic
    return np.log10(powerperhalo)

"""def type1_windheatperhalo(zrange, logminmass, kinetic_frac=0.005):
    # quasars live in 12.5 halos, or minimum mass 12.2
    halodens_type1 = luminosityfunction.hmf_zrange(zrange=zrange, logminmass=12.2)
    # fraction of quasars in halos more massive than M
    massive_frac = luminosityfunction.hmf_zrange(zrange=zrange, logminmass=logminmass) / halodens_type1
    # energy released in massive halos is massive_frac * energy released by all quasars
    epervolume = massive_frac * windheating(zrange=zrange, kinetic_frac=kinetic_frac)
    return np.log10(epervolume / luminosityfunction.hmf_zrange(zrange=zrange, logminmass=logminmass))


def hod_density_above_m(hodparams, fduty, eff_z, logm_min):
    from halomodelpy import hod_model
    hod = hod_model.zheng_hod(hodparams, param_ids=['M', 'sigM', 'M0', 'M1', 'alpha'])
    number_per_volume = fduty * hod['hod'] * cosmo.hmf_z(hod['mgrid'], eff_z)
    goodidx = np.where(hod['mgrid'] > logm_min)
    mgrid = 10**hod['mgrid'][goodidx]
    number_per_volume = number_per_volume[goodidx]

    dens_above_m = np.trapz(number_per_volume, x=np.log(mgrid))
    return dens_above_m"""



def desiqso_hod_dens(z, logminmass=12., logmaxmass=16.):
    """
    number density of quasars in halos between logminmass and logmaxmass is integral of QSO HOD * HMF
    assuming QSO HOD of Prada+23
    :param z: redshift
    :param logminmass: log minimum halo mass (hubble units)
    :return:
    """
    from scipy.signal import savgol_filter
    qsohod = pd.read_csv('results/hod/desi_qso/desi_qso_hod.csv', names=['M', 'hod'])
    sortidx = np.argsort(qsohod['M'])
    m, hod = np.array(qsohod['M'][sortidx]), np.array(qsohod['hod'][sortidx])
    # smooth with savitsky golay filter above 10^13 Msun where hod is noisy
    satregion = m >= 1e13
    cenregion = m < 1e13
    hodcen = hod[cenregion]
    hodsat = hod[satregion]
    hodsat = 10 ** savgol_filter(np.log10(hodsat), 10, 1)
    hod = np.array(list(hodcen) + list(hodsat))

    inmassbin = np.where((m > 10 ** logminmass) & (m < 10 ** logmaxmass))[0]
    m, hod = m[inmassbin], hod[inmassbin]
    # integrate hod * hmf
    return hubbleunits.add_h_to_density(np.trapz(hod * cosmo.hmf_z(np.log10(m), z), x=np.log(m)))

def desiqso_massive_frac(z, logminmass, logmaxmass=16.):
    """
    fraction of all QSOs which live in halos between logminmass and logmaxmass
    integral of HOD*HMF from logminmass to inf, divided by integral from 0 to inf
    :param z: redshift
    :param logminmass: log minimum halo mass (hubble units)
    :return:
    """
    totaldens = desiqso_hod_dens(z, logminmass=12., logmaxmass=16.)
    massivedens = desiqso_hod_dens(z, logminmass=logminmass, logmaxmass=logmaxmass)
    return massivedens / totaldens

def qso_windheatperhalo(zrange, logminmass, kinetic_frac=0.005, power=True, logmaxmass=16.):
    """
    Estimate total heat energy (power) deposited by quasars per number of halos more massive than logminmass
    :param zrange: tuple (min redshift, max redshift)
    :param logminmass: minimum mass of halos above which to calculate density
    :param kinetic_frac: bolometric luminosity / wind coupling coefficient
    :param power: True for heat energy divided by cosmic time elapsed (erg/s), False for total energy in erg
    :return: Heat energy (erg) or power (erg/s) per massive halo
    """
    # global heat energy deposited by AGN in zrange per volume
    agn_windenergy_per_volume = windheating(zrange, kinetic_frac=kinetic_frac)
    midz = zrange[0] + (zrange[1] - zrange[0]) / 2.
    # fraction of luminosity density contributed by quasars (Lbol > 10^45 erg/s) compared to all AGN
    qso_frac_all_agn = luminosityfunction.qso_luminosity_density(45, midz) / \
                       luminosityfunction.qso_luminosity_density(40, midz)
    # fraction of quasars which are in halos more massive than logminmass
    qso_frac_in_massive_halos = desiqso_massive_frac(midz, logminmass, logmaxmass=logmaxmass)
    # number density of halos more massive than logminmass
    halodens = luminosityfunction.hmf_zrange(zrange, logminmass, logmaxmass=logmaxmass)

    if power:
        # time elapsed in redshift interval
        tcosmic = (apcosmo.age(zrange[0]) - apcosmo.age(zrange[1])).to('s').value
    else:
        tcosmic = 1.

    # heat (power) per halo is
    # (heat/volume of all AGN) * (QSO/AGN luminosity density fraction) * (fraction of QSOs in halos > M_min)
    # per number of halos / (optionally divided by cosmic time interval for power)
    return np.log10(agn_windenergy_per_volume * qso_frac_all_agn * qso_frac_in_massive_halos / halodens / tcosmic)


#def jet_wind_power_ratio_frombias(zrangeqso, logminmass, rg_dndz_or_zrange, bias_and_err, kinetic_frac=0.005):
    #windpower = qso_windheatperhalo(zrange=zrangeqso, logminmass=logminmass, kinetic_frac=kinetic_frac)
    #heat_per_massive_halo_frombias(dndz_or_zrange=rg_dndz_or_zrange, bias_and_err=bias_and_err)

def jet_wind_power_ratio_hoddraws(zrangeqso, logminmass_cut, rg_dndz_or_zrange, hoddraws, f_or_l_min, f_or_l_max=None,
                                  lftype='agn', lumbound=True, min_kinfrac=0.001, max_kinfrac=0.005,
                                  logmaxmass=16.):
    heat, loerr, hierr = heat_per_massive_halo_from_hod_draws(dndz_or_zrange=rg_dndz_or_zrange, hoddraws=hoddraws,
                                         logminmass_cut=logminmass_cut, f_or_l_min=f_or_l_min, f_or_l_max=f_or_l_max,
                                         lftype=lftype, lumbound=lumbound, logmaxmass=logmaxmass)
    max_wind = qso_windheatperhalo(zrange=zrangeqso, logminmass=logminmass_cut, kinetic_frac=max_kinfrac,
                                   logmaxmass=logmaxmass)
    min_wind = qso_windheatperhalo(zrange=zrangeqso, logminmass=logminmass_cut, kinetic_frac=min_kinfrac,
                                   logmaxmass=logmaxmass)

    max_ratio = 10 ** (heat + hierr) / 10 ** min_wind
    min_ratio = 10 ** (heat - loerr) / 10 ** max_wind

    return min_ratio, max_ratio

def jet_wind_power_ratio_frombias(zrangeqso, logminmass_cut, rg_dndz_or_zrange, bias_and_err, sigM, f_or_l_min,
                                  f_or_l_max=None, lftype='agn', lumbound=True, min_kinfrac=0.001, max_kinfrac=0.005,
                                  logmaxmass=16.):
    heat, loerr, hierr = heat_per_massive_halo_frombias(dndz_or_zrange=rg_dndz_or_zrange, bias_and_err=bias_and_err,
                                                        sigM=sigM,
                                         logminmass_cut=logminmass_cut, f_or_l_min=f_or_l_min, f_or_l_max=f_or_l_max,
                                         lftype=lftype, lumbound=lumbound, logmaxmass=logmaxmass)
    max_wind = qso_windheatperhalo(zrange=zrangeqso, logminmass=logminmass_cut, kinetic_frac=max_kinfrac,
                                   logmaxmass=logmaxmass)
    min_wind = qso_windheatperhalo(zrange=zrangeqso, logminmass=logminmass_cut, kinetic_frac=min_kinfrac,
                                   logmaxmass=logmaxmass)

    max_ratio = 10 ** (heat + hierr) / 10 ** min_wind
    min_ratio = 10 ** (heat - loerr) / 10 ** max_wind

    return min_ratio, max_ratio

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