import pandas as pd
from SkyTools import fluxutils
import astropy.units as u
from halomodelpy import luminosityfunction
from halomodelpy import hubbleunits
import numpy as np


ragn1_13 = pd.read_csv('AGN_LF_Kondapally22-main/radio-excess_AGN/LF_radio_excess_AGN_1.0_z_1.3_tab.csv')
ragn13_17 = pd.read_csv('AGN_LF_Kondapally22-main/radio-excess_AGN/LF_radio_excess_AGN_1.3_z_1.7_tab.csv')
ragn17_21 = pd.read_csv('AGN_LF_Kondapally22-main/radio-excess_AGN/LF_radio_excess_AGN_1.7_z_2.1_tab.csv')
ragn21_25 = pd.read_csv('AGN_LF_Kondapally22-main/radio-excess_AGN/LF_radio_excess_AGN_2.1_z_2.5_tab.csv')
lfs = [ragn1_13, ragn13_17, ragn17_21, ragn21_25]
zcenters = np.array([1.15, 1.5, 1.9, 2.3])

def integrate_lf(l_grid, logl_tab, log_rho):
    # fill -infinity for right extrapolation, 10**-inf = 0
    logrho_interp = np.interp(l_grid, np.array(logl_tab), np.array(log_rho), right=-np.inf)
    return np.trapz(10**logrho_interp, x=l_grid)


def density_brighter_than_flux(fluxcut=2.):
    l150 = np.log10(fluxutils.luminosity_at_rest_nu(fluxcut, -0.7, .144, .15, zcenters, flux_unit=u.mJy, energy=False))
    dens = []
    for j in range(len(zcenters)):
        dens.append(hubbleunits.add_h_to_density(
            integrate_lf(np.linspace(l150[j], 30, 10), lfs[j]['logL'], lfs[j]['rho_agn'])))
    return np.array(dens)


def interp_lf_brighter(z, fluxcut=2.):
    dens = density_brighter_than_flux(fluxcut=fluxcut)
    return 10**np.interp(z, zcenters, np.log10(dens))

def occupation_frac(logminmass, z):
    dens = interp_lf_brighter(z)
    halodens = luminosityfunction.int_hmf(z, logminmass)
    focc = dens/halodens
    return focc