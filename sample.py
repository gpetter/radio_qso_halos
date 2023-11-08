from astropy.table import Table, vstack
from mocpy import MOC
import astropy.units as u
import pymangle
import numpy as np
import healpy as hp
from SkyTools import healpixhelper as myhp
from SkyTools import coordhelper
from SkyTools import fluxutils
from SkyTools import catalog_utils
from plotscripts import radio_lum

from colossus.cosmology import cosmology
cosmo = cosmology.setCosmology('planck18')
apcosmo = cosmo.toAstropy()

freq_dict = {'LoLSS_DR1': 54, 'LOTSS_DR2': 144, 'Apertif': 1355, 'FIRST': 1400, 'VLASS': 3000}
rms_dict = {'LOTSS_DR2': 0.15, 'FIRST': 0.15}

# Equation 1 in Hardcastle 2018 or Shimwell 2018
# determines whether radio source is resolved or not
def lofar_resolved(catalog):
	rms_map = hp.read_map('masks/LOTSS_DR2_rms.fits')
	rms = rms_map[hp.ang2pix(nside=hp.npix2nside(len(rms_map)), theta=catalog['RA'], phi=catalog['DEC'], lonlat=True)]
	resolved = catalog['F_144'] / catalog['PeakF_144'] > \
			   (1.25 + 3.1 * (catalog['PeakF_144'] / catalog['PeakFerr_144']) ** (-0.53))
	catalog['Resolved'] = np.zeros(len(catalog))
	catalog['Resolved'][np.where(resolved)] = 1.
	return catalog

def lotss_dr2resolved(catalog):
	catalog['Resolved'] = np.zeros(len(catalog))
	resolved = np.log(catalog['F_144'] / catalog['PeakF_144']) > 0.42 + (1.08 /
									(1 + ((catalog['F_144'] / catalog['Ferr_144'])/96.57) ** 2.49))
	catalog['Resolved'][resolved] = 1.
	return catalog

def physical_size(angsizes, zs):
	from colossus.cosmology import cosmology
	cosmo = cosmology.setCosmology('planck18')
	apcosmo = cosmo.toAstropy()
	ang_diam_dists = apcosmo.angular_diameter_distance(zs).to(u.kpc).value
	phys_sizes = (angsizes * u.arcsec).to(u.rad) * ang_diam_dists
	return phys_sizes

def stardensmap():
	pixweight = Table.read('../data/desi_targets/syst_maps/pixweight-1-dark.fits')

	stars = np.empty(hp.nside2npix(256))
	stars[hp.nest2ring(256, pixweight['HPXPIXEL'])] = pixweight['STARDENS']
	return stars

def in_cmblensingmask(ras, decs):
	lensmask = hp.read_map('/home/graysonpetter/ssd/Dartmouth/data/lensing_maps/PlanckPR4/derived/1024/mask.fits')
	return lensmask[hp.ang2pix(nside=hp.npix2nside(len(lensmask)), theta=ras, phi=decs, lonlat=True)] == 1


def in_lotss_dr2(ras, decs, galcut=20, northonly=True):
	moc = MOC.from_fits('/home/graysonpetter/ssd/Dartmouth/data/radio_cats/LOTSS_DR2/lotss_dr2_hips_moc.fits')
	inmoc = moc.contains(np.array(ras) * u.deg, np.array(decs) * u.deg)

	rmsmap = hp.read_map('masks/LOTSS_DR2_rms.fits')
	rms = rmsmap[hp.ang2pix(nside=hp.npix2nside(len(rmsmap)), theta=ras, phi=decs, lonlat=True)]
	goodrms = rms < 0.2 # mJy

	l, b = coordhelper.equatorial_to_galactic(ras, decs)
	goodbs = np.abs(b) > galcut
	good_idxs = inmoc & goodrms & goodbs
	if northonly:
		innorth = (ras > 90) & (ras < 290)
		good_idxs = good_idxs & innorth


	return good_idxs

def cat_in_lotss(cat):
	return cat[in_lotss_dr2(cat['RA'], cat['DEC'])]


def in_ls_dr8(ra, dec):
	import astropy.units as u
	northfoot = MOC.load('../data/footprints/legacySurvey/dr8_photoz_duncan/desi_lis_dr8_pzn.rcf.moc.fits')
	southfoot = MOC.load('../data/footprints/legacySurvey/dr8_photoz_duncan/desi_lis_dr8_pzs.rcf.moc.fits')
	inbad = (ra < 243.4) & (ra > 242) & (dec < 35.3) & (dec > 34.2)
	infoot = (northfoot.contains(ra * u.deg, dec * u.deg) | southfoot.contains(ra * u.deg,
																			   dec * u.deg)) & np.logical_not(inbad)

	return infoot

def cat_in_ls_dr8(t):
	infoot = in_ls_dr8(t['RA'], t['DEC'])
	return t[infoot]


def in_eboss(ras, decs, northonly=True):

	ebossmoc = pymangle.Mangle('../data/footprints/eBOSS/eBOSS_QSOandLRG_fullfootprintgeometry_noveto.ply')
	good_idxs = ebossmoc.contains(ras, decs)
	if northonly:
		innorth = (ras > 90) & (ras < 290)
		good_idxs = good_idxs & innorth
	return good_idxs

def cat_in_eboss(cat):
	return cat[in_eboss(cat['RA'], cat['DEC'])]



def in_boss(ras, decs):

	bossmoc = pymangle.Mangle('../data/footprints/BOSS/bosspoly.ply')
	ngcmoc = pymangle.Mangle('../data/footprints/BOSS/geometry/boss_survey_ngc_good2.ply')
	badfield = pymangle.Mangle('../data/footprints/BOSS/geometry/badfield_mask_postprocess_pixs8.ply')
	badphot = pymangle.Mangle('../data/footprints/BOSS/geometry/badfield_mask_unphot-ugriz_pix.ply')
	star = pymangle.Mangle('../data/footprints/BOSS/geometry/allsky_bright_star_mask_pix.ply')
	good_idxs = bossmoc.contains(ras, decs) & ngcmoc.contains(ras, decs) & \
				(np.logical_not(badfield.contains(ras, decs))) & (np.logical_not(badphot.contains(ras, decs))) & \
				(np.logical_not(star.contains(ras, decs)))
	return good_idxs


def cat_in_boss(cat):
	return cat[in_boss(cat['RA'], cat['DEC'])]

def in_desi_elg_foot(ras, decs):
	elgrand = Table.read('/home/graysonpetter/ssd/Dartmouth/data/lss/desiELG_edr/desiELG_edr_randoms.fits')
	dens = myhp.healpix_density_map(elgrand['RA'], elgrand['DEC'], nsides=32)
	dens[np.where(dens > 0)] = 1
	return myhp.inmask((ras, decs), dens, return_bool=True)

def cat_in_desi_elg(cat):
	return cat[in_desi_elg_foot(cat['RA'], cat['DEC'])]

def in_vlass_epoch2(ras, decs):
	moc1 = MOC.from_fits('/home/graysonpetter/ssd/Dartmouth/data/footprints/VLASS/vlass_2.1.moc.fits')
	moc2 = MOC.from_fits('/home/graysonpetter/ssd/Dartmouth/data/footprints/VLASS/vlass_2.2.moc.fits')

	good_idxs = moc1.contains(np.array(ras) * u.deg, np.array(decs) * u.deg) | \
				moc2.contains(np.array(ras) * u.deg, np.array(decs) * u.deg)
	return good_idxs

def in_goodwise(ras, decs):
	pixweight = Table.read('/home/graysonpetter/ssd/Dartmouth/data/desi_targets/syst_maps/pixweight-1-dark.fits')
	import healpy as hp
	w2depth = np.empty(hp.nside2npix(256))
	w2depth[hp.nest2ring(256, pixweight['HPXPIXEL'])] = pixweight['PSFDEPTH_W2']
	w2depth = 22.5 - 2.5 * np.log10(5 / np.sqrt(w2depth)) - 3.339
	w2depth[np.where(np.logical_not(np.isfinite(w2depth)))] = 100.
	depths = w2depth[hp.ang2pix(256, ras, decs, lonlat=True)]

	lams, betas = coordhelper.equatorial_to_ecliptic(ras, decs)

	return (depths > 17.3) & (betas < 75)

def cat_in_goodwise(cat):
	return cat[in_goodwise(cat['RA'], cat['DEC'])]

def in_goodclustering_area(ra, dec):
	return in_lotss_dr2(ra, dec) & in_goodwise(ra, dec) & in_ls_dr8(ra, dec)

def cat_in_goodclustering_area(cat):
	cat = cat_in_lotss(cat)
	cat = cat_in_goodwise(cat)
	cat = cat_in_ls_dr8(cat)
	return cat


def lotss_randoms():
	from SkyTools import random_catalogs
	randra, randdec = random_catalogs.uniform_sphere(1000, density=True, lat_range=(15, 90))
	rand = Table()
	rand['RA'] = randra
	rand['DEC'] = randdec
	rand['weight'] = np.ones(len(rand))
	rand = cat_in_goodclustering_area(rand)
	return rand

def rg_mask(nside):
	mask = np.zeros(hp.nside2npix(nside))
	l, b = hp.pix2ang(nside, np.arange(len(mask)), lonlat=True)
	ra, dec = coordhelper.galactic_to_equatorial(l, b)
	idx = in_goodclustering_area(ra, dec)
	mask[idx] = 1.
	return mask


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
		#qsocat = qsocat[np.where(in_mocs)]
		#rmsmask = hp.read_map('masks/LOTSS_DR2_noise_mask.fits')
		#goodrms = rmsmask[hp.ang2pix(hp.npix2nside(len(rmsmask)), ras, decs, lonlat=True)]
	
	if 'LoLSS_DR1' in radio_names:
		moc = MOC.from_fits('../data/radio_cats/LoLSS_DR1/Moc.fits')
		good_idxs = moc.contains(np.array(ras) * u.deg, np.array(decs) * u.deg)
	#qsocat = qsocat[np.where(in_mocs)]
	#rmsmask = hp.read_map('masks/LOTSS_DR2_noise_mask.fits')
	#goodrms = rmsmask[hp.ang2pix(hp.npix2nside(len(rmsmask)), ras, decs, lonlat=True)]

		#good_idxs = good_idxs * goodrms
	
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



def radio_match(qsosample, radio_names, sep, alpha_fid=-0.8):
	#qsocat = mask_sample(qsosample, radio_names)
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
		#qsocat['angsize_%s' % radio_freq][qsoidx] = radiocat['Maj'][radidx]
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
				#lofar_resolved(qsocat)
				qsocat['PeakF_%s' % freq_dict[radio_name]] = np.full(len(qsocat), np.nan)
				qsocat['PeakFerr_%s' % freq_dict[radio_name]] = np.full(len(qsocat), np.nan)
				qsocat['PeakF_%s' % radio_freq][qsoidx] = radiocat['Peak_flux'][radidx]
				qsocat['PeakFerr_%s' % radio_freq][qsoidx] = radiocat['E_Peak_flux'][radidx]
				qsocat['Phys_size_144'] = np.full(len(qsocat), np.nan)
				qsocat['Phys_size_144'][qsoidx] = physical_size(qsocat['angsize_144'][qsoidx], qsocat['Z'][qsoidx])
				#qsocat = lofar_resolved(qsocat)
				qsocat = lotss_dr2resolved(qsocat)

				#qsocat['L_144'] = np.log10(150. * 10**6 * (qsocat['F_LOTSS_DR2'] * u.mJy * 4 * np.pi * (lum_dists * (1 + qsocat['Z']) ** (-(alpha_fid + 1) / 2.)) ** 2).to(u.erg).value)
				#s_1400_lotss = qsocat['F_LOTSS_DR2'] * (1400./150.) ** alpha_fid
				#qsocat['L_1400_lotss'] = np.log10(1.4e9 * (s_1400_lotss * u.mJy * 4 * np.pi * (lum_dists * (1 + qsocat['Z']) ** (-(alpha_fid + 1) / 2.)) ** 2).to(u.erg).value)
				qsocat['L_144'] = np.log10(fluxutils.luminosity_at_rest_nu(qsocat['F_144'], alpha=alpha_fid,
																  nu_obs=.144, nu_rest_want=.144,
																  z=qsocat['Z'], flux_unit=u.mJy))

				qsocat['SFR_lotss'] = fluxutils.best23_lum150_2sfr(np.log10(fluxutils.luminosity_at_rest_nu(qsocat['F_144'], alpha=alpha_fid,
																  nu_obs=.144, nu_rest_want=.144,
																  z=qsocat['Z'], flux_unit=u.mJy, energy=False)))
				qsocat['L_1400_lotss'] = np.log10(fluxutils.luminosity_at_rest_nu(qsocat['F_144'], alpha=alpha_fid,
																  nu_obs=.144, nu_rest_want=1.4,
																  z=qsocat['Z'], flux_unit=u.mJy))


			if radio_name == 'FIRST':
				#qsocat['L_1400_fid'] = np.log10(1.4e9 * (qsocat['F_FIRST'] * u.mJy * 4 * np.pi * (lum_dists * (1 + qsocat['Z']) ** (-(alpha_fid + 1) / 2.)) ** 2).to(u.erg).value)
				qsocat['L_1400'] = np.log10(fluxutils.luminosity_at_rest_nu(qsocat['F_1400'], alpha=alpha_fid,
																				  nu_obs=1.4, nu_rest_want=1.4,
																				  z=qsocat['Z'], flux_unit=u.mJy))
				#freq1, freq2 = freq_dict['LOTSS_DR2'], freq_dict['FIRST']
				#qsocat['alpha_%s_%s' % (freq1, freq2)] = np.log10(qsocat['F_%s' % freq2] / qsocat['F_%s' % freq1]) / np.log10(float(freq2)/float(freq1))
			if radio_name == 'VLASS':
				#qsocat['L_3000_fid'] = np.log10(3.e9 * (qsocat['F_VLASS'] * u.mJy * 4 * np.pi * (lum_dists * (1 + qsocat['Z']) ** (-(alpha_fid + 1) / 2.)) ** 2).to(u.erg).value)
				#s_1400_vlass = qsocat['F_VLASS'] * (1400./3000.) ** alpha_fid
				#qsocat['L_1400_vlass'] = np.log10(1.4e9 * (s_1400_vlass * u.mJy * 4 * np.pi * (lum_dists * (1 + qsocat['Z']) ** (-(alpha_fid + 1) / 2.)) ** 2).to(u.erg).value)
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
					np.log10(float(freq2)/float(freq1))




	qsocat.write('catalogs/masked/%s.fits' % qsosample, overwrite=True)


def lotss_deep_match(qsosample, sep, alpha_fid=-0.8):
	lockman_ctr = (161.75, 58.083)
	lockmanmask = myhp.mask_from_pointings(([lockman_ctr[0]], [lockman_ctr[1]]), 2048, pointing_radius=2.)
	highresmask = myhp.mask_from_pointings(([lockman_ctr[0]], [lockman_ctr[1]]), 2048, pointing_side=2.)

	radiocat = Table.read('../data/radio_cats/lofar_deep/lockman_pybdsf_source.fits')
	highrescat = Table.read('../data/radio_cats/lofar_highres/lofar_highres.fits')
	highrescat = highrescat[myhp.inmask((highrescat['RA'], highrescat['DEC']), highresmask)]

	highresrmsmap = myhp.healpix_average_in_pixels(highrescat['RA'], highrescat['DEC'], nsides=256,
												   values=np.log10(1000.*highrescat['E_Peak_flux']))
	highresrmsmap = 10**highresrmsmap
	
	#qsocat = mask_sample(qsosample, radio_names)
	qsocat = Table.read('catalogs/masked/%s.fits' % qsosample)
	qsocat = qsocat[myhp.inmask((qsocat['RA'], qsocat['DEC']), lockmanmask)]

	qsocat['det_deep'] = np.zeros(len(qsocat))
	qsocat['det_highres'] = np.full(len(qsocat), np.nan)
	qsocat['F_ILT'] = np.full(len(qsocat), np.nan)
	qsocat['Ferr_ILT'] = np.full(len(qsocat), np.nan)
	qsocat['Peak_ILT'] = np.full(len(qsocat), np.nan)
	qsocat['Peakerr_ILT'] = np.full(len(qsocat), np.nan)
	qsocat['det_highres'][myhp.inmask((qsocat['RA'], qsocat['DEC']), highresmask)] = 0.

	#qsocat['Peak_%s' % freq_dict[radio_name]] = np.full(len(qsocat), np.nan)

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

def overlap_weights_1d(prop_arr, prop_range, nbins):
	hists = []
	for j in range(len(prop_arr)):
		hist, binedges = np.histogram(prop_arr[j], bins=nbins, range=prop_range, density=True)
		hists.append(hist)
	hists = np.array(hists)
	min_hist = np.amin(hists, axis=0)
	weights = []
	for j in range(len(prop_arr)):
		dist_ratio = min_hist / hists[j]
		dist_ratio[np.where(np.isnan(dist_ratio) | np.isinf(dist_ratio))] = 0
		weights.append(dist_ratio[np.digitize(prop_arr[j], bins=binedges)-1])
	return weights



def overlap_weights_2d(prop1_arr, prop2_arr, prop1_range, prop2_range, nbins):
	from scipy import stats, interpolate
	hists, bin_locs = [], []

	for j in range(len(prop1_arr)):
		thishist = stats.binned_statistic_2d(prop1_arr[j], prop2_arr[j], None, statistic='count',
											 bins=[nbins[0], nbins[1]],
											 range=[[prop1_range[0] - 0.001, prop1_range[1] + 0.001],
													[prop2_range[0] - 0.001, prop2_range[1] + 0.001]],
											 expand_binnumbers=True)
		normed_hist = thishist[0] / np.sum(thishist[0])
		hists.append(normed_hist)
		bin_locs.append(np.array(thishist[3]) - 1)

	hists = np.array(hists)
	min_hist = np.amin(hists, axis=0)
	weights = []

	for j in range(len(prop1_arr)):
		dist_ratio = min_hist / hists[j]
		dist_ratio[np.where(np.isnan(dist_ratio) | np.isinf(dist_ratio))] = 0
		bin_idxs = bin_locs[j]
		weights.append(dist_ratio[bin_idxs[0], bin_idxs[1]])
	return weights


def match_random_zdist(samplecat, randomcat):
	zhist, edges = np.histogram(samplecat['Z'], bins=15, range=(0.7, 2.3), density=True)
	randhist, edges = np.histogram(randomcat['Z'], bins=15, range=(0.7, 2.3), density=True)
	ratio = zhist/randhist
	sampleweights = ratio[np.digitize(randomcat['Z'], bins=edges) - 1]
	subrandcat = randomcat[np.random.choice(len(randomcat), size=(10*len(samplecat)), replace=False, p=(sampleweights / np.sum(sampleweights)))]
	return subrandcat

def weight_common_dndz(cat1, cat2, zbins=15):
	minz1, maxz1 = np.min(cat1['Z']), np.max(cat2['Z'])
	minz2, maxz2 = np.min(cat2['Z']), np.max(cat2['Z'])
	minz = np.min([minz1, minz2])
	maxz = np.max([maxz1, maxz2])
	zhist1, edges = np.histogram(cat1['Z'], bins=zbins, range=(minz, maxz), density=True)
	zhist2, edges = np.histogram(cat2['Z'], bins=zbins, range=(minz, maxz), density=True)
	dndz_product = np.sqrt(zhist1 * zhist2)
	ratio1 = zhist1/dndz_product
	ratio2 = zhist2/dndz_product
	cat1['zweight'] = ratio1[np.digitize(cat1['Z'], bins=edges) - 1]
	cat2['zweight'] = ratio2[np.digitize(cat2['Z'], bins=edges) - 1]
	cat1['weight'] *= cat1['zweight']
	cat2['weight'] *= cat1['zweight']
	return cat1, cat2


def select_radioloud(radio_name, fcut, cat, randcat, detect_thresh=6.):
	freq = freq_dict[radio_name]
	cat = cat[np.where(cat['det_%s' % freq] > -1)]
	radiocat = cat[np.where((cat['F_%s' % freq] > fcut) & (cat['det_%s' % freq] == 1))]
	radioquiet = cat[np.where((cat['det_%s' % freq] == 0) | (cat['F_%s' % freq] < fcut))]
	randcat = randcat[np.where(randcat['in_%s' % freq] == 1)]
	rmsmap = hp.read_map('masks/%s_rms.fits' % radio_name)
	cat = cat[np.where(myhp.coords2mapvalues(cat['RA'], cat['DEC'], rmsmap) < (fcut / detect_thresh))]
	radiocat = radiocat[np.where(myhp.coords2mapvalues(radiocat['RA'], radiocat['DEC'], rmsmap) <
								 (fcut / detect_thresh))]
	randcat = randcat[np.where(myhp.coords2mapvalues(randcat['RA'], randcat['DEC'], rmsmap) < (fcut / detect_thresh))]
	return radiocat, radioquiet, cat, randcat


def rolling_percentile_selection(cat, prop, minpercentile, maxpercentile=100, nzbins=100):
	"""
	choose highest nth percentile of e.g. luminosity in bins of redshift
	"""
	minz, maxz = np.min(cat['Z']), np.max(cat['Z'])
	zbins = np.linspace(minz, maxz, nzbins)
	idxs = []
	for j in range(len(zbins)-1):
		catinbin = cat[np.where((cat['Z'] > zbins[j]) & (cat['Z'] <= zbins[j+1]))]
		thresh = np.percentile(catinbin[prop], minpercentile)
		hithresh = np.percentile(catinbin[prop], maxpercentile)
		idxs += list(np.where((cat[prop] > thresh) & (cat[prop] <= hithresh) &
							  (cat['Z'] > zbins[j]) & (cat['Z'] <= zbins[j+1]))[0])

	newcat = cat[np.array(idxs)]
	return newcat


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

def qsocat(eboss=True, boss=False):
	if eboss:
		dat = Table.read('catalogs/masked/eBOSS_QSO.fits')
		rand = Table.read('catalogs/masked/eBOSS_QSO_randoms.fits')
		if boss:
			dat = vstack((dat, Table.read('catalogs/masked/BOSS_QSO.fits')))
			rand = vstack((rand, Table.read('catalogs/masked/BOSS_QSO_randoms.fits')))
	else:
		dat = Table.read('catalogs/masked/BOSS_QSO.fits')
		rand = Table.read('catalogs/masked/BOSS_QSO_randoms.fits')
	return dat, rand

def eboss_qso(minz=None, maxz=None):
	#dat = Table.read('catalogs/masked/eBOSS_QSO.fits')
	#rand = Table.read('catalogs/masked/eBOSS_QSO_randoms.fits')
	dat = Table.read('../data/lss/eBOSS_QSO/eBOSS_QSO.fits')
	rand = Table.read('../data/lss/eBOSS_QSO/eBOSS_QSO_randoms.fits')
	if minz is not None:
		dat = dat[np.where(dat['Z'] > minz)]
		rand = rand[np.where(rand['Z'] > minz)]
	if maxz is not None:
		dat = dat[np.where(dat['Z'] < maxz)]
		rand = rand[np.where(rand['Z'] < maxz)]
	return dat, rand

def boss_qso(minz=None, maxz=None):
	dat = Table.read('catalogs/masked/BOSS_QSO.fits')
	rand = Table.read('catalogs/masked/BOSS_QSO_randoms.fits')
	if minz is not None:
		dat = dat[np.where(dat['Z'] > minz)]
		rand = rand[np.where(rand['Z'] > minz)]
	if maxz is not None:
		dat = dat[np.where(dat['Z'] < maxz)]
		rand = rand[np.where(rand['Z'] < maxz)]
	return dat, rand

def boss_gals(minz=None, maxz=None):
	"""
	BOSS CMASS + LOWZ galaxies and randoms
	"""
	dat = Table.read('../data/lss/BOSStot/BOSStot.fits')
	rand = Table.read('../data/lss/BOSStot/BOSStot_randoms.fits')
	if minz is not None:
		dat = dat[np.where(dat['Z'] > minz)]
		rand = rand[np.where(rand['Z'] > minz)]
	if maxz is not None:
		dat = dat[np.where(dat['Z'] < maxz)]
		rand = rand[np.where(rand['Z'] < maxz)]
	return dat, rand

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





def hzrg_sample(fcut=2., sep_cw=5, yint=0.15, sep_2mass=5, jcut=16, w2faint=17.5, maxflux=1000, majmax=30, zphotcut=0.5):
	from SkyTools import fluxutils
	lotss = Table.read('catalogs/LoTSS.fits')
	lotss = cat_in_goodclustering_area(lotss)
	# cut local zphot detections
	lotss = lotss[np.where((lotss['zphot'] > zphotcut) | (np.logical_not(np.isfinite(lotss['zphot']))))]

	lotss = lotss[np.where(lotss['Total_flux'] < maxflux)]
	lotss = lotss[np.where(lotss['Maj'] < majmax)]
	#lotss = lotss[np.where((lotss['Jmag'].mask == True) | (np.array(lotss['Jmag'], dtype=float) > jcut) | (lotss['sep_2mass'] > sep_2mass))]
	# detected in WISE to avoid lobes
	lotss = lotss[np.where((lotss['sep_cw'] < sep_cw))]
	lotss = lotss[np.where((lotss['W2_cw'] < w2faint))]

	#lotss = lotss[np.where(((lotss['W1_cw'] - lotss['W2_cw']) > (5.25 - 0.3 * lotss['W2_cw'])))]
	lotss = lotss[np.where((lotss['W1_cw'] - lotss['W2_cw']) > ((17 - lotss['W2_cw'])/4.+yint))]

	lotss.remove_columns(['RA', 'DEC'])
	lotss.rename_columns(['RA_cw', 'DEC_cw'], ['RA', 'DEC'])

	#lotss = lotss[np.where( (lotss['W1_cw'] - lotss['W2_cw'] > colorcut) | (lotss['W1_cw'] > w1cut) |
	# (lotss['W1_cw'].mask == True) | (lotss['W2_cw'].mask == True))]
	lotss['r75'] = np.array(fluxutils.r75_assef(lotss['W1_cw'], lotss['W2_cw']), dtype=int)
	lotss['r90'] = np.array(fluxutils.r90_assef(lotss['W1_cw'], lotss['W2_cw']), dtype=int)
	lotss = lotss[lotss['Total_flux'] > fcut]
	lotss['weight'] = np.ones(len(lotss))
	return lotss

def izrg_sample(fcut=5., sep_cw=5,  w2faint=17.5, maxflux=1000, majmax=30):
	lotss = Table.read('catalogs/LoTSS.fits')
	lotss = cat_in_goodclustering_area(lotss)
	lotss = lotss[np.where(lotss['Total_flux'] < maxflux)]
	lotss = lotss[np.where(lotss['Maj'] < majmax)]
	# detected in WISE to avoid lobes
	lotss = lotss[np.where((lotss['sep_cw'] < sep_cw))]
	lotss = lotss[np.where((lotss['W2_cw'] < w2faint))]

	lotss = lotss[np.where((lotss['W1_cw'] - lotss['W2_cw']) < ((17 - lotss['W2_cw']) / 4. + 0.15))]
	lotss = lotss[np.where((lotss['W1_cw'] - lotss['W2_cw']) < ((lotss['W2_cw'] - 17) / 3. + 0.75))]

	lotss.remove_columns(['RA', 'DEC'])
	lotss.rename_columns(['RA_cw', 'DEC_cw'], ['RA', 'DEC'])

	lotss = lotss[lotss['Total_flux'] > fcut]
	lotss['weight'] = np.ones(len(lotss))
	return lotss

def lzrg_sample(fcut=20., sep_cw=7.,  w2faint=17.5, maxflux=2000, majmax=45):
	lotss = Table.read('catalogs/LoTSS.fits')
	lotss = cat_in_goodclustering_area(lotss)
	lotss = lotss[np.where(lotss['Total_flux'] < maxflux)]
	lotss = lotss[np.where(lotss['Maj'] < majmax)]
	# detected in WISE to avoid lobes
	lotss = lotss[np.where((lotss['sep_cw'] < sep_cw))]
	lotss = lotss[np.where((lotss['W2_cw'] < w2faint))]

	lotss = lotss[np.where((lotss['W1_cw'] - lotss['W2_cw']) < ((17 - lotss['W2_cw']) / 4. - 0.15))]
	lotss = lotss[np.where((lotss['W1_cw'] - lotss['W2_cw']) > ((lotss['W2_cw'] - 17) / 3. + 0.65))]
	lotss = lotss[np.where((lotss['W1_cw'] - lotss['W2_cw']) < ((lotss['W2_cw'] - 17) / 3. + 1.15))]

	lotss.remove_columns(['RA', 'DEC'])
	lotss.rename_columns(['RA_cw', 'DEC_cw'], ['RA', 'DEC'])

	lotss = lotss[lotss['Total_flux'] > fcut]
	lotss['weight'] = np.ones(len(lotss))
	return lotss

def lzrg_zphot_sample(lcut=25, minz=0.25, maxz=0.5, maxflux=2000.):
	lotss = Table.read('catalogs/LoTSS.fits')
	lotss = cat_in_goodclustering_area(lotss)
	lotss = lotss[np.where(lotss['Total_flux'] < maxflux)]
	lotss = lotss[np.where((lotss['zphot'] > minz) & (lotss['zphot'] < maxz))]
	lotss = lotss[np.where(lotss['L150'] > lcut)]

	return lotss



def noz_rg_sample(fcut=2., sep_cw=7, w2cut=17.5, maxflux=1000, majmax=15):
	from SkyTools import fluxutils
	lotss = Table.read('catalogs/LoTSS.fits')
	lotss = cat_in_goodclustering_area(lotss)
	lotss = lotss[np.where(lotss['Total_flux'] < maxflux)]
	lotss = lotss[np.where(lotss['Maj'] < majmax)]
	#lotss = lotss[np.where((lotss['Jmag'].mask == True) | (np.array(lotss['Jmag'], dtype=float) > jcut) | (lotss['sep_2mass'] > sep_2mass))]

	lotss = lotss[np.where((lotss['sep_cw'] > sep_cw) | (np.logical_not(np.isfinite(lotss['sep_cw']))))]
	lotss = lotss[np.where((lotss['W2_cw'] > w2cut) | (np.logical_not(np.isfinite(lotss['W2_cw']))))]

	lotss = lotss[lotss['Total_flux'] > fcut]
	lotss['weight'] = np.ones(len(lotss))
	return lotss


def match2bootes(cat, sep):
	bootes = Table.read('/home/graysonpetter/ssd/Dartmouth/data/radio_cats/LoTSS_deep/classified/bootes.fits')
	bootidx, catidx = coordhelper.match_coords((bootes['RA'], bootes['DEC']), (cat['RA'], cat['DEC']),
											   max_sep=sep, symmetric=False)
	bootes = bootes[bootidx]
	return bootes



def rg_redshifts():
	from SkyTools import get_redshifts
	rgs = lotss_rg_sample()
	rgs = get_redshifts.match_to_spec_surveys(rgs, seplimit=3)
	rgs.write('catalogs/rgs_specz.fits', overwrite=True)

def treat_dndz_pdf(cat, sep=3, ndraws=100):
	bootes = match2bootes(cat, sep=sep)
	specs = bootes[np.where(bootes['f_zbest'] == 1)]
	phots = bootes[np.where(bootes['f_zbest'] == 0)]

	singlephots = phots[np.where(phots['z2med'] == -99)]
	doublephots = phots[np.where(phots['z2med'] > 0)]
	maxpossiblez = 3.5

	speczs = list(np.repeat(specs['z_best'], ndraws))
	#specweights = list(ndraws*np.ones_like(speczs))

	singlezs, singleweights = [], []
	for j in range(len(singlephots)):
		thisrow = singlephots[j]
		uperr = (thisrow['z1max'] - thisrow['z1med']) / 1.3
		loerr = (thisrow['z1med'] - thisrow['z1min']) / 1.3
		weight = thisrow['z1area']

		b = np.random.normal(loc=thisrow['z1med'], scale=uperr, size=ndraws)
		above = b[np.where(b > thisrow['z1med'])]
		c = np.random.normal(loc=thisrow['z1med'], scale=loerr, size=ndraws)
		below = c[np.where((c > 0) & (c < thisrow['z1med']))]
		tot = np.concatenate((above, below))
		try:
			singlezs += list(np.random.choice(tot, int(weight * ndraws), replace=False))
		except:
			pass


	doublezs1, doublezs2, doubleweights1, doubleweights2 = [], [], [], []
	for j in range(len(doublephots)):
		thisrow = doublephots[j]
		uperr = (thisrow['z1max'] - thisrow['z1med']) / 1.3
		loerr = (thisrow['z1med'] - thisrow['z1min']) / 1.3
		weight = thisrow['z1area']
		if thisrow['z1med'] < maxpossiblez:

			b = np.random.normal(loc=thisrow['z1med'], scale=uperr, size=ndraws)
			above = b[np.where(b > thisrow['z1med'])]
			c = np.random.normal(loc=thisrow['z1med'], scale=loerr, size=ndraws)
			below = c[np.where((c > 0) & (c < thisrow['z1med']))]
			tot1 = np.concatenate((above, below))
			try:
				doublezs1 += list(np.random.choice(tot1, int(weight * ndraws), replace=False))
			except:
				pass

		uperr = (thisrow['z2max'] - thisrow['z2med']) / 1.3
		loerr = (thisrow['z2med'] - thisrow['z2min']) / 1.3
		weight = thisrow['z2area']

		if thisrow['z2med'] < maxpossiblez:

			b = np.random.normal(loc=thisrow['z2med'], scale=uperr, size=ndraws)
			above = b[np.where(b > thisrow['z2med'])]
			c = np.random.normal(loc=thisrow['z2med'], scale=loerr, size=ndraws)
			below = c[np.where((c > 0) & (c < thisrow['z2med']))]
			tot2 = np.concatenate((above, below))
			try:
				doublezs2 += list(np.random.choice(tot2, int(weight * ndraws), replace=False))
			except:
				pass


	finalzs = np.array(speczs + singlezs + doublezs1 + doublezs2)
	return finalzs

def redshift_dist(cat, sep, bootesonly=True):
	if bootesonly:
		deepcat = Table.read('/home/graysonpetter/ssd/Dartmouth/data/radio_cats/LoTSS_deep/classified/bootes.fits')
	else:
		deepcat = Table.read('/home/graysonpetter/ssd/Dartmouth/data/radio_cats/LoTSS_deep/classified/combined.fits')
	bootidx, catidx = coordhelper.match_coords((deepcat['RA'], deepcat['DEC']), (cat['RA'], cat['DEC']),
											   max_sep=sep, symmetric=False)
	deepcat = deepcat[bootidx]
	deepcat = deepcat[np.where(deepcat['z_best'] < 4)]
	zs = deepcat['z_best']
	#speczs = bootes['Z'][np.where(bootes['Z'] > 0)]
	return zs

def spline_dndz(dndz, n_newzs, spline_k=4, smooth=0.05):
	from halomodelpy import redshift_helper
	smooth_dndz = redshift_helper.spline_dndz(dndz, n_newzs=n_newzs, spline_k=spline_k, smooth=smooth)
	return smooth_dndz


def tomographer_dndz(which):
	import pandas as pd
	if which == 'hi':
		nz = pd.read_csv('results/tomographer/ddz_hzrg.csv')
	elif which == 'mid':
		nz = pd.read_csv('results/tomographer/ddz_izrg.csv')
	else:
		nz = pd.read_csv('results/tomographer/ddz_lzrg.csv')
	zs, dndz_b, dndz_berr = nz['z'], nz['dNdz_b'], nz['dNdz_b_err']

	bconst_int = np.trapz(dndz_b, zs)
	bconst_dndz = dndz_b / bconst_int
	bconst_dndz_err = dndz_berr / bconst_int

	dndz_growth = dndz_b * cosmo.growthFactor(zs)

	dndz_growth_err = dndz_berr * cosmo.growthFactor(zs)



	growth_int = np.trapz(dndz_growth, zs)
	dndz_growth /= growth_int
	dndz_growth_err /= growth_int

	return zs, dndz_growth, dndz_growth_err


def estimate_rg_magbias(fluxcut, magdiff=0.1, nboots=100):
	lotss_liberal = lotss_rg_sample(fluxcut * .5)
	mags = -2.5 * np.log10(lotss_liberal['Total_flux'])
	magcut = -2.5 * np.log10(fluxcut)

	ncut = len(np.where(mags < magcut)[0])

	n_brighter = len(np.where(mags < (magcut - magdiff))[0])
	n_fainter = len(np.where(mags < (magcut + magdiff))[0])

	dlogn_dm_1 = (np.log10(n_brighter) - np.log10(ncut)) / (-magdiff)
	dlogn_dm_2 = (np.log10(n_fainter) - np.log10(ncut)) / (magdiff)

	smu = np.mean([dlogn_dm_1, dlogn_dm_2])

	smus = []
	for j in range(nboots):

		lotss_boot = lotss_liberal[np.random.choice(len(lotss_liberal), len(lotss_liberal), replace=True)]
		mags = -2.5 * np.log10(lotss_boot['Total_flux'])
		ncut = len(np.where(mags < magcut)[0])

		n_brighter = len(np.where(mags < (magcut - magdiff))[0])
		n_fainter = len(np.where(mags < (magcut + magdiff))[0])
		dlogn_dm_1 = (np.log10(n_brighter) - np.log10(ncut)) / (-magdiff)
		dlogn_dm_2 = (np.log10(n_fainter) - np.log10(ncut)) / (magdiff)

		smus.append(np.mean([dlogn_dm_1, dlogn_dm_2]))
	return smu, np.std(smus)







def maketomographer_files(nside=128, whichsample='hi'):
	if whichsample == "hi":
		rgs = hzrg_sample()
	elif whichsample == "mid":
		rgs = izrg_sample()
	else:
		rgs = lzrg_sample()
	mask = np.zeros(hp.nside2npix(nside))
	l, b = hp.pix2ang(nside, np.arange(len(mask)), lonlat=True)
	ra, dec = coordhelper.galactic_to_equatorial(l, b)
	inmoc = in_lotss_dr2(ra, dec)
	mask[inmoc] = 1
	rgs = rgs['RA', 'DEC']
	rgs.write('results/tomographer/lotss_rg.fits', overwrite=True)
	hp.write_map('results/tomographer/mask.fits', mask, overwrite=True)