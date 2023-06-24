from astropy.table import Table, vstack
from mocpy import MOC
import astropy.units as u
from astropy.coordinates import SkyCoord
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

def in_lotss_dr2(ras, decs):
	moc = MOC.from_fits('/home/graysonpetter/ssd/Dartmouth/data/radio_cats/LOTSS_DR2/lotss_dr2_hips_moc.fits')
	good_idxs = moc.contains(np.array(ras) * u.deg, np.array(decs) * u.deg)
	return good_idxs

def in_eboss(ras, decs):
	import pymangle
	ebossmoc = pymangle.Mangle('../data/footprints/eBOSS/eBOSS_QSOandLRG_fullfootprintgeometry_noveto.ply')
	good_idxs = ebossmoc.contains(ras, decs)
	return good_idxs

def in_vlass_epoch2(ras, decs):
	moc1 = MOC.from_fits('/home/graysonpetter/ssd/Dartmouth/data/footprints/VLASS/vlass_2.1.moc.fits')
	moc2 = MOC.from_fits('/home/graysonpetter/ssd/Dartmouth/data/footprints/VLASS/vlass_2.2.moc.fits')

	good_idxs = moc1.contains(np.array(ras) * u.deg, np.array(decs) * u.deg) | \
				moc2.contains(np.array(ras) * u.deg, np.array(decs) * u.deg)
	return good_idxs

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

def desiqso():
	dat = Table.read('catalogs/masked/desiQSO_edr.fits')
	rand = Table.read('catalogs/masked/desiQSO_edr_randoms.fits')
	return dat, rand

def lotss_rg_sample(fcut=2., sep_cw=5, sep_2mass=5, w1cut=17, colorcut=0.6, jcut=16, w1faint=18, maxflux=1000, majmax=60):
	from SkyTools import fluxutils
	lotss = Table.read('../data/radio_cats/LOTSS_DR2/LOTSS_DR2_2mass_cw.fits')
	lotss = lotss[np.where(lotss['Total_flux'] < maxflux)]
	lotss = lotss[np.where(lotss['Maj'] < majmax)]
	lotss = lotss[np.where((lotss['Jmag'].mask == True) | (np.array(lotss['Jmag'], dtype=float) > jcut) | (lotss['sep_2mass'] > sep_2mass))]
	# detected in WISE to avoid lobes
	lotss = lotss[np.where((lotss['sep_cw'] < sep_cw))]

	lotss = lotss[np.where(((lotss['W1_cw'] - lotss['W2_cw']) > (5.25 - 0.3 * lotss['W2_cw'])))]

	#lotss = lotss[np.where( (lotss['W1_cw'] - lotss['W2_cw'] > colorcut) | (lotss['W1_cw'] > w1cut) |
	# (lotss['W1_cw'].mask == True) | (lotss['W2_cw'].mask == True))]
	lotss['r75'] = np.array(fluxutils.r75_assef(lotss['W1_cw'], lotss['W2_cw']), dtype=int)
	lotss['r90'] = np.array(fluxutils.r90_assef(lotss['W1_cw'], lotss['W2_cw']), dtype=int)
	lotss = lotss[lotss['Total_flux'] > fcut]
	return lotss



def redshift_dist(cat, sep):
	bootes = Table.read('/home/graysonpetter/ssd/Dartmouth/data/radio_cats/LoTSS_deep/classified/bootes.fits')
	bootidx, catidx = coordhelper.match_coords((bootes['RA'], bootes['DEC']), (cat['RA'], cat['DEC']),
											   max_sep=sep, symmetric=False)
	bootes = bootes[bootidx]
	bootes = bootes[np.where(bootes['z_best'] < 4)]
	zs = bootes['z_best']
	#speczs = bootes['Z'][np.where(bootes['Z'] > 0)]
	return zs


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

	plotdir = '/home/graysonpetter/Dropbox/radioplots/'
	plt.scatter(lowz['W2_cw'], lowz['W1_cw'] - lowz['W2_cw'], s=20, c='cornflowerblue', marker='o')

	plt.scatter(midz['W2_cw'], midz['W1_cw'] - midz['W2_cw'], s=20, c='orange', marker='o')
	plt.scatter(hiz['W2_cw'], hiz['W1_cw'] - hiz['W2_cw'], s=20, c='firebrick', marker='o')
	plt.scatter(star['W2_cw'], star['W1_cw'] - star['W2_cw'], s=10, c='blue', marker='*')
	plt.scatter(0, 0, s=100, c='b', marker='*', label="SF-only")
	plt.plot(np.linspace(10, 13.86, 5), 0.65 * np.ones(5), ls='--', c='k')
	plt.plot(np.linspace(13.86, 20, 100), 0.65 * np.exp(0.153 * np.square(np.linspace(13.86, 20, 100) - 13.86)), c='k',
			 ls='--')
	plt.ylim(-0.5, 2)
	plt.text(15.2, 1.8, 'R90 AGN', fontsize=15)
	plt.text(12.2, 0.4, r'$z < 0.5$', color='cornflowerblue', fontsize=20)
	plt.text(14.5, -0.3, r'$0.5 < z < 1$', color='orange', fontsize=20)
	plt.text(17.5, 1.6, r'$z > 1$', color='firebrick', fontsize=20)
	plt.xlim(12, 18.5)
	plt.plot(np.linspace(10, 20, 100), (5.25 - 0.3 * np.linspace(10, 20, 100)), c='grey', ls='--')
	plt.xlabel('W2')
	plt.ylabel(r'W1 $-$ W2')
	plt.legend()
	plt.title(r'$S_{150 \ \mathrm{MHz}} >$ 2 mJy')
	plt.savefig(plotdir + 'wise_diagram.pdf')
	plt.close('all')

def hostgals():
	import matplotlib.pyplot as plt
	lotz = lotss_rg_sample()
	from SkyTools import coordhelper
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

	plotdir = '/home/graysonpetter/Dropbox/radioplots/'
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