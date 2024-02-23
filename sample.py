from astropy.table import Table, vstack, hstack
from mocpy import MOC
import astropy.units as u
import pymangle
import numpy as np
import healpy as hp
from SkyTools import healpixhelper as myhp
from SkyTools import coordhelper
from SkyTools import fluxutils
from halomodelpy import redshift_helper
import params

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
	# for some reason there are patches where Duncan doesn't provide photo zs, identified by hand
	inbad = ((ra < 243.4) & (ra > 242) & (dec < 35.3) & (dec > 34.2)) | \
			((ra < 147) & (ra > 145.7) & (dec < 57.5) & (dec > 56.85)) | \
			((ra < 175.4) & (ra > 174.5) & (dec < 43.45) & (dec > 42.75)) | \
			((ra < 150.66) & (ra > 150) & (dec < 33) & (dec > 32.6)) | \
			((ra < 150.16) & (ra > 149.5) & (dec < 33.75) & (dec > 33.1)) | \
			((ra < 166) & (ra > 165.5) & (dec < 61.8) & (dec > 61.5))
	infoot = (northfoot.contains(ra * u.deg, dec * u.deg) |
							southfoot.contains(ra * u.deg, dec * u.deg)) & np.logical_not(inbad)
	goodls = hp.read_map('masks/good_LS.fits')
	zdepth = hp.read_map('masks/LS_zdepth.fits')
	bitmaskmap = hp.read_map('masks/maskbits_map.fits')
	depths = myhp.coords2mapvalues(ra, dec, zdepth)
	good_z = (depths > 23.)

	outside_bitmask = np.logical_not(myhp.inmask((ra, dec), bitmaskmap, return_bool=True))

	infoot = myhp.inmask((ra, dec), goodls, return_bool=True) & infoot & good_z & outside_bitmask


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


"""def lotss_randoms():
	from SkyTools import random_catalogs
	randra, randdec = random_catalogs.uniform_sphere(1000, density=True, lat_range=(15, 90))
	rand = Table()
	rand['RA'] = randra
	rand['DEC'] = randdec
	rand['weight'] = np.ones(len(rand))
	rand = cat_in_goodclustering_area(rand)
	return rand"""

def lotss_randoms(nrand):
	rand = Table.read('../data/randoms/ls/dr8/randoms-inside-dr8-0.31.0-1.fits')
	#rand = rand[np.where((rand['MASKBITS'] == 0) & (rand['WISEMASK_W1'] == 0) & (rand['WISEMASK_W2'] == 0))]
	rand = rand[np.random.choice(len(rand), int(nrand * 10), replace=False)]
	rand = cat_in_goodclustering_area(rand)
	rand['weight'] = np.ones(len(rand))
	rand = rand[np.random.choice(len(rand), int(nrand), replace=False)]
	rand = rand['RA', 'DEC', 'weight']
	return rand

def rg_mask(nside):
	mask = np.zeros(hp.nside2npix(nside))
	l, b = hp.pix2ang(nside, np.arange(len(mask)), lonlat=True)
	ra, dec = coordhelper.galactic_to_equatorial(l, b)
	idx = in_goodclustering_area(ra, dec)
	mask[idx] = 1.
	return mask




	
	


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


def match_random_zdist(samplecat, randomcat, nbins=15):
	zhist, edges = np.histogram(samplecat['Z'], bins=nbins, range=(0.7, 2.3), density=True)
	randhist, edges = np.histogram(randomcat['Z'], bins=nbins, range=(0.7, 2.3), density=True)
	ratio = zhist/randhist
	sampleweights = ratio[np.digitize(randomcat['Z'], bins=edges) - 1]
	subrandcat = randomcat[np.random.choice(len(randomcat), size=(10*len(samplecat)), replace=False, p=(sampleweights / np.sum(sampleweights)))]
	return subrandcat

def weight_common_dndz(cat1, cat2, zbins=15):
	minz1, maxz1 = np.min(cat1['Z']), np.max(cat1['Z'])
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



def supercede_catwise(lotss, supercede_sep=2.5, w2mag=10):
	lotss.rename_columns(['W1_uw', 'W2_uw'], ['W1', 'W2'])
	supercede_idx = np.where((lotss['sep_cw'] < supercede_sep) & (lotss['W2_cw'] > w2mag))
	lotss['W1'][supercede_idx] = lotss['W1_cw'][supercede_idx]
	lotss['W2'][supercede_idx] = lotss['W2_cw'][supercede_idx]
	return lotss

def hzrg_cut(lotss, fcut=None):
	lotss = Table(lotss, copy=True)
	if fcut is None:
		fcut = params.hzrg_fluxcut
	# cut local zphot detections
	lotss = lotss[np.where((lotss['zphot'] > params.hzrg_minzphot) | (np.logical_not(np.isfinite(lotss['zphot']))))]

	lotss = lotss[np.where(lotss['Total_flux'] < params.hzrg_maxflux)]
	lotss = lotss[np.where(lotss['LAS'] < params.lasmax)]
	lotss = supercede_catwise(lotss, supercede_sep=params.supercede_cw_sep)
	#lotss = lotss[np.where((lotss['L150'] > params.lumcut) | (np.logical_not(np.isfinite(lotss['zphot']))))]


	if params.w2faint is not None:
		# detected in WISE to avoid lobes
		# if catwise:
		#	lotss = lotss[np.where((lotss['sep_%s' % wisekey] < sep_cw))]
		lotss = lotss[np.where((lotss['W2'] < params.w2faint))]
		lotss = lotss[np.where(
			(lotss['W1'] - lotss['W2']) > params.hzrg_cut_eqn(lotss['W2']))]
	else:
		lotss = lotss[np.where((
									   (lotss['W1_%s'] - lotss['W2_%s']) > params.hzrg_cut_eqn(lotss['W2'])) |
							   (np.logical_not(np.isfinite(lotss['W1']))) |
							   (np.logical_not(np.isfinite(lotss['W2']))))]

	lotss['r75'] = np.array(fluxutils.r75_assef(lotss['W1'], lotss['W2']), dtype=int)
	lotss['r90'] = np.array(fluxutils.r90_assef(lotss['W1'], lotss['W2']), dtype=int)
	lotss = lotss[lotss['Total_flux'] > fcut]
	lotss['weight'] = np.ones(len(lotss))
	return lotss

def hzrg_sample(fcut=None):
	lotss = Table.read('catalogs/LoTSS.fits')
	lotss = cat_in_goodclustering_area(lotss)
	lotss = hzrg_cut(lotss, fcut=fcut)
	return lotss

def izrg_cut(lotss, lummin=None, lummax=np.inf, minz=None, maxz=None):
	lotss = Table(lotss, copy=True)
	lotss = lotss[np.where(lotss['Total_flux'] < params.izrg_maxflux)]
	#lotss = lotss[np.where(lotss['LAS'] < params.lasmax)]
	lotss['weight'] = np.ones(len(lotss))

	if lummin is None:
		lummin = params.lumcut
	if minz is None:
		minz = params.izrg_minzphot
	if maxz is None:
		maxz = params.izrg_maxzphot

	lotss = supercede_catwise(lotss, supercede_sep=params.supercede_cw_sep)

	#lotss = lotss[np.where((lotss['W2'] < params.w2faint))]

	nothzrg = (lotss['W1'] - lotss['W2']) < params.hzrg_cut_eqn(lotss['W2'])
	#notlzrg = (lotss['W1'] - lotss['W2']) < ((lotss['W2'] - 17) / 3. + 0.75)
	isr90qso = fluxutils.r90_assef(lotss['W1'], lotss['W2'])

	lotss = lotss[np.where(
		(nothzrg))]
	#|
	#())]

	if params.izrg_minzphot is not None:
		lotss = lotss[np.where(((lotss['zphot'] > minz) & (lotss['zphot'] < maxz)))]
		lotss = lotss[np.where((lotss['L150'] > lummin) & (lotss['L150'] < lummax))]
		return lotss


	lotss = lotss[lotss['Total_flux'] > params.izrg_minflux]

	return lotss


def izrg_sample(lummin=None, lummax=np.inf, minz=None, maxz=None):
	lotss = Table.read('catalogs/LoTSS.fits')
	lotss = cat_in_goodclustering_area(lotss)
	return izrg_cut(lotss, lummin=lummin, lummax=lummax, minz=minz, maxz=maxz)

def lzrg_cut(lotss):
	lotss = Table(lotss, copy=True)
	if params.lzrg_maxflux is not None:
		lotss = lotss[np.where(lotss['Total_flux'] < params.lzrg_maxflux)]
	#lotss = lotss[np.where(lotss['LAS'] < params.lasmax)]

	lotss = supercede_catwise(lotss, supercede_sep=params.supercede_cw_sep)
	# if using Duncan photometric redshifts (Hardcastle+23 catalog) to select low-redshift luminous AGN,
	# instead of using WISE colors like higher-redshift samples
	if params.lzrg_minzphot is not None:
		lotss = lotss[np.where((lotss['zphot'] > params.lzrg_minzphot) & (lotss['zphot'] < params.lzrg_maxzphot))]
		# still throw out HzRGs
		#lotss = lotss[np.where((lotss['W1'] - lotss['W2']) < params.hzrg_cut_eqn(lotss['W2']))]
		lotss = lotss[np.where(lotss['L150'] > params.lumcut)]

	# otherwise use WISE color cuts to try to select 0.25 < z < 0.5 radio galaxies
	else:
		# detected in WISE to avoid lobes
		#lotss = lotss[np.where((lotss['sep_cw'] < sep_cw))]

		lotss = lotss[np.where((lotss['W1_cw'] - lotss['W2_cw']) < ((17 - lotss['W2_cw']) / 4. - 0.15))]
		lotss = lotss[np.where((lotss['W1_cw'] - lotss['W2_cw']) > ((lotss['W2_cw'] - 17) / 3. + 0.65))]
		lotss = lotss[np.where((lotss['W1_cw'] - lotss['W2_cw']) < ((lotss['W2_cw'] - 17) / 3. + 1.15))]

		lotss.remove_columns(['RA', 'DEC'])
		lotss.rename_columns(['RA_cw', 'DEC_cw'], ['RA', 'DEC'])

		lotss = lotss[lotss['Total_flux'] > params.lzrg_minflux]
	lotss['weight'] = np.ones(len(lotss))
	return lotss

def lzrg_sample():
	lotss = Table.read('catalogs/LoTSS.fits')
	lotss = cat_in_goodclustering_area(lotss)
	lotss = lzrg_cut(lotss)
	return lotss



def total_clustering_sample():
	lotss = Table.read('catalogs/LoTSS.fits')
	lotss = cat_in_goodclustering_area(lotss)
	lzrg = lzrg_cut(lotss)
	izrg = izrg_cut(lotss)
	hzrg = hzrg_cut(lotss)
	print('%s LzRGs' % len(lzrg))
	print('%s IzRGs' % len(izrg))
	print('%s HzRGs' % len(hzrg))
	print('%s Total' % (len(lzrg) + len(izrg) + len(hzrg)))



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


def match2bootes(cat, sep, stack=False):
	bootes = Table.read('/home/graysonpetter/ssd/Dartmouth/data/radio_cats/LoTSS_deep/classified/bootes.fits')
	bootidx, catidx = coordhelper.match_coords((bootes['RA'], bootes['DEC']), (cat['RA'], cat['DEC']),
											   max_sep=sep, symmetric=False)
	bootes = bootes[bootidx]
	if stack:
		cat = cat[catidx]
		bootes = hstack([cat, bootes])
	return bootes

def match2combined(cat, sep, stack=False):
	comb = Table.read('/home/graysonpetter/ssd/Dartmouth/data/radio_cats/LoTSS_deep/classified/combined.fits')
	idx, catidx = coordhelper.match_coords((comb['RA'], comb['DEC']), (cat['RA'], cat['DEC']),
											   max_sep=sep, symmetric=False)
	comb = comb[idx]
	if stack:
		cat = cat[catidx]
		comb = hstack([cat, comb])
	return comb


def rg_redshifts():
	from SkyTools import get_redshifts
	rgs = lotss_rg_sample()
	rgs = get_redshifts.match_to_spec_surveys(rgs, seplimit=3)
	rgs.write('catalogs/rgs_specz.fits', overwrite=True)

def treat_dndz_pdf(cat, sep=3., ndraws=100, bootesonly=False):
	if bootesonly:
		photcat = match2bootes(cat, sep=sep)
	else:
		photcat = match2combined(cat, sep=sep)

	specs = photcat[np.where(photcat['f_zbest'] == 1)]
	phots = photcat[np.where(photcat['f_zbest'] == 0)]

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

def savgol_dndz(dndz):
	from scipy.signal import savgol_filter
	from halomodelpy import redshift_helper
	smoothed = savgol_filter(dndz[1], int(len(dndz[1]) / 10.), polyorder=1)
	dndz = redshift_helper.norm_z_dist((dndz[0], smoothed))
	return dndz

def hzrg_dndz(hzrgcat, nzbins=30, zrange=(0.1, 4.)):
	zs = treat_dndz_pdf(cat=hzrgcat, sep=2., bootesonly=False)
	dndz = redshift_helper.dndz_from_z_list(zs, nbins=nzbins, zrange=zrange)
	dndz = savgol_dndz(dndz)
	return dndz, np.median(zs)

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


def estimate_rg_magbias(whichsamp, magdiff=0.1, nboots=100):
	if whichsamp == 'hzrg':
		lotss = hzrg_sample(fcut=(0.75*params.hzrg_fluxcut))
		fluxcut=2.

	elif whichsamp == 'izrg':
		lotss = izrg_sample(fcut=0.75*5.)
		fluxcut=5.
	else:
		print('hzrg or izrg')
		return
	mags = -2.5 * np.log10(lotss['Total_flux'])
	magcut = -2.5 * np.log10(fluxcut)

	ncut = len(np.where(mags < magcut)[0])

	n_brighter = len(np.where(mags < (magcut - magdiff))[0])
	n_fainter = len(np.where(mags < (magcut + magdiff))[0])

	dlogn_dm_1 = (np.log10(n_brighter) - np.log10(ncut)) / (-magdiff)
	dlogn_dm_2 = (np.log10(n_fainter) - np.log10(ncut)) / (magdiff)

	smu = np.mean([dlogn_dm_1, dlogn_dm_2])

	smus = []
	for j in range(nboots):

		lotss_boot = lotss[np.random.choice(len(lotss), len(lotss), replace=True)]
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