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







def legacysurvey_mask(syst_label):
	"""
		Read in angular systematics map produced by DESI (Myers+23)
		:param syst_label:
		:return:
	"""
	pixweight = Table.read('/home/graysonpetter/ssd/Dartmouth/data/desi_targets/syst_maps/pixweight-1-dark.fits')
	systmap = np.empty(hp.nside2npix(256))
	systmap[hp.nest2ring(256, pixweight['HPXPIXEL'])] = pixweight['%s' % syst_label]
	if syst_label == 'PSFDEPTH_W2':
		systmap = 22.5 - 2.5 * np.log10(5 / np.sqrt(systmap)) - 3.339
	elif (syst_label == 'PSFDEPTH_Z') | (syst_label == 'PSFDEPTH_R') | (syst_label == 'PSFDEPTH_G'):
		systmap = 22.5 - 2.5 * np.log10(5 / np.sqrt(systmap))
	return systmap

def in_cmblensingmask(ras, decs):
	"""
	Filter coordinates inside Planck CMB lensing mask, works as an effective mask of the Galactic plane
	:return:
	"""
	lensmask = hp.read_map('/home/graysonpetter/ssd/Dartmouth/data/lensing_maps/PlanckPR4/derived/1024/mask.fits')
	return lensmask[hp.ang2pix(nside=hp.npix2nside(len(lensmask)), theta=ras, phi=decs, lonlat=True)] == 1


def in_lotss_dr2(ras, decs, northonly=True):
	"""
	Filter coordinates to inside LoTSS DR2 footprint
	:param ras: right ascension array (deg)
	:param decs: declination array (deg)
	:param northonly: bool, only use footprint in North Galactic Cap
	:return:
	"""
	moc = MOC.from_fits('/home/graysonpetter/ssd/Dartmouth/data/radio_cats/LOTSS_DR2/lotss_dr2_hips_moc.fits')
	inmoc = moc.contains(np.array(ras) * u.deg, np.array(decs) * u.deg)

	rmsmap = hp.read_map('masks/LOTSS_DR2_rms.fits')
	rms = rmsmap[hp.ang2pix(nside=hp.npix2nside(len(rmsmap)), theta=ras, phi=decs, lonlat=True)]
	goodrms = rms < 0.2 # mJy

	good_idxs = inmoc & goodrms
	if northonly:
		innorth = (ras > 90) & (ras < 290)
		good_idxs = good_idxs & innorth


	return good_idxs




def in_ls_dr8(ra, dec):
	"""
	Filter coordinates inside Legacy Survey DR8 footprint, since we are using photo-z
	:param ra:
	:param dec:
	:return:
	"""
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




def in_eboss(ras, decs, northonly=True):
	"""
	Filter coordinates in eBOSS quasar footprint
	:param ras:
	:param decs:
	:param northonly:
	:return:
	"""

	ebossmoc = pymangle.Mangle('../data/footprints/eBOSS/eBOSS_QSOandLRG_fullfootprintgeometry_noveto.ply')
	good_idxs = ebossmoc.contains(ras, decs)
	if northonly:
		innorth = (ras > 90) & (ras < 290)
		good_idxs = good_idxs & innorth
	return good_idxs





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






def in_goodwise(ras, decs):
	"""
	Filter coordinates inside footprint where WISE depth is complete to some limit
	:param ras:
	:param decs:
	:return:
	"""
	w2depth = hp.read_map('masks/wisedepth.fits')
	depths = w2depth[hp.ang2pix(hp.npix2nside(len(w2depth)), ras, decs, lonlat=True)]
	return (depths > 17.3)






def outside_galaxy(ras, decs, galcut=0, betacut=90, ebvcut=0.1, stardenscut=2000.):
	"""
	Filter coordinates by galactic/ecliptic cuts, or reddening, stellar density cuts
	:param ras:
	:param decs:
	:param galcut:
	:param betacut:
	:param ebvcut:
	:param stardenscut:
	:return:
	"""
	stardens = hp.read_map('masks/stardens.fits')
	ebv = hp.read_map('masks/ebv.fits')

	densities = stardens[hp.ang2pix(hp.npix2nside(len(stardens)), ras, decs, lonlat=True)]
	ebvs = ebv[hp.ang2pix(hp.npix2nside(len(ebv)), ras, decs, lonlat=True)]

	l, b = coordhelper.equatorial_to_galactic(ras, decs)
	lams, betas = coordhelper.equatorial_to_ecliptic(ras, decs)
	goodbs = np.abs(b) > galcut
	goodbetas = (betas < betacut)

	gooddens = (densities < stardenscut)
	goodebv = (ebvs < ebvcut)
	return goodebv & gooddens & goodbs & goodbetas


def cat_outside_galaxy(cat, galcut=0, betacut=90, ebvcut=0.1, stardenscut=2000.):
	cat = cat[outside_galaxy(cat['RA'], cat['DEC'],
							 galcut=galcut, betacut=betacut, ebvcut=ebvcut, stardenscut=stardenscut)]
	return cat


def cat_in_lotss(cat):
	return cat[in_lotss_dr2(cat['RA'], cat['DEC'])]


def cat_in_ls_dr8(t):
	infoot = in_ls_dr8(t['RA'], t['DEC'])
	return t[infoot]

def cat_in_eboss(cat):
	return cat[in_eboss(cat['RA'], cat['DEC'])]

def cat_in_goodwise(cat):
	return cat[in_goodwise(cat['RA'], cat['DEC'])]

def cat_in_boss(cat):
	return cat[in_boss(cat['RA'], cat['DEC'])]


def in_goodclustering_area(ra, dec):
	return in_lotss_dr2(ra, dec) & in_goodwise(ra, dec) & in_ls_dr8(ra, dec) & outside_galaxy(ra, dec)

def cat_in_goodclustering_area(cat):
	cat = cat_in_lotss(cat)
	cat = cat_in_goodwise(cat)
	cat = cat_in_ls_dr8(cat)
	cat = cat_outside_galaxy(cat)
	return cat


def gen_lotss_randoms():
	from SkyTools import random_catalogs
	randra, randdec = random_catalogs.uniform_sphere(1000, density=True, lat_range=(15, 90))
	rand = Table()
	rand['RA'] = randra
	rand['DEC'] = randdec
	rand['weight'] = np.ones(len(rand))
	rand = cat_in_goodclustering_area(rand)
	return rand

def lotss_randoms(nrand):
	"""
	Using randoms provided by Legacy Survey DR8
	randomly sample and apply masks
	:param nrand:
	:return:
	"""
	rand = Table.read('catalogs/randoms_dr8.fits')
	rand = rand[np.where((rand['WISEMASK_W1'] == 0) & (rand['WISEMASK_W2'] == 0))]
	rand = rand['RA', 'DEC']
	rand = rand[np.random.choice(len(rand), int(nrand * 10), replace=False)]
	rand = cat_in_goodclustering_area(rand)
	rand['weight'] = np.ones(len(rand))
	rand = rand[np.random.choice(len(rand), int(nrand), replace=False)]
	return rand

def rg_mask(nside):
	"""
	For CMB lensing analysis
	Make a map at NSIDE, and apply same masks to centers of pixels as the clustering samples
	:param nside:
	:return:
	"""
	mask = np.zeros(hp.nside2npix(nside))
	l, b = hp.pix2ang(nside, np.arange(len(mask)), lonlat=True)
	ra, dec = coordhelper.galactic_to_equatorial(l, b)
	idx = in_goodclustering_area(ra, dec)
	mask[idx] = 1.
	return mask



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
	#lotss = lotss[np.where(np.logical_not(lotss['Resolved']))]
	lotss = supercede_catwise(lotss, supercede_sep=params.supercede_cw_sep)
	#lotss = lotss[np.where((lotss['L150'] > params.lumcut) | (np.logical_not(np.isfinite(lotss['zphot']))))]

	if params.use_wiseflags:
		lotss = lotss[np.where(lotss['abfl'] == "00")]


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
	if params.izrg_maxflux is not None:
		lotss = lotss[np.where(lotss['Total_flux'] < params.izrg_maxflux)]
	lotss['weight'] = np.ones(len(lotss))

	if lummin is None:
		lummin = params.lumcut
	if minz is None:
		minz = params.izrg_minzphot
	if maxz is None:
		maxz = params.izrg_maxzphot

	lotss = supercede_catwise(lotss, supercede_sep=params.supercede_cw_sep)

	lotss = lotss[np.where((lotss['W2'] < params.w2faint))]

	if params.use_wiseflags:
		lotss = lotss[np.where(lotss['abfl'] == "00")]

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

	if params.use_wiseflags:
		lotss = lotss[np.where(lotss['abfl'] == "00")]
	# if using Duncan photometric redshifts (Hardcastle+23 catalog) to select low-redshift luminous AGN,
	# instead of using WISE colors like higher-redshift samples
	if params.lzrg_minzphot is not None:
		lotss = lotss[np.where(lotss['W2'] < params.w2faint_lzrg)]
		lotss = lotss[np.where((lotss['zphot'] > params.lzrg_minzphot) & (lotss['zphot'] < params.lzrg_maxzphot))]
		# still throw out HzRGs
		lotss = lotss[np.where((lotss['W1'] - lotss['W2']) < params.hzrg_cut_eqn(lotss['W2']))]
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

	lotss = Table.read('catalogs/LoTSS.fits')
	lotss = cat_in_goodclustering_area(lotss)
	lotss = lotss[np.where(lotss['Total_flux'] < maxflux)]
	lotss = lotss[np.where(lotss['Maj'] < majmax)]

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





def treat_dndz_pdf(cat, sep=2., minz=0.1, maxz=3.):
	"""
	Infer redshift distribution from Duncan+21 photo-z "PDFs"
	Given are first two peaks in redshift PDF, with medians, 80% confidence intervals and associated areas
	this will attempt to reconstruct full dn/dz of a sample, by bootstrapping
	:param cat:
	:param sep:
	:return:
	"""
	from scipy.stats import norm
	match = match2combined(cat, sep=sep)
	# objects with spec-zs
	specs = match[np.where((match['f_zbest'] == 1) | (match['hasz'] == 1))]
	# with only photo-zs
	phot = match[np.where((match['f_zbest'] == 0) & (match['hasz'] == 0))]


	# remove bad estimates, here we are assuming galaxies at z>3 are infeasible to detect with WISE
	phot = phot[(phot['z1min'] > minz) & (phot['z1med'] > minz) & (phot['z1max'] < maxz) & (phot['z1med'] < maxz)]

	z1med, z1min, z1max = phot['z1med'], phot['z1min'], phot['z1max']
	z1med_weight = phot['z1area']
	# Duncan quotes 80% confidence interval, corresponds to 1.28 sigma
	# chance of drawing a min or max bound should be weighted down by G(1.28)/G(0)
	z1min_weight = phot['z1area'] * norm.pdf(1.28) / norm.pdf(0)

	# set with a measured second peak
	withz2 = phot[(phot['z2min'] > minz) & (phot['z2med'] > minz) & (phot['z2max'] < maxz) & (phot['z2med'] < maxz)]
	frac_withz2 = len(withz2) / len(phot)

	z2med, z2min, z2max = withz2['z2med'], withz2['z2min'], withz2['z2max']
	z2med_weight = frac_withz2 * withz2['z2area']
	z2min_weight = frac_withz2 * withz2['z2area'] * norm.pdf(1.28) / norm.pdf(0)


	zs = np.concatenate((z1med, z1min, z1max, z2med, z2min, z2max))
	zweight = np.concatenate((z1med_weight, z1min_weight, z1min_weight, z2med_weight, z2min_weight, z2min_weight))
	zweight /= np.sum(zweight)

	# randomly draw from phot-z medians, mins and maxes for first two peaks, weighted appopriately
	drawn_photz = np.random.choice(zs, 10000, replace=True, p=zweight)

	frac_with_spec = len(specs) / len(phot)
	drawn_specz_repeated = np.random.choice(specs['z_best'], int(10000*frac_with_spec), replace=True)

	allzs = np.concatenate((drawn_specz_repeated, drawn_photz))
	allzs = allzs[allzs < 4.]
	return allzs

def savgol_dndz(dndz):
	"""
	Smooth redshift distribution with a linear Savitsky-Golay filter to smooth over aliasing "spikes"
	:param dndz:
	:return:
	"""
	from scipy.signal import savgol_filter
	from halomodelpy import redshift_helper
	smoothed = savgol_filter(dndz[1], int(len(dndz[1]) / 10.), polyorder=1)
	dndz = redshift_helper.norm_z_dist((dndz[0], smoothed))
	return dndz





def redshift_dist(cat, sep=2., bootesonly=True):
	"""
	Get list of "best" redshift estimates for a given sample by matching to deep fields (Duncan+21, Best+23)
	:param cat:
	:param sep:
	:param bootesonly:
	:return:
	"""
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

def get_dndz(cat):
	"""
	Estimate dn/dz for a given sample by directly taking histogram of "best" redshift estimates in deep fields
	:param cat:
	:return:
	"""
	zs = redshift_dist(cat)
	dndz = redshift_helper.dndz_from_z_list(zs, nbins=params.nzbins, zrange=params.zbin_range)
	# dont need beyond where dndz->0, chop off
	minidx = np.min(np.nonzero(dndz[1]))
	maxidx = np.max(np.nonzero(dndz[1])) + 1
	dndz = dndz[0][minidx:maxidx], dndz[1][minidx:maxidx]
	# increase resolution of dn/dz
	dndz = redshift_helper.fill_in_coarse_dndz(dndz, np.linspace(np.min(dndz[0]), np.max(dndz[0]), params.nzbins))
	return dndz, np.median(zs)


def hzrg_dndz(hzrgcat, pdfs=True, bootesonly=False):
	"""
	As the high-z sample is mostly photo-z, worry about unphysical spikes in dn/dz, so smooth with a filter
	:param hzrgcat:
	:param pdfs:
	:param bootesonly:
	:return:
	"""
	if pdfs:
		zs = treat_dndz_pdf(cat=hzrgcat, sep=2.)
	else:
		zs = redshift_dist(hzrgcat, bootesonly=bootesonly)
	dndz = redshift_helper.dndz_from_z_list(zs, nbins=params.nzbins, zrange=params.zbin_range)
	dndz = savgol_dndz(dndz)
	return dndz, np.median(zs)

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