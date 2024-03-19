import numpy as np
import healpy as hp


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
	ang_diam_dists = apcosmo.angular_diameter_distance(zs).to(u.kpc).value
	phys_sizes = (angsizes * u.arcsec).to(u.rad) * ang_diam_dists
	return phys_sizes


def in_vlass_epoch2(ras, decs):
	moc1 = MOC.from_fits('/home/graysonpetter/ssd/Dartmouth/data/footprints/VLASS/vlass_2.1.moc.fits')
	moc2 = MOC.from_fits('/home/graysonpetter/ssd/Dartmouth/data/footprints/VLASS/vlass_2.2.moc.fits')

	good_idxs = moc1.contains(np.array(ras) * u.deg, np.array(decs) * u.deg) | \
				moc2.contains(np.array(ras) * u.deg, np.array(decs) * u.deg)
	return good_idxs


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


"""def treat_dndz_pdf(cat, sep=2., ndraws=100, bootesonly=False):

	if bootesonly:
		photcat = match2bootes(cat, sep=sep)
	else:
		photcat = match2combined(cat, sep=sep)
	maxpossiblez = 3.
	# objects with spec-zs
	specs = photcat[np.where((photcat['f_zbest'] == 1) | (photcat['hasz'] == 1))]
	# with only photo-zs
	photcat = photcat[np.where((photcat['f_zbest'] == 0) & (photcat['hasz'] == 0))]

	# throwing out systems with uninformative PDFS
	# require positive minimum bound
	photcat = photcat[np.where(
							((photcat['z1min'] > 0.1) & ((photcat['z2min'] > 0.1) | (photcat['z2min'] < -50)))
							)]
	photcat = photcat[np.where(
		(photcat['z1med'] > 0.8) | ((photcat['z1med'] - photcat['z1min']) < 0.1)
	)]
	# require reasonable maximum bound
	photcat = photcat[np.where(
							((photcat['z1max'] < maxpossiblez) & ((photcat['z2max'] < maxpossiblez)))
							)]
	# throw out things with very broad PDFs
	#photcat = photcat[np.where(
	#						((photcat['z1max']-photcat['z1med']) < 1) & ((((photcat['z1med']-photcat['z1min']) < 0.3)) |
	#																	 (photcat['z1med'] > 1))
	#						)]
	#photcat = photcat[np.where(
	#	(photcat['z2med'] < -50) |
	#	(((photcat['z2max'] - photcat['z2med']) < 0.3) & ((photcat['z2med'] - photcat['z2min']) < 0.3))
	#)]




	# systems where there is only one peak in the photoz PDF
	singlephots = photcat[np.where(photcat['z2med'] == -99)]
	# sources where there are two peaks
	doublephots = photcat[np.where(photcat['z2med'] > 0)]


	speczs = list(np.repeat(specs['z_best'], ndraws))
	#specweights = list(ndraws*np.ones_like(speczs))

	singlezs, singleweights = [], []
	for j in range(len(singlephots)):
		thisrow = singlephots[j]
		# Duncan is reporting  bounds of 80% confidence interval
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
	return finalzs"""