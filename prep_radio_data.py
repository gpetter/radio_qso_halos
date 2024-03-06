from SkyTools import healpixhelper, coordhelper, fluxutils, bitmaskhelper
import numpy as np
import healpy as hp
import astropy.units as u
from mocpy import MOC
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table, vstack
import os
import glob

def prepfor_wisematch():
    """
    Process raw Hardcastle+23 catalog, output catalog ready to update by crossmatching with CatWISE2020
    :return:
    """
    t = Table.read('../data/radio_cats/LOTSS_DR2/combined-release-v1.1-LM_opt_mass.fits')
    # only intersted in AGN which are bright, and want flux-complete sample, make cut on flux to reduce size, 1.5 mJy
    t = t[np.where(t['Total_flux'] > 1.5)]
    # make a new "best" RA/DEC column, using optical/IR position when available, else falling back to radio position
    t.rename_columns(['RA', 'DEC'], ['radRA', 'radDEC'])
    t.rename_columns(['ra', 'dec'], ['ls_ra', 'ls_dec'])
    t['RA'] = t['radRA']
    t['DEC'] = t['radDEC']
    good_id = np.where(np.isfinite(t['ID_RA']))
    t['RA'][good_id] = t['ID_RA'][good_id]
    t['DEC'][good_id] = t['ID_DEC'][good_id]

    # write out, take to TOPCAT to match with CatWISE using best coordinates
    t.write('catalogs/H23_prepped.fits', overwrite=True)


def cut_duplicates():
    """
    Prep LoTSS DR2 catalog
    If two or more sources are matched to the same WISE source, only set WISE counterpart be assigned to the closest one
    :return:
    """
    lotss = Table.read('catalogs/H23_cw.fits')
    lotss = lotss[np.where(lotss['Total_flux'] > 1.5)]

    uniq, counts = np.unique(lotss['objID_cw'], return_counts=True)
    dup = uniq[counts > 1]
    #dup_idx = np.where(np.in1d(lotss['objID'], dup))
    seps = lotss['sep_cw']

    for duplic in dup:
        idx = np.where(lotss['objID_cw'] == duplic)[0]
        #numdups = len(idx[0])
        newseps = seps[idx]
        if len(newseps) > 0:
            bestidx = np.argmin(newseps)
            badidx = np.where(newseps > newseps[bestidx])[0]
            seps[idx[badidx]] = np.nan
    lotss['RA'] = np.array(lotss['RA'])
    lotss['DEC'] = np.array(lotss['DEC'])
    lotss['RA_cw'] = np.array(lotss['RA_cw'])
    lotss['DEC_cw'] = np.array(lotss['DEC_cw'])
    # cut negative redshifts
    lotss['zphot'][np.where(lotss['zphot'] < 0.0001)] = np.nan

    # https://catalog.unwise.me/catalogs.html
    lotss['W1_uw'] = lotss['mag_w1'] - 2.699
    lotss['W2_uw'] = lotss['mag_w2'] - 3.339
    lotss['W3'] = lotss['mag_w3'] - 5.174
    lotss['W4'] = lotss['mag_w4'] - 6.62

    #use_cw_w1 = np.where((np.logical_not(np.isfinite(lotss['W1']))) & (lotss['sep_cw'] < 2.5))
    # use CatWISE instead of unWISE when a good match
    #use_cw = np.where((lotss['sep_cw'] < 2.5))
    #lotss['W1'][use_cw] = lotss['W1_cw'][use_cw]
    #lotss['W2'][use_cw] = lotss['W2_cw'][use_cw]
    #lotss['RA'][use_cw] = lotss['RA_cw'][use_cw]
    #lotss['DEC'][use_cw] = lotss['DEC_cw'][use_cw]

    lotss['L150'] = np.log10(fluxutils.luminosity_at_rest_nu(lotss['Total_flux'], alpha=-0.7, nu_obs=.144,
                                                             nu_rest_want=.15, z=lotss['zphot'],
                                                             flux_unit=u.mJy, energy=False))
    lotss['RA', 'DEC', 'Total_flux',
          'Peak_flux', 'Maj', 'W1_uw', 'W2_uw', 'W1_cw', 'W2_cw', 'W3', 'W4',
          'zphot', 'L150', 'z_desi', 'Resolved', 'LAS', 'abfl', 'sep_cw'].write('catalogs/LoTSS.fits', overwrite=True)


def make_legacysurvey_mask():
    """
    Using Legacy Survey DR8 data for photometric redshifts, so want area with best LS data
    :return:
    """
    rand = Table.read('../data/randoms/ls/dr8/randoms-inside-dr8-0.31.0-1.fits')
    rand = rand[(rand['NOBS_G'] >= 3) & (rand['NOBS_R'] >= 3) & (rand['NOBS_Z'] >= 3)]
    dens = healpixhelper.healpix_density_map(rand['RA'], rand['DEC'], 256)
    dens[np.where(dens >= 1)] = 1
    hp.write_map('masks/good_LS.fits', dens, overwrite=True)

def make_ls_depth_map():
    """
    Using Legacy Survey DR8 data for photometric redshifts, so want area with best LS data
    :return:
    """
    rand = Table.read('../data/randoms/ls/dr8/randoms-inside-dr8-0.31.0-1.fits')
    rand = rand[np.where(rand['PSFDEPTH_Z'] > 0)]
    depths = -2.5*(np.log10(5/np.sqrt(rand['PSFDEPTH_Z']))-9)
    meddepth = healpixhelper.healpix_median_in_pixels(256, (rand['RA'], rand['DEC']), depths)
    hp.write_map('masks/LS_zdepth.fits', meddepth, overwrite=True)

def make_wisemap():
    pixweight = Table.read('/home/graysonpetter/ssd/Dartmouth/data/desi_targets/syst_maps/pixweight-1-dark.fits')
    import healpy as hp
    w2depth = np.empty(hp.nside2npix(256))
    w2depth[hp.nest2ring(256, pixweight['HPXPIXEL'])] = pixweight['PSFDEPTH_W2']
    w2depth = 22.5 - 2.5 * np.log10(5 / np.sqrt(w2depth)) - 3.339
    #w2depth[np.where(np.logical_not(np.isfinite(w2depth)))] = 100.
    hp.write_map('masks/wisedepth.fits', w2depth, overwrite=True)

def make_stardensmap():
    pixweight = Table.read('../data/desi_targets/syst_maps/pixweight-1-dark.fits')

    stars = np.empty(hp.nside2npix(256))
    stars[hp.nest2ring(256, pixweight['HPXPIXEL'])] = pixweight['STARDENS']
    hp.write_map('masks/stardens.fits', stars, overwrite=True)

def make_ebv_map():
    pixweight = Table.read('../data/desi_targets/syst_maps/pixweight-1-dark.fits')

    ebv = np.empty(hp.nside2npix(256))
    ebv[hp.nest2ring(256, pixweight['HPXPIXEL'])] = pixweight['EBV']
    hp.write_map('masks/ebv.fits', ebv, overwrite=True)

def make_bitmask_map(nside=4096):
    randfiles = glob.glob('../data/randoms/ls/dr8/randoms-inside-dr8-0.31.0-*.fits')
    tottab = Table()
    for randfile in randfiles:
        t = Table.read(randfile)
        t = t[np.where(t['MASKBITS'] != 0)]
        maskidxs = []
        for k in range(len(t)):

            bits = bitmaskhelper.parse_bitmask(t['MASKBITS'][k], 15)
            if (1 in bits) | (8 in bits) | (9 in bits) | (12 in bits):
                maskidxs.append(k)
        maskidxs = np.array(maskidxs)
        t = t[maskidxs]
        t = t['RA', 'DEC']
        tottab = vstack((tottab, t))
    densmap = healpixhelper.healpix_density_map(tottab['RA'], tottab['DEC'], nsides=nside)
    densmap[np.where(densmap > 0)] = 1
    hp.write_map('masks/maskbits_map.fits', densmap, overwrite=True)



def highres_lotss_dr2_noise_map(outputmap_nside=4096):
    scratch_space = '.'
    curdr = os.getcwd()
    os.chdir(scratch_space)

    rmsmap = np.zeros(hp.nside2npix(outputmap_nside))

    t = Table.read(curdr + '/selection/lotss_dr2_mosaic_ids.fits')

    try:
        for j in range(len(t)):
            os.system(
                'wget https://lofar-surveys.org/public/DR2/mosaics/%s/mosaic.rms.fits' % t['mosaic_id'][j].strip())
            rmsfile = fits.open('mosaic.rms.fits')
            header = rmsfile[0].header
            w = WCS(header)
            npix_x, npix_y = header['NAXIS1'], header['NAXIS2']
            observed_pix = np.where(np.isfinite(rmsfile[0].data))

            worldcoords = w.wcs_pix2world(observed_pix[3], observed_pix[2], 0, 0, 0)
            pixras, pixdecs = worldcoords[0], worldcoords[1]
            rmsvals = rmsfile[0].data[observed_pix]
            # replace high regions of RMS due to source detection with background
            # foo, sigmed, foo2 = astrostats.sigma_clipped_stats(rmsvals)
            # rmsvals[np.where(rmsvals > 5*sigmed)] = sigmed

            hp_idxs = hp.ang2pix(nside=outputmap_nside, theta=pixras, phi=pixdecs, lonlat=True)
            avgrms = np.bincount(hp_idxs, weights=rmsvals, minlength=hp.nside2npix(outputmap_nside)) / \
                     np.bincount(hp_idxs, minlength=hp.nside2npix(outputmap_nside))

            idxs = np.where(avgrms > 0)
            rmsmap[idxs] = avgrms[idxs]
            print(j)

            os.remove('mosaic.rms.fits')

        os.chdir(curdr)
        hp.write_map('masks/lotss_dr2_highres_rms_map.fits', rmsmap, overwrite=True)
    except:
        os.chdir(curdr)
        hp.write_map('masks/lotss_dr2_highres_rms_map.fits', rmsmap, overwrite=True)


def lotss_dr2_background_rms():
    from plotscripts import hp_maps
    rmshires = np.log10(hp.read_map('masks/lotss_dr2_highres_rms_map.fits'))
    lores = 1000. * 10 ** healpixhelper.ud_grade_median(rmshires, nside_out=128)

    hp.write_map('masks/LOTSS_DR2_rms.fits', lores, overwrite=True)
    hp_maps.lotss_rms(lores)

def prep_randoms():
    """
    Save time by pre-trimming randoms provided by Legacy Survey DR8, to only those in LoTSS DR2 footprint
    :return:
    """
    rand = Table.read('../data/randoms/ls/dr8/randoms-inside-dr8-0.31.0-1.fits')
    moc = MOC.from_fits('/home/graysonpetter/ssd/Dartmouth/data/radio_cats/LOTSS_DR2/lotss_dr2_hips_moc.fits')
    inmoc = moc.contains(np.array(rand['RA']) * u.deg, np.array(rand['DEC']) * u.deg)
    rand = rand[inmoc]
    rand.write('catalogs/randoms_dr8.fits', overwrite=True)

"""
# replace rms at positions of radio detections with local background rms
def background_rms(inner_radius=0.04):
	rms = hp.read_map('masks/lotss_dr2_highres_rms_map.fits')
	nside = hp.npix2nside(len(rms))
	rms[np.where(rms == 0)] = np.nan
	#lotss = Table.read('../data/radio_cats/LOTSS_DR2/LOTSS_DR2.fits')
	#for j in range(len(lotss)):
	#	ra, dec = lotss['RA'][j], lotss['DEC'][j]
	#	annulus = myhp.query_annulus_coord(nside, (ra, dec), inner_radius, 0.1)
	#	medrms = np.nanmedian(rms[annulus])
	#	indisc = myhp.query_disc_coord(nside, (ra, dec), inner_radius)
	#	indisc_unmasked = indisc[np.where(np.isfinite(rms[indisc]))]
	#	rms[indisc_unmasked] = medrms
	hp.write_map('masks/LOTSS_DR2_rms.fits', rms*1000., overwrite=True)

def lowres_noisemap(lofar_name, nsides=64):
	catalog = Table.read('../data/radio_cats/%s/%s.fits' % (lofar_name, lofar_name))
	# convert coordinates to healpix pixels
	pix_of_sources = hp.ang2pix(nsides, catalog['RA'], catalog['DEC'], lonlat=True)
	# number of pixels for healpix map with nsides
	npix = hp.nside2npix(nsides)
	medmap = 10 ** np.array(stats.binned_statistic(x=pix_of_sources, values=np.log10(catalog['E_Peak_flux']),
									statistic='median', bins=np.linspace(-0.5, npix-0.5, npix+1))[0])

	hp.write_map('masks/%s_rms.fits' % lofar_name, np.array(medmap), overwrite=True)"""
def download_first_moc():
    from astroquery.cds import cds
    moc_FIRST = cds.find_datasets(meta_data="ID=*first14*", return_moc=True)
    moc_FIRST.write('../data/radio_cats/FIRST/FIRST_moc.fits', format='fits')

def make_apertif_mask(nside):
    tiles1 = Table.read('../data/radio_cats/Apertif/pointings.fits')
    mask = healpixhelper.mask_from_pointings((tiles1['centeralpha'], tiles1['centerdelta']), nside, pointing_radius=0.4)

    hp.write_map('masks/Apertif.fits', mask, overwrite=True)

def apertif_rms():
    apertif = Table.read('../data/radio_cats/Apertif/Apertif.fits')
    rms = 10**healpixhelper.healpix_average_in_pixels(apertif['RA'], apertif['DEC'], 128, np.log10(apertif['Islrms']))

    hp.write_map('masks/Apertif_rms.fits', rms, overwrite=True)


def lolss_rms_map(nside=2048):
    rms = np.zeros(hp.nside2npix(nside))
    ls, bs = hp.pix2ang(nside, np.arange(len(rms)), lonlat=True)

    m = MOC.from_fits('../data/radio_cats/LoLSS_DR1/Moc.fits')

    inmoc = m.contains(ls * u.deg, bs * u.deg)
    rms[inmoc] = 1.6
    hp.write_map('masks/LoLSS_DR1_rms.fits', rms, overwrite=True)

def make_vlass_mask(nside):
    tiles1 = Table.read('../data/radio_cats/VLASS/QL1_tiles.fits')
    tiles2 = Table.read('../data/radio_cats/VLASS/QL2_tiles.fits')
    rmsmap2 = healpixhelper.mask_from_pointings((tiles2['OBSRA'], tiles2['OBSDEC']), nside, pointing_side=1.,
                                                fill_values=tiles2['Mean_isl_rms'] / 1000.)
    hp.write_map('masks/VLASS_rms.fits', rmsmap2, overwrite=True)
    del rmsmap2

    mask1 = healpixhelper.mask_from_pointings((tiles1['OBSRA'], tiles1['OBSDEC']), nside, pointing_side=1.)
    mask2 = healpixhelper.mask_from_pointings((tiles2['OBSRA'], tiles2['OBSDEC']), nside, pointing_side=1.)
    mask = mask1 * mask2
    hp.write_map('masks/VLASS.fits', mask, overwrite=True)


def construct_clean_vlass(flux_cut=0, apply_mask=False):
    epoch1 = Table.read('../data/radio_cats/VLASS/Epoch1_QL.fits')
    epoch2 = Table.read('../data/radio_cats/VLASS/Epoch2_QL.fits')

    epoch1 = epoch1[
        np.where((epoch1['Duplicate_flag'] < 2) & ((epoch1['Quality_flag'] == 0) | (epoch1['Quality_flag'] == 4)))]
    epoch1 = epoch1[np.where(epoch1['Total_flux'] > flux_cut)]

    epoch2 = epoch2[
        np.where((epoch2['Duplicate_flag'] < 2) & ((epoch2['Quality_flag'] == 0) | (epoch2['Quality_flag'] == 4)))]
    epoch2 = epoch2[np.where(epoch2['Total_flux'] > flux_cut)]
    epoch1, epoch2 = coordhelper.match_coords(epoch1, epoch2, find_common_footprint=False)
    epoch2 = epoch2['RA', 'DEC', 'Total_flux', 'E_Total_flux', 'Peak_flux', 'E_Peak_flux', 'Maj']
    epoch2.write('../data/radio_cats/VLASS/VLASS.fits', overwrite=True)
    if apply_mask:
        mask = hp.read_map('masks/VLASS.fits')
        inmask = np.where(mask[hp.ang2pix(4096, epoch2['RA'], epoch2['DEC'], lonlat=True)] == 1)
        epoch2 = epoch2[inmask]

        epoch2.write('catalogs/VLASS/combined_masked.fits', overwrite=True)

def vlass_ep2():
    epoch2 = Table.read('../data/radio_cats/VLASS/Epoch2_QL_cw_2mass.fits')
    # 3 sigma 120 muJy
    epoch2 = epoch2[np.where(epoch2['Total_flux'] > 0.360)]
    epoch2 = epoch2[
        np.where((epoch2['Duplicate_flag'] < 2) & ((epoch2['Quality_flag'] == 0) | (epoch2['Quality_flag'] == 4)))]
    epoch2 = epoch2['RA', 'DEC', 'Total_flux', 'E_Total_flux', 'Peak_flux', 'E_Peak_flux', 'Maj', 'W1_cw', 'W2_cw', 'sep_cw', 'Jmag', 'sep_2mass']
    epoch2.write('catalogs/VLASS/VLASS.fits', overwrite=True)

def lotss_photo_zs():
    lotss = Table.read('../data/radio_cats/LOTSS_DR2/LOTSS_DR2.fits')
    lotss = coordhelper.match2duncan(lotss, sep=5.)
    lotss.write('catalogs/LoTSS_DR2_photz.fits', overwrite=True)