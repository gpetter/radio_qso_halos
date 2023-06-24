from SkyTools import healpixhelper, coordhelper
import numpy as np
import healpy as hp
import astropy.units as u
from mocpy import MOC
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
import os


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