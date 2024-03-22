import measurements
import sample
import numpy as np
import prep_radio_data
import radiogalaxy_sample
import params



# Step 0, download Hardcastle+23 catalog, match to CatWISE2020 (in TOPCAT is easiest)
#prep_radio_data.prepfor_wisematch()

# Step 1, make LoTSS catalog
#prep_radio_data.cut_duplicates()
#prep_radio_data.make_legacysurvey_mask()
#prep_radio_data.make_ls_depth_map()
#prep_radio_data.make_bitmask_map()
#prep_radio_data.prep_randoms()

# 2. Create redshift samples, plot sample properties
#radiogalaxy_sample.wisediagram()
#radiogalaxy_sample.both_redshift_dist()
#radiogalaxy_sample.lum_redshift()
#radiogalaxy_sample.hostgals()
#print('HzRG mag bias is %s pm %s' % sample.estimate_rg_magbias('hzrg'))
#print('IzRG mag bias is %s pm %s' % sample.estimate_rg_magbias('izrg'))

# 3. Check redshift distributions with tomographer.org
#sample.maketomographer_files(128, 'lo')


#measurements.autocorr_lzrgs(params.hodscales)
#measurements.lumtrend(params.linscales, nzbins=15)

# 5. Autoclustering of IzRGs
#measurements.autocorr_izrgs(params.hodscales)

# 6. Autoclustering of HzRGs, CMB lensing of HzRGs, and cross-correlations with eBOSS quasars
#measurements.autocorr_hzrgs(params.hodscales)
#measurements.lens_hzrg(params.lensscales)
#measurements.hzrg_xcorr(rpscales=params.linscales)

# 7. Fit autocorrelations with HOD
#measurements.hodfit('lzrg', nwalkers=10, niter=5000,
#                    freeparam_ids=['M', 'sigM', 'M1', 'alpha'], inital_params=[13.25, 0.5, #14, .8])
#measurements.hodfit('izrg', nwalkers=10, niter=5000,
#                    freeparam_ids=['M', 'sigM', 'M1', 'alpha'], inital_params=[13.25, 0.5, #14, .8])
#measurements.hodfit('hzrg', nwalkers=10, niter=5000,
#                    freeparam_ids=['M', 'sigM', 'M1', 'alpha'], inital_params=[13, 0.6, 15., #1.2])


# 8. Energy injection
#measurements.halopower(mcut=13.)
measurements.halopower_ratio()