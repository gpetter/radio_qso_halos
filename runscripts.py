import measurements
import sample
import numpy as np
import prep_radio_data
import radiogalaxy_sample
import params



#sample.total_clustering_sample()
# Step 0, download Hardcastle+23 catalog, match to CatWISE2020

# Step 1, make LoTSS catalog
#prep_radio_data.prepfor_wisematch()
#prep_radio_data.cut_duplicates()
#prep_radio_data.make_legacysurvey_mask()
#prep_radio_data.make_ls_depth_map()
#prep_radio_data.make_bitmask_map()

# 2. Create redshift samples, plot
#radiogalaxy_sample.wisediagram_both()
radiogalaxy_sample.both_redshift_dist()
radiogalaxy_sample.lum_redshift()
radiogalaxy_sample.hostgals()
#print('HzRG mag bias is %s pm %s' % sample.estimate_rg_magbias('hzrg'))
#print('IzRG mag bias is %s pm %s' % sample.estimate_rg_magbias('izrg'))

# 3. Check redshift distributions with tomographer.org
#sample.maketomographer_files(128, 'lo')

# 4. Cross correlate LzRGs with BOSS galaxies
measurements.lzrg_xcorr(params.linscales)
measurements.autocorr_lzrgs(params.hodscales)
#measurements.lumtrend(params.linscales, nzbins=15)

# 5. Autoclustering of IzRGs
measurements.autocorr_izrgs(params.hodscales)

# 6. Autoclustering of HzRGs, CMB lensing of HzRGs, and cross-correlations with eBOSS quasars
#measurements.autocorr_hzrgs(params.hodscales)
#measurements.lens_hzrg(lensscales)
#measurements.hzrg_xcorr(rpscales=linscales)

# 7. Fit autocorrelations with HOD
#measurements.hodfit('lzrg', niter=1000, freeparam_ids=['M', 'sigM', 'M1', 'alpha'], inital_params=[13, 1., 13.5, .8])
#measurements.hodfit('izrg', niter=1000, freeparam_ids=['M', 'sigM', 'M1', 'alpha'], inital_params=[13, 1., 13.5, .8])
#measurements.hodfit('hzrg', niter=1000, freeparam_ids=['M', 'sigM', 'M1', 'alpha'], inital_params=[13, 0.5, 14, .8])


# 8. Determine duty cycle and energy injection
#measurements.duty_cycle()
#measurements.halopower()
#measurements.halopower_ratio()