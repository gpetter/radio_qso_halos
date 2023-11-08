import measurements
import sample
import numpy as np
import prep_radio_data
import radiogalaxy_sample

linscales = np.logspace(0.7, 1.4, 11)
lensscales = np.logspace(0.7, 1.4, 8)
hodscales = np.logspace(-0.5, 1.4, 21)

# Step 0, download Hardcastle+23 catalog, match to CatWISE2020

# Step 1, make LoTSS catalog
#prep_radio_data.cut_duplicates()

# 2. Create redshift samples, plot
#radiogalaxy_sample.wisediagram_both()
#radiogalaxy_sample.both_redshift_dist()
#radiogalaxy_sample.lum_redshift()

# 3. Check redshift distributions with tomographer.org
#sample.maketomographer_files(128, 'lo')

# 4. Cross correlate LzRGs with BOSS galaxies
#measurements.lzrg_xcorr(linscales)

# 5. Autoclustering of IzRGs
#measurements.autocorr_izrgs(hodscales)

# 6. Autoclustering of HzRGs, CMB lensing of HzRGs, and cross-correlations with eBOSS quasars
#measurements.autocorr_hzrgs(hodscales)
#measurements.lens_hzrg(lensscales)
#measurements.hzrg_xcorr(rpscales=linscales)

# 7. Fit autocorrelations with HOD
#measurements.izrghod_fit(niter=1000, freeparam_ids=['M', 'sigM', 'M1', 'alpha'], inital_params=[13, 1., 13.5, .8])
#measurements.hzrghod_fit(niter=1000, freeparam_ids=['M', 'sigM', 'M1', 'alpha'], inital_params=[13, 0.5, 14, .8])

# 8. Determine duty cycle and energy injection
#measurements.duty_cycle()
#measurements.halopower()