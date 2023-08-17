import measurements
import sample
import numpy as np
import prep_radio_data

linscales = np.logspace(0.7, 1.4, 11)
hodscales = np.logspace(-0.5, 1.4, 21)

#sample.maketomographer_files(128, 'lo')
#measurements.autocorr_izrgs(hodscales)
#measurements.lens_izrg()
#measurements.lzrg_xcorr(linscales)

#measurements.autocorr_hzrgs(hodscales)
#measurements.lens_hzrg()
#measurements.izrghod_fit(niter=5000, freeparam_ids=['M', 'sigM', 'M1', 'alpha'], inital_params=[13, 1., 13.5, .8])
#measurements.hzrghod_fit(niter=1000, freeparam_ids=['M', 'sigM', 'M1', 'alpha'], inital_params=[13, 0.5, 13.5, .8])

#measurements.hzrg_xcorr(rpscales=scales)
#measurements.desi_elg_xcorr(scales)

#measurements.duty_cycle()
#measurements.haloenergy()
measurements.halopower()