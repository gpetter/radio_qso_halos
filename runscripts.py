import measurements
import sample
import numpy as np
import prep_radio_data

scales = np.logspace(0.7, 1.4, 11)


#measurements.autocorr_izrgs(np.logspace(-0.5, 1.4, 21))
#measurements.lens_izrg()


#measurements.autocorr_hzrgs(scales)
#measurements.lens_hzrg()

#measurements.hzrg_xcorr(rpscales=scales)
#measurements.desi_elg_xcorr(scales)

#measurements.duty_cycle()
measurements.haloenergy()