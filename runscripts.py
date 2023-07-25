import measurements
import sample
import numpy as np
import prep_radio_data

#prep_radio_data.apertif_rms()
#prep_radio_data.lotss_photo_zs()
scales = np.logspace(0.7, 1.4, 11)
#measurements.hzrg_xcorr(rpscales=scales, fcut=2)
#measurements.desi_elg_xcorr(scales)
#measurements.haloenergy()




#measurements.autocorr_izrgs(np.logspace(-0.5, 1.4, 21))
#measurements.lens_izrg()


#measurements.autocorr_hzrgs(scales)
measurements.lens_hzrg()


#measurements.eboss_qso_autocf(scales)
#measurements.fluxcut(scales)
#measurements.lens_hiz(fcut=2)
