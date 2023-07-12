import measurements
import sample
import numpy as np
import prep_radio_data

#prep_radio_data.apertif_rms()
#prep_radio_data.lotss_photo_zs()
scales = np.logspace(0.7, 1.4, 11)
#measurements.eboss_qso_autocf(scales)
#measurements.fluxcut(scales)
measurements.lens_hiz(fcut=10)
#measurements.extended_jets(scales)
#measurements.lotss_selected_xcorr(rpscales=scales, fcut=10)
#measurements.bal_qso(scales)
#sample.radio_match('eBOSS_QSO', ['LOTSS_DR2', 'Apertif', 'FIRST'], 5.)
#sample.mask_randoms('eBOSS_QSO')
#sample.lotss_deep_match('eBOSS_QSO', 5.)
#measurements.radio_loud_lum(scales, 25)
#measurements.qsolum_allz(scales, npercentiles=10, boss=False)
#measurements.quasarlum_zbins(scales, 5)
#measurements.radio_luminosity(scales)