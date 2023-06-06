import measurements
import sample
import numpy as np
import prep_radio_data

#prep_radio_data.apertif_rms()

scales = np.logspace(0.5, 1.5, 10)
#measurements.eboss_qso_autocf(scales)
#measurements.extended_jets(scales)
measurements.lotss_selected_xcorr()
#measurements.bal_qso(scales)
#sample.radio_match('eBOSS_QSO', ['LOTSS_DR2', 'Apertif', 'FIRST'], 5.)
#sample.mask_randoms('desi_QSO')
#sample.lotss_deep_match('eBOSS_QSO', 5.)
#measurements.radio_loud_lum(scales, 25)
#measurements.qsolum_allz(scales, npercentiles=10, boss=False)
#measurements.quasarlum_zbins(scales, 5)