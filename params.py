import numpy as np

linscales = np.logspace(0.7, 1.4, 8)
lensscales = np.logspace(0.7, 1.4, 8)
hodscales = np.logspace(-0.6, 1.5, 21)
#hodscales = np.logspace(-0.5, 1.4, 16)

# luminosity cut in log(W/Hz) for the lower-z samples with photo-zs
lumcut=25.25

use_wiseflags = True

# min and max photo-z cuts for "LzRG" sample
lzrg_minzphot = 0.25
lzrg_maxzphot = 0.5
lzrg_minflux = 20.
lzrg_maxflux = None

# min and max photo-z cuts for "IzRG" sample
izrg_minzphot=0.5
izrg_maxzphot=0.8
izrg_maxflux=None
izrg_minflux = 5.


# minimum photo-z cut to remove interlopers in "HzRG" sample
hzrg_minzphot = 1.
# flux cut for HzRGs in mJy
hzrg_fluxcut = 2.5
hzrg_maxflux = 5000.

lasmax = 100.

# W2 Vega faint magnitude limit
w2faint = 17.25
w2faint_lzrg = 16.5

# let CatWISE magnitudes supercede unWISE when separation is less than this in arcsec
supercede_cw_sep = 2.5

eboss_qso_zbins = [1., 1.5, 2.]

nzbins = 30
zbin_range = (0.1, 4.)

prior_dict = {'M': (13, 1), 'sigM': (0.5, 0.5), 'M1': (14, 1), 'alpha': (1, 0.5)}

def izrg_cut_eqn(w2):
    return (w2 - 17.) / 3. + 0.75
def hzrg_cut_eqn(w2):
    return (17. - w2)/4. + 0.15

min_kinfrac = 0.001
max_kinfrac = 0.005