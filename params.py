import numpy as np

linscales = np.logspace(0.7, 1.4, 11)
lensscales = np.logspace(0.7, 1.4, 8)
hodscales = np.logspace(-0.5, 1.4, 21)

# luminosity cut in log(W/Hz) for the lower-z samples with photo-zs
lumcut=25.

# min and max photo-z cuts for "LzRG" sample
lzrg_minzphot = 0.25
lzrg_maxzphot = 0.5
lzrg_minflux = 20.
lzrg_maxflux = 2000.

# min and max photo-z cuts for "IzRG" sample
izrg_minzphot=0.5
izrg_maxzphot=0.8
izrg_maxflux=1000.
izrg_minflux = 5.


# minimum photo-z cut to remove interlopers in "HzRG" sample
hzrg_minzphot = 0.9
# flux cut for HzRGs in mJy
hzrg_fluxcut = 2.
hzrg_maxflux = 1000.

lasmax = 30.

# W2 Vega faint magnitude limit
w2faint = 17.5

# let CatWISE magnitudes supercede unWISE when separation is less than this in arcsec
supercede_cw_sep = 2.5

def izrg_cut_eqn(w2):
    return (w2 - 17.) / 3. + 0.75
def hzrg_cut_eqn(w2):
    return (17. - w2)/4. + 0.15
