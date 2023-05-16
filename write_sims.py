from analytic.constants import *
from analytic.solver import Evolve_RG
import pickle
import itertools
import schwimmbad


zs = np.array([0.5, 1., 1.5, 2., 2.5, 3.])
ms = np.logspace(12.5, 13.5, 5)
qs = np.logspace(35., 40.5, 12)


ts = np.logspace(-5,np.log10(500), 50)
outdict = {'logt_myr': np.log10(ts)}
paramarr = list(itertools.product(*[ms,qs]))
print('%s number of runs' % len(paramarr))

def runsim(j):
	try:
		paramtup = paramarr[j]
		m, q = paramtup
		env = Evolve_RG('universal',M500=m, z=0)
		env.solve(q,ts*Myr)
		env.findsynch(144e6)
		newdicts = []
		for z in zs:
			env.findcorrection((144e6, 1400e6), z)
			newdicts.append({'z':z, 'Q': np.log10(q), 'Menv': np.log10(m), 'r': np.log10(env.R/kpc), 'L_144': np.log10(env.corr_synch[:, 0]), 'L_1400': np.log10(env.corr_synch[:, 1])})
		return newdicts
	except:
		return [None]


if __name__ == "__main__":
	pool = schwimmbad.choose_pool(mpi=False, processes=12)
	runs = pool.map(runsim, np.arange(len(paramarr)))
	flat_list = [item for sublist in runs for item in sublist]
	outdict['runs'] = flat_list
	with open('sims/test.pickle', 'wb') as f:
		pickle.dump(outdict, f)