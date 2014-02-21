"""
Compare poes and disagg_poes
"""

import datetime
import numpy as np
import openquake.hazardlib as oqhazlib
import hazard.rshalib as rshalib


gmpe = rshalib.gsim.AkkarBommer2010()
gsim = oqhazlib.gsim.akkar_bommer_2010.AkkarBommer2010()

M = 6.
d = np.array([10.])
h = 0
vs30 = np.array([800])
kappa = None
sof = "normal"
im = "PGA"
T = 0
damping = 5.
imls = np.arange(0.1, 1.0, 0.1)
truncation_level = 3
num_epsilons = 6

print gmpe(M, d, h=h, vs30=vs30, kappa=kappa, mechanism=sof, imt=im, T=T, damping=damping)

sctx, rctx, dctx, imt = gmpe._get_contexts_and_imt(M, d, h, vs30, kappa, sof, im, T, damping)

start = datetime.datetime.now()
for i in range(100):
	poes = gsim.get_poes(sctx, rctx, dctx, imt, imls, truncation_level)
end = datetime.datetime.now()
print end - start
start = datetime.datetime.now()
for i in range(100):
	disagg_poes_by_eps = gsim.disaggregate_poe(sctx, rctx, dctx, imt, imls, truncation_level, num_epsilons)
end = datetime.datetime.now()
print end - start
disagg_poes = np.sum(disagg_poes_by_eps, axis=1)

print imls
print poes
print disagg_poes
print np.allclose(poes, disagg_poes)

## Conclusion: disaggregate_poe gives same result as get_poes
