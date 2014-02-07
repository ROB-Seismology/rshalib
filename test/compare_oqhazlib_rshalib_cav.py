"""
Test CAV implementation in openquake.hazardlib
"""

import numpy as np
import openquake.hazardlib as oqhazlib
import hazard.rshalib as rshalib


gmpe = rshalib.gsim.AkkarBommer2010()
oqgmpe = oqhazlib.gsim.akkar_bommer_2010.AkkarBommer2010()

M = 5.5
d = np.array([10])
h = 0
vs30 = np.array([800])
kappa = None
mechanism = "normal"
im = "SA"
T = 0.5
damping = 5
imls = np.arange(0.1, 1.0, 0.1)
truncation_level = 3
cav_min = 0.16
depsilon = 0.02
eps_correlation_model = "EUS"

sctx, rctx, dctx, imt = gmpe._get_contexts_and_imt(M, d, h, vs30, kappa, mechanism, im, T, damping)

rshalib_poes = gmpe.get_exceedance_probability(imls, M, d, imt=im, T=T, vs30=vs30, kappa=kappa, mechanism=mechanism, truncation_level=truncation_level)
oqhazlib_poes = oqgmpe.get_poes(sctx, rctx, dctx, imt, imls, truncation_level)
#print rshalib_poes
#print oqhazlib_poes

print

rshalib_poes_cav = gmpe.get_exceedance_probability_cav(imls, M, d, imt=im, T=T, vs30=vs30, kappa=kappa, mechanism=mechanism, truncation_level=truncation_level, cav_min=cav_min, depsilon=depsilon, eps_correlation_model=eps_correlation_model)
oqhazlib_poes_cav = oqgmpe.get_poes_cav(sctx, rctx, dctx, imt, imls, truncation_level, cav_min=cav_min, depsilon=depsilon, eps_correlation_model=eps_correlation_model)
print rshalib_poes_cav
print oqhazlib_poes_cav