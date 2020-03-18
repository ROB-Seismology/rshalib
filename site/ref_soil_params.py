"""
Reference soil parameters.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from . import vs30


# TODO: add ref_kappa
# TODO: add ref_thickness


REF_VS30 = vs30.ROCK
REF_VS30MEASURED = False
REF_Z1PT0 = 100.
REF_Z2PT5 = 2.
REF_KAPPA = None

REF_SOIL_PARAMS = {"vs30": REF_VS30,
					"vs30measured": REF_VS30MEASURED,
					"z1pt0": REF_Z1PT0,
					"z2pt5": REF_Z2PT5,
					"kappa": REF_KAPPA}
