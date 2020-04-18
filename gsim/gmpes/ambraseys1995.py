# -*- coding: iso-Latin-1 -*-
"""
Ambraseys (1995) GMPE
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.constants import g

from ..gmpe import (GMPE, IMTUndefinedError)


__all__ = ['Ambraseys1995DDGMPE']


class Ambraseys1995DDGMPE(GMPE):
	"""
	Ambraseys (1995) depth-dependent
		Magnitude scale: MS
		Magnitude range: 4.0 - 7.3
		Distance range: 0 - 260 km
	"""
	def __init__(self):
		imt_periods = {}
		imt_periods["PGA"] = [0]
		distance_metric = "Hypocentral"
		Mmin, Mmax = 4.0, 7.3
		dmin, dmax = 0., 260.
		Mtype = "MS"
		dampings = [5]
		name = "Ambraseys1995DD"
		short_name = "Am_1995DD"
		GMPE.__init__(self, imt_periods, distance_metric, Mmin, Mmax, dmin, dmax,
					Mtype, dampings, name, short_name)

		## Coefficients
		self.a = {}
		self.a['PGA'] = np.array([-1.06])
		self.b = {}
		self.b['PGA'] = np.array([0.245])
		self.c = {}
		self.c['PGA'] = np.array([-0.00045])
		self.d = {}
		self.d['PGA'] = np.array([-1.016])
		self.sigma = {}
		self.sigma['PGA'] = np.array([.25])

		## Unit conversion
		self.imt_scaling = {"g": 1.0, "mg": 1E+3, "ms2": g,
							"gal": g*100, "cms2": g*100}

	def __call__(self, M, d, h=0, imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="rock", vs30=None, kappa=None, mechanism="normal",
				damping=5):
		"""
		"""
		scale_factor = self.imt_scaling[imt_unit.lower()]

		imt = imt.upper()
		if not self.has_imt(imt):
			raise IMTUndefinedError(imt)
		if imt == "PGA":
			A = self.a[imt]
			B = self.b[imt]
			C = self.c[imt]
			D = self.d[imt]
			S = self.sigma[imt]
		else:
			pass

		r = np.sqrt(d*d + h*h)

		log_y = A + B*M + C*r + D*np.log10(r) + epsilon*S
		y = 10**log_y
		y *= scale_factor

		return y

	def log_sigma(self, M=5., d=10., h=0., imt="PGA", T=0, soil_type="rock",
				vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		"""
		imt = imt.upper()
		if not self.has_imt(imt):
			raise IMTUndefinedError(imt)
		if imt == "PGA":
			return np.array(self.sigma[imt][0])
		else:
			pass

	def is_rake_dependent(self):
		"""
		Indicate whether or not GMPE depends on rake of the source
		"""
		return False


if __name__ == "__main__":
	pass

