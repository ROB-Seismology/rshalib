# -*- coding: iso-Latin-1 -*-
"""
McGuire (1974) GMPE
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np
from scipy.constants import g

from ...utils import interpolate
from .base import *



__all__ = ['McGuire1974GMPE']


class McGuire1974GMPE(GMPE):
	"""
	Class representing McGuire (1974) GMPE (as described in McGuire, 1977)
	Completely untested.
	Only interesting for its different IMT's (PGA, PGV, PGD, and PSV).
	No standard deviation!

		Magnitude scale: ML
		Magnitude range: 5.3 - 7.6
		Distance metric: Hypocentral
		Distance range: 14 - 125 km
		Intensity measure types: PGA, PGV, PGD, (P)SV
		PSV period range: 0.1 - 8 s
		Damping for PSV: 2 percent [0, 5, and 10 are also in original
			report, but not in 1977 publication]
		Soil classes: None
		Fault types: None
	"""
	def __init__(self):
		psv_freqs = np.array([0.125, 0.15, 0.2, 0.25, 0.32, 0.5, 0.64, 1., 1.3,
							1.8, 2., 2.5, 3.2, 5., 6.7, 10.])
		imt_periods = {}
		imt_periods["PGA"] = np.array([0.])
		imt_periods["PGV"] = np.array([0.])
		imt_periods["PGD"] = np.array([0.])
		imt_periods["PSV"] = 1. / psv_freqs
		distance_metric = "Hypocentral"
		Mmin, Mmax = 5.3, 7.6
		dmin, dmax = 14., 125.
		Mtype = "ML"
		#dampings = [0, 2, 5, 10]
		dampings = [2]
		name = "McGuire1974"
		short_name = "McG_1974"
		GMPE.__init__(self, imt_periods, distance_metric, Mmin, Mmax, dmin, dmax,
					Mtype, dampings, name, short_name)

		## Coefficients
		self.a = {}
		self.a["PGA"] = 472
		self.a["PGV"] = 5.64
		self.a["PGD"] = 0.393
		## TO DO: value below needs to be checked
		self.a["PSV"] = 5.64

		self.b = {}
		self.b["PGA"] = np.array([0.278])
		self.b["PGV"] = np.array([0.401])
		self.b["PGD"] = np.array([0.434])
		self.b["PSV"] = np.array([0.4, 0.45, 0.55, 0.53, 0.45, 0.44, 0.4, 0.36,
								0.38, 0.4, 0.35, 0.34, 0.3, 0.25, 0.17, 0.24])

		self.c = {}
		self.c["PGA"] = np.array([1.301])
		self.c["PGV"] = np.array([1.202])
		self.c["PGD"] = np.array([0.885])
		self.c["PSV"] = np.array([0.7, 0.68, 0.92, 0.72, 0.62, 0.5, 0.45, 0.58,
								0.9, 1.2, 1.16, 1.2, 1.32, 1.25, 1.32,1.3])

		## Unit conversion
		## PSV: cm/sec, PGD: cm
		self.imt_scaling = {}
		self.imt_scaling["PGA"] = {"g": 0.01/g, "mg": 10./g, "ms2": 1E-2, "gal": 1.0}
		self.imt_scaling["PGV"] = {"ms": 1E-2, "cms": 1.0}
		self.imt_scaling["PGD"] = {"m": 1E-2, "cm": 1.0}
		self.imt_scaling["PSV"] = self.imt_scaling["PGV"]

	def __call__(self, M, d, h=0, imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type=None, vs30=None, kappa=None, mechanism="normal",
				damping=2):
		"""
		Return ground motion for given magnitude, distance, depth,
		soil type, and fault mechanism.

		:param M:
			Float or float array, magnitude(s).
			Note that if d is an array, M must be a float.
		:param d:
			Float or float array, distance(s) in km.
			Note that if M is an array, d must be a float.
		:param h:
			float, focal depth in km
			(default: 0., i.e. assume d is hypocentral distance).
		:param imt:
			str, one of the supported intensity measure types: "PGA", "PGV",
			"PGD" or "PSV"
			(default: "PGA").
		:param T:
			float, spectral period of considered IMT.
			Only relevant if IMT == "PSV"
			(default: 0).
		:param imt_unit:
			str, unit in which intensities should be expressed
			(default: "g")
		:param epsilon:
			float, number of standard deviations above or below the mean.
			Ignored because this GMPE does not specify a standard deviation.
			(default: 0).
		:param soil_type:
			str, zoil type. Ignored in this GMPE
			(default: "rock").
		:param vs30:
			float, shear-wave velocity in the upper 30 m (in m/s). Ignored.
			(default: None).
		:param kappa:
			float, kappa value, in seconds. Ignored in this GMPE
			(default: None)
		:param mechanism:
			str, fault mechanism: either "normal", "reverse" or "strike-slip".
			Ignored in this GMPE
			(default: "normal")
		:param damping:
			float, damping in percent. The only supported value is 2
			(although the paper also mentions undamped, 5, and 10)
			(default: 2).

		:return:
			Returns a float array with ground-motion intensities
		"""
		imt = imt.upper()
		scale_factor = self.imt_scaling[imt][imt_unit.lower()]

		if not self.has_imt(imt):
			raise IMTUndefinedError(imt)
		if imt in ("PGA", "PGV", "PGD"):
			A = self.a[imt]
			B = self.b[imt]
			C = self.c[imt]
		elif imt == "PSV":
			if T < self.Tmin(imt) or T > self.Tmax(imt):
				raise PeriodUndefinedError(imt, T)
				#return None
			elif damping not in self.dampings:
				raise DampingNotSupportedError(damping)
			else:
				psv_periods = self.imt_periods[imt]
				A = self.a[imt]
				B = interpolate(psv_periods, self.b[imt], [T])
				C = interpolate(psv_periods, self.c[imt], [T])

		ah = A * 10**(B*M) * (d + 25)**-C
		ah *= scale_factor
		return ah

	def log_sigma(self, M=5., d=10., h=0., imt="PGA", T=0, soil_type="rock",
				vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		Return standard deviation in log10 space
		Note that this value is independent of data scaling (gal, g, m/s**2, ...)

		:param M:
			Float or float array, magnitude(s).
			Note that if d is an array, M must be a float.
		:param d:
			Float or float array, distance(s) in km.
			Note that if M is an array, d must be a float.
		:param h:
			float, focal depth in km
			(default: 0., i.e. assume d is hypocentral distance).
		:param imt:
			str, one of the supported intensity measure types: "PGA", "PGV",
			"PGD" or "PSV"
			(default: "PGA").
		:param T:
			float, spectral period of considered IMT.
			Only relevant if IMT == "PSV"
			(default: 0).
		:param soil_type:
			str, zoil type. Ignored in this GMPE (default: "rock").
		:param vs30:
			float, shear-wave velocity in the upper 30 m (in m/s). Ignored.
			(default: None).
		:param kappa:
			float, kappa value, in seconds. Ignored in this GMPE
			(default: None)
		:param mechanism:
			str, fault mechanism: either "normal", "reverse" or "strike-slip".
			Ignored in this GMPE
			(default: "normal")
		:param damping:
			float, damping in percent. The only supported value is 2
			(although the paper also mentions undamped, 5, and 10)
			(default: 2).
		"""
		if not self.has_imt(imt):
			raise IMTUndefinedError(imt)
		if imt == "PSV":
			if T < self.Tmin(imt) or T > self.Tmax(imt):
				raise PeriodUndefinedError(imt, T)
			elif damping not in self.dampings:
				raise DampingNotSupportedError(damping)
		return np.array([1E-4])

	def is_rake_dependent(self):
		"""
		Indicate whether or not GMPE depends on rake of the source
		"""
		return False


if __name__ == "__main__":
	pass

