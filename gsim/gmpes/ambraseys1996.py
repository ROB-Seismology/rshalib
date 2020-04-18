# -*- coding: iso-Latin-1 -*-
"""
Ambraseys et al (1996) GMPE
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np
from scipy.constants import g

from ...utils import interpolate
from ..gmpe import *



__all__ = ['AmbraseysEtAl1996GMPE']


class AmbraseysEtAl1996GMPE(GMPE):
	"""
	Ambraseys et al. (1996)
		Magnitude scale: MS
		Magnitude range: 4 - 7.9
		Distance metric: Joyner-Boore
		Distance range: 0 - 260 km
		Intensity measure types: PGA, (P?)SA
		Original IMT unit: g
		SA period range: 0.1 - 2 s
		Dampings for SA: 5
		Soil classes:
			rock (vs30 >= 750 m/s)
			stiff (360 <= vs30 < 750 m/s)
			soft (vs30 < 360 m/s)
		Fault types: None
	"""
	def __init__(self):
		imt_periods = {}
		imt_periods["PGA"] = [0]
		imt_periods["SA"] = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17,
							0.18, 0.19, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30,
							0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46,
							0.48, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
							0.85, 0.90, 0.95, 1.00, 1.10, 1.20, 1.30, 1.40,
							1.50, 1.60, 1.70, 1.80, 1.90, 2.00]
		distance_metric = "Joyner-Boore"
		Mmin, Mmax = 4, 7.9
		dmin, dmax = 0., 260.
		Mtype = "MS"
		dampings = [5]
		name = "AmbraseysEtAl1996"
		short_name = "Am_1996"
		GMPE.__init__(self, imt_periods, distance_metric, Mmin, Mmax, dmin, dmax,
						Mtype, dampings, name, short_name)

		## Coefficients
		self.c1 = {}
		self.c1["PGA"] = np.array([-1.48])
		self.c1["SA"] = np.array([-0.84, -0.86, -0.87, -0.87, -0.94, -0.98,
									-1.05, -1.08, -1.13, -1.19, -1.21, -1.28,
									-1.37, -1.40, -1.46, -1.55, -1.63, -1.65,
									-1.69, -1.82, -1.94, -1.99, -2.05, -2.11,
									-2.17, -2.25, -2.38, -2.49, -2.58, -2.67,
									-2.75, -2.86, -2.93, -3.03, -3.10, -3.17,
									-3.30, -3.38, -3.43, -3.52, -3.61, -3.68,
									-3.74, -3.79, -3.80, -3.79])
		self.c2 = {}
		self.c2["PGA"] = np.array([0.266])
		self.c2["SA"] = np.array([0.219, 0.221, 0.231, 0.238, 0.244, 0.247,
									0.252, 0.258, 0.268, 0.278, 0.284, 0.295,
									0.308, 0.318, 0.326, 0.338, 0.349, 0.351,
									0.354, 0.364, 0.377, 0.384, 0.393, 0.401,
									0.410, 0.420, 0.434, 0.438, 0.451, 0.463,
									0.477, 0.485, 0.492, 0.502, 0.503, 0.508,
									0.513, 0.513, 0.514, 0.522, 0.524, 0.520,
									0.517, 0.514, 0.508, 0.503])
		self.h0 = {}
		self.h0["PGA"] = np.array([3.5])
		self.h0["SA"] = np.array([4.5, 4.5, 4.7, 5.3, 4.9, 4.7, 4.4, 4.3, 4.0,
									3.9, 4.2, 4.1, 3.9, 4.3, 4.4, 4.2, 4.2, 4.4,
									4.5, 3.9, 3.6, 3.7, 3.9, 3.7, 3.5, 3.3, 3.1,
									2.5, 2.8, 3.1, 3.5, 3.7, 3.9, 4.0, 4.0, 4.3,
									4.0, 3.6, 3.6, 3.4, 3.0, 2.5, 2.5, 2.4, 2.8,
									3.2])
		self.c4 = {}
		self.c4["PGA"] = np.array([-0.922])
		self.c4["SA"] = np.array([-0.954, -0.945, -0.960, -0.981, -0.955, -0.938,
									-0.907, -0.896, -0.901, -0.907, -0.922, -0.911,
									-0.916, -0.942, -0.946, -0.933, -0.932, -0.939,
									-0.936, -0.900, -0.888, -0.897, -0.908, -0.911,
									-0.920, -0.913, -0.911, -0.881, -0.901, -0.914,
									-0.942, -0.925, -0.920, -0.920, -0.892, -0.885,
									-0.857, -0.851, -0.848, -0.839, -0.817, -0.781,
									-0.759, -0.730, -0.724, -0.728])
		self.ca = {}
		self.ca["PGA"] = np.array([0.117])
		self.ca["SA"] = np.array([0.078, 0.098, 0.111, 0.131, 0.136, 0.143,
									0.152, 0.140, 0.129, 0.133, 0.135, 0.120,
									0.124, 0.134, 0.134, 0.133, 0.125, 0.118,
									0.124, 0.132, 0.139, 0.147, 0.153, 0.149,
									0.150, 0.147, 0.134, 0.124, 0.122, 0.116,
									0.113, 0.127, 0.124, 0.124, 0.121, 0.128,
									0.123, 0.128, 0.115, 0.109, 0.109, 0.108,
									0.105, 0.104, 0.103, 0.101])
		self.cs = {}
		self.cs["PGA"] = np.array([0.124])
		self.cs["SA"] = np.array([0.027, 0.036, 0.052, 0.068, 0.077, 0.085,
									0.101, 0.102, 0.107, 0.130, 0.142, 0.143,
									0.155, 0.163, 0.158, 0.148, 0.161, 0.163,
									0.160, 0.164, 0.172, 0.180, 0.187, 0.191,
									0.197, 0.201, 0.203, 0.212, 0.215, 0.214,
									0.212, 0.218, 0.218, 0.225, 0.217, 0.219,
									0.206, 0.214, 0.200, 0.197, 0.204, 0.206,
									0.206, 0.204, 0.194, 0.182])
		self.sigma = {}
		self.sigma["PGA"] = np.array([0.25])
		self.sigma["SA"] = np.array([0.27, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27,
									0.27, 0.27, 0.28, 0.27, 0.28, 0.28, 0.28,
									0.29, 0.30, 0.31, 0.31, 0.31, 0.31, 0.31,
									0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32,
									0.32, 0.33, 0.32, 0.32, 0.32, 0.32, 0.32,
									0.32, 0.32, 0.31, 0.31, 0.31, 0.31, 0.31,
									0.31, 0.32, 0.32, 0.32])

		## Unit conversion
		self.imt_scaling = {"g": 1.0, "mg": 1E+3, "ms2": g,
							"gal": g*100, "cms2": g*100}

	def __call__(self, M, d, h=0, imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="rock", vs30=None, kappa=None, mechanism="normal",
				damping=5):
		"""
		Compute ground motion for given magnitude, distance, depth,
		soil type, and fault mechanism.

		:param M:
			Float or float array, magnitude(s).
			Note that if d is an array, M must be a float.
		:param d:
			Float or float array, distance(s) in km.
			Note that if M is an array, d must be a float.
		:param h:
			float, depth in km. Ignored in this GMPE.
		:param imt:
			str, one of the supported intensity measure types: "PGA" or "SA"
			(default: "PGA").
		:param T:
			float, spectral period of considered IMT. Ignored if IMT == "PGA"
			(default: 0).
		:param imt_unit:
			str, unit in which intensities should be expressed
			(default: "g")
		:param epsilon:
			float, number of standard deviations above or below the mean
			(default: 0).
		:param soil_type:
			str, either "rock", "stiff" or "soft"
			(default: "rock").
				Rock: VS >= 750 m/s
				Stiff soil: 360 <= VS < 750 m/s
				Soft soil: 180 <= VS < 360 m/s
		:param vs30:
			float, shear-wave velocity in the upper 30 m (in m/s).
			If not None, it takes precedence over the soil_type parameter
			(default: None).
		:param mechanism:
			str, fault mechanism: either "normal", "reverse" or "strike-slip".
			Ignored in this GMPE.
		:param damping:
			float, damping in percent. Only supported value is 5.

		:return:
			Returns a float array with ground-motion values
		"""
		scale_factor = self.imt_scaling[imt_unit.lower()]

		imt = imt.upper()
		if not self.has_imt(imt):
			raise IMTUndefinedError(imt)
		if imt == "PGA":
			C1 = self.c1[imt]
			C2 = self.c2[imt]
			h = self.h0[imt]
			C4 = self.c4[imt]
			CA = self.ca[imt]
			CS = self.cs[imt]
			S = self.sigma[imt]
		else:
			#try:
			#	len(T)
			#except:
			#	T = np.array([T])
			#else:
			#	T = np.array(T)
			if T < self.Tmin(imt) or T > self.Tmax(imt):
				raise PeriodUndefinedError(imt, T)
				#return None
			elif damping not in self.dampings:
				raise DampingNotSupportedError(damping)
			else:
				sa_periods = self.imt_periods["SA"]
				C1 = interpolate(sa_periods, self.c1[imt], [T])
				C2 = interpolate(sa_periods, self.c2[imt], [T])
				h = interpolate(sa_periods, self.h0[imt], [T])
				C4 = interpolate(sa_periods, self.c4[imt], [T])
				CA = interpolate(sa_periods, self.ca[imt], [T])
				CS = interpolate(sa_periods, self.cs[imt], [T])
				S = interpolate(sa_periods, self.sigma[imt], [T])

		if vs30 != None:
			if vs30 >= 750:
				soil_type = "rock"
			elif 360 <= vs30 < 750:
				soil_type = "stiff"
			elif 180 <= vs30 < 360:
				soil_type = "soft"
			else:
				raise VS30OutOfRangeError(vs30)

		if soil_type.lower() == "soft":
			SA, SS = 0, 1
		elif soil_type.lower() == "stiff":
			SA, SS = 1, 0
		else:
			# default is rock
			SA, SS = 0, 0

		r = np.sqrt(d*d + h*h)

		log_y = C1 + C2*M + C4*np.log10(r) + CA*SA + CS*SS + epsilon*S
		y = 10**log_y
		y *= scale_factor

		return y

	def log_sigma(self, M=5., d=10., h=0., imt="PGA", T=0, soil_type="rock",
					vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		Return standard deviation in log10 space
		Note that this value is independent of data scaling (gal, g, m/s**2, ...)

		:param M:
			Float or float array, magnitude(s). Ignored in this GMPE.
			(default: 5.)
		:param d:
			Float or float array, distance(s) in km. Ignored in this GMPE.
			(default: 10.)
		:param imt:
			str, one of the supported intensity measure types: "PGA" or "SA"
			(default: "PGA")
		:param T:
			float, spectral period of considered IMT. Ignored if IMT == "PGA"
			(default: 0)
		:param soil_type:
			str, either "rock", "stiff" or "soft"
				Rock: VS >= 750 m/s
				Stiff soil: 360 <= VS < 750 m/s
				Soft soil: 180 <= VS < 360 m/s
			(default: "rock").
		:param vs30:
			float, shear-wave velocity in the upper 30 m (in m/s).
			If not None, it takes precedence over the soil_type parameter
			(default: None).
		:param kappa:
			float, kappa value, in seconds. Ignored in this GMPE
			(default: None)
		:param mechanism:
			str, focal mechanism. Ignored in this GMPE
			(default: None)
		:param damping:
			float, damping in percent. Only supported value is 5
		"""
		imt = imt.upper()
		if not self.has_imt(imt):
			raise IMTUndefinedError(imt)
		if imt == "PGA":
			return np.array(self.sigma[imt][0])
		else:
			if T < self.Tmin(imt) or T > self.Tmax(imt):
				raise PeriodUndefinedError(imt, T)
				#return None
			elif damping not in self.dampings:
				raise DampingNotSupportedError(damping)
			else:
				return interpolate(self.imt_periods["SA"], self.sigma["SA"], [T])

	def is_rake_dependent(self):
		"""
		Indicate whether or not GMPE depends on rake of the source
		"""
		return False

	def get_crisis_periods(self):
		"""
		Return array of max. 40 spectral periods to be used with CRISIS
		"""
		return np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
						0.19, 0.2, 0.22, 0.24, 0.26, 0.28, 0.30, 0.325, 0.35,
						0.375, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
						0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])

	def plot_figure4(self):
		"""
		Plot Figure 4 in the paper of Ambraseys et al. (1996)
		"""
		self.plot_distance(mags=[5., 6., 7.], dmin=1, dmax=1E3, ymin=1E-3, ymax=1)

	def plot_figure17(self, soil_type="rock"):
		"""
		Plot Figure 17 in the paper of Ambraseys et al. (1996)

		:param soil_type:
			str, either "rock", "stiff" or "soft" (default: "rock").
		"""
		self.plot_spectrum(mags=[5.], d=10, plot_style="lin", Tmin=0, Tmax=2.0,
							amax=0.4, include_pgm=False, soil_type=soil_type,
							want_minor_grid=True)

	def plot_figure18(self, soil_type="rock"):
		"""
		Plot Figure 18 in the paper of Ambraseys et al. (1996)

		:param soil_type:
			str, either "rock", "stiff" or "soft" (default: "rock").
		"""
		self.plot_spectrum(mags=[7.], d=10, plot_style="lin", Tmin=0, Tmax=2.0,
							amax=1.15, include_pgm=False, soil_type=soil_type,
							want_minor_grid=True)

	def plot_figure19(self, soil_type="rock"):
		"""
		Plot Figure 19 in the paper of Ambraseys et al. (1996)

		:param soil_type:
			str, either "rock", "stiff" or "soft" (default: "rock").
		"""
		self.plot_spectrum(mags=[7.], d=40, plot_style="lin", Tmin=0, Tmax=2.0,
							amax=0.37, include_pgm=False, soil_type=soil_type,
							want_minor_grid=True)


if __name__ == "__main__":
	pass

