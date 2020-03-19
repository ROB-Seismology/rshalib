# -*- coding: iso-Latin-1 -*-

"""
base GMPE class
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys

import numpy as np

#from openquake.hazardlib.imt import PGD, PGV, PGA, SA, MMI

from ...utils import interpolate, logrange



__all__ = ['GMPE', 'CRISIS_DISTANCE_METRICS',
			'IMTUndefinedError', 'PeriodUndefinedError',
			'DampingNotSupportedError', 'VS30OutOfRangeError',
			'SoilTypeNotSupportedError', 'adjust_hard_rock_to_rock',
			'adjust_host_to_target', 'adjust_components',
			'convert_distance_metric', 'sd2psa', 'sd2psv']


def sd2psa(sd, T):
	"""
	Convert spectral displacement to (pseudo)spectral acceleration

	:param sd:
		Float or numpy array, spectral displacement(s)
	:param T:
		float, spectral period
	"""
	omega = 2 * np.pi / T
	return sd * omega * omega


def sd2psv(sd, T):
	"""
	Convert spectral displacement to (pseudo)spectral velocity

	:param sd:
		Float or numpy array, spectral displacement(s)
	:param T:
		float, spectral period
	"""
	omega = 2 * np.pi / T
	return sd * omega


## Map distance metrics to constants used in CRISIS
CRISIS_DISTANCE_METRICS = {
	"Hypocentral": 1,
	"Epicentral": 2,
	"Joyner-Boore": 3,
	"Rupture": 4}


## Error definitions
class IMTUndefinedError(Exception):
	def __init__(self, imt):
		super(IMTUndefinedError, self).__init__("")
		self.imt = imt

	def __str__(self):
		return "IMT %s not defined!" % self.imt


class PeriodUndefinedError(Exception):
	def __init__(self, imt, period):
		super(PeriodUndefinedError, self).__init__("")
		self.imt = imt
		self.period = period

	def __str__(self):
		return "Period %s s not defined for imt %s!" % (self.imt, self.period)


class DampingNotSupportedError(Exception):
	def __init__(self, damping):
		super(DampingNotSupportedError, self).__init__("")
		self.damping = damping

	def __str__(self):
		return "Damping %s%% not supported!" % self.damping


class VS30OutOfRangeError(Exception):
	def __init__(self, vs30):
		super(VS30OutOfRangeError, self).__init__("")
		self.vs30 = vs30

	def __str__(self):
		return "VS30 = %s m/s out of range!" % self.vs30


class SoilTypeNotSupportedError(Exception):
	def __init__(self, soil_type):
		super(SoilTypeNotSupportedError, self).__init__("")
		self.soil_type = soil_type

	def __str__(self):
		return "Soil type %s not supported!" % self.soil_type


class GMPE(object):
	def __init__(self, imt_periods, distance_metric, Mmin, Mmax, dmin, dmax, Mtype,
				dampings, name, short_name):
		"""
		Base class for a GMPE (Ground-Motion Prediction Equation) or
		attenuation law.

		:param imt_periods:
			dict, mapping intensity measure types (e.g. "PGA", "SA", "PGV",
			"PGD") to lists or arrays of periods in seconds (float values).
			Periods must be monotonically increasing or decreasing
		:param distance_metric:
			str, Distance metric used in this GMPE: "Hypocentral",
			"Epicentral", "Joyner-Boore" or "Rupture"
		:param Mmin:
			float, Lower bound of valid magnitude range
		:param Mmax:
			float, Upper bound of valid magnitude range
		:param dmin:
			float, Lower bound of valid distance range in km
		:param dmax:
			float, Upper bound of valid distance range in km
		:param Mtype:
			float, Magnitude type used in this GMPE: "MW", "MS", ...
		:param dampings:
			list of damping values supported by this GMPE
		:param name:
			str, full name of this GMPE
		:param short_name:
			str, abbreviated name of this GMPE

		Note: Objects derived from this class should implement __call__ and
		log_sigma methods
		"""
		self.imt_periods = {}
		for imt in imt_periods.keys():
			self.imt_periods[imt] = np.array(imt_periods[imt])
		self.distance_metric = distance_metric
		self.Mmin = Mmin
		self.Mmax = Mmax
		self.dmin = dmin
		self.dmax = dmax
		self.Mtype = Mtype
		self.dampings = dampings
		self.name = name
		self.short_name = short_name

		self.__name__ = self.__class__.__name__

	def __repr__(self):
		return '<GMPE %s>' % self.name

	def has_imt(self, imt):
		"""
		Check if given intensity measure type is supported.

		:param imt:
			str, intensity measure type: e.g., "PGA", "PGV", "PGD", "SA"
		"""
		return imt.upper() in self.imt_periods

	@property
	def imts(self):
		"""
		List of supported intensity measure types.
		"""
		return self.imt_periods.keys()

	@property
	def sa_periods(self):
		"""
		List of spectral periods for IMT SA
		"""
		return self.imt_periods["SA"]

	@property
	def sa_freqs(self):
		"""
		List of spectral frequencies for IMT SA
		"""
		return 1. / self.sa_periods

	def freqs(self, imt="SA"):
		"""
		Return spectral frequencies for given IMT

		:param imt: intensity measure type (default: "SA")
		"""
		return 1. / self.imt_periods[imt]

	def Tmin(self, imt="SA"):
		"""
		Return minimum period in seconds  for given IMT

		:param imt:
			str, intensity measure type (default: "SA")
		"""
		return self.imt_periods[imt].min()

	def Tmax(self, imt="SA"):
		"""
		Return maximum period in seconds for given IMT

		:param imt:
			str, intensity measure type (default: "SA")
		"""
		return self.imt_periods[imt].max()

	def fmin(self, imt="SA"):
		"""
		Return minimum frequency in Hz for given IMT. If Tmax is 0, return
		value is 34 Hz

		:param imt:
			str, intensity measure type (default: "SA")
		"""
		try:
			return 1.0 / self.Tmax(imt=imt)
		except:
			return 34

	def fmax(self, imt="SA"):
		"""
		Return maximum frequency in Hz for given IMT. If Tmin is 0, return
		value is 34 Hz

		:param imt:
			str, intensity measure type (default: "SA")
		"""
		try:
			return 1.0 / self.Tmin(imt=imt)
		except:
			return 34

	def get_epsilon(self, iml, M, d, h=0., imt="PGA", T=0., imt_unit="g",
					soil_type="rock", vs30=None, kappa=None, mechanism="normal",
					damping=5):
		"""
		Determine epsilon for given intensity measure level(s), magnitude,
		and distance

		:param iml:
			Float or numpy array, intensity measure level(s)
		:param M:
			float, magnitude.
		:param d:
			float, distance in km.
		:param h:
			float, depth in km. Ignored if distance metric of GMPE is
			epicentral or Joyner-Boore
			(default: 0).
		:param imt:
			str, one of the supported intensity measure types: "PGA" or "SA"
			(default: "SA").
		:param T:
			float, spectral period of considered IMT. Ignored if IMT is
			"PGA" or "PGV"
			(default: 0).
		:param imt_unit:
			str, unit in which intensities should be expressed,
			depends on IMT
			(default: "g")
		:param soil_type:
			str, one of the soil types supported by the GMPE
			(default: "rock").
		:param vs30:
			float, shear-wave velocity in the upper 30 m (in m/s).
			If not None, it takes precedence over the soil_type parameter
			(default: None).
		:param kappa:
			float, kappa value, in seconds
			(default: None)
		:param mechanism:
			str, fault mechanism: either "normal", "reverse" or "strike-slip".
			(default: "normal").
		:param damping:
			float, damping in percent.
			(default: 5.).

		:return:
			float array, epsilon value(s)
		"""
		im_med = self.__call__(M, d, h=h, imt=imt, T=T, imt_unit=imt_unit,
								soil_type=soil_type, vs30=vs30, kappa=kappa,
								mechanism=mechanism, damping=damping)
		log_sigma = self.log_sigma(M, d, h=h, imt=imt, T=T, soil_type=soil_type,
									vs30=vs30, kappa=kappa, mechanism=mechanism,
									damping=damping)
		epsilon = (np.log10(iml) - np.log10(im_med)) / log_sigma
		return epsilon

	def get_exceedance_probability(self, iml, M, d, h=0., imt="PGA", T=0.,
									imt_unit="g", soil_type="rock", vs30=None,
									kappa=None, mechanism="normal", damping=5,
									truncation_level=3):
		"""
		Compute probability of exceedance for given intensity measure level(s),
		magnitude, and distance

		:param iml:
			Float or numpy array, intensity measure level(s)
		:param M:
			float, magnitude.
		:param d:
			float, distance in km.
		:param h:
			float, depth in km. Ignored if distance metric of GMPE is
			epicentral or Joyner-Boore
			(default: 0).
		:param imt:
			str, one of the supported intensity measure types: "PGA" or "SA"
			(default: "PGA").
		:param T:
			float, spectral period of considered IMT. Ignored if IMT is
			"PGA" or "PGV"
			(default: 0).
		:param imt_unit:
			str, unit in which intensities are expressed,
			depends on IMT
			(default: "g")
		:param soil_type:
			str, one of the soil types supported by the GMPE
			(default: "rock").
		:param vs30:
			float, shear-wave velocity in the upper 30 m (in m/s).
			If not None, it takes precedence over the soil_type parameter
			(default: None).
		:param kappa:
			float, kappa value, in seconds (default: None)
		:param mechanism:
			str, fault mechanism: either "normal", "reverse" or "strike-slip".
			(default: "normal").
		:param damping:
			float, damping in percent
			(default: 5.).
		:param truncation_level:
			float, number of standard deviations at which to truncate GMPE
			(default: 3)

		:return:
			float array, probability(ies) of exceedance for each iml
		"""
		import scipy.stats
		log_iml = np.log10(iml)
		median = self.__call__(M, d, h=h, imt=imt, T=T, imt_unit=imt_unit,
							epsilon=0, soil_type=soil_type, vs30=vs30,
							kappa=kappa, mechanism=mechanism, damping=damping)
		log_median = np.log10(median)
		if truncation_level == 0:
			return (log_iml > log_median) * 1.0
		else:
			log_sigma = self.log_sigma(M, d, h=h, imt=imt, T=T, soil_type=soil_type,
									vs30=vs30, kappa=kappa, mechanism=mechanism,
									damping=damping)
			if truncation_level is None:
				dist = scipy.stats.norm(log_median, log_sigma)
			else:
				dist = scipy.stats.truncnorm(-truncation_level, truncation_level,
											log_median, log_sigma)
			return 1 - dist.cdf(log_iml)

	def get_exceedance_probability_cav(self, iml, M, d, h=0., imt="PGA", T=0.,
									imt_unit="g", soil_type="rock", vs30=None,
									kappa=None, mechanism="normal", damping=5,
									truncation_level=3, cav_min=0.16, depsilon=0.02,
									eps_correlation_model="EUS"):
		"""
		Compute joint probability of exceeding given intensity level(s)
		and exceeding a minimum CAV value

		:param iml:
			Float or numpy array, intensity measure level(s)
		:param M:
			float, magnitude.
		:param d:
			float, distance in km.
		:param h:
			float, depth in km. Ignored if distance metric of GMPE is
			epicentral or Joyner-Boore
			(default: 0).
		:param imt:
			str, one of the supported intensity measure types: "PGA" or "SA"
			(default: "PGA").
		:param T:
			float, spectral period of considered IMT. Ignored if IMT is
			"PGA" or "PGV"
			(default: 0).
		:param imt_unit:
			str, unit in which intensities are expressed,
			depends on IMT
			(default: "g")
		:param soil_type:
			str, one of the soil types supported by the GMPE
			(default: "rock").
		:param vs30:
			float, shear-wave velocity in the upper 30 m (in m/s).
			If not None, it takes precedence over the soil_type parameter
			(default: None).
		:param kappa:
			float, kappa value, in seconds (default: None)
		:param mechanism:
			str, fault mechanism: either "normal", "reverse" or "strike-slip".
			(default: "normal").
		:param damping:
			float, damping in percent
			(default: 5.).
		:param truncation_level:
			float, number of standard deviations at which to truncate GMPE
			(default: 3).
		:param cav_min:
			float, CAV threshold
			(default: 0.16 g.s).
		:param depsilon:
			float, bin width used to discretize epsilon pga. Should be
			sufficiently small for PGA, but is less sensitive for SA
			(default: 0.02).
		:param eps_correlation_model:
			Str, name of model used for correlation of epsilon values
			of PGA and SA, either "WUS" or "EUS"
			(default: "EUS").

		:return:
			float array, joint probability(ies) for each iml
		"""
		import scipy.stats
		from ..utils import interpolate
		from ..cav import calc_cav_exceedance_prob

		## TODO: if soil_type is used, convert to vs30
		if vs30 is None and soil_type == "rock":
			vs30 = 800

		iml = np.asarray(iml)

		## Truncated normal distribution of epsilon values
		## and corresponding probabilities
		if imt == "SA":
			#pga_truncation_level = truncation_level + 1
			pga_truncation_level = truncation_level
		else:
			pga_truncation_level = truncation_level
		neps = int(pga_truncation_level / depsilon) * 2 + 1
		eps_pga_list = np.linspace(-pga_truncation_level, pga_truncation_level, neps)
		eps_dist = scipy.stats.truncnorm(-pga_truncation_level, pga_truncation_level)
		prob_eps_list = eps_dist.pdf(eps_pga_list) * depsilon
		#prob_eps_list /= np.cumsum(prob_eps_list)

		## Pre-calculate PGA and CAV exceedance probabilities for epsilon PGA
		pga_eps_pga_list = np.zeros_like(eps_pga_list)
		cav_exceedance_prob = np.zeros_like(eps_pga_list)
		for e, eps_pga in enumerate(eps_pga_list):
			## Determine PGA corresponding to eps_pga
			pga = self.__call__(M, d, h=h, imt="PGA", T=0, epsilon=eps_pga,
								imt_unit=imt_unit, soil_type=soil_type, vs30=vs30,
								kappa=kappa, mechanism=mechanism, damping=damping)
			pga_eps_pga_list[e] = pga
			## CAV exceedance probability for PGA
			cav_exceedance_prob[e] = calc_cav_exceedance_prob(pga, M, vs30,
										cav_min=cav_min, duration_dependent=True)

		joint_exceedance_prob = np.zeros_like(iml)

		if imt == "PGA":
			## Integrate explicitly over epsilon
			for e in range(len(eps_pga_list)):
				## Sum only epsilon contributions larger than considered iml
				## in order to obtain exceedance probability (survival function)
				joint_exceedance_prob[pga_eps_pga_list[e] > iml] += (prob_eps_list[e]
															* cav_exceedance_prob[e])

		elif imt == "SA":
			## Correlation coefficients between PGA and SA (Table 3-1)
			#if cav_min > 0:
			b1_freqs = np.array([0.5, 1, 2.5, 5, 10, 20, 25, 35])
			b1_WUS = np.array([0.59, 0.59, 0.6, 0.633, 0.787, 0.931, 0.956, 0.976])
			b1_EUS = np.array([0.5, 0.55, 0.6, 0.75, 0.88, 0.9, 0.91, 0.93])
			if eps_correlation_model == "WUS":
				b1 = interpolate(b1_freqs, b1_WUS, [1./T])[0]
			elif eps_correlation_model == "EUS":
				b1 = interpolate(b1_freqs, b1_EUS, [1./T])[0]

				#print("PGA truncation level: ", truncation_level / b1)
			#else:
				## This makes:
				## eps_sa = 0
				## sigma_ln_sa_given_pga = sigma_ln_sa
				## ln_sa_given_pga = ln_sa_med
			#	b1 = 0

			## Determine median SA and sigma
			sa_med = self.__call__(M, d, h=h, imt="SA", T=T, imt_unit=imt_unit,
									soil_type=soil_type, vs30=vs30, kappa=kappa,
									mechanism=mechanism, damping=damping)
			ln_sa_med = np.log(sa_med)
			sigma_log_sa = self.log_sigma(M, d, h=h, imt="SA", T=T,
										soil_type=soil_type, vs30=vs30, kappa=kappa,
										mechanism=mechanism, damping=damping)
			sigma_ln_sa = np.log(10) * sigma_log_sa

			## Determine true sa_epsilon corresponding to iml
			#true_eps_sa = self.get_epsilon(iml, M, d, h=h, imt=imt, T=T, imt_unit=imt_unit, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
			#prob_true_eps_sa = eps_dist.pdf(true_eps_sa)
			#prob_true_eps_sa /= np.cumsum(prob_true_eps_sa)

			ln_iml = np.log(iml)

			## Eq. 3-3
			sigma_ln_sa_given_pga = np.sqrt(1 - b1**2) * sigma_ln_sa

			## Integrate over eps_pga (inner integral of Eq. 4.1)
			for e, eps_pga in enumerate(eps_pga_list):
				## Determine epsilon value of SA, and sigma
				## Eq. 3-1
				eps_sa = b1 * eps_pga

				## Eq. 3-2
				ln_sa_given_pga = ln_sa_med + eps_sa * sigma_ln_sa

				## Determine probability of exceedance of SA given PGA
				## Eq. 4-3
				eps_sa_dot = (ln_iml - ln_sa_given_pga) / sigma_ln_sa_given_pga

				## Original implementation
				## This has two problems: exceeds non-CAV filtered poes at large
				## accelerations, and is not correctly truncated
				## Eq. 4-2
				poe_sa_given_pga = 1.0 - eps_dist.cdf(eps_sa_dot)
				#poe_sa_given_pga = 1.0 - scipy.stats.truncnorm.cdf(ln_iml, -truncation_level, truncation_level, loc=ln_sa_given_pga, scale=sigma_ln_sa_given_pga)
				joint_exceedance_prob += (prob_eps_list[e] * cav_exceedance_prob[e] * poe_sa_given_pga)

				## replace poe_sa_given_pga with prob_sa_given_pga
				#prob_sa_given_pga = eps_dist.pdf(eps_sa_dot) * depsilon
				#idxs = np.where(ln_sa_given_pga > ln_iml)
				#joint_exceedance_prob[idxs] += (prob_eps_list[e] * cav_exceedance_prob[e] * prob_sa_given_pga[idxs])
				#joint_exceedance_prob[idxs] += (cav_exceedance_prob[e] * prob_sa_given_pga[idxs])

				## This does not take into account sigma_ln_sa_given_pga
				#joint_exceedance_prob[ln_sa_given_pga > ln_iml] += (prob_eps_list[e] * cav_exceedance_prob[e])

				#prob_eps = scipy.stats.truncnorm.pdf(??, -pga_truncation_level, pga_truncation_level, ln_sa_given_pga, sigma_ln_sa_given_pga) * depsilon
				#prob_eps = scipy.stats.truncnorm.pdf(eps_pga, -pga_truncation_level, pga_truncation_level) * depsilon
				#joint_exceedance_prob[ln_sa_given_pga > ln_iml] += (prob_eps * cav_exceedance_prob[e])

			## Workaround to make sure SA values are properly truncated
			sa_exceedance_prob = self.get_exceedance_probability(iml, M, d, h=h,
												imt=imt, T=T, imt_unit=imt_unit,
												soil_type=soil_type, vs30=vs30,
												kappa=kappa, mechanism=mechanism,
												damping=damping,
												truncation_level=truncation_level)
			joint_exceedance_prob = np.minimum(joint_exceedance_prob, sa_exceedance_prob)

		return joint_exceedance_prob

	def is_depth_dependent(self):
		"""
		Indicate whether or not GMPE depends on depth of the source
		"""
		if self.distance_metric in ("Hypocentral", "Rupture"):
			return True
		elif self.distance_metric in ("Epicentral", "Joyner-Boore"):
			return False

	def sigma_depends_on_magnitude(self, imt="PGA", T=0., soil_type="rock",
							vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		Indicate whether or not standard deviation depends on magnitude
		"""
		sigma1 = self.log_sigma(self.Mmin, d=10, h=0, T=T, imt=imt,
								soil_type=soil_type, vs30=vs30, kappa=kappa,
								mechanism=mechanism, damping=damping)
		sigma2 = self.log_sigma(self.Mmax, d=10, h=0, T=T, imt=imt,
								soil_type=soil_type, vs30=vs30, kappa=kappa,
								mechanism=mechanism, damping=damping)
		if np.allclose(sigma1, sigma2):
			return False
		else:
			return True

	def sigma_depends_on_distance(self, imt="PGA", T=0., soil_type="rock",
							vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		Indicate whether or not standard deviation depends on magnitude
		"""
		sigma1 = self.log_sigma(6.0, d=self.dmin, h=0, T=T, imt=imt,
								soil_type=soil_type, vs30=vs30, kappa=kappa,
								mechanism=mechanism, damping=damping)
		sigma2 = self.log_sigma(6.0, d=self.dmax, h=0, T=T, imt=imt,
								soil_type=soil_type, vs30=vs30, kappa=kappa,
								mechanism=mechanism, damping=damping)
		if np.allclose(sigma1, sigma2):
			return False
		else:
			return True

	def check_imt_unit_scaling(self):
		"""
		Check if scaling factors for different imt units are OK.
		"""
		from scipy.constants import g

		M, d = 6., 25.
		for imt in self.imts:
			if imt in ("PGA", "SA"):
				imt_units = ["g", "mg", "ms2", "cms2", "gal"]
				conv_factors = np.array([1.0, 1E-3, 1./g, 0.01/g, 0.01/g])
			elif imt in ("PGV", "SV"):
				imt_units = ["ms", "cms"]
				conv_factors = np.array([1.0, 1E-2])
			elif imt in ("PGD", "SD"):
				imt_units = ["m", "cm"]
				conv_factors = np.array([1.0, 1E-2])

			values = np.zeros(len(imt_units), 'd')
			if imt in ("PGA", "PGV", "PGD"):
				for i, imt_unit in enumerate(imt_units):
					values[i] = self.__call__(M, d, imt=imt, imt_unit=imt_unit)
			elif imt in ("SA", "SV", "SD"):
				for i, imt_unit in enumerate(imt_units):
					values[i] = self.__call__(M, d, imt=imt, T=1, imt_unit=imt_unit)
			values *= conv_factors
			allclose = np.allclose(values, np.ones(len(values), 'd')*values[0])
			print(imt, allclose)
			if not allclose:
				print(values)

	def get_spectrum(self, M, d, h=0, imt="SA", periods=[], imt_unit="g",
					epsilon=0, soil_type="rock", vs30=None, kappa=None,
					mechanism="normal", damping=5, include_pgm=False, pgm_period=0):
		"""
		Compute response spectrum for given magnitude, distance, depth
		and number of standard deviations.

		:param M:
			float, magnitude.
		:param d:
			float, distance in km.
		:param h:
			float, depth in km. Ignored if distance metric of GMPE is
			epicentral or Joyner-Boore
			(default: 0).
		:param imt:
			str, one of the supported intensity measure types: "PGA" or "SA"
			(default: "SA").
		:param imt_unit:
			str, unit in which intensities should be expressed,
			depends on IMT
			(default: "g")
		:param epsilon:
			float, number of standard deviations above or below the mean
			(default: 0).
		:param soil_type:
			str, one of the soil types supported by the GMPE
			(default: "rock").
		:param vs30:
			float, shear-wave velocity in the upper 30 m (in m/s).
			If not None, it takes precedence over the soil_type parameter
			(default: None).
		:param kappa:
			float, kappa value, in seconds
			(default: None)
		:param mechanism:
			str, fault mechanism: either "normal", "reverse" or "strike-slip".
			(default: "normal").
		:param damping:
			float, damping in percent.
			(default: 5)
		:param include_pgm:
			bool, whether or not to include peak ground motion
			(default: False).
		:param pgm_period:
			float, period (in s) at which to include PGM
			(default: 0)
		"""
		# TODO: return ResponseSpectrum
		if periods in (None, []):
			periods = self.imt_periods[imt]
		intensities = [self.__call__(M, d, h=h, imt=imt, T=T, imt_unit=imt_unit,
									epsilon=epsilon, soil_type=soil_type, vs30=vs30,
									kappa=kappa, mechanism=mechanism, damping=damping)[0]
									for T in periods]
		if include_pgm:
			try:
				pgm = {"SA": "PGA", "PSV": "PGV", "SD": "PGD"}[imt]
			except:
				pass
			else:
				if self.has_imt(pgm):
					[pgm_Avalue] = self.__call__(M, d, h=h, imt=pgm, T=0,
										imt_unit=imt_unit, epsilon=epsilon,
										soil_type=soil_type, vs30=vs30, kappa=kappa,
										mechanism=mechanism, damping=damping)
					intensities.insert(0, pgm_Avalue)
					periods = np.concatenate([[pgm_period], periods])
		intensities = np.array(intensities)
		return (periods, intensities)

	def get_crisis_periods(self):
		"""
		Return array of max. 40 spectral periods to be used with CRISIS.

		This method needs to be overridden if subclass has more than 40 periods.
		"""
		if len(self.sa_periods) <= 40:
			return self.sa_periods

	def write_crisis_atn(self, Mmin=None, Mmax=None, Mstep=0.5,
						rmin=None, rmax=None, nr=50, h=0,
						imt_periods={"PGA": [0]}, imt_unit="g", num_sigma=3,
						soil_type="rock", vs30=None, kappa=None,
						mechanism="normal", damping=5, filespec=None):
		"""
		Generate attenuation table to be used with CRISIS

		:param Mmin:
			float, minimum magnitude to include in the table.
			If None, the lower bound of the valid magnitude range for this GMPE
			will be used. The value will be rounded down to a multiple of Mstep.
			(default: None).
		:param Mmax:
			float, maximum magnitude to include in the table.
			If None, the upper bound of the valid magnitude range for this GMPE
			will be used. The value will be rounded up to a multiple of Mstep.
			(default: None).
		:param Mstep:
			float, magnitude step
			(default: 0.5).
		:param rmin:
			float, minimum distance in km to include in the table.
			If None, the lower bound of the valid distance range for this GMPE
			will be used
			(default: None).
		:param rmax:
			float, maximum distance in km to include in the table.
			If None, the upper bound of the valid distance range for this GMPE
			will be used
			(default: None).
		:param nr:
			Int, number of distance values. The maximum is 50
			(default: 50).
		:param h:
			float, depth in km. Ignored if distance metric of GMPE is epicentral
			or Joyner-Boore
			(default: 0).
		:param imt_periods:
			Dictionary mapping intensity measure types (e.g. "PGA", "SA", "PGV",
			"PGD") to lists or arrays of periods in seconds (float values).
			If empty or None, a maximum of 40 spectral periods will be generated
			for SA
			(default: {"PGA" : [0]}
		:param imt_unit:
			str, unit in which intensities should be expressed,
			depends on IMT
			(default: "g")
		:param num_sigma:
			float, truncation level in number of standard deviations
			above the mean
			(default: 3).
		:param soil_type:
			str, soil type. Note that valid soil types depend on the GMPE
			(default: "rock").
		:param vs30:
			float, shear-wave velocity in the upper 30 m (in m/s).
			If not None, it takes precedence over the soil_type parameter
			(default: None).
		:param kappa:
			float, kappa value in seconds
			(default: None)
		:param mechanism:
			str, fault mechanism: either "normal", "reverse" or "strike-slip".
			Ignored if GMPE is not dependent on rake
			(default: normal).
		:param damping:
			float, damping in percent
			(default: 5).
		:param filespec:
			str, full path specification to output file. Extension ".ATN"
			will be appended if necessary. If filespec is empty or None,
			write to standard output.
			(default: None)
		"""
		if filespec:
			pathname, extension = os.path.splitext(filespec)
			if extension.upper() != ".ATN":
				filespec = filespec + ".ATN"
			of = open(filespec, "w")
		else:
			of = sys.stdout

		if Mmin is None:
			Mmin = self.Mmin
		if Mmax is None:
			Mmax = self.Mmax
		if rmin is None:
			rmin = self.dmin
		if rmax is None:
			rmax = self.dmax

		Mmin = np.floor(Mmin / Mstep) * Mstep
		Mmax = np.ceil(Mmax / Mstep) * Mstep
		Mags = np.arange(Mmin, Mmax+Mstep, Mstep)
		nmag = len(Mags)

		## Avoid math domain errors with 0
		if rmin == 0:
			rmin = 0.1
		distances = logrange(rmin, rmax, nr)

		of.write("%.1f\t%.1f\t%d\n" % (Mmin, Mmax, nmag))
		of.write("%.2f\t%.2f\t%d\t%d\n"
				% (rmin, rmax, nr, CRISIS_DISTANCE_METRICS[self.distance_metric]))

		## Make sure we only have 40 periods
		if imt_periods in ({}, None):
			imt_periods = {"SA" : self.get_crisis_periods()}

		num_periods = 0
		for imt in imt_periods.keys():
			if imt_periods[imt] in ([], None):
				imt_periods[imt] = self.imt_periods[imt]
			num_periods += len(imt_periods[imt])

		if num_periods > 40:
			print("Warning: Too many (%d) periods. CRISIS only supports 40!"
				% num_periods)

		## Periods should be in ascending order
		## Sorting IMT's makes sure PGA comes before SA
		all_periods = []
		for imt in sorted(imt_periods.keys()):
			for T in sorted(imt_periods[imt]):
				## CRISIS does not support non-numeric structural periods,
				## so different IMT's may not have duplicate periods!
				if not T in all_periods:
					all_periods.append(T)
				else:
					raise Exception("Duplicate period found: %s (%s s)"
									% (imt, T))

				## Note: CRISIS does not support distance-dependent uncertainty,
				## so we take 10 km (or 20 km in the case of Toro)
				if "ToroEtAl2002" in self.name:
					d_for_sigma = 20
				else:
					d_for_sigma = 10

				sigma_depends_on_magnitude = self.sigma_depends_on_magnitude(imt=imt,
												T=T, soil_type=soil_type, vs30=vs30,
												kappa=kappa, mechanism=mechanism,
												damping=damping)
				if not sigma_depends_on_magnitude:
					log_sigma = self.log_sigma(M=0, d=d_for_sigma, h=h, imt=imt,
												T=T, soil_type=soil_type, vs30=vs30,
												kappa=kappa, mechanism=mechanism,
												damping=damping)
					ln_sigma = log_sigma * np.log(10)
				else:
					ln_sigma = -1.0
				#if num_imts > 1:
				#	of.write("%s-%.3E\t%.3E\t%.2f\n" % (imt, T, ln_sigma, num_sigma*-1))
				of.write("%.3E\t%.3E\t%.2f\n" % (T, ln_sigma, num_sigma*-1))
				for M in Mags:
					intensities = self.__call__(M, distances, h=h, T=T, imt=imt,
												imt_unit=imt_unit, epsilon=0,
												soil_type=soil_type, vs30=vs30,
												kappa=kappa, mechanism=mechanism,
												damping=damping)
					s = "\t".join(["%.4E" % val for val in intensities])
					of.write("%s\n" % s)
				if sigma_depends_on_magnitude:
					for M in Mags:
						log_sigma = self.log_sigma(M=M, d=d_for_sigma, h=h, T=T,
												imt=imt, soil_type=soil_type,
												vs30=vs30, kappa=kappa,
												mechanism=mechanism, damping=damping)
						ln_sigma = log_sigma * np.log(10)
						of.write("%.3E\n" % ln_sigma)

	def plot_pdf(self, M, d, h=0, imt="PGA", T=0, imt_unit="g",
				sigma_truncations=[2,3], soil_type="rock", vs30=None, kappa=None,
				mechanism="normal", damping=5, plot_style="lin", amax=None,
				fig_filespec=None, title="", lang="en"):
		"""
		Plot probability density function versus ground motion.

		:param M:
			float, magnitude.
		:param d:
			float, distance in km.
		:param h:
			float, depth in km. Ignored if distance metric of GMPE is
			epicentral or Joyner-Boore
			(default: 0).
		:param imt:
			str, one of the supported intensity measure types: "PGA" or "SA"
			(default: "PGA").
		:param T:
			float, spectral period of considered IMT. Ignored if IMT == "PGA"
			(default: 0).
		:param imt_unit:
			str, unit in which intensities should be expressed,
			depends on IMT
			(default: "g")
		:param sigma_truncations:
			List with truncation levels to plot separately
			(default: [2,3]).
		:param soil_type:
			str, one of the soil types supported by the GMPE
			(default: "rock").
		:param vs30:
			float, shear-wave velocity in the upper 30 m (in m/s).
			If not None, it takes precedence over the soil_type parameter
			(default: None).
		:param kappa:
			float, kappa value, in seconds (default: None)
		:param mechanism:
			str, fault mechanism: either "normal", "reverse" or "strike-slip"
			(default: "normal").
		:param damping:
			float, damping in percent.
		:param plot_style:
			str, plot style: either "lin" or "log"
			(default: "lin")
		:param amax:
			float, maximum intensity value
			(default: None).
		:param fig_filespec:
			str, full path to output file for figure. If not specified,
			figure will be shown on the screen
			(default: None).
		:param title:
			str, plot title
			(default: "")
		:param lang:
			str, shorthand for language of annotations.
			Currently only "en" and "nl" are supported
			(default: "en").
		"""
		import pylab
		from matplotlib.ticker import MultipleLocator
		import stats.ValuePdf as ValuePdf

		if not isinstance(sigma_truncations, (list, tuple)):
			sigma_truncations = [sigma_truncations]
		colors = ("r", "g", "b", "c", "m", "y", "k")

		ah_mean = self.__call__(M, d, h=h, imt=imt, T=T, imt_unit=imt_unit,
								soil_type=soil_type, vs30=vs30, kappa=kappa,
								mechanism=mechanism, damping=damping)
		log_sigma = self.log_sigma(M, d, h=h, imt=imt, T=T, soil_type=soil_type,
								vs30=vs30, kappa=kappa, mechanism=mechanism,
								damping=damping)
		log_ah_mean = np.log10(ah_mean)

		if plot_style == "lin":
			## Linear horizontal axis
			pdf_unb = ValuePdf.LogNormalValuePdf(log_ah_mean, log_sigma, base=10,
									musigma_log=True, num_sigma=5, normalize=False)
			pdf_unb_area = np.cumsum(pdf_unb.pdf)
			pdf_list = []
			for epsilon in sigma_truncations:
				#max_val = pdf_unb.geometric_mean() * pdf_unb.geometric_sigma()**sigma
				max_val = 10**(log_ah_mean + log_sigma * epsilon)
				pdf = pdf_unb.copy()
				pdf.truncate(max_val=max_val)
				pdf.pad()
				pdf.pdf *= pdf_unb_area
				pdf_list.append(pdf)

			label = {"en": "Unbounded", "nl": "Onbegrensd"}[lang.lower()]
			pylab.plot(pdf_unb.values, pdf_unb.pdf, colors[0], linewidth=2, label=label)
			i = 1
			for epsilon, pdf_truncated in zip(sigma_truncations, pdf_list):
				label = {"en": "Truncated at", "nl": "Afgeknot op"}[lang.lower()] + " +%d sigma" % epsilon
				pylab.plot(pdf_truncated.values, pdf_truncated.pdf, colors[i], linewidth=2, label=label)
				i += 1
			label = get_imt_label(imt, lang.lower()) + " (%s)" % imt_unit
			pylab.xlabel(label, fontsize="x-large")
			xmin, xmax, ymin, ymax = pylab.axis()
			if amax is None:
				amax = xmax
			pylab.axis((0, amax, ymin, ymax))

		else:
			## Logarithmic horizontal axis
			pdf_unb = ValuePdf.NormalValuePdf(log_ah_mean, log_sigma, num_sigma=5, normalize=True)
			pdf_list = []
			for epsilon in sigma_truncations:
				max_val = log_ah_mean + log_sigma * epsilon
				pdf = pdf_unb.copy()
				pdf.truncate(max_val=max_val)
				pdf.pad()
				pdf_list.append(pdf)
			label= {"en": "Unbounded", "nl": "Onbegrensd"}[lang.lower()]
			pylab.plot(pdf_unb.values, pdf_unb.pdf, colors[0], linewidth=2, label=label)
			i = 1
			for epsilon, pdf_truncated in zip(sigma_truncations, pdf_list):
				label = {"en": "Truncated at", "nl": "Afgeknot op"}[lang.lower()] + " +%d sigma" % epsilon
				pylab.plot(pdf_truncated.values, pdf_truncated.pdf, colors[i], linewidth=2, label=label)
				i += 1
			label = "Log " + get_imt_label(imt, lang.lower()) + " (%s)" % imt_unit
			pylab.xlabel(label, fontsize="x-large")
			xmin, xmax, ymin, ymax = pylab.axis()
			pylab.axis((xmin, 3.0, ymin, ymax))

		## Plot decorations
		pylab.ylabel({"en": "Probability density", "nl": "Kansdichtheid"}[lang.lower()], fontsize="x-large")
		pylab.grid(True)
		if not title:
			title = "%s" % self.name + {"en": " GMPE", "nl": " dempingswet"}[lang.lower()]
		title += "\nM=%.1f, r=%.1f km, h=%d km" % (M, d, int(round(h)))
		title += ", %s" % imt.upper()
		if len(self.imt_periods[imt]) > 1:
			title += " (T=%.1f s)" % T
		pylab.title(title)
		pylab.legend(loc=0)
		ax = pylab.gca()
		minorLocator = MultipleLocator(0.1)
		ax.xaxis.set_minor_locator(minorLocator)
		for label in ax.get_xticklabels() + ax.get_yticklabels():
			label.set_size('large')
		if fig_filespec:
			pylab.savefig(fig_filespec, dpi=300)
			pylab.clf()
		else:
			pylab.show()

	def plot_distance(self, mags, dmin=None, dmax=None, distance_metric=None,
						h=0, imt="PGA", T=0, imt_unit="g", epsilon=0,
						soil_type="rock", vs30=None, kappa=None,
						mechanism="normal", damping=5,
						plot_style="loglog", amin=None, amax=None, color='k',
						fig_filespec=None, title="", want_minor_grid=False,
						legend_location=0, lang="en"):
		"""
		Plot ground motion versus distance for this GMPE.
		Horizontal axis: distances.
		Vertical axis: ground motion values.

		:param mags:
			list of floats, magnitudes to plot
		:param dmin:
			float, lower distance in km. If None, use the lower bound of
			the distance range of each GMPE
			(default: None).
		:param dmax:
			float, upper distance in km. If None, use the lower bound of
			the valid distance range of each GMPE
			(default: None).
		:param distance_metric:
			str, distance_metric to plot (options: "rjb", "rrup")
			(default: None, distance metrics of gmpes are used)
		:param h:
			float, depth in km. Ignored if distance metric of GMPE is
			epicentral or Joyner-Boore
			(default: 0).
		:param imt:
			str, one of the supported intensity measure types.
			(default: "PGA").
		:param T:
			float, period to plot
			(default: 0).
		:param imt_unit:
			str, unit in which intensities should be expressed,
			depends on IMT
			(default: "g")
		:param epsilon:
			float, number of standard deviations above or below the mean
			to plot in addition to the mean
			(default: 0).
		:param soil_type:
			str, one of the soil types supported by the particular GMPE
			(default: "rock").
		:param vs30:
			float, shear-wave velocity in the upper 30 m (in m/s).
			If not None, it takes precedence over the soil_type parameter
			(default: None).
		:param kappa:
			float, kappa value, in seconds (default: None)
		:param mechanism:
			str, fault mechanism: either "normal", "reverse" or "strike-slip"
			(default: "normal").
		:param damping:
			float, damping in percent
			(default: 5).
		:param plot_style:
			str, plotting style ("lin", "loglin", "linlog" or "loglog").
			First term refers to horizontal axis, second term to vertical axis.
			(default: "loglog").
		:param amin:
			float, lower ground-motion value to plot
			(default: None).
		:param amax:
			upper ground-motion value to plot
			(default: None).
		:param color:
			matplotlib color specification
			(default: 'k')
		:param fig_filespec:
			str, full path specification of output file
			(default: None).
		:param title:
			str, plot title
			(default: "")
		:param want_minor_grid:
			bool, whether or not to plot minor gridlines
			(default: False).
		:param legend_location:
			Integer, location of legend (matplotlib location code):
			"best" 	0
			"upper right" 	1
			"upper left" 	2
			"lower left" 	3
			"lower right" 	4
			"right" 	5
			"center left" 	6
			"center right" 	7
			"lower center" 	8
			"upper center" 	9
			"center" 	10
			(default: 0)
		:param lang:
			str, shorthand for language of annotations.
			Currently only "en" and "nl" are supported
			(default: "en").
		"""
		from .plot import plot_distance

		plot_distance([self], mags=mags, dmin=dmin, dmax=dmax,
					distance_metric=distance_metric, h=h, imt=imt, T=T,
					imt_unit=imt_unit, epsilon=epsilon, soil_type=soil_type,
					vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping,
					plot_style=plot_style, amin=amin, amax=amax, colors=[color],
					fig_filespec=fig_filespec, title=title,
					want_minor_grid=want_minor_grid, legend_location=legend_location,
					lang=lang)

	def plot_spectrum(self, mags, d, h=0, imt="SA", Tmin=None, Tmax=None,
					imt_unit="g", epsilon=0, soil_type="rock", vs30=None,
					kappa=None, mechanism="normal", damping=5, plot_freq=False,
					plot_style="loglog", amin=None, amax=None, color='k',
					label=None, fig_filespec=None, title="",
					want_minor_grid=False, include_pgm=True, pgm_freq=50,
					legend_location=0, lang="en"):
		"""
		Plot ground motion spectrum for this GMPE.
		Horizontal axis: spectral periods or frequencies.
		Vertical axis: ground motion values.

		:param mags:
			list of floats, magnitudes to plot
		:param d:
			float, distance in km.
		:param h:
			float, depth in km. Ignored if distance metric of GMPE is
			epicentral or Joyner-Boore
			(default: 0).
		:param imt:
			str, one of the supported intensity measure types.
			(default: "SA").
		:param Tmin:
			float, lower period to plot. If None, lower bound of valid
			period range is used
			(default: None).
		:param Tmax:
			float, upper period to plot. If None, upper bound of valid
			period range is used
			(default: None).
		:param imt_unit:
			str, unit in which intensities should be expressed,
			depends on IMT
			(default: "g")
		:param epsilon:
			float, number of standard deviations above or below the mean
			to plot in addition to the mean
			(default: 0).
		:param soil_type:
			str, one of the soil types supported by the particular GMPE
			(default: "rock").
		:param vs30:
			float, shear-wave velocity in the upper 30 m (in m/s).
			If not None, it takes precedence over the soil_type parameter
			(default: None).
		:param kappa:
			float, kappa value, in seconds
			(default: None)
		:param mechanism:
			str, fault mechanism: either "normal", "reverse" or "strike-slip"
			(default: "normal").
		:param damping:
			float, damping in percent (default: 5).
		:param include_pgm:
			bool, whether or not to include peak ground motion in the plot,
			if possible (plot_freq == False and plot_style in ("lin", "linlog")
			(default: True).
		:param pgm_freq:
			float, frequency (in Hz) at which to plot PGM if horizontal
			axis is logarithmic or is in frequencies
			(default: 50)
		:param plot_freq:
			bool, whether or not to plot frequencies instead of periods
			(default: False).
		:param plot_style:
			str, plotting style ("lin", "loglin", "linlog" or "loglog").
			First term refers to horizontal axis, second term to vertical axis.
			(default: "loglog").
		:param amin:
			float, lower ground-motion value to plot
			(default: None).
		:param amax:
			upper ground-motion value to plot
			(default: None).
		:param color:
			matplotlib color specification
			(default: 'k')
		:param label:
			str, label to plot in legend
			(default: None)
		:param fig_filespec:
			str, full path specification of output file
			(default: None).
		:param title:
			str, plot title
			(default: "")
		:param want_minor_grid:
			bool, whether or not to plot minor gridlines
			(default: False).
		:param legend_location:
			Integer, location of legend (matplotlib location code):
			"best" 	0
			"upper right" 	1
			"upper left" 	2
			"lower left" 	3
			"lower right" 	4
			"right" 	5
			"center left" 	6
			"center right" 	7
			"lower center" 	8
			"upper center" 	9
			"center" 	10
			(default: 0)
		:param lang:
			str, shorthand for language of annotations.
			Currently only "en" and "nl" are supported
			(default: "en").
		"""
		from .plot import plot_spectrum

		if Tmin is None:
			Tmin = self.Tmin(imt)
		if Tmax is None:
			Tmax = self.Tmax(imt)
		plot_spectrum([self], mags=mags, d=d, h=h, imt=imt, Tmin=Tmin, Tmax=Tmax,
					imt_unit=imt_unit, epsilon=epsilon, soil_type=soil_type,
					vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping,
					include_pgm=include_pgm, pgm_freq=pgm_freq, plot_freq=plot_freq,
					plot_style=plot_style, amin=amin, amax=amax, colors=[color],
					labels=[label], fig_filespec=fig_filespec, title=title,
					want_minor_grid=want_minor_grid, legend_location=legend_location,
					lang=lang)


def adjust_hard_rock_to_rock(imt, periods, gm, gm_logsigma=None):
	"""
	Adjust hard rock (vs30=2800 m/s) to rock (vs30=760 m/s, kappa=0.03)
	according to Table 9 in Drouet et al. (2010)
	Applicable to Toro (2002) and Campbell (2003) GMPE's

	:param imt:
		str, intensity measure type, either "PGA" or "SA"
	:param periods:
		List, periods for given imt
	:param gm:
		Float array: ground motions corresponding to given periods
	:param gm_logsigma:
		Float array: log of standard deviations of ground motions
		corresponding to given periods
		This term is currently discarded
		(default: None)

	:return:
		Float array, adjusted ground motions
	"""
	# TODO: not clear how to interpolate between PGA and 1st spectral period
	T_AFrock = np.array([0.01, 0.03, 0.04, 0.1, 0.2, 0.4, 1.0, 2.0])
	AFrock = np.array([0.735106, 0.423049, 0.477379, 0.888509, 1.197291,
						1.308267, 1.265762, 1.215779])
	AFrock_sigma = np.array([0.338916, 0.289785, 0.320650, 0.352442, 0.281552,
							0.198424, 0.154327, 0.155520])

	if imt == "PGA":
		AF = AFrock[0]
		AF_sigma = AFrock_sigma[0]
	elif imt == "SA":
		AF = interpolate(T_AFrock, AFrock, periods)
		AF_sigma = interpolate(T_AFrock, AFrock_sigma, periods)

	#for t, af in zip(periods, AF):
	#	print(t, af)

	adjusted_gm = gm * AF
	# TODO: check sigma computation
	#adjusted_gm_logsigma = np.log10(np.exp(np.log(10**gm_log_sigma) + AF_sigma))

	#return (adjusted_gm, adjusted_gm_logsigma)
	return adjusted_gm


def adjust_host_to_target(imt, periods, gm, M, d, host_vs30, host_kappa,
						target_vs30, target_kappa, region="ena"):
	"""
	Perform host-to-target adjustments for VS30 and kappa using a combination of
	generic rock profiles and (inverse) random vibration theory.

	First, a transfer function is computed between a generic host rock profile
	and a generic target rock profile. The transfer function magnitudes are
	multiplied with the Fourier amplitude spectrum, which is obtained with
	inverse random vibration theory. The adjusted FAS is then converted back to
	a response spectrum using random vibration theory

	:param imt:
		str, intensity measure type, only "SA" is supported. For PGA, include
		it in SA at an appropriate period (between 1./34 and 1.100)
	:param periods:
		List or array, periods for given imt
	:param gm:
		Float array: ground motions corresponding to given periods
	:param M:
		float, magnitude
	:param d:
		float, distance
	:param host_vs30:
		float, vs30 of host rock (in m/s)
	:param host_kappa:
		float, kappa of host rock (in s)
	:param target_vs30:
		float, vs30 of target rock (in m/s)
	:param target_kappa:
		float, kappa of target rock (in s)
	:param region:
		str, either "ena" or "wna"
		(default: "ena")

	:return:
		Float array, adjusted ground motions
	"""
	import pyrvt
	from ..siteresponse import get_host_to_target_tf

	if imt != "SA":
		raise Exception("Host-to-target adjustment not supported for %s!" % imt)

	if target_kappa < host_kappa:
		raise Exception("Manual intervention needed if target kappa < host kappa!")

	if host_vs30 > 2700:
		print("Warning: host vs30 will be clipped to 2700 m/s!")
		host_vs30 = min(host_vs30, 2700)

	if target_vs30 > 2700:
		print("Warning: target vs30 will be clipped to 2700 m/s!")
		target_vs30 = min(target_vs30, 2700)

	freqs = 1./np.asarray(periods)
	irvt = pyrvt.motions.CompatibleRvtMotion(freqs, gm, magnitude=M, distance=d,
											region=region)
	tf = get_host_to_target_tf(irvt.freqs, host_vs30, host_kappa, target_vs30,
								target_kappa)
	irvt.fourier_amps *= tf.magnitudes
	rvt = pyrvt.motions.RvtMotion(irvt.freqs, irvt.fourier_amps, irvt.duration)
	adjusted_gm = rvt.compute_osc_resp(freqs)

	return adjusted_gm


def adjust_faulting_style(imt, periods, gm, mechanism):
	"""
	Adjust style of faulting for GMPE's that do not include a
	style-of-faulting term according to the report by Drouet et al. (2010)
	Applicable to Toro (2002) and Campbell (2003) GMPE's

	:param imt:
		str, intensity measure type, either "PGA" or "SA"
	:param periods:
		List, periods for given imt
	:param gm:
		Float array: ground motions corresponding to given periods
	:param mechanism:
		str, fault mechanism, either "normal", "strike-slip" or "reverse"

	:return:
		Float array, adjusted ground motions
	"""
	## Note: it appears the two reports by Drouet et al. (2010) are in
	## contradiction regarding the formulas for normal and reverse faulting.
	## It is very likely that the formulas on page 13 of deliverable 4.2,
	## which are those used in OpenSHA and oqhazlib have been interchanged,
	## because the adjustment leads to highest accelerations for normal
	## events, and lowest accelerations for reverse events, whereas all
	## the plots in Drouet et al. show the opposite...
	## The formulas on page 10 of the GMPE report (revised version 2)
	## are probably the correct ones.

	# TODO: not clear how to interpolate between PGA and 1st spectral period
	T_FRSS = np.array([0., 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
					0.19, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36,
					0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.55, 0.60, 0.65,
					0.70, 1.60, 1.70, 1.80, 1.90, 2.0])
	FRSS = np.array([1.22, 1.08, 1.10, 1.11, 1.13, 1.13, 1.15, 1.16, 1.16, 1.17,
					1.18, 1.19, 1.20, 1.20, 1.21, 1.22, 1.23, 1.23, 1.23, 1.23,
					1.23, 1.23, 1.24, 1.23, 1.24, 1.23, 1.23, 1.23, 1.22, 1.21,
					1.20, 1.19, 1.18, 1.17, 1.16, 1.14])
	Fnss = 0.95
	pN, pR = 0.01, 0.81

	if imt == "PGA":
		frss = FRSS[0]
	else:
		frss = interpolate(T_FRSS, FRSS, periods)

	#for t, f in zip(periods, frss):
	#	print(t, f)

	if mechanism == "strike-slip":
		adjustment = frss ** -pR * Fnss ** -pN
	elif mechanism == "normal":
		#adjustment = frss ** (1 - pR) * Fnss ** -pN
		adjustment = frss ** -pR * Fnss ** (1 - pN)
	elif mechanism == "reverse":
		adjustment = frss ** (1 - pR) * Fnss ** -pN
		#adjustment = frss ** -pR * Fnss ** (1 - pN)

	adjusted_gm = gm * adjustment

	return adjusted_gm


def adjust_components(component_type, periods, gm):
	"""
	Adjust component type to GM (geometric mean) according to the report
	by Drouet et al. (2010).
	Conversions are valid for periods from 0.02 to 5 s

	:param component_type:
		str, component type, one of the following:
			"GR": rotated geometrical mean
			"AM": arithmetic mean
			"LA": larger envelope
			"RA": random horizontal
			"LP": larger PGA
	:param periods:
		list or array, spectral periods (in s)
	:param gm:
		float array, ground motions corresponding to given periods

	:return:
		float array, adjusted ground motions
	"""
	c1 = {"GR": 1.0, "AM": 1.0, "LE": 1.1, "RA": 1.0, "LP": 1.1}[component_type]
	c2 = {"GR": 1.0, "AM": 1.0, "LE": 1.2, "RA": 1.0, "LP": 1.0}[component_type]

	adjustment = np.ones_like(gm)
	adjustment[np.where(0.02 <= periods <= 0.15)] = c1
	ids = np.where(0.15 < periods <= 0.8)
	adjustment[ids] = c1 + (c2 - c1) * (np.log(periods[ids] / 0.15) / np.log(0.8 / 0.15))
	adjustment[np.where(0.8 < periods <= 5.0)] = c2

	adjusted_gm = gm * adjustment

	return adjusted_gm


def convert_distance_metric(distances, metric_from, metric_to, mag):
	"""
	Adjust distance metric between rjb and rrup according to
	Atkinson and Boore (2011).
	Conversions are valid for magnitudes from 4 to 8,
	focal depths from 8 to 12, and dips from 45 to 90.
	Hypocenter is assumed to be at center of rupture.

	:param distances:
		np array, distances to convert
	:param metric_from:
		str, distance metric of distances
	:param metric_to:
		str, distance metric to convert distances to
	:param mag:
		float, magnitude to use for conversion

	:return:
		np array, converted distances
	"""
	if metric_to is None or metric_from == metric_to:
		return distances
	else:
		ztor = 21. - 2.5 * mag
		if metric_from == "Rupture" and metric_to == "Joyner-Boore":
			return np.sqrt(distances ** 2 - ztor ** 2)
		elif metric_from == "Joyner-Boore" and metric_to == "Rupture":
			return np.sqrt(distances ** 2 + ztor ** 2)
		else:
			# TODO: support other conversion
			raise Exception("Conversion not supported")


if __name__ == "__main__":
	pass

