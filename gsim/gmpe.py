# -*- coding: iso-Latin-1 -*-

## Import standard python modules
import os
import sys
import platform

## Import third-party modules
## Kludge because matplotlib is broken on seissrv3. Sigh...
if platform.uname()[1] == "seissrv3":
	import matplotlib
	matplotlib.use('AGG')
import numpy as np
from scipy.constants import g
import pylab
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

## Import ROB modules
from ..utils import interpolate, logrange

import openquake.hazardlib as nhlib
from openquake.hazardlib.imt import PGD, PGV, PGA, SA


def sd2psa(sd, T):
	"""
	Convert spectral displacement to (pseudo)spectral acceleration

	:param sd:
		Float or numpy array, spectral displacement(s)
	:param T:
		Float, spectral period
	"""
	omega = 2 * np.pi / T
	return sd * omega * omega


def sd2psv(sd, T):
	"""
	Convert spectral displacement to (pseudo)spectral velocity

	:param sd:
		Float or numpy array, spectral displacement(s)
	:param T:
		Float, spectral period
	"""
	omega = 2 * np.pi / T
	return sd * omega


## Map distance metrics to constants used in CRISIS
CRISIS_DistanceMetrics = {
	"Hypocentral": 1,
	"Epicentral": 2,
	"Joyner-Boore": 3,
	"Rupture": 4}


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
	def __init__(self, imt_periods, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings, name, short_name):
		"""
		Base class for a GMPE (Ground-Motion Prediction Equation) or attenuation
		law.

		:param imt_periods:
			Dictionary mapping intensity measure types (e.g. "PGA", "SA", "PGV",
			"PGD") to lists or arrays of periods in seconds (float values).
			Periods must be monotonically increasing or decreasing
		:param distance_metric:
			String, Distance metric used in this GMPE: "Hypocentral",
			"Epicentral", "Joyner-Boore" or "Rupture"
		:param Mmin:
			Float, Lower bound of valid magnitude range
		:param Mmax:
			Float, Upper bound of valid magnitude range
		:param dmin:
			Float, Lower bound of valid distance range in km
		:param dmax:
			Float, Upper bound of valid distance range in km
		:param Mtype:
			Float, Magnitude type used in this GMPE: "MW", "MS", ...
		:param dampings:
			List of damping values supported by this GMPE
		:param name:
			String containing full name of this GMPE
		:param short_name:
			String, abbreviated name of this GMPE

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

	def has_imt(self, imt):
		"""
		Check if given intensity measure type is supported.

		:param imt:
			String, intensity measure type: e.g., "PGA", "PGV", "PGD", "SA"
		"""
		return self.imt_periods.has_key(imt.upper())

	@property
	def imts(self):
		"""
		List supported intensity measure types.
		"""
		return self.imt_periods.keys()

	@property
	def sa_periods(self):
		"""
		List spectral periods for IMT SA
		"""
		return self.imt_periods["SA"]

	@property
	def sa_freqs(self):
		"""
		List spectral frequencies for IMT SA
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
			String, intensity measure type (default: "SA")
		"""
		return self.imt_periods[imt].min()

	def Tmax(self, imt="SA"):
		"""
		Return maximum period in seconds for given IMT

		:param imt:
			String, intensity measure type (default: "SA")
		"""
		return self.imt_periods[imt].max()

	def fmin(self, imt="SA"):
		"""
		Return minimum frequency in Hz for given IMT. If Tmax is 0, return
		value is 34 Hz

		:param imt:
			String, intensity measure type (default: "SA")
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
			String, intensity measure type (default: "SA")
		"""
		try:
			return 1.0 / self.Tmin(imt=imt)
		except:
			return 34

	def get_epsilon(self, iml, M, d, h=0., imt="PGA", T=0., imt_unit="g", soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		Determine epsilon for given intensity measure level(s), magnitude, and distance

		:param iml:
			Float or numpy array, intensity measure level(s)
		:param M:
			Float, magnitude.
		:param d:
			Float, distance in km.
		:param h:
			Float, depth in km. Ignored if distance metric of GMPE is epicentral
			or Joyner-Boore (default: 0).
		:param imt:
			String, one of the supported intensity measure types: "PGA" or "SA"
			(default: "SA").
		:param T:
			Float, spectral period of considered IMT. Ignored if IMT is
			"PGA" or "PGV" (default: 0).
		:param imt_unit:
			String, unit in which intensities should be expressed, depends on
			IMT (default: "g")
		:param soil_type:
			String, one of the soil types supported by the GMPE (default: "rock").
		:param vs30:
			Float, shear-wave velocity in the upper 30 m (in m/s). If not None,
			it takes precedence over the soil_type parameter (default: None).
		:param kappa:
			Float, kappa value, in seconds (default: None)
		:param mechanism:
			String, fault mechanism: either "normal", "reverse" or "strike-slip".
			(default: "normal").
		:param damping:
			Float, damping in percent.

		:return:
			float array, epsilon value(s)
		"""
		im_med = self.__call__(M, d, h=h, imt=imt, T=T, imt_unit=imt_unit, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
		log_sigma = self.log_sigma(M, d, h=h, imt=imt, T=T, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
		epsilon = (np.log10(iml) - np.log10(im_med)) / log_sigma
		return epsilon

	def get_exceedance_probability(self, iml, M, d, h=0., imt="PGA", T=0., imt_unit="g", soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5, truncation_level=3):
		"""
		Compute probability of exceedance for given intensity measure level(s),
		magnitude, and distance

		:param iml:
			Float or numpy array, intensity measure level(s)
		:param M:
			Float, magnitude.
		:param d:
			Float, distance in km.
		:param h:
			Float, depth in km. Ignored if distance metric of GMPE is epicentral
			or Joyner-Boore (default: 0).
		:param imt:
			String, one of the supported intensity measure types: "PGA" or "SA"
			(default: "PGA").
		:param T:
			Float, spectral period of considered IMT. Ignored if IMT is
			"PGA" or "PGV" (default: 0).
		:param imt_unit:
			String, unit in which intensities are expressed, depends on
			IMT (default: "g")
		:param soil_type:
			String, one of the soil types supported by the GMPE (default: "rock").
		:param vs30:
			Float, shear-wave velocity in the upper 30 m (in m/s). If not None,
			it takes precedence over the soil_type parameter (default: None).
		:param kappa:
			Float, kappa value, in seconds (default: None)
		:param mechanism:
			String, fault mechanism: either "normal", "reverse" or "strike-slip".
			(default: "normal").
		:param damping:
			Float, damping in percent (default: 5.).
		:param truncation_level:
			Float, number of standard deviations at which to truncate GMPE
			(default: 3)

		:return:
			float array, probability(ies) of exceedance for each iml
		"""
		import scipy.stats
		log_iml = np.log10(iml)
		median = self.__call__(M, d, h=h, imt=imt, T=T, imt_unit=imt_unit, epsilon=0, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
		log_median = np.log10(median)
		if truncation_level == 0:
			return (log_iml > log_median) * 1.0
		else:
			log_sigma = self.log_sigma(M, d, h=h, imt=imt, T=T, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
			if truncation_level is None:
				dist = scipy.stats.norm(log_median, log_sigma)
			else:
				dist = scipy.stats.truncnorm(-truncation_level, truncation_level, log_median, log_sigma)
			return 1 - dist.cdf(log_iml)

	def get_exceedance_probability_cav(self, iml, M, d, h=0., imt="PGA", T=0., imt_unit="g", soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5, truncation_level=3, cav_min=0.16, depsilon=0.02, eps_correlation_model="EUS"):
		"""
		Compute joint probability of exceeding given intensity level(s)
		and exceeding a minimum CAV value

		:param iml:
			Float or numpy array, intensity measure level(s)
		:param M:
			Float, magnitude.
		:param d:
			Float, distance in km.
		:param h:
			Float, depth in km. Ignored if distance metric of GMPE is epicentral
			or Joyner-Boore (default: 0).
		:param imt:
			String, one of the supported intensity measure types: "PGA" or "SA"
			(default: "PGA").
		:param T:
			Float, spectral period of considered IMT. Ignored if IMT is
			"PGA" or "PGV" (default: 0).
		:param imt_unit:
			String, unit in which intensities are expressed, depends on
			IMT (default: "g")
		:param soil_type:
			String, one of the soil types supported by the GMPE (default: "rock").
		:param vs30:
			Float, shear-wave velocity in the upper 30 m (in m/s). If not None,
			it takes precedence over the soil_type parameter (default: None).
		:param kappa:
			Float, kappa value, in seconds (default: None)
		:param mechanism:
			String, fault mechanism: either "normal", "reverse" or "strike-slip".
			(default: "normal").
		:param damping:
			Float, damping in percent (default: 5.).
		:param truncation_level:
			Float, number of standard deviations at which to truncate GMPE
			(default: 3).
		:param cav_min:
			Float, CAV threshold (default: 0.16 g.s).
		:param depsilon:
			Float, bin width used to discretize epsilon pga. Should be
			sufficiently small for PGA (default: 0.02).
		:param eps_correlation_model:
			Str, name of model used for correlation of epsilon values
			of PGA and SA, either "WUS" or "EUS" (default: "EUS").

		:return:
			float array, joint probability(ies) for each iml
		"""
		import scipy.stats
		from scitools.numpytools import seq
		from ..utils import interpolate
		from ..cav import calc_CAV_exceedance_prob

		## TODO: if soil_type is used, convert to vs30
		if soil_type == "rock":
			vs30 = 800

		iml = np.asarray(iml)
		dist = scipy.stats.truncnorm(-truncation_level, truncation_level)
		eps_pga_list = seq(-truncation_level, truncation_level, depsilon)
		prob_eps_list = dist.pdf(eps_pga_list) * depsilon

		## Pre-calculate CAV exceedance probabilities for epsilon PGA
		pga_eps_pga_list = np.zeros_like(eps_pga_list)
		cav_exceedance_prob = np.zeros_like(eps_pga_list)
		for e, eps_pga in enumerate(eps_pga_list):
			## Determine PGA corresponding to eps_pga
			pga = self.__call__(M, d, h=h, imt="PGA", T=0, epsilon=eps_pga, imt_unit=imt_unit, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
			pga_eps_pga_list[e] = pga
			## CAV exceedance probability for PGA
			cav_exceedance_prob[e] = calc_CAV_exceedance_prob(pga, M, vs30, CAVmin=0.16, duration_dependent=True)

		if imt == "PGA":
			## The following is not correct (no explicit integration over epsilon)
			#cav_exceedance_prob = calc_CAV_exceedance_prob(iml, M, vs30, CAVmin=cav_min, duration_dependent=True)
			#pga_exceedance_prob = self.get_exceedance_probability(iml, M, d, h=h, imt="PGA", T=0., imt_unit=imt_unit, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping, truncation_level=truncation_level)
			#exceedance_prob = cav_exceedance_prob * pga_exceedance_prob

			## Integrate explicitly over epsilon
			exceedance_prob = np.zeros_like(iml)
			for e in range(len((eps_pga_list))):
				exceedance_prob[pga_eps_pga_list[e] > iml] += (prob_eps_list[e] * cav_exceedance_prob[e])

		else:
			sa_exceedance_prob = self.get_exceedance_probability(iml, M, d, h=h, imt=imt, T=T, imt_unit=imt_unit, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping, truncation_level=truncation_level)

			## Correlation coefficients between PGA and SA (Table 3-1)
			b1_freqs = np.array([0.5, 1, 2.5, 5, 10, 20, 25, 35])
			b1_WUS = np.array([0.59, 0.59, 0.6, 0.633, 0.787, 0.931, 0.956, 0.976])
			b1_EUS = np.array([0.5, 0.55, 0.6, 0.75, 0.88, 0.9, 0.91, 0.93])
			if eps_correlation_model == "WUS":
				b1 = interpolate(1./b1_freqs, b1_WUS, [T])[0]
			elif eps_correlation_model == "EUS":
				b1 = interpolate(1./b1_freqs, b1_EUS, [T])[0]

			## Determine median SA and sigma
			sa_med = self.__call__(M, d, h=h, imt="SA", T=T, imt_unit=imt_unit, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
			sigma_log_sa = self.log_sigma(M, d, h=h, imt="SA", T=T, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
			sigma_ln_sa = np.log(10) * sigma_log_sa

			## Loop over eps_pga
			exceedance_prob = 0
			for e, eps_pga in enumerate(eps_pga_list):
				## Determine epsilon value of SA, and sigma
				## Eq. 3-1
				eps_sa = b1 * eps_pga
				## Eq. 3-2
				ln_sa_given_pga = np.log(sa_med) + eps_sa * sigma_ln_sa
				## Eq. 3-3
				sigma_ln_sa_given_pga = np.sqrt(1 - b1**2) * sigma_ln_sa

				## Determine probability of exceedance of SA given PGA
				## Eq. 4-3
				eps_sa_dot = (np.log(iml) - ln_sa_given_pga) / sigma_ln_sa_given_pga
				## Eq. 4-2
				prob_sa_given_pga = 1.0 - scipy.stats.norm.cdf(eps_sa_dot)

				exceedance_prob += (prob_eps_list[e] * cav_exceedance_prob[e] * prob_sa_given_pga)

			## iml values close to truncation boundaries should have lower exceecance rates
			## This also constrains iml between (-truncation_level, truncation_level)
			exceedance_prob = np.minimum(exceedance_prob, sa_exceedance_prob)

		return exceedance_prob


	def is_depth_dependent(self):
		"""
		Indicate whether or not GMPE depends on depth of the source
		"""
		if self.distance_metric in ("Hypocentral", "Rupture"):
			return True
		elif self.distance_metric in ("Epicentral", "Joyner-Boore"):
			return False

	def sigma_depends_on_magnitude(self, imt="PGA", T=0., soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		Indicate whether or not standard deviation depends on magnitude
		"""
		sigma1 = self.log_sigma(self.Mmin, d=10, h=0, T=T, imt=imt, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
		sigma2 = self.log_sigma(self.Mmax, d=10, h=0, T=T, imt=imt, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
		if np.allclose(sigma1, sigma2):
			return False
		else:
			return True

	def sigma_depends_on_distance(self, imt="PGA", T=0., soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		Indicate whether or not standard deviation depends on magnitude
		"""
		sigma1 = self.log_sigma(6.0, d=self.dmin, h=0, T=T, imt=imt, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
		sigma2 = self.log_sigma(6.0, d=self.dmax, h=0, T=T, imt=imt, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
		if np.allclose(sigma1, sigma2):
			return False
		else:
			return True

	def check_imt_unit_scaling(self):
		"""
		Check if scaling factors for different imt units are OK.
		"""
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
			print imt, allclose
			if not allclose:
				print values

	def get_spectrum(self, M, d, h=0, imt="SA", periods=[], imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		Return (periods, intensities) tuple for given magnitude, distance, depth
		and number of standard deviations.

		:param M:
			Float, magnitude.
		:param d:
			Float, distance in km.
		:param h:
			Float, depth in km. Ignored if distance metric of GMPE is epicentral
			or Joyner-Boore (default: 0).
		:param imt:
			String, one of the supported intensity measure types: "PGA" or "SA"
			(default: "SA").
		:param imt_unit:
			String, unit in which intensities should be expressed, depends on
			IMT (default: "g")
		:param epsilon:
			Float, number of standard deviations above or below the mean
			(default: 0).
		:param soil_type:
			String, one of the soil types supported by the GMPE (default: "rock").
		:param vs30:
			Float, shear-wave velocity in the upper 30 m (in m/s). If not None,
			it takes precedence over the soil_type parameter (default: None).
		:param kappa:
			Float, kappa value, in seconds (default: None)
		:param mechanism:
			String, fault mechanism: either "normal", "reverse" or "strike-slip".
			(default: "normal").
		:param damping:
			Float, damping in percent.
		"""
		if periods in (None, []):
			periods = self.imt_periods[imt]
		intensities = [self.__call__(M, d, h=h, imt=imt, T=T, imt_unit=imt_unit, epsilon=epsilon, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)[0] for T in periods]
		intensities = np.array(intensities)
		return (periods, intensities)


	def get_CRISIS_periods(self):
		"""
		Return array of max. 40 spectral periods to be used with CRISIS.

		This method needs to be overridden if subclass has more than 40 periods.
		"""
		if len(self.sa_periods) <= 40:
			return self.sa_periods

	def writeCRISIS_ATN(self, Mmin=None, Mmax=None, Mstep=0.5, rmin=None, rmax=None, nr=50, h=0, imt_periods={"PGA": [0]}, imt_unit="g", num_sigma=3, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5, filespec=None):
		"""
		Generate attenuation table to be used with CRISIS

		:param Mmin:
			Float, minimum magnitude to include in the table.
			If None, the lower bound of the valid magnitude range for this GMPE
			will be used. The value will be rounded down to a multiple of Mstep.
			(default: None).
		:param Mmax:
			Float, maximum magnitude to include in the table.
			If None, the upper bound of the valid magnitude range for this GMPE
			will be used. The value will be rounded up to a multiple of Mstep.
			(default: None).
		:param Mstep:
			Float, magnitude step (default: 0.5).
		:param rmin:
			Float, minimum distance in km to include in the table.
			If None, the lower bound of the valid distance range for this GMPE
			will be used (default: None).
		:param rmax:
			Float, maximum distance in km to include in the table.
			If None, the upper bound of the valid distance range for this GMPE
			will be used (default: None).
		:param nr:
			Int, number of distance values. The maximum is 50 (default: 50).
		:param h:
			Float, depth in km. Ignored if distance metric of GMPE is epicentral
			or Joyner-Boore (default: 0).
		:param imt_periods:
			Dictionary mapping intensity measure types (e.g. "PGA", "SA", "PGV",
			"PGD") to lists or arrays of periods in seconds (float values).
			If empty or None, a maximum of 40 spectral periods will be generated
			for SA (default: {"PGA" : [0]}
		:param imt_unit:
			String, unit in which intensities should be expressed, depends on
			IMT (default: "g")
		:param num_sigma:
			Float, truncation level in number of standard deviations above the
			mean (default: 3).
		:param soil_type:
			String, soil type. Note that valid soil types depend on the GMPE
			(default: "rock").
		:param vs30:
			Float, shear-wave velocity in the upper 30 m (in m/s). If not None,
			it takes precedence over the soil_type parameter (default: None).
		:param kappa:
			Float, kappa value in seconds (default: None)
		:param mechanism:
			String, fault mechanism: either "normal", "reverse" or "strike-slip".
			Ignored if GMPE is not dependent on rake (default: normal).
		:param damping:
			Float, damping in percent (default: 5).
		:param filespec:
			String, full path specification to output file. Extension ".ATN"
			will be appended if necessary. If filespec is empty or None,
			write to standard output.
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
		of.write("%.2f\t%.2f\t%d\t%d\n" % (rmin, rmax, nr, CRISIS_DistanceMetrics[self.distance_metric]))

		## Make sure we only have 40 periods
		if imt_periods in ({}, None):
			imt_periods = {"SA" : self.get_CRISIS_periods()}

		num_periods = 0
		for imt in imt_periods.keys():
			if imt_periods[imt] in ([], None):
				imt_periods[imt] = self.imt_periods[imt]
			num_periods += len(imt_periods[imt])

		if num_periods > 40:
			print("Warning: Too many (%d) periods. CRISIS only supports 40!" % num_periods)

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
					raise Exception("Duplicate period found: %s (%s s)" % (imt, T))

				## Note: CRISIS does not support distance-dependent uncertainty,
				## so we take 10 km (or 20 km in the case of Toro)
				if "ToroEtAl2002" in self.name:
					d_for_sigma = 20
				else:
					d_for_sigma = 10

				sigma_depends_on_magnitude = self.sigma_depends_on_magnitude(imt=imt, T=T, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
				if not sigma_depends_on_magnitude:
					log_sigma = self.log_sigma(M=0, d=d_for_sigma, h=h, imt=imt, T=T, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
					ln_sigma = log_sigma * np.log(10)
				else:
					ln_sigma = -1.0
				#if num_imts > 1:
				#	of.write("%s-%.3E\t%.3E\t%.2f\n" % (imt, T, ln_sigma, num_sigma*-1))
				of.write("%.3E\t%.3E\t%.2f\n" % (T, ln_sigma, num_sigma*-1))
				for M in Mags:
					intensities = self.__call__(M, distances, h=h, T=T, imt=imt, imt_unit=imt_unit, epsilon=0, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
					s = "\t".join(["%.4E" % val for val in intensities])
					of.write("%s\n" % s)
				if sigma_depends_on_magnitude:
					for M in Mags:
						log_sigma = self.log_sigma(M=M, d=d_for_sigma, h=h, T=T, imt=imt, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
						ln_sigma = log_sigma * np.log(10)
						of.write("%.3E\n" % ln_sigma)

	def plot_pdf(self, M, d, h=0, imt="PGA", T=0, imt_unit="g", sigma_truncations=[2,3], soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5, plot_style="lin", amax=None, fig_filespec=None, title="", lang="en"):
		"""
		Plot probability density function versus ground motion.

		:param M:
			Float, magnitude.
		:param d:
			Float, distance in km.
		:param h:
			Float, depth in km. Ignored if distance metric of GMPE is epicentral
			or Joyner-Boore (default: 0).
		:param imt:
			String, one of the supported intensity measure types: "PGA" or "SA"
			(default: "PGA").
		:param T:
			Float, spectral period of considered IMT. Ignored if IMT == "PGA"
			(default: 0).
		:param imt_unit:
			String, unit in which intensities should be expressed, depends on
			IMT (default: "g")
		:param sigma_truncations:
			List with truncation levels to plot separately
			(default: [2,3]).
		:param soil_type:
			String, one of the soil types supported by the GMPE (default: "rock").
		:param vs30:
			Float, shear-wave velocity in the upper 30 m (in m/s). If not None,
			it takes precedence over the soil_type parameter (default: None).
		:param kappa:
			Float, kappa value, in seconds (default: None)
		:param mechanism:
			String, fault mechanism: either "normal", "reverse" or "strike-slip"
			(default: "normal").
		:param damping:
			Float, damping in percent.
		:param plot_style:
			String, plot style: either "lin" or "log"
		:param amax:
			Float, maximum intensity value (default: None).
		:param fig_filespec:
			String, full path to output file for figure. If not specified,
			figure will be shown on the screen (default: None).
		:param title:
			String, plot title (default: "")
		:param lang:
			String, shorthand for language of annotations. Currently only
			"en" and "nl" are supported (default: "en").
		"""
		import stats.ValuePdf as ValuePdf

		if not isinstance(sigma_truncations, (list, tuple)):
			sigma_truncations = [sigma_truncations]
		colors = ("r", "g", "b", "c", "m", "y", "k")

		ah_mean = self.__call__(M, d, h=h, imt=imt, T=T, imt_unit=imt_unit, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
		log_sigma = self.log_sigma(M, d, h=h, imt=imt, T=T, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
		log_ah_mean = np.log10(ah_mean)

		if plot_style == "lin":
			## Linear horizontal axis
			pdf_unb = ValuePdf.LogNormalValuePdf(log_ah_mean, log_sigma, base=10, musigma_log=True, num_sigma=5, normalize=False)
			pdf_unb_area = np.add.reduce(pdf_unb.pdf)
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

	def plot_distance(self, mags, dmin=None, dmax=None, distance_metric=None, h=0, imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5, plot_style="loglog", amin=None, amax=None, color='k', fig_filespec=None, title="", want_minor_grid=False, legend_location=0, lang="en"):
		"""
		Plot ground motion versus distance for this GMPE.
		Horizontal axis: distances.
		Vertical axis: ground motion values.

		:param mags:
			list of floats, magnitudes to plot
		:param dmin:
			Float, lower distance in km. If None, use the lower bound of the
			distance range of each GMPE (default: None).
		:param dmax:
			Float, upper distance in km. If None, use the lower bound of the
			valid distance range of each GMPE (default: None).
		:param distance_metric:
			str, distance_metric to plot (options: "rjb", "rrup")
			(default: None, distance metrics of gmpes are used)
		:param h:
			Float, depth in km. Ignored if distance metric of GMPE is epicentral
			or Joyner-Boore (default: 0).
		:param imt:
			String, one of the supported intensity measure types.
			(default: "PGA").
		:param T:
			Float, period to plot (default: 0).
		:param imt_unit:
			String, unit in which intensities should be expressed, depends on
			IMT (default: "g")
		:param epsilon:
			Float, number of standard deviations above or below the mean to
			plot in addition to the mean (default: 0).
		:param soil_type:
			String, one of the soil types supported by the particular GMPE
			(default: "rock").
		:param vs30:
			Float, shear-wave velocity in the upper 30 m (in m/s). If not None,
			it takes precedence over the soil_type parameter (default: None).
		:param kappa:
			Float, kappa value, in seconds (default: None)
		:param mechanism:
			String, fault mechanism: either "normal", "reverse" or "strike-slip"
			(default: "normal").
		:param damping:
			Float, damping in percent (default: 5).
		:param plot_style:
			String, plotting style ("lin", "loglin", "linlog" or "loglog").
			First term refers to horizontal axis, second term to vertical axis.
			(default: "loglog").
		:param amin:
			Float, lower ground-motion value to plot (default: None).
		:param amax:
			upper ground-motion value to plot (default: None).
		:param color:
			matplotlib color specification (default: 'k')
		:param fig_filespec:
			String, full path specification of output file (default: None).
		:param title:
			String, plot title (default: "")
		:param want_minor_grid:
			Boolean, whether or not to plot minor gridlines (default: False).
		:param legend_location:
			Integer, location of legend (matplotlib location code) (default=0):
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
		:param lang:
			String, shorthand for language of annotations. Currently only
			"en" and "nl" are supported (default: "en").
		"""
		plot_distance([self], mags=mags, dmin=dmin, dmax=dmax, distance_metric=distance_metric, h=h, imt=imt, T=T, imt_unit=imt_unit, epsilon=epsilon, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping, plot_style=plot_style, amin=amin, amax=amax, colors=[color], fig_filespec=fig_filespec, title=title, want_minor_grid=want_minor_grid, legend_location=legend_location, lang=lang)

	def plot_spectrum(self, mags, d, h=0, imt="SA", Tmin=None, Tmax=None, imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5, plot_freq=False, plot_style="loglog", amin=None, amax=None, color='k', label=None, fig_filespec=None, title="", want_minor_grid=False, include_pgm=True, legend_location=None, lang="en"):
		"""
		Plot ground motion spectrum for this GMPE.
		Horizontal axis: spectral periods or frequencies.
		Vertical axis: ground motion values.

		:param mags:
			list of floats, magnitudes to plot
		:param d:
			Float, distance in km.
		:param h:
			Float, depth in km. Ignored if distance metric of GMPE is epicentral
			or Joyner-Boore (default: 0).
		:param imt:
			String, one of the supported intensity measure types.
			(default: "SA").
		:param Tmin:
			Float, lower period to plot. If None, lower bound of valid period
			range is used (default: None).
		:param Tmax:
			Float, upper period to plot. If None, upper bound of valid period
			range is used (default: None).
		:param imt_unit:
			String, unit in which intensities should be expressed, depends on
			IMT (default: "g")
		:param epsilon:
			Float, number of standard deviations above or below the mean to
			plot in addition to the mean (default: 0).
		:param soil_type:
			String, one of the soil types supported by the particular GMPE
			(default: "rock").
		:param vs30:
			Float, shear-wave velocity in the upper 30 m (in m/s). If not None,
			it takes precedence over the soil_type parameter (default: None).
		:param kappa:
			Float, kappa value, in seconds (default: None)
		:param mechanism:
			String, fault mechanism: either "normal", "reverse" or "strike-slip"
			(default: "normal").
		:param damping:
			Float, damping in percent (default: 5).
		:param include_pgm:
			Boolean, whether or not to include peak ground motion in the plot,
			if possible (plot_freq == False and plot_style in ("lin", "linlog")
			(default: True).
		:param plot_freq:
			Boolean, whether or not to plot frequencies instead of periods
			(default: False).
		:param plot_style:
			String, plotting style ("lin", "loglin", "linlog" or "loglog").
			First term refers to horizontal axis, second term to vertical axis.
			(default: "loglog").
		:param amin:
			Float, lower ground-motion value to plot (default: None).
		:param amax:
			upper ground-motion value to plot (default: None).
		:param color:
			matplotlib color specification (default: 'k')
		:param label:
			String, label to plot in legend (default: None)
		:param fig_filespec:
			String, full path specification of output file (default: None).
		:param title:
			String, plot title (default: "")
		:param want_minor_grid:
			Boolean, whether or not to plot minor gridlines (default: False).
		:param legend_location:
			Integer, location of legend (matplotlib location code) (default=None):
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
		:param lang:
			String, shorthand for language of annotations. Currently only
			"en" and "nl" are supported (default: "en").
		"""
		if Tmin is None:
			Tmin = self.Tmin(imt)
		if Tmax is None:
			Tmax = self.Tmax(imt)
		plot_spectrum([self], mags=mags, d=d, h=h, imt=imt, Tmin=Tmin, Tmax=Tmax, imt_unit=imt_unit, epsilon=epsilon, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping, include_pgm=include_pgm, plot_freq=plot_freq, plot_style=plot_style, amin=amin, amax=amax, colors=[color], labels=[label], fig_filespec=fig_filespec, title=title, want_minor_grid=want_minor_grid, legend_location=legend_location, lang=lang)


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
		GMPE.__init__(self, imt_periods, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings, name, short_name)

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
		self.imt_scaling = {"g": 1.0, "mg": 1E+3, "ms2": g, "gal": g*100, "cms2": g*100}

	def __call__(self, M, d, h=0, imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
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

	def log_sigma(self, M=5., d=10., h=0., imt="PGA", T=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		"""
		imt = imt.upper()
		if not self.has_imt(imt):
			raise IMTUndefinedError(imt)
		if imt == "PGA":
			return self.sigma[imt][0]
		else:
			pass

	def is_rake_dependent(self):
		"""
		Indicate whether or not GMPE depends on rake of the source
		"""
		return False


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
		imt_periods["SA"] = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00]
		distance_metric = "Joyner-Boore"
		Mmin, Mmax = 4, 7.9
		dmin, dmax = 0., 260.
		Mtype = "MS"
		dampings = [5]
		name = "AmbraseysEtAl1996"
		short_name = "Am_1996"
		GMPE.__init__(self, imt_periods, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings, name, short_name)

		## Coefficients
		self.c1 = {}
		self.c1["PGA"] = np.array([-1.48])
		self.c1["SA"] = np.array([-0.84, -0.86, -0.87, -0.87, -0.94, -0.98, -1.05, -1.08, -1.13, -1.19, -1.21, -1.28, -1.37, -1.40, -1.46, -1.55, -1.63, -1.65, -1.69, -1.82, -1.94, -1.99, -2.05, -2.11, -2.17, -2.25, -2.38, -2.49, -2.58, -2.67, -2.75, -2.86, -2.93, -3.03, -3.10, -3.17, -3.30, -3.38, -3.43, -3.52, -3.61, -3.68, -3.74, -3.79, -3.80, -3.79])
		self.c2 = {}
		self.c2["PGA"] = np.array([0.266])
		self.c2["SA"] = np.array([0.219, 0.221, 0.231, 0.238, 0.244, 0.247, 0.252, 0.258, 0.268, 0.278, 0.284, 0.295, 0.308, 0.318, 0.326, 0.338, 0.349, 0.351, 0.354, 0.364, 0.377, 0.384, 0.393, 0.401, 0.410, 0.420, 0.434, 0.438, 0.451, 0.463, 0.477, 0.485, 0.492, 0.502, 0.503, 0.508, 0.513, 0.513, 0.514, 0.522, 0.524, 0.520, 0.517, 0.514, 0.508, 0.503])
		self.h0 = {}
		self.h0["PGA"] = np.array([3.5])
		self.h0["SA"] = np.array([4.5, 4.5, 4.7, 5.3, 4.9, 4.7, 4.4, 4.3, 4.0, 3.9, 4.2, 4.1, 3.9, 4.3, 4.4, 4.2, 4.2, 4.4, 4.5, 3.9, 3.6, 3.7, 3.9, 3.7, 3.5, 3.3, 3.1, 2.5, 2.8, 3.1, 3.5, 3.7, 3.9, 4.0, 4.0, 4.3, 4.0, 3.6, 3.6, 3.4, 3.0, 2.5, 2.5, 2.4, 2.8, 3.2])
		self.c4 = {}
		self.c4["PGA"] = np.array([-0.922])
		self.c4["SA"] = np.array([-0.954, -0.945, -0.960, -0.981, -0.955, -0.938, -0.907, -0.896, -0.901, -0.907, -0.922, -0.911, -0.916, -0.942, -0.946, -0.933, -0.932, -0.939, -0.936, -0.900, -0.888, -0.897, -0.908, -0.911, -0.920, -0.913, -0.911, -0.881, -0.901, -0.914, -0.942, -0.925, -0.920, -0.920, -0.892, -0.885, -0.857, -0.851, -0.848, -0.839, -0.817, -0.781, -0.759, -0.730, -0.724, -0.728])
		self.ca = {}
		self.ca["PGA"] = np.array([0.117])
		self.ca["SA"] = np.array([0.078, 0.098, 0.111, 0.131, 0.136, 0.143, 0.152, 0.140, 0.129, 0.133, 0.135, 0.120, 0.124, 0.134, 0.134, 0.133, 0.125, 0.118, 0.124, 0.132, 0.139, 0.147, 0.153, 0.149, 0.150, 0.147, 0.134, 0.124, 0.122, 0.116, 0.113, 0.127, 0.124, 0.124, 0.121, 0.128, 0.123, 0.128, 0.115, 0.109, 0.109, 0.108, 0.105, 0.104, 0.103, 0.101])
		self.cs = {}
		self.cs["PGA"] = np.array([0.124])
		self.cs["SA"] = np.array([0.027, 0.036, 0.052, 0.068, 0.077, 0.085, 0.101, 0.102, 0.107, 0.130, 0.142, 0.143, 0.155, 0.163, 0.158, 0.148, 0.161, 0.163, 0.160, 0.164, 0.172, 0.180, 0.187, 0.191, 0.197, 0.201, 0.203, 0.212, 0.215, 0.214, 0.212, 0.218, 0.218, 0.225, 0.217, 0.219, 0.206, 0.214, 0.200, 0.197, 0.204, 0.206, 0.206, 0.204, 0.194, 0.182])
		self.sigma = {}
		self.sigma["PGA"] = np.array([0.25])
		self.sigma["SA"] = np.array([0.27, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27, 0.27, 0.28, 0.27, 0.28, 0.28, 0.28, 0.29, 0.30, 0.31, 0.31, 0.31, 0.31, 0.31, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.33, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.32, 0.32, 0.32])

		## Unit conversion
		self.imt_scaling = {"g": 1.0, "mg": 1E+3, "ms2": g, "gal": g*100, "cms2": g*100}

	def __call__(self, M, d, h=0, imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		Return ground motion for given magnitude, distance, depth, soil type,
		and fault mechanism.

		:param M:
			Float or float array, magnitude(s).
			Note that if d is an array, M must be a float.
		:param d:
			Float or float array, distance(s) in km.
			Note that if M is an array, d must be a float.
		:param h:
			Float, depth in km. Ignored in this GMPE.
		:param imt:
			String, one of the supported intensity measure types: "PGA" or "SA"
			(default: "PGA").
		:param T:
			Float, spectral period of considered IMT. Ignored if IMT == "PGA"
			(default: 0).
		:param imt_unit:
			String, unit in which intensities should be expressed (default: "g")
		:param epsilon:
			Float, number of standard deviations above or below the mean
			(default: 0).
		:param soil_type:
			String, either "rock", "stiff" or "soft" (default: "rock").
				Rock: VS >= 750 m/s
				Stiff soil: 360 <= VS < 750 m/s
				Soft soil: 180 <= VS < 360 m/s
		:param vs30:
			Float, shear-wave velocity in the upper 30 m (in m/s). If not None,
			it takes precedence over the soil_type parameter (default: None).
		:param mechanism:
			String, fault mechanism: either "normal", "reverse" or "strike-slip".
			Ignored in this GMPE.
		:param damping:
			Float, damping in percent. Only supported value is 5.

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

	def log_sigma(self, M=5., d=10., h=0., imt="PGA", T=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
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
			String, one of the supported intensity measure types: "PGA" or "SA"
			(default: "PGA")
		:param T:
			Float, spectral period of considered IMT. Ignored if IMT == "PGA"
			(default: 0)
		:param soil_type:
			String, either "rock", "stiff" or "soft" (default: "rock").
				Rock: VS >= 750 m/s
				Stiff soil: 360 <= VS < 750 m/s
				Soft soil: 180 <= VS < 360 m/s
		:param vs30:
			Float, shear-wave velocity in the upper 30 m (in m/s). If not None,
			it takes precedence over the soil_type parameter (default: None).
		:param kappa:
			Float, kappa value, in seconds. Ignored in this GMPE (default: None)
		:param mechanism:
			String, focal mechanism. Ignored in this GMPE (default: None)
		:param damping:
			Float, damping in percent. Only supported value is 5
		"""
		imt = imt.upper()
		if not self.has_imt(imt):
			raise IMTUndefinedError(imt)
		if imt == "PGA":
			return self.sigma[imt][0]
		else:
			if T < self.Tmin(imt) or T > self.Tmax(imt):
				raise PeriodUndefinedError(imt, T)
				#return None
			elif damping not in self.dampings:
				raise DampingNotSupportedError(damping)
			else:
				return interpolate(self.imt_periods["SA"], self.sigma["SA"], [T])[0]

	def is_rake_dependent(self):
		"""
		Indicate whether or not GMPE depends on rake of the source
		"""
		return False

	def get_CRISIS_periods(self):
		"""
		Return array of max. 40 spectral periods to be used with CRISIS
		"""
		return np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.22, 0.24, 0.26, 0.28, 0.30, 0.325, 0.35, 0.375, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])

	def plot_Figure4(self):
		"""
		Plot Figure 4 in the paper of Ambraseys et al. (1996)
		"""
		self.plot_distance(mags=[5., 6., 7.], dmin=1, dmax=1E3, amin=1E-3, amax=1)

	def plot_Figure17(self, soil_type="rock"):
		"""
		Plot Figure 17 in the paper of Ambraseys et al. (1996)

		:param soil_type:
			String, either "rock", "stiff" or "soft" (default: "rock").
		"""
		self.plot_spectrum(mags=[5.], d=10, plot_style="lin", Tmin=0, Tmax=2.0, amax=0.4, include_pgm=False, soil_type=soil_type, want_minor_grid=True)

	def plot_Figure18(self, soil_type="rock"):
		"""
		Plot Figure 18 in the paper of Ambraseys et al. (1996)

		:param soil_type:
			String, either "rock", "stiff" or "soft" (default: "rock").
		"""
		self.plot_spectrum(mags=[7.], d=10, plot_style="lin", Tmin=0, Tmax=2.0, amax=1.15, include_pgm=False, soil_type=soil_type, want_minor_grid=True)

	def plot_Figure19(self, soil_type="rock"):
		"""
		Plot Figure 19 in the paper of Ambraseys et al. (1996)

		:param soil_type:
			String, either "rock", "stiff" or "soft" (default: "rock").
		"""
		self.plot_spectrum(mags=[7.], d=40, plot_style="lin", Tmin=0, Tmax=2.0, amax=0.37, include_pgm=False, soil_type=soil_type, want_minor_grid=True)


class BergeThierry2003GMPE(GMPE):
	"""
	Berge-Thierry et al. (2003)
		Magnitude scale: MS
		Magnitude range: 4 - 7.9
		Distance metric: Hypocentral
		Distance range: 4 - 330 km
		Intensity measure types: (P)SA (PGA ~ SA at 34 Hz, 5% damping)
		Original IMT unit: cm/s2 (not explicitly mentioned in paper!)
		SA period range: 0.0294 - 10 s
		Dampings for SA: 5, 7, 10, 20
		Soil classes:
			rock (vs30 >= 800 m/s)
			alluvium (300 <= vs30 < 800 m/s)
		Fault types: None
	"""
	def __init__(self):
		freqs = np.array([1.0000E-001, 1.1100E-001, 1.2500E-001, 1.4300E-001, 1.6700E-001, 1.8200E-001, 2.0000E-001, 2.2200E-001, 2.5000E-001, 2.6300E-001, 2.7800E-001, 2.9400E-001, 3.0000E-001, 3.1200E-001, 3.3300E-001, 3.5700E-001, 3.8500E-001, 4.0000E-001, 4.1700E-001, 4.5500E-001, 5.0000E-001, 5.5600E-001, 6.0000E-001, 6.2500E-001, 6.6700E-001, 7.0000E-001, 7.1400E-001, 7.6900E-001, 8.0000E-001, 8.3300E-001, 9.0000E-001, 9.0900E-001, 1.0000E+000, 1.1000E+000, 1.1110E+000, 1.1760E+000, 1.2000E+000, 1.2500E+000, 1.3000E+000, 1.3330E+000, 1.4000E+000, 1.4290E+000, 1.4710E+000, 1.5000E+000, 1.5150E+000, 1.5620E+000, 1.6000E+000, 1.6130E+000, 1.6670E+000, 1.7000E+000, 1.7240E+000, 1.7860E+000, 1.8000E+000, 1.8520E+000, 1.9000E+000, 1.9230E+000, 2.0000E+000, 2.0830E+000, 2.1000E+000, 2.1740E+000, 2.2000E+000, 2.2730E+000, 2.3000E+000, 2.3810E+000, 2.4000E+000, 2.5000E+000, 2.6000E+000, 2.6320E+000, 2.7000E+000, 2.7780E+000, 2.8000E+000, 2.9000E+000, 2.9410E+000, 3.0000E+000, 3.1250E+000, 3.1550E+000, 3.3000E+000, 3.3330E+000, 3.4480E+000, 3.5710E+000, 3.6000E+000, 3.8000E+000, 3.8500E+000, 4.0000E+000, 4.1670E+000, 4.2000E+000, 4.4000E+000, 4.5500E+000, 4.6000E+000, 4.8000E+000, 5.0000E+000, 5.2500E+000, 5.2630E+000, 5.5000E+000, 5.5560E+000, 5.7500E+000, 5.8820E+000, 6.0000E+000, 6.2500E+000, 6.5000E+000, 6.6670E+000, 6.7500E+000, 7.0000E+000, 7.1430E+000, 7.2500E+000, 7.5000E+000, 7.6920E+000, 7.7500E+000, 8.0000E+000, 8.3330E+000, 8.5000E+000, 9.0000E+000, 9.0910E+000, 9.5000E+000, 1.0000E+001, 1.0500E+001, 1.1000E+001, 1.1111E+001, 1.1500E+001, 1.1765E+001, 1.2000E+001, 1.2500E+001, 1.3000E+001, 1.3333E+001, 1.3500E+001, 1.4000E+001, 1.4286E+001, 1.4500E+001, 1.5000E+001, 1.5385E+001, 1.6000E+001, 1.6667E+001, 1.7000E+001, 1.8000E+001, 1.8868E+001, 2.0000E+001, 2.2000E+001, 2.5000E+001, 2.8000E+001, 2.9412E+001, 3.1000E+001, 3.3333E+001, 3.4000E+001])
		imt_periods = {}
		imt_periods["SA"] = 1./freqs
		distance_metric = "Hypocentral"
		Mmin, Mmax = 4, 7.9
		dmin, dmax = 4., 330.
		Mtype = "MS"
		dampings = [5, 7, 10, 20]
		name = "Berge-Thierry2003"
		short_name = "BT_2003"
		GMPE.__init__(self, imt_periods, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings, name, short_name)

		## Coefficients
		self.a = np.zeros((len(self.dampings), len(freqs)), 'd')
		self.b = np.zeros((len(self.dampings), len(freqs)), 'd')
		self.c1 = np.zeros((len(self.dampings), len(freqs)), 'd')
		self.c2 = np.zeros((len(self.dampings), len(freqs)), 'd')
		self.sigma = np.zeros((len(self.dampings), len(freqs)), 'd')
		for i in range(len(self.dampings)):
			if i == 0:
				self.a[i] = [6.0860E-001, 6.1220E-001, 6.1450E-001, 6.1750E-001, 6.1600E-001, 6.1060E-001, 5.9900E-001, 5.8560E-001, 5.7220E-001, 5.7120E-001, 5.7270E-001, 5.7150E-001, 5.7050E-001, 5.6860E-001, 5.6830E-001, 5.6660E-001, 5.6770E-001, 5.6540E-001, 5.6410E-001, 5.6170E-001, 5.6220E-001, 5.6200E-001, 5.5800E-001, 5.5570E-001, 5.5270E-001, 5.4940E-001, 5.4810E-001, 5.4440E-001, 5.4090E-001, 5.3610E-001, 5.2780E-001, 5.2730E-001, 5.1990E-001, 5.1040E-001, 5.0980E-001, 5.0400E-001, 5.0100E-001, 4.9400E-001, 4.8750E-001, 4.8470E-001, 4.7710E-001, 4.7320E-001, 4.6880E-001, 4.6550E-001, 4.6370E-001, 4.5960E-001, 4.5690E-001, 4.5590E-001, 4.5160E-001, 4.4920E-001, 4.4720E-001, 4.4250E-001, 4.4180E-001, 4.3940E-001, 4.3790E-001, 4.3720E-001, 4.3230E-001, 4.2390E-001, 4.2220E-001, 4.1650E-001, 4.1480E-001, 4.0890E-001, 4.0700E-001, 4.0340E-001, 4.0280E-001, 3.9970E-001, 3.9310E-001, 3.9090E-001, 3.8670E-001, 3.8220E-001, 3.8070E-001, 3.7600E-001, 3.7520E-001, 3.7420E-001, 3.6900E-001, 3.6710E-001, 3.6020E-001, 3.5900E-001, 3.5550E-001, 3.5110E-001, 3.5010E-001, 3.4420E-001, 3.4300E-001, 3.3650E-001, 3.3440E-001, 3.3400E-001, 3.3030E-001, 3.2710E-001, 3.2540E-001, 3.1960E-001, 3.1670E-001, 3.1490E-001, 3.1470E-001, 3.1090E-001, 3.0890E-001, 3.0160E-001, 2.9740E-001, 2.9520E-001, 2.9390E-001, 2.9550E-001, 2.9550E-001, 2.9500E-001, 2.9330E-001, 2.9150E-001, 2.9030E-001, 2.8870E-001, 2.8670E-001, 2.8630E-001, 2.8310E-001, 2.8060E-001, 2.7930E-001, 2.7830E-001, 2.7830E-001, 2.7760E-001, 2.7860E-001, 2.7810E-001, 2.8090E-001, 2.8190E-001, 2.8480E-001, 2.8560E-001, 2.8580E-001, 2.8660E-001, 2.8570E-001, 2.8710E-001, 2.8790E-001, 2.9090E-001, 2.9200E-001, 2.9240E-001, 2.9330E-001, 2.9440E-001, 2.9650E-001, 2.9600E-001, 2.9600E-001, 2.9690E-001, 2.9810E-001, 2.9920E-001, 3.0160E-001, 3.0330E-001, 3.0680E-001, 3.0830E-001, 3.0970E-001, 3.1140E-001, 3.1180E-001]
				self.b[i] = [1.5630E-003, 1.6370E-003, 1.7210E-003, 1.7660E-003, 1.8800E-003, 1.9410E-003, 2.1050E-003, 2.4490E-003, 2.7110E-003, 2.6620E-003, 2.5730E-003, 2.5410E-003, 2.5330E-003, 2.5360E-003, 2.4490E-003, 2.2770E-003, 2.0060E-003, 1.9210E-003, 1.8290E-003, 1.6520E-003, 1.3750E-003, 1.2450E-003, 1.0850E-003, 9.8440E-004, 9.1240E-004, 8.2720E-004, 7.6760E-004, 5.3290E-004, 4.8600E-004, 4.4790E-004, 4.0740E-004, 3.9080E-004, 2.5160E-004, 8.3930E-005, 3.2820E-005, -1.4330E-004, -1.9320E-004, -2.5680E-004, -3.1220E-004, -3.0090E-004, -2.0190E-004, -1.7000E-004, -1.6680E-004, -1.5500E-004, -1.5490E-004, -1.6660E-004, -1.9950E-004, -1.9530E-004, -2.1750E-004, -2.5220E-004, -2.7020E-004, -3.3800E-004, -3.6010E-004, -4.3300E-004, -5.0500E-004, -5.3960E-004, -5.6800E-004, -5.4840E-004, -5.4040E-004, -5.8160E-004, -5.9310E-004, -6.1180E-004, -6.1670E-004, -6.5130E-004, -6.6130E-004, -7.0780E-004, -7.9550E-004, -8.0740E-004, -8.6350E-004, -9.0390E-004, -9.1140E-004, -9.6980E-004, -1.0060E-003, -1.0100E-003, -9.4680E-004, -9.2720E-004, -8.7370E-004, -8.5200E-004, -7.8360E-004, -7.5300E-004, -7.5200E-004, -7.2000E-004, -7.0750E-004, -5.7500E-004, -5.9880E-004, -6.1710E-004, -6.6780E-004, -6.9180E-004, -6.7500E-004, -6.7190E-004, -6.8890E-004, -7.3370E-004, -7.3690E-004, -7.8260E-004, -7.7930E-004, -7.3410E-004, -6.9860E-004, -7.0970E-004, -8.0560E-004, -1.0520E-003, -1.2180E-003, -1.2560E-003, -1.3070E-003, -1.3220E-003, -1.3380E-003, -1.3970E-003, -1.4250E-003, -1.4400E-003, -1.4150E-003, -1.4360E-003, -1.4380E-003, -1.4380E-003, -1.4420E-003, -1.4220E-003, -1.4740E-003, -1.5430E-003, -1.5280E-003, -1.5360E-003, -1.5710E-003, -1.5830E-003, -1.5520E-003, -1.5310E-003, -1.5200E-003, -1.5400E-003, -1.5550E-003, -1.5830E-003, -1.5700E-003, -1.5460E-003, -1.5130E-003, -1.5150E-003, -1.5070E-003, -1.4720E-003, -1.4600E-003, -1.4320E-003, -1.4290E-003, -1.3410E-003, -1.2820E-003, -1.1190E-003, -9.8220E-004, -9.5470E-004, -9.4220E-004, -9.3340E-004, -9.3030E-004]
				self.c1[i] = [-2.6680E+000, -2.5920E+000, -2.4910E+000, -2.3730E+000, -2.2010E+000, -2.0720E+000, -1.8860E+000, -1.6620E+000, -1.4170E+000, -1.3500E+000, -1.3000E+000, -1.2430E+000, -1.2200E+000, -1.1790E+000, -1.1300E+000, -1.0710E+000, -1.0190E+000, -9.7870E-001, -9.4060E-001, -8.6560E-001, -7.9630E-001, -7.2580E-001, -6.5640E-001, -6.1860E-001, -5.6040E-001, -5.0950E-001, -4.8700E-001, -4.1130E-001, -3.6790E-001, -3.1330E-001, -2.2070E-001, -2.1230E-001, -1.1620E-001, 8.1950E-003, 2.0060E-002, 9.6150E-002, 1.2640E-001, 1.9060E-001, 2.5450E-001, 2.8710E-001, 3.5480E-001, 3.8570E-001, 4.2840E-001, 4.5730E-001, 4.7250E-001, 5.1060E-001, 5.3830E-001, 5.4780E-001, 5.8950E-001, 6.1570E-001, 6.3570E-001, 6.8440E-001, 6.9410E-001, 7.2710E-001, 7.5220E-001, 7.6420E-001, 8.1500E-001, 8.7950E-001, 8.9310E-001, 9.4400E-001, 9.6010E-001, 1.0110E+000, 1.0290E+000, 1.0700E+000, 1.0780E+000, 1.1190E+000, 1.1790E+000, 1.1990E+000, 1.2370E+000, 1.2750E+000, 1.2860E+000, 1.3260E+000, 1.3370E+000, 1.3520E+000, 1.3970E+000, 1.4120E+000, 1.4660E+000, 1.4770E+000, 1.5060E+000, 1.5420E+000, 1.5500E+000, 1.5990E+000, 1.6090E+000, 1.6510E+000, 1.6770E+000, 1.6830E+000, 1.7260E+000, 1.7580E+000, 1.7700E+000, 1.8140E+000, 1.8430E+000, 1.8670E+000, 1.8680E+000, 1.9010E+000, 1.9150E+000, 1.9570E+000, 1.9780E+000, 1.9890E+000, 1.9960E+000, 1.9970E+000, 2.0040E+000, 2.0090E+000, 2.0180E+000, 2.0270E+000, 2.0320E+000, 2.0400E+000, 2.0500E+000, 2.0520E+000, 2.0660E+000, 2.0760E+000, 2.0810E+000, 2.0770E+000, 2.0750E+000, 2.0720E+000, 2.0590E+000, 2.0540E+000, 2.0200E+000, 2.0110E+000, 1.9870E+000, 1.9760E+000, 1.9680E+000, 1.9540E+000, 1.9470E+000, 1.9330E+000, 1.9260E+000, 1.9010E+000, 1.8880E+000, 1.8810E+000, 1.8650E+000, 1.8510E+000, 1.8260E+000, 1.8140E+000, 1.8090E+000, 1.7850E+000, 1.7660E+000, 1.7400E+000, 1.7010E+000, 1.6530E+000, 1.5930E+000, 1.5730E+000, 1.5580E+000, 1.5410E+000, 1.5370E+000]
				self.c2[i] = [-2.4850E+000, -2.4080E+000, -2.3080E+000, -2.1920E+000, -2.0280E+000, -1.9040E+000, -1.7290E+000, -1.5200E+000, -1.3030E+000, -1.2420E+000, -1.1940E+000, -1.1350E+000, -1.1110E+000, -1.0690E+000, -1.0140E+000, -9.4950E-001, -8.9800E-001, -8.5420E-001, -8.1580E-001, -7.3950E-001, -6.6600E-001, -5.8340E-001, -5.0260E-001, -4.5830E-001, -3.9350E-001, -3.3850E-001, -3.1550E-001, -2.3770E-001, -1.8910E-001, -1.3380E-001, -3.8750E-002, -2.9000E-002, 8.2900E-002, 2.0220E-001, 2.1300E-001, 2.8420E-001, 3.1450E-001, 3.7820E-001, 4.3800E-001, 4.6810E-001, 5.3540E-001, 5.6760E-001, 6.0810E-001, 6.3630E-001, 6.5200E-001, 6.9100E-001, 7.1870E-001, 7.2820E-001, 7.6820E-001, 7.9260E-001, 8.1170E-001, 8.5710E-001, 8.6610E-001, 8.9590E-001, 9.2080E-001, 9.3240E-001, 9.7970E-001, 1.0450E+000, 1.0580E+000, 1.1090E+000, 1.1250E+000, 1.1740E+000, 1.1910E+000, 1.2260E+000, 1.2320E+000, 1.2670E+000, 1.3210E+000, 1.3390E+000, 1.3720E+000, 1.4050E+000, 1.4150E+000, 1.4520E+000, 1.4610E+000, 1.4720E+000, 1.5120E+000, 1.5250E+000, 1.5730E+000, 1.5810E+000, 1.6050E+000, 1.6380E+000, 1.6450E+000, 1.6880E+000, 1.6970E+000, 1.7360E+000, 1.7580E+000, 1.7620E+000, 1.7920E+000, 1.8150E+000, 1.8250E+000, 1.8610E+000, 1.8810E+000, 1.8930E+000, 1.8940E+000, 1.9190E+000, 1.9300E+000, 1.9700E+000, 1.9940E+000, 2.0060E+000, 2.0130E+000, 2.0070E+000, 2.0090E+000, 2.0100E+000, 2.0140E+000, 2.0220E+000, 2.0280E+000, 2.0330E+000, 2.0400E+000, 2.0420E+000, 2.0510E+000, 2.0540E+000, 2.0560E+000, 2.0470E+000, 2.0450E+000, 2.0340E+000, 2.0160E+000, 2.0100E+000, 1.9810E+000, 1.9730E+000, 1.9460E+000, 1.9350E+000, 1.9270E+000, 1.9110E+000, 1.9060E+000, 1.8930E+000, 1.8870E+000, 1.8610E+000, 1.8490E+000, 1.8420E+000, 1.8290E+000, 1.8170E+000, 1.7960E+000, 1.7880E+000, 1.7840E+000, 1.7650E+000, 1.7490E+000, 1.7290E+000, 1.7000E+000, 1.6650E+000, 1.6180E+000, 1.6020E+000, 1.5890E+000, 1.5760E+000, 1.5730E+000]
				self.sigma[i] = [4.1830E-001, 4.2360E-001, 4.2840E-001, 4.2990E-001, 4.2840E-001, 4.2600E-001, 4.2330E-001, 4.2780E-001, 4.3440E-001, 4.3650E-001, 4.3590E-001, 4.3400E-001, 4.3290E-001, 4.3010E-001, 4.2550E-001, 4.2020E-001, 4.1300E-001, 4.1100E-001, 4.0930E-001, 4.0550E-001, 4.0300E-001, 3.9420E-001, 3.9190E-001, 3.9190E-001, 3.9320E-001, 3.9410E-001, 3.9350E-001, 3.9200E-001, 3.8770E-001, 3.8380E-001, 3.8040E-001, 3.7940E-001, 3.7370E-001, 3.7440E-001, 3.7470E-001, 3.7580E-001, 3.7460E-001, 3.7140E-001, 3.7030E-001, 3.7000E-001, 3.6800E-001, 3.6670E-001, 3.6550E-001, 3.6490E-001, 3.6470E-001, 3.6250E-001, 3.6090E-001, 3.6040E-001, 3.6030E-001, 3.6090E-001, 3.6100E-001, 3.5920E-001, 3.5870E-001, 3.5740E-001, 3.5700E-001, 3.5680E-001, 3.5550E-001, 3.5560E-001, 3.5550E-001, 3.5490E-001, 3.5470E-001, 3.5270E-001, 3.5210E-001, 3.5130E-001, 3.5120E-001, 3.5170E-001, 3.5070E-001, 3.5000E-001, 3.4920E-001, 3.4840E-001, 3.4810E-001, 3.4710E-001, 3.4690E-001, 3.4740E-001, 3.4870E-001, 3.4910E-001, 3.4830E-001, 3.4770E-001, 3.4580E-001, 3.4470E-001, 3.4440E-001, 3.4290E-001, 3.4220E-001, 3.3940E-001, 3.3710E-001, 3.3670E-001, 3.3200E-001, 3.2920E-001, 3.2810E-001, 3.2610E-001, 3.2500E-001, 3.2620E-001, 3.2620E-001, 3.2710E-001, 3.2670E-001, 3.2470E-001, 3.2300E-001, 3.2150E-001, 3.2000E-001, 3.2010E-001, 3.1960E-001, 3.1910E-001, 3.1730E-001, 3.1660E-001, 3.1610E-001, 3.1560E-001, 3.1420E-001, 3.1370E-001, 3.1210E-001, 3.1060E-001, 3.1030E-001, 3.0790E-001, 3.0730E-001, 3.0430E-001, 3.0190E-001, 3.0170E-001, 3.0160E-001, 3.0200E-001, 3.0420E-001, 3.0530E-001, 3.0580E-001, 3.0610E-001, 3.0720E-001, 3.0660E-001, 3.0590E-001, 3.0470E-001, 3.0460E-001, 3.0480E-001, 3.0590E-001, 3.0590E-001, 3.0470E-001, 3.0410E-001, 3.0400E-001, 3.0270E-001, 3.0220E-001, 3.0090E-001, 3.0150E-001, 2.9820E-001, 2.9470E-001, 2.9350E-001, 2.9280E-001, 2.9240E-001, 2.9230E-001]
			elif i == 1:
				self.a[i] = [6.057000E-01,6.085000E-01,6.105000E-01,6.127000E-01,6.105000E-01,6.050000E-01,5.947000E-01,5.828000E-01,5.717000E-01,5.705000E-01,5.711000E-01,5.687000E-01,5.677000E-01,5.660000E-01,5.655000E-01,5.641000E-01,5.654000E-01,5.636000E-01,5.629000E-01,5.601000E-01,5.597000E-01,5.595000E-01,5.557000E-01,5.540000E-01,5.501000E-01,5.472000E-01,5.457000E-01,5.402000E-01,5.371000E-01,5.330000E-01,5.255000E-01,5.248000E-01,5.169000E-01,5.080000E-01,5.074000E-01,5.017000E-01,4.984000E-01,4.918000E-01,4.856000E-01,4.825000E-01,4.749000E-01,4.710000E-01,4.662000E-01,4.634000E-01,4.620000E-01,4.580000E-01,4.552000E-01,4.540000E-01,4.501000E-01,4.478000E-01,4.457000E-01,4.399000E-01,4.390000E-01,4.365000E-01,4.350000E-01,4.339000E-01,4.288000E-01,4.223000E-01,4.208000E-01,4.151000E-01,4.134000E-01,4.081000E-01,4.064000E-01,4.025000E-01,4.016000E-01,3.987000E-01,3.928000E-01,3.907000E-01,3.866000E-01,3.824000E-01,3.812000E-01,3.761000E-01,3.742000E-01,3.722000E-01,3.667000E-01,3.653000E-01,3.600000E-01,3.589000E-01,3.553000E-01,3.510000E-01,3.498000E-01,3.431000E-01,3.417000E-01,3.369000E-01,3.338000E-01,3.335000E-01,3.298000E-01,3.270000E-01,3.255000E-01,3.202000E-01,3.170000E-01,3.146000E-01,3.145000E-01,3.098000E-01,3.080000E-01,3.016000E-01,2.979000E-01,2.957000E-01,2.941000E-01,2.944000E-01,2.945000E-01,2.941000E-01,2.927000E-01,2.912000E-01,2.900000E-01,2.882000E-01,2.866000E-01,2.864000E-01,2.841000E-01,2.819000E-01,2.809000E-01,2.804000E-01,2.806000E-01,2.804000E-01,2.804000E-01,2.803000E-01,2.825000E-01,2.833000E-01,2.852000E-01,2.862000E-01,2.866000E-01,2.868000E-01,2.868000E-01,2.879000E-01,2.888000E-01,2.916000E-01,2.926000E-01,2.932000E-01,2.945000E-01,2.954000E-01,2.966000E-01,2.968000E-01,2.970000E-01,2.980000E-01,2.991000E-01,3.005000E-01,3.028000E-01,3.041000E-01,3.068000E-01,3.080000E-01,3.091000E-01,3.106000E-01,3.110000E-01]
				self.b[i] = [1.555000E-03,1.618000E-03,1.700000E-03,1.745000E-03,1.846000E-03,1.916000E-03,2.061000E-03,2.341000E-03,2.541000E-03,2.504000E-03,2.447000E-03,2.424000E-03,2.415000E-03,2.388000E-03,2.295000E-03,2.109000E-03,1.859000E-03,1.804000E-03,1.706000E-03,1.524000E-03,1.293000E-03,1.120000E-03,9.923000E-04,8.977000E-04,8.059000E-04,6.960000E-04,6.340000E-04,4.648000E-04,3.997000E-04,3.336000E-04,2.994000E-04,2.890000E-04,1.602000E-04,3.432000E-06,-3.176000E-05,-2.118000E-04,-2.596000E-04,-3.123000E-04,-3.449000E-04,-3.381000E-04,-2.753000E-04,-2.390000E-04,-2.305000E-04,-2.223000E-04,-2.211000E-04,-2.413000E-04,-2.575000E-04,-2.537000E-04,-2.929000E-04,-3.308000E-04,-3.468000E-04,-3.773000E-04,-3.915000E-04,-4.431000E-04,-5.160000E-04,-5.477000E-04,-5.909000E-04,-6.366000E-04,-6.445000E-04,-6.885000E-04,-7.025000E-04,-7.138000E-04,-7.167000E-04,-7.306000E-04,-7.303000E-04,-7.918000E-04,-8.838000E-04,-9.062000E-04,-9.418000E-04,-9.608000E-04,-9.697000E-04,-1.015000E-03,-1.023000E-03,-1.018000E-03,-9.515000E-04,-9.357000E-04,-8.907000E-04,-8.854000E-04,-8.420000E-04,-7.886000E-04,-7.816000E-04,-7.346000E-04,-7.123000E-04,-6.353000E-04,-6.328000E-04,-6.448000E-04,-6.894000E-04,-7.078000E-04,-6.973000E-04,-6.887000E-04,-7.128000E-04,-7.271000E-04,-7.291000E-04,-7.540000E-04,-7.478000E-04,-7.050000E-04,-6.848000E-04,-6.937000E-04,-7.836000E-04,-9.667000E-04,-1.094000E-03,-1.129000E-03,-1.201000E-03,-1.226000E-03,-1.241000E-03,-1.289000E-03,-1.311000E-03,-1.317000E-03,-1.306000E-03,-1.332000E-03,-1.335000E-03,-1.356000E-03,-1.363000E-03,-1.359000E-03,-1.385000E-03,-1.441000E-03,-1.444000E-03,-1.448000E-03,-1.458000E-03,-1.456000E-03,-1.444000E-03,-1.412000E-03,-1.399000E-03,-1.416000E-03,-1.427000E-03,-1.458000E-03,-1.460000E-03,-1.453000E-03,-1.435000E-03,-1.425000E-03,-1.414000E-03,-1.395000E-03,-1.386000E-03,-1.354000E-03,-1.343000E-03,-1.282000E-03,-1.242000E-03,-1.088000E-03,-9.771000E-04,-9.498000E-04,-9.382000E-04,-9.287000E-04,-9.264000E-04]
				self.c1[i] = [-2.653000E+00,-2.573000E+00,-2.471000E+00,-2.349000E+00,-2.174000E+00,-2.046000E+00,-1.871000E+00,-1.665000E+00,-1.444000E+00,-1.378000E+00,-1.325000E+00,-1.261000E+00,-1.239000E+00,-1.198000E+00,-1.147000E+00,-1.089000E+00,-1.040000E+00,-1.004000E+00,-9.679000E-01,-8.899000E-01,-8.192000E-01,-7.459000E-01,-6.783000E-01,-6.433000E-01,-5.801000E-01,-5.306000E-01,-5.074000E-01,-4.274000E-01,-3.844000E-01,-3.335000E-01,-2.459000E-01,-2.363000E-01,-1.367000E-01,-1.704000E-02,-5.691000E-03,6.829000E-02,9.968000E-02,1.633000E-01,2.253000E-01,2.583000E-01,3.271000E-01,3.585000E-01,4.015000E-01,4.272000E-01,4.401000E-01,4.771000E-01,5.058000E-01,5.163000E-01,5.573000E-01,5.827000E-01,6.031000E-01,6.570000E-01,6.674000E-01,7.003000E-01,7.252000E-01,7.390000E-01,7.910000E-01,8.474000E-01,8.594000E-01,9.108000E-01,9.270000E-01,9.740000E-01,9.902000E-01,1.030000E+00,1.039000E+00,1.080000E+00,1.136000E+00,1.154000E+00,1.191000E+00,1.226000E+00,1.235000E+00,1.279000E+00,1.295000E+00,1.315000E+00,1.362000E+00,1.373000E+00,1.419000E+00,1.429000E+00,1.459000E+00,1.494000E+00,1.503000E+00,1.556000E+00,1.567000E+00,1.603000E+00,1.632000E+00,1.638000E+00,1.678000E+00,1.707000E+00,1.718000E+00,1.759000E+00,1.789000E+00,1.815000E+00,1.817000E+00,1.853000E+00,1.864000E+00,1.901000E+00,1.921000E+00,1.933000E+00,1.944000E+00,1.951000E+00,1.956000E+00,1.959000E+00,1.968000E+00,1.976000E+00,1.982000E+00,1.991000E+00,1.999000E+00,2.000000E+00,2.009000E+00,2.019000E+00,2.023000E+00,2.018000E+00,2.016000E+00,2.011000E+00,2.003000E+00,1.997000E+00,1.970000E+00,1.963000E+00,1.944000E+00,1.932000E+00,1.924000E+00,1.913000E+00,1.902000E+00,1.890000E+00,1.883000E+00,1.860000E+00,1.850000E+00,1.843000E+00,1.826000E+00,1.813000E+00,1.795000E+00,1.782000E+00,1.776000E+00,1.755000E+00,1.737000E+00,1.712000E+00,1.678000E+00,1.637000E+00,1.590000E+00,1.573000E+00,1.560000E+00,1.545000E+00,1.541000E+00]
				self.c2[i] = [-2.472000E+00,-2.391000E+00,-2.290000E+00,-2.171000E+00,-2.003000E+00,-1.881000E+00,-1.715000E+00,-1.522000E+00,-1.326000E+00,-1.265000E+00,-1.214000E+00,-1.149000E+00,-1.126000E+00,-1.084000E+00,-1.029000E+00,-9.651000E-01,-9.153000E-01,-8.764000E-01,-8.404000E-01,-7.612000E-01,-6.854000E-01,-6.035000E-01,-5.256000E-01,-4.853000E-01,-4.156000E-01,-3.621000E-01,-3.380000E-01,-2.520000E-01,-2.059000E-01,-1.544000E-01,-6.381000E-02,-5.351000E-02,5.974000E-02,1.743000E-01,1.845000E-01,2.554000E-01,2.867000E-01,3.479000E-01,4.049000E-01,4.358000E-01,5.050000E-01,5.373000E-01,5.796000E-01,6.050000E-01,6.183000E-01,6.566000E-01,6.856000E-01,6.961000E-01,7.349000E-01,7.579000E-01,7.768000E-01,8.275000E-01,8.368000E-01,8.669000E-01,8.904000E-01,9.036000E-01,9.527000E-01,1.010000E+00,1.022000E+00,1.073000E+00,1.089000E+00,1.134000E+00,1.149000E+00,1.183000E+00,1.191000E+00,1.226000E+00,1.277000E+00,1.294000E+00,1.327000E+00,1.358000E+00,1.366000E+00,1.405000E+00,1.419000E+00,1.435000E+00,1.476000E+00,1.486000E+00,1.526000E+00,1.534000E+00,1.561000E+00,1.592000E+00,1.600000E+00,1.647000E+00,1.656000E+00,1.688000E+00,1.714000E+00,1.718000E+00,1.747000E+00,1.768000E+00,1.777000E+00,1.809000E+00,1.831000E+00,1.847000E+00,1.848000E+00,1.877000E+00,1.888000E+00,1.923000E+00,1.943000E+00,1.956000E+00,1.966000E+00,1.966000E+00,1.966000E+00,1.967000E+00,1.971000E+00,1.977000E+00,1.983000E+00,1.991000E+00,1.996000E+00,1.996000E+00,2.002000E+00,2.005000E+00,2.006000E+00,1.996000E+00,1.993000E+00,1.982000E+00,1.970000E+00,1.963000E+00,1.939000E+00,1.932000E+00,1.912000E+00,1.900000E+00,1.893000E+00,1.881000E+00,1.872000E+00,1.861000E+00,1.854000E+00,1.831000E+00,1.820000E+00,1.814000E+00,1.800000E+00,1.789000E+00,1.774000E+00,1.764000E+00,1.759000E+00,1.742000E+00,1.728000E+00,1.709000E+00,1.682000E+00,1.652000E+00,1.616000E+00,1.603000E+00,1.592000E+00,1.579000E+00,1.576000E+00]
				self.sigma[i] = [4.157000E-01,4.203000E-01,4.244000E-01,4.254000E-01,4.233000E-01,4.212000E-01,4.194000E-01,4.229000E-01,4.281000E-01,4.297000E-01,4.287000E-01,4.270000E-01,4.259000E-01,4.234000E-01,4.190000E-01,4.138000E-01,4.073000E-01,4.060000E-01,4.045000E-01,4.006000E-01,3.987000E-01,3.919000E-01,3.894000E-01,3.891000E-01,3.906000E-01,3.909000E-01,3.904000E-01,3.883000E-01,3.846000E-01,3.811000E-01,3.775000E-01,3.768000E-01,3.720000E-01,3.719000E-01,3.722000E-01,3.732000E-01,3.722000E-01,3.696000E-01,3.681000E-01,3.676000E-01,3.658000E-01,3.645000E-01,3.634000E-01,3.628000E-01,3.625000E-01,3.607000E-01,3.593000E-01,3.589000E-01,3.585000E-01,3.586000E-01,3.585000E-01,3.569000E-01,3.565000E-01,3.555000E-01,3.556000E-01,3.555000E-01,3.541000E-01,3.536000E-01,3.537000E-01,3.532000E-01,3.526000E-01,3.514000E-01,3.511000E-01,3.502000E-01,3.500000E-01,3.495000E-01,3.490000E-01,3.485000E-01,3.475000E-01,3.470000E-01,3.468000E-01,3.459000E-01,3.458000E-01,3.458000E-01,3.464000E-01,3.464000E-01,3.456000E-01,3.451000E-01,3.434000E-01,3.418000E-01,3.414000E-01,3.397000E-01,3.392000E-01,3.371000E-01,3.350000E-01,3.346000E-01,3.307000E-01,3.278000E-01,3.270000E-01,3.246000E-01,3.236000E-01,3.237000E-01,3.237000E-01,3.232000E-01,3.230000E-01,3.214000E-01,3.199000E-01,3.184000E-01,3.170000E-01,3.162000E-01,3.161000E-01,3.155000E-01,3.144000E-01,3.133000E-01,3.128000E-01,3.120000E-01,3.107000E-01,3.103000E-01,3.086000E-01,3.076000E-01,3.072000E-01,3.051000E-01,3.048000E-01,3.023000E-01,3.002000E-01,3.003000E-01,3.002000E-01,3.004000E-01,3.016000E-01,3.023000E-01,3.027000E-01,3.033000E-01,3.034000E-01,3.032000E-01,3.028000E-01,3.020000E-01,3.020000E-01,3.021000E-01,3.027000E-01,3.029000E-01,3.027000E-01,3.024000E-01,3.021000E-01,3.011000E-01,3.001000E-01,2.991000E-01,2.990000E-01,2.968000E-01,2.944000E-01,2.934000E-01,2.928000E-01,2.924000E-01,2.923000E-01]
			elif i == 2:
				self.a[i] = [6.016000E-01,6.039000E-01,6.055000E-01,6.064000E-01,6.036000E-01,5.983000E-01,5.894000E-01,5.800000E-01,5.710000E-01,5.693000E-01,5.687000E-01,5.655000E-01,5.647000E-01,5.631000E-01,5.620000E-01,5.610000E-01,5.622000E-01,5.611000E-01,5.605000E-01,5.577000E-01,5.567000E-01,5.565000E-01,5.531000E-01,5.510000E-01,5.466000E-01,5.434000E-01,5.418000E-01,5.355000E-01,5.321000E-01,5.282000E-01,5.218000E-01,5.211000E-01,5.134000E-01,5.041000E-01,5.033000E-01,4.970000E-01,4.937000E-01,4.875000E-01,4.829000E-01,4.797000E-01,4.721000E-01,4.686000E-01,4.636000E-01,4.607000E-01,4.592000E-01,4.549000E-01,4.523000E-01,4.514000E-01,4.478000E-01,4.452000E-01,4.430000E-01,4.376000E-01,4.367000E-01,4.337000E-01,4.315000E-01,4.302000E-01,4.251000E-01,4.195000E-01,4.183000E-01,4.132000E-01,4.114000E-01,4.068000E-01,4.054000E-01,4.010000E-01,4.003000E-01,3.971000E-01,3.920000E-01,3.901000E-01,3.862000E-01,3.822000E-01,3.810000E-01,3.762000E-01,3.744000E-01,3.720000E-01,3.670000E-01,3.658000E-01,3.605000E-01,3.592000E-01,3.555000E-01,3.518000E-01,3.508000E-01,3.444000E-01,3.427000E-01,3.387000E-01,3.351000E-01,3.344000E-01,3.305000E-01,3.271000E-01,3.261000E-01,3.216000E-01,3.183000E-01,3.153000E-01,3.150000E-01,3.099000E-01,3.087000E-01,3.037000E-01,3.007000E-01,2.985000E-01,2.961000E-01,2.954000E-01,2.948000E-01,2.943000E-01,2.930000E-01,2.923000E-01,2.917000E-01,2.898000E-01,2.888000E-01,2.883000E-01,2.866000E-01,2.846000E-01,2.839000E-01,2.837000E-01,2.838000E-01,2.834000E-01,2.838000E-01,2.836000E-01,2.850000E-01,2.854000E-01,2.868000E-01,2.873000E-01,2.878000E-01,2.879000E-01,2.888000E-01,2.901000E-01,2.907000E-01,2.925000E-01,2.935000E-01,2.943000E-01,2.958000E-01,2.968000E-01,2.974000E-01,2.981000E-01,2.983000E-01,2.994000E-01,3.003000E-01,3.015000E-01,3.037000E-01,3.047000E-01,3.068000E-01,3.078000E-01,3.087000E-01,3.100000E-01,3.103000E-01]
				self.b[i] = [1.546000E-03,1.602000E-03,1.675000E-03,1.731000E-03,1.805000E-03,1.885000E-03,2.003000E-03,2.192000E-03,2.340000E-03,2.320000E-03,2.273000E-03,2.264000E-03,2.247000E-03,2.201000E-03,2.114000E-03,1.935000E-03,1.721000E-03,1.662000E-03,1.572000E-03,1.396000E-03,1.200000E-03,9.805000E-04,8.553000E-04,7.865000E-04,6.959000E-04,5.713000E-04,5.235000E-04,3.829000E-04,3.197000E-04,2.541000E-04,1.893000E-04,1.743000E-04,4.406000E-05,-8.638000E-05,-1.098000E-04,-2.454000E-04,-2.815000E-04,-3.171000E-04,-3.647000E-04,-3.680000E-04,-3.434000E-04,-3.263000E-04,-3.008000E-04,-2.962000E-04,-2.952000E-04,-3.028000E-04,-3.172000E-04,-3.224000E-04,-3.611000E-04,-3.837000E-04,-3.900000E-04,-4.151000E-04,-4.271000E-04,-4.684000E-04,-5.227000E-04,-5.476000E-04,-5.946000E-04,-6.572000E-04,-6.731000E-04,-7.291000E-04,-7.399000E-04,-7.642000E-04,-7.781000E-04,-7.858000E-04,-7.913000E-04,-8.462000E-04,-9.111000E-04,-9.293000E-04,-9.619000E-04,-9.795000E-04,-9.846000E-04,-1.005000E-03,-1.008000E-03,-1.000000E-03,-9.567000E-04,-9.458000E-04,-8.899000E-04,-8.788000E-04,-8.603000E-04,-8.333000E-04,-8.257000E-04,-7.758000E-04,-7.570000E-04,-7.091000E-04,-6.839000E-04,-6.855000E-04,-7.052000E-04,-7.067000E-04,-7.102000E-04,-7.100000E-04,-7.286000E-04,-7.342000E-04,-7.343000E-04,-7.255000E-04,-7.206000E-04,-6.930000E-04,-6.888000E-04,-6.872000E-04,-7.653000E-04,-8.808000E-04,-9.679000E-04,-9.972000E-04,-1.070000E-03,-1.103000E-03,-1.114000E-03,-1.141000E-03,-1.173000E-03,-1.176000E-03,-1.186000E-03,-1.204000E-03,-1.215000E-03,-1.247000E-03,-1.254000E-03,-1.265000E-03,-1.289000E-03,-1.313000E-03,-1.333000E-03,-1.331000E-03,-1.334000E-03,-1.322000E-03,-1.314000E-03,-1.295000E-03,-1.292000E-03,-1.304000E-03,-1.312000E-03,-1.325000E-03,-1.330000E-03,-1.333000E-03,-1.332000E-03,-1.327000E-03,-1.308000E-03,-1.301000E-03,-1.295000E-03,-1.280000E-03,-1.257000E-03,-1.213000E-03,-1.178000E-03,-1.048000E-03,-9.667000E-04,-9.475000E-04,-9.345000E-04,-9.250000E-04,-9.223000E-04]
				self.c1[i] = [-2.632000E+00,-2.549000E+00,-2.446000E+00,-2.318000E+00,-2.141000E+00,-2.016000E+00,-1.854000E+00,-1.671000E+00,-1.475000E+00,-1.411000E+00,-1.352000E+00,-1.285000E+00,-1.263000E+00,-1.222000E+00,-1.168000E+00,-1.111000E+00,-1.062000E+00,-1.030000E+00,-9.964000E-01,-9.180000E-01,-8.459000E-01,-7.708000E-01,-7.044000E-01,-6.670000E-01,-6.019000E-01,-5.508000E-01,-5.280000E-01,-4.443000E-01,-4.006000E-01,-3.530000E-01,-2.699000E-01,-2.599000E-01,-1.604000E-01,-4.195000E-02,-3.007000E-02,4.509000E-02,7.674000E-02,1.374000E-01,1.908000E-01,2.244000E-01,2.944000E-01,3.246000E-01,3.667000E-01,3.932000E-01,4.064000E-01,4.447000E-01,4.714000E-01,4.807000E-01,5.219000E-01,5.485000E-01,5.688000E-01,6.197000E-01,6.297000E-01,6.637000E-01,6.923000E-01,7.074000E-01,7.585000E-01,8.106000E-01,8.218000E-01,8.691000E-01,8.856000E-01,9.289000E-01,9.433000E-01,9.856000E-01,9.938000E-01,1.035000E+00,1.084000E+00,1.101000E+00,1.135000E+00,1.170000E+00,1.179000E+00,1.221000E+00,1.237000E+00,1.259000E+00,1.303000E+00,1.314000E+00,1.359000E+00,1.369000E+00,1.401000E+00,1.432000E+00,1.440000E+00,1.493000E+00,1.506000E+00,1.538000E+00,1.569000E+00,1.576000E+00,1.615000E+00,1.645000E+00,1.654000E+00,1.691000E+00,1.721000E+00,1.750000E+00,1.752000E+00,1.789000E+00,1.797000E+00,1.828000E+00,1.846000E+00,1.858000E+00,1.875000E+00,1.885000E+00,1.892000E+00,1.896000E+00,1.905000E+00,1.910000E+00,1.913000E+00,1.922000E+00,1.928000E+00,1.930000E+00,1.938000E+00,1.947000E+00,1.951000E+00,1.947000E+00,1.946000E+00,1.942000E+00,1.934000E+00,1.928000E+00,1.911000E+00,1.906000E+00,1.891000E+00,1.882000E+00,1.875000E+00,1.865000E+00,1.851000E+00,1.839000E+00,1.833000E+00,1.816000E+00,1.806000E+00,1.799000E+00,1.784000E+00,1.772000E+00,1.759000E+00,1.746000E+00,1.741000E+00,1.722000E+00,1.706000E+00,1.685000E+00,1.655000E+00,1.621000E+00,1.585000E+00,1.571000E+00,1.560000E+00,1.547000E+00,1.543000E+00]
				self.c2[i] = [-2.453000E+00,-2.370000E+00,-2.267000E+00,-2.143000E+00,-1.973000E+00,-1.854000E+00,-1.700000E+00,-1.529000E+00,-1.352000E+00,-1.292000E+00,-1.234000E+00,-1.167000E+00,-1.145000E+00,-1.104000E+00,-1.045000E+00,-9.850000E-01,-9.343000E-01,-8.996000E-01,-8.649000E-01,-7.849000E-01,-7.083000E-01,-6.274000E-01,-5.523000E-01,-5.105000E-01,-4.387000E-01,-3.839000E-01,-3.599000E-01,-2.702000E-01,-2.233000E-01,-1.736000E-01,-8.733000E-02,-7.703000E-02,3.263000E-02,1.474000E-01,1.580000E-01,2.307000E-01,2.618000E-01,3.194000E-01,3.684000E-01,4.003000E-01,4.701000E-01,5.007000E-01,5.429000E-01,5.698000E-01,5.835000E-01,6.232000E-01,6.504000E-01,6.593000E-01,6.973000E-01,7.218000E-01,7.410000E-01,7.887000E-01,7.979000E-01,8.289000E-01,8.555000E-01,8.696000E-01,9.189000E-01,9.716000E-01,9.827000E-01,1.029000E+00,1.045000E+00,1.085000E+00,1.099000E+00,1.137000E+00,1.144000E+00,1.180000E+00,1.225000E+00,1.240000E+00,1.271000E+00,1.302000E+00,1.311000E+00,1.347000E+00,1.361000E+00,1.380000E+00,1.418000E+00,1.427000E+00,1.467000E+00,1.476000E+00,1.505000E+00,1.533000E+00,1.541000E+00,1.586000E+00,1.597000E+00,1.625000E+00,1.653000E+00,1.659000E+00,1.688000E+00,1.711000E+00,1.718000E+00,1.747000E+00,1.770000E+00,1.790000E+00,1.792000E+00,1.823000E+00,1.830000E+00,1.858000E+00,1.875000E+00,1.887000E+00,1.902000E+00,1.907000E+00,1.911000E+00,1.913000E+00,1.918000E+00,1.920000E+00,1.922000E+00,1.930000E+00,1.933000E+00,1.935000E+00,1.939000E+00,1.943000E+00,1.944000E+00,1.934000E+00,1.932000E+00,1.925000E+00,1.913000E+00,1.907000E+00,1.890000E+00,1.885000E+00,1.869000E+00,1.861000E+00,1.854000E+00,1.845000E+00,1.832000E+00,1.820000E+00,1.815000E+00,1.798000E+00,1.789000E+00,1.782000E+00,1.768000E+00,1.758000E+00,1.748000E+00,1.738000E+00,1.733000E+00,1.718000E+00,1.705000E+00,1.690000E+00,1.666000E+00,1.641000E+00,1.613000E+00,1.602000E+00,1.593000E+00,1.581000E+00,1.579000E+00]
				self.sigma[i] = [4.124000E-01,4.161000E-01,4.196000E-01,4.200000E-01,4.175000E-01,4.160000E-01,4.148000E-01,4.170000E-01,4.210000E-01,4.216000E-01,4.204000E-01,4.188000E-01,4.178000E-01,4.156000E-01,4.112000E-01,4.062000E-01,4.011000E-01,4.000000E-01,3.991000E-01,3.952000E-01,3.941000E-01,3.898000E-01,3.866000E-01,3.863000E-01,3.870000E-01,3.872000E-01,3.869000E-01,3.841000E-01,3.811000E-01,3.783000E-01,3.745000E-01,3.739000E-01,3.705000E-01,3.692000E-01,3.693000E-01,3.695000E-01,3.690000E-01,3.672000E-01,3.659000E-01,3.650000E-01,3.631000E-01,3.622000E-01,3.611000E-01,3.604000E-01,3.600000E-01,3.589000E-01,3.578000E-01,3.573000E-01,3.563000E-01,3.558000E-01,3.553000E-01,3.542000E-01,3.540000E-01,3.534000E-01,3.535000E-01,3.533000E-01,3.521000E-01,3.513000E-01,3.512000E-01,3.507000E-01,3.503000E-01,3.495000E-01,3.492000E-01,3.486000E-01,3.484000E-01,3.470000E-01,3.462000E-01,3.460000E-01,3.454000E-01,3.445000E-01,3.443000E-01,3.436000E-01,3.434000E-01,3.434000E-01,3.435000E-01,3.434000E-01,3.421000E-01,3.417000E-01,3.403000E-01,3.390000E-01,3.386000E-01,3.366000E-01,3.360000E-01,3.343000E-01,3.328000E-01,3.324000E-01,3.292000E-01,3.265000E-01,3.256000E-01,3.231000E-01,3.214000E-01,3.207000E-01,3.206000E-01,3.195000E-01,3.191000E-01,3.174000E-01,3.160000E-01,3.149000E-01,3.135000E-01,3.125000E-01,3.118000E-01,3.114000E-01,3.104000E-01,3.097000E-01,3.093000E-01,3.080000E-01,3.069000E-01,3.065000E-01,3.055000E-01,3.041000E-01,3.038000E-01,3.020000E-01,3.017000E-01,3.007000E-01,2.990000E-01,2.989000E-01,2.985000E-01,2.986000E-01,2.991000E-01,2.996000E-01,2.999000E-01,3.001000E-01,2.998000E-01,2.997000E-01,2.997000E-01,2.991000E-01,2.991000E-01,2.992000E-01,2.996000E-01,2.998000E-01,2.998000E-01,2.998000E-01,2.996000E-01,2.988000E-01,2.979000E-01,2.971000E-01,2.969000E-01,2.954000E-01,2.940000E-01,2.932000E-01,2.928000E-01,2.924000E-01,2.923000E-01]
			elif i == 3:
				self.a[i] = [5.925000E-01,5.929000E-01,5.929000E-01,5.917000E-01,5.874000E-01,5.833000E-01,5.775000E-01,5.716000E-01,5.650000E-01,5.630000E-01,5.608000E-01,5.581000E-01,5.572000E-01,5.561000E-01,5.544000E-01,5.536000E-01,5.541000E-01,5.539000E-01,5.533000E-01,5.500000E-01,5.483000E-01,5.457000E-01,5.428000E-01,5.401000E-01,5.358000E-01,5.322000E-01,5.307000E-01,5.248000E-01,5.216000E-01,5.184000E-01,5.126000E-01,5.117000E-01,5.032000E-01,4.932000E-01,4.922000E-01,4.866000E-01,4.840000E-01,4.789000E-01,4.742000E-01,4.712000E-01,4.647000E-01,4.620000E-01,4.580000E-01,4.554000E-01,4.540000E-01,4.497000E-01,4.466000E-01,4.456000E-01,4.418000E-01,4.395000E-01,4.378000E-01,4.337000E-01,4.328000E-01,4.292000E-01,4.259000E-01,4.245000E-01,4.196000E-01,4.148000E-01,4.138000E-01,4.097000E-01,4.083000E-01,4.044000E-01,4.031000E-01,3.993000E-01,3.985000E-01,3.943000E-01,3.901000E-01,3.887000E-01,3.858000E-01,3.829000E-01,3.820000E-01,3.783000E-01,3.769000E-01,3.748000E-01,3.706000E-01,3.696000E-01,3.645000E-01,3.635000E-01,3.601000E-01,3.565000E-01,3.557000E-01,3.498000E-01,3.484000E-01,3.445000E-01,3.405000E-01,3.398000E-01,3.348000E-01,3.314000E-01,3.303000E-01,3.267000E-01,3.234000E-01,3.195000E-01,3.193000E-01,3.153000E-01,3.144000E-01,3.114000E-01,3.093000E-01,3.078000E-01,3.051000E-01,3.033000E-01,3.024000E-01,3.019000E-01,3.007000E-01,3.001000E-01,2.996000E-01,2.984000E-01,2.972000E-01,2.970000E-01,2.959000E-01,2.942000E-01,2.936000E-01,2.930000E-01,2.930000E-01,2.927000E-01,2.925000E-01,2.922000E-01,2.926000E-01,2.927000E-01,2.932000E-01,2.936000E-01,2.939000E-01,2.947000E-01,2.954000E-01,2.959000E-01,2.962000E-01,2.970000E-01,2.975000E-01,2.980000E-01,2.990000E-01,2.997000E-01,3.007000E-01,3.015000E-01,3.018000E-01,3.027000E-01,3.032000E-01,3.039000E-01,3.048000E-01,3.060000E-01,3.074000E-01,3.080000E-01,3.086000E-01,3.095000E-01,3.097000E-01]
				self.b[i] = [1.525000E-03,1.556000E-03,1.609000E-03,1.661000E-03,1.706000E-03,1.760000E-03,1.833000E-03,1.893000E-03,1.958000E-03,1.949000E-03,1.911000E-03,1.891000E-03,1.878000E-03,1.828000E-03,1.738000E-03,1.596000E-03,1.429000E-03,1.368000E-03,1.320000E-03,1.199000E-03,1.015000E-03,8.037000E-04,6.527000E-04,5.784000E-04,4.681000E-04,3.901000E-04,3.543000E-04,2.194000E-04,1.612000E-04,1.020000E-04,-3.932000E-06,-1.064000E-05,-1.151000E-04,-2.150000E-04,-2.291000E-04,-3.079000E-04,-3.132000E-04,-3.247000E-04,-3.494000E-04,-3.676000E-04,-3.968000E-04,-4.047000E-04,-4.174000E-04,-4.263000E-04,-4.309000E-04,-4.296000E-04,-4.326000E-04,-4.349000E-04,-4.512000E-04,-4.615000E-04,-4.701000E-04,-5.022000E-04,-5.113000E-04,-5.323000E-04,-5.484000E-04,-5.598000E-04,-5.875000E-04,-6.383000E-04,-6.494000E-04,-7.040000E-04,-7.219000E-04,-7.723000E-04,-7.889000E-04,-8.265000E-04,-8.347000E-04,-8.677000E-04,-8.879000E-04,-8.934000E-04,-9.025000E-04,-9.195000E-04,-9.250000E-04,-9.438000E-04,-9.515000E-04,-9.537000E-04,-9.484000E-04,-9.463000E-04,-9.226000E-04,-9.164000E-04,-8.986000E-04,-8.742000E-04,-8.699000E-04,-8.203000E-04,-8.095000E-04,-7.856000E-04,-7.709000E-04,-7.649000E-04,-7.287000E-04,-7.225000E-04,-7.208000E-04,-7.181000E-04,-7.165000E-04,-7.081000E-04,-7.079000E-04,-7.010000E-04,-6.995000E-04,-7.030000E-04,-7.024000E-04,-7.108000E-04,-7.266000E-04,-7.561000E-04,-7.895000E-04,-8.025000E-04,-8.364000E-04,-8.560000E-04,-8.729000E-04,-9.003000E-04,-9.194000E-04,-9.265000E-04,-9.532000E-04,-9.686000E-04,-9.738000E-04,-1.004000E-03,-1.010000E-03,-1.023000E-03,-1.044000E-03,-1.063000E-03,-1.072000E-03,-1.074000E-03,-1.081000E-03,-1.086000E-03,-1.090000E-03,-1.099000E-03,-1.107000E-03,-1.109000E-03,-1.109000E-03,-1.104000E-03,-1.104000E-03,-1.105000E-03,-1.106000E-03,-1.108000E-03,-1.106000E-03,-1.103000E-03,-1.099000E-03,-1.091000E-03,-1.076000E-03,-1.061000E-03,-1.035000E-03,-9.816000E-04,-9.480000E-04,-9.331000E-04,-9.262000E-04,-9.145000E-04,-9.120000E-04]
				self.c1[i] = [-2.586000E+00,-2.494000E+00,-2.383000E+00,-2.248000E+00,-2.068000E+00,-1.957000E+00,-1.826000E+00,-1.681000E+00,-1.524000E+00,-1.465000E+00,-1.403000E+00,-1.341000E+00,-1.319000E+00,-1.281000E+00,-1.222000E+00,-1.166000E+00,-1.112000E+00,-1.085000E+00,-1.053000E+00,-9.764000E-01,-9.003000E-01,-8.119000E-01,-7.450000E-01,-7.032000E-01,-6.376000E-01,-5.869000E-01,-5.660000E-01,-4.847000E-01,-4.423000E-01,-3.997000E-01,-3.204000E-01,-3.097000E-01,-2.044000E-01,-8.749000E-02,-7.589000E-02,-7.736000E-03,1.854000E-02,7.174000E-02,1.223000E-01,1.546000E-01,2.192000E-01,2.457000E-01,2.837000E-01,3.090000E-01,3.216000E-01,3.611000E-01,3.911000E-01,4.009000E-01,4.408000E-01,4.648000E-01,4.821000E-01,5.253000E-01,5.350000E-01,5.720000E-01,6.043000E-01,6.187000E-01,6.668000E-01,7.153000E-01,7.249000E-01,7.667000E-01,7.810000E-01,8.198000E-01,8.334000E-01,8.720000E-01,8.803000E-01,9.236000E-01,9.655000E-01,9.785000E-01,1.006000E+00,1.035000E+00,1.043000E+00,1.080000E+00,1.094000E+00,1.113000E+00,1.153000E+00,1.163000E+00,1.207000E+00,1.216000E+00,1.247000E+00,1.278000E+00,1.285000E+00,1.334000E+00,1.346000E+00,1.379000E+00,1.414000E+00,1.420000E+00,1.461000E+00,1.489000E+00,1.498000E+00,1.529000E+00,1.557000E+00,1.590000E+00,1.591000E+00,1.622000E+00,1.629000E+00,1.652000E+00,1.666000E+00,1.677000E+00,1.697000E+00,1.711000E+00,1.719000E+00,1.723000E+00,1.733000E+00,1.738000E+00,1.741000E+00,1.750000E+00,1.757000E+00,1.759000E+00,1.766000E+00,1.776000E+00,1.779000E+00,1.781000E+00,1.781000E+00,1.781000E+00,1.778000E+00,1.778000E+00,1.771000E+00,1.769000E+00,1.763000E+00,1.759000E+00,1.755000E+00,1.746000E+00,1.736000E+00,1.730000E+00,1.727000E+00,1.717000E+00,1.712000E+00,1.708000E+00,1.698000E+00,1.691000E+00,1.680000E+00,1.670000E+00,1.665000E+00,1.653000E+00,1.643000E+00,1.631000E+00,1.614000E+00,1.591000E+00,1.571000E+00,1.563000E+00,1.556000E+00,1.546000E+00,1.543000E+00]
				self.c2[i] = [-2.412000E+00,-2.320000E+00,-2.211000E+00,-2.080000E+00,-1.907000E+00,-1.801000E+00,-1.675000E+00,-1.538000E+00,-1.391000E+00,-1.333000E+00,-1.271000E+00,-1.209000E+00,-1.188000E+00,-1.150000E+00,-1.089000E+00,-1.031000E+00,-9.757000E-01,-9.464000E-01,-9.122000E-01,-8.307000E-01,-7.531000E-01,-6.623000E-01,-5.906000E-01,-5.457000E-01,-4.751000E-01,-4.215000E-01,-3.992000E-01,-3.136000E-01,-2.688000E-01,-2.241000E-01,-1.401000E-01,-1.286000E-01,-1.845000E-02,9.790000E-02,1.092000E-01,1.757000E-01,2.011000E-01,2.524000E-01,3.005000E-01,3.317000E-01,3.951000E-01,4.215000E-01,4.591000E-01,4.841000E-01,4.965000E-01,5.359000E-01,5.652000E-01,5.746000E-01,6.124000E-01,6.350000E-01,6.512000E-01,6.910000E-01,7.000000E-01,7.342000E-01,7.645000E-01,7.781000E-01,8.243000E-01,8.710000E-01,8.802000E-01,9.200000E-01,9.338000E-01,9.707000E-01,9.835000E-01,1.020000E+00,1.027000E+00,1.067000E+00,1.105000E+00,1.117000E+00,1.142000E+00,1.167000E+00,1.175000E+00,1.207000E+00,1.219000E+00,1.236000E+00,1.272000E+00,1.280000E+00,1.320000E+00,1.328000E+00,1.355000E+00,1.383000E+00,1.389000E+00,1.433000E+00,1.443000E+00,1.472000E+00,1.501000E+00,1.507000E+00,1.540000E+00,1.565000E+00,1.572000E+00,1.598000E+00,1.621000E+00,1.648000E+00,1.649000E+00,1.676000E+00,1.681000E+00,1.700000E+00,1.713000E+00,1.722000E+00,1.738000E+00,1.749000E+00,1.754000E+00,1.757000E+00,1.764000E+00,1.767000E+00,1.769000E+00,1.775000E+00,1.781000E+00,1.782000E+00,1.787000E+00,1.794000E+00,1.795000E+00,1.794000E+00,1.793000E+00,1.790000E+00,1.785000E+00,1.783000E+00,1.775000E+00,1.773000E+00,1.766000E+00,1.762000E+00,1.758000E+00,1.749000E+00,1.740000E+00,1.734000E+00,1.731000E+00,1.722000E+00,1.717000E+00,1.713000E+00,1.704000E+00,1.698000E+00,1.688000E+00,1.679000E+00,1.675000E+00,1.665000E+00,1.658000E+00,1.649000E+00,1.636000E+00,1.618000E+00,1.602000E+00,1.596000E+00,1.589000E+00,1.581000E+00,1.579000E+00]
				self.sigma[i] = [4.057000E-01,4.070000E-01,4.085000E-01,4.083000E-01,4.057000E-01,4.047000E-01,4.042000E-01,4.045000E-01,4.044000E-01,4.036000E-01,4.021000E-01,4.006000E-01,4.000000E-01,3.985000E-01,3.951000E-01,3.915000E-01,3.885000E-01,3.874000E-01,3.865000E-01,3.847000E-01,3.838000E-01,3.815000E-01,3.796000E-01,3.789000E-01,3.788000E-01,3.785000E-01,3.783000E-01,3.762000E-01,3.747000E-01,3.729000E-01,3.696000E-01,3.692000E-01,3.660000E-01,3.634000E-01,3.631000E-01,3.616000E-01,3.611000E-01,3.600000E-01,3.588000E-01,3.580000E-01,3.560000E-01,3.552000E-01,3.542000E-01,3.535000E-01,3.532000E-01,3.522000E-01,3.515000E-01,3.513000E-01,3.502000E-01,3.495000E-01,3.492000E-01,3.484000E-01,3.482000E-01,3.474000E-01,3.466000E-01,3.462000E-01,3.453000E-01,3.446000E-01,3.445000E-01,3.439000E-01,3.437000E-01,3.430000E-01,3.428000E-01,3.421000E-01,3.419000E-01,3.411000E-01,3.401000E-01,3.398000E-01,3.394000E-01,3.390000E-01,3.389000E-01,3.384000E-01,3.382000E-01,3.377000E-01,3.366000E-01,3.363000E-01,3.350000E-01,3.347000E-01,3.339000E-01,3.329000E-01,3.327000E-01,3.306000E-01,3.300000E-01,3.286000E-01,3.270000E-01,3.267000E-01,3.244000E-01,3.224000E-01,3.218000E-01,3.193000E-01,3.172000E-01,3.150000E-01,3.148000E-01,3.132000E-01,3.129000E-01,3.117000E-01,3.107000E-01,3.101000E-01,3.087000E-01,3.073000E-01,3.066000E-01,3.063000E-01,3.051000E-01,3.044000E-01,3.039000E-01,3.028000E-01,3.020000E-01,3.018000E-01,3.011000E-01,3.002000E-01,2.997000E-01,2.986000E-01,2.984000E-01,2.975000E-01,2.968000E-01,2.961000E-01,2.956000E-01,2.955000E-01,2.950000E-01,2.947000E-01,2.947000E-01,2.946000E-01,2.944000E-01,2.943000E-01,2.943000E-01,2.944000E-01,2.945000E-01,2.946000E-01,2.947000E-01,2.948000E-01,2.949000E-01,2.949000E-01,2.947000E-01,2.944000E-01,2.941000E-01,2.938000E-01,2.937000E-01,2.933000E-01,2.929000E-01,2.928000E-01,2.926000E-01,2.924000E-01,2.923000E-01]

		## Unit conversion
		## Original unit is cm/s2 (gal)
		self.imt_scaling = {"g": 0.01/g, "mg": 10./g, "ms2": 1E-2, "gal": 1.0, "cms2": 1.0}

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		Return ground motion for given magnitude, distance, depth, soil type,
		and fault mechanism.

		:param M:
			Float or float array, magnitude(s).
			Note that if d is an array, M must be a float.
		:param d:
			Float or float array, distance(s) in km.
			Note that if M is an array, d must be a float.
		:param h:
			Float, focal depth in km (default: 0., i.e. assume d is hypocentral
			distance).
		:param imt:
			String, one of the supported intensity measure types: "PGA" or "SA"
			(default: "PGA").
		:param T:
			Float, spectral period of considered IMT. Ignored if IMT == "PGA"
			(default: 0).
		:param imt_unit:
			String, unit in which intensities should be expressed (default: "g")
		:param epsilon:
			Float, number of standard deviations above or below the mean
			(default: 0).
		:param soil_type:
			String, either "rock" or "alluvium" (default: "rock"):
				Rock: VS >= 800 m/s
				Alluvium: 300 <= VS < 800 m/s
		:param vs30:
			Float, shear-wave velocity in the upper 30 m (in m/s). If not None,
			it takes precedence over the soil_type parameter (default: None).
		:param kappa:
			Float, kappa value, in seconds. Ignored in this GMPE (default: None)
		:param mechanism:
			String, fault mechanism: either "normal", "reverse" or "strike-slip".
			Ignored in this GMPE.
		:param damping:
			Float, damping in percent. Supported value are 5, 7, 10, and 20.

		:return:
			Returns a float array with ground-motion intensities
		"""
		scale_factor = self.imt_scaling[imt_unit.lower()]

		imt = imt.upper()
		if imt == "PGA":
			imt = "SA"
			T = self.imt_periods[imt][-1]
			damping_index = 0
			## Note: slicing below [-1:] is necessary to ensure result is array!
			A = self.a[damping_index][-1:]
			B = self.b[damping_index][-1:]
			C1 = self.c1[damping_index][-1:]
			C2 = self.c2[damping_index][-1:]
			S = self.sigma[damping_index][-1:]
		else:
			if not self.has_imt(imt):
				raise IMTUndefinedError(imt)

			if T < self.Tmin(imt) or T > self.Tmax(imt):
				raise PeriodUndefinedError(imt, T)
				#return None

			try:
				damping_index = self.dampings.index(damping)
			except:
				raise DampingNotSupportedError(damping)
			else:
				a = self.a[damping_index]
				b = self.b[damping_index]
				c1 = self.c1[damping_index]
				c2 = self.c2[damping_index]
				sigma = self.sigma[damping_index]
				sa_periods = self.imt_periods["SA"]
				A = interpolate(sa_periods, a, [T])
				B = interpolate(sa_periods, b, [T])
				C1 = interpolate(sa_periods, c1, [T])
				C2 = interpolate(sa_periods, c2, [T])
				S = interpolate(sa_periods, sigma, [T])

		if vs30 != None:
			if vs30 >= 800:
				soil_type = "rock"
			elif 300 <= vs30 < 800:
				soil_type = "alluvium"
			else:
				raise VS30OutOfRangeError(vs30)

		if soil_type.lower() == "alluvium":
			C = C2
		else:
			# default is rock
			C = C1

		r = np.sqrt(d*d + h*h)
		log_ah = A*M + B*r - np.log10(r) + C + epsilon * S
		ah = (10**log_ah)
		ah *= scale_factor

		return ah

	def log_sigma(self, M=5., d=10., h=0., imt="PGA", T=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
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
			Float, focal depth in km (default: 0., i.e. assume d is hypocentral
			distance).
		:param imt:
			String, one of the supported intensity measure types: "PGA" or "SA"
			(default: "PGA").
		:param T:
			Float, spectral period of considered IMT. Ignored if IMT == "PGA"
			(default: 0).
		:param soil_type:
			String, either "rock" or "alluvium" (default: "rock"):
				Rock: VS >= 800 m/s
				Alluvium: 300 <= VS < 800 m/s
		:param vs30:
			Float, shear-wave velocity in the upper 30 m (in m/s). If not None,
			it takes precedence over the soil_type parameter (default: None).
		:param kappa:
			Float, kappa value, in seconds. Ignored in this GMPE (default: None)
		:param mechanism:
			String, fault mechanism: either "normal", "reverse" or "strike-slip".
			Ignored in this GMPE.
		:param damping:
			Float, damping in percent. Supported value are 5, 7, 10, and 20.
		"""
		imt = imt.upper()
		if imt == "PGA":
			damping_index = 0
			return self.sigma[damping_index][-1]
		else:
			if not self.has_imt(imt):
				raise IMTUndefinedError(imt)
			if T < self.Tmin(imt) or T > self.Tmax(imt):
				raise PeriodUndefinedError(imt, T)
			try:
				damping_index = self.dampings.index(damping)
			except:
				raise DampingNotSupportedError(damping)
			else:
				sigma = self.sigma[damping_index]
				return interpolate(self.imt_periods["SA"], sigma, [T])[0]

	def is_rake_dependent(self):
		"""
		Indicate whether or not GMPE depends on rake of the source
		"""
		return False

	def get_CRISIS_periods(self):
		"""
		Return array of max. 40 spectral periods to be used with CRISIS
		"""
		freqs = np.array([0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,10.0,15.0,20.0,25.0,30.0,34.0])
		periods = 1.0 / freqs
		periods = periods[::-1]
		return periods

	def plot_Figure3(self, soil_type="rock"):
		"""
		Plot Figure 3 in the paper of Berge-Thierry et al. (2003)

		:param soil_type:
			String, either "rock" or "alluvium" (default: "rock").
		"""
		self.plot_spectrum(mags=[6.5], d=20, plot_style="loglog", plot_freq=True, Tmin=1E-2, Tmax=10, amin=1E-2, amax=10, include_pgm=False, soil_type=soil_type, want_minor_grid=True)

	def plot_Figure7(self, soil_type="rock"):
		"""
		Plot Figure 7 in the paper of Berge-Thierry et al. (2003)

		:param soil_type:
			String, either "rock" or "alluvium" (default: "rock").
		"""
		self.plot_spectrum(mags=[5.5], d=20, plot_style="loglog", plot_freq=True, Tmin=1E-2, Tmax=10, amin=1E-2, amax=10, include_pgm=False, soil_type=soil_type, want_minor_grid=True)

	def plot_Figure10(self):
		"""
		Plot Figure 10 in the paper of Berge-Thierry et al. (2003)
		"""
		self.plot_spectrum(mags=[6.], d=30, plot_style="loglog", plot_freq=True, Tmin=1E-2, Tmax=10, epsilon=1, amin=1E-4, amax=5, include_pgm=False, soil_type="rock", want_minor_grid=True)

	def plot_Figure11(self):
		"""
		Plot Figure 11 in the paper of Berge-Thierry et al. (2003)
		"""
		self.plot_spectrum(mags=[6.], d=30, plot_style="loglog", plot_freq=True, Tmin=1E-2, Tmax=10, epsilon=1, amin=1E-4, amax=5, include_pgm=False, soil_type="alluvium", want_minor_grid=True)


class CauzziFaccioli2008GMPE(GMPE):
	"""
	Cauzzi & Faccioli (2008)
		Magnitude scale: MW
		Magnitude range: 5.0 - 7.2
		Distance metric: Hypocentral
		Distance range: 6 - 150 km
		Intensity measure types: PGA, PSD
		Original IMT unit: cm, m/s2
		PSD period range: 0.05 - 20 s
		Dampings for SD: 5, 10, 20, and 30%
		Soil classes:
			type A (rock): vs30 >= 800 m/s
			type B: 360 <= vs30 < 800 m/s
			type C: 180 <= vs30 < 360 m/s
			type D: vs30 < 180 m/s
		Fault types: normal, reverse, strike-slip

		Note: It is advised to use rake-dependency carefully, based on the
		variation observed in sigma with respect to unspecified mechanism.

		Note: A call to this GMPE takes 1 extra parameter 'explicit_vs30'.

		Note: In Nhlib distances are clipped at 15 km (as per Ezio Faccioli's
		personal communication.)
	"""
	def __init__(self):
		imt_periods = {}
		imt_periods["PGA"] = np.array([0.])
		imt_periods["SD"] = np.arange(0.05, 20.01, 0.05)
		imt_periods["SA"] = imt_periods["SD"]
		distance_metric = "Hypocentral"
		Mmin, Mmax = 5., 7.2
		dmin, dmax = 6., 150.
		Mtype = "MW"
		dampings = [5]
		name = "Cauzzi&Faccioli2008"
		short_name = "CF_2008"
		GMPE.__init__(self, imt_periods, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings, name, short_name)

		## Coefficients
		self.a1 = {}
		self.a1["PGA"] = np.array([-1.296])
		self.a1["SD"] = np.array([-2.885,-1.908,-1.991,-2.186,-2.299,-2.320,-2.353,-2.463,-2.497,-2.541,-2.553,-2.611,-2.599,-2.626,-2.697,-2.739,-2.791,-2.797,-2.831,-2.863,-2.868,-2.894,-2.936,-2.953,-2.978,-3.005,-3.031,-3.047,-3.065,-3.090,-3.117,-3.137,-3.143,-3.158,-3.165,-3.163,-3.162,-3.152,-3.143,-3.142,-3.147,-3.161,-3.176,-3.187,-3.193,-3.196,-3.206,-3.212,-3.217,-3.221,-3.224,-3.236,-3.260,-3.283,-3.303,-3.316,-3.333,-3.351,-3.369,-3.388,-3.409,-3.430,-3.445,-3.457,-3.471,-3.486,-3.502,-3.520,-3.541,-3.560,-3.585,-3.614,-3.642,-3.667,-3.693,-3.719,-3.746,-3.772,-3.795,-3.815,-3.834,-3.853,-3.872,-3.889,-3.906,-3.925,-3.945,-3.963,-3.982,-4.000,-4.017,-4.035,-4.052,-4.071,-4.092,-4.111,-4.126,-4.144,-4.161,-4.180,-4.201,-4.222,-4.243,-4.263,-4.281,-4.298,-4.314,-4.329,-4.341,-4.353,-4.363,-4.376,-4.388,-4.401,-4.413,-4.426,-4.439,-4.450,-4.459,-4.468,-4.476,-4.486,-4.495,-4.505,-4.514,-4.523,-4.533,-4.543,-4.554,-4.565,-4.576,-4.587,-4.597,-4.606,-4.614,-4.622,-4.631,-4.640,-4.648,-4.656,-4.665,-4.672,-4.679,-4.685,-4.692,-4.697,-4.703,-4.708,-4.713,-4.718,-4.724,-4.728,-4.733,-4.737,-4.742,-4.746,-4.750,-4.753,-4.758,-4.763,-4.767,-4.770,-4.773,-4.776,-4.780,-4.783,-4.787,-4.789,-4.792,-4.794,-4.797,-4.800,-4.803,-4.806,-4.809,-4.812,-4.814,-4.817,-4.819,-4.822,-4.824,-4.827,-4.830,-4.833,-4.835,-4.837,-4.839,-4.840,-4.840,-4.841,-4.841,-4.841,-4.842,-4.843,-4.845,-4.846,-4.847,-4.849,-4.850,-4.851,-4.852,-4.853,-4.854,-4.855,-4.855,-4.856,-4.856,-4.856,-4.856,-4.856,-4.856,-4.856,-4.855,-4.854,-4.853,-4.851,-4.849,-4.848,-4.847,-4.846,-4.845,-4.845,-4.844,-4.844,-4.843,-4.843,-4.842,-4.842,-4.841,-4.840,-4.839,-4.839,-4.838,-4.838,-4.838,-4.838,-4.838,-4.837,-4.837,-4.837,-4.836,-4.836,-4.836,-4.835,-4.833,-4.832,-4.831,-4.830,-4.828,-4.827,-4.826,-4.825,-4.823,-4.822,-4.821,-4.820,-4.819,-4.818,-4.817,-4.815,-4.814,-4.812,-4.811,-4.810,-4.808,-4.807,-4.806,-4.805,-4.804,-4.802,-4.801,-4.799,-4.798,-4.796,-4.794,-4.793,-4.791,-4.789,-4.787,-4.784,-4.781,-4.779,-4.776,-4.773,-4.771,-4.769,-4.767,-4.765,-4.763,-4.760,-4.758,-4.755,-4.752,-4.750,-4.747,-4.745,-4.742,-4.740,-4.737,-4.735,-4.733,-4.730,-4.728,-4.725,-4.723,-4.720,-4.718,-4.716,-4.713,-4.711,-4.709,-4.707,-4.705,-4.704,-4.702,-4.700,-4.699,-4.697,-4.695,-4.694,-4.692,-4.691,-4.690,-4.689,-4.688,-4.687,-4.686,-4.684,-4.683,-4.681,-4.680,-4.679,-4.677,-4.676,-4.674,-4.673,-4.672,-4.670,-4.669,-4.668,-4.666,-4.665,-4.663,-4.662,-4.660,-4.659,-4.656,-4.654,-4.652,-4.649,-4.647,-4.645,-4.642,-4.640,-4.637,-4.635,-4.633,-4.630,-4.628,-4.626,-4.624,-4.622,-4.620,-4.618,-4.616,-4.613,-4.611,-4.609,-4.607,-4.605,-4.602,-4.600,-4.598,-4.595,-4.593,-4.591,-4.588,-4.586,-4.584,-4.581,-4.579,-4.576,-4.573,-4.571,-4.568,-4.566,-4.563,-4.560,-4.558,-4.555,-4.553,-4.551,-4.548,-4.546,-4.544,-4.541,-4.539,-4.537,-4.535,-4.533])
		self.a1["SA"] = self.a1["SD"]
		self.a2 = {}
		self.a2["PGA"] = np.array([0.556])
		self.a2["SD"] = np.array([0.524,0.488,0.522,0.558,0.592,0.614,0.630,0.652,0.664,0.674,0.681,0.694,0.700,0.710,0.727,0.740,0.755,0.761,0.769,0.777,0.780,0.788,0.799,0.804,0.809,0.815,0.821,0.827,0.833,0.840,0.848,0.854,0.857,0.861,0.864,0.865,0.867,0.868,0.868,0.869,0.870,0.873,0.876,0.878,0.878,0.879,0.880,0.880,0.880,0.880,0.880,0.881,0.884,0.888,0.890,0.892,0.895,0.897,0.900,0.903,0.905,0.908,0.910,0.912,0.914,0.916,0.919,0.922,0.925,0.927,0.931,0.935,0.939,0.943,0.946,0.950,0.954,0.958,0.961,0.965,0.968,0.971,0.974,0.977,0.980,0.983,0.986,0.988,0.990,0.993,0.995,0.998,1.000,1.003,1.005,1.008,1.009,1.012,1.014,1.017,1.019,1.022,1.025,1.028,1.030,1.032,1.035,1.036,1.038,1.039,1.041,1.043,1.044,1.046,1.047,1.049,1.051,1.052,1.053,1.054,1.055,1.056,1.056,1.057,1.058,1.059,1.060,1.061,1.062,1.063,1.065,1.066,1.067,1.067,1.068,1.069,1.070,1.070,1.071,1.072,1.073,1.073,1.074,1.074,1.075,1.075,1.075,1.076,1.076,1.076,1.077,1.077,1.077,1.077,1.078,1.078,1.078,1.078,1.079,1.079,1.079,1.079,1.079,1.079,1.080,1.080,1.080,1.080,1.081,1.081,1.081,1.082,1.082,1.082,1.083,1.083,1.084,1.084,1.084,1.085,1.085,1.085,1.086,1.086,1.086,1.087,1.087,1.087,1.087,1.087,1.087,1.087,1.087,1.087,1.087,1.087,1.087,1.087,1.087,1.087,1.087,1.088,1.088,1.088,1.088,1.088,1.088,1.088,1.088,1.088,1.088,1.088,1.088,1.088,1.088,1.088,1.088,1.088,1.088,1.087,1.087,1.087,1.087,1.087,1.087,1.087,1.087,1.086,1.086,1.086,1.086,1.086,1.086,1.086,1.086,1.086,1.086,1.086,1.086,1.086,1.086,1.086,1.086,1.086,1.086,1.086,1.085,1.085,1.085,1.085,1.085,1.085,1.085,1.085,1.084,1.084,1.084,1.084,1.084,1.084,1.084,1.084,1.083,1.083,1.083,1.083,1.083,1.083,1.083,1.082,1.082,1.082,1.082,1.082,1.082,1.081,1.081,1.081,1.081,1.081,1.080,1.080,1.080,1.079,1.079,1.079,1.078,1.078,1.078,1.077,1.077,1.077,1.076,1.076,1.075,1.075,1.074,1.074,1.074,1.073,1.073,1.072,1.072,1.072,1.071,1.071,1.071,1.070,1.070,1.070,1.069,1.069,1.069,1.069,1.068,1.068,1.068,1.068,1.067,1.067,1.067,1.067,1.067,1.067,1.067,1.066,1.066,1.066,1.066,1.066,1.066,1.066,1.066,1.065,1.065,1.065,1.065,1.065,1.065,1.065,1.065,1.065,1.064,1.064,1.064,1.064,1.064,1.063,1.063,1.063,1.063,1.062,1.062,1.062,1.062,1.061,1.061,1.061,1.061,1.060,1.060,1.060,1.060,1.060,1.059,1.059,1.059,1.059,1.058,1.058,1.058,1.058,1.057,1.057,1.057,1.057,1.056,1.056,1.056,1.056,1.055,1.055,1.055,1.055,1.054,1.054,1.054,1.054,1.053,1.053,1.053,1.053,1.052,1.052,1.052,1.052,1.051,1.051,1.051,1.051])
		self.a2["SA"] = self.a2["SD"]
		self.a3 = {}
		self.a3["PGA"] = np.array([-1.582])
		self.a3["SD"] = np.array([-1.713,-1.714,-1.601,-1.516,-1.477,-1.461,-1.453,-1.418,-1.407,-1.389,-1.383,-1.368,-1.366,-1.361,-1.361,-1.361,-1.369,-1.376,-1.372,-1.369,-1.366,-1.366,-1.369,-1.366,-1.359,-1.355,-1.357,-1.361,-1.364,-1.367,-1.370,-1.371,-1.370,-1.369,-1.369,-1.370,-1.371,-1.371,-1.372,-1.369,-1.368,-1.367,-1.365,-1.362,-1.356,-1.350,-1.342,-1.335,-1.327,-1.320,-1.313,-1.307,-1.301,-1.296,-1.291,-1.288,-1.284,-1.278,-1.273,-1.267,-1.260,-1.256,-1.252,-1.248,-1.245,-1.242,-1.239,-1.236,-1.233,-1.229,-1.225,-1.221,-1.217,-1.214,-1.211,-1.207,-1.204,-1.200,-1.197,-1.195,-1.192,-1.190,-1.187,-1.184,-1.181,-1.177,-1.174,-1.170,-1.166,-1.162,-1.159,-1.156,-1.154,-1.150,-1.145,-1.141,-1.138,-1.134,-1.130,-1.127,-1.124,-1.120,-1.116,-1.113,-1.109,-1.106,-1.102,-1.099,-1.096,-1.094,-1.091,-1.089,-1.087,-1.084,-1.082,-1.078,-1.075,-1.072,-1.069,-1.065,-1.061,-1.057,-1.054,-1.050,-1.047,-1.043,-1.040,-1.037,-1.034,-1.031,-1.029,-1.026,-1.023,-1.020,-1.017,-1.014,-1.011,-1.008,-1.005,-1.003,-1.000,-0.997,-0.995,-0.992,-0.989,-0.987,-0.985,-0.983,-0.981,-0.979,-0.978,-0.977,-0.975,-0.974,-0.973,-0.972,-0.970,-0.969,-0.967,-0.966,-0.964,-0.963,-0.962,-0.960,-0.959,-0.958,-0.958,-0.957,-0.956,-0.956,-0.955,-0.955,-0.955,-0.955,-0.955,-0.955,-0.955,-0.955,-0.955,-0.955,-0.955,-0.955,-0.954,-0.954,-0.954,-0.954,-0.954,-0.954,-0.954,-0.954,-0.954,-0.954,-0.954,-0.954,-0.954,-0.954,-0.954,-0.954,-0.954,-0.954,-0.954,-0.954,-0.954,-0.954,-0.954,-0.954,-0.954,-0.954,-0.954,-0.955,-0.955,-0.955,-0.956,-0.956,-0.957,-0.957,-0.957,-0.957,-0.957,-0.958,-0.958,-0.958,-0.958,-0.958,-0.958,-0.958,-0.958,-0.958,-0.958,-0.959,-0.959,-0.960,-0.960,-0.961,-0.961,-0.962,-0.962,-0.963,-0.963,-0.964,-0.965,-0.966,-0.966,-0.967,-0.968,-0.969,-0.970,-0.971,-0.972,-0.973,-0.974,-0.974,-0.975,-0.976,-0.977,-0.977,-0.978,-0.979,-0.980,-0.981,-0.982,-0.983,-0.984,-0.985,-0.986,-0.987,-0.987,-0.988,-0.989,-0.990,-0.991,-0.992,-0.993,-0.994,-0.995,-0.996,-0.997,-0.998,-0.999,-0.999,-1.000,-1.001,-1.002,-1.003,-1.004,-1.005,-1.006,-1.006,-1.007,-1.008,-1.009,-1.009,-1.010,-1.011,-1.011,-1.012,-1.012,-1.013,-1.013,-1.014,-1.014,-1.015,-1.016,-1.016,-1.017,-1.017,-1.018,-1.019,-1.019,-1.020,-1.021,-1.021,-1.022,-1.023,-1.023,-1.024,-1.024,-1.025,-1.026,-1.026,-1.027,-1.027,-1.028,-1.029,-1.029,-1.030,-1.031,-1.031,-1.032,-1.033,-1.033,-1.034,-1.035,-1.036,-1.036,-1.037,-1.038,-1.039,-1.040,-1.040,-1.041,-1.042,-1.043,-1.043,-1.044,-1.045,-1.046,-1.047,-1.047,-1.048,-1.049,-1.050,-1.050,-1.051,-1.052,-1.053,-1.054,-1.054,-1.055,-1.056,-1.057,-1.057,-1.058,-1.059,-1.060,-1.060,-1.061,-1.062,-1.063,-1.063,-1.064,-1.065,-1.066,-1.067,-1.068,-1.068,-1.069,-1.070,-1.071,-1.072,-1.073,-1.073,-1.074,-1.075,-1.076,-1.077,-1.078,-1.079,-1.080,-1.081,-1.082,-1.083,-1.083,-1.084,-1.085,-1.086,-1.087,-1.088,-1.089,-1.090])
		self.a3["SA"] = self.a3["SD"]
		self.aB = {}
		self.aB["PGA"] = np.array([0.22])
		self.aB["SD"] = np.array([0.159,0.219,0.252,0.280,0.247,0.203,0.195,0.174,0.177,0.186,0.183,0.179,0.161,0.148,0.145,0.138,0.141,0.145,0.150,0.146,0.145,0.139,0.133,0.128,0.126,0.124,0.125,0.122,0.121,0.117,0.112,0.108,0.101,0.099,0.098,0.095,0.093,0.087,0.083,0.083,0.084,0.084,0.085,0.086,0.084,0.080,0.076,0.075,0.072,0.070,0.069,0.070,0.072,0.073,0.074,0.074,0.076,0.074,0.074,0.073,0.074,0.074,0.076,0.077,0.079,0.080,0.081,0.083,0.085,0.087,0.089,0.091,0.092,0.092,0.094,0.095,0.096,0.095,0.094,0.093,0.091,0.090,0.088,0.086,0.083,0.081,0.080,0.080,0.079,0.078,0.078,0.079,0.079,0.080,0.081,0.082,0.081,0.080,0.080,0.081,0.081,0.081,0.081,0.080,0.080,0.079,0.078,0.077,0.076,0.076,0.076,0.076,0.076,0.077,0.076,0.076,0.075,0.074,0.073,0.073,0.072,0.071,0.071,0.071,0.071,0.071,0.071,0.071,0.072,0.072,0.073,0.074,0.075,0.075,0.075,0.075,0.074,0.074,0.073,0.072,0.071,0.070,0.070,0.069,0.069,0.069,0.069,0.069,0.070,0.070,0.071,0.072,0.072,0.073,0.073,0.073,0.073,0.073,0.073,0.073,0.073,0.073,0.073,0.072,0.072,0.072,0.072,0.072,0.071,0.071,0.071,0.071,0.071,0.070,0.070,0.069,0.069,0.068,0.068,0.068,0.067,0.067,0.067,0.066,0.066,0.066,0.066,0.066,0.065,0.065,0.065,0.065,0.065,0.065,0.065,0.065,0.064,0.065,0.065,0.065,0.065,0.065,0.065,0.064,0.064,0.063,0.063,0.062,0.062,0.061,0.060,0.060,0.059,0.058,0.057,0.056,0.055,0.054,0.053,0.053,0.052,0.051,0.051,0.051,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.051,0.051,0.051,0.052,0.052,0.052,0.053,0.053,0.054,0.055,0.055,0.056,0.056,0.057,0.057,0.058,0.058,0.058,0.059,0.059,0.060,0.060,0.061,0.061,0.062,0.063,0.063,0.064,0.065,0.065,0.066,0.066,0.067,0.067,0.068,0.068,0.068,0.068,0.069,0.069,0.069,0.069,0.069,0.069,0.069,0.070,0.070,0.071,0.071,0.072,0.072,0.073,0.073,0.074,0.074,0.075,0.075,0.076,0.076,0.077,0.077,0.077,0.078,0.078,0.078,0.079,0.079,0.079,0.080,0.080,0.081,0.081,0.081,0.082,0.082,0.083,0.083,0.084,0.084,0.085,0.085,0.085,0.085,0.085,0.086,0.086,0.086,0.086,0.086,0.087,0.087,0.087,0.087,0.087,0.087,0.087,0.087,0.088,0.088,0.088,0.088,0.088,0.088,0.089,0.089,0.089,0.089,0.089,0.090,0.090,0.090,0.090,0.090,0.090,0.090,0.090,0.090,0.090,0.090,0.090,0.090,0.090,0.090,0.090,0.090,0.090,0.090,0.090,0.090,0.090,0.091,0.091,0.091,0.091,0.091,0.091,0.091,0.092,0.092,0.092,0.092,0.092,0.092,0.093,0.093,0.093,0.093,0.093,0.093,0.094,0.094,0.094,0.094,0.094,0.095,0.095,0.095,0.095,0.095,0.095,0.096,0.096])
		self.aB["SA"] = self.aB["SD"]
		self.aC = {}
		self.aC["PGA"] = np.array([0.304])
		self.aC["SD"] = np.array([0.191,0.218,0.315,0.421,0.435,0.433,0.457,0.446,0.447,0.448,0.451,0.445,0.433,0.425,0.416,0.404,0.403,0.408,0.413,0.407,0.402,0.390,0.381,0.372,0.369,0.367,0.371,0.366,0.361,0.357,0.351,0.342,0.332,0.329,0.327,0.320,0.314,0.304,0.298,0.295,0.294,0.293,0.294,0.293,0.291,0.285,0.279,0.273,0.267,0.262,0.261,0.260,0.259,0.258,0.257,0.255,0.253,0.249,0.246,0.243,0.241,0.239,0.238,0.237,0.235,0.234,0.233,0.233,0.234,0.234,0.235,0.235,0.236,0.236,0.238,0.239,0.240,0.239,0.237,0.235,0.232,0.228,0.225,0.222,0.217,0.215,0.213,0.212,0.211,0.210,0.210,0.209,0.209,0.210,0.210,0.211,0.210,0.209,0.209,0.209,0.210,0.210,0.210,0.209,0.208,0.208,0.207,0.206,0.206,0.206,0.206,0.206,0.206,0.206,0.205,0.205,0.204,0.202,0.201,0.200,0.199,0.198,0.197,0.196,0.196,0.195,0.194,0.193,0.193,0.192,0.192,0.192,0.192,0.192,0.192,0.191,0.191,0.190,0.189,0.188,0.188,0.187,0.186,0.186,0.186,0.187,0.187,0.188,0.189,0.189,0.190,0.192,0.193,0.193,0.194,0.194,0.195,0.195,0.196,0.196,0.196,0.195,0.195,0.195,0.194,0.194,0.194,0.193,0.193,0.192,0.192,0.191,0.191,0.190,0.190,0.189,0.188,0.188,0.187,0.186,0.186,0.185,0.185,0.184,0.184,0.184,0.183,0.183,0.183,0.182,0.182,0.182,0.181,0.181,0.181,0.181,0.181,0.180,0.180,0.180,0.180,0.180,0.180,0.179,0.179,0.178,0.177,0.177,0.176,0.175,0.174,0.173,0.173,0.172,0.170,0.170,0.169,0.168,0.167,0.167,0.166,0.166,0.166,0.165,0.165,0.165,0.165,0.165,0.165,0.165,0.164,0.164,0.164,0.164,0.164,0.164,0.164,0.165,0.165,0.165,0.165,0.166,0.166,0.166,0.166,0.167,0.167,0.168,0.168,0.168,0.169,0.169,0.170,0.170,0.170,0.171,0.171,0.171,0.172,0.172,0.173,0.173,0.173,0.174,0.175,0.175,0.176,0.176,0.177,0.177,0.178,0.178,0.179,0.179,0.179,0.179,0.180,0.180,0.180,0.180,0.180,0.181,0.181,0.181,0.182,0.182,0.183,0.184,0.184,0.185,0.186,0.186,0.187,0.187,0.188,0.189,0.189,0.189,0.190,0.190,0.191,0.191,0.192,0.192,0.193,0.193,0.193,0.194,0.194,0.195,0.195,0.196,0.196,0.196,0.197,0.197,0.198,0.198,0.198,0.199,0.199,0.199,0.199,0.200,0.200,0.200,0.201,0.201,0.201,0.201,0.202,0.202,0.202,0.202,0.202,0.203,0.203,0.203,0.203,0.204,0.204,0.204,0.205,0.205,0.205,0.205,0.206,0.206,0.206,0.206,0.207,0.207,0.207,0.207,0.207,0.208,0.208,0.208,0.208,0.208,0.208,0.208,0.208,0.209,0.209,0.209,0.209,0.209,0.209,0.210,0.210,0.210,0.211,0.211,0.211,0.212,0.212,0.212,0.213,0.213,0.214,0.214,0.214,0.214,0.215,0.215,0.215,0.216,0.216,0.216,0.217,0.217,0.217,0.218,0.218,0.218,0.219,0.219,0.219,0.219])
		self.aC["SA"] = self.aC["SD"]
		self.aD = {}
		self.aD["PGA"] = np.array([0.332])
		self.aD["SD"] = np.array([0.166,0.160,0.293,0.395,0.441,0.455,0.502,0.534,0.584,0.636,0.679,0.692,0.695,0.696,0.704,0.704,0.709,0.709,0.703,0.686,0.671,0.657,0.639,0.623,0.609,0.594,0.585,0.570,0.559,0.551,0.543,0.532,0.515,0.504,0.497,0.489,0.482,0.472,0.465,0.460,0.459,0.459,0.459,0.457,0.453,0.446,0.440,0.437,0.431,0.425,0.424,0.421,0.421,0.420,0.421,0.422,0.421,0.416,0.412,0.407,0.402,0.399,0.397,0.395,0.393,0.392,0.391,0.392,0.392,0.390,0.389,0.388,0.386,0.383,0.382,0.379,0.376,0.372,0.369,0.366,0.362,0.360,0.357,0.354,0.350,0.347,0.346,0.345,0.344,0.343,0.342,0.341,0.341,0.341,0.341,0.341,0.339,0.337,0.336,0.334,0.333,0.331,0.329,0.326,0.324,0.322,0.320,0.318,0.316,0.314,0.314,0.313,0.313,0.312,0.311,0.309,0.308,0.307,0.306,0.305,0.304,0.303,0.303,0.302,0.301,0.301,0.300,0.300,0.300,0.300,0.300,0.301,0.302,0.302,0.302,0.302,0.303,0.303,0.302,0.302,0.301,0.301,0.300,0.300,0.300,0.300,0.300,0.300,0.301,0.302,0.303,0.303,0.304,0.305,0.306,0.306,0.307,0.308,0.308,0.309,0.309,0.310,0.311,0.311,0.311,0.311,0.311,0.310,0.310,0.310,0.310,0.310,0.310,0.310,0.310,0.310,0.310,0.310,0.310,0.310,0.310,0.310,0.310,0.310,0.309,0.309,0.309,0.309,0.308,0.308,0.308,0.308,0.308,0.307,0.307,0.307,0.306,0.306,0.306,0.305,0.305,0.305,0.305,0.305,0.304,0.304,0.303,0.303,0.302,0.302,0.301,0.300,0.300,0.299,0.298,0.297,0.296,0.296,0.295,0.294,0.294,0.294,0.293,0.294,0.294,0.294,0.294,0.294,0.294,0.293,0.293,0.293,0.293,0.293,0.292,0.292,0.292,0.292,0.292,0.293,0.293,0.293,0.293,0.293,0.293,0.294,0.294,0.294,0.294,0.295,0.295,0.295,0.296,0.296,0.296,0.296,0.297,0.297,0.298,0.298,0.298,0.299,0.299,0.299,0.300,0.300,0.301,0.301,0.302,0.302,0.303,0.303,0.303,0.303,0.304,0.304,0.304,0.304,0.305,0.305,0.305,0.305,0.306,0.306,0.307,0.307,0.308,0.308,0.309,0.309,0.310,0.310,0.310,0.311,0.311,0.311,0.312,0.312,0.312,0.313,0.313,0.313,0.314,0.314,0.314,0.314,0.315,0.315,0.315,0.316,0.316,0.317,0.317,0.318,0.318,0.319,0.319,0.320,0.320,0.320,0.320,0.321,0.321,0.321,0.321,0.322,0.322,0.322,0.322,0.322,0.322,0.322,0.323,0.323,0.323,0.323,0.323,0.324,0.324,0.324,0.324,0.324,0.325,0.325,0.325,0.325,0.325,0.325,0.326,0.326,0.326,0.326,0.326,0.326,0.326,0.326,0.326,0.326,0.326,0.326,0.326,0.326,0.326,0.326,0.326,0.326,0.327,0.327,0.327,0.327,0.327,0.328,0.328,0.328,0.328,0.328,0.329,0.329,0.329,0.330,0.330,0.330,0.330,0.330,0.331,0.331,0.331,0.331,0.331,0.331,0.332,0.332,0.332,0.332,0.333,0.333,0.333,0.333,0.333,0.333])
		self.aD["SA"] = self.aD["SD"]
		## Sigma for unspecified focal mechanism
		self.SigmaTU = {}
		self.SigmaTU["PGA"] = np.array([0.344])
		self.SigmaTU["SD"] = np.array([0.364,0.376,0.375,0.376,0.367,0.358,0.357,0.357,0.353,0.357,0.357,0.355,0.353,0.353,0.351,0.348,0.349,0.353,0.353,0.355,0.354,0.354,0.353,0.352,0.352,0.352,0.352,0.351,0.351,0.349,0.346,0.344,0.341,0.339,0.337,0.335,0.333,0.332,0.330,0.329,0.329,0.328,0.328,0.327,0.325,0.323,0.322,0.320,0.319,0.318,0.317,0.315,0.313,0.312,0.311,0.310,0.309,0.308,0.307,0.306,0.306,0.305,0.304,0.303,0.302,0.301,0.299,0.298,0.297,0.295,0.295,0.294,0.294,0.294,0.294,0.294,0.294,0.293,0.293,0.293,0.292,0.292,0.291,0.291,0.290,0.289,0.289,0.288,0.288,0.287,0.287,0.286,0.286,0.285,0.284,0.283,0.283,0.282,0.281,0.280,0.280,0.279,0.278,0.277,0.277,0.276,0.275,0.274,0.274,0.273,0.273,0.272,0.272,0.272,0.271,0.271,0.270,0.270,0.270,0.269,0.269,0.269,0.268,0.268,0.268,0.267,0.267,0.267,0.266,0.266,0.266,0.266,0.266,0.265,0.265,0.265,0.265,0.265,0.265,0.265,0.265,0.265,0.265,0.265,0.264,0.264,0.264,0.264,0.263,0.263,0.263,0.263,0.262,0.262,0.262,0.262,0.262,0.262,0.261,0.261,0.261,0.261,0.260,0.260,0.260,0.260,0.259,0.259,0.259,0.259,0.259,0.258,0.258,0.258,0.258,0.257,0.257,0.257,0.257,0.256,0.256,0.255,0.255,0.255,0.255,0.254,0.254,0.254,0.253,0.253,0.253,0.252,0.252,0.252,0.252,0.251,0.251,0.251,0.251,0.251,0.250,0.250,0.250,0.250,0.250,0.250,0.249,0.249,0.249,0.249,0.249,0.249,0.249,0.249,0.248,0.248,0.248,0.248,0.248,0.248,0.248,0.248,0.248,0.248,0.247,0.247,0.247,0.247,0.247,0.247,0.247,0.246,0.246,0.246,0.246,0.246,0.246,0.246,0.246,0.245,0.245,0.245,0.245,0.245,0.245,0.245,0.245,0.244,0.244,0.244,0.244,0.244,0.244,0.243,0.243,0.243,0.243,0.243,0.243,0.242,0.242,0.242,0.242,0.242,0.242,0.241,0.241,0.241,0.241,0.241,0.241,0.241,0.240,0.240,0.240,0.240,0.240,0.240,0.240,0.239,0.239,0.239,0.239,0.239,0.239,0.239,0.238,0.238,0.238,0.238,0.238,0.238,0.238,0.237,0.237,0.237,0.237,0.237,0.237,0.237,0.237,0.237,0.237,0.237,0.237,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.235,0.235,0.235,0.235,0.235,0.235,0.235,0.235,0.235,0.235,0.235,0.235,0.235,0.235,0.235,0.235,0.235,0.235,0.235,0.235,0.235,0.235,0.235,0.235,0.235,0.235,0.235,0.235,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.236,0.237,0.237,0.237,0.237,0.237,0.237,0.237,0.237,0.237,0.237,0.237,0.237,0.237,0.238,0.238,0.238,0.238,0.238,0.238,0.238,0.238,0.238,0.238,0.239,0.239,0.239])
		self.SigmaTU["SA"] = self.SigmaTU["SD"]
		## Rake-dependent terms
		self.aN = {}
		self.aN["PGA"] = np.array([-0.06])
		self.aN["SD"] = np.array([-0.100,-0.089,-0.073,-0.041,-0.051,-0.042,-0.029,-0.025,-0.010,-0.005,-0.001,-0.001,0.006,0.002,0.003,0.004,0.003,0.011,0.016,0.020,0.018,0.017,0.017,0.022,0.025,0.028,0.028,0.027,0.025,0.022,0.017,0.014,0.009,0.001,-0.008,-0.014,-0.019,-0.024,-0.030,-0.034,-0.039,-0.045,-0.049,-0.055,-0.060,-0.064,-0.066,-0.068,-0.072,-0.076,-0.078,-0.080,-0.081,-0.081,-0.080,-0.080,-0.080,-0.079,-0.078,-0.078,-0.078,-0.077,-0.076,-0.075,-0.075,-0.075,-0.073,-0.071,-0.071,-0.070,-0.070,-0.070,-0.069,-0.069,-0.070,-0.070,-0.070,-0.069,-0.069,-0.069,-0.069,-0.068,-0.067,-0.066,-0.065,-0.063,-0.062,-0.061,-0.059,-0.058,-0.057,-0.056,-0.055,-0.054,-0.053,-0.052,-0.051,-0.051,-0.050,-0.050,-0.049,-0.049,-0.048,-0.047,-0.046,-0.045,-0.044,-0.043,-0.042,-0.041,-0.039,-0.038,-0.036,-0.034,-0.032,-0.030,-0.028,-0.026,-0.024,-0.023,-0.021,-0.019,-0.018,-0.016,-0.015,-0.014,-0.013,-0.012,-0.011,-0.009,-0.008,-0.007,-0.005,-0.003,-0.002,-0.001,0.000,0.001,0.001,0.002,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.008,0.009,0.009,0.010,0.011,0.011,0.012,0.013,0.014,0.014,0.015,0.016,0.016,0.017,0.018,0.018,0.019,0.019,0.019,0.020,0.019,0.019,0.019,0.018,0.018,0.018,0.017,0.017,0.017,0.016,0.016,0.015,0.015,0.015,0.015,0.015,0.015,0.015,0.015,0.015,0.015,0.015,0.015,0.016,0.016,0.016,0.016,0.016,0.016,0.017,0.017,0.018,0.018,0.019,0.019,0.020,0.021,0.021,0.021,0.022,0.022,0.022,0.023,0.023,0.023,0.024,0.024,0.024,0.024,0.025,0.025,0.025,0.025,0.025,0.025,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.027,0.027,0.027,0.027,0.027,0.027,0.027,0.027,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.026,0.025,0.025,0.025,0.025,0.025,0.025,0.024,0.024,0.024,0.024,0.024,0.024,0.024,0.024,0.023,0.023,0.023,0.023,0.023,0.023,0.023,0.023,0.023,0.023,0.023,0.023,0.022,0.022,0.022,0.022,0.022,0.022,0.022,0.021,0.021,0.021,0.021,0.020,0.020,0.020,0.020,0.019,0.019,0.019,0.019,0.019,0.018,0.018,0.018,0.018,0.018,0.018,0.018,0.018,0.018,0.017,0.017,0.017,0.017,0.017,0.017,0.017,0.017,0.017,0.017,0.017,0.017,0.017,0.017,0.017,0.017,0.017,0.017,0.017,0.017,0.017,0.017,0.017,0.017,0.017,0.017,0.017,0.017,0.017,0.016,0.016,0.016,0.016,0.016,0.016,0.016,0.016,0.016,0.015,0.015,0.015,0.015,0.015,0.015,0.014,0.014,0.014,0.014,0.014,0.014,0.014,0.014,0.014,0.014,0.014,0.014,0.014,0.013,0.013,0.013,0.013,0.013,0.013,0.013,0.013,0.013,0.013,0.013,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012,0.012])
		self.aN["SA"] = self.aN["SD"]
		self.aR = {}
		self.aR["PGA"] = np.array([0.094])
		self.aR["SD"] = np.array([0.132,0.127,0.119,0.101,0.085,0.065,0.057,0.051,0.037,0.028,0.026,0.017,0.008,0.005,0.002,-0.004,-0.006,-0.006,-0.005,-0.003,-0.002,-0.003,-0.004,-0.006,-0.008,-0.009,-0.010,-0.010,-0.009,-0.009,-0.007,-0.005,-0.004,-0.003,-0.001,0.001,0.003,0.005,0.008,0.010,0.012,0.015,0.017,0.020,0.023,0.027,0.029,0.032,0.036,0.040,0.043,0.045,0.048,0.051,0.053,0.056,0.058,0.060,0.062,0.064,0.065,0.066,0.066,0.065,0.065,0.065,0.064,0.062,0.061,0.060,0.059,0.057,0.057,0.056,0.056,0.055,0.055,0.054,0.053,0.053,0.052,0.051,0.050,0.048,0.047,0.045,0.043,0.041,0.039,0.037,0.036,0.034,0.032,0.030,0.028,0.026,0.024,0.021,0.019,0.016,0.014,0.012,0.010,0.008,0.007,0.005,0.004,0.002,0.001,-0.001,-0.002,-0.003,-0.004,-0.005,-0.007,-0.008,-0.010,-0.011,-0.012,-0.014,-0.015,-0.017,-0.018,-0.020,-0.022,-0.024,-0.027,-0.029,-0.032,-0.034,-0.037,-0.039,-0.041,-0.044,-0.046,-0.048,-0.049,-0.051,-0.053,-0.054,-0.056,-0.057,-0.059,-0.060,-0.062,-0.063,-0.064,-0.066,-0.067,-0.069,-0.070,-0.071,-0.073,-0.074,-0.076,-0.077,-0.079,-0.080,-0.082,-0.083,-0.084,-0.086,-0.087,-0.088,-0.090,-0.091,-0.092,-0.093,-0.094,-0.095,-0.096,-0.096,-0.097,-0.098,-0.098,-0.099,-0.099,-0.100,-0.100,-0.101,-0.101,-0.102,-0.102,-0.102,-0.102,-0.103,-0.103,-0.103,-0.103,-0.103,-0.103,-0.104,-0.104,-0.104,-0.104,-0.104,-0.104,-0.105,-0.105,-0.105,-0.105,-0.105,-0.105,-0.105,-0.105,-0.105,-0.105,-0.105,-0.105,-0.105,-0.105,-0.105,-0.105,-0.105,-0.105,-0.105,-0.105,-0.105,-0.104,-0.104,-0.104,-0.104,-0.104,-0.104,-0.105,-0.105,-0.104,-0.104,-0.104,-0.104,-0.104,-0.104,-0.104,-0.104,-0.103,-0.103,-0.103,-0.103,-0.102,-0.102,-0.102,-0.102,-0.102,-0.101,-0.101,-0.101,-0.101,-0.101,-0.101,-0.101,-0.101,-0.101,-0.101,-0.100,-0.100,-0.100,-0.100,-0.100,-0.100,-0.100,-0.100,-0.100,-0.100,-0.100,-0.100,-0.100,-0.099,-0.099,-0.099,-0.099,-0.099,-0.099,-0.099,-0.099,-0.099,-0.099,-0.099,-0.099,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.098,-0.097,-0.097,-0.097,-0.097,-0.097,-0.097,-0.097,-0.097,-0.097,-0.097,-0.097,-0.096,-0.096,-0.096,-0.096,-0.096,-0.096,-0.096,-0.096,-0.096,-0.096,-0.095,-0.095,-0.095,-0.095,-0.095,-0.095,-0.095,-0.095,-0.095,-0.094,-0.094,-0.094,-0.094,-0.094,-0.094,-0.093,-0.093,-0.093,-0.093,-0.093,-0.093,-0.092,-0.092,-0.092,-0.092,-0.092,-0.092,-0.091,-0.091,-0.091,-0.091,-0.091,-0.091,-0.090,-0.090,-0.090,-0.090,-0.090,-0.090,-0.090,-0.090,-0.089,-0.089,-0.089,-0.089,-0.089,-0.089,-0.089,-0.088,-0.088,-0.088,-0.088,-0.088,-0.088,-0.087,-0.087,-0.087,-0.087,-0.087,-0.087,-0.087])
		self.aR["SA"] = self.aR["SD"]
		self.aS = {}
		self.aS["PGA"] = np.array([-0.013])
		self.aS["SD"] = np.array([-0.010,-0.014,-0.018,-0.024,-0.014,-0.010,-0.012,-0.011,-0.011,-0.010,-0.010,-0.006,-0.005,-0.003,-0.002,0.001,0.002,-0.002,-0.004,-0.006,-0.006,-0.005,-0.005,-0.006,-0.006,-0.007,-0.006,-0.006,-0.006,-0.004,-0.003,-0.003,-0.001,0.001,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.010,0.011,0.011,0.012,0.011,0.010,0.010,0.010,0.009,0.008,0.008,0.007,0.005,0.004,0.002,0.001,0.000,-0.002,-0.003,-0.004,-0.004,-0.004,-0.004,-0.004,-0.004,-0.004,-0.004,-0.003,-0.003,-0.002,-0.001,-0.001,-0.001,0.000,0.000,0.001,0.001,0.001,0.001,0.002,0.002,0.002,0.003,0.003,0.003,0.004,0.004,0.005,0.005,0.005,0.006,0.007,0.007,0.008,0.008,0.009,0.009,0.010,0.011,0.012,0.013,0.013,0.014,0.014,0.014,0.014,0.015,0.015,0.015,0.015,0.015,0.015,0.014,0.014,0.014,0.014,0.014,0.014,0.014,0.014,0.014,0.014,0.015,0.015,0.016,0.016,0.017,0.017,0.018,0.019,0.019,0.020,0.020,0.020,0.021,0.021,0.022,0.022,0.022,0.023,0.023,0.023,0.024,0.024,0.024,0.024,0.024,0.025,0.025,0.025,0.026,0.026,0.026,0.027,0.027,0.027,0.027,0.028,0.028,0.028,0.028,0.029,0.029,0.029,0.030,0.030,0.031,0.031,0.031,0.032,0.032,0.033,0.033,0.033,0.034,0.034,0.034,0.035,0.035,0.035,0.035,0.035,0.036,0.036,0.036,0.036,0.036,0.036,0.036,0.036,0.036,0.036,0.036,0.036,0.036,0.036,0.035,0.035,0.035,0.035,0.035,0.035,0.035,0.034,0.034,0.034,0.034,0.034,0.034,0.034,0.034,0.033,0.033,0.033,0.033,0.033,0.033,0.033,0.033,0.033,0.033,0.033,0.033,0.033,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.033,0.033,0.033,0.033,0.033,0.033,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.032,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031,0.031])
		self.aS["SA"] = self.aS["SD"]
		self.SigmaTM = {}
		self.SigmaTM["PGA"] = np.array([0.341])
		self.SigmaTM["SD"] = np.array([0.357,0.370,0.370,0.374,0.366,0.357,0.358,0.357,0.354,0.359,0.359,0.357,0.355,0.355,0.353,0.350,0.350,0.355,0.355,0.356,0.356,0.356,0.354,0.353,0.354,0.353,0.353,0.353,0.352,0.350,0.347,0.345,0.343,0.340,0.338,0.336,0.334,0.333,0.331,0.330,0.329,0.329,0.328,0.326,0.324,0.322,0.321,0.319,0.318,0.316,0.314,0.312,0.311,0.309,0.308,0.307,0.306,0.305,0.304,0.303,0.303,0.302,0.301,0.300,0.299,0.298,0.296,0.295,0.294,0.293,0.292,0.292,0.292,0.293,0.293,0.292,0.292,0.292,0.291,0.291,0.291,0.290,0.290,0.289,0.289,0.288,0.288,0.288,0.287,0.287,0.287,0.286,0.285,0.285,0.284,0.284,0.283,0.282,0.281,0.281,0.280,0.279,0.278,0.278,0.277,0.276,0.276,0.275,0.274,0.273,0.273,0.273,0.272,0.272,0.272,0.271,0.271,0.271,0.270,0.270,0.270,0.269,0.269,0.269,0.268,0.268,0.267,0.267,0.266,0.266,0.266,0.266,0.265,0.265,0.265,0.265,0.264,0.264,0.264,0.264,0.264,0.264,0.263,0.263,0.263,0.262,0.262,0.262,0.261,0.261,0.261,0.260,0.260,0.260,0.260,0.259,0.259,0.258,0.258,0.258,0.257,0.257,0.257,0.256,0.256,0.255,0.255,0.255,0.254,0.254,0.254,0.253,0.253,0.253,0.253,0.252,0.252,0.251,0.251,0.251,0.250,0.250,0.249,0.249,0.249,0.248,0.248,0.248,0.247,0.247,0.247,0.246,0.246,0.246,0.246,0.245,0.245,0.245,0.245,0.244,0.244,0.244,0.244,0.244,0.243,0.243,0.243,0.243,0.243,0.242,0.242,0.242,0.242,0.242,0.242,0.242,0.242,0.242,0.242,0.242,0.242,0.241,0.241,0.241,0.241,0.241,0.241,0.241,0.241,0.240,0.240,0.240,0.240,0.240,0.240,0.240,0.239,0.239,0.239,0.239,0.239,0.239,0.239,0.239,0.239,0.239,0.239,0.238,0.238,0.238,0.238,0.238,0.238,0.237,0.237,0.237,0.237,0.237,0.237,0.237,0.236,0.236,0.236,0.236,0.236,0.236,0.235,0.235,0.235,0.235,0.235,0.235,0.235,0.234,0.234,0.234,0.234,0.234,0.234,0.234,0.234,0.233,0.233,0.233,0.233,0.233,0.233,0.233,0.232,0.232,0.232,0.232,0.232,0.232,0.232,0.232,0.231,0.231,0.231,0.231,0.231,0.231,0.231,0.231,0.231,0.231,0.231,0.231,0.231,0.231,0.231,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.230,0.231,0.231,0.231,0.231,0.231,0.231,0.231,0.231,0.231,0.231,0.231,0.231,0.231,0.232,0.232,0.232,0.232,0.232,0.232,0.232,0.232,0.232,0.232,0.232,0.233,0.233,0.233,0.233,0.233,0.233,0.233,0.233,0.233,0.233,0.234,0.234,0.234,0.234,0.234,0.234,0.234,0.234,0.235,0.235])
		self.SigmaTM["SA"] = self.SigmaTM["SD"]
		## Explicit vs30
		#self.bV = {}
		#self.bV["PGA"] = np.array([-0.432])
		#self.bV["SD"] = np.array([-0.186,-0.110,-0.357,-0.585,-0.721,-0.798,-0.892,-0.951,-0.996,-1.043,-1.085,-1.099,-1.112,-1.129,-1.123,-1.114,-1.108,-1.103,-1.086,-1.055,-1.026,-1.000,-0.972,-0.941,-0.918,-0.898,-0.883,-0.860,-0.845,-0.834,-0.822,-0.803,-0.783,-0.768,-0.758,-0.746,-0.734,-0.723,-0.712,-0.701,-0.697,-0.694,-0.691,-0.686,-0.680,-0.673,-0.666,-0.655,-0.644,-0.636,-0.634,-0.627,-0.624,-0.621,-0.619,-0.617,-0.613,-0.606,-0.601,-0.594,-0.589,-0.585,-0.582,-0.577,-0.574,-0.571,-0.569,-0.568,-0.565,-0.561,-0.559,-0.556,-0.552,-0.550,-0.547,-0.545,-0.542,-0.538,-0.534,-0.530,-0.526,-0.523,-0.519,-0.516,-0.512,-0.510,-0.508,-0.506,-0.505,-0.504,-0.502,-0.499,-0.497,-0.495,-0.493,-0.492,-0.490,-0.488,-0.486,-0.484,-0.482,-0.480,-0.477,-0.475,-0.474,-0.473,-0.473,-0.472,-0.471,-0.471,-0.471,-0.471,-0.471,-0.471,-0.471,-0.471,-0.470,-0.469,-0.468,-0.467,-0.466,-0.465,-0.464,-0.463,-0.461,-0.459,-0.457,-0.456,-0.453,-0.451,-0.449,-0.448,-0.447,-0.447,-0.447,-0.448,-0.449,-0.449,-0.449,-0.450,-0.450,-0.451,-0.452,-0.453,-0.453,-0.454,-0.455,-0.457,-0.458,-0.460,-0.461,-0.463,-0.465,-0.467,-0.469,-0.471,-0.473,-0.474,-0.475,-0.476,-0.477,-0.478,-0.479,-0.479,-0.479,-0.480,-0.480,-0.480,-0.480,-0.480,-0.479,-0.479,-0.479,-0.479,-0.478,-0.478,-0.478,-0.478,-0.477,-0.477,-0.476,-0.476,-0.475,-0.474,-0.473,-0.472,-0.472,-0.471,-0.470,-0.470,-0.469,-0.469,-0.468,-0.468,-0.467,-0.466,-0.466,-0.465,-0.465,-0.464,-0.463,-0.463,-0.463,-0.463,-0.462,-0.462,-0.461,-0.461,-0.460,-0.460,-0.459,-0.458,-0.457,-0.457,-0.456,-0.456,-0.456,-0.455,-0.455,-0.455,-0.455,-0.455,-0.455,-0.456,-0.456,-0.456,-0.456,-0.456,-0.456,-0.456,-0.455,-0.455,-0.455,-0.455,-0.455,-0.455,-0.454,-0.454,-0.454,-0.454,-0.454,-0.454,-0.454,-0.454,-0.454,-0.454,-0.454,-0.454,-0.454,-0.454,-0.454,-0.454,-0.454,-0.454,-0.454,-0.455,-0.455,-0.455,-0.455,-0.455,-0.455,-0.456,-0.456,-0.456,-0.456,-0.456,-0.456,-0.456,-0.456,-0.456,-0.457,-0.457,-0.457,-0.457,-0.457,-0.457,-0.457,-0.458,-0.458,-0.458,-0.458,-0.459,-0.459,-0.459,-0.460,-0.460,-0.460,-0.461,-0.461,-0.461,-0.462,-0.462,-0.463,-0.463,-0.463,-0.464,-0.464,-0.464,-0.465,-0.465,-0.466,-0.467,-0.467,-0.468,-0.468,-0.469,-0.469,-0.469,-0.470,-0.470,-0.470,-0.471,-0.471,-0.471,-0.472,-0.472,-0.472,-0.473,-0.473,-0.473,-0.473,-0.473,-0.473,-0.474,-0.474,-0.474,-0.474,-0.474,-0.475,-0.475,-0.475,-0.475,-0.475,-0.475,-0.475,-0.476,-0.476,-0.476,-0.476,-0.476,-0.476,-0.476,-0.477,-0.477,-0.477,-0.477,-0.477,-0.478,-0.478,-0.478,-0.478,-0.479,-0.479,-0.479,-0.479,-0.479,-0.480,-0.480,-0.480,-0.480,-0.480,-0.481,-0.481,-0.481,-0.481,-0.482,-0.482,-0.482,-0.483,-0.483,-0.483,-0.484,-0.484,-0.484,-0.485,-0.485,-0.486,-0.486,-0.486,-0.487,-0.487,-0.487,-0.488,-0.488,-0.488,-0.489,-0.489,-0.489,-0.489,-0.490,-0.490,-0.491,-0.491,-0.491,-0.492,-0.492,-0.492,-0.492,-0.493,-0.493])
		#self.bV["SA"] = self.bV["SD"]
		#self.Va = {}
		#self.Va["PGA"] = np.array([1319.1])
		#self.Va["SD"] = np.array([3120.000,2505.000,1890.000,1220.300,948.520,821.610,770.300,715.890,704.360,693.490,675.830,662.050,641.590,625.100,623.620,617.340,622.530,633.640,647.640,647.280,651.350,648.020,648.220,648.030,652.470,656.820,670.910,678.160,683.390,682.560,680.080,677.830,669.710,668.930,668.710,665.350,662.340,653.540,649.200,651.830,653.750,654.370,656.830,659.300,659.440,652.510,647.220,645.870,641.060,637.650,637.240,638.720,640.900,640.550,641.340,641.240,643.370,640.130,637.650,636.950,637.270,637.280,637.290,638.620,639.330,639.460,640.440,643.280,645.950,648.080,651.010,653.940,656.140,658.020,661.580,664.940,667.260,667.530,665.460,663.230,658.680,654.220,649.450,643.460,635.420,630.300,627.690,625.700,624.100,621.880,622.990,624.950,627.600,630.360,633.020,635.580,634.430,633.470,634.310,636.770,639.750,641.750,642.880,641.670,640.470,639.280,637.680,636.720,636.820,637.000,637.780,639.460,640.950,641.460,640.470,639.580,638.170,635.740,634.390,632.300,630.760,630.620,630.820,631.580,632.600,634.160,635.200,636.460,638.680,641.210,644.510,647.970,650.650,651.850,651.380,649.980,648.620,648.130,646.090,643.010,640.810,638.820,636.300,635.220,634.400,634.550,634.430,635.330,636.040,636.800,638.780,640.210,641.780,642.290,642.150,642.180,641.900,641.750,641.720,641.330,640.940,640.520,640.570,639.710,638.970,638.470,637.800,637.640,636.650,636.090,635.880,635.440,635.030,633.990,633.230,631.860,630.780,629.580,628.420,627.840,627.030,626.250,625.660,625.450,625.310,624.470,624.370,624.190,623.770,623.140,622.860,622.090,621.750,621.680,621.700,621.200,621.140,621.250,621.360,621.620,621.010,621.230,621.150,620.520,619.320,618.170,616.820,614.950,613.770,612.660,611.370,609.650,607.920,605.790,603.460,601.230,599.370,597.620,595.720,594.130,592.770,591.490,590.290,589.360,588.720,588.220,587.770,587.730,587.340,586.910,586.590,586.390,585.700,585.550,585.480,585.540,585.720,586.050,585.990,586.630,587.230,587.750,588.260,588.840,589.550,590.340,591.420,591.990,592.870,593.740,594.660,595.720,596.480,597.280,598.200,599.200,600.090,600.890,601.780,602.790,603.760,604.850,606.070,607.410,608.890,610.570,612.190,613.820,614.660,615.790,616.940,618.050,618.870,619.440,619.890,620.290,620.010,620.390,620.500,620.490,619.870,619.990,620.440,621.070,621.880,622.260,623.380,624.600,625.760,626.770,627.660,628.590,629.530,629.950,631.040,631.820,632.380,632.970,633.510,634.100,634.690,634.570,635.000,635.450,635.990,636.570,637.190,637.940,638.180,639.040,639.860,640.650,641.590,642.590,643.620,644.660,645.640,646.260,646.190,646.540,646.910,647.340,647.760,648.180,648.630,649.160,649.670,649.960,650.280,650.470,649.990,650.140,650.290,650.560,650.870,651.210,651.540,651.890,652.240,652.680,652.520,653.030,653.650,654.160,654.570,654.850,655.120,655.370,655.610,655.850,656.100,656.160,656.180,656.160,655.960,655.780,655.580,655.380,655.170,655.010,654.980,655.010,655.690,655.620,655.470,655.320,655.210,655.130,655.080,655.030,655.150,655.320,655.550,655.830,656.130,656.380,656.660,656.970,657.940,658.290,658.610,658.890,659.140,659.400,659.730,660.100,660.450,660.850,661.190,661.530,661.880,662.230,662.640,662.990,663.290,663.600,663.930,664.110,664.920,665.080])
		#self.Va["SA"] = self.Va["SD"]
		self.bV800 = {}
		self.bV800["PGA"] = np.array([-0.764])
		self.bV800["SD"] = np.array([-0.602,-0.666,-0.804,-0.948,-0.964,-0.923,-0.949,-0.922,-0.925,-0.950,-0.959,-0.947,-0.915,-0.894,-0.881,-0.855,-0.854,-0.860,-0.864,-0.840,-0.820,-0.793,-0.770,-0.742,-0.726,-0.711,-0.711,-0.696,-0.689,-0.681,-0.671,-0.655,-0.633,-0.624,-0.622,-0.614,-0.607,-0.593,-0.583,-0.580,-0.579,-0.578,-0.580,-0.580,-0.577,-0.567,-0.557,-0.549,-0.537,-0.528,-0.525,-0.521,-0.520,-0.515,-0.513,-0.509,-0.505,-0.494,-0.485,-0.476,-0.469,-0.463,-0.458,-0.453,-0.449,-0.445,-0.441,-0.440,-0.439,-0.437,-0.437,-0.435,-0.434,-0.432,-0.432,-0.432,-0.430,-0.426,-0.421,-0.416,-0.410,-0.403,-0.397,-0.390,-0.381,-0.376,-0.374,-0.372,-0.370,-0.368,-0.367,-0.366,-0.367,-0.367,-0.368,-0.369,-0.368,-0.366,-0.365,-0.365,-0.366,-0.366,-0.365,-0.363,-0.362,-0.361,-0.360,-0.358,-0.358,-0.358,-0.358,-0.359,-0.360,-0.359,-0.357,-0.355,-0.353,-0.349,-0.347,-0.344,-0.341,-0.339,-0.338,-0.337,-0.335,-0.334,-0.333,-0.331,-0.330,-0.329,-0.329,-0.329,-0.329,-0.328,-0.327,-0.325,-0.324,-0.323,-0.321,-0.318,-0.316,-0.315,-0.313,-0.312,-0.311,-0.311,-0.311,-0.312,-0.313,-0.314,-0.315,-0.317,-0.318,-0.320,-0.321,-0.321,-0.322,-0.322,-0.323,-0.323,-0.322,-0.322,-0.322,-0.321,-0.320,-0.320,-0.320,-0.319,-0.318,-0.318,-0.317,-0.317,-0.317,-0.316,-0.315,-0.315,-0.315,-0.314,-0.313,-0.313,-0.312,-0.311,-0.311,-0.310,-0.310,-0.310,-0.309,-0.309,-0.308,-0.308,-0.307,-0.307,-0.307,-0.307,-0.306,-0.306,-0.306,-0.306,-0.306,-0.306,-0.306,-0.306,-0.306,-0.305,-0.304,-0.303,-0.302,-0.300,-0.299,-0.298,-0.297,-0.295,-0.293,-0.291,-0.289,-0.287,-0.286,-0.284,-0.283,-0.281,-0.280,-0.279,-0.279,-0.278,-0.278,-0.278,-0.278,-0.278,-0.278,-0.277,-0.277,-0.277,-0.277,-0.277,-0.277,-0.277,-0.277,-0.277,-0.278,-0.279,-0.279,-0.280,-0.281,-0.281,-0.282,-0.282,-0.283,-0.284,-0.285,-0.286,-0.287,-0.288,-0.289,-0.289,-0.290,-0.291,-0.292,-0.293,-0.294,-0.295,-0.295,-0.296,-0.297,-0.298,-0.300,-0.301,-0.302,-0.303,-0.305,-0.306,-0.307,-0.308,-0.308,-0.309,-0.310,-0.310,-0.311,-0.311,-0.312,-0.312,-0.312,-0.313,-0.314,-0.314,-0.315,-0.316,-0.318,-0.319,-0.320,-0.321,-0.322,-0.323,-0.324,-0.325,-0.326,-0.327,-0.328,-0.329,-0.329,-0.330,-0.331,-0.332,-0.333,-0.333,-0.334,-0.335,-0.336,-0.337,-0.338,-0.338,-0.339,-0.340,-0.341,-0.342,-0.343,-0.344,-0.345,-0.346,-0.346,-0.347,-0.347,-0.348,-0.348,-0.349,-0.349,-0.350,-0.351,-0.351,-0.351,-0.352,-0.352,-0.352,-0.353,-0.353,-0.354,-0.354,-0.355,-0.355,-0.356,-0.356,-0.357,-0.357,-0.358,-0.359,-0.359,-0.360,-0.360,-0.361,-0.361,-0.362,-0.362,-0.363,-0.363,-0.363,-0.363,-0.363,-0.363,-0.364,-0.364,-0.364,-0.364,-0.364,-0.365,-0.365,-0.365,-0.365,-0.366,-0.366,-0.366,-0.367,-0.367,-0.368,-0.368,-0.369,-0.369,-0.370,-0.371,-0.371,-0.372,-0.373,-0.373,-0.374,-0.374,-0.375,-0.375,-0.376,-0.376,-0.377,-0.378,-0.378,-0.379,-0.380,-0.380,-0.381,-0.382,-0.382,-0.383,-0.383,-0.384,-0.384])
		self.bV800["SA"] = self.bV800["SD"]

		## Unit conversion
		## Original unit is cm
		self.imt_scaling = {}
		self.imt_scaling["PGA"] = {"g": 1./g, "mg": 1E+3/g, "ms2": 1., "gal": 100.0, "cms2": 100.0}
		self.imt_scaling["SD"] = {"m": 1E-2, "cm": 1.0}
		self.imt_scaling["SA"] = {"g": 0.01/g, "mg": 10./g, "ms2": 1E-2, "gal": 1.0, "cms2": 1.0}

	def __call__(self, M, d, h=0, imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="rock", vs30=None, explicit_vs30=False, kappa=None, mechanism="normal", damping=5):
		"""
		Return ground motion for given magnitude, distance, depth, soil type,
		and fault mechanism.

		:param M:
			Float or float array, magnitude(s).
			Note that if d is an array, M must be a float.
		:param d:
			Float or float array, distance(s) in km.
			Note that if M is an array, d must be a float.
		:param h:
			Float, focal depth in km (default: 0). Ignored in this GMPE.
		:param imt:
			String, one of the supported intensity measure types: "PGA",
			"PHV" or "SA" (default: "PGA").
		:param T:
			Float, spectral period of considered IMT. Ignored if IMT is
			"PGA" or "PGV" (default: 0).
		:param imt_unit:
			String, unit in which intensities should be expressed (default: "g")
		:param epsilon:
			Float, number of standard deviations above or below the mean
			(default: 0).
		:param soil_type:
			String, either "rock", "stiff" or "soft" (default: "rock"):
				Rock: VS >= 750 m/s
				Stiff soil: 360 <= VS < 750 m/s
				Soft soil: VS < 360 m/s
		:param vs30:
			Float, shear-wave velocity in the upper 30 m (in m/s). If not None,
			it takes precedence over the soil_type parameter (default: None).
		:param explicit_vs30:
			Bool, whether or not vs30 value must be used explicitly rather than
			converted to soil_type (default: False)
		:param kappa:
			Float, kappa value, in seconds. Ignored in this GMPE (default: None)
		:param mechanism:
			String, fault mechanism: either "normal", "reverse" or "strike-slip"
			(default: "normal").
		:param damping:
			Float, damping in percent. The only upported value is 5.

		:return:
			Returns a float array with ground-motion intensities
		"""
		imt = imt.upper()
		scale_factor = self.imt_scaling[imt][imt_unit.lower()]

		if not self.has_imt(imt):
			raise IMTUndefinedError(imt)
		if imt == "PGA":
			a1 = self.a1[imt]
			a2 = self.a2[imt]
			a3 = self.a3[imt]
			if not explicit_vs30:
				aB = self.aB[imt]
				aC = self.aC[imt]
				aD = self.aD[imt]
			else:
				bV800 = self.bV800[imt]
			if mechanism == "normal":
				aN = self.aN[imt]
			elif mechanism == "reverse":
				aR = self.aR[imt]
			elif mechanism == "strike-slip":
				aS = self.aS[imt]
			if mechanism in ("normal", "reverse", "strike-slip"):
				sigma = self.SigmaTM[imt]
			else:
				sigma = self.SigmaTU[imt]
		elif imt in ("SD", "SA"):
			if T < self.Tmin(imt) or T > self.Tmax(imt):
				raise PeriodUndefinedError(imt, T)
				#return None
			elif damping not in self.dampings:
				raise DampingNotSupportedError(damping)
			else:
				sa_periods = self.imt_periods["SA"]
				a1 = interpolate(sa_periods, self.a1[imt], [T])
				a2 = interpolate(sa_periods, self.a2[imt], [T])
				a3 = interpolate(sa_periods, self.a3[imt], [T])
				if not explicit_vs30:
					aB = interpolate(sa_periods, self.aB[imt], [T])
					aC = interpolate(sa_periods, self.aC[imt], [T])
					aD = interpolate(sa_periods, self.aD[imt], [T])
				else:
					bV800 = interpolate(sa_periods, self.bV800[imt], [T])
				if mechanism == "normal":
					aN = interpolate(sa_periods, self.aN[imt], [T])
				elif mechanism == "reverse":
					aR = interpolate(sa_periods, self.aR[imt], [T])
				elif mechanism == "strike-slip":
					aS = interpolate(sa_periods, self.aS[imt], [T])
				if mechanism in ("normal", "reverse", "strike-slip"):
					sigma = interpolate(sa_periods, self.SigmaTM[imt], [T])
				else:
					sigma = interpolate(sa_periods, self.SigmaTU[imt], [T])

		## Common terms
		log_ah = a1 + a2*M + a3*np.log10(d) + epsilon * sigma

		## Site term
		## Option 1: soil types or vs30 converted to soil type
		if vs30 != None:
			if vs30 >= 800:
				soil_type = "rock"
			elif 360 <= vs30 < 800:
				soil_type = "typeB"
			elif 180 <= vs30 < 360:
				soil_type = "typeC"
			else:
				soil_type = "typeD"

		if soil_type.lower() in ("rock", "typea"):
			SB, SC, SD = 0., 0., 0.
		elif soil_type.lower() == "typeb":
			SB, SC, SD = 1., 0., 0.
		elif soil_type.lower() == "typec":
			SB, SC, SD = 0., 1., 0.
		elif soil_type.lower() == "typed":
			SB, SC, SD = 0., 0., 1.

		if not explicit_vs30:
			log_ah += aB*SB + aC*SC + aD*SD
		## Option 2: explicit vs30
		else:
			# TODO: explicit vs30 not checked yet
			log_ah += bV800 * np.log10(vs30 / 800.)

		## Fault-mechanism term

		if mechanism == "normal":
			log_ah += aN
		elif mechanism == "reverse":
			log_ah += aR
		elif mechanism == "strike-slip":
			log_ah += aS

		ah = 10**log_ah

		## Convert SD to PSA
		# TODO: needs to be tested!
		if imt == "SA":
			ah = sd2psa(ah, T)
		## Scaling factor depending on imt_unit
		ah *= scale_factor

		return ah

	def log_sigma(self, M=5., d=10., h=0., imt="PGA", T=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
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
			Float, focal depth in km (default: 0). Ignored in this GMPE.
		:param imt:
			String, one of the supported intensity measure types: "PGA",
			"PHV" or "SA" (default: "PGA").
		:param T:
			Float, spectral period of considered IMT. Ignored if IMT is
			"PGA" or "PGV" (default: 0).
		:param soil_type:
			String, either "rock", "stiff" or "soft" (default: "rock"):
				Rock: VS >= 750 m/s
				Stiff soil: 360 <= VS < 750 m/s
				Soft soil: VS < 360 m/s
		:param vs30:
			Float, shear-wave velocity in the upper 30 m (in m/s). If not None,
			it takes precedence over the soil_type parameter (default: None).
		:param explicit_vs30:
			Bool, whether or not vs30 value must be used explicitly rather than
			converted to soil_type (default: False)
		:param kappa:
			Float, kappa value, in seconds. Ignored in this GMPE (default: None)
		:param mechanism:
			String, fault mechanism: either "normal", "reverse" or "strike-slip"
			(default: "normal").
		:param damping:
			Float, damping in percent. The only upported value is 5.
		"""
		imt = imt.upper()
		if not self.has_imt(imt):
			raise IMTUndefinedError(imt)
		if imt == "PGA":
			if not mechanism in ("normal", "reverse", "strike-slip"):
				return self.SigmaTU[imt][0]
			else:
				return self.SigmaTM[imt][0]
		elif imt in ("SD", "SA"):
			if T < self.Tmin(imt) or T > self.Tmax(imt):
				raise PeriodUndefinedError(imt, T)
				#return None
			elif damping not in self.dampings:
				raise DampingNotSupportedError(damping)
			else:
				if not mechanism in ("normal", "reverse", "strike-slip"):
					return interpolate(self.imt_periods[imt], self.SigmaTU[imt], [T])[0]
				else:
					return interpolate(self.imt_periods[imt], self.SigmaTM[imt], [T])[0]

	def plot_Figure4(self, M=5., d=25.):
		"""
		Plot Figure 4 in the original paper of Cauzzi & Faccioli (2008)
		Use as follows:
		for d in [25., 50., 100.]:
			cf2008.plot_Figure4(M)

		:param M:
			Float, moment magnitude (either 5., 5.5, 6. or 7.)
		:param d:
			Float, focal distance in km (either 25., 50. or 100)
		"""
		if M == 5.:
			amax = 0.25
		elif M == 5.5:
			amax = 0.8
		elif M == 6.:
			amax = 2.5
		elif M == 7.:
			amax = 30
		title = "Cauzzi & Faccioli (2008) - Figure 4"
		self.plot_spectrum(mags=[M], d=d, imt="SD", imt_unit="cm", Tmin=0.05, Tmax=20, plot_style="linlin", amin=0, amax=amax, mechanism=None,want_minor_grid=True, title=title, legend_location=1)

	def plot_Figure13(self, M=5.):
		"""
		Plot Figure 13 in the original paper of Cauzzi & Faccioli (2008)

		Note: distances in this figure are Joyner-Boore distances!
		We don't know the conversion used for M=6.
		Still, I can't reproduce the figure.
		However, I can reproduce a figure for PGA in a report by Ameri et al.

		:param M:
			Int, moment magnitude (either 5 or 7)
		"""
		if M == 5:
			dmin, dmax = 11.9, 100.2
			amin, amax = 0.01, 2.
		elif M == 7:
			dmin, dmax = 14.9, 100.6
			amin, amax = 0.1, 5.
		else:
			dmin, dmax = 10., 100.
			amin, amax = 0.01, 5.
		self.plot_distance(mags=[M], dmin=dmin, dmax=dmax, epsilon=1, amin=amin, amax=amax, mechanism="strike-slip", plot_style="loglog", imt_unit="ms2", want_minor_grid=True)

	def plot_Ameri_Figure11(self):
		"""
		Plot Figure 11 in the report by Ameri et al.
		"Strong-motion parameters of the Mw=6.3 Abruzzo (Central Italy) earthquake"
		"""
		self.plot_distance(mags=[6.3], dmin=1., dmax=200, amin=0.01, amax=50, imt_unit="ms2", plot_style="loglog", epsilon=1, want_minor_grid=True)


class AkkarBommer2010GMPE(GMPE):
	"""
	Akkar & Bommer (2010)
		Magnitude scale: MW
		Magnitude range: 5.0 - 7.6
		Distance metric: Joyner-Boore
		Distance range: 0 - 100 km
		Intensity measure types: PGA, PGV, (P)SA
		Original IMT unit: cm/s2
		SA period range: 0.05 - 3 s
		Dampings for SA: 5
		Soil classes:
			rock (vs30 >= 750 m/s)
			stiff (360 <= vs30 < 750 m/s)
			soft (vs30 < 360 m/s)
		Fault types: normal, reverse, strike-slip
	"""
	def __init__(self):
		imt_periods = {}
		imt_periods["PGA"] = np.array([0.])
		imt_periods["PGV"] = np.array([0.])
		imt_periods["SA"] = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95, 3])
		distance_metric = "Joyner-Boore"
		Mmin, Mmax = 5., 7.6
		dmin, dmax = 0., 100.
		Mtype = "MW"
		dampings = [5]
		name = "Akkar&Bommer2010"
		short_name = "AkB_2010"
		GMPE.__init__(self, imt_periods, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings, name, short_name)

		## Coefficients
		self.b1 = {}
		self.b1["PGA"] = np.array([1.04159])
		self.b1["PGV"] = np.array([-2.12833])
		self.b1["SA"] = np.array([2.11528, 2.11994, 1.64489, 0.92065, 0.13978, -0.84006, -1.32207, -1.7032, -1.97201, -2.76925, -3.51672, -3.92759, -4.4949, -4.62925, -4.95053, -5.32863, -5.75799, -5.82689, -5.90592, -6.17066, -6.60337, -6.90379, -6.9618, -6.99236, -6.74613, -6.51719, -6.55821, -6.61945, -6.62737, -6.71787, -6.80776, -6.83632, -6.88684, -6.946, -7.09166, -7.22818, -7.29772, -7.35522, -7.40716, -7.50404, -7.55598, -7.53463, -7.50811, -8.09168, -8.11057, -8.16272, -7.94704, -7.96679, -7.97878, -7.88403, -7.68101, -7.72574, -7.53288, -7.41587, -7.34541, -7.24561, -7.07107, -6.99332, -6.95669, -6.92924])
		self.b2 = {}
		self.b2["PGA"] = np.array([0.91333])
		self.b2["PGV"] = np.array([1.21448])
		self.b2["SA"] = np.array([0.72571, 0.75179, 0.83683, 0.96815, 1.13068, 1.37439, 1.47055, 1.5593, 1.61645, 1.83268, 2.02523, 2.08471, 2.21154, 2.21764, 2.29142, 2.38389, 2.50635, 2.50287, 2.51405, 2.58558, 2.69584, 2.77044, 2.75857, 2.73427, 2.62375, 2.51869, 2.52238, 2.52611, 2.49858, 2.49486, 2.50291, 2.51009, 2.54048, 2.57151, 2.62938, 2.66824, 2.67565, 2.67749, 2.68206, 2.71004, 2.72737, 2.71709, 2.71035, 2.91159, 2.92087, 2.93325, 2.85328, 2.85363, 2.849, 2.81817, 2.7572, 2.82043, 2.74824, 2.69012, 2.65352, 2.61028, 2.56123, 2.52699, 2.51006, 2.45899])
		self.b3 = {}
		self.b3["PGA"] = np.array([-0.0814])
		self.b3["PGV"] = np.array([-0.08137])
		self.b3["SA"] = np.array([-0.07351, -0.07448, -0.07544, -0.07903, -0.08761, -0.10349, -0.10873, -0.11388, -0.11742, -0.13202, -0.14495, -0.14648, -0.15522, -0.15491, -0.15983, -0.16571, -0.17479, -0.17367, -0.17417, -0.17938, -0.18646, -0.19171, -0.1889, -0.18491, -0.17392, -0.1633, -0.16307, -0.16274, -0.1591, -0.15689, -0.15629, -0.15676, -0.15995, -0.16294, -0.16794, -0.17057, -0.17004, -0.16934, -0.16906, -0.1713, -0.17291, -0.17221, -0.17212, -0.1892, -0.19044, -0.19155, -0.18539, -0.18561, -0.18527, -0.1832, -0.17905, -0.18717, -0.18142, -0.17632, -0.17313, -0.16951, -0.16616, -0.16303, -0.16142, -0.15513])
		self.b4 = {}
		self.b4["PGA"] = np.array([-2.92728])
		self.b4["PGV"] = np.array([-2.46942])
		self.b4["SA"] = np.array([-3.33201, -3.10538, -2.75848, -2.49264, -2.33824, -2.19123, -2.12993, -2.12718, -2.16619, -2.12969, -2.04211, -1.88144, -1.79031, -1.798, -1.81321, -1.77273, -1.77068, -1.76295, -1.79854, -1.80717, -1.73843, -1.71109, -1.66588, -1.5912, -1.52886, -1.46527, -1.48223, -1.48257, -1.4331, -1.35301, -1.31227, -1.3326, -1.40931, -1.47676, -1.54037, -1.54273, -1.50936, -1.46988, -1.43816, -1.44395, -1.45794, -1.46662, -1.49679, -1.55644, -1.59537, -1.60461, -1.57428, -1.57833, -1.57728, -1.60381, -1.65212, -1.88782, -1.89525, -1.87041, -1.86079, -1.85612, -1.90422, -1.89704, -1.90132, -1.76801])
		self.b5 = {}
		self.b5["PGA"] = np.array([0.2812])
		self.b5["PGV"] = np.array([0.22349])
		self.b5["SA"] = np.array([0.33534, 0.30253, 0.2549, 0.2179, 0.20089, 0.18139, 0.17485, 0.17137, 0.177, 0.16877, 0.15617, 0.13621, 0.12916, 0.13495, 0.1392, 0.13273, 0.13096, 0.13059, 0.13535, 0.13599, 0.12485, 0.12227, 0.11447, 0.10265, 0.09129, 0.08005, 0.08173, 0.08213, 0.07577, 0.06379, 0.05697, 0.0587, 0.0686, 0.07672, 0.08428, 0.08325, 0.07663, 0.07065, 0.06525, 0.06602, 0.06774, 0.0694, 0.07429, 0.08428, 0.09052, 0.09284, 0.09077, 0.09288, 0.09428, 0.09887, 0.1068, 0.14049, 0.14356, 0.14283, 0.1434, 0.14444, 0.15127, 0.15039, 0.15081, 0.13314])
		self.b6 = {}
		self.b6["PGA"] = np.array([7.86638])
		self.b6["PGV"] = np.array([6.41443])
		self.b6["SA"] = np.array([7.74734, 8.21405, 8.31786, 8.21914, 7.20688, 6.54299, 6.24751, 6.57173, 6.78082, 7.17423, 6.7617, 6.10103, 5.19135, 4.46323, 4.27945, 4.37011, 4.62192, 4.65393, 4.8454, 4.97596, 5.04489, 5.00975, 5.08902, 5.03274, 5.08347, 5.14423, 5.29006, 5.3349, 5.19412, 5.1575, 5.27441, 5.54539, 5.93828, 6.36599, 6.82292, 7.11603, 7.31928, 7.25988, 7.25344, 7.26059, 7.4032, 7.46168, 7.51273, 7.77062, 7.87702, 7.91753, 7.61956, 7.59643, 7.50338, 7.53947, 7.61893, 8.12248, 7.92236, 7.49999, 7.26668, 7.11861, 7.36277, 7.45038, 7.60234, 7.2195])
		self.b7 = {}
		self.b7["PGA"] = np.array([0.08753])
		self.b7["PGV"] = np.array([0.20354])
		self.b7["SA"] = np.array([0.04707, 0.02667, 0.02578, 0.06557, 0.0981, 0.12847, 0.16213, 0.21222, 0.24121, 0.25944, 0.26498, 0.27718, 0.28574, 0.30348, 0.31516, 0.32153, 0.3352, 0.34849, 0.35919, 0.36619, 0.37278, 0.37756, 0.38149, 0.3812, 0.38782, 0.38862, 0.38677, 0.38625, 0.38285, 0.37867, 0.37267, 0.36952, 0.36531, 0.35936, 0.35284, 0.34775, 0.34561, 0.34142, 0.3372, 0.33298, 0.3301, 0.32645, 0.32439, 0.31354, 0.30997, 0.30826, 0.32071, 0.31801, 0.31401, 0.31104, 0.30875, 0.31122, 0.30935, 0.30688, 0.30635, 0.30534, 0.30508, 0.30362, 0.29987, 0.29772])
		self.b8 = {}
		self.b8["PGA"] = np.array([0.01527])
		self.b8["PGV"] = np.array([0.08484])
		self.b8["SA"] = np.array([-0.02426, -0.00062, 0.01703, 0.02105, 0.03919, 0.0434, 0.06695, 0.09201, 0.11675, 0.13562, 0.14446, 0.15156, 0.15239, 0.15652, 0.16333, 0.17366, 0.1848, 0.19061, 0.19411, 0.19519, 0.19461, 0.19423, 0.19402, 0.19309, 0.19392, 0.19273, 0.19082, 0.19285, 0.19161, 0.18812, 0.18568, 0.18149, 0.17617, 0.17301, 0.16945, 0.16743, 0.1673, 0.16325, 0.16171, 0.15839, 0.15496, 0.15337, 0.15264, 0.1443, 0.1443, 0.14412, 0.14321, 0.14301, 0.14324, 0.14332, 0.14343, 0.14255, 0.14223, 0.14074, 0.14052, 0.13923, 0.13933, 0.13776, 0.13584, 0.13198])
		self.b9 = {}
		self.b9["PGA"] = np.array([-0.04189])
		self.b9["PGV"] = np.array([-0.05856])
		self.b9["SA"] = np.array([-0.0426, -0.04906, -0.04184, -0.02098, -0.04853, -0.05554, -0.04722, -0.05145, -0.05202, -0.04283, -0.04259, -0.03853, -0.03423, -0.04146, -0.0405, -0.03946, -0.03786, -0.02884, -0.02209, -0.02269, -0.02613, -0.02655, -0.02088, -0.01623, -0.01826, -0.01902, -0.01842, -0.01607, -0.01288, -0.01208, -0.00845, -0.00533, -0.00852, -0.01204, -0.01386, -0.01402, -0.01526, -0.01563, -0.01848, -0.02258, -0.02626, -0.0292, -0.03484, -0.03985, -0.04155, -0.04238, -0.04963, -0.0491, -0.04812, -0.0471, -0.04607, -0.05106, -0.05024, -0.04887, -0.04743, -0.04731, -0.04522, -0.04203, -0.03863, -0.03855])
		self.b10 = {}
		self.b10["PGA"] = np.array([0.08015])
		self.b10["PGV"] = np.array([0.01305])
		self.b10["SA"] = np.array([0.08649, 0.0791, 0.0784, 0.08438, 0.08577, 0.09221, 0.09003, 0.09903, 0.09943, 0.08579, 0.06945, 0.05932, 0.05111, 0.04661, 0.04253, 0.03373, 0.02867, 0.02475, 0.02502, 0.02121, 0.01115, 0.0014, 0.00148, 0.00413, 0.00413, -0.00369, -0.00897, -0.00876, -0.00564, -0.00215, -0.00047, -0.00006, -0.00301, -0.00744, -0.01387, -0.01492, -0.01192, -0.00703, -0.00351, -0.00486, -0.00731, -0.00871, -0.01225, -0.01927, -0.02322, -0.02626, -0.02342, -0.0257, -0.02643, -0.02769, -0.02819, -0.02966, -0.0293, -0.02963, -0.02919, -0.02751, -0.02776, -0.02615, -0.02487, -0.02469])
		self.SigmaIntraEvent = {}
		self.SigmaIntraEvent["PGA"] = np.array([0.261])
		self.SigmaIntraEvent["PGV"] = np.array([0.2562])
		self.SigmaIntraEvent["SA"] = np.array([0.272, 0.2728, 0.2788, 0.2821, 0.2871, 0.2902, 0.2983, 0.2998, 0.3037, 0.3078, 0.307, 0.3007, 0.3004, 0.2978, 0.2973, 0.2927, 0.2917, 0.2915, 0.2912, 0.2895, 0.2888, 0.2896, 0.2871, 0.2878, 0.2863, 0.2869, 0.2885, 0.2875, 0.2857, 0.2839, 0.2845, 0.2844, 0.2841, 0.284, 0.284, 0.2834, 0.2828, 0.2826, 0.2832, 0.2835, 0.2836, 0.2832, 0.283, 0.283, 0.283, 0.2829, 0.2815, 0.2826, 0.2825, 0.2818, 0.2818, 0.2838, 0.2845, 0.2854, 0.2862, 0.2867, 0.2869, 0.2874, 0.2872, 0.2876])
		self.SigmaInterEvent = {}
		self.SigmaInterEvent["PGA"] = np.array([0.0994])
		self.SigmaInterEvent["PGV"] = np.array([0.1083])
		self.SigmaInterEvent["SA"] = np.array([0.1142, 0.1167, 0.1192, 0.1081, 0.099, 0.0976, 0.1054, 0.1101, 0.1123, 0.1163, 0.1274, 0.143, 0.1546, 0.1626, 0.1602, 0.1584, 0.1543, 0.1521, 0.1484, 0.1483, 0.1465, 0.1427, 0.1435, 0.1439, 0.1453, 0.1427, 0.1428, 0.1458, 0.1477, 0.1468, 0.145, 0.1457, 0.1503, 0.1537, 0.1558, 0.1582, 0.1592, 0.1611, 0.1642, 0.1657, 0.1665, 0.1663, 0.1661, 0.1627, 0.1627, 0.1633, 0.1632, 0.1645, 0.1665, 0.1681, 0.1688, 0.1741, 0.1759, 0.1772, 0.1783, 0.1794, 0.1788, 0.1784, 0.1783, 0.1785])
		self.SigmaTot = {}
		self.SigmaTot["PGA"] = np.array([0.279287236])
		self.SigmaTot["PGV"] = np.array([0.278])
		self.SigmaTot["SA"] = np.array([0.295001085, 0.296713212, 0.303212928, 0.302102665, 0.303689661, 0.306172827, 0.316373276, 0.319377598, 0.323797746, 0.329038797, 0.332384958, 0.332970704, 0.337848072, 0.339298688, 0.337714865, 0.332812034, 0.32999603, 0.328795772, 0.326833291, 0.325273946, 0.323832812, 0.322848958, 0.320965201, 0.321770182, 0.321060399, 0.320429243, 0.321906959, 0.322356774, 0.321620553, 0.319608276, 0.319319981, 0.319549448, 0.321407685, 0.32292366, 0.323928449, 0.324565556, 0.32453117, 0.325293667, 0.327358947, 0.328372867, 0.328863513, 0.328417311, 0.328143581, 0.326435736, 0.326435736, 0.326648588, 0.325386678, 0.326990841, 0.327915385, 0.328129319, 0.328488478, 0.332946317, 0.334486263, 0.335936006, 0.337196278, 0.338202972, 0.338054803, 0.338268119, 0.338045456, 0.338490783])

		## Unit conversion
		## Original unit is cm/s2 (gal)
		self.imt_scaling = {}
		self.imt_scaling["PGA"] = {"g": 0.01/g, "mg": 10./g, "ms2": 1E-2, "gal": 1.0, "cms2": 1.0}
		self.imt_scaling["PGV"] = {"ms": 1E-2, "cms": 1.0}
		self.imt_scaling["SA"] = self.imt_scaling["PGA"]

	def __call__(self, M, d, h=0, imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		Return ground motion for given magnitude, distance, depth, soil type,
		and fault mechanism.

		:param M:
			Float or float array, magnitude(s).
			Note that if d is an array, M must be a float.
		:param d:
			Float or float array, distance(s) in km.
			Note that if M is an array, d must be a float.
		:param h:
			Float, focal depth in km (default: 0). Ignored in this GMPE.
		:param imt:
			String, one of the supported intensity measure types: "PGA",
			"PHV" or "SA" (default: "PGA").
		:param T:
			Float, spectral period of considered IMT. Ignored if IMT is
			"PGA" or "PGV" (default: 0).
		:param imt_unit:
			String, unit in which intensities should be expressed (default: "g")
		:param epsilon:
			Float, number of standard deviations above or below the mean
			(default: 0).
		:param soil_type:
			String, either "rock", "stiff" or "soft" (default: "rock"):
				Rock: VS >= 750 m/s
				Stiff soil: 360 <= VS < 750 m/s
				Soft soil: VS < 360 m/s
		:param vs30:
			Float, shear-wave velocity in the upper 30 m (in m/s). If not None,
			it takes precedence over the soil_type parameter (default: None).
		:param kappa:
			Float, kappa value, in seconds. Ignored in this GMPE (default: None)
		:param mechanism:
			String, fault mechanism: either "normal", "reverse" or "strike-slip"
			(default: "normal").
		:param damping:
			Float, damping in percent. The only upported value is 5.

		:return:
			Returns a float array with ground-motion intensities
		"""
		imt = imt.upper()
		scale_factor = self.imt_scaling[imt][imt_unit.lower()]

		if not self.has_imt(imt):
			raise IMTUndefinedError(imt)
		if imt in ("PGA", "PGV"):
			b1 = self.b1[imt]
			b2 = self.b2[imt]
			b3 = self.b3[imt]
			b4 = self.b4[imt]
			b5 = self.b5[imt]
			b6 = self.b6[imt]
			b7 = self.b7[imt]
			b8 = self.b8[imt]
			b9 = self.b9[imt]
			b10 = self.b10[imt]
			sigma = self.SigmaTot[imt]
		else:
			if T < self.Tmin(imt) or T > self.Tmax(imt):
				raise PeriodUndefinedError(imt, T)
				#return None
			elif damping not in self.dampings:
				raise DampingNotSupportedError(damping)
			else:
				sa_periods = self.imt_periods["SA"]
				b1 = interpolate(sa_periods, self.b1[imt], [T])
				b2 = interpolate(sa_periods, self.b2[imt], [T])
				b3 = interpolate(sa_periods, self.b3[imt], [T])
				b4 = interpolate(sa_periods, self.b4[imt], [T])
				b5 = interpolate(sa_periods, self.b5[imt], [T])
				b6 = interpolate(sa_periods, self.b6[imt], [T])
				b7 = interpolate(sa_periods, self.b7[imt], [T])
				b8 = interpolate(sa_periods, self.b8[imt], [T])
				b9 = interpolate(sa_periods, self.b9[imt], [T])
				b10 = interpolate(sa_periods, self.b10[imt], [T])
				sigma = interpolate(sa_periods, self.SigmaTot[imt], [T])

		if vs30 != None:
			if vs30 >= 750:
				soil_type = "rock"
			elif 360 <= vs30 < 750:
				soil_type = "stiff"
			elif vs30 < 360:
				soil_type = "soft"

		if soil_type.lower() == "soft":
			SS, SA = 1, 0
		elif soil_type.lower() == "stiff":
			SS, SA = 0, 1
		else:
			# default is rock
			SS, SA = 0, 0

		if mechanism.lower() == "normal":
			FN, FR = 1, 0
		elif mechanism.lower() == "reverse":
			FN, FR = 0, 1
		elif mechanism.lower() == "strike-slip":
			FN, FR = 0, 0

		log_ah = b1 + b2*M + b3*M*M + (b4 + b5*M)*np.log10(np.sqrt(d*d + b6*b6)) + b7*SS + b8*SA + b9*FN + b10*FR + epsilon * sigma
		ah = (10**log_ah)
		ah *= scale_factor

		return ah

	def log_sigma(self, M=5., d=10., h=0., imt="PGA", T=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
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
			Float, focal depth in km (default: 0). Ignored in this GMPE.
		:param imt:
			String, one of the supported intensity measure types: "PGA",
			"PHV" or "SA" (default: "PGA").
		:param T:
			Float, spectral period of considered IMT. Ignored if IMT is
			"PGA" or "PGV" (default: 0).
		:param soil_type:
			String, either "rock", "stiff" or "soft" (default: "rock"):
				Rock: VS >= 750 m/s
				Stiff soil: 360 <= VS < 750 m/s
				Soft soil: VS < 360 m/s
		:param vs30:
			Float, shear-wave velocity in the upper 30 m (in m/s). If not None,
			it takes precedence over the soil_type parameter (default: None).
		:param kappa:
			Float, kappa value, in seconds. Ignored in this GMPE (default: None)
		:param mechanism:
			String, fault mechanism: either "normal", "reverse" or "strike-slip"
			(default: "normal").
		:param damping:
			Float, damping in percent. The only upported value is 5.
		"""
		imt = imt.upper()
		if not self.has_imt(imt):
			raise IMTUndefinedError(imt)
		if imt in ("PGA", "PGV"):
			return self.SigmaTot[imt][0]
		else:
			if T < self.Tmin(imt) or T > self.Tmax(imt):
				raise PeriodUndefinedError(imt, T)
				#return None
			elif damping not in self.dampings:
				raise DampingNotSupportedError(damping)
			else:
				return interpolate(self.imt_periods[imt], self.SigmaTot[imt], [T])[0]

	def is_rake_dependent(self):
		"""
		Indicate whether or not GMPE depends on rake of the source
		"""
		return True

	def get_CRISIS_periods(self):
		"""
		Return array of max. 40 spectral periods to be used with CRISIS
		"""
		return self.imt_periods["SA"]

	def plot_Figure9(self):
		"""
		Plot Figure 9 in the paper of Akkar & Bommer (2010)
		"""
		self.plot_spectrum(mags=[5., 6.3, 7.6], d=10, plot_style="lin", plot_freq=False, Tmin=0, Tmax=3, epsilon=0, amin=0, amax=625, imt_unit="cms2", include_pgm=False, soil_type="rock", mechanism="strike-slip", want_minor_grid=False, legend_location=1)

	def plot_Figure10(self):
		"""
		Plot Figure 10 in the paper of Akkar & Bommer (2010)
		"""
		self.plot_spectrum(mags=[5., 6.3, 7.6], d=10, plot_style="lin", plot_freq=False, Tmin=0, Tmax=3, epsilon=1, amin=0, amax=1250, imt_unit="cms2", include_pgm=False, soil_type="rock", mechanism="strike-slip", want_minor_grid=False, legend_location=1)

	def plot_Figure11(self, soil_type="rock", mechanism="strike-slip"):
		"""
		Plot Figure 11 in the paper of Akkar & Bommer (2010)

		:param soil_type:
			String, either "rock", "stiff" or "soft" (default: "rock")
		:param mechanism:
			String, fault mechanism: either "normal", "reverse" or "strike-slip"
			(default: strike-slip)
		"""
		self.plot_distance(mags=[5., 7.6], dmin=1, dmax=1E2, amin=1E-1, amax=100, imt="PGV", imt_unit="cms", plot_style="loglog", soil_type=soil_type, mechanism=mechanism, want_minor_grid=True)


class BommerEtAl2011GMPE(AkkarBommer2010GMPE):
	"""
	Bommer et al. (2011)
	This GMPE is based on Akkar & Bommer (2010), but extended to higher
	frequencies. Coefficients for PGA have changed as well.
	The coefficients have been copied from the opensha package in
	OpenQuake, where this GMPE is known as Akb_2010.

		Magnitude scale: MW
		Magnitude range: 5 - 7.6
		Distance metric: Joyner-Boore
		Distance range: 0 - 100 km
		Intensity measure types: PGA, PGV, (P)SA
		Original IMT unit: cm/s2
		SA period range: 0.01 - 4 s
		Dampings for SA: 5
		Soil classes:
			rock (vs30 >= 750 m/s)
			stiff (360 <= vs30 < 750 m/s)
			soft (vs30 < 360 m/s)
		Fault types: normal, reverse, strike-slip

	Note: In OpenQuake, the period range is also extended to 4 s, using
	the factor T3sec_TO_T4sec_DECAYFACTOR = 0.8. This is not part of the
	original publication, and is not implemented here.
	"""
	def __init__(self):
		AkkarBommer2010GMPE.__init__(self)

		## Override name
		self.name = "BommerEtAl2011"
		self.short_name = "B_2011"

		## Override SA periods and coefficients
		self.imt_periods["SA"] = np.array([0.01000, 0.02000, 0.03000, 0.04000, 0.05000, 0.10000, 0.15000, 0.20000, 0.25000, 0.30000, 0.35000, 0.40000, 0.45000, 0.50000, 0.55000, 0.60000, 0.65000, 0.70000, 0.75000, 0.80000, 0.85000, 0.90000, 0.95000, 1.00000, 1.05000, 1.10000, 1.15000, 1.20000, 1.25000, 1.30000, 1.35000, 1.40000, 1.45000, 1.50000, 1.55000, 1.60000, 1.65000, 1.70000, 1.75000, 1.80000, 1.85000, 1.90000, 1.95000, 2.00000, 2.05000, 2.10000, 2.15000, 2.20000, 2.25000, 2.30000, 2.35000, 2.40000, 2.45000, 2.50000, 2.55000, 2.60000, 2.65000, 2.70000, 2.75000, 2.80000, 2.85000, 2.90000, 2.95000, 3.00000])

		## Coefficients
		self.b1["PGA"] = np.array([1.43525])
		self.b1["SA"] = np.array([1.43153, 1.48690, 1.64821, 2.08925, 2.49228, 2.11994, 1.64489, 0.92065, 0.13978, -0.84006, -1.32207, -1.70320, -1.97201, -2.76925, -3.51672, -3.92759, -4.49490,-4.62925, -4.95053, -5.32863, -5.75799, -5.82689, -5.90592, -6.17066, -6.60337, -6.90379, -6.96180,-6.99236, -6.74613, -6.51719, -6.55821, -6.61945, -6.62737, -6.71787, -6.80776, -6.83632, -6.88684,-6.94600, -7.09166, -7.22818, -7.29772, -7.35522, -7.40716, -7.50404, -7.55598, -7.53463, -7.50811,-8.09168, -8.11057, -8.16272, -7.94704, -7.96679, -7.97878, -7.88403, -7.68101, -7.72574, -7.53288,-7.41587, -7.34541, -7.24561, -7.07107, -6.99332, -6.95669, -6.92924])
		self.b2["PGA"] = np.array([0.74866])
		self.b2["SA"] = np.array([0.75258, 0.75966, 0.73507, 0.65032, 0.58575, 0.75179, 0.83683, 0.96815, 1.13068, 1.37439, 1.47055, 1.55930, 1.61645, 1.83268, 2.02523, 2.08471, 2.21154, 2.21764, 2.29142, 2.38389, 2.50635, 2.50287, 2.51405, 2.58558, 2.69584, 2.77044, 2.75857, 2.73427, 2.62375, 2.51869, 2.52238, 2.52611, 2.49858, 2.49486, 2.50291, 2.51009, 2.54048, 2.57151, 2.62938, 2.66824, 2.67565, 2.67749, 2.68206, 2.71004, 2.72737, 2.71709, 2.71035, 2.91159, 2.92087, 2.93325, 2.85328, 2.85363, 2.84900, 2.81817, 2.75720, 2.82043, 2.74824, 2.69012, 2.65352, 2.61028, 2.56123, 2.52699, 2.51006, 2.45899])
		self.b3["PGA"] = np.array([-0.06520])
		self.b3["SA"] = np.array([-0.06557, -0.06767, -0.06700, -0.06218, -0.06043, -0.07448, -0.07544, -0.07903, -0.08761, -0.10349, -0.10873, -0.11388,-0.11742, -0.13202, -0.14495, -0.14648, -0.15522, -0.15491, -0.15983, -0.16571, -0.17479, -0.17367,-0.17417, -0.17938, -0.18646, -0.19171, -0.18890, -0.18491, -0.17392, -0.16330, -0.16307, -0.16274,-0.15910, -0.15689, -0.15629, -0.15676, -0.15995, -0.16294, -0.16794, -0.17057, -0.17004, -0.16934,-0.16906, -0.17130, -0.17291, -0.17221, -0.17212, -0.18920, -0.19044, -0.19155, -0.18539, -0.18561,-0.18527, -0.18320, -0.17905, -0.18717, -0.18142, -0.17632, -0.17313, -0.16951, -0.16616, -0.16303,-0.16142,-0.15513])
		self.b4["PGA"] = np.array([-2.72950])
		self.b4["SA"] = np.array([-2.73290, -2.82146, -2.89764, -3.02618, -3.20215, -3.10538, -2.75848, -2.49264, -2.33824, -2.19123, -2.12993, -2.12718, -2.16619, -2.12969, -2.04211, -1.88144, -1.79031, -1.79800, -1.81321, -1.77273, -1.77068, -1.76295, -1.79854, -1.80717, -1.73843, -1.71109, -1.66588, -1.59120, -1.52886, -1.46527, -1.48223, -1.48257, -1.43310, -1.35301, -1.31227, -1.33260, -1.40931, -1.47676, -1.54037, -1.54273, -1.50936, -1.46988, -1.43816, -1.44395, -1.45794, -1.46662, -1.49679, -1.55644, -1.59537, -1.60461, -1.57428, -1.57833, -1.57728, -1.60381, -1.65212, -1.88782, -1.89525, -1.87041, -1.86079, -1.85612, -1.90422, -1.89704, -1.90132, -1.76801])
		self.b5["PGA"] = np.array([0.25139])
		self.b5["SA"] = np.array([0.25170, 0.26510, 0.27607, 0.28999, 0.31485, 0.30253, 0.25490, 0.21790, 0.20089, 0.18139, 0.17485, 0.17137, 0.17700, 0.16877, 0.15617, 0.13621, 0.12916, 0.13495, 0.13920, 0.13273, 0.13096, 0.13059, 0.13535, 0.13599, 0.12485, 0.12227, 0.11447, 0.10265, 0.09129, 0.08005, 0.08173, 0.08213, 0.07577, 0.06379, 0.05697, 0.05870, 0.06860, 0.07672, 0.08428, 0.08325, 0.07663, 0.07065, 0.06525, 0.06602, 0.06774, 0.06940, 0.07429, 0.08428, 0.09052, 0.09284, 0.09077, 0.09288, 0.09428, 0.09887, 0.10680, 0.14049, 0.14356, 0.14283, 0.14340, 0.14444, 0.15127, 0.15039, 0.15081, 0.13314])
		self.b6["PGA"] = np.array([7.74959])
		self.b6["SA"] = np.array([7.73304, 7.20661, 6.87179, 7.42328, 7.75532, 8.21405, 8.31786, 8.21914, 7.20688, 6.54299, 6.24751, 6.57173, 6.78082, 7.17423, 6.76170, 6.10103, 5.19135, 4.46323, 4.27945, 4.37011, 4.62192, 4.65393, 4.84540, 4.97596, 5.04489, 5.00975, 5.08902, 5.03274, 5.08347, 5.14423, 5.29006, 5.33490, 5.19412, 5.15750, 5.27441, 5.54539, 5.93828, 6.36599, 6.82292, 7.11603, 7.31928, 7.25988, 7.25344, 7.26059, 7.40320, 7.46168, 7.51273, 7.77062,  7.87702, 7.91753, 7.61956, 7.59643, 7.50338, 7.53947, 7.61893, 8.12248, 7.92236, 7.49999, 7.26668, 7.11861, 7.36277, 7.45038, 7.60234, 7.21950])
		self.b7["PGA"] = np.array([0.08320])
		self.b7["SA"] = np.array([0.08105, 0.07825, 0.06376, 0.05045, 0.03798, 0.02667, 0.02578, 0.06557, 0.09810, 0.12847, 0.16213, 0.21222, 0.24121, 0.25944, 0.26498, 0.27718, 0.28574, 0.30348, 0.31516, 0.32153, 0.33520, 0.34849, 0.35919, 0.36619, 0.37278, 0.37756, 0.38149, 0.38120, 0.38782, 0.38862, 0.38677, 0.38625, 0.38285, 0.37867, 0.37267, 0.36952, 0.36531, 0.35936, 0.35284, 0.34775, 0.34561, 0.34142, 0.33720, 0.33298, 0.33010, 0.32645, 0.32439, 0.31354, 0.30997, 0.30826, 0.32071, 0.31801, 0.31401, 0.31104, 0.30875, 0.31122, 0.30935, 0.30688, 0.30635, 0.30534, 0.30508, 0.30362, 0.29987, 0.29772])
		self.b8["PGA"] = np.array([0.00766])
		self.b8["SA"] = np.array([0.00745, 0.00618,-0.00528,-0.02091,-0.03143,-0.00062, 0.01703, 0.02105, 0.03919, 0.04340, 0.06695, 0.09201, 0.11675, 0.13562, 0.14446, 0.15156, 0.15239, 0.15652, 0.16333, 0.17366, 0.18480, 0.19061, 0.19411, 0.19519, 0.19461, 0.19423, 0.19402, 0.19309, 0.19392, 0.19273, 0.19082, 0.19285, 0.19161, 0.18812, 0.18568, 0.18149, 0.17617, 0.17301, 0.16945, 0.16743, 0.16730, 0.16325, 0.16171, 0.15839, 0.15496, 0.15337, 0.15264, 0.14430, 0.14430, 0.14412, 0.14321, 0.14301, 0.14324, 0.14332, 0.14343, 0.14255, 0.14223, 0.14074, 0.14052, 0.13923, 0.13933, 0.13776, 0.13584, 0.13198])
		self.b9["PGA"] = np.array([-0.05823])
		self.b9["SA"] = np.array([-0.05886, -0.06111, -0.06189, -0.06278, -0.06708, -0.04906, -0.04184, -0.02098, -0.04853, -0.05554, -0.04722, -0.05145, -0.05202, -0.04283, -0.04259, -0.03853, -0.03423, -0.04146, -0.04050, -0.03946, -0.03786, -0.02884, -0.02209, -0.02269, -0.02613, -0.02655, -0.02088, -0.01623, -0.01826, -0.01902, -0.01842, -0.01607, -0.01288, -0.01208, -0.00845, -0.00533, -0.00852, -0.01204, -0.01386, -0.01402, -0.01526, -0.01563, -0.01848, -0.02258, -0.02626, -0.02920, -0.03484, -0.03985, -0.04155, -0.04238, -0.04963, -0.04910, -0.04812, -0.04710, -0.04607, -0.05106, -0.05024, -0.04887, -0.04743, -0.04731, -0.04522, -0.04203, -0.03863, -0.03855])
		self.b10["PGA"] = np.array([0.07087])
		self.b10["SA"] = np.array([0.07169, 0.06756, 0.06529, 0.05935, 0.06382, 0.07910, 0.07840, 0.08438, 0.08577, 0.09221, 0.09003, 0.09903, 0.09943, 0.08579, 0.06945, 0.05932, 0.05111, 0.04661, 0.04253, 0.03373, 0.02867, 0.02475, 0.02502, 0.02121, 0.01115, 0.00140, 0.00148, 0.00413, 0.00413, -0.00369, -0.00897, -0.00876, -0.00564, -0.00215, -0.00047, -0.00006, -0.00301, -0.00744, -0.01387, -0.01492, -0.01192, -0.00703, -0.00351, -0.00486, -0.00731, -0.00871, -0.01225, -0.01927, -0.02322, -0.02626, -0.02342, -0.02570, -0.02643, -0.02769, -0.02819, -0.02966, -0.02930, -0.02963, -0.02919, -0.02751, -0.02776, -0.02615, -0.02487, -0.02469])
		self.SigmaIntraEvent["PGA"] = np.array([0.26110])
		self.SigmaIntraEvent["SA"] = np.array([0.26160, 0.26350, 0.26750, 0.27090, 0.27280, 0.27280, 0.27880, 0.28210, 0.28710, 0.29020, 0.29830, 0.29980, 0.30370, 0.30780, 0.30700, 0.30070, 0.30040, 0.29780, 0.29730, 0.29270, 0.29170, 0.29150, 0.29120, 0.28950, 0.28880, 0.28960, 0.28710, 0.28780, 0.28630, 0.28690, 0.28850, 0.28750, 0.28570, 0.28390, 0.28450, 0.28440, 0.28410, 0.28400, 0.28400, 0.28340, 0.28280, 0.28260, 0.28320, 0.28350, 0.28360, 0.28320, 0.28300, 0.28300, 0.28300, 0.28290, 0.28150, 0.28260, 0.28250, 0.28180, 0.28180, 0.28380, 0.28450, 0.28540, 0.28620, 0.28670, 0.28690, 0.28740, 0.28720, 0.28760])
		self.SigmaInterEvent["PGA"] = np.array([0.10560])
		self.SigmaInterEvent["SA"] = np.array([0.10510, 0.11140, 0.11370, 0.11520, 0.11810, 0.11670, 0.11920, 0.10810, 0.09900, 0.09760, 0.10540, 0.11010, 0.11230, 0.11630, 0.12740, 0.14300, 0.15460, 0.16260, 0.16020, 0.15840, 0.15430, 0.15210, 0.14840, 0.14830, 0.14650, 0.14270, 0.14350, 0.14390, 0.14530, 0.14270, 0.14280, 0.14580, 0.14770, 0.14680, 0.14500, 0.14570, 0.15030, 0.15370, 0.15580, 0.15820, 0.15920, 0.16110, 0.16420, 0.16570, 0.16650, 0.16630, 0.16610, 0.16270, 0.16270, 0.16330, 0.16320, 0.16450, 0.16650, 0.16810, 0.16880, 0.17410, 0.17590, 0.17720, 0.17830, 0.17940, 0.17880, 0.17840, 0.17830, 0.17850])
		self.SigmaTot["PGA"] = np.array([0.28160])
		self.SigmaTot["SA"] = np.array([0.28192, 0.28608, 0.29066, 0.29438, 0.29727, 0.29671, 0.30321, 0.30210, 0.30369, 0.30617, 0.31637, 0.31938, 0.32380, 0.32904, 0.33238, 0.33297, 0.33785, 0.33930, 0.33771, 0.33281, 0.33000, 0.32880, 0.32683, 0.32527, 0.32383, 0.32280, 0.32097, 0.32177, 0.32106, 0.32043, 0.32191, 0.32236, 0.32162, 0.31961, 0.31932, 0.31950, 0.32141, 0.32292, 0.32393, 0.32457, 0.32453, 0.32529, 0.32736, 0.32837, 0.32886, 0.32842, 0.32814, 0.32644, 0.32644, 0.32665, 0.32539, 0.32699, 0.32792, 0.32813, 0.32849, 0.33290, 0.33449, 0.33594, 0.33720, 0.33820, 0.33810, 0.33827, 0.33800, 0.33849])

		# TODO: this factor comes from OpenQuake, but is not used currently
		self.T3sec_TO_T4sec_DECAYFACTOR = 0.8

	def plot_Figure4a(self, soil_type="stiff", mechanism="strike-slip"):
		"""
		Plot Figure 4a in the paper of Bommer et al. (2011)

		:param soil_type:
			String, either "rock", "stiff" or "soft" (default: "stiff")
		:param mechanism:
			String, fault mechanism: either "normal", "reverse" or "strike-slip"
			(default: strike-slip)
		"""
		self.plot_spectrum(mags=[5.5, 6.5, 7.5], d=10, plot_style="loglin", plot_freq=False, Tmin=0.01, Tmax=3, epsilon=0, amin=0, amax=0.8, imt_unit="g", include_pgm=True, soil_type=soil_type, mechanism=mechanism, want_minor_grid=False, legend_location=0)

	def plot_Figure4b(self, d=5., soil_type="rock", mechanism="reverse"):
		"""
		Plot Figure 4b in the paper of Bommer et al. (2011)

		:param d:
			Float, horizontal distance in km (default: 5.)
		:param soil_type:
			String, either "rock", "stiff" or "soft" (default: "stiff")
		:param mechanism:
			String, fault mechanism: either "normal", "reverse" or "strike-slip"
			(default: strike-slip)
		"""
		self.plot_spectrum(mags=[6.5], d=d, plot_style="loglin", plot_freq=False, Tmin=0.01, Tmax=3, epsilon=0, amin=0, amax=1.0, imt_unit="g", include_pgm=True, soil_type=soil_type, mechanism=mechanism, want_minor_grid=False, legend_location=0)


AkkarBommer2010SHAREGMPE = BommerEtAl2011GMPE

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
		Damping for PSV: 2 percent [0, 5, and 10 are also in original report, but not in 1977 publication]
		Soil classes: None
		Fault types: None
	"""
	def __init__(self):
		psv_freqs = np.array([0.125, 0.15, 0.2, 0.25, 0.32, 0.5, 0.64, 1., 1.3, 1.8, 2., 2.5, 3.2, 5., 6.7, 10.])
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
		GMPE.__init__(self, imt_periods, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings, name, short_name)

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
		self.b["PSV"] = np.array([0.4, 0.45, 0.55, 0.53, 0.45, 0.44, 0.4, 0.36, 0.38, 0.4, 0.35, 0.34, 0.3, 0.25, 0.17, 0.24])

		self.c = {}
		self.c["PGA"] = np.array([1.301])
		self.c["PGV"] = np.array([1.202])
		self.c["PGD"] = np.array([0.885])
		self.c["PSV"] = np.array([0.7, 0.68, 0.92, 0.72, 0.62, 0.5, 0.45, 0.58, 0.9, 1.2, 1.16, 1.2, 1.32, 1.25, 1.32,1.3])

		## Unit conversion
		## PSV: cm/sec, PGD: cm
		self.imt_scaling = {}
		self.imt_scaling["PGA"] = {"g": 0.01/g, "mg": 10./g, "ms2": 1E-2, "gal": 1.0}
		self.imt_scaling["PGV"] = {"ms": 1E-2, "cms": 1.0}
		self.imt_scaling["PGD"] = {"m": 1E-2, "cm": 1.0}
		self.imt_scaling["PSV"] = self.imt_scaling["PGV"]

	def __call__(self, M, d, h=0, imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type=None, vs30=None, kappa=None, mechanism="normal", damping=2):
		"""
		Return ground motion for given magnitude, distance, depth, soil type,
		and fault mechanism.

		:param M:
			Float or float array, magnitude(s).
			Note that if d is an array, M must be a float.
		:param d:
			Float or float array, distance(s) in km.
			Note that if M is an array, d must be a float.
		:param h:
			Float, focal depth in km (default: 0., i.e. assume d is hypocentral
			distance).
		:param imt:
			String, one of the supported intensity measure types: "PGA", "PGV",
			"PGD" or "PSV"
			(default: "PGA").
		:param T:
			Float, spectral period of considered IMT. Only relevant if IMT ==
			"PSV" (default: 0).
		:param imt_unit:
			String, unit in which intensities should be expressed (default: "g")
		:param epsilon:
			Float, number of standard deviations above or below the mean.
			Ignored because this GMPE does not specify a standard deviation.
			(default: 0).
		:param soil_type:
			String, zoil type. Ignored in this GMPE (default: "rock").
		:param vs30:
			Float, shear-wave velocity in the upper 30 m (in m/s). Ignored.
			(default: None).
		:param kappa:
			Float, kappa value, in seconds. Ignored in this GMPE (default: None)
		:param mechanism:
			String, fault mechanism: either "normal", "reverse" or "strike-slip".
			Ignored in this GMPE (default: "normal".
		:param damping:
			Float, damping in percent. The only supported value is 2 (although
			the paper also mentions undamped, 5, and 10) (default: 2).

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

	def log_sigma(self, M=5., d=10., h=0., imt="PGA", T=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
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
			Float, focal depth in km (default: 0., i.e. assume d is hypocentral
			distance).
		:param imt:
			String, one of the supported intensity measure types: "PGA", "PGV",
			"PGD" or "PSV"
			(default: "PGA").
		:param T:
			Float, spectral period of considered IMT. Only relevant if IMT ==
			"PSV" (default: 0).
		:param soil_type:
			String, zoil type. Ignored in this GMPE (default: "rock").
		:param vs30:
			Float, shear-wave velocity in the upper 30 m (in m/s). Ignored.
			(default: None).
		:param kappa:
			Float, kappa value, in seconds. Ignored in this GMPE (default: None)
		:param mechanism:
			String, fault mechanism: either "normal", "reverse" or "strike-slip".
			Ignored in this GMPE (default: "normal".
		:param damping:
			Float, damping in percent. The only supported value is 2 (although
			the paper also mentions undamped, 5, and 10) (default: 2).
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


class NhlibGMPE(GMPE):
	"""
	Class to implement GMPEs from Nhlib into this module.
	"""

	IMT_DICT = {"PGD": PGD, "PGV": PGV, "PGA": PGA, "SA": SA}

	def __init__(self, name, short_name, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings=[5], imt_periods=None):
		"""
		:param name:
			String, defines name of GMPE. The names of all available GMPEs in
			Nhlib can be retrieved by nhlib.gsim.get_available_gsims().keys().
		:param imt_periods:
			Dict, String: List of Floats, mapping name of imt (PGD, PGV, PGA or
			SA) to periods (Default: None). Must only be provided if spectral
			periods cannot be retrieved from COEFFS attribute of GMPE.

		See class GMPE for other params.
		"""
		## get dict mapping gmpe names to classes
		self.gmpe = nhlib.gsim.get_available_gsims()[name]

		## get imt periods
		if not imt_periods:
			imt_periods = {}
			for imt in self.gmpe.DEFINED_FOR_INTENSITY_MEASURE_TYPES:
				if imt == SA:
					if name in ("AtkinsonBoore2006", "AtkinsonBoore2006Prime"):
						imt_periods['SA'] = [sa.period for sa in sorted(self.gmpe.COEFFS_BC.sa_coeffs.keys())]
					elif "adjusted" in name:
						(vs30, kappa) = self.gmpe.COEFFS.keys()[0]
						imt_periods['SA'] = [sa.period for sa in sorted(self.gmpe.COEFFS[(vs30, kappa)].sa_coeffs.keys())]
					else:
						imt_periods['SA'] = [sa.period for sa in sorted(self.gmpe.COEFFS.sa_coeffs.keys())]
				else:
					imt_periods[imt.__name__] = [0]

		if name in ("ToroEtAl2002", "ToroEtAl2002SHARE"):
			## Remove T=3 and T=4 seconds, which are not supported by the
			## original publication
			imt_periods['SA'] = imt_periods['SA'][:-2]

		GMPE.__init__(self, imt_periods, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings, name, short_name)

		## Unit conversion
		self.imt_scaling = {}
		self.imt_scaling["PGA"] = {"g": 1.0, "mg": 1E+3, "ms2": g, "gal": g*100, "cms2": g*100}
		self.imt_scaling["SA"] = self.imt_scaling["PGA"]
		self.imt_scaling["PGV"] = {"ms": 1E-2, "cms": 1.0}
		self.imt_scaling["PGD"] = {"m": 1E-2, "cm": 1.0}

	def _get_nhlib_mean_and_stddevs(self, M, d, h=0., imt="PGA", T=0, vs30=None, kappa=None, mechanism="normal", damping=5):
		## Convert arguments to correct shape
		if isinstance(M, (int, float)):
			magnitude_is_array = False
			Mags = [M]
		else:
			magnitude_is_array = True
			Mags = M

		if isinstance(d, (int, float)):
			distance_is_array = False
			d = np.array([d], dtype='d')
		else:
			if isinstance(d, list):
				d = np.array(d, dtype='d')
			else:
				d = d.astype('d')
			distance_is_array = True

		ln_means, ln_stddevs = [], []
		for M in Mags:
			## get sctx, rctx, dctx and imt
			sctx, rctx, dctx, imt_object = self._get_contexts_and_imt(M, d, h, vs30, kappa, mechanism, imt, T, damping)

			## get mean and sigma
			ln_mean, [ln_sigma] = self.gmpe().get_mean_and_stddevs(sctx, rctx, dctx, imt_object, ['Total'])

			#if not distance_is_array:
			#	ln_mean = ln_mean[0]
			#	ln_sigma = ln_sigma[0]

			ln_means.append(ln_mean)
			ln_stddevs.append(ln_sigma)

		if not magnitude_is_array:
			ln_means = ln_means[0]
			ln_stddevs = ln_stddevs[0]
		else:
			ln_means = np.array(ln_means)
			ln_stddevs = np.array(ln_stddevs)

		return (ln_means, ln_stddevs)

	def _get_contexts_and_imt(self, M, d, h, vs30, kappa, mechanism, imt, T, damping):
		"""
		Return site, rupture and distance context, and imt objects.

		See :meth:`__call__` for params.
		"""
		## set site context
		sctx = nhlib.gsim.base.SitesContext()
		if not vs30:
			vs30 = 800.
		sctx.vs30 = np.ones_like(d, dtype="d")
		sctx.vs30 *= vs30
		if kappa:
			sctx.kappa = np.ones_like(d, dtype='d')
			sctx.kappa *= kappa
#		sctx.vs30measured = None
#		sctx.z1pt0 = None
#		sctx.z2pt5 = None

		## set rupture context
		rctx = nhlib.gsim.base.RuptureContext()
		rctx.mag = M
#		rctx.dip = None
		rctx.rake = {'normal': -90., 'reverse': 90., 'strike-slip': 0.}[mechanism]
#		rctx.ztor = None
		rctx.hypo_depth = h
#		rctx.width = None

		## set distance context
		dctx = nhlib.gsim.base.DistancesContext()
		setattr(dctx, {'Hypocentral': 'rhypo', 'Epicentral': 'repi', 'Joyner-Boore': 'rjb', 'Rupture': 'rrup'}[self.distance_metric], d)
		## For Chiou & Youngs (2008) GMPE, set rx negative to exclude hanging-wall effect

		## TODO: implement other attributes of contexts
		## One option would be to add (some of) these parameters to the __call__ function

		if self.name in ("ChiouYoungs2008", "AbrahamsonSilva2008"):
			## First 3 parameters are only needed for hanging-wall effect
			setattr(dctx, "rx", np.ones_like(getattr(dctx, "rrup")) * -1)
			setattr(dctx, "rjb", np.ones_like(getattr(dctx, "rrup")))
			setattr(rctx, "dip", 45)
			## This one is only for Abrahamson & Silva (2008)
			setattr(rctx, "width", 20.)
			## Other required parameters
			setattr(rctx, "ztor", 5.)
			## The following parameters probably only concern site effects
			setattr(sctx, "z1pt0", np.array([0.034]))
			# TODO: check whether vs30measured should be a boolean
			setattr(sctx, "vs30measured", np.zeros_like(sctx.vs30, dtype=bool))

		## set imt
		if imt == "SA":
			imt_object = self.IMT_DICT[imt](T, damping)
		else:
			imt_object = self.IMT_DICT[imt]()

		return sctx, rctx, dctx, imt_object

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0, vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		See class GMPE for params.

		Note: soil_type not supported. Should be implemented for each subclass.
		"""
		ln_means, ln_stddevs = self._get_nhlib_mean_and_stddevs(M, d, h=h, imt=imt, T=T, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)

		## number of standard deviations
		if epsilon:
			unc = ln_stddevs * epsilon
		else:
			unc = 0.

		## convert to accelerations
		imls = np.exp(ln_means + unc)

		## apply scale factor
		scale_factor = self.imt_scaling[imt.upper()][imt_unit.lower()]
		imls *= scale_factor

		return imls

	def log_sigma(self, M, d, h=0., imt="PGA", T=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		Return sigma as log10.

		See method self.__call__ for params.
		"""
		#TODO: check whether we should support soil_type !
		_, ln_stddevs = self._get_nhlib_mean_and_stddevs(M, d, h=h, imt=imt, T=T, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)

		log_sigmas = np.log10(np.exp(ln_stddevs))

		return log_sigmas

	def is_rake_dependent(self):
		"""
		Indicate whether or not GMPE depends on rake of the source
		"""
		M, d = 6.0, 15.0
		val1 = self.__call__(M, d, vs30=800., mechanism="normal")
		val2 = self.__call__(M, d, vs30=800., mechanism="reverse")
		if np.allclose(val1, val2):
			return False
		else:
			return True


class AbrahamsonSilva2008(NhlibGMPE):
	def __init__(self):
		name, short_name = "AbrahamsonSilva2008", "AS_2008"
		distance_metric = "Rupture"
		Mmin, Mmax = 5.0, 8.5
		dmin, dmax = 0., 200
		Mtype = "MW"
		dampings = [5.]

		NhlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		if vs30 is None:
			if soil_type == ("rock"):
				vs30 = 800
			else:
				raise SoilTypeNotSupportedError(soil_type)
		return NhlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T, imt_unit=imt_unit, epsilon=epsilon, vs30=vs30, mechanism=mechanism, damping=damping)

	def plot_Figure1_Abrahamson_2008(self, imt="PGA", T=0):
		"""
		Plot Figure 1 in the paper of Abrahamson et al. (2008)
		Note that the figure does not match, because they use RJB distance,
		and we ignore hanging-wall effects (and possibly other reasons...)

		:param imt:
			String, intensity measure type, either "PGA" or "SA" (default: "PGA")
		:param T:
			Float, spectral period, either 0. or 1. (default: 1.)
		"""
		self.plot_distance(mags=[5., 6., 7., 8.], plot_style="loglog", amin=0.001, amax=1, dmin=1, dmax=200, mechanism="strike-slip", vs30=760.)

	def plot_Figure7_Abrahamson_2008(self):
		"""
		Plot Figure 7 in the paper of Abrahamson et al. (2008)
		Note that the figure does not match, because they use RJB distance,
		and we ignore hanging-wall effects (and possibly other reasons...)
		"""
		self.plot_spectrum(mags=[5., 6., 7., 8.], d=10., Tmin=0.01, Tmax=10, amin=1E-3, amax=1, mechanism="strike-slip", vs30=760.)


class AkkarBommer2010(NhlibGMPE):
	def __init__(self):
		name, short_name = "AkkarBommer2010", "AB_2010"
		distance_metric = "Joyner-Boore"
		Mmin, Mmax = 5.0, 7.6
		dmin, dmax = 0., 100.
		Mtype = "MW"
		dampings = [5.]

		NhlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type == "rock":
				vs30 = 800
			elif soil_type == "stiff":
				vs30 = 550
			elif soil_type == "soft":
				vs30 = 300
			else:
				raise SoilTypeNotSupportedError(soil_type)
		return NhlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T, imt_unit=imt_unit, epsilon=epsilon, vs30=vs30, mechanism=mechanism, damping=damping)


class AkkarEtAl2013(NhlibGMPE):
	def __init__(self):
		name, short_name = "AkkarEtAl2013", "A_2013"
		distance_metric = "Joyner-Boore"
		Mmin, Mmax = 4., 7.6
		dmin, dmax = 0., 200.
		Mtype = "MW"
		dampings = [5.]

		NhlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		if vs30 is None:
			if soil_type == "rock":
				vs30 = 750.
			else:
				raise SoilTypeNotSupportedError(soil_type)
		return NhlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T, imt_unit=imt_unit, epsilon=epsilon, vs30=vs30, mechanism=mechanism, damping=damping)

	def plot_figure_10(self, period):
		"""
		Reproduce Figure 10 of Akkart et al. (2013).

		:param period:
			float, period (options: 0 (=PGA), 0.2, 1. or 4.)
		"""
		self.plot_distance(mags=[6.0, 6.5, 7.0, 7.5, 8.0], dmin=1., dmax=200., distance_metric="Joyner-Boore", h=0, imt={0: "PGA", 0.2: "SA", 1.: "SA", 4.: "SA"}[period], T=period, imt_unit="g", mechanism="strike-slip", vs30=750., damping=5, plot_style="loglog", amin={0: 0.001, 0.2: 0.001, 1.: 0.0001, 4.: 0.0001}[period], amax=2., want_minor_grid=True)


class AtkinsonBoore2006(NhlibGMPE):
	def __init__(self):
		name, short_name = "AtkinsonBoore2006", "AB_2006"
		distance_metric = "Rupture"
		Mmin, Mmax = 3.5, 8.0
		dmin, dmax = 1., 1000.
		Mtype = "MW"
		dampings = [5.]

		NhlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type == "hard rock":
				vs30 = 2000.
			elif soil_type == "rock":
				vs30 = 760.
			else:
				raise SoilTypeNotSupportedError(soil_type)
		return NhlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T, imt_unit=imt_unit, epsilon=epsilon, vs30=vs30, mechanism=mechanism, damping=damping)

	def plot_Figure4(self, imt="PGA", T=0):
		"""
		Reproduce Figure 4 in the original publication of Atkinson & Boore (2006)

		Not sure why, but I can't reproduce these plots...

		:param imt:
			String, intensity measure type, either "PGA" or "SA"
		:param T:
			Float, period, either 0.2, 1 or 2 s
		"""
		if imt == "SA" and T in (1, 2):
			amin, amax = 1E-2, 1E+3
		else:
			amin, amax = 1E-1, 1E+4
		self.plot_distance(mags=[5., 6., 7., 8.], imt=imt, T=T, plot_style="loglog", vs30=1999., imt_unit="cms2", amin=amin, amax=amax)

	def plot_Figure_Boore_notes(self, T=0.2):
		"""
		Reproduce figures in the notes by Boore
		http://www.daveboore.com/pubs_online/ab06_gmpes_programs_and_tables.pdf

		:param T:
			Float, period, either, 0.2, 1 or 5 s
		"""
		VS30 = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 1999, 2000, 2500]
		M5values, M6values, M7values = [], [], []
		for vs30 in VS30:
			values = self.__call__([5.,6.,7.], 15., vs30=vs30, imt="SA", T=T, imt_unit="cms2")
			M5values.append(values[0])
			M6values.append(values[1])
			M7values.append(values[2])
		pylab.loglog(VS30, M5values, 'b', label="M=5")
		pylab.loglog(VS30, M6values, 'g', label="M=6")
		pylab.loglog(VS30, M7values, 'r', label="M=7")
		pylab.grid(True)
		pylab.grid(True, which="minor")
		pylab.legend()
		pylab.title("T = %.1f Hz" % (1./T))
		if T == 0.2:
			amin, amax = 10, 5000
		elif T == 1:
			amin, amax = 1, 500
		elif T == 5:
			amin, amax = 0.1, 50
		pylab.axis((150, 2500, amin, amax))
		pylab.xlabel("V30")
		pylab.ylabel("y (cm/s2)")
		pylab.show()


class AtkinsonBoore2006Prime(NhlibGMPE):
	def __init__(self):
		name, short_name = "AtkinsonBoore2006Prime", "AB_2006'"
		distance_metric = "Rupture"
		Mmin, Mmax = 3.5, 8.0
		dmin, dmax = 1., 1000.
		Mtype = "MW"
		dampings = [5.]

		NhlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type == "hard rock":
				vs30 = 2000.
			elif soil_type == "rock":
				vs30 = 760.
			else:
				raise SoilTypeNotSupportedError(soil_type)
		return NhlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T, imt_unit=imt_unit, epsilon=epsilon, vs30=vs30, mechanism=mechanism, damping=damping)

	def plot_Figure10(self, period=0.1):
		"""
		Plot figure 10 in Atkinson and Boore (2011).

		:param period:
			float, 0.1 or 1. (default: 0.1)
		"""
		self.plot_distance(mags=[5., 6., 7.5], dmin=5., dmax=500., distance_metric="Joyner-Boore", h=0, imt="SA", T=period, imt_unit="cms2", mechanism="normal", damping=5, plot_style="loglog", amin={0.1: 0.5, 1.: 0.2}[period], amax={0.1: 5000, 1.: 2000}[period], want_minor_grid=True)


class BindiEtAl2011(NhlibGMPE):
	def __init__(self):
		name, short_name = "BindiEtAl2011", "Bi_2011"
		distance_metric = "Joyner-Boore"
		Mmin, Mmax = 4., 6.9
		dmin, dmax = 0., 200.
		Mtype = "MW"
		dampings = [5.]

		NhlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type in ["A", "rock"]:
				vs30 = 800.
			elif soil_type == "B":
				vs30 = 360.
			elif soil_type == "C":
				vs30 = 180.
			elif soil_type == "D":
				vs30 = 179.
			else:
				raise SoilTypeNotSupportedError(soil_type)
		return NhlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T, imt_unit=imt_unit, epsilon=epsilon, vs30=vs30, mechanism=mechanism, damping=damping)

	def plotFigure11(self, imt="PGA", soil_type="A"):
		"""
		Plot figure 11 in Bindi et al 2011, p. 33.
		Note that we use strike-slip mechanism instead of unspecified
		(does not seem to be possible with nhlib).

		:param imt:
			String, intensity measure type, either "PGA" or "SA" (default: "PGA")
		:param soil_type:
			String, one of the soil types A (=rock), B and C (default: "A")
		"""
		if imt == "PGA":
			T = 0.
		if imt == "SA":
			T = 1.
		title = "Bindi et al. (2011) - Figure 11 - soil type %s" % soil_type
		self.plot_distance(mags=[6.3], dmin=0, dmax=300, imt=imt, T=T, imt_unit="ms2", epsilon=1, soil_type=soil_type, amin=0.01, amax=50., color="r", want_minor_grid=True, title=title)

	def plotFigure12(self, imt="PGA", soil_type="A"):
		"""
		Plot figure 12 in Bindi et al 2011, p. 34.
		Note that we use strike-slip mechanism instead of unspecified
		(does not seem to be possible with nhlib).

		:param imt:
			String, intensity measure type, either "PGA" or "SA" (default: "PGA")
		:param soil_type:
			String, one of the soil types A (=rock), B and C (default: "A")
		"""
		if imt == "PGA":
			T = 0.
			amax = 5
		if imt == "SA":
			T = 1.
			amax = 2
		title = "Bindi et al. (2011) - Figure 12 - soil type %s" % soil_type
		self.plot_distance(mags=[4.6], dmin=0, dmax=300, imt=imt, T=T, imt_unit="ms2", epsilon=1, soil_type=soil_type, amin=0.001, amax=amax, color="r", want_minor_grid=True, title=title)


class BooreAtkinson2008(NhlibGMPE):
	"""
	Valid VS30 range: 180 - 1300 m/s
	"""
	def __init__(self):
		name, short_name = "BooreAtkinson2008", "BA_2008"
		distance_metric = "Joyner-Boore"
		#Mmin, Mmax = 4.27, 7.9
		#dmin, dmax = 0., 280.
		Mmin, Mmax = 5.0, 8.0
		dmin, dmax = 0., 200.
		Mtype = "MW"
		dampings = [5.]

		NhlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type == "rock":
				vs30 = 760.
			else:
				raise SoilTypeNotSupportedError(soil_type)
		elif vs30 > 1500:
			raise VS30OutOfRangeError(vs30)
		return NhlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T, imt_unit=imt_unit, epsilon=epsilon, vs30=vs30, mechanism=mechanism, damping=damping)

	def plot_Figure10(self, r=0.):
		"""
		Plot Figure 10 in the paper of Boore & Atkinson (2008)
		Note that we use strike-slip mechanism instead of unspecified
		(does not seem to be possible with nhlib).

		:param r:
			Float, distance, either 0., 30. or 200.
		"""
		self.plot_spectrum(mags=[5., 6., 7., 8.], dmin=1., dmax=r, soil_type="rock", Tmin=1E-2, Tmax=10, amin=1E-2, amax=2000, imt_unit="cms2", mechanism="strike-slip")

	def plot_Figure11(self, T=0.2):
		"""
		Plot Figure 10 in the paper of Boore & Atkinson (2008)
		Note that we use strike-slip mechanism instead of unspecified
		(does not seem to be possible with nhlib).

		:param T:
			Float, period, either 0.2 or 3.0
		"""
		if T == 0.2:
			amin, amax = 1, 1E4
		elif T == 3.0:
			amin, amax = 0.05, 500
		self.plot_distance(mags=[5., 6., 7., 8.], imt="SA", T=T, soil_type="rock", mechanism="strike-slip", imt_unit="cms2", amin=amin, amax=amax, plot_style="loglog", dmin=0.05, dmax=400)

	def plot_Figure12(self, T=0.2):
		"""
		Plot Figure 10 in the paper of Boore & Atkinson (2008)
		Note that we use strike-slip mechanism instead of unspecified
		(does not seem to be possible with nhlib).

		:param T:
			Float, period, either 0.2 or 3.0
		"""
		if T == 0.2:
			amin, amax = 20, 2000
		elif T == 3.0:
			amin, amax = 5, 500
		for vs30 in (180, 250, 360):
			self.plot_distance(mags=[7.], imt="SA", T=T, vs30=vs30, mechanism="strike-slip", imt_unit="cms2", amin=amin, amax=amax, plot_style="loglog", dmin=0.5, dmax=200)


class BooreAtkinson2008Prime(NhlibGMPE):
	def __init__(self):
		name, short_name = "BooreAtkinson2008Prime", "BA_2008'"
		distance_metric = "Joyner-Boore"
		#Mmin, Mmax = 4.27, 7.9
		#dmin, dmax = 0., 280.
		Mmin, Mmax = 4.0, 8.0
		dmin, dmax = 0.0, 200.
		Mtype = "MW"
		dampings = [5.]

		NhlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type == "rock":
				vs30 = 760.
			else:
				raise SoilTypeNotSupportedError(soil_type)
		elif vs30 > 1500:
			raise VS30OutOfRangeError(vs30)
		return NhlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T, imt_unit=imt_unit, epsilon=epsilon, vs30=vs30, mechanism=mechanism, damping=damping)

	def plot_Figure7(self, period=0.3):
		"""
		Plot figure 7 in Atkinson and Boore (2011).

		:param period:
			float, 0.3 or 1. (default: 0.3)

		NOTE: hack is required: unspecified rake must be used (see commented
		line in hazardlib files of BooreAtkinson2008 and BooreAtkinson2008Prime)
		"""
		plot_distance([self, BooreAtkinson2008()], mags=[4.], dmin=1., dmax=200., h=0, imt="SA", T=period, imt_unit="cms2", mechanism="normal", damping=5, plot_style="loglog", amin={0.3: 0.01, 1.: 0.001}[period], amax={0.3: 1000, 1.: 100}[period], want_minor_grid=True)


class Campbell2003(NhlibGMPE):
	def __init__(self):
		name, short_name = "Campbell2003", "C_2003"
		distance_metric = "Rupture"
		Mmin, Mmax = 5.0, 8.2
		dmin, dmax = 0., 1000.
		Mtype = "MW"
		dampings = [5.]

		NhlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="hard rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		"""
		if vs30 is None:
			if not soil_type in ("generic rock", "hard rock"):
				raise SoilTypeNotSupportedError(soil_type)
		return NhlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T, imt_unit=imt_unit, epsilon=epsilon, vs30=vs30, mechanism=mechanism, damping=damping)

	def plot_Figure2(self, imt="PGA", T=0):
		"""
		Plot Figure 2 in the paper of Campbell (2003)

		:param imt:
			String, intensity measure type, either "PGA" or "SA" (default: "PGA")
		:param T:
			Float, period, either 0 (PGA), 0.2 (SA), 1.0 (SA) or 3.0 (SA)
		"""
		if (imt, T) == ("SA", 3.0):
			amin, amax = 1E-5, 1E0
		else:
			amin, amax = 1E-4, 1E1
		self.plot_distance(mags=[5., 6., 7., 8.], imt=imt, T=T, soil_type="hard rock", dmin=1, amin=amin, amax=amax)

	def plot_Figure3(self, r=3.):
		"""
		Plot Figure 3 in the paper of Campbell (2003)
		Note that PGA is not included in the plot.

		:param r:
			Float, either 3. or 30.
		"""
		self.plot_spectrum(mags=[5., 6., 7., 8.], d=r, soil_type="hard rock", Tmin=1E-3, Tmax=1E1, amin=1E-3, amax=1E+1)

	def plot_Figure13b_Drouet(self):
		"""
		Plot Figure 13b in the SHARE report by Drouet et al.
		"""
		self.plot_spectrum(mags=[6.], d=20., soil_type="hard rock", Tmin=1E-2, Tmax=8, imt_unit="ms2", amin=0.1, amax=10, plot_style="loglog")


class Campbell2003SHARE(NhlibGMPE):
	def __init__(self):
		name, short_name = "Campbell2003SHARE", "C_2003_SHARE"
		distance_metric = "Rupture"
		Mmin, Mmax = 5.0, 8.2
		dmin, dmax = 0., 1000.
		Mtype = "MW"
		dampings = [5.]

		NhlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type != "rock":
				raise SoilTypeNotSupportedError(soil_type)
		return NhlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T, imt_unit=imt_unit, epsilon=epsilon, vs30=vs30, mechanism=mechanism, damping=damping)

	def plot_Figure13b_Drouet(self):
		"""
		Plot Figure 13b in the SHARE report by Drouet et al.
		Seems to correspond more or less with kappa = 0.03
		"""
		self.plot_spectrum(mags=[6.], d=20., soil_type="rock", Tmin=1E-2, Tmax=8, imt_unit="ms2", amin=0.1, amax=10, plot_style="loglog")


class Campbell2003adjusted(NhlibGMPE):
	def __init__(self):
		name, short_name = "Campbell2003adjusted", "C_2003adj"
		distance_metric = "Rupture"
		Mmin, Mmax = 5.0, 8.2
		dmin, dmax = 0., 1000.
		Mtype = "MW"
		dampings = [5.]

		NhlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=0.03, mechanism="normal", damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type != "rock":
				raise SoilTypeNotSupportedError(soil_type)
		return NhlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T, imt_unit=imt_unit, epsilon=epsilon, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)


class CauzziFaccioli2008(NhlibGMPE):
	def __init__(self):
		name, short_name = "CauzziFaccioli2008", "CF_2008"
		distance_metric = "Hypocentral"
		Mmin, Mmax = 5.0, 7.2
		dmin, dmax = 6., 150.
		Mtype = "MW"
		dampings = [5.]

		NhlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type in ("rock", "typeA"):
				vs30 = 801
			elif soil_type == "typeB":
				vs30 = 600
			elif soil_type == "typeC":
				vs30 = 270
			elif soil_type == "typeD":
				vs30 = 150
		return NhlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T, imt_unit=imt_unit, epsilon=epsilon, vs30=vs30, mechanism=mechanism, damping=damping)


class ChiouYoungs2008(NhlibGMPE):
	def __init__(self):
		name, short_name = "ChiouYoungs2008", "CY_2008"
		# Note: this does not implement the hanging-wall effect, which also depends on Rx and RJB
		distance_metric = "Rupture"
		Mmin, Mmax = 4.3, 7.9
		dmin, dmax = 0.2, 70.
		Mtype = "MW"
		dampings = [5.]

		NhlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type == ("rock"):
				vs30 = 800
			else:
				raise SoilTypeNotSupportedError(soil_type)
		return NhlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T, imt_unit=imt_unit, epsilon=epsilon, vs30=vs30, mechanism=mechanism, damping=damping)


class FaccioliEtAl2010(NhlibGMPE):
	def __init__(self):
		name, short_name = "FaccioliEtAl2010", "F_2010"
		distance_metric = "Rupture"
		Mmin, Mmax = 4.5, 7.6
		# TODO: check valid distance range!
		dmin, dmax = 1., 150.
		Mtype = "MW"
		dampings = [5.]

		NhlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type in ("rock", "typeA"):
				vs30 = 801
			elif soil_type == "typeB":
				vs30 = 600
			elif soil_type == "typeC":
				vs30 = 270
			elif soil_type == "typeD":
				vs30 = 150
		return NhlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T, imt_unit=imt_unit, epsilon=epsilon, vs30=vs30, mechanism=mechanism, damping=damping)


class ToroEtAl2002(NhlibGMPE):
	def __init__(self):
		name, short_name = "ToroEtAl2002", "T_2002"
		distance_metric = "Joyner-Boore"
		Mmin, Mmax = 5.0, 8.0
		dmin, dmax = 1., 1000.
		Mtype = "MW"
		dampings = [5.]

		NhlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="hard rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		"""
		if vs30 is None:
			if not soil_type in ("generic rock", "hard rock"):
				raise SoilTypeNotSupportedError(soil_type)
		return NhlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T, imt_unit=imt_unit, epsilon=epsilon, vs30=vs30, mechanism=mechanism, damping=damping)

	def plot_Figure13a_Drouet(self):
		"""
		Plot Figure 13a in the SHARE report by Drouet et al.
		"""
		self.plot_spectrum(mags=[6.], d=20., Tmin=1E-2, Tmax=8, imt_unit="ms2", amin=0.1, amax=10, plot_style="loglog", soil_type="hard rock")


class ToroEtAl2002SHARE(NhlibGMPE):
	def __init__(self):
		name, short_name = "ToroEtAl2002SHARE", "T_2002_SHARE"
		distance_metric = "Joyner-Boore"
		Mmin, Mmax = 5.0, 8.0
		dmin, dmax = 1., 100.
		Mtype = "MW"
		dampings = [5.]

		NhlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type != "rock":
				raise SoilTypeNotSupportedError(soil_type)
		return NhlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T, imt_unit=imt_unit, epsilon=epsilon, vs30=vs30, mechanism=mechanism, damping=damping)

	def plot_Figure13a_Drouet(self):
		"""
		Plot Figure 13a in the SHARE report by Drouet et al.
		Not sure which kappa value this should correspond to (it should be 0.03)
		"""
		self.plot_spectrum(mags=[6.], d=20., Tmin=1E-2, Tmax=8, imt_unit="ms2", amin=0.1, amax=10, plot_style="loglog", soil_type="rock")


class ToroEtAl2002adjusted(NhlibGMPE):
	def __init__(self):
		name, short_name = "ToroEtAl2002adjusted", "T_2002adj"
		distance_metric = "Joyner-Boore"
		Mmin, Mmax = 5.0, 8.0
		dmin, dmax = 1., 1000.
		Mtype = "MW"
		dampings = [5.]

		NhlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="hard rock", vs30=None, kappa=0.03, mechanism="normal", damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type == "rock":
				vs30, kappa = 800, 0.03
		return NhlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T, imt_unit=imt_unit, epsilon=epsilon, vs30=vs30, mechanism=mechanism, damping=damping, kappa=kappa)


class ZhaoEtAl2006Asc(NhlibGMPE):
	def __init__(self):
		name, short_name = "ZhaoEtAl2006Asc", "Z_2006_Asc"
		distance_metric = "Rupture"
		Mmin, Mmax = 5.0, 8.3
		dmin, dmax = 0., 400.
		Mtype = "MW"
		dampings = [5.]
		imt_periods = {}
		## Note: attribute name for periods in ZhaoEtAl2006Asc is different, therefore they are provided here
		imt_periods["PGA"] = [0]
		imt_periods["SA"] = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00]

		NhlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax, dmin, dmax, Mtype, dampings, imt_periods)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type == "hard rock":
				vs30 = 1101
			elif soil_type in ("rock", "SC I"):
				vs30 = 850
			elif soil_type in ("hard soil", "SC II"):
				vs30 = 450
			elif soil_type in ("medium soil", "SC III"):
				vs30 = 250
			elif soil_type in ("soft soil", "SC IV"):
				vs30 = 200
		return NhlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T, imt_unit=imt_unit, epsilon=epsilon, vs30=vs30, mechanism=mechanism, damping=damping)

	def plot_Figure2a(self):
		"""
		Plot Figure 2a in the paper of Zhao et al. (2006)
		"""
		self.plot_distance(mags=[7.], mechanism="strike-slip", soil_type="SC II", epsilon=1, dmin=0.3, dmax=400, amin=2E-3, amax=3.0, plot_style="loglog")

	def plot_Figure3a(self):
		"""
		Plot Figure 3a in the paper of Zhao et al. (2006)
		Note that it is not possible to exactly reproduce the figure, which uses
		"mean site conditions"
		"""
		self.plot_distance(mags=[5., 6., 7., 8.], mechanism="strike-slip", soil_type="SC II", dmin=1, dmax=400, amin=5E-4, amax=3.0, plot_style="loglog")


def adjust_hard_rock_to_rock(imt, periods, gm, gm_logsigma=None):
	"""
	Adjust hard rock (vs30=2800 m/s) to rock (vs30=760 m/s, kappa=0.03)
	according to Table 9 in Drouet et al. (2010)
	Applicable to Toro (2002) and Campbell (2003) GMPE's

	:param imt:
		String, intensity measure type, either "PGA" or "SA"
	:param periods:
		List, periods for given imt
	:param gm:
		Float array: ground motions corresponding to given periods
	:param gm_logsigma:
		Float array: log of standard deviations of ground motions
		corresponding to given periods
		This term is currently discarded

	:return:
		Float array, adjusted ground motions
	"""
	# TODO: not clear how to interpolate between PGA and 1st spectral period
	T_AFrock = np.array([0.01, 0.03, 0.04, 0.1, 0.2, 0.4, 1.0, 2.0])
	AFrock = np.array([0.735106, 0.423049, 0.477379, 0.888509, 1.197291, 1.308267, 1.265762, 1.215779])
	AFrock_sigma = np.array([0.338916, 0.289785, 0.320650, 0.352442, 0.281552, 0.198424, 0.154327, 0.155520])

	if imt == "PGA":
		AF = AFrock[0]
		AF_sigma = AFrock_sigma[0]
	elif imt == "SA":
		AF = interpolate(T_AFrock, AFrock, periods)
		AF_sigma = interpolate(T_AFrock, AFrock_sigma, periods)

	#for t, af in zip(periods, AF):
	#	print t, af

	adjusted_gm = gm * AF
	# TODO: check sigma computation
	#adjusted_gm_logsigma = np.log10(np.exp(np.log(10**gm_log_sigma) + AF_sigma))

	#return (adjusted_gm, adjusted_gm_logsigma)
	return adjusted_gm


def adjust_faulting_style(imt, periods, gm, mechanism):
	"""
	Adjust style of faulting for GMPE's that do not include a style-of-faulting
	term according to the report by Drouet et al. (2010)
	Applicable to Toro (2002) and Campbell (2003) GMPE's

	:param imt:
		String, intensity measure type, either "PGA" or "SA"
	:param periods:
		List, periods for given imt
	:param gm:
		Float array: ground motions corresponding to given periods
	:param mechanism:
		String, fault mechanism, either "normal", "strike-slip" or "reverse"

	:return:
		Float array, adjusted ground motions
	"""
	## Note: it appears the two reports by Drouet et al. (2010) are in
	## contradiction regarding the formulas for normal and reverse faulting.
	## It is very likely that the formulas on page 13 of deliverable 4.2,
	## which are those used in OpenSHA and nhlib have been interchanged,
	## because the adjustment leads to highest accelerations for normal
	## events, and lowest accelerations for reverse events, whereas all
	## the plots in Drouet et al. show the opposite...
	## The formulas on page 10 of the GMPE report (revised version 2)
	## are propably the correct ones.

	# TODO: not clear how to interpolate between PGA and 1st spectral period
	T_FRSS = np.array([0., 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.55, 0.60, 0.65, 0.70, 1.60, 1.70, 1.80, 1.90, 2.0])
	FRSS = np.array([1.22, 1.08, 1.10, 1.11, 1.13, 1.13, 1.15, 1.16, 1.16, 1.17, 1.18, 1.19, 1.20, 1.20, 1.21, 1.22, 1.23, 1.23, 1.23, 1.23, 1.23, 1.23, 1.24, 1.23, 1.24, 1.23, 1.23, 1.23, 1.22, 1.21, 1.20, 1.19, 1.18, 1.17, 1.16, 1.14])
	Fnss = 0.95
	pN, pR = 0.01, 0.81

	if imt == "PGA":
		frss = FRSS[0]
	else:
		frss = interpolate(T_FRSS, FRSS, periods)

	#for t, f in zip(periods, frss):
	#	print t, f

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
	Conversions are valid from 0.02 to 5 s

	:param component_type:
		String, component type, one of the following:
			"GR": rotated geometrical mean
			"AM": arithmetic mean
			"LA": larger envelope
			"RA": random horizontal
			"LP": larger PGA
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
	Conversion are valid for magnitudes from 4 to 8, focal depths from 8 to 12,
	and dips from 45 to 90. Hypocenter is assumed to be at center of rupture.

	:param distances:
		np array, distances to convert
	:param metric_from:
		str, distance metric of distances
	:param metric_to:
		str, distance metric to convert distances to
	:param mag:
		float, magnitude to use for conversion
	"""
	if metric_from == None or metric_from == metric_to:
		return distances
	else:
		ztor = 21. - 2.5 * mag
		if metric_from == "Rupture" and metric_to == "Joyner-Boore":
			return np.sqrt(distances ** 2 - ztor ** 2)
		elif metric_from == "Joyner-Boore" and metric_to == "Rupture":
			return np.sqrt(distances ** 2 + ztor ** 2)
		else:
			raise "conversion not supported" # TODO: support other conversion


def plot_distance(gmpe_list, mags, dmin=None, dmax=None, distance_metric=None, h=0, imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5, plot_style="loglog", amin=None, amax=None, colors=None, fig_filespec=None, title="", want_minor_grid=False, legend_location=0, lang="en"):
	"""
	Function to plot ground motion versus distance for one or more GMPE's.
	Horizontal axis: distances.
	Vertical axis: ground motion values.

	:param gmpe_list:
		list of GMPE objects.
	:param mags:
		list of floats, magnitudes to plot
	:param dmin:
		Float, lower distance in km. If None, use the lower bound of the
		distance range of each GMPE (default: None).
	:param dmax:
		Float, upper distance in km. If None, use the lower bound of the
		valid distance range of each GMPE (default: None).
	:param distance_metric:
		str, distance_metric to plot (options: "Joyner-Boore", "Rupture")
		(default: None, distance metrics of gmpes are used)
	:param h:
		Float, depth in km. Ignored if distance metric of GMPE is epicentral
		or Joyner-Boore (default: 0).
	:param imt:
		String, one of the supported intensity measure types.
		(default: "PGA").
	:param T:
		Float, period to plot (default: 0).
	:param imt_unit:
		String, unit in which intensities should be expressed, depends on
		IMT (default: "g")
	:param epsilon:
		Float, number of standard deviations above or below the mean to
		plot in addition to the mean (default: 0).
	:param soil_type:
		String, one of the soil types supported by the particular GMPE
		(default: "rock").
	:param vs30:
		Float, shear-wave velocity in the upper 30 m (in m/s). If not None,
		it takes precedence over the soil_type parameter (default: None).
	:param kappa:
		Float, kappa value, in seconds (default: None)
	:param mechanism:
		String, fault mechanism: either "normal", "reverse" or "strike-slip"
		(default: "normal").
	:param damping:
		Float, damping in percent (default: 5).
	:param plot_style:
		String, plotting style ("lin", "loglin", "linlog" or "loglog").
		First term refers to horizontal axis, second term to vertical axis.
		(default: "loglog").
	:param amin:
		Float, lower ground-motion value to plot (default: None).
	:param amax:
		upper ground-motion value to plot (default: None).
	:param colors:
		List of matplotlib color specifications (default: None)
	:param fig_filespec:
		String, full path specification of output file (default: None).
	:param title:
		String, plot title (default: "")
	:param want_minor_grid:
		Boolean, whether or not to plot minor gridlines (default: False).
	:param legend_location:
		Integer, location of legend (matplotlib location code) (default=0):
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
	:param lang:
		String, shorthand for language of annotations. Currently only
		"en" and "nl" are supported (default: "en").
	"""
	linestyles = ("", "--", ":", "-.")
	if not colors:
		colors = ("k", "r", "g", "b", "c", "m", "y")

	if plot_style.lower() in ("lin", "linlin"):
		plotfunc = pylab.plot
	elif plot_style.lower() == "linlog":
		plotfunc = pylab.semilogy
	elif plot_style.lower() == "loglin":
		plotfunc = pylab.semilogx
	elif plot_style.lower() == "loglog":
		plotfunc = pylab.loglog

	for i, gmpe in enumerate(gmpe_list):
		if dmin is None:
			dmin = gmpe.dmin
		if dmax is None:
			dmax = gmpe.dmax
		## Avoid math domain errors with 0
		if dmin == 0:
			dmin = 0.1
		distances = logrange(max(dmin, gmpe.dmin), min(dmax, gmpe.dmax), 25)
		for j, M in enumerate(mags):
			converted_distances = convert_distance_metric(distances, distance_metric, gmpe.distance_metric, M)
			Avalues = gmpe(M, converted_distances, h=h, imt=imt, T=T, imt_unit=imt_unit, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
			style = colors[i%len(colors)] + linestyles[j%len(linestyles)]
			plotfunc(distances, Avalues, style, linewidth=3, label=gmpe.name+" (M=%.1f)" % M)
			if epsilon:
				## Fortunately, log_sigma is independent of scale factor!
				## Thus, the following are equivalent:
				#log_sigma = gmpe.log_sigma(M, imt=imt, T=T, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
				#Asigmavalues = 10**(np.log10(Avalues) + log_sigma)
				Asigmavalues = gmpe(M, converted_distances, h=h, imt=imt, T=T, imt_unit=imt_unit, epsilon=epsilon, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
				plotfunc(distances, Asigmavalues, style, linewidth=1, label=gmpe.name+" (M=%.1f) $\pm %d \sigma$" % (M, epsilon))
				#Asigmavalues = 10**(np.log10(Avalues) - log_sigma)
				Asigmavalues = gmpe(M, converted_distances, h=h, imt=imt, T=T, imt_unit=imt_unit, epsilon=-epsilon, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
				plotfunc(distances, Asigmavalues, style, linewidth=1, label='_nolegend_')

	## Plot decoration
	if distance_metric:
		pylab.xlabel(" ".join([distance_metric, "distance (km)"]), fontsize="x-large")
	else:
		distance_metrics = set()
		for gmpe in gmpe_list:
			distance_metrics.add(gmpe.distance_metric)
		if len(distance_metrics) > 1:
			pylab.xlabel("Distance (km)", fontsize="x-large")
		else:
			pylab.xlabel(" ".join([gmpe.distance_metric, "distance (km)"]), fontsize="x-large")
	imt_label = get_imt_label(imt, lang.lower()) + " (%s)" % imt_unit_to_plot_label.get(imt_unit, imt_unit)
	pylab.ylabel(imt_label, fontsize="x-large")
	pylab.grid(True)
	if want_minor_grid:
		pylab.grid(True, which="minor")
	title += "\n%s" % imt
	if not imt in ("PGA", "PGV", "PGD"):
		title += " (T=%s s)" % T
	pylab.title(title)
	font = FontProperties(size='large')
	pylab.legend(loc=legend_location, prop=font)
	xmin, xmax, ymin, ymax = pylab.axis()
	if amin is None:
		amin = ymin
	if amax is None:
		amax = ymax
	pylab.axis((dmin, dmax, amin, amax))
	ax = pylab.gca()
	for label in ax.get_xticklabels() + ax.get_yticklabels():
		label.set_size('large')
	if fig_filespec:
		pylab.savefig(fig_filespec, dpi=300)
		pylab.clf()
	else:
		pylab.show()


def plot_spectrum(gmpe_list, mags, d, h=0, imt="SA", Tmin=None, Tmax=None, imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None, mechanism="normal", damping=5, include_pgm=True, plot_freq=False, plot_style="loglog", amin=None, amax=None, colors=None, labels=None, fig_filespec=None, title="", want_minor_grid=False, legend_location=None, lang="en"):
	"""
	Function to plot ground motion spectrum for one or more GMPE's.
	Horizontal axis: spectral periods or frequencies.
	Vertical axis: ground motion values.

	:param gmpe_list:
		list of GMPE objects.
	:param mags:
		lsit of floats, magnitudes to plot
	:param d:
		Float, distance in km.
	:param h:
		Float, depth in km. Ignored if distance metric of GMPE is epicentral
		or Joyner-Boore (default: 0).
	:param imt:
		String, one of the supported intensity measure types.
		(default: "SA").
	:param Tmin:
		Float, lower period to plot. If None, lower bound of valid period
		range is used (default: None).
	:param Tmax:
		Float, upper period to plot. If None, upper bound of valid period
		range is used (default: None).
	:param imt_unit:
		String, unit in which intensities should be expressed, depends on
		IMT (default: "g")
	:param epsilon:
		Float, number of standard deviations above or below the mean to
		plot in addition to the mean (default: 0).
	:param soil_type:
		String, one of the soil types supported by the particular GMPE
		(default: "rock").
	:param vs30:
		Float, shear-wave velocity in the upper 30 m (in m/s). If not None,
		it takes precedence over the soil_type parameter (default: None).
	:param kappa:
		Float, kappa value, in seconds (default: None)
	:param mechanism:
		String, fault mechanism: either "normal", "reverse" or "strike-slip"
		(default: "normal").
	:param damping:
		Float, damping in percent (default: 5).
	:param include_pgm:
		Boolean, whether or not to include peak ground motion in the plot,
		if possible (plot_freq == False and plot_style in ("lin", "linlog")
		(default: True).
	:param plot_freq:
		Boolean, whether or not to plot frequencies instead of periods
		(default: False).
	:param plot_style:
		String, plotting style ("lin", "loglin", "linlog" or "loglog").
		First term refers to horizontal axis, second term to vertical axis.
		(default: "loglog").
	:param amin:
		Float, lower ground-motion value to plot (default: None).
	:param amax:
		upper ground-motion value to plot (default: None).
	:param colors:
		List of matplotlib color specifications (default: None).
	:param labels:
		List of labels for each GMPE (defaultl: None)
	:param fig_filespec:
		String, full path specification of output file (default: None).
	:param title:
		String, plot title (default: "")
	:param want_minor_grid:
		Boolean, whether or not to plot minor gridlines (default: False).
	:param legend_location:
		Integer, location of legend (matplotlib location code) (default=None):
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
	:param lang:
		String, shorthand for language of annotations. Currently only
		"en" and "nl" are supported (default: "en").
	"""
	linestyles = ("", "--", ":", "-.")
	if not colors:
		colors = ("k", "r", "g", "b", "c", "m", "y")

	if plot_style.lower() in ("lin", "linlin"):
		plotfunc = pylab.plot
	elif plot_style.lower() == "linlog":
		plotfunc = pylab.semilogy
	elif plot_style.lower() == "loglin":
		plotfunc = pylab.semilogx
	elif plot_style.lower() == "loglog":
		plotfunc = pylab.loglog

	for i, gmpe in enumerate(gmpe_list):
		periods = gmpe.imt_periods[imt]
		if Tmin is None or gmpe.Tmin(imt) < Tmin:
			Tmin = gmpe.Tmin(imt)
		if Tmax is None or gmpe.Tmax(imt) > Tmax:
			Tmax = gmpe.Tmax(imt)
		if plot_freq:
			freqs = gmpe.freqs(imt)
			xvalues = freqs
		else:
			xvalues = periods
		for j, M in enumerate(mags):
			periods, Avalues = gmpe.get_spectrum(M, d, h=h, imt=imt, imt_unit=imt_unit, epsilon=0, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
			#Asigma_values = gmpe.get_spectrum(M, d, h=h, imt=imt, imt_unit=imt_unit, epsilon=num_sigma, soil_type=soil_type, mechanism=mechanism, damping=damping)
			Asigma_values = np.array([gmpe.log_sigma(M, d, h=h, imt=imt, T=T, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)[0] for T in periods])

			#non_zero_Avalues, non_zero_xvalues, non_zero_Asigma_values = [], [], []
			#for a, x, sigma in zip(Avalues, xvalues, Asigma_values):
			#	if a:
			#		non_zero_Avalues.append(a)
			#		non_zero_xvalues.append(x)
			#		non_zero_Asigma_values.append(sigma)

			style = linestyles[j] + colors[i]
			if isinstance(labels, (list, tuple)) and len(labels) > i and labels[i] != None:
				gmpe_label = labels[i]
			else:
				gmpe_label = gmpe.name
			if gmpe.is_rake_dependent():
				gmpe_label += " - %s" % mechanism
			plotfunc(xvalues, Avalues, style, linewidth=3, label=gmpe_label+" (M=%.1f)" % M)

			pgm = None
			if include_pgm:
				try:
					pgm = {"SA": "PGA", "PSV": "PGV", "SD": "PGD"}[imt]
				except:
					pass
				else:
					if gmpe.has_imt(pgm) and plot_style in ("lin", "linlog") and plot_freq == False:
						[pgm_T], [pgm_Avalue] = gmpe.get_spectrum(M, d, h=h, imt=pgm, imt_unit=imt_unit, epsilon=0, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
						pgm_sigma = gmpe.log_sigma(M, d, h=h, imt=pgm, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
						Tmin = pgm_T
						# TODO: add outline color and symbol size
						plotfunc(pgm_T, pgm_Avalue, colors[j]+"o", label="_nolegend_")
					else:
						pgm = None

			if epsilon:
				sigma_values = 10**(np.log10(Avalues) + epsilon * Asigma_values)
				plotfunc(xvalues, sigma_values, style, linewidth=1, label=gmpe_label+" (M=%.1f) $\pm %d \sigma$" % (M, epsilon))
				sigma_values = 10**(np.log10(Avalues) - epsilon * Asigma_values)
				plotfunc(xvalues, sigma_values, style, linewidth=1, label='_nolegend_')

				if pgm:
					for sign in (1.0, -1.0):
						pgm_sigma_value = 10**(np.log10(pgm_Avalue) + epsilon * sign * pgm_sigma)
						# TODO: add outline color and symbol size
						plotfunc(pgm_T, pgm_sigma_value, "o", label="_nolegend_")

	## PLot decoration
	pylab.grid(True)
	if want_minor_grid:
		pylab.grid(True, which="minor")
	title += "\nd=%.1f km, h=%d km" % (d, int(round(h)))
	pylab.title(title)
	font = FontProperties(size='medium')
	xmin, xmax, ymin, ymax = pylab.axis()
	if amin is None:
		amin = ymin
	if amax is None:
		amax = ymax
	if plot_freq:
		pylab.xlabel("Frequency (Hz)", fontsize="x-large")
		if legend_location == None:
			legend_location = 4
		pylab.axis((1./Tmax, 1./Tmin, amin, amax))
	else:
		pylab.xlabel("Period (s)", fontsize="x-large")
		if legend_location == None:
			legend_location = 3
		pylab.axis((Tmin, Tmax, amin, amax))
	imt_label = get_imt_label(imt, lang.lower()) + " (%s)" % imt_unit_to_plot_label.get(imt_unit, imt_unit)
	pylab.ylabel(imt_label, fontsize="x-large")
	pylab.legend(loc=legend_location, prop=font)
	ax = pylab.gca()
	for label in ax.get_xticklabels() + ax.get_yticklabels():
		label.set_size('large')
	if fig_filespec:
		pylab.savefig(fig_filespec, dpi=300)
		pylab.clf()
	else:
		pylab.show()


## Dictionary to convert IMT units to plot labels
imt_unit_to_plot_label = {}
imt_unit_to_plot_label["g"] = "g"
imt_unit_to_plot_label["gal"] = "gal"
imt_unit_to_plot_label["ms2"] = "$m/s^2$"
imt_unit_to_plot_label["cms2"] = "$cm/s^2$"
imt_unit_to_plot_label["ms"] = "m/s"
imt_unit_to_plot_label["cms"] = "cm/s"
imt_unit_to_plot_label["m"] = "m"
imt_unit_to_plot_label["cm"] = "cm"


def get_imt_label(imt, lang="en"):
	"""
	Return plot label for a particular IMT

	:param imt:
		String, Intensity measure type
	:param lang:
		String, shorthand for language of annotations. Currently only
		"en" and "nl" are supported (default: "en").

	:return:
		String, plot axis label.
	"""
	imt_label = {}
	imt_label["PGA"] = {"en": "Peak Ground Acceleration", "nl": "Piekgrondversnelling"}
	imt_label["PGV"] = {"en": "Peak Ground Velocity", "nl": "Piekgrondsnelheid"}
	imt_label["PGD"] = {"en": "Peak Ground Displacement", "nl": "Piekgrondverplaatsing"}
	imt_label["SA"] = {"en": "Spectral Acceleration", "nl": "Spectrale versnelling"}
	imt_label["PSV"] = {"en": "Spectral Velocity", "nl": "Spectrale snelheid"}
	imt_label["SD"] = {"en": "Spectral Displacement", "nl": "Spectrale verplaatsing"}
	return imt_label[imt][lang]


if __name__ == "__main__":
	pass

