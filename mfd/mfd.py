# -*- coding: iso-Latin-1 -*-

"""
This module extends functionality of nhlib.mfd
"""

import datetime
from lxml import etree

import numpy as np
import pylab
from matplotlib.font_manager import FontProperties

import openquake.hazardlib as nhlib

from ..nrml import ns
from ..nrml.common import *



class MFD(object):
	"""
	Generic class containing methods that are common for
	:class:`EvenlyDiscretizedMFD` and :class:`TruncatedGRMFD`
	"""
	def __init__(self):
		pass

	def __len__(self):
		return int(round((self.max_mag - self.get_min_mag_edge()) / self.bin_width))

	def get_magnitude_bin_centers(self):
		"""
		Return center values of magnitude bins

		:return:
			numpy float array
		"""
		return np.array(zip(*self.get_annual_occurrence_rates())[0])

	def get_magnitude_bin_edges(self):
		"""
		Return left edge value of magnitude bins

		:return:
			numpy float array
		"""
		return self.get_magnitude_bin_centers() - self.bin_width / 2

	def get_magnitude_index(self, M):
		"""
		Determine index of given magnitude (edge) value

		:param M:
			Float, magnitude value (left edge of bin)

		:return:
			Int, index
		"""
		return int(round((M - self.get_min_mag_edge()) / self.bin_width))

	def get_cumulative_rates(self):
		"""
		Return cumulative annual occurrence rates

		:return:
			numpy float array
		"""
		return np.add.accumulate(self.occurrence_rates[::-1])[::-1]

	def is_magnitude_compatible(self, M):
		"""
		Determine whether a particular magnitude (edge) value is compatible
		with bin width of MFD

		:param M:
			Float, magnitude value (left edge of bin)

		:return:
			Bool
		"""
		foffset = (M - self.get_min_mag_edge()) / self.bin_width
		offset = int(round(foffset))
		if not np.allclose(foffset, offset):
			return False
		else:
			return True

	def is_compatible(self, other_mfd):
		"""
		Determine if MFD is compatible with another one, in terms of
		bin width, modulus of magnitude, and magnitude type

		:param other_mfd:
			instance of :class:`EvenlyDiscretizedMFD` or :clas:`TruncatedGRMFD`

		:return:
			Bool
		"""
		if other_mfd.Mtype != self.Mtype:
			return False
		if not np.allclose(other_mfd.bin_width, self.bin_width):
			return False
		elif not self.is_magnitude_compatible(other_mfd.get_min_mag_edge()):
			return False
		else:
			return True

	def to_evenly_discretized_mfd(self, max_mag=None):
		"""
		Convert to an EvenlyDiscretizedMFD

		:param max_mag:
			Float, maximum magnitude (default: None)

		:return:
			instance of :class:`EvenlyDiscretizedMFD`
		"""
		if max_mag is None:
			occurrence_rates = self.occurrence_rates
		else:
			if max_mag <= self.max_mag:
				occurrence_rates = np.array(self.occurrence_rates)[self.get_magnitude_bin_centers() < max_mag]
			else:
				dmag = max_mag - self.max_mag
				num_zeros = int(np.round(dmag / self.bin_width))
				occurrence_rates = np.append(self.occurrence_rates, np.zeros(num_zeros))
		return EvenlyDiscretizedMFD(self.get_min_mag_center(), self.bin_width, occurrence_rates, Mtype=self.Mtype)

	def get_num_earthquakes(self, completeness, end_date):
		"""
		Return array with number of earthquakes per magnitude bin,
		taking into account completeness

		:param completeness:
			instance of :class:`Completeness`
		:param end_date:
			datetime.date or Int, end date with respect to which observation periods
			will be determined

		:return:
			numpy float array
		"""
		magnitudes = self.get_magnitude_bin_edges()
		timespans = completeness.get_completeness_timespans(magnitudes, end_date)
		return np.array(self.occurrence_rates) * timespans

	def export_csv(self, filespec, cumul=True):
		"""
		Export magnitudes and occurrence rates to a csv file

		:param filespec:
			str, full path to csv file
		:param cumul:
			bool, whether or not cumulative rates should be exported
			(default: True)
		"""
		f = open(filespec, "w")
		f.write("Magnitudes,Occurence Rates\n")
		if cumul:
			rates = self.get_cumulative_rates()
		else:
			rates = self.occurrence_rates
		magnitudes = self.get_magnitude_bin_centers()
		for mag, rate in zip(magnitudes, rates):
			f.write("%.2f,%f\n" % (mag, rate))
		f.close()

	def sample_inter_event_times(self, timespan, skip_time=0, method="poisson", random_seed=None):
		"""
		Generate random inter-event times for each magnitude bin according to a
		Poisson or random process.

		See: http://preshing.com/20111007/how-to-generate-random-timings-for-a-poisson-process/

		:param timespan:
			int, time span (in years) in which random timings are generated
		:param skip_time:
			int, time span (in years) to skip before starting to keep the
			samples
			(default: 0)
		:param method:
			str, sampling method, either "poisson", "random"
			(default: "poisson")
		:param random_seed:
			int, seed for the random number generator (default: None)

		:return:
			nested list with inter-event times for each magnitude bin
		"""
		import random
		rnd = random.Random()
		rnd.seed(random_seed)

		inter_event_times = []
		for mag, rate in self.get_annual_occurrence_rates():
			if method in ("random", "time-dependent"):
				## In a Poisson distribution, variance is equal to mean,
				## so we use the corresponding standard deviation
				# TODO: check error propgation for inverse
				sigma = 1. / np.sqrt(rate)
			## Should we re-initialize random number generator for each bin?
			inter_event_times.append([])
			total_time, iet = 0, -1
			#while total_time <= timespan:
			while total_time < (timespan + skip_time):
				if iet >= 0 and total_time >= skip_time:
					if len(inter_event_times[-1]) == 0:
						next_event_time = total_time - skip_time
					else:
						next_event_time = iet
					inter_event_times[-1].append(next_event_time)
				if method == "poisson":
					#prob = rnd.random()
					## From http://preshing.com/20111007/how-to-generate-random-timings-for-a-poisson-process/
					#next_event_time = -np.log(1.0 - prob) / rate
					iet = rnd.expovariate(rate)
				elif method == "random":
					epsilon = rnd.normalvariate(0, 1)
					iet = (1. / rate) + sigma * epsilon
				elif method == "time-dependent":
					# TODO
					raise Exception("Time-dependent sampling not yet implemented!")
				total_time += iet

		return np.array(inter_event_times)

	def generate_random_catalog(self, timespan, skip_time=0, method="poisson", start_year=1, lons=None, lats=None, random_seed=None):
		"""
		Generate random catalog by random sampling.
		See :meth:`sample_inter_event_times`
		Optionally position can be sampled from a discrete distribution.

		:param timespan:
			int, time span (in years) of catalog
		:param method:
			str, sampling method, either "poisson", "random"
			(default: "poisson")
		:param start_year:
			int, start year of generated catalog (default: 1).
			Note that the year 0 does not exist!
		:param lons:
			list or array, longitudes to sample from (default: None)
		:param lats:
			list or array, latitudes to sample from (default: None)
		:param random_seed:
			int, seed for the random number generator (default: None)

		:return:
			instance of :class:`EQCatalog`
		"""
		import random
		import mx.DateTime as mxDateTime
		rnd = random.Random()
		rnd.seed(random_seed)
		from eqcatalog.eqcatalog import EQCatalog, LocalEarthquake
		inter_event_times = self.sample_inter_event_times(timespan, skip_time, method, random_seed)
		eq_list = []
		ID = 0
		depth = 0.
		name = ""
		try:
			num_lon_lats = len(lons)
		except:
			num_lon_lats = 0
		start_date = mxDateTime.Date(start_year, 1, 1)
		for M, iets in zip(self.get_magnitude_bin_centers(), inter_event_times):

			fractional_years = np.add.accumulate(iets)
			for year in fractional_years:
				days_in_year = (mxDateTime.Date(int(year), 1, 1)
							- mxDateTime.Date(int(year), 12, 31)).days
				days = int((year - int(year)) * days_in_year)
				date = start_date + mxDateTime.RelativeDate(years=int(year), days=days)
				if num_lon_lats == 0:
					lon, lat = 0., 0.
				else:
					idx = rnd.randint(0, num_lon_lats-1)
					lon, lat = lons[idx], lats[idx]
				eq = LocalEarthquake(ID, date, None, lon, lat, depth, {self.Mtype: M}, name=name)
				eq_list.append(eq)
				ID += 1
		end_date = start_date + mxDateTime.RelativeDate(years=timespan) - mxDateTime.RelativeDate(days=1)
		return EQCatalog(eq_list, start_date, end_date)

	def get_incremental_moment_rates(self):
		"""
		Calculate moment rates corresponding to each magnitude bin

		:return:
			float array, moment rates in N.m
		"""
		magnitudes, occurrence_rates = zip(*self.get_annual_occurrence_rates())
		magnitudes, occurrence_rates = np.array(magnitudes), np.array(occurrence_rates)
		moments = 10 ** (1.5 * (magnitudes + 6.06))
		moment_rates = moments * occurrence_rates
		return moment_rates

	def get_total_moment_rate(self):
		"""
		Calculate total moment rate

		:return:
			Float, total moment rate in N.m/yr
		"""
		return np.add.reduce(self.get_incremental_moment_rates())

	def get_return_periods(self):
		"""
		Return return periods for each magnitude bin

		:return:
			float array, return periods in yr
		"""
		return 1. / np.array(self.occurrence_rates)

	def get_return_period(self, M):
		"""
		Report return period for particular magnitude

		:param M:
			float, magnitude value (left edge of bin)

		:return:
			float, return period in yr
		"""
		idx = self.get_magnitude_index(M)
		return 1. / self.occurrence_rates[idx]


class EvenlyDiscretizedMFD(nhlib.mfd.EvenlyDiscretizedMFD, MFD):
	"""
	Evenly Discretized Magnitude-Frequency Distribution

	:param min_mag:
		Positive float value representing the middle point of the first
		bin in the histogram.
	:param bin_width:
		A positive float value -- the width of a single histogram bin.
	:param occurrence_rates:
		The list of non-negative float values representing the actual
		annual occurrence rates. The resulting histogram has as many bins
		as this list length.
		Note that the property is always stored as a python list, even
		if a numpy array is specified!
	:param Mtype:
		String, magnitude type, either "MW" or "MS" (default: "MW")
	"""
	def __init__(self, min_mag, bin_width, occurrence_rates, Mtype="MW"):
		## Note: Convert occurrence_rates to list to avoid problems in oqhazlib!
		nhlib.mfd.EvenlyDiscretizedMFD.__init__(self, min_mag, bin_width, list(occurrence_rates))
		self.Mtype = Mtype

	def __div__(self, other):
		if isinstance(other, (int, float)):
			occurrence_rates = np.array(self.occurrence_rates) / other
			return EvenlyDiscretizedMFD(self.min_mag, self.bin_width, occurrence_rates, self.Mtype)
		else:
			raise TypeError("Divisor must be integer or float")

	def __mul__(self, other):
		if isinstance(other, (int, float)):
			occurrence_rates = np.array(self.occurrence_rates) * other
			return EvenlyDiscretizedMFD(self.min_mag, self.bin_width, occurrence_rates, self.Mtype)
		else:
			raise TypeError("Multiplier must be integer or float")

	def __add__(self, other):
		if isinstance(other, (TruncatedGRMFD, EvenlyDiscretizedMFD, YoungsCoppersmith1985MFD)):
			return sum_MFDs([self, other])
		else:
			raise TypeError("Operand must be MFD")

	def __sub__(self, other):
		# TODO: this doesn't work if max_mags are different, need to either pad or truncate
		if isinstance(other, (TruncatedGRMFD, EvenlyDiscretizedMFD, YoungsCoppersmith1985MFD)):
			if not self.is_compatible(other):
				raise Exception("MFD's not compatible")
			if self.get_min_mag() <= other.get_min_mag() and self.max_mag >= other.max_mag:
				occurrence_rates = np.array(self.occurrence_rates)
				start_index = self.get_magnitude_index(other.get_min_mag())
				occurrence_rates[start_index:start_index+len(other)] -= np.array(other.occurrence_rates)
				# Replace negative values with zeros
				occurrence_rates[np.where(occurrence_rates < 0)] = 0
				return EvenlyDiscretizedMFD(self.min_mag, self.bin_width, occurrence_rates)
			else:
				raise Exception("Second MFD must fully overlap with first one!")
		else:
			raise TypeError("Operand must be MFD")

	@property
	def max_mag(self):
		return self.get_min_mag_edge() + len(self.occurrence_rates) * self.bin_width

	def get_copy(self):
		"""
		Return a copy of the MFD

		:return:
			instance of :class:`EvenlyDiscretizedMFD`
		"""
		return self.to_evenly_discretized_mfd()

	def get_min_mag_edge(self):
		"""
		Return left edge of minimum magnitude bin

		:return:
			Float
		"""
		return self.min_mag - self.bin_width / 2

	def get_min_mag_center(self):
		"""
		Return center value of minimum magnitude bin

		:return:
			Float
		"""
		return self.min_mag

	def get_center_magnitudes(self):
		"""
		Return array with center magnitudes of each bin

		:return:
			ndarray, magnitudes
		"""
		magnitudes = np.arange(len(self.occurrence_rates), dtype='f')
		magnitudes = self.get_min_mag_center() + magnitudes * self.bin_width
		return magnitudes

	def set_min_mag(self, min_mag):
		"""
		Set minimum magnitude

		:param min_mag:
			float, new minimum magnitude (left edge, not center of bin !)
		"""
		assert self.get_min_mag_edge() <= min_mag <= self.max_mag
		mag_idx = self.get_magnitude_index(min_mag)
		self.occurrence_rates = self.occurrence_rates[mag_idx:]
		self.min_mag = min_mag + self.bin_width / 2

	#def get_magnitude_bin_edges(self):
	#	return np.array(zip(*self.get_annual_occurrence_rates())[0])

	#def get_magnitude_bin_centers(self):
	#	return self.get_magnitude_bin_edges() + self.bin_width / 2

	#def get_cumulative_rates(self):
	#	return np.add.accumulate(self.occurrence_rates[::-1])[::-1]

	def divide(self, weights):
		"""
		Divide MFD into a number of MFD's that together sum up to the original MFD

		:param weights:
			list or array containing weight of each sub-MFD

		:return:
			List containing instances of :class:`EvenlyDiscretizedMFD`
		"""
		weights = np.array(weights, dtype='d')
		weights /= np.add.reduce(weights)
		mfd_list = []
		for w in weights:
			occurrence_rates = np.array(self.occurrence_rates) * w
			mfd = EvenlyDiscretizedMFD(self.min_mag, self.bin_width, occurrence_rates, self.Mtype)
			mfd_list.append(mfd)
		return mfd_list

	def split(self, M):
		"""
		Split MFD at a particular magnitude

		:param M:
			Float, magnitude value where MFD should be split

		:return:
			List containing 2 instances of :class:`EvenlyDiscretizedMFD`
		"""
		if not self.is_magnitude_compatible(M):
			raise Exception("Magnitude value not compatible!")
		elif self.get_min_mag_edge() < M < self.max_mag:
			index = int(round((M - self.get_min_mag_edge()) / self.bin_width))
			occurrence_rates1 = self.occurrence_rates[:index]
			occurrence_rates2 = self.occurrence_rates[index:]
			mfd1 = EvenlyDiscretizedMFD(self.min_mag, self.bin_width, occurrence_rates1, self.Mtype)
			mfd2 = EvenlyDiscretizedMFD(M+self.bin_width/2, self.bin_width, occurrence_rates2, self.Mtype)
			return [mfd1, mfd2]
		else:
			raise Exception("Split magnitude not in valid range!")

	def extend(self, other_mfd):
		"""
		Extend MFD with another one that covers larger magnitudes.

		:param other_mfd:
			instance of :class:`EvenlyDiscretizedMFD` or :class:`TruncatedGRMFD`
			The minimum magnitude of other_mfd should be equal to or larger than
			the maximum magnitude of this MFD.

		Note:
			Bins between both MFD's will be filled with zero incremental
			occurrence rates!
		"""
		magnitude_bin_edges = other_mfd.get_magnitude_bin_edges()
		occurrence_rates = other_mfd.occurrence_rates
		if not np.allclose(other_mfd.bin_width, self.bin_width):
			raise Exception("Bin width not compatible!")
		fgap = ((magnitude_bin_edges[0] - self.max_mag) / self.bin_width)
		gap = int(round(fgap))
		if not np.allclose(fgap, gap):
			raise Exception("Bin width not compatible!")

		num_empty_bins = gap + 1
		if num_empty_bins >= 0:
			occurrence_rates = np.concatenate([self.occurrence_rates, np.zeros(num_empty_bins, dtype='d'), occurrence_rates])
			self.occurrence_rates = list(occurrence_rates)
		else:
			raise Exception("Magnitudes must not overlap with MFD magnitude range. Sum MFD's instead")

	def append_characteristic_eq(self, Mc, return_period):
		"""
		Append magnitude-frequency of a characteristic earthquake

		:param Mc:
			Float, magnitude of characteristic earthquake, must be multiple
			of current bin width
		:param return_period:
			Float, return period in yr of characteristic earthquake
		"""
		Mc = np.floor(Mc / self.bin_width) * self.bin_width
		if self.is_magnitude_compatible(Mc):
			characteristic_mfd = CharacteristicMFD(Mc, return_period, self.bin_width, Mtype=self.Mtype)
			self.extend(characteristic_mfd)
		else:
			raise Exception("Characteristic magnitude should be multiple of bin width!")

	def to_truncated_GR_mfd(self, completeness, end_date, method="Weichert", b_val=None, verbose=False):
		"""
		Calculate truncated Gutenberg-Richter MFD using maximum likelihood estimation
		for variable observation periods for different magnitude increments.
		Adapted from calB.m and calBfixe.m Matlab modules written by Philippe Rosset (ROB, 2004),
		which is based on the method by Weichert, 1980 (BSSA, 70, Nr 4, 1337-1346).

		:param completeness:
			instance of :class:`Completeness`, necessary for Weichert method
		:param end_date:
			datetime.date or Int, end date with respect to which observation periods
			will be determined in Weichert method
		:param method:
			String, computation method (default: "Weichert":
			- "Weichert"
			- "LSQc" (least squares on cumulative values)
			- "LSQi" (least squares on incremental values)
		:param b_val:
			Float, fixed b value to constrain MLE estimation (default: None)
		:param verbose:
			Bool, whether some messages should be printed or not (default: False)

		:return:
			instance of :class:`TruncatedGRMFD`
		"""
		from eqcatalog.calcGR import calcGR_Weichert, calcGR_LSQ
		magnitudes = self.get_magnitude_bin_edges()
		if method == "Weichert":
			## Number of earthquakes in each bin according to completeness
			bins_N = self.get_num_earthquakes(completeness, end_date)
			a, b, stda, stdb = calcGR_Weichert(magnitudes, bins_N, completeness, end_date, b_val=b_val, verbose=verbose)
		elif method[:3] == "LSQ":
			if method == "LSQc":
				occurrence_rates = self.get_cumulative_rates()
			elif method == "LSQi":
				occurrence_rates = np.array(self.occurrence_rates)
			a, b, stda, stdb = calcGR_LSQ(magnitudes, occurrence_rates, b_val=b_val, verbose=verbose)
			if method == "LSQi":
				## Compute a value for cumulative MFD from discrete a value
				a2 = a
				dM = self.bin_width
				a = a2 + get_a_separation(b, dM)
		return TruncatedGRMFD(self.get_min_mag_edge(), self.max_mag, self.bin_width, a, b, a_sigma=stda, b_sigma=stdb, Mtype=self.Mtype)

	def get_max_mag_observed(self):
		"""
		Return maximum observed magnitude (max. magnitude with non-zero
		occurrence_rate)
		"""
		mag_bin_centers = self.get_magnitude_bin_centers()
		Mmax = mag_bin_centers[np.array(self.occurrence_rates) > 0][-1]
		return Mmax

	def get_Bayesian_Mmax_pdf(self, prior_model="CEUS_COMP", Mmax_obs=None, n=None, Mmin_n=4.5, b_val=None, bin_width=None, truncation=(5.5, 8.25), completeness=None, end_date=None, verbose=True):
		"""
		Compute Mmax distribution following Bayesian approach.

		:param prior_model:
			String, indicating which prior model should be considered, one of:
			- "EPRI_extended": extended crust in EPRI (1994)
			- "EPRI_non_extended": non-extended crust in EPRI (1994)
			- "CEUS_COMP": composite prior in CEUS (2012)
			- "CEUS_MESE": Mesozoic and younger extension in CEUS (2012)
			- "CEUS_NMESE": Non-Mesozoic and younger extension in CEUS (2012)
			(default: "CEUS_COMP")
		:param Mmax_obs:
			Float: maximum observed magnitude (default: None, will use highest
				magnitude bin center with non-zero occurrence rate)
		:param n:
			Int, number of earthquakes above minimu magnitude relevant for PSHA
			(default: None, will compute this parameter from MFD and Mmin)
		:param Mmin_n:
			Float, lower magnitude, used to count n, the number of earthquakes
			between Mmin and Mmax_obs (corresponds to lower magnitude in PSHA).
			If None, will use min_mag of MFD.
			(default: 4.5)
		:param b_val:
			Float, b value of MFD (default: None, will compute b value from MFD
			using Weichert method)
		:param bin_width:
			Float, magnitude bin width. If None, will take bin_width from MFD
			(default: None)
		:param truncation:
			Int or tuple, representing truncation of prior distribution.
			If int, truncation is interpreted as the number of standard deviations.
			If tuple, elements are interpreted as minimum and maximum magnitude of
			the distribution
			(default: (5.5, 8.25), corresponding to the truncation applied in CEUS)
		:param completeness:
			instance of :class:`Completeness` containing initial years of completeness
			and corresponding minimum magnitudes (default: None)
			This parameter is required if n and/or b_val have to be computed
		:param end_date:
			datetime.date or Int, end date of observation period corresponding to MFD
			This parameter is required if n and/or b_val have to be computed
		:param verbose:
			Bool, whether or not to print additional information (default: True)

		:return:
			(prior, likelihood, posterior, params) tuple
			- prior: instance of :class:`MmaxPMF`, prior distribution
			- likelihood: numpy array
			- posterior: instance of :class:`MmaxPMF`, posterior distribution
			- params: (observed Mmax, n, a, b) tuple
		"""
		from matplotlib import mlab
		from ..utils import seq
		from ..pmf import MmaxPMF

		## Global prior distributions
		if prior_model == "EPRI_extended":
			mean, sigma = 6.4, 0.8
		elif prior_model == "EPRI_non_extended":
			mean, sigma = 6.3, 0.5
		elif prior_model == "CEUS_COMP":
			mean, sigma = 7.2, 0.64
		elif prior_model == "CEUS_MESE":
			mean, sigma = 7.35, 0.75
		elif prior_model == "CEUS_NMESE":
			mean, sigma = 6.7, 0.61

		if not bin_width:
			bin_width = self.bin_width

		if isinstance(truncation, (int, float)):
			Mmin, Mmax = mean - truncation * sigma, mean + truncation * sigma
		else:
			Mmin, Mmax = truncation[:2]

		magnitudes = seq(Mmin, Mmax, bin_width)
		prior = mlab.normpdf(magnitudes, mean, sigma)
		prior /= np.sum(prior)

		## Regional likelihood function
		likelihood = np.ones_like(magnitudes, dtype='d')
		if Mmax_obs is None:
			Mmax_obs = self.get_max_mag_observed()
		## Note: this is more robust than magnitudes[magnitudes >= Mmax_obs]
		Mmax_obs_index = (np.abs(magnitudes - Mmax_obs)).argmin()
		if n is None:
			if not Mmin_n:
				Mmin_n = self.get_min_mag_edge()

			bins_N = self.get_num_earthquakes(completeness, end_date)
			n = np.add.reduce(bins_N[self.get_magnitude_bin_centers() > Mmin_n])
		if not b_val:
			## Set maximum magnitude of MFD to mean prior magnitude
			imfd = self.to_evenly_discretized_mfd(max_mag=mean)
			## Compute b value using Weichert method. Note that this will
			## use the minimum magnitude of the MFD rather than the imposed
			## Mmin. Weichert computations are usually more robust with the
			## lowest minimum magnitude
			gr_mfd = imfd.to_truncated_GR_mfd(completeness, end_date, method="Weichert", verbose=verbose)
			b_val = gr_mfd.b_val
			a_val = gr_mfd.a_val
		else:
			a_val = None
		if not np.isnan(b_val):
			beta = b_val * np.log(10)
			if verbose:
				print("Maximum observed magnitude: %.1f" % Mmax_obs)
				print("n(M > Mmin): %d" % n)
			likelihood = np.zeros_like(magnitudes)
			likelihood[Mmax_obs_index:] = (1 - np.exp(-beta * (magnitudes[Mmax_obs_index:] - Mmin_n))) ** -n
		## If b value is NaN, likelihood = ones, and posterior = prior

		## Posterior
		posterior = prior * likelihood
		posterior /= np.sum(posterior)

		## Replace zero probabilities with very small values to avoid error in PMF
		prior_pmf = MmaxPMF(magnitudes, prior.clip(1E-8))
		#likelihood_pmf = MmaxPMF(magnitudes, likelihood.clip(1E-8))
		likelihood /= np.sum(likelihood)
		posterior_pmf = MmaxPMF(magnitudes, posterior.clip(1E-8))
		params = (Mmax_obs, n, a_val, b_val)

		return prior_pmf, likelihood, posterior_pmf, params

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML evenlyDiscretizedIncrementalMFD element)

		:param encoding:
			String, unicode encoding (default: 'latin1')
		"""
		edi_elem = etree.Element(ns.EVENLY_DISCRETIZED_INCREMENTAL_MFD)
		edi_elem.set(ns.MINIMUM_MAGNITUDE, str(self.min_mag))
		edi_elem.set(ns.BIN_WIDTH, str(self.bin_width))
		#edi_elem.set(ns.OCCURRENCE_RATES, " ".join(map(str, self.occurrence_rates)))
		occur_rates_elem = etree.SubElement(edi_elem, ns.OCCURRENCE_RATES)
		occur_rates_elem.text = " ".join(map(str, self.occurrence_rates))
		return edi_elem

	def plot(self, color='k', style="o", label="", discrete=True, cumul_or_inc="both", completeness=None, end_year=None, Mrange=(), Freq_range=(), title="", lang="en", fig_filespec=None, ax=None, fig_width=0, dpi=300):
		"""
		Plot magnitude-frequency distribution

		:param color:
			matplotlib color specification (default: 'k')
		:param style:
			matplotlib symbol style or line style (default: 'o')
		:param label:
			String, plot labels (default: "")
		:param discrete:
			Bool, whether or not to plot discrete MFD (default: True)
		:param cumul_or_inc:
			String, either "cumul", "inc" or "both", indicating
			whether to plot cumulative MFD, incremental MFD or both
			(default: "both")
		:param completeness:
			instance of :class:`Completeness`, used to plot completeness
			limits (default: None)
		:param end_year:
			Int, end year of catalog (used when plotting completeness limits)
			(default: None, will use current year)
		:param Mrange:
			(Mmin, Mmax) tuple, minimum and maximum magnitude in X axis
			(default: ())
		:param Freq_range:
			(Freq_min, Freq_max) tuple, minimum and maximum values in frequency
			(Y) axis (default: ())
		:param title:
			String, plot title (default: "")
		:param lang:
			String, language of plot axis labels (default: "en")
		:param fig_filespec:
			String, full path to output image file, if None plot to screen
			(default: None)
		:param ax:
			instance of :class:`~matplotlib.axes.Axes` in which plot will
			be made (default: None, will create new figure and axes)
		:param fig_width:
			Float, figure width in cm, used to recompute :param:`dpi` with
			respect to default figure width (default: 0)
		:param dpi:
			Int, image resolution in dots per inch (default: 300)

		:return:
			if both :param:`ax` and :param:`fig_filespec` are None, a
			(ax, fig) tuple will be returned
		"""
		return plot_MFD([self], colors=[color], styles=[style], labels=[label], discrete=[discrete], cumul_or_inc=[cumul_or_inc], completeness=completeness, end_year=end_year, Mrange=Mrange, Freq_range=Freq_range, title=title, lang=lang, fig_filespec=fig_filespec, ax=ax, fig_width=fig_width, dpi=dpi)



class CharacteristicMFD(EvenlyDiscretizedMFD):
	"""
	MFD representing a characteristic earthquake, implemented as an
	evenly discretized MFD with one magnitude bin. The characteristic
	magnitde is taken to correspond to the left edge of the bin.

	:param char_mag:
		Float, magnitude of characteristic earthquake
	:param return_period:
		Float, return period of characteristic earthquake in year
	:param bin_width:
		Float, magnitude bin width
	:param M_sigma:
		Float, standard deviation on magnitude (default: 0.3)
	:param num_sigma:
		Float, number of standard deviations to spread occurrence rates over
		(default: 0)
	"""
	def __init__(self, char_mag, return_period, bin_width, M_sigma=0.3, num_sigma=0):
		self.char_mag = char_mag
		self.char_return_period = return_period
		self.M_sigma = M_sigma
		self.num_sigma = num_sigma

		Mmin, occurrence_rates = self._get_evenly_discretized_mfd_params(num_sigma, bin_width=bin_width)
		EvenlyDiscretizedMFD.__init__(self, Mmin, bin_width, occurrence_rates, Mtype="MW")

	def _get_evenly_discretized_mfd_params(self, num_sigma, bin_width=None):
		"""
		Compute parameters for constructing evenly discretized MFD

		:param num_sigma:
			Float, number of standard deviations to spread occurrence rates over
		:param bin_width:
			Float,  magnitude bin width
			(default: None, will use currently set bin_width)

		:return:
			(Mmin, occurrence_rates) tuple
			Note: Mmin corresponds to magnitude bin center!
		"""
		from matplotlib import mlab
		from scipy.stats import norm

		if not bin_width:
			bin_width = self.bin_width
		if self.M_sigma and num_sigma:
			Mmin = self.char_mag - self.M_sigma * num_sigma
			Mmax = self.char_mag + self.M_sigma * num_sigma
			Mmin = np.floor(Mmin / bin_width) * bin_width
			Mmax = np.ceil(Mmax / bin_width) * bin_width
			magnitudes = np.arange(Mmin, Mmax, bin_width)
			probs = mlab.normpdf(magnitudes + bin_width/2, self.char_mag, self.M_sigma)
			probs /= np.sum(probs)
			occurrence_rates = (1./self.char_return_period) * probs
			## CRISIS formula
			#EM = char_mag
			#s = M_sigma * num_sigma
			#Mu = Mmax
			#M0 = Mmin
			#probs = (norm.cdf(Mu, EM, s) - norm.cdf(magnitudes+bin_width/2, EM, s)) / (norm.cdf(Mu, EM, s) - norm.cdf(M0, EM, s))
			#cumul_rates = (1./return_period) * probs
			#occurrence_rates = cumul_rates[:-1] - cumul_rates[1:]
			#occurrence_rates = np.append(occurrence_rates, cumul_rates[-1:])
		else:
			Mmin = self.char_mag
			occurrence_rates = np.array([1./self.char_return_period])
		return Mmin + bin_width/2, occurrence_rates

	def set_num_sigma(self, num_sigma):
		"""
		Set number of standard deviations around characteristic magnitude

		:param num_sigma:
			Float, number of standard deviations to spread occurrence rates over
		"""
		Mmin, occurrence_rates = self._get_evenly_discretized_mfd_params(num_sigma)
		self.min_mag = Mmin
		self.modify_set_occurrence_rates(occurrence_rates)

	def __div__(self, other):
		if isinstance(other, (int, float)):
			return_period = self.return_period * other
			return CharacteristicMFD(self.char_mag, self.return_period, self.bin_width, self.M_sigma, self.num_sigma)
		else:
			raise TypeError("Divisor must be integer or float")

	def __mul__(self, other):
		if isinstance(other, (int, float)):
			return_period = self.return_period / other
			return CharacteristicMFD(self.char_mag, self.return_period, self.bin_width, self.M_sigma, self.num_sigma)
		else:
			raise TypeError("Multiplier must be integer or float")

	@property
	def return_period(self):
		return 1. / np.sum(self.occurrence_rates)


class TruncatedGRMFD(nhlib.mfd.TruncatedGRMFD, MFD):
	"""
	Truncated or modified Gutenberg-Richter MFD

	:param min_mag:
		The lowest possible magnitude for this MFD. The first bin in the
		:meth:`result histogram <get_annual_occurrence_rates>` will be aligned
		to make its left border match this value.
	:param max_mag:
		The highest possible magnitude. The same as for ``min_mag``: the last
		bin in the histogram will correspond to the magnitude value equal to
		``max_mag - bin_width / 2``.
	:param bin_width:
		A positive float value -- the width of a single histogram bin.
	:param a_val:
		Float, the cumulative ``a`` value (``10 ** a`` is the number
		of earthquakes per year with magnitude greater than or equal to 0),
	:param b_val:
		Float, Gutenberg-Richter ``b`` value -- the decay rate
		of exponential distribution. It describes the relative size distribution
		of earthquakes: a higher ``b`` value indicates a relatively larger
		proportion of small events and vice versa.
	:param a_sigma:
		Float, standard deviation of the a value (default: 0).
	:param b_sigma:
		Float, standard deviation of the b value (default: 0).
	:param Mtype:
		String, magnitude type, either "MW" or "MS" (default: "MW")

	Note:
		Values for ``min_mag`` and ``max_mag`` don't have to be aligned with
		respect to ``bin_width``. They get rounded accordingly anyway so that
		both are divisible by ``bin_width`` just before converting a function
		to a histogram. See :meth:`_get_min_mag_and_num_bins`.
	"""
	def __init__(self, min_mag, max_mag, bin_width, a_val, b_val, a_sigma=0, b_sigma=0, Mtype="MW"):
		nhlib.mfd.TruncatedGRMFD.__init__(self, min_mag, max_mag, bin_width, a_val, b_val)
		self.a_sigma = a_sigma
		self.b_sigma = b_sigma
		self.Mtype = Mtype

	def __div__(self, other):
		if isinstance(other, (int, float)):
			N0 = 10**self.a_val
			a_val = np.log10(N0 / float(other))
			## a_sigma does not change
			a_sigma = self.a_sigma
			return TruncatedGRMFD(self.min_mag, self.max_mag, self.bin_width, a_val, self.b_val, a_sigma, self.b_sigma, self.Mtype)
		else:
			raise TypeError("Divisor must be integer or float")

	def __mul__(self, other):
		if isinstance(other, (int, float)):
			N0 = 10**self.a_val
			a_val = np.log10(N0 * float(other))
			## a_sigma does not change
			a_sigma = self.a_sigma
			return TruncatedGRMFD(self.min_mag, self.max_mag, self.bin_width, a_val, self.b_val, a_sigma, self.b_sigma, self.Mtype)
		else:
			raise TypeError("Multiplier must be integer or float")

	def __add__(self, other):
		if isinstance(other, (TruncatedGRMFD, EvenlyDiscretizedMFD)):
			return sum_MFDs([self, other])
		else:
			raise TypeError("Operand must be MFD")

	def __sub__(self, other):
		if isinstance(other, TruncatedGRMFD):
			if self.min_mag == other.min_mag and self.max_mag == other.max_mag and self.b_val == other.b_val and self.Mtype == other.Mtype:
				## Note: bin width does not have to be the same here
				N0 = 10 ** self.a_val - 10 ** other.a_val
				a_val = np.log10(N0)
				## Error propagation, see http://chemwiki.ucdavis.edu/Analytical_Chemistry/Quantifying_Nature/Significant_Digits/Propagation_of_Error
				a_sigma = 0.434 * ((self.get_N0_sigma() + other.get_N0_sigma()) / N0)
				b_sigma = np.mean(self.b_sigma, other.b_sigma)
				return TruncatedGRMFD(self.min_mag, self.max_mag, self.bin_width, a_val, self.b_val, a_sigma, b_sigma, self.Mtype)
		elif isinstance(other, EvenlyDiscretizedMFD):
			return self.to_evenly_discretized_mfd().__sub__(other)
		else:
			raise TypeError("Operand must be MFD")

	@property
	def occurrence_rates(self):
		return np.array(zip(*self.get_annual_occurrence_rates())[1])

	@property
	def beta(self):
		return np.log(10) * self.b_val

	@property
	def alpha(self):
		return np.log(10) * self.a_val

	def get_copy(self):
		"""
		Return a copy of the MFD

		:return:
			instance of :class:`TruncatedGRMFD`
		"""
		return self.to_truncated_GR_mfd()

	def get_min_mag_edge(self):
		"""
		Return left edge of minimum magnitude bin

		:return:
			Float
		"""
		return self.min_mag

	def get_min_mag_center(self):
		"""
		Return center value of minimum magnitude bin

		:return:
			Float
		"""
		return self.min_mag + self.bin_width / 2

	def get_center_magnitudes(self):
		"""
		Return array with center magnitudes of each bin

		:return:
			ndarray, magnitudes
		"""
		magnitudes = np.arange(self.get_min_mag_center(), self.max_mag, self.bin_width)
		return magnitudes

	def set_min_mag(self, min_mag):
		"""
		Set minimum magnitude

		:param min_mag:
			float, new minimum magnitude (left edge)
		"""
		assert min_mag <= self.max_mag
		self.min_mag = min_mag

	def get_cumulative_rates(self):
		"""
		Return cumulative annual occurrence rates

		:return:
			numpy float array
		"""
		a, b = self.a_val, self.b_val
		min_mag, max_mag = self.get_min_mag_edge(), self.max_mag
		mags = self.get_magnitude_bin_edges()
		return (10**(a-b*min_mag))*((10**(-1*b*mags)-10**(-1*b*max_mag))/(10**(-1*b*min_mag)-10**(-1*b*max_mag)))
		## Note: the following is identical
		#return np.add.accumulate(self.occurrence_rates[::-1])[::-1]

	def get_N0_sigma(self):
		"""
		Return standard deviation on annual occurrence rate of M=0,
		computed from a_sigma

		:return:
			Float
		"""
		N = 10**self.a_val
		return N * 2.303 * self.a_sigma

	def divide(self, weights):
		"""
		Divide MFD into a number of MFD's that together sum up to the original MFD

		:param weights:
			list or array containing weight of each sub-MFD

		:return:
			List containing instances of :class:`TruncatedGRMFD`
		"""
		weights = np.array(weights, dtype='d')
		weights /= np.add.reduce(weights)
		N0 = 10**self.a_val
		avalues = np.log10(weights * N0)
		## a_sigma does not change
		a_sigma = self.a_sigma
		return [TruncatedGRMFD(self.min_mag, self.max_mag, self.bin_width, aw, self.b_val, a_sigma, self.b_sigma, self.Mtype) for aw in avalues]

	def split(self, M):
		"""
		Split MFD at a particular magnitude

		:param M:
			Float, magnitude value where MFD should be split

		:return:
			List containing 2 instances of :class:`TruncatedGRMFD`
		"""
		if not self.is_magnitude_compatible(M):
			raise Exception("Magnitude value not compatible!")
		elif self.get_min_mag_edge() < M < self.max_mag:
			mfd1 = TruncatedGRMFD(self.min_mag, M, self.bin_width, self.a_val, self.b_val, self.a_sigma, self.b_sigma, self.Mtype)
			mfd2 = TruncatedGRMFD(M, self.max_mag, self.bin_width, self.a_val, self.b_val, self.a_sigma, self.b_sigma, self.Mtype)
			return [mfd1, mfd2]
		else:
			raise Exception("Split magnitude not in valid range!")

	def extend(self, other_mfd):
		"""
		Extend MFD with another one that covers larger magnitudes.

		:param other_mfd:
			instance of :class:`TruncatedGRMFD`
			The minimum magnitude of other_mfd should be equal to the
			maximum magnitude of this MFD.

		Note:
			If other_mfd is instance of :class:`EvenlyDiscretizedGRMFD`
			or if its min_mag is larger than max_mag of this MFD, an
			exception will be raised, prompting to convert to an instance
			of :class:`EvenlyDiscretizedGRMFD` first.
		"""
		if isinstance(other_mfd, TruncatedGRMFD) and other_mfd.b_val == self.bval and other_mfd.min_mag == self.max_mag:
			self.max_mag = other_mfd.max_mag
		else:
			raise Exception("MFD objects not compatible. Convert to EvenlyDiscretizedMFD")

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML truncGutenbergRichterMFD element)

		:param encoding:
			String, unicode encoding (default: 'latin1')
		"""
		tgr_elem = etree.Element(ns.TRUNCATED_GUTENBERG_RICHTER_MFD)
		tgr_elem.set(ns.A_VALUE, str(self.a_val))
		tgr_elem.set(ns.B_VALUE, str(self.b_val))
		tgr_elem.set(ns.MINIMUM_MAGNITUDE, str(self.min_mag))
		tgr_elem.set(ns.MAXIMUM_MAGNITUDE, str(self.max_mag))

		return tgr_elem

	def plot(self, color='k', style="-", label="", discrete=False, cumul_or_inc="cumul", completeness=None, end_year=None, Mrange=(), Freq_range=(), title="", lang="en", fig_filespec=None, ax=None, fig_width=0, dpi=300):
		"""
		Plot magnitude-frequency distribution

		:param color:
			matplotlib color specification (default: 'k')
		:param style:
			matplotlib symbol style or line style (default: '-')
		:param label:
			String, plot labels (default: "")
		:param discrete:
			Bool, whether or not to plot discrete MFD (default: False)
		:param cumul_or_inc:
			String, either "cumul", "inc" or "both", indicating
			whether to plot cumulative MFD, incremental MFD or both
			(default: "cumul")
		:param completeness:
			instance of :class:`Completeness`, used to plot completeness
			limits (default: None)
		:param end_year:
			Int, end year of catalog (used when plotting completeness limits)
			(default: None, will use current year)
		:param Mrange:
			(Mmin, Mmax) tuple, minimum and maximum magnitude in X axis
			(default: ())
		:param Freq_range:
			(Freq_min, Freq_max) tuple, minimum and maximum values in frequency
			(Y) axis (default: ())
		:param title:
			String, plot title (default: "")
		:param lang:
			String, language of plot axis labels (default: "en")
		:param fig_filespec:
			String, full path to output image file, if None plot to screen
			(default: None)
		:param ax:
			instance of :class:`~matplotlib.axes.Axes` in which plot will
			be made (default: None, will create new figure and axes)
		:param fig_width:
			Float, figure width in cm, used to recompute :param:`dpi` with
			respect to default figure width (default: 0)
		:param dpi:
			Int, image resolution in dots per inch (default: 300)

		:return:
			if both :param:`ax` and :param:`fig_filespec` are None, a
			(ax, fig) tuple will be returned
		"""
		return plot_MFD([self], colors=[color], styles=[style], labels=[label], discrete=[discrete], cumul_or_inc=[cumul_or_inc], completeness=completeness, end_year=end_year, Mrange=Mrange, Freq_range=Freq_range, title=title, lang=lang, fig_filespec=fig_filespec, ax=ax, fig_width=fig_width, dpi=dpi)

	def to_truncated_GR_mfd(self, min_mag=None, max_mag=None, bin_width=None):
		"""
		Copy to another instance of :class:`TruncatedGRMFD`
		Optionally, non-defining parameters can be changed

		:param min_mag:
			Float, lower magnitude (bin edge) of MFD (default: None)
		:param max_mag:
			Float, upper magnitude of MFD (default: None)
		:param bin_width:
			Float, magnitude bin width of MFD (default: None)
		"""
		if min_mag is None:
			min_mag = self.min_mag
		if max_mag is None:
			max_mag = self.max_mag
		if bin_width is None:
			bin_width = self.bin_width
		return TruncatedGRMFD(min_mag, max_mag, bin_width, self.a_val, self.b_val, self.a_sigma, self.b_sigma, self.Mtype)

	def get_mfd_from_b_val(self, b_val):
		"""
		Construct new MFD from a given b value, assuming uncertainties
		on a and b values are correlated. This supposes that the a and
		b values of the current MFD correspond to the mean values, and
		requires that the standard deviations of the a and b values are
		nonzero.

		:param b_val:
			Float, imposed b value

		:return:
			instance of :class:`TruncatedGRMFD`
		"""
		if self.b_sigma == 0 or self.a_sigma == 0:
			raise Exception("a_sigma and b_sigma must not be zero!")
		epsilon = (b_val - self.b_val) / self.b_sigma
		a_val = self.a_val + self.a_sigma * epsilon
		return TruncatedGRMFD(self.min_mag, self.max_mag, self.bin_width, a_val, b_val, Mtype=self.Mtype)

	def sample_from_b_val(self, num_samples, num_sigma=2, random_seed=None):
		"""
		Generate MFD's by Monte Carlo sampling of the b value.
		The corresponding a value is computed from a_sigma (see
		:meth:`get_mfd_from_b_val`

		:param num_samples:
			Int, number of MFD samples to generate
		:param num_sigma:
			Float, number of standard deviations on b value (default: 2)
		:param random_seed:
			None or int, seed to initialize internal state of random number
			generator (default: None, will seed from current time)

		:return:
			List with instances of :class:`TruncatedGRMFD`
		"""
		import scipy.stats

		mfd_samples = []
		np.random.seed(seed=random_seed)
		mu, sigma = self.b_val, self.b_sigma
		b_values = scipy.stats.truncnorm.rvs(-num_sigma, num_sigma, mu, sigma, size=num_samples)
		for b_val in b_values:
			mfd_samples.append(self.get_mfd_from_b_val(b_val))
		return mfd_samples

	@classmethod
	def construct_FentonEtAl2006MFD(self, min_mag, max_mag, bin_width, area, b_val=0.7991):
		"""
		Construct "minimum" MFD for SCR according to Fenton et al. (2006),
		based on surface area

		:param min_mag:
			Float, Minimum magnitude (default: None, take min_mag from current MFD).
		:param max_mag:
			Maximum magnitude (default: None, take max_mag from current MFD).
		:param bin_width:
			Float, Magnitude interval for evenly discretized magnitude frequency
			distribution (default: None, take bin_width from current MFD.
		:param area:
			Float, area of region in square km
		:param b_val:
			Float, Parameter of the truncated gutenberg richter model.
			(default: 0.7991)
		"""
		beta, std_beta = 1.84, 0.24
		stdb = std_beta / np.log(10)
		if b_val is None:
			b_val = beta / np.log(10)
		lamda = 0.004 * area / 1E6
		try:
			a_val = a_from_lambda(lamda, 6.0, b_val)
		except:
			raise()
		## stda is not defined
		stda = 0
		return TruncatedGRMFD(min_mag, max_mag, bin_width, a_val, b_val, stda, stdb, Mtype="MS")

	@classmethod
	def construct_Johnston1994MFD(self, min_mag, max_mag, bin_width, area, region="total"):
		"""
		Construct "minimum" MFD for SCR according to Johnston (1994),
		based on surface area

		:param min_mag:
			Float, Minimum magnitude (default: None, take min_mag from current MFD).
		:param max_mag:
			Maximum magnitude (default: None, take max_mag from current MFD).
		:param bin_width:
			Float, Magnitude interval for evenly discretized magnitude frequency
			distribution (default: None, take bin_width from current MFD.
		:param area:
			Float, area of region in square km
		:param region:
			str, SCR region ("africa", "australia", "europe", "china", "india",
			"north america", "na extended", "na non-extended", "south america",
			"total", "total extended", or "total non-extended")
			(default: "total")
		"""
		## Table 4-7, page 4-60
		if region.lower() == "africa":
			a, b = 2.46, 0.982
			## Note: since a_normalized = a + log10(100000 / area),
			## standard deviation on a_normalized is same as on a
			stda, stdb = 0.073, 0.119
		elif region.lower() == "antarctica":
			a, b = 1.27, 1.0
			stda, stdb = None, None
		elif region.lower() in ("asia", "russia"):
			a, b = 2.09, 1.16
			stda, stdb = None, None
		elif region.lower() == "australia":
			a, b = 2.29, 0.896
			stda, stdb = 0.051, 0.077
		elif region.lower() == "europe":
			a, b = 3.32, 1.156
			stda, stdb = 0.069, 0.106
		elif region.lower() == "china":
			a, b = 2.96, 1.029
			stda, stdb = 0.096, 0.109
		elif region.lower() == "india":
			a, b = 3.02, 0.966
			stda, stdb = 0.101, 0.154
		elif region.lower() == "north america":
			a, b = 1.12, 0.728
			stda, stdb = 0.056, 0.067
		elif region.lower() == "na extended":
			a, b = 1.33, 0.747
			stda, stdb = 0.063, 0.076
		elif region.lower() == "na non-extended":
			a, b = 1.32, 0.790
			stda, stdb = 0.107, 0.158
		elif region.lower() == "south america":
			a, b = 3.46, 1.212
			stda, stdb = 0.130, 0.270
		elif region.lower() == "total":
			a, b = 2.46, 0.975
			stda, stdb = 0.030, 0.047
		elif region.lower() == "total extended":
			a, b = 2.36, 0.887
			stda, stdb = 0.039, 0.054
		elif region.lower() == "total non-extended":
			a, b = 3.26, 1.186
			stda, stdb = 0.049, 0.094

		mfd = TruncatedGRMFD(min_mag, max_mag, bin_width, a, b, stda, stdb, Mtype="MW")
		return mfd * (area / 1E5)

	def get_Mmax_from_moment_rate(self, moment_rate):
		"""
		Determine maximum possible magnitude that is in agreement with
		the MFD and a given moment rate

		:param moment_rate:
			float, moment rate in N.m/yr

		:return:
			float, maximum magnitude
		"""
		for max_mag in np.arange(self.min_mag + self.bin_width, 10, self.bin_width):
			mfd = TruncatedGRMFD(self.min_mag, max_mag, self.bin_width, self.a_val, self.b_val)
			if mfd.get_total_moment_rate() > moment_rate:
				break
		return max_mag

	def print_report(self):
		"""
		Report cumulative frequencies for whole magnitude units
		"""
		mags = self.get_magnitude_bin_edges()
		idxs = [0]
		second_idx = int(round((np.ceil(mags[0]) - mags[0]) / self.bin_width))
		idxs += range(second_idx, len(self), int(round(1./self.bin_width)))
		if idxs[1] == 0:
			idxs = idxs[1:]

		if not self.a_sigma:
			cumul_rates = self.get_cumulative_rates()
			for i in idxs:
				M = mags[i]
				rate = cumul_rates[i]
				if rate > 1:
					print("%s>=%.1f: %.1f per year" % (self.Mtype, M, rate))
				else:
					print("%s>=%.1f: 1 every %.0f years" % (self.Mtype, M, 1./rate))
		else:
			mfd1 = self.get_mfd_from_b_val(self.b_val + self.b_sigma)
			mfd2 = self.get_mfd_from_b_val(self.b_val - self.b_sigma)
			cumul_rates1 = mfd1.get_cumulative_rates()
			cumul_rates2 = mfd2.get_cumulative_rates()
			for i in idxs:
				M = mags[i]
				rate1, rate2 = cumul_rates1[i], cumul_rates2[i]
				if rate1 > 1 or rate2 > 1:
					print("%s>=%.1f: %.1f - %.1f per year" % ((self.Mtype, M) + tuple(np.sort([rate1, rate2]))))
				else:
					print("%s>=%.1f: 1 every %.0f - %.0f years" % ((self.Mtype, M) + tuple(np.sort([1./rate1, 1./rate2]))))



class YoungsCoppersmith1985MFD(nhlib.mfd.YoungsCoppersmith1985MFD, EvenlyDiscretizedMFD):
	"""
	Class implementing the MFD for the 'Characteristic Earthquake Model'
	by Youngs & Coppersmith (1985)

	:param min_mag:
		The lowest possible magnitude for this MFD. The first bin in the
		:meth:`result histogram <get_annual_occurrence_rates>` will be aligned
		to make its left border match this value.
	:param a_val:
		Float, the cumulative ``a`` value (``10 ** a`` is the number
		of earthquakes per year with magnitude greater than or equal to 0),
	:param b_val:
		Float, Gutenberg-Richter ``b`` value -- the decay rate
		of exponential distribution. It describes the relative size distribution
		of earthquakes: a higher ``b`` value indicates a relatively larger
		proportion of small events and vice versa.
	:param char_mag:
		The characteristic magnitude defining the middle point of the
		characteristic distribution. That is the boxcar function representing
		the characteristic distribution is defined in the range
		[char_mag - 0.25, char_mag + 0.25].
	:param char_rate:
		The characteristic rate associated to the characteristic magnitude,
		to be distributed over the domain of the boxcar function representing
		the characteristic distribution (that is: char_rate / 0.5)
	:param bin_width:
		A positive float value -- the width of a single histogram bin.
	"""
	def __init__(self, min_mag, a_val, b_val, char_mag, char_rate, bin_width):
		super(YoungsCoppersmith1985MFD, self).__init__(min_mag, a_val, b_val, char_mag, char_rate, bin_width)
		self.Mtype = "MW"

	@property
	def occurrence_rates(self):
		return np.array(zip(*self.get_annual_occurrence_rates())[1])

	@property
	def beta(self):
		return np.log(10) * self.b_val

	@property
	def alpha(self):
		return np.log(10) * self.a_val

	@property
	def max_mag(self):
		return self.get_min_mag_edge() + len(self.occurrence_rates) * self.bin_width

	def get_min_mag_edge(self):
		"""
		Return left edge of minimum magnitude bin

		:return:
			Float
		"""
		return self.min_mag

	def get_min_mag_center(self):
		"""
		Return center value of minimum magnitude bin

		:return:
			Float
		"""
		return self.min_mag + self.bin_width / 2

	def plot(self, color='k', style="-", label="", discrete=True, cumul_or_inc="both", completeness=None, end_year=None, Mrange=(), Freq_range=(), title="", lang="en", fig_filespec=None, ax=None, fig_width=0, dpi=300):
		"""
		Plot magnitude-frequency distribution

		:param color:
			matplotlib color specification (default: 'k')
		:param style:
			matplotlib symbol style or line style (default: '-')
		:param label:
			String, plot labels (default: "")
		:param discrete:
			Bool, whether or not to plot discrete MFD (default: True)
		:param cumul_or_inc:
			String, either "cumul", "inc" or "both", indicating
			whether to plot cumulative MFD, incremental MFD or both
			(default: "both")
		:param completeness:
			instance of :class:`Completeness`, used to plot completeness
			limits (default: None)
		:param end_year:
			Int, end year of catalog (used when plotting completeness limits)
			(default: None, will use current year)
		:param Mrange:
			(Mmin, Mmax) tuple, minimum and maximum magnitude in X axis
			(default: ())
		:param Freq_range:
			(Freq_min, Freq_max) tuple, minimum and maximum values in frequency
			(Y) axis (default: ())
		:param title:
			String, plot title (default: "")
		:param lang:
			String, language of plot axis labels (default: "en")
		:param fig_filespec:
			String, full path to output image file, if None plot to screen
			(default: None)
		:param ax:
			instance of :class:`~matplotlib.axes.Axes` in which plot will
			be made (default: None, will create new figure and axes)
		:param fig_width:
			Float, figure width in cm, used to recompute :param:`dpi` with
			respect to default figure width (default: 0)
		:param dpi:
			Int, image resolution in dots per inch (default: 300)

		:return:
			if both :param:`ax` and :param:`fig_filespec` are None, a
			(ax, fig) tuple will be returned
		"""
		return plot_MFD([self], colors=[color], styles=[style], labels=[label], discrete=[discrete], cumul_or_inc=[cumul_or_inc], completeness=completeness, end_year=end_year, Mrange=Mrange, Freq_range=Freq_range, title=title, lang=lang, fig_filespec=fig_filespec, ax=ax, fig_width=fig_width, dpi=dpi)


def sum_MFDs(mfd_list, weights=[]):
	"""
	Sum two or more MFD's

	:param mfd_list:
		List containing instances of :class:`EvenlyDiscretizedMFD` or
		:class:`TruncatedGRMFD`

	:param weights:
		List or array containing weights of each MFD (default: [])

	:return:
		instance of :class:`TruncatedGRMFD` (if all MFD's in list are
		TruncatedGR, and have same min_mag, max_mag, and b_val) or else
		instance of :class:`EvenlyDiscretizedMFD`

	Note:
		Weights will be normalized!
	"""
	if weights in ([], None):
		weights = np.ones(len(mfd_list), 'd')
	total_weight = np.add.reduce(weights)
	weights = (np.array(weights) / total_weight) * len(mfd_list)
	bin_width = min([mfd.bin_width for mfd in mfd_list])
	Mtype = mfd_list[0].Mtype
	for mfd in mfd_list:
		if mfd.bin_width != bin_width:
			raise Exception("Bin widths not compatible!")
		if mfd.Mtype != Mtype:
			raise Exception("Magnitude types not compatible!")
	all_min_mags = set([mfd.get_min_mag_edge() for mfd in mfd_list])
	all_max_mags = set([mfd.max_mag for mfd in mfd_list])
	## If all MFD's are TruncatedGR, and have same min_mag, max_mag, and b_val
	## return TrucatedGR, else return EvenlyDiscretized
	is_truncated = np.array([isinstance(mfd, TruncatedGRMFD) for mfd in mfd_list])
	if is_truncated.all():
		all_bvals = set([mfd.b_val for mfd in mfd_list])
		if len(all_min_mags) == len(all_max_mags) == len(all_bvals) == 1:
			## TruncatedGR's can be summed into another TruncatedGR object
			all_avals = np.array([mfd.a_val for mfd in mfd_list])
			N0 = 10**all_avals
			N0sum = np.add.reduce(N0 * weights)
			a = np.log10(N0sum)
			## Error propagation, see http://chemwiki.ucdavis.edu/Analytical_Chemistry/Quantifying_Nature/Significant_Digits/Propagation_of_Error
			Nsum_sigma = np.sum([mfd.get_N0_sigma() * w for (mfd, w) in zip(mfd_list, weights)])
			a_sigma = 0.434 * (Nsum_sigma / N0sum)
			b_sigma = np.mean([mfd.b_sigma for mfd in mfd_list])
			mfd = mfd_list[0]
			return TruncatedGRMFD(mfd.min_mag, mfd.max_mag, mfd.bin_width, a, mfd.b_val, a_sigma, b_sigma, mfd.Mtype)
		else:
			## TruncatedGR's can be summed after conversion to EvenlyDiscretized
			pass

	min_mag = min(all_min_mags)
	max_mag = max(all_max_mags)
	num_bins = int(round((max_mag - min_mag) / bin_width))
	occurrence_rates = np.zeros(num_bins, 'd')
	for i, mfd in enumerate(mfd_list):
		start_index = int(round((mfd.get_min_mag_edge() - min_mag) / bin_width))
		end_index = start_index + len(mfd.occurrence_rates)
		occurrence_rates[start_index:end_index] += (np.array(mfd.occurrence_rates) * weights[i])
	return EvenlyDiscretizedMFD(min_mag+bin_width/2, bin_width, occurrence_rates, Mtype)


def plot_MFD(mfd_list, colors=[], styles=[], labels=[], discrete=[], cumul_or_inc=[], completeness=None, end_year=None, Mrange=(), Freq_range=(), title="", lang="en", legend_location=1, fig_filespec=None, ax=None, fig_width=0, dpi=300):
	"""
	Plot one or more magnitude-frequency distributions

	:param mfd_list:
		List with instance of :class:`EvenlyDiscretizedMFD` or :class:`TruncatedGRMFD`
	:param colors:
		List with matplotlib color specifications, one for each mfd
		(default: [])
	:param styles:
		List with matplotlib symbol styles or line styles, one for each mfd
		(default: [])
	:param labels:
		List with plot labels, one for each mfd (default: [])
	:param discrete:
		List of bools, whether or not to plot discrete MFD's (default: [])
	:param cumul_or_inc:
		List of strings, either "cumul", "inc" or "both", indicating
		whether to plot cumulative MFD, incremental MFD or both
		(default: [])
	:param completeness:
		instance of :class:`Completeness`, used to plot completeness
		limits (default: None)
	:param end_year:
		Int, end year of catalog (used when plotting completeness limits)
		(default: None, will use current year)
	:param Mrange:
		(Mmin, Mmax) tuple, minimum and maximum magnitude in X axis
		(default: ())
	:param Freq_range:
		(Freq_min, Freq_max) tuple, minimum and maximum values in frequency
		(Y) axis (default: ())
	:param title:
		String, plot title (default: "")
	:param lang:
		String, language of plot axis labels (default: "en")
	:param legend_location:
		int or str, matplotlib specification for legend location (default: 1)
	:param fig_filespec:
		String, full path to output image file, if None plot to screen
		(default: None)
	:param ax:
		instance of :class:`~matplotlib.axes.Axes` in which plot will
		be made (default: None, will create new figure and axes)
	:param fig_width:
		Float, figure width in cm, used to recompute :param:`dpi` with
		respect to default figure width (default: 0)
	:param dpi:
		Int, image resolution in dots per inch (default: 300)

	:return:
		if both :param:`ax` and :param:`fig_filespec` are None, a
		(ax, fig) tuple will be returned
	"""
	if ax is None:
		## Note: clf call seems to create a figure as well
		pylab.clf()
		fig = pylab.gcf()
		#fig = pylab.figure()
		ax = fig.add_subplot(1, 1, 1)
		interactive = True
	else:
		ax.cla()
		fig = pylab.gcf()
		interactive = False

	if not colors:
		colors = ("r", "g", "b", "c", "m", "k")

	if not labels:
		labels = [""] * len(mfd_list)

	if isinstance(discrete, bool):
		discrete = [discrete] * len(mfd_list)

	if isinstance(cumul_or_inc, (str, unicode)):
		cumul_or_inc = [cumul_or_inc] * len(mfd_list)

	## Plot
	## Line below removed because matplotlib crashes if this function is
	## called more than once
	#fig = pylab.figure()

	for i, mfd in enumerate(mfd_list):
		color = colors[i % len(colors)]

		try:
			want_discrete = discrete[i]
		except:
			if isinstance(mfd, TruncatedGRMFD):
				want_discrete = False
			else:
				want_discrete = True

		try:
			cumul_or_inc[i]
		except:
			if isinstance(mfd, TruncatedGRMFD):
				want_cumulative = True
				want_incremental = False
			else:
				want_cumulative = True
				want_incremental = True
		else:
			if cumul_or_inc[i] == "cumul":
				want_cumulative = True
				want_incremental = False
			elif cumul_or_inc[i] == "inc":
				want_cumulative = False
				want_incremental = True
			else:
				want_cumulative = True
				want_incremental = True

		## Discrete MFD
		if want_discrete:
			try:
				symbol = styles[i]
			except:
				symbol = 'o'
			else:
				if symbol in ("", None, "-", "--", ":", ":."):
					symbol = "o"

			## Cumulative
			if want_cumulative:
				label = labels[i]
				if want_incremental:
					label += " (cumul.)"
				obj_list = ax.semilogy(mfd.get_magnitude_bin_edges(), mfd.get_cumulative_rates(), symbol, label=label)
				pylab.setp(obj_list, markersize=10.0, markeredgewidth=1.0, markeredgecolor='k', markerfacecolor=color)

			## Incremental
			if want_incremental:
				label = labels[i] + " (inc.)"
				obj_list = ax.semilogy(mfd.get_magnitude_bin_centers(), mfd.occurrence_rates, symbol, label=label)
				pylab.setp(obj_list, markersize=10.0, markeredgewidth=1.0, markeredgecolor=color, markerfacecolor="None")

		## Continuous MFD
		else:
			try:
				linestyle = styles[i]
			except:
				linestyle = "-"
			else:
				if linestyle in ("", None) or not linestyle in ("-", "--", ":", ":."):
					linestyle = "-"

			## Cumulative
			if want_cumulative:
				label = labels[i]
				if want_incremental:
					label += " (cumul.)"
				ax.semilogy(mfd.get_magnitude_bin_edges(), mfd.get_cumulative_rates(), color=color, linestyle=linestyle, lw=3, label=label)

			## Incremental
			if want_incremental:
				label = labels[i] + " (inc.)"
				ax.semilogy(mfd.get_magnitude_bin_centers(), mfd.occurrence_rates, color=color, linestyle=linestyle, lw=1, label=label)

	if not Mrange:
		Mrange = pylab.axis()[:2]
	if not Freq_range:
		Freq_range = pylab.axis()[2:]

	## Plot limits of completeness
	if completeness:
		annoty = Freq_range[0] * 10**0.5
		bbox_props = dict(boxstyle="round,pad=0.4", fc="w", ec="k", lw=1)
		## Make sure min_mags is not sorted in place,
		## otherwise completeness object may misbehave
		min_mags = np.sort(completeness.min_mags)
		if not end_year:
			end_year = datetime.date.today().year
		for i in range(1, len(min_mags)):
			ax.plot([min_mags[i], min_mags[i]], Freq_range, 'k--', lw=1, label="_nolegend_")
			ax.annotate("", xy=(min_mags[i-1], annoty), xycoords='data', xytext=(min_mags[i], annoty), textcoords='data', arrowprops=dict(arrowstyle="<->"),)
			label = "%s - %s" % (completeness.get_initial_completeness_year(min_mags[i-1]), end_year)
			ax.text(np.mean([min_mags[i-1], min_mags[i]]), annoty*10**-0.25, label, ha="center", va="center", size=12, bbox=bbox_props)
		ax.annotate("", xy=(min_mags[i], annoty), xycoords='data', xytext=(min(mfd.max_mag, Mrange[1]), annoty), textcoords='data', arrowprops=dict(arrowstyle="<->"),)
		label = "%s - %s" % (completeness.get_initial_completeness_year(min_mags[i]), end_year)
		ax.text(np.mean([min_mags[i], mfd.max_mag]), annoty*10**-0.25, label, ha="center", va="center", size=12, bbox=bbox_props)

	## Apply plot limits
	ax.axis((Mrange[0], Mrange[1], Freq_range[0], Freq_range[1]))

	ax.set_xlabel("Magnitude ($M_%s$)" % mfd.Mtype[1].upper(), fontsize="x-large")
	label = {"en": "Annual number of earthquakes", "nl": "Aantal aardbevingen per jaar", "fr": u"Nombre de séismes par année"}[lang.lower()]
	ax.set_ylabel(label, fontsize="x-large")
	ax.set_title(title, fontsize='x-large')
	ax.grid(True)
	font = FontProperties(size='medium')
	ax.legend(loc=legend_location, prop=font)
	for label in ax.get_xticklabels() + ax.get_yticklabels():
		label.set_size('large')

	if fig_filespec:
		default_figsize = pylab.rcParams['figure.figsize']
		default_dpi = pylab.rcParams['figure.dpi']
		if fig_width:
			fig_width /= 2.54
			dpi = dpi * (fig_width / default_figsize[0])

		fig.savefig(fig_filespec, dpi=dpi)
	elif interactive:
		pylab.show()
	else:
		return ax, fig


def alphabetalambda(a, b, M=0):
	"""
	Calculate alpha, beta, lambda from a, b, and M.
	(assuming truncated Gutenberg-Richter MFD)

	:param a:
		Float, a value of Gutenberg-Richter relation
	:param b:
		Float, b value of Gutenberg-Richter relation
	:param M:
		Float, magnitude for which to compute lambda (default: 0)

	:return:
		(alpha, beta, lambda) tuple
	"""
	alpha = a * np.log(10)
	beta = b * np.log(10)
	lambda0 = np.exp(alpha - beta*M)
	# This is identical
	# lambda0 = 10**(a - b*M)
	return (alpha, beta, lambda0)


def a_from_lambda(lamda, M, b):
	"""
	Compute a value from lambda value for a particular magnitude
	(assuming truncated Gutenberg-Richter MFD)

	:param lamda:
		Float, Frequency (1/yr) for magnitude M
	:param M:
		Float, magnitude
	:param b:
		Float, b value

	:return:
		Float, a value
	"""
	return np.log10(lamda) + b * M


def get_a_separation(b, dM):
	"""
	Compute separation between a values of cumulative and discrete
	(assuming truncated Gutenberg-Richter MFD)

	:param b:
		Float, b value
	:param dM:
		Float, magnitude bin width
	"""
	return b * dM - np.log10(10**(b*dM) - 1)



# TODO: the following functions are probably obsolete

def split_mfd_fault_bg(aValue, bValue, Mmin, Mmin_fault, Mmax, bin_width=0.1):
	"""
	Split MFD over a background zone (up to Mmin_fault) and a fault source

	:param aValue:
		a value of the MFD
	:param bValue:
		b value of the MFD
	:param Mmin:
		minimum magnitude of the MFD
	:param Mmin_fault:
		minimum magnitude on the fault
	:param Mmax:
		maximum magnitude of the MFD
	:param bin_width:
		bin width in magnitude units of the MFD

	Return value:
		tuple (mfd_bg, mfd_fault) with nhlib MFD objects for background zone and
		fault source. mfd_bg is an instance of nhlib.mfd.EvenlyDiscretizedMFD
		and mfd_fault is an instance of nhlib.mfd.TruncatedGRMFD
	"""
	## Construct summed MFD
	mfd_summed = TruncatedGRMFD(Mmin, Mmax, bin_width, aValue, bValue)
	## Note: get_annual_occurrence_rates() returns non-cumulative rates !
	hist = mfd_summed.get_annual_occurrence_rates()
	print hist
	Mbins, rates_summed = zip(*hist)
	Mbins, rates_summed = np.array(Mbins), np.array(rates_summed)

	## Determine first bin of fault MFD
	print Mbins
	print Mmin_fault
	index = np.where(Mbins > Mmin_fault)[0][0]

	## Construct MFD's
	mfd_fault = TruncatedGRMFD(Mmin_fault, Mmax, bin_width, aValue, bValue)

	rates_bg = rates_summed.copy()
	#rates_bg = rates_bg[:index] - rates_bg[index]
	rates_bg = rates_bg[:index]
	mfd_bg = EvenlyDiscretizedMFD(Mbins[0], bin_width, rates_bg)

	## Note that this is equivalent to
	# mfd_bg = TruncatedGRMFD(Mmin, Mmin_fault, bin_width, aValue, bValue)

	## Check that 2 MFD's sum up to overall MFD
	hist = mfd_fault.get_annual_occurrence_rates()
	Mbins_fault, rates_fault = zip(*hist)
	#rates_summed2 = np.zeros(len(Mbins), 'd')
	#rates_summed2[index:] = rates_fault
	#rates_summed2[:index] = rates_fault[0]
	#rates_summed2[:index] += rates_bg
	rates_summed2 = np.concatenate((rates_bg, rates_fault))

	if np.allclose(rates_summed, rates_summed2):
		return (mfd_bg, mfd_fault)
	else:
		raise Exception("Summed rates do not match!")


def divide_mfd_fault_bg(aValue, bValue, Mmin, Mmin_fault, Mmax_fault, Mmax, bin_width=0.1):
	"""
	Divide MFD over a background zone and a fault source (between Mmin_fault and
	Mmax_fault)

	:param aValue:
		a value of the MFD
	:param bValue:
		b value of the MFD
	:param Mmin:
		minimum magnitude of the MFD
	:param Mmin_fault:
		minimum magnitude on the fault
	:param Mmax_fault:
		maximum magnitude on the fault
	:param Mmax:
		maximum magnitude of the MFD
	:param bin_width:
		bin width in magnitude units of the MFD

	Return value:
		tuple (mfd_bg, mfd_fault) with nhlib MFD objects for background zone and
		fault source. mfd_bg and mfd_fault are instances of
		nhlib.mfd.EvenlyDiscretizedMFD
	"""
	## Construct summed MFD
	mfd_summed = TruncatedGRMFD(Mmin, Mmax, bin_width, aValue, bValue)
	hist = mfd_summed.get_annual_occurrence_rates()
	Mbins, rates_summed = zip(*hist)
	Mbins, rates_summed = np.array(Mbins), np.array(rates_summed)

	## Determine first and last bin of fault MFD
	start_index = np.where(Mbins > Mmin_fault)[0][0]
	end_index = np.where(Mbins > Mmax_fault)[0][0]
	#print Mbins[start_index: end_index]

	## Construct MFD's
	rates_bg = np.copy(rates_summed)
	#rates_fault = rates_summed[start_index: end_index] - rates_summed[end_index]
	rates_fault = rates_summed[start_index: end_index]
	#rates_bg[start_index : end_index] = rates_bg[end_index]
	rates_bg[start_index : end_index] *= 0.
	#rates_bg[:start_index] -= rates_fault[0]

	mfd_bg = EvenlyDiscretizedMFD(Mbins[0], bin_width, rates_bg)
	mfd_fault = EvenlyDiscretizedMFD(Mbins[start_index], bin_width, rates_fault)

	## Check that 2 MFD's sum up to overall MFD
	# TODO: try with bin values instead of cumulative values
	#rates_summed2 = np.zeros(len(Mbins), 'd')
	#rates_summed2[end_index:] = rates_bg[end_index:]
	#rates_summed2[start_index: end_index] = rates_bg[end_index]
	#rates_summed2[start_index: end_index] += rates_fault
	#rates_summed2[:start_index] += rates_fault[0]
	#rates_summed2[:start_index] += rates_bg[:start_index]
	rates_summed2 = rates_bg.copy()
	rates_summed2[start_index: end_index] += rates_fault

	if np.allclose(rates_summed, rates_summed2):
		return (mfd_bg, mfd_fault)
	else:
		raise Exception("Summed rates do not match!")


def divide_mfd_faults(aValue, bValue, Mmin, Mmax_catalog, Mmax_faults, Mmax_rates, weights, bin_width=0.1):
	"""
	Divide catalog MFD over a number of fault sources, and add rate of Mmax as a
	characteristic earthquake
	:param aValue:
		a value of the MFD
	:param bValue:
		b value of the MFD
	:param Mmin:
		minimum magnitude of the MFD
	:param Mmax_catalog:
		maximum magnitude of the catalog MFD
	:param Mmax_faults:
		list of maximum magnitudes for each fault
	:param Mmax_rates:
		list of occurrence rates (annual frequencies) for the Mmax on each fault
	:param weights:
		list of weights of each fault in the catalog MFD (based on moment rates)
	:param bin_width:
		bin width in magnitude units of the MFD

	Return value:
		list of nhlib MFD objects (instances of instances of
		nhlib.mfd.EvenlyDiscretizedMFD) for each fault
	"""
	mfd_list = []
	for weight, Mmax, Mmax_rate in zip(weights, Mmax_faults, Mmax_rates):
		a = aValue + np.log10(weight)
		num_bins1 = int(round((Mmax_catalog - Mmin) / bin_width))
		Mbins1 = np.linspace(Mmin, Mmax_catalog, num_bins1)
		rates1 = 10**(a - bValue * Mbins1)
		num_bins2 = int(round((Mmax - Mmax_catalog) / bin_width))
		Mbins2 = np.linspace(Mmax_catalog + bin_width, Mmax, num_bins2)
		rates2 = np.zeros(num_bins2, 'd')
		#rates2 += Mmax_rate
		rates2[-1] = Mmax_rate
		rates = np.concatenate((rates1, rates2))
		Mbins = np.concatenate((Mbins1, Mbins2)) + (bin_width / 2)
		#print Mbins
		#print rates
		mfd = EvenlyDiscretizedMFD(Mmin + (bin_width / 2), bin_width, rates)
		mfd_list.append(mfd)

	return mfd_list



