# -*- coding: iso-Latin-1 -*-

"""
Base MFD class
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np



__all__ = ['MFD', 'sum_mfds']


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
		min_mag = self.get_min_mag_center()
		return min_mag + np.arange(len(self)) * self.bin_width
		#return np.array(list(zip(*self.get_annual_occurrence_rates()))[0])

	def get_magnitude_bin_edges(self):
		"""
		Return left edge value of magnitude bins

		:return:
			numpy float array
		"""
		return self.get_magnitude_bin_centers() - self.bin_width / 2
	
	@property
	def center_magnitudes(self):
		return self.get_magnitude_bin_centers()
	
	@property
	def edge_magnitudes(self):
		return self.get_magnitude_bin_edges()

	def get_magnitude_index(self, M):
		"""
		Determine index of given magnitude (edge) value

		:param M:
			float, magnitude value (left edge of bin)

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
		return np.cumsum(self.occurrence_rates[::-1])[::-1]

	def is_magnitude_compatible(self, M):
		"""
		Determine whether a particular magnitude (edge) value is compatible
		with bin width of MFD

		:param M:
			float, magnitude value (left edge of bin)

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
			instance of :class:`EvenlyDiscretizedMFD` or :class:`TruncatedGRMFD`

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
			float, maximum magnitude (default: None)

		:return:
			instance of :class:`EvenlyDiscretizedMFD`
		"""
		from .evenly_discretized import EvenlyDiscretizedMFD

		if max_mag is None:
			occurrence_rates = self.occurrence_rates
		else:
			if max_mag <= self.max_mag:
				idxs = self.get_magnitude_bin_centers() < max_mag
				occurrence_rates = np.array(self.occurrence_rates)[idxs]
			else:
				dmag = max_mag - self.max_mag
				num_zeros = int(np.round(dmag / self.bin_width))
				occurrence_rates = np.append(self.occurrence_rates, np.zeros(num_zeros))
		return EvenlyDiscretizedMFD(self.get_min_mag_center(), self.bin_width,
									occurrence_rates, Mtype=self.Mtype)

	def get_num_earthquakes(self, completeness, end_date):
		"""
		Return array with number of earthquakes per magnitude bin,
		taking into account completeness

		:param completeness:
			instance of :class:`eqcatalog.Completeness`
		:param end_date:
			datetime.date or int, end date with respect to which
			observation periods will be determined

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

	def sample_inter_event_times(self, timespan, skip_time=0, method="poisson",
								random_seed=None):
		"""
		Generate random inter-event times for each magnitude bin according
		to a Poisson or random process.

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
			int, seed for the random number generator
			(default: None)

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

	def generate_random_catalog(self, timespan, skip_time=0, method="poisson",
								start_year=1, lons=None, lats=None, random_seed=None):
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
			int, start year of generated catalog.
			Note that the year 0 does not exist!
			(default: 1)
		:param lons:
			list or array, longitudes to sample from
			(default: None)
		:param lats:
			list or array, latitudes to sample from
			(default: None)
		:param random_seed:
			int, seed for the random number generator
			(default: None)

		:return:
			instance of :class:`EQCatalog`
		"""
		import random
		from eqcatalog import EQCatalog, LocalEarthquake
		from eqcatalog.time import date_from_fractional_year

		rnd = random.Random()
		rnd.seed(random_seed)

		inter_event_times = self.sample_inter_event_times(timespan, skip_time,
															method, random_seed)
		eq_list = []
		ID = 0
		depth = 0.
		name = ""
		try:
			num_lon_lats = len(lons)
		except:
			num_lon_lats = 0
		start_date = np.datetime64('%d-01-01' % start_year, dtype='M8[D]')
		for M, iets in zip(self.get_magnitude_bin_centers(), inter_event_times):

			fractional_years = np.add.accumulate(iets)
			eq_dates = date_from_fractional_year(fractional_years + start_year)
			for date in eq_dates:
				if num_lon_lats == 0:
					lon, lat = 0., 0.
				else:
					idx = rnd.randint(0, num_lon_lats-1)
					lon, lat = lons[idx], lats[idx]
				eq = LocalEarthquake(ID, date, None, lon, lat, depth,
									{self.Mtype: M}, name=name)
				eq_list.append(eq)
				ID += 1
		end_year = start_year + timespan -1
		end_date = np.datetime64('%d-12-31' % end_year, dtype='M8[D]')
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
			float, total moment rate in N.m/yr
		"""
		return np.sum(self.get_incremental_moment_rates())

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



def sum_mfds(mfd_list, weights=[]):
	"""
	Sum two or more MFD's

	:param mfd_list:
		List containing instances of :class:`TruncatedGRMFD` or
		(descendants of) :class:`EvenlyDiscretizedMFD`
	:param weights:
		List or array containing weights of each MFD (default: [])

	:return:
		instance of :class:`TruncatedGRMFD` (if all MFD's in list are
		TruncatedGR, and have same min_mag, max_mag, and b_val) or else
		instance of :class:`EvenlyDiscretizedMFD`

	Note:
		Weights will be normalized!
	"""
	from .truncated_gr import TruncatedGRMFD
	from .evenly_discretized import EvenlyDiscretizedMFD

	if weights in ([], None):
		weights = np.ones(len(mfd_list), 'd')
	total_weight = np.sum(weights)
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
	## return TruncatedGR, else return EvenlyDiscretized
	is_truncated = np.array([isinstance(mfd, TruncatedGRMFD) for mfd in mfd_list])
	if is_truncated.all():
		all_bvals = set([mfd.b_val for mfd in mfd_list])
		if len(all_min_mags) == len(all_max_mags) == len(all_bvals) == 1:
			## TruncatedGR's can be summed into another TruncatedGR object
			all_avals = np.array([mfd.a_val for mfd in mfd_list])
			N0 = 10**all_avals
			N0sum = np.sum(N0 * weights)
			a = np.log10(N0sum)
			## Error propagation, see http://chemwiki.ucdavis.edu/Analytical_Chemistry/Quantifying_Nature/Significant_Digits/Propagation_of_Error
			Nsum_sigma = np.sum([mfd.get_N0_sigma() * w
								for (mfd, w) in zip(mfd_list, weights)])
			a_sigma = 0.434 * (Nsum_sigma / N0sum)
			b_sigma = np.mean([mfd.b_sigma for mfd in mfd_list])
			mfd = mfd_list[0]
			return TruncatedGRMFD(mfd.min_mag, mfd.max_mag, mfd.bin_width, a,
									mfd.b_val, a_sigma, b_sigma, mfd.Mtype)
		else:
			## TruncatedGR's can be summed after conversion to EvenlyDiscretized
			pass

	# TODO: take into account (characteristic) MFDs that may be shifted half a bin width!
	min_mag = min(all_min_mags)
	max_mag = max(all_max_mags)
	num_bins = int(round((max_mag - min_mag) / bin_width))
	occurrence_rates = np.zeros(num_bins, 'd')
	for i, mfd in enumerate(mfd_list):
		start_index = int(round((mfd.get_min_mag_edge() - min_mag) / bin_width))
		end_index = start_index + len(mfd.occurrence_rates)
		occurrence_rates[start_index:end_index] += (np.array(mfd.occurrence_rates)
													* weights[i])
	return EvenlyDiscretizedMFD(min_mag+bin_width/2, bin_width, occurrence_rates,
								Mtype)


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
		tuple (mfd_bg, mfd_fault) with oqhazlib MFD objects for background zone and
		fault source. mfd_bg is an instance of oqhazlib.mfd.EvenlyDiscretizedMFD
		and mfd_fault is an instance of oqhazlib.mfd.TruncatedGRMFD
	"""
	## Construct summed MFD
	mfd_summed = TruncatedGRMFD(Mmin, Mmax, bin_width, aValue, bValue)
	## Note: get_annual_occurrence_rates() returns non-cumulative rates !
	hist = mfd_summed.get_annual_occurrence_rates()
	print(hist)
	Mbins, rates_summed = zip(*hist)
	Mbins, rates_summed = np.array(Mbins), np.array(rates_summed)

	## Determine first bin of fault MFD
	print(Mbins)
	print(Mmin_fault)
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
		tuple (mfd_bg, mfd_fault) with oqhazlib MFD objects for background zone and
		fault source. mfd_bg and mfd_fault are instances of
		oqhazlib.mfd.EvenlyDiscretizedMFD
	"""
	## Construct summed MFD
	mfd_summed = TruncatedGRMFD(Mmin, Mmax, bin_width, aValue, bValue)
	hist = mfd_summed.get_annual_occurrence_rates()
	Mbins, rates_summed = zip(*hist)
	Mbins, rates_summed = np.array(Mbins), np.array(rates_summed)

	## Determine first and last bin of fault MFD
	start_index = np.where(Mbins > Mmin_fault)[0][0]
	end_index = np.where(Mbins > Mmax_fault)[0][0]
	#print(Mbins[start_index: end_index])

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
		list of oqhazlib MFD objects (instances of instances of
		oqhazlib.mfd.EvenlyDiscretizedMFD) for each fault
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
		#print(Mbins)
		#print(rates)
		mfd = EvenlyDiscretizedMFD(Mmin + (bin_width / 2), bin_width, rates)
		mfd_list.append(mfd)

	return mfd_list



