# -*- coding: iso-Latin-1 -*-

"""
Evenly Discretized MFD
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import int

import numpy as np

from .. import oqhazlib

from .base import (MFD, sum_mfds)



__all__ = ['EvenlyDiscretizedMFD']


class EvenlyDiscretizedMFD(oqhazlib.mfd.EvenlyDiscretizedMFD, MFD):
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
		str, magnitude type, either "MW" or "MS"
		(default: "MW")
	"""
	def __init__(self, min_mag, bin_width, occurrence_rates, Mtype="MW"):
		## Note: Convert occurrence_rates to list to avoid problems in oqhazlib!
		oqhazlib.mfd.EvenlyDiscretizedMFD.__init__(self, min_mag, bin_width,
													list(occurrence_rates))
		self.Mtype = Mtype

	def __repr__(self):
		txt = '<EvenlyDiscretizedMFD | %s=%.2f:%.2f:%.2f>'
		txt %= (self.Mtype, self.min_mag, self.max_mag, self.bin_width)
		return txt

	def __div__(self, other):
		if isinstance(other, (int, float)):
			occurrence_rates = np.array(self.occurrence_rates) / other
			return EvenlyDiscretizedMFD(self.min_mag, self.bin_width,
										occurrence_rates, self.Mtype)
		else:
			raise TypeError("Divisor must be integer or float")

	def __mul__(self, other):
		if isinstance(other, (int, float)):
			occurrence_rates = np.array(self.occurrence_rates) * other
			return EvenlyDiscretizedMFD(self.min_mag, self.bin_width,
										occurrence_rates, self.Mtype)
		else:
			raise TypeError("Multiplier must be integer or float")

	def __add__(self, other):
		if isinstance(other, MFD):
			return sum_mfds([self, other])
		else:
			raise TypeError("Operand must be MFD")

	def __sub__(self, other):
		# TODO: this doesn't work if max_mags are different, need to either pad or truncate
		if isinstance(other, MFD):
			if not self.is_compatible(other):
				raise Exception("MFD's not compatible")
			if (self.get_min_mag() <= other.get_min_mag()
				and self.max_mag >= other.max_mag):
				occurrence_rates = np.array(self.occurrence_rates)
				start_index = self.get_magnitude_index(other.get_min_mag())
				occurrence_rates[start_index:start_index+len(other)] -= np.array(
															other.occurrence_rates)
				## Replace negative values with zeros
				occurrence_rates[np.where(occurrence_rates < 0)] = 0
				return EvenlyDiscretizedMFD(self.min_mag, self.bin_width,
											occurrence_rates)
			else:
				raise Exception("Second MFD must fully overlap with first one!")
		else:
			raise TypeError("Operand must be MFD")

	@property
	def max_mag(self):
		return self.get_min_mag_edge() + len(self.occurrence_rates) * self.bin_width

	def copy(self):
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
		Divide MFD into a number of MFD's that together sum up to the
		original MFD

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
			mfd = EvenlyDiscretizedMFD(self.min_mag, self.bin_width,
										occurrence_rates, self.Mtype)
			mfd_list.append(mfd)
		return mfd_list

	def split(self, M):
		"""
		Split MFD at a particular magnitude

		:param M:
			float, magnitude value where MFD should be split

		:return:
			List containing 2 instances of :class:`EvenlyDiscretizedMFD`
		"""
		if not self.is_magnitude_compatible(M):
			raise Exception("Magnitude value not compatible!")
		elif self.get_min_mag_edge() < M < self.max_mag:
			index = int(round((M - self.get_min_mag_edge()) / self.bin_width))
			occurrence_rates1 = self.occurrence_rates[:index]
			occurrence_rates2 = self.occurrence_rates[index:]
			mfd1 = EvenlyDiscretizedMFD(self.min_mag, self.bin_width,
										occurrence_rates1, self.Mtype)
			mfd2 = EvenlyDiscretizedMFD(M+self.bin_width/2, self.bin_width,
										occurrence_rates2, self.Mtype)
			return [mfd1, mfd2]
		else:
			raise Exception("Split magnitude not in valid range!")

	def extend(self, other_mfd):
		"""
		Extend MFD with another one that covers larger magnitudes.

		:param other_mfd:
			instance of :class:`EvenlyDiscretizedMFD` or :class:`TruncatedGRMFD`
			The minimum magnitude of other_mfd should be equal to or
			larger than the maximum magnitude of this MFD.

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

		num_empty_bins = gap
		if num_empty_bins >= 0:
			occurrence_rates = np.concatenate([self.occurrence_rates,
											np.zeros(num_empty_bins, dtype='d'),
											occurrence_rates])
			self.occurrence_rates = list(occurrence_rates)
		else:
			raise Exception("Magnitudes must not overlap with MFD magnitude "
							"range. Sum MFDs instead")

	def modify_set_bin_width(self, bin_width):
		"""
		Modify bin width in place

		:param bin_width:
			float, new magnitude bin width
		"""
		from ..utils import interpolate

		edge_mags = self.get_magnitude_bin_edges()
		rebinned_edge_mags = np.arange(edge_mags[0], edge_mags[-1] + self.bin_width/2,
										bin_width)
		rebinned_cum_rates = interpolate(edge_mags, self.get_cumulative_rates(),
										rebinned_edge_mags)
		rebinned_occurrence_rates = np.diff(rebinned_cum_rates[::-1])[::-1]
		rebinned_occurrence_rates = np.concatenate([rebinned_occurrence_rates,
													rebinned_cum_rates[-1:]])
		self.min_mag = rebinned_edge_mags[0] + bin_width/2
		self.bin_width = bin_width
		self.modify_set_occurrence_rates(rebinned_occurrence_rates)

	def append_characteristic_eq(self, Mc, return_period, M_sigma=0.3, num_sigma=0):
		"""
		Append magnitude-frequency of a characteristic earthquake

		:param Mc:
			float, magnitude of characteristic earthquake
		:param return_period:
			float, return period in yr of characteristic earthquake
		:param M_sigma:
		:param num_sigma:
			see :class:`CharacteristicMFD`
		"""
		characteristic_mfd = CharacteristicMFD(Mc, return_period, self.bin_width,
						Mtype=self.Mtype, M_sigma=M_sigma, num_sigma=num_sigma,
						force_bin_alignment=True)
		dM = characteristic_mfd.char_mag - Mc
		if not np.allclose(dM, 0):
			print("Warning: Mc has been changed by %f to align with bin width!" % dM)
		self.extend(characteristic_mfd)

	def to_truncated_GR_mfd(self, completeness, end_date, method="Weichert",
							b_val=None, Mmax=None, verbose=False):
		"""
		Calculate truncated Gutenberg-Richter MFD using maximum likelihood
		estimation for variable observation periods for different
		magnitude increments. Adapted from calB.m and calBfixe.m Matlab
		modules written by Philippe Rosset (ROB, 2004), which is based
		on the method by Weichert, 1980 (BSSA, 70, Nr 4, 1337-1346).

		:param completeness:
			instance of :class:`Completeness`, necessary for Weichert method
		:param end_date:
			datetime.date or int, end date with respect to which observation
			periods will be determined in Weichert method
		:param method:
			str, computation method:
			- "Weichert"
			- "LSQc": least squares on cumulative values
			- "LSQi": least squares on incremental values
			- "wLSQc": weighted least squares on cumulative values
			- "wLSQi": weighted least squares on incremental values
			(default: "Weichert")
		:param b_val:
			float, fixed b value to constrain MLE estimation
			(default: None)
		:param Mmax:
			float, maximum magnitude of fitted GR MFD
			(default: None, will use currently set max_mag)
		:param verbose:
			bool, whether some messages should be printed or not
			(default: False)

		:return:
			instance of :class:`TruncatedGRMFD`
		"""
		from eqcatalog.calcGR import calcGR_Weichert, calcGR_LSQ
		from .truncated_gr import TruncatedGRMFD

		magnitudes = self.get_magnitude_bin_edges()
		if Mmax is None:
			max_mag = self.max_mag
		else:
			max_mag = Mmax
		if method == "Weichert":
			## Number of earthquakes in each bin according to completeness
			bins_N = self.get_num_earthquakes(completeness, end_date)
			if Mmax != None:
				num_empty_bins = int(round((Mmax - self.max_mag) / self.bin_width)) + 1
				bins_N = np.concatenate([bins_N, np.zeros(num_empty_bins)])
				extra_mags = magnitudes[-1] + (np.arange(1, num_empty_bins+1)
											* self.bin_width)
				magnitudes = np.concatenate([magnitudes, extra_mags])
			a, b, stda, stdb = calcGR_Weichert(magnitudes, bins_N, completeness,
										end_date, b_val=b_val, verbose=verbose)
		elif "LSQ" in method:
			weights = None
			if method[0] == 'w' and completeness and end_date:
				## Use weighted LSQC
				weights = self.get_num_earthquakes(completeness, end_date)
			if method[-1] == 'c':
				occurrence_rates = self.get_cumulative_rates()
			elif method[-1] == 'i':
				occurrence_rates = np.array(self.occurrence_rates)
			a, b, stda, stdb = calcGR_LSQ(magnitudes, occurrence_rates, b_val=b_val,
										weights=weights, verbose=verbose)
			if method[-1] == 'i':
				## Compute a value for cumulative MFD from discrete a value
				a2 = a
				dM = self.bin_width
				a = a2 + get_a_separation(b, dM)
		return TruncatedGRMFD(self.get_min_mag_edge(), max_mag, self.bin_width,
							a, b, a_sigma=stda, b_sigma=stdb, Mtype=self.Mtype)

	def get_max_mag_observed(self):
		"""
		Return maximum observed magnitude (max. magnitude with non-zero
		occurrence_rate)
		"""
		mag_bin_centers = self.get_magnitude_bin_centers()
		Mmax = mag_bin_centers[np.array(self.occurrence_rates) > 0][-1]
		return Mmax

	def get_bayesian_mmax_pdf(self, prior_model="CEUS_COMP", Mmax_obs=None,
							n=None, Mmin_n=4.5, b_val=None, bin_width=None,
							truncation=(5.5, 8.25), completeness=None,
							end_date=None, verbose=True):
		"""
		Compute Mmax distribution following Bayesian approach.

		:param prior_model:
			str, indicating which prior model should be considered, one of:
			- "EPRI_extended": extended crust in EPRI (1994)
			- "EPRI_non_extended": non-extended crust in EPRI (1994)
			- "CEUS_COMP": composite prior in CEUS (2012)
			- "CEUS_MESE": Mesozoic and younger extension in CEUS (2012)
			- "CEUS_NMESE": Non-Mesozoic and younger extension in CEUS (2012)
			(default: "CEUS_COMP")
		:param Mmax_obs:
			Float: maximum observed magnitude
			(default: None, will use highest magnitude bin center
			with non-zero occurrence rate))
		:param n:
			int, number of earthquakes above minimu magnitude relevant for PSHA
			(default: None, will compute this parameter from MFD and Mmin)
		:param Mmin_n:
			float, lower magnitude, used to count n, the number of earthquakes
			between Mmin and Mmax_obs (corresponds to lower magnitude in PSHA).
			If None, will use min_mag of MFD.
			(default: 4.5)
		:param b_val:
			float, b value of MFD
			(default: None, will compute b value from MFD using Weichert
			method)
		:param bin_width:
			float, magnitude bin width. If None, will take bin_width from MFD
			(default: None)
		:param truncation:
			Int or tuple, representing truncation of prior distribution.
			If int, truncation is interpreted as the number of standard deviations.
			If tuple, elements are interpreted as minimum and maximum
			magnitude of the distribution
			(default: (5.5, 8.25), corresponding to the truncation applied in CEUS)
		:param completeness:
			instance of :class:`Completeness` containing initial years of
			completeness and corresponding minimum magnitudes
			This parameter is required if n and/or b_val have to be computed
			(default: None)
		:param end_date:
			datetime.date or int, end date of observation period corresponding
			to MFD. This parameter is required if n and/or b_val have to be
			computed
		:param verbose:
			bool, whether or not to print additional information
			(default: True)

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
			gr_mfd = imfd.to_truncated_GR_mfd(completeness, end_date,
											method="Weichert", verbose=verbose)
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
			likelihood[Mmax_obs_index:] = \
				(1 - np.exp(-beta * (magnitudes[Mmax_obs_index:] - Mmin_n))) ** -n
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
			str, unicode encoding
			(default: 'latin1')
		"""
		from lxml import etree
		from ..nrml import ns

		edi_elem = etree.Element(ns.EVENLY_DISCRETIZED_INCREMENTAL_MFD)
		edi_elem.set(ns.MINIMUM_MAGNITUDE, str(self.min_mag))
		edi_elem.set(ns.BIN_WIDTH, str(self.bin_width))
		#edi_elem.set(ns.OCCURRENCE_RATES, " ".join(map(str, self.occurrence_rates)))
		occur_rates_elem = etree.SubElement(edi_elem, ns.OCCURRENCE_RATES)
		occur_rates_elem.text = " ".join(map(str, self.occurrence_rates))
		return edi_elem

	def plot(self, label="", color='k', style=None, lw_or_ms=None,
			discrete=True, cumul_or_inc="both",
			completeness=None, end_year=None,
			title="", lang="en",
			fig_filespec=None, **kwargs):
		"""
		Plot magnitude-frequency distribution

		:param color:
			matplotlib color specification
			(default: 'k')
		:param label:
			str, legend label
			(default: "")
		:param style:
			matplotlib marker style (for discrete plots) or line style
			(for non-discrete plots)
			(default: None = 'o'/'-' for :param:`discrete` True/False)
		:param lw_or_ms:
			float, marker size (discrete plots) or line width (non-discrete)
			(default: None = 8 / 2.5 for :param:`discrete` True/False)
		:param discrete:
			bool, whether or not to plot discrete MFD
			(default: True)
		:param cumul_or_inc:
			str, either "cumul", "inc" or "both", indicating
			whether to plot cumulative MFD, incremental MFD or both
			(default: "both")
		:param completeness:
			instance of :class:`Completeness`, used to plot completeness
			limits
			(default: None)
		:param end_year:
			int, end year of catalog (used when plotting completeness limits)
			(default: None, will use current year)
		:param title:
			str, plot title
			(default: "")
		:param lang:
			str, language of plot axis labels
			(default: "en")
		:param fig_filespec:
			str, full path to output image file, if None plot to screen
			(default: None)
		:kwargs:
			additional keyword arguments understood by
			:func:`generic_mpl.plot_xy`

		:return:
			matplotlib Axes instance
		"""
		from .plot import plot_mfds

		return plot_mfds([self], colors=[color], labels=[label],
						styles=[style], lw_or_ms=[lw_or_ms],
						discrete=[discrete], cumul_or_inc=[cumul_or_inc],
						completeness=completeness, end_year=end_year,
						title=title, lang=lang,
						fig_filespec=fig_filespec, **kwargs)
