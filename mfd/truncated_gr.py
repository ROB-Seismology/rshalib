# -*- coding: iso-Latin-1 -*-

"""
Truncated Gutenberg-Richter MFD
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import int


import numpy as np

import openquake.hazardlib as oqhazlib

from .base import MFD



__all__ = ['TruncatedGRMFD']


class TruncatedGRMFD(oqhazlib.mfd.TruncatedGRMFD, MFD):
	"""
	Truncated or modified Gutenberg-Richter MFD

	:param min_mag:
		The lowest possible magnitude for this MFD. The first bin in the
		:meth:`result histogram <get_annual_occurrence_rates>` will be
		aligned to make its left border match this value.
	:param max_mag:
		The highest possible magnitude. The same as for ``min_mag``: the
		last bin in the histogram will correspond to the magnitude value
		equal to ``max_mag - bin_width / 2``.
	:param bin_width:
		A positive float value -- the width of a single histogram bin.
	:param a_val:
		float, the cumulative ``a`` value (``10 ** a`` is the number
		of earthquakes per year with magnitude greater than or equal to 0)
	:param b_val:
		float, Gutenberg-Richter ``b`` value -- the decay rate
		of exponential distribution. It describes the relative size
		distribution of earthquakes: a higher ``b`` value indicates a
		relatively larger proportion of small events and vice versa.
	:param a_sigma:
		float, standard deviation of the a value
		(default: 0).
	:param b_sigma:
		float, standard deviation of the b value
		(default: 0).
	:param Mtype:
		str, magnitude type, either "MW" or "MS"
		(default: "MW")

	Note:
		Values for ``min_mag`` and ``max_mag`` don't have to be aligned
		with respect to ``bin_width``. They get rounded accordingly
		anyway so that both are divisible by ``bin_width`` just before
		converting a function to a histogram.
		See :meth:`_get_min_mag_and_num_bins`.
	"""
	def __init__(self, min_mag, max_mag, bin_width, a_val, b_val,
				a_sigma=0, b_sigma=0, Mtype="MW"):
		oqhazlib.mfd.TruncatedGRMFD.__init__(self, min_mag, max_mag, bin_width,
											a_val, b_val)
		self.a_sigma = a_sigma
		self.b_sigma = b_sigma
		self.Mtype = Mtype

	def __div__(self, other):
		if isinstance(other, (int, float)):
			N0 = 10**self.a_val
			a_val = np.log10(N0 / float(other))
			## a_sigma does not change
			a_sigma = self.a_sigma
			return TruncatedGRMFD(self.min_mag, self.max_mag, self.bin_width,
							a_val, self.b_val, a_sigma, self.b_sigma, self.Mtype)
		else:
			raise TypeError("Divisor must be integer or float")

	def __mul__(self, other):
		if isinstance(other, (int, float)):
			N0 = 10**self.a_val
			a_val = np.log10(N0 * float(other))
			## a_sigma does not change
			a_sigma = self.a_sigma
			return TruncatedGRMFD(self.min_mag, self.max_mag, self.bin_width,
							a_val, self.b_val, a_sigma, self.b_sigma, self.Mtype)
		else:
			raise TypeError("Multiplier must be integer or float")

	def __add__(self, other):
		if isinstance(other, MFD):
			return sum_MFDs([self, other])
		else:
			raise TypeError("Operand must be MFD")

	def __sub__(self, other):
		if isinstance(other, TruncatedGRMFD):
			if (self.min_mag == other.min_mag
				and self.max_mag == other.max_mag
				and self.b_val == other.b_val
				and self.Mtype == other.Mtype):
				## Note: bin width does not have to be the same here
				N0 = 10 ** self.a_val - 10 ** other.a_val
				a_val = np.log10(N0)
				## Error propagation, see http://chemwiki.ucdavis.edu/Analytical_Chemistry/Quantifying_Nature/Significant_Digits/Propagation_of_Error
				a_sigma = 0.434 * ((self.get_N0_sigma() + other.get_N0_sigma()) / N0)
				b_sigma = np.mean(self.b_sigma, other.b_sigma)
				return TruncatedGRMFD(self.min_mag, self.max_mag, self.bin_width,
								a_val, self.b_val, a_sigma, b_sigma, self.Mtype)
		elif isinstance(other, MFD):
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
		magnitudes = np.arange(self.get_min_mag_center(), self.max_mag,
								self.bin_width)
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
		return ((10**(a-b*min_mag))
				* ((10**(-1*b*mags) - 10**(-1*b*max_mag))
				/ (10**(-1*b*min_mag) - 10**(-1*b*max_mag))))
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
		Divide MFD into a number of MFD's that together sum up to the
		original MFD

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
		return [TruncatedGRMFD(self.min_mag, self.max_mag, self.bin_width, aw,
							self.b_val, a_sigma, self.b_sigma, self.Mtype)
							for aw in avalues]

	def split(self, M):
		"""
		Split MFD at a particular magnitude

		:param M:
			float, magnitude value where MFD should be split

		:return:
			List containing 2 instances of :class:`TruncatedGRMFD`
		"""
		if not self.is_magnitude_compatible(M):
			raise Exception("Magnitude value not compatible!")
		elif self.get_min_mag_edge() < M < self.max_mag:
			mfd1 = TruncatedGRMFD(self.min_mag, M, self.bin_width, self.a_val,
							self.b_val, self.a_sigma, self.b_sigma, self.Mtype)
			mfd2 = TruncatedGRMFD(M, self.max_mag, self.bin_width, self.a_val,
							self.b_val, self.a_sigma, self.b_sigma, self.Mtype)
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
		if (isinstance(other_mfd, TruncatedGRMFD)
			and other_mfd.b_val == self.bval
			and other_mfd.min_mag == self.max_mag):
			self.max_mag = other_mfd.max_mag
		else:
			raise Exception("MFD objects not compatible. "
							"Convert to EvenlyDiscretizedMFD")

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML truncGutenbergRichterMFD element)

		:param encoding:
			str, unicode encoding
			(default: 'latin1')
		"""
		from lxml import etree
		from ..nrml import ns

		tgr_elem = etree.Element(ns.TRUNCATED_GUTENBERG_RICHTER_MFD)
		tgr_elem.set(ns.A_VALUE, str(self.a_val))
		tgr_elem.set(ns.B_VALUE, str(self.b_val))
		tgr_elem.set(ns.MINIMUM_MAGNITUDE, str(self.min_mag))
		tgr_elem.set(ns.MAXIMUM_MAGNITUDE, str(self.max_mag))

		return tgr_elem

	def plot(self, color='k', style="-", label="", discrete=False,
			cumul_or_inc="cumul", completeness=None, end_year=None,
			Mrange=(), Freq_range=(), title="", lang="en", y_log_labels=True,
			fig_filespec=None, ax=None, fig_width=0, dpi=300):
		"""
		Plot magnitude-frequency distribution

		:param color:
			matplotlib color specification
			(default: 'k')
		:param style:
			matplotlib symbol style or line style
			(default: '-')
		:param label:
			str, plot labels
			(default: "")
		:param discrete:
			Bool, whether or not to plot discrete MFD
			(default: False)
		:param cumul_or_inc:
			str, either "cumul", "inc" or "both", indicating
			whether to plot cumulative MFD, incremental MFD or both
			(default: "cumul")
		:param completeness:
			instance of :class:`Completeness`, used to plot completeness
			limits
			(default: None)
		:param end_year:
			int, end year of catalog (used when plotting completeness limits)
			(default: None, will use current year)
		:param Mrange:
			(Mmin, Mmax) tuple, minimum and maximum magnitude in X axis
			(default: ())
		:param Freq_range:
			(Freq_min, Freq_max) tuple, minimum and maximum values in
			frequency (Y) axis
			(default: ())
		:param title:
			str, plot title
			(default: "")
		:param lang:
			str, language of plot axis labels
			(default: "en")
		:param y_log_labels:
			bool, whether or not Y axis labels are plotted as 10 to a power
			(default: True)
		:param fig_filespec:
			str, full path to output image file, if None plot to screen
			(default: None)
		:param ax:
			instance of :class:`~matplotlib.axes.Axes` in which plot will
			be made
			(default: None, will create new figure and axes)
		:param fig_width:
			float, figure width in cm, used to recompute :param:`dpi` with
			respect to default figure width
			(default: 0)
		:param dpi:
			int, image resolution in dots per inch
			(default: 300)

		:return:
			if both :param:`ax` and :param:`fig_filespec` are None, a
			(ax, fig) tuple will be returned
		"""
		from .plot import plot_mfds

		return plot_mfds([self], colors=[color], styles=[style], labels=[label],
						discrete=[discrete], cumul_or_inc=[cumul_or_inc],
						completeness=completeness, end_year=end_year,
						Mrange=Mrange, Freq_range=Freq_range, title=title,
						lang=lang, y_log_labels=y_log_labels,
						fig_filespec=fig_filespec, ax=ax, fig_width=fig_width,
						dpi=dpi)

	def to_truncated_GR_mfd(self, min_mag=None, max_mag=None, bin_width=None):
		"""
		Copy to another instance of :class:`TruncatedGRMFD`
		Optionally, non-defining parameters can be changed

		:param min_mag:
			float, lower magnitude (bin edge) of MFD
			(default: None)
		:param max_mag:
			float, upper magnitude of MFD
			(default: None)
		:param bin_width:
			float, magnitude bin width of MFD
			(default: None)
		"""
		if min_mag is None:
			min_mag = self.min_mag
		if max_mag is None:
			max_mag = self.max_mag
		if bin_width is None:
			bin_width = self.bin_width
		return TruncatedGRMFD(min_mag, max_mag, bin_width, self.a_val,
							self.b_val, self.a_sigma, self.b_sigma, self.Mtype)

	def get_mfd_from_b_val(self, b_val):
		"""
		Construct new MFD from a given b value, assuming uncertainties
		on a and b values are correlated. This supposes that the a and
		b values of the current MFD correspond to the mean values, and
		requires that the standard deviations of the a and b values are
		nonzero.

		:param b_val:
			float, imposed b value

		:return:
			instance of :class:`TruncatedGRMFD`
		"""
		if self.b_sigma == 0 or self.a_sigma == 0:
			raise Exception("a_sigma and b_sigma must not be zero!")
		epsilon = (b_val - self.b_val) / self.b_sigma
		a_val = self.a_val + self.a_sigma * epsilon
		return TruncatedGRMFD(self.min_mag, self.max_mag, self.bin_width, a_val,
							b_val, Mtype=self.Mtype)

	def sample_from_b_val(self, num_samples, num_sigma=2, random_seed=None):
		"""
		Generate MFD's by Monte Carlo sampling of the b value.
		The corresponding a value is computed from a_sigma (see
		:meth:`get_mfd_from_b_val`

		:param num_samples:
			int, number of MFD samples to generate
		:param num_sigma:
			float, number of standard deviations on b value
			(default: 2)
		:param random_seed:
			None or int, seed to initialize internal state of random number
			generator
			(default: None, will seed from current time)

		:return:
			List with instances of :class:`TruncatedGRMFD`
		"""
		import scipy.stats

		mfd_samples = []
		np.random.seed(seed=random_seed)
		mu, sigma = self.b_val, self.b_sigma
		b_values = scipy.stats.truncnorm.rvs(-num_sigma, num_sigma, mu, sigma,
											size=num_samples)
		for b_val in b_values:
			mfd_samples.append(self.get_mfd_from_b_val(b_val))
		return mfd_samples

	@classmethod
	def construct_FentonEtAl2006MFD(self, min_mag, max_mag, bin_width, area,
									b_val=0.7991):
		"""
		Construct "minimum" MFD for SCR according to Fenton et al. (2006),
		based on surface area

		:param min_mag:
			float, Minimum magnitude
			If None, take min_mag from current MFD
		:param max_mag:
			Maximum magnitude
			If None, take max_mag from current MFD
		:param bin_width:
			float, Magnitude interval for evenly discretized magnitude
			frequency distribution
			If None, take bin_width from current MFD
		:param area:
			float, area of region in square km
		:param b_val:
			float, Parameter of the truncated Gutenberg-Richter model.
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
		return TruncatedGRMFD(min_mag, max_mag, bin_width, a_val, b_val,
							stda, stdb, Mtype="MS")

	@classmethod
	def construct_Johnston1994MFD(self, min_mag, max_mag, bin_width, area,
								region="total"):
		"""
		Construct "minimum" MFD for SCR according to Johnston (1994),
		based on surface area

		:param min_mag:
			float, Minimum magnitude
			If None, take min_mag from current MFD
		:param max_mag:
			Maximum magnitude
			If None, take max_mag from current MFD
		:param bin_width:
			float, Magnitude interval for evenly discretized magnitude
			frequency distribution
			If None, take bin_width from current MFD
		:param area:
			float, area of region in square km
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

		mfd = TruncatedGRMFD(min_mag, max_mag, bin_width, a, b, stda, stdb,
							Mtype="MW")
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
			mfd = TruncatedGRMFD(self.min_mag, max_mag, self.bin_width,
								self.a_val, self.b_val)
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
					print("%s >= %.1f: %.1f per year" % (self.Mtype, M, rate))
				else:
					print("%s >= %.1f: 1 every %.0f years"
						% (self.Mtype, M, 1./rate))
		else:
			mfd1 = self.get_mfd_from_b_val(self.b_val + self.b_sigma)
			mfd2 = self.get_mfd_from_b_val(self.b_val - self.b_sigma)
			cumul_rates1 = mfd1.get_cumulative_rates()
			cumul_rates2 = mfd2.get_cumulative_rates()
			for i in idxs:
				M = mags[i]
				rate1, rate2 = cumul_rates1[i], cumul_rates2[i]
				if rate1 > 1 or rate2 > 1:
					print("%s >= %.1f: %.1f - %.1f per year"
						% ((self.Mtype, M) + tuple(np.sort([rate1, rate2]))))
				else:
					print("%s >= %.1f: 1 every %.1f - %.1f years"
						% ((self.Mtype, M) + tuple(np.sort([1./rate1, 1./rate2]))))



def alphabetalambda(a, b, M=0):
	"""
	Calculate alpha, beta, lambda from a, b, and M.
	(assuming truncated Gutenberg-Richter MFD)

	:param a:
		float, a value of Gutenberg-Richter relation
	:param b:
		float, b value of Gutenberg-Richter relation
	:param M:
		float, magnitude for which to compute lambda (default: 0)

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
		float, Frequency (1/yr) for magnitude M
	:param M:
		float, magnitude
	:param b:
		float, b value

	:return:
		float, a value
	"""
	return np.log10(lamda) + b * M


def get_a_separation(b, dM):
	"""
	Compute separation between a values of cumulative and discrete
	(assuming truncated Gutenberg-Richter MFD)

	:param b:
		float, b value
	:param dM:
		float, magnitude bin width
	"""
	return b * dM - np.log10(10**(b*dM) - 1)
