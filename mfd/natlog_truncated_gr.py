"""
Truncated Gutenberg-Richter MFD in terms of the natural logarithm

Mainly useful as a placeholder for the different formulas,
and to demonstrate the relation between a, b, alpha and beta values
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import int


import numpy as np

from .. import oqhazlib

from .base import MFD
from .truncated_gr import TruncatedGRMFD



__all__ = ['NatLogTruncatedGRMFD']


class NatLogTruncatedGRMFD(MFD):
	"""
	"""
	def __init__(self, min_mag, max_mag, bin_width, alpha, beta,
				cov=np.mat(np.zeros((2, 2))), Mtype='MW'):
		self.min_mag = np.round(min_mag / bin_width) * bin_width
		self.max_mag = np.round(max_mag / bin_width) * bin_width
		self.bin_width = bin_width
		self.alpha = alpha
		self.beta = beta
		self.cov = cov
		self.Mtype = Mtype

		self.check_constraints()

	def __repr__(self):
		txt = '<NatLogTruncatedGRMFD | %s=%.2f:%.2f:%.2f | alpha=%.2f, beta=%.2f>'
		txt %= (self.Mtype, self.min_mag, self.max_mag, self.bin_width,
				self.alpha, self.beta)
		return txt

	## Methods required by oqhazlib.mfd.MFD

	def check_constraints(self):
		"""
		Checks the following constraints:

		* Bin width is greater than 0.
		* Maximum magnitude is greater than minimum magnitude
		  by at least one bin width (or equal to that value).
		* ``b`` value is positive.
		"""
		if not self.bin_width > 0:
			raise ValueError('bin width %g must be positive' % self.bin_width)

		if not self.max_mag >= self.min_mag + self.bin_width:
			raise ValueError('maximum magnitude %g must be higher than '
							 'minimum magnitude %g by '
							 'bin width %g at least'
							 % (self.max_mag, self.min_mag, self.bin_width))

		if not 0 < self.beta:
			raise ValueError('beta %g must be non-negative' % self.beta)

	def _get_min_mag_and_num_bins(self):
		"""
		Count the number of bins in the histogram and return it
		along with the first bin center abscissa (magnitude) value.

		Rounds ``min_mag`` and ``max_mag`` with respect to ``bin_width``
		to make the distance between them include integer number of bins.

		:returns:
			A tuple of two items: first bin center and total number of bins.
		"""
		min_mag = self.get_min_mag_center()
		max_mag = self.get_max_mag_center()
		# here we use math round on the result of division and not just
		# cast it to integer because for some magnitude values that can't
		# be represented as an IEEE 754 double precisely the result can
		# look like 7.999999999999 which would become 7 instead of 8
		# being naively casted to int so we would lose the last bin.
		num_bins = int(round((max_mag - min_mag) / self.bin_width)) + 1
		return min_mag, num_bins

	def get_min_max_mag(self):
		"""
		Return the minum and maximum magnitude (bin centers)
		"""
		return (self.get_min_mag_center(), self.get_max_mag_center())

	@property
	def a_val(self):
		return np.log(np.exp(self.alpha) / self.beta) / np.log(10)

	@property
	def b_val(self):
		return self.beta / np.log(10)

	@property
	def alpha_sigma(self):
		return np.sqrt(self.cov[0,0])

	@property
	def beta_sigma(self):
		return np.sqrt(self.cov[1,1])

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

	def get_max_mag_center(self):
		"""
		Return center value of maximum magnitude bin

		:return:
			float
		"""
		return self.max_mag - self.bin_width / 2.0

	def get_incremental_rates(self):
		"""
		"""
		mags = self.get_magnitude_bin_centers()
		rates = (np.exp(self.alpha - self.beta * mags)
				* 2 * np.sinh(self.beta * self.bin_width / 2.) / self.beta)
		return rates

	@property
	def occurrence_rates(self):
		return self.get_incremental_rates()

	def get_annual_occurrence_rates(self):
		"""
		"""
		mags = self.get_magnitude_bin_centers()
		rates = self.get_incremental_rates()
		return list(zip(mags, rates))

	def get_lambda0(self):
		"""
		Cumulative frequency above lower magnitude
		"""
		## Eq. 3 in Stromeyer & Gruenthal (2015)
		beta = self.beta
		Mmin, Mmax = self.min_mag, self.max_mag
		return (np.exp(self.alpha)
				* (np.exp(-beta * Mmin) - np.exp(-beta * Mmax))
				/ beta)

	def get_cumulative_rates(self):
		"""
		"""
		mags = self.get_magnitude_bin_edges()
		Mmin, Mmax = self.min_mag, self.max_mag
		lambda0 = self.get_lambda0()
		beta = self.beta
		nom = np.exp(-beta * (mags - Mmin)) - np.exp(-beta * (Mmax - Mmin))
		denom = 1 - np.exp(-beta * (Mmax - Mmin))
		return lambda0 * nom / denom

	def get_pdf(self):
		"""
		"""
		beta = self.beta
		mags = self.get_magnitude_bin_centers()
		Mmin, Mmax = self.min_mag, self.max_mag
		nom = beta * np.exp(-beta * mags)
		denom = np.exp(-beta * Mmin) - np.exp(-beta * Mmax)
		return nom / denom

	def to_truncated_gr_mfd(self):
		"""
		"""
		return TruncatedGRMFD(self.min_mag, self.max_mag, self.bin_width,
							self.a_val, self.b_val,
							#cov=self.cov*np.log(10),
							Mtype=self.Mtype)

	def construct_mfd_at_epsilon(self, epsilon):
		"""
		:return:
			instance of :class:`EvenlyDiscretizedMFD`
		"""
		from eqcatalog.calcGR_MLE import construct_mfd_at_epsilon

		mfd = construct_mfd_at_epsilon(self.alpha, self.beta, self.cov, epsilon,
										self.min_mag, self.max_mag, self.bin_width,
										precise=True, log10=False)
		return mfd

	def get_uncertainty_pmf(self, num_discretizations=5, Mmax_pmf=None):
		"""
		"""
		from eqcatalog.calcGR_MLE import construct_mfd_pmf

		if Mmax_pmf is None:
			Mmax_pmf = self.max_mag
		mfds, weights = construct_mfd_pmf(self.alpha, self.beta, self.cov,
										self.min_mag, Mmax_pmf, self.bin_width,
										num_discretizations, precise=True,
										log10=False)
		return (mfds, weights)

	def plot(self, label="", color='k', style=None, lw_or_ms=None,
			discrete=False, cumul_or_inc="cumul",
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
