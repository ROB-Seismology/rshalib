"""
Truncated Gutenberg-Richter MFD in terms of the natural logarithm
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import int


import numpy as np

from .. import oqhazlib

from .base import (MFD, sum_mfds)
from .truncated_gr import TruncatedGRMFD



__all__ = ['TruncatedGRMFD']


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

	def __len__(self):
		return (self.max_mag - self.min_mag) / self.bin_width

	@property
	def a_val(self):
		return np.log(np.exp(alpha) / self.beta) / np.log(10)

	@property
	def b_val(self):
		return self.beta / np.log(10)

	def get_magnitude_bin_edges(self):
		return self.min_mag + np.arange(len(self)) * self.bin_width

	def get_magnitude_bin_centers(self):
		return self.get_magnitude_bin_edges() + self.bin_width / 2.

	def get_incremental_rates(self):
		mags = self.get_magnitude_bin_centers()
		rates = (np.exp(self.alpha - self.beta * mags)
				* 2 * np.sinh(self.beta * self.bin_width / 2.) / self.beta)
		return rates

	def get_lambda0(self):
		"""
		Cumulative frequency above lower magnitude
		"""
		beta = self.beta
		return (np.exp(self.alpha)
				* (np.exp(-beta * self.min_mag) - np.exp(-beta * self.max_mag))
				/ beta)

	def get_cumulative_rates(self):
		mags = self.get_magnitude_bin_edges()
		Mmin, Mmax = self.min_mag, self.max_mag
		lambda0 = self.get_lambda0()
		beta = self.beta
		nom = np.exp(-beta * (mags - Mmin)) - np.exp(-beta * (Mmax - Mmin))
		denom = 1 - np.exp(-beta * (Mmax - Mmin))
		return lambda0 * nom / denom

	def get_pdf(self):
		beta = self.beta
		mags = self.get_magnitude_bin_centers()
		Mmin, Mmax = self.min_mag, self.max_mag
		nom = beta * np.exp(-beta * mags)
		denom = np.exp(-beta * Mmin) - np.exp(-beta * Mmax)
		return nom / denom

	def to_truncated_gr_mfd(self):
		return TruncatedGRMFD(self.min_mag, self.max_mag, self.bin_width,
							self.a_val, self.b_val, cov=self.cov*np.log(10))
