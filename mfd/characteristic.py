# -*- coding: iso-Latin-1 -*-

"""
Characteristic MFD
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import int

import numpy as np

from .evenly_discretized import EvenlyDiscretizedMFD



__all__ = ['CharacteristicMFD']


class CharacteristicMFD(EvenlyDiscretizedMFD):
	"""
	MFD representing a characteristic earthquake, implemented as an
	evenly discretized MFD with one magnitude bin. The characteristic
	magnitude is taken to correspond to the left edge of the bin.

	:param char_mag:
		float, magnitude of characteristic earthquake
	:param return_period:
		float, return period of characteristic earthquake in year
	:param bin_width:
		float, magnitude bin width
	:param M_sigma:
		float, standard deviation on magnitude
		(default: 0.3)
	:param num_sigma:
		float, number of standard deviations to spread occurrence rates over
		(default: 0)
	:param Mtype:
		str, magnitude type
		(default: "MW")
	:param force_bin_alignment:
		bool, whether or not to enforce bin edges to aligh with bin width.
		If True, characteristic magnitude may be raised by up to half a
		bin width.
		(default: True)
	"""
	def __init__(self, char_mag, return_period, bin_width, M_sigma=0.3, num_sigma=0,
				Mtype="MW", force_bin_alignment=True):
		#self.char_mag = char_mag - bin_width / 2.
		if force_bin_alignment:
			self.char_mag = self._align_char_mag(char_mag, bin_width)
		else:
			self.char_mag = char_mag
		self.char_return_period = return_period
		self.M_sigma = M_sigma
		self.num_sigma = num_sigma

		Mmin, occurrence_rates = self._get_evenly_discretized_mfd_params(num_sigma,
																bin_width=bin_width)
		EvenlyDiscretizedMFD.__init__(self, Mmin, bin_width, occurrence_rates,
									Mtype=Mtype)

	def __repr__(self):
		txt = '<CharacteristicMFD | %s=%.2f +/- %.2f (x %s)>'
		txt %= (self.Mtype, self.char_mag, self.M_sigma, self.num_sigma)
		return txt

	def modify_set_char_mag(self, char_mag, force_bin_alignment=True):
		"""
		Set characteristic magnitude

		:param char_mag:
			float, characteristic magnitude
		:param force_bin_alignment:
			bool, whether or not to enforce bin edges to aligh with bin width.
			If True, characteristic magnitude may be raised by up to half a
			bin width.
			(default: True)

		:return:
			None, instance is modified in place
		"""
		#self.char_mag = char_mag - self.bin_width / 2.
		if force_bin_alignment:
			self.char_mag = self._align_char_mag(char_mag, bin_width)
		else:
			self.char_mag = char_mag
		Mmin, occurrence_rates = self._get_evenly_discretized_mfd_params(self.num_sigma,
																		bin_width=None)
		self.min_mag = Mmin
		self.modify_set_occurrence_rates(occurrence_rates)

	def _align_char_mag(self, char_mag, bin_width):
		"""
		Align characteristic magnitude to center of bin

		:param char_mag:
			float, characteristic magnitude
		:param bin_width:
			float, MFD bin width

		:return:
			float, aligned characteristic magnitude
		"""
		div, mod = divmod(char_mag, bin_width)
		if np.allclose(mod, bin_width):
			div += 1
			mod = 0
		return (div * bin_width) + bin_width / 2.

	def _get_evenly_discretized_mfd_params(self, num_sigma, bin_width=None):
		"""
		Compute parameters for constructing evenly discretized MFD

		:param num_sigma:
			float, number of standard deviations to spread occurrence rates over
		:param bin_width:
			float,  magnitude bin width
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
			#Mmin = np.floor(Mmin / bin_width) * bin_width
			#Mmax = np.ceil(Mmax / bin_width) * bin_width
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
		#return Mmin + bin_width/2, occurrence_rates
		return Mmin, occurrence_rates

	def set_num_sigma(self, num_sigma):
		"""
		Set number of standard deviations around characteristic magnitude

		:param num_sigma:
			float, number of standard deviations to spread occurrence rates over

		:return:
			None, instance is modified in place
		"""
		Mmin, occurrence_rates = self._get_evenly_discretized_mfd_params(num_sigma)
		self.min_mag = Mmin
		self.modify_set_occurrence_rates(occurrence_rates)
		self.num_sigma = num_sigma

	def __div__(self, other):
		if isinstance(other, (int, float)):
			return_period = self.return_period * other
			return CharacteristicMFD(self.char_mag, self.return_period,
									self.bin_width, self.M_sigma, self.num_sigma)
		else:
			raise TypeError("Divisor must be integer or float")

	def __mul__(self, other):
		if isinstance(other, (int, float)):
			return_period = self.return_period / other
			return CharacteristicMFD(self.char_mag, self.return_period,
									self.bin_width, self.M_sigma, self.num_sigma)
		else:
			raise TypeError("Multiplier must be integer or float")

	@property
	def return_period(self):
		return 1. / np.sum(self.occurrence_rates)

	def adjust_for_dsha(self):
		"""
		Adjust characteristic MFD for use in DSHA:
		set :prop:`num_sigma` to 0
		set :prop`occurrence_rates` to [1.0]
		Adjustment is done in place.
		"""
		self.set_num_sigma(0)
		self.modify_set_occurrence_rates([1.0])
