# -*- coding: iso-Latin-1 -*-

"""
Youngs & Coppersmith (1985) characteristic MFD
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from .. import oqhazlib

from .evenly_discretized import EvenlyDiscretizedMFD



__all__ = ['YoungsCoppersmith1985MFD']


class YoungsCoppersmith1985MFD(oqhazlib.mfd.YoungsCoppersmith1985MFD, EvenlyDiscretizedMFD):
	"""
	Class implementing the MFD for the 'Characteristic Earthquake Model'
	by Youngs & Coppersmith (1985)

	Note: the parameters of the Gutenberg-Richter and characteristic parts
	must match, so it is best to create this MFD using
	:meth:`from_characteristic_rate` or :meth:`from_toal_moment_rate` !

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
	Mtype = "MW"

	def __repr__(self):
		txt = '<YoungsCoppersmith1985MFD | %s=%.2f:%.2f+/-0.25:%.2f | a=%.2f, b=%.2f>'
		txt %= (self.Mtype, self.min_mag, self.char_mag, self.bin_width,
				self.a_val, self.b_val)
		return txt

	@property
	def occurrence_rates(self):
		return np.array(list(zip(*self.get_annual_occurrence_rates()))[1])

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
