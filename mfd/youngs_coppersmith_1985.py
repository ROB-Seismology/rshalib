# -*- coding: iso-Latin-1 -*-

"""
Youngs & Coppersmith (1985) characteristic MFD
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import openquake.hazardlib as oqhazlib

from .evenly_discretized import EvenlyDiscretizedMFD



__all__ = ['YoungsCoppersmith1985MFD']


class YoungsCoppersmith1985MFD(oqhazlib.mfd.YoungsCoppersmith1985MFD, EvenlyDiscretizedMFD):
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
		super(YoungsCoppersmith1985MFD, self).__init__(min_mag, a_val, b_val,
												char_mag, char_rate, bin_width)
		self.Mtype = "MW"

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

	def plot(self, color='k', style="-", label="", discrete=True,
			cumul_or_inc="both", completeness=None, end_year=None,
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
			str, plot labels (default: "")
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
		:param Mrange:
			(Mmin, Mmax) tuple, minimum and maximum magnitude in X axis
			(default: ())
		:param Freq_range:
			(Freq_min, Freq_max) tuple, minimum and maximum values in
			frequency (Y) axis
			(default: ())
		:param title:
			str, plot title (default: "")
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
		#TODO: is there a difference with plot method of EvenlyDiscretizedMFD??
		from .plot import plot_mfds

		return plot_mfds([self], colors=[color], styles=[style], labels=[label],
						discrete=[discrete], cumul_or_inc=[cumul_or_inc],
						completeness=completeness, end_year=end_year,
						Mrange=Mrange, Freq_range=Freq_range,
						title=title, lang=lang, y_log_labels=y_log_labels,
						fig_filespec=fig_filespec,
						ax=ax, fig_width=fig_width, dpi=dpi)
