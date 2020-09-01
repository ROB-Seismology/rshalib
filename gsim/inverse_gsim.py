"""
Inverse GSIM
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from functools import partial

from .. import oqhazlib


__all__ = ['InverseGSIM']


class InverseGSIM():
	"""
	Class implementing inverse Ground Shaking Intensity Model,
	which can be used to find magnitude, given intensity and distance.

	:param gsim_name:
		str, name of GSIM (IPE or GMPE)
	:param stddev_type:
		instance of :class:`oqhazlib.const.StdDev`, type of standard
		deviation
	"""
	def __init__(self, gsim_name, stddev_type=oqhazlib.const.StdDev.TOTAL):
		self.gsim = oqhazlib.gsim.get_available_gsims()[gsim_name]()
		self.imt = list(self.gsim.DEFINED_FOR_INTENSITY_MEASURE_TYPES)[0]()
		self.stddev_type = stddev_type

	def __repr__(self):
		return '<InverseGSIM %s>' % self.gsim

	def get_distance_metrics(self):
		"""
		:return:
			list of strings, required distance metrics for this GSIM
		"""
		return list(self.gsim.REQUIRES_DISTANCES)

	def get_imts(self):
		"""
		:return:
			list with instances of :class:`oqhazlib.imt.IMT` supported
			by this GSIM
		"""
		return [imt() for imt in self.gsim.DEFINED_FOR_INTENSITY_MEASURE_TYPES]

	def get_intensities_from_mag(self, mag, sctx, rctx, dctx, epsilon=0):
		"""
		Compute predicted intensities for given magnitude (= forward
		application of IPE)

		:param mag:
			float, magnitude
		:param sctx:
			dict, site context
		:param rctx:
			dict, rupture context
		:param dctx:
			dict, distance context
		:param epsilon:
			float, number of standard deviations above/below mean
			intensity
			(default: 0)

		:return:
			1D array, predicted intensities
		"""
		from copy import deepcopy
		rctx = deepcopy(rctx)
		rctx.mag = mag
		mmi, [sigma] = self.gsim.get_mean_and_stddevs(sctx, rctx, dctx, self.imt,
														[self.stddev_type])
		return self.gsim.to_imt_unit_values(mmi + epsilon * sigma)

	def get_prediction_mse(self, mag, observed_intensities, sctx, rctx, dctx,
							epsilon=0):
		"""
		Compute predicted intensities for given magnitude, and determine
		mean square error with respect to observed intensities

		:param mag:
			float, magnitude
		:param observed intensities
			1D array, observed intensities
		:param sctx:
		:param rctx:
		:param dctx:
			dicts, site, rupture and distance contexts
		:param epsilon:
			float, number of standard deviations above/below mean
			intensity
			(default: 0)

		:return:
			1D array, mean square errors between predicted and observed
			intensities
		"""
		predicted_intensities = self.get_intensities_from_mag(mag, sctx, rctx,
														dctx, epsilon=epsilon)
		mse = np.mean((predicted_intensities - observed_intensities)**2)
		return mse

	def find_mag_from_intensity(self, observed_intensities, site_model, rupture,
								mag_bounds=(3, 9.5), epsilon=0):
		"""
		Given intensities observed at different sites, find magnitude

		:param observed_intensities:
			1D array, observed intensities
		:param site_model:
			instance of :class:`rshalib.site.SoilSiteModel`
		:param rupture:
			instance of :class:`oqhazlib.source.Rupture`
		:param mag_bounds:
			(lower_mag, upper_mag) tuple, magnitude range to limit
			search to
			(default: (3, 9.5))
		:param epsilon:
			float, number of standard deviations above/below mean
			intensity
			(default: 0)

		:return:
			float, magnitude
			or None if no solution is found
		"""
		from scipy.optimize import minimize_scalar
		from .oqhazlib_gmpe import make_gsim_contexts

		sctx, rctx, dctx = make_gsim_contexts(self.gsim, site_model, rupture)
		minimize_func = partial(self.get_prediction_mse,
							observed_intensities=observed_intensities,
							sctx=sctx, rctx=rctx, dctx=dctx, epsilon=epsilon)
		result = minimize_scalar(minimize_func, bounds=mag_bounds,
								method='bounded')
		if result.success:
			return result.x
		else:
			print(result.message)
