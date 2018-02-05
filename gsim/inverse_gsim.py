
import numpy as np
from scipy.optimize import minimize_scalar
from functools import partial

import openquake.hazardlib as oqhazlib



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

	def get_distance_metrics(self):
		return list(self.gsim.REQUIRES_DISTANCES)

	def get_imts(self):
		return [imt() for imt in self.gsim.DEFINED_FOR_INTENSITY_MEASURE_TYPES]

	def get_intensities_from_mag(self, mag, sctx, rctx, dctx, epsilon=0):
		"""
		Compute predicted intensities for given magnitude (= forward
		application of IPE)
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
		"""
		predicted_intensities = self.get_intensities_from_mag(mag, sctx, rctx,
														dctx, epsilon=epsilon)
		mse = np.mean((predicted_intensities - observed_intensities)**2)
		return mse

	def find_mag_from_intensity(self, observed_intensities, site_model, rupture,
								mag_bounds=(3, 9.5), epsilon=0):
		"""
		Given intensities observed at different sites, find magnitude
		"""
		sctx, rctx, dctx = self.gsim.make_contexts(site_model, rupture)
		minimize_func = partial(self.get_prediction_mse,
							observed_intensities=observed_intensities,
							sctx=sctx, rctx=rctx, dctx=dctx, epsilon=epsilon)
		result = minimize_scalar(minimize_func, bounds=mag_bounds,
								method='bounded')
		if result.success:
			return result.x
		else:
			print result.message
