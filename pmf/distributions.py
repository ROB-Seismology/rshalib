# -*- coding: utf-8 -*-

"""
Classes representing source-model elements in Openquake/nhlib. Where possible,
the classes are inherited from nhlib classes. All provide methods to create
XML elements, which are used to write a complete source-model NRML file.
Thus, objects instantiated from these classes can be used directly in nhlib,
as well as to generate input files for OpenQuake.
"""

from lxml import etree
import decimal
from decimal import Decimal

import numpy as np
from scipy import stats

import openquake.hazardlib as nhlib

from ..nrml import ns
from ..nrml.common import *
from ..geo import NodalPlane
from ..utils import interpolate, wquantiles



class PMF(nhlib.pmf.PMF):
	def __init__(self, data):
		super(PMF, self).__init__(data)

	@classmethod
	def from_values_and_weights(cls, values, weights):
		"""
		Construct new PMF from list of values and weights

		:param values:
			list or array of values (can be any type)
		:param weights:
			list or array of corresponding weights

		:return:
			new instance of same class
		"""
		assert len(values) > 0 and len(values) == len(weights)
		weights = np.array([Decimal(str(w)) for w in weights])
		weights /= np.sum(weights)
		## If sum is not one, adjust last weight
		if np.sum(weights) != 1.0:
			weights[-1] = Decimal(1) - np.sum(weights[:-1])

		data = zip(weights, values)
		return cls(data)

	@property
	def weights(self):
		return np.array([item[0] for item in self.data])

	@property
	def values(self):
		return [item[1] for item in self.data]

	def __len__(self):
		return len(self.data)

	def __iter__(self):
		for weight, value in self.data:
			yield (value, weight)


class NumericPMF(PMF):
	"""
	Class representing a PMF where values correspond to numbers,
	possibly with regular spacing
	"""
	def __init__(self, data):
		super(NumericPMF, self).__init__(data)

	@property
	def values(self):
		return np.array([item[1] for item in self.data])

	def min(self):
		return self.values[np.where(self.weights > 1E-8)].min()

	def max(self):
		return self.values.max()

	def get_mean(self):
		"""
		Return weighted mean of PMF
		"""
		return np.average(self.values, weights=self.weights)

	def get_median(self):
		"""
		Return weighted median of PMF
		"""
		[median] = self.get_percentiles([50])
		return median

	def get_percentiles(self, percentiles, interpol=False):
		"""
		Compute weighted percentiles of PMF

		:param percentiles:
			list or array of percentiles in the range 0-1 or 0-100
			(determined automatically from max. value)
		:param interpol:
			bool, whether or not percentile intercept should be
			interpolated (default: False)

		:return:
			array containing interpolated values corresponding to
			percentile intercepts
		"""
		quantile_levels = np.array(percentiles, 'f') / 100.
		weights = self.weights.astype('d')
		percentile_intercepts = wquantiles(self.values, weights, quantile_levels, interpol=interpol)
		return percentile_intercepts

	def rebin_equal_weight(self, num_bins=5, precision=4):
		"""
		Rebin PMF into bins with (approximately) equal weight,
		such that bin values are a multiple of the original bin_width
		(assumed to correspond to value spacing), rounded up

		:param num_bins:
			Int, (maximum) number of bins in output PMF
		:param precision:
			Integer, decimal precision of weights (default: 4)

		:return:
			instance of :class:`PMF` or subclass
			Note that actual number of bins may be less than specified
			if one or more bins have much higher weight than the average.
		"""
		assert num_bins > 1
		values = self.values
		weights = self.weights.astype('d')
		bin_width = values[1] - values[0]
		cumul_weights = np.add.accumulate(weights)

		start_index = max(0, np.where(weights > 1E-6)[0][0])
		bin_edge_indexes = set([start_index])
		for bin_edge_weight in np.linspace(1.0/num_bins, 1.0, num_bins):
			index = np.argmin(np.abs(cumul_weights - bin_edge_weight))
			bin_edge_indexes.add(index)
		bin_edge_indexes = sorted(bin_edge_indexes)

		bin_edges = values[bin_edge_indexes]
		bin_centers = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2
		## We use ceil because max_mag in MFD has zero occurrence rate
		bin_centers_rounded = np.ceil(bin_centers / bin_width) * bin_width

		bin_cumul_weights = cumul_weights[bin_edge_indexes]
		bin_weights = bin_cumul_weights[1:] - bin_cumul_weights[:-1]
		bin_weights = bin_weights.clip(Decimal('1E-%d' % precision))

		decimal.getcontext().prec = precision
		bin_weights = np.array([Decimal(val) for val in bin_weights])
		bin_weights = adjust_decimal_weights(bin_weights, precision)

		return self.__class__(bin_centers_rounded, bin_weights)

	def rebin_equal_length(self, num_bins=5, precision=4):
		"""
		Rebin PMF into bins with equal length
		such that bin values are a multiple of the original bin_width
		(assumed to correspond to value spacing), rounded up

		:param num_bins:
			Int, (maximum) number of bins in output PMF
		:param precision:
			Integer, decimal precision of weights (default: 4)

		:return:
			instance of :class:`PMF` or subclass
			Note that actual number of bins may be less than specified.
		"""
		values = self.values
		weights = self.weights.astype('d')
		bin_width = values[1] - values[0]
		cumul_weights = np.add.accumulate(weights)

		start_index = max(0, np.where(weights > 1E-6)[0][0])
		end_index = min(len(weights) - 1, np.where(weights > 1E-6)[0][-1])
		total_num_bins = end_index - start_index + 1
		int_bin_interval = float(total_num_bins) / num_bins
		## Make sure int_bin_interval is odd
		if np.ceil(int_bin_interval) % 2 == 1:
			int_bin_interval = int(np.ceil(int_bin_interval))
		else:
			int_bin_interval = int(np.floor(int_bin_interval))

		bin_edge_indexes = range(start_index, end_index+1, int_bin_interval)
		if bin_edge_indexes[-1] != end_index:
			bin_edge_indexes.append(end_index)

		bin_edges = values[bin_edge_indexes]
		bin_centers = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2
		bin_centers_rounded = np.floor(bin_centers / bin_width) * bin_width

		bin_weights = np.zeros_like(bin_centers_rounded)
		for i in range(len(bin_weights)):
			start_index = bin_edge_indexes[i]
			end_index = bin_edge_indexes[i+1]
			if i + 1 == len(bin_weights):
				end_index += 1
			bin_weights[i] = np.sum(weights[start_index:end_index])
		bin_cumul_weights = cumul_weights[bin_edge_indexes]

		decimal.getcontext().prec = precision
		bin_weights = np.array([Decimal(val) for val in bin_weights])
		bin_weights = adjust_decimal_weights(bin_weights, precision)

		return self.__class__(bin_centers_rounded, bin_weights)


class GMPEPMF(PMF):
	"""
	Class representing GMPE uncertainties to be used in a logic tree

	:param gmpe_names:
		list of GMPE names
	:param weights:
		list of corresponding weights
	"""
	def __init__(self, gmpe_names, weights):
		if len(gmpe_names) != len(weights):
			raise Exception("Number of weights and number of GMPE names must be identical!")
		data = data = PMF.from_values_and_weights(gmpe_names, weights).data
		super(GMPEPMF, self).__init__(data)

	@property
	def gmpe_names(self):
		return self.values


class SourceModelPMF(PMF):
	"""
	Class representing source-model uncertainties to be used in a logic tree

	:param source_models:
		list of instances of :class:`SourceModel`
	:param weights:
		list of corresponding weights
	"""
	def __init__(self, source_models, weights):
		if len(source_models) != len(weights):
			raise Exception("Number of weights and number of source models must be identical!")
		data = PMF.from_values_and_weights(source_models, weights).data
		super(SourceModelPMF, self).__init__(data)

	@property
	def source_models(self):
		return self.values


class MmaxPMF(NumericPMF):
	"""
	Class representing Mmax uncertainties to be used in a logic tree

	:param max_mags:
		list of Mmax values
	:param weights:
		list of corresponding weights
	:param absolute:
		Bool, whether Mmax values are absolute (True) or relative (False) values
		(default: False)
	"""
	def __init__(self, max_mags, weights, absolute=True):
		if len(max_mags) != len(weights):
			raise Exception("Number of weights and number of magnitudes must be identical!")
		data = PMF.from_values_and_weights(max_mags, weights).data
		super(MmaxPMF, self).__init__(data)
		self.absolute = absolute

	@property
	def max_mags(self):
		return self.values


class MFDPMF(PMF):
	"""
	Class representing MFD uncertainties to be used in a logic tree

	:param mfd_values:
		list with MFD values, either:
		- relative increments to a b value
		- (a, b) tuples
		- arrays of incremental occurrence rates
	:param weights:
		list of corresponding weights
	"""
	def __init__(self, mfd_values, weights):
		if len(mfd_values) != len(weights):
			raise Exception("Number of weights and number of MFD values must be identical!")
		data = data = PMF.from_values_and_weights(mfd_values, weights).data
		super(MFDPMF, self).__init__(data)

	@property
	def mfd_values(self):
		return self.values

	def is_bGRRelative(self):
		"""
		Determine whether MFD values are relative b values
		"""
		if isinstance(self.values[0], (int, float)):
			return True
		else:
			return False

	def is_abGRAbsolute(self):
		"""
		Determine whether MFD values are absolute (a, b) values
		"""
		try:
			if len(self.values[0]) == 2:
				return True
			else:
				return False
		except:
			return False

	def is_incrementalMFDRates(self):
		"""
		Determine whether MFD values are incremental occurrence rates
		for an EvenlyDiscretizedMFD. If the length of items in MFD values
		is larger than two, they are considered to be occurrence rates.
		"""
		try:
			if len(self.values[0]) > 2:
				return True
			else:
				return False
		except:
			return False


class NodalPlaneDistribution(PMF):
	"""
	Class representing a nodal-plane distribution

	:param nodal_planes:
		list of NodalPlane objects
	:param weights:
		list of corresponding weights
	"""
	def __init__(self, nodal_planes, weights):
		if len(nodal_planes) != len(weights):
			raise Exception("Number of weights and number of nodal planes must be identical!")
		data = zip(weights, nodal_planes)
		super(NodalPlaneDistribution, self).__init__(data)

	@property
	def nodal_planes(self):
		return self.values

	def __iter__(self):
		for nodal_plane, weight in zip(self.nodal_planes, self.weights):
			yield (nodal_plane, weight)

	def __len__(self):
		return len(self.nodal_planes)

	def get_strike_weights(self):
		"""
		Return weights for each strike

		:return:
			Tuple of arrays (strikes, weights)
		"""
		strikes, weights = [], []
		for nodal_plane, weight in self:
			strike = nodal_plane.strike
			try:
				idx = strikes.index(strike)
			except ValueError:
				strikes.append(strike)
				weights.append(weight)
			else:
				weights[idx] += weight
		return np.array(strikes), np.array(weights)

	def get_dip_weights(self):
		"""
		Return weights for each dip

		:return:
			Tuple of arrays (dips, weights)
		"""
		dips, weights = [], []
		for nodal_plane, weight in self:
			dip = nodal_plane.dip
			try:
				idx = dips.index(dip)
			except ValueError:
				dips.append(dip)
				weights.append(weight)
			else:
				weights[idx] += weight
		return np.array(dips), np.array(weights)

	def get_rake_weights(self):
		"""
		Return weights for each faulting style

		:return:
			Tuple of arrays (rakes, weights)
		"""
		rakes, weights = np.array([-90, 90, 0]), np.zeros(3, 'f')
		for nodal_plane, weight in self:
			if -135 <= nodal_plane.rake <= -45:
				weights[0] += float(weight)
			elif 45 <= nodal_plane.rake <= 135:
				weights[1] += float(weight)
			else:
				weights[2] += float(weight)
		return rakes, weights

	def get_main_rake(self):
		"""
		Return the "main" rake.
		If nodal-plane distribution has different rakes, take the one
		with the highest weight or the first one if they all have the
		same weight

		:return:
			Int: -90 for normal, 90 for reverse, or 0 for strike-slip
		"""
		rakes, weights = self.get_rake_weights()
		rake = rakes[weights.argmax()]
		return rake

	def get_num_rakes(self):
		"""
		Return number of faulting styles in distribution

		:return:
			Int, number of faulting styles
		"""
		rakes, weights = self.get_rake_weights()
		return len(np.where(weights > 0.)[0])

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML nodalPlaneDist element)

		:param encoding:
			String, unicode encoding (default: 'latin1')
		"""
		npd_elem = etree.Element(ns.NODAL_PLANE_DISTRIBUTION)
		for weight, np in zip(self.weights, self.nodal_planes):
			np_elem = etree.SubElement(npd_elem, ns.NODAL_PLANE)
			np_elem.set(ns.PROBABILITY, str(weight))
			np_elem.set(ns.STRIKE, str(np.strike))
			np_elem.set(ns.DIP, str(np.dip))
			np_elem.set(ns.RAKE, str(np.rake))
		return npd_elem

	def print_distribution(self):
		"""
		Print strike, dip, rake and weight of all nodal planes
		in distribution
		"""
		print "Strike Dip Rake Weight"
		for nodal_plane, weight in zip(self.nodal_planes, self.weights):
			print "   %3d  %2d %4d %s" % (nodal_plane.strike, nodal_plane.dip, nodal_plane.rake, weight)


class HypocentralDepthDistribution(PMF):
	"""
	Class representing a hypocentral depth distribution

	:param hypo_depths:
		list of hypocentral depths
	:param weights:
		list of corresponding weights
	"""
	def __init__(self, hypo_depths, weights):
		if len(hypo_depths) != len(weights):
			raise Exception("Number of weights and number of hypocentral depths must be identical!")
		data = zip(weights, hypo_depths)
		super(HypocentralDepthDistribution, self).__init__(data)

	def __iter__(self):
		for hypo_depth, weight in zip(self.hypo_depths, self.weights):
			yield (hypo_depth, weight)

	def __len__(self):
		return len(self.hypo_depths)

	@property
	def hypo_depths(self):
		return np.array(self.values)

	def get_mean_depth(self, weighted=True):
		"""
		Return mean depth.

		:param weighted:
			Boolean, indicating whether mean should be weighted (default: True).

		:return:
			Float, mean depth
		"""
		if weighted:
			return np.average(self.hypo_depths, weights=self.weights)
		else:
			return np.mean(self.hypo_depths)

	def get_min_depth(self):
		"""
		Return minimum (shallowest) depth

		:return:
			Float, minimum depth
		"""
		return self.hypo_depths.min()

	def get_max_depth(self):
		"""
		Return maximum (deepest) depth

		:return:
			Float, maximum depth
		"""
		return self.hypo_depths.max()

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML hypoDepthDist element)

		:param encoding:
			String, unicode encoding (default: 'latin1')
		"""
		hdd_elem = etree.Element(ns.HYPOCENTRAL_DEPTH_DISTRIBUTION)
		for weight, depth in zip(self.weights, self.hypo_depths):
			hd_elem = etree.SubElement(hdd_elem, ns.HYPOCENTRAL_DEPTH)
			hd_elem.set(ns.PROBABILITY, str(weight))
			hd_elem.set(ns.DEPTH, str(depth))
		return hdd_elem

	def print_distribution(self):
		"""
		Print strike, dip, rake and weight of all nodal planes
		in distribution
		"""
		print "Depth Weight"
		for depth, weight in zip(self.hypo_depths, self.weights):
			print "%5.2f %s" % (depth, weight)


def adjust_decimal_weights(weights, precision=4):
	"""
	Ensure weights add up exactly to 1 by, if necessary, modifying
	the last weight in the list

	:param weights:
		list or array of decimal weights
	:param precision:
		Integer, decimal precision of weights (default: 4)

	:return:
		list or array of decimal weights
	"""
	if len(weights) > 1:
		## quantize fails if weight = 1 (precision != number of decimal digits)
		## We suppose that if there is only 1 weight, it is equal to 1
		decimal.getcontext().prec = precision
		quantize_str = '0.' + '0' * precision
		weights = [w.quantize(Decimal(quantize_str), rounding=decimal.ROUND_HALF_EVEN) for w in weights]
		## Increase precision by 1 to check if sum == 1 !
		decimal.getcontext().prec = precision + 1
		if np.sum(weights) != 1:
			weights[-1] = Decimal(1) - np.sum(weights[:-1])
	return weights


def create_nodal_plane_distribution(strike_range, dip_range, rake_range, precision=4):
	"""
	Generate a uniform nodal plane distribution from strike, dip, and rake
	ranges.

	:param strike_range:
		- Tuple (min_strike, max_strike, delta_strike) in degrees
		- Single strike value (integer or float)
	:param dip_range:
		- Tuple (min_dip, max_dip, delta_dip) in degrees
		- Single dip value (integer or float)
	:param rake_range:
		- Tuple (min_rake, max_rake, delta_rake) in degrees
		- Single rake value (integer or float)
	:param precision:
		Integer, decimal precision of weights (default: 4)

	:return:
		instance of :class:`NodalPlaneDistribution`
	"""
	if isinstance(strike_range, (int, float)):
		strikes, strike_weights = [strike_range], [Decimal(1.0)]
	elif len(strike_range) == 3:
		min_strike, max_strike, delta_strike = strike_range
		strikes, strike_weights = get_uniform_distribution(min_strike, max_strike, delta_strike, precision=precision*2)
	else:
		raise Exception("strike_range tuple must contain 3 elements: min, max, and delta.")

	if isinstance(dip_range, (int, float)):
		dips, dip_weights = [dip_range], [Decimal(1.0)]
	elif len(dip_range) == 3:
		min_dip, max_dip, delta_dip = dip_range
		dips, dip_weights = get_uniform_distribution(min_dip, max_dip, delta_dip, precision=precision*2)
	else:
		raise Exception("dip_range tuple must contain 3 elements: min, max, and delta.")

	if isinstance(rake_range, (int, float)):
		rakes, rake_weights = [rake_range], [Decimal(1.0)]
	elif len(rake_range) == 3:
		min_rake, max_rake, delta_rake = rake_range
		rakes, rake_weights = get_uniform_distribution(min_rake, max_rake, delta_rake, precision=precision*2)
	else:
		raise Exception("rake_range tuple must contain 3 elements: min, max, and delta.")

	nodal_planes, nodal_plane_weights = [], []
	for strike, strike_weight in zip(strikes, strike_weights):
		for dip, dip_weight in zip(dips, dip_weights):
			for rake, rake_weight in zip(rakes, rake_weights):
				nodal_planes.append(NodalPlane(strike, dip, rake))
				nodal_plane_weights.append(strike_weight * rake_weight * dip_weight)
	nodal_plane_weights = adjust_decimal_weights(nodal_plane_weights, precision)

	return NodalPlaneDistribution(nodal_planes, nodal_plane_weights)


def get_normal_distribution(min, max, num_bins, sigma_range=2, precision=4, centers=True):
	"""
	Get normal distribution with bin centers as values.

	:param min:
		Float for minimum value of distribution.
	:param max:
		Float for maximum value of distribution.
	:param num_bins:
		Integer with number of bins for distribution.
	:param sigma_range:
		Sigma range for distribution between min and max values (Default: 2).
	:param precision:
		Integer, decimal precision of weights (default: 4)
	:param centers:
		Bool, whether min and max are to be considered as bin centers
		(True) or bin edges (False)
		(default: True)

	:return:
		tuple (values, weights)
		- values: Numpy array with distribution values, centers of bins (length = num_bins).
		- weights: Numpy array with weights of distribution values
	"""
	decimal.getcontext().prec = precision
	val_range = float(max - min)
	mean = (min + max) / 2.
	if val_range == 0. or num_bins == 1.:
		bin_centers = [mean]
		weights = [Decimal(1)]
	else:
		# Assume value range corresponds to +/- sigma_range
		sigma = val_range / (sigma_range * 2)
		bin_width = val_range / num_bins
		dist = stats.truncnorm(-sigma_range, sigma_range, mean, sigma)
		if centers:
			bin_centers = np.linspace(min, max, num_bins)
			weights = dist.pdf(bin_centers)
		else:
			bin_edges = np.linspace(min, max, num_bins+1)
			bin_centers = bin_edges[:-1] + bin_width / 2
			## Compute probability of area covered by each bin
			bin_edge_probs = dist.cdf(bin_edges)
			weights = bin_edge_probs[1:] - bin_edge_probs[:-1]

		## Normalize
		weights = np.array([Decimal(w) for w in weights])
		weights /= sum(weights)
		## Ensure sum == 1
		weights = adjust_decimal_weights(weights, precision)
	return bin_centers, weights


def get_normal_distribution_bin_edges(min, max, num_bins, sigma_range=2, precision=4):
	"""
	Get normal distribution with bin edges as values.

	:param min:
		Float for minimum value of distribution.
	:param max:
		Float for maximum value of distribution.
	:param num_bins:
		Integer with number of bins for distribution.
	:param sigma_range:
		Sigma range for distribution between min and max values (Default: 2).
	:param precision:
		Integer, decimal precision of weights (default: 4)

	:return:
		tuple (values, weights)
		- values: Numpy array with distribution values, edges of bins (length = num_bins).
		- weights: Numpy array with weights of distribution values
	"""
	decimal.getcontext().prec = precision
	val_range = max - min
	mean = (min + max) / 2.
	sigma = val_range / (sigma_range * 2)
	bin_edges = np.linspace(min, max, num_bins+1)
	weights = stats.distributions.norm.pdf(bin_edges, loc=mean, scale=sigma)
	weights = np.array([Decimal(w) for w in weights])
	weights /= sum(weights)

	## Ensure sum == 1
	weights = adjust_decimal_weights(weights, precision)
	return bin_edges, weights


def get_uniform_distribution(min, max, delta, precision=4):
	"""
	Get uniform distribution.

	:param min:
		Float for minimum value of distribution.
	:param max:
		Float for maximum value of distribution.
	:param delta:
		Float for step between distribution values.
	:param precision:
		Integer, decimal precision of weights (default: 4)

	:return:
		tuple (values, weights)
		- values: Numpy array with distribution values (equally spaced by delta from min to max).
		- weights: Numpy array with weights of distribution values. See get_uniform_weights.
	"""
	if delta == 0:
		values = [np.mean([min, max])]
	else:
		# TODO: check if this works as expected !
		max = min + np.floor((max - min) / delta) * delta
		values = np.arange(min, max+1, delta)
		#values = np.arange(min, max+delta, delta)
	weights = get_uniform_weights(len(values), precision)
	return values, weights


def get_uniform_weights(num_weights, precision=4):
	"""
	Get uniform weights.

	:param num_weights:
		Integer, number of weights.
	:param precision:
		Integer, decimal precision (default: 4)

	:return:
		Numpy array (length equal to num_weights) of equal weights summing up to 1.0.
	"""
	decimal.getcontext().prec = precision
	weights = np.array([Decimal(1) for i in range(num_weights)])
	weights /= Decimal(num_weights)

	## Ensure sum == 1
	weights = adjust_decimal_weights(weights, precision)
	return weights



if __name__ == '__main__':
	pass
