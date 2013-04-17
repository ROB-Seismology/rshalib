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



class NodalPlaneDistribution(nhlib.pmf.PMF):
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
		self.nodal_planes = nodal_planes
		self.weights = np.array(weights)
		data = zip(weights, nodal_planes)
		super(NodalPlaneDistribution, self).__init__(data)

	def __iter__(self):
		for nodal_plane, weight in zip(self.nodal_planes, self.weights):
			yield (nodal_plane, weight)

	def __len__(self):
		return len(self.nodal_planes)

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
			print "   %3d  %2d %4d %.4f" % (nodal_plane.strike, nodal_plane.dip, nodal_plane.rake, weight)


class HypocentralDepthDistribution(nhlib.pmf.PMF):
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
		self.hypo_depths = np.array(hypo_depths)
		self.weights = np.array(weights)
		data = zip(weights, hypo_depths)
		super(HypocentralDepthDistribution, self).__init__(data)

	def __iter__(self):
		for hypo_depth, weight in zip(self.hypo_depths, self.weights):
			yield (hypo_depth, weight)

	def __len__(self):
		return len(self.hypo_depths)

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
			print "%5.2f %.4f" % (depth, weight)


def create_nodal_plane_distribution(strike_range, dip_range, rake_range):
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

	:return:
		instance of :class:`NodalPlaneDistribution`
	"""
	if isinstance(strike_range, (int, float)):
		strikes, strike_weights = [strike_range], [Decimal(1.0)]
	elif len(strike_range) == 3:
		min_strike, max_strike, delta_strike = strike_range
		strikes, strike_weights = get_uniform_distribution(min_strike, max_strike, delta_strike)
	else:
		raise Exception("strike_range tuple must contain 3 elements: min, max, and delta.")

	if isinstance(dip_range, (int, float)):
		dips, dip_weights = [dip_range], [Decimal(1.0)]
	elif len(dip_range) == 3:
		min_dip, max_dip, delta_dip = dip_range
		dips, dip_weights = get_uniform_distribution(min_dip, max_dip, delta_dip)
	else:
		raise Exception("dip_range tuple must contain 3 elements: min, max, and delta.")

	if isinstance(rake_range, (int, float)):
		rakes, rake_weights = [rake_range], [Decimal(1.0)]
	elif len(rake_range) == 3:
		min_rake, max_rake, delta_rake = rake_range
		rakes, rake_weights = get_uniform_distribution(min_rake, max_rake, delta_rake)
	else:
		raise Exception("rake_range tuple must contain 3 elements: min, max, and delta.")

	nodal_planes, nodal_plane_weights = [], []
	for strike, strike_weight in zip(strikes, strike_weights):
		for dip, dip_weight in zip(dips, dip_weights):
			for rake, rake_weight in zip(rakes, rake_weights):
				nodal_planes.append(NodalPlane(strike, dip, rake))
				nodal_plane_weights.append(strike_weight * rake_weight * dip_weight)

	return NodalPlaneDistribution(nodal_planes, nodal_plane_weights)


def get_normal_distribution(min, max, num_bins, sigma_range=2):
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
	:return values:
		Numpy array with distribution values, centers of bins (length = num_bins).
	:return weights:
		Numpy array with weights of distribution values
	"""
	#decimal.getcontext().prec = 4
	val_range = max - min
	mean = (min + max) / 2.
	if val_range == 0. or num_bins == 1.:
		bin_centers = [mean]
		weights = [1.]
	else:
		# Assume depth range corresponds to +/ sigma_range
		sigma = val_range / (sigma_range * 2)
		bin_width = val_range / num_bins
		bin_edges = np.linspace(min, max, num_bins+1)
		bin_centers = bin_edges[:-1] + bin_width / 2
		bin_edge_probs = stats.distributions.norm.cdf(bin_edges, loc=mean, scale=sigma)
		## Compute probability of area covered by each bin
		weights = bin_edge_probs[1:] - bin_edge_probs[:-1]
		## Normalize
		weights = np.array([Decimal(w) for w in weights])
		weights /= sum(weights)
		## Check if and set sum == 1
		if sum(weights) != 1.0:
			weights[-1] = 1.0 - sum(weights[:-1])
	return bin_centers, weights


def get_normal_distribution_bin_edges(min, max, num_bins, sigma_range=2):
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
	:return values:
		Numpy array with distribution values, edges of bins (length = num_bins).
	:return weights:
		Numpy array with weights of distribution values
	"""
	#decimal.getcontext().prec = 4
	val_range = max - min
	mean = (min + max) / 2.
	sigma = val_range / (sigma_range * 2)
	bin_edges = np.linspace(min, max, num_bins+1)
	weights = stats.distributions.norm.pdf(bin_edges, loc=mean, scale=sigma)
	weights = np.array([Decimal(w) for w in weights])
	weights /= sum(weights)
	## Check if and set sum == 1
	if sum(weights) != 1.0:
		weights[-1] = 1 - sum(weights[:-1])
	return bin_edges, weights


def get_uniform_distribution(min, max, delta):
	"""
	Get uniform distribution.

	:param min:
		Float for minimum value of distribution.
	:param max:
		Float for maximum value of distribution.

	:param delta:
		Float for step between distribution values.
	:return values:
		Numpy array with distribution values (equally spaced by delta from min to max).
	:return weights:
		Numpy array with weights of distribution values. See get_uniform_weights.
	"""
	if delta == 0:
		values = [np.mean([min, max])]
	else:
		values = np.arange(min, max+delta, delta)
	weights = get_uniform_weights(len(values))
	return values, weights


def get_uniform_weights(num_weights):
	"""
	Get uniform weights.

	:param num_weights:
		Integer with number of weights.
	:return:
		Numpy array (length equal to num_weights) of equal weights summing up to 1.0.
	"""
	#decimal.getcontext().prec = 4
	weights = np.array([Decimal(1) for i in range(num_weights)])
	weights /= Decimal(num_weights)

	## Kludge, because it is simply not possible to generate uniform weights
	## for 3 items that sum up to 1.0 ...
	if sum(weights) != 1.0:
		weights[-1] = 1 - sum(weights[:-1])
	return weights



if __name__ == '__main__':
	pass
