"""
Base class for deaggregation results
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from .base_array import *



__all__ = ['DeaggBase']


class DeaggBase(object):
	"""
	Base class for deaggregation results

	:param bin_edges:
		6-tuple, containing:
			- magnitude bin edges
			- distance bin edges
			- longitude bin edges
			- latitude bin edges
			- epsilon bin edges
			- tectonic region types

	:param deagg_matrix:
		instance of :class:`ExceedanceRateMatrix` or :class:`ProbabilityMatrix`,
		6-or-more-dimensional array containing deaggregation values, with
		the 6 last dimensions corresponding to:
			- magnitude bins
			- distance bins
			- longitude bins
			- latitude bins
			- epsilon bins
			- tectonic-region-type bins

	:param timespan:
		float, time span in Poisson formula.
	"""
	def __init__(self, bin_edges, deagg_matrix, timespan):
		assert len(bin_edges) == 6, "bin_edges must contain 6 elements!"
		self.bin_edges = list(bin_edges)
		assert deagg_matrix.shape[-6] == max(1, len(bin_edges[0]) - 1), (
						"Number of magnitude bins not in accordance "
						"with specified bin edges!")
		assert deagg_matrix.shape[-5] == max(1, len(bin_edges[1]) - 1), (
						"Number of distance bins not in accordance "
						"with specified bin edges!")
		assert deagg_matrix.shape[-4] == max(1, len(bin_edges[2]) - 1), (
						"Number of longitude bins not in accordance "
						"with specified bin edges!")
		assert deagg_matrix.shape[-3] == max(1, len(bin_edges[3]) - 1), (
						"Number of latitude bins not in accordance "
						"with specified bin edges!")
		assert deagg_matrix.shape[-2] == max(1, len(bin_edges[4]) - 1), (
						"Number of epsilon bins not in accordance with "
						"specified bin edges!")
		assert deagg_matrix.shape[-1] == max(1, len(bin_edges[5])), (
						"Number of tectonic-region-type bins not in "
						"accordance with specified bin edges!")
		assert isinstance(deagg_matrix, DeaggMatrix), ("deagg_matrix must be "
						"instance of DeaggMatrix!")
		self.deagg_matrix = deagg_matrix
		self.timespan = timespan

	@property
	def matrix(self):
		return np.asarray(self.deagg_matrix)

	@property
	def nmags(self):
		return self.matrix.shape[-6]

	@property
	def ndists(self):
		return self.matrix.shape[-5]

	@property
	def nlons(self):
		return self.matrix.shape[-4]

	@property
	def nlats(self):
		return self.matrix.shape[-3]

	@property
	def neps(self):
		return self.matrix.shape[-2]

	@property
	def ntrts(self):
		return self.matrix.shape[-1]

	@property
	def mag_bin_edges(self):
		return self.bin_edges[0]

	@property
	def mag_bin_widths(self):
		mag_bin_edges = self.mag_bin_edges
		return np.diff(mag_bin_edges)

	@property
	def mag_bin_width(self):
		mag_bin_edges = self.mag_bin_edges
		return mag_bin_edges[1] - mag_bin_edges[0]

	@property
	def mag_bin_centers(self):
		return self.mag_bin_edges[:-1] + self.mag_bin_widths / 2

	@property
	def dist_bin_edges(self):
		return self.bin_edges[1]

	@property
	def dist_bin_widths(self):
		dist_bin_edges = self.dist_bin_edges
		return np.diff(dist_bin_edges)

	@property
	def dist_bin_width(self):
		dist_bin_edges = self.dist_bin_edges
		return dist_bin_edges[1] - dist_bin_edges[0]

	@property
	def dist_bin_centers(self):
		return self.dist_bin_edges[:-1] + self.dist_bin_widths / 2

	@property
	def lon_bin_edges(self):
		return self.bin_edges[2]

	@property
	def lon_bin_widths(self):
		lon_bin_edges = self.lon_bin_edges
		return np.diff(lon_bin_edges)

	@property
	def lon_bin_width(self):
		lon_bin_edges = self.lon_bin_edges
		return lon_bin_edges[1] - lon_bin_edges[0]

	@property
	def lon_bin_centers(self):
		return self.lon_bin_edges[:-1] + self.lon_bin_widths / 2

	@property
	def lat_bin_edges(self):
		return self.bin_edges[3]

	@property
	def lat_bin_widths(self):
		lat_bin_edges = self.lat_bin_edges
		return np.diff(lat_bin_edges)

	@property
	def lat_bin_width(self):
		lat_bin_edges = self.lat_bin_edges
		return lat_bin_edges[1] - lat_bin_edges[0]

	@property
	def lat_bin_centers(self):
		return self.lat_bin_edges[:-1] + self.lat_bin_widths / 2

	@property
	def eps_bin_edges(self):
		return self.bin_edges[4]

	@property
	def eps_bin_widths(self):
		eps_bin_edges = self.eps_bin_edges
		return np.diff(eps_bin_edges)

	@property
	def eps_bin_width(self):
		eps_bin_edges = self.eps_bin_edges
		return eps_bin_edges[1] - eps_bin_edges[0]

	@property
	def eps_bin_centers(self):
		return self.eps_bin_edges[:-1] + self.eps_bin_widths / 2

	@property
	def trt_bins(self):
		return self.bin_edges[5]

	@property
	def min_mag(self):
		return self.mag_bin_edges[0]

	@property
	def max_mag(self):
		return self.mag_bin_edges[-1]

	@property
	def min_dist(self):
		return self.dist_bin_edges[0]

	@property
	def max_dist(self):
		return self.dist_bin_edges[-1]

	@property
	def min_lon(self):
		return self.lon_bin_edges[0]

	@property
	def max_lon(self):
		return self.lon_bin_edges[-1]

	@property
	def min_lat(self):
		return self.lat_bin_edges[0]

	@property
	def max_lat(self):
		return self.lat_bin_edges[-1]

	@property
	def min_eps(self):
		return self.eps_bin_edges[0]

	@property
	def max_eps(self):
		return self.eps_bin_edges[-1]

	def collapse_axis(self, axis):
		"""
		Collapse a particular axis to a single value

		:param axis:
			Int, axis index
		"""
		self.deagg_matrix = self.deagg_matrix.fold_axis(axis, keepdims=True)
		self.bin_edges[axis] = self.bin_edges[axis][0::len(self.bin_edges[axis])-1]

	def collapse_coords(self):
		"""
		Collapse coordinate axes to single values
		"""
		self.collapse_axis(-3)
		self.collapse_axis(-4)

	def combine_trts(self, trt_dict):
		"""
		Combine different trt's into one.

		:param trt_dict:
			dict, mapping new trt names to lists of old trt names
		"""
		old_trt_bins = self.trt_bins
		new_trt_bins = old_trt_bins[:]
		for new_trt in trt_dict.keys():
			for combined_trt in trt_dict[new_trt]:
				new_trt_bins.remove(combined_trt)
			new_trt_bins.append(new_trt)
		shape = list(self.deagg_matrix.shape)
		shape[-1] = len(new_trt_bins)
		old_deagg_matrix = self.deagg_matrix
		new_deagg_matrix = old_deagg_matrix.__class__(np.zeros(shape,
													dtype=old_deagg_matrix.dtype))
		for i, trt in enumerate(new_trt_bins):
			if trt in trt_dict.keys():
				for combined_trt in trt_dict[trt]:
					idx = old_trt_bins.index(combined_trt)
					new_deagg_matrix[..., i] += old_deagg_matrix[..., idx]
			else:
				idx = old_trt_bins.index(trt)
				new_deagg_matrix[..., i] = old_deagg_matrix[..., idx]

		self.deagg_matrix = new_deagg_matrix
		self.bin_edges[-1] = new_trt_bins

	def sort_trts(self):
		"""
		Adjust deaggregation matrix such that trt's are ordered
		alphabetically
		"""
		idxs = np.argsort(self.trt_bins)
		self.deagg_matrix = self.deagg_matrix[..., idxs]
		self.bin_edges[-1] = sorted(self.trt_bins)

	def get_axis_bin_edges(self, axis):
		"""
		Return bin edges for given axis.

		:param axis:
			Int, axis index
		"""
		return self.bin_edges[axis]

	def get_axis_bin_widths(self, axis):
		"""
		Return bin widths for given axis.

		:param axis:
			Int, axis index
		"""
		if axis == 0:
			return self.mag_bin_widths
		elif axis == 1:
			return self.dist_bin_widths
		elif axis == 2:
			return self.eps_bin_widths
		elif axis == 3:
			return self.lon_bin_widths
		elif axis == 4:
			return self.lat_bin_widths

	def get_axis_bin_centers(self, axis):
		"""
		Return bin centers for given axis.

		:param axis:
			Int, axis index
		"""
		if axis == 0:
			return self.mag_bin_centers
		elif axis == 1:
			return self.dist_bin_centers
		elif axis == 2:
			return self.eps_bin_centers
		elif axis == 3:
			return self.lon_bin_centers
		elif axis == 4:
			return self.lat_bin_centers

	def get_mag_pmf(self):
		"""
		Fold full deaggregation matrix to magnitude PMF.

		:returns:
			1D array, a histogram representing magnitude PMF.
		"""
		return self.deagg_matrix.fold_axes([-5,-4,-3,-2,-1])

	def get_dist_pmf(self):
		"""
		Fold full deaggregation matrix to distance PMF.

		:returns:
			1D array, a histogram representing distance PMF.
		"""
		return self.deagg_matrix.fold_axes([-6,-4,-3,-2,-1])

	def get_eps_pmf(self):
		"""
		Fold full deaggregation matrix to epsilon PMF.

		:returns:
			1D array, a histogram representing epsilon PMF.
		"""
		return self.deagg_matrix.fold_axes([-6,-5,-4,-3,-1])

	def get_trt_pmf(self):
		"""
		Fold full deaggregation matrix to tectonic region type PMF.

		:returns:
			1D array, a histogram representing tectonic region type PMF.
		"""
		return self.deagg_matrix.fold_axes([-6,-5,-4,-3,-2])

	def get_mag_dist_pmf(self):
		"""
		Fold full deaggregation matrix to magnitude / distance PMF.

		:returns:
			2D array, first dimension represents magnitude histogram bins,
			second one -- distance histogram bins.
		"""
		return self.deagg_matrix.fold_axes([-4,-3,-2,-1])

	def get_mag_dist_eps_pmf(self):
		"""
		Fold full deaggregation matrix to magnitude / distance /epsilon PMF.

		:returns:
			3D array, first dimension represents magnitude histogram bins,
			second one -- distance histogram bins, third one -- epsilon
			histogram bins.
		"""
		return self.deagg_matrix.fold_axes([-4,-3,-1])

	def get_lon_lat_pmf(self):
		"""
		Fold full deaggregation matrix to longitude / latitude PMF.

		:returns:
			2D array, first dimension represents longitude histogram bins,
			second one -- latitude histogram bins.
		"""
		return self.deagg_matrix.fold_axes([-6,-5,-2,-1])

	def get_mag_lon_lat_pmf(self):
		"""
		Fold full deaggregation matrix to magnitude / longitude / latitude PMF.

		:returns:
			3D array, first dimension represents magnitude histogram bins,
			second one -- longitude histogram bins, third one -- latitude
			histogram bins.
		"""
		return self.deagg_matrix.fold_axes([-5,-2,-1])

	def get_lon_lat_trt_pmf(self):
		"""
		Fold full deaggregation matrix to longitude / latitude / trt PMF.

		:returns:
			3D array, first dimension represents longitude histogram bins,
			second one -- latitude histogram bins, third one -- trt histogram bins.
		"""
		return self.deagg_matrix.fold_axes([-6,-5,-1])

	def get_fractional_contribution_matrix(self):
		"""
		Convert deaggregation values to fractional contributions.

		:return:
			ndarray with same dimensions as self.deagg_matrix
		"""
		return self.deagg_matrix.to_fractional_contribution_matrix()
