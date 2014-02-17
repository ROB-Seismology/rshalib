import numpy as np

import openquake.hazardlib.calc.disagg as disagg

from plot import plot_deaggregation
from hazard_curve import Poisson, HazardCurve


class DeaggMatrix(np.ndarray):
	"""
	Base class for deaggregation matrix, subclassed from numpy ndarray

	:param data:
		ndarray, n-d float array containing exceedance rates or
		probabilities of exceedance
	"""
	def __new__(cls, data):
		obj = np.asarray(data).view(cls)
		return obj

	def __array_finalize__(self, obj):
		if obj is None:
			return

	@property
	def matrix(self):
		return np.asarray(self)

	@property
	def num_axes(self):
		return len(self.shape)


class ExceedanceRateMatrix(DeaggMatrix):
	"""
	Class representing a deaggregation matrix containing exceedance rates

	:param data:
		ndarray, n-d float array containing exceedance rates
	"""
	def get_total_exceedance_rate(self, timespan=None):
		"""
		Return total exceedance rate

		:param timespan:
			Float, time span in Poisson formula (default: None).
			This parameter is supported for compatibility with a similar
			method in ProbabilityMatrix, but ignored.
		"""
		return np.sum(self.matrix)

	def get_total_probability(self, timespan):
		"""
		Return total probability

		:param timespan:
			Float, time span in Poisson formula.
		"""
		return Poisson(life_time=timespan, return_period=1./self.get_total_exceedance_rate())

	def to_probability_matrix(self, timespan):
		"""
		Convert to probability matrix

		:param timespan:
			Float, time span in Poisson formula.

		:return:
			instance of :class:`ProbabilityMatrix`
		"""
		return ProbabilityMatrix(Poisson(life_time=timespan, return_period=1./self.matrix))

	def to_exceedance_rate_matrix(self, timespan=None):
		"""
		Convert to exceedance-rate matrix

		:param timespan:
			Float, time span in Poisson formula (default: None).
			This parameter is supported for compatibility with a similar
			method in ProbabilityMatrix, but ignored.

		:return:
			instance of :class:`ExceedanceRateMatrix`
		"""
		return ExceedanceRateMatrix(self.matrix)

	def to_normalized_matrix(self):
		"""
		Convert to a normalized matrix, containing percentual contribution
		to hazard

		:return:
			ndarray
		"""
		return self.matrix / self.get_total_exceedance_rate()

	def fold_axis(self, axis):
		"""
		Fold matrix along one axis.
		Total exceedance rate in given axis is computed as sum of all
		exceedance rates.

		:param axis:
			Int, index of axis to fold

		:return:
			instance of :class:`ExceedanceRateMatrix`
		"""
		if axis < 0:
			axis = self.num_axes + axis
		return self.sum(axis=axis)

	def fold_axes(self, axes):
		"""
		Fold matrix along multiple axes

		:param axes:
			List of integers, indexes of axes to fold

		:return:
			instance of :class:`ExceedanceRateMatrix`
		"""
		matrix = self
		axes = [{True: self.num_axes + axis, False: axis}[axis < 0] for axis in axes]
		for axis in sorted(axes)[::-1]:
			matrix = matrix.fold_axis(axis)
		return matrix


class ProbabilityMatrix(DeaggMatrix):
	"""
	Class representing a deaggregation matrix containing probabilities of exceedance

	:param data:
		ndarray, n-d float array containing exceedance rates
	"""
	#def __new__(cls, data, timespan):
	#	obj = DeaggMatrix.__new__(cls, data)
	#	obj.timespan = timespan
	#	return obj

	def get_total_exceedance_rate(self, timespan):
		"""
		Return total exceedance rate

		:param timespan:
			Float, time span in Poisson formula.
		"""
		return 1. / Poisson(life_time=timespan, prob=self.get_total_probability())

	def get_total_probability(self, timespan=None):
		"""
		Return total probability

		:param timespan:
			Float, time span in Poisson formula (default: None).
			This parameter is supported for compatibility with a similar
			method in ExceedanceRateMatrix, but ignored.
		"""
		return 1 - np.prod(1 - self.matrix)

	def to_probability_matrix(self, timespan=None):
		"""
		Convert to probability matrix

		:param timespan:
			Float, time span in Poisson formula.
			This parameter is supported for compatibility with a similar
			method in ExceedanceRateMatrix, but ignored.

		:return:
			instance of :class:`ProbabilityMatrix`
		"""
		return ProbabilityMatrix(self.matrix)

	def to_exceedance_rate_matrix(self, timespan):
		"""
		Convert to exceedance-rate matrix

		:param timespan:
			Float, time span in Poisson formula.

		:return:
			instance of :class:`ExceedanceRateMatrix`
		"""
		return ExceedanceRateMatrix(1. / Poisson(life_time=timespan, prob=self.matrix))

	def to_normalized_matrix(self):
		"""
		Convert to a normalized matrix, containing percentual contribution
		to hazard

		:return:
			ndarray
		"""
		ln_non_exceedance_probs = np.log(1. - self.matrix)
		return ln_non_exceedance_probs / np.sum(ln_non_exceedance_probs)

	def fold_axis(self, axis):
		"""
		Fold matrix along one axis.
		Total probability in given axis is computed as product of all
		non-exceedance probabilities.

		:param axis:
			Int, index of axis to fold

		:return:
			instance of :class:`ProbabilityMatrix`
		"""
		if axis < 0:
			axis = self.num_axes + axis
		return 1 - np.prod(1 - self, axis=axis)

	def fold_axes(self, axes):
		"""
		Fold matrix along multiple axes

		:param axes:
			List of integers, indexes of axes to fold

		:return:
			instance of :class:`ProbabilityMatrix`
		"""
		matrix = self
		axes = [{True: self.num_axes + axis, False: axis}[axis < 0] for axis in axes]
		for axis in sorted(axes)[::-1]:
			matrix = matrix.fold_axis(axis)
		return matrix


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
		Float, time span in Poisson formula.
	"""
	def __init__(self, bin_edges, deagg_matrix, timespan):
		assert len(bin_edges) == 6, "bin_edges must contain 6 elements!"
		self.bin_edges = bin_edges
		assert deagg_matrix.shape[-6] == max(1, len(bin_edges[0]) - 1), "Number of magnitude bins not in accordance with specified bin edges!"
		assert deagg_matrix.shape[-5] == max(1, len(bin_edges[1]) - 1), "Number of distance bins not in accordance with specified bin edges!"
		assert deagg_matrix.shape[-4] == max(1, len(bin_edges[2]) - 1), "Number of longitude bins not in accordance with specified bin edges!"
		assert deagg_matrix.shape[-3] == max(1, len(bin_edges[3]) - 1), "Number of latitude bins not in accordance with specified bin edges!"
		assert deagg_matrix.shape[-2] == max(1, len(bin_edges[4]) - 1), "Number of epsilon bins not in accordance with specified bin edges!"
		assert deagg_matrix.shape[-1] == max(1, len(bin_edges[5])), "Number of tectonic-region-type bins not in accordance with specified bin edges!"
		assert isinstance(deagg_matrix, (ExceedanceRateMatrix, ProbabilityMatrix)), "deagg_matrix must be instance of ExceedanceRateMatrix or ProbabilityMatrix!"
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
		return mag_bin_edges[1:] - mag_bin_edges[:-1]

	@property
	def mag_bin_centers(self):
		return self.mag_bin_edges[:-1] + self.mag_bin_widths / 2

	@property
	def dist_bin_edges(self):
		return self.bin_edges[1]

	@property
	def dist_bin_widths(self):
		dist_bin_edges = self.dist_bin_edges
		return dist_bin_edges[1:] - dist_bin_edges[:-1]

	@property
	def dist_bin_centers(self):
		return self.dist_bin_edges[:-1] + self.dist_bin_widths / 2

	@property
	def lon_bin_edges(self):
		return self.bin_edges[2]

	@property
	def lon_bin_widths(self):
		lon_bin_edges = self.lon_bin_edges
		return lon_bin_edges[1:] - lon_bin_edges[:-1]

	@property
	def lon_bin_centers(self):
		return self.lon_bin_edges[:-1] + self.lon_bin_widths / 2

	@property
	def lat_bin_edges(self):
		return self.bin_edges[3]

	@property
	def lat_bin_widths(self):
		lat_bin_edges = self.lat_bin_edges
		return lat_bin_edges[1:] - lat_bin_edges[:-1]

	@property
	def lat_bin_centers(self):
		return self.lat_bin_edges[:-1] + self.lat_bin_widths / 2

	@property
	def eps_bin_edges(self):
		return self.bin_edges[4]

	@property
	def eps_bin_widths(self):
		eps_bin_edges = self.eps_bin_edges
		return eps_bin_edges[1:] - eps_bin_edges[:-1]

	@property
	def eps_bin_centers(self):
		return self.eps_bin_edges[:-1] + self.eps_bin_widths / 2

	@property
	def trt_bins(self):
		return self.bin_edges[5]

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

	def to_percent_contribution(self):
		"""
		Convert deaggregation values to percent contribution.

		:return:
			ndarray with same dimensions as self.deagg_matrix
		"""
		return self.deagg_matrix.to_normalized_matrix()


class DeaggregationSlice(DeaggBase):
	"""
	Class representing a full deaggregation result for a single intensity level,
	as computed by nhlib.

	Deaggregation values represent conditional probability distribution or
	exceedance rate as a function of:
		- rupture magnitude,
		- joyner-boore distance from rupture surface to site,
		- longitude and latitude of surface projection of rupture closest point
		  to site,
		- epsilon: number of standard deviations by which an intensity measure
		  level deviates from the median value predicted by a gsim, given
		  the rupture parameters.
		- rupture tectonic region type
	given the event that an intensity measure type ``imt`` exceeds an intensity
	measure level ``iml`` at a geographical location ``site``.

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
		6-D array containing deaggregation values, with dimensions corresponding to:
			- magnitude bins
			- distance bins
			- longitude bins
			- latitude bins
			- epsilon bins
			- tectonic-region-type bins

	:param site:
		instance of :class:`SHASite`: site where hazard was computed

	:param imt:
		str, intensity measure type

	:param iml:
		float, intensity level

	:period:
		float, spectral period

	:param timespan:
		Float, time span in Poisson formula.
	"""
	def __init__(self, bin_edges, deagg_matrix, site, imt, iml, period, timespan):
		DeaggBase.__init__(self, bin_edges, deagg_matrix, timespan)
		self.site = site
		self.imt = imt
		self.iml = iml
		self.period = period

	def to_exceedance_rate(self):
		"""
		Convert deaggregation slice to exceedance rate

		:return:
			instance of :class:`DeaggregationSlice`
		"""
		deagg_matrix = self.deagg_matrix.to_exceedance_rate_matrix(self.timespan)
		return DeaggregationSlice(self.bin_edges, deagg_matrix, self.site, self.imt, self.iml, self.period, self.timespan)

	def to_probability(self):
		"""
		Convert deaggregation slice to probability of exceecance

		:return:
			ndarray, 6-D
		"""
		deagg_matrix = self.deagg_matrix.to_probability_matrix(self.timespan)
		return DeaggregationSlice(self.bin_edges, deagg_matrix, self.site, self.imt, self.iml, self.period, self.timespan)

	def get_total_probability(self):
		"""
		Return total probability of exceedance
		"""
		return self.deagg_matrix.get_total_probability(self.timespan)

	def get_total_exceedance_rate(self):
		"""
		Return total exceedance rate
		"""
		return self.deagg_matrix.get_total_exceedance_rate(self.timespan)

	def get_return_period(self):
		"""
		Return return period corresponding to total probability
		"""
		return 1./self.get_total_exceedance_rate()

	def plot_mag_dist_pmf(self, title="", fig_filespec=None):
		"""
		Plot magnitude / distance PMF.
		"""
		mag_dist_pmf = self.get_mag_dist_pmf().to_normalized_matrix()
		if self.neps > 1:
			eps_pmf = self.get_eps_pmf().to_normalized_matrix()
		else:
			eps_pmf = None
		eps_bin_edges = self.eps_bin_edges[1:]
		return_period = self.get_return_period()
		plot_deaggregation(mag_dist_pmf, self.mag_bin_edges, self.dist_bin_edges, return_period, eps_values=eps_pmf, eps_bin_edges=eps_bin_edges, mr_style="2D", site_name=self.site.name, struc_period=self.period, title_comment=title, fig_filespec=fig_filespec)

	def get_modal_eq_scenario(self):
		"""
		Determine modal earthquake scenario (having largest contribution)

		:return:
			(mag, dist) tuple
		"""
		mag_dist_pmf = self.get_mag_dist_pmf()
		mag_index, dist_index = np.unravel_index(mag_dist_pmf.argmax(), mag_dist_pmf.shape)
		return (self.mag_bin_centers[mag_index], self.dist_bin_centers[dist_index])

	def get_mean_eq_scenario(self):
		"""
		Determine mean earthquake scenario

		:return:
			(mag, dist) tuple
		"""
		return (self.get_mean_magnitude(), self.get_mean_distance())

	def get_mean_magnitude(self):
		"""
		Return mean magnitude according to formula in DOE-STD-1023-95,
		appendix A, page 5.
		"""
		deagg_matrix = self.deagg_matrix.to_exceedance_rate_matrix(self.timespan)
		deagg_matrix = deagg_matrix.fold_axes([1,2,3,4,5])
		return float(np.sum(self.mag_bin_centers * deagg_matrix)) / self.get_total_exceedance_rate()

	def get_mean_distance(self):
		"""
		Return mean distance according to formula in DOE-STD-1023-95,
		appendix A, page 6.
		"""
		deagg_matrix = self.deagg_matrix.to_exceedance_rate_matrix(self.timespan)
		deagg_matrix = deagg_matrix.fold_axes([0,2,3,4,5])
		return np.exp(float(np.sum(np.log(self.dist_bin_centers) * deagg_matrix)) / self.get_total_exceedance_rate())

	def get_contribution_above_threshold(self, threshold, axis):
		"""
		Determine contribution of bins above a threshold value.

		:param threshold:
			Float, threshold value in given axis
		:param axis:
			Int, index of axis

		:return:
			Float, percent contribution
		"""
		bin_indexes = np.where(self.get_axis_bin_centers(axis) > threshold)
		deagg_matrix = self.deagg_matrix.to_exceedance_rate_matrix(self.timespan)
		axes_to_fold = np.arange(6)
		axes_to_fold = np.delete(axes_to_fold, axis)
		deagg_matrix = deagg_matrix.fold_axes(axes_to_fold)
		return float(np.sum(deagg_matrix[bin_indexes])) / self.get_total_exceedance_rate()

	def get_contribution_above_magnitude(self, mag):
		"""
		Determine contribution of magnitudes above a threshold magnitude.

		:param mag:
			Float, threshold magnitude

		:return:
			Float, percent contribution
		"""
		return self.get_contribution_above_threshold(mag, axis=0)

	def get_contribution_above_distance(self, dist):
		"""
		Determine contribution of distances above a threshold distance.

		:param dist:
			Float, threshold distance

		:return:
			Float, percent contribution
		"""
		return self.get_contribution_above_threshold(dist, axis=1)

	def get_trt_slice(self, trt):
		"""
		Obtain deaggregation slice for a given trt

		:param trt:
			str, tectonic region type

		:return:
			instance of :class:`DeaggregationSlice`
		"""
		trt_idx = self.trt_bins.index(trt)
		bin_edges = (self.mag_bin_edges, self.dist_bin_edges, self.lon_bin_edges, self.lat_bin_edges, self.eps_bin_edges, [trt])
		deagg_matrix = self.deagg_matrix[:,:,:,:,:,trt_idx:trt_idx+1]
		return DeaggregationSlice(bin_edges, deagg_matrix, self.site, self.imt, self.iml, self.period, self.timespan)

	def rebin(self, new_bin_edges, axis=0):
		"""
		Rebin deaggregation slice along a given axis

		:param new_bin_edges:
			array-like, new bin edges
		:param axis:
			Int, axis index

		:return:
			instance of :class:`DeaggregationSlice`
		"""
		#TODO: renormalization!
		from ..utils import interpolate
		rebinned_shape = list(self.deagg_matrix.shape)
		rebinned_shape[axis] = len(new_bin_edges) - 1
		rebinned_deagg = np.zeros(rebinned_shape, 'd')
		other_axes = [i for i in range(6) if not i == axis]
		old_bin_edges = self.bin_edges[axis]
		for i in range(rebinned_shape[other_axes[0]]):
			for j in range(rebinned_shape[other_axes[1]]):
				for k in range(rebinned_shape[other_axes[2]]):
					for l in range(rebinned_shape[other_axes[3]]):
						for m in range(rebinned_shape[other_axes[4]]):
							## Construct list with axis indexes
							ax = [0] * 6
							ax[other_axes[0]] = i
							ax[other_axes[1]] = j
							ax[other_axes[2]] = k
							ax[other_axes[3]] = l
							ax[other_axes[4]] = m
							ax[axis] = slice(None)
							interpol_deagg = interpolate(old_bin_edges[:-1], self.deagg_matrix[ax[0], ax[1], ax[2], ax[3], ax[4], ax[5]], new_bin_edges[:-1])
							interpol_deagg = self.deagg_matrix.__class__(interpol_deagg)
							## Renormalize
							#total = self.deagg_matrix[ax[0], ax[1], ax[2], ax[3], ax[4], ax[5]].get_total_exceedance_rate(self.timespan)
							#rebinned_total = interpol_deagg.get_total_exceedance_rate(self.timespan)
							#if rebinned_total != 0:
							#	interpol_deagg = interpol_deagg * (total / rebinned_total)
							rebinned_deagg[ax[0], ax[1], ax[2], ax[3], ax[4], ax[5]] = interpol_deagg
		rebinned_deagg = self.deagg_matrix.__class__(rebinned_deagg)
		bin_edges = list(self.bin_edges)
		bin_edges[axis] = new_bin_edges
		bin_edges = tuple(bin_edges)
		return DeaggregationSlice(bin_edges, rebinned_deagg, self.site, self.imt, self.iml, self.period, self.timespan)

	def rebin_magnitudes(self, mag_bin_edges):
		"""
		Rebin magnitude bins

		:param mag_bin_edges:
			array-like, new magnitude bin edges
		:param axis:
			Int, axis index

		:return:
			instance of :class:`DeaggregationSlice`
		"""
		return self.rebin(mag_bin_edges, axis=0)

	def rebin_distances(self, dist_bin_edges):
		"""
		Rebin distance bins

		:param dist_bin_edges:
			array-like, new distance bin edges
		:param axis:
			Int, axis index

		:return:
			instance of :class:`DeaggregationSlice`
		"""
		return self.rebin(dist_bin_edges, axis=1)

	def write_nrml(self, nrml_filespec, sourceModelTreePath=None, gsimTreePath=None):
		"""
		"""
		from ..openquake.IO import write_disaggregation_slice
		poe = self.deagg_matrix.get_total_probability(timespan=self.timespan)
		matrix = self.deagg_matrix.to_probability_matrix(timespan=self.timespan)
		write_disaggregation_slice(self.site, self.imt, self.period, self.iml, poe, self.timespan, self.bin_edges, matrix, nrml_filespec, sourceModelTreePath, gsimTreePath)


class DeaggregationCurve(DeaggBase):
	"""
	Class representing a full deaggregation result for a range of intensities

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
		7-D array containing deaggregation values, with dimensions corresponding to:
			- intensity levels
			- magnitude bins
			- distance bins
			- longitude bins
			- latitude bins
			- epsilon bins
			- tectonic-region-type bins

	:param site:
		instance of :class:`SHASite`: site where hazard was computed

	:param imt:
		str, intensity measure type

	:param intensities:
		float array, intensity levels

	:period:
		float, spectral period

	:param timespan:
		Float, time span in Poisson formula.
	"""
	def __init__(self, bin_edges, deagg_matrix, site, imt, intensities, period, timespan):
		self.site = site
		self.imt = imt
		self.period = period

		## Make sure intensities are ordered from small to large
		if intensities[0] > intensities[-1]:
			DeaggBase.__init__(self, bin_edges, deagg_matrix[::-1], timespan)
			self.intensities = intensities[::-1]
		else:
			DeaggBase.__init__(self, bin_edges, deagg_matrix, timespan)
			self.intensities = intensities

	def __len__(self):
		return len(self.intensities)

	def __iter__(self):
		for iml_index in range(len(self.intensities)):
			yield self.get_slice(iml_index=iml_index)

	def __getitem__(self, iml_index):
		return self.get_slice(iml_index=iml_index)

	def get_intensity_bin_centers(self):
		"""
		Return center values of intensity bins
		Note: in contrast to deaggregation bins, the number of intensity bin
		centers returned is equal to the number of intensities.
		"""
		imls = self.intensities
		log_imls = np.log(imls)
		log_iml_widths = log_imls[1:] - log_imls[:-1]
		intensity_bin_centers = np.exp(log_imls[:-1] + log_iml_widths / 2.)
		intensity_bin_centers = np.append(intensity_bin_centers, np.exp(log_imls[-1] + log_iml_widths[-1] / 2))
		return intensity_bin_centers

	def get_slice(self, iml=None, iml_index=None):
		"""
		Get deaggregation slice for a particular intensity level.

		:param iml:
			Float, intensity level (default: None)
		:param iml_index:
			Int, intensity index (default: None)

		:return:
			instance of :class:`DeaggregationSlice`
		"""
		if iml is not None:
			iml_index = np.argmin(np.abs(self.intensities - iml))
		else:
			iml = self.intensities[iml_index]
		matrix = self.deagg_matrix[iml_index]
		return DeaggregationSlice(self.bin_edges, matrix, self.site, self.imt, iml, self.period, self.timespan)

	def get_hazard_curve(self):
		"""
		Get hazard curve corresponding to total exceedance rate or
		probability for each intensity level.

		:return:
			instance of :class:`HazardCurve`
		"""
		exceedance_rates = []
		for slice in self:
			exceedance_rates.append(slice.get_total_exceedance_rate())
		model_name = ""
		filespec = ""
		return HazardCurve(model_name, filespec, self.site, self.period, self.imt, self.intensities, "g", self.timespan, poes=None, exceedance_rates=exceedance_rates, variances=None, site_name="")

	def get_occurrence_rates(self):
		"""
		Calculate rate of occurrence for each intensity interval (difference
		between exceedance rates of intensity below and above)
		Note: the number of occurrence rates returned is equal to the number
		of exceedance rates (or intensities).

		:return:
			instance of :class:`ExceedanceRateMatrix`
		"""
		exceedance_rates = self.deagg_matrix.to_exceedance_rate_matrix(self.timespan)
		occurrence_rates = exceedance_rates[:-1] - exceedance_rates[1:]
		occurrence_rates = np.append(occurrence_rates, exceedance_rates[-1:], axis=0)
		return ExceedanceRateMatrix(occurrence_rates)

	@classmethod
	def from_deaggregation_slices(self, deagg_slices):
		"""
		Construct new instance of :class:`DeaggregationCurve` from a number of
		deaggregation slices.

		:param deagg_slices:
			list of instances of :class:`DeaggregationSlice`
		"""
		imls = np.array([ds.iml for ds in deagg_slices])
		iml_indexes = np.argsort(imls)
		imls = imls[iml_indexes]
		deagg_matrixes = [deagg_slices[iml_index].deagg_matrix[np.newaxis] for iml_index in iml_indexes]
		deagg_matrix = np.concatenate(deagg_matrixes)
		deagg_matrix = deagg_slices[0].deagg_matrix.__class__(deagg_matrix)
		# TODO: check that bin_edges etc. are identical
		bin_edges = deagg_slices[0].bin_edges
		site = deagg_slices[0].site
		imt = deagg_slices[0].imt
		period = deagg_slices[0].period
		timespan = deagg_slices[0].timespan
		return DeaggregationCurve(bin_edges, deagg_matrix, site, imt, imls, period, timespan)

	def filter_cav(self, vs30, CAVmin=0.16, gmpe_name=""):
		"""
		Apply CAV filtering to deaggregation curve, according to EPRI report
		by Abrahamson et al. (2006).

		:param vs30:
			Float, shear-wave velocity in the top 30 m (m/s)
		:param CAVmin:
			Float, minimum CAV value in g.s (default: 0.16)
		:param gmpe_name:
			Str, name of GMPE (needed when imt is spectral)

		:return:
			instance of :class:`DeaggregationCurve`
		"""
		import scipy.stats
		from ..cav import calc_ln_SA_given_PGA, calc_ln_PGA_given_SA, calc_CAV_exceedance_prob

		num_intensities = len(self.intensities)

		## Reduce to magnitude-distance pmf, and store in a new DeaggregationCurve object
		deagg_matrix = self.get_mag_dist_pmf()[:,:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
		bin_edges = (self.mag_bin_edges, self.dist_bin_edges, np.array([0]), np.array([0]), np.array([0]), np.array([0]))
		CAV_deagg_curve = DeaggregationCurve(bin_edges, deagg_matrix, self.site, self.imt, self.intensities, self.period, self.timespan)

		intensities = self.get_intensity_bin_centers()

		if self.imt == "PGA":
			## Calculate CAV exceedance probabilities corresponding to PGA
			num_intensities = len(self.intensities)
			CAV_exceedance_probs = np.zeros((num_intensities, self.nmags), 'd')
			for k in range(num_intensities):
				zk = intensities[k]
				CAV_exceedance_probs[k] = calc_CAV_exceedance_prob(zk, self.mag_bin_centers, vs30, CAVmin)
			CAV_exceedance_probs = CAV_exceedance_probs[:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis]

		elif self.imt == "SA":
			from ..gsim import gmpe as gmpe_module
			gmpe = getattr(gmpe_module, gmpe_name)()

			## Compute PGA given SA
			"""
			pga_given_sa = np.zeros((num_intensities, self.nmags, self.ndists), 'd')
			prob_pga_given_sa = np.ones((num_intensities, self.nmags, self.ndists), 'd')
			T = self.period
			sa = intensities
			ln_sa = np.log(sa)
			for k in range(num_intensities):
				for r in range(self.ndists):
					R = self.dist_bin_centers[r]
					M = self.mag_bin_centers
					ln_pga_given_sa, sigma_ln_pga_given_sa = calc_ln_PGA_given_SA(sa[k], M, R, T, vs30, gmpe_name)
					# Note: ln_sa in line below cannot be correct
					epsilon_pga_given_sa = (ln_sa[k] - ln_pga_given_sa) / sigma_ln_pga_given_sa
					prob_pga_given_sa[k,:,r] = scipy.stats.norm.pdf(epsilon_pga_given_sa)
					#prob_pga_given_sa[k,:,r] = 1.0 - scipy.stats.norm.cdf(epsilon_pga_given_sa)
					pga_given_sa[k,:,r] = np.exp(ln_pga_given_sa)
			"""

			#pga = np.zeros((num_intensities, self.nmags, self.ndists), 'd')
			pga_given_sa = np.zeros((num_intensities, self.nmags, self.ndists), 'd')
			prob_sa_given_pga = np.ones((num_intensities, self.nmags, self.ndists), 'd')
			T = self.period
			sa = intensities
			ln_sa = np.log(sa)
			for k in range(num_intensities):
				for r in range(self.ndists):
					R = self.dist_bin_centers[r]
					M = self.mag_bin_centers
					## Compute epsilon
					#epsilon = gmpe.get_epsilon(sa[k], M, R, imt=self.imt, T=T, vs30=vs30)
					## Compute PGA corresponding to epsilon
					#pga[k,:,r] = gmpe(M, R, imt="PGA", T=0, epsilon=epsilon, vs30=vs30)
					ln_pga_given_sa, sigma_ln_pga_given_sa = calc_ln_PGA_given_SA(sa[k], M, R, T, vs30, gmpe_name)
					pga_given_sa[k,:,r] = np.exp(ln_pga_given_sa)
					## Compute SA given PGA
					## This is ignored at the moment
					#ln_sa_given_pga, sigma_ln_sa_given_pga = calc_ln_SA_given_PGA(pga[k,:,r], M, R, T, vs30, gmpe_name)
					## Compute epsilon of SA given PGA
					#epsilon_sa_given_pga = (ln_sa[k] - ln_sa_given_pga) / sigma_ln_sa_given_pga
					## Compute probability that SA is equal to (or greater than?) sa given PGA
					#prob_sa_given_pga[k,:,r] = scipy.stats.norm.pdf(epsilon_sa_given_pga)
					#prob_sa_given_pga[k,:,r] = 1.0 - scipy.stats.norm.cdf(epsilon_sa_given_pga)

			## Calculate CAV exceedance probabilities corresponding to PGA
			CAV_exceedance_probs = np.zeros((num_intensities, self.nmags, self.ndists), 'd')
			for k in range(num_intensities):
				for r in range(self.ndists):
					zk = pga_given_sa[k,:,r]
					#zk = pga[k,:,r]
					CAV_exceedance_probs[k,:,r] = calc_CAV_exceedance_prob(zk, self.mag_bin_centers, vs30, CAVmin)
			#CAV_exceedance_probs *= prob_sa_given_pga
			CAV_exceedance_probs = CAV_exceedance_probs[:,:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]

			#prob_sa_given_pga = prob_sa_given_pga[:,:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]

		## Calculate filtered occurrence rates
		deagg_occurrence_rates = CAV_deagg_curve.get_occurrence_rates()
		CAV_deagg_occurrence_rates = deagg_occurrence_rates * CAV_exceedance_probs

		## Convert occurrence rates back to exceedance rates
		CAV_deagg_exceedance_rates = np.add.accumulate(CAV_deagg_occurrence_rates[::-1], axis=0)[::-1]
		#if self.imt == "SA":
		#	CAV_deagg_exceedance_rates *= prob_sa_given_pga
		CAV_deagg_curve.deagg_matrix = ExceedanceRateMatrix(CAV_deagg_exceedance_rates)

		return CAV_deagg_curve


class SpectralDeaggregationCurve(DeaggBase):
	"""
	Class representing a full deaggregation result for a range of intensities
	and a range of periods

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
		8-D array containing deaggregation values, with dimensions corresponding to:
			- spectral periods
			- intensity levels
			- magnitude bins
			- distance bins
			- longitude bins
			- latitude bins
			- epsilon bins
			- tectonic-region-type bins

	:param site:
		instance of :class:`SHASite`: site where hazard was computed

	:param imt:
		str, intensity measure type

	:param intensities:
		float array, intensity levels

	:periods:
		float array, spectral periods

	:param timespan:
		Float, time span in Poisson formula.
	"""
	def __init__(self, bin_edges, deagg_matrix, site, imt, intensities, periods, timespan):
		self.site = site
		self.imt = imt
		self.periods = np.array(periods)

		## Make sure intensities are ordered from small to large
		if intensities[0,0] > intensities[0,-1]:
			DeaggBase.__init__(self, bin_edges, deagg_matrix[:,::-1], timespan)
			self.intensities = intensities[:,::-1]
		else:
			DeaggBase.__init__(self, bin_edges, deagg_matrix, timespan)
			self.intensities = intensities

	def __iter__(self):
		for period_index in range(len(self.periods)):
			yield self.get_curve(period_index=period_index)

	def __len__(self):
		return len(self.periods)

	def __getitem__(self, period_index):
		return self.get_curve(period_index=period_index)

	def get_curve(self, period=None, period_index=None):
		"""
		Get deaggregation curve for a particular spectral period.

		:param period:
			Float, spectral period (default: None)
		:param iml_index:
			Int, period index (default: None)

		:return:
			instance of :class:`DeaggregationCurve`
		"""
		if period is not None:
			period_index = np.argmin(np.abs(self.periods - period))
		else:
			period = self.periods[period_index]
		matrix = self.deagg_matrix[period_index]
		intensities = self.intensities[period_index]
		return DeaggregationCurve(self.bin_edges, matrix, self.site, self.imt, intensities, period, self.timespan)

	@classmethod
	def from_deaggregation_curves(self, deagg_curves):
		"""
		Construct new instance of :class:`SpectralDeaggregationCurve` from a
		number of deaggregation curves.

		:param deagg_curves:
			list of instances of :class:`DeaggregationCurve`
		"""
		periods = np.array([dc.period for dc in deagg_curves])
		period_indexes = np.argsort(periods)
		periods = periods[period_indexes]
		deagg_matrixes = [deagg_curves[period_index].deagg_matrix[np.newaxis] for period_index in period_indexes]
		deagg_matrix = np.concatenate(deagg_matrixes)
		deagg_matrix = deagg_curves[0].deagg_matrix.__class__(deagg_matrix)
		intensity_arrays = [deagg_curves[period_index].intensities[np.newaxis,:] for period_index in period_indexes]
		intensities = np.concatenate(intensity_arrays, axis=0)
		# TODO: check that bin_edges etc. are identical
		bin_edges = deagg_curves[0].bin_edges
		site = deagg_curves[0].site
		imt = deagg_curves[0].imt
		timespan = deagg_curves[0].timespan
		return SpectralDeaggregationCurve(bin_edges, deagg_matrix, site, imt, intensities, periods, timespan)

	def filter_cav(self, vs30, CAVmin=0.16, gmpe_name=""):
		"""
		Apply CAV filtering to each spectral deaggregation curve,
		and reconstruct into a spectral deaggregation curve

		:param vs30:
			Float, shear-wave velocity in the top 30 m (m/s)
		:param CAVmin:
			Float, minimum CAV value in g.s (default: 0.16)
		:param gmpe_name:
			Str, name of GMPE (needed when imt is spectral)

		:return:
			instance of :class:`SpectralDeaggregationCurve`
		"""
		CAV_curves = []
		for curve in self:
			CAV_deagg_curve = curve.filter_cav(vs30, CAVmin=CAVmin, gmpe_name=gmpe_name)
			CAV_curves.append(CAV_deagg_curve)
		return SpectralDeaggregationCurve.from_deaggregation_curves(CAV_curves)

	def get_conditional_mean_spectrum(self):
		"""
		Compute conditional mean spectrum as outlined in e.g., Baker (2011)
		"""
		pass

	def write_nrml(self, nrml_filespec, sourceModelTreePath=None, gsimTreePath=None):
		"""
		"""
		from lxml import etree
		from ..nrml import ns
		nrml_file = open(nrml_filespec, "w")
		root = etree.Element("nrml", nsmap=ns.NSMAP)
		sdc_elem = etree.SubElement(root, "spectralDeaggregationCurve")
		if sourceModelTreePath:
			sdc_elem.set("sourceModelTreePath", sourceModelTreePath)
		if gsimTreePath:
			sdc_elem.set("gsimTreePath", gsimTreePath)
		lon, lat = self.site[0], self.site[1]
		sdc_elem.set("lon", str(lon))
		sdc_elem.set("lat", str(lat))
		sdc_elem.set("investigationTime", str(self.timespan))
		mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, trts = self.bin_edges
		sdc_elem.set("magBinEdges", ", ".join(map(str, mag_bins)))
		sdc_elem.set("distBinEdges", ", ".join(map(str, dist_bins)))
		sdc_elem.set("lonBinEdges", ", ".join(map(str, lon_bins)))
		sdc_elem.set("latBinEdges", ", ".join(map(str, lat_bins)))
		sdc_elem.set("epsBinEdges", ", ".join(map(str, eps_bins)))
		sdc_elem.set("tectonicRegionTypes", ", ".join(trts))
		dims = ",".join(map(str, self.matrix.shape[2:]))
		sdc_elem.set("dims", dims)
		for dc in self:
			dc_elem = etree.SubElement(sdc_elem, "deaggregationCurve")
			dc_elem.set("imt", str(dc.imt))
			dc_elem.set("saPeriod", str(dc.period))
			for ds in dc:
				ds_elem = etree.SubElement(dc_elem, "deaggregationSlice")
				ds_elem.set("iml", str(ds.iml))
				poe = ds.deagg_matrix.get_total_probability(timespan=self.timespan)
				ds_elem.set("poE", str(poe))
				matrix = ds.deagg_matrix.to_probability_matrix(timespan=self.timespan)
				for index, value in np.ndenumerate(matrix):
					if not np.allclose(value, 0.):
						index = ",".join(map(str, index))
						value = str(value)
						prob = etree.SubElement(ds_elem, "prob")
						prob.set("index", index)
						prob.set("value", value)
		nrml_file.write(etree.tostring(root, pretty_print=True, xml_declaration=True, encoding="UTF-8"))
		nrml_file.close()
