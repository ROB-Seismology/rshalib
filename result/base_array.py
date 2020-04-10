"""
Base arrays for hazard results

Contain rules for calculating with exceedance rates and exceedance
probabilities:
- exceedance rates are additive
- non-exceedance probabilities are multiplicative
- logs of non-exceedance probabilities are additive
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.stats import mstats

from ..poisson import poisson_conv
from ..utils import interpolate, wquantiles


__all__ = ['is_empty_array', 'as_array',
			'HazardCurveArray', 'ExceedanceRateArray', 'ProbabilityArray',
			'DeaggMatrix', 'ExceedanceRateMatrix', 'ProbabilityMatrix',
			'FractionalContributionMatrix']



def is_empty_array(ar):
	"""
	Determine whether or not a given array is empty, i.e. if:
	- array is None or []
	- all elements in array are None

	:param ar:
		numpy array, list or None

	:return:
		bool
	"""
	if ar is None or len(ar) == 0 or np.all(ar == None):
		return True
	else:
		return False


def as_array(values):
	"""
	Convert values to array if it is not None or already a numpy array
	"""
	if is_empty_array(values):
		values = None
	else:
		values = {True: values,
				False: np.array(values, dtype='d')}[isinstance(values, np.ndarray)]
	return values


## Hazard curve arrays
## Note: ExceedanceRateArray and ProbabilityArray have the same methods
## with the same arguments, so that code using these classes can be agnostic
## which one of the two it is


class HazardCurveArray(np.ndarray):
	"""
	Base class for hazard curve array, subclassed from numpy ndarray

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
	def array(self):
		return np.asarray(self)

	@property
	def num_axes(self):
		return len(self.shape)


class ExceedanceRateArray(HazardCurveArray):
	"""
	Class representing a hazard curve containing exceedance rates

	:param data:
		ndarray, n-d float array containing exceedance rates
	"""
	def to_exceedance_rates(self, timespan=None):
		"""
		Return array of exceedance rates

		:param timespan:
			None, parameter is present to be compatible with the
			same method in :class:`ProbabilityArray`, but is ignored

		:return:
			ndarray
		"""
		return self.array

	def to_exceedance_rate_array(self, timespan=None):
		"""
		Copy to another exceedance-rate array.
		This method is present to be compatible with
		:class:`ProbabilityArray`

		:param timespan:
			None, parameter is present to be compatible with the
			same method in :class:`ProbabilityArray`, but is ignored

		:return:
			instance of :class:`ExceedanceRateArray`
		"""
		return ExceedanceRateArray(self.array.copy())

	def to_probabilities(self, timespan):
		"""
		Compute exceedance probabilities for given time span

		:param timespan:
			float, time span

		:return:
			ndarray
		"""
		return poisson_conv(t=timespan, tau=1./self.array)

	def to_probability_array(self, timespan):
		"""
		Compute exceedance probability array for given time span

		:param timespan:
			float, time span

		:return:
			instance of :class:`ProbabilityArray`
		"""
		return ProbabilityArray(self.to_probabilities(timespan))

	def to_return_periods(self, timespan=None):
		"""
		Convert exceedance rates to return periods

		:param timespan:
			None, parameter is present to be compatible with the
			same method in :class:`ProbabilityArray`, but is ignored

		:return:
			ndarray
		"""
		return 1./self.array

	def mean(self, axis, weights=None):
		"""
		Compute (weighted) mean exceedance rate

		:param axis:
			int, array axis to compute mean for
		:param weights:
			ndarray, optional weights for each element in array axis
			(default: None)

		:return:
			instance of :class:`ExceedanceRateArray`
		"""
		if weights is None:
			return self.__class__(np.mean(self.array, axis=axis))
		else:
			return self.__class__(np.average(self.array, axis=axis,
											weights=weights))

	def mean_and_variance(self, axis, weights=None):
		"""
		Compute the weighted average and variance.

		:param axis:
		:param weights:
			see :meth:`mean`

		:return:
			(mean, variance) tuple of instances of
			:class:`ExceedanceRateArray`
		"""
		mean = self.mean(axis, weights=weights)
		if weights is None:
			variance = np.var(self.array, axis=axis)
		else:
			variance = np.average((self.array - mean)**2, axis=axis,
									weights=weights)
		variance = self.__class__(variance)
		return (mean, variance)

	def scoreatpercentile(self, axis, percentile_levels, weights=None,
						interpol=True):
		"""
		Compute percentile exceedance rates

		:param axis:
			int, array axis to compute percentiles for
		:param percentile_levels:
			list of percentile levels (in the range 0 -100)
		:param weights
			ndarray, optional weights for each element in array axis
			(default: None)
		:param interpol:
			bool, whether or not to interpolate percentile intercept
			in case :param:`weights` is not None
			(default: True)

		:return:
			instance of :class:`ExceedanceRateArray`
		"""
		quantile_levels = np.array(percentile_levels, 'f') / 100.
		if weights is None:
			percentiles = np.apply_along_axis(mstats.mquantiles, axis, self.array,
											quantile_levels)
		else:
			percentiles = np.apply_along_axis(wquantiles, axis, self.array,
											weights, quantile_levels, interpol)
		## Rotate percentile axis to last position
		# TODO: consider first position for percentile axis
		ndim = self.array.ndim
		percentiles = np.rollaxis(percentiles, axis, ndim)
		return self.__class__(percentiles)


class ProbabilityArray(HazardCurveArray):
	"""
	Class representing a hazard curve containing probabilities of
	exceedance

	:param data:
		ndarray, n-d float array containing exceedance probabilities
	"""
	def __add__(self, other):
		"""
		Sum exceedance probabilities
		"""
		assert isinstance(other, ProbabilityArray)
		return ProbabilityArray(1 - ((1 - self.array) * (1 - other.array)))

	def __sub__(self, other):
		"""
		Subtract exceedance probabilities
		"""
		assert isinstance(other, ProbabilityArray)
		return ProbabilityArray(1 - ((1 - self.array) / (1 - other.array)))

	def __mul__(self, other):
		"""
		Multiply exceedance probabilities with scalar or other
		probability array
		"""
		assert (np.isscalar(other) or other.shape == ()
				or isinstance(other, ProbabilityArray))
		#return ProbabilityArray(1 - np.exp(np.log(1 - self.array) * float(number)))
		return ProbabilityArray(1 - (1 - self.array) ** other)

	def __div__(self, other):
		"""
		Divide exceedance probabilities by scalar or other
		probability array
		"""
		assert (np.isscalar(other) or other.shape == ()
				or isinstance(other, ProbabilityArray))
		#return ProbabilityArray(1 - np.exp(np.log(1 - self.array) / float(number)))
		return self.__mul__(1./other)

	def to_exceedance_rates(self, timespan):
		"""
		Compute exceedance rates:

		:param timespan:
			float, time span

		:return:
			ndarray
		"""
		## Ignore division warnings
		np.seterr(divide='ignore', invalid='ignore')

		return 1. / poisson_conv(t=timespan, poe=self.array)

	def to_exceedance_rate_array(self, timespan):
		"""
		Compute exceedance-rate array

		:param timespan:
			float, time span

		:return:
			instance of :class:`ExceedanceRateArray`
		"""
		return ExceedanceRateArray(self.to_exceedance_rates(timespan))

	def to_probabilities(self, timespan=None):
		"""
		Return array of probabilities

		:param timespan:
			None, parameter is present to be compatible with the
			same method in :class:`ExceedanceRateArray`, but is ignored

		:return:
			ndarray
		"""
		return self.array

	def to_probability_array(self, timespan=None):
		"""
		Copy to another probability array.
		This method is present to be compatible with
		:class:`ExceedanceRateArray`

		:param timespan:
			None, parameter is present to be compatible with the
			same method in :class:`ExceedanceRateArray`, but is ignored

		:return:
			instance of :class:`ProbabilityArray`
		"""
		return ProbabilityArray(self.array.copy())

	def to_return_periods(self, timespan):
		"""
		Convert probabilities to return periods

		:param timespan:
			float, time span

		:return:
			ndarray
		"""
		return poisson_conv(poe=self.array, t=timespan)

	def mean(self, axis, weights=None):
		"""
		Compute mean probability of exceedance

		:param axis:
			int, array axis to compute mean for
		:param weights:
			ndarray, optional weights for each element in array axis
			(default: None)

		:return:
			instance of :class:`ProbabilityArray`
		"""
		## exceedance probabilities are not additive, but non-exceedance
		## probabilities are multiplicative, so the logs of non-exceedance
		## probabilities are additive. So, we can take the mean of these logs,
		## take the exponent to obtain the mean non-exceedance probability,
		## then convert back to exceedance probability

		# TODO: this is different from openquake.engine.calculators.post_processing,
		# where the simple mean of the poes is computed.
		# In practice, the differences appear to be minor
		return self.__class__(1 - np.exp(np.average(np.log(1 - self.array),
											axis=axis, weights=weights)))

	def mean_and_variance(self, axis, weights=None):
		"""
		Compute the weighted average and variance.

		:param axis:
		:param weights:
			see :meth:`mean`

		:return:
			(mean, variance) tuple of instances of
			:class:`ProbabilityArray`
		"""
		log_non_exceedance_probs = np.log(1 - self.array)
		mean = np.average(log_non_exceedance_probs, axis=axis, weights=weights)
		_mean = np.expand_dims(mean, axis)
		variance = np.average((log_non_exceedance_probs - _mean)**2, axis=axis,
								weights=weights)
		#np.sum(weights * (log_non_exceedance_probs - _mean)**2, axis=axis) / np.sum(weights)
		mean = self.__class__(1 - np.exp(mean))
		# TODO: from Wikipedia, but probably not correct
		variance = (np.exp(variance) - 1.) * np.exp(2 * mean.array + variance)
		#variance = np.exp(variance)
		variance = self.__class__(variance)
		return (mean, variance)

	def scoreatpercentile(self, axis, percentile_levels, weights=None,
						interpol=True):
		"""
		Compute percentile exceedance probabilities

		:param axis:
			int, array axis to compute percentiles for
		:param percentile_levels:
			list of percentile levels (in the range 0 -100)
		:param weights
			ndarray, optional weights for each element in array axis
			(default: None)
		:param interpol:
			bool, whether or not to interpolate percentile intercept
			in case :param:`weights` is not None
			(default: True)

		:return:
			instance of :class:`ExceedanceRateArray`
		"""
		quantile_levels = np.array(percentile_levels, 'f') / 100.
		## Note: for probabilities, we need to take 1 - quantile_levels!
		quantile_levels = 1 - quantile_levels
		if weights is None:
			percentiles = np.apply_along_axis(mstats.mquantiles, axis,
										np.log(1 - self.array), quantile_levels)
		else:
			percentiles = np.apply_along_axis(wquantiles, axis,
											np.log(1 - self.array), weights,
											quantile_levels, interpol)
		## Rotate percentile axis to last position
		ndim = self.array.ndim
		percentiles = np.rollaxis(percentiles, axis, ndim)
		return self.__class__(1 - np.exp(percentiles))


## Deaggregation arrays

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

	def slice_axis(self, axis, start=None, stop=None):
		"""
		Slice deaggregation matrix along a given axis.

		:param axis:
			Int, index of axis to slice
		:param start:
			Int, start index (default: None)
		:param stop:
			Int, end index (default: None)

		:return:
			instance of subclass of :class:`DeaggMatrix`
		"""
		if start is None:
			start = 0
		if stop is None:
			stop = self.shape[axis]
		idxs = range(start, stop)
		return self.take(idxs, axis=axis)

	def interpolate_axis(self, axis, axis_values, target_values):
		"""
		Interpolate deaggregation matrix along a given axis.

		:param axis:
			Int, index of axis to slice
		:param axis_values:
			1-D array, x values corresponding to axis.
			Length should correspond to length of matrix along given axis
		:param target_values:
			1-D array, x values for which axis will be interpolated

		:return:
			instance of subclass of :class:`DeaggMatrix`
		"""
		def interp_wrapper(yin, xin, xout):
			return interpolate(xin, yin, xout)

		interp_matrix = np.apply_along_axis(interp_wrapper, axis, self,
											axis_values, target_values)
		return self.__class__(interp_matrix)

	def rebin_axis(self, axis, axis_bin_edges, target_bin_edges):
		"""
		Rebin deaggregation matrix along a given axis.

		:param axis:
			Int, index of axis to slice
		:param axis_bin_edges:
			1-D array, bin edges corresponding to axis.
			Length should correspond to length of matrix along given axis + 1
		:param target_bin_edges:
			1-D array, bin edges for which axis will be rebinned

		:return:
			instance of subclass of :class:`DeaggMatrix`
		"""
		zero = np.zeros(1)
		def interp_wrapper(yin, xin, xout):
			yin = np.insert(np.cumsum(yin), 0, zero)
			yout = interpolate(xin, yin, xout)
			yout = np.diff(yout)
			return yout

		rebinned_matrix = np.apply_along_axis(interp_wrapper, axis, self,
											axis_bin_edges, target_bin_edges)
		return self.__class__(rebinned_matrix)


class FractionalContributionMatrix(DeaggMatrix):
	"""
	Class representing a deaggregation matrix containing fractional
	contributions

	:param data:
		ndarray, n-d float array containing fractional contributions
	"""
	def get_total_contribution(self):
		"""
		Return total contribution
		"""
		return np.sum(self.matrix)

	def fold_axis(self, axis, keepdims=False):
		"""
		Fold matrix along one axis.
		Total contribution / exceedance rate in given axis is computed
		as sum of all contributions / exceedance rates.

		:param axis:
			Int, index of axis to fold
		:param keepdims:
			Bool, whether or not folded matrix should have same dimensions
			as original matrix (default: False)

		:return:
			instance of :class:`FractionalContributionMatrix` or
			:class:`ExceedanceRateMatrix`
		"""
		if axis < 0:
			axis = self.num_axes + axis
		#return self.sum(axis=axis, keepdims=keepdims)
		sum = self.sum(axis=axis)
		if keepdims:
			sum = self.__class__(np.expand_dims(sum, axis))
		return sum

	def fold_axes(self, axes):
		"""
		Fold matrix along multiple axes

		:param axes:
			List of integers, indexes of axes to fold

		:return:
			instance of :class:`FractionalContributionMatrix` or
			:class:`ExceedanceRateMatrix`
		"""
		matrix = self
		axes = [{True: self.num_axes + axis, False: axis}[axis < 0] for axis in axes]
		for axis in sorted(axes)[::-1]:
			matrix = matrix.fold_axis(axis)
		return matrix

	def to_fractional_contribution_matrix(self):
		"""
		Renormalize.

		:return:
			instance of :class:`FractionalContributionMatrix`
		"""
		fractions = self.matrix / self.get_total_contribution()
		return FractionalContributionMatrix(fractions)


class ExceedanceRateMatrix(FractionalContributionMatrix):
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
		return poisson_conv(t=timespan, tau=1./self.get_total_exceedance_rate())

	def to_probability_matrix(self, timespan):
		"""
		Convert to probability matrix

		:param timespan:
			Float, time span in Poisson formula.

		:return:
			instance of :class:`ProbabilityMatrix`
		"""
		return ProbabilityMatrix(poisson_conv(t=timespan, tau=1./self.matrix))

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

	def to_fractional_contribution_matrix(self):
		"""
		Convert to a normalized matrix, containing fractional contribution
		to hazard

		:return:
			ndarray
		"""
		fractions = self.matrix / self.get_total_exceedance_rate()
		return FractionalContributionMatrix(fractions)


class ProbabilityMatrix(DeaggMatrix):
	"""
	Class representing a deaggregation matrix containing probabilities
	of exceedance

	:param data:
		ndarray, n-d float array containing exceedance probabilities
	"""
	def __add__(self, other):
		"""
		Sum exceedance probabilities
		"""
		assert isinstance(other, ProbabilityMatrix)
		return ProbabilityMatrix(1 - ((1 - self.matrix) * (1 - other.matrix)))

	def __sub__(self, other):
		"""
		Subtract exceedance probabilities
		"""
		assert isinstance(other, ProbabilityMatrix)
		return ProbabilityMatrix(1 - ((1 - self.matrix) / (1 - other.matrix)))

	def __mul__(self, other):
		"""
		Multiply exceedance probabilities with scalar or other
		probability matrix
		"""
		assert (np.isscalar(other) or other.shape == ()
				or isinstance(other, ProbabilityMatrix))
		#return ProbabilityMatrix(1 - np.exp(np.log(1 - self.matrix) * float(number)))
		return ProbabilityMatrix(1 - (1 - self.matrix) ** other)

	def __div__(self, other):
		"""
		Divide exceedance probabilities with scalar or other
		probability matrix
		"""
		assert (np.isscalar(other) or other.shape == ()
				or isinstance(other, ProbabilityMatrix))
		#return ProbabilityMatrix(1 - np.exp(np.log(1 - self.matrix) / float(number)))
		return self.__mul__(1./other)

	def get_total_exceedance_rate(self, timespan):
		"""
		Return total exceedance rate

		:param timespan:
			Float, time span in Poisson formula.
		"""
		return 1. / poisson_conv(t=timespan, poe=self.get_total_probability())

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
		## Ignore division warnings
		np.seterr(divide='ignore', invalid='ignore')

		return ExceedanceRateMatrix(1. / poisson_conv(t=timespan, poe=self.matrix))

	def to_fractional_contribution_matrix(self):
		"""
		Convert to a normalized matrix, containing fractional contribution
		to hazard

		:return:
			ndarray
		"""
		ln_non_exceedance_probs = np.log(1. - self.matrix)
		fractions = ln_non_exceedance_probs / np.sum(ln_non_exceedance_probs)
		return FractionalContributionMatrix(fractions)

	def fold_axis(self, axis, keepdims=False):
		"""
		Fold matrix along one axis.
		Total probability in given axis is computed as product of all
		non-exceedance probabilities.

		:param axis:
			Int, index of axis to fold
		:param keepdims:
			Bool, whether or not folded matrix should have same dimensions
			as original matrix (default: False)

		:return:
			instance of :class:`ProbabilityMatrix`
		"""
		if axis < 0:
			axis = self.num_axes + axis
		#return 1 - np.prod(1 - self, axis=axis, keepdims=keepdims)
		prod = 1 - np.prod(1 - self, axis=axis)
		if keepdims:
			prod = self.__class__(np.expand_dims(prod, axis))
		return prod

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
