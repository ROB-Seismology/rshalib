"""
Base arrays for hazard results
"""

from __future__ import absolute_import, division, print_function, unicode_literals

#from decimal import Decimal
import numpy as np

from ..poisson import poisson_conv


# TODO: common base class for ExceedanceRateArray, ExceedanceRateMatrix etc.


__all__ = ['DeaggMatrix', 'ExceedanceRateMatrix', 'ProbabilityMatrix',
			'FractionalContributionMatrix']



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
		from ..utils import interpolate

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
		from ..utils import interpolate

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
		ndarray, n-d float array containing probabilities
	"""
	#def __new__(cls, data, timespan):
	#	obj = DeaggMatrix.__new__(cls, data)
	#	obj.timespan = timespan
	#	return obj

	def __add__(self, other):
		assert isinstance(other, ProbabilityMatrix)
		return ProbabilityMatrix(1 - ((1 - self.matrix) * (1 - other.matrix)))

	def __sub__(self, other):
		assert isinstance(other, ProbabilityMatrix)
		return ProbabilityMatrix(1 - ((1 - self.matrix) / (1 - other.matrix)))

	def __mul__(self, other):
		#assert isinstance(number, (int, float, Decimal))
		#return ProbabilityMatrix(1 - np.exp(np.log(1 - self.matrix) * float(number)))
		return ProbabilityMatrix(1 - (1 - self.matrix) ** other)

	def __div__(self, other):
		#assert isinstance(number, (int, float, Decimal))
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


