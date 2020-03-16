"""
Utilities
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


__all__ = ['interpolate', 'logrange', 'wquantiles', 'seq']


def interpolate(xin, yin, xout, lib='numpy', left=None, right=None, **kwargs):
	"""
	Wrapper for linear interpolation function in numpy or scipy

	Automatically reorders xin/yin if necessary, and ignores inf values

	:param xin:
		1D array, input X values
	:param yin:
		1D array, input Y values
	:param xout:
		1D array, X values that should be interpolated
	:param lib:
		str, which library to use for the interpolation: 'numpy',
		'scipy' or 'cwp'
		(default: 'numpy')
	:param left:
		float, value to return for x < xout[0], e.g. np.nan
		Note that scipy also supports the string "extrapolate", but
		for both left and right values
		(default: None, will use yin[0])
	:param right:
		float, value to return for x > xout[0], e.g. np.nan
		(default: None, will use yin[-1])
	:param kwargs:
		additional keyword-arguments understood by :func:`np.interp`
		or :class:`scipy.interpolate.interp1d`

	:return:
		1D array, interpolated Y values
	"""
	## Ignore inf values
	if np.isinf(xin).any():
		print("Warning: inf values in xin ignored for interpolation!")
		idxs = -np.isinf(xin)
		xin = xin[idxs]
		yin = yin[idxs]
	if np.isinf(yin).any():
		print("Warning: inf values in yin ignored for interpolation!")
		idxs = -np.isinf(yin)
		xin = xin[idxs]
		yin = yin[idxs]

	## Scipy and numpy interpolation don't work if input arrays are
	## in decreasing order
	if lib in ('numpy', 'scipy'):
		if np.all(np.diff(xin) <= 0):
			xin, yin = xin[::-1], yin[::-1]

	## Out-of-bounds values
	if left is None:
		left = yin[0]
	if right is None:
		right = yin[-1]

	## Numpy
	if lib == 'numpy':
		yout = np.interp(xout, xin, yin, left=left, right=right, **kwargs)

	## SciPy
	elif lib == 'scipy':
		from scipy.interpolate import interp1d
		kwargs['bounds_error'] = kwargs.get('bounds_error', False)
		if 'extrapolate' in (left, right):
			fill_value = 'extrapolate'
		else:
			fill_value = (left, right)
		kwargs['fill_value'] = kwargs.get('fill_value', fill_value)
		interpolator = interp1d(xin, yin, **kwargs)
		yout = interpolator(xout)

	## CWP intlin
	elif lib == 'cwp':
		from geosurvey.cwp import intlin
		yout = intlin(xin, yin, xout, yinl=left, yinr=right)

	return yout


def logrange(min, max, n):
	"""
	Generate logarithmically-spaced range

	:param min:
		Float, minimum value
	:param max:
		Float, maximum value (inclusive)
	:param n:
		Int, number of values to generate

	:return:
		Float array
	"""
	logs = np.linspace(np.log10(min), np.log10(max), n)
	return 10**logs


def wquantiles(data, weights, quantile_levels, interpol=True):
	"""
	Compute weighted quantiles of a 1-D array

	:param data:
		1-D array containing data values
	:param weights:
		1-D array containing weights
	:param quantile_levels:
		list or array containing quantile levels (in range 0 - 1)
	:param interpol:
		bool, whether or not percentile intercept should be
		interpolated (default: True)

	:return:
		1-D array containing data values corresponding to quantiles
	"""
	## Sort data and weights
	ind_sorted = np.argsort(data)
	sorted_data = data[ind_sorted]
	sorted_weights = weights[ind_sorted]

	## Compute the auxiliary arrays
	Sn = np.cumsum(sorted_weights)
	Pn = (Sn-0.5*sorted_weights)/np.sum(sorted_weights)

	## Get quantile intercepts
	if interpol:
		## Interpolate the values of the weighted quantiles
		quantile_intercepts = np.interp(quantile_levels, Pn, sorted_data)
	else:
		## Alternatively, if percentile should correspond to a value in the list
		idxs = np.searchsorted(Pn, quantile_levels)
		quantile_intercepts = sorted_data[idxs]

	return quantile_intercepts


def seq(start, stop, step):
	"""
	Alternative to scitools.numpytools.seq function to avoid
	some nasty side effects of importing this module:
	- modification of matplotlib defaults (e.g., backend);
	- immediate disappearance of matplotlib plots when pylab.show()
	  is called.

	Generate sequence of numbers between start and stop (inclusive!),
	with increment of step. More robust alternative to arange, where the
	returned sequence normally should not include stop, but sometimes it
	does in an unpredictable way (particulary with a step of 0.1).
	Compare for instance:
	np.arange(5.1, 6.1, 0.1):
	array([ 5.1,  5.2,  5.3,  5.4,  5.5,  5.6,  5.7,  5.8,  5.9,  6. ])

	np.arange(5.1, 6.2, 0.1):
	array([ 5.1,  5.2,  5.3,  5.4,  5.5,  5.6,  5.7,  5.8,  5.9,  6. ,  6.1,  6.2])


	Note that, if stop is not exactly aligned to start + n*dm,
	rounding will apply.

	:param start:
		int or float, start of sequence
	:param end:
		int or float, end of sequence
	:param step:
		int or float, spacing between values

	:return:
		1-D numpy array, int if both start and step are integers,
		float otherwise
	"""
	diff = float(stop - start)
	nvals = int(round(diff / step)) + 1
	return start + np.arange(nvals) * step
