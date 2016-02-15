import numpy as np
import numpy.ma as ma
import matplotlib



def interpolate(xin, yin, xout):
	"""
	Wrapper for linear interpolation function
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

	## Scipy and numpy interpolation don't work as exceedance rates
	## are in decreasing order,
	if np.all(np.diff(xin) <= 0):
		xin, yin = xin[::-1], yin[::-1]
	## SciPy
	#from scipy.interpolate import interp1d
	#interpolator = interp1d(xin, yin, bounds_error=False)
	#yout = interpolator(xout)

	## Numpy
	yout = np.interp(xout, xin, yin, left=yin[0], right=yin[-1])

	## CWP intlin
	#from geosurvey.cwp import *
	#yout = intlin(xin, yin, xout)

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


class LevelNorm(matplotlib.colors.Normalize):
	"""
	Normalize a given value to the 0-1 range according to pre-defined levels,
	for use in matplotlib plotting functions involving color maps.

	:param levels:
		list or array, containing a number of levels in ascending order,
		including the maximum value. These levels will be uniformly spaced
		in the color domain.
	"""
	def __init__(self, levels):
		vmin = levels[0]
		vmax = levels[1]
		self.levels = levels
		matplotlib.colors.Normalize.__init__(self, vmin, vmax)

	def __call__(self, value, clip=None):
		level_values = np.linspace(0, 1, len(self.levels))
		out_values = interpolate(self.levels, level_values, value)
		try:
			mask = value.mask
		except:
			mask = None
		out_values = ma.masked_array(out_values, mask)
		return out_values


