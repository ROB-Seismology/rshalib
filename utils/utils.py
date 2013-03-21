import numpy as np
import numpy.ma as ma
import matplotlib



def interpolate(xin, yin, xout):
	"""
	Wrapper for linear interpolation function
	"""
	## Scipy and numpy interpolation don't work as exceedance rates
	## are in decreasing order
	if np.all(np.diff(xin) < 0):
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
	step = (np.log10(max) - np.log10(min)) / (n - 1)
	logs = np.array([np.log10(min) + i * step for i in range(n)])
	return 10**logs


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


