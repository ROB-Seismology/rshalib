import numpy as np
from scipy.interpolate import interp1d

from geosurvey.cwp import *


def interpolate(xin, yin, xout):
	"""
	Wrapper for linear interpolation function
	"""
	## Scipy and numpy interpolation don't work as exceedance rates
	## are in decreasing order
	#if np.all(np.diff(xin) < 0):
	#	xin, yin = xin[::-1], yin[::-1]
	## SciPy
	#interpolator = interp1d(xin, yin, bounds_error=False)
	#yout = interpolator(xout)

	## Numpy
	#yout = np.interp(xout, xin, yin, left=yin[0], right=yin[-1])

	## CWP intlin
	yout = intlin(xin, yin, xout)

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


