"""
result submodule
"""

from __future__ import absolute_import, division, print_function, unicode_literals



## Reloading mechanism
try:
	reloading
except NameError:
	## Module is imported for the first time
	reloading = False
else:
	## Module is reloaded
	reloading = True
	try:
		## Python 3
		from importlib import reload
	except ImportError:
		## Python 2
		pass


if not reloading:
	from . import plot
else:
	reload(plot)
from .plot import (plot_hazard_curve, plot_hazard_spectrum, plot_histogram,
					plot_deaggregation)

if not reloading:
	from . import hazard_curve
else:
	reload(hazard_curve)
from .hazard_curve import *

if not reloading:
	from . import deaggregation
else:
	reload(deaggregation)
from .deaggregation import (ExceedanceRateMatrix, ProbabilityMatrix,
	FractionalContributionMatrix, DeaggregationSlice, DeaggregationCurve,
	SpectralDeaggregationCurve, get_mean_deaggregation_slice,
	get_mean_deaggregation_curve)

