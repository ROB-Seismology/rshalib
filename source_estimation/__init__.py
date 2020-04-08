"""
Estimation of source parameters from intensity/ground-motion distribution
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
	from . import bakun_wentworth_1997
else:
	reload(bakun_wentworth_1997)
from .bakun_wentworth_1997 import *

if not reloading:
	from . import probabilistic
else:
	reload(probabilistic)
from .probabilistic import *
