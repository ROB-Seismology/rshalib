"""
site submodule
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


## vs30
if not reloading:
	from . import vs30
else:
	reload(vs30)

## ref_soil_params (depends on vs30)
if not reloading:
	from . import ref_soil_params
else:
	reload(ref_soil_params)
from .ref_soil_params import REF_SOIL_PARAMS

## site (depends on ref_soil_params)
if not reloading:
	from . import site
else:
	reload(site)
from .site import *
