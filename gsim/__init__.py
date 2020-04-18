"""
gsim submodule
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


## inverse_gsim (no internal dependencies)
if not reloading:
	from . import inverse_gsim
else:
	reload(inverse_gsim)
from .inverse_gsim import *

## gsim_model (no internal dependencies)
if not reloading:
	from . import gsim_model
else:
	reload(gsim_model)
from .gsim_model import *

## plot (no internal dependencies)
if not reloading:
	from . import plot
else:
	reload(plot)
from .plot import *

## gmpe (depends on plot)
if not reloading:
	from . import gmpe
else:
	reload(gmpe)
from .gmpe import *

## oqhazlib_gmpe (depends on gmpe)
if not reloading:
	from . import oqhazlib_gmpe
else:
	reload(oqhazlib_gmpe)
from .oqhazlib_gmpe import *

if not reloading:
	from . import gmpes
else:
	reload(gmpes)
from .gmpes import *
