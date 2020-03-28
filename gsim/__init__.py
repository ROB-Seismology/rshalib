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


if not reloading:
	from . import inverse_gsim
else:
	reload(inverse_gsim)
from .inverse_gsim import *

if not reloading:
	from . import gsim_model
else:
	reload(gsim_model)
from .gsim_model import *

if not reloading:
	from . import gmpe
else:
	reload(gmpe)
from .gmpe import *
