"""
gmpe submodule
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
from .plot import *

if not reloading:
	from . import base
else:
	reload(base)
from .base import *

if not reloading:
	from . import oqhazlib_gmpe
else:
	reload(oqhazlib_gmpe)
from .oqhazlib_gmpe import *

if not reloading:
	from . import bergethierry2003
else:
	reload(bergethierry2003)
from .bergethierry2003 import *
