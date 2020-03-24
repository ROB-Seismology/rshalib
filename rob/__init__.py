"""
ROB-specific hazard stuff
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
	from . import rob_source_models
else:
	reload(rob_source_models)
from .rob_source_models import *

if not reloading:
	from . import gmpe_lt
else:
	reload(gmpe_lt)
from .gmpe_lt import *

if not reloading:
	from . import ec8
else:
	reload(ec8)
from .ec8 import *
