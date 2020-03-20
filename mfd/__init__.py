"""
This module extends functionality of oqhazlib.mfd
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

#TODO: maybe also define __all__ ??

if not reloading:
	from . import base
else:
	reload(base)
from .base import *

if not reloading:
	from . import plot
else:
	reload(plot)
from .plot import *

if not reloading:
	from . import evenly_discretized
else:
	reload(evenly_discretized)
from .evenly_discretized import *

if not reloading:
	from . import truncated_gr
else:
	reload(truncated_gr)
from .truncated_gr import *

## depends on evenly_discretized
if not reloading:
	from . import characteristic
else:
	reload(characteristic)
from .characteristic import *

## depends on evenly_discretized
if not reloading:
	from . import youngs_coppersmith_1985
else:
	reload(youngs_coppersmith_1985)
from .youngs_coppersmith_1985 import *
