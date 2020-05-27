"""
Reference spectra (RG1.60, Eurocode 8, ...)
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
	from . import rg160
else:
	reload(rg160)
from .rg160 import *

if not reloading:
	from . import en1998
else:
	reload(en1998)
from .en1998 import *

if not reloading:
	from . import en1998update
else:
	reload(en1998update)
