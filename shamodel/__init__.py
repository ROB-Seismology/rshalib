"""
shamodel submodule
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
	from . import base
else:
	reload(base)

if not reloading:
	from . import dshamodel
else:
	reload(dshamodel)
from .dshamodel import *

if not reloading:
	from . import pshamodel
else:
	reload(pshamodel)
from .pshamodel import PSHAModel, PSHAModelTree, DecomposedPSHAModelTree
