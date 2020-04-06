"""
OpenQuake-specific input and output
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
	from . import config
else:
	## Reloading config may in some cases result in the following error:
	## super(OQ_Params, self).__init__(ini_filespec, configspec=configspec, write_empty_values=True, list_values=False)
	## TypeError: super(type, obj): obj must be an instance or subtype of type
	reload(config)
from .config import *

if not reloading:
	from . import IO
else:
	reload(IO)
from .IO import *
