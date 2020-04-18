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


## GMPEs
if not reloading:
	from . import akkarbommer2010
else:
	reload(akkarbommer2010)
from .akkarbommer2010 import *

if not reloading:
	from . import ambraseys1995
else:
	reload(ambraseys1995)
from .ambraseys1995 import *

if not reloading:
	from . import ambraseys1996
else:
	reload(ambraseys1996)
from .ambraseys1996 import *

if not reloading:
	from . import bergethierry2003
else:
	reload(bergethierry2003)
from .bergethierry2003 import *

if not reloading:
	from . import bommer2011
else:
	reload(bommer2011)
from .bommer2011 import *

if not reloading:
	from . import cauzzifaccioli2008
else:
	reload(cauzzifaccioli2008)
from .cauzzifaccioli2008 import *

if not reloading:
	from . import mcguire1974
else:
	reload(mcguire1974)
from .mcguire1974 import *
