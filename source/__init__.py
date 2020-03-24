"""
source submodule

Classes representing source-model elements in Openquake/oqhazlib.
Where possible, the classes are inherited from oqhazlib classes.
All provide methods to create XML elements, which are used to write
a complete source-model NRML file.
Thus, objects instantiated from these classes can be used directly
in oqhazlib, as well as to generate input files for OpenQuake.
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


## rupture (no internal dependencies)
if not reloading:
	from . import rupture
else:
	reload(rupture)
from .rupture import *

## rupture_source (no internal dependencies)
if not reloading:
	from . import rupture_source
else:
	reload(rupture_source)
from .rupture_source import *

## point (depends on rupture_source)
if not reloading:
	from . import point
else:
	reload(point)
from .point import *

## area (depends on rupture_source, point)
if not reloading:
	from . import area
else:
	reload(area)
from .area import *

## simple_fault (depends on rupture_source, point)
if not reloading:
	from . import simple_fault
else:
	reload(simple_fault)
from .simple_fault import *

## complex_fault (depends on rupture_source)
if not reloading:
	from . import complex_fault
else:
	reload(complex_fault)
from .complex_fault import *

## characteristic (depends on rupture_source)
if not reloading:
	from . import characteristic
else:
	reload(characteristic)
from .characteristic import *


"""
## source_model
if not reloading:
	from . import source_model
else:
	reload(source_model)
from .source_model import SourceModel

if not reloading:
	from . import grid_source_model
else:
	reload(grid_source_model)
from .grid_source_model import SimpleUniformGridSourceModel

if not reloading:
	from . import read_from_gis
else:
	reload(read_from_gis)
from .read_from_gis import (import_source_model_from_gis,
							import_source_from_gis_record,
							import_point_or_area_source_from_gis_record,
							import_simple_fault_source_from_gis_record)

if not reloading:
	from . import smoothed_seismicity
else:
	reload(smoothed_seismicity)
from .smoothed_seismicity import SmoothedSeismicity

if not reloading:
	from . import fault_network
else:
	reload(fault_network)
from .fault_network import FaultNetwork
"""