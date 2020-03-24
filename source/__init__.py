"""
source submodule
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
	from . import rupture
else:
	reload(rupture)
from .rupture import Rupture

if not reloading:
	from . import source
else:
	reload(source)
from .source import (PointSource, AreaSource, SimpleFaultSource,
                    ComplexFaultSource, CharacteristicFaultSource,
                    RuptureSource)

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
