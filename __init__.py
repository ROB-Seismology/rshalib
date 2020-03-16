# rshalib: The ROB Seismic Hazard Assessment Library
# Copyright (C) 2013 Royal Observatory of Belgium
# Main developer: Kris Vanneste, ROB
# Contributing developers: Bart Vleminckx, ROB

"""
rshalib: The ROB Seismic Hazard Assessment Library

rshalib is a python library for building and running seismic hazard models,
and plotting the results.

rshalib is built on top of oq-hazardlib (formerly nhlib), the OpenQuake
hazard library, developed by GEM, and available at:
https://github.com/gem/oq-hazardlib

Models built with rshalib can be:
	- run in place with oq-hazardlib
	- exported to CRISIS2007 .DAT files
	- exported to Openquake NRML format
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


## Import submodules
## Because of the reloads, the order of imported submodules is important,
## as this may result in the following type of error:

## super(OQ_Params, self).__init__(ini_filespec, configspec=configspec, write_empty_values=True, list_values=False)
## TypeError: super(type, obj): obj must be an instance or subtype of type

## Therefore, submodules should be loaded in order of inter-dependency,
## i.e. a submodule depending on another one, should be loaded after that one.


## nrml (no internal dependencies)
if not reloading:
	from . import nrml
else:
	reload(nrml)

## utils( no internal dependencies)
if not reloading:
	from . import utils
else:
	reload(utils)

## poisson (no internal dependencies)
if not reloading:
	from . import poisson
else:
	reload(poisson)

## cav (no internal dependencies)
if not reloading:
	from . import cav
else:
	reload(cav)

## calc (no internal dependencies)
if not reloading:
	from . import calc
else:
	reload(calc)

## geo (depends on nrml)
if not reloading:
	from . import geo
else:
	reload(geo)

## site (depends on geo, nrml)
if not reloading:
	from . import site
else:
	reload(site)

## gsim (depends on utils, cav)
if not reloading:
	from . import gsim
else:
	reload(gsim)

## pmf (depends on geo, nrml, utils)
if not reloading:
	from . import pmf
else:
	reload(pmf)

## mfd (depends on nrml)
if not reloading:
	from . import mfd
else:
	reload(mfd)

## source (depends on geo, mfd, nrml)
if not reloading:
	from . import source
else:
	reload(source)

## rob (depends on geo, mfd, pmf, source)
if not reloading:
	from . import rob
else:
	reload(rob)

## result (depends on nrml, pmf, site, utils)
if not reloading:
	from . import result
else:
	reload(result)

## siteresponse (depends on calc, utils, result)
if not reloading:
	from . import siteresponse
else:
	reload(siteresponse)

## crisis (depends on mfd, result, source)
if not reloading:
	from . import crisis
else:
	reload(crisis)

## openquake (depends on nrml, result, site)
if not reloading:
	from . import openquake
else:
	reload(openquake)

## logictree (depends on nrml, pmf, source)
if not reloading:
	from .import logictree
else:
	reload(logictree)

## shamodel (depends on calc, crisis, geo, gsim, logictree, openquake, pmf,
## result, site, source)
if not reloading:
	from . import shamodel
else:
	reload(shamodel)
