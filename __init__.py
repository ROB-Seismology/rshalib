# rshalib: The ROB Seismic Hazard Assessment Library
# Copyright (C) 2013 Royal Observatory of Belgium
# Main developer: Kris Vanneste, ROB
# Contributing developers: Bart Vleminckx, ROB

"""
rshalib: The ROB Seismic Hazard Assessment Library

rshalib is a python library for building and running seismic hazard model,
and plotting the results.

rshalib is built on top of oq-hazardlib (formerly nhlib), the OpenQuake
hazard library, developed by GEM, and available at:
https://github.com/gem/oq-hazardlib

Models built with rshalib can be:
	- run in place with oq-hazardlib
	- exported to CRISIS2007 .DAT files
	- exported to Openquake NRML format
"""

## Because of the reloads, the order of imported submodules is important,
## as this may result in the following type of error:

## super(OQ_Params, self).__init__(ini_filespec, configspec=configspec, write_empty_values=True, list_values=False)
## TypeError: super(type, obj): obj must be an instance or subtype of type

## Therefore, submodules should be loaded in order of inter-dependency,
## i.e. a submodule depending on another one, should be loaded after that one.


## No internal dependencies
import nrml
reload(nrml)

## No internal dependencies
import utils
reload(utils)

## No internal dependencies
import cav
reload(cav)

## Depends on nrml
import geo
reload(geo)

## Depends on geo, nrml
import site
reload(site)

## Depends on utils
import gsim
reload(gsim)

## Depends on geo, nrml, utils
import pmf
reload(pmf)

## Depends on nrml
import mfd
reload(mfd)

## Depends on geo, mfd, nrml
import source
reload(source)

## Depends on geo, mfd, pmf, source
import rob
reload(rob)

## Depends on nrml, pmf, site, utils
import result
reload(result)

## Depends on mfd, result, source
import crisis
reload(crisis)

## Depends on nrml, result, site
import openquake
reload(openquake)

## Depends on nrml, pmf, source
import logictree
reload(logictree)

## Depends on crisis, geo, gsim, logictree, openquake, pmf, result, site, source
import shamodel
reload(shamodel)
