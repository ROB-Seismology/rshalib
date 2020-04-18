# The Hazard Library
# Copyright (C) 2012 GEM Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Package :mod:`openquake.hazardlib.gsim` contains base and specific
implementations of ground shaking intensity models. See
:mod:`openquake.hazardlib.gsim.base`.
"""
import os
import inspect
import importlib
from collections import OrderedDict
from openquake.hazardlib.gsim.base import (
    GMPE, IPE, GroundShakingIntensityModel)


def get_available_gsims():
	'''
	Return an ordered dictionary with the available GSIM classes, keyed
	by class name.
	'''
	gsims = {}
	for fname in os.listdir(os.path.dirname(__file__)):
		if fname.endswith('.py'):
			modname, _ext = os.path.splitext(fname)
			mod = importlib.import_module('.' + modname, package=__name__)
			for cls in mod.__dict__.values():
				if inspect.isclass(cls) and issubclass(
					cls, GroundShakingIntensityModel) and cls not in (
						GroundShakingIntensityModel, GMPE, IPE):
					gsims[cls.__name__] = cls
	return OrderedDict((k, gsims[k]) for k in sorted(gsims))
