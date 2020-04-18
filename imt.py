"""
Intensity measure types
"""

from __future__ import absolute_import, division, print_function, unicode_literals


from . import oqhazlib
from openquake.hazardlib.imt import *



__all__ = list(oqhazlib.imt.__all__) + ['from_tuple']


def from_tuple(imt_tuple):
	"""
	Construct Intensity Measure Type from tuple

	:param imt_tuple:
		tuple of 1 or 3 elements:
		(imt string)
		(imt string, float period, float damping)

	:return:
		instance of :class:`oqhazlib.imt._IMT`
	"""
	imt = getattr(oqhazlib.imt, imt_tuple[0])(*imt_tuple[1:])
	return imt
