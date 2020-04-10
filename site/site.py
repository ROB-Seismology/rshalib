# -*- coding: utf-8 -*-

"""
Classes representing sites in Openquake. Where possible,
the classes are inherited from oqhazlib classes. All provide methods to create
XML elements, which are used to write a complete source-model NRML file.
Thus, objects instantiated from these classes can be used directly in oqhazlib,
as well as to generate input files for OpenQuake.
"""

# TODO: add kappa and thickness as soil_params

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from .. import (oqhazlib, OQ_VERSION)
from ..geo import Point
from .ref_soil_params import REF_SOIL_PARAMS



__all__ = ['GenericSite', 'SoilSite']


class GenericSite(Point):
	"""
	Class representing a generic site (without soil characteristics) for SHA,
	derived from oqhazlib.geo.Point, but which can also be sliced, indexed, and
	looped over like a (longitude, latitude, depth) tuple.

	:param longitude:
		float, longitude
	:param latitude:
		float, latitude
	:param depth:
		float, depth in km
		(default: 0.)
	:param name:
		str, site name
		(default: "")
	"""
	def __init__(self, longitude, latitude, depth=0., name=""):
		super(GenericSite, self).__init__(longitude, latitude, depth)
		if not name:
			self.name = self.get_name_from_position()
		else:
			self.name = name

	def __str__(self):
		return self.name

	def __repr__(self):
		return '<GenericSite %s>' % self.name

	@property
	def lon(self):
		return self.longitude

	@property
	def lat(self):
		return self.latitude

	def get_name_from_position(self):
		"""
		Construct generic site name from position (lon, lat, and
		optionally depth)

		:return:
			str, site name
		"""
		NorS = 'S' if self.latitude < 0 else 'N'
		EorW = 'W' if self.longitude < 0 else 'E'
		name = "%.3f %s, %.3f %s"
		name %= (abs(self.longitude), EorW, abs(self.latitude), NorS)
		if self.depth:
			name += ", %.1f" % self.depth
		return name

	def to_soil_site(self, soil_params=REF_SOIL_PARAMS):
		"""
		Convert to a SoilSite object, representing a site with
		soil characteristics

		:param soil_params:
			dict, mapping soil parameters to values:
			- vs30:
				float, average shear-wave velocity down to 30 m depth, in m/s
				(default: value defined in vs30.rock)
			- vs30measured:
				bool, indicating whether vs30 was actually measured
				(default: False)
			- z1pt0
			- z2pt5
			- kappa (optional)

		:return:
			instance of :class:`SoilSite`
		"""
		return SoilSite(self.longitude, self.latitude, self.depth, soil_params,
						self.name)

## alias for backwards compatibility
SHASite = GenericSite


class SoilSite(oqhazlib.site.Site):
	"""
	Class representing a site with soil characteristics.
	This class extends :class:`oqhazlib.site.Site` by providing longitude,
	latitude, depth, and name properties

	:param longitude:
		float, longitude
	:param latitude:
		float, latitude
	:param depth:
		float, depth
		(default: 0.)
	:param soil_params:
		dict, containing soil parameters (vs30, vs30measured, z1pt0,
		z2pt5, and kappa)
		(default: REF_SOIL_PARAMS)
	:param name:
		str, site name
		(default: "")
	"""
	def __init__(self, longitude, latitude, depth=0., soil_params=REF_SOIL_PARAMS,
				name=""):
		location = Point(longitude, latitude, depth)
		if OQ_VERSION >= '2.9.0':
			soil_params = soil_params.copy()
			self.kappa = soil_params.pop('kappa', np.nan)
		super(SoilSite, self).__init__(location, **soil_params)
		self.name = name

	def __repr__(self):
		return '<SoilSite %s>' % self.name

	@property
	def lon(self):
		return self.location.longitude

	@property
	def lat(self):
		return self.location.latitude

	@property
	def longitude(self):
		return self.location.longitude

	@property
	def latitude(self):
		return self.location.latitude

	@property
	def depth(self):
		return self.location.depth

	def to_generic_site(self):
		"""
		Convert to a GenericSite object, representing a site without
		soil characteristics

		:return:
			instance of :class:`GenericSite`
		"""
		return GenericSite(self.lon, self.lat, self.depth, self.name)
