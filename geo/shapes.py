# -*- coding: utf-8 -*-

"""
Classes representing source-model elements in Openquake/oqhazlib. Where possible,
the classes are inherited from oqhazlib classes. All provide methods to create
XML elements, which are used to write a complete source-model NRML file.
Thus, objects instantiated from these classes can be used directly in oqhazlib,
as well as to generate input files for OpenQuake.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from lxml import etree

import numpy as np

from .. import oqhazlib
from ..nrml import ns
#from ..nrml.common import *



__all__ = ['Point', 'Line', 'Polygon', 'NodalPlane']


class Polygon(oqhazlib.geo.Polygon):
	def __init__(self, points):
		super(Polygon, self).__init__(points=points)
		## Store points as a property, as they may contain a z coordinate
		self.points = points

	def __len__(self):
		return len(self.lons)

	def __iter__(self):
		for pt in self.points:
			yield pt

	def __getitem__(self, item):
		"""
		Allow slicing
		"""
		return self.points.__getitem__(item)

	def to_line(self):
		"""
		Convert polygon into a :class:`Line` object
		"""
		return Line(self.points)

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML areaBoundary element)

		:param encoding:
			String, unicode encoding (default: 'latin1')
		"""
		polygon_elem = etree.Element(ns.POLYGON)
		exterior_elem = etree.SubElement(polygon_elem, ns.EXTERIOR)
		linearRing_elem = etree.SubElement(exterior_elem, ns.LINEAR_RING)
		posList_elem = etree.SubElement(linearRing_elem, ns.POSITION_LIST)
		posList_elem.text = " ".join(["%s %s" % (pt.longitude, pt.latitude) for pt in self])
		return polygon_elem


class Line(oqhazlib.geo.Line):
	def __init__(self, points):
		super(Line, self).__init__(points)

	def __iter__(self):
		for pt in self.points:
			yield pt

	def __getitem__(self, item):
		"""
		Allow slicing.
		"""
		return self.points.__getitem__(item)

	@classmethod
	def from_lon_lats(cls, lons, lats):
		"""
		Construct from longitudes and latitudes

		:param lons:
			list or array of floats, longitudes
		:param lats:
			list or array of floats, latitudes

		:return:
			instance of :class:`Line`
		"""
		points = [Point(lon, lat) for (lon, lat) in zip(lons, lats)]
		return cls(points)

	@property
	def lons(self):
		return [pt.longitude for pt in self.points]

	@property
	def lats(self):
		return [pt.latitude for pt in self.points]

	def to_polygon(self):
		"""
		Convert line into a :class:`Polygon` object
		"""
		return Polygon(self.points)

	def reverse_direction(self):
		"""
		Reverse line direction

		:return:
			None, list of points is reversed in place
		"""
		self.points = self.points[::-1]

	def project(self, vertical_distance, dip):
		"""
		Project line up- or downward along dip

		:param vertical_distance:
			float, vertical distance in km, negative/positive is up/downward
		:param dip:
			float, (fault) dip in degrees

		:return:
			instance of :class:`Line`
		"""
		hori_distance = vertical_distance / np.tan(np.radians(dip))
		dip_direction = (self.average_azimuth() + 90) % 360
		lons, lats = np.array(self.lons), np.array(self.lats)
		proj_lons, proj_lats = oqhazlib.geo.geodetic.point_at(lons, lats,
													dip_direction, hori_distance)
		return self.from_lon_lats(proj_lons, proj_lats)

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML root element)

		:param encoding:
			String, unicode encoding (default: 'latin1')
		"""
		lineString_elem = etree.Element(ns.LINE_STRING)
		posList_elem = etree.SubElement(lineString_elem, ns.POSITION_LIST)
		posList_elem.text = " ".join(["%s %s" % (pt.longitude, pt.latitude) for pt in self])
		return lineString_elem


class Point(oqhazlib.geo.Point):
	def __init__(self, longitude, latitude, depth=0.):
		super(Point, self).__init__(longitude, latitude, depth)

	def __getitem__(self, item):
		return self.to_tuple().__getitem__(item)

	def __iter__(self):
		return self.to_tuple().__iter__()

	def __str__(self):
		return self.to_tuple().__str__()

	def __len__(self):
		return self.to_tuple().__len__()

	def to_tuple(self):
		"""
		Convert point to a (longitude, latitude, depth) tuple
		"""
		return (self.longitude, self.latitude, self.depth)

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML root element)

		:param encoding:
			String, unicode encoding (default: 'latin1')
		"""
		point_elem = etree.Element(ns.POINT)
		position_elem = etree.SubElement(point_elem, ns.POSITION)
		position_elem.text = "%s %s" % (self.longitude, self.latitude)
		return point_elem


class NodalPlane(oqhazlib.geo.NodalPlane):
	"""
	Class representing a nodal plane of a focal mechanism

	:param strike:
		Float, strike in degrees
	:param dip:
		Float, dip in degrees
	:param rake:
		Float, rake in degrees
	"""
	def __init__(self, strike, dip, rake):
		super(NodalPlane, self).__init__(strike=strike, dip=dip, rake=rake)


if __name__ == '__main__':
	pass
