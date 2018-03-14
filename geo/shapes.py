# -*- coding: utf-8 -*-

"""
Classes representing source-model elements in Openquake/nhlib. Where possible,
the classes are inherited from nhlib classes. All provide methods to create
XML elements, which are used to write a complete source-model NRML file.
Thus, objects instantiated from these classes can be used directly in nhlib,
as well as to generate input files for OpenQuake.
"""

from lxml import etree

import openquake.hazardlib as nhlib

from ..nrml import ns
from ..nrml.common import *


class Polygon(nhlib.geo.Polygon):
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

	"""
	def __iter__(self):
		self._current_index = 0
		return self

	def next(self):
		if self._current_index >= len(self.lons):
			raise StopIteration
		else:
			lon, lat = self.lons[self._current_index], self.lats[self._current_index]
			self._current_index += 1
			return nhlib.geo.Point(lon, lat)
	"""

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


class Line(nhlib.geo.Line):
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
		"""
		self.points = self.points[::-1]

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


class Point(nhlib.geo.Point):
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


class NodalPlane(nhlib.geo.NodalPlane):
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
