# -*- coding: utf-8 -*-

"""
ComplexFaultSource class
"""

from __future__ import absolute_import, division, print_function, unicode_literals

try:
	## Python 2
	basestring
except:
	## Python 3
	basestring = str


## Note: Don't use np as abbreviation for nodalplane!!
import numpy as np

from .. import oqhazlib

from ..mfd import *
from ..geo import Line
from .rupture_source import RuptureSource



__all__ = ['ComplexFaultSource']


class ComplexFaultSource(oqhazlib.source.ComplexFaultSource, RuptureSource):
	"""
	Class representing a complex fault source.

	:param source_id:
		source identifier
	:param name:
		full name of source
	:param tectonic_region_type:
		tectonic region type known to oqhazlib/OQ.
		See :class:`oqhazlib.const.TRT`
	:param mfd:
		magnitude-frequency distribution (instance of :class:`TruncatedGRMFD`
		or :class:`EvenlyDiscretizedMFD`)
	:param rupture_mesh_spacing:
		The desired distance between two adjacent points in source's
		ruptures' mesh, in km. Mainly this parameter allows to balance
		the trade-off between time needed to compute the :meth:`distance
		<oqhazlib.geo.surface.base.BaseSurface.get_min_distance>` between
		the rupture surface and a site and the precision of that computation.
	:param magnitude_scaling_relationship:
		Instance of subclass of :class:`oqhazlib.scalerel.base.BaseMSR` to
		describe how does the area of the rupture depend on magnitude and rake.
	:param rupture_aspect_ratio:
		float number representing how much source's ruptures are more wide
		than tall. Aspect ratio of 1 means ruptures have square shape,
		value below 1 means ruptures stretch vertically more than horizontally
		and vice versa.
	:param edges:
		A list of :class:`Line` objects, representing fault source geometry. See
		:meth:`oqhazlib.geo.surface.complex_fault.ComplexFaultSurface.from_fault_data`
	:param rake:
		Angle describing fault rake in decimal degrees.

	"""
	# TODO: add bg_zone parameter as for SimpleFaultSource
	def __init__(self, source_id, name, tectonic_region_type, mfd,
				rupture_mesh_spacing, magnitude_scaling_relationship,
				rupture_aspect_ratio, edges, rake):
		"""
		"""
		if isinstance(magnitude_scaling_relationship, (str, unicode)):
			magnitude_scaling_relationship = getattr(oqhazlib.scalerel, magnitude_scaling_relationship)()
		super(ComplexFaultSource, self).__init__(source_id=source_id,
				name=name,
				tectonic_region_type=tectonic_region_type,
				mfd=mfd,
				rupture_mesh_spacing=rupture_mesh_spacing,
				magnitude_scaling_relationship=magnitude_scaling_relationship,
				rupture_aspect_ratio=rupture_aspect_ratio,
				edges=edges,
				rake=rake)

	def __repr__(self):
		return '<ComplexFaultSource #%s>' % self.source_id

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML complexFaultSource element)

		:param encoding:
			unicode encoding
			(default: 'latin1')
		"""
		from lxml import etree
		from ..nrml import ns
		from ..nrml.common import xmlstr

		cfs_elem = etree.Element(ns.COMPLEX_FAULT_SOURCE)
		cfs_elem.set(ns.ID, xmlstr(self.source_id, encoding=encoding))
		cfs_elem.set(ns.NAME, xmlstr(self.name, encoding=encoding))
		cfs_elem.set(ns.TECTONIC_REGION_TYPE, xmlstr(self.tectonic_region_type,
													encoding=encoding))

		cfg_elem = etree.SubElement(cfs_elem, ns.COMPLEX_FAULT_GEOMETRY)
		for i, edge in enumerate(self.edges):
			if i == 0:
				edge_elem = etree.SubElement(cfg_elem, ns.FAULT_TOP_EDGE)
			elif i == len(self.edges) - 1:
				edge_elem = etree.SubElement(cfg_elem, ns.FAULT_BOTTOM_EDGE)
			else:
				edge_elem = etree.SubElement(cfg_elem, ns.INTERMEDIATE_EDGE)
			edge_elem.append(edge.create_xml_element(encoding=encoding))

		magScaleRel_elem = etree.SubElement(cfs_elem, ns.MAGNITUDE_SCALING_RELATIONSHIP)
		magScaleRel_elem.text = self.magnitude_scaling_relationship.__class__.__name__

		ruptAspectRatio_elem = etree.SubElement(cfs_elem, ns.RUPTURE_ASPECT_RATIO)
		ruptAspectRatio_elem.text = str(self.rupture_aspect_ratio)

		cfs_elem.append(self.mfd.create_xml_element(encoding=encoding))

		rake_elem = etree.SubElement(cfs_elem, ns.RAKE)
		rake_elem.text = str(self.rake)

		return cfs_elem

	@property
	def min_mag(self):
		"""
		Return Mmin specified in the source's MFD
		"""
		if isinstance(self.mfd, TruncatedGRMFD):
			return self.mfd.min_mag
		elif isinstance(self.mfd, EvenlyDiscretizedMFD):
			return self.mfd.min_mag - self.mfd.bin_width / 2.

	@property
	def max_mag(self):
		"""
		Return Mmax specified in the source's MFD
		"""
		return self.mfd.max_mag

	@property
	def longitudes(self):
		"""
		Return list of longitudes in the source's ComplexFaultGeometry object
		"""
		lons = []
		for edge in self.edges:
			lons.append(edge.lons)
		return lons

	@property
	def latitudes(self):
		"""
		Return list of latitudes in the source's ComplexFaultGeometry object
		"""
		lats = []
		for edge in self.edges:
			lats.append(edge.lats)
		return lats

	def to_characteristic_source(self):
		"""
		Convert to a characteristic fault source

		:return:
			instance of :class:`CharacteristicFaultSource`
		"""
		from .characteristic import CharacteristicFaultSource

		surface = oqhazlib.geo.surface.ComplexFaultSurface.from_fault_data(
			self.edges, self.rupture_mesh_spacing)

		return CharacteristicFaultSource(self.source_id, self.name,
			self.tectonic_region_type, self.mfd, surface, self.rake)



if __name__ == '__main__':
	pass