# -*- coding: utf-8 -*-

"""
CharacteristicFaultSource class
"""

from __future__ import absolute_import, division, print_function, unicode_literals


## Note: Don't use np as abbreviation for nodalplane!!
import numpy as np

from .. import (oqhazlib, OQ_VERSION)

from ..mfd import *
from ..geo import Point, Line, Polygon
from .rupture_source import RuptureSource

# TODO: need common base class for SimpleFaultSource and CharacteristicFaultSource


__all__ = ['CharacteristicFaultSource']


class CharacteristicFaultSource(RuptureSource, oqhazlib.source.CharacteristicFaultSource):
	"""
	Class representing a characteristic source, this is a fault surface
	with seismic events rupturing the entire fault surface
	independently of their magnitude.
	Thus, rupture mesh spacing, magnitude scaling relationship and rupture
	aspect ratio need not be specified.
	We do not support the case where characteristic fault sources are defined
	by multiple planar surfaces, but only the cases with simple fault surfaces
	or complex fault surfaces.

	:param timespan:
		float, timespan for Poisson temporal occurrence model.
		Introduced in more recent versions of OpenQuake
		(default: 1)
	"""
	def __init__(self, source_id, name, tectonic_region_type, mfd, surface, rake,
				timespan=1):
		self.timespan = timespan
		## OQ version dependent keyword arguments
		oqver_kwargs = {}
		if OQ_VERSION >= '2.9.0':
			oqver_kwargs['temporal_occurrence_model'] = self.tom
		super(CharacteristicFaultSource, self).__init__(source_id=source_id,
								name=name,
								tectonic_region_type=tectonic_region_type,
								mfd=mfd,
								surface=surface,
								rake=rake,
								**oqver_kwargs)

	def __repr__(self):
		return '<CharacteristicFaultSource #%s>' % self.source_id

	@property
	def tom(self):
		"""
		Temporal occurrence model
		"""
		return oqhazlib.tom.PoissonTOM(self.timespan)

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML characteristicFaultSource element)

		:param encoding:
			unicode encoding
			(default: 'latin1')
		"""
		from lxml import etree
		from ..nrml import ns
		from ..nrml.common import xmlstr

		cfs_elem = etree.Element(ns.CHARACTERISTIC_FAULT_SOURCE)
		cfs_elem.set(ns.ID, xmlstr(self.source_id, encoding=encoding))
		cfs_elem.set(ns.NAME, xmlstr(self.name, encoding=encoding))
		cfs_elem.set(ns.TECTONIC_REGION_TYPE, xmlstr(self.tectonic_region_type,
													encoding=encoding))

		if isinstance(self.surface, oqhazlib.geo.surface.ComplexFaultSurface):
			fg_elem = etree.SubElement(cfs_elem, ns.COMPLEX_FAULT_GEOMETRY)
			for i, edge in enumerate(self.edges):
				if i == 0:
					edge_elem = etree.SubElement(fg_elem, ns.FAULT_TOP_EDGE)
				elif i == len(self.edges) - 1:
					edge_elem = etree.SubElement(fg_elem, ns.FAULT_BOTTOM_EDGE)
				else:
					edge_elem = etree.SubElement(fg_elem, ns.INTERMEDIATE_EDGE)
				edge_elem.append(edge.create_xml_element(encoding=encoding))

		elif isinstance(self.surface, oqhazlib.geo.surface.SimpleFaultSurface):
			fg_elem = etree.SubElement(cfs_elem, ns.SIMPLE_FAULT_GEOMETRY)
			fg_elem.append(self.fault_trace.create_xml_element(encoding=encoding))
			dip_elem = etree.SubElement(fg_elem, ns.DIP)
			dip_elem.text = str(self.dip)
			usd_elem = etree.SubElement(fg_elem, ns.UPPER_SEISMOGENIC_DEPTH)
			usd_elem.text = str(self.upper_seismogenic_depth)
			lsd_elem = etree.SubElement(fg_elem, ns.LOWER_SEISMOGENIC_DEPTH)
			lsd_elem.text = str(self.lower_seismogenic_depth)

		cfs_elem.append(self.mfd.create_xml_element(encoding=encoding))

		rake_elem = etree.SubElement(cfs_elem, ns.RAKE)
		rake_elem.text = str(self.rake)

		return cfs_elem

	@property
	def upper_seismogenic_depth(self):
		return self.surface.get_top_edge_depth()

	@property
	def dip(self):
		return self.surface.get_dip()

	@property
	def width(self):
		return self.surface.get_width()

	@property
	def lower_seismogenic_depth(self):
		# TODO: check if this also works for complex fault surfaces
		return self.upper_seismogenic_depth + self.width * np.sin(np.radians(self.dip))

	@property
	def fault_trace(self):
		lons = self.surface.mesh.lons[0,:]
		lats = self.surface.mesh.lats[0,:]
		surface_trace = Line([Point(lon, lat) for (lon, lat) in zip(lons, lats)])
		return surface_trace

	@property
	def edges(self):
		edges = []
		num_edges = self.surface.mesh.shape[0]
		for i in range(num_edges):
			lons = self.surface.mesh.lons[i,:]
			lats = self.surface.mesh.lats[i,:]
			depths = self.surface.mesh.depths[i,:]
			edge = Line([Point(lon, lat, depth)
						for (lon, lat, depth) in zip(lons, lats, depths)])
			edges.append(edge)
		return edges

	def get_length(self):
		"""
		Compute length of fault based on its mesh

		:return:
			Float, fault length
		"""
		fault_mesh = self.surface.get_mesh()
		submesh = fault_mesh[:2, :]
		_, lengths, _, _ = submesh.get_cell_dimensions()
		return np.sum(lengths)

	def get_rupture(self, timespan=1):
		"""
		Get rupture corresponding to characteristic earthquake

		:param timespan:
			Float, time interval for Poisson distribution, in years
			(default: 1)

		:return:
			instance of ProbabilisticRupture
		"""
		## Create MFD with only 1 bin corresponding to Mmax,
		mfd_copy = self.mfd
		max_mag, max_mag_rate = self.mfd.get_annual_occurrence_rates()[-1]
		self.mfd = EvenlyDiscretizedMFD(max_mag, self.mfd.bin_width, [max_mag_rate])
		rup = list(self.iter_ruptures(timespan))[0]
		## Restore original MFD
		self.mfd = mfd_copy
		return rup

	def get_polygon(self):
		"""
		Construct polygonal outline of fault

		:return:
			instance of :class:`Polygon`
		"""
		rupture = self.get_rupture()
		lons, lats, depths = self.get_rupture_bounds(rupture)
		points = [Point(lon, lat, depth)
				for lon, lat, depth in zip(lons, lats, depths)]
		return Polygon(points)

	def get_surface(self):
		"""
		Get fault surface object

		:return:
			instance of :class:`openquake.hazardlib.geo.surface.simple_fault.SimpleFaultSurface`
		"""
		rupture = self.get_rupture()
		return rupture.surface

	def to_lbm_data(self, geom_type="line"):
		"""
		Convert to layeredbasemap line or polygon

		:param geom_type:
			str, "line" or "polygon"
			(default: "line")

		:return:
			layeredbasemap LineData or PolygonData object
		"""
		import mapping.layeredbasemap as lbm

		if geom_type == "line":
			ft = self.fault_trace
			geom = lbm.LineData(ft.lons, ft.lats, label=self.source_id)
		elif geom_type == "polygon":
			pg = self.get_polygon()
			geom = lbm.PolygonData(pg.lons, pg.lats, label=self.source_id)

		geom.value = {}
		attrs = ("source_id", "name", "tectonic_region_type", "rake")
		for attr in attrs:
			setattr(geom, attr, getattr(self, attr))

		return geom



if __name__ == '__main__':
	pass
