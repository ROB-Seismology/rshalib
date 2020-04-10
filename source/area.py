# -*- coding: utf-8 -*-

"""
AreaSource class
"""

from __future__ import absolute_import, division, print_function, unicode_literals


## Note: Don't use np as abbreviation for nodalplane!!
import numpy as np

from .. import (oqhazlib, OQ_VERSION)

from ..msr import get_oq_msr
from ..mfd import (TruncatedGRMFD, EvenlyDiscretizedMFD)
from .rupture_source import RuptureSource
from .point import PointSource



__all__ = ['AreaSource']


class AreaSource(RuptureSource, oqhazlib.source.AreaSource):
	"""
	Class representing an area source, i.e. a polygonal geographical region
	where seismicity is assumed to be uniform.
	An area source is similar to a point source, except for the geometry

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
		Float number representing how much source's ruptures are more wide
		than tall. Aspect ratio of 1 means ruptures have square shape,
		value below 1 means ruptures stretch vertically more than horizontally
		and vice versa.
	:param upper_seismogenic_depth:
		Minimum depth an earthquake rupture can reach, in km.
	:param lower_seismogenic_depth:
		Maximum depth an earthquake rupture can reach, in km.
	:param nodal_plane_distribution:
		:class:`NodalPlaneDistribution` object with values that are instances
		of :class:`NodalPlane`. Shows the distribution of probability for
		rupture to have the certain nodal plane.
	:param hypocenter_distribution:
		:class:`HypocentralDepthDistribution` with values being float numbers
		in km representing the depth of the hypocenter. Latitude and longitude
		of the hypocenter is always set to ones of ``location``.
	:param polygon:
		An instance of :class:`oqhazlib.geo.polygon.Polygon` that defines
		source's area.
	:param area_discretization:
		Float number, polygon area discretization spacing in kilometers.
		See :meth:`iter_ruptures`.
	:param timespan:
		float, timespan for Poisson temporal occurrence model.
		Introduced in more recent versions of OpenQuake
		(default: 1)
"""
	def __init__(self, source_id, name, tectonic_region_type, mfd,
				rupture_mesh_spacing, magnitude_scaling_relationship,
				rupture_aspect_ratio,
				upper_seismogenic_depth, lower_seismogenic_depth,
				nodal_plane_distribution, hypocenter_distribution,
				polygon, area_discretization, timespan=1):
		"""
		"""
		self.timespan = timespan
		magnitude_scaling_relationship = get_oq_msr(magnitude_scaling_relationship)

		## OQ version dependent keyword arguments
		oqver_kwargs = {}
		if OQ_VERSION >= '2.9.0':
			oqver_kwargs['temporal_occurrence_model'] = self.tom
		super(AreaSource, self).__init__(source_id=source_id,
				name=name,
				tectonic_region_type=tectonic_region_type,
				mfd=mfd,
				rupture_mesh_spacing=rupture_mesh_spacing,
				magnitude_scaling_relationship=magnitude_scaling_relationship,
				rupture_aspect_ratio=rupture_aspect_ratio,
				upper_seismogenic_depth=upper_seismogenic_depth,
				lower_seismogenic_depth=lower_seismogenic_depth,
				nodal_plane_distribution=nodal_plane_distribution,
				hypocenter_distribution=hypocenter_distribution,
				polygon=polygon,
				area_discretization=area_discretization,
				**oqver_kwargs)

	def __repr__(self):
		return '<AreaSource #%s>' % self.source_id

	@property
	def tom(self):
		"""
		Temporal occurrence model
		"""
		try:
			return self.temporal_occurrence_model
		except AttributeError:
			return oqhazlib.tom.PoissonTOM(self.timespan)

	def set_timespan(self, timespan):
		"""
		Modify timespan

		:param timespan:
			float, timespan for Poisson temporal occurrence model
		"""
		self.timespan = timespan
		if OQ_VERSION >= '2.9.0':
			self.tom.time_span = timespan

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML areaSource element)

		:param encoding:
			unicode encoding
			(default: 'latin1')
		"""
		from lxml import etree
		from ..nrml import ns
		from ..nrml.common import xmlstr

		areaSource_elem = etree.Element(ns.AREA_SOURCE)
		areaSource_elem.set(ns.ID, xmlstr(self.source_id, encoding=encoding))
		areaSource_elem.set(ns.NAME, xmlstr(self.name, encoding=encoding))
		areaSource_elem.set(ns.TECTONIC_REGION_TYPE,
							xmlstr(self.tectonic_region_type, encoding=encoding))

		areaGeometry_elem = etree.SubElement(areaSource_elem, ns.AREA_GEOMETRY)
		areaGeometry_elem.append(self.polygon.create_xml_element(encoding=encoding))
		upperSeismoDepth_elem = etree.SubElement(areaGeometry_elem,
												ns.UPPER_SEISMOGENIC_DEPTH)
		upperSeismoDepth_elem.text = str(self.upper_seismogenic_depth)
		lowerSeismoDepth_elem = etree.SubElement(areaGeometry_elem,
												ns.LOWER_SEISMOGENIC_DEPTH)
		lowerSeismoDepth_elem.text = str(self.lower_seismogenic_depth)

		magScaleRel_elem = etree.SubElement(areaSource_elem,
											ns.MAGNITUDE_SCALING_RELATIONSHIP)
		magScaleRel_elem.text = self.magnitude_scaling_relationship.__class__.__name__

		ruptAspectRatio_elem = etree.SubElement(areaSource_elem, ns.RUPTURE_ASPECT_RATIO)
		ruptAspectRatio_elem.text = str(self.rupture_aspect_ratio)

		areaSource_elem.append(self.mfd.create_xml_element(encoding=encoding))
		areaSource_elem.append(self.nodal_plane_distribution.create_xml_element(
																encoding=encoding))
		areaSource_elem.append(self.hypocenter_distribution.create_xml_element(
																encoding=encoding))

		return areaSource_elem

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
		Return list of longitudes in the source's Polygon object
		"""
		return self.polygon.lons

	@property
	def latitudes(self):
		"""
		Return list of latitudes in the source's Polygon object
		"""
		return self.polygon.lats

	def get_Johnston1994_mfd(self, min_mag=None, max_mag=None, bin_width=None,
							region="total"):
		"""
		Construct "minimum" MFD for SCR according to Johnston (1994),
		based on surface area

		:param min_mag:
			float, Minimum magnitude
			(default: None, take min_mag from current MFD).
		:param max_mag:
			Maximum magnitude
			(default: None, take max_mag from current MFD).
		:param bin_width:
			float, Magnitude interval for evenly discretized magnitude
			frequency distribution
			(default: None, take bin_width from current MFD.
		:param region:
			str, SCR region ("africa", "australia", "europe", "china", "india",
			"north america", "na extended", "na non-extended", "south america",
			"total", "total extended", or "total non-extended")
			(default: "total")

		:return:
			instance of :class:`TruncatedGRMFD`
		"""
		if bin_width is None:
			bin_width = self.mfd.bin_width
		if min_mag is None:
			min_mag = self.mfd.get_min_mag_edge()
		if max_mag is None:
			max_mag = self.mfd.max_mag

		return TruncatedGRMFD.construct_Johnston1994_mfd(min_mag, max_mag,
											bin_width, self.get_area(), region)

	def get_FentonEtAl2006_mfd(self, min_mag=None, max_mag=None, bin_width=None,
								b_val=0.7991):
		"""
		Construct "minimum" MFD for SCR according to Fenton et al. (2006),
		based on surface area

		:param min_mag:
			float, Minimum magnitude
			(default: None, take min_mag from current MFD).
		:param max_mag:
			Maximum magnitude
			(default: None, take max_mag from current MFD).
		:param bin_width:
			float, Magnitude interval for evenly discretized magnitude
			frequency distribution
			(default: None, take bin_width from current MFD.
		:param b_val:
			float, Parameter of the truncated Gutenberg-Richter model.
			(default: 0.7991)

		:return:
			instance of :class:`TruncatedGRMFD`
		"""
		if bin_width is None:
			bin_width = self.mfd.bin_width
		if min_mag is None:
			min_mag = self.mfd.get_min_mag_edge()
		if max_mag is None:
			max_mag = self.mfd.max_mag

		return TruncatedGRMFD.construct_FentonEtAl2006_mfd(min_mag, max_mag,
											bin_width, self.get_area(), b_val)

	def to_ogr_geometry(self):
		"""
		Create OGR Geometry object
		"""
		import osr, ogr

		## Construct WGS84 projection system corresponding to earthquake coordinates
		wgs84 = osr.SpatialReference()
		## Note: str for PY2/3 compatibility
		wgs84.SetWellKnownGeogCS(str("WGS84"))

		# Create ring
		ring = ogr.Geometry(ogr.wkbLinearRing)

		# Add points
		for lon, lat in zip(self.longitudes, self.latitudes):
			ring.AddPoint(lon, lat)

		# Create polygon
		poly = ogr.Geometry(ogr.wkbPolygon)
		poly.AssignSpatialReference(wgs84)
		poly.AddGeometry(ring)
		poly.CloseRings()

		return poly

	def get_area(self):
		"""
		Compute area of source in square km

		:return:
			float, source area in square km
		"""
		import osr
		from mapping.geotools.coordtrans import WGS84, get_utm_spec, get_utm_srs

		poly = self.to_ogr_geometry()
		centroid = poly.Centroid()
		utm_spec = get_utm_spec(centroid.GetX(), centroid.GetY())
		utm_srs = get_utm_srs(utm_spec)
		coordTrans = osr.CoordinateTransformation(WGS84, utm_srs)
		poly.Transform(coordTrans)
		return poly.GetArea() / 1E6

	def get_centroid(self):
		"""
		Compute centroid of area source

		:return:
			(float, float) tuple: longitude, latitude of centroid
		"""
		centroid = self.to_ogr_geometry().Centroid()
		return (centroid.GetX(), centroid.GetY())

	def get_discretized_mesh(self):
		"""
		Compute discretized mesh

		:return:
			instance of :class:`openquake.hazardlib.geo.Mesh`
		"""
		mesh = self.polygon.discretize(self.area_discretization)
		return mesh

	def get_discretized_locations(self):
		"""
		Compute discretized locations:

		:return:
			list of (lon, lat) tuples
		"""
		mesh = self.get_discretized_mesh()
		return [(pt.longitude, pt.latitude) for pt in mesh]

	def get_bounding_box(self):
		"""
		Determine rectangular bounding box

		:return:
			(west, east, south, north) tuple
		"""
		w = self.longitudes.min()
		e = self.longitudes.max()
		s = self.latitudes.min()
		n = self.latitudes.max()
		return (w, e, s, n)

	def get_normalized_a_val(self, area=1E+5):
		"""
		Report a value normalized by area

		:param area:
			normalized surface area in sq. km
			(default: 1E+5)
		"""
		return self.mfd.a_val + np.log10(area / self.get_area())

	def to_point_sources(self):
		"""
		Decompose area source into point sources.

		:return:
			list with instances of :class:`PointSource`. Name and source_id
			will be identical to that of the area source.
		"""
		polygon_mesh = self.polygon.discretize(self.area_discretization)
		rate_scaling_factor = 1.0 / len(polygon_mesh)
		fmt = "_#%%0%dd" % len(str(len(polygon_mesh)))
		point_sources = []
		for i, epicenter in enumerate(polygon_mesh):
			mfd = self.mfd * rate_scaling_factor
			source_id = self.source_id + fmt % i
			ptsrc = PointSource(source_id, self.name, self.tectonic_region_type,
								mfd, self.rupture_mesh_spacing,self.magnitude_scaling_relationship,
								self.rupture_aspect_ratio, self.upper_seismogenic_depth,
								self.lower_seismogenic_depth, epicenter,
								self.nodal_plane_distribution, self.hypocenter_distribution,
								timespan=self.timespan)
			point_sources.append(ptsrc)
		return point_sources

	def get_seismogenic_thickness(self):
		"""
		Return seismogenic thickness in km
		"""
		return self.lower_seismogenic_depth - self.upper_seismogenic_depth

	def get_moment_rate_from_strain_rate(self, strain_rate, rigidity=3E+10):
		"""
		Given the strain rate, determine the corresponding moment rate
		in the area source according to the Kostrov formula

		:param strain_rate:
			float, strain rate in 1/yr
		:param rigidity:
			rigidity or shear modulus in N/m**2 or Pascal
			(default: 3E+10)

		:return:
			float, moment rate in N.m/yr
		"""
		return (strain_rate * 2 * rigidity * self.get_area() * 1E+6
				* self.get_seismogenic_thickness() * 1E+3)

	def get_Mmax_from_strain_rate(self, strain_rate, rigidity=3E+10):
		"""
		Determine maximum possible magnitude that is in agreement with
		the MFD of the source and a given moment rate
		Note: MFD of source must be truncated Gutenberg-Richter MFD !

		:param strain_rate:
			float, strain rate in 1/yr
		:param rigidity:
			rigidity or shear modulus in N/m**2 or Pascal
			(default: 3E+10)

		:return:
			float, maximum magnitude
		"""
		assert isinstance(self.mfd, TruncatedGRMFD)
		moment_rate = self.get_moment_rate_from_strain_rate(strain_rate, rigidity)
		Mmax = self.mfd.get_Mmax_from_moment_rate(moment_rate)
		return Mmax

	def to_lbm_data(self):
		"""
		Convert to layeredbasemap polygon

		:return:
			layeredbasemap PolygonData object
		"""
		import mapping.layeredbasemap as lbm

		pg = lbm.PolygonData(self.longitudes, self.latitudes, label=self.source_id)
		pg.value = {}
		attrs = ("source_id", "name", "tectonic_region_type", "rupture_mesh_spacing",
				"magnitude_scaling_relationship", "rupture_aspect_ratio",
				"upper_seismogenic_depth", "lower_seismogenic_depth",
				"area_discretization")
		for attr in attrs:
			setattr(pg, attr, getattr(self, attr))
		return pg

	def contains_sites(self, sites):
		"""
		Determine whether given sites are located in this area source

		:param sites:
			instance of :class:`rshalib.site.SoilSiteModel`
			or list with instances of :class:`rshalib.site.SHASite`

		:return:
			array of bool values
		"""
		from ..site import SoilSiteModel
		if isinstance(sites, SoilSiteModel):
			mesh = site_model.mesh
		else:
			lons = np.array([site.longitude for site in sites])
			lats = np.array([site.latitude for site in sites])
			depths = np.array([site.depth for site in sites])
			mesh = oqhazlib.geo.mesh.Mesh(lons, lats, depths)
		return self.polygon.intersects(mesh)



if __name__ == '__main__':
	pass
