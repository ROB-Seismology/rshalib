# -*- coding: utf-8 -*-

"""
PointSource class
"""

from __future__ import absolute_import, division, print_function, unicode_literals


## Note: Don't use np as abbreviation for nodalplane!!
import numpy as np
import pylab

from .. import (oqhazlib, OQ_VERSION)

from ..msr import get_oq_msr
from ..mfd import (TruncatedGRMFD, EvenlyDiscretizedMFD)
from ..geo import Point, NodalPlane
from ..pmf import HypocentralDepthDistribution, NodalPlaneDistribution
from .rupture_source import RuptureSource



__all__ = ['PointSource']


class PointSource(oqhazlib.source.PointSource, RuptureSource):
	"""
	Class representing a point source, corresponding to a single
	geographic site

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
	:param upper_seismogenic_depth:
		Minimum depth an earthquake rupture can reach, in km.
	:param lower_seismogenic_depth:
		Maximum depth an earthquake rupture can reach, in km.
	:param location:
		:class:`Point` object representing the location of the seismic source.
		The depth value of that point is ignored.
	:param nodal_plane_distribution:
		:class:`NodalPlaneDistribution` object with values that are instances
		of :class:`NodalPlane`. Shows the distribution of probability for
		rupture to have the certain nodal plane.
	:param hypocenter_distribution:
		:class:`HypocentralDepthDistribution` with values being float numbers
		in km representing the depth of the hypocenter. Latitude and longitude
		of the hypocenter is always set to ones of ``location``.
	:param timespan:
		float, timespan for Poisson temporal occurrence model.
		Introduced in more recent versions of OpenQuake
		(default: 1)
	"""
	def __init__(self, source_id, name, tectonic_region_type, mfd,
				rupture_mesh_spacing, magnitude_scaling_relationship,
				rupture_aspect_ratio,
				upper_seismogenic_depth, lower_seismogenic_depth, location,
				nodal_plane_distribution, hypocenter_distribution,
				timespan=1):
		"""
		"""
		self.timespan = timespan
		magnitude_scaling_relationship = get_oq_msr(magnitude_scaling_relationship)

		## OQ version dependent keyword arguments
		oqver_kwargs = {}
		if OQ_VERSION >= '2.9.0':
			oqver_kwargs['temporal_occurrence_model'] = self.tom
		super(PointSource, self).__init__(source_id=source_id,
				name=name,
				tectonic_region_type=tectonic_region_type,
				mfd=mfd,
				rupture_mesh_spacing=rupture_mesh_spacing,
				magnitude_scaling_relationship=magnitude_scaling_relationship,
				rupture_aspect_ratio=rupture_aspect_ratio,
				upper_seismogenic_depth=upper_seismogenic_depth,
				lower_seismogenic_depth=lower_seismogenic_depth,
				location=location,
				nodal_plane_distribution=nodal_plane_distribution,
				hypocenter_distribution=hypocenter_distribution,
				**oqver_kwargs)

	def __repr__(self):
		return '<PointSource #%s>' % self.source_id

	@property
	def tom(self):
		"""
		Temporal occurrence model
		"""
		return oqhazlib.tom.PoissonTOM(self.timespan)

	@classmethod
	def from_eq_record(cls, eq, Mtype="MW", Mrelation={},
				tectonic_region_type="Stable Shallow Crust",
				magnitude_scaling_relationship=oqhazlib.scalerel.WC1994(),
				rupture_mesh_spacing=1., rupture_aspect_ratio=1.,
				upper_seismogenic_depth=5., lower_seismogenic_depth=25.,
				nodal_plane_distribution=None, hypocenter_distribution=None,
				synthetic=False, timespan=1):
		"""
		Construct point source from earthquake object

		Note:
		order of precedence for hypocenter_distribution:
		- earthquake depth, if is not zero and lies between upper and
		lower seismogenic depth, or
		- provided hypocenter_distribution, or
		- average of lower and upper seismogenic depth

		order of precedence of nodal_plane_distribution is:
		- earthquake focal mechanism in ROB database, or
		- provided nodal_plane_distribution, or
		- N-S and E-W planes dipping 45 degrees and with strike-slip
		style of faulting


		:param eq:
			instance of :class:`LocalEarthquake`

		:param nodal_plane_distribution:
			if eq does not have a focal mechanism, use this distribution.

		...

		:param synthetic:
			bool, whether catalog is synthetic or not, to avoid lookup
			in focal mechanisms database
			(default: False)

		:return:
			instance of :class:`PointSource`
		"""
		source_id = eq.ID
		name = eq.name
		location = Point(eq.lon, eq.lat)
		M = np.round(eq.get_M(Mtype, Mrelation), decimals=1)
		mfd = EvenlyDiscretizedMFD(M, 0.1, [1.], Mtype)
		depth = eq.depth
		if depth and upper_seismogenic_depth < depth < lower_seismogenic_depth:
			#depth = min(lower_seismogenic_depth, max(5, eq.depth))
			hdd = HypocentralDepthDistribution([depth], [1])
		elif hypocenter_distribution:
			hdd = hypocenter_distribution
		else:
			depth = (lower_seismogenic_depth + upper_seismogenic_depth) / 2.
			hdd = HypocentralDepthDistribution([depth], [1])
		if not synthetic:
			focmec_rec = eq.get_focal_mechanism()
		else:
			focmec_rec = False
		if focmec_rec:
			focmec = focmec_rec.get_focmec()
			np1 = NodalPlane(*[round(angle.deg()) for angle in focmec.sdr1()])
			np2 = NodalPlane(*[round(angle.deg()) for angle in focmec.sdr2()])
			npd = NodalPlaneDistribution([np1, np2], [0.5] * 2)
		elif nodal_plane_distribution:
			npd = nodal_plane_distribution
		else:
			np1 = NodalPlane(0, 45, 0)
			np2 = NodalPlane(90, 45, 0)
			np3 = NodalPlane(180, 45, 0)
			np4 = NodalPlane(270, 45, 0)
			npd = NodalPlaneDistribution([np1, np2, np3, np4], [0.25] * 4)

		return cls(source_id, name, tectonic_region_type, mfd,
					rupture_mesh_spacing, magnitude_scaling_relationship,
					rupture_aspect_ratio, upper_seismogenic_depth,
					lower_seismogenic_depth, location, npd, hdd,
					timespan=timespan)

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML pointSource element)

		:param encoding:
			unicode encoding
			(default: 'latin1')
		"""
		from lxml import etree
		from ..nrml import ns
		from ..nrml.common import xmlstr

		#TODO: temporal_occurrence_model in recent versions of OQ??
		pointSource_elem = etree.Element(ns.POINT_SOURCE)
		pointSource_elem.set(ns.ID, xmlstr(self.source_id, encoding=encoding))
		pointSource_elem.set(ns.NAME, xmlstr(self.name, encoding=encoding))
		pointSource_elem.set(ns.TECTONIC_REGION_TYPE,
							xmlstr(self.tectonic_region_type, encoding=encoding))

		pointGeometry_elem = etree.SubElement(pointSource_elem, ns.POINT_GEOMETRY)
		pointGeometry_elem.append(self.location.create_xml_element(encoding=encoding))
		upperSeismoDepth_elem = etree.SubElement(pointGeometry_elem,
												ns.UPPER_SEISMOGENIC_DEPTH)
		upperSeismoDepth_elem.text = str(self.upper_seismogenic_depth)
		lowerSeismoDepth_elem = etree.SubElement(pointGeometry_elem,
												ns.LOWER_SEISMOGENIC_DEPTH)
		lowerSeismoDepth_elem.text = str(self.lower_seismogenic_depth)

		magScaleRel_elem = etree.SubElement(pointSource_elem,
											ns.MAGNITUDE_SCALING_RELATIONSHIP)
		magScaleRel_elem.text = self.magnitude_scaling_relationship.__class__.__name__

		ruptAspectRatio_elem = etree.SubElement(pointSource_elem, ns.RUPTURE_ASPECT_RATIO)
		ruptAspectRatio_elem.text = str(self.rupture_aspect_ratio)

		pointSource_elem.append(self.mfd.create_xml_element(encoding=encoding))
		pointSource_elem.append(self.nodal_plane_distribution.create_xml_element(
																encoding=encoding))
		pointSource_elem.append(self.hypocenter_distribution.create_xml_element(
																encoding=encoding))

		return pointSource_elem

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
		Return list of longitudes in the source's Location object
		"""
		return [self.location.longitude]

	@property
	def latitudes(self):
		"""
		Return list of latitudes in the source's Location object
		"""
		return [self.location.latitude]

	def to_ogr_geometry(self):
		"""
		Create OGR Geometry object
		"""
		import osr, ogr

		## Construct WGS84 projection system corresponding to earthquake coordinates
		wgs84 = osr.SpatialReference()
		wgs84.SetWellKnownGeogCS(bytes("WGS84"))

		## Create point object
		point = ogr.Geometry(ogr.wkbPoint)
		point.AssignSpatialReference(wgs84)

		## Set point coordinates
		point.SetPoint(0, self.location.longitude, self.location.latitude)

		return point

	def get_centroid(self):
		"""
		Compute centroid

		:return:
			(float, float) tuple: longitude, latitude of centroid
		"""
		return (self.location.longitude, self.location.latitude)

	def get_bounding_box(self):
		"""
		Determine rectangular bounding box

		:return:
			(west, east, south, north) tuple
		"""
		w = self.location.longitude
		e = self.location.longitude
		s = self.location.latitude
		n = self.location.latitude
		return (w, e, s, n)

	def get_seismogenic_thickness(self):
		"""
		Return seismogenic thickness in km
		"""
		return self.lower_seismogenic_depth - self.upper_seismogenic_depth

	def to_lbm_data(self):
		"""
		Convert to layeredbasemap point

		:return:
			layeredbasemap PointData object
		"""
		import mapping.layeredbasemap as lbm

		pt = lbm.PointData(self.location.longitude, self.location.latitude,
							label=self.source_id)
		attrs = ("source_id", "name", "tectonic_region_type",
				"rupture_mesh_spacing", "magnitude_scaling_relationship",
				"rupture_aspect_ratio", "upper_seismogenic_depth",
				"lower_seismogenic_depth")
		for attr in attrs:
			setattr(pt, attr, getattr(self, attr))
		return pt



if __name__ == '__main__':
	pass
