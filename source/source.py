# -*- coding: utf-8 -*-

"""
Classes representing source-model elements in Openquake/nhlib. Where possible,
the classes are inherited from nhlib classes. All provide methods to create
XML elements, which are used to write a complete source-model NRML file.
Thus, objects instantiated from these classes can be used directly in nhlib,
as well as to generate input files for OpenQuake.
"""

from lxml import etree
## Note: Don't import numpy as np to avoid conflicts with np as abbreviation
## for nodalplane
import numpy

import openquake.hazardlib as nhlib

from ..nrml import ns
from ..nrml.common import *
from ..mfd import *
from ..geo.angle import *
from ..geo import Point, Line, Polygon


class PointSource(nhlib.source.PointSource):
	"""
	Class representing a point source, corresponding to a single geographic site

	:param source_id:
		source identifier
	:param name:
		full name of source
	:param tectonic_region_type:
		tectonic region type known to nhlib/OQ. See :class:`nhlib.const.TRT`
	:param mfd:
		magnitude-frequency distribution (instance of :class:`TruncatedGRMFD`
		or :class:`EvenlyDiscretizedMFD`)
	:param rupture_mesh_spacing:
		The desired distance between two adjacent points in source's
		ruptures' mesh, in km. Mainly this parameter allows to balance
		the trade-off between time needed to compute the :meth:`distance
		<nhlib.geo.surface.base.BaseSurface.get_min_distance>` between
		the rupture surface and a site and the precision of that computation.
	:param magnitude_scaling_relationship:
		Instance of subclass of :class:`nhlib.scalerel.base.BaseMSR` to
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
	"""
	def __init__(self, source_id, name, tectonic_region_type, mfd, rupture_mesh_spacing, magnitude_scaling_relationship, rupture_aspect_ratio, upper_seismogenic_depth, lower_seismogenic_depth, location, nodal_plane_distribution, hypocenter_distribution):
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
				hypocenter_distribution=hypocenter_distribution)

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML pointSource element)

		:param encoding:
			unicode encoding (default: 'latin1')
		"""
		pointSource_elem = etree.Element(ns.POINT_SOURCE)
		pointSource_elem.set(ns.ID, xmlstr(self.source_id, encoding=encoding))
		pointSource_elem.set(ns.NAME, xmlstr(self.name, encoding=encoding))
		pointSource_elem.set(ns.TECTONIC_REGION_TYPE, xmlstr(self.tectonic_region_type, encoding=encoding))

		pointGeometry_elem = etree.SubElement(pointSource_elem, ns.POINT_GEOMETRY)
		pointGeometry_elem.append(self.location.create_xml_element(encoding=encoding))
		upperSeismoDepth_elem = etree.SubElement(pointGeometry_elem, ns.UPPER_SEISMOGENIC_DEPTH)
		upperSeismoDepth_elem.text = str(self.upper_seismogenic_depth)
		lowerSeismoDepth_elem = etree.SubElement(pointGeometry_elem, ns.LOWER_SEISMOGENIC_DEPTH)
		lowerSeismoDepth_elem.text = str(self.lower_seismogenic_depth)

		magScaleRel_elem = etree.SubElement(pointSource_elem, ns.MAGNITUDE_SCALING_RELATIONSHIP)
		magScaleRel_elem.text = self.magnitude_scaling_relationship.__class__.__name__

		ruptAspectRatio_elem = etree.SubElement(pointSource_elem, ns.RUPTURE_ASPECT_RATIO)
		ruptAspectRatio_elem.text = str(self.rupture_aspect_ratio)

		pointSource_elem.append(self.mfd.create_xml_element(encoding=encoding))
		pointSource_elem.append(self.nodal_plane_distribution.create_xml_element(encoding=encoding))
		pointSource_elem.append(self.hypocenter_distribution.create_xml_element(encoding=encoding))

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


class AreaSource(nhlib.source.AreaSource):
	"""
	Class representing an area source, i.e. a polygonal geographical region
	where seismicity is assumed to be uniform.
	An area source is similar to a point source, except for the geometry

	:param source_id:
		source identifier
	:param name:
		full name of source
	:param tectonic_region_type:
		tectonic region type known to nhlib/OQ. See :class:`nhlib.const.TRT`
	:param mfd:
		magnitude-frequency distribution (instance of :class:`TruncatedGRMFD`
		or :class:`EvenlyDiscretizedMFD`)
	:param rupture_mesh_spacing:
		The desired distance between two adjacent points in source's
		ruptures' mesh, in km. Mainly this parameter allows to balance
		the trade-off between time needed to compute the :meth:`distance
		<nhlib.geo.surface.base.BaseSurface.get_min_distance>` between
		the rupture surface and a site and the precision of that computation.
	:param magnitude_scaling_relationship:
		Instance of subclass of :class:`nhlib.scalerel.base.BaseMSR` to
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
		An instance of :class:`nhlib.geo.polygon.Polygon` that defines
		source's area.
	:param area_discretization:
		Float number, polygon area discretization spacing in kilometers.
		See :meth:`iter_ruptures`.
	"""

	def __init__(self, source_id, name, tectonic_region_type, mfd, rupture_mesh_spacing, magnitude_scaling_relationship, rupture_aspect_ratio, upper_seismogenic_depth, lower_seismogenic_depth, nodal_plane_distribution, hypocenter_distribution, polygon, area_discretization):
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
				area_discretization=area_discretization)

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML areaSource element)

		:param encoding:
			unicode encoding (default: 'latin1')
		"""
		areaSource_elem = etree.Element(ns.AREA_SOURCE)
		areaSource_elem.set(ns.ID, xmlstr(self.source_id, encoding=encoding))
		areaSource_elem.set(ns.NAME, xmlstr(self.name, encoding=encoding))
		areaSource_elem.set(ns.TECTONIC_REGION_TYPE, xmlstr(self.tectonic_region_type, encoding=encoding))

		areaGeometry_elem = etree.SubElement(areaSource_elem, ns.AREA_GEOMETRY)
		areaGeometry_elem.append(self.polygon.create_xml_element(encoding=encoding))
		upperSeismoDepth_elem = etree.SubElement(areaGeometry_elem, ns.UPPER_SEISMOGENIC_DEPTH)
		upperSeismoDepth_elem.text = str(self.upper_seismogenic_depth)
		lowerSeismoDepth_elem = etree.SubElement(areaGeometry_elem, ns.LOWER_SEISMOGENIC_DEPTH)
		lowerSeismoDepth_elem.text = str(self.lower_seismogenic_depth)

		magScaleRel_elem = etree.SubElement(areaSource_elem, ns.MAGNITUDE_SCALING_RELATIONSHIP)
		magScaleRel_elem.text = self.magnitude_scaling_relationship.__class__.__name__

		ruptAspectRatio_elem = etree.SubElement(areaSource_elem, ns.RUPTURE_ASPECT_RATIO)
		ruptAspectRatio_elem.text = str(self.rupture_aspect_ratio)

		areaSource_elem.append(self.mfd.create_xml_element(encoding=encoding))
		areaSource_elem.append(self.nodal_plane_distribution.create_xml_element(encoding=encoding))
		areaSource_elem.append(self.hypocenter_distribution.create_xml_element(encoding=encoding))

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


class SimpleFaultSource(nhlib.source.SimpleFaultSource):
	"""
	Class representing a simple fault source.

	:param source_id:
		source identifier
	:param name:
		full name of source
	:param tectonic_region_type:
		tectonic region type known to nhlib/OQ. See :class:`nhlib.const.TRT`
	:param mfd:
		magnitude-frequency distribution (instance of :class:`TruncatedGRMFD`
		or :class:`EvenlyDiscretizedMFD`)
	:param rupture_mesh_spacing:
		The desired distance between two adjacent points in source's
		ruptures' mesh, in km. Mainly this parameter allows to balance
		the trade-off between time needed to compute the :meth:`distance
		<nhlib.geo.surface.base.BaseSurface.get_min_distance>` between
		the rupture surface and a site and the precision of that computation.
	:param magnitude_scaling_relationship:
		Instance of subclass of :class:`nhlib.scalerel.base.BaseMSR` to
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
	:param fault_trace:
		An instance of :class:`Line` representing the line of intersection
		between the fault plane and the Earth's surface.
	:param dip:
		Angle between earth surface and fault plane in decimal degrees.
	:param rake:
		Angle describing fault rake in decimal degrees.
	:param slip_rate:
		Slip rate in mm/yr (default: NaN)
	:param bg_zone:
		String, ID of background zone (default: None)
	"""
	def __init__(self, source_id, name, tectonic_region_type, mfd, rupture_mesh_spacing, magnitude_scaling_relationship, rupture_aspect_ratio, upper_seismogenic_depth, lower_seismogenic_depth, fault_trace, dip, rake, slip_rate=numpy.NaN, bg_zone=None):
		super(SimpleFaultSource, self).__init__(source_id=source_id,
				name=name,
				tectonic_region_type=tectonic_region_type,
				mfd=mfd,
				rupture_mesh_spacing=rupture_mesh_spacing,
				magnitude_scaling_relationship=magnitude_scaling_relationship,
				rupture_aspect_ratio=rupture_aspect_ratio,
				upper_seismogenic_depth=upper_seismogenic_depth,
				lower_seismogenic_depth=lower_seismogenic_depth,
				fault_trace=fault_trace,
				dip=dip,
				rake=rake)
		self.slip_rate = slip_rate
		self.bg_zone = bg_zone

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML simpleFaultSource element)

		:param encoding:
			unicode encoding (default: 'latin1')
		"""
		sfs_elem = etree.Element(ns.SIMPLE_FAULT_SOURCE)
		sfs_elem.set(ns.ID, xmlstr(self.source_id, encoding=encoding))
		sfs_elem.set(ns.NAME, xmlstr(self.name, encoding=encoding))
		sfs_elem.set(ns.TECTONIC_REGION_TYPE, xmlstr(self.tectonic_region_type, encoding=encoding))

		sfg_elem = etree.SubElement(sfs_elem, ns.SIMPLE_FAULT_GEOMETRY)
		sfg_elem.append(self.fault_trace.create_xml_element(encoding=encoding))
		dip_elem = etree.SubElement(sfg_elem, ns.DIP)
		dip_elem.text = str(self.dip)
		usd_elem = etree.SubElement(sfg_elem, ns.UPPER_SEISMOGENIC_DEPTH)
		usd_elem.text = str(self.upper_seismogenic_depth)
		lsd_elem = etree.SubElement(sfg_elem, ns.LOWER_SEISMOGENIC_DEPTH)
		lsd_elem.text = str(self.lower_seismogenic_depth)

		magScaleRel_elem = etree.SubElement(sfs_elem, ns.MAGNITUDE_SCALING_RELATIONSHIP)
		magScaleRel_elem.text = self.magnitude_scaling_relationship.__class__.__name__

		ruptAspectRatio_elem = etree.SubElement(sfs_elem, ns.RUPTURE_ASPECT_RATIO)
		ruptAspectRatio_elem.text = str(self.rupture_aspect_ratio)

		sfs_elem.append(self.mfd.create_xml_element(encoding=encoding))

		rake_elem = etree.SubElement(sfs_elem, ns.RAKE)
		rake_elem.text = str(self.rake)

		return sfs_elem

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
		Return list of longitudes in the source's SimpleFaultGeometry object
		"""
		return self.fault_trace.lons

	@property
	def latitudes(self):
		"""
		Return list of latitudes in the source's SimpleFaultGeometry object
		"""
		return self.fault_trace.lats

	def get_depth_range(self):
		"""
		Compute depth range in km

		:return:
			Float, depth range in km
		"""
		return self.lower_seismogenic_depth - self.upper_seismogenic_depth

	def get_width(self):
		"""
		Compute fault width in km

		:return:
			Float, fault width in km

		Note: don't change to m, in order to keep compatibility with mtoolkit!
		"""
		return self.get_depth_range() / numpy.sin(numpy.radians(self.dip))

	def get_projected_width(self):
		"""
		Compute width of fault projected at the surface in km

		:return:
			Float, projected fault width in km
		"""
		return self.get_depth_range() * numpy.cos(numpy.radians(self.dip))

	def get_length(self):
		"""
		Compute length of fault in km

		:return:
			Float, fault length in km
		"""
		lons, lats = self.fault_trace.lons, self.fault_trace.lats
		lons1, lats1 = lons[:-1], lats[:-1]
		lons2, lats2 = lons[1:], lats[1:]
		distances = nhlib.geo.geodetic.geodetic_distance(lons1, lats1, lons2, lats2)
		return numpy.add.reduce(distances)

	def get_area(self):
		"""
		Compute area of fault in square km

		:return:
			Float, fault area in square km
		"""
		return self.get_length() * self.get_width()

	def _get_trace_azimuths_and_distances(self):
		"""
		Compute point-to-point azimuth and distance along fault trace

		:return:
			Tuple (azimuths, distances):
			azimuths: numpy array with point-to-point azimuths in degrees
			distances: numpy array with point-to-point distances in km
		"""
		lons, lats = self.fault_trace.lons, self.fault_trace.lats
		lons1, lats1 = lons[:-1], lats[:-1]
		lons2, lats2 = lons[1:], lats[1:]
		distances = nhlib.geo.geodetic.geodetic_distance(lons1, lats1, lons2, lats2)
		azimuths = nhlib.geo.geodetic.azimuth(lons1, lats1, lons2, lats2)
		return (azimuths, distances)

	def get_mean_strike(self):
		"""
		Compute mean strike in degrees, supposing fault trace points are ordered
		in the opposite direction (left-hand rule).

		:return:
			Float, mean strike in degrees
		"""
		azimuths, distances = self._get_trace_azimuths_and_distances()
		weights = distances / numpy.add.reduce(distances)
		mean_strike = mean_angle(azimuths, weights)
		## Constrain to (0, 360)
		mean_strike = (mean_strike + 360) % 360
		#mean_strike = numpy.average(azimuths, weights=weights)
		return mean_strike

	def get_std_strike(self):
		"""
		Compute standard deviation of strike from surface trace

		:return:
			Float, standard deviation of strike in degrees
		"""
		mean_strike = self.get_mean_strike()
		#mean_strike_angle = 90 - mean_strike
		azimuths, distances = self._get_trace_azimuths_and_distances()
		weights = distances / numpy.add.reduce(distances)
		azimuth_deltas = [delta_angle(mean_strike, az) for az in azimuths]

		variance = numpy.dot(weights, numpy.array(azimuth_deltas)**2) / weights.sum()
		return numpy.sqrt(variance)

	def get_min_max_strike(self):
		"""
		Compute minimum and maximum strike from surface trace

		:return:
			Tuple (min_strike, max_strike) in degrees
		"""
		azimuths, _ = self._get_trace_azimuths_and_distances()
		az_min, az_max = azimuths.min(), azimuths.max()
		if az_max - az_min > 180:
			tmpAzimuths = numpy.array(azimuths)
			for i in range(len(tmpAzimuths)):
				if tmpAzimuths[i] <= 180:
					tmpAzimuths[i] += 360
			az_min, az_max = min(tmpAzimuths), max(tmpAzimuths)
			az_max -= 360
		return (az_min, az_max)

	def get_polygon(self):
		"""
		Construct polygonal outline of fault

		:return:
			instance of :class:`Polygon`
		"""
		width = self.get_projected_width()
		mean_strike = self.get_mean_strike()
		perpendicular_direction = mean_strike + 90.
		z0, z1 = self.upper_seismogenic_depth, self.lower_seismogenic_depth
		top_edge = [Point(pt.longitude, pt.latitude, z0) for pt in self.fault_trace]
		bottom_edge = []
		for pt in top_edge:
			lon, lat = nhlib.geo.geodetic.point_at(pt[0], pt[1], perpendicular_direction, width)
			bottom_edge.append(Point(lon, lat, z1))
		bottom_edge.reverse()
		return Polygon(top_edge + bottom_edge)

	def calc_Mmax_Wells_Coppersmith(self):
		"""
		Compute Mmax from Wells & Coppersmith (1994) scaling relation

		:return:
			Float, maximum magnitude
		"""
		from nhlib.scalerel.wc1994 import WC1994
		wc = WC1994()
		max_mag = wc.get_median_mag(self.get_area(), self.rake)
		return max_mag

	def get_MFD_characteristic(self, bin_width=None, M_sigma=0.03, num_sigma=0):
		"""
		Construct MFD corresponding to a characteristic Mmax
		:param bin_width:
			Float, Magnitude interval for evenly discretized magnitude frequency
			distribution (default: None, take bin_width from current MFD.
		:param M_sigma:
			Float, standard deviation on magnitude (default: 0.3)
		:param num_sigma:
			Float, number of standard deviations to spread occurrence rates over
			(default: 0)

		:return:
			instance of :class:`EvenlyDiscretizedMFD`
		"""
		if bin_width is None:
			bin_width = self.mfd.bin_width

		max_mag = self.max_mag - bin_width
		return_period = self.get_Mmax_return_period()
		MFD = CharacteristicMFD(max_mag, return_period, bin_width, M_sigma=M_sigma, num_sigma=num_sigma)
		return MFD

	def get_MFD_Anderson_Luco(self, min_mag=None, max_mag=None, bin_width=None, b_val=None, aseismic_coef=0., strain_drop=None, mu=3E+10, arbitrary_surface=False):
		"""
		Compute MFD according to Anderson & Luco (1983), based on slip rate

		:param min_mag:
			Float, Minimum magnitude (default: None, take min_mag from current MFD).
		:param max_mag:
			Maximum magnitude (default: None, take max_mag from current MFD).
		:param bin_width:
			Float, Magnitude interval for evenly discretized magnitude frequency
			distribution (default: None, take bin_width from current MFD.
		:param b_val:
			Float, Parameter of the truncated gutenberg richter model.
			(default: None, take b_val from current MFD)
		:param aseismic_coef:
			Float, The proportion of the fault slip that is released aseismically.
			(default: 0.)
		:param strain_drop:
			Float, strain drop (dimensionless), this is the ratio between
			rupture displacement and rupture length. If not provided, it will
			be computed from Mmax (default: None)
		:param mu:
			rigidity or shear modulus in N/m**2 (default: 3E+10)
		:param arbitrary_surface:
			Boolean indicating whether rupture surface is arbitrary or
			corresponds to max_mag (default: False)

		:return:
			instance of :class:`EvenlyDiscretizedMFD`
		"""
		if b_val is None:
			b_val = self.mfd.b_val
		if bin_width is None:
			bin_width = self.mfd.bin_width
		if min_mag is None:
			min_mag = self.mfd.get_min_mag_edge()
		if max_mag is None:
			#max_mag = self.mfd.max_mag - bin_width / 2.
			max_mag = self.mfd.max_mag
		if not strain_drop:
			strain_drop = self.get_Mmax_strain_drop()
		## For seismic moment in units of dyn-cm
		c, d = 16.05, 1.5

		## Using mtoolkit, returns EvenlyDiscretizedMFD
		#from mtoolkit.scientific.fault_calculator import get_mfd, MOMENT_SCALING
		#from mtoolkit.geo.tectonic_region import TectonicRegionBuilder
		#moment_scaling = (c, d)
		#min_mag = min_mag + bin_width / 2.
		#trt = TectonicRegionBuilder.create_tect_region_by_name(self.tectonic_region_type)
		## Override displacement-length ratio with actual value computed from Mmax
		#trt._dlr = {'value': [strain_drop], 'weight': [1.0]}
		#trt._smod = {'value': [mu/1E+9], 'weight': [1.0]}
		#occurrence_rates = get_mfd(self.slip_rate, aseismic_coef, trt, self, b_val, min_mag, bin_width, max_mag, self.rake, moment_scaling, arbitrary_surface)
		#return EvenlyDiscretizedMFD(min_mag, bin_width, occurrence_rates)

		dbar = d * np.log(10.0)
		bbar = b_val * np.log(10.0)
		seismic_slip = self.slip_rate * (1.0 - aseismic_coef)

		## Compute cumulative value for M=0
		mag_value = 0
		if arbitrary_surface:
			max_mag_moment = 10 ** ((max_mag + 10.73) * 1.5)
			arbitrary_surface_param = max_mag_moment / ((mu * 10) * (self.get_area() * 1E10))
			N0 = (((dbar - bbar) / (dbar)) * ((seismic_slip / 10.) / arbitrary_surface_param) *
						np.exp(bbar * (max_mag - mag_value)))
		else:
			beta = np.sqrt(strain_drop * (10.0 ** c) / ((mu * 10) * (self.get_width() * 1E5)))
			N0 = (((dbar - bbar) / (dbar)) * ((seismic_slip / 10.) / beta) *
						np.exp(bbar * (max_mag - mag_value)) *
						np.exp(-(dbar / 2.) * max_mag))

		a_val = np.log10(N0)
		return TruncatedGRMFD(min_mag, max_mag, bin_width, a_val, b_val)

	def get_MFD_Youngs_Coppersmith(self, min_mag=None, bin_width=None, b_val=None):
		"""
		Compute MFD according to Youngs & Coppersmith (1985), based on
		frequency of Mmax

		:param min_mag:
			Float, Minimum magnitude (default: None, take min_mag from current MFD).
		:param bin_width:
			Float, Magnitude interval for evenly discretized magnitude frequency
			distribution (default: None, take bin_width from current MFD).
		:param b_val:
			Float, Parameter of the truncated gutenberg richter model.
			(default: None, take b_val from current MFD)

		:return:
			instance of :class:`YoungsCoppersmith1985MFD`
		"""
		if b_val is None:
			b_val = self.mfd.b_val
		if bin_width is None:
			bin_width = self.mfd.bin_width
		if min_mag is None:
			min_mag = self.mfd.get_min_mag_center()
		char_mag = self.max_mag
		char_rate = 1. / self.get_Mmax_return_period()
		MFD = YoungsCoppersmith1985MFD.from_characteristic_rate(min_mag, b_val, char_mag, char_rate, bin_width)
		return MFD

	def get_moment_rate(self, mu=3E+10):
		"""
		Compute moment rate from slip rate and fault dimensions

		:param mu:
			rigidity or shear modulus in N/m**2 (default: 3E+10)

		:return:
			Float, moment rate in N.m/yr
		"""
		return mu * self.get_area() * 1E+6 * self.slip_rate * 1E-3

	def get_Mmax_moment(self):
		"""
		Compute seismic moment corresponding to maximum magnitude,
		assuming Mmax corresponds to max_mag of MFD minus one bin width

		:return:
			Float, seismic moment in N.m
		"""
		return 10 ** (1.5 * ((self.max_mag - self.mfd.bin_width) + 6.06))

	def get_Mmax_return_period(self, mu=3E+10):
		"""
		Compute predicted return period for Mmax based on slip rate
		assuming Mmax corresponds to max_mag of MFD minus one bin width

		:param mu:
			rigidity or shear modulus in N/m**2 (default: 3E+10)

		:return:
			Float, return period in yr
		"""
		return self.get_Mmax_moment() / self.get_moment_rate(mu=mu)

	def get_Mmax_moment_rate(self, mu=3E+10):
		"""
		Compute moment rate corresponding to maximum magnitude
		assuming Mmax corresponds to max_mag of MFD minus one bin width

		:param mu:
			rigidity or shear modulus in N/m**2 (default: 3E+10)

		:return:
			Float, moment rate in N.m/yr
		"""
		return self.get_Mmax_moment() / self.get_Mmax_return_period(mu=mu)

	def get_Mmax_slip(self, mu=3E+10):
		"""
		Compute slip corresponding to maximum magnitude, assuming it would
		occur over the entire fault plane, and assuming Mmax corresponds to
		max_mag of MFD minus one bin width

		:param mu:
			rigidity or shear modulus in N/m**2 (default: 3E+10)

		:return:
			Float, slip in m
		"""
		return self.get_Mmax_moment() / (mu * self.get_area() * 1E+6)

	def get_Mmax_strain_drop(self, mu=3E+10):
		"""
		Compute strain drop corresponding to maximum magnitude, assuming
		it would occur over the entire fault plane, and assuming Mmax
		corresponds to max_mag of MFD minus one bin width

		:return:
			Float, strain drop (dimensionless)
		"""
		return self.get_Mmax_slip(mu=mu) / (self.get_length() * 1E+3)

	def get_strike_slip_rate(self):
		"""
		Compute strike-slip component of slip rate.

		:return:
			Float, strike-slip component of slip rate (+ = left-lateral,
				- = right-lateral)
		"""
		return self.slip_rate * np.cos(np.radians(self.rake))

	def get_dip_slip_rate(self):
		"""
		Compute dip-slip component of slip rate.

		:return:
			Float, dip-slip component of slip rate (+ = reverse,
				- = normal)
		"""
		return self.slip_rate * np.sin(np.radians(self.rake))

	def get_vertical_dip_slip_rate(self):
		"""
		Compute vertical component of dip-slip component of slip rate.

		:return:
			Float, vertical component of dip-slip rate (+ = reverse,
				- = normal)
		"""
		return self.get_dip_slip_rate() * np.sin(np.radians(self.dip))

	def get_horizontal_dip_slip_rate(self):
		"""
		Compute horizontal component of dip-slip component of slip rate.

		:return:
			Float, vertical component of dip-slip rate (+ = shortening,
				- = extension)
		"""
		return self.get_dip_slip_rate() * np.cos(np.radians(self.dip))


class ComplexFaultSource(nhlib.source.ComplexFaultSource):
	"""
	Class representing a complex fault source.

	:param source_id:
		source identifier
	:param name:
		full name of source
	:param tectonic_region_type:
		tectonic region type known to nhlib/OQ. See :class:`nhlib.const.TRT`
	:param mfd:
		magnitude-frequency distribution (instance of :class:`TruncatedGRMFD`
		or :class:`EvenlyDiscretizedMFD`)
	:param rupture_mesh_spacing:
		The desired distance between two adjacent points in source's
		ruptures' mesh, in km. Mainly this parameter allows to balance
		the trade-off between time needed to compute the :meth:`distance
		<nhlib.geo.surface.base.BaseSurface.get_min_distance>` between
		the rupture surface and a site and the precision of that computation.
	:param magnitude_scaling_relationship:
		Instance of subclass of :class:`nhlib.scalerel.base.BaseMSR` to
		describe how does the area of the rupture depend on magnitude and rake.
	:param rupture_aspect_ratio:
		Float number representing how much source's ruptures are more wide
		than tall. Aspect ratio of 1 means ruptures have square shape,
		value below 1 means ruptures stretch vertically more than horizontally
		and vice versa.
	:param edges:
		A list of :class:`Line` objects, representing fault source geometry. See
		:meth:`nhlib.geo.surface.complex_fault.ComplexFaultSurface.from_fault_data`
	:param rake:
		Angle describing fault rake in decimal degrees.

	"""
	def __init__(self, source_id, name, tectonic_region_type, mfd, rupture_mesh_spacing, magnitude_scaling_relationship, rupture_aspect_ratio, edges, rake):
		super(ComplexFaultSource, self).__init__(source_id=source_id,
				name=name,
				tectonic_region_type=tectonic_region_type,
				mfd=mfd,
				rupture_mesh_spacing=rupture_mesh_spacing,
				magnitude_scaling_relationship=magnitude_scaling_relationship,
				rupture_aspect_ratio=rupture_aspect_ratio,
				edges=edges,
				rake=rake)

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML complexFaultSource element)

		:param encoding:
			unicode encoding (default: 'latin1')
		"""
		cfs_elem = etree.Element(ns.COMPLEX_FAULT_SOURCE)
		cfs_elem.set(ns.ID, xmlstr(self.source_id, encoding=encoding))
		cfs_elem.set(ns.NAME, xmlstr(self.name, encoding=encoding))
		cfs_elem.set(ns.TECTONIC_REGION_TYPE, xmlstr(self.tectonic_region_type, encoding=encoding))


		cfg_elem = etree.Element(ns.COMPLEX_FAULT_GEOMETRY)
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


if __name__ == '__main__':
	pass
