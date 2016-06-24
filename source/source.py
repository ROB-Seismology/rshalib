# -*- coding: utf-8 -*-

"""
Classes representing source-model elements in Openquake/oqhazlib. Where possible,
the classes are inherited from oqhazlib classes. All provide methods to create
XML elements, which are used to write a complete source-model NRML file.
Thus, objects instantiated from these classes can be used directly in oqhazlib,
as well as to generate input files for OpenQuake.
"""

from lxml import etree
## Note: Don't import numpy as np to avoid conflicts with np as abbreviation
## for nodalplane
import numpy
import pylab

import openquake.hazardlib as oqhazlib

from ..nrml import ns
from ..nrml.common import *
from ..mfd import *
from ..geo.angle import *
from ..geo import Point, Line, Polygon, NodalPlane
from ..pmf import HypocentralDepthDistribution, NodalPlaneDistribution



class RuptureSource():
	"""
	Class containing methods to explore rupture data
	independent of source type.

	The number of ruptures generated in an area source is equal to
	the number of area discretization points * number of magnitude bins *
	number of hypocentral depths * number of nodal planes
	"""
	def get_ruptures_Poisson(self, mag=None, strike=None, dip=None, rake=None, timespan=1):
		"""
		Generate ruptures according to Poissonian temporal occurrence model,
		optionally filtering by magnitude, strike, dip and rake.

		:param timespan:
			Float, time interval for Poisson distribution, in years
			(default: 1)
		:param mag:
			Float, magnitude value (center of bin) of ruptures to plot
			(default: None)
		:param strike:
			Float, strike in degrees of ruptures to plot (default: None)
		:param dip:
			Float, dip in degrees of ruptures to plot (default: None)
		:param rake:
			Float, rake in degrees of ruptures to plot (default: None)

		:return:
			list of instances of :class:`ProbabilisticRupture`
		"""
		tom = oqhazlib.tom.PoissonTOM(timespan)
		ruptures = list(self.iter_ruptures(tom))

		## Filter by magnitude, strik, dip, and rake
		if mag is not None:
			ruptures = [rup for rup in ruptures if np.allclose(rup.mag, mag)]
		if strike is not None:
			ruptures = [rup for rup in ruptures if np.allclose(rup.surface.get_strike(), strike)]
		if dip is not None:
			ruptures = [rup for rup in ruptures if np.allclose(rup.surface.get_dip(), dip)]
		if rake is not None:
			ruptures = [rup for rup in ruptures if np.allclose(rup.rake, rake)]
		return ruptures

	def get_stochastic_event_set_Poisson(self, timespan):
		"""
		Generate a random set of ruptures following a Poissonian temporal
		occurrence model. The stochastic event set represents a possible
		realization of the seismicity as described by the source,
		in the given time span.

		:param timespan:
			Float, time interval for Poisson distribution, in years

		:return:
			list of instances of :class:`ProbabilisticRupture`
		"""
		event_set = list(oqhazlib.calc.stochastic_event_set_poissonian([self], timespan))
		return event_set

	def get_rupture_bounds(self, rupture):
		"""
		Determine 3-D coordinates of rupture bounds

		:param rupture:
			instance of :class:`ProbabilisticRupture`

		:return:
			(lons, lats, depths) tuple of numpy float arrays
			depths are in km
		"""
		if isinstance(self, (PointSource, AreaSource)):
			corner_indexes = [0,1,3,2,0]
			lons = rupture.surface.corner_lons[corner_indexes]
			lats = rupture.surface.corner_lats[corner_indexes]
			depths = rupture.surface.corner_depths[corner_indexes]
		elif isinstance(self, (SimpleFaultSource, CharacteristicFaultSource)):
			mesh = rupture.surface.mesh
			all_lons, all_lats, all_depths = mesh.lons.flatten(), mesh.lats.flatten(), mesh.depths.flatten()
			top_edge_indexes = np.where(all_depths == rupture.surface.get_top_edge_depth())[0]
			bottom_edge_indexes = np.where(all_depths == all_depths.max())[0]
			edge_indexes = np.concatenate([top_edge_indexes, bottom_edge_indexes[::-1], top_edge_indexes[:1]])
			lons = all_lons[edge_indexes]
			lats = all_lats[edge_indexes]
			depths = all_depths[edge_indexes]
			#polygon = rupture.surface.mesh.get_convex_hull()
			#lons.append(polygon.lons)
			#lats.append(polygon.lats)
		# TODO: ComplexFaultRupture
		return lons, lats, depths

	def get_rupture_centers(self, ruptures):
		"""
		Determine 3-D coordinates of rupture centers

		:param ruptures:
			list of instances of :class:`ProbabilisticRupture`

		:return:
			(lons, lats, depths) tuple of numpy float arrays
			depths are in km
		"""
		lons = np.array([rup.hypocenter.longitude for rup in ruptures])
		lats = np.array([rup.hypocenter.latitude for rup in ruptures])
		if isinstance(self, (PointSource, AreaSource)):
			## rup.hypocenter does not appear to have correct depth information
			depths = np.array([rup.surface.get_middle_point().depth for rup in ruptures])
		if isinstance(self, (SimpleFaultSource, ComplexFaultSource, CharacteristicFaultSource)):
			depths = np.array([rup.hypocenter.depth for rup in ruptures])
		return lons, lats, depths

	def plot_rupture_mags_vs_occurrence_rates(self, color_param=None, timespan=1):
		ruptures = self.get_ruptures_Poisson(timespan=timespan)
		mags = np.array([rup.mag for rup in ruptures])
		occurrences = np.array([rup.occurrence_rate for rup in ruptures])
		if color_param == "depth":
			c = self.get_rupture_centers(ruptures)[2]
			vmin, vmax = self.upper_seismogenic_depth, self.lower_seismogenic_depth
		elif color_param == "strike":
			c = np.array([rup.surface.get_strike() for rup in ruptures])
			vmin, vmax = 0, 360
		elif color_param == "dip":
			c = np.array([rup.surface.get_dip() for rup in ruptures])
			vmin, vmax = 0, 90
		elif color_param == "rake":
			c = np.array([rup.rake for rup in ruptures])
			vmin, vmax = -90, 90
		else:
			c = 'b'
			vmin, vmax = None, None
		ax = pylab.subplot(111)
		pylab.scatter(mags, occurrences, marker='o', c=c, cmap="jet", vmin=vmin, vmax=vmax)
		ax.set_yscale('log')
		ax.set_ylim(occurrences.min(), occurrences.max())
		pylab.xlabel("Magnitude")
		pylab.ylabel("Occurrence rate (1/yr)")
		pylab.show()


	def plot_rupture_bounds_3d(self, ruptures, fill=False):
		"""
		Plot rupture bounds in 3 dimensions.
		Note that lon, lat coordinates are transformed to UTM coordinates
		in order to obtain metric coordinates

		:param ruptures:
			list of instances of :class:`ProbabilisticRupture`.
		:param fill:
			Bool, whether or not to plot ruptures with a transparent fill
			(default: False)
		"""
		import mpl_toolkits.mplot3d.axes3d as p3
		import mapping.geo.coordtrans as coordtrans

		## Determine UTM zone and hemisphere
		utm_spec = coordtrans.get_utm_spec(*self.get_centroid())

		fig = pylab.figure()
		ax = p3.Axes3D(fig)
		for rup in ruptures:
			if -135 <= rup.rake <= -45:
				color = (0.,1.,0.)
			elif 45 <= rup.rake <= 135:
				color = (1.,0.,0.)
			else:
				color = (0.,0.,1.)
			lons, lats, depths = self.get_rupture_bounds(rup)
			coord_list = zip(lons, lats)
			utm_coord_list = coordtrans.lonlat_to_utm(coord_list, utm_spec)
			x, y = zip(*utm_coord_list)
			x, y = np.array(x) / 1000, np.array(y) / 1000
			ax.plot3D(x, y, -depths, color=color)
			if fill:
				tri = p3.art3d.Poly3DCollection([zip(x, y, -depths)])
				tri.set_color(color + (0.05,))
				tri.set_edgecolor('k')
				ax.add_collection3d(tri)

		## Plot source outline
		if isinstance(self, PointSource):
			origin = (self.location.longitude, self.location.latitude)
			src_longitudes, src_latitudes = [], []
			dist = 10
			num_azimuths = 37
			for azimuth in np.linspace(0, 360, num_azimuths):
				lon, lat = oqhazlib.geo.geodetic.point_at(origin[0], origin[1], azimuth, dist)
				src_longitudes.append(lon)
				src_latitudes.append(lat)
			src_depths = np.zeros(num_azimuths)
		elif isinstance(self, AreaSource):
			src_longitudes = self.polygon.lons
			src_latitudes = self.polygon.lats
			src_depths = np.zeros_like(src_longitudes)
		elif isinstance(self, SimpleFaultSource):
			polygon = self.get_polygon()
			src_longitudes = polygon.lons
			src_latitudes = polygon.lats
			src_depths = np.array([pt.depth for pt in polygon])

		coord_list = zip(src_longitudes, src_latitudes)
		utm_coord_list = coordtrans.lonlat_to_utm(coord_list, utm_spec)
		x, y = zip(*utm_coord_list)
		x, y = np.array(x) / 1000, np.array(y) / 1000
		ax.plot3D(x, y, -src_depths, 'k', lw=3)

		## Upper and lower seismogenic depth
		ax.plot3D(x, y, np.ones_like(x) * -self.upper_seismogenic_depth, 'k--', lw=3)
		ax.plot3D(x, y, np.ones_like(x) * -self.lower_seismogenic_depth, 'k--', lw=3)

		## Highlight first rupture if source is a fault
		if isinstance(self, (SimpleFaultSource, ComplexFaultSource)):
			lons, lats, depths = self.get_rupture_bounds(ruptures[0])
			coord_list = zip(lons, lats)
			utm_coord_list = coordtrans.lonlat_to_utm(coord_list, utm_spec)
			x, y = zip(*utm_coord_list)
			x, y = np.array(x) / 1000, np.array(y) / 1000
			ax.plot3D(x, y, -depths, 'm--', lw=3)

		## Plot decoration
		ax.set_xlabel("Easting (km)")
		ax.set_ylabel("Northing (km)")
		ax.set_zlabel("Depth (km)")
		pylab.title(self.name)
		pylab.show()

	def plot_rupture_map(self, mag, bounds=True, strike=None, dip=None, rake=None, timespan=1):
		"""
		Plot map showing rupture bounds or rupture centers.

		:param mag:
			Float, magnitude value (center of bin) of ruptures to plot
		:param bounds:
			Bool, whether rupture bounds (True) or centers (False) should
			be plotted (default: True)
		:param strike:
			Float, strike in degrees of ruptures to plot (default: None)
		:param dip:
			Float, dip in degrees of ruptures to plot (default: None)
		:param rake:
			Float, rake in degrees of ruptures to plot (default: None)
		:param timespan:
			Float, time interval for Poisson distribution, in years
			(default: 1)
		"""
		from mpl_toolkits.basemap import Basemap

		## Generate ruptures, and filter according to magnitude bin, strike and rake
		ruptures = self.get_ruptures_Poisson(timespan=timespan)
		ruptures = [rup for rup in ruptures if np.allclose(rup.mag, mag)]
		if strike is not None:
			ruptures = [rup for rup in ruptures if np.allclose(rup.surface.strike, strike)]
		if dip is not None:
			ruptures = [rup for rup in ruptures if np.allclose(rup.surface.get_dip(), dip)]
		if rake is not None:
			ruptures = [rup for rup in ruptures if np.allclose(rup.rake, rake)]

		## Get source coordinates
		if isinstance(self, PointSource):
			src_longitudes = [self.location.longitude]
			src_latitudes = [self.location.latitude]
		elif isinstance(self, AreaSource):
			src_longitudes = self.polygon.lons
			src_latitudes = self.polygon.lats
		elif isinstance(self, SimpleFaultSource):
			polygon = self.get_polygon()
			src_longitudes = polygon.lons
			src_latitudes = polygon.lats

		## Determine map bounds and central longitude and latitude
		llcrnrlon, llcrnrlat = min(src_longitudes), min(src_latitudes)
		urcrnrlon, urcrnrlat = max(src_longitudes), max(src_latitudes)
		lon_0 = (llcrnrlon + urcrnrlon) / 2.
		lat_0 = (llcrnrlat + urcrnrlat) / 2.

		## Initiate map
		map = Basemap(projection="aea", resolution='h', llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, lon_0=lon_0, lat_0=lat_0)
		map.drawcoastlines()
		map.drawcountries()

		## Plot ruptures
		for rup in ruptures:
			if -135 <= rup.rake <= -45:
				color = 'g'
			elif 45 <= rup.rake <= 135:
				color = 'r'
			else:
				color = 'b'
			if bounds == True:
				lons, lats, depths = self.get_rupture_bounds(rup)
				x, y = map(lons, lats)
				map.plot(x, y, color)
		if bounds == False:
			lons, lats, depths = self.get_rupture_centers(ruptures)
			x, y = map(lons, lats)
			map.plot(x, y, 'o', color=color)

		## Highlight first rupture if source is a fault
		if isinstance(self, (SimpleFaultSource, ComplexFaultSource)):
			lons, lats, depths = self.get_rupture_bounds(ruptures[0])
			x, y = map(lons, lats)
			map.plot(x, y, 'm--', lw=3)

		## Plot source outline
		if isinstance(self, PointSource):
			pass
		elif isinstance(self, AreaSource):
			x, y = map(src_longitudes, src_latitudes)
			map.plot(x, y, 'k', lw=3)
		if isinstance(self, SimpleFaultSource):
			x, y = map(src_longitudes, src_latitudes)
			map.plot(x, y, 'k--', lw=2)
			x, y = map(self.longitudes, self.latitudes)
			map.plot(x, y, 'k', lw=3)

		map.drawmapboundary()
		pylab.title(self.name)
		pylab.show()


class PointSource(oqhazlib.source.PointSource, RuptureSource):
	"""
	Class representing a point source, corresponding to a single geographic site

	:param source_id:
		source identifier
	:param name:
		full name of source
	:param tectonic_region_type:
		tectonic region type known to oqhazlib/OQ. See :class:`oqhazlib.const.TRT`
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
		if isinstance(magnitude_scaling_relationship, (str, unicode)):
			magnitude_scaling_relationship = getattr(oqhazlib.scalerel, magnitude_scaling_relationship)()
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

	@classmethod
	def from_eq_record(self, eq, Mtype="MW", Mrelation={},
				tectonic_region_type="Stable Shallow Crust",
				magnitude_scaling_relationship=oqhazlib.scalerel.WC1994(),
				rupture_mesh_spacing=1., rupture_aspect_ratio=1.,
				upper_seismogenic_depth=5., lower_seismogenic_depth=25.,
				nodal_plane_distribution=None, hypocenter_distribution=None,
				synthetic=False):
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

		return PointSource(source_id, name, tectonic_region_type, mfd,
					rupture_mesh_spacing, magnitude_scaling_relationship,
					rupture_aspect_ratio, upper_seismogenic_depth,
					lower_seismogenic_depth, location, npd, hdd)

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

	def to_ogr_geometry(self):
		"""
		Create OGR Geometry object
		"""
		import osr, ogr

		## Construct WGS84 projection system corresponding to earthquake coordinates
		wgs84 = osr.SpatialReference()
		wgs84.SetWellKnownGeogCS("WGS84")

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
			(Float, Float) tuple: longitude, latitude of centroid
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


class AreaSource(oqhazlib.source.AreaSource, RuptureSource):
	"""
	Class representing an area source, i.e. a polygonal geographical region
	where seismicity is assumed to be uniform.
	An area source is similar to a point source, except for the geometry

	:param source_id:
		source identifier
	:param name:
		full name of source
	:param tectonic_region_type:
		tectonic region type known to oqhazlib/OQ. See :class:`oqhazlib.const.TRT`
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
	"""

	def __init__(self, source_id, name, tectonic_region_type, mfd, rupture_mesh_spacing, magnitude_scaling_relationship, rupture_aspect_ratio, upper_seismogenic_depth, lower_seismogenic_depth, nodal_plane_distribution, hypocenter_distribution, polygon, area_discretization):
		if isinstance(magnitude_scaling_relationship, (str, unicode)):
			magnitude_scaling_relationship = getattr(oqhazlib.scalerel, magnitude_scaling_relationship)()
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

	def get_MFD_Johnston1994(self, min_mag=None, max_mag=None, bin_width=None, region="total"):
		"""
		Construct "minimum" MFD for SCR according to Johnston (1994),
		based on surface area

		:param min_mag:
			Float, Minimum magnitude (default: None, take min_mag from current MFD).
		:param max_mag:
			Maximum magnitude (default: None, take max_mag from current MFD).
		:param bin_width:
			Float, Magnitude interval for evenly discretized magnitude frequency
			distribution (default: None, take bin_width from current MFD.
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

		return mfd.TruncatedGRMFD.construct_Johnston1994MFD(min_mag, max_mag, bin_width, self.get_area(), region)

	def get_MFD_FentonEtAl2006(self, min_mag=None, max_mag=None, bin_width=None, b_val=0.7991):
		"""
		Construct "minimum" MFD for SCR according to Fenton et al. (2006),
		based on surface area

		:param min_mag:
			Float, Minimum magnitude (default: None, take min_mag from current MFD).
		:param max_mag:
			Maximum magnitude (default: None, take max_mag from current MFD).
		:param bin_width:
			Float, Magnitude interval for evenly discretized magnitude frequency
			distribution (default: None, take bin_width from current MFD.
		:param b_val:
			Float, Parameter of the truncated gutenberg richter model.
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

		return mfd.TruncatedGRMFD.construct_FentonEtAl2006MFD(min_mag, max_mag, bin_width, self.get_area(), b_val)

	def to_ogr_geometry(self):
		"""
		Create OGR Geometry object
		"""
		import osr, ogr

		## Construct WGS84 projection system corresponding to earthquake coordinates
		wgs84 = osr.SpatialReference()
		wgs84.SetWellKnownGeogCS("WGS84")

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
			Float, fault source in square km
		"""
		import osr
		from mapping.geo.coordtrans import wgs84, get_utm_spec, get_utm_srs

		poly = self.to_ogr_geometry()
		centroid = poly.Centroid()
		utm_spec = get_utm_spec(centroid.GetX(), centroid.GetY())
		utm_srs = get_utm_srs(utm_spec)
		coordTrans = osr.CoordinateTransformation(wgs84, utm_srs)
		poly.Transform(coordTrans)
		return poly.GetArea() / 1E6

	def get_centroid(self):
		"""
		Compute centroid of area source

		:return:
			(Float, Float) tuple: longitude, latitude of centroid
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
			normalized surface area in sq. km (default: 1E+5)
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
								self.nodal_plane_distribution, self.hypocenter_distribution)
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
			float, rigidity (default: 3E+10)

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
			float, rigidity (default: 3E+10)

		:return:
			float, maximum magnitude
		"""
		assert isinstance(self.mfd, TruncatedGRMFD)
		moment_rate = self.get_moment_rate_from_strain_rate(strain_rate, rigidity)
		Mmax = self.mfd.get_Mmax_from_moment_rate(moment_rate)
		return Mmax


class SimpleFaultSource(oqhazlib.source.SimpleFaultSource, RuptureSource):
	"""
	Class representing a simple fault source.

	:param source_id:
		source identifier
	:param name:
		full name of source
	:param tectonic_region_type:
		tectonic region type known to oqhazlib/OQ. See :class:`oqhazlib.const.TRT`
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
	# TODO: SlipratePMF
	# TODO: add aseismic_coef and strain_drop parameters (see get_MFD_Anderson_Luco method)
	def __init__(self, source_id, name, tectonic_region_type, mfd, rupture_mesh_spacing, magnitude_scaling_relationship, rupture_aspect_ratio, upper_seismogenic_depth, lower_seismogenic_depth, fault_trace, dip, rake, slip_rate=numpy.NaN, bg_zone=None):
		if isinstance(magnitude_scaling_relationship, (str, unicode)):
			magnitude_scaling_relationship = getattr(oqhazlib.scalerel, magnitude_scaling_relationship)()
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

	def get_bounding_box(self):
		"""
		Determine rectangular bounding box

		:return:
			(west, east, south, north) tuple
		"""
		polygon = self.get_polygon()
		w = polygon.lons.min()
		e = polygon.lons.max()
		s = polygon.lats.min()
		n = polygon.lats.max()
		return (w, e, s, n)

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
		return self.get_width() * numpy.cos(numpy.radians(self.dip))

	def get_length(self):
		"""
		Compute length of fault in km

		:return:
			Float, fault length in km
		"""
		lons, lats = self.fault_trace.lons, self.fault_trace.lats
		lons1, lats1 = lons[:-1], lats[:-1]
		lons2, lats2 = lons[1:], lats[1:]
		distances = oqhazlib.geo.geodetic.geodetic_distance(lons1, lats1, lons2, lats2)
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
		distances = oqhazlib.geo.geodetic.geodetic_distance(lons1, lats1, lons2, lats2)
		azimuths = oqhazlib.geo.geodetic.azimuth(lons1, lats1, lons2, lats2)
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
			lon, lat = oqhazlib.geo.geodetic.point_at(pt[0], pt[1], perpendicular_direction, width)
			bottom_edge.append(Point(lon, lat, z1))
		bottom_edge.reverse()
		return Polygon(top_edge + bottom_edge + [top_edge[0]])

	def calc_Mmax_Wells_Coppersmith(self):
		"""
		Compute Mmax from Wells & Coppersmith (1994) scaling relation

		:return:
			Float, maximum magnitude
		"""
		from openquake.hazardlib.scalerel.wc1994 import WC1994
		wc = WC1994()
		max_mag = wc.get_median_mag(self.get_area(), self.rake)
		return max_mag

	def get_MFD_characteristic(self, bin_width=None, M_sigma=0.3, num_sigma=1):
		"""
		Construct MFD corresponding to a characteristic Mmax
		:param bin_width:
			Float, Magnitude interval for evenly discretized magnitude frequency
			distribution (default: None, take bin_width from current MFD.
		:param M_sigma:
			Float, standard deviation on magnitude (default: 0.3)
		:param num_sigma:
			Float, number of standard deviations to spread occurrence rates over
			(default: 1)

		:return:
			instance of :class:`EvenlyDiscretizedMFD`
		"""
		if bin_width is None:
			bin_width = self.mfd.bin_width

		#char_mag = self.max_mag - bin_width
		char_mag = self.max_mag
		return_period = self.get_Mmax_return_period()
		MFD = CharacteristicMFD(char_mag, return_period, bin_width, M_sigma=M_sigma, num_sigma=num_sigma)
		return MFD

	def get_MFD_Anderson_Luco(self, min_mag=None, max_mag=None, bin_width=None, b_val=None, aseismic_coef=0., strain_drop=None, mu=3E+10, arbitrary_surface=True):
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
			rupture displacement and rupture length. This parameter is only
			required if :param:`arbitrary_surface` is set to True.
			If not provided, it will be computed from Mmax
			(default: None)
		:param mu:
			rigidity or shear modulus in N/m**2 (default: 3E+10)
		:param arbitrary_surface:
			Boolean indicating whether rupture surface is arbitrary or
			corresponds to max_mag (default: True)

		:return:
			instance of :class:`TruncatedGRMFD`
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
		if not strain_drop and not arbitrary_surface:
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

	def to_ogr_geometry(self):
		"""
		Create OGR Geometry object
		"""
		# TODO: complete
		import osr, ogr

		## Construct WGS84 projection system corresponding to earthquake coordinates
		wgs84 = osr.SpatialReference()
		wgs84.SetWellKnownGeogCS("WGS84")

		## Create line
		line = ogr.Geometry(type=ogr.wkbLineString)

		## Add points
		for lon, lat in zip(self.longitudes, self.latitudes):
			line.AddPoint_2D(lon, lat)

		line.AssignSpatialReference(wgs84)

		return line

	def get_centroid(self):
		"""
		Compute centroid of area source

		:return:
			(Float, Float) tuple: longitude, latitude of centroid
		"""
		centroid = self.to_ogr_geometry().Centroid()
		return (centroid.GetX(), centroid.GetY())

	def to_characteristic_source(self, convert_mfd=False):
		"""
		Convert to a characteristic fault source

		:param convert_mfd:
			bool, whether or not to convert fault MFD to
			instance of :class:`CharacteristicMFD`
			(default: False)

		:return:
			instance of :class:`CharacteristicSource`
		"""
		surface = oqhazlib.geo.surface.SimpleFaultSurface.from_fault_data(
			self.fault_trace, self.upper_seismogenic_depth,
			self.lower_seismogenic_depth, self.dip, self.rupture_mesh_spacing)

		if convert_mfd and not isinstance(self.mfd, CharacteristicMFD):
			mfd = self.get_MFD_characteristic()
		else:
			mfd = self.mfd

		return CharacteristicFaultSource(self.source_id, self.name,
			self.tectonic_region_type, mfd, surface, self.rake)

	def get_subfaults(self, as_num=1, ad_num=1):
		"""
		Divide fault into subfaults (to be used for dislocation modeling).

		:param as_num:
			int, number of subfaults along strike (default: 1)
		:param ad_num:
			int, number of subfaults along dip (default: 1)

		:return:
			list with instances of :class:`eqgeology.FocMec.ElasticSubFault`
		"""
		from eqgeology.FocMec.okada import ElasticSubFault
		from thirdparty.PyVisvalingamWhyatt.polysimplify import VWSimplifier

		subfault_width = self.get_width() / ad_num

		lons, lats = self.fault_trace.lons, self.fault_trace.lats
		pts = zip(lons, lats)
		if as_num >= len(pts):
			print("Warning: maximum possible number of subfaults along strike is %d" % (len(pts)-1))
			reduced_pts = pts
			as_num = len(pts) - 1
		else:
			simplifier = VWSimplifier(pts)
			reduced_pts = simplifier.from_number(as_num+1)
			## In some cases, as_num is not honored by VWSimplifier
			as_num = len(reduced_pts) - 1

		perpendicular_direction = self.get_mean_strike() + 90.

		subfaults = []
		for i in range(as_num):
			start, end = reduced_pts[i:i+2]
			center_lon = np.mean([start[0], end[0]])
			center_lat = np.mean([start[1], end[1]])
			[subfault_length] = oqhazlib.geo.geodetic.geodetic_distance(start[0:1], start[-1:], end[0:1], end[-1:])
			[subfault_strike] = oqhazlib.geo.geodetic.azimuth(start[0:1], start[-1:], end[0:1], end[-1:])
			for j in range(ad_num):
				top_depth = j * (subfault_width * np.sin(np.radians(self.dip)))
				top_dist = j * (subfault_width * np.cos(np.radians(self.dip)))
				top_lon, top_lat = oqhazlib.geo.geodetic.point_at(center_lon, center_lat, perpendicular_direction, top_dist)

				subfault = ElasticSubFault()
				subfault.strike = subfault_strike
				subfault.dip = self.dip
				subfault.rake = self.rake
				subfault.length = subfault_length * 1E+3
				subfault.width = subfault_width * 1E+3
				subfault.depth = top_depth * 1E+3
				subfault.slip = self.get_Mmax_slip()
				subfault.longitude = top_lon
				subfault.latitude = top_lat
				subfault.coordinate_specification = "top center"
				subfaults.append(subfault)

		return subfaults


class ComplexFaultSource(oqhazlib.source.ComplexFaultSource, RuptureSource):
	"""
	Class representing a complex fault source.

	:param source_id:
		source identifier
	:param name:
		full name of source
	:param tectonic_region_type:
		tectonic region type known to oqhazlib/OQ. See :class:`oqhazlib.const.TRT`
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
	:param edges:
		A list of :class:`Line` objects, representing fault source geometry. See
		:meth:`oqhazlib.geo.surface.complex_fault.ComplexFaultSurface.from_fault_data`
	:param rake:
		Angle describing fault rake in decimal degrees.

	"""
	# TODO: add bg_zone parameter as for SimpleFaultSource
	def __init__(self, source_id, name, tectonic_region_type, mfd, rupture_mesh_spacing, magnitude_scaling_relationship, rupture_aspect_ratio, edges, rake):
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
		surface = oqhazlib.geo.surface.ComplexFaultSurface.from_fault_data(
			self.edges, self.rupture_mesh_spacing)

		return CharacteristicFaultSource(self.source_id, self.name,
			self.tectonic_region_type, self.mfd, surface, self.rake)


class CharacteristicFaultSource(oqhazlib.source.CharacteristicFaultSource, RuptureSource):
	"""
	Class representing a characteristic source, this is a fault surface
	with seismic events rupturing the entire fault surface
	independently of their magnitude.
	Thus, rupture mesh spacing, magnitude scaling relationship and rupture
	aspect ratio need not be specified.
	We do not support the case where characteristic fault sources are defined
	by multiple planar surfaces, but only the cases with simple fault surfaces
	or complex fault surfaces.
	"""
	def __init__(self, source_id, name, tectonic_region_type, mfd, surface, rake):
		super(CharacteristicFaultSource, self).__init__(source_id, name,
								tectonic_region_type, mfd, surface, rake)

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML characteristicFaultSource element)

		:param encoding:
			unicode encoding (default: 'latin1')
		"""
		cfs_elem = etree.Element(ns.CHARACTERISTIC_FAULT_SOURCE)
		cfs_elem.set(ns.ID, xmlstr(self.source_id, encoding=encoding))
		cfs_elem.set(ns.NAME, xmlstr(self.name, encoding=encoding))
		cfs_elem.set(ns.TECTONIC_REGION_TYPE, xmlstr(self.tectonic_region_type, encoding=encoding))

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
		return self.upper_seismogenic_depth + self.width * numpy.sin(numpy.radians(self.dip))

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
			edge = Line([Point(lon, lat, depth) for (lon, lat, depth) in zip(lons, lats, depths)])
			edges.append(edge)
		return edges

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
		points = [Point(lon, lat, depth) for lon, lat, depth in zip(lons, lats, depths)]
		return Polygon(points)



if __name__ == '__main__':
	pass
