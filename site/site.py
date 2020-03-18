# -*- coding: utf-8 -*-

"""
Classes representing source-model elements in Openquake. Where possible,
the classes are inherited from oqhazlib classes. All provide methods to create
XML elements, which are used to write a complete source-model NRML file.
Thus, objects instantiated from these classes can be used directly in oqhazlib,
as well as to generate input files for OpenQuake.
"""

# TODO: add kappa and thickness as soil_params

from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import int

from lxml import etree

import numpy as np

import openquake.hazardlib as oqhazlib
from openquake.hazardlib.geo.geodetic import geodetic_distance

from .ref_soil_params import REF_SOIL_PARAMS
from ..nrml import ns
from ..nrml.common import (create_nrml_root, xmlstr)
from ..geo import Point, Polygon

try:
	## Python 2
	basestring
except:
	## Python 3
	basestring = str



__all__ = ['GenericSite', 'GenericSiteModel', 'SoilSite', 'SoilSiteModel']


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


class GenericSiteModel(oqhazlib.geo.Mesh):
	"""
	Class representing a generic site model.
	Subclasses from :class:`openquake.hazardlib.geo.Mesh`, so sites
	are always stored internally as lons, lats and depths.

	SiteModel is initiated, in order of priority, by:
	(1) sites (and optionally site names) or
	(2) lons and lats (and depths), or
	(3) grid outline and grid spacing.

	:param sites:
		- list of instances of :class:`GenericSite` or :class:`SoilSite`
		- instance of :class:`GenericSiteModel`
		(default: None)
	:param lons:
		1-D or 2-D array, lons of sites
		(default: None)
	:param lats:
		1-D or 2-D array, lats of sites
		(default: None)
	:param depths:
		1-D or 2-D array, depths of sites (in km)
		or float, uniform depth for all sites (e.g., for grid model)
		(default: None, zero depths)
	:param site_names:
		list of str, names of sites
		(length must equal that of sites)
		Discarded if site model is constructed from grid specification
		(default: None)
	:param grid_outline
		- list of 2 (lon, lat) tuples, corresponding to lower left
		and lower right corners
		- list of 4 (lon, lat) tuples, corresponding to lower left,
		lower right, upper right and upper left corners
		- list of instances or subclasses of :class:`rshalib.geo.Point`,
		corresponding to ll, lr, ur and ul corners
		- list of 4 floats, corresponding to west, east, south, north
		grid limits
		(default: None)
	:param grid_spacing:
		- float, uniform grid spacing in degrees
		- list of 2 floats, corresponding to north-south and east-west
		spacing in degrees
		- str, consisting of a number with 'km' appended,
		uniform kilometric grid spacing
		(default: None)
	:param name:
		str, site model name
		(default: '')
	"""
	def __init__(self, sites=None, site_names=None,
				lons=None, lats=None, depths=None,
				grid_outline=None, grid_spacing=None,
				name=''):
		"""
		"""
		if sites:
			## (1) sites (and site names)
			self._set_sites(sites, site_names)

		elif not (lons is None or lats is None):
			## (2) lons and lats (and depths)
			assert len(lons) == len(lats)
			assert site_names is None or len(site_names) == len(lons)
			super(GenericSiteModel, self).__init__(lons, lats, depths)
			self.site_names = site_names
			self.grid_outline = None
			self.grid_spacing = None
		else:
			## (3) grid outline and grid spacing
			if depths is None or np.isscalar(depths):
				depth = depths
			else:
				depth = depths[0]
			self._set_grid(grid_outline, grid_spacing, depth=depth)

		self.name = name

	def __repr__(self):
		return '<GenericSiteModel %s (n=%d)>' % (self.name, len(self))

	def __getitem__(self, index):
		"""
		:param index:
			int, index in flattened array of mesh

		:return:
			instance of :class:`GenericSite`
		"""
		lon = self.lons.flat[index]
		lat = self.lats.flat[index]
		depth = self.depths.flat[index] if self.depths is not None else 0.
		name = self.site_names[index] if self.site_names is not None else ''

		return GenericSite(lon, lat, depth, name=name)

	def __iter__(self):
		"""
		:return:
			iterator, (lon, lat) or (lon, lat, depth) for each site
		"""
		for i in xrange(len(self)):
			yield self.__getitem__(i)

	def __contains__(self, site):
		"""
		Determine if site model contains a given site (based on position
		only).

		:param site:
			instance of :class:`GenericSite` or :class:`SoilSite`

		:return:
			bool
		"""
		idx = np.argmin(np.abs(self.lons - site.lon))
		if np.allclose([self.lons[idx], self.lats[idx]], [site.lon, site.lat], atol=1E-6):
			return True
		else:
			return False

	@staticmethod
	def _parse_sites(sites, site_names=None):
		"""
		Convert sites to longitudes, latitudes and depths

		:param sites:
			list with instances of :class:`GenericSite` or :class:`SoilSite`
		:param site_names:
			list of strings, site names
			(default: None)

		:return:
			(lons, lats, depths, site_names) tuple
		"""
		lons = np.array([site.lon for site in sites])
		lats = np.array([site.lat for site in sites])
		depths = np.array([site.depth for site in sites])
		if np.allclose(depths, 0.):
			depths = None

		if site_names is None and hasattr(sites[0], 'name'):
			site_names = [site.name for site in sites]

		return (lons, lats, depths, site_names)

	def _set_sites(self, sites, site_names=None):
		"""
		Set longitudes, latitudes, depths and site names from list of sites

		:param sites:
		:param site_names:
			see :meth`_parse_sites`

		:return:
			None, GenericSiteModel is modified in place
		"""
		lons, lats, depths, site_names = self._parse_sites(sites, site_names)
		super(GenericSiteModel, self).__init__(lons=lons, lats=lats, depths=depths)
		self.site_names = site_names
		self.grid_outline = None
		self.grid_spacing = None

	@classmethod
	def from_sites(cls, sites, site_names=None):
		"""
		Generate generic site model from list of sites

		:param sites:
			list with instances of :class:`GenericSite` or :class:`SoilSite`
		:param site_names:
			list of strings, site names
			(default: None)

		:return:
			instance of :class:`GenericSiteModel`
		"""
		lons, lats, depths, site_names = cls._parse_sites(sites, site_names)
		return cls(lons=lons, lats=lats, depths=depths, site_names=site_names)

	def copy(self):
		"""
		Copy site model

		:return:
			instance of :class:`GenericSiteModel`
		"""
		return self.from_sites(self)

	@staticmethod
	def _parse_grid_outline(grid_outline):
		"""
		Parse grid_outline

		:param grid_outline:
			- list of 2 (lon, lat) tuples, corresponding to lower left
			and lower right corners
			- list of 4 (lon, lat) tuples, corresponding to lower left,
			lower right, upper right and upper left corners
			- list of instances or subclasses of :class:`rshalib.geo.Point`,
			corresponding to ll, lr, ur and ul corners
			- list of 4 floats, corresponding to west, east, south, north
			grid limits

		:return:
			list of 4 (lon, lat) tuples, corresponding to lower left,
			lower right, upper right and upper left corners
		"""
		if len(grid_outline) < 2:
			raise Exception("grid_outline must contain 2 points at least")
		elif len(grid_outline) == 2:
			llc, urc = grid_outline
			lrc = (urc[0], llc[1])
			ulc = (llc[0], urc[1])
			grid_outline = [llc, lrc, urc, ulc]
		elif len(grid_outline) == 4:
			if np.isscalar(grid_outline[0]):
				w, e, s, n = grid_outline
				llc = (w, s)
				lrc = (e, s)
				urc = (e, n)
				ulc = (w, n)
				grid_outline = [llc, lrc, urc, ulc]
			elif len(grid_outline[0] == 2) or isinstance(grid_outline[0], Point):
				grid_outline = [(site[0], site[1]) for site in grid_outline]
		else:
			raise Exception("grid_outline not understood!")

		return grid_outline

	@staticmethod
	def _parse_grid_spacing(grid_spacing):
		"""
		Parse grid_spacing

		:param grid_spacing:
			- float, uniform grid spacing in degrees
			- list of 2 floats, corresponding to north-south and east-west
			spacing in degrees
			- str, consisting of a number with 'km' appended,
			uniform kilometric grid spacing

		:return:
			list of 2 floats or str
		"""
		if isinstance(grid_spacing, (int, float)):
			grid_spacing = (grid_spacing, grid_spacing)
		elif isinstance(grid_spacing, basestring):
			assert grid_spacing.endswith("km")
			grid_spacing = grid_spacing
		elif len(grid_spacing) == 2 and isinstance(grid_spacing[0], (int, float)):
			grid_spacing = grid_spacing
		else:
			raise Exception("grid_spacing not understood!")

		return grid_spacing

	@staticmethod
	def _create_grid(grid_outline, grid_spacing):
		"""
		Create lons and lats based on grid_outline and grid_spacing.

		:param grid_outline:
			list of 4 (lon, lat) tuples, corresponding to lower left,
			lower right, upper right and upper left corners
		:param grid_spacing:
			list of 2 floats, north-south and east-west spacing in degrees
			or str ending with 'km', uniform kilometric spacing

		:return:
			(lons, lats) tuple of 1D or 2D(?) arrays
		"""
		if isinstance(grid_spacing, basestring):
			grid_spacing = float(grid_spacing[:-2])
			polygon = Polygon([Point(lon, lat) for (lon, lat) in grid_outline])
			grid = polygon.discretize(grid_spacing)
			lons = grid.lons
			lats = grid.lats
		else:
			from ..utils import seq

			grid_outline = np.array(grid_outline)
			slons = seq(grid_outline[:,0].min(), grid_outline[:,0].max(),
						grid_spacing[0])
			slats = seq(grid_outline[:,1].min(), grid_outline[:,1].max(),
						grid_spacing[1])
			grid = np.dstack(np.meshgrid(slons, slats[::-1]))
			lons = grid[:,:,0]
			lats = grid[:,:,1]

		return (lons, lats)

	def _set_grid(self, grid_outline, grid_spacing, depth=None):
		"""
		Set longitudes, latitudes and depths from grid outline and spacing

		:param grid_outline:
			see :meth`_parse_grid_outline`
		:param grid_spacing:
			see :meth`_parse_grid_spacing`
		:param depth:
			float, uniform depth (in km) for all grid points
			(default: None)

		:return:
			None, GenericSiteModel is modified in place
		"""
		self.grid_outline = grid_outline
		self.grid_spacing = grid_spacing
		grid_outline = self._parse_grid_outline(grid_outline)
		grid_spacing = self._parse_grid_spacing(grid_spacing)
		lons, lats = self._create_grid(grid_outline, grid_spacing)
		if depth is not None:
			depths = np.ones_like(lons) * depth
		else:
			depths = None
		super(GenericSiteModel, self).__init__(lons=lons, lats=lats, depths=depths)
		self.site_names = None

	@classmethod
	def from_grid_spec(cls, grid_outline, grid_spacing, depth=None):
		"""
		Generate generic site model from grid outline and spacing

		:param grid_outline:
			- float, uniform grid spacing in degrees
			- list of 2 floats, corresponding to north-south and east-west
			spacing in degrees
			- str, consisting of a number with 'km' appended,
			uniform kilometric grid spacing
		:param grid_spacing:
			- float, uniform grid spacing in degrees
			- list of 2 floats, corresponding to north-south and east-west
			spacing in degrees
			- str, consisting of a number with 'km' appended,
			uniform kilometric grid spacing
		:param depth:
			float, uniform depth (in km) for all grid points
			(default: None)

		:return:
			instance of :class:`GenericSiteModel`
		"""
		_grid_outline = grid_outline
		_grid_spacing = grid_spacing
		grid_outline = cls._parse_grid_outline(_grid_outline)
		grid_spacing = cls._parse_grid_spacing(_grid_spacing)
		lons, lats = cls._create_grid(grid_outline, grid_spacing)
		if depth is not None:
			depths = np.ones_like(lons) * depth
		else:
			depths = None

		site_model = cls(lons=lons, lats=lats, depths=depths, site_names=None)
		site_model.grid_outline = _grid_outline
		site_model.grid_spacing = _grid_spacing
		return site_model

	@classmethod
	def from_polygon(cls, polygon, spacing, depth=None):
		"""
		Construct simple site model from polygon and spacing.

		:param polygon:
			instance of :class:`oqhazlib.geo.polygon.Polygon`
		:param spacing:
			float, spacing in km for discretizing polygonµ
		:param depth:
			float, depth in km
			(default: None)

		:return:
			instance of :class:`GenericSiteModel`
		"""
		mesh = polygon.discretize(spacing)
		if depth:
			depths = [depth] * len(mesh)
		else:
			depths = None
		return cls(lons=mesh.lons, lats=mesh.lats, depths=depths)

	def get_region(self):
		"""
		:return:
			(float, float, float, float) tuple, (w, e, s, n) of bounding box
		"""
		return (self.lons.min(), self.lons.max(), self.lats.min(), self.lats.max())

	def is_grid(self):
		"""
		Determine whether site model is a 2-D grid (in degree space) or not
		"""
		return len(self.shape) == 2

	@property
	def slons(self):
		"""
		:return:
			1d np array, set of lons if model is a lon lat grid, otherwise None
		"""
		if self.is_grid():
			return self.lons[0,:]
		else:
			return None

	@property
	def slats(self):
		"""
		:return:
			1d np array, set of lats if model is a lon lat grid, otherwise None
		"""
		if self.is_grid():
			return self.lats[:,0][::-1]
		else:
			return None

	def get_geographic_distances(self, lon, lat):
		"""
		Get the geographic distance each site in the model to a given site.

		:param lon:
			float, lon of site
		:param lat:
			float, lat of site

		:return:
			np array with same shape as underlying mesh, distances in km
		"""
		return geodetic_distance(lon, lat, self.lons, self.lats)

	def get_nearest_site_index(self, lon, lat, flat_index=True):
		"""
		Get index of closest site.

		:param lon:
			float, lon of site.
		:param lat:
			float, lat of site.
		:param flat_index:
			bool, whether or not index should correspond to position
			in flattened array. Only relevant if site model is a 2-D grid
			(default: True)

		:return:
			int, index of site
		"""
		mesh = self.__class__(lons=np.array([lon]), lats=np.array([lat]))
		i = self._geodetic_min_distance(mesh, True)
		if not flat_index:
			i = np.unravel_index(i, self.shape)
		return i

	def get_nearest_site(self, lon, lat):
		"""
		Get closest site

		:param lon:
		:param lat:
			see :meth:`get_nearest_site_index`

		:return:
			instance of :class:`GenericSite`
		"""
		return self.__getitem__(self.get_nearest_site_index(lon, lat, flat_index=True))

	def clip(self):
		"""
		Clip the site model by the grid outline if it is a degree spaced grid.
		Note that site names will get lost!

		:return:
			instance of GenericSiteModel
		"""
		if len(self.shape) != 2:
			return self
		else:
			grid_outline = self._parse_grid_outline(self.grid_outline)
			mask = Polygon([Point(*point) for point in grid_outline]).intersects(self)
			lons = self.lons[mask]
			lats = self.lats[mask]
			if self.depths != None:
				depths = self.depths[mask]
			else:
				depths = None
			return self.__class__(lons=lons, lats=lats, depths=depths)

	def get_sites(self, clip=False):
		"""
		Get sites of site model.

		:param clip:
			bool, whether to clip site model or not
			(default: False)

		:return:
			list of instances of GenericSite
		"""
		if clip == True:
			site_model = self.clip()
		else:
			site_model = self
		sites = [site_model.__getitem__(i) for i in range(len(site_model))]
		return sites

	def to_soil_site_model(self, ref_soil_params=REF_SOIL_PARAMS, name=''):
		"""
		Get soil site model from generic site model with reference
		soil parameters for each site.

		:param ref_soil_params:
			dict, reference value for each soil parameter needed by
			soil site model
			(default: defaults specified as REF_SOIL_PARAMS in
			ref_soil_params module)
		:param name:
			str, name of soil site model
			(default: "")

		:return:
			instance of SoilSiteModel
		"""
		name = name or self.name
		soil_sites = [site.to_soil_site(ref_soil_params) for site in self.get_sites()]
		return SoilSiteModel(soil_sites, name=name)

	def plot(self, region=None, projection="merc", resolution="i",
			site_style='default', border_style='default', title=None):
		"""
		Plot map of site model

		:param region:
			(W, E, S, N) tuple
			(default: None, will take bounding box of site model)
		:param projection:
			str, name of projection
			(default: "merc")
		:param resolution:
			char, coastline / country border resolution
			(default: "i")
		:param site_style:
			instance of :class:`layeredbasemap.PointStyle`
			(default: 'default')
		:param border_style:
			instance of :class:`layeredbasemap.LineStyle`
			(default: 'default')
		:param title:
			str, plot title
			(default: None)
		"""
		from mapping.layeredbasemap.layered_basemap import MapLayer, LayeredBasemap
		from mapping.layeredbasemap.data_types import MultiPointData, BuiltinData
		from mapping.layeredbasemap.styles import PointStyle, LineStyle

		map_layers = []

		site_data = MultiPointData(self.lons, self.lats)
		if site_style == "default":
			site_style = PointStyle(shape=".", size=5)
		site_layer = MapLayer(site_data, site_style, legend_label='Sites')
		map_layers.append(site_layer)

		if border_style:
			if border_style == "default":
				border_style = LineStyle()
			coastline_layer = MapLayer(BuiltinData("coastlines"), border_style)
			map_layers.append(coastline_layer)
			country_layer = MapLayer(BuiltinData("countries"), border_style)
			map_layers.append(country_layer)

		title = self.name if title is None else title
		region = self.get_region() if region is None else region
		map = LayeredBasemap(layers=map_layers, region=region, title=title,
							projection=projection, resolution=resolution)
		map.plot()

## alias for backwards compatibility
SHASiteModel = GenericSiteModel


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


class SoilSiteModel(oqhazlib.site.SiteCollection):
	"""
	Class representing a soil site model.

	:param sites:
		list with instances of :class:`SoilSite`
	:param name:
		str, site model name
	"""

	def __init__(self, sites, name):
		self.name = name
		super(SoilSiteModel, self).__init__(sites=sites)
		self.site_names = [site.name for site in sites]

	@property
	def num_sites(self):
		return self.__len__()

	def __contains__(self, site):
		"""
		Determine if site model contains a given site (based on position
		only).

		:param site:
			instance of :class:`GenericSite` or :class:`SoilSite`

		:return:
			bool
		"""
		idx = np.argmin(np.abs(self.lons - site.lon))
		if np.allclose([self.lons[idx], self.lats[idx]], [site.lon, site.lat], atol=1E-6):
			return True
		else:
			return False

	def __repr__(self):
		return '<SoilSiteModel %s (n=%d)>' % (self.name, len(self))

	def __getitem__(self, index):
		"""
		:param index:
			int, site index in site model

		:return:
			instance of :class:`SoilSite`
		"""
		lon = self.mesh.lons[index]
		lat = self.mesh.lats[index]
		## Note: depth requires patch in oqhazlib
		depth = self.mesh.depths[index] if self.mesh.depths is not None else 0.
		vs30 = self.vs30[index]
		vs30measured = self.vs30measured[index]
		z1pt0 = self.z1pt0[index]
		z2pt5 = self.z2pt5[index]
		kappa = self.kappa[index]
		soil_params = {"vs30": vs30, "vs30measured": vs30measured,
						"z1pt0": z1pt0, "z2pt5": z2pt5, "kappa": kappa}
		name = self.site_names[index] if self.site_names is not None else ''

		site = SoilSite(lon, lat, depth, soil_params, name)
		return site

	## Note: we don't override __iter__ method of SiteCollection

	@property
	def lons(self):
		return self.mesh.lons

	@property
	def lats(self):
		return self.mesh.lats

	def get_region(self):
		"""
		Determine bounding box

		:return:
			(lonmin, lonmax, latmin, latmax) tuple
		"""
		return (self.lons.min(), self.lons.max(), self.lats.min(), self.lats.max())

	@classmethod
	def from_generic_site_model(cls, site_model, soil_params=REF_SOIL_PARAMS,
								name=''):
		"""
		Generate soil site model from generic site model and soil parameters

		:param soil_params:
			dict, containing soil parameters (vs30, vs30measured, z1pt0,
			z2pt5, and kappa)
			(default: REF_SOIL_PARAMS)
		:param name:
			str, name of soil site model
			(default: "")

		:return:
			instance of :class:`SoilSiteModel`
		"""
		return sha_site_model.to_soil_site_model(ref_soil_params=soil_params, name=name)

	@classmethod
	def from_polygon(cls, polygon, spacing, depth=None,
					soil_params=REF_SOIL_PARAMS, name=""):
		"""
		Genereate soil site model from polygon, spacing and soil parameters

		:param polygon:
			instance of :class:`oqhazlib.geo.polygon.Polygon`
		:param spacing:
			float, spacing in km for discretizing polygon
		:param depth:
			float, depth in km
			(default: None)
		:param soil_params:
		:param name:
			see :meth:`from_generic_site_model`

		:return:
			instance of :class:`SoilSiteModel`
		"""
		site_model = GenericSiteModel.from_polygon(polygon, spacing, depth=depth)
		return self.from_generic_site_model(site_model, soil_params=soil_params,
											name=name)

	def get_sites(self):
		"""
		:return:
			list with instances of :class:`SoilSite`
		"""
		sites = [self.__getitem__(i) for i in range(self.num_sites)]
		return sites

	def get_generic_sites(self):
		"""
		:return:
			list with instances of :class:`GenericSite`
		"""
		sites = []
		for pt, site_name in zip(self.mesh, self.site_names):
			sites.append(GenericSite(pt.longitude, pt.latitude, pt.depth, site_name))
		return sites

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML SiteModel element)

		:param encoding:
			String, unicode encoding (default: 'latin1')
		"""
		soil_site_model_elem = etree.Element(ns.SITE_MODEL)

		kappa = self.kappa.all() and not np.isnan(self.kappa).all()

		for i in range(self.num_sites):
			site_elem = etree.SubElement(soil_site_model_elem, ns.SITE)
			site_elem.set(ns.LON, str(self.mesh.lons[i]))
			site_elem.set(ns.LAT, str(self.mesh.lats[i]))
			site_elem.set(ns.VS30, str(self.vs30[i]))
			site_elem.set(ns.VS30TYPE, xmlstr({True: 'measured', False: 'inferred'}[self.vs30measured[i]]))
			site_elem.set(ns.Z1PT0, str(self.z1pt0[i]))
			site_elem.set(ns.Z2PT5, str(self.z2pt5[i]))
			if kappa:
				site_elem.set(ns.KAPPA, str(self.kappa[i]))

		return soil_site_model_elem

	def print_xml(self, encoding='latin1'):
		"""
		Print XML to screen

		:param encoding:
			str, unicode encoding (default: 'utf-8')
		"""
		tree = create_nrml_root(self, encoding=encoding)
		print(etree.tostring(tree, xml_declaration=True, encoding=encoding,
							pretty_print=True))

	def write_xml(self, filespec, encoding='latin1', pretty_print=True):
		"""
		Write site model to XML file

		:param filespec:
			str, full path to XML output file
		:param encoding:
			str, unicode encoding (default: 'utf-8')
		:param pretty_print:
			bool, indicating whether or not to indent each element
			(default: True)
		"""
		tree = create_nrml_root(self, encoding=encoding)
		tree.write(open(filespec, 'w'), xml_declaration=True, encoding=encoding,
					pretty_print=pretty_print)
