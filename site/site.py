# -*- coding: utf-8 -*-

"""
Classes representing source-model elements in Openquake/nhlib. Where possible,
the classes are inherited from nhlib classes. All provide methods to create
XML elements, which are used to write a complete source-model NRML file.
Thus, objects instantiated from these classes can be used directly in nhlib,
as well as to generate input files for OpenQuake.
"""


# TODO: add kappa and thickness as soil_params


import numpy as np

from lxml import etree

import openquake.hazardlib as nhlib

from openquake.hazardlib.geo.geodetic import geodetic_distance

from ref_soil_params import REF_SOIL_PARAMS
from ..geo import Point, Polygon
from ..nrml import ns
from ..nrml.common import *


class SHASite(Point):
	"""
	Class representing a simple site (without soil characteristics) in SHA,
	derived from nhlib.geo.Point, but which can also be sliced, indexed, and
	looped over like a (longitude, latitude, depth) tuple.

	:param longitude:
		Float, longitude
	:param latitude:
		Float, latitude
	:param depth:
		Float, depth (default: 0.0)
	:param name:
		Site name (default: "")
	"""
	def __init__(self, longitude, latitude, depth=0.0, name=""):
		super(SHASite, self).__init__(longitude, latitude, depth)
		if not name:
			self.name = self.get_name_from_position()
		else:
			self.name = name

	def __str__(self):
		return self.name

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
		if self.longitude < 0:
			NorS = "S"
		else:
			NorS = "N"
		if self.latitude < 0:
			EorW = "W"
		else:
			EorW = "E"
		name = "%.3f %s, %.3f %s"
		name %= (abs(self.longitude), EorW, abs(self.latitude), NorS)
		if self.depth:
			name += ", %.1f" % self.depth
		return name

	def to_soil_site(self, soil_params=REF_SOIL_PARAMS):
		"""
		Convert to a SoilSite object, representing a site with
		soil characteristics

		:param vs30:
			Float, average shear-wave velocity down to 30 m depth, in m/s
			(default: value defined in vs30.rock)
		:param vs30measured:
			Bool, indicating whether vs30 was actually measured
			(default: False)
		:param z1pt0:
		:param z2pt5:

		:return:
			instance of :class:`SoilSite`
		"""
		return SoilSite(self.longitude, self.latitude, self.depth, soil_params, self.name)


class SHASiteModel(nhlib.geo.Mesh):
	"""
	Class representing a site model. Subclasses from :class:`openquake.hazardlib.geo.Mesh`.
	"""

	def __init__(self, lons=None, lats=None, depths=None, sites=None, names=None, grid_outline=None, grid_spacing=None):
		"""
		SiteModel is initiated by (1) lons and lats (and depths), (2) sites (and names) or (3) grid outline and grid spacing. Priority is given in this order.

		:param lons:
			nd np array, lons of sites (default: None)
		:param lats:
			nd np array, lats of sites (default: None)
		:param depths:
			nd np array, depths of sites (default: None, zero depths)
		:param sites:
			list of (float, float) tuples, (lon, lat) for each site
			or list of instances of SHASite
			or list of instances of SoilSite
			or instance of SHASiteModel
			(default: None)
		:param names:
			list of str, names of sites (length must equal that of sites)
			(default: None)
		:param grid_outline
			list of (float, float) tuples, (lon, lat) for each site of grid outline (if two these are llc and urc)
			or list of instances of (subclasses) of :class:`rshalib.geo.Point`
			or (float, float, float, float) tuple, (w lon, e lon, s lat, n lat)
			(default: None)
		:param grid_spacing:
			float, grid spacing in degrees for both lon and lats
			or (float, float) tuple, grid spacing in degrees for lon and lat separatly
			or str, grid spacing in km (string must end with "km")
			(default: None)
		"""
		if not (lons is None or lats is None): ## (1) lons and lats (and depths)
			super(SHASiteModel, self).__init__(lons, lats, depths)
			self.sites = None
			self.names = None
			self.grid_outline = None
			self.grid_spacing = None
		elif sites: ## (2) sites (and names)
			assert names == None or len(names) == len(sites)
			self.names = names
			self._set_sites(sites)
			self.grid_outline = None
			self.grid_spacing = None
		else: ## (3) grid outline and grid spacing
			self.names = None
			self._set_grid_outline(grid_outline)
			self._set_grid_spacing(grid_spacing)
			self._set_grid()
			self.depths = None

	def __iter__(self):
		"""
		:return:
			iterator, (lon, lat) or (lon, lat, depth) for each site
		"""
		lons = self.lons.flat
		lats = self.lats.flat
		if not self.depths is None:
			depths = self.depths.flat
			for i in xrange(len(self)):
				yield (lons[i], lats[i], depths[i])
		else:
			for i in xrange(len(self)):
				yield (lons[i], lats[i])

	def __contains__(self, site):
		"""
		Determine if site model contains a given site (based on position
		only).

		:param site:
			instance of :class:`SHASite` or :class:`SoilSite`

		:return:
			bool
		"""
		idx = np.argmin(np.abs(self.lons - site.lon))
		if np.allclose([self.lons[idx], self.lats[idx]], [site.lon, site.lat], atol=1E-6):
			return True
		else:
			return False

	def _set_sites(self, sites):
		"""
		Sets lons, lats (and depths and names) from sites. Handles (lon, lat) or (lon, lat, depth) tuples, instances of SHASite, instances of Soilsites or instance of SHASiteModel.
		"""
		self.lons = np.zeros(len(sites))
		self.lats = np.zeros(len(sites))
		self.depths = np.zeros(len(sites))
		names = [""] * len(sites)
		for i in xrange(len(sites)):
			self.lons[i] = sites[i][0]
			self.lats[i] = sites[i][1]
			if len(sites[i]) == 3:
				self.depths[i] = sites[i][2]
			if hasattr(sites[i], "name"):
				names[i] = sites[i].name
		if np.allclose(self.depths, 0.):
			self.depths = None
		if len(set(names)) != 1 or (len(set(names)) == 1 and names[0] != ""):
			self.names = names

	def _set_grid_outline(self, grid_outline):
		"""
		Sets grid_outline.
		"""
		if len(grid_outline) < 2:
			raise Exception("grid_outline must contain 2 points at least")
		elif len(grid_outline) == 2:
			llc, urc = grid_outline
			lrc = (urc[0], llc[1])
			ulc = (llc[0], urc[1])
			self.grid_outline = [llc, lrc, urc, ulc]
		elif len(grid_outline) == 4 and isinstance(grid_outline[0], (int, float)):
			w, e, s, n = grid_outline
			llc = (w, s)
			lrc = (e, s)
			urc = (e, n)
			ulc = (w, n)
			self.grid_outline = [llc, lrc, urc, ulc]
		else:
			self.grid_outline = [(site[0], site[1]) for site in grid_outline]

	def _set_grid_spacing(self, grid_spacing):
		"""
		Sets grid_spacing.
		"""
		if isinstance(grid_spacing, (int, float)):
			self.grid_spacing = (grid_spacing, grid_spacing)
		else:
			if isinstance(grid_spacing, (str, unicode)):
				 assert grid_spacing.endswith("km")
			self.grid_spacing = grid_spacing

	def _set_grid(self):
		"""
		Sets lons and lats by grid_outline and grid_spacing.
		"""
		if isinstance(self.grid_spacing, (str, unicode)):
			grid = Polygon([Point(lon, lat) for (lon, lat) in self.grid_outline]).discretize(float(self.grid_spacing[:-2]))
			self.lons = grid.lons
			self.lats = grid.lats
		else:
			from ..utils import seq

			slons = seq(np.array(self.grid_outline)[:,0].min(), np.array(self.grid_outline)[:,0].max(), self.grid_spacing[0])
			slats = seq(np.array(self.grid_outline)[:,1].min(), np.array(self.grid_outline)[:,1].max(), self.grid_spacing[1])
			grid = np.dstack(np.meshgrid(slons, slats[::-1]))
			self.lons = grid[:,:,0]
			self.lats = grid[:,:,1]

	@classmethod
	def from_grid_spec(cls, grid_outline, grid_spacing):
		pass

	@classmethod
	def from_polygon(cls, polygon, spacing, depth=None):
		"""
		Construct SHA site model from polygon.

		:param polygon:
			instance of :class:`oqhazlib.geo.polygon.Polygon`
		:param spacing:
			float, spacing in km for discretizing polygonµ
		:param depth:
			float, depth in km
			(default: None)

		:return:
			instance of :class:`SHASiteModel`
		"""
		mesh = polygon.discretize(spacing)
		if depth:
			depths = [depth] * len(mesh)
		else:
			depths = None
		return cls(lons=mesh.lons, lats=mesh.lats, depths=depths)

	@property
	def region(self):
		"""
		:return:
			(float, float, float, float) tuple, (w, e, s, n) of bounding box
		"""
		return (self.lons.min(), self.lons.max(), self.lats.min(), self.lats.max())

	@property
	def slons(self):
		"""
		:return:
			1d np array, set of lons if model is a lon lat grid, otherwise None
		"""
		if len(self.shape) == 2:
			return self.lons[0,:]
		else:
			return None

	@property
	def slats(self):
		"""
		:return:
			1d np array, set of lats if model is a lon lat grid, otherwise None
		"""
		if len(self.shape) == 2:
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
			np array like self.shape, distances in km
		"""
		return geodetic_distance(lon, lat, self.lons, self.lats)

	def get_site(self, lon, lat, index=False):
		"""
		Get (closest) site.

		:param lon:
			float, lon of site.
		:param lat:
			float, lat of site.
		:param index:
			bool, give index instead of site (default: False)

		:return:
			tuple (int, int) or int, index of site
		"""
		i = np.unravel_index(self._geodetic_min_distance(type(self)(lons=np.array([lon]), lats=np.array([lat])), True)[0], self.shape)
		if index == False:
			if self.depths != None:
				depth = self.depths[i]
			else:
				depth = 0
			if self.names != None:
				name = self.names[i]
			else:
				name = ""
			return SHASite(self.lons[i], self.lats[i], depth, name)
		else:
			return i

	def clip(self):
		"""
		Clip the site model by the grid outline if it is a degree spaced grid.

		:return:
			instance of SHASiteModel
		"""
		if len(self.shape) != 2:
			return self
		else:
			mask = Polygon([Point(*point) for point in self.grid_outline]).intersects(self)
			lons = self.lons[mask]
			lats = self.lats[mask]
			if self.depths != None:
				depths = self.depths[mask]
			else:
				depths = None
			return type(self)(lons=lons, lats=lats, depths=depths)

	def get_sites(self, clip=False):
		"""
		Get sites of site model.

		:param clip:
			bool, clip site model or not (default: False)

		:return:
			list of instances of SHASite
		"""
		if clip == True:
			site_model = self.clip()
		else:
			site_model = self
		sites = []
		for i, point in enumerate(site_model):
			if self.names != None:
				name = self.names[i]
			else:
				name = ""
			sites.append(SHASite(*point, name=name))
		return sites

	def to_soil_site_model(self, name="", ref_soil_params=REF_SOIL_PARAMS):
		"""
		Get soil site model from site model with reference soil parameters for each site.

		:param name:
			str, name of soil site model (default: "")
		:param ref_soil_params:
			dict, reference value for each soil parameter needed by soil site model (default: defaults specified as REF_SOIL_PARAMS in ref_soil_params module)

		:return:
			instance of SoilSiteModel
		"""
		soil_sites = [site.to_soil_site(ref_soil_params) for site in self.get_sites()]
		return SoilSiteModel(name, soil_sites)

	def plot(self):
		"""
		"""
		from mapping.layeredbasemap.layered_basemap import MapLayer, LayeredBasemap
		from mapping.layeredbasemap.data_types import MultiPointData, BuiltinData
		from mapping.layeredbasemap.styles import PointStyle, LineStyle

		map_layers = []
		map_layers.extend([MapLayer(data=MultiPointData(self.lons, self.lats), style=PointStyle(shape=".", size=5))])
		map_layers.extend([MapLayer(BuiltinData("coastlines"), LineStyle()), MapLayer(BuiltinData("countries"), LineStyle())])
		map = LayeredBasemap(layers=map_layers, region=self.region, projection="merc",
			resolution="i", title="Test")
		map.plot()


class SoilSite(nhlib.site.Site):
	"""
	Class representing a site with soil characteristics.
	This class extends :class:`nhlib.site.Site` by providing longitude,
	latitude, depth, and name properties

	:param longitude:
		Float, longitude
	:param latitude:
		Float, latitude
	:param depth:
		Float, depth (default: 0.0)
	:param soil_params:
		Dict, containing soil parameters (vs30, vs30measured, z1pt0, z2pt5, and kappa)
		(default: REF_SOIL_PARAMS)
	:param name:
		Site name (default: "")
	"""
	def __init__(self, longitude, latitude, depth=0, soil_params=REF_SOIL_PARAMS, name=""):
		location = Point(longitude, latitude, depth)
		super(SoilSite, self).__init__(location, **soil_params)
		self.name = name
		#SHASite.__init__(self, longitude, latitude, depth, name)

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

	def to_sha_site(self):
		"""
		Convert to a SHASite object, representing a site without
		soil characteristics

		:return:
			instance of :class:`SHASite`
		"""
		return SHASite(self.lon, self.lat, self.depth, self.name)


class SoilSiteModel(nhlib.site.SiteCollection):
	"""
	Class representing a soil site model.

	:param name:
		String, site model name
	:param sites:
		list of site objects (instances of :class:`Site`)
	"""

	def __init__(self, name, sites):
		self.name = name
		self.num_sites = len(sites)
		super(SoilSiteModel, self).__init__(sites=sites)
		self.site_names = [site.name for site in sites]

	def __len__(self):
		return self.num_sites

	def __contains__(self, site):
		"""
		Determine if site model contains a given site (based on position
		only).

		:param site:
			instance of :class:`SHASite` or :class:`SoilSite`

		:return:
			bool
		"""
		idx = np.argmin(np.abs(self.lons - site.lon))
		if np.allclose([self.lons[idx], self.lats[idx]], [site.lon, site.lat], atol=1E-6):
			return True
		else:
			return False

	@property
	def lons(self):
		return self.mesh.lons

	@property
	def lats(self):
		return self.mesh.lats

	def get_bbox(self):
		"""
		Determine bounding box:

		:return:
			(lonmin, lonmax, latmin, latmax) tuple
		"""
		return (self.lons.min(), self.lons.max(), self.lats.min(), self.lats.max())

	@classmethod
	def from_polygon(cls, polygon, spacing, soil_params=REF_SOIL_PARAMS,
					depth=None, name=""):
		"""
		:param polygon:
			instance of :class:`oqhazlib.geo.polygon.Polygon`
		:param spacing:
			float, spacing in km for discretizing polygonµ
		:param soil_params:
			dict, containing soil parameters (vs30, vs30measured, z1pt0, z2pt5, and kappa)
			(default: REF_SOIL_PARAMS)
		:param depth:
			float, depth in km
			(default: None)
		:param name:
			str, name of soil site model
			(default: "")

		:return:
			instance of :class:`SoilSiteModel`
		"""
		sha_site_model = SHASiteModel.from_polygon(polygon, spacing, depth=depth)
		return sha_site_model.to_soil_site_model(name=name, ref_soil_params=soil_params)

	def get_sites(self):
		"""
		:return:
			list with instances of :class:`SoilSite`
		"""
		sites = []
		for i in range(self.num_sites):
			lon = self.mesh.lons[i]
			lat = self.mesh.lats[i]
			## Note: depth requires patch in oqhazlib
			if self.mesh.depths:
				depth = self.mesh.depths[i]
			else:
				depth = 0
			vs30 = self.vs30[i]
			vs30measured = self.vs30measured[i]
			z1pt0 = self.z1pt0[i]
			z2pt5 = self.z2pt5[i]
			kappa = self.kappa[i]
			soil_params = {"vs30": vs30, "vs30measured": vs30measured, "z1pt0": z1pt0, "z2pt5": z2pt5, "kappa": kappa}
			try:
				name = self.site_names[i]
			except:
				name = ""
			site = SoilSite(lon, lat, depth, soil_params, name)
			sites.append(site)
		return sites

	def get_sha_sites(self):
		"""
		:return:
			list with instances of :class:`SHASite`
		"""
		sites = []
		for pt, site_name in zip(self.mesh, self.site_names):
			sites.append(SHASite(pt.longitude, pt.latitude, pt.depth, site_name))
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

	def print_xml(self):
		"""
		Print XML to screen
		"""
		encoding='latin1'
		tree = create_nrml_root(self, encoding=encoding)
		print etree.tostring(tree, xml_declaration=True, encoding=encoding, pretty_print=True)

	def write_xml(self, filespec, encoding='latin1', pretty_print=True):
		"""
		Write site model to XML file

		:param filespec:
			String, full path to XML output file
		:param encoding:
			String, unicode encoding (default: 'utf-8')
		:param pretty_print:
			boolean indicating whether or not to indent each element
			(default: True)
		"""
		tree = create_nrml_root(self, encoding=encoding)
		tree.write(open(filespec, 'w'), xml_declaration=True, encoding=encoding, pretty_print=pretty_print)


if __name__ == '__main__':
	#soil_site_model = create_site_model('test', [(2.15, 49.15), (2.65, 48.65)], True)
	#soil_site_model.write_xml()
	Mmin, Mmax = 5.0, 7.7
	Mmin_fault, Mmax_fault = 5.5, 6.2
	bin_width = 0.1
	aValue, bValue = 5.392, 1.261
	mfd_bg, mfd_fault = split_mfd_fault_bg(aValue, bValue, Mmin, Mmin_fault, Mmax, bin_width)
	mfd_bg, mfd_fault = divide_mfd_fault_bg(aValue, bValue, Mmin, Mmin_fault, Mmax_fault, Mmax, bin_width)
	print mfd_bg.get_annual_occurrence_rates()
	print mfd_fault.get_annual_occurrence_rates()

	mfd_list = divide_mfd_faults(aValue, bValue, Mmin, 5.7, [6.3, 7.0], [0.7, 0.3], [1E-4, 1E-4], bin_width)
	print mfd_list[0].get_annual_occurrence_rates()
