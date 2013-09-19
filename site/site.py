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
			if longitude < 0:
				NorS = "S"
			else:
				NorS = "N"
			if latitude < 0:
				EorW = "W"
			else:
				EorW = "E"
			self.name = "%.3f %s, %.3f %s" % (abs(longitude), NorS, abs(latitude), EorW)
			if depth:
				self.name += ", %.1f" % depth
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


class SHASiteModel(object):
	"""
	"""
	# TODO: support names
	# TODO: subclass from hazardlib Mesh
	# TODO: plot method
	# TODO: iter method
	
	def __init__(self, sites=None, grid_outline=None, grid_spacing=None):
		"""
		"""
		assert sites != None or (grid_outline != None and grid_spacing != None)
		if sites != None:
			self._set_sites(sites)
			self.grid_outline = None
			self.grid_spacing = None
		else:
			self._set_grid_outline(grid_outline)
			self._set_grid_spacing(grid_spacing)
			self._set_grid()
			self.sites = None
	
	def __len__(self):
		"""
		"""
		return np.prod(self.shape)
	
	def _set_sites(self, sites):
		"""
		"""
		self.lons = np.zeros(len(sites))
		self.lats = np.zeros(len(sites))
		for i in xrange(len(sites)):
			self.lons[i] = sites[i][0]
			self.lats[i] = sites[i][1]
	
	def _set_grid_outline(self, grid_outline):
		"""
		"""
		if len(grid_outline) == 2:
			llc, urc = grid_outline
			lrc = (urc[0], llc[1])
			ulc = (llc[0], urc[1])
			self.grid_outline = np.array([llc, lrc, urc, ulc])
		if len(grid_outline) == 4 and isinstance(grid_outline[0], (int, float)):
			w, e, s, n = grid_outline
			llc = (w, s)
			lrc = (e, s)
			urc = (e, n)
			ulc = (w, n)
			self.grid_outline = np.array([llc, lrc, urc, ulc])
		else:
			self.grid_outline = np.array([(point[0], point[1]) for point in grid_outline])
	
	def _set_grid_spacing(self, grid_spacing):
		"""
		"""
		if isinstance(grid_spacing, (int, float)):
			self.grid_spacing = (grid_spacing, grid_spacing)
		else:
			self.grid_spacing = grid_spacing
	
	def _set_grid(self):
		"""
		"""
		if isinstance(self.grid_spacing, (str, unicode)):
			grid = Polygon([Point(lon, lat) for (lon, lat) in self.grid_outline]).discretize(float(self.grid_spacing))
			self.lons = grid.lons
			self.lats = grid.lats
		else:
			slons = np.arange(self.grid_outline[:,0].min(), self.grid_outline[:,0].max(),
				self.grid_spacing[0]) + self.grid_spacing[0] / 2.
			slats = np.arange(self.grid_outline[:,1].min(), self.grid_outline[:,1].max(),
				self.grid_spacing[1]) + self.grid_spacing[1] / 2.
			grid = np.dstack(np.meshgrid(slons, slats[::-1]))
			self.lons = grid[:,:,0]
			self.lats = grid[:,:,1]
	
	@property
	def shape(self):
		"""
		"""
		return self.lons.shape

	def get_sites(self):
		"""
		"""
		return [SHASite(self.lons[i], self.lats[i]) for i in np.ndindex(self.shape)]
	
	def to_soil_site_model(self, ref_soil_params=REF_SOIL_PARAMS):
		"""
		"""
		soil_sites = [SoilSite(self.lons[i], self.lats[i], soil_params=ref_soil_params) for i in np.ndindex(self.shape)]
		return SoilSiteModel("", soil_sites)


class SoilSite(nhlib.site.Site, SHASite):
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
	:param vs30:
		Float, average shear-wave velocity down to 30 m depth, in m/s
		(default: value defined in vs30.rock)
	:param vs30measured:
		Bool, indicating whether vs30 was actually measured
		(default: False)
	:param z1pt0:
	:param z2pt5:
	:param name:
		Site name (default: "")
	"""
	def __init__(self, longitude, latitude, depth=0, soil_params=REF_SOIL_PARAMS, name=""):
		location = Point(longitude, latitude, depth)
		nhlib.site.Site.__init__(self, location, **soil_params)
		SHASite.__init__(self, longitude, latitude, depth, name)


class SoilSiteModel(nhlib.site.SiteCollection):
	"""
	Class representing a complete site model

	:param name:
		String, site model name
	:param sources:
		list of site objects (instances of :class:`Site`)
	"""

	def __init__(self, name, sites):
		self.name = name
		self.num_sites = len(sites)
		super(SoilSiteModel, self).__init__(sites=sites)

	def __len__(self):
		return self.num_sites

	@property
	def sites(self):
		"""
		:return:
			list with instances of :class:`SHASite`
		"""
		return [SHASite(pt.longitude, pt.latitude, pt.depth) for pt in self.mesh]

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML SiteModel element)

		:param encoding:
			String, unicode encoding (default: 'latin1')
		"""
		soil_site_model_elem = etree.Element(ns.SITE_MODEL)

		for i in range(self.num_sites):
			site_elem = etree.SubElement(site_model_elem, ns.SITE)
			site_elem.set(ns.LON, str(self.mesh.lons[i]))
			site_elem.set(ns.LAT, str(self.mesh.lats[i]))
			site_elem.set(ns.VS30, str(self.vs30[i]))
			site_elem.set(ns.VS30TYPE, xmlstr({True: 'measured', False: 'inferred'}[self.vs30measured[i]]))
			site_elem.set(ns.Z1PT0, str(self.z1pt0[i]))
			site_elem.set(ns.Z2PT5, str(self.z2pt5[i]))

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
