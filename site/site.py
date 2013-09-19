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
from ..geo import Point
import vs30


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

	def to_soil_site(self, vs30=vs30.rock, vs30measured=False, z1pt0=100., z2pt5=2.):
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
		return SoilSite(self.longitude, self.latitude, self.depth, vs30, vs30measured, z1pt0, z2pt5, self.name)


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
	def __init__(self, longitude, latitude, depth=0, vs30=vs30.rock, vs30measured=False, z1pt0=100., z2pt5=2., name=""):
		location = Point(longitude, latitude, depth)
		nhlib.site.Site.__init__(self, location, vs30, vs30measured, z1pt0, z2pt5)
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


def create_soil_site_model(name, lons_lats, grid=False, grid_spacing=0.1, vs30=vs30.rock, vs30measured=False, z1pt0=1., z2pt5=2.):
	"""
	Create site model from site longitudes and latitudes.

	:param lons_lats:
		List with (lon, lat) tuples for sites.
	:param grid:
		Boolean, if True lons_lats contains only two elements (left and lower right corner) from which rectangular grid is created.
	:param vs30:
		See class Site (Default: value defined in vs30.rock).
	:param vs30measured:
		See class Site (Default: False).
	:param z1pt0:
		See class Site (Default: 1.).
	:param z2pt5:
		See class Site (Default: 2.).
	"""
	sites = []
	if grid:
		min_lon, max_lon = lons_lats[0][0], lons_lats[1][0]
		min_lat, max_lat = lons_lats[1][1], lons_lats[0][1]
		lons = np.arange(min_lon, max_lon, grid_spacing)
		lats = np.arange(min_lat, max_lat, grid_spacing)
		lons_lats = []
		for lat in lats[::-1]:
			for lon in lons:
				lons_lats.append((lon, lat))
	for lon_lat in lons_lats:
		location = Point(lon_lat[0], lon_lat[1])
		sites.append(nhlib.site.Site(location, vs30, vs30measured, z1pt0, z2pt5))
	soil_site_model = SoilSiteModel(name, sites)
	return soil_site_model


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
