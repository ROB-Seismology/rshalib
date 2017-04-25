# -*- coding: utf-8 -*-

"""
Classes representing source-model elements in Openquake/nhlib. Where possible,
the classes are inherited from nhlib classes. All provide methods to create
XML elements, which are used to write a complete source-model NRML file.
Thus, objects instantiated from these classes can be used directly in nhlib,
as well as to generate input files for OpenQuake.
"""

from lxml import etree

import openquake.hazardlib as oqhazlib
from openquake.hazardlib.scalerel import WC1994

from ..nrml import ns
from ..nrml.common import *
from ..mfd import *
from source import (PointSource, AreaSource, SimpleFaultSource, ComplexFaultSource,
					CharacteristicFaultSource)

import jsonpickle
import base64, zlib
import numpy

class NumpyFloatHandler(jsonpickle.handlers.BaseHandler):
    def flatten(self, obj, data):
        """
        Converts and round to float an encod
        """
        return round(obj,6)

jsonpickle.handlers.registry.register(numpy.float, NumpyFloatHandler)
jsonpickle.handlers.registry.register(numpy.float32, NumpyFloatHandler)
jsonpickle.handlers.registry.register(numpy.float64, NumpyFloatHandler)


class NDArrayHandler(jsonpickle.handlers.BaseHandler):
	"""
	A JSON-pickler for NumPy arrays.

	The raw bytes are compressed using zlib and then base 64 encoded.
	"""

	def flatten(self, arr, data):
		dtype = arr.dtype.str
		if dtype != '|O4':
			data['dtype'] = dtype
			data['bytes'] = base64.b64encode(zlib.compress(arr.tostring()))
			data['shape'] = arr.shape
		else:
			## Object arrays
			p = jsonpickle.pickler.Pickler()
			data = p.flatten(arr.tolist())
		return data

	def restore(self, data):
		byte_str = zlib.decompress(base64.b64decode(data['bytes']))
		array = numpy.fromstring(byte_str, dtype=data['dtype'])
		return array.reshape(data['shape'])

jsonpickle.handlers.registry.register(numpy.ndarray, NDArrayHandler)


class SourceModel():
	"""
	Class representing a complete source model

	:param name:
		String, source model name
	:param sources:
		list of source objects (instances of :class:`PointSource`,
		:class:`AreaSource`, :class:`SimpleFaultSource` and/or
		:class:`ComplexFaultSource`)
	:param description:
		String, containing optional description (e.g. logic-tree path)
		(default: "")
	"""
	def __init__(self, name, sources, description=""):
		self.name = name
		self.sources = sources
		self.description = description
		self.validate()

	# Note: __getattr__ causes recursion when used in conjunction with
	# multiprocesing
	#def __getattr__(self, name):
		"""
		Make sources accessible with their ID as object properties
		"""
	#	try:
	#		index = self.source_ids.index(name)
	#	except ValueError:
	#		raise AttributeError(name)
	#	else:
	#		return self.sources[index]

	def __getitem__(self, index_or_name):
		"""
		Make sources accessible with their ID as key
		"""
		if isinstance(index_or_name, int):
			index = index_or_name
		elif isinstance(index_or_name, (str, unicode)):
			name = index_or_name
			source_ids = self.source_ids
			if name in source_ids:
				index = source_ids.index(name)
			else:
				raise KeyError(name)
		else:
			raise KeyError(index_or_name)

		return self.sources[index]

	def __len__(self):
		return len(self.sources)

	def __iter__(self):
		for source in self.sources:
			yield source

	@property
	def source_ids(self):
		return [src.source_id for src in self.sources]

	def to_json(self):
		return jsonpickle.encode(self)

	@classmethod
	def from_json(self, json_string):
		return jsonpickle.decode(json_string)

	@classmethod
	def from_eq_catalog(self, eq_catalog, Mtype="MW", Mrelation={}, area_source_model_name=None,
				tectonic_region_type="Stable Shallow Crust",
				magnitude_scaling_relationship=WC1994(),
				rupture_mesh_spacing=1., rupture_aspect_ratio=1.,
				upper_seismogenic_depth=5., lower_seismogenic_depth=25.,
				nodal_plane_distribution=None, hypocenter_distribution=None,
				synthetic=False):
		"""
		Construct point source model from earthquake catalog.

		Note: if area_source_model is provided, it overrides trt, usd,
		lsd, npd and hdd. See :meth:`from_eq_record` for order of
		precedence for determining nodal_plane_distribution and
		hypocenter_distribution

		:param eq_catalog:
			instance of :class:`EQCatalog`

		...

		:param synthetic:
			bool, whether catalog is synthetic or not, to avoid lookup
			in focal mechanisms database
			(default: False)

		:return:
			instance of :class:`SourceModel`
		"""
		if isinstance(magnitude_scaling_relationship, (str, unicode)):
			magnitude_scaling_relationship = getattr(oqhazlib.scalerel, magnitude_scaling_relationship)()
		src_list = []
		if area_source_model_name:
			#from ..rob import create_rob_source_model
			from read_from_gis import read_rob_source_model
			zone_catalogs = eq_catalog.split_into_zones(area_source_model_name, verbose=False)
			#source_model = create_rob_source_model(area_source_model_name, min_mag=2., verbose=False)
			source_model = read_rob_source_model(area_source_model_name, min_mag=2., verbose=False)
			for zone_id in zone_catalogs.keys():
				zone_catalog = zone_catalogs[zone_id]
				source_zone = source_model[zone_id]
				for eq in zone_catalog:
					tectonic_region_type = source_zone.tectonic_region_type
					upper_seismogenic_depth = source_zone.upper_seismogenic_depth
					lower_seismogenic_depth = source_zone.lower_seismogenic_depth
					nodal_plane_distribution = source_zone.nodal_plane_distribution
					hypocenter_distribution = source_zone.nodal_plane_distribution

					## TODO: nodal plane distribution from as model
					pt_src = PointSource.from_eq_record(eq, Mtype, Mrelation,
						tectonic_region_type, magnitude_scaling_relationship,
						rupture_mesh_spacing, rupture_aspect_ratio,
						upper_seismogenic_depth, lower_seismogenic_depth,
						nodal_plane_distribution, hypocenter_distribution,
						synthetic=synthetic)
					src_list.append(pt_src)

		else:
			for eq in eq_catalog:
				pt_src = PointSource.from_eq_record(eq, Mtype, Mrelation,
					tectonic_region_type, magnitude_scaling_relationship,
					rupture_mesh_spacing, rupture_aspect_ratio,
					upper_seismogenic_depth, lower_seismogenic_depth,
					nodal_plane_distribution, hypocenter_distribution,
					synthetic=synthetic)
				src_list.append(pt_src)

		return SourceModel(eq_catalog.name, src_list)

	def append(self, source):
		"""
		Append source object

		:param source:
			instance of :class:`PointSource`, :class:`AreaSource`,
			:class:`SimpleFaultSource` and/or :class:`ComplexFaultSource`
		"""
		self.sources.append(source)
		self.validate()

	def extend(self, sources):
		"""
		Extend with list of source objects

		:param sources:
			list of source objects (instances of :class:`PointSource`,
			:class:`AreaSource`, :class:`SimpleFaultSource` and/or
			:class:`ComplexFaultSource`)
		"""
		self.sources.extend(sources)
		self.validate()

	def validate(self):
		"""
		Make sure there are no duplicate source ID's
		"""
		source_ids = []
		for source in self.sources:
			if not source.source_id in source_ids:
				source_ids.append(source.source_id)
			else:
				raise NRMLError("Duplicate source id found: %s" % source.source_id)

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML SourceModel element)

		:param encoding:
			String, unicode encoding (default: 'latin1')
		"""
		sourceModel_elem = etree.Element(ns.SOURCE_MODEL)
		sourceModel_elem.set(ns.NAME, self.name)

		for source in self.sources:
			sourceModel_elem.append(source.create_xml_element(encoding=encoding))

		return sourceModel_elem

	def write_xml(self, filespec, encoding='latin1', pretty_print=True):
		"""
		Write source model to XML file

		:param filespec:
			String, full path to XML output file
		:param encoding:
			String, unicode encoding (default: 'utf-8')
		:param pretty_print:
			boolean, indicating whether or not to indent each element
			(default: True)
		"""
		tree = create_nrml_root(self, encoding=encoding)
		tree.write(open(filespec, 'w'), xml_declaration=True, encoding=encoding, pretty_print=pretty_print)

	def print_xml(self):
		"""
		Print XML to screen
		"""
		encoding='latin1'
		tree = create_nrml_root(self, encoding=encoding)
		print etree.tostring(tree, xml_declaration=True, encoding=encoding, pretty_print=True)

	@property
	def min_mag(self):
		return min([source.min_mag for source in self.sources])

	@property
	def max_mag(self):
		return max([source.max_mag for source in self.sources])

	@property
	def lower_seismogenic_depth(self):
		return max([source.lower_seismogenic_depth for source in self.sources])

	@property
	def upper_seismogenic_depth(self):
		return min([source.upper_seismogenic_depth for source in self.sources])

	def get_sources_by_trt(self, trt):
		"""
		Fetch sources of a given tectonic region type

		:param trt:
			str, tectonic region type
			If None or empty string, all sources will be selected.

		:return:
			list with source objects
		"""
		if not trt:
			return self.sources
		else:
			return [source for source in self.sources if src.tectonic_region_type == trt]

	def get_sources_by_type(self, source_type):
		"""
		Fetch sources of a given type

		:param source_type:
			str, one of "point", "area", "fault", "simple_fault", "complex_fault"
			or "non_area"
			If None, all sources will be selected

		:return:
			list with source objects
		"""
		if not source_type:
			return self.sources
		else:
			meth_name = "get_%s_sources" % source_type
			return getattr(self, meth_name)()

	def get_point_sources(self):
		return [source for source in self.sources if isinstance(source, PointSource)]

	def get_area_sources(self):
		return [source for source in self.sources if isinstance(source, AreaSource)]

	def get_non_area_sources(self):
		return [source for source in self.sources if not isinstance(source, AreaSource)]

	def get_simple_fault_sources(self):
		return [source for source in self.sources if isinstance(source, SimpleFaultSource)]

	def get_complex_fault_sources(self):
		return [source for source in self.sources if isinstance(source, ComplexFaultSource)]

	def get_fault_sources(self):
		return [source for source in self.sources if isinstance(source, (SimpleFaultSource, ComplexFaultSource, CharacteristicFaultSource))]

	def set_fault_MFDs_from_BG_zones(self):
		"""
		Set MFD's of fault sources in the model from MFD of background zone,
		weighted by moment rate of fault sources. min_mag and max_mag of original
		fault MFD's will be preserved.
		"""
		fault_sources = self.get_fault_sources()
		fault_source_groups = {}
		for fault in fault_sources:
			if fault.bg_zone in fault_source_groups.keys():
				fault_source_groups[fault.bg_zone].append(fault)
			else:
				fault_source_groups[fault.bg_zone] = [fault]

		for bg_zone_name, fault_source_group in fault_source_groups.items():
			bg_zone = getattr(self, bg_zone_name)
			moment_rates = numpy.zeros(len(fault_source_group), 'd')
			for i, fault in enumerate(fault_source_group):
				moment_rates[i] = fault.get_moment_rate()
			moment_rate_weights = moment_rates / numpy.add.reduce(moment_rates)
			fault_mfds = bg_zone.mfd.divide(moment_rate_weights)
			for fault, fault_mfd in zip(fault_source_group, fault_mfds):
				min_mag, max_mag = fault.mfd.min_mag, fault.mfd.max_mag
				fault.mfd = fault_mfd
				fault.mfd.min_mag = min_mag
				fault.mfd.max_mag = max_mag

	def get_bounding_box(self):
		"""
		Determine rectangular bounding box of source model

		:return:
			(west, east, south, north) tuple
		"""
		# TODO: implement for complex fault sources as well
		regions = [src.get_bounding_box() for src in self.sources]
		regions = numpy.array(regions)
		w, _, s, _ = regions.min(axis=0)
		_, e, _, n = regions.max(axis=0)
		return (w, e, s, n)

	def decompose_area_sources(self):
		"""
		Create a new source model where all area sources are replaced with
		point sources.

		:return:
			instance of :class:`SourceModel`
		"""
		sources = []
		for src in self.sources:
			if isinstance(src, AreaSource):
				sources.extend(src.to_point_sources())
			else:
				sources.append(src)
		return SourceModel(self.name, sources, self.description)

	def get_tectonic_region_types(self):
		"""
		Return list of unique tectonic region types in the source model

		:return:
			list of strings
		"""
		trts = set([src.tectonic_region_type for src in self.sources])
		return list(trts)

	def split_by_trt(self):
		"""
		Split source model by tectonic region type

		:return:
			dict mapping tectonic region types (strings) to instances of
			:class:`SourceModel`
		"""
		trts = self.get_tectonic_region_types()
		source_model_dict = {}
		for trt in trts:
			source_model_dict[trt] = []
		for src in self.sources:
			source_model_dict[src.tectonic_region_type].append(src)
		for trt in trts:
			trt_short_name = ''.join([word[0].capitalize() for word in trt.split()])
			name = "%s -- %s" % (self.name, trt_short_name)
			sources = source_model_dict[trt]
			source_model_dict[trt] = SourceModel(name, sources, self.description)
		return source_model_dict

	def split_by_source(self):
		"""
		Split source model into source models containing one source each

		:return:
			list with instances of :class:`SourceModel`
		"""
		source_model_list = []
		for src in self.sources:
			name = "%s -- %s" % (self.name, src.source_id)
			source_model = SourceModel(name, [src], self.description)
			source_model_list.append(source_model)
		return source_model_list

	def get_num_decomposed_sources(self):
		"""
		Return list with number of decomposed sources for each source.
		"""
		num_decomposed_sources = []
		for src in self.sources:
			if isinstance(src, AreaSource):
				polygon_mesh = src.polygon.discretize(src.area_discretization)
				num_decomposed_sources.append(len(polygon_mesh))
			else:
				num_decomposed_sources.append(1)
		return num_decomposed_sources

	def is_background_source(self, src_id):
		"""
		Determine if a particular source represents a background source

		:param src_id:
			str, source ID

		:return:
			bool
		"""
		src = self.__getitem__(src_id)
		if not isinstance(src, AreaSource):
			return False
		else:
			for flt in self.get_simple_fault_sources():
				if flt.bg_zone == src_id:
					return True
			else:
				return False

	def get_total_fault_length(self):
		"""
		Report total length of (simple) faults in source model

		:return:
			float, fault length in km
		"""
		flt_len = 0
		for flt in self.get_simple_fault_sources():
			flt_len += flt.get_length()
		return flt_len

	def get_total_fault_area(self):
		"""
		Report total area of (simple) faults in source model

		:return:
			float, fault area in square km
		"""
		total_area = 0
		for flt in self.get_simple_fault_sources():
			total_area += flt.get_area()
		return total_area

	def get_total_area(self):
		"""
		Report total surface area of area sources in source model

		:return:
			float, surface area in square km
		"""
		total_area = 0
		for src in self.get_area_sources():
			total_area += src.get_area()
		return total_area

	def get_average_seismogenic_thickness(self):
		"""
		Determine average seismogenic thickness of all area sources,
		weighted by area.
		"""
		seismogenic_thicknesses = [src.get_seismogenic_thickness() for src in self.get_area_sources()]
		areas = [src.get_area() for src in self.get_area_sources()]
		return numpy.average(seismogenic_thicknesses, weights=areas)

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
		return (strain_rate * 2 * rigidity * self.get_total_area() * 1E+6
				* self.get_average_seismogenic_thickness() * 1E+3)

	def get_summed_mfd(self):
		"""
		Compute summed MFD of all sources in model

		:return:
			instance of :class:`EvenlyDiscretizedMFD` or :class:`TruncatedGRMFD`
		"""
		mfd_list = [src.mfd for src in self.sources]
		return sum_MFDs(mfd_list)

	def get_summed_fault_mfd(self):
		"""
		Compute summed MFD of all fault sources in model

		:return:
			instance of :class:`EvenlyDiscretizedMFD` or :class:`TruncatedGRMFD`
		"""
		mfd_list = [src.mfd for src in self.get_fault_sources()]
		return sum_MFDs(mfd_list)

	def get_summed_area_source_mfd(self):
		"""
		Compute summed MFD of all area sources in model

		:return:
			instance of :class:`EvenlyDiscretizedMFD` or :class:`TruncatedGRMFD`
		"""
		mfd_list = [src.mfd for src in self.get_area_sources()]
		return sum_MFDs(mfd_list)

	def get_plot(self, region=None, projection="merc", resolution="i", graticule_interval=(1., 1.),
				point_source_style="default", area_source_style="default",
				fault_source_style="default", countries_style="default",
				coastline_style="default", label_style="default",
				title=None, ax=None, **kwargs):
		"""
		Plot hazard map

		:param region:
			(west, east, south, north) tuple specifying rectangular region
			to plot in geographic coordinates
			(default: None)
		:param projection:
			string, map projection. See Basemap documentation
			(default: "merc")
		:param resolution:
			char, resolution of builtin shorelines / country borders:
			'c' (crude), 'l' (low), 'i' (intermediate), 'h' (high), 'f' (full)
			(default: 'i')
		:param graticule_interval:
			(dlon, dlat) tuple of floats, spacing of grid lines (meridians,
			parallels) to draw over the map
			(default: (1., 1.)
		:param point_source_style:
			instance of :class:`LayeredBasemap.PointStyle`, defining
			style for plotting point sources
			(default: "default")
		:param area_source_style:
			instance of :class:`LayeredBasemap.PolygonStyle`, defining
			style for plotting area sources
			(default: "default")
		:param fault_source_style:
			instance of :class:`LayeredBasemap.LineStyle`, defining
			style for plotting fault sources
			(default: "default")
		:param countries_style:
			instance of :class:`LayeredBasemap.LineStyle`, defining
			style for plotting country borders
			(default: "default")
		:param coastline_style:
			instance of :class:`LayeredBasemap.LineStyle`, defining
			style for plotting coastlines
			(default: "default")
		:param label_style:
			instance of :class:`LayeredBasemap.TextStyle`, defining
			style for plotting source labels
			(default: "default")
		:param title:
			str, map title. If empty string, no title will be plotted.
			If None, default title will be used
			(default: None)
		:param ax:
			matplotlib Axes instance
			(default: None)
		:param kwargs:
			additional keyword arguments to be passed to LayeredBasemap
			constructor

		:return:
			instance of :class:`LayeredBasemap.LayeredBasemap`, where
			additional layers may be added before final plotting.
		"""
		import mapping.Basemap as lbm

		## Construct default styles:
		if label_style == "default":
			label_style = lbm.TextStyle(color='k', font_size=8, font_weight="bold",
					horizontal_alignment="center", vertical_alignment="center")
		if point_source_style == "default":
			point_source_style = lbm.PointStyle(shape='*', fill_color='yellow')
			point_source_style.label_style = label_style
		if area_source_style == "default":
			area_source_style = lbm.PolygonStyle(line_width=2, fill_color="none")
			area_source_style.label_style = label_style
		if fault_source_style == "default":
			fault_source_style = lbm.LineStyle(line_width=3, line_color='purple')
			fault_source_style.label_style = label_style
		source_model_style = lbm.CompositeStyle(line_style=fault_source_style,
											polygon_style=area_source_style,
											point_style=point_source_style)
		if countries_style == "default":
			countries_style = lbm.LineStyle(line_width=2, line_color="w")
		if coastline_style == "default":
			coastline_style = lbm.LineStyle(line_width=2, line_color="w")

		## Compute map limits
		if not region:
			region = self.get_bounding_box()

		map_layers = []

		## Coastlines and national boundaries
		if coastline_style:
			map_layers.append(lbm.MapLayer(lbm.BuiltinData("coastlines"), coastline_style, name="coastlines"))
		if countries_style:
			map_layers.append(lbm.MapLayer(lbm.BuiltinData("countries"), countries_style, name="countries"))

		## Source model
		legend_label = {}
		#legend_label["polygons"] = "Area sources"
		#legend_label["lines"] = "Fault sources"
		#legend_label["points"] = "Point sources"

		# TODO: add ComplexFaultSource
		# TODO: how to handle CharacteristicSource (doesn't have get_polygon method)?
		polygon_data = lbm.MultiPolygonData([], [])
		line_data = lbm.MultiLineData([], [])
		point_data = lbm.MultiPointData([], [])
		for source in self.sources:
			if isinstance(source, AreaSource):
				polygon_data.append(lbm.PolygonData(source.longitudes, source.latitudes, label=source.source_id))
				if not legend_label.has_key("polygons"):
					legend_label["polygons"] = "Area sources"
			elif isinstance(source, (SimpleFaultSource, CharacteristicFaultSource)):
				pg = source.get_polygon()
				polygon_data.append(lbm.PolygonData(pg.lons, pg.lats))
				fault_trace = source.fault_trace
				line_data.append(lbm.LineData(fault_trace.lons, fault_trace.lats, label=source.source_id))
				if not legend_label.has_key("lines"):
					legend_label["lines"] = "Fault sources"
			elif isinstance(source, PointSource):
				point_data.append(lbm.PointData(source.location.longitude, source.location.latitude, label=source.source_id))
				if not legend_label.has_key("points"):
					legend_label["points"] = "Point sources"
			else:
				print("Warning: Skipped plotting source %s, source type not supported" % source.source_id)
		sm_data = lbm.CompositeData(lines=line_data, polygons=polygon_data,
									points=point_data)
		sm_style = source_model_style
		map_layers.append(lbm.MapLayer(sm_data, sm_style, legend_label=legend_label, name="source_model"))

		## Title
		if title is None:
			title = self.name

		legend_style = lbm.LegendStyle(location=0)
		graticule_style = lbm.GraticuleStyle(annot_axes="SE")
		map = lbm.LayeredBasemap(map_layers, title, projection, region=region, graticule_interval=graticule_interval, resolution=resolution, graticule_style=graticule_style, legend_style=legend_style, ax=ax, **kwargs)
		return map

	@classmethod
	def from_nrml_file(cls, nrml_filespec, rupture_mesh_spacing, area_discretization):
		"""
		Read source model from NRML file

		:param nrml_filespec:
			string, full path to NRML file describing source model
		:param rupture_mesh_spacing:
			float, the desired distance between two adjacent points in source's
			ruptures' mesh, in km.
		:param area_discretization:
			float, polygon area discretization spacing in kilometers.

		:return:
			instance of :class:`SourceModel`
		"""
		from openquake.nrmllib.hazard.parsers import SourceModelParser
		import openquake.engine.input.source as input

		## Replace oqhazlib modules with rshalib modules
		from .. import geo
		from .. import mfd
		from .. import pmf
		from .. import source
		input.geo = geo
		input.mfd = mfd
		input.pmf = pmf
		input.source = source

		## Parse NRML and create NRML model objects
		parser = SourceModelParser(nrml_filespec)
		src_model = parser.parse()

		## Convert NRML model objects to rshalib objects
		sources = []
		for src_nrml in src_model:
			src = input.nrml_to_hazardlib(src_nrml, rupture_mesh_spacing, None,
										area_discretization)
			sources.append(src)
		description = "Imported from %s" % nrml_filespec
		return cls(src_model.name, sources, description=description)



if __name__ == '__main__':
	import openquake.hazardlib as nhlib
	## Test construction of source model with all 4 source typologies
	name = "Roer Valley Graben"
	trt = "Active Shallow Crust"
	mfd = TruncatedGRMFD(4.0, 6.7, 0.1, 1.2, 0.9)
	strike, dip, rake = 0, 60, -90
	np = NodalPlane(strike, dip, rake)
	npd = NodalPlaneDistribution([np], [1.0])
	hdd = HypocentralDepthDistribution([5., 10., 15., 20.], [0.125, 0.375, 0.375, 0.125])
	usd = 0.0
	lsd = 25.0
	msr = nhlib.scalerel.WC1994()
	rar = 1.0
	area_discretization = 1.0
	rupture_mesh_spacing = 1.0

	source_id = "1"
	point_list = [nhlib.geo.Point(4.0, 50.0), nhlib.geo.Point(5.0, 50.0), nhlib.geo.Point(5.0, 51.0)]
	polygon = Polygon(point_list)
	area_source = AreaSource(source_id=source_id, name=name, tectonic_region_type=trt,
		mfd=mfd, rupture_mesh_spacing=rupture_mesh_spacing, magnitude_scaling_relationship=msr,
		rupture_aspect_ratio=rar, upper_seismogenic_depth=usd, lower_seismogenic_depth=lsd,
		nodal_plane_distribution=npd, hypocenter_distribution=hdd, polygon=polygon,
		area_discretization=area_discretization)

	source_id = "2"
	location = Point(4.0, 50.0)
	point_source = PointSource(source_id=source_id, name=name, tectonic_region_type=trt,
		mfd=mfd, rupture_mesh_spacing=rupture_mesh_spacing, magnitude_scaling_relationship=msr,
		rupture_aspect_ratio=rar, upper_seismogenic_depth=usd, lower_seismogenic_depth=lsd,
		location=location, nodal_plane_distribution=npd, hypocenter_distribution=hdd)

	source_id = "3"
	fault_trace = Line(point_list)
	mfd = EvenlyDiscretizedMFD(4.1, 0.2, [1.0, 0.8, 0.6, 0.4, 0.2, 0.1])
	simple_fault_source = SimpleFaultSource(source_id=source_id, name=name,
		tectonic_region_type=trt, mfd=mfd, rupture_mesh_spacing=rupture_mesh_spacing,
		magnitude_scaling_relationship=msr, rupture_aspect_ratio=rar,
		upper_seismogenic_depth=usd, lower_seismogenic_depth=lsd,
		fault_trace=fault_trace, dip=dip, rake=rake)

	source_id = "4"
	edges = [Line(point_list), Line(point_list)]
	complex_fault_source = ComplexFaultSource(source_id=source_id, name=name,
		tectonic_region_type=trt, mfd=mfd, rupture_mesh_spacing=rupture_mesh_spacing,
		magnitude_scaling_relationship=msr, rupture_aspect_ratio=rar,
		edges=edges, rake=rake)

	sm = SourceModel("Test", [area_source, point_source, simple_fault_source, complex_fault_source])
	sm.print_xml()

	## Test iteration
	for source in sm:
		print source.source_id
	sources = list(sm)
	print sources[0].source_id

