# -*- coding: utf-8 -*-

"""
Classes representing source-model elements in Openquake/nhlib. Where possible,
the classes are inherited from nhlib classes. All provide methods to create
XML elements, which are used to write a complete source-model NRML file.
Thus, objects instantiated from these classes can be used directly in nhlib,
as well as to generate input files for OpenQuake.
"""

from lxml import etree

from ..nrml import ns
from ..nrml.common import *
from source import PointSource, AreaSource, SimpleFaultSource, ComplexFaultSource

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

	def __getattr__(self, name):
		"""
		Make sources accessible with their ID as object properties
		"""
		try:
			index = self.source_ids.index(name)
		except ValueError:
			raise AttributeError(name)
		else:
			return self.sources[index]

	def __getitem__(self, index_or_name):
		"""
		Make sources accessible with their ID as key
		"""
		if isinstance(index_or_name, int):
			index = index_or_name
		elif isinstance(index_or_name, (str, unicode)):
			name = index_or_name
			if name in self.source_ids:
				index = self.source_ids.index(name)
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

	def to_json(self):
		return jsonpickle.encode(self)

	@classmethod
	def from_json(self, json_string):
		return jsonpickle.decode(json_string)

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
		self.source_ids = []
		for source in self.sources:
			if not source.source_id in self.source_ids:
				self.source_ids.append(source.source_id)
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

	def get_point_sources(self):
		return [source for source in self.sources if isinstance(source, PointSource)]

	def get_area_sources(self):
		return [source for source in self.sources if isinstance(source, AreaSource)]

	def get_simple_fault_sources(self):
		return [source for source in self.sources if isinstance(source, SimpleFaultSource)]

	def get_complex_fault_sources(self):
		return [source for source in self.sources if isinstance(source, ComplexFaultSource)]

	def get_fault_sources(self):
		return [source for source in self.sources if isinstance(source, (SimpleFaultSource, ComplexFaultSource))]

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

