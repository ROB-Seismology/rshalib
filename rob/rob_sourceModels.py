"""
Module to create nrml objects and files for ROB source models
"""


### imports
import numpy as np
import os
import decimal
from matplotlib import pyplot

from openquake.hazardlib.scalerel import WC1994

#import mapping.MIPython as MI
from eqcatalog.source_models import rob_source_models_dict

from ..mfd import TruncatedGRMFD, EvenlyDiscretizedMFD
from ..geo import Point, Line, Polygon, NodalPlane, mean_angle
from ..pmf.distributions import *
from ..source import AreaSource, SimpleFaultSource, SourceModel


# set precision for weights calculation
decimal.getcontext().prec = 4


def create_rob_source_model(source_model_name, min_mag=4.0, mfd_bin_width=0.1, column_map={}, **kwargs):
	"""
	Create OQ/nhlib source model from one of the known ROB source models
	(stored in MapInfo tables).

	:param source_model_name:
		String with name of ROB source model in ROB source models dict.
	:param min_mag:
		Float specifying minimum magnitude. Sources with Mmax <= Mmin are
		discarded (Default: 4.0)
	:param mfd_bin_width:
		Float, bin width of MFD (default: 0.1)
	:param column_map:
		Dict, mapping OQ/nhlib source parameters (keys) to MapInfo column names
		(values), thus overriding the values specified in the default column_map
		in rob_source_models_dict (Default: {}).
	:param kwargs:
		Keyword arguments supported by create_rob_area_source or
		create_rob_simple_fault_source

	:return:
		SourceModel object.
	"""
	from mapping.geo.readGIS import read_GIS_file

	## Initialze MapInfo
	#miApp = MI.Application(maximize=False)

	rob_source_model = rob_source_models_dict[source_model_name]
	#source_rec_table = miApp.OpenTable(rob_source_model['tab_filespec'])
	source_records = read_GIS_file(rob_source_model['tab_filespec'])

	## Override default column map
	default_column_map = rob_source_model['column_map']
	for key in default_column_map:
		if not key in column_map:
			column_map[key] = default_column_map[key]
	if not column_map.has_key("min_mag") or column_map["min_mag"] < min_mag:
		column_map["min_mag"] = min_mag

	sources = []
	for source_rec in source_records:
		source_id = source_rec.get(column_map["id"])
		max_mag = source_rec.get(column_map["max_mag"], column_map["max_mag"])
		if max_mag > min_mag:
			if source_rec.get(column_map['b_val'], column_map['b_val']) > 0.0:
				sources.append(create_rob_source(source_rec, column_map, mfd_bin_width=mfd_bin_width, **kwargs))
			else:
				print("Discarding source %s: zero or negative b value" % source_id)
		else:
			print("Discarding source %s: Mmax (%s) <= Mmin (%s)" % (source_id, max_mag, min_mag))
	#source_rec_table.Close()
	source_model = SourceModel(source_model_name, sources)
	return source_model


def create_rob_source(source_rec, column_map, **kwargs):
	"""
	Create source from ROB GIS-data (MapInfo table record).
	This is a wrapper function for create_rob_area_source and
	create_rob_simple_fault_source

	:param source_rec:
		MIPython Record object for a ROB source.
	:param column_map:
		Dictionary for mapping variables (keys) to source_rec columns (values).
	:param kwargs:
		Keyword arguments supported by create_rob_area_source or
		create_rob_simple_fault_source

	:return:
		AreaSource or SimpleFaultSource object.
	"""
	## Decide which function to use based on MapInfo object type
	#function = {MI.OBJ_TYPE_REGION: create_rob_area_source, MI.OBJ_TYPE_PLINE: create_rob_simple_fault_source}[source_rec.obj_type]
	function = {"POLYGON": create_rob_area_source, "LINESTRING": create_rob_simple_fault_source}[source_rec["obj"].GetGeometryName()]
	return function(source_rec, column_map=column_map, **kwargs)


def create_rob_area_source(
	source_rec,
	mfd=None,
	rupture_mesh_spacing=1.,
	magnitude_scaling_relationship=WC1994(),
	rupture_aspect_ratio=1.,
	strike_delta=45.,
	nodal_plane_distribution=None,
	hypo_num_bins=5,
	hypocentral_distribution=None,
	polygon=None,
	area_discretization=1.,
	mfd_bin_width = 0.1,
	column_map={},
	**kwargs
	):
	"""
	Create area source from ROB GIS-data (MapInfo table record).

	:param source_rec:
		MIPython Record object, for a ROB area source.
	:param mfd:
		See class AreaSource (Default: None).
	:param rupture_mesh_spacing:
		See class AreaSource (Default: 1.).
	:param magnitude_scaling_relationship:
		See class AreaSource (Default: WC1994).
	:param rupture_aspect_ratio:
		See class AreaSource (Default: 1.0).
	:param strike_delta:
		Float with strike delta for nodal plane distribution (Default: 45.).
	:param nodal_plane_distribution:
		See class AreaSource (Default: None).
	:param hypo_num_bins:
		Integer with number of bins for hypocentral depth distribution (Default: 5)
	:param hypocenter_distribution:
		See class AreaSource (Default: None).
	:param polygon:
		See class AreaSource (Default: None).
	:param area_discretization:
		See class AreaSource (Default: 1.0).
	:param mfd_bin_width:
		Float, bin width of MFD (default: 0.1)
	:param column_map:
		Dict, mapping OQ/nhlib source parameters (keys) to MapInfo column names
		(values) (Default: {}).
	:param kwargs:
		Possible extra parameters that will be ignored

	:return:
		AreaSource object.
	"""
	## set id
	source_id = source_rec.get(column_map['id'], column_map['id'])
	## set name
	name = source_rec.get(column_map['name'], column_map['name'])
	name = name.decode('latin1')
	## set tectonic region type
	tectonic_region_type = source_rec.get(column_map['tectonic_region_type'], column_map['tectonic_region_type'])
	## set mfd
	if not mfd:
		a_val = source_rec.get(column_map['a_val'], column_map['a_val'])
		b_val = source_rec.get(column_map['b_val'], column_map['b_val'])
		min_mag = source_rec.get(column_map['min_mag'], column_map['min_mag'])
		max_mag = source_rec.get(column_map['max_mag'], column_map['max_mag'])
		mfd = TruncatedGRMFD(min_mag, max_mag, mfd_bin_width, a_val, b_val)
	## set upper seismogenic depth
	upper_seismogenic_depth = source_rec.get(column_map['upper_seismogenic_depth'], column_map['upper_seismogenic_depth'])
	## set lower seismogenic depth
	lower_seismogenic_depth = source_rec.get(column_map['lower_seismogenic_depth'], column_map['lower_seismogenic_depth'])
	## set nodal plane distribution
	if not nodal_plane_distribution:
		# TODO: This may fail when e.g., min_strike=355 and max_strike=5
		min_strike = source_rec.get(column_map['min_strike'], column_map['min_strike'])
		max_strike = source_rec.get(column_map['max_strike'], column_map['max_strike'])
		if min_strike == 0. and max_strike == 360.:
			max_strike -= strike_delta
		## If min_strike and max_strike are 180 degrees apart, assume they correspond
		## to real fault directions, and allow only these values
		if max_strike - min_strike == 180.:
			strike_delta = 180.
		strikes, weights = get_uniform_distribution(min_strike, max_strike, strike_delta)
		dip = source_rec.get(column_map['dip'], column_map['dip'])
		rake = source_rec.get(column_map['rake'], column_map['rake'])
		nodal_planes = [NodalPlane(strike, dip, rake) for strike in strikes]
		nodal_plane_distribution = NodalPlaneDistribution(nodal_planes, weights)
	## set hypocenter distribution
	if not hypocentral_distribution:
		min_hypo_depth = source_rec.get(column_map['min_hypo_depth'], column_map['min_hypo_depth'])
		max_hypo_depth = source_rec.get(column_map['max_hypo_depth'], column_map['max_hypo_depth'])
		hypo_depths, weights = get_normal_distribution(min_hypo_depth, max_hypo_depth, num_bins=hypo_num_bins)
		hypocentral_distribution = HypocentralDepthDistribution(hypo_depths, weights)
	## set polygon
	if not polygon:
		"""
		miApp = source_rec._MIapp
		coordsys = miApp.GetCurrentCoordsys()
		miApp.SetCoordsys(MI.Coordsys(1,0))
		points = []
		## If polygon contains more than one polyline, take the outer one
		mi_polygon = max(source_rec.GetGeography())
		## Reverse polygon if clockwise
		if source_rec.IsClockwise():
			mi_polygon.reverse()
		for node in mi_polygon:
			points.append(Point(node.x, node.y))
		polygon = Polygon(points)
		miApp.SetCoordsys(coordsys)
		"""
		points = []
		zone_poly = source_rec['obj']
		## Assume outer outline corresponds to first linear ring
		linear_ring = zone_poly.GetGeometryRef(0)
		points = linear_ring.GetPoints()
		polygon = Polygon([Point(*pt) for pt in points])

	## initiate AreaSource object
	area_source = AreaSource(
		source_id,
		name,
		tectonic_region_type,
		mfd,
		rupture_mesh_spacing,
		magnitude_scaling_relationship,
		rupture_aspect_ratio,
		upper_seismogenic_depth,
		lower_seismogenic_depth,
		nodal_plane_distribution,
		hypocentral_distribution,
		polygon,
		area_discretization)
	## return AreaSource object
	return area_source


def create_rob_simple_fault_source(
	source_rec,
	mfd=None,
	rupture_mesh_spacing=1.,
	magnitude_scaling_relationship=WC1994(),
	rupture_aspect_ratio=1.,
	fault_trace=None,
	mfd_bin_width = 0.1,
	column_map={},
	**kwargs
	):
	"""
	Create simple fault source from ROB GIS-data (MapInfo table record).

	:param source_rec:
		MIPython Record object, for a ROB simple fault source.
	:param mfd:
		See class SimpleFaultSource (Default: None).
	:param rupture_mesh_spacing:
		See class SimpleFaultSource (Default: 0.1).
	:param magnitude_scaling_relationship:
		See class SimpleFaultSource (Default: WC1994).
	:param rupture_aspect_ratio:
		See class SimpleFaultSource (Default: 1.0).
	:param fault_trace:
		See class SimpleFaultSource (Default: None).
	:param mfd_bin_width:
		Float, bin width of MFD (default: 0.1)
	:param column_map:
		Dict, for mapping variables (keys) to source_rec columns (values) (Default: {}).
	:param kwargs:
		Possible extra parameters that will be ignored

	:return:
		SimpleFaultSource object.
	"""
	## ID
	source_id = source_rec.get(column_map['id'], column_map['id'])
	## Name
	name = source_rec.get(column_map['name'], column_map['name'])
	name = name.decode('latin1')
	## Tectonic region type
	tectonic_region_type = source_rec.get(column_map['tectonic_region_type'], column_map['tectonic_region_type'])

	## MFD
	if not mfd:
		# TODO: it could be dangerous to construct MFD from default values in column map
		a_val = source_rec.get(column_map['a_val'], column_map['a_val'])
		b_val = source_rec.get(column_map['b_val'], column_map['b_val'])
		min_mag = source_rec.get(column_map['min_mag'], column_map['min_mag'])
		max_mag = source_rec.get(column_map['max_mag'], column_map['max_mag'])
		max_mag_rounded = np.ceil(max_mag / mfd_bin_width) * mfd_bin_width
		## Make sure maximum magnitude is smaller than MFD.max_mag
		if np.allclose(max_mag, max_mag_rounded):
			max_mag_rounded += mfd_bin_width
		mfd = TruncatedGRMFD(min_mag, max_mag_rounded, mfd_bin_width, a_val, b_val)

	## Upper seismogenic depth
	upper_seismogenic_depth = source_rec.get(column_map['upper_seismogenic_depth'], column_map['upper_seismogenic_depth'])
	## Lower seismogenic depth
	lower_seismogenic_depth = source_rec.get(column_map['lower_seismogenic_depth'], column_map['lower_seismogenic_depth'])

	## Fault trace
	if not fault_trace:
		"""
		miApp = source_rec._MIapp
		coordsys = miApp.GetCurrentCoordsys()
		miApp.SetCoordsys(MI.Coordsys(1,0))
		points = []
		for node in source_rec.GetGeography()[0]:
			points.append(Point(node.x, node.y))
		fault_trace = Line(points)
		miApp.SetCoordsys(coordsys)
		"""

		points = []
		zone_poly = source_rec['obj']
		linear_ring = zone_poly.GetGeometryRef(0)
		points = linear_ring.GetPoints()
		fault_trace = Line([Point(*pt) for pt in points])

	## Dip
	if isinstance(column_map['dip'], (int, float)):
		dip = column_map['dip']
	else:
		dip = source_rec.get(column_map['dip'])
		if dip is None:
			max_dip = source_rec.get(column_map['dip']+"Max")
			min_dip = source_rec.get(column_map['dip']+"Min")
			if None in (min_dip, max_dip):
				raise Exception("Dip not defined")
			else:
				dip = (min_dip + max_dip) / 2.

	## Rake
	if isinstance(column_map['rake'], (int, float)):
		rake = column_map['rake']
	else:
		rake = source_rec.get(column_map['rake'])
		if rake is None:
			max_rake = source_rec.get(column_map['rake']+"Max")
			min_rake = source_rec.get(column_map['rake']+"Min")
			if None in (min_rake, max_rake):
				raise Exception("Rake not defined")
			else:
				rake = mean_angle([min_rake, max_rake])
				## Constrain to (-180, 180)
				if rake > 180:
					rake -= 360.

	## Slip rate
	if isinstance(column_map['slip_rate'], (int, float)):
		slip_rate = column_map['slip_rate']
	else:
		slip_rate = source_rec.get(column_map['slip_rate'])
		if slip_rate is None:
			max_slip_rate = source_rec.get(column_map['slip_rate']+"Max")
			min_slip_rate = source_rec.get(column_map['slip_rate']+"Min")
			if None in (min_slip_rate, max_slip_rate):
				print("Warning: Slip rate not defined")
			else:
				slip_rate = (min_slip_rate + max_slip_rate) / 2.

	## Initiate SimpleFaultSource object
	simple_fault_source = SimpleFaultSource(
		source_id,
		name,
		tectonic_region_type,
		mfd,
		rupture_mesh_spacing,
		magnitude_scaling_relationship,
		rupture_aspect_ratio,
		upper_seismogenic_depth,
		lower_seismogenic_depth,
		fault_trace,
		dip,
		rake,
		slip_rate)

	## return SimpleFaultSource object
	return simple_fault_source




if __name__ == '__main__':
	"""
	"""
	## write rob source models
	source_model = create_rob_source_model('Seismotectonic')
	source_model.print_xml()
#	source_model.write_xml('D:\Temp\Seismotectonic.xml')
#	source_model = create_rob_source_model('TwoZone')
#	source_model.write_xml('D:\Temp\TwoZone_split.xml')
#	source_model = create_rob_source_model('TwoZone')
#	source_model.write_xml('D:\Temp\TwoZone_split.xml')
#	source_model = create_rob_source_model('RVRS_CSS')
#	source_model.write_xml('D:\Temp\RVRS_CSS.xml')

	# show distributions
#	x,y = get_normal_distribution(5., 20., 15., sigma_range=1)
#	print x,y
#	pyplot.plot(x,y,color='r')

#	x,y = get_normal_distribution(5., 20., 15., sigma_range=4)
#	print x,y
#	pyplot.plot(x,y,color='b')

#	pyplot.show()
