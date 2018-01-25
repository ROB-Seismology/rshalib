"""
Module to create nrml objects and files for ROB source models
"""


### imports
import numpy as np
import os
import decimal

from openquake.hazardlib.scalerel import WC1994

from ..mfd import TruncatedGRMFD, EvenlyDiscretizedMFD
from ..geo import Point, Line, Polygon, NodalPlane, mean_angle
from ..pmf.distributions import *
from ..source import AreaSource, SimpleFaultSource, SourceModel


# set precision for weights calculation
#decimal.getcontext().prec = 4


from ..source import import_source_model_from_gis


def read_rob_source_model(
	source_model_name,
	column_map={},
	source_ids=[],
	source_catalogs={},
	overall_catalog=None,
	catalog_params={},
	encoding="latin-1",
	verbose=True,
	**kwargs):
	"""
	Read well-known ROB source model

	:param source_model_name:
		str, name of ROB source model
	:param column_map:
		dict, mapping source parameter names to GIS columns or scalars
		(default: {})
	:param source_ids:
		list of source IDs to import
		(default: [], import all sources in GIS file)
	:param source_catalogs:
		dict, mapping source ID's to instances of :class:`eqcatalog.EQCatalog`
		(default: {})
	:param overall_catalog:
		instance of :class:`eqcatalog.EQCatalog`
		source catalogs will be extracted from overall catalog if they
		are not specified in :param:`source_catalogs`
		(default: None)
	:param catalog_params:
		dict, defining catalog processing parameters, in particular the keys
		'Mtype', 'Mrelation', and 'completeness'
		(default: {})
	:param encoding:
		str, unicode encoding in GIS file
		(default: "latin-1")
	:param verbose:
		bool, whether or not to print information during reading
		(default: True)
	:param kwargs:
		dict, mapping parameter names to values, overriding entries
		in :param:`column_map` or in GIS file or in default kwargs for
		particular source

	:return:
		instance of :class:`SourceModel`
	"""
	from eqcatalog.source_models import rob_source_models_dict

	rob_source_model = rob_source_models_dict[source_model_name]

	gis_filespec = rob_source_model['gis_filespec']

	## Override default column map
	## Copy dict to avoid side effects in calling function
	column_map = column_map.copy()
	default_column_map = rob_source_model['column_map']
	for key in default_column_map:
		if not key in column_map:
			column_map[key] = default_column_map[key]

	return import_source_model_from_gis(gis_filespec, name=source_model_name,
				column_map=column_map, source_ids=source_ids, source_catalogs=source_catalogs,
				overall_catalog=overall_catalog, catalog_params=catalog_params,
				encoding=encoding, verbose=verbose, **kwargs)


def create_rob_source_model(source_model_name, min_mag=4.0, mfd_bin_width=0.1, column_map={}, source_catalogs={}, catalog_params={}, fix_mi_lambert=True, verbose=True, **kwargs):
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
	:param source_catalogs:
		Dict, mapping source IDs to earthquake catalogs.
		If specified, catalog will be used to determine max_mag and MFD
		of each source (default: {})
	:param catalog_params:
		Dict, defining catalog parameters, in particular the keys
		'Mtype', 'Mrelation', and 'completeness' (default: {})
	:param fix_mi_lambert:
		bool, whether or not to apply spatial reference system fix for
		old MapInfo files in Lambert 1972 system
		(default: True)
	:param verbose:
		Boolean, whether or not to print information (default: True)
	:param kwargs:
		Keyword arguments supported by create_rob_area_source or
		create_rob_simple_fault_source

	:return:
		SourceModel object.
	"""
	print("Warning: this function is deprecated. Use read_rob_source_model instead!")
	from eqcatalog.source_models import rob_source_models_dict
	from mapping.geotools.readGIS import read_GIS_file

	rob_source_model = rob_source_models_dict[source_model_name]
	source_records = read_GIS_file(rob_source_model['gis_filespec'], encoding=None, fix_mi_lambert=fix_mi_lambert, verbose=verbose)

	## Override default column map
	## Copy dict to avoid side effects in calling function
	column_map = column_map.copy()
	default_column_map = rob_source_model['column_map']
	for key in default_column_map:
		if not key in column_map:
			column_map[key] = default_column_map[key]
	column_map["min_mag"] = min_mag

	sources = []
	for source_rec in source_records:
		source_id = source_rec.get(column_map["id"])
		source_catalog = source_catalogs.get(source_id)
		#max_mag = source_rec.get(column_map["max_mag"], column_map["max_mag"])
		#if max_mag > min_mag:
		#	if source_rec.get(column_map['b_val'], column_map['b_val']) > 0.0:
		#		sources.append(create_rob_source(source_rec, column_map, mfd_bin_width=mfd_bin_width, **kwargs))
		#	else:
		#		print("Discarding source %s: zero or negative b value" % source_id)
		#else:
		#	print("Discarding source %s: Mmax (%s) <= Mmin (%s)" % (source_id, max_mag, min_mag))
		sources.append(create_rob_source(source_rec, column_map, mfd_bin_width=mfd_bin_width, catalog=source_catalog, catalog_params=catalog_params, verbose=verbose, **kwargs))
	source_model = SourceModel(source_model_name, sources)
	return source_model


def create_rob_source(source_rec, column_map, catalog=None, catalog_params={}, verbose=True, **kwargs):
	"""
	Create source from ROB GIS-data (MapInfo table record).
	This is a wrapper function for create_rob_area_source and
	create_rob_simple_fault_source

	:param source_rec:
		MIPython Record object for a ROB source.
	:param column_map:
		Dictionary for mapping variables (keys) to source_rec columns (values).
	:param catalog:
		instance of :class:`EQCatalog` that will be used to determine
		max_mag and MFD (default: None)
	:param catalog_params:
		Dict, defining catalog parameters, in particular the keys
		'Mtype', 'Mrelation', and 'completeness' (default: {})
	:param verbose:
		Boolean, whether or not to print information (default: True)
	:param kwargs:
		Keyword arguments supported by create_rob_area_source or
		create_rob_simple_fault_source

	:return:
		AreaSource or SimpleFaultSource object.
	"""
	print("Warning: this function is deprecated!")

	## Decide which function to use based on object type
	function = {"POLYGON": create_rob_area_source, "LINESTRING": create_rob_simple_fault_source}[source_rec["obj"].GetGeometryName()]
	return function(source_rec, column_map=column_map, catalog=catalog, catalog_params=catalog_params, verbose=verbose, **kwargs)


def create_rob_area_source(
	source_rec,
	mfd=None,
	rupture_mesh_spacing=1.,
	magnitude_scaling_relationship=WC1994(),
	rupture_aspect_ratio=1.,
	strike_delta=90.,
	dip_delta=30.,
	nodal_plane_distribution=None,
	hypo_bin_width=5,
	hypocentral_distribution=None,
	polygon=None,
	area_discretization=5.,
	mfd_bin_width=0.1,
	column_map={},
	catalog=None,
	catalog_params={},
	verbose=True,
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
		Float with strike delta for nodal plane distribution (default: 90.).
	:param dip_delta:
		Float with dip delta for nodal plane distribution (default: 30.)
	:param nodal_plane_distribution:
		See class AreaSource (Default: None).
	:param hypo_bin_width:
		Float, bin width (in km) for hypocentral depth distribution (Default: 5)
	:param hypocentral_distribution:
		See class AreaSource (Default: None).
	:param polygon:
		See class AreaSource (Default: None).
	:param area_discretization:
		See class AreaSource (Default: 5.0).
	:param mfd_bin_width:
		Float, bin width of MFD (default: 0.1)
	:param column_map:
		Dict, mapping OQ/nhlib source parameters (keys) to MapInfo column names
		(values) (Default: {}).
	:param catalog:
		instance of :class:`EQCatalog` that will be used to determine
		max_mag and MFD (default: None)
	:param catalog_params:
		Dict, defining catalog parameters, in particular the keys
		'Mtype', 'Mrelation', and 'completeness' (default: {})
	:param verbose:
		Boolean, whether or not to print information (default: True)
	:param kwargs:
		Possible extra parameters that will be ignored
	:return:
		AreaSource object.
	"""
	print("Warning: this function is deprecated!")

	import osr
	from mapping.geotools.coordtrans import wgs84, lambert1972
	coordTrans = osr.CoordinateTransformation(wgs84, lambert1972)

	## ID and name
	source_id = str(source_rec.get(column_map['id'], column_map['id']))
	name = str(source_rec.get(column_map['name'], column_map['name']))
	name = name.decode('latin1')
	if verbose:
		print source_id

	## Tectonic region type
	tectonic_region_type = source_rec.get(column_map['tectonic_region_type'], column_map['tectonic_region_type'])

	## upper and lower seismogenic depth
	upper_seismogenic_depth = float(source_rec.get(column_map['upper_seismogenic_depth'], column_map['upper_seismogenic_depth']))
	lower_seismogenic_depth = float(source_rec.get(column_map['lower_seismogenic_depth'], column_map['lower_seismogenic_depth']))

	## nodal plane distribution
	if not nodal_plane_distribution:
		## Strike
		# TODO: This may fail when e.g., min_strike=355 and max_strike=5
		min_strike = float(source_rec.get(column_map['min_strike'], column_map['min_strike']))
		max_strike = float(source_rec.get(column_map['max_strike'], column_map['max_strike']))
		if min_strike > max_strike:
			min_strike, max_strike = max_strike, min_strike
		if min_strike == 0. and max_strike == 360.:
			max_strike -= strike_delta
		## If min_strike and max_strike are 180 degrees apart, assume they correspond
		## to real fault directions, and allow only these values
		if max_strike - min_strike == 180.:
			strike_delta = 180.
		strikes, strike_weights = get_uniform_distribution(min_strike, max_strike, strike_delta)

		## Dip
		min_dip = float(source_rec.get(column_map['min_dip'], column_map['min_dip']))
		if min_dip == 0:
			min_dip = 1E-6
		max_dip = float(source_rec.get(column_map['max_dip'], column_map['max_dip']))
		if max_dip == 0:
			max_dip = 1E-6
		if min_dip > max_dip:
			min_dip, max_dip = max_dip, min_dip
		dips, dip_weights = get_uniform_distribution(min_dip, max_dip, dip_delta)

		## Rake
		Ss = float(source_rec.get(column_map['Ss'], column_map['Ss']))
		Nf = float(source_rec.get(column_map['Nf'], column_map['Nf']))
		Tf = float(source_rec.get(column_map['Tf'], column_map['Tf']))
		rake_weights = np.array([Ss, Nf, Tf], 'i') / Decimal(100.)
		rakes = np.array([0, -90, 90])[rake_weights > 0]
		rake_weights = rake_weights[rake_weights > 0]

		## NPD
		nodal_planes, nodal_plane_weights = [], []
		for strike, strike_weight in zip(strikes, strike_weights):
			for dip, dip_weight in zip(dips, dip_weights):
				for rake, rake_weight in zip(rakes, rake_weights):
					nodal_planes.append(NodalPlane(strike, dip, rake))
					nodal_plane_weights.append(strike_weight * rake_weight * dip_weight)
		nodal_plane_distribution = NodalPlaneDistribution(nodal_planes, nodal_plane_weights)

	## hypocenter distribution
	if not hypocentral_distribution:
		min_hypo_depth = float(source_rec.get(column_map['min_hypo_depth'], column_map['min_hypo_depth']))
		min_hypo_depth = max(min_hypo_depth, upper_seismogenic_depth)
		max_hypo_depth = float(source_rec.get(column_map['max_hypo_depth'], column_map['max_hypo_depth']))
		max_hypo_depth = min(max_hypo_depth, lower_seismogenic_depth)
		num_bins = (max_hypo_depth - min_hypo_depth) / hypo_bin_width + 1
		hypo_depths, weights = get_normal_distribution(min_hypo_depth, max_hypo_depth, num_bins=num_bins)
		hypocentral_distribution = HypocentralDepthDistribution(hypo_depths, weights)

	## polygon
	if not polygon:
		points = []
		zone_poly = source_rec['obj']
		## Assume outer outline corresponds to first linear ring
		linear_ring = zone_poly.GetGeometryRef(0)
		## In some versions of ogr, GetPoints method does not exist
		#points = linear_ring.GetPoints()
		points = [linear_ring.GetPoint(i) for i in range(linear_ring.GetPointCount())]
		polygon = Polygon([Point(*pt) for pt in points])

		## Calculate area
		#zone_poly.Transform(coordTrans)
		#area = zone_poly.GetArea() / 1E6

	## instantiate AreaSource object
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

	## MFD
	if not mfd:
		## Determine MFD from catalog if one is specified
		if catalog is not None:
			min_mag = column_map['min_mag']

			## Lower magnitude to compute MFD
			if catalog_params.has_key('completeness'):
				min_mag_mfd = catalog_params['completeness'].min_mag
			else:
				min_mag_mfd = min_mag

			## Use b value specified in column map (but not in GIS table)
			if isinstance(column_map['b_val'], (int, float)):
				b_val = column_map['b_val']
			else:
				b_val = None

			## Use max_mag specified in column map or determine from catalog using EPRI method
			max_mag = None
			if isinstance(column_map['max_mag'], (int, float)):
				max_mag = column_map['max_mag']
			if max_mag is None:
				max_mag_pmf = catalog.get_Bayesian_Mmax_pdf(Mmin_n=min_mag, dM=mfd_bin_width, verbose=False, **catalog_params)
				max_mag = max_mag_pmf.get_percentile(50)
			max_mag = np.ceil(max_mag / mfd_bin_width) * mfd_bin_width
			if max_mag <= min_mag:
				raise Exception("Mmax of source %s not larger than Mmin!" % source_id)

			try:
				## Weichert computation
				## Note: using lowest magnitude in completeness object
				## is more robust than using min_mag
				mfd = catalog.get_estimated_MFD(min_mag_mfd, max_mag, mfd_bin_width, method="Weichert", b_val=b_val, verbose=False, **catalog_params)
			except ValueError as err:
				print("Warning: Weichert MFD computation: %s" % err.args[0])
				try:
					## Fall back to minimum MFD for SCR by Johnston et al. (1994)
					mfd = area_source.get_MFD_Johnston1994(min_mag, max_mag, mfd_bin_width)
				except ValueError as err:
					mfd = None
			else:
				mfd.min_mag = min_mag

		else:
			## Read parameters from GIS table
			a_val = source_rec.get(column_map['a_val'], column_map['a_val'])
			b_val = source_rec.get(column_map['b_val'], column_map['b_val'])
			a_sigma = source_rec.get(column_map['a_sigma'], column_map['a_sigma'])
			b_sigma = source_rec.get(column_map['b_sigma'], column_map['b_sigma'])
			min_mag = source_rec.get(column_map['min_mag'], column_map['min_mag'])
			max_mag = source_rec.get(column_map['max_mag'], column_map['max_mag'])
			if max_mag <= min_mag:
				raise Exception("Mmax of source %s not larger than Mmin!" % source_id)
			try:
				mfd = TruncatedGRMFD(min_mag, max_mag, mfd_bin_width, a_val, b_val, a_sigma, b_sigma)
			except ValueError:
				try:
					## Fall back to minimum MFD for SCR by Johnston et al. (1994)
					mfd = area_source.get_MFD_Johnston1994(min_mag, max_mag, mfd_bin_width)
				except ValueError:
					mfd = None

		area_source.mfd = mfd

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
	catalog=None,
	catalog_params={},
	verbose=True,
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
	:param catalog:
		instance of :class:`EQCatalog` that will be used to determine
		max_mag and MFD (default: None)
		Currently ignored
	:param catalog_params:
		Dict, defining catalog parameters, in particular the keys
		'Mtype', 'Mrelation', and 'completeness' (default: {})
	:param verbose:
		Boolean, whether or not to print information (default: True)
	:param kwargs:
		Possible extra parameters that will be ignored

	:return:
		SimpleFaultSource object.
	"""
	print("Warning: this function is deprecated!")

	## ID and name
	source_id = str(source_rec.get(column_map['id'], column_map['id']))
	name = str(source_rec.get(column_map['name'], column_map['name']))
	name = name.decode('latin1')
	if verbose:
		print source_id

	## Tectonic region type
	tectonic_region_type = source_rec.get(column_map['tectonic_region_type'], column_map['tectonic_region_type'])

	## MFD
	if not mfd:
		# TODO: it could be dangerous to construct MFD from default values in column map
		a_val = float(source_rec.get(column_map['a_val'], column_map['a_val']))
		b_val = float(source_rec.get(column_map['b_val'], column_map['b_val']))
		min_mag = float(source_rec.get(column_map['min_mag'], column_map['min_mag']))
		max_mag = float(source_rec.get(column_map['max_mag'], column_map['max_mag']))
		max_mag_rounded = np.ceil(max_mag / mfd_bin_width) * mfd_bin_width
		## Make sure maximum magnitude is smaller than MFD.max_mag
		if np.allclose(max_mag, max_mag_rounded):
			max_mag_rounded += mfd_bin_width
		try:
			mfd = TruncatedGRMFD(min_mag, max_mag_rounded, mfd_bin_width, a_val, b_val)
		except ValueError:
			mfd = None

	## Upper seismogenic depth
	upper_seismogenic_depth = float(source_rec.get(column_map['upper_seismogenic_depth'], column_map['upper_seismogenic_depth']))
	## Lower seismogenic depth
	lower_seismogenic_depth = float(source_rec.get(column_map['lower_seismogenic_depth'], column_map['lower_seismogenic_depth']))

	## Fault trace
	if not fault_trace:
		points = []
		linear_ring = source_rec['obj']
		## In some versions of ogr, GetPoints method does not exist
		#points = linear_ring.GetPoints()
		points = [linear_ring.GetPoint(i) for i in range(linear_ring.GetPointCount())]
		fault_trace = Line([Point(*pt) for pt in points])

	## Dip
	max_dip = float(source_rec.get(column_map['min_dip'], column_map['min_dip']))
	min_dip = float(source_rec.get(column_map['max_dip'], column_map['max_dip']))
	if None in (min_dip, max_dip):
		raise Exception("Dip not defined")
	else:
		dip = (min_dip + max_dip) / 2.

	## Rake
	max_rake = float(source_rec.get(column_map['min_rake'], column_map['min_rake']))
	min_rake = float(source_rec.get(column_map['max_rake'], column_map['max_rake']))
	if None in (min_rake, max_rake):
		raise Exception("Rake not defined")
	else:
		rake = mean_angle([min_rake, max_rake])
		## Constrain to (-180, 180)
		if rake > 180:
			rake -= 360.

	## Slip rate
	max_slip_rate = float(source_rec.get(column_map['min_slip_rate'], column_map['min_slip_rate']))
	min_slip_rate = float(source_rec.get(column_map['max_slip_rate'], column_map['max_slip_rate']))
	if None in (min_slip_rate, max_slip_rate):
		print("Warning: Slip rate not defined")
	else:
		slip_rate = (min_slip_rate + max_slip_rate) / 2.

	## Background zone
	bg_zone = source_rec.get(column_map['bg_zone'], column_map['bg_zone'])

	## Instantiate SimpleFaultSource object
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
		slip_rate,
		bg_zone)

	return simple_fault_source




if __name__ == '__main__':
	"""
	"""
	from matplotlib import pyplot

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
