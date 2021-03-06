"""
Read source models from GIS files
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
from decimal import Decimal
import sys

if sys.version_info.major == 2:
	PY2 = True
else:
	PY2 = False
	basestring = str

import numpy as np

from .. import oqhazlib

from ..mfd import TruncatedGRMFD, EvenlyDiscretizedMFD
from ..geo import Point, Line, Polygon, NodalPlane, mean_angle
from ..pmf import (NumericPMF, NodalPlaneDistribution,
					HypocentralDepthDistribution, adjust_decimal_weights,
					get_normal_distribution, get_uniform_distribution)

from mapping.geotools.read_gis import read_gis_file


__all__ = ['import_source_model_from_gis',
			'import_source_from_gis_record',
			'import_point_or_area_source_from_gis_record',
			'import_simple_fault_source_from_gis_record']


COMMON_SOURCE_PARAMS = [
	'id',
	'name',
	'tectonic_region_type',
	'rupture_mesh_spacing',
	'rupture_aspect_ratio',
	'magnitude_scaling_relationship',
	'upper_seismogenic_depth',
	'lower_seismogenic_depth',
	'a_val',
	'b_val',
	'mfd_bin_width',
	'min_mag',
	'max_mag',
	'max_mag_distribution']

POINT_SOURCE_PARAMS = COMMON_SOURCE_PARAMS + [
	'min_dip',
	'max_dip',
	'dip_delta',
	'min_strike',
	'max_strike',
	'strike_delta',
	'strike_dip_distribution',
	'Nf',
	'Tf',
	'Ss',
	'hypo_distribution',
	'min_hypo_depth',
	'max_hypo_depth',
	'hypo_bin_width']

AREA_SOURCE_PARAMS = POINT_SOURCE_PARAMS + [
	'area_discretization']

SIMPLE_FAULT_SOURCE_PARAMS = COMMON_SOURCE_PARAMS + [
	'dip',
	'dip_distribution',
	'rake',
	'rake_distribution',
	'slip_rate',
	'slip_rate_distribution',
	'bg_zone']


DEFAULT_POINT_SOURCE_COLUMN_MAP = dict((p,p) for p in POINT_SOURCE_PARAMS)
DEFAULT_AREA_SOURCE_COLUMN_MAP = dict((p,p) for p in AREA_SOURCE_PARAMS)
DEFAULT_SIMPLE_FAULT_SOURCE_COLUMN_MAP = dict((p,p) for p in SIMPLE_FAULT_SOURCE_PARAMS)


def import_param(
	source_rec,
	column_map,
	param_name,
	type=None,
	encoding='latin1'):
	"""
	Read a particular parameter from a GIS record, optionally coerced
	into a particular type

	:param source_rec:
		GIS record as returned by :func:`mapping.geotools.read_gis.read_gis_file`
	:param column_map:
		dict, mapping source parameter names to GIS columns or scalars
	:param param_name:
		str, name of parameter to read (must be key in :param:`column_map`)
	:param type:
		python type to coerce returned value into (may be useful if
		numeric values are stored as strings in GIS record)
		(default: None, will return values as type stored in GIS record)
	:param encoding:
		str, encoding to use for decoding unicode strings
		(default: 'latin1')

	:return:
		value read from GIS record or scalar value provided in
		:param:`column_map` or else None if :param:`param_name` is not
		in :param:`column_map` or in GIS record
	"""
	val = None
	if param_name in column_map:
		val = source_rec.get(column_map[param_name], column_map[param_name])
	if val is not None and type:
		if PY2 and isinstance(val, unicode) and type == str:
			val = val.decode(encoding)
		else:
			val = type(val)
	return val


def import_distribution(
	source_rec,
	column_map,
	param_name):
	"""
	Read a distribution for a particular parameter from a GIS record

	:param source_rec:
		GIS record as returned by :func:`mapping.geotools.read_gis.read_gis_file`
	:param column_map:
		dict, mapping source parameter names to GIS columns or scalars
	:param param_name:
		str, name of parameter to read (must be key in :param:`column_map`)

	:return:
		instance of :class:`pmf.NumericPMF`
	"""
	dist = None
	values, weights = [], []
	for value_col, weight_col in column_map[param_name]:
		value = float(source_rec.get(value_col, value_col))
		weight = float(source_rec.get(weight_col, weight_col))
		if weight != 0:
			values.append(value)
			weights.append(weight)
	if values:
		dist = NumericPMF.from_values_and_weights(values, weights)
	return dist


def read_gr_mfd_params(
	source_rec,
	column_map,
	min_mag=None,
	max_mag=None,
	mfd_bin_width=None,
	a_val=None,
	b_val=None,
	a_sigma=None,
	b_sigma=None):
	"""
	Read truncated Gutenberg-Richter MFD from GIS record

	:param source_rec:
		GIS record as returned by :func:`mapping.geotools.read_gis.read_gis_file`
	:param column_map:
		dict, mapping source parameter names to GIS columns or scalars
	:param min_mag:
		float, minimum magnitude
		(default: None)
	:param max_mag:
		float, maximum magnitude
		(default: None)
	:param mfd_bin_width:
		float, magnitude bin width for MFD
		(default: None)
	:param a_val:
		float, a-value
		(default: None)
	:param b_val:
		float, b-value
		(default: None)
	:param a_sigma:
		float, standard deviation on a-value
		(default: None)
	:param b_sigma:
		float, standard deviation on b-value
		(default: None)

	:return:
		(min_mag, max_mag, mfd_bin_width, a_val, b_val, a_sigma, b_sigma)
	"""
	if mfd_bin_width is None:
		mfd_bin_width = import_param(source_rec, column_map, 'mfd_bin_width', float)
	if min_mag is None:
		min_mag = import_param(source_rec, column_map, 'min_mag', float)
	if max_mag is None:
		max_mag = import_param(source_rec, column_map, 'max_mag', float)
	if max_mag and mfd_bin_width:
		max_mag = np.ceil(max_mag / mfd_bin_width) * mfd_bin_width
	if a_val is None:
		a_val = import_param(source_rec, column_map, 'a_val', float)
	if b_val is None:
		b_val = import_param(source_rec, column_map, 'b_val', float)
	if a_sigma is None:
		a_sigma = import_param(source_rec, column_map, 'a_sigma', float)
	if b_sigma is None:
		b_sigma = import_param(source_rec, column_map, 'b_sigma', float)

	return (min_mag, max_mag, mfd_bin_width, a_val, b_val, a_sigma, b_sigma)


def read_gr_mfd(
	source_rec,
	column_map,
	min_mag=None,
	max_mag=None,
	mfd_bin_width=None,
	a_val=None,
	b_val=None,
	a_sigma=None,
	b_sigma=None):
	"""
	Read truncated Gutenberg-Richter MFD from GIS record

	:param source_rec:
		GIS record as returned by :func:`mapping.geotools.read_gis.read_gis_file`
	:param column_map:
		dict, mapping source parameter names to GIS columns or scalars
	:param min_mag:
		float, minimum magnitude
		(default: None)
	:param max_mag:
		float, maximum magnitude
		(default: None)
	:param mfd_bin_width:
		float, magnitude bin width for MFD
		(default: None)
	:param a_val:
		float, a-value
		(default: None)
	:param b_val:
		float, b-value
		(default: None)
	:param a_sigma:
		float, standard deviation on a-value
		(default: None)
	:param b_sigma:
		float, standard deviation on b-value
		(default: None)

	:return:
		instance of :class:`TruncatedGRMFD`
	"""
	source_id = import_param(source_rec, column_map, 'id', str)
	(min_mag, max_mag, mfd_bin_width, a_val, b_val, a_sigma, b_sigma) = read_gr_mfd_params(
		source_rec, column_map, min_mag, max_mag, mfd_bin_width, a_val, b_val, a_sigma, b_sigma)
	if max_mag <= min_mag:
		print("Warning: Mmax (%s) of source %s not higher than Mmin!"
				% (max_mag, source_id))
		max_mag = min_mag + mfd_bin_width
	mfd = TruncatedGRMFD(min_mag, max_mag, mfd_bin_width, a_val, b_val, a_sigma, b_sigma)
	return mfd


def get_gr_mfd_from_catalog(
	catalog,
	catalog_params,
	min_mag,
	max_mag,
	mfd_bin_width=0.1,
	b_val=None,
	verbose=False
	):
	"""
	Determine truncated Gutenberg-Richter MFD from catalog

	:param catalog:
		instance of :class:`eqcatalog.EQCatalog`
	:param catalog_params:
		dict, defining catalog processing parameters, in particular the keys
		'Mtype', 'Mrelation', and 'completeness'
		(default: {})
	:param min_mag:
		float, minimum magnitude
	:param max_mag:
		float, maximum magnitude
		If None, will be taken as median value of Bayesian Mmax
		distribution determined from catalog
	:param mfd_bin_width:
		float, magnitude bin width of MFD
		(default: 0.1)
	:param b_val:
		float, b-value to impose
		(default: None)
	:param verbose:
		bool, whether or not to print information (default: False)

	:return:
		instance of :class:`TruncatedGRMFD`
	"""
	if min_mag is None:
		raise Exception("min_mag must be specified if MFD is determined "
						"from catalog")

	## Lower magnitude to compute MFD
	## Note: using lowest magnitude in completeness object
	## is more robust than using min_mag
	if 'completeness' in catalog_params:
		min_mag_mfd = catalog_params['completeness'].min_mag
	else:
		min_mag_mfd = min_mag

	## b_val can be None

	## Use max_mag specified or determine from catalog using EPRI method
	if max_mag is None:
		max_mag_pmf = catalog.get_Bayesian_Mmax_pdf(Mmin_n=min_mag, dM=mfd_bin_width,
										b_val=b_val, verbose=verbose, **catalog_params)
		max_mag = max_mag_pmf.get_percentile(50)
	max_mag = np.ceil(max_mag / mfd_bin_width) * mfd_bin_width
	if max_mag <= min_mag:
		print("Warning: Mmax (%s) not higher than Mmin!" % max_mag)
		max_mag = min_mag + mfd_bin_width

	try:
		## Weichert computation
		mfd = catalog.get_estimated_mfd(min_mag_mfd, max_mag, mfd_bin_width,
				method="Weichert", b_val=b_val, verbose=False, **catalog_params)
	except ValueError as err:
		print("Warning: Weichert MFD computation: %s" % err.args[0])
		raise
	else:
		mfd.min_mag = min_mag
		return mfd


def import_point_or_area_source_from_gis_record(
	source_rec,
	column_map=DEFAULT_AREA_SOURCE_COLUMN_MAP,
	tectonic_region_type="",
	upper_seismogenic_depth=None,
	lower_seismogenic_depth=None,
	rupture_mesh_spacing=1.,
	rupture_aspect_ratio=1.,
	magnitude_scaling_relationship='WC1994',
	nodal_plane_distribution=None,
	min_strike=None,
	max_strike=None,
	strike_delta=90.,
	min_dip=None,
	max_dip=None,
	dip_delta=30.,
	Ss=None,
	Nf=None,
	Tf=None,
	hypocentral_distribution=None,
	min_hypo_depth=None,
	max_hypo_depth=None,
	hypo_bin_width=5.,
	area_discretization=5.,
	mfd=None,
	min_mag=4.0,
	max_mag=None,
	mfd_bin_width=0.1,
	a_val=None,
	b_val=None,
	a_sigma=None,
	b_sigma=None,
	catalog=None,
	catalog_params={},
	encoding="latin1",
	verbose=True,
	**kwargs):
	"""
	Create area or point source from GIS record.

	Philosophy:
		parameters specified as kwargs take precedence over those
		in column_map

	:param source_rec:
		GIS record as returned by :func:`mapping.geotools.read_gis.read_gis_file`
		representing area source.
	:param column_map:
		dict, mapping source parameter names to GIS columns or scalars
		(default: DEFAULT_AREA_SOURCE_COLUMN_MAP).
	:param tectonic_region_type:
		str, tectonic region type
		(default: "")
	:param upper_seismogenic_depth:
		float, upper seismogenic depth in km
		(default: None)
	:param lower_seismogenic_depth:
		float, lower seismogenic depth in km
		(default: None)
	:param rupture_mesh_spacing:
		float, rupture mesh spacing in km
		(default: 1.0)
	:param rupture_aspect_ratio:
		float, characteristic length/width ratio of modeled ruptures
		(default: 1.0)
	:param magnitude_scaling_relationship:
		str or instance of class:`openquake.hazardlib.scalerel.base.BaseMSR`
		magnitude-area scaling relationship
		(default: 'WC1994')
	:param nodal_plane_distribution:
		instance of :class:`NodalPlaneDistribution`
		(default: None)
	:param min_strike:
		float, minimum strike in degrees for nodal plane distribution
		(default: None)
	:param max_strike:
		float, maximum strike in degrees for nodal plane distribution
		(default: None)
	:param strike_delta:
		float, strike increment in degrees for nodal plane distribution
		(default: None)
	:param min_dip:
		float, minimum dip angle in degrees for nodal plane distribution
		(default: None)
	:param max_dip:
		float, maximum dip angle in degrees for nodal plane distribution
		(default: None)
	:param dip_delta:
		float, dip increment in degrees for nodal plane distribution
		(default: None)
	:param Ss:
		int, strike-slip percentage [0, 100] in nodal plane distribution
		(default: None)
	:param Nf:
		int, normal-faulting percentage [0, 100] in nodal plane distribution
		(default: None)
	:param Tf:
		int, thrust-faulting percentage [0, 100] in nodal plane distribution
		(default: None)
	:param hypocentral_distribution:
		instance of :class:`HypocentralDepthDistribution`
		(default: None)
	:param min_hypo_depth:
		float, minimum hypocentral depth in km for hypocentral depth distribution
		(default: None)
	:param max_hypo_depth:
		float, maximum hypocentral depth in km for hypocentral depth distribution
		(default: None)
	:param hypo_bin_width:
		float, bin width in km for hypocentral depth distribution
		(default: 5.)
	:param area_discretization:
		float, grid spacing in km for discretizing area into point sources
		(default: 5.)
	:param mfd:
		instance of :class:`oqhazlib.mfd.base.BaseMFD`, magnitude-frequency
		distribution.
		Note: takes precedence over other mfd parameters
		(default: None)
	:param min_mag:
		float, minimum magnitude
		(default: 4.0)
	:param max_mag:
		float, maximum magnitude
		(default: None)
	:param mfd_bin_width:
		float, magnitude bin width for MFD
		(default: 0.1)
	:param a_val:
		float, a-value
		(default: None)
	:param b_val:
		float, b-value
		(default: None)
	:param a_sigma:
		float, standard deviation on a-value
		(default: None)
	:param b_sigma:
		float, standard deviation on b-value
		(default: None)
	:param catalog:
		instance of :class:`EQCatalog` that will be used to determine
		MFD (default: None)
		(default: None)
	:param catalog_params:
		dict, defining catalog processing parameters, in particular the keys
		'Mtype', 'Mrelation', and 'completeness'
		(default: {})
	:param bg_zone:
		str, name of background zone
		(default: "")
	:param encoding:
		str, unicode encoding in GIS record
		(default: "latin1")
	:param verbose:
		bool, whether or not to print information during reading
		(default: True)
	:param kwargs:
		dict, possible extra parameters that will be ignored

	:return:
		instance of :class:`PointSource` or :class:`AreaSource`
	"""
	from . import PointSource, AreaSource

	## ID and name
	source_id = import_param(source_rec, column_map, 'id', str, encoding=encoding)
	name = import_param(source_rec, column_map, 'name', str, encoding=encoding)
	if verbose:
		print(source_id)

	## Tectonic region type
	if not tectonic_region_type:
		tectonic_region_type = import_param(source_rec, column_map,
								'tectonic_region_type', str, encoding=encoding)

	## Upper and lower seismogenic depth
	if upper_seismogenic_depth is None:
		upper_seismogenic_depth = import_param(source_rec, column_map,
											'upper_seismogenic_depth', float)
	if not lower_seismogenic_depth:
		lower_seismogenic_depth = import_param(source_rec, column_map,
											'lower_seismogenic_depth', float)

	## Magnitude scaling relationship
	if not magnitude_scaling_relationship:
		magnitude_scaling_relationship = import_param(source_rec, column_map,
									'magnitude_scaling_relationship', str)
	if isinstance(magnitude_scaling_relationship, basestring):
		magnitude_scaling_relationship = getattr(oqhazlib.scalerel,
											magnitude_scaling_relationship)()

	## Nodal plane distribution
	if not nodal_plane_distribution:
		## Strike and dip combined (SHARE...)
		if 'strike_dip_distribution' in column_map:
			strike_dips, strike_dip_weights = [], []
			for strike_col, dip_col, weight_col in column_map['strike_dip_distribution']:
				strike = int(source_rec.get(strike_col, 0))
				dip = int(source_rec.get(dip_col, 90))
				weight = float(source_rec.get(weight_col, weight_col))
				if weight != 0:
					strike_dips.append((strike, dip))
					strike_dip_weights.append(Decimal(weight))
			if not strike_dips or strike_dips == [(0, 0)]:
				strike_dips = [(0, 90)]
				strike_dip_weights = [Decimal(1)]

		else:
			## Strike
			if min_strike is None:
				min_strike = import_param(source_rec, column_map, 'min_strike', float)
			if max_strike is None:
				max_strike = import_param(source_rec, column_map, 'max_strike', float)
			if strike_delta is None:
				strike_delta = import_param(source_rec, column_map, 'strike_delta', float)
			# TODO: This may fail when e.g., min_strike=355 and max_strike=5
			if min_strike > max_strike:
				min_strike, max_strike = max_strike, min_strike
			if min_strike == 0. and max_strike == 360.:
				max_strike -= strike_delta
			## If min_strike and max_strike are 180 degrees apart, assume they correspond
			## to real fault directions, and allow only these values
			if max_strike - min_strike == 180.:
				strike_delta = 180.
			strikes, strike_weights = get_uniform_distribution(min_strike, max_strike,
																strike_delta)

			## Dip
			if min_dip is None:
				min_dip = import_param(source_rec, column_map, 'min_dip', float)
			if max_dip is None:
				max_dip = import_param(source_rec, column_map, 'max_dip', float)
			if dip_delta is None:
				dip_delta = import_param(source_rec, column_map, 'dip_delta', float)
			## oq-hazardlib doesn't like zero dips
			min_dip = max(min_dip, 1E-6)
			max_dip = max(max_dip, 1E-6)
			if min_dip > max_dip:
				min_dip, max_dip = max_dip, min_dip
			dips, dip_weights = get_uniform_distribution(min_dip, max_dip, dip_delta)

			strike_dips, strike_dip_weights = [], []
			for strike, strike_weight in zip(strikes, strike_weights):
				for dip, dip_weight in zip(dips, dip_weights):
					strike_dips.append((strike, dip))
					strike_dip_weights.append(strike_weight * dip_weight)

		## Rake
		if Ss is None:
			Ss = import_param(source_rec, column_map, 'Ss', float)
		if Nf is None:
			Nf = import_param(source_rec, column_map, 'Nf', float)
		if Tf is None:
			Tf = import_param(source_rec, column_map, 'Tf', float)
		rake_weights = np.array([Ss, Nf, Tf], 'i') / Decimal(100.)
		rakes = np.array([0, -90, 90])[rake_weights > 0]
		rake_weights = rake_weights[rake_weights > 0]

		## Construct NPD
		nodal_planes, np_weights = [], []
		#for strike, strike_weight in zip(strikes, strike_weights):
		#	for dip, dip_weight in zip(dips, dip_weights):
		for (strike, dip), strike_dip_weight in zip(strike_dips, strike_dip_weights):
				for rake, rake_weight in zip(rakes, rake_weights):
					nodal_planes.append(NodalPlane(strike, dip, rake))
					#np_weights.append(strike_weight * dip_weight * rake_weight)
					np_weights.append(strike_dip_weight * rake_weight)
		np_weights = adjust_decimal_weights(np_weights)

		nodal_plane_distribution = NodalPlaneDistribution(nodal_planes, np_weights)

	## Hypocenter distribution
	if not hypocentral_distribution:
		if 'hypo_distribution' in column_map:
			hypo_dist = import_distribution(source_rec, column_map, 'hypo_distribution')
			if hypo_dist:
				hypocentral_distribution = HypocentralDepthDistribution(hypo_dist.values,
																	hypo_dist.weights)

		if not hypocentral_distribution:
			if min_hypo_depth is None:
				min_hypo_depth = import_param(source_rec, column_map,
											'min_hypo_depth', float)
			min_hypo_depth = max(min_hypo_depth, upper_seismogenic_depth)
			if max_hypo_depth is None:
				max_hypo_depth = import_param(source_rec, column_map,
											'max_hypo_depth', float)
			max_hypo_depth = min(max_hypo_depth, lower_seismogenic_depth)

			if hypo_bin_width is None:
				hypo_bin_width = import_param(source_rec, column_map,
											'hypo_bin_width', float)
			num_bins = max(1, (max_hypo_depth - min_hypo_depth) / hypo_bin_width + 1)
			hypo_depths, weights = get_normal_distribution(min_hypo_depth, max_hypo_depth,
														num_bins=int(num_bins))
			hypocentral_distribution = HypocentralDepthDistribution(hypo_depths, weights)

	## Compare hypocentral distribution with upper/lower seismogenic depth
	if hypocentral_distribution.values[0] <= upper_seismogenic_depth:
		msg = "Warning: usd (%.1f) >= min. hypo depth (%.1f) in src %s"
		msg %= (upper_seismogenic_depth, hypocentral_distribution.values[0], source_id)
		print(msg)
		if upper_seismogenic_depth == 0:
			hypocentral_distribution.values[0] -= 1
		else:
			upper_seismogenic_depth = max(0, hypocentral_distribution.values[0] - 1)

	if hypocentral_distribution.values[-1] >= lower_seismogenic_depth:
		msg = "Warning: lsd (%.1f) <= max. hypo depth (%.1f) in src %s"
		msg %= (lower_seismogenic_depth, hypocentral_distribution.values[-1], source_id)
		print(msg)
		lower_seismogenic_depth = hypocentral_distribution.values[-1] + 1

	## Geometry
	geom = source_rec['obj']
	if geom.GetGeometryName() == "POINT":
		x, y = geom.GetGeometryRef(0).GetPoint(0)
		point = Point(x, y)

		## Instantiate PointSource object
		source = PointSource(source_id, name, tectonic_region_type, mfd,
			rupture_mesh_spacing, magnitude_scaling_relationship,
			rupture_aspect_ratio, upper_seismogenic_depth, lower_seismogenic_depth,
			nodal_plane_distribution, hypocentral_distribution, point)
	else:
		## Assume outer outline corresponds to first linear ring
		linear_ring = geom.GetGeometryRef(0)
		## In some versions of ogr, GetPoints method does not exist
		#points = linear_ring.GetPoints()
		points = [linear_ring.GetPoint(i) for i in range(linear_ring.GetPointCount())]
		polygon = Polygon([Point(*pt) for pt in points])

		if area_discretization is None:
			area_discretization = import_param(source_rec, column_map,
												'area_discretization', float)

		## Instantiate AreaSource object
		## Note: we create area_source first because in some cases it may be
		## required to determine the MFD
		source = AreaSource(source_id, name, tectonic_region_type, mfd,
			rupture_mesh_spacing, magnitude_scaling_relationship,
			rupture_aspect_ratio, upper_seismogenic_depth, lower_seismogenic_depth,
			nodal_plane_distribution, hypocentral_distribution, polygon,
			area_discretization)

	## Override max_mag with mean from max_mag_distribution (SHARE...)
	max_mag_dist = None
	if 'max_mag_distribution' in column_map:
		max_mag_dist = import_distribution(source_rec, column_map,
											'max_mag_distribution')
		if max_mag_dist:
			max_mag = max_mag_dist.get_mean()

	## MFD
	if not mfd:
		if catalog is not None:
			## Determine MFD from catalog if one is specified
			## We do not read from GIS record (or column_map) in this case
			try:
				mfd = get_gr_mfd_from_catalog(catalog, catalog_params, min_mag,
											max_mag, mfd_bin_width, b_val, verbose)
			except ValueError:
				pass

		else:
			## Read parameters from GIS table
			try:
				mfd = read_gr_mfd(source_rec, column_map, min_mag=min_mag,
						max_mag=max_mag, mfd_bin_width=mfd_bin_width,
						a_val=a_val, b_val=b_val, a_sigma=a_sigma, b_sigma=b_sigma)
			except ValueError:
				pass

		if not mfd:
			try:
				## Fall back to average SCR MFD by Johnston et al. (1994) for area sources
				mfd = source.get_Johnston1994_mfd(min_mag, max_mag, mfd_bin_width)
			except (AttributeError, ValueError):
				mfd = None
				print("Warning: MFD could not be determined for source %s"
						% source_id)
			else:
				print("Warning: Falling back to average SCR MFD for source %s"
						% source_id)

		## Sum weighted MFDs with different Mmax to incremental MFD
		if mfd and max_mag_dist:
			mfd_list = []
			for Mmax, weight in max_mag_dist:
				mfd2 = mfd.copy()
				mfd2.max_mag = Mmax
				mfd2 *= float(weight)
				mfd_list.append(mfd2)
			mfd = np.sum(mfd_list)

	source.mfd = mfd

	return source


import_point_source_from_gis_record = import_point_or_area_source_from_gis_record
import_area_source_from_gis_record = import_point_or_area_source_from_gis_record


def import_simple_fault_source_from_gis_record(
	source_rec,
	column_map=DEFAULT_SIMPLE_FAULT_SOURCE_COLUMN_MAP,
	tectonic_region_type="",
	upper_seismogenic_depth=None,
	lower_seismogenic_depth=None,
	rupture_mesh_spacing=1.,
	rupture_aspect_ratio=1.,
	magnitude_scaling_relationship='WC1994',
	dip=None,
	rake=None,
	slip_rate=None,
	mfd=None,
	min_mag=4.0,
	max_mag=None,
	mfd_bin_width=0.1,
	a_val=None,
	b_val=None,
	a_sigma=None,
	b_sigma=None,
	catalog=None,
	catalog_params={},
	bg_zone="",
	encoding="latin1",
	verbose=True,
	**kwargs
	):
	"""
	Create simple fault source from GIS record.

	:param source_rec:
		GIS record as returned by :func:`mapping.geotools.read_gis.read_gis_file`
		representing simple fault source.
	:param column_map:
		dict, mapping source parameter names to GIS columns or scalars
		(default: DEFAULT_SIMPLE_FAULT_SOURCE_COLUMN_MAP)
	:param tectonic_region_type:
		str, tectonic region type
		(default: "")
	:param upper_seismogenic_depth:
		float, upper seismogenic depth in km
		(default: None)
	:param lower_seismogenic_depth:
		float, lower seismogenic depth in km
		(default: None)
	:param rupture_mesh_spacing:
		float, rupture mesh spacing in km
		(default: 1.0)
	:param rupture_aspect_ratio:
		float, characteristic length/width ratio of modeled ruptures
		(default: 1.0)
	:param magnitude_scaling_relationship:
		str or instance of class:`openquake.hazardlib.scalerel.base.BaseMSR`
		magnitude-area scaling relationship
		(default: 'WC1994')
	:param dip:
		float, dip angle in degrees
		Note that dip distribution can be specified through column map
		(default: None)
	:param rake:
		float, rake in degrees
		Note that rake distribution can be specified through column map
		(default: None)
	:param slip_rate:
		float, slip rate in mm/yr
		Note that slip rate distribution can be specified through column map
		(default: None)
	:param mfd:
		instance of :class:`oqhazlib.mfd.base.BaseMFD`, magnitude-frequency
		distribution.
		Note: takes precedence over other mfd parameters
		(default: None)
	:param min_mag:
		float, minimum magnitude
		(default: 4.0)
	:param max_mag:
		float, maximum magnitude
		(default: None)
	:param mfd_bin_width:
		float, magnitude bin width for MFD
		(default: 0.1)
	:param a_val:
		float, a-value
		(default: None)
	:param b_val:
		float, b-value
		(default: None)
	:param a_sigma:
		float, standard deviation on a-value
		(default: None)
	:param b_sigma:
		float, standard deviation on b-value
		(default: None)
	:param catalog:
		instance of :class:`EQCatalog` that will be used to determine
		MFD (default: None)
		(default: None)
	:param catalog_params:
		dict, defining catalog processing parameters, in particular the keys
		'Mtype', 'Mrelation', and 'completeness'
		(default: {})
	:param bg_zone:
		str, name of background zone
		(default: "")
	:param encoding:
		str, unicode encoding in GIS record
		(default: "latin1")
	:param verbose:
		bool, whether or not to print information during reading
		(default: True)
	:param kwargs:
		dict, possible extra parameters that will be ignored

	:return:
		instance of :class:`SimpleFaultSource`
	"""
	from . import SimpleFaultSource

	## ID and name
	source_id = import_param(source_rec, column_map, 'id', str, encoding=encoding)
	name = import_param(source_rec, column_map, 'name', str, encoding=encoding)
	if verbose:
		print(source_id)

	## Tectonic region type
	if not tectonic_region_type:
		tectonic_region_type = import_param(source_rec, column_map,
								'tectonic_region_type', str, encoding=encoding)

	## Upper and lower seismogenic depth
	if upper_seismogenic_depth is None:
		upper_seismogenic_depth = import_param(source_rec, column_map,
											'upper_seismogenic_depth', float)
	if not lower_seismogenic_depth:
		lower_seismogenic_depth = import_param(source_rec, column_map,
											'lower_seismogenic_depth', float)

	## Magnitude scaling relationship
	if not magnitude_scaling_relationship:
		magnitude_scaling_relationship = import_param(source_rec, column_map,
									'magnitude_scaling_relationship', str)
	if isinstance(magnitude_scaling_relationship, basestring):
		magnitude_scaling_relationship = getattr(oqhazlib.scalerel,
											magnitude_scaling_relationship)()

	## Fault trace
	points = []
	linear_ring = source_rec['obj']
	## In some versions of ogr, GetPoints method does not exist
	#points = linear_ring.GetPoints()
	points = [linear_ring.GetPoint(i) for i in range(linear_ring.GetPointCount())]
	fault_trace = Line([Point(*pt) for pt in points])

	## Dip
	if dip is None:
		if "dip_distribution" in column_map:
			dip_dist = import_distribution(source_rec, column_map, 'dip_distribution')
			if dip_dist:
				dip = dip_dist.get_mean()
		else:
			dip = import_param(source_rec, column_map, 'dip', float)
	if dip is None:
		raise Exception("Dip not defined for source %s" % source_id)

	## Rake
	if rake is None:
		if "rake_distribution" in column_map:
			rake_dist = import_distribution(source_rec, column_map, 'rake_distribution')
			if rake_dist:
				weights = list(map(float, rake_dist.weights))
				rake = mean_angle(rake_dist.values, weights=weights)
		else:
			rake = import_param(source_rec, column_map, 'rake', float)
	if rake is None:
		raise Exception("Rake not defined for source %s" % source_id)
	else:
		## Constrain to (-180, 180)
		if rake > 180:
			rake -= 360.

	## Slip rate
	if 'slip_rate_distribution' in column_map:
		slip_rate_dist = import_distribution(source_rec, column_map,
											'slip_rate_distribution')
		if slip_rate_dist:
			slip_rate = slip_rate_dist.get_mean()
	if slip_rate is None:
		slip_rate = import_param(source_rec, column_map, 'slip_rate', float)
	if slip_rate is None:
		print("Warning: Slip rate not defined for source %s" % source_id)

	"""
	if min_slip_rate is None:
		min_slip_rate = import_param(source_rec, column_map, 'min_slip_rate', float)
	if max_slip_rate is None:
		max_slip_rate = import_param(source_rec, column_map, 'max_slip_rate', float)
	if None in (min_slip_rate, max_slip_rate):
		print("Warning: Slip rate not defined for source %s" % source_id)
		slip_rate = None
	else:
		slip_rate = (min_slip_rate + max_slip_rate) / 2.
	"""

	## Background zone
	if not bg_zone:
		bg_zone = import_param(source_rec, column_map, 'bg_zone', str)

	## Instantiate SimpleFaultSource object
	## Dummy MFD needed to create SimpleFaultSource, will be overwritten later
	if not mfd:
		dummy_mfd = TruncatedGRMFD(4.0, 5.0, 0.1, 2.3, 0.95)
	else:
		dummy_mfd = mfd
	simple_fault_source = SimpleFaultSource(source_id, name, tectonic_region_type,
		dummy_mfd, rupture_mesh_spacing, magnitude_scaling_relationship,
		rupture_aspect_ratio, upper_seismogenic_depth, lower_seismogenic_depth,
		fault_trace, dip, rake, slip_rate, bg_zone)

	## Override max_mag if there is a distribution (SHARE...)
	if 'max_mag_distribution' in column_map:
		max_mag_dist = import_distribution(source_rec, column_map,
											'max_mag_distribution')
		if max_mag_dist:
			max_mag = max_mag_dist.get_mean()

	## MFD
	if not mfd:
		if catalog is not None:
			## Determine MFD from catalog if one is specified
			## We do not read from GIS record (or column_map) in this case
			try:
				mfd = get_gr_mfd_from_catalog(catalog, catalog_params, min_mag,
											max_mag, mfd_bin_width, b_val, verbose)
			except ValueError:
				pass
		else:
			## Read parameters from GIS table
			try:
				mfd = read_gr_mfd(source_rec, column_map, min_mag=min_mag,
						max_mag=max_mag, mfd_bin_width=mfd_bin_width,
						a_val=a_val, b_val=b_val, a_sigma=a_sigma, b_sigma=b_sigma)
			except ValueError:
				mfd = None

		if not mfd:
			try:
				## Fall back to characteristic MFD
				mfd_params = read_gr_mfd_params(source_rec, column_map, min_mag=min_mag,
						max_mag=max_mag, mfd_bin_width=mfd_bin_width,
						a_val=a_val, b_val=b_val, a_sigma=a_sigma, b_sigma=b_sigma)

				if mfd_bin_width is None:
					mfd_bin_width = mfd_params[2]
				if max_mag is None:
					max_mag = mfd_params[1]
				if not max_mag:
					max_mag = simple_fault_source.calc_Mmax_Wells_Coppersmith()
				if max_mag and mfd_bin_width:
					max_mag = np.ceil(max_mag / mfd_bin_width) * mfd_bin_width
				simple_fault_source.mfd.max_mag = max_mag
				mfd = simple_fault_source.get_characteristic_mfd(bin_width=mfd_bin_width,
													force_bin_alignment=False)
			except:
				mfd = None
				print("Warning: MFD could not be determined for source %s"
						% source_id)
			else:
				print("Warning: Falling back to characteristic MFD for source %s"
						% source_id)

	simple_fault_source.mfd = mfd

	return simple_fault_source


def import_source_from_gis_record(
	source_rec,
	column_map=None,
	catalog=None,
	catalog_params={},
	verbose=True,
	encoding="latin1",
	**kwargs):
	"""
	Wrapper function to create various types of sources from GIS records

	:param source_rec:
		GIS record as returned by :func:`mapping.geotools.read_gis.read_gis_file`
	:param column_map:
		dict, mapping source parameter names to GIS columns or scalars
		(default: None, will use default column map for given type of source)
	:param catalog:
		instance of :class:`eqcatalog.EQCatalog`
		(default: None)
	:param catalog_params:
		dict, defining catalog processing parameters, in particular the keys
		'Mtype', 'Mrelation', and 'completeness'
		(default: {})
	:param encoding:
		str, unicode encoding in GIS record
		(default: "latin1")
	:param verbose:
		bool, whether or not to print information during reading
		(default: True)
	:param kwargs:
		dict, mapping parameter names to values, overriding entries
		in :param:`column_map` or in GIS file or in default kwargs for
		particular source
	"""
	## Decide which function to use based on object type
	obj_type = source_rec["obj"].GetGeometryName()
	if obj_type in ("POINT", "POLYGON"):
		if obj_type == "POINT":
			default_column_map = DEFAULT_POINT_SOURCE_COLUMN_MAP
		else:
			default_column_map = DEFAULT_AREA_SOURCE_COLUMN_MAP
		func = import_point_or_area_source_from_gis_record
	elif obj_type in "LINESTRING":
		default_column_map = DEFAULT_SIMPLE_FAULT_SOURCE_COLUMN_MAP
		func = import_simple_fault_source_from_gis_record
	else:
		print("Warning: %s geometry not supported!" % obj_type)

	if column_map is None:
		column_map = default_column_map.copy()
	else:
		column_map = dict((k,w) for (k,w) in column_map.items() if k in default_column_map)
		kwargs = dict((k,w) for (k,w) in kwargs.items())

	return func(source_rec, column_map=column_map, catalog=catalog,
					catalog_params=catalog_params, encoding=encoding,
					verbose=verbose, **kwargs)


def import_source_model_from_gis(
	gis_filespec,
	name="",
	column_map=None,
	source_ids=[],
	source_catalogs={},
	overall_catalog=None,
	catalog_params={},
	encoding="latin1",
	verbose=True,
	**kwargs):
	"""
	Import source model from GIS file

	:param gis_filespec:
		str, full path to GIS file containing source model
	:param name:
		str, name of source model
		(default: "")
	:param column_map:
		dict, mapping source parameter names to GIS columns or scalars
		(default: None)
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
		#(default: "guess", will try to guess, but this may fail)
		(default: "latin1")
	:param verbose:
		bool, whether or not to print information during reading
		(default: True)
	:param kwargs:
		dict, mapping parameter names to values, overriding entries
		in :param:`column_map` or in GIS file or in default kwargs for
		particular source

	:return:
		instance of :class:`..source.SourceModel`
	"""
	from .source_model import SourceModel

	sources = []
	source_records = read_gis_file(gis_filespec, encoding=None, verbose=verbose)
	for source_rec in source_records:
		source_id = import_param(source_rec, column_map, 'id', str)
		if source_ids and not source_id in source_ids:
			continue
		catalog = source_catalogs.get(source_id)
		if not catalog and overall_catalog:
			## Extract source catalog from overall catalog
			geom = source_rec['obj']
			if geom.GetGeometryName() == "POLYGON":
				catalog = overall_catalog.subselect_polygon(geom)

		source = import_source_from_gis_record(source_rec, column_map, catalog=catalog,
						catalog_params=catalog_params, verbose=verbose, **kwargs)
		sources.append(source)
	if not name:
		name = os.path.splitext(os.path.split(gis_filespec)[1])[0]
	source_model = SourceModel(name, sources)
	return source_model
