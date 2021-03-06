"""
Functions to read and write CRISIS input and output files
"""
### imports
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np

from ..utils import interpolate
from ..mfd import TruncatedGRMFD, CharacteristicMFD
from ..site import GenericSite
from ..source import PointSource, AreaSource, SimpleFaultSource, ComplexFaultSource
from ..result import *



__all__ = ['write_DAT_2007', 'write_ASC', 'read_DAT', 'read_GRA',
		'read_GRA_multi', 'read_MAP', 'read_DES', 'read_DES_full',
		'read_batch', 'get_crisis_rupture_area_parameters']


def str2float(s, replace_nan=None, replace_inf=None):
	"""
	Convert string to float, taking into account NaN and inf values.
	This is handled in Python from version 2.6 onwards
	Optionally, values can be specified to replace occurrences of NaN and inf
	"""
	try:
		return float(s)
	except ValueError:
		if s.lower() == "nan":
			return {None: np.NaN}.get(replace_nan, replace_nan)
		elif s.lower() in ("inf", "+inf"):
			return {None: np.inf}.get(replace_inf, replace_inf)
		elif s.lower() in ("-inf"):
			return {None: -np.inf}.get(replace_inf, -replace_inf)


def write_DAT_2007(filespec, source_model, ground_motion_model, gsim_atn_map,
					return_periods, grid_outline, grid_spacing=0.5, sites=[],
					sites_filespec='', imt_periods={"PGA": [0]},
					intensities=None, min_intensities={"PGA": [1E-3]}, max_intensities={"PGA": [1.0]},
					num_intensities=100, imt_unit="g", model_name="",
					truncation_level=3., integration_distance=200.,
					source_discretization=(1.0, 5.0), vs30=800., kappa=None,
					mag_scale_rel="WC1994", atn_Mmax=None,
					output={"gra": True, "map": True, "fue": False,
							"des": False, "smx": True, "eps": False,
							"res_full": False},
					deagg_dist_metric="Hypocentral", map_filespec="", cities_filespec="",
					overwrite=False):
	"""
	Write input file for CRISIS2007

	:param filespec:
		String, full path specification for output file
	:param source_model:
		oqhazlib_nrml.SourceModel object
		Note that geometry of area sources must be specified in counterclockwise
		order.

	:param ground_motion_model:
		oqhazlib_rob.GroundMotionModel object, mapping tectonic region types to
		gsims (GMPE's) (1 gsim for each trt).
	:param gsim_atn_map:
		Dictionary mapping gsims (in ground_motion_model) to paths of attenuation
		tables (.ATN files). If the tables do not exist, they will be generated
		automatically. Note that tables will be written for each fault type
		("normal", "reverse", and "strike-slip") if the gsim is rake dependent.
	:param return_periods:
		List, tuple or array of max. 5 return periods for computation of hazard maps
	:param grid_outline:
		List of (float, float) tuples or oqhazlib_rob.GenericSite objects, containing
		(longitude, latitude) of points defining outline of grid where hazard
		will be computed. If only 2 points are given, they define the lower left
		and upper richt corners. If there are more than 2 points, grid_outline
		is considered as a grid reduction polygon, inside which a regular grid
		will be computed. In that case, vertexes must be specified in counter-
		clockwise order
		(default: []).
	:param grid_spacing:
		Float or tuple of floats, defining grid spacing in degrees (in longitude
		and latitude if tuple)
		(default: 0.5).
	:param sites:
		List of (float, float) tuples or oqhazlib_rob.GenericSite objects, defining
		(longitude, latitude) of sites
		(default: []).
	:param sites_filespec:
		String, full path to .ASC file containing individual sites to compute
		(default: '').
	:param imt_periods:
		Dictionary mapping intensity measure types (e.g. "PGA", "SA", "PGV",
		"PGD") to lists or arrays of periods in seconds (float values).
		Maximum number of periods in CRISIS is 40.
		(default: {"PGA": [0]})
	:param intensities:
		List of floats or array, defining equal intensities for all intensity
		measure types and periods. When given, params min_intensities,
		max_intensities and num_intensities are not set.
		(default: None)
	:param min_intensities:
		Dictionary mapping intensity measure types (e.g. "PGA", "SA", "PGV",
		"PGD") to lists or arrays with minimum intensity for each spectral
		period. If list contains only one value, it will be extended to the
		number of periods for that IMT.
		(default: {"PGA": [1E-3]})
	:param max_intensities:
		Dictionary mapping intensity measure types (e.g. "PGA", "SA", "PGV",
		"PGD") to lists or arrays with maximum intensity for each spectral
		period. If list contains only one value, it will be extended to the
		number of periods for that IMT.
		(default: {"PGA": [1.0})
	:param num_intensities:
		Integer, number of (logarithmically-spaced) intensities to be computed
		by CRISIS. The maximum is 100. If param intensities is given,
		num_intensities is set to 100 to give best possible interpolation.
		(default: 100)
	:param imt_unit:
		String, intensity measure unit
		Note: it may be necessary to specify this as a dictionary, if different
		IMT's are used with different units.
		(default: "g")
	:param model_name:
		String, name of model
		(default: "")
	:param truncation_level":
		Float, truncation level in number of standard deviations of GSIM
		(default: 3.)
	:param integration_distance:
		Float, maximum distance in km for spatial integration
		(default: 200)
	:param source_discretization:
		Tuple of 2 Floats (min_triangle_size,  min_dist_triangle_ratio), which
		control the discretization of (area) sources. CRISIS subdivides sources
		into triangles, until one of 2 conditions is met (see CRISIS help file):
		- size of triangle is smaller than min_triangle_size (square root of
		  the triangle area)
		- ratio between site-to-source distance and triangle size is larger than
		  min_dist_triangle_ratio
		Some possibilities:
		- (11, 3): CRISIS default, OK when site is outside source zones
		- (2.5, 4): Values used in ROB calculations for EC8
		- (1.0, 5): Recommended values when site is close to source
		(default: 1.0, 5)
	:param vs30:
		Float, shear-wave velocity in the upper 30 m (in m/s). This is used to
		determine the correct soil type for each gsim
		(default: 800, which should correspond to "rock" in most cases).
	:param kappa:
		Float, kappa value in seconds
		(default: None)
	:param mag_scale_rel:
		String, name of magnitude-area scaling relationship to be used,
		one of "WC1994", "Brune1970" or "Singh1980"
		(default: "").
		If empty, the scaling relationships associated with the individual
		source objects will be used.
	:param atn_Mmax:
		Float, maximum magnitude in attenuation tables
		(default: None, will determine from source model)
	:param output:
		Dict with boolean values, indicating which output files to generate
		Keys correspond to output file types:
		"gra": whether or not to output .GRA file (exceedance rates for each
			site)
			(default: True)
		"map": whether or not to output .MAP file (intensities corresponding to
			fixed return period)
			(default: True)
		"fue": whether or not to output .FUE file (exceedance rates deaggregated
			by source)
			(default: True)
		"des": whether or not to output .DES file (exceedance rates deaggregated
			by magnitude and distance)
			(default: True)
		"smx": whether or not to output .SMX file (maximum earthquakes)
			(default: True)
		"eps": whether or not to output .EPS file (exceedance rates deaggregated
			by epsilon
			(default: True)
		"res_full": whether or not .RES file should include input data +
			exceedance rates (True) or only input data (False)
			(default: False)
	:param deagg_dist_metric:
		String, distance metric for deaggregation: "Hypocentral", "Epicentral",
		"Joyner-Boore" or "Rupture"
		(default: "Hypocentral")
	:param map_filespec:
		String, full path specification for map file to draw as background
		(default: "")
	:param cities_filespec:
		String, full path specification for cities file to draw as background
		(default: "")
	:param overwrite:
		Boolean, whether or not to overwrite existing input files
		(default: False)
	"""
	from ..mfd.truncated_gr import alphabetalambda
	from ..gsim import CRISIS_DISTANCEMETRICS
	from shapely.geometry.polygon import LinearRing

	if intensities:
		num_intensities = 100

	## Generate attenuation tables if they don't exist.
	## Note that we need 3 tables, one for each fault mechanism
	gsims = gsim_atn_map.keys()
	#atn_folder = os.path.split(atn_filespecs[0])[0]
	gsims_num_rakes = []
	for gsim in gsims:
		atn_filespecs = []
		# HACK for different name structure in attenuation
		if hasattr(att, gsim+"GMPE"):
			gsimObj = getattr(att, gsim+"GMPE")()
		else:
			gsimObj = getattr(att, gsim)()
		if gsimObj.is_rake_dependent():
			mechanisms = ["normal", "reverse", "strike-slip"]
			gsims_num_rakes.append(3)
		else:
			mechanisms = [""]
			gsims_num_rakes.append(1)
		for mechanism in mechanisms:
			atn_filespec = os.path.splitext(gsim_atn_map[gsim])[0]
			if mechanism:
				atn_filespec += "_%s" % mechanism
			else:
				## Set mechanism to arbitrary type to avoid problem with
				## unspecified mechanism in oqhazlib GMPE's
				mechanism = "normal"
			atn_filespec += "_VS30=%s" % vs30
			if kappa:
				atn_filespec += "_kappa=%s" % kappa
			atn_filespec += ".ATN"
			atn_filespecs.append(atn_filespec)
			Mmin = source_model.min_mag
			Mmax = max(atn_Mmax, source_model.max_mag)
			Mstep, num_distances = 0.5, 50
			if not os.path.exists(atn_filespec) or overwrite:
				# TODO: check for gsim names in attenuation
				if gsimObj.distance_metric in ("Hypocentral", "Rupture"):
					## Rhypo, Rrup
					## Determine minimum distance as minimum depth
					## Make a list of sources to which curreng gsim applies
					src_list = []
					for src in source_model:
						trt = src.tectonic_region_type
						if gsim in ground_motion_model[trt]:
							src_list.append(src)
					min_distances = []
					for src in src_list:
						if isinstance(src, (PointSource, AreaSource)):
							min_distances.append(src.hypocenter_distribution.min())
						elif isinstance(src, (SimpleFaultSource, ComplexFaultSource)):
							min_distances.append(src.upper_seismogenic_depth)
					if len(min_distances) > 0:
						min_distance = min(min_distances)
						min_distance = max(min_distance, 1.)
					else:
						min_distance = 1.0
				else:
					## Repi, RJB
					min_distance = 1.0
				# TODO: damping?
				gsimObj.writeCRISIS_ATN(Mmin, Mmax, Mstep, min_distance,
										integration_distance, num_distances,
										h=0, imt_periods=imt_periods,
										imt_unit=imt_unit,
										num_sigma=truncation_level,
										vs30=vs30, kappa=kappa, mechanism=mechanism,
										damping=5, filespec=atn_filespec)
			else:
				print("Warning: CRISIS ATN file %s exists! "
					"Set overwrite=True to overwrite." % atn_filespec)
	gsim_atn_map[gsim] = atn_filespecs
	gsims_num_rakes = np.array(gsims_num_rakes)
	num_atn_tables = np.sum(gsims_num_rakes)

	## Fill list of return periods with zeros to make a total of 5
	return_periods = list(return_periods)
	return_periods.extend([0] * (5 - len(return_periods)))

	## Write input file
	## Skip if file exists and overwrite is not True
	if os.path.exists(filespec) and not overwrite:
		print("Warning: CRISIS DAT file %s exists! "
				"Set overwrite=True to overwrite." % filespec)
		return

	of = open(filespec, "w")
	of.write("[CRISIS2007 format. "
			"Be careful when editing this file by hand]\n")

	## Which output files to generate
	if grid_outline and len(grid_outline) > 1:
		input_sites = 0
	elif sites:
		input_sites = 1
	of.write("%d,%d,%d,%d,%d,%d,%d,%d\n"
			% (output['res_full'], output['gra'], output['map'], output['fue'],
				output['des'], input_sites, output['smx'], output['eps']))

	## Model name
	if model_name:
		of.write("%s\n" % model_name)
	else:
		of.write("%s source model - %s GMPE\n"
				% (source_model.name, ground_motion_model.name))

	## Construct list of all periods
	all_periods = []
	for imt in imt_periods.keys():
		for T in imt_periods[imt]:
			## CRISIS does not support non-numeric spectral periods,
			## so different IMT's may not have duplicate periods!
			if not T in all_periods:
				all_periods.append(T)
			else:
				raise Exception("Duplicate period found: %s (%s s)"
								% (imt, T))

	## Dimensions
	## Compute total number of sources taking into account nodal plane
	## and hypodepth distributions
	num_sources = 0
	for source in source_model:
		## Index of attenuation table
		gsim_name = ground_motion_model[source.tectonic_region_type]
		gsims_index = gsims.index(gsim_name)

		## First determine number of rakes (faulting styles) and focal depths
		if isinstance(source, (SimpleFaultSource, ComplexFaultSource)):
			num_rakes = 1
		else:
			num_rakes = min(gsims_num_rakes[gsims_index],
							source.nodal_plane_distribution.get_num_rakes())

		if isinstance(source, (SimpleFaultSource, ComplexFaultSource)):
			num_depths = 1
		else:
			if hasattr(att, gsim_name+"GMPE"):
				gsimObj = getattr(att, gsim_name+"GMPE")()
			else:
				gsimObj = getattr(att, gsim_name)()
			if gsimObj.distance_metric in ("Joyner-Boore", "Epicentral"):
				num_depths = 1
			else:
				num_depths = len(source.hypocenter_distribution)

		num_sources += (num_depths * num_rakes)

	of.write("%d,%d,%d,%d\n"
			% (num_sources, num_atn_tables, len(all_periods), num_intensities))

	## IMTLS
	## Periods should be in ascending order
	## Sorting IMT's makes sure PGA comes before SA
	for imt in sorted(imt_periods.keys()):
		periods = imt_periods[imt]
		if intensities:
			Imin = [intensities[0]]
		else:
			Imin = min_intensities[imt]
		if len(Imin) == 1:
			Imin = list(Imin) * len(periods)
		if intensities:
			Imax = [intensities[-1]]
		else:
			Imax = max_intensities[imt]
		if len(Imax) == 1:
			Imax = list(Imax) * len(periods)
		for T in sorted(periods):
			k = periods.index(T)
			of.write("%.3E,%.2E,%.2E,%s\n" % (T, Imin[k], Imax[k], imt_unit))

	## Integration parameters, return periods, and distance metric for deaggregation
	min_triangle_size, min_dist_triangle_ratio = source_discretization
	deagg_dist_metric = CRISIS_DISTANCEMETRICS[deagg_dist_metric]
	of.write("%s,%s,%s,%s,%s,%s,%s,%s,%d\n"
			% ((integration_distance, min_dist_triangle_ratio, min_triangle_size)
			+ tuple(return_periods[:5]) + (deagg_dist_metric,)))

	## Grid or sites where to compute hazard
	if grid_outline and len(grid_outline) > 1:
		if not isinstance(grid_spacing, (tuple, list)):
			grid_spacing = (grid_spacing, grid_spacing)
		lons, lats = zip(*grid_outline)[:2]
		min_lon, max_lon = min(lons), max(lons)
		min_lat, max_lat = min(lats), max(lats)
		## Round min/max lon/lat to multiples of grid_spacing
		## Make sure dlon/dlat is multiple of grid_spacing
		#min_lon = np.ceil((min_lon / grid_spacing[0])) * grid_spacing[0]
		#max_lon = np.floor((max_lon / grid_spacing[0])) * grid_spacing[0]
		#min_lat = np.ceil((min_lat / grid_spacing[1])) * grid_spacing[1]
		#max_lat = np.floor((max_lat / grid_spacing[1])) * grid_spacing[1]
		dlon, dlat = max_lon - min_lon, max_lat - min_lat
		dlon = np.ceil((dlon / grid_spacing[0])) * grid_spacing[0]
		dlat = np.ceil((dlat / grid_spacing[1])) * grid_spacing[1]
		grid_origin = (min_lon, min_lat)
		#num_grid_lons = ((max_lon - min_lon) / grid_spacing[0]) + 1
		#num_grid_lats = ((max_lat - min_lat) / grid_spacing[1]) + 1
		num_grid_lons = (dlon / grid_spacing[0])
		num_grid_lats = (dlat / grid_spacing[1])
		grid_numlines = (num_grid_lons, num_grid_lats)
		of.write("%.3f,%.3f,%.3f,%.3f,%d,%d\n"
				% (grid_origin + grid_spacing + grid_numlines))
	elif sites:
		if not sites_filespec:
			sites_filespec = os.path.join(os.path.dirname(filespec), 'sites.ASC')
		of.write("%s\n" % sites_filespec)
		if not os.path.exists(sites_filespec):
			write_ASC(sites_filespec, sites)
		else:
			if overwrite:
				write_ASC(sites_filespec, sites)
	else:
		raise Exception("Must specify either sites or grid_outline!")

	## Grid reduction polygon
	if grid_outline and len(grid_outline) > 2:
		of.write(" 1\n")
		of.write("%d" % len(grid_outline))
		## Check that vertices are in CCW order
		ring = LinearRing(grid_outline)
		if not ring.is_ccw:
			grid_outline.reverse()
		for pt in grid_outline:
			of.write("%s,%s\n" % (pt[0], pt[1]))
	else:
		of.write(" 0\n")

	## Attenuation tables
	for gsim in gsim_atn_map.keys():
		for atn_filespec in gsim_atn_map[gsim]:
			of.write("%s\n" % atn_filespec)

	## Source geometry and activity
	for source in source_model:
		## Index of attenuation table
		gsim_name = ground_motion_model[source.tectonic_region_type]
		gsims_index = gsims.index(gsim_name)

		## First determine number of rakes (faulting styles) and focal depths
		#if hasattr(source, 'rake'):
		if isinstance(source, (SimpleFaultSource, ComplexFaultSource)):
			num_rakes = 1
			rake_weights, rakes = np.array([1.]), np.array([source.rake])
		else:
			num_rakes = source.nodal_plane_distribution.get_num_rakes()
			rakes, rake_weights = source.nodal_plane_distribution.get_rake_weights()
			## If GMPE is not rake-dependent, set num_rakes to 1, and sum rake_weights
			if gsims_num_rakes[gsims_index] < num_rakes:
				num_rakes = 1
				rake_weights[:] = [np.sum(rake_weights), 0., 0.]
		non_zero_rake_indexes = np.where(rake_weights > 0)[0]


		if isinstance(source, (SimpleFaultSource, ComplexFaultSource)):
			num_depths = 1
		else:
			## Collapse number of depths if GMPE distance metric is epicentral
			## or Joyner-Boore
			if hasattr(att, gsim_name+"GMPE"):
				gsimObj = getattr(att, gsim_name+"GMPE")()
			else:
				gsimObj = getattr(att, gsim_name)()
			if gsimObj.distance_metric in ("Joyner-Boore", "Epicentral"):
				num_depths = 1
			else:
				num_depths = len(source.hypocenter_distribution)

		## Loop over rakes
		for r in range(num_rakes):
			try:
				rake = rakes[non_zero_rake_indexes][r]
				rake_weight = np.float(rake_weights[non_zero_rake_indexes][r])
			except TypeError:
				rake = rakes[non_zero_rake_indexes]
				rake_weight = np.float(rake_weights[non_zero_rake_indexes])

			if -135 <= rake < -45:
				sof = "normal"
				rake_index = 1
			elif 45 <= rake <= 135:
				sof = "reverse"
				rake_index = 2
			else:
				sof = "ss"
				rake_index = 3

			## Loop over focal depths
			for d in range(num_depths):
				if isinstance(source, (AreaSource, PointSource)):
					hypo_depth = source.hypocenter_distribution.hypo_depths[d]
					if num_depths == 1:
						depth_weight = 1.0
					else:
						depth_weight = np.float(source.hypocenter_distribution.weights[d])
				else:
					depth_weight = 1.0

				## Name
				name = source.source_id
				if isinstance(source, (AreaSource, PointSource)):
					if num_rakes > 0:
						name += "_%s" % sof
					if num_depths > 0:
						name += "_%dkm" % hypo_depth
				of.write("%s\n" % name)

				## Main rake
				#if hasattr(source, 'rake'):
				#	rake = source.rake
				#else:
					## If nodal-plane distribution has different rakes, take the one
					## with the highest weight or the first one if they all have the
					## same weight
				#	rake = source.nodal_plane_distribution.get_main_rake()

					# TODO: if there is only 1 rake, generate corresponding ATN table.
					# else, if GMPE is not rake dependent, generate an "averaged" ATN
					# table !

				## GMPE
				## TODO: magnitude scaling in CRISIS is different for line sources!
				## Magnitude scaling relationship
				if mag_scale_rel:
					msr = mag_scale_rel
				else:
					## Convert class name to string
					msr = repr(source.magnitude_scaling_relationship).split(" ")[0].split(".")[-1]
				k1, k2 = get_crisis_rupture_area_parameters(msr, rake)
				## Index of attenuation table
				#if -135 <= rake <= -45:
				#	rake_index = 1
				#elif 45 <= rake <= 135:
				#	rake_index = 2
				#else:
				#	rake_index = 3
				atn_index = np.sum(gsims_num_rakes[:gsims_index])
				atn_index += min(gsims_num_rakes[gsims_index], rake_index)
				if isinstance(source, AreaSource):
					source_type = 0
				elif isinstance(source, SimpleFaultSource):
					## We model fault sources as dipping area sources.
					source_type = 0
				elif isinstance(source, PointSource):
					source_type = 2
				else:
					raise Exception("Complex fault sources are "
									"not supported in CRISIS!")
				is_alive = True
				if isinstance(source.mfd, TruncatedGRMFD):
					mfd_type = 1
				elif isinstance(source.mfd, CharacteristicMFD):
					mfd_type = 2
				else:
					raise Exception("CRISIS does not support EvenlyDiscretizedMFD !")
				of.write("%d,%d,%d,%s,%s,%s\n"
						% (mfd_type, source_type, atn_index, is_alive, k1, k2))

				## Geometry
				if not isinstance(source, PointSource):
					if isinstance(source, AreaSource):
						vertexes = source.polygon.points
					elif isinstance(source, SimpleFaultSource):
						vertexes = source.get_polygon().points
					## Test if first and last point are different
					if vertexes[0] == vertexes[-1]:
						vertexes = vertexes[:-1]
					## Test if vertexes in polygon are in CCW order
					ring = LinearRing(vertexes)
					if not ring.is_ccw:
						vertexes.reverse()

					of.write(" %d\n" % len(vertexes))
					if isinstance(source, AreaSource):
						#z = source.hypocenter_distribution.mean()
						for point in vertexes:
							of.write("%s,%s,%.1f\n"
									% (point.longitude, point.latitude, hypo_depth))
					elif isinstance(source, SimpleFaultSource):
						for point in vertexes:
							of.write("%s,%s,%.1f\n"
									% (point.longitude, point.latitude, point.depth))
				else:
					point = source.location
					of.write(" %d\n" % 1.)
					of.write("%s,%s,%.1f\n"
							% (point.longitude, point.latitude, hypo_depth))

				## MFD
				#print(rake_weight, depth_weight)
				mfd = source.mfd * (rake_weight * depth_weight)
				if isinstance(mfd, TruncatedGRMFD):
					a, b = mfd.a_val, mfd.b_val
					Mmin = mfd.min_mag
					abl = alphabetalambda(a, b, Mmin)
					beta = abl[1]
					beta_cov = 0
					lbd = abl[2]
					Mmax_expected = mfd.max_mag
					Mmax_stdev = 0
					Mmax_lower_limit = mfd.max_mag
					Mmax_upper_limit = mfd.max_mag
					of.write("%.6f,%.6f,%s,%.2f,%.2f,%.2f,%.2f,%.2f\n"
							% (lbd, beta, beta_cov, Mmax_expected, Mmax_stdev,
								Mmax_lower_limit, Mmin, Mmax_upper_limit))
				elif isinstance(mfd, CharacteristicMFD):
					#Mmin = mfd.get_min_mag_edge()
					#Mmax = mfd.get_magnitude_bin_edges()[-1]
					Mmin = mfd.char_mag - mfd.M_sigma * mfd.num_sigma
					Mmax = mfd.char_mag + mfd.M_sigma * mfd.num_sigma + mfd.bin_width
					M_sigma = mfd.M_sigma
					return_period = mfd.return_period
					D = mfd.char_mag
					## time-dependent parameters
					F, T00 = 0, 1
					of.write("%s, %s, %s, %s, %.2f, %.2f, %.2f\n"
							% (return_period, T00, D, F, M_sigma, Mmin, Mmax))
				else:
					raise Exception("CRISIS does not support EvenlyDiscretizedMFD !")
				of.write(" 0\n")

	## Additional background maps
	if map_filespec:
		of.write("%s\n" % map_filespec)
	else:
		of.write("Empty\n")
	if cities_filespec:
		of.write("%s\n" % cities_filespec)
	else:
		of.write("Empty\n")
	of.write(" 0\n")

	of.close()


def write_ASC(asc_filespec, sites):
	"""
	Write Crisis asc sites file.

	:param asc_filespec:
		String, defining filespec for asc file.
	:param sites:
		List of (float, float) tuples, defining (longitude, latitude)
		of sites.
	"""
	f = open(asc_filespec, 'w')
	f.write('%d' % len(sites))
	for site in sites:
		f.write('\n, %s, %s, %s' % (site.name, site.longitude, site.latitude))
	f.close()


def read_batch(batch_filespec):
	"""
	Read Crisis batch file.

	:param batch_filespec:
		str, full path to batch file

	:return:
		(filespecs, weights) tuple
	"""
	filespecs, weights = [], []
	f = open(batch_filespec)
	for line in f:
		filespec, weight = line.split(',')
		filespec = os.path.splitext(filespec)[0]
		filespecs.append(gra_filespec)
		weights.append(float(weight))
	f.close()

	return filespecs, np.array(weights)


def read_DAT(filespec):
	"""
	Read spectral periods and intensity_unit from CRISIS .DAT file

	:paam filespec:
		str, full path to file to be read

	return:
		tuple (period array, intensity_unit, model_name)
	"""
	# IDEA: we could also determine number of sites and number of intensities,
	# which would facilitate reading read_GRA

	if filespec[-4:].lower() != ".dat":
		filespec += ".dat"

	periods = []
	f = open(filespec)
	for linenr, line in enumerate(f):
		if linenr == 2:
			model_name = line.strip()
		if linenr == 3:
			num_periods = int(line.split(',')[2].strip())
		elif 4 <= linenr < 4 + num_periods:
			columns = line.split(',')
			periods.append(float(columns[0].strip()))
		if linenr == 4:
			try:
				intensity_unit = columns[3].strip()
			except:
				intensity_unit = ""
	f.close()

	periods = np.array(periods, 'f')
	return (periods, intensity_unit, model_name)


def read_GRA(filespec, sites=None, out_periods=[], in_periods=[],
			intensity_unit="", convert_to_g=True, IMT="", avoid_zeros=True,
			model_name="", site_names=[], verbose=False):
	"""
	Read CRISIS .GRA or .AGR file (CRISIS2007 format)

	:param filespec:
		str, full path to file to be read
	:param sites:
		list of sites to extract, either site indexes or (lon, lat) tuples
		(default: [] or None, will extract all sites)
	:param out_periods:
		list or array, spectral periods for which exceedance means and
		variances need to be interpolated
		(default: [] or None, output only calculated spectral periods)
	:param in_periods:
		list or array, spectral periods for which exceedance means
		and variances are reported in the GRA file. If not specified,
		these are determined from the corresponding DAT file
		(default: [])
	:param intensity_unit:
		str, intensity unit. Determined from corresponding DAT file
		if not specified
		(default: "")
	:param convert_to_g:
		bool indicating whether intensities must be converted to g,
		based on their intensity unit
		(default: True)
	:param IMT:
		str, intensity measure type. Defaults to "PGA" if there is
		only 1 spectral period with value 0 or ~1/34, else "SA"
		(default: "")
	:param avoid_zeros:
		bool, if True, replace zero values with very small values:
		1E-13 for exceedance mean, and 1E-18 for exceedance variance
		(default: True)
	:param model_name:
		str, name for this model run
		(default: "")
	:param site_names:
		list of site names.
		(default: [])
	:param verbose:
		bool indicating whether or not information should be printed
		during parsing
		(default: False)

	The return value is a SpectralHazardCurveField object with the
	following properties:
		sites: list with (lon, lat) tuples of all sites
		periods: array with spectral periods
		intensities: 2-D array [k, l] with fixed intensity values for
			which exceedance rate was calculated. These can be different
			for each spectral period [k]
		exceedance_rates: 3-dimensional [i, k, l] array of mean exceedance rate
		variances: 3-dimensional [i, k, l] array of variance of exceedance rate
			i: sites
			k: spectral periods
			l: intensities
	If AGR file, the return value is a SpectralHazardCurveFieldTree
		object with the following additional properties:
		mean: 3-D array [i, k, l]
		percentile_levels: 1-D array with percentile levels
		percentiles:
			4-dimensional [i, k, l, p] array of percentiles of exceedance rate
				i: sites
				k: spectral periods
				l: intensities
				p: percentile values (P5, P16, P50, P84, P96)
	"""
	def append_period_exceedance():
		site_exceedance_means.append(np.array(period_exceedance_means, 'd'))
		site_exceedance_variances.append(np.array(period_exceedance_variances, 'd'))
		if period_exceedance_percentiles != []:
			site_exceedance_percentiles.append(np.array(period_exceedance_percentiles, 'd'))
			#site_exceedance_percentiles = np.array(site_exceedance_percentiles, 'd')

	def append_site_exceedance():
		## Convert list of period_exceedances for previous site to array, and append site
		if want_site:
			all_exceedance_means.append(np.array(site_exceedance_means, 'd'))
			all_exceedance_variances.append(np.array(site_exceedance_variances, 'd'))
			if site_exceedance_percentiles != []:
				all_exceedance_percentiles.append(np.array(site_exceedance_percentiles, 'd'))
				#all_exceedance_percentiles = np.array(all_exceedance_percentiles)


	if filespec[-4:].lower() not in (".gra", ".agr"):
		filespec += ".gra"

	if type(out_periods) != type(None):
		out_periods = list(out_periods)

	zero_period = 1./34
	if in_periods in ([], None):
		try:
			in_periods, unit, description = read_DAT(filespec[:-4])
		except IOError:
			print("%s.DAT file not found" % filespec[:-4])
			print("Assuming spectral period is %.4f s (PGA), "
					"or reading from .AGR file if possible" % zero_period)
			if not intensity_unit:
				print("  and intensity unit is 'g'")
				intensity_unit = "g"
			in_periods = np.array([zero_period], 'd')
		else:
			if not intensity_unit:
				intensity_unit = {True: "g", False: unit}[unit == ""]
			if not model_name:
				model_name = description
	if not intensity_unit:
		intensity_unit = "g"
	if not IMT:
		IMT = {True: "SA", False: "PGA"}[len(in_periods) > 1]

	all_sites = []
	all_site_names = []
	all_intensities = []
	all_exceedance_means = []
	all_exceedance_variances = []
	all_exceedance_percentiles = []
	in_periods_agr = []
	percentile_levels = np.array([5, 16, 50, 84, 95])
	site_nr, period_nr = 0, 0
	f = open(filespec)
	for linenr, line in enumerate(f):
		if linenr == 2:
			## Determine percentile levels if we are dealing with an AGR file
			try:
				s = line.split("percentiles")[1]
			except IndexError:
				pass
			else:
				s = s.replace('(', '').replace(')', '').strip()
				percentile_levels = np.array(list(map(float, s.split(','))))
				if percentile_levels.max() < 1.01:
					percentile_levels *= 100
					percentile_levels.astype('i')
		if linenr > 5:
			columns = line.split()
			if len(columns) in (2, 3) or len(columns) >= 8:
				## Start of either a new site or a new spectral period
				#if (len(columns) == 2 and not columns[0] == "INTENSITY") or columns[0] == "SITE:":
				if line[:2] == "  " and columns[0] != "INTENSITY":
					## Lon, lat coordinates of a new site
					if columns[0] == "SITE:":
						## CRISIS2003 format
						lon, lat = float(columns[1]), float(columns[2])
						try:
							site_name = columns[3]
						except:
							site_name = ""
					else:
						## CRISIS2007 format
						lon, lat = float(columns[0]), float(columns[1])
						try:
							site_name = columns[2]
						except:
							site_name = ""
					want_site = False
					if not sites or site_nr in sites or (lon, lat) in sites:
						want_site = True
					if want_site:
						all_sites.append((lon, lat))
						if site_name:
							all_site_names.append(site_name)
					site_nr += 1
					if verbose:
						print("Site nr: %d" % site_nr)

					if period_nr != 0:
						## Append last spectral period to previous site
						append_period_exceedance()
						append_site_exceedance()

						if site_nr == 2:
							## Append intensities for last period of previous site 1 if there is more than 1 site
							if verbose:
								print("Appending intensities for period #%d "
										"of previous site 1" % period_nr)
							all_intensities.append(np.array(period_intensities, 'd'))

					# Reset site_exceedance, period_exceedance, and period_intensities lists
					site_exceedance_means, site_exceedance_variances, site_exceedance_percentiles = [], [], []
					period_exceedance_means, period_exceedance_variances, period_exceedance_percentiles = [], [], []
					period_intensities = []
					period_nr = 0

				elif columns[0] == "INTENSITY":
					## New spectral period
					## GRA or AGR files normally do not include the value of the spectral periods,
					## but AGR files created with write_AGR do, so we don't need a RES or other
					## file to figure out what the spectral periods are
					try:
						in_periods_agr.append(float(columns[-1].split('=')[1]))
					except:
						pass
					period_nr += 1
					if verbose:
						print("  Period nr: %d" % period_nr)
					if period_exceedance_means:
						## All spectral periods, except first and last one
						append_period_exceedance()
						if site_nr == 1:
							#period_intensities = np.array(period_intensities, 'd')
							if verbose:
								print("Appending period intensities for period #%d"
										% (period_nr-1))
							all_intensities.append(np.array(period_intensities, 'd'))

					# Reset period_exceedance, and period_intensities lists
					period_exceedance_means, period_exceedance_variances, period_exceedance_percentiles = [], [], []
					period_intensities = []

				elif len(columns) == 3:
					## Intensity, Exceedance rate, and variance values
					intensity, exceedance, variance = [str2float(s, replace_nan=1E-18)
														for s in columns]
					if site_nr == 1:
						period_intensities.append(float(columns[0]))
					period_exceedance_means.append(exceedance)
					period_exceedance_variances.append(variance)
					period_exceedance_percentiles = []
					num_percentiles = 0

				elif len(columns) >= 8:
					## AGR files: intensity, exceedance rate, variance, and percentile values
					# TODO: allow variable number of percentiles
					intensity, exceedance, variance, p5, p16, p50, p84, p95 = \
						[str2float(s, replace_nan=1E-18) for s in columns[:8]]
					percentiles = np.array([p5, p16, p50, p84, p95], 'd')
					if site_nr == 1:
						period_intensities.append(float(columns[0]))
					period_exceedance_means.append(exceedance)
					period_exceedance_variances.append(variance)
					period_exceedance_percentiles.append(percentiles)
					num_percentiles = 5

	f.close()
	if verbose:
		print("Finished reading file")

	## Append last spectral period
	append_period_exceedance()
	if site_nr == 1:
		#period_intensities = np.array(period_intensities, 'd')
		if verbose:
			print("Appending period intensities for period #%d" % period_nr)
		all_intensities.append(np.array(period_intensities, 'd'))
	all_intensities = np.array(all_intensities)

	## Append last site, and convert to 3-D array
	append_site_exceedance()
	all_exceedance_means = np.array(all_exceedance_means, 'd')
	all_exceedance_variances = np.array(all_exceedance_variances, 'd')
	if site_exceedance_percentiles != []:
		all_exceedance_percentiles = np.array(all_exceedance_percentiles)

	## Set in_periods to in_periods_agr for AGR files
	if in_periods_agr != []:
		in_periods = np.array(in_periods_agr)

	## Remove zeros, replace with very small values
	if avoid_zeros:
		if 0.0 in all_exceedance_means:
			try:
				minval = all_exceedance_means[np.where(all_exceedance_means > 0)].min()
			except ValueError:
				minval = 1E-15
			all_exceedance_means = np.maximum(all_exceedance_means, minval)
		if 0.0 in all_exceedance_variances:
			try:
				minval = all_exceedance_variances[np.where(all_exceedance_variances > 0)].min()
			except ValueError:
					minval = 1E-15
			all_exceedance_variances = np.maximum(all_exceedance_variances, minval)
		if 0.0 in all_exceedance_percentiles:
			try:
				minval = all_exceedance_percentiles[np.where(all_exceedance_percentiles > 0)].min()
			except ValueError:
				minval = 1E-15
			all_exceedance_percentiles = np.maximum(all_exceedance_percentiles, minval)

	## Determine IMT if not specified
	if not IMT:
		if len(in_periods) > 1:
			IMT = "SA"
		elif in_periods[0] == 0 or abs(in_periods[0] - zero_period) < 0.001:
			IMT = "PGA"

	## Apply scale factor to intensities, based on intensity_unit
	if convert_to_g and IMT in ("PGA", "SA") and intensity_unit in ("gal", "mg", "ms2"):
		scale_factor = {"mg": 1E-3, "ms2": 1/9.81, "gal": 1./981}[intensity_unit]
		if verbose:
			print("Applying scale factor %s to intensities" % scale_factor)
		all_intensities *= scale_factor
		intensity_unit = "g"

	## Handle site names: if no site names are provided in the function arguments,
	## use those read from file (if any)
	if not site_names:
		site_names = all_site_names
	if len(site_names) == 0:
		site_names = [None] * len(all_sites)

	## Convert to GenericSite objects
	all_sites = [GenericSite(lon, lat, name=name)
				for ((lon, lat), name) in zip(all_sites, site_names)]

	## Create SpectralHazardCurveField(Tree) object
	all_exceedance_means = ExceedanceRateArray(all_exceedance_means)
	if not model_name:
		model_name = os.path.splitext(os.path.basename(filespec))[0]
	if all_exceedance_percentiles == []:
		filespecs = [filespec]*len(in_periods),
		shcf = SpectralHazardCurveField(all_exceedance_means, all_sites, in_periods,
										all_intensities, intensity_unit, IMT,
										model_name=model_name, filespecs=filespecs,
										variances=all_exceedance_variances)
	else:
		num_sites, num_periods, num_intensities = len(all_sites), len(in_periods), all_intensities.shape[-1]
		shape = (num_sites, 1, num_periods, num_intensities)
		mean = all_exceedance_means
		all_exceedance_means = all_exceedance_means.reshape(shape)
		all_exceedance_variances = all_exceedance_variances.reshape(shape)
		shcf = SpectralHazardCurveFieldTree(all_exceedance_means, [model_name], [1.],
											all_sites, in_periods,
											all_intensities, intensity_unit, IMT,
											model_name=model_name, filespecs=[filespec],
											variances=all_exceedance_variances,
											mean=mean, percentile_levels=percentile_levels,
											percentiles=all_exceedance_percentiles)

	## If necessary, interpolate exceedances for other spectral periods
	if out_periods:
		shcf = shcf.interpolate_periods(out_periods)

	f.close()
	return shcf


def read_GRA_multi(filespecs, sites=None, out_periods=[], intensity_unit="",
					convert_to_g=True, IMT="", avoid_zeros=True, model_name="",
					branch_names=[], weights=[], site_names=[]):
	"""
	Read multiple CRISIS .GRA files

	:param filespecs:
		list with full path specification of files to be read
	:param sites:
		list of sites to extract, either site indexes or (lon, lat) tuples
		(default: [] or None, will extract all sites)
	:param out_periods:
		list or array, spectral periods for which exceedance means
		and variances need to be interpolated
		(default: [] or None, output only calculated spectral periods)
	:param intensity_unit:
		str, intensity unit. Determined from corresponding DAT file if
		not specified
		(default: "")
	:param convert_to_g:
		bool indicating whether intensities must be converted to g,
		based on their intensity unit
		(default: True)
	:param IMT:
		str, intensity measure type. Defaults to "PGA" if there is only
		1 spectral period with value 0 or ~1/34, else "SA"
		(default: "")
	:param avoid_zeros:
		bool, if True, replace zero values with very small values:
		1E-13 for exceedance mean, and 1E-18 for exceedance variance
		(default: True)
	:param model_name:
		str, name for this logic-tree model
		(default: "")
	:param branch_names:
		list with name of each branch
			(default: [])
	:param weights:
		list with weight of each branch
		(default:[])
	:param site_names:
		list of site names
		(default: [])

	The return value is a SpectralHazardCurveFieldTree object with the
	following properties:
		sites: list with (lon, lat) tuples of all sites
		periods: array with spectral periods
		intensities: 2-D array [k, l] with fixed intensity values for
			which exceedance rate was calculated. These can be different
			for each spectral period [k]
		exceedance_rates: 4-dimensional [i, j, k, l] array of mean
			exceedance rate
		variances: 4-dimensional [i, j, k, l] array of variance of
			exceedance rate
			i: sites
			j: models or logic-tree branches (i.e. corresponding to different filespecs)
			k: spectral periods
			l: intensities

	Note that this function is not meant to read multiple .AGR files
	If input files have different intensity ranges for one or more
	spectral periods, their exceedances will be interpolated to the
	intensity values of the first input file
	"""
	print("Reading %d files" % len(filespecs))
	in_periods, unit, description = read_DAT(os.path.splitext(filespecs[0])[0])
	if not intensity_unit:
		intensity_unit = unit
	shcf = read_GRA(filespecs[0], sites=sites, in_periods=in_periods,
					intensity_unit=intensity_unit, convert_to_g=convert_to_g,
					IMT=IMT, avoid_zeros=avoid_zeros, verbose=False)

	if not (out_periods in ([], None) or len(out_periods) == 0):
		shcf = shcf.interpolate_periods(out_periods)

	num_models = len(filespecs)
	num_sites = shcf.num_sites
	num_periods = shcf.num_periods
	num_intensities = shcf.num_intensities
	exc_shape = (num_sites, num_models, num_periods, num_intensities)
	all_exceedance_means = np.zeros(exc_shape, 'd')
	all_exceedance_means[:,0,:,:] = shcf.exceedance_rates
	all_exceedance_means = ExceedanceRateArray(all_exceedance_means)
	all_exceedance_variances = np.zeros(exc_shape, 'd')
	all_exceedance_variances[:,0,:,:] = shcf.variances

	if branch_names in (None, []):
		common_path = os.path.commonprefix(filespecs)
		branch_names = [os.path.splitext(filespec[len(common_path):])[0]
						for filespec in filespecs]
	if weights in (None, []):
		weights = np.ones(num_models, 'f') / num_models
	shcft = SpectralHazardCurveFieldTree(all_exceedance_means, branch_names,
										weights, shcf.sites, shcf.periods,
										shcf.intensities, shcf.intensity_unit, shcf.IMT,
										model_name=model_name, filespecs=filespecs,
										timespan=shcf.timespan,
										variances=all_exceedance_variances)

	for j, filespec in enumerate(filespecs[1:]):
		shcf = read_GRA(filespec, sites=sites, in_periods=in_periods,
						intensity_unit=intensity_unit, convert_to_g=convert_to_g,
						IMT=IMT, avoid_zeros=avoid_zeros, verbose=False)

		if not (out_periods in ([], None) or len(out_periods) == 0):
			shcf = shcf.interpolate_periods(out_periods)
		## Check if intensities array is same as in 1st file
		if (shcf.intensities != shcft.intensities).any():
			## Interpolate for intensities used in 1st file
			print("Warning: intensities array in file %d different from "
				"file 0! Will interpolate" % (j + 1))
			if j == 0:
				print(shcft.intensities)
				print
				print(shcf.intensities)
			for k in range(len(out_periods)):
				## We check the 1st and last intensity values of each spectral period
				if ((abs(shcf.intensities[k,0] - shcft.intensities[k,0]) > 1E-6)
					or (abs(shcf.intensities[k,-1] - shcft.intensities[k,-1]) > 1E-6)):
					for i in range(num_sites):
						shcft._hazard_values[i,j+1,k] = interpolate(shcf.intensities[k],
																shcf._hazard_values[i,k],
																shcft.intensities[k])
						shcft.variances[i,j+1,k] = interpolate(shcf.intensities[k],
																shcf.variances[i,k],
																shcft.intensities[k])
		else:
			shcft._hazard_values[:,j+1] = shcf._hazard_values
			shcft.variances[:,j+1] = shcf.variances
		## Overwrite model name
		if shcf.model_name:
			shcft.branch_names[j+1] = shcf.model_name

	print("  done")
	return shcft


def read_MAP(filespec, period_spec=0, intensity_unit="", convert_to_g=True,
			IMT="", model_name="", verbose=False):
	"""
	Read CRISIS .MAP file (CRISIS2007 format) in site mode, i.e.
	only one spectral ordinate is read for each site.

	:param filespec:
		str, full path to file to be read
	:param period_spec:
		period index (integer) or period (float) to extract
		If None, all periods are passed through
		(default: 0)
	:param intensity_unit:
		str, intensity unit. Determined from corresponding DAT file
		if not specified
		(default: "")
	:param convert_to_g:
		bool indicating whether intensities must be converted to g,
		based on their intensity unit
		(default: True)
	:param IMT:
		str, intensity measure type. Defaults to "PGA" if there is only
		1 spectral period with value 0 or ~1/34, else "SA"
		(default: "")
	:param model_name:
		str, model name
		(default: "")
	:param verbose:
		bool, whether or not to print some information
		(default: False)

	:return:
		instance of :class:`HazardMapSet` (if only one period is requested)
		or instance of :class:`UHSFieldSet` (if all periods are requested)
	"""
	if os.path.splitext(filespec)[-1].upper() != ".MAP":
		filespec += ".map"

	## Determine spectral periods
	zero_period = 1./34
	try:
		periods, unit, description = read_DAT(filespec[:-4])
	except IOError:
		print("%s.DAT file not found" % filespec[:-4])
		print("Assuming spectral period is %.4f s (PGA), or reading "
			"from .AGR file if possible" % zero_period)
		if not intensity_unit:
			print("  and intensity unit is 'g'")
			intensity_unit = "g"
		periods = np.array([zero_period], 'd')
	else:
		if not intensity_unit:
			intensity_unit = {True: "g", False: unit}[unit == ""]
		if not model_name:
			model_name = description
	if not IMT:
		IMT = {True: "SA", False: "PGA"}[len(periods) > 1]

	sites = []
	lons, lats = [], []
	intensities = []

	f = open(filespec)
	header_linenr = 10000
	for linenr, line in enumerate(f):
		## Determine return periods
		words = line.split()
		if len(words) > 0:
			if words[0] == "Long.":
				header_linenr = linenr
				words = words[3:8]
				return_periods = [int(float(s)) for s in words]
				return_periods = [rp for rp in return_periods if not rp == 0]
				for rp in return_periods:
					intensities.append([])

			## Determine sites
			if linenr > header_linenr:
				lon, lat, nt, a0, a1, a2, a3, a4 = words[:8]
				site = (float(lon), float(lat))
				if not site in sites:
					sites.append(site)
				else:
					continue
	f.seek(0)

	num_sites, num_periods, num_return_periods = (len(sites), len(periods),
												len(return_periods))
	if verbose:
		print("Found %d sites and %d periods" % (num_sites, num_periods))

	intensities = np.zeros((num_return_periods, num_sites, num_periods), 'd')
	prev_site = None
	site_index = -1
	for linenr, line in enumerate(f):
		if linenr > 15:
			lon, lat, nt, a0, a1, a2, a3, a4 = line.split()[:8]
			site = (lon, lat)
			if site != prev_site:
				site_index += 1
			if site_index == num_sites:
				site_index = 0
			a = np.array([a0, a1, a2, a3, a4][:num_return_periods], 'd')
			period_index = int(nt) - 1
			intensities[:, site_index, period_index] = a
			prev_site = site
	f.close()

	## Apply scale factor to intensities, based on intensity_unit
	if convert_to_g and IMT in ("PGA", "SA") and intensity_unit in ("gal", "mg", "ms2"):
		scale_factor = {"mg": 1E-3, "ms2": 1/9.81, "gal": 1./981}[intensity_unit]
		if verbose:
			print("Applying scale factor %s to intensities" % scale_factor)
		intensities *= scale_factor
		intensity_unit = "g"

	hs = HazardSpectrum(periods)
	if period_spec != None:
		period_index = hs.period_index(period_spec)
		result = HazardMapSet(sites, periods[period_index],
							intensities[:,:,period_index], intensity_unit, IMT,
							model_name=model_name, filespecs=[filespec],
							return_periods=return_periods)
	else:
		result = UHSFieldSet(sites, periods, intensities, intensity_unit, IMT,
							model_name=model_name, filespecs=[filespec],
							return_periods=return_periods)

	return result


def read_DES(filespec, site, intensity=None, return_period=None, period_index=0,
			rebin_magnitudes=[], rebin_distances=[], intensity_unit="g",
			verbose=True):
	"""
	Read CRISIS deaggregation by Magnitude and distance results (.DES) file
	The results are only read for a particular intensity value or for
	the intensity corresponding to a particular return period.
	Use :func:`read_DES_full` to read the complete deaggregation results.

	:param filespec:
		Str, full path to file to be read
	:param site:
		site index (int) or tuple with site coordinates
	:param intensity:
		Float, intensity value for which to read M,r deaggregation results
		(default: None)
	:param return_period:
		Float, return period for which to read M,r deaggregation results
		The corresponding intensity is interpolated from the .GRA file,
		and results are given for the nearest intensity value
		(default: None)
	:param period_index:
		Int, index of spectral period for which to read M, r deaggregation
		results
		(default: 0)
	:param rebin_magnitudes:
		array of magnitudes to rebin M, r deaggregation results
		(default: [], i.e. no rebinning, use magnitude values from the .DES file)
	:param rebin_distances:
		array of distances to rebin M, r deaggregation results
		(default: [], i.e. no rebinning, use distance values from the .DES file)
	:param intensity_unit:
		str, intensity unit
		(default: "g")
	:param verbose:
		bool, whether or not to print some information about what data
		is actually read
		(default: True)

	:return:
		instance of :class:`DeaggregationSlice`

	Note:
		either intensity or return_period must be specified
	"""
	if filespec[-4:].lower() != ".des":
		filespec += ".des"

	if not (intensity or return_period):
		raise Exception("Need to specify either intensity or return period!")

	## Read necessary information from .GRA file
	shcf = read_GRA(os.path.splitext(filespec)[0])
	struc_periods = shcf.periods
	sites = shcf.sites
	intensities = shcf.intensities
	if len(intensities.shape) > 1:
		intensities = intensities[period_index]
	exceedance_means = shcf.exceedance_rates

	## Determine site index
	site_nr = shcf.site_index(site)
	if verbose:
		print("Site nr: %d" % site_nr)

	## Determine intensity index
	if return_period:
		intensity = interpolate(exceedance_means[site_nr,period_index,:],
								intensities, [1.0 / return_period])[0]
		if verbose:
			print("Interpolated intensity: %.3f g" % intensity)
	if intensity:
		intensity_nr = abs(intensities - intensity).argmin()
	if verbose:
		print("Intensity nr: %d (value: %.3f g)"
				% (intensity_nr, intensities[intensity_nr]))

	## Read number of magnitudes and number of distances from header
	f = open(filespec)
	for linenr, line in enumerate(f):
		if "Number of magnitudes" in line:
			MagNum = int(line.split(':')[1].strip())
		if "Number of distances" in line:
			DistNum = int(line.split(':')[1].strip())
			start_nr = linenr + 1
			break
	f.seek(0)

	## Determine line numbers to read
	rec_len = MagNum + 3
	period_len = rec_len * len(intensities)
	site_len = period_len * len(struc_periods) + 2
	site_start = start_nr + site_nr * site_len
	period_start = site_start + period_index * period_len
	rec_start = period_start + intensity_nr * rec_len + 3
	rec_end = rec_start + MagNum + 1
	if verbose:
		print("DES Reading lines %d : %d" % (rec_start, rec_end))

	## Read M,r values for particular site, intensity, and spectral period
	magnitudes = np.zeros(MagNum, 'f')
	values = np.zeros((MagNum-1, DistNum-1), 'd')
	for linenr, line in enumerate(f):
		if rec_start == linenr:
			distances = [float(s) for s in line.split()]
			distances = np.array(distances)
			i = 0
		elif rec_start < linenr < rec_end:
			words = line.split()
			magnitudes[i] = float(words[0])
			if i < MagNum - 1:
				## value for last magnitude is always zero
				for j, word in enumerate(words[2:]):
					## value for first distance is always zero
					values[i,j] = float(word)
			i += 1
		elif linenr > rec_end:
			break
	f.close()

	## Rebin magnitudes and/or distances
	if rebin_magnitudes not in (None, []):
		rebin_values = np.zeros((len(rebin_magnitudes)), DistNum, 'd')
		for d in range(DistNum):
			rebin_values[d] = interpolate(magnitudes, values[:,d], rebin_magnitudes)
			## Renormalize
			total, rebin_total = np.sum(values[:,d]), np.sum(rebin_values[:,d])
			if rebin_total != 0:
				rebin_values[:,d] = rebin_values[:,d] * (total / rebin_total)
		values = rebin_values
		magnitudes = rebin_magnitudes
		MagNum = len(rebin_magnitudes)

	if rebin_distances not in (None, []):
		rebin_values = np.zeros(MagNum, (len(rebin_distances)), 'd')
		for m in range(MagNum):
			rebin_values[:,m] = interpolate(distances, values[m], rebin_distances)
			## Renormalize
			total, rebin_total = np.sum(values[m]), np.sum(rebin_values[m])
			if rebin_total != 0:
				rebin_values[m] = rebin_values[m] * (total / rebin_total)
		values = rebin_values
		distances = rebin_distances

	values = values[:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
	values = ExceedanceRateMatrix(values)
	bin_edges = (magnitudes, distances, np.array([0]), np.array([0]),
				np.array([0]), np.array([0]))
	site = GenericSite(*sites[site_nr])
	period = struc_periods[period_index]
	imt = shcf.IMT
	iml = intensity
	time_span = 50
	return DeaggregationSlice(bin_edges, values, site, iml, intensity_unit, imt,
							period, return_period, time_span)


def read_DES_full(filespec, site=0, rebin_magnitudes=[], rebin_distances=[],
					intensity_unit='g', verbose=True):
	"""
	Read CRISIS deaggregation by Magnitude and distance results (.DES) file
	In contrast to read_DES, the results are read completely for a particular site

	:param filespec:
		full path to file to be read
	:param site:
		site index or tuple with site coordinates
		(default: 0)
	:param rebin_magnitudes:
		array of magnitudes to rebin M, r deaggregation results
		(default: [], i.e. no rebinning, use magnitude values from the .DES file)
	:param rebin_distances:
		array of distances to rebin M, r deaggregation results
		(default: [], i.e. no rebinning, use distance values from the .DES file)
	:param intensity_unit:
		str, intensity unit
		(default: "g")
	:param verbose:
		bool, whether or not to print some information about what data
		is actually read
		(default: True)

	Return value:
		tuple of (magnitudes, distances, deagg_exceedances, intensities)
			magnitudes: array of magnitude values (lower bounds of magnitude bins)
			distances: array of distance values (lower bounds of distance bins)
			deagg_exceedances: 4-D [k,l,r,m] array of M,r- deaggregated exceedance rates
				for a range of spectral periods (k) and intensities (l)
			intensities: 1-D or 2-D array of intensity values
	"""
	if filespec[-4:].lower() != ".des":
		filespec += ".des"

	## Read necessary information from .GRA file
	shcf = read_GRA(os.path.splitext(filespec)[0])
	struc_periods = shcf.periods
	sites = shcf.sites
	intensities = shcf.intensities
	exceedance_means = shcf.exceedance_rates

	num_intensities = intensities.shape[-1]

	## Determine site index
	site_nr = shcf.site_index(site)
	if verbose:
		print("Site nr: %d" % site_nr)

	## Read number of magnitudes and number of distances from header
	f = open(filespec)
	for linenr, line in enumerate(f):
		if "Number of magnitudes" in line:
			MagNum = int(line.split(':')[1].strip())
		if "Number of distances" in line:
			DistNum = int(line.split(':')[1].strip())
			start_nr = linenr + 1
			break
	f.seek(0)

	## Determine line numbers to read
	rec_len = MagNum + 3
	period_len = rec_len * num_intensities
	site_len = period_len * len(struc_periods) + 2
	site_start = start_nr + site_nr * site_len
	site_end = site_start + site_len
	if verbose:
		print("DES Reading lines %d : %d" % (site_start, site_end))

	## Read M,r values for particular site
	magnitudes = np.zeros(MagNum, 'f')
	deagg_exceedances = np.zeros((len(struc_periods), num_intensities, MagNum - 1,
								DistNum - 1) ,'d')

	i = 0
	for linenr, line in enumerate(f):
		if site_start <= linenr < site_end:
			columns = line.split()
			if len(columns) > 2:
				i += 1
				if "INTENSITY" in line:
					i = 0
					period_index, intensity_index = int(columns[1]) - 1, int(columns[3]) - 1
				elif i == 1 and linenr == start_nr + 2 + i:
					distances = [float(s) for s in columns]
					distances = np.array(distances)
				elif i > 1:
					if len(columns):
						Mag_index = i - 2
						if linenr == start_nr + 2 + i:
							magnitudes[Mag_index] = float(columns[0])
						## value for last magnitude is always zero
						if Mag_index < MagNum - 1:
							## value for first distance is always zero
							values = np.array([float(s) for s in columns[2:]], 'd')
							deagg_exceedances[period_index, intensity_index, Mag_index] = values
		elif linenr == site_end:
			break
	f.close()

	## Rebin magnitudes and/or distances
	if rebin_magnitudes not in (None, []):
		rebin_values = np.zeros((len(struc_periods), num_intensities, DistNum,
								len(rebin_magnitudes)) ,'d')
		for k in range(len(struc_periods)):
			for l in range(num_intensities):
				for d in range(DistNum):
					rebin_values[k,l,d] = interpolate(magnitudes,
													deagg_exceedances[k,l,d],
													rebin_magnitudes,
													lib='scipy', kind='cubic')
					## Renormalize
					total, rebin_total = np.sum(deagg_exceedances[k,l,d]), np.sum(rebin_values[k,l,d])
					if rebin_total != 0:
						rebin_values[k,l,d] = rebin_values[k,l,d] * (total / rebin_total)
		deagg_exceedances = rebin_values
		magnitudes = rebin_magnitudes
		MagNum = len(rebin_magnitudes)

	if rebin_distances not in (None, []):
		rebin_values = np.zeros((len(struc_periods), num_intensities,
								len(rebin_distances), MagNum),'d')
		for k in range(len(struc_periods)):
			for l in range(num_intensities):
				for m in range(MagNum):
					rebin_values[k,l,:,m] = interpolate(distances,
														deagg_exceedances[k,l,:,m],
														rebin_distances,
														lib='scipy', kind='cubic')
					## Renormalize
					total, rebin_total = np.sum(deagg_exceedances[k,l,:,m]), np.sum(rebin_values[k,l,:,m])
					if rebin_total != 0:
						rebin_values[k,l,:,m] = (rebin_values[k,l,:,m]
												* (total / rebin_total))
		deagg_exceedances = rebin_values
		distances = rebin_distances

	deagg_exceedances = deagg_exceedances[:,:,:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
	deagg_exceedances = ExceedanceRateMatrix(deagg_exceedances)
	bin_edges = (magnitudes, distances, np.array([0]), np.array([0]), np.array([0]),
				np.array([0]))
	site = GenericSite(*sites[site_nr])
	imt = shcf.IMT
	time_span = 50
	return_periods = np.zeros_like(intensities[0])  ## Dummy return periods
	return SpectralDeaggregationCurve(bin_edges, deagg_exceedances, site,
									intensities, intensity_unit, imt,
									struc_periods, return_periods, timespan=time_span)


# TODO: implement length parameters
def get_crisis_rupture_area_parameters(scale_rel="WC1994", rake=None):
	"""
	Return k1, k2 constants used by CRISIS to relate rupture area with
	magnitude, according to the formula: area = k1 * exp(k2 * M)

	:param scale_rel:
		String, name of scaling relationship, one of "WC1994",
		"Brune1970", "Singh1980" or "PointMSR" (default: "WC1994").
	:param rake:
		Float, defining rake (default: None).

	:return:
		(k1, k2) tuple of floats
	"""
	# TODO: add PeerMSR (see oqhazlib.scalerel.PeerMSR)
	if scale_rel == "WC1994":
		if rake is None:
			return (0.01015, 1.04768)
		elif (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
			return (0.01100, 1.03616)
		elif rake > 0:
			return (0.00517, 1.12827)
		else:
			return (0.02072, 0.94406)
	elif scale_rel == "Brune1970":
		return (0.00381, 1.15130)
	elif scale_rel == "Singh1980":
		return (0.00564, 1.15130)
	elif scale_rel == "PointMSR":
		return (1E-4, 0.)



if __name__ == "__main__":
	"""
	## create hazard curve tree plot
	import hazard.psha.CRISISvsOQ.LogicTree as lt
	root_dir = r"X:\PSHA\CRISISvsOQ\crisis"
	filespecs = lt.slice_logictree(mode="PGA", site="Doel", zone_model=None, att_law=None, attlaw_sigma=None, Mmax_increment=None, seismicity_models=None)
	filespecs = filespecs*2
	shcft = read_GRA_multi(filespecs, intensity_unit="mg")
	shcft.plot(title="Crisis results for Doel (PGA)")
	"""
	hms = read_MAP(r"X:\PSHA\CRISISvsOQ\crisis\Grid\Belgium_Seismotectonic_Akb2010.map", model_name="CRISIS", verbose=True)
	hm = hms.getHazardMap(0)
	hm.plot(amax=0.14)
