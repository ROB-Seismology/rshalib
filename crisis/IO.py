### imports

import os

import numpy as np

from ..result import *
from ..mfd import alphabetalambda, TruncatedGRMFD, CharacteristicMFD
from ..source import PointSource, AreaSource, SimpleFaultSource, ComplexFaultSource


def s2float(s, replace_nan=None, replace_inf=None):
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


def writeCRISIS2007(filespec, source_model, ground_motion_model, gsim_atn_map,
					return_periods, grid_outline, grid_spacing=0.5, sites=[],
					sites_filespec='', imt_periods={"PGA": [0]},
					intensities=None, min_intensities={"PGA": [1E-3]}, max_intensities={"PGA": [1.0]},
					num_intensities=100, imt_unit="g", model_name="",
					truncation_level=3., integration_distance=200.,
					source_discretization=(1.0, 5.0), vs30=800.,
					mag_scale_rel="WC1994",
					output={"gra": True, "map": True, "fue": False, "des": False, "smx": True, "eps": False, "res_full": False},
					deagg_dist_metric="Hypocentral", map_filespec="", cities_filespec="",
					overwrite=False):
	"""
	Write input file for CRISIS2007

	:param filespec:
		String, full path specification for output file
	:param source_model:
		nhlib_nrml.SourceModel object
		Note that geometry of area sources must be specified in counterclockwise
		order.

	:param ground_motion_model:
		nhlib_rob.GroundMotionModel object, mapping tectonic region types to
		gsims (GMPE's) (1 gsim for each trt).
	:param gsim_atn_map:
		Dictionary mapping gsims (in ground_motion_model) to paths of attenuation
		tables (.ATN files). If the tables do not exist, they will be generated
		automatically. Note that tables will be written for each fault type
		("normal", "reverse", and "strike-slip") if the gsim is rake dependent.
	:param return_periods:
		List, tuple or array of max. 5 return periods for computation of hazard maps
	:param grid_outline:
		List of (float, float) tuples or nhlib_rob.PSHASite objects, containing
		(longitude, latitude) of points defining outline of grid where hazard
		will be computed. If only 2 points are given, they define the lower left
		and upper richt corners. If there are more than 2 points, grid_outline
		is considered as a grid reduction polygon, inside which a regular grid
		will be computed. In that case, vertexes must be specified in counter-
		clockwise order (default: []).
	:param grid_spacing:
		Float or tuple of floats, defining grid spacing in degrees (in longitude
		and latitude if tuple) (default: 0.5).
	:param sites:
		List of (float, float) tuples or nhlib_rob.PSHASite objects, defining
		(longitude, latitude) of sites (default: []).
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
		max_intensities and num_intensities are not set. (default: None)
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
		by CRISIS. The maximum is 100 (default: 100). If param intensities is
		given num_intensities is set to 100 to give best possible interpolation.
	:param imt_unit:
		String, intensity measure unit (default: "g")
		Note: it may be necessary to specify this as a dictionary, if different
		IMT's are used with different units.
	:param model_name:
		String, name of model (default: "")
	:param truncation_level":
		Float, truncation level in number of standard deviations of GSIM
	:param integration_distance:
		Float, maximum distance in km for spatial integration (default: 200)
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
		determine the correct soil type for each gsim (default: 800, which
		should correspond to "rock" in most cases).
	:param mag_scale_rel:
		String, name of magnitude-area scaling relationship to be used,
		one of "WC1994", "Brune1970" or "Singh1980" (default: "").
		If empty, the scaling relationships associated with the individual
		source objects will be used.
	:param output:
		Dict with boolean values, indicating which output files to generate
		Keys correspond to output file types:
		"gra": whether or not to output .GRA file (exceedance rates for each
			site) (default: True)
		"map": whether or not to output .MAP file (intensities corresponding to
			fixed return period) (default: True)
		"fue": whether or not to output .FUE file (exceedance rates deaggregated
			by source) (default: True)
		"des": whether or not to output .DES file (exceedance rates deaggregated
			by magnitude and distance) (default: True)
		"smx": whether or not to output .SMX file (maximum earthquakes)
			(default: True)
		"eps": whether or not to output .EPS file (exceedance rates deaggregated
			by epsilon (default: True)
		"res_full": whether or not .RES file should include input data +
			exceedance rates (True) or only input data (False) (default: False)
	:param deagg_dist_metric:
		String, distance metric for deaggregation: "Hypocentral", "Epicentral",
		"Joyner-Boore" or "Rupture" (default: "Hypocentral")
	:param map_filespec:
		String, full path specification for map file to draw as background
		(default: "")
	:param cities_filespec:
		String, full path specification for cities file to draw as background
		(default: "")
	:param overwrite:
		Boolean, whether or not to overwrite existing input files (default: False)
	"""
	from ..gsim import gmpe as att
	from shapely.geometry.polygon import LinearRing

	if intensities:
		num_intensities = 100

	## Generate attenuation tables if they don't exist.
	## Note that we need 3 tables, one for each fault mechanism
	gsims, atn_filespecs = gsim_atn_map.keys(), gsim_atn_map.values()
	#atn_folder = os.path.split(atn_filespecs[0])[0]
	gsims_num_rakes = []
	for gsim in gsims:
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
				## unspecified mechanism in nhlib GMPE's
				mechanism = "normal"
			atn_filespec += ".ATN"
			Mmin, Mmax = source_model.min_mag, source_model.max_mag
			Mstep, num_distances = 0.5, 50
			if not os.path.exists(atn_filespec) or overwrite:
				# TODO: check for gsim names in attenuation
				if gsimObj.distance_metric in ("Hypocentral", "Rupture"):
					## Rhypo, Rrup
					min_distance = source_model.upper_seismogenic_depth
				else:
					## Repi, RJB
					min_distance = 0.1
				# TODO: damping?
				gsimObj.writeCRISIS_ATN(Mmin, Mmax, Mstep, min_distance,
										integration_distance, num_distances,
										h=0, imt_periods=imt_periods,
										imt_unit=imt_unit,
										num_sigma=truncation_level,
										vs30=vs30, mechanism=mechanism,
										damping=5, filespec=atn_filespec)
			else:
				print("Warning: CRISIS ATN file %s exists! Set overwrite=True to overwrite." % atn_filespec)
	gsims_num_rakes = np.array(gsims_num_rakes)
	num_atn_tables = np.add.reduce(gsims_num_rakes)

	## Fill list of return periods with zeros to make a total of 5
	return_periods = list(return_periods)
	return_periods.extend([0] * (5 - len(return_periods)))

	## Write input file
	## Skip if file exists and overwrite is not True
	if os.path.exists(filespec) and not overwrite:
		print("Warning: CRISIS DAT file %s exists! Set overwrite=True to overwrite." % filespec)
		return

	of = open(filespec, "w")
	of.write("[CRISIS2007 format. Be careful when editing this file by hand]\n")

	## Which output files to generate
	if grid_outline and len(grid_outline) > 1:
		input_sites = 0
	elif sites:
		input_sites = 1
	of.write("%d,%d,%d,%d,%d,%d,%d,%d\n" % (output['res_full'], output['gra'], output['map'], output['fue'], output['des'], input_sites, output['smx'], output['eps']))

	## Model name
	if model_name:
		of.write("%s\n" % model_name)
	else:
		of.write("%s zone model - %s attenuation law\n" % (source_model.name, ground_motion_model.name))

	## Construct list of all periods
	all_periods = []
	for imt in imt_periods.keys():
		for T in imt_periods[imt]:
			## CRISIS does not support non-numeric structural periods,
			## so different IMT's may not have duplicate periods!
			if not T in all_periods:
				all_periods.append(T)
			else:
				raise Exception("Duplicate period found: %s (%s s)" % (imt, T))

	## Dimensions
	## Compute total number of sources taking into account nodal plane and hypodepth distributions
	num_sources = 0
	for source in source_model:
		## Index of attenuation table
		gsims_index = gsims.index(ground_motion_model[source.tectonic_region_type])

		## First determine number of rakes (faulting styles) and focal depths
		if isinstance(source, (SimpleFaultSource, ComplexFaultSource)):
			num_rakes = 1
		else:
			num_rakes = min(gsims_num_rakes[gsims_index], source.nodal_plane_distribution.get_num_rakes())

		if isinstance(source, (SimpleFaultSource, ComplexFaultSource)):
			num_depths = 1
		else:
			num_depths = len(source.hypocenter_distribution)

		num_sources += (num_depths * num_rakes)

	of.write("%d,%d,%d,%d\n" % (num_sources, num_atn_tables, len(all_periods), num_intensities))

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
	deagg_dist_metric = att.CRISIS_DistanceMetrics[deagg_dist_metric]
	of.write("%s,%s,%s,%s,%s,%s,%s,%s,%d\n" % ((integration_distance, min_dist_triangle_ratio, min_triangle_size) + tuple(return_periods[:5]) + (deagg_dist_metric,)))

	## Grid or sites where to compute hazard
	if grid_outline and len(grid_outline) > 1:
		if not isinstance(grid_spacing, (tuple, list)):
			grid_spacing = (grid_spacing, grid_spacing)
		lons, lats = zip(*grid_outline)[:2]
		min_lon, max_lon = min(lons), max(lons)
		min_lat, max_lat = min(lats), max(lats)
		## Round min/max lon/lat to multiples of grid_spacing
		min_lon = np.ceil((min_lon / grid_spacing[0])) * grid_spacing[0]
		max_lon = np.floor((max_lon / grid_spacing[0])) * grid_spacing[0]
		min_lat = np.ceil((min_lat / grid_spacing[1])) * grid_spacing[1]
		max_lat = np.floor((max_lat / grid_spacing[1])) * grid_spacing[1]
		grid_origin = (min_lon, min_lat)
		num_grid_lons = ((max_lon - min_lon) / grid_spacing[0]) + 1
		num_grid_lats = ((max_lat - min_lat) / grid_spacing[1]) + 1
		grid_numlines = (num_grid_lons, num_grid_lats)
		of.write("%.3f,%.3f,%.3f,%.3f,%d,%d\n" % (grid_origin + grid_spacing + grid_numlines))
	elif sites:
		if not sites_filespec:
			sites_filespec = os.path.join(os.path.dirname(filespec), 'sites.ASC')
		of.write("%s\n" % sites_filespec)
		if not os.path.exists(sites_filespec):
			writeCRISIS_ASC(sites_filespec, sites)
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
	for gsim, num_rakes in zip(gsims, gsims_num_rakes):
		if num_rakes == 3:
			mechanisms = ["normal", "reverse", "strike-slip"]
		else:
			mechanisms = [""]
		for mechanism in mechanisms:
			atn_filespec = os.path.splitext(gsim_atn_map[gsim])[0]
			if mechanism:
				atn_filespec += "_%s" % mechanism
			atn_filespec += ".ATN"
			of.write("%s\n" % atn_filespec)

	## Source geometry and activity
	for source in source_model:
		## Index of attenuation table
		gsims_index = gsims.index(ground_motion_model[source.tectonic_region_type])

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
				atn_index = np.add.reduce(gsims_num_rakes[:gsims_index])
				atn_index += min(gsims_num_rakes[gsims_index], rake_index)
				if isinstance(source, AreaSource):
					source_type = 0
				elif isinstance(source, SimpleFaultSource):
					## We model fault sources as dipping area sources.
					source_type = 0
				elif isinstance(source, PointSource):
					source_type = 2
				else:
					raise Exception("Complex fault sources are not supported in CRISIS!")
				is_alive = True
				if isinstance(source.mfd, TruncatedGRMFD):
					mfd_type = 1
				elif isinstance(source.mfd, CharacteristicMFD):
					mfd_type = 2
				else:
					raise Exception("CRISIS does not support EvenlyDiscretizedMFD !")
				of.write("%d,%d,%d,%s,%s,%s\n" % (mfd_type, source_type, atn_index, is_alive, k1, k2))

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
						#z = source.hypocenter_distribution.get_mean_depth()
						for point in vertexes:
							of.write("%s,%s,%.1f\n" % (point.longitude, point.latitude, hypo_depth))
					elif isinstance(source, SimpleFaultSource):
						for point in vertexes:
							of.write("%s,%s,%.1f\n" % (point.longitude, point.latitude, point.depth))
				else:
					point = source.location
					of.write(" %d\n" % 1.)
					of.write("%s,%s,%.1f\n" % (point.longitude, point.latitude, hypo_depth))

				## MFD
				#print rake_weight, depth_weight
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
					of.write("%.6f,%.6f,%s,%.2f,%.2f,%.2f,%.2f,%.2f\n" % (lbd, beta, beta_cov, Mmax_expected, Mmax_stdev, Mmax_lower_limit, Mmin, Mmax_upper_limit))
				elif isinstance(mfd, CharacteristicMFD):
					Mmin = mfd.get_min_mag_edge()
					Mmax = mfd.get_magnitude_bin_edges()[-1]
					M_sigma = mfd.M_sigma
					return_period = mfd.return_period
					D = mfd.char_mag
					## time-dependent parameters
					F, T00 = 0, 1
					of.write("%s, %s, %s, %s, %.2f, %.2f, %.2f\n" % (return_period, T00, D, F, M_sigma, Mmin, Mmax))
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


def writeCRISIS_ASC(asc_filespec, sites):
	"""
	Write Crisis asc sites file.

	:param asc_filespec:
		String, defining filespec for asc file.
	:param sites:
		List of (float, float) tuples, defining (longitude, latitude) of sites.
	"""
	f = open(asc_filespec, 'w')
	f.write('%d' % len(sites))
	for site in sites:
		f.write('\n, %s, %s, %s' % (site.name, site.longitude, site.latitude))
	f.close()


def readCRISIS_DAT(filespec):
	"""
	Read structural periods and intensity_unit from CRISIS .DAT file
	Parameters:
		filespec: full path to file to be read
	Return value:
		tuple (period array, intensity_unit)
	"""
	# IDEA: we could also determine number of sites and number of intensities,
	# which would facilitate reading readCRISIS_GRA

	if filespec[-4:].lower() != ".dat":
		filespec += ".dat"

	periods = []
	f = open(filespec)
	for linenr, line in enumerate(f):
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
	return (periods, intensity_unit)


def readCRISIS_GRA(filespec, sites=None, out_periods=[], in_periods=[], intensity_unit="", convert_to_g=True, IMT="", avoid_zeros=True, model_name="", site_names=[], verbose=False):
	"""
	Read CRISIS .GRA or .AGR file (CRISIS2007 format)
	Parameters:
		filespec: full path to file to be read
		sites: list of sites to extract, either site indexes or (lon, lat) tuples
			(default: [] or None, will extract all sites)
		out_periods: structural periods for which exceedance means and variances need to be
			interpolated (default: [] or None, output only calculated structural periods)
		in_periods: structural periods for which exceedance means and variances are reported
			in the GRA file. If not specified, these are determined from the corresponding
			DAT file (default: [])
		intensity_unit: intensity unit. Determined from corresponding DAT file if not
			specified (default: "")
		convert_to_g: boolean indicating whether intensities must be converted to g,
			based on their intensity unit (default: True)
		IMT: intensity measure type. Defaults to "PGA" if there is only 1 structural period
			with value 0 or ~1/34, else "SA" (default: "")
		avoid_zeros: if True, replace zero values with very small values:
			1E-13 for exceedance mean, and 1E-18 for exceedance variance
		model_name: name for this model run (default: "")
		site_names: list of site names. (default: [])
		verbose: boolean indicating whether or not information should be printed during
			parsing (default: False)
	The return value is a SpectralHazardCurveField object with the following properties:
		sites: list with (lon, lat) tuples of all sites
		periods: array with structural periods
		intensities: 2-D array [k, l] with fixed intensity values for which exceedance
			rate was calculated. These can be different for each structural period [k]
		exceedance_rates: 3-dimensional [i, k, l] array of mean exceedance rate
		variances: 3-dimensional [i, k, l] array of variance of exceedance rate
			i: sites
			k: structural periods
			l: intensities
	If AGR file, the return value is a SpectralHazardCurveFieldTree object with the
		following additional properties:
		mean: 3-D array [i, k, l]
		percentile_levels: 1-D array with percentile levels
		percentiles:
			4-dimensional [i, k, l, p] array of percentiles of exceedance rate
				i: sites
				k: structural periods
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
			in_periods, unit = readCRISIS_DAT(filespec[:-4])
		except IOError:
			print("%s.DAT file not found" % filespec[:-4])
			print("Assuming structural period is %.4f s (PGA), or reading from .AGR file if possible" % zero_period)
			if not intensity_unit:
				print("  and intensity unit is 'g'")
				intensity_unit = "g"
			in_periods = np.array([zero_period], 'd')
		else:
			if not intensity_unit:
				intensity_unit = {True: "g", False: unit}[unit == ""]
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
				percentile_levels = np.array(map(float, s.split(',')))
				if percentile_levels.max() < 1.01:
					percentile_levels *= 100
					percentile_levels.astype('i')
		if linenr > 5:
			columns = line.split()
			if len(columns) in (2, 3) or len(columns) >= 8:
				## Start of either a new site or a new structural period
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
						## Append last structural period to previous site
						append_period_exceedance()
						append_site_exceedance()

						if site_nr == 2:
							## Append intensities for last period of previous site 1 if there is more than 1 site
							if verbose:
								print("Appending intensities for period #%d of previous site 1" % period_nr)
							all_intensities.append(np.array(period_intensities, 'd'))

					# Reset site_exceedance, period_exceedance, and period_intensities lists
					site_exceedance_means, site_exceedance_variances, site_exceedance_percentiles = [], [], []
					period_exceedance_means, period_exceedance_variances, period_exceedance_percentiles = [], [], []
					period_intensities = []
					period_nr = 0

				elif columns[0] == "INTENSITY":
					## New structural period
					## GRA or AGR files normally do not include the value of the structural periods,
					## but AGR files created with writeCRISIS_AGR do, so we don't need a RES or other
					## file to figure out what the structural periods are
					try:
						in_periods_agr.append(float(columns[-1].split('=')[1]))
					except:
						pass
					period_nr += 1
					if verbose:
						print("  Period nr: %d" % period_nr)
					if period_exceedance_means:
						## All structural periods, except first and last one
						append_period_exceedance()
						if site_nr == 1:
							#period_intensities = np.array(period_intensities, 'd')
							if verbose:
								print("Appending period intensities for period #%d" % (period_nr-1))
							all_intensities.append(np.array(period_intensities, 'd'))

					# Reset period_exceedance, and period_intensities lists
					period_exceedance_means, period_exceedance_variances, period_exceedance_percentiles = [], [], []
					period_intensities = []

				elif len(columns) == 3:
					## Intensity, Exceedance rate, and variance values
					intensity, exceedance, variance = [s2float(s, replace_nan=1E-18) for s in columns]
					if site_nr == 1:
						period_intensities.append(float(columns[0]))
					period_exceedance_means.append(exceedance)
					period_exceedance_variances.append(variance)
					period_exceedance_percentiles = []
					num_percentiles = 0

				elif len(columns) >= 8:
					## AGR files: intensity, exceedance rate, variance, and percentile values
					# TODO: allow variable number of percentiles
					intensity, exceedance, variance, p5, p16, p50, p84, p95 = [s2float(s, replace_nan=1E-18) for s in columns[:8]]
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

	## Append last structural period
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

	## Create SpectralHazardCurveField(Tree) object
	if not model_name:
		model_name = os.path.splitext(os.path.basename(filespec))[0]
	if all_exceedance_percentiles == []:
		shcf = SpectralHazardCurveField(model_name, [filespec]*len(in_periods), all_sites, in_periods, IMT, all_intensities, intensity_unit, exceedance_rates=all_exceedance_means, variances=all_exceedance_variances, site_names=site_names)
	else:
		num_sites, num_periods, num_intensities = len(all_sites), len(in_periods), all_intensities.shape[-1]
		shape = (num_sites, 1, num_periods, num_intensities)
		mean = all_exceedance_means
		all_exceedance_means = all_exceedance_means.reshape(shape)
		all_exceedance_variances = all_exceedance_variances.reshape(shape)
		shcf = SpectralHazardCurveFieldTree(model_name, [model_name], [filespec], [1.], all_sites, in_periods, IMT, all_intensities, intensity_unit, exceedance_rates=all_exceedance_means, variances=all_exceedance_variances, mean=mean, percentile_levels=percentile_levels, percentiles=all_exceedance_percentiles, site_names=site_names)

	## If necessary, interpolate exceedances for other spectral periods
	if out_periods:
		shcf = shcf.interpolate_periods(out_periods)

	f.close()
	return shcf


def readCRISIS_GRA_multi(filespecs, sites=None, out_periods=[], intensity_unit="", convert_to_g=True, IMT="", avoid_zeros=True, model_name="", branch_names=[], weights=[], site_names=[]):
	"""
	Read multiple CRISIS .GRA files
	Parameters:
		filespecs: list with full path specification of files to be read
		sites: list of sites to extract, either site indexes or (lon, lat) tuples
			(default: [] or None, will extract all sites)
		out_periods: structural periods for which exceedance means and variances need to be
			interpolated (default: [] or None, output only calculated structural periods)
		intensity_unit: intensity unit. Determined from corresponding DAT file if not
			specified (default: "")
		convert_to_g: boolean indicating whether intensities must be converted to g,
			based on their intensity unit (default: True)
		IMT: intensity measure type. Defaults to "PGA" if there is only 1 structural period
			with value 0 or ~1/34, else "PGA" (default: "")
		avoid_zeros: if True, replace zero values with very small values:
			1E-13 for exceedance mean, and 1E-18 for exceedance variance
		model_name: name for this logic-tree model (default: "")
		branch_names: list with name of each branch (default: [])
		weights: list with weight of each branch (default:[])
		site_names: list of site names (default: [])
	The return value is a SpectralHazardCurveFieldTree object with the following properties:
		sites: list with (lon, lat) tuples of all sites
		periods: array with structural periods
		intensities: 2-D array [k, l] with fixed intensity values for which exceedance
			rate was calculated. These can be different for each structural period [k]
		exceedance_rates: 4-dimensional [i, j, k, l] array of mean exceedance rate
		variances: 4-dimensional [i, j, k, l] array of variance of exceedance rate
			i: sites
			j: models or logic-tree branches (i.e. corresponding to different filespecs)
			k: structural periods
			l: intensities
	Note that this function is not meant to read multiple .AGR files
	If input files have different intensity ranges for one or more structural periods,
	their exceedances will be interpolated to the intensity values of the first input file
	"""
	print("Reading %d files" % len(filespecs))
	in_periods, unit = readCRISIS_DAT(os.path.splitext(filespecs[0])[0])
	if not intensity_unit:
		intensity_unit = unit
	shcf = readCRISIS_GRA(filespecs[0], sites=sites, in_periods=in_periods, intensity_unit=intensity_unit, convert_to_g=convert_to_g, IMT=IMT, avoid_zeros=avoid_zeros, verbose=False)

	if not (out_periods in ([], None) or len(out_periods) == 0):
		shcf = shcf.interpolate_periods(out_periods)

	num_models = len(filespecs)
	num_sites = shcf.num_sites
	num_periods = shcf.num_periods
	num_intensities = shcf.num_intensities
	all_exceedance_means = np.zeros((num_sites, num_models, num_periods, num_intensities), 'd')
	all_exceedance_means[:,0,:,:] = shcf.exceedance_rates
	all_exceedance_variances = np.zeros((num_sites, num_models, num_periods, num_intensities), 'd')
	all_exceedance_variances[:,0,:,:] = shcf.variances

	if branch_names in (None, []):
		common_path = os.path.commonprefix(filespecs)
		branch_names = [os.path.splitext(filespec[len(common_path):])[0] for filespec in filespecs]
	if weights in (None, []):
		weights = np.ones(num_models, 'f') / num_models
	shcft = SpectralHazardCurveFieldTree(model_name, branch_names, filespecs, weights, shcf.sites, shcf.periods, shcf.IMT, shcf.intensities, shcf.intensity_unit, shcf.timespan,exceedance_rates=all_exceedance_means, variances=all_exceedance_variances, site_names=shcf.site_names)

	for j, filespec in enumerate(filespecs[1:]):
		shcf = readCRISIS_GRA(filespec, sites=sites, in_periods=in_periods, intensity_unit=intensity_unit, convert_to_g=convert_to_g, IMT=IMT, avoid_zeros=avoid_zeros, verbose=False)

		if not (out_periods in ([], None) or len(out_periods) == 0):
			shcf = shcf.interpolate_periods(out_periods)
		## Check if intensities array is same as in 1st file
		if (shcf.intensities != shcft.intensities).any():
			## Interpolate for intensities used in 1st file
			print("Warning: intensities array in file %d different from file 0! Will interpolate" % (j + 1))
			if j == 0:
				print shcft.intensities
				print
				print shcf.intensities
			for k in range(len(out_periods)):
				## We check the 1st and last intensity values of each structural period
				if (abs(shcf.intensities[k,0] - shcft.intensities[k,0]) > 1E-6) or (abs(shcf.intensities[k,-1] - shcft.intensities[k,-1]) > 1E-6):
					for i in range(num_sites):
						shcft._exceedance_rates[i,j+1,k] = interpolate(shcf.intensities[k], shcf.exceedance_rates[i,k], shcft.intensities[k])
						shcft.variances[i,j+1,k] = interpolate(shcf.intensities[k], shcf.variances[i,k], shcft.intensities[k])
		else:
			shcft._exceedance_rates[:,j+1] = shcf.exceedance_rates
			shcft.variances[:,j+1] = shcf.variances

	print("  done")
	return shcft


def readCRISIS_MAP(filespec, period_spec=0, intensity_unit="", convert_to_g=True, IMT="", model_name="", verbose=False):
	"""
	Read CRISIS .MAP file (CRISIS2007 format) in site mode, i.e. only one spectral ordinate is read
	for each site.
	Parameters:
		filespec: full path to file to be read
		period_spec: period index (integer) or period (float) to extract
			(default: 0). If None, all periods are passed through
		intensity_unit: intensity unit. Determined from corresponding DAT file if not
			specified (default: "")
		convert_to_g: boolean indicating whether intensities must be converted to g,
			based on their intensity unit (default: True)
		IMT: intensity measure type. Defaults to "PGA" if there is only 1 structural period
			with value 0 or ~1/34, else "PGA" (default: "")
		spectral_ordinate: number of the spectral ordinate (sequential, starting from 1) that is requested
			(default: 1)
	Return value:
		HazardMapSet object (if only one period is requested), or
		UHSFieldSet object (if all periods are requested)
	"""
	if os.path.splitext(filespec)[-1].upper() != ".MAP":
		filespec += ".map"

	## Determine spectral periods
	zero_period = 1./34
	try:
		periods, unit = readCRISIS_DAT(filespec[:-4])
	except IOError:
		print("%s.DAT file not found" % filespec[:-4])
		print("Assuming structural period is %.4f s (PGA), or reading from .AGR file if possible" % zero_period)
		if not intensity_unit:
			print("  and intensity unit is 'g'")
			intensity_unit = "g"
		periods = np.array([zero_period], 'd')
	else:
		if not intensity_unit:
			intensity_unit = {True: "g", False: unit}[unit == ""]
	if not IMT:
		IMT = {True: "SA", False: "PGA"}[len(periods) > 1]

	sites = []
	lons, lats = [], []
	intensities = []

	f = open(filespec)
	for linenr, line in enumerate(f):
		## Determine return periods
		if linenr == 15:
			words = line.split()[3:8]
			return_periods = [int(float(s)) for s in words]
			return_periods = [rp for rp in return_periods if not rp == 0]
			for rp in return_periods:
				intensities.append([])

		## Determine sites
		if linenr > 15:
			lon, lat, nt, a0, a1, a2, a3, a4 = line.split()[:8]
			site = (float(lon), float(lat))
			if not site in sites:
				sites.append(site)
			else:
				continue
	f.seek(0)

	num_sites, num_periods, num_return_periods = len(sites), len(periods), len(return_periods)
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
		result = HazardMapSet(model_name, [filespec], sites, periods[period_index], IMT, intensities[:,:,period_index], intensity_unit=intensity_unit, return_periods=return_periods)
	else:
		result = UHSFieldSet(model_name, [filespec], sites, periods, IMT, intensities, intensity_unit=intensity_unit, return_periods=return_periods)
	return result


# TODO: implement length parameters
def get_crisis_rupture_area_parameters(scale_rel="WC1994", rake=None):
	"""
	Return constants used by CRISIS to relate rupture area with magnitude.

	:param scale_rel:
		String, name of scaling relationship, one of "WC1994",
		"Brune1970" or "Singh1980" (default: "WC1994").
	:param rake:
		Float, defining rake (default: None).
	"""
	# TODO: add PeerMSR (see nhlib.scalerel.PeerMSR)
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



if __name__ == "__main__":
	"""
	## create hazard curve tree plot
	import hazard.psha.CRISISvsOQ.LogicTree as lt
	root_dir = r"X:\PSHA\CRISISvsOQ\crisis"
	filespecs = lt.slice_logictree(mode="PGA", site="Doel", zone_model=None, att_law=None, attlaw_sigma=None, Mmax_increment=None, seismicity_models=None)
	filespecs = filespecs*2
	shcft = readCRISIS_GRA_multi(filespecs, intensity_unit="mg")
	shcft.plot(title="Crisis results for Doel (PGA)")
	"""
	hms = readCRISIS_MAP(r"X:\PSHA\CRISISvsOQ\crisis\Grid\Belgium_Seismotectonic_Akb2010.map", model_name="CRISIS", verbose=True)
	hm = hms.getHazardMap(0)
	hm.plot(amax=0.14)
