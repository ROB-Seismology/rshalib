"""
Functions related to multiprocessing
"""

import os
import multiprocessing
import numpy as np
import openquake.hazardlib as oqhazlib


# TODO: use logging facilities instead of print statements


def run_parallel(func, job_arg_list, num_processes, shared_arr=None, verbose=True):
	"""
	Generic function to run multiple jobs in parallel with a given
	function.

	:param func:
		function object
		Some restrictions apply:
		- the function must be importable, so class methods or functions
			created in another function or method cannot be used.
	:param job_arg_list:
		list of tuples with function arguments for each job
		Some restrictions apply:
		- objects passed to multiprocessing should not override the
			__getattr__ method
		- objects inheriting from classes with _slots_ loose any
			new attributes set in the inherited class
		- IMT objects are mangled up (don't know why)
		- ...
	:param num_processes:
		int, number of parallel processes. Actual number of processes
			may be lower depending on available cores
	:param verbose:
		bool, whether or not to print some progress information

	:return:
		list with values returned by func for each job in order
	"""
	num_processes = min(multiprocessing.cpu_count(), num_processes)
	num_processes = min(num_processes, len(job_arg_list))
	if verbose:
		print("Starting %d parallel processes" % num_processes)
	if shared_arr is None:
		pool = multiprocessing.Pool(processes=num_processes)
	else:
		pool = multiprocessing.Pool(processes=num_processes, initializer=init_shared_arr, initargs=(shared_arr,))
	result = pool.map(func, job_arg_list)
	return result


def init_shared_arr(shared_arr_):
	"""
	Make shared array available
	"""
	global shared_arr
	shared_arr = shared_arr_ # must be inhereted, not passed as an argument


def calc_shcf_by_source((psha_model, source, cav_min, verbose)):
	"""
	Stand-alone function that will compute hazard curves for a single
	source.
	Note:
		- all necessary parameters have to provided, so as not to depend
		  on variables defined elsewhere
		- all parameters are combined in a single tuple to facilitate
		  passing of arguments to pool.map()

	:param psha_model:
		instance of :class:`PSHAModel`
	:param source:
		instance of :class:`PointSource`, :class:`AreaSource`,
		:class:`SimpleFaultSource` or :class:`ComplexFaultSource`
	:param cav_min:
		float, CAV threshold in g.s
	:param verbose:
		bool, whether or not to print some progress information

	:return:
		3-D numpy array [i,k,l] with probabilities of non-exceedance
	"""
	if verbose:
		print source.source_id
	sources = [source]
	sites = psha_model.get_soil_site_model()
	gsims = psha_model._get_trt_gsim_dict()
	imts = psha_model._get_imtls()
	tom = psha_model.poisson_tom

	total_sites = len(sites)
	shape = (total_sites, len(imts), len(imts[imts.keys()[0]]))
	curves = np.ones(shape)

	## Copied from openquake.hazardlib
	sources_sites = ((source, sites) for source in sources)
	for source, s_sites in psha_model.source_site_filter(sources_sites):
		try:
			ruptures_sites = ((rupture, s_sites)
							  for rupture in source.iter_ruptures(tom))
			for rupture, r_sites in psha_model.rupture_site_filter(ruptures_sites):
				prob = rupture.get_probability_one_or_more_occurrences()
				gsim = gsims[rupture.tectonic_region_type]
				sctx, rctx, dctx = gsim.make_contexts(r_sites, rupture)
				for k, imt in enumerate(imts):
					poes = gsim.get_poes_cav(sctx, rctx, dctx, imt, imts[imt],
										 psha_model.truncation_level, cav_min=cav_min)
					curves[:,k,:] *= r_sites.expand(
						(1 - prob) ** poes, total_sites, placeholder=1
					)
		except Exception, err:
			msg = 'An error occurred with source id=%s. Error: %s'
			msg %= (source.source_id, err.message)
			raise RuntimeError(msg)
	return curves


def calc_shcf_psha_model((psha_model, sample_idx, cav_min, combine_pga_and_sa, verbose)):
	"""
	Stand-alone function that will compute hazard curves for a single
	logic-tree sample.
	Note:
		- all necessary parameters have to provided, so as not to depend
		  on variables defined elsewhere
		- all parameters are combined in a single tuple to facilitate
		  passing of arguments to pool.map()

	:param psha_model:
		instace of :class:`PSHAModel`
	:param sample_idx:
		str, sample index (formatted with adequate number of leading zeros,
		and starting from 1)
	:param cav_min:
		float, CAV threshold in g.s
	:param combine_pga_and_sa:
		bool, whether or not to combine PGA and SA, if present
	:param verbose:
		Bool, whether or not to print some progress information

	:return:
		0 (successful execution) or 1 (error occurred)
	"""
	## Run
	if verbose:
		print("Starting hazard-curve calculation of sample %s..." % sample_idx)

	try:
		im_shcf_dict = psha_model.calc_shcf(cav_min=cav_min, combine_pga_and_sa=combine_pga_and_sa)
	except Exception, err:
		msg = 'Warning: An error occurred with sample %s. Error: %s'
		msg %= (sample_idx, err.message)
		#raise RuntimeError(msg)
		print(msg)
		return 1
	else:
		## Write XML file, creating directory if necessary
		# TODO: replace with psha_model.write_oq_shcf()
		hc_folder = psha_model.oq_root_folder
		subfolders = ["computed_output", "classical", "calc_oqhazlib"]
		for im, shcf in im_shcf_dict.items():
			if im == "SA":
				subfolders.append("hazard_curve_multi")
				xml_filename = "hazard_curve_multi-rlz-%s.xml" % (sample_idx)
			else:
				subfolders.extend(["hazard_curve", im])
				xml_filename = "hazard_curve-rlz-%s.xml" % (sample_idx)
			for subfolder in subfolders:
				hc_folder = os.path.join(hc_folder, subfolder)
				if not os.path.exists(hc_folder):
					os.mkdir(hc_folder)
			xml_filespec = os.path.join(hc_folder, xml_filename)
			shcf.write_nrml(xml_filespec)

		return 0


def deaggregate_by_source((psha_model, source, src_idx, deagg_matrix_shape, site_imtls, deagg_site_model, mag_bins, dist_bins, eps_bins, lon_bins, lat_bins, dtype, verbose)):
	"""
	Stand-alone function that will deaggregate for a single source.

	:param psha_model:
		instance of :class:`PSHAModel`
	:param source:
		instance of :class:`PointSource`, :class:`AreaSource`,
		:class:`SimpleFaultSource` or :class:`ComplexFaultSource`
	:param src_idx:
		int, source index in 8-D deaggregation matrix
	:param shared_deagg_matrix_shape:
		tuple describing shape of full deaggregation matrix in shared
		memory
	:param site_imtls:
		nested dictionary mapping (lon, lat) tuples to dictionaries
		mapping oqhazlib IMT objects to 1-D arrays of intensity measure
		levels
	:param deagg_site_model:
		list with instances of :class:`SHASite` or instance of
		:class:`SHASiteModel`for which deaggregation will be performed.
		Note that instances of class:`SoilSite` will
		not work with multiprocessing
	:param deagg_imt_periods:
		dictionary mapping instances of :class:`IMT` to lists of spectral
		periods.
	:param mag_bins:
		list or array with magnitude bin edges.
	:param dist_bins:
		list or array with distance bin edges
	:param eps_bins:
		list or array with epsilon bin edges
	:param lon_bins:
		list or array with longitude bin edges
	:param lat_bins:
		list or array with latitude bin edges
	:param dtype:
		str, precision of deaggregation matrix
	:param verbose:
		Bool, whether or not to print some progress information

	:return:
		0 (successful execution) or 1 (error occurred)
	"""
	# TODO: make param deagg_site_model in agreement with deaggregate_psha_model
	import openquake.hazardlib as oqhazlib
	from openquake.hazardlib.site import SiteCollection

	try:
		## Initialize deaggregation matrix with ones representing NON-exceedance probabilities !
		src_deagg_matrix = np.ones(deagg_matrix_shape[:-1], dtype=dtype)

		## Perform deaggregation
		tom = psha_model.poisson_tom
		gsims = psha_model._get_trt_gsim_dict()
		source_site_filter = psha_model.source_site_filter
		rupture_site_filter = psha_model.rupture_site_filter

		n_epsilons = len(eps_bins) - 1

		sources = [source]
		sources_sites = ((source, deagg_site_model) for source in sources)
		for src_idx, (source, s_sites) in \
				enumerate(source_site_filter(sources_sites)):

			if verbose:
				print source.source_id

			tect_reg = source.tectonic_region_type
			gsim = gsims[tect_reg]

			ruptures_sites = ((rupture, s_sites)
							  for rupture in source.iter_ruptures(tom))
			for rupture, r_sites in rupture_site_filter(ruptures_sites):
				## Extract rupture parameters of interest
				mag_idx = np.digitize([rupture.mag], mag_bins)[0] - 1

				sitemesh = r_sites.mesh
				sctx, rctx, dctx = gsim.make_contexts(r_sites, rupture)
				if hasattr(dctx, "rjb"):
					jb_dists = getattr(dctx, "rjb")
				else:
					jb_dists = rupture.surface.get_joyner_boore_distance(sitemesh)
				closest_points = rupture.surface.get_closest_points(sitemesh)
				lons = [pt.longitude for pt in closest_points]
				lats = [pt.latitude for pt in closest_points]

				dist_idxs = np.digitize(jb_dists, dist_bins) - 1
				lon_idxs = np.digitize(lons, lon_bins) - 1
				lat_idxs = np.digitize(lats, lat_bins) - 1

				## Compute probability of one or more rupture occurrences
				prob_one_or_more = rupture.get_probability_one_or_more_occurrences()

				## compute conditional probability of exceeding iml given
				## the current rupture, and different epsilon level, that is
				## ``P(IMT >= iml | rup, epsilon_bin)`` for each of epsilon bins
				for site_idx, site in enumerate(r_sites):
					dist_idx = dist_idxs[site_idx]
					lon_idx = lon_idxs[site_idx]
					lat_idx = lat_idxs[site_idx]
					site_key = (site.location.longitude, site.location.latitude)
					imtls = site_imtls[site_key]
					imts = imtls.keys()
					sctx2, rctx2, dctx2 = gsim.make_contexts(SiteCollection([site]), rupture)
					for imt_idx, imt_tuple in enumerate(imts):
						imls = imtls[imt_tuple]
						## Reconstruct imt from tuple
						imt = getattr(oqhazlib.imt, imt_tuple[0])(imt_tuple[1], imt_tuple[2])
						## In contrast to what is stated in the documentation,
						## disaggregate_poe does handle more than one iml
						poes_given_rup_eps = gsim.disaggregate_poe(
							sctx2, rctx2, dctx2, imt, imls, psha_model.truncation_level, n_epsilons
						)

						## Probability of non-exceedance
						pone = (1. - prob_one_or_more) ** poes_given_rup_eps

						try:
							src_deagg_matrix[site_idx, imt_idx, :, mag_idx, dist_idx, lon_idx, lat_idx, :] *= pone
						except IndexError:
							## May fail if rupture extent is beyond (lon,lat) range
							pass

		## Update shared matrix
		with shared_arr.get_lock(): # synchronize access
			for site_idx, site_key in enumerate(sorted(site_imtls.keys())):
				shared_deagg_matrix = np.frombuffer(shared_arr.get_obj()) # no data copying
				shared_deagg_matrix = shared_deagg_matrix.reshape(deagg_matrix_shape)
				shared_deagg_matrix[site_idx,:,:,:,:,:,:,:,src_idx] *= src_deagg_matrix[site_idx]

	except Exception, err:
		msg = 'Warning: An error occurred with source %s. Error: %s'
		msg %= (source.source_id, err.message)
		#raise RuntimeError(msg)
		print(msg)
		return 1
	else:
		#return deagg_matrix_dict
		return 0


def deaggregate_psha_model((psha_model, sample_idx, deagg_sites, deagg_imt_periods, mag_bin_width, distance_bin_width, num_epsilon_bins, coordinate_bin_width, dtype, verbose)):
	"""
	Stand-alone function that will deaggregate a single logic-tree sample.
	Intensity measure levels corresponding to psha_model.return_periods
	will be interpolated first, so the hazard curves must have been
	computed before.
	Note:
		- all necessary parameters have to provided, so as not to depend
		  on variables defined elsewhere
		- all parameters are combined in a single tuple to facilitate
		  passing of arguments to pool.map()

	:param psha_model:
		instace of :class:`PSHAModel`
	:param sample_idx:
		str, sample index (formatted with adequate number of leading zeros,
		and starting from 1)
	:param deagg_sites:
		list with instances of :class:`SHASite` for which deaggregation
		will be performed. Note that instances of class:`SoilSite` will
		not work with multiprocessing
	:param deagg_imt_periods:
		dictionary mapping instances of :class:`IMT` to lists of spectral
		periods.
	:param mag_bin_width:
		Float, magnitude bin width (None will take MFD bin width
		of first source)
	:param dist_bin_width:
		Float, distance bin width in km
	:param n_epsilons:
		Int, number of epsilon bins (None will result in bins
		corresponding to integer epsilon values)
	:param coord_bin_width:
		Float, lon/lat bin width in decimal degrees
	:param dtype:
		str, precision of deaggregation matrix
	:param verbose:
		Bool, whether or not to print some progress information

	:return:
		0 (successful execution) or 1 (error occurred)

	The deaggregation results will be written to XML files in the output
	folder of the PSHA model.
	"""
	# TODO: perhaps we need to add calc_id parameter
	if verbose:
		print psha_model.name

	## Determine intensity levels from saved hazard curves
	curve_name = "rlz-%s" % sample_idx
	site_imtls = psha_model._interpolate_oq_site_imtls(curve_name, deagg_sites,
												deagg_imt_periods, calc_id=None)

	## Deaggregation
	if verbose:
		print("Starting deaggregation of sample %s..." % sample_idx)

	try:
		spectral_deagg_curve_dict = psha_model.deaggregate(site_imtls, mag_bin_width, distance_bin_width, num_epsilon_bins, coordinate_bin_width, dtype, verbose=False)
	except Exception, err:
		msg = 'Warning: An error occurred with sample %s. Error: %s'
		msg %= (sample_idx, err.message)
		#raise RuntimeError(msg)
		print(msg)
		return 1
	else:
		## Write XML file(s), creating directory if necessary
		for (lon, lat) in spectral_deagg_curve_dict.keys():
			sdc = spectral_deagg_curve_dict[(lon, lat)]
			psha_model.read_oq_disagg_matrix_multi(sdc, calc_id="oqhazlib")

		## Don't return deaggregation results to preserve memory
		return 0


def square(x):
	"""
	Example function
	"""
	return x * x

def np_array(x):
	"""
	Example function
	"""
	return np.array(range(x))

def do_nothing(sample_idx, psha_model):
	"""
	Dummy function for testing multiprocessing.
	"""
	print("Doing nothing #%s" % (sample_idx,))



if __name__ == "__main__":
	print run_parallel(np_array, range(10), 3)
