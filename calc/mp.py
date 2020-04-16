"""
Functions related to multiprocessing
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys
import multiprocessing
from functools import partial
import traceback

import numpy as np

from .. import (oqhazlib, OQ_VERSION)


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
		- class methods and ad-hoc functions
		- ...
	:param num_processes:
		int, number of parallel processes. Actual number of processes
			may be lower depending on available cores
	:param shared_arr:
		array that will be shared between different processes
		(default: None)
	:param verbose:
		bool, whether or not to print some progress information

	:return:
		list with values returned by func for each job in order
	"""
	from functools import partial

	# TODO: logging with mp.get_logger().info(), mp.get_logger().error(), ...
	num_processes = min(multiprocessing.cpu_count(), num_processes)
	num_processes = min(num_processes, len(job_arg_list))
	if verbose:
		print("Starting %d parallel processes" % num_processes)
	if shared_arr is None:
		pool = multiprocessing.Pool(processes=num_processes)
	else:
		pool = multiprocessing.Pool(processes=num_processes, initializer=init_shared_arr,
									initargs=(shared_arr,))
	## In PY3.3, we can use pool.starmap
	if sys.version_info[:2] >= (3, 3):
		result = pool.starmap(func, job_arg_list, chunksize=1)
	else:
		job_arg_list = [(func, job_args) for job_args in job_arg_list]
		result = pool.map(mp_func_wrapper, job_arg_list, chunksize=1)
		## functools.partial instead of mp_func_wrapper doesn't work for multiple args
		#result = pool.map(partial(func), job_arg_list, chunksize=1)
	pool.close()
	return result


def mp_func_wrapper(func_args_tuple):
	"""
	Wrapper function to be used with multiprocessing pool.map
	This function is needed when multiple arguments need to be passed
	to the function used in a multiprocessing pool.

	:param func_args_tuple:
		(func, args) tuple:
		- func: function object
		- args: tuple or list with arguments for :param:`func`

	:return:
		return value of :param:`func`
	"""
	func, args = func_args_tuple
	return func(*args)


def init_shared_arr(shared_arr_):
	"""
	Make shared array available
	"""
	global shared_arr
	shared_arr = shared_arr_ # must be inherited, not passed as an argument


def calc_shcf_by_source(psha_model, source, cav_min, verbose):
	"""
	Stand-alone function that will compute hazard curves for a single
	source.
	Note:
		- all necessary parameters have to provided, so as not to depend
		  on variables defined elsewhere

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
	from ..gsim import make_gsim_contexts

	if verbose:
		print(source.source_id)

	sources = [source]
	sites = psha_model.get_soil_site_model()
	gsims = psha_model._get_trt_gsim_dict()
	imts = psha_model._get_imtls()

	total_sites = len(sites)
	shape = (total_sites, len(imts), len(imts[list(imts.keys())[0]]))
	curves = np.ones(shape, dtype=np.float64)

	## Copied from openquake.hazardlib
	if OQ_VERSION >= '2.9.0':
		source_site_filter = psha_model.source_site_filter(sites)([source])
	else:
		sources_sites = ((source, sites) for source in sources)
		source_site_filter = psha_model.source_site_filter(sources_sites)
	for _, s_sites in source_site_filter:
		if OQ_VERSION >= '2.9.0':
			tom = None
		else:
			tom = psha_model.poisson_tom
		try:
			for rupture in source.iter_ruptures(tom):
				poe_rup = rupture.get_probability_one_or_more_occurrences()

				if OQ_VERSION >= '2.9.0':
					r_sites = psha_model.rupture_site_filter(rupture, sites=s_sites)
					#r_sites = s_sites
				else:
					try:
						[(_, r_sites)] = psha_model.rupture_site_filter([(rupture,
																		s_sites)])
					except:
						r_sites = None
				if r_sites is None:
					continue

				gsim = gsims[rupture.tectonic_region_type]
				try:
					sctx, rctx, dctx = make_gsim_contexts(gsim, r_sites, rupture,
										max_distance=psha_model.integration_distance)
				except:
					## Rupture probably too far
					continue

				if cav_min > 0 and not hasattr(sctx, "vs30"):
					## Set vs30 explicitly for GMPEs that do not require vs30
					setattr(sctx, "vs30", getattr(r_sites, "vs30"))

				for k, imt in enumerate(imts):
					if OQ_VERSION >= '2.9.0':
						poes = gsim.get_poes(sctx, rctx, dctx, imt, imts[imt],
										 psha_model.truncation_level)
					else:
						poes = gsim.get_poes_cav(sctx, rctx, dctx, imt, imts[imt],
										 psha_model.truncation_level, cav_min=cav_min)

					exceedances = (1 - poe_rup) ** poes

					if OQ_VERSION >= '2.9.0':
						if len(r_sites) == total_sites:
							site_indices = None
						else:
							site_indices = r_sites.indices
						if site_indices is not None:
							curves[site_indices,k,:] *= exceedances
						else:
							curves[:,k,:] *= exceedances
					else:
						curves[:,k,:] *= r_sites.expand(exceedances, total_sites,
														placeholder=1)

		except Exception as err:
			msg = 'Warning: An error occurred with source %s:\n%s'
			msg %= (source.source_id, traceback.format_exc())
			raise RuntimeError(msg)
			#print(msg)
			#return 1
		else:
			return 0

	return curves


def calc_shcf_psha_model(psha_model, curve_name, curve_path, cav_min,
						combine_pga_and_sa, calc_id, verbose):
	"""
	Stand-alone function that will compute hazard curves for a single
	PSHA model, e.g., a logic-tree sample.
	Note:
		- all necessary parameters have to provided, so as not to depend
		  on variables defined elsewhere

	:param psha_model:
		instace of :class:`PSHAModel`
	:param curve_name:
		str, identifying hazard curve (e.g., "rlz-01", "mean", "quantile_0.84")
	:param curve_path:
		str, path to hazard curve relative to main hazard-curve folder
	:param cav_min:
		float, CAV threshold in g.s
	:param combine_pga_and_sa:
		bool, whether or not to combine PGA and SA, if present
	:param calc_id:
		int or str, OpenQuake calculation ID
	:param verbose:
		Bool, whether or not to print some progress information

	:return:
		0 (successful execution) or 1 (error occurred)
	"""
	## Run
	if verbose:
		print("Starting hazard-curve calculation of curve %s..." % curve_name)

	try:
		im_shcf_dict = psha_model.calc_shcf(cav_min=cav_min,
										combine_pga_and_sa=combine_pga_and_sa)
	except Exception as err:
		msg = 'Warning: An error occurred with curve %s. Error: %s'
		msg %= (curve_name, err)
		#raise RuntimeError(msg)
		print(msg)
		return 1
	else:
		## Write XML file, creating directory if necessary
		for im, shcf in im_shcf_dict.items():
			if im == "SA":
				psha_model.write_oq_shcf(shcf, curve_name, curve_path=curve_path,
										calc_id=calc_id)
			else:
				hcf = shcf.get_hazard_curve_field(period_spec=0)
				psha_model.write_oq_hcf(hcf, curve_name, curve_path=curve_path,
										calc_id=calc_id)

		return 0


def deaggregate_by_source(psha_model, source, src_idx, deagg_matrix_shape,
						site_imtls, deagg_site_model, mag_bins, dist_bins,
						eps_bins, lon_bins, lat_bins, dtype, verbose):
	"""
	Stand-alone function that will deaggregate for a single source.
	The computed non-exceedance probabilities will be multiplied with
	the full 9-D deaggregation matrix in shared memory.
	Note:
		- all necessary parameters have to provided, so as not to depend
		  on variables defined elsewhere

	:param psha_model:
		instance of :class:`PSHAModel`
	:param source:
		instance of :class:`PointSource`, :class:`AreaSource`,
		:class:`SimpleFaultSource` or :class:`ComplexFaultSource`
	:param src_idx:
		int, source index in shared 9-D deaggregation matrix
	:param shared_deagg_matrix_shape:
		tuple describing shape of full deaggregation matrix in shared
		memory
	:param site_imtls:
		nested dictionary mapping (lon, lat) tuples to dictionaries
		mapping oqhazlib IMT objects to 1-D arrays of intensity measure
		levels
	:param deagg_site_model:
		instance of :class:`rshalib.site.SoilSiteModel`, sites for which
		deaggregation will be performed.
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
	from openquake.hazardlib.site import SiteCollection
	from ..gsim import make_gsim_contexts

	try:
		## Initialize deaggregation matrix with ones representing
		## NON-exceedance probabilities !
		src_deagg_matrix = np.ones(deagg_matrix_shape[:-1], dtype=dtype)

		## Perform deaggregation
		#tom = psha_model.poisson_tom
		gsims = psha_model._get_trt_gsim_dict()
		source_site_filter = psha_model.source_site_filter
		rupture_site_filter = psha_model.rupture_site_filter

		n_epsilons = len(eps_bins) - 1

		if OQ_VERSION >= '2.9.0':
			from scipy.stats import truncnorm
			trunc_norm = truncnorm(-psha_model.truncation_level,
									psha_model.truncation_level)

		#sources = [source]
		#sources_sites = ((source, deagg_site_model) for source in sources)
		#for s, (source, s_sites) in \
		#		enumerate(source_site_filter(sources_sites)):

		if OQ_VERSION >= '2.9.0':
			source_site_filter = source_site_filter(deagg_site_model)([source])
		else:
			sources_sites = [(source, deagg_site_model)]
			source_site_filter = source_site_filter(sources_sites)
		for _, s_sites in source_site_filter:
			if OQ_VERSION >= '2.9.0':
				tom = None
			else:
				tom = psha_model.poisson_tom

			if verbose:
				print(source.source_id)

			trt = source.tectonic_region_type
			gsim = gsims[trt]

			#ruptures_sites = ((rupture, s_sites)
			#				  for rupture in source.iter_ruptures(tom))
			#for rupture, r_sites in rupture_site_filter(ruptures_sites):
			for rupture in source.iter_ruptures(tom):
				if OQ_VERSION >= '2.9.0':
					r_sites = rupture_site_filter(rupture, sites=s_sites)
				else:
					try:
						[(_, r_sites)] = rupture_site_filter([(rupture, s_sites)])
					except:
						r_sites = None
				if r_sites is None:
					continue

				## Extract rupture parameters of interest
				mag_idx = np.digitize([rupture.mag], mag_bins)[0] - 1

				sitemesh = r_sites.mesh
				#sctx, rctx, dctx = gsim.make_contexts(r_sites, rupture)
				#if hasattr(dctx, "rjb"):
				#	jb_dists = getattr(dctx, "rjb")
				#else:
				jb_dists = rupture.surface.get_joyner_boore_distance(sitemesh)
				closest_points = rupture.surface.get_closest_points(sitemesh)
				lons = [pt.longitude for pt in closest_points]
				lats = [pt.latitude for pt in closest_points]

				dist_idxs = np.digitize(jb_dists, dist_bins) - 1
				lon_idxs = np.digitize(lons, lon_bins) - 1
				lat_idxs = np.digitize(lats, lat_bins) - 1

				## Compute probability of one or more rupture occurrences
				if OQ_VERSION < '2.9.0':
					poe_rup = rupture.get_probability_one_or_more_occurrences()

				## compute conditional probability of exceeding iml given
				## the current rupture, and different epsilon level, that is
				## ``P(IMT >= iml | rup, epsilon_bin)`` for each of epsilon bins
				for site_idx, site in enumerate(r_sites):
					dist_idx = dist_idxs[site_idx]
					lon_idx = lon_idxs[site_idx]
					lat_idx = lat_idxs[site_idx]
					site_key = (site.location.longitude, site.location.latitude)
					imtls = site_imtls[site_key]
					imts = list(imtls.keys())
					site_col = SiteCollection([site])

					try:
						sctx, rctx, dctx = make_gsim_contexts(gsim, site_col, rupture,
											max_distance=psha_model.integration_distance)
					except:
						## Rupture probably too far
						continue

					for imt_idx, imt_tuple in enumerate(imts):
						imls = imtls[imt_tuple]
						## Reconstruct imt from tuple
						imt = getattr(oqhazlib.imt, imt_tuple[0])(*imt_tuple[1:])
						## In contrast to what is stated in the documentation,
						## disaggregate_poe does handle more than one iml
						if OQ_VERSION >= '3.2.0':
							pone = gsim.disaggregate_pne(rupture, sctx, dctx,
								imt, imls, trunc_norm, eps_bins)
						elif OQ_VERSION >= '2.9.0':
							pone = gsim.disaggregate_pne(rupture, sctx, rctx,
								dctx, imt, imls, trunc_norm, eps_bins)
						else:
							poes_given_rup_eps = gsim.disaggregate_poe(sctx, rctx,
									dctx, imt, imls, psha_model.truncation_level,
									n_epsilons)

							## Probability of non-exceedance
							pone = (1. - poe_rup) ** poes_given_rup_eps

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

	except Exception as err:
		msg = 'Warning: An error occurred with source %s:\n%s'
		msg %= (source.source_id, traceback.format_exc())
		raise RuntimeError(msg)
		#print(msg)
		#return 1
	else:
		return 0


def deaggregate_psha_model(psha_model, curve_name, curve_path, deagg_sites,
							deagg_imt_periods, mag_bin_width, distance_bin_width,
							num_epsilon_bins, coordinate_bin_width, dtype,
							calc_id, interpolate_rp, verbose):
	"""
	Stand-alone function that will deaggregate a single PSHA model,
	e.g. a logic-tree sample.
	Intensity measure levels corresponding to psha_model.return_periods
	will be interpolated first, so the hazard curves must have been
	computed before.
	Note:
		- all necessary parameters have to provided, so as not to depend
		  on variables defined elsewhere

	:param psha_model:
		instace of :class:`PSHAModel`
	:param curve_name:
		str, identifying hazard curve (e.g., "rlz-01", "mean", "quantile_0.84")
	:param curve_path:
		str, path to hazard curve relative to main hazard-curve folder
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
	:param calc_id:
		int or str, OpenQuake calculation ID
	:param interpolate_rp:
		bool, whether or not to interpolate intensity levels corresponding
		to return periods from the corresponding curve first.
		If False, deaggregation will be performed for all intensity levels
		available for a given spectral period.
	:param verbose:
		Bool, whether or not to print some progress information

	:return:
		0 (successful execution) or 1 (error occurred)

	The deaggregation results will be written to XML files in the output
	folder of the PSHA model.
	"""
	if verbose:
		print(psha_model.name)

	if interpolate_rp:
		## Determine intensity levels from saved hazard curves
		site_imtls = psha_model._interpolate_oq_site_imtls(curve_name, curve_path,
								deagg_sites, deagg_imt_periods, calc_id=calc_id)
	else:
		## Deaggregate for all available intensity levels
		site_imtls = psha_model._get_deagg_site_imtls(deagg_sites, deagg_imt_periods)
		## Override return_periods property with fake return periods
		psha_model.return_periods = np.zeros(psha_model.num_intensities)

	## Deaggregation
	if verbose:
		print("Starting deaggregation of curve %s..." % curve_name)

	try:
		spectral_deagg_curve_dict = psha_model.deaggregate(site_imtls, mag_bin_width,
												distance_bin_width, num_epsilon_bins,
												coordinate_bin_width, dtype, verbose=False)
	except Exception as err:
		msg = 'Warning: An error occurred with curve %s. Error: %s'
		msg %= (curve_path, err)
		#raise RuntimeError(msg)
		print(msg)
		return 1
	else:
		## Write XML file(s), creating directory if necessary
		for (lon, lat) in spectral_deagg_curve_dict.keys():
			sdc = spectral_deagg_curve_dict[(lon, lat)]
			psha_model.write_oq_disagg_matrix_multi(sdc, curve_name,
										curve_path=curve_path, calc_id=calc_id)

		## Don't return deaggregation results to preserve memory
		return 0


def calc_gmf_with_fixed_epsilon(
		rupture, sites, imt_tuple, gsim, truncation_level,
		total_residual_epsilons=None,
		intra_residual_epsilons=None,
		inter_residual_epsilons=None,
		integration_distance=None):
	"""
	Standalone function that computes ground-motion field for a single rupture.
	Modified from function ground_motion_field_with_residuals in old version of
	oq-hazardlib. In newer versions, this function has been removed.

	Original documentation
	A simplified version of ``ground_motion_fields`` where: the values
	due to uncertainty (total, intra-event or inter-event residual
	epsilons) are given in input; only one intensity measure type is
	considered.

	See :func:``openquake.hazardlib.calc.gmf.ground_motion_fields`` for
	the description of most of the input parameters.

	:param total_residual_epsilons:
		a 2d numpy array of floats with the epsilons needed to compute the
		total residuals in the case the GSIM provides only total standard
		deviation.
	:param intra_residual_epsilons:
		a 2d numpy array of floats with the epsilons needed to compute the
		intra event residuals
	:param inter_residual_epsilons:
		a 2d numpy array of floats with the epsilons needed to compute the
		intra event residuals
		Note that intra- and inter-event residuals should not be opposite
		in sign.

	:returns:
		a 1d numpy array of floats, representing ground shaking intensity
		for all sites in the collection.
	"""
	from openquake.hazardlib.const import StdDev
	from ..gsim import make_gsim_contexts

	## Reconstruct imt from tuple
	imt = getattr(oqhazlib.imt, imt_tuple[0])(*imt_tuple[1:])

	if integration_distance:
		if OQ_VERSION >= '3.2.0':
			## No-op, filtering is handled in ContextMaker
			rupture_site_filter = lambda rupture, sites: sites
		elif OQ_VERSION >= '2.9.0':
			rupture_site_filter = partial(oqhazlib.calc.filters.filter_sites_by_distance_to_rupture,
										integration_distance=integration_distance)
		else:
			rupture_site_filter = oqhazlib.calc.filters.rupture_site_distance_filter(
															integration_distance)
		## Trim *_epsilons arrays to sites within integration_distance
		jb_dist = rupture.surface.get_joyner_boore_distance(sites.mesh)
		indices = (jb_dist <= integration_distance)
		if total_residual_epsilons is not None:
			total_residual_epsilons = total_residual_epsilons[indices]
		if intra_residual_epsilons is not None:
			intra_residual_epsilons = intra_residual_epsilons[indices]
		if inter_residual_epsilons is not None:
			inter_residual_epsilons = inter_residual_epsilons[indices]
	else:
		rupture_site_filter = oqhazlib.calc.filters.source_site_noop_filter

	total_sites = len(sites)

	if OQ_VERSION >= '2.9.0':
		sites = rupture_site_filter(rupture, sites=sites)
	else:
		ruptures_sites = list(rupture_site_filter([(rupture, sites)]))
		[(rupture, sites)] = ruptures_sites
	if not sites:
		return np.zeros(total_sites)

	try:
		sctx, rctx, dctx = make_gsim_contexts(gsim, sites, rupture,
								max_distance=integration_distance)
	except:
		## Rupture probably too far
		return np.zeros(total_sites)

	if truncation_level == 0:
		mean, _stddevs = gsim.get_mean_and_stddevs(sctx, rctx, dctx, imt,
												   stddev_types=[])
		gmf = gsim.to_imt_unit_values(mean)

	elif gsim.DEFINED_FOR_STANDARD_DEVIATION_TYPES == set([StdDev.TOTAL]):
		assert total_residual_epsilons is not None

		mean, [stddev_total] = gsim.get_mean_and_stddevs(
			sctx, rctx, dctx, imt, [StdDev.TOTAL]
		)
		#stddev_total = stddev_total.reshape(stddev_total.shape + (1, ))
		total_residual = stddev_total * total_residual_epsilons
		gmf = gsim.to_imt_unit_values(mean + total_residual)
	else:
		assert inter_residual_epsilons is not None
		assert intra_residual_epsilons is not None
		mean, [stddev_inter, stddev_intra] = gsim.get_mean_and_stddevs(
			sctx, rctx, dctx, imt, [StdDev.INTER_EVENT, StdDev.INTRA_EVENT]
		)

		intra_residual = stddev_intra * intra_residual_epsilons
		inter_residual = stddev_inter * inter_residual_epsilons

		## We have to take into account the sign of epsilon,
		## otherwise, residuals will always be positive!
		## Assume sign of intra_residual_epsilons and inter_residual_epsilons
		## cannot be opposite
		intra_sign = np.sign(intra_residual_epsilons)
		inter_sign = np.sign(inter_residual_epsilons)
		assert (intra_sign != -inter_sign).any()
		#sign = np.sign(intra_sign + inter_sign)

		## Convert residuals to imaginary part of complex numbers
		intra_residual = 0 + 1j * intra_residual
		inter_residual = 0 + 1j * inter_residual

		gmf = gsim.to_imt_unit_values(
			#mean + sign * np.sqrt(intra_residual**2 + inter_residual)**2)
			mean + np.imag(np.sqrt(intra_residual**2 + inter_residual)**2))

	placeholder = 0.
	if sites.indices is not None:
		if OQ_VERSION >= '2.9.0':
			_gmf = np.ones(total_sites) * placeholder
			if OQ_VERSION >= '3.2.0':
				_gmf.put(sctx.sids, gmf)
			else:
				_gmf.put(sites.indices, gmf)
			gmf = _gmf
		else:
			gmf = sites.expand(gmf, total_sites, placeholder=placeholder)

	return gmf


def calc_random_gmf(
		rupture,
		sites,
		imt_tuple,
		gsim,
		truncation_level,
		num_realizations,
		shared_arr_idx,
		shared_arr_shape,
		correlation_model=None,
		integration_distance=None,
		random_seed=None):
	"""
	Compute random ground-motion field for a single rupture and IMT.
	Results are stored in a shared array.

	:param imt_tuple:
		(string IMT, float period) tuple
	:param truncation_level:
		float, uncertainty range in number of standard deviations
	:param num_realizations:
		int, number of random fields to generate
	:param shared_arr_idx:
		(r,g,k) tuple containing rupture index (r), gmpe index (g) and
		imt index (k) in shared array
	:param shared_arr_shape:
		tuple containing shape of shared array
	:param integration_distance:
		float, integratiion distance in km
		(default: None)
	:param random_seed:
		int, random seed (default: None)

	See :func:``openquake.hazardlib.calc.gmf.ground_motion_fields`` for
	the description of remaining input parameters.

	:return:
		0 (successful execution) or 1 (error occurred)
	"""
	from openquake.hazardlib.calc import ground_motion_fields

	if truncation_level > 0:
		np.random.seed(seed=random_seed)

	## Reconstruct imt from tuple
	imt = getattr(oqhazlib.imt, imt_tuple[0])(*imt_tuple[1:])

	if integration_distance:
		if OQ_VERSION >= '3.2.0':
			from openquake.hazardlib.gsim.base import ContextMaker
			from openquake.hazardlib.calc.filters import IntegrationDistance
			trt = 'default'
			maximum_distance = {trt: [(rupture.mag, integration_distance)]}
			maximum_distance = IntegrationDistance(maximum_distance)
			ctx_maker = ContextMaker([gsim], maximum_distance=maximum_distance)
			sites, dctx = ctx_maker.filter(sites, rupture)
			site_idxs = sites.array['sids']
		elif OQ_VERSION >= '2.9.0':
			sites = oqhazlib.calc.filters.filter_sites_by_distance_to_rupture(
									rupture, integration_distance, sites)
			site_idxs = sites.indices
		else:
			rupture_site_filter = oqhazlib.calc.filters.rupture_site_distance_filter(
															integration_distance)
			site_idxs = None
	else:
		rupture_site_filter = oqhazlib.calc.filters.source_site_noop_filter

	try:
		if OQ_VERSION >= '2.9.0':
			gmf_dict = ground_motion_fields(rupture, sites, [imt], gsim,
										truncation_level, num_realizations,
										correlation_model)
		else:
			gmf_dict = ground_motion_fields(rupture, sites, [imt], gsim,
										truncation_level, num_realizations,
										correlation_model, rupture_site_filter)

		with shared_arr.get_lock(): # synchronize access
			shared_gmf_matrix = np.frombuffer(shared_arr.get_obj()) # no data copying
			shared_gmf_matrix = shared_gmf_matrix.reshape(shared_arr_shape)
			r, g, k = shared_arr_idx
			if site_idxs is not None:
				shared_gmf_matrix[r,g,site_idxs,:,k] = gmf_dict[imt]
			else:
				shared_gmf_matrix[r,g,:,:,k] = gmf_dict[imt]

	except Exception as err:
		msg = 'Warning: An error occurred with field #%s:\n%s'
		msg %= (shared_arr_idx, traceback.format_exc())
		raise RuntimeError(msg)
		#print(msg)
		#return 1
	else:
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
	## Test
	result = run_parallel(np_array, range(10), 3)
	for ar in result:
		print(ar)
