"""
Stand-alone functions needed for multiprocessing
"""

import os
import multiprocessing
import numpy as np
import openquake.hazardlib as oqhazlib



def run_parallel(func, job_arg_list, num_processes):
	"""
	Wrapper
	"""
	num_processes = min(multiprocessing.cpu_count(), num_processes)
	pool = multiprocessing.Pool(processes=num_processes)
	result = pool.map(func, job_arg_list)
	return result


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


def calc_hc_source((psha_model, source, cav_min, verbose)):
	"""
	Stand-alone function that will compute hazard curves for a single
	source.

	:return:
		3-D numpy array [i,k,l] with probabilities of non-exceedance
	"""
	if verbose:
		print source.source_id
	sources = [source]
	sites = psha_model.get_soil_site_model()
	gsims = psha_model._get_nhlib_trts_gsims_map()
	imts = psha_model._get_nhlib_imts()
	tom = psha_model.poisson_tom

	# TODO: shared memory array to store curves?
	total_sites = len(sites)
	shape = (total_sites, len(imts), len(imts[imts.keys()[0]]))
	curves = np.ones(shape)

	## Copied from openquake.hazardlib
	sources_sites = ((source, sites) for source in sources)
	for source, s_sites in psha_model.source_site_filter(sources_sites):
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
	return curves


def run_psha_model((psha_model, sample_idx, cav_min, verbose)):
	"""
	Stand-alone function that will compute hazard curves for a single
	logic-tree sample.

	:param psha_model:
		instace of :class:`PSHAModel`
	:param sample_idx:
		str, sample index (formatted with adequate number of leading zeros)
	:param cav_min:
		float, CAV threshold in g.s
	:param verbose:
		Bool, whether or not to print some progress information

	:return:
		None
	"""
	## Run
	if verbose:
		print("Starting hazard-curve calculation of sample %s..." % sample_idx)
	shcfs = psha_model.run_nhlib_shcf(plot=False, write=False, cav_min=cav_min)

	## Write XML file, creating directory if necessary
	hc_folder = psha_model.output_dir
	for subfolder in ("classical", "calc_oqhazlib",  "hazard_curve_multi"):
		hc_folder = os.path.join(hc_folder, subfolder)
		if not os.path.exists(hc_folder):
			os.mkdir(hc_folder)
	# TODO: number of leading zeros, cf. above ...
	xml_filename = "hazard_curve_multi-rlz-%s.xml" % (lon, lat, sample_idx)
	xml_filespec = os.path.join(deagg_folder, xml_filename)
	shcfs.write_nrml(xml_filespec, psha_model.smlt_path, psha_model.gmpelt_path)

	return None


def deaggregate_psha_model((psha_model, sample_idx, hc_folder, deagg_sites, deagg_imt_periods, mag_bin_width, distance_bin_width, num_epsilon_bins, coordinate_bin_width, verbose)):
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
		str, sample index (formatted with adequate number of leading zeros)
	:param hc_folder:
		str, full path to top folder containing hazard curves for
		different IMT's
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
	:param verbose:
		Bool, whether or not to print some progress information

	:return:
		None

	The deaggregation results will be written to XML files in the output
	folder of the PSHA model.
	"""
	from ..openquake import parse_hazard_curves

	if verbose:
		print psha_model.name

	## Determine intensity levels from saved hazard curves
	site_imtls = {}
	for site in deagg_sites:
		site_imtls[(site.lon, site.lat)] = {}

	for im in sorted(deagg_imt_periods.keys()):
		for T in sorted(deagg_imt_periods[im]):
			if im == "PGA":
				imt = getattr(oqhazlib.imt, im)()
			else:
				imt = getattr(oqhazlib.imt, im)(T, 5.)

			## Determine imls from hazard curves
			if im == "PGA":
				im_hc_folder = os.path.join(hc_folder, im)
			else:
				im_hc_folder = os.path.join(hc_folder, "%s-%s" % (im, T))

			hc_filename = "hazard_curve-rlz-%s.xml" % sample_idx
			hc_filespec = os.path.join(im_hc_folder, hc_filename)
			hcf = parse_hazard_curves(hc_filespec)
			sha_sites = psha_model.get_soil_site_model().get_sha_sites()
			hcf.set_site_names(sha_sites)
			for site in deagg_sites:
				hc = hcf.getHazardCurve(site.name)
				imls = hc.interpolate_return_periods(psha_model.return_periods)
				#print imt, imls
				site_imtls[(site.lon, site.lat)][imt] = imls

	## Deaggregation
	if verbose:
		print("Starting deaggregation of sample %s..." % sample_idx)
	spectral_deagg_curve_dict = psha_model.deaggregate(site_imtls, mag_bin_width, distance_bin_width, num_epsilon_bins, coordinate_bin_width, verbose=False)

	## Write XML file(s), creating directory if necessary
	for (lon, lat) in spectral_deagg_curve_dict.keys():
		spectral_deagg_curve = spectral_deagg_curve_dict[(lon, lat)]
		deagg_folder = psha_model.output_dir
		for subfolder in ("disaggregation", "calc_oqhazlib",  "disagg_matrix_multi"):
			deagg_folder = os.path.join(deagg_folder, subfolder)
			if not os.path.exists(deagg_folder):
				os.mkdir(deagg_folder)
		xml_filename = "disagg_matrix_multi-lon_%s-lat_%s-rlz-%s.xml" % (lon, lat, sample_idx)
		xml_filespec = os.path.join(deagg_folder, xml_filename)
		spectral_deagg_curve.write_nrml(xml_filespec, psha_model.smlt_path, psha_model.gmpelt_path)

	## Don't return anything to preserve memory
	return None


def do_nothing(sample_idx, psha_model):
	"""
	Dummy function for testing multiprocessing.
	"""
	print("Doing nothing #%s" % (sample_idx,))



if __name__ == "__main__":
	print run_parallel(np_array, range(10), 3)
