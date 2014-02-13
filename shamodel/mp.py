"""
Stand-alone functions needed for multiprocessing
"""

import os
import openquake.hazardlib as oqhazlib


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
		int, sample index
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

			# TODO: number of leading zeros depends on num_lt_samples
			hc_filename = "hazard_curve-rlz-%03d.xml" % (sample_idx + 1)
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
		print("Starting deaggregation of sample %d..." % sample_idx)
	spectral_deagg_curve_dict = psha_model.deaggregate(site_imtls, mag_bin_width, distance_bin_width, num_epsilon_bins, coordinate_bin_width, verbose=verbose)

	## Write XML file(s), creating directory if necessary
	for (lon, lat) in spectral_deagg_curve_dict.keys():
		spectral_deagg_curve = spectral_deagg_curve_dict[(lon, lat)]
		deagg_folder = psha_model.output_dir
		for subfolder in ("disaggregation", "calc_oqhazlib",  "disagg_matrix_multi"):
			deagg_folder = os.path.join(deagg_folder, subfolder)
			if not os.path.exists(deagg_folder):
				os.mkdir(deagg_folder)
		# TODO: number of leading zeros, cf. above ...
		xml_filename = "disagg_matrix_multi-lon_%s-lat_%s-rlz-%03d.xml" % (lon, lat, sample_idx + 1)
		xml_filespec = os.path.join(deagg_folder, xml_filename)
		spectral_deagg_curve.write_xml(xml_filespec, psha_model.smlt_path, psha_model.gmpelt_path)

	## Don't return anything to preserve memory


def do_nothing(sample_idx, psha_model):
	"""
	Dummy function for testing multiprocessing.
	"""
	print("Doing nothing #%d" % (sample_idx,))


