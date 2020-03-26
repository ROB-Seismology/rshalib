"""
DecomposedPSHAModelTree class
"""

from __future__ import absolute_import, division, print_function, unicode_literals


### imports
import os
from collections import OrderedDict

import numpy as np

from openquake.hazardlib.imt import PGA, SA, PGV, PGD, MMI

from ..calc import mp
from ..geo import *
from ..site import *
from ..result import SpectralHazardCurveField, SpectralHazardCurveFieldTree, SpectralDeaggregationCurve
from ..source import SourceModel
from ..gsim import GroundMotionModel
from .pshamodeltree import PSHAModelTree



__all__ = ['DecomposedPSHAModelTree']


class DecomposedPSHAModelTree(PSHAModelTree):
	"""
	Special version of PSHAModelTree that is computed in a source-centric way.
	Instead of computing hazard curves for a complete source model
	corresponding to sampled or enumerated branches, all realizations
	for each source are computed separately, in order to save computation
	time.

	Parameters are identical to :class:`PSHAModelTree`
	"""
	def __init__(self, name, source_model_lt, gmpe_lt, root_folder,
				site_model, ref_soil_params=REF_SOIL_PARAMS,
				imt_periods={'PGA': [0]}, intensities=None,
				min_intensities=0.001, max_intensities=1., num_intensities=100,
				return_periods=[], time_span=50.,
				truncation_level=3., integration_distance=200.,
				num_lt_samples=1, random_seed=42):
		"""
		"""
		PSHAModelTree.__init__(self, name, source_model_lt, gmpe_lt, root_folder,
							site_model, ref_soil_params, imt_periods, intensities,
							min_intensities, max_intensities, num_intensities,
							return_periods, time_span, truncation_level,
							integration_distance, num_lt_samples, random_seed)

	def _get_curve_path(self, source_model_name, trt, source_id, gmpe_name):
		"""
		Construct subfolder path for decomposed calculation

		:param source_model_name:
			str, name of source model
		:param trt:
			str, tectonic region type
		:param source_id:
			str, source ID
		:param gmpe_name:
			str, name of GMPE

		:return:
			str, subfolder path
		"""
		trt_short_name = ''.join([word[0].capitalize() for word in trt.split()])
		curve_path = os.path.sep.join([source_model_name, trt_short_name,
										source_id, gmpe_name])
		return curve_path

	def iter_psha_models(self, source_type=None):
		"""
		Loop over decomposed PSHA models

		:param source_type:
			str, one of "point", "area", "fault", "simple_fault",
			"complex_fault" or "non_area"
			(default: None, will use all sources)

		:return:
			generator object yielding instances of :class:`PSHAModel`
		"""
		gmpe_system_def = self.gmpe_lt.gmpe_system_def
		for source_model in self.source_models:
			for src in source_model.get_sources_by_type(source_type):
				for (modified_src, branch_path, branch_weight) in \
						self.source_model_lt.enumerate_source_realizations(
														source_model.name, src):
					branch_path = [b.split('--')[-1] for b in branch_path]
					somo_name = "%s--%s" % (source_model.name, src.source_id)
					curve_name = '--'.join(branch_path)
					partial_source_model = SourceModel(somo_name+'--'+curve_name,
														[modified_src], "")
					trt = src.tectonic_region_type
					for gmpe_name in gmpe_system_def[trt].gmpe_names:
						gmpe_model = GroundMotionModel("", {trt: gmpe_name})
						model_name = somo_name + " -- " + gmpe_name
						psha_model = self._get_psha_model(partial_source_model,
														gmpe_model, model_name)
						yield psha_model

	def calc_shcf_mp(self, cav_min=0, num_cores=None, combine_pga_and_sa=True,
					calc_id="oqhazlib", overwrite=True, verbose=True):
		"""
		Compute spectral hazard curve fields using multiprocessing.
		The results are written to XML files in a folder structure:
		source_model_name / trt_short_name / source_id / gmpe_name

		:param cav_min:
			float, CAV threshold in g.s
			(default: 0)
		:param num_cores:
			int, number of CPUs to be used. Actual number of cores used
			may be lower depending on available cores and memory
			(default: None, will determine automatically)
		:param combine_pga_and_sa:
			bool, whether or not to combine PGA and SA, if present
			(default: True)
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: "oqhazlib")
		:param overwrite:
			bool, whether or not to overwrite existing files. This allows to
			skip computed results after an interruption (default: True)
		:param verbose:
			bool, whether or not to print some progress information
			(default: True)
		"""
		## Area sources:
		## multiprocesing is applied to decomposed area sources in each PSHA model
		for psha_model in self.iter_psha_models("area"):
			if verbose:
				print(psha_model.name)
			curve_name_parts = psha_model.source_model.name.split('--')
			source_model_name = curve_name_parts[0]
			curve_name = '--'.join(curve_name_parts[2:])
			src = psha_model.source_model.sources[0]
			trt = src.tectonic_region_type
			gmpe_name = psha_model.ground_motion_model[trt]

			if overwrite is False:
				## Skip if files already exist and overwrite is False
				im_imls = self._get_im_imls(combine_pga_and_sa=combine_pga_and_sa)
				files_exist = []
				for im in im_imls.keys():
					# TODO: different filespecs for different ims?
					xml_filespec = self.get_oq_shcf_filespec_decomposed(
										source_model_name, trt, src.source_id,
										gmpe_name, curve_name, calc_id=calc_id)
					files_exist.append(os.path.exists(xml_filespec))
				if np.all(files_exist):
					continue

			shcf_dict = psha_model.calc_shcf_mp(cav_min=cav_min,
											decompose_area_sources=True,
											num_cores=num_cores,
											combine_pga_and_sa=combine_pga_and_sa)

			for im in shcf_dict.keys():
				shcf = shcf_dict[im]
				self.write_oq_shcf(shcf, source_model_name, trt, src.source_id,
									gmpe_name, curve_name, calc_id=calc_id)


		## Non-area sources:
		## multiprocessing is applied to PSHA models not containing area sources
		psha_models = list(self.iter_psha_models("non_area"))

		## Create list with arguments for each job
		job_args = []
		for psha_model in psha_models:
			if verbose:
				print(psha_model.name)
			curve_name_parts = psha_model.source_model.name.split('--')
			source_model_name = curve_name_parts[0]
			curve_name = '--'.join(curve_name_parts[2:])
			src = psha_model.source_model.sources[0]
			trt = src.tectonic_region_type
			gmpe_name = psha_model.ground_motion_model[trt]
			curve_path = self._get_curve_path(source_model_name, trt, src.source_id,
												gmpe_name)

			if overwrite is False:
				## Skip if files already exist and overwrite is False
				im_imls = self._get_im_imls(combine_pga_and_sa=combine_pga_and_sa)
				files_exist = []
				for im in im_imls.keys():
					# TODO: different filespecs for different ims?
					xml_filespec = self.get_oq_shcf_filespec_decomposed(
										source_model_name, trt, src.source_id,
										gmpe_name, curve_name, calc_id=calc_id)
					files_exist.append(os.path.exists(xml_filespec))
				if np.all(files_exist):
					continue

			job_args.append((psha_model, curve_name, curve_path, cav_min,
							combine_pga_and_sa, calc_id, verbose))

			## Create folder before starting mp to avoid race conditions
			hc_folder = self.get_oq_hc_folder_decomposed(source_model_name, trt,
										src.source_id, gmpe_name, calc_id=calc_id)
			self.create_folder_structure(hc_folder)

		## Launch multiprocessing
		if len(job_args) > 0:
			mp.run_parallel(mp.calc_shcf_psha_model, job_args, num_cores,
							verbose=verbose)

	def deaggregate_mp(self, sites, imt_periods,
						mag_bin_width=None, dist_bin_width=10., n_epsilons=None,
						coord_bin_width=1.0, dtype='d', num_cores=None,
						calc_id="oqhazlib", interpolate_rp=True, overwrite=True,
						verbose=False):
		"""
		Compute spectral deaggregation curves using multiprocessing.
		The results are written to XML files in a folder structure:
		source_model_name / trt_short_name / source_id / gmpe_name

		:param sites:
			list with instances of :class:`GenericSite` for which deaggregation
			will be performed. Note that instances of class:`SoilSite` will
			not work with multiprocessing
		:param imt_periods:
			dictionary mapping intensity measure strings to lists of spectral
			periods.
		:param mag_bin_width:
			Float, magnitude bin width
			(default: None, will take MFD bin width of first source)
		:param dist_bin_width:
			Float, distance bin width in km
			(default: 10.)
		:param n_epsilons:
			Int, number of epsilon bins
			(default: None, will result in bins corresponding to integer
			epsilon values)
		:param coord_bin_width:
			Float, lon/lat bin width in decimal degrees
			(default: 1.)
		:param dtype:
			str, precision of deaggregation matrix
			(default: 'd')
		:param num_cores:
			int, number of CPUs to be used. Actual number of cores used
			may be lower depending on available cores and memory
			(default: None, will determine automatically)
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: "oqhazlib")
		:param interpolate_rp:
			bool, whether or not to interpolate intensity levels corresponding
			to return periods from the overall mean hazard curve first.
			If False, deaggregation will be performed for all intensity levels
			available for a given spectral period.
			(default: True).
		:param overwrite:
			bool, whether or not to overwrite existing files. This allows to
			skip computed results after an interruption
			(default: True)
		:param verbose:
			bool, whether or not to print some progress information
			(default: True)
		"""
		## Convert sites to GenericSite objects if necessary, because SoilSites
		## cause problems when used in conjunction with multiprocessing
		## (notably the name attribute cannot be accessed, probably due to
		## the use of __slots__ in parent class)
		## Note that this is similar to the deepcopy problem with MFD objects.
		deagg_sites = []
		site_model = self.get_soil_site_model()
		for site in sites:
			if isinstance(site, SoilSite):
				site = site.to_sha_site()
			if site in site_model:
				deagg_sites.append(site)

		## Determine intensity levels for which to perform deaggregation
		if interpolate_rp:
			## Determine intensity levels corresponding to return periods
			## from mean hazard curve
			site_imtls = self._interpolate_oq_site_imtls(deagg_sites, imt_periods,
														calc_id=calc_id)
			return_periods = self.return_periods
		else:
			## Deaggregate for all available intensity levels
			site_imtls = self._get_deagg_site_imtls(deagg_sites, imt_periods)
			## Fake return periods
			return_periods = np.zeros(self.num_intensities)


		## Deaggregate area sources:
		## multiprocesing is applied to decomposed area sources in each PSHA model
		for psha_model in self.iter_psha_models("area"):
			if verbose:
				print(psha_model.name)

			curve_name_parts = psha_model.source_model.name.split('--')
			source_model_name = curve_name_parts[0]
			curve_name = '--'.join(curve_name_parts[2:])
			src = psha_model.source_model.sources[0]
			trt = src.tectonic_region_type
			gmpe_name = psha_model.ground_motion_model[trt]
			curve_path = self._get_curve_path(source_model_name, trt, src.source_id,
												gmpe_name)
			## Override return_periods property
			psha_model.return_periods = return_periods

			if overwrite is False:
				## Skip if files already exist and overwrite is False
				files_exist = []
				for (lon, lat) in site_imtls.keys():
					site = GenericSite(lon, lat)
					xml_filespec = self.get_oq_sdc_filespec_decomposed(
										source_model_name, trt, src.source_id,
										gmpe_name, curve_name, site, calc_id=calc_id)
					files_exist.append(os.path.exists(xml_filespec))
				if np.all(files_exist):
					continue

			sdc_dict = psha_model.deaggregate_mp(site_imtls, decompose_area_sources=True,
								mag_bin_width=mag_bin_width, dist_bin_width=dist_bin_width,
								n_epsilons=n_epsilons, coord_bin_width=coord_bin_width,
								dtype=dtype, num_cores=num_cores, verbose=verbose)

			## Write XML file(s), creating directory if necessary
			for (lon, lat) in sdc_dict.keys():
				sdc = sdc_dict[(lon, lat)]
				self.write_oq_disagg_matrix_multi(sdc, source_model_name, trt,
												src.source_id, gmpe_name, curve_name,
												calc_id=calc_id)

		## Deaggregate non-area sources:
		## multiprocessing is applied to PSHA models not containing area sources
		psha_models = list(self.iter_psha_models("non_area"))

		## Create list with arguments for each job
		job_args = []
		for psha_model in psha_models:
			if verbose:
				print(psha_model.name)
			curve_name_parts = psha_model.source_model.name.split('--')
			source_model_name = curve_name_parts[0]
			curve_name = '--'.join(curve_name_parts[2:])
			src = psha_model.source_model.sources[0]
			trt = src.tectonic_region_type
			gmpe_name = psha_model.ground_motion_model[trt]
			curve_path = self._get_curve_path(source_model_name, trt, src.source_id,
											gmpe_name)

			if overwrite is False:
				## Skip if files already exist and overwrite is False
				im_imls = self._get_im_imls(combine_pga_and_sa=True)
				files_exist = []
				for (lon, lat) in site_imtls.keys():
					site = GenericSite(lon, lat)
					xml_filespec = self.get_oq_sdc_filespec_decomposed(source_model_name,
												trt, src.source_id, gmpe_name, curve_name,
												site, calc_id=calc_id)
					files_exist.append(os.path.exists(xml_filespec))
				if np.all(files_exist):
					continue

			deagg_sites = [GenericSite(lon, lat) for (lon, lat) in site_imtls.keys()]
			job_args.append((psha_model, curve_name, curve_path, deagg_sites,
							imt_periods, mag_bin_width, dist_bin_width, n_epsilons,
							coord_bin_width, dtype, calc_id, interpolate_rp, verbose))

			## Create folder before starting mp to avoid race conditions
			hc_folder = self.get_oq_hc_folder_decomposed(source_model_name, trt,
									src.source_id, gmpe_name, calc_id=calc_id)
			self.create_folder_structure(hc_folder)

		## Launch multiprocessing
		if len(job_args) > 0:
			mp.run_parallel(mp.deaggregate_psha_model, job_args, num_cores,
							verbose=verbose)

	def _interpolate_oq_site_imtls(self, sites, imt_periods, curve_name="",
									curve_path="", calc_id=None):
		"""
		Determine intensity levels corresponding to psha-model return periods
		from saved hazard curves. Mainly useful as helper function for
		deaggregation.

		:param sites:
			list with instances of :class:`GenericSite` or instance of
			:class:`GenericSiteModel`. Note that instances
			of class:`SoilSite` will not work with multiprocessing
		:param imt_periods:
			dictionary mapping intensity measure strings to lists of spectral
			periods.
		:param curve_name:
			str, identifying hazard curve
			(e.g., "rlz-01", "mean", "quantile_0.84")
			(default: "", will compute overall mean hazard curve)
		:param curve_path:
			str, path to hazard curve relative to main hazard-curve folder
			(default: "")
		:param calc_id:
			str, calculation ID.
			(default: None, will determine from folder structure)

		:return:
			nested dictionary mapping (lon, lat) tuples to dictionaries
			mapping oqhazlib IMT objects to 1-D arrays of intensity measure
			levels
		"""
		site_imtls = OrderedDict()
		for site in sites:
			try:
				lon, lat = site.lon, site.lat
			except AttributeError:
				lon, lat = site.location.longitude, site.location.latitude
			site_imtls[(lon, lat)] = OrderedDict()

		shcf = None
		if curve_name:
			## Read hazard_curve_multi if it exists
			try:
				shcf = self.read_oq_shcf(curve_name, curve_path=curve_path,
										calc_id=calc_id)
			except:
				pass
		if shcf is None:
			## Compute mean hazard curve
			print("Computing mean hazard curve...")
			shcf = self.get_oq_mean_shcf(calc_id=calc_id)

		for im in sorted(imt_periods.keys()):
			for T in sorted(imt_periods[im]):
				imt = self._construct_imt(im, T)
				hcf = shcf.getHazardCurveField(period_spec=T)
				for i, site in enumerate(sites):
					try:
						site_name = site.name
					except AttributeError:
						site_name = sites.site_names[i]
						lon, lat = site.location.longitude, site.location.latitude
					else:
						lon, lat = site.lon, site.lat
					hc = hcf.getHazardCurve(site_name)
					imls = hc.interpolate_return_periods(self.return_periods)
					site_imtls[(lon, lat)][imt] = imls

		return site_imtls

	def get_oq_hc_folder_decomposed(self, source_model_name, trt, source_id,
									gmpe_name, calc_id=None):
		"""
		Return path to hazard_curve folder for a decomposed computation

		:param source_model_name:
			str, name of source model
		:param trt:
			str, tectonic region type
		:param source_id:
			str, source ID
		:param gmpe_name:
			str, name of GMPE
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)

		return:
			str, full path to hazard-curve folder
		"""
		hc_folder = self.get_oq_hc_folder(calc_id=calc_id, multi=True)
		trt_short_name = ''.join([word[0].capitalize() for word in trt.split()])
		hc_folder = os.path.join(hc_folder, source_model_name, trt_short_name,
								source_id, gmpe_name)
		return hc_folder

	def get_oq_disagg_folder_decomposed(self, source_model_name, trt, source_id,
										gmpe_name, calc_id=None):
		"""
		Return path to disaggregation folder for a decomposed computation

		:param source_model_name:
			str, name of source model
		:param trt:
			str, tectonic region type
		:param source_id:
			str, source ID
		:param gmpe_name:
			str, name of GMPE
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)

		return:
			str, full path to disaggregation folder
		"""
		deagg_folder = self.get_oq_disagg_folder(calc_id=calc_id, multi=True)
		trt_short_name = ''.join([word[0].capitalize() for word in trt.split()])
		deagg_folder = os.path.join(deagg_folder, source_model_name,
									trt_short_name, source_id, gmpe_name)
		return deagg_folder

	def get_oq_shcf_filespec_decomposed(self, source_model_name, trt, source_id,
										gmpe_name, curve_name, calc_id="oqhazlib"):
		"""
		Get full path to decomposed spectral hazard curve field xml file

		:param source_model_name:
			str, name of source model
		:param trt:
			str, tectonic region type
		:param source_id:
			str, source ID
		:param gmpe_name:
			str, name of GMPE
		:param curve_name:
			str, identifying hazard curve (e.g., "Mmax01--MFD03")
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: "oqhazlib")

		:return:
			str, full path to spectral hazard curve field file
		"""
		hc_folder = self.get_oq_hc_folder_decomposed(source_model_name, trt,
										source_id, gmpe_name, calc_id=calc_id)
		xml_filename = "hazard_curve_multi-%s.xml" % curve_name
		xml_filespec = os.path.join(hc_folder, xml_filename)
		return xml_filespec

	def write_oq_shcf(self, shcf, source_model_name, trt, source_id, gmpe_name,
					curve_name, calc_id="oqhazlib"):
		"""
		Write spectral hazard curve field

		:param shcf:
			instance of :class:`rshalib.result.SpectralHazardCurveField`
		:param source_model_name:
			str, name of source model
		:param trt:
			str, tectonic region type
		:param source_id:
			str, source ID
		:param gmpe_name:
			str, name of GMPE
		:param curve_name:
			str, identifying hazard curve (e.g., "Mmax01--MFD03")
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: "oqhazlib")
		"""
		xml_filespec = self.get_oq_shcf_filespec_decomposed(source_model_name, trt,
												source_id, gmpe_name, curve_name,
												calc_id=calc_id)
		hc_folder = os.path.split(xml_filespec)[0]
		self.create_folder_structure(hc_folder)
		smlt_path = getattr(self, "smlt_path", "--".join(
									[source_model_name, source_id, curve_name]))
		gmpelt_path = getattr(self, "gmpelt_path", gmpe_name)
		shcf.write_nrml(xml_filespec, smlt_path=smlt_path, gmpelt_path=gmpelt_path)

	def get_oq_sdc_filespec_decomposed(self, source_model_name, trt, source_id,
								gmpe_name, curve_name, site, calc_id="oqhazlib"):
		"""
		Get full path to decomposed spectral deaggregation curve xml file

		:param source_model_name:
			str, name of source model
		:param trt:
			str, tectonic region type
		:param source_id:
			str, source ID
		:param gmpe_name:
			str, name of GMPE
		:param curve_name:
			str, identifying hazard curve (e.g., "Mmax01--MFD03")
		:param site:
			instace of :class:`rshalib.site.GenericSite`
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: "oqhazlib")
		"""
		disagg_folder = self.get_oq_disagg_folder_decomposed(source_model_name,
									trt, source_id, gmpe_name, calc_id=calc_id)
		xml_filename = "disagg_matrix_multi-lon_%s-lat_%s-%s.xml"
		xml_filename %= (site.lon, site.lat, curve_name)
		xml_filespec = os.path.join(disagg_folder, xml_filename)
		return xml_filespec

	def write_oq_disagg_matrix_multi(self, sdc, source_model_name, trt, source_id,
									gmpe_name, curve_name, calc_id="oqhazlib"):
		"""
		Write OpenQuake multi-deaggregation matrix. Folder structure
		will be created, if necessary.

		:param sdc:
			instance of :class:`SpectralDeaggregationCurve`
		:param source_model_name:
			str, name of source model
		:param trt:
			str, tectonic region type
		:param source_id:
			str, source ID
		:param gmpe_name:
			str, name of GMPE
		:param curve_name:
			str, identifying hazard curve (e.g., "Mmax01--MFD03")
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: "oqhazlib")
		"""
		xml_filespec = self.get_oq_sdc_filespec_decomposed(source_model_name, trt,
												source_id, gmpe_name, curve_name,
												sdc.site, calc_id=calc_id)
		disagg_folder = os.path.split(xml_filespec)[0]
		self.create_folder_structure(disagg_folder)
		smlt_path = getattr(self, "smlt_path", "--".join(
									[source_model_name, source_id, curve_name]))
		gmpelt_path = getattr(self, "gmpelt_path", gmpe_name)
		sdc.write_nrml(xml_filespec, smlt_path, gmpelt_path)

	def read_oq_realization_by_source(self, source_model_name, src, smlt_path,
									gmpelt_path, calc_id=None):
		"""
		Read results of a particular logictree sample for 1 source

		:param source_model_name:
			str, name of source model
		:param src:
			instance of :class:`rshalib.source.[Point|Area|SimpleFault|ComplexFault]Source`
		:param smlt_path:
			list of branch ids (strings), source-model logic tree path
		:param gmpelt_path:
			list of branch ids (strings), ground-motion logic tree path
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)

		:return:
			(shcf, weight) tuple:
			- shcf: instance of :class:`rshalib.result.SpectralHazardCurveField`
			- weight: decimal
		"""
		for branch_id in gmpelt_path:
			gmpe_branch = self.gmpe_lt.get_branch_by_id(branch_id)
			trt = gmpe_branch.parent_branchset.applyToTectonicRegionType
			gmpe_name = gmpe_branch.value
			weight = gmpe_branch.weight
			if src.tectonic_region_type == trt:
				branch_path = []
				for branch_id in smlt_path[1:]:
					smlt_branch = self.source_model_lt.get_branch_by_id(branch_id)
					if smlt_branch.parent_branchset.filter_source(src):
						branch_path.append(branch_id)
						weight *= smlt_branch.weight
				branch_path = [bp.split('--')[-1] for bp in branch_path]
				curve_name = '--'.join(branch_path)
				curve_path = self._get_curve_path(source_model_name, trt,
												src.source_id, gmpe_name)
				shcf = self.read_oq_shcf(curve_name, curve_path, calc_id=calc_id)
				return shcf, weight

	def read_oq_realization(self, source_model_name, smlt_path, gmpelt_path,
							calc_id=None):
		"""
		Read results of a particular logic-tree sample (by summing hazard
		curves of individual sources)

		:param source_model_name:
			str, name of source model
		:param smlt_path:
			list of branch ids (strings), source-model logic tree path
		:param gmpelt_path:
			list of branch ids (strings), ground-motion logic tree path
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)

		:return:
			instance of :class:`rshalib.result.SpectralHazardCurveField`
		"""
		source_model = self.get_source_model_by_name(source_model_name)
		summed_shcf = None
		for src in source_model.sources:
			shcf, weight = self.read_oq_realization_by_source(source_model_name,
									src, smlt_path, gmpelt_path, calc_id=calc_id)
			if shcf:
				if summed_shcf is None:
					summed_shcf = shcf
				else:
					summed_shcf += shcf
		return summed_shcf

	def read_oq_shcft(self, skip_samples=0, write_xml=False, calc_id=None):
		"""
		Read results corresponding to a number of logic-tree samples

		:param skip_samples:
			int, number of samples to skip (default: 0)
		:param write_xml:
			bool, whether or not to write spectral hazard curve fields
			corresponding to different logic-tree realizations to xml
			(default: False)
		:param calc_id:
			int or str, OpenQuake calculation ID (default: None, will
				be determined automatically)

		:return:
			instance of :class:`rshalib.result.SpectralHazardCurveFieldTree`
		"""
		shcf_list, weights, branch_names = [], [], []
		for sample_idx, (sm_name, smlt_path, gmpelt_path, weight) in \
				enumerate(self.sample_logic_tree_paths(self.num_lt_samples,
												skip_samples=skip_samples)):
			num_lt_samples = self.num_lt_samples or self.get_num_paths()
			fmt = "%%0%dd" % len(str(num_lt_samples))
			curve_name = "rlz-" + fmt % (sample_idx + 1 + skip_samples)
			xml_filespec = self.get_oq_shcf_filespec(curve_name, calc_id=calc_id)

			if write_xml is False and os.path.exists(xml_filespec):
				shcf = self.read_oq_shcf(curve_name, calc_id=calc_id)
			else:
				sm_name = os.path.splitext(sm_name)[0]
				shcf = self.read_oq_realization(sm_name, smlt_path, gmpelt_path,
												calc_id=calc_id)
				shcf.model_name = "%s, LT sample %s"
				shcf.model_name %= (self.name, fmt % (sample_idx + 1 + skip_samples))
				self.smlt_path = " -- ".join(smlt_path)
				self.gmpelt_path = " -- ".join(gmpelt_path)
				self.write_oq_shcf(shcf, "", "", "", "", curve_name, calc_id=calc_id)
			shcf_list.append(shcf)
			weights.append(weight)
			self.smlt_path = ""
			self.gmpelt_path = ""
		shcft = SpectralHazardCurveFieldTree.from_branches(shcf_list, self.name,
														branch_names=branch_names,
														weights=weights)
		return shcft

	def read_oq_source_realizations(self, source_model_name, src, gmpe_name="",
									calc_id=None, verbose=False):
		"""
		Read results for all realizations of a particular source

		:param source_model_name:
			str, name of source model
		:param src:
			instance of
			:class:`rshalib.source.[Point|Area|SimpleFault|ComplexFault]Source`
		:param gmpe_name:
			str, name of GMPE (default: "", will read all GMPEs)
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)
		:param verbose:
			bool, whether or not to print some progress information
			(default: False)

		:return:
			(shcf_list, weights) tuple:
			- shcf_list: list of instances of :class:`SpectralHazardCurveField`
			- weights: list with corresponding weights (decimals)
		"""
		shcf_list, weights = [], []
		trt = src.tectonic_region_type
		if not gmpe_name:
			gmpe_weight_iterable = self.gmpe_lt.gmpe_system_def[trt]
		else:
			## use dummy weight
			gmpe_weight_iterable = [(gmpe_name, 1)]
		for gmpe_name, gmpe_weight in gmpe_weight_iterable:
			for (branch_path, smlt_weight) in \
					self.source_model_lt.enumerate_branch_paths_by_source(
														source_model_name,src):
				branch_path = [b.branch_id.split('--')[-1] for b in branch_path]
				curve_name = '--'.join(branch_path)
				curve_path = self._get_curve_path(source_model_name, trt,
												src.source_id, gmpe_name)
				shcf = self.read_oq_shcf(curve_name, curve_path, calc_id=calc_id)
				shcf_list.append(shcf)
				weights.append(gmpe_weight * smlt_weight)
		return shcf_list, weights

	def read_oq_source_shcft(self, source_model_name, src, calc_id=None,
							verbose=False):
		"""
		Read results for all realizations of a particular source

		:param source_model_name:
			str, name of source model
		:param src:
			instance of
			:class:`rshalib.source.[Point|Area|SimpleFault|ComplexFault]Source`
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)
		:param verbose:
			bool, whether or not to print some progress information
			(default: False)

		:return:
			instance of :class:`SpectralHazardCurveField`
		"""
		shcf_list, weights = self.read_oq_source_realizations(source_model_name,
											src, calc_id=calc_id, verbose=verbose)
		shcft = SpectralHazardCurveFieldTree.from_branches(shcf_list, src.name,
															weights=weights)
		return shcft

	def enumerate_correlated_sources(self, source_model, trt=None):
		"""
		Enumerate correlated sources for a particular source model

		:param source_model:
			instance of :class:`rshalib.source.SourceModel`
		:param trt:
			str, tectonic region type (default: None)

		:return:
			generator object yielding lists of sources
		"""
		for src_ids in self.source_model_lt.list_correlated_sources(source_model):
			sources = [source_model[src_id] for src_id in src_ids]
			if trt:
				sources = [src for src in sources if src.tectonic_region_type == trt]
				if len(sources) == 0:
					continue
			yield sources

	def read_oq_correlated_source_realizations(self, source_model_name, src_list,
												gmpe_name="", calc_id=None):
		"""
		Read results for all realizations of a list of correlated sources

		:param source_model_name:
			str, name of source model
		:param src_list:
			list with instances of
			:class:`rshalib.source.[Point|Area|SimpleFault|ComplexFault]Source`
		:param gmpe_name:
			str, name of GMPE (default: "", will read all GMPEs)
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)

		:return:
			(shc_list, weights) tuple:
			- shc_list: list of instances of :class:`SpectralHazardCurveField`
			- weights: list with corresponding weights (decimals)
		"""
		from ..openquake import parse_spectral_hazard_curve_field

		shcf_list, weights = [], []
		src0 = src_list[0]
		trt = src0.tectonic_region_type
		if not gmpe_name:
			gmpe_weight_iterable = self.gmpe_lt.gmpe_system_def[trt]
		else:
			## use dummy weight
			gmpe_weight_iterable = [(gmpe_name, 1)]
		for gmpe_name, gmpe_weight in gmpe_weight_iterable:
			for (branch_path, smlt_weight) in \
					self.source_model_lt.enumerate_branch_paths_by_source(
														source_model_name, src0):
				branch_path = [b.branch_id.split('--')[-1] for b in branch_path]
				curve_name = '--'.join(branch_path)
				## Sum identical samples for each of the correlated sources
				for i, src in enumerate(src_list):
					hc_folder = self.get_oq_hc_folder_decomposed(source_model_name,
									trt, src.source_id, gmpe_name, calc_id=calc_id)
					xml_filename = "hazard_curve_multi-%s.xml" % curve_name
					#print(xml_filename)
					xml_filespec = os.path.join(hc_folder, xml_filename)
					shcf = parse_spectral_hazard_curve_field(xml_filespec)
					if i == 0:
						summed_shcf = shcf
					else:
						summed_shcf += shcf
				summed_shcf.set_site_names(self.get_generic_sites())
				shcf_list.append(summed_shcf)
				weights.append(gmpe_weight * smlt_weight)
		return shcf_list, weights

	def get_oq_mean_shcf_by_source(self, source_model_name, src, gmpe_name="",
									write_xml=False, calc_id=None):
		"""
		Compute or read mean spectral hazard curve field for a particular
		source

		:param source_model_name:
			str, name of source model
		:param src:
			instance of
			:class:`rshalib.source.[Point|Area|SimpleFault|ComplexFault]Source`
		:param gmpe_name:
			str, name of GMPE (default: "", will read all GMPEs)
		:param write_xml:
			bool, whether or not to write mean spectral hazard curve field
			to xml. If mean shcf already exists, it will be overwritten
			(default: False)
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)

		:return:
			instance of :class:`SpectralHazardCurveField`
		"""
		curve_name = "mean"
		trt = src.tectonic_region_type
		curve_path = self._get_curve_path(source_model_name, trt, src.source_id,
											gmpe_name)
		xml_filespec = self.get_oq_shcf_filespec(curve_name, curve_path=curve_path,
												calc_id=calc_id)

		if write_xml is False and os.path.exists(xml_filespec):
			mean_shcf = self.read_oq_shcf(curve_name, curve_path=curve_path,
											calc_id=calc_id)
		else:
			shcf_list, weights = self.read_oq_source_realizations(source_model_name,
										src, gmpe_name=gmpe_name, calc_id=calc_id)
			mean_shcf = None
			for i in range(len(shcf_list)):
				shcf = shcf_list[i]
				weight = float(weights[i])
				if i == 0:
					mean_shcf = shcf * weight
				else:
					mean_shcf += (shcf * weight)
			mean_shcf.model_name = "%s weighted mean" % src.source_id

			self.write_oq_shcf(mean_shcf, source_model_name, trt, src.source_id,
								gmpe_name, curve_name, calc_id=calc_id)

		return mean_shcf

	def get_oq_mean_shcf_by_correlated_sources(self, source_model_name, src_list,
									gmpe_name="", write_xml=False, calc_id=None):
		"""
		Compute or read mean spectral hazard curve field for a list of
		correlated sources

		:param source_model_name:
			str, name of source model
		:param src_list:
			list with
			instances of :class:`rshalib.source.[Point|Area|SimpleFault|ComplexFault]Source`
		:param gmpe_name:
			str, name of GMPE (default: "", will read all GMPEs)
		:param write_xml:
			bool, whether or not to write mean spectral hazard curve field
			to xml. If mean shcf already exists, it will be overwritten
			(default: False)
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)

		:return:
			instance of :class:`SpectralHazardCurveField`
		"""
		curve_name = "mean"
		src0 = src_list[0]
		trt = src0.tectonic_region_type
		curve_path = self._get_curve_path(source_model_name, trt, src0.source_id,
										gmpe_name)
		xml_filespec = self.get_oq_shcf_filespec(curve_name, curve_path=curve_path,
												calc_id=calc_id)

		if write_xml is False and os.path.exists(xml_filespec):
			mean_shcf = self.read_oq_shcf(curve_name, curve_path=curve_path,
										calc_id=calc_id)
		else:
			shcf_list, weights = self.read_oq_correlated_source_realizations(
											source_model_name, src_list,
											gmpe_name=gmpe_name, calc_id=calc_id)
			mean_shcf = None
			for i in range(len(shcf_list)):
				shcf = shcf_list[i]
				weight = float(weights[i])
				if i == 0:
					mean_shcf = shcf * weight
				else:
					mean_shcf += (shcf * weight)
			mean_shcf.model_name = "%s weighted mean"
			mean_shcf.model_name %= '+'.join([src.source_id for src in src_list])

			self.write_oq_shcf(mean_shcf, source_model_name, trt, src0.source_id,
								gmpe_name, curve_name, calc_id=calc_id)

		return mean_shcf

	def get_oq_mean_shcf_by_source_model(self, source_model, trt="", gmpe_name="",
									write_xml=False, respect_gm_trt_correlation=False,
									calc_id=None):
		"""
		Compute or read mean spectral hazard curve field for a particular
		source model by summing mean shcf's of individual sources

		:param source_model:
			instance of :class:`rshalib.source.SourceModel`
		:param trt:
			str, tectonic region type
			(default: "")
		:param gmpe_name:
			str, name of GMPE
			(default: "", will read all GMPEs)
		:param write_xml:
			bool, whether or not to write mean spectral hazard curve field to xml.
			If mean shcf already exists, it will be overwritten
			(default: False)
		:param respect_gm_trt_correlation:
			bool, whether or not mean should be computed separately for each trt,
			in order to respect the correlation between sources in each trt in the
			ground_motion logic tree.
			(default: False)
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)

		:return:
			instance of :class:`SpectralHazardCurveField`
		"""
		curve_name = "mean"
		curve_path = self._get_curve_path(source_model.name, trt, "", gmpe_name)
		xml_filespec = self.get_oq_shcf_filespec(curve_name, curve_path=curve_path,
												calc_id=calc_id)

		if write_xml is False and os.path.exists(xml_filespec):
			summed_shcf = self.read_oq_shcf(curve_name, curve_path=curve_path,
											calc_id=calc_id)
		else:
			if respect_gm_trt_correlation:
				## More explicit calculation
				## Calculate mean for each trt separately, in order to respect
				## correlation between sources in each trt in the ground-motion
				## logic tree
				summed_shcf = None
				for _trt in source_model.get_tectonic_region_types():
					if not trt or _trt == trt:
						trt_shcf = None
						for _gmpe_name, gmpe_weight in self.gmpe_lt.gmpe_system_def[trt]:
							if not gmpe_name or _gmpe_name == gmpe_name:
								gmpe_shcf = None
								for src_list in self.enumerate_correlated_sources(
															source_model, _trt):
									if len(src_list) == 1:
										[src] = src_list
										shcf = self.get_oq_mean_shcf_by_source(
														source_model.name, src,
														gmpe_name=gmpe_name,
														write_xml=write_xml,
														calc_id=calc_id)
									else:
										shcf = self.get_oq_mean_shcf_by_correlated_sources(
													source_model.name, src_list,
													gmpe_name=gmpe_name,
													write_xml=write_xml,
													calc_id=calc_id)
									if shcf:
										if gmpe_shcf is None:
											gmpe_shcf = shcf
										else:
											gmpe_shcf += shcf
								gmpe_shcf *= gmpe_weight
								if trt_shcf is None:
									trt_shcf = gmpe_shcf
								else:
									trt_shcf += gmpe_shcf
						if summed_shcf is None:
							summed_shcf = trt_shcf
						else:
							summed_shcf += trt_shcf

			else:
				## Simpler calculation:
				## Compute mean for each source and sum
				summed_shcf = None
				## Note that correlation of sources does not matter for computing the mean
				## It may even cause problems
				#for src_list in self.enumerate_correlated_sources(source_model, trt=trt):
				#	if len(src_list) == 1:
				#		[src] = src_list
				#		shcf = self.get_oq_mean_shcf_by_source(source_model.name, src, gmpe_name=gmpe_name, write_xml=write_xml, calc_id=calc_id)
				#	else:
				#		shcf = self.get_oq_mean_shcf_by_correlated_sources(source_model.name, src_list, gmpe_name=gmpe_name, write_xml=write_xml, calc_id=calc_id)
				for src in [src for src in source_model
							if src.tectonic_region_type == trt or trt == ""]:
					shcf = self.get_oq_mean_shcf_by_source(source_model.name, src,
															gmpe_name=gmpe_name,
															write_xml=write_xml,
															calc_id=calc_id)
					if summed_shcf is None:
						summed_shcf = shcf
					else:
						summed_shcf += shcf

			summed_shcf.model_name = "%s weighted mean" % source_model.name

			self.write_oq_shcf(summed_shcf, source_model.name, trt, "", gmpe_name,
							curve_name, calc_id=calc_id)

		return summed_shcf

	def get_oq_mean_shcf(self, trt="", gmpe_name="", write_xml=False,
						respect_gm_trt_correlation=False, calc_id=None):
		"""
		Read mean spectral hazard curve field of entire logic tree.
		If mean shcf does not exist, it will be computed from the decomposed
		shcf's. If it exists, it will be read if write_xml is False

		:param trt:
			str, tectonic region type (default: "")
		:param gmpe_name:
			str, name of GMPE (default: "", will read all GMPEs)
		:param write_xml:
			bool, whether or not to write mean spectral hazard curve field
			to xml. If mean shcf already exists, it will be overwritten
			(default: False)
		:param respect_gm_trt_correlation:
			bool, whether or not mean should be computed separately for
			each trt, in order to respect the correlation between sources
			in each trt in the ground_motion logic tree.
			(default: False)
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)

		:return:
			instance of :class:`SpectralHazardCurveField`
		"""
		curve_name = "mean"
		if trt or gmpe_name:
			curve_path = self._get_curve_path("", trt, "", gmpe_name)
		else:
			curve_path = ""
		xml_filespec = self.get_oq_shcf_filespec(curve_name, curve_path=curve_path,
												calc_id=calc_id)

		if write_xml is False and os.path.exists(xml_filespec):
			mean_shcf = self.read_oq_shcf(curve_name, curve_path=curve_path,
										calc_id=calc_id)
		else:
			mean_shcf = None
			for source_model, somo_weight in self.source_model_lt.source_model_pmf:
				somo_weight = float(somo_weight)
				source_model_shcf = self.get_oq_mean_shcf_by_source_model(
										source_model, trt=trt, gmpe_name=gmpe_name,
										write_xml=write_xml,
										respect_gm_trt_correlation=respect_gm_trt_correlation,
										calc_id=calc_id)
				if mean_shcf is None:
					mean_shcf = source_model_shcf * somo_weight
				else:
					mean_shcf += (source_model_shcf * somo_weight)
			mean_shcf.model_name = "Logic-tree weighted mean"
			self.write_oq_shcf(mean_shcf, "", trt, "", gmpe_name, curve_name,
							calc_id=calc_id)
		return mean_shcf

	def calc_oq_shcf_percentiles_decomposed(self, percentile_levels):
		"""
		Compute percentiles of spectral hazard curve fields

		:param percentile_levels:
			list or array with percentile levels in the range 0 - 100

		:return:
			percentiles of hazard values: 4-D array [i,k,l,p]
		"""
		total_percs = None
		for source_model, somo_weight in self.source_model_lt.source_model_pmf:
			print(source_model.name)
			somo_percs = None
			for src in source_model.sources:
				print(src.source_id)
				src_shcft = self.read_oq_source_shcft(source_model.name, src)
				percs = src_shcft.calc_percentiles_epistemic(percentile_levels,
													weighted=True, interpol=True)
				if somo_percs is None:
					somo_percs = percs
				else:
					somo_percs += percs
			if total_percs is None:
				total_percs = somo_percs * somo_weight
			else:
				total_percs += (somo_percs * somo_weight)
		return total_percs

	def calc_shcf_stats(self, num_samples):
		pass

	# TODO: methods to compute minimum / maximum scenarios

	def read_oq_deagg_realization_by_source(self, source_model_name, src,
											smlt_path, gmpelt_path, site,
											calc_id=None, dtype='d'):
		"""
		Read deaggregation results of a particular logictree sample
		for 1 source

		:param source_model_name:
			str, name of source model
		:param src:
			instance of
			:class:`rshalib.source.[Point|Area|SimpleFault|ComplexFault]Source`
		:param smlt_path:
			list of branch ids (strings), source-model logic tree path
		:param gmpelt_path:
			list of branch ids (strings), ground-motion logic tree path
		:param site:
			instance of :class:`GenericSite` or :class:`SoilSite`
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)
		:param dtype:
			str, precision of deaggregation matrix
			(default: 'd')

		:return:
			(sdc, weight) tuple:
			- sdc: instance of :class:`rshalib.result.SpectralDeaggregationCurve`
			- weight: decimal
		"""
		from ..openquake import parse_spectral_hazard_curve_field

		for branch_id in gmpelt_path:
			gmpe_branch = self.gmpe_lt.get_branch_by_id(branch_id)
			trt = gmpe_branch.parent_branchset.applyToTectonicRegionType
			gmpe_name = gmpe_branch.value
			weight = gmpe_branch.weight
			if src.tectonic_region_type == trt:
				branch_path = []
				for branch_id in smlt_path[1:]:
					smlt_branch = self.source_model_lt.get_branch_by_id(branch_id)
					if smlt_branch.parent_branchset.filter_source(src):
						branch_path.append(branch_id)
						weight *= smlt_branch.weight
				branch_path = [bp.split('--')[-1] for bp in branch_path]
				curve_name = '--'.join(branch_path)
				curve_path = self._get_curve_path(source_model_name, trt,
												src.source_id, gmpe_name)
				sdc = self.read_oq_disagg_matrix_multi(curve_name, site,
										curve_path, calc_id=calc_id, dtype=dtype)
				return sdc, weight

	def read_oq_deagg_realization(self, source_model_name, smlt_path, gmpelt_path,
								site, calc_id=None, dtype='d'):
		"""
		Read deaggregation results of a particular logic-tree sample
		(by summing deaggregation curves of individual sources).

		:param source_model_name:
			str, name of source model
		:param smlt_path:
			list of branch ids (strings), source-model logic tree path
		:param gmpelt_path:
			list of branch ids (strings), ground-motion logic tree path
		:param site:
			instance of :class:`GenericSite` or :class:`SoilSite`
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)
		:param dtype:
			str, precision of deaggregation matrix
			(default: 'd')

		:return:
			instance of :class:`rshalib.result.SpectralDeaggregationCurve`
		"""
		import gc

		source_model = self.get_source_model_by_name(source_model_name)
		for i, src in enumerate(source_model.sources):
			sdc, weight = self.read_oq_deagg_realization_by_source(source_model_name,
												src, smlt_path, gmpelt_path,
												site, calc_id=calc_id, dtype=dtype)
			if i == 0:
				## Create empty deaggregation matrix
				bin_edges = self.get_deagg_bin_edges(sdc.mag_bin_width,
								sdc.dist_bin_width, sdc.lon_bin_width, sdc.neps)
				mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, trts = bin_edges
				num_periods = len(sdc.periods)
				num_intensities = len(sdc.return_periods)
				summed_deagg_matrix = SpectralDeaggregationCurve.construct_empty_deagg_matrix(
											num_periods, num_intensities, bin_edges,
											sdc.deagg_matrix.__class__, dtype)

			max_mag_idx = min(sdc.nmags, len(mag_bins) - 1)
			min_lon_idx = int((sdc.min_lon - lon_bins[0]) / sdc.lon_bin_width)
			max_lon_idx = min_lon_idx + sdc.nlons
			min_lat_idx = int((sdc.min_lat - lat_bins[0]) / sdc.lat_bin_width)
			max_lat_idx = min_lat_idx + sdc.nlats
			try:
				trt_idx = trts.index(src.source_id)
			except:
				trt_idx = trts.index(src.tectonic_region_type)

			summed_deagg_matrix[:,:,:max_mag_idx,:,min_lon_idx:max_lon_idx,min_lat_idx:max_lat_idx,:,trt_idx] += sdc.deagg_matrix[:,:,:max_mag_idx,:,:,:,:,0]
			del sdc.deagg_matrix
			gc.collect()
		#intensities = np.zeros(sdc.intensities.shape)
		summed_sdc = SpectralDeaggregationCurve(bin_edges, summed_deagg_matrix,
									sdc.site, sdc.imt, sdc.intensities,
									sdc.periods, sdc.return_periods, sdc.timespan)
		summed_sdc.model_name = "%s weighted mean" % source_model_name

		return summed_sdc

	def get_oq_mean_sdc_from_lt_samples(self, site, interpolate_rp=True,
										interpolate_matrix=False, skip_samples=0,
										write_xml=False, calc_id=None, dtype='d'):
		"""
		Read or compute mean spectral deaggregation curve based on
		logic-tree samples

		:param site:
			instance of :class:`GenericSite` or :class:`SoilSite`
		:param interpolate_rp:
			bool, whether or not to interpolate each logic-tree sample at
			the return periods defined in logic tree
			(default: True)
		:param interpolate_matrix:
			bool, whether or not the deaggregation matrix should be
			interpolated at the intensities interpolated from the
			hazard curve. If False, the nearest slices will be selected
			(default: False)
		:param skip_samples:
			int, number of samples to skip
			(default: 0)
		:param write_xml:
			bool, whether or not to write spectral hazard curve fields
			corresponding to different logic-tree realizations to xml
			(default: False)
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)
		:param dtype:
			str, precision of deaggregation matrix
			(default: 'd')

		:return:
			instance of :class:`rshalib.result.SpectralDeaggregationCurve`
		"""
		mean_curve_name = "mean_lt"
		curve_path = ""
		xml_filespec = self.get_oq_sdc_filespec(mean_curve_name, site,
										curve_path=curve_path, calc_id=calc_id)
		if write_xml is False and os.path.exists(xml_filespec):
			mean_sdc = self.read_oq_disagg_matrix_multi(mean_curve_name, site,
							curve_path=curve_path, calc_id=calc_id, dtype=dtype)
		else:
			mean_sdc = None
			for sample_idx, (sm_name, smlt_path, gmpelt_path, weight) in \
					enumerate(self.sample_logic_tree_paths(self.num_lt_samples,
													skip_samples=skip_samples)):
				sm_name = os.path.splitext(sm_name)[0]
				num_lt_samples = self.num_lt_samples or self.get_num_paths()
				fmt = "%%0%dd" % len(str(num_lt_samples))
				curve_name = "rlz-" + fmt % (sample_idx + 1 + skip_samples)
				curve_path = ""
				xml_filespec = self.get_oq_sdc_filespec(curve_name, site,
										curve_path=curve_path, calc_id=calc_id)

				if write_xml is False and os.path.exists(xml_filespec):
					summed_sdc = self.read_oq_disagg_matrix_multi(curve_name, site,
													curve_path=curve_path,
													calc_id=calc_id, dtype=dtype)
				else:
					summed_sdc = self.read_oq_deagg_realization(sm_name, smlt_path,
									gmpelt_path, site, calc_id=calc_id, dtype=dtype)

				if interpolate_rp:
					## Read shcf corresponding to logic-tree sample, and use it to slice
					## spectral deaggregation curve at return periods defined in logic tree
					shcf_filespec = self.get_oq_shcf_filespec(curve_name,
										curve_path=curve_path, calc_id=calc_id)
					if write_xml is False and os.path.exists(shcf_filespec):
						shcf = self.read_oq_shcf(curve_name, curve_path=curve_path,
												calc_id=calc_id)
					else:
						shcf = self.read_oq_realization(sm_name, smlt_path,
													gmpelt_path, calc_id=calc_id)
						self.write_oq_shcf(shcf, "", "", "", "", curve_name,
											calc_id=calc_id)
					shc = shcf.getSpectralHazardCurve(site_spec=(site.lon, site.lat))
					summed_sdc = summed_sdc.slice_return_periods(self.return_periods,
										shc, interpolate_matrix=interpolate_matrix)

				if not os.path.exists(xml_filespec):
					self.smlt_path = smlt_path
					self.gmpelt_path = gmpelt_path
					self.write_oq_disagg_matrix_multi(summed_sdc, "", "", "", "",
													curve_name, calc_id=calc_id)

				if mean_sdc is None:
					mean_sdc = summed_sdc
					mean_sdc.model_name = "Logic-tree weighted mean"
				else:
					mean_sdc.deagg_matrix += summed_sdc.deagg_matrix

			mean_sdc.deagg_matrix /= (sample_idx + 1)
			self.write_oq_disagg_matrix_multi(mean_sdc, "", "", "", "",
											mean_curve_name, calc_id=calc_id)

		return mean_sdc

	def read_oq_source_deagg_realizations(self, source_model_name, src, site,
										gmpe_name="", calc_id=None, dtype='d',
										verbose=False):
		"""
		Read deaggregation results for all realizations of a particular
		source

		:param source_model_name:
			str, name of source model
		:param src:
			instance of
			:class:`rshalib.source.[Point|Area|SimpleFault|ComplexFault]Source`
		:param site:
			instance of :class:`GenericSite` or :class:`SoilSite`
		:param gmpe_name:
			str, name of GMPE
			(default: "", will read all GMPEs)
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)
		:param dtype:
			str, precision of deaggregation matrix
			(default: 'd')
		:param verbose:
			bool, whether or not to print some progress information
			(default: False)

		:return:
			generator yielding (sdc, weight) tuples:
			- sdc: instance of :class:`SpectralDeaggregationCurve`
			- weight: corresponding weight (decimal)
		"""
		trt = src.tectonic_region_type
		if not gmpe_name:
			gmpe_weight_iterable = self.gmpe_lt.gmpe_system_def[trt]
		else:
			## use dummy weight
			gmpe_weight_iterable = [(gmpe_name, 1)]
		for gmpe_name, gmpe_weight in gmpe_weight_iterable:
			for (branch_path, smlt_weight) in \
					self.source_model_lt.enumerate_branch_paths_by_source(
														source_model_name, src):
				branch_path = [b.branch_id.split('--')[-1] for b in branch_path]
				curve_name = '--'.join(branch_path)
				curve_path = self._get_curve_path(source_model_name, trt,
												src.source_id, gmpe_name)
				sdc = self.read_oq_disagg_matrix_multi(curve_name, site,
										curve_path, calc_id=calc_id, dtype=dtype)
				weight = gmpe_weight * smlt_weight
				yield (sdc, weight)

	def get_oq_mean_sdc_by_source(self, source_model_name, src, site,
						gmpe_name="", mean_shc=None, interpolate_matrix=False,
						calc_id=None, dtype='d', write_xml=False, verbose=False):
		"""
		Read or compute mean spectral deaggregation curve for a particular
		source

		:param source_model_name:
			str, name of source model
		:param src:
			instance of
			:class:`rshalib.source.[Point|Area|SimpleFault|ComplexFault]Source`
		:param site:
			instance of :class:`GenericSite` or :class:`SoilSite`
		:param gmpe_name:
			str, name of GMPE
			(default: "", will read all GMPEs)
		:param mean_shc:
			instance of :class:`rshalib.result.SpectralHazardCurve`
			If specified, sdc will be reduced to intensities corresponding
			to return periods of PSHA model tree
			(default: None).
		:param interpolate_matrix:
			bool, whether or not the deaggregation matrix should be
			interpolated at the intensities interpolated from the mean
			hazard curve. If False, the nearest slices will be selected
			(default: False)
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)
		:param dtype:
			str, precision of deaggregation matrix
			(default: 'd')
		:param write_xml:
			bool, whether or not to write mean spectral deaggregation curve
			to xml. If mean sdc exists, it will be overwritten
			(default: False)
		:param verbose:
			bool, wheter or not to print some progress info
			(default: False)

		:return:
			instance of :class:`SpectralDeaggregationCurve`
		"""
		import gc

		curve_name = "mean"
		trt = src.tectonic_region_type
		curve_path = self._get_curve_path(source_model_name, trt, src.source_id, "")
		xml_filespec = self.get_oq_sdc_filespec(curve_name, site,
										curve_path=curve_path, calc_id=calc_id)

		if write_xml is False and os.path.exists(xml_filespec):
			mean_sdc = self.read_oq_disagg_matrix_multi(curve_name, site,
							curve_path=curve_path, calc_id=calc_id, dtype=dtype)
		else:
			for i, (sdc, weight) in \
					enumerate(self.read_oq_source_deagg_realizations(source_model_name,
												src, site, gmpe_name=gmpe_name,
												calc_id=calc_id, dtype=dtype)):
				weight = float(weight)
				if verbose:
					print(i)
				if i == 0:
					## Create empty deaggregation matrix
					## max_mag may be different
					mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, trts = sdc.bin_edges
					min_mag = mag_bins[0]
					max_mag = self.source_model_lt.get_source_max_mag(source_model_name,
																	src)
					dmag = (np.ceil((max_mag - min_mag) / sdc.mag_bin_width)
							* sdc.mag_bin_width)
					nmags = int(round(dmag / sdc.mag_bin_width))
					mag_bins = min_mag + sdc.mag_bin_width * np.arange(nmags + 1)
					#mag_bins = sdc.mag_bin_width * np.arange(
					#	int(np.floor(min_mag / sdc.mag_bin_width)),
					#	int(np.ceil(max_mag / sdc.mag_bin_width) + 1)
					#)
					bin_edges = (mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, trts)

					num_periods = len(sdc.periods)
					num_intensities = len(sdc.return_periods)
					mean_deagg_matrix = SpectralDeaggregationCurve.construct_empty_deagg_matrix(
													num_periods, num_intensities, bin_edges,
													sdc.deagg_matrix.__class__, dtype)

				mean_deagg_matrix[:,:,:min(nmags, sdc.nmags)] += (sdc.deagg_matrix[:,:,:min(nmags, sdc.nmags)] * weight)
				del sdc.deagg_matrix
				gc.collect()

			mean_sdc = SpectralDeaggregationCurve(bin_edges, mean_deagg_matrix,
											sdc.site, sdc.imt, sdc.intensities,
											sdc.periods, sdc.return_periods,
											sdc.timespan)
			mean_sdc.model_name = "%s weighted mean" % src.source_id
			if mean_shc:
				mean_sdc = mean_sdc.slice_return_periods(self.return_periods,
								mean_shc, interpolate_matrix=interpolate_matrix)

			self.write_oq_disagg_matrix_multi(mean_sdc, source_model_name,
										src.tectonic_region_type, src.source_id,
										"", curve_name, calc_id=calc_id)

		return mean_sdc

	def get_oq_mean_sdc_by_source_model(self, source_model_name, site, trt="",
										gmpe_name="", mean_shc=None,
										interpolate_matrix=False, calc_id=None,
										dtype='d', write_xml=False, verbose=False):
		"""
		Read or compute mean spectral deaggregation curve for a particular source model

		:param source_model_name:
			str, name of source model
		:param site:
			instance of :class:`GenericSite` or :class:`SoilSite`
		:param trt:
			str, tectonic region type
			(default: "")
		:param gmpe_name:
			str, name of GMPE
			(default: "", will read all GMPEs)
		:param mean_shc:
			instance of :class:`rshalib.result.SpectralHazardCurve`
			If specified, sdc will be reduced to intensities corresponding
			to return periods of PSHA model tree
			(default: None).
		:param interpolate_matrix:
			bool, whether or not the deaggregation matrix should be
			interpolated at the intensities interpolated from the mean
			hazard curve. If False, the nearest slices will be selected
			(default: False)
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)
		:param dtype:
			str, precision of deaggregation matrix
			(default: 'd')
		:param write_xml:
			bool, whether or not to write mean spectral deaggregation curve
			to xml. If mean sdc exists, it will be overwritten
			(default: False)
		:param verbose:
			bool, whether or not to print some progress info
			(default: False)

		:return:
			instance of :class:`SpectralDeaggregationCurve`
		"""
		import gc

		curve_name = "mean"
		curve_path = self._get_curve_path(source_model_name, trt, "", gmpe_name)
		xml_filespec = self.get_oq_sdc_filespec(curve_name, site,
										curve_path=curve_path, calc_id=calc_id)

		if write_xml is False and os.path.exists(xml_filespec):
			summed_sdc = self.read_oq_disagg_matrix_multi(curve_name, site,
										curve_path=curve_path, calc_id=calc_id)
		else:
			source_model = self.get_source_model_by_name(source_model_name)
			sources = source_model.get_sources_by_trt(trt)
			for i, src in enumerate(sources):
				if verbose:
					print(src.source_id)
				sdc = self.get_oq_mean_sdc_by_source(source_model_name, src, site,
											gmpe_name=gmpe_name, mean_shc=mean_shc,
											interpolate_matrix=interpolate_matrix,
											calc_id=calc_id, dtype=dtype,
											write_xml=write_xml, verbose=verbose)
				if i == 0:
					## Create empty deaggregation matrix
					bin_edges = self.get_deagg_bin_edges(sdc.mag_bin_width,
											sdc.dist_bin_width, sdc.lon_bin_width,
											sdc.neps)
					mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, trts = bin_edges
					trts = [src.source_id for src in sources]
					bin_edges = (mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, trts)
					num_periods = len(sdc.periods)
					num_intensities = len(sdc.return_periods)
					summed_deagg_matrix = SpectralDeaggregationCurve.construct_empty_deagg_matrix(
												num_periods, num_intensities, bin_edges,
												sdc.deagg_matrix.__class__, dtype)

				max_mag_idx = min(sdc.nmags, len(mag_bins) - 1)
				min_lon_idx = int((sdc.min_lon - lon_bins[0]) / sdc.lon_bin_width)
				max_lon_idx = min_lon_idx + sdc.nlons
				min_lat_idx = int((sdc.min_lat - lat_bins[0]) / sdc.lat_bin_width)
				max_lat_idx = min_lat_idx + sdc.nlats
				#trt_idx = trts.index(src.source_id)
				summed_deagg_matrix[:,:,:max_mag_idx,:,min_lon_idx:max_lon_idx,min_lat_idx:max_lat_idx,:,i] += sdc.deagg_matrix[:,:,:max_mag_idx,:,:,:,:,0]
				del sdc.deagg_matrix
				gc.collect()
			#intensities = np.zeros(sdc.intensities.shape)
			summed_sdc = SpectralDeaggregationCurve(bin_edges, summed_deagg_matrix,
												sdc.site, sdc.imt, sdc.intensities,
												sdc.periods, sdc.return_periods,
												sdc.timespan)
			summed_sdc.model_name = "%s weighted mean" % source_model_name

			self.write_oq_disagg_matrix_multi(summed_sdc, source_model.name, trt,
									"", gmpe_name, curve_name, calc_id=calc_id)

		return summed_sdc

	def get_oq_mean_sdc(self, site, trt="", gmpe_name="", mean_shc=None,
						interpolate_matrix=False, calc_id=None, dtype='d',
						write_xml=False, verbose=False):
		"""
		Read mean spectral deaggregation curve of the entire logic tree.
		If mean sdc does not exist, it will be computed from the decomposed
		deaggregation curves. If it exists, it will be read if write_xml
		is False

		:param site:
			instance of :class:`GenericSite` or :class:`SoilSite`
		:param trt:
			str, tectonic region type
			(default: "")
		:param gmpe_name:
			str, name of GMPE
			(default: "", will read all GMPEs)
		:param mean_shc:
			instance of :class:`rshalib.result.SpectralHazardCurve`
			If specified, sdc will be reduced to intensities corresponding
			to return periods of PSHA model tree
			(default: None).
		:param interpolate_matrix:
			bool, whether or not the deaggregation matrix should be
			interpolated at the intensities interpolated from the mean
			hazard curve. If False, the nearest slices will be selected
			(default: False)
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)
		:param dtype:
			str, precision of deaggregation matrix
			(default: 'd')
		:param write_xml:
			bool, whether or not to write mean spectral deaggregation curve
			to xml. If mean sdc exists, it will be overwritten
			(default: False)
		:param verbose:
			bool, whether or not to print some progress info
			(default: False)

		:return:
			instance of :class:`SpectralDeaggregationCurve`
		"""
		import gc

		curve_name = "mean"
		if trt or gmpe_name:
			curve_path = self._get_curve_path("", trt, "", gmpe_name)
		else:
			curve_path = ""
		xml_filespec = self.get_oq_sdc_filespec(curve_name, site,
										curve_path=curve_path, calc_id=calc_id)

		if write_xml is False and os.path.exists(xml_filespec):
			mean_sdc = self.read_oq_disagg_matrix_multi(curve_name, site,
										curve_path=curve_path, calc_id=calc_id)
		else:
			for i, (source_model, somo_weight) in \
					enumerate(self.source_model_lt.source_model_pmf):
				somo_weight = float(somo_weight)
				if verbose:
					print(source_model.name)
				sdc = self.get_oq_mean_sdc_by_source_model(source_model.name,
									site, mean_shc=mean_shc,
									interpolate_matrix=interpolate_matrix,
									calc_id=calc_id, dtype=dtype,
									write_xml=write_xml, verbose=verbose)
				if i == 0:
					## Create empty deaggregation matrix
					bin_edges = self.get_deagg_bin_edges(sdc.mag_bin_width,
									sdc.dist_bin_width, sdc.lon_bin_width, sdc.neps)
					num_periods = len(sdc.periods)
					num_intensities = len(sdc.return_periods)
					mean_deagg_matrix = SpectralDeaggregationCurve.construct_empty_deagg_matrix(
											num_periods, num_intensities, bin_edges,
											sdc.deagg_matrix.__class__, dtype)

				trt_bins = bin_edges[-1]
				if sdc.trt_bins == trt_bins:
					## trt bins correspond to source IDs
					mean_deagg_matrix[:,:,:,:,:,:,:,:] += (sdc.deagg_matrix * somo_weight)
				else:
					## trt bins correspond to tectonic region types
					for trt_idx, _trt in enumerate(trt_bins):
						src_idxs = []
						for src_idx, src_id in enumerate(sdc.trt_bins):
							src = source_model[src_id]
							if src.tectonic_region_type == _trt:
								src_idxs.append(src_idx)
						src_idxs = np.array(src_idxs)
						## Loop needed to avoid running out of memory...
						for t in range(num_periods):
							for l in range(num_intensities):
								# Note: something very strange happens here: if we slice t, l, and
								# src_idxs simultaneously, src_idxs becomes first dimension!
								mean_deagg_matrix[t,l,:,:,:,:,:,trt_idx] += (sdc.deagg_matrix[t,l][:,:,:,:,:,src_idxs].fold_axis(-1) * somo_weight)
						#mean_deagg_matrix[:,:,:,:,:,:,:,trt_idx] += (sdc.deagg_matrix[:,:,:,:,:,:,:,src_idxs].fold_axis(-1) * somo_weight)

				del sdc.deagg_matrix
				gc.collect()

			#intensities = np.zeros(sdc.intensities.shape)
			mean_sdc = SpectralDeaggregationCurve(bin_edges, mean_deagg_matrix,
												sdc.site, sdc.imt, sdc.intensities,
												sdc.periods, sdc.return_periods,
												sdc.timespan)
			mean_sdc.model_name = "Logic-tree weighted mean"

			self.write_oq_disagg_matrix_multi(mean_sdc, "", trt, "", gmpe_name,
											curve_name, calc_id=calc_id)

		return mean_sdc

	def delete_oq_shcf_stats(self, percentile_levels=[5, 16, 50, 84, 95],
							calc_id="oqhazlib", verbose=True, dry_run=False):
		"""
		Delete all xml files with hazard-curve statistics

		:param percentile_levels:
			list or array with percentile levels in the range 0 - 100
			(default: [5, 16, 50, 84, 100])
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: "oqhazlib")
		:param verbose:
			bool, whether or not to print filenames to be deleted
			(default: True)
		:param dry_run:
			bool, whether or not to run without actually deleting files
			(default: False, will delete xml files)
		"""
		curve_names = ["mean"]
		for perc_level in percentile_levels:
			curve_names.append("quantile-%.2f" % (perc_level / 100.))

		xml_filespecs = []
		for curve_name in curve_names:
			xml_filespec = self.get_oq_shcf_filespec_decomposed("", "", "", "",
													curve_name, calc_id=calc_id)
			xml_filespecs.append(xml_filespec)
			for source_model, _ in self.source_model_lt.source_model_pmf:
				xml_filespec = self.get_oq_shcf_filespec_decomposed(source_model.name,
										"", "", "", curve_name, calc_id=calc_id)
				xml_filespecs.append(xml_filespec)
				for trt in source_model.get_tectonic_region_types():
					xml_filespec = self.get_oq_shcf_filespec_decomposed(source_model.name,
												trt, "", "", curve_name, calc_id=calc_id)
					xml_filespecs.append(xml_filespec)
					gmpe_names = [""] + self.gmpe_lt.gmpe_system_def[trt].gmpe_names
					for gmpe_name in gmpe_names:
						xml_filespec = self.get_oq_shcf_filespec_decomposed(
											source_model.name, trt, "", gmpe_name,
											curve_name, calc_id=calc_id)
						xml_filespecs.append(xml_filespec)
						## If there is only one TRT
						xml_filespec = self.get_oq_shcf_filespec_decomposed(
											source_model.name, "", "", gmpe_name,
											curve_name, calc_id=calc_id)
						xml_filespecs.append(xml_filespec)
						for src_list in self.enumerate_correlated_sources(source_model,
																			trt):
							for src in src_list:
								xml_filespec = self.get_oq_shcf_filespec_decomposed(
											source_model.name, trt, src.source_id,
											gmpe_name, curve_name, calc_id=calc_id)
								xml_filespecs.append(xml_filespec)

			for trt in self.gmpe_lt.tectonicRegionTypes:
				xml_filespec = self.get_oq_shcf_filespec_decomposed("", trt, "",
												"", curve_name, calc_id=calc_id)
				xml_filespecs.append(xml_filespec)
				gmpe_names = self.gmpe_lt.gmpe_system_def[trt].gmpe_names
				for gmpe_name in gmpe_names:
					xml_filespec = self.get_oq_shcf_filespec_decomposed("", trt,
									"", gmpe_name, curve_name, calc_id=calc_id)
					xml_filespecs.append(xml_filespec)
					## If there is only one TRT
					xml_filespec = self.get_oq_shcf_filespec_decomposed("", "",
									"", gmpe_name, curve_name, calc_id=calc_id)
					xml_filespecs.append(xml_filespec)

		for xml_filespec in set(xml_filespecs):
			if os.path.exists(xml_filespec):
				if verbose:
					print("Deleting %s" % xml_filespec)
				if not dry_run:
					os.unlink(xml_filespec)

	def delete_oq_shcf_samples(self, calc_id="oqhazlib", verbose=True,
								dry_run=False):
		"""
		Delete xml files corresponding to logictree samples (hazard curves)

		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: "oqhazlib")
		:param verbose:
			bool, whether or not to print filenames to be deleted
			(default: True)
		:param dry_run:
			bool, whether or not to run without actually deleting files
			(default: False, will delete xml files)
		"""
		num_lt_samples = self.num_lt_samples or self.get_num_paths()
		for sample_idx in range(num_lt_samples):
			fmt = "%%0%dd" % len(str(num_lt_samples))
			curve_name = "rlz-" + fmt % (sample_idx + 1)
			xml_filespec = self.get_oq_shcf_filespec(curve_name, calc_id=calc_id)
			if os.path.exists(xml_filespec):
				if verbose:
					print("Deleting %s" % xml_filespec)
				if not dry_run:
					os.unlink(xml_filespec)

	def delete_oq_sdc_stats(self, calc_id="oqhazlib", verbose=True,
							dry_run=False):
		"""
		Delete all xml files with deaggregation statistics

		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: "oqhazlib")
		:param verbose:
			bool, whether or not to print filenames to be deleted
			(default: True)
		:param dry_run:
			bool, whether or not to run without actually deleting files
			(default: False, will delete xml files)
		"""
		curve_names = ["mean"]

		xml_filespecs = []
		for site in self.get_sites():
			for curve_name in curve_names:
				xml_filespec = self.get_oq_sdc_filespec_decomposed("", "", "",
										"", curve_name, site, calc_id=calc_id)
				xml_filespecs.append(xml_filespec)
				for source_model, _ in self.source_model_lt.source_model_pmf:
					xml_filespec = self.get_oq_sdc_filespec_decomposed(source_model.name,
										"", "", "", curve_name, site, calc_id=calc_id)
					xml_filespecs.append(xml_filespec)
					for trt in source_model.get_tectonic_region_types():
						xml_filespec = self.get_oq_sdc_filespec_decomposed(
											source_model.name, trt, "", "",
											curve_name, site, calc_id=calc_id)
						xml_filespecs.append(xml_filespec)
						gmpe_names = [""] + self.gmpe_lt.gmpe_system_def[trt].gmpe_names
						for gmpe_name in gmpe_names:
							xml_filespec = self.get_oq_sdc_filespec_decomposed(
											source_model.name, trt, "", gmpe_name,
											curve_name, site, calc_id=calc_id)
							xml_filespecs.append(xml_filespec)
							for src_list in self.enumerate_correlated_sources(
															source_model, trt):
								for src in src_list:
									xml_filespec = self.get_oq_sdc_filespec_decomposed(
											source_model.name, trt, src.source_id,
											gmpe_name, curve_name, site, calc_id=calc_id)
									xml_filespecs.append(xml_filespec)

		for xml_filespec in xml_filespecs:
			if os.path.exists(xml_filespec):
				if verbose:
					print("Deleting %s" % xml_filespec)
				if not dry_run:
					os.unlink(xml_filespec)

	def delete_oq_sdc_samples(self, calc_id="oqhazlib", verbose=True,
							dry_run=False):
		"""
		Delete xml files corresponding to logictree samples (deaggregation)

		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: "oqhazlib")
		:param verbose:
			bool, whether or not to print filenames to be deleted
			(default: True)
		:param dry_run:
			bool, whether or not to run without actually deleting files
			(default: False, will delete xml files)
		"""
		num_lt_samples = self.num_lt_samples or self.get_num_paths()
		for site in self.get_sites():
			for sample_idx in range(num_lt_samples):
				fmt = "%%0%dd" % len(str(num_lt_samples))
				curve_name = "rlz-" + fmt % (sample_idx + 1)
				xml_filespec = self.get_oq_sdc_filespec(curve_name, site,
														calc_id=calc_id)
				if os.path.exists(xml_filespec):
					if verbose:
						print("Deleting %s" % xml_filespec)
					if not dry_run:
						os.unlink(xml_filespec)

	def iter_oq_shcf_files(self, calc_id="oqhazlib"):
		"""
		Iterate over spectral hazard curve field files computed with OpenQuake

		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: "oqhazlib")

		:return:
			generator object, yielding full path to xml file (str)
		"""
		gmpe_system_def = self.gmpe_lt.gmpe_system_def
		for source_model in self.source_models:
			for src in source_model.sources:
				trt = src.tectonic_region_type
				for (modified_src, branch_path, branch_weight) in \
						self.source_model_lt.enumerate_source_realizations(
														source_model.name, src):
					branch_path = [b.split('--')[-1] for b in branch_path]
					curve_name = '--'.join(branch_path)
					for gmpe_name in gmpe_system_def[trt].gmpe_names:
						xml_filespec = self.get_oq_shcf_filespec_decomposed(
											source_model.name, trt, src.source_id,
											gmpe_name, curve_name, calc_id=calc_id)
						yield xml_filespec

	def iter_oq_sdc_files(self, calc_id="oqhazlib"):
		"""
		Iterate over spectral deaggregation curve files computed with OpenQuake

		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: "oqhazlib")

		:return:
			generator object, yielding full path to xml file (str)
		"""
		gmpe_system_def = self.gmpe_lt.gmpe_system_def
		for source_model in self.source_models:
			for src in source_model.sources:
				trt = src.tectonic_region_type
				for (modified_src, branch_path, branch_weight) in \
						self.source_model_lt.enumerate_source_realizations(
														source_model.name, src):
					branch_path = [b.split('--')[-1] for b in branch_path]
					curve_name = '--'.join(branch_path)
					for gmpe_name in gmpe_system_def[trt].gmpe_names:
						xml_filespec = self.get_oq_sdc_filespec_decomposed(
											source_model.name, trt, src.source_id,
											gmpe_name, curve_name, calc_id=calc_id)
						yield xml_filespec

	def get_oq_shcf_computation_time(self, calc_id="oqhazlib"):
		"""
		Determine computation time of OpenQuake hazard-curve computation

		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: "oqhazlib")

		:return:
			(datetime.datetime, datetime.datetime, datetime.timedelta) tuple,
			representing start time, end time and time interval
		"""
		from datetime import datetime

		for i, xml_filespec in enumerate(self.iter_oq_shcf_files(calc_id=calc_id)):
			mtime = os.path.getmtime(xml_filespec)
			if i == 0:
				min_time = max_time = mtime
			else:
				min_time = min(mtime, min_time)
				max_time = max(mtime, max_time)

		max_time = datetime.fromtimestamp(max_time)
		min_time = datetime.fromtimestamp(min_time)
		time_delta = max_time - min_time
		return (min_time, max_time, time_delta)

	def get_oq_sdc_computation_time(self, calc_id="oqhazlib"):
		"""
		Determine computation time of OpenQuake deaggregation-curve computation

		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: "oqhazlib")

		:return:
			datetime.timedelta object
		"""
		from datetime import datetime

		for i, xml_filespec in enumerate(self.iter_oq_sdc_files(calc_id=calc_id)):
			mtime = os.path.getmtime(xml_filespec)
			if i == 0:
				min_time = max_time = mtime
			else:
				min_time = min(mtime, min_time)
				max_time = max(mtime, max_time)

		max_time = datetime.fromtimestamp(max_time)
		min_time = datetime.fromtimestamp(min_time)
		time_delta = max_time - min_time
		return time_delta

	def to_psha_model_tree(self):
		"""
		Convert back to standard PSHA model tree

		:return:
			instance of :class:`PSHAModelTree`
		"""
		return PSHAModelTree(self.name, self.source_model_lt, self.gmpe_lt,
						self.root_folder, self.site_model, self.ref_soil_params,
						self.imt_periods, self.intensities, self.min_intensities,
						self.max_intensities, self.num_intensities,
						self.return_periods, self.time_span,
						self.truncation_level, self.integration_distance,
						self.num_lt_samples, self.random_seed)

	def plot_source_model_sensitivity(self, fig_folder, sites=[], somo_colors={},
									somo_shortnames={}, plot_hc=True, plot_uhs=True,
									plot_barchart=True, hc_periods=[0, 0.2, 1],
									barchart_periods=[0, 0.2, 2], Tmax=10, amax={},
									calc_id=None, recompute=False, fig_site_id="name"):
		"""
		Generate plots showing source-model sensitivity

		:param fig_folder:
			str, full path to folder where figures will be saved
		:param sites:
			list with instances of :class:`GenericSite`, sites for which
			to plot sensitivity
			(default: [], will plot sensitivity for all sites)
		:param somo_colors:
			dict, mapping source model names to matplotlib color definitions
			If empty, will be generated automatically. Note that the color red
			will be reserved for the weighted mean.
			(default: {})
		:param somo_shortnames:
			dict, mapping source model names to short names used in barcharts.
			If empty, will be generated automatically
			(default: {})
		:param plot_hc:
			bool, whether or not hazard curves should be plotted
			(default: True)
		:param plot_uhs:
			bool, whether or not UHS should be plotted
			(default: True)
		:param plot_barchart:
			bool, whether or not barcharts should be plotted
			(default: True)
		:param hc_periods:
			list or array, spectral periods for which to plot hazard curves
			(default: [0, 0.2, 1])
		:param barchart_periods:
			list or array, spectral periods for which to plot barcharts
			(default: [0, 0.2, 1])
		:param Tmax:
			float, maximum period in X axis of UHS
		:param amax:
			dict, mapping return periods to maximum spectral acceleration
			in Y axis of UHS
			(default: {})
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)
		:param recompute:
			bool, whether or not mean spectral hazard curve fields should
			be recomputed in case they already exist
			(default: False)
		:param fig_site_id:
			str, site attribute to use in file names
			(default: "name")
		"""
		from ..result import HazardCurveCollection, UHSCollection
		from ..plot.barchart import plot_nested_variation_barchart

		num_source_models = len(self.source_models)

		if somo_colors == {}:
			all_colors = ["green", "magenta", "blue", "cyan", "yellow", "black"]
			for s, source_model in enumerate(self.source_models):
				somo_colors[source_model.name] = all_colors[s]

		if somo_shortnames == {}:
			for source_model in self.source_models:
				somo_shortnames[source_model.name] = ''.join(
					[word[0].capitalize() for word in source_model.name.split('_')])

		bc_colors = {}
		for source_model in self.source_models:
			bc_colors[somo_shortnames[source_model.name]] = somo_colors[source_model.name]

		if sites in (None, []):
			sites = self.get_sites()

		## Read or compute mean spectral hazard curve field
		mean_somo_shcf_list = []
		for source_model in self.source_models:
			shcf = self.get_oq_mean_shcf_by_source_model(source_model,
											calc_id=calc_id, write_xml=recompute)
			mean_somo_shcf_list.append(shcf)
		mean_shcf = self.get_oq_mean_shcf(calc_id=calc_id, write_xml=recompute)

		## Plot hazard curves
		if plot_hc:
			for site in sites:
				mean_somo_shc_list = [shcf.getSpectralHazardCurve(site_spec=site.name)
										for shcf in mean_somo_shcf_list]
				mean_shc = mean_shcf.getSpectralHazardCurve(site_spec=site.name)
				for period in hc_periods:
					hc_list, labels, colors = [], [], []
					for s, source_model in enumerate(self.source_models):
						hc = mean_somo_shc_list[s].getHazardCurve(period_spec=float(period))
						hc_list.append(hc)
						labels.append(source_model.name)
						colors.append(somo_colors[source_model.name])
					mean_hc = mean_shc.getHazardCurve(period_spec=float(period))
					hc_list.append(mean_hc)
					labels.append("Weighted mean")
					colors.append("red")
					hcc = HazardCurveCollection(hc_list, labels=labels, colors=colors)
					title = "Mean hazard curves, %s, T=%s s" % (site.name, period)
					fig_filename = "SHC_somo_mean_site_%s=%s_T=%s.png"
					fig_filename %= (fig_site_id, getattr(site, fig_site_id), period)
					fig_filespec = os.path.join(fig_folder, fig_filename)
					hcc.plot(title=title, fig_filespec=fig_filespec)

		## Compute UHS
		mean_somo_uhsfs_list = [shcf.interpolate_return_periods(self.return_periods)
								for shcf in mean_somo_shcf_list]
		mean_uhsfs = mean_shcf.interpolate_return_periods(self.return_periods)

		for return_period in self.return_periods:
			mean_somo_uhsf_list = [uhsfs.getUHSField(return_period=return_period)
									for uhsfs in mean_somo_uhsfs_list]
			mean_uhsf = mean_uhsfs.getUHSField(return_period=return_period)

			for site in sites:
				uhs_list, labels, colors = [], [], []
				for s, source_model in enumerate(self.source_models):
					mean_somo_uhs = mean_somo_uhsf_list[s].getUHS(site_spec=site.name)
					uhs_list.append(mean_somo_uhs)
					labels.append(source_model.name)
					colors.append(somo_colors[source_model.name])
					## Export somo mean to csv
					#csv_filename = "UHS_mean_%s_site_%s=%s_Tr=%.Eyr.csv" % (source_model.name, fig_site_id, getattr(site, fig_site_id), return_period)
					#mean_somo_uhs.export_csv(os.path.join(fig_folder, csv_filename))

				mean_uhs = mean_uhsf.getUHS(site_spec=site.name)
				uhs_list.append(mean_uhs)
				labels.append("Weighted mean")
				colors.append("red")

				## Plot UHS
				if plot_uhs:
					uhsc = UHSCollection(uhs_list, colors=colors, labels=labels)
					title = "Mean UHS, %s, Tr=%.E yr" % (site.name, return_period)
					fig_filename = "UHS_somo_mean_site_%s=%s_Tr=%.Eyr.png"
					fig_filename %= (fig_site_id, getattr(site, fig_site_id), return_period)
					fig_filespec = os.path.join(fig_folder, fig_filename)
					uhsc.plot(title=title, Tmax=Tmax, amax=amax.get(return_period, None),
							legend_location=1, fig_filespec=fig_filespec)

				## Plot barchart
				if plot_barchart:
					category_value_dict = OrderedDict()
					for period in barchart_periods:
						period_label = "T=%s s" % period
						mean_value = float("%.3f" % mean_uhs[period])
						category_value_dict[period_label] = OrderedDict()
						#for source_model_name, uhs in zip(labels, uhs_list)[:3]:
						for s, source_model in enumerate(self.source_models):
							uhs = uhs_list[s]
							somo_shortname = somo_shortnames[source_model.name]
							category_value_dict[period_label][somo_shortname] = uhs[period] - mean_value
					fig_filename = "Barchart_somo_site_%s=%s_Tr=%.Eyr.png"
					fig_filename %= (fig_site_id, getattr(site, fig_site_id), return_period)
					fig_filespec = os.path.join(fig_folder, fig_filename)
					ylabel = "SA (g)"
					title = "Source-model sensitivity, %s, Tr=%.E yr"
					title %= (site.name, return_period)
					plot_nested_variation_barchart(0, category_value_dict, ylabel,
								title, color=bc_colors, fig_filespec=fig_filespec)

	def plot_source_sensitivity(self, fig_folder, sites=[], sources_to_combine={},
								plot_hc=True, plot_uhs=True, hc_periods=[0, 0.2, 1],
								Tmax=10, amax={}, calc_id=None, recompute=False,
								fig_site_id="name"):
		"""
		Generate plots showing source sensitivity

		:param fig_folder:
			str, full path to folder where figures will be saved
		:param sites:
			list with instances of :class:`GenericSite`, sites for which
			to plot sensitivity
			(default: [], will plot sensitivity for all sites)
		:param sources_to_combine:
			dict mapping source model names to dicts mapping in turn
			names to lists of source ids that should be combined in
			the plot
			(default: {})
		:param plot_hc:
			bool, whether or not hazard curves should be plotted
			(default: True)
		:param plot_uhs:
			bool, whether or not UHS should be plotted
			(default: True)
		:param hc_periods:
			list or array, spectral periods for which to plot hazard curves
			(default: [0, 0.2, 1])
		:param Tmax:
			float, maximum period in X axis of UHS
			(default: 10)
		:param amax:
			dict, mapping return periods to maximum spectral acceleration
			in Y axis of UHS
			(default: {})
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)
		:param recompute:
			bool, whether or not mean spectral hazard curve fields should
			be recomputed in case they already exist
			(default: False)
		:param fig_site_id:
			str, site attribute to use in file names
			(default: "name")
		"""
		from ..result import HazardCurveCollection, UHSCollection

		src_colors = ["aqua", "blue", "fuchsia", "green", "lime", "maroon",
					"navy", "olive", "orange", "purple", "silver", "teal",
					"yellow", "slateblue", "saddlebrown", "forestgreen"]

		if sites in (None, []):
			sites = self.get_sites()

		## Read or compute mean spectral hazard curve fields
		for source_model in self.source_models:
			mean_somo_shcf = self.get_oq_mean_shcf_by_source_model(source_model,
											calc_id=calc_id, write_xml=recompute)
			somo_sources_to_combine = sources_to_combine.get(source_model.name, {})
			all_somo_sources_to_combine = sum(somo_sources_to_combine.values(), [])
			for src in source_model.sources:
				if not src.source_id in all_somo_sources_to_combine:
					somo_sources_to_combine[src.source_id] = [src.source_id]

			## Organize combined and uncombined sources
			mean_src_shcf_dict = {}
			for combined_src_id, src_id_list in somo_sources_to_combine.items():
				combined_shcf = None
				for src_id in src_id_list:
					src = source_model[src_id]
					shcf = self.get_oq_mean_shcf_by_source(source_model.name, src,
											calc_id=calc_id, write_xml=recompute)
					## Sum remote sources
					if combined_shcf is None:
						combined_shcf = shcf
					else:
						combined_shcf += shcf
				mean_src_shcf_dict[combined_src_id] = combined_shcf

			## Plot hazard curves
			if plot_hc:
				for site in sites:
					mean_somo_shc = mean_somo_shcf.getSpectralHazardCurve(
															site_spec=site.name)
					for period in hc_periods:
						mean_somo_hc = mean_somo_shc.getHazardCurve(
														period_spec=float(period))
						hc_list = [mean_somo_hc]
						labels = ["All sources"]
						colors = ["red"]
						for s, combined_src_id in enumerate(sorted(mean_src_shcf_dict.keys())):
							mean_src_shc = mean_src_shcf_dict[combined_src_id].getHazardCurve(
												site_spec=site.name, period_spec=float(period))
							hc_list.append(mean_src_shc)
							labels.append(combined_src_id)
							colors.append(src_colors[s])
						hcc = HazardCurveCollection(hc_list, labels=labels,
													colors=colors)
						title = "Source hazard curves, %s, T=%s s"
						title %= (site.name, period)
						fig_filename = "SHC_%s_sources_site_%s=%s_T=%s.png"
						fig_filename %= (source_model.name, fig_site_id, getattr(site, fig_site_id), period)
						fig_filespec = os.path.join(fig_folder, fig_filename)
						hcc.plot(title=title, fig_filespec=fig_filespec)

			## Compute UHS
			if plot_uhs:
				return_periods = self.return_periods
				mean_somo_uhsfs = mean_somo_shcf.interpolate_return_periods(
																return_periods)
				mean_src_uhsfs_dict = {}
				for combined_src_id in mean_src_shcf_dict.keys():
					mean_src_uhsfs_dict[combined_src_id] = mean_src_shcf_dict[combined_src_id].interpolate_return_periods(return_periods)

				for return_period in self.return_periods:
					mean_somo_uhsf = mean_somo_uhsfs.getUHSField(
													return_period=return_period)
					mean_src_uhsf_dict = {}
					for combined_src_id in mean_src_uhsfs_dict.keys():
						mean_src_uhsf_dict[combined_src_id] = mean_src_uhsfs_dict[combined_src_id].getUHSField(return_period=return_period)

					for site in sites:
						mean_somo_uhs = mean_somo_uhsf.getUHS(site_spec=site.name)
						uhs_list = [mean_somo_uhs]
						labels = ["All sources"]
						colors = ["red"]
						for s, combined_src_id in enumerate(sorted(mean_src_uhsf_dict.keys())):
							mean_src_uhsf = mean_src_uhsf_dict[combined_src_id]
							mean_src_uhs = mean_src_uhsf.getUHS(site_spec=site.name)
							uhs_list.append(mean_src_uhs)
							labels.append(combined_src_id)
							colors.append(src_colors[s])

						## Plot UHS
						uhsc = UHSCollection(uhs_list, labels=labels, colors=colors)
						title = "Source UHS, %s, %s, Tr=%.E yr"
						title %= (source_model.name, site.name, return_period)
						fig_filename = "UHS_%s_sources_site_%s=%s_Tr=%.Eyr.png"
						fig_filename %= (source_model.name, fig_site_id, getattr(site, fig_site_id), return_period)
						fig_filespec = os.path.join(fig_folder, fig_filename)
						uhsc.plot(title=title, Tmax=Tmax, amax=amax.get(return_period, None), legend_location=1, fig_filespec=fig_filespec)

	def plot_gmpe_sensitivity(self, fig_folder, sites=[], gmpe_colors={},
							gmpe_shortnames={}, somo_colors={}, somo_shortnames={},
							plot_hc=True, plot_uhs=True, plot_barchart=True,
							plot_by_source_model=False, hc_periods=[0, 0.2, 1],
							barchart_periods=[0, 0.2, 2], Tmax=10, amax={},
							calc_id=None, recompute=False, fig_site_id="name"):
		"""
		Generate plots showing GMPE sensitiviy

		:param fig_folder:
			str, full path to folder where figures will be saved
		:param sites:
			list with instances of :class:`GenericSite`, sites for which
			to plot sensitivity
			(default: [], will plot sensitivity for all sites)
		:param gmpe_colors:
			dict, mapping GMPE names to matplotlib color definitions
			If empty, will be generated automatically. Note that the color
			red will be reserved for the weighted mean.
			(default: {})
		:param gmpe_shortnames:
			dict, mapping GMPE names to short names used in barcharts.
			If empty, will be generated automatically
			(default: {})
		:param somo_colors:
			dict, mapping source model names to matplotlib color
			definitions to be used in barcharts
			If empty, will be generated automatically. Note that the
			color red will be reserved for the weighted mean.
			(default: {})
		:param somo_shortnames:
			dict, mapping source model names to short names used in barcharts.
			If empty, will be generated automatically
			(default: {})
		:param plot_hc:
			bool, whether or not hazard curves should be plotted
			(default: True)
		:param plot_uhs:
			bool, whether or not UHS should be plotted
			(default: True)
		:param plot_barchart:
			bool, whether or not barcharts should be plotted
			(default: True)
		:param plot_by_source_model:
			bool, whether or not hazard curves and UHS should be plotted
			by source model in addition to the plots by TRT
			(default: False)
		:param hc_periods:
			list or array, spectral periods for which to plot hazard curves
			(default: [0, 0.2, 1])
		:param barchart_periods:
			list or array, spectral periods for which to plot barcharts
			(default: [0, 0.2, 1])
		:param Tmax:
			float, maximum period in X axis of UHS
			(default: 10)
		:param amax:
			dict, mapping return periods to maximum spectral acceleration
			in Y axis of UHS
			(default: {})
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)
		:param recompute:
			bool, whether or not mean spectral hazard curve fields should
			be recomputed in case they already exist
			(default: False)
		:param fig_site_id:
			str, site attribute to use in file names
			(default: "name")
		"""
		from ..result import HazardCurveCollection, UHSCollection
		from ..plot.barchart import plot_nested_variation_barchart

		if gmpe_colors == {}:
			for trt in self.gmpe_lt.tectonicRegionTypes:
				all_colors = ["green", "magenta", "blue", "cyan", "yellow",
							"gray", "black"]
				for g, (gmpe_name, gmpe_weight) in \
						enumerate(self.gmpe_lt.gmpe_system_def[trt]):
					gmpe_colors[gmpe_name] = all_colors[g]

		if somo_shortnames == {}:
			for source_model in self.source_models:
				somo_shortnames[source_model.name] = ''.join(
					[word[0].capitalize() for word in source_model.name.split('_')])

		if gmpe_shortnames == {}:
			for trt in self.gmpe_lt.tectonicRegionTypes:
				for gmpe_name, gmpe_weight in self.gmpe_lt.gmpe_system_def[trt]:
					try:
						gmpe_short_name = getattr(rshalib.gsim, gmpe_name)().short_name
					except:
						gmpe_short_name = "".join(
							[c for c in gmpe_name if c.isupper() or c.isdigit()])
					gmpe_shortnames[gmpe_name] = gmpe_short_name

		if somo_colors == {}:
			all_colors = ["green", "magenta", "blue", "cyan", "yellow", "black"]
			for s, source_model in enumerate(self.source_models):
				somo_colors[source_model.name] = all_colors[s]

		bc_colors = {}
		for source_model in self.source_models:
			bc_colors[somo_shortnames[source_model.name]] = somo_colors[source_model.name]
		bc_colors["Avg"] = "red"

		if sites in (None, []):
			sites = self.get_sites()

		all_trts = self.gmpe_lt.tectonicRegionTypes
		if (len(all_trts) == 2
			and self.gmpe_lt.gmpe_system_def[all_trts[0]]
			== self.gmpe_lt.gmpe_system_def[all_trts[1]]):
			# TODO: ideally, this should also work if more than 2 trt's have same GMPE system def
			trt = ""
			trt_shortnames = [''.join([word[0].capitalize() for word in all_trts[i].split()]) for i in range(len(all_trts))]
			trt_shortnames = {trt: '+'.join([trtsn for trtsn in sorted(trt_shortnames)])}
			trts = [trt]
			gmpe_system_def = {trt: self.gmpe_lt.gmpe_system_def[all_trts[0]]}
		else:
			trts = all_trts
			trt_shortnames = {}
			for trt in trts:
				trt_shortnames[trt] = ''.join(
								[word[0].capitalize() for word in trt.split()])
			gmpe_system_def = self.gmpe_lt.gmpe_system_def

		return_periods = self.return_periods

		## Initialize category_value_dict for barchart plot
		category_value_dict = {}
		mean_value_dict = {}
		for site in sites:
			category_value_dict[site.name] = {}
			mean_value_dict[site.name] = {}
			for return_period in return_periods:
				category_value_dict[site.name][return_period] = {}
				mean_value_dict[site.name][return_period] = {}
				for period in barchart_periods:
					category_value_dict[site.name][return_period][period] = {}
					mean_value_dict[site.name][return_period][period] = {}
					for trt in trts:
						category_value_dict[site.name][return_period][period][trt] = OrderedDict()
						for gmpe_name, gmpe_weight in gmpe_system_def[trt]:
							gmpe_short_name = gmpe_shortnames[gmpe_name]
							category_value_dict[site.name][return_period][period][trt][gmpe_short_name] = OrderedDict()

		## By TRT
		for trt in trts:
			trt_short_name = trt_shortnames[trt]
			trt_gmpes = [gmpe_name for gmpe_name, gmpe_weight in gmpe_system_def[trt]]
			mean_trt_shcf = self.get_oq_mean_shcf(trt=trt, write_xml=recompute,
												calc_id=calc_id)
			gmpe_shcf_list = []
			for gmpe_name in trt_gmpes:
				shcf = self.get_oq_mean_shcf(trt=trt, gmpe_name=gmpe_name,
											calc_id=calc_id, write_xml=recompute)
				gmpe_shcf_list.append(shcf)

			## Plot hazard curves
			if plot_hc:
				for site in sites:
					mean_trt_shc = mean_trt_shcf.getSpectralHazardCurve(site_spec=site.name)
					gmpe_shc_list = [shcf.getSpectralHazardCurve(site_spec=site.name)
									for shcf in gmpe_shcf_list]
					for period in hc_periods:
						hc_list, labels, colors = [], [], []
						for g, gmpe_name in enumerate(trt_gmpes):
							hc = gmpe_shc_list[g].getHazardCurve(
														period_spec=float(period))
							hc_list.append(hc)
							labels.append(gmpe_name)
							colors.append(gmpe_colors[gmpe_name])
						mean_trt_hc = mean_trt_shc.getHazardCurve(
														period_spec=float(period))
						hc_list.append(mean_trt_hc)
						labels.append("Weighted mean")
						colors.append("red")
						hcc = HazardCurveCollection(hc_list, labels=labels,
													colors=colors)
						title = "Mean %s hazard curves, %s, T=%s s"
						title %= (trt_short_name, site.name, period)
						fig_filename = "SHC_%s_site_%s=%s_T=%s.png"
						fig_filename %= (trt_short_name, fig_site_id, getattr(site, fig_site_id), period)
						fig_filespec = os.path.join(fig_folder, fig_filename)
						hcc.plot(title=title, fig_filespec=fig_filespec)

			## Compute UHS and collect mean values by TRT
			mean_trt_uhsfs = mean_trt_shcf.interpolate_return_periods(return_periods)
			gmpe_uhsfs_list = [shcf.interpolate_return_periods(return_periods)
								for shcf in gmpe_shcf_list]
			for return_period in return_periods:
				mean_trt_uhsf = mean_trt_uhsfs.getUHSField(return_period=return_period)
				gmpe_uhsf_list = [uhsfs.getUHSField(return_period=return_period)
								for uhsfs in gmpe_uhsfs_list]
				for site in sites:
					uhs_list, labels, colors = [], [], []
					for g, (gmpe_name, gmpe_weight) in enumerate(gmpe_system_def[trt]):
						gmpe_short_name = gmpe_shortnames[gmpe_name]
						gmpe_uhs = gmpe_uhsf_list[g].getUHS(site_spec=site.name)
						uhs_list.append(gmpe_uhs)
						labels.append(gmpe_name)
						colors.append(gmpe_colors[gmpe_name])

					mean_trt_uhs = mean_trt_uhsf.getUHS(site_spec=site.name)
					uhs_list.append(mean_trt_uhs)
					labels.append("Weighted mean")
					colors.append("red")

					## Plot UHS
					if plot_uhs:
						uhsc = UHSCollection(uhs_list, colors=colors, labels=labels)
						title = "UHS, %s, %s, Tr=%.E yr"
						title %= (trt_short_name, site.name, return_period)
						fig_filename = "UHS_%s_site_%s=%s_Tr=%.Eyr.png"
						fig_filename %= (trt_short_name, fig_site_id, getattr(site, fig_site_id), return_period)
						fig_filespec = os.path.join(fig_folder, fig_filename)
						uhsc.plot(title=title, Tmax=Tmax, amax=amax.get(return_period, None), legend_location=1, fig_filespec=fig_filespec)

					for period in barchart_periods:
						mean_value_dict[site.name][return_period][period][trt] = mean_trt_uhs[period]

		## Collect mean values per TRT/GMPE
		for trt in trts:
			for gmpe_name, gmpe_weight in gmpe_system_def[trt]:
				gmpe_short_name = gmpe_shortnames[gmpe_name]
				mean_trt_gmpe_shcf = self.get_oq_mean_shcf(trt=trt,
										gmpe_name=gmpe_name, write_xml=recompute,
										calc_id=calc_id)
				mean_trt_gmpe_uhsfs = mean_trt_gmpe_shcf.interpolate_return_periods(
																	return_periods)
				for return_period in return_periods:
					mean_trt_gmpe_uhsf = mean_trt_gmpe_uhsfs.getUHSField(
													return_period=return_period)
					for site in sites:
						mean_trt_gmpe_uhs = mean_trt_gmpe_uhsf.getUHS(
															site_spec=site.name)
						for period in barchart_periods:
							category_value_dict[site.name][return_period][period][trt][gmpe_short_name]["Avg"] = mean_trt_gmpe_uhs[period]

		## By source model
		for source_model in self.source_models:
			somo_short_name = somo_shortnames[source_model.name]
			if trts == [""]:
				somo_trts = [""]
			else:
				somo_trts = source_model.get_tectonic_region_types()
			for trt in somo_trts:
				trt_short_name = trt_shortnames[trt]
				trt_gmpes = [gmpe_name for gmpe_name, gmpe_weight in gmpe_system_def[trt]]
				gmpe_shcf_list = []
				trt_shcf = self.get_oq_mean_shcf_by_source_model(source_model,
														trt=trt, calc_id=calc_id,
														write_xml=recompute)
				for gmpe_name, gmpe_weight in gmpe_system_def[trt]:
					shcf = self.get_oq_mean_shcf_by_source_model(source_model,
												trt=trt, gmpe_name=gmpe_name,
												calc_id=calc_id, write_xml=recompute)
					gmpe_shcf_list.append(shcf)

				if plot_hc and plot_by_source_model:
					for site in sites:
						trt_shc = trt_shcf.getSpectralHazardCurve(
															site_spec=site.name)
						gmpe_shc_list = [shcf.getSpectralHazardCurve(site_spec=site.name)
										for shcf in gmpe_shcf_list]
						for period in hc_periods:
							hc_list, labels, colors = [], [], []
							for g, gmpe_name in enumerate(trt_gmpes):
								hc = gmpe_shc_list[g].getHazardCurve(
														period_spec=float(period))
								hc_list.append(hc)
								labels.append(gmpe_name)
								colors.append(gmpe_colors[gmpe_name])
							trt_hc = trt_shc.getHazardCurve(period_spec=float(period))
							hc_list.append(trt_hc)
							labels.append("Weighted mean")
							colors.append("red")
							hcc = HazardCurveCollection(hc_list, labels=labels,
														colors=colors)
							title = "%s, %s, %s, T=%s s"
							title %= (source_model.name, trt_short_name, site.name, period)
							fig_filename = "SHC_%s_%s_site_%s=%s_T=%s.png"
							fig_filename %= (source_model.name, trt_short_name, fig_site_id, getattr(site, fig_site_id), period)
							fig_filespec = os.path.join(fig_folder, fig_filename)
							hcc.plot(title=title, fig_filespec=fig_filespec)

				## Compute UHS
				gmpe_uhsfs_list = [shcf.interpolate_return_periods(return_periods)
									for shcf in gmpe_shcf_list]
				trt_uhsfs = trt_shcf.interpolate_return_periods(return_periods)

				for return_period in return_periods:
					gmpe_uhsf_list = [uhsfs.getUHSField(return_period=return_period)
									for uhsfs in gmpe_uhsfs_list]
					trt_uhsf = trt_uhsfs.getUHSField(return_period=return_period)

					for site in sites:
						uhs_list, labels, colors = [], [], []
						labels = []
						for g, (gmpe_name, gmpe_weight) in enumerate(gmpe_system_def[trt]):
							gmpe_short_name = gmpe_shortnames[gmpe_name]
							gmpe_uhs = gmpe_uhsf_list[g].getUHS(site_spec=site.name)
							uhs_list.append(gmpe_uhs)
							labels.append(gmpe_name)
							colors.append(gmpe_colors[gmpe_name])

							for period in barchart_periods:
								category_value_dict[site.name][return_period][period][trt][gmpe_short_name][somo_short_name] = gmpe_uhs[period]

						trt_uhs = trt_uhsf.getUHS(site_spec=site.name)
						uhs_list.append(trt_uhs)
						labels.append("Weighted mean")
						colors.append("red")

						## Plot UHS
						if plot_uhs and plot_by_source_model:
							uhsc = UHSCollection(uhs_list, colors=colors,
												labels=labels)
							title = "UHS, %s, %s, %s, Tr=%.E yr"
							title %= (source_model.name, trt_short_name, site.name, return_period)
							fig_filename = "UHS_%s_%s_site_%s=%s_Tr=%.Eyr.png"
							fig_filename %= (source_model.name, trt_short_name, fig_site_id, getattr(site, fig_site_id), return_period)
							fig_filespec = os.path.join(fig_folder, fig_filename)
							uhsc.plot(title=title, Tmax=Tmax,
									amax=amax.get(return_period, None),
									legend_location=1, fig_filespec=fig_filespec)

		## Barchart plot
		if plot_barchart:
			for site in sites:
				for return_period in return_periods:
					for period in barchart_periods:
						for trt in trts:
							trt_short_name = trt_shortnames[trt]
							fig_filename = "Barchart_gmpe_site_%s=%s_%s_Tr=%.Eyr_T=%ss.png"
							fig_filename %= (fig_site_id, getattr(site, fig_site_id), trt_short_name, return_period, period)
							fig_filespec = os.path.join(fig_folder, fig_filename)
							if period == 0:
								ylabel = "PGA (g)"
							else:
								ylabel = "SA (g)"
							title = "GMPE sensitivity, %s, %s, Tr=%.E yr, T=%s s"
							title %= (site.name, trt_short_name, return_period, period)
							cv_dict = category_value_dict[site.name][return_period][period][trt]
							mean_value = mean_value_dict[site.name][return_period][period][trt]
							plot_nested_variation_barchart(mean_value, cv_dict,
												ylabel, title, color=bc_colors,
												fig_filespec=fig_filespec)


if __name__ == '__main__':
	"""
	"""
	pass

