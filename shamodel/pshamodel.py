"""
:mod:`rhsalib.shamodel.pshamodel` exports :class:`rhlib.pshamodel.PSHAModel` and :class:`rhlib.pshamodel.PSHAModelTree`
"""

# TODO: check if documentation is compatibele with Sphinx
# NOTE: damping for spectral periods is fixed at 5.

### imports
import numpy as np
import os
from collections import OrderedDict
import copy
import random
from random import choice

import openquake.hazardlib as nhlib
from openquake.hazardlib.imt import PGA, SA, PGV, PGD, MMI

from base import SHAModelBase
from ..geo import *
from ..site import *
from ..result import SpectralHazardCurveField, SpectralHazardCurveFieldTree, Poisson, ProbabilityMatrix, DeaggregationSlice
from ..logictree import GroundMotionSystem, SeismicSourceSystem
from ..crisis import write_DAT_2007
from ..openquake import OQ_Params
from ..source import SourceModel
from ..gsim import GroundMotionModel
from ..pmf import get_uniform_weights



# TODO: make distinction between imt (PGA, SA) and im (SA(0.5, 5.0), SA(1.0, 5.0))


## Minimum and maximum values for random number generator
MIN_SINT_32 = -(2**31)
MAX_SINT_32 = (2**31) - 1


class PSHAModelBase(SHAModelBase):
	"""
	Base class for PSHA models, holding common attributes and methods.

	:param output_dir:
		String, defining full path to output directory.
	:param imt_periods:
		see :class:`..site.SHASiteModel`
	:param intensities:
		List of floats or array, defining equal intensities for all intensity measure types and periods (default: None).
		When given, params min_intensities, max_intensities and num_intensities are not set.
	:param min_intensities:
		Dictionary mapping intensity measure types (e.g. "PGA", "SA", "PGV", "PGD") to lists or arrays (one for each period) of minimum intensities (float values).
	:param max_intensities:
		Dictionary mapping intensity measure types (e.g. "PGA", "SA", "PGV", "PGD") to lists or arrays (one for each period) of maximum intensities (float values).
	:param num_intensities:
		Float, defining number of intensities (default: 100).
	:param return_periods:
		List of floats, defining return periods.
	:param time_span:
		Float, defining time span in years (default 50.).
	:param truncation_level:
		see :class:`..site.SHASiteModel`
	:param integration_distance:
		see :class:`..site.SHASiteModel`
	"""

	def __init__(self, name, output_dir, sites, grid_outline, grid_spacing, soil_site_model, ref_soil_params, imt_periods, intensities, min_intensities, max_intensities, num_intensities, return_periods, time_span, truncation_level, integration_distance):
		"""
		"""
		SHAModelBase.__init__(self, name, sites, grid_outline, grid_spacing, soil_site_model, ref_soil_params, imt_periods, truncation_level, integration_distance)

		self.output_dir = output_dir
		self.intensities = intensities
		if self.intensities:
			self.min_intensities = None
			self.max_intensities = None
			self.num_intensities = len(self.intensities)
		else:
			self.min_intensities = self._get_intensities_limits(min_intensities)
			self.max_intensities = self._get_intensities_limits(max_intensities)
			self.num_intensities = num_intensities
		self.return_periods = np.array(return_periods)
		self.time_span = time_span

	def _get_intensities_limits(self, intensities_limits):
		"""
		Return dict, defining minimum or maximum intensity for intensity type and period.

		:param intensities_limits:
			dict or float
		"""
		if not isinstance(intensities_limits, dict):
			intensities_limits = {imt: [intensities_limits]*len(periods) for imt, periods in self.imt_periods.items()}
		return intensities_limits

	def _get_nhlib_params(self):
		"""
		Construct dict with nhlib params for both PSHAModel and PSHAModelTree.

		:return:
			dict with following key, value pairs:
			'soil_site_model': soil site model object
			'imts': dictionary mapping intensity measure type objects to intensities
			'ssdf': source_to_site distance filter
			'rsdf': rupture_to_site distance filter
		"""
		nhlib_params = {}
		nhlib_params['soil_site_model'] = self.get_soil_site_model()
		nhlib_params['imts'] = self._get_nhlib_imts()
		nhlib_params['ssdf'] = nhlib.calc.filters.source_site_distance_filter(self.integration_distance)
		nhlib_params['rsdf'] = nhlib.calc.filters.rupture_site_distance_filter(self.integration_distance)
		return nhlib_params

	def _get_imt_intensities(self):
		"""
		Construct a dictionary containing a 2-D array [k, l] of intensities for each IMT.

		:return:
			dict {IMT (string): intensities (2-D numpy array of floats)}
		"""
		imtls = {}
		for imt, periods in self.imt_periods.items():
			if len(periods) > 1:
				imls = np.zeros((len(periods), self.num_intensities))
				for k, period in enumerate(periods):
					if self.intensities:
						imls[k,:] = np.array(self.intensities)
					else:
						imls[k,:] = np.logspace(np.log10(self.min_intensities[imt][k]), np.log10(self.max_intensities[imt][k]), self.num_intensities)
				imtls[imt] = imls
			else:
				if self.intensities:
					imtls[imt] = np.array(self.intensities).reshape(1, self.num_intensities)
				else:
					imtls[imt] = np.logspace(np.log10(self.min_intensities[imt][0]), np.log10(self.max_intensities[imt][0]), self.num_intensities).reshape(1, self.num_intensities)
		return imtls

	def _get_nhlib_imts(self):
		"""
		Construct a dictionary mapping nhlib intensity measure type objects
		to 1-D arrays of intensity measure levels. This dictionary can be passed
		as an argument to the nhlib.calc.hazard_curves_poissonian function.

		:return:
			dict {:mod:`nhlib.imt` object: 1-D numpy array of floats}
		"""
		imtls = {}
		for imt, periods in self.imt_periods.items():
			if len(periods) > 1:
				for k, period in enumerate(periods):
					if self.intensities:
						imtls[eval(imt)(period, 5.)] = np.array(self.intensities)
					else:
						imtls[eval(imt)(period, 5.)] = np.logspace(np.log10(self.min_intensities[imt][k]), np.log10(self.max_intensities[imt][k]), self.num_intensities)
			else:
				if self.intensities:
					imtls[eval(imt)()] = np.array(self.intensities)
				else:
					imtls[eval(imt)()] = np.logspace(np.log10(self.min_intensities[imt][0]), np.log10(self.max_intensities[imt][0]), self.num_intensities)
		return imtls

	def _get_openquake_imts(self):
		"""
		Construct a dictionary mapping intensity measure type strings
		to 1-D arrays of intensity measure levels. This dictionary can be
		passed to :class:`OQParams`.`set_imts` function, which is used to
		generate the configuration file for OpenQuake.

		:return:
			dict {imt (string): 1-D numpy array of floats}
		"""
		# TODO: probably better to move this into config.py, where we had a similar method
		imtls = {}
		for imt, periods in self.imt_periods.items():
			if len(periods) > 1:
				for k, period in enumerate(periods):
					if self.intensities:
						imtls[imt + "(%s)" % period] = list(self.intensities)
					else:
						imtls[imt + "(%s)" % period] = list(np.logspace(np.log10(self.min_intensities[imt][k]), np.log10(self.max_intensities[imt][k]), self.num_intensities))
			else:
				if self.intensities:
					imtls[imt] = list(self.intensities)
				else:
					imtls[imt] = list(np.logspace(np.log10(self.min_intensities[imt][0]), np.log10(self.max_intensities[imt][0]), self.num_intensities))
		return imtls

	def _degree_to_km(self, degree, lat=0.):
		"""
		Convert distance in arc degrees to distance in km assuming a spherical earth.
		Distance is along a great circle, unless latitude is specified.

		:param degree:
			Float, distance in arc degrees.
		:param lat:
			Float, latitude in degrees (default: 0.).
		"""
		return (40075./360.) * degree * np.cos(np.radians(lat))

	def _km_to_degree(self, km, lat=0.):
		"""
		Convert distance in km to distance in arc degrees assuming a spherical earth

		:param km:
			Float, distance in km.
		:param lat:
			Float, latitude in degrees (default: 0.).
		"""
		return km / ((40075./360.) * np.cos(np.radians(lat)))

	def _get_grid_spacing_km(self):
		"""
		Return grid spacing in km
		"""
		grid_outline = self.sha_site_model.grid_outline
		grid_spacing = self.sha_site_model.grid_spacing
		if isinstance(grid_spacing, (str, unicode)) and grid_spacing[-2:] == 'km':
			grid_spacing_km = float(grid_spacing[:-2])
		else:
			central_latitude = np.mean([site[1] for site in grid_outline])
			grid_spacing_km1 = self._degree_to_km(grid_spacing[0], central_latitude)
			grid_spacing_km2 = self._degree_to_km(grid_spacing[1])
			grid_spacing_km = min(grid_spacing_km1, grid_spacing_km2)

		return grid_spacing_km

	def _get_grid_spacing_degrees(self, adjust_lat=True):
		"""
		Return grid spacing in degrees as a tuple
		"""
		grid_outline = self.sha_site_model.grid_outline
		grid_spacing = self.sha_site_model.grid_spacing
		central_latitude = np.mean([site[1] for site in grid_outline])
		if isinstance(grid_spacing, (str, unicode)) and grid_spacing[-2:] == 'km':
			grid_spacing_km = float(grid_spacing[:-2])
			grid_spacing_lon = self._km_to_degree(grid_spacing_km, central_latitude)
			if adjust_lat:
				grid_spacing_lat = self._km_to_degree(grid_spacing_km)
				grid_spacing = (grid_spacing_lon, grid_spacing_lat)
			else:
				grid_spacing = (grid_spacing_lon, grid_spacing_lon)
		elif isinstance(grid_spacing, (int, float)):
			if adjust_lat:
				grid_spacing = (grid_spacing, grid_spacing * np.cos(np.radians(central_latitude)))
			else:
				grid_spacing = (grid_spacing, grid_spacing)
		else:
			grid_spacing = grid_spacing

		return grid_spacing

	def _soil_site_model_or_ref_soil_params(self, output_dir, params):
		"""
		Write nrml file for soil site model if present and set file param, or set ref soil params
		"""
		if self.soil_site_model:
			file_name = (self.soil_site_model.name or "soil_site_model") + ".xml"
			self.soil_site_model.write_xml(os.path.join(output_dir, file_name))
			params.set_soil_site_model_or_reference_params(soil_site_model_file=file_name)
		else:
			params.set_soil_site_model_or_reference_params(
				reference_vs30_value=self.ref_soil_params["vs30"],
				reference_vs30_type={True: 'measured', False:'inferred'}[self.ref_soil_params["vs30measured"]],
				reference_depth_to_1pt0km_per_sec=self.ref_soil_params["z1pt0"],
				reference_depth_to_2pt5km_per_sec=self.ref_soil_params["z2pt5"],
				reference_kappa=self.ref_soil_params.get("kappa", None))


class PSHAModel(PSHAModelBase):
	"""
	Class representing a single PSHA model.

	:param source_model:
		SourceModel object.
	:param ground_motion_model:
		GroundMotionModel object.

	See :class:`PSHAModelBase` for other arguments.
	"""

	def __init__(self, name, source_model, ground_motion_model, output_dir, sites=[], grid_outline=[], grid_spacing=0.5, soil_site_model=None, ref_soil_params=REF_SOIL_PARAMS, imt_periods={'PGA': [0]}, intensities=None, min_intensities=0.001, max_intensities=1., num_intensities=100, return_periods=[], time_span=50., truncation_level=3., integration_distance=200.):

		"""
		"""
		# TODO: consider moving 'name' parameter to third position, to be in accordance with order of parameters in docstring.
		PSHAModelBase.__init__(self, name, output_dir, sites, grid_outline, grid_spacing, soil_site_model, ref_soil_params, imt_periods, intensities, min_intensities, max_intensities, num_intensities, return_periods, time_span, truncation_level, integration_distance)
		self.source_model = source_model
		self.ground_motion_model = ground_motion_model

	def run_nhlib_shcf(self, plot=False, write=False, nrml_base_filespec=""):
		"""
		Run PSHA model with nhlib, and store result in one or more
		SpectralHazardCurfeField objects.

		:param plot:
			Boolean, defining whether to plot results or not
			(default: True).
		:param write:
			Boolean, defining whether to write results or not
			(default: False).
		:param nrml_base_filespec:
			String, base file specification for NRML output file
			(default: "").

		:return:
			dict {imt (string) : SpectralHazardCurveField object}
		"""
		if not nrml_base_filespec:
			nrml_base_filespec = os.path.join(self.output_dir, '%s' % self.name)
		else:
			nrml_base_filespec = os.path.splitext(nrml_base_filespec)[0]
		nhlib_params = self._get_nhlib_params()
		hazard_result = self.run_nhlib_poes(nhlib_params)
		imtls = self._get_imt_intensities()
		shcfs = {}
		site_names = [site.name for site in self.get_sites()]
		for imt, periods in self.imt_periods.items():
			# TODO: add method to PSHAModelBase to associate nhlib/OQ imt's with units
			shcf = SpectralHazardCurveField(self.name, [''], self.get_sites(), periods, imt, imtls[imt], 'g', self.time_span, poes=hazard_result[imt], site_names=site_names)
			nrml_filespec = nrml_base_filespec + '_%s.xml' % imt
			shcfs[imt] = shcf
			if plot:
				shcf.plot()
			if write:
				shcf.write_nrml(nrml_filespec)
		return shcfs

	def run_nhlib_poes(self, nhlib_params=None):
		"""
		Run PSHA model with nhlib. Output is a dictionary mapping intensity
		measure types to probabilities of exceedance (poes).

		:param nhlib_params:
			dict containing parameters specific for nhlib, namely 'soil_site_model',
			'imts', 'ssdf', and 'rsdf'. See :class:`PSHAModelBase`.`_get_nhlib_params`
			for an explanation of these keys.

		:return:
			dict {imt (string) : poes (2-D numpy array of poes)}
		"""
		if not nhlib_params:
			nhlib_params = self._get_nhlib_params()
		num_sites = len(nhlib_params['soil_site_model'])
		hazard_curves = nhlib.calc.hazard_curves_poissonian(self.source_model, nhlib_params['soil_site_model'], nhlib_params['imts'], self.time_span, self._get_nhlib_trts_gsims_map(), self.truncation_level, nhlib_params['ssdf'], nhlib_params['rsdf'])
		hazard_result = {}
		for imt, periods in self.imt_periods.items():
			if len(periods) > 1:
				poes = np.zeros((num_sites, len(periods), self.num_intensities))
				for k, period in enumerate(periods):
					poes[:,k,:] = hazard_curves[eval(imt)(period, 5.)]
				hazard_result[imt] = poes
			else:
				hazard_result[imt] = hazard_curves[eval(imt)()].reshape(num_sites, 1, self.num_intensities)
		return hazard_result

	def deagg_nhlib(self, site_index, imt, iml, n_epsilons=None, mag_bin_width=None, dist_bin_width=10., coord_bin_width=1.0):
		"""
		Run deaggregation with nhlib

		:param site_index:
			Int, index of site
		:param imt:
			Instance of :class:`nhlib.imt._IMT`, intensity measure type
		:param iml:
			Float, intensity measure level
		:param n_epsilons:
			Int, number of epsilon bins (default: None, will result in bins
				corresponding to integer epsilon values)
		:param mag_bin_width:
			Float, magnitude bin width (default: None, will take MFD bin width
				of first source)
		:param dist_bin_width:
			Float, distance bin width in km (default: 10.)
		:param coord_bin_width:
			Float, lon/lat bin width in decimal degrees (default: 1.)

		:return:
			instance of :class:`DeaggregationSlice`
		"""
		if not n_epsilons:
			n_epsilons = 2 * int(np.ceil(self.truncation_level))
		if not mag_bin_width:
			mag_bin_width = self.source_model[0].mfd.bin_width
		site = self.get_sites()[site_index]
		nhlib_site = nhlib.site.Site(Point(*site), **self.ref_soil_params)
		#imt = self._get_nhlib_imts()
		ssdf = nhlib.calc.filters.source_site_distance_filter(self.integration_distance)
		rsdf = nhlib.calc.filters.rupture_site_distance_filter(self.integration_distance)

		#tom = nhlib.tom.PoissonTOM(self.time_span)
		#bin_edges, deagg_matrix = nhlib.calc.disaggregation(self.source_model, nhlib_site, imt, iml, self._get_nhlib_trts_gsims_map(), tom, self.truncation_level, n_epsilons, mag_bin_width, dist_bin_width, coord_bin_width, ssdf, rsdf)
		bin_edges, deagg_matrix = nhlib.calc.disaggregation_poissonian(self.source_model, nhlib_site, imt, iml, self._get_nhlib_trts_gsims_map(), self.time_span, self.truncation_level, n_epsilons, mag_bin_width, dist_bin_width, coord_bin_width, ssdf, rsdf)
		deagg_matrix = ProbabilityMatrix(deagg_matrix)
		imt_name = str(imt).split('(')[0]
		if imt_name == "SA":
			period = imt.period
		else:
			period = 0
		return DeaggregationSlice(bin_edges, deagg_matrix, site, imt_name, iml, period, self.time_span)

	def _get_implicit_openquake_params(self):
		"""
		Return a dictionary of implicit openquake parameters that are
		defined in source objects
		(rupture_mesh_spacing, area_source_discretization, mfd_bin_width).
		Warnings will be generated if one or more sources have different
		parameters than the first source.
		"""
		all_sources = self.source_model.sources
		rupture_mesh_spacing = all_sources[0].rupture_mesh_spacing
		mfd_bin_width = all_sources[0].mfd.bin_width
		for src in all_sources[1:]:
			if src.rupture_mesh_spacing != rupture_mesh_spacing:
				print("Warning: rupture mesh spacing of src %s different from that of 1st source!" % src.source_id)
			if src.mfd.bin_width != mfd_bin_width:
				print("Warning: mfd bin width of src %s different from that of 1st source!" % src.source_id)

		area_sources = self.source_model.get_area_sources()
		if len(area_sources) > 0:
			area_source_discretization = area_sources[0].area_discretization
			for src in area_sources[1:]:
				if src.area_discretization != area_source_discretization:
					print("Warning: area discretization of src %s different from that of 1st source!" % src.source_id)
		else:
			area_source_discretization = 5.

		params = {}
		params['rupture_mesh_spacing'] = rupture_mesh_spacing
		params['width_of_mfd_bin'] = mfd_bin_width
		params['area_source_discretization'] = area_source_discretization

		return params

	def write_openquake(self, calculation_mode='classical', user_params=None, **kwargs):
		"""
		Write PSHA model input for OpenQuake.

		:param calculation_mode:
			str, calculation mode of OpenQuake (options: "classical" or
				"disaggregation") (default: "classical")
		:param user_params:
			{str, val} dict, defining respectively parameters and value for OpenQuake (default: None).
		"""
		# TODO: depending on how we implement deaggregation, calculation_mode may be dropped in the future
		## set OQ_params object and override with params from user_params
		params = OQ_Params(calculation_mode=calculation_mode, description=self.name)
		implicit_params = self._get_implicit_openquake_params()
		for key in implicit_params:
			setattr(params, key, implicit_params[key])
		if user_params:
			for key in user_params:
				setattr(params, key, user_params[key])

		if calculation_mode == "classical":
			params.mean_hazard_curves = False
			params.quantile_hazard_curves = []
			params.number_of_logic_tree_samples = 1

		## set sites or grid_outline
		if self.sha_site_model and self.sha_site_model.grid_outline:
			grid_spacing_km = self._get_grid_spacing_km()
			params.set_grid_or_sites(grid_outline=self.sha_site_model.grid_outline, grid_spacing=grid_spacing_km)
		else:
			params.set_grid_or_sites(sites=self.get_sites())

		## write nrml file for source model
		self.source_model.write_xml(os.path.join(self.output_dir, self.source_model.name + '.xml'))

		## write nrml file for soil site model if present and set file param, or set ref soil params
		self._soil_site_model_or_ref_soil_params(self.output_dir, params)

		## validate source model logic tree and write nrml file
		source_model_lt = SeismicSourceSystem(self.source_model.name, self.source_model)
		source_model_lt.validate()
		source_model_lt_file_name = 'source_model_lt.xml'
		source_model_lt.write_xml(os.path.join(self.output_dir, source_model_lt_file_name))
		params.source_model_logic_tree_file = source_model_lt_file_name

		## create ground_motion_model logic tree and write nrml file
		ground_motion_model_lt = self.ground_motion_model.get_optimized_model(self.source_model).to_ground_motion_system()
		ground_motion_model_lt_file_name = 'ground_motion_model_lt.xml'
		ground_motion_model_lt.write_xml(os.path.join(self.output_dir, ground_motion_model_lt_file_name))
		params.gsim_logic_tree_file = ground_motion_model_lt_file_name

		## convert return periods and time_span to poes
		if not self.return_periods in ([], None):
			if calculation_mode == "classical":
				params.poes = Poisson(life_time=self.time_span, return_period=self.return_periods)
			elif calculation_mode == "disaggregation":
				params.poes_disagg = Poisson(life_time=self.time_span, return_period=self.return_periods)

		## set other params
		params.intensity_measure_types_and_levels = self._get_openquake_imts()
		params.investigation_time = self.time_span
		params.truncation_level = self.truncation_level
		params.maximum_distance = self.integration_distance
		params.number_of_logic_tree_samples = 0

		if calculation_mode == "disaggregation":
			params.poes_disagg = kwargs["poes_disagg"]
			params.mag_bin_width = kwargs["mag_bin_width"]
			params.distance_bin_width = kwargs["distance_bin_width"]
			params.coordinate_bin_width =kwargs["coordinate_bin_width"]
			params.num_epsilon_bins = kwargs["num_epsilon_bins"]

		# validate and write oq params to ini file
		params.validate()
		params.write_config(os.path.join(self.output_dir, 'job.ini'))

	def write_crisis(self, filespec="", atn_folder="", site_filespec="", atn_Mmax=None, overwrite=False):
		"""
		Write full PSHA model input for Crisis.

		:param filespec:
			String, full path to CRISIS input .DAT file
			(default: "").
		:param atn_folder:
			String, full path to folder with attenuation tables (.ATN files)
			(default: "").
		:param site_filespec:
			String, full path to .ASC file containing sites where hazard
			will be computed
			(default: "")
		:param atn_Mmax:
			Float, maximum magnitude in attenuation table(s)
			(default: None, will determine automatically from source model)
		:param overwrite:
			Boolean, whether or not to overwrite existing input files (default: False)

		:return:
			String, full path to CRISIS input .DAT file
		"""
		## Raise exception if model contains sites with different
		## vs30 and/or kappa
		if self.soil_site_model:
			if len(set(self.soil_site_model.vs30)) > 1 or len(set(self.soil_site_model.kappa)) > 1:
				raise Exception("CRISIS2007 does not support sites with different VS30 and/or kappa!")

		## Construct default filenames and paths if none are specified
		if not filespec:
			filespec = os.path.join(self.output_dir, self.name + '.dat')
		if not atn_folder:
			atn_folder = os.path.join(self.output_dir, 'gsims')
		if not os.path.exists(atn_folder):
			os.mkdir(atn_folder)
		if not site_filespec:
			site_filespec = os.path.join(self.output_dir, 'sites.ASC')

		## Map gsims to attenuation tables
		gsim_atn_map = {}
		for gsim in self._get_used_gsims():
			gsim_atn_map[gsim] = os.path.join(atn_folder, gsim + '.ATN')

		## Convert grid spacing if necessary
		if isinstance(self.grid_spacing, (str, unicode)):
			grid_spacing = self._get_grid_spacing_degrees()
		else:
			grid_spacing = self.grid_spacing

		## Write input file. This will also write the site file and attenuation
		## tables if necessary.
		write_DAT_2007(filespec, self.source_model, self.ground_motion_model, gsim_atn_map, self.return_periods, self.grid_outline, grid_spacing, self.get_sites(), site_filespec, self.imt_periods, self.intensities, self.min_intensities, self.max_intensities, self.num_intensities, 'g', self.name, self.truncation_level, self.integration_distance, source_discretization=(1.0, 5.0), vs30=self.ref_soil_params["vs30"], kappa=self.ref_soil_params["kappa"], mag_scale_rel=None, atn_Mmax=atn_Mmax, output={"gra": True, "map": True, "fue": True, "des": True, "smx": True, "eps": True, "res_full": False}, map_filespec="", cities_filespec="", overwrite=overwrite)

		## Return name of output file
		return filespec

	def _get_nhlib_trts_gsims_map(self):
		"""
		Return {str, GroundShakingIntensityModel object} dict, defining respectively tectonic region types and gsim for nhlib.
		"""
		#return {trt: NHLIB_GSIMS_MAP[self.ground_motion_model[trt]]() for trt in self._get_used_trts()}
		return {trt: nhlib.gsim.get_available_gsims()[self.ground_motion_model[trt]]() for trt in self._get_used_trts()}

	def _get_used_trts(self):
		"""
		Return list of strings, defining tectonic region types used in source model.
		"""
		used_trts = set()
		for source in self.source_model:
			used_trts.add(source.tectonic_region_type)
		return list(used_trts)

	def _get_used_gsims(self):
		"""
		Return list of strings, defining gsims of tectonic region types used in source model.
		"""
		used_gsims = set()
		for used_trt in self._get_used_trts():
			used_gsims.add(self.ground_motion_model[used_trt])
		return list(used_gsims)


class PSHAModelTree(PSHAModelBase):
	"""
	Class representing a PSHA model logic tree.

	:param source_model_lt:
		:class:`LogicTree` object, defining source model logic tree.
	:param ground_motion_models:
		List of :class:`GroundMotionModel` objects.
	:param soil_site_model:
		SoilSiteModel object
	:param lts_sampling_method:
		String, defining sampling method for logic trees (options: 'random' and 'enumerated') (default: 'random').
	:param num_lts_samples:
		Integer, defining times to sample logic trees (default: 1).

	See :class:`PSHAModelBase` for other arguments.
	"""
	def __init__(self, name, source_model_lt, gmpe_lt, output_dir, sites=[], grid_outline=[], grid_spacing=0.5, soil_site_model=None, ref_soil_params=REF_SOIL_PARAMS, imt_periods={'PGA': [0]}, intensities=None, min_intensities=0.001, max_intensities=1., num_intensities=100, return_periods=[], time_span=50., truncation_level=3., integration_distance=200., num_lt_samples=1, random_seed=42):
		"""
		"""
		from openquake.engine.input.logictree import LogicTreeProcessor
		PSHAModelBase.__init__(self, name, output_dir, sites, grid_outline, grid_spacing, soil_site_model, ref_soil_params, imt_periods, intensities, min_intensities, max_intensities, num_intensities, return_periods, time_span, truncation_level, integration_distance)
		self.source_model_lt = source_model_lt
		self.gmpe_lt = gmpe_lt.get_optimized_system(self.source_models)
		self.num_lt_samples = num_lt_samples
		#self.lts_sampling_method = lts_sampling_method
		#if self.lts_sampling_method == 'enumerated':
		#	self.enumerated_lts_samples = self._enumerate_lts_samples()
		self.random_seed = random_seed
		self.ltp = LogicTreeProcessor(self.source_model_lt, self.gmpe_lt)
		self._init_rnd()

	@property
	def source_models(self):
		return self.source_model_lt.source_models

	def _init_rnd(self):
		"""
		Initialize random number generator with random seed
		"""
		self.rnd = random.Random()
		self.rnd.seed(self.random_seed)

	def plot_diagram(self):
		"""
		Plot a diagram of the logic tree(s) using networkx
		"""
		# TODO
		pass

	def get_num_paths(self):
		"""
		Return total number of paths in the two logic trees.
		"""
		num_smlt_paths = self.source_model_lt.get_num_paths()
		num_gmpelt_paths = self.gmpe_logic_tree.get_num_paths()
		return num_smlt_paths * num_gmpelt_paths

	def sample_logic_trees(self, num_samples=1, enumerate_gmpe_lt=False, verbose=False):
		"""
		Sample both source-model and GMPE logic trees, in a way that is
		similar to :meth:`_initialize_realizations_montecarlo` of
		:class:`BaseHazardCalculator` in oq-engine

		:param num_samples:
			int, number of random samples
			If zero, :meth:`enumerate_logic_trees` will be called
			(default: 1)
		:param enumerate_gmpe_lt:
			bool, whether or not to enumerate the GMPE logic tree
			(default: False)
		:param verbose:
			bool, whether or not to print some information (default: False)

		:return:
			list of instances of :class:`PSHAModel`
		"""
		if num_samples == 0:
			return self.enumerate_logic_trees()

		psha_models = []

		if enumerate_gmpe_lt:
			gmpe_models, _ = self.enumerate_gmpe_lt(verbose=verbose)

		for i in xrange(num_samples):
			## Generate 2nd-order random seeds
			smlt_random_seed = self.rnd.randint(MIN_SINT_32, MAX_SINT_32)
			gmpelt_random_seed = self.rnd.randint(MIN_SINT_32, MAX_SINT_32)

			## Call OQ logictree processor
			sm_name, smlt_path = self.ltp.sample_source_model_logictree(smlt_random_seed)
			gmpelt_path = self.ltp.sample_gmpe_logictree(gmpelt_random_seed)

			## Convert to objects
			source_model = self._smlt_sample_to_source_model(sm_name, smlt_path, verbose=verbose)
			if not enumerate_gmpe_lt:
				gmpe_models = [self._gmpe_sample_to_gmpe_model(gmpelt_path)]

			for gmpe_model in gmpe_models:
				## Convert to PSHA model
				name = "%s, LT sample %04d (SM_LTP: %s; GMPE_LTP: %s)" % (self.name, i+1, " -- ".join(smlt_path), " -- ".join(gmpelt_path))
				psha_model = self._get_psha_model(source_model, gmpe_model, name)
				psha_models.append(psha_model)

			## Update the seed for the next realization
			seed = self.rnd.randint(MIN_SINT_32, MAX_SINT_32)
			self.rnd.seed(seed)

		return psha_models

	def enumerate_logic_trees(self):
		"""
		Enumerate both source-model and GMPE logic trees, in a way that is
		similar to :meth:`_initialize_realizations_enumeration` of
		:class:`BaseHazardCalculator` in oq-engine

		:return:
			tuple of:
			- list of instances of :class:`PSHAModel`
			- list of corresponding weights
		"""
		psha_models, weights = [], []
		for i, path_info in enumerate(self.ltp.enumerate_paths()):
			sm_name, weight, smlt_path, gmpelt_path = path_info
			source_model = self._smlt_sample_to_source_model(sm_name, smlt_path, verbose=verbose)
			gmpe_model = self._gmpe_sample_to_gmpe_model(path)
			name = "%s, LT enum %04d (SM_LTP: %s; GMPE_LTP: %s)" % (self.name, i, source_model.description, gmpe_model.name)
			psha_model = self._get_psha_model(source_model, gmpe_model, name)
			psha_models.append(psha_model)
			weights.append(weight)
		return psha_models, weights

	def _get_psha_model(self, source_model, gmpe_model, name):
		"""
		Convert a logic-tree sample, consisting of a source model and a
		GMPE model, to a PSHAModel object.

		:param source_model:
			instance of :class:`SourceModel`
		:param gmpe_model:
			instance of :class:`GroundMotionModel`, mapping tectonic
			region type to GMPE name
		:param name:
			string, name of PSHA model
		"""
		# TODO: adjust output_dir based on path?
		output_dir = self.output_dir
		optimized_gmpe_model = gmpe_model.get_optimized_model(source_model)
		psha_model = PSHAModel(name, source_model, optimized_gmpe_model, output_dir,
			sites=self.get_sites(), grid_outline=self.grid_outline, grid_spacing=self.grid_spacing,
			soil_site_model=self.soil_site_model, ref_soil_params=self.ref_soil_params,
			imt_periods=self.imt_periods, intensities=self.intensities,
			min_intensities=self.min_intensities, max_intensities=self.max_intensities,
			num_intensities=self.num_intensities, return_periods=self.return_periods,
			time_span=self.time_span, truncation_level=self.truncation_level,
			integration_distance=self.integration_distance)
		return psha_model

	def sample_source_model_lt(self, num_samples=1, verbose=False, show_plot=False):
		"""
		Sample source-model logic tree

		:param num_samples:
			int, number of random samples.
			If zero, :meth:`enumerate_source_model_lt` will be called
			(default: 1)
		:param verbose:
			bool, whether or not to print some information (default: False)
		:param show_plot:
			bool, whether or not to plot a diagram of the sampled branch path
			(default: False)

		:return:
			list of instances of :class:`SourceModel`, one for each sample
		"""
		if num_samples == 0:
			return self.enumerate_source_model_lt(verbose=verbose, show_plot=show_plot)

		modified_source_models = []
		for i in xrange(num_samples):
			## Generate 2nd-order random seed
			random_seed = self.rnd.randint(MIN_SINT_32, MAX_SINT_32)
			## Call OQ logictree processor
			sm_name, path = self.ltp.sample_source_model_logictree(random_seed)
			if verbose:
				print sm_name, path
			if show_plot:
				self.source_model_lt.plot_diagram(highlight_path=path)
			## Apply uncertainties
			source_model = self._smlt_sample_to_source_model(sm_name, path, verbose=verbose)
			modified_source_models.append(source_model)
		return modified_source_models

	def enumerate_source_model_lt(self, verbose=False, show_plot=False):
		"""
		Enumerate source-model logic tree

		:param verbose:
			bool, whether or not to print some information (default: False)
		:param show_plot:
			bool, whether or not to plot a diagram of the sampled branch path
			(default: False)

		:return:
			tuple of:
			- list of instances of :class:`SourceModel`, one for each sample
			- list of corresponding weights
		"""
		weights, modified_source_models = [], []
		for smlt_path_weight, smlt_branches in self.source_model_lt.root_branchset.enumerate_paths():
			smlt_path = [branch.branch_id for branch in smlt_branches]
			sm_name = os.path.splitext(smlt_branches[0].value)[0]
			if verbose:
				print smlt_path_weight, sm_name, smlt_path
			if show_plot:
				self.source_model_lt.plot_diagram(highlight_path=smlt_path)
			## Apply uncertainties
			source_model = self._smlt_sample_to_source_model(sm_name, smlt_path, verbose=verbose)
			modified_source_models.append(source_model)
			weights.append(smlt_path_weight)
		return modified_source_models, weights

	def _smlt_sample_to_source_model(self, sm_name, path, verbose=False):
		"""
		Convert sample from source-model logic tree to a new source model
		object, applying the sampled uncertainties to each source.

		:param sm_name:
			string, name of source model
		:param path:
			list of branch ID's, representing the path through the
			source-model logic tree
		:param verbose:
			bool, whether or not to print some information (default: False)

		:return:
			instance of :class:`SourceModel`
		"""
		for sm in self.source_models:
			if sm.name == os.path.splitext(sm_name)[0]:
				modified_sources = []
				for src in sm:
					## Note: copy MFD explicitly, as not all source attributes are
					## instantiated properly when deepcopy is used!
					modified_src = copy.copy(src)
					modified_src.mfd = src.mfd.get_copy()
					apply_uncertainties = self.ltp.parse_source_model_logictree_path(path)
					apply_uncertainties(modified_src)
					if verbose:
						print "  %s" % src.source_id
						if hasattr(src.mfd, 'a_val'):
							print "    %.2f %.3f %.3f  -->  %.2f %.3f %.3f" % (src.mfd.max_mag, src.mfd.a_val, src.mfd.b_val, modified_src.mfd.max_mag, modified_src.mfd.a_val, modified_src.mfd.b_val)
						elif hasattr(src.mfd, 'occurrence_rates'):
							print "    %s  -->  %s" % (src.mfd.occurrence_rates, modified_src.mfd.occurrence_rates)
					modified_sources.append(modified_src)
				break
		description = " -- ".join(path)
		return SourceModel(sm.name, modified_sources, description)

	def sample_gmpe_lt(self, num_samples=1, verbose=False, show_plot=False):
		"""
		Sample GMPE logic tree

		:param num_samples:
			int, number of random samples
			If zero, :meth:`enumerate_gmpe_lt` will be called
			(default: 1)
		:param verbose:
			bool, whether or not to print some information (default: False)
		:param show_plot:
			bool, whether or not to plot a diagram of the sampled branch path
			(default: False)

		:return:
			list of instances of :class:`GroundMotionModel`, one for each sample
		"""
		if num_samples == 0:
			return self.enumerate_gmpe_lt(verbose=verbose, show_plot=show_plot)

		gmpe_models = []
		for i in xrange(num_samples):
			## Generate 2nd-order random seed
			random_seed = self.rnd.randint(MIN_SINT_32, MAX_SINT_32)
			## Call OQ logictree processor
			path = self.ltp.sample_gmpe_logictree(random_seed)
			if verbose:
				print path
			if show_plot:
				self.gmpe_lt.plot_diagram(highlight_path=path)
			## Convert to GMPE model
			gmpe_model = self._gmpe_sample_to_gmpe_model(path)
			gmpe_models.append(gmpe_model)
			if verbose:
				print gmpe_model
		return gmpe_models

	def enumerate_gmpe_lt(self, verbose=False, show_plot=False):
		"""
		Enumerate GMPE logic tree

		:param verbose:
			bool, whether or not to print some information (default: False)
		:param show_plot:
			bool, whether or not to plot a diagram of the sampled branch path
			(default: False)

		:return:
			tuple of:
			- list of instances of :class:`GroundMotionModel`, one for each sample
			- list of corresponding weights
		"""
		gmpe_models, weights = [], []
		for gmpelt_path_weight, gmpelt_branches in self.gmpe_lt.root_branchset.enumerate_paths():
			gmpelt_path = [branch.branch_id for branch in gmpelt_branches]
			if verbose:
				print gmpelt_path_weight, gmpelt_path
			if show_plot:
				self.gmpe_lt.plot_diagram(highlight_path=gmpelt_path)
			gmpe_model = self._gmpe_sample_to_gmpe_model(gmpelt_path)
			gmpe_models.append(gmpe_model)
			weights.append(gmpelt_path_weight)
		return gmpe_models, weights

	def _gmpe_sample_to_gmpe_model(self, path):
		"""
		Convert sample from GMPE logic tree to a ground-motion model

		:param path:
			list of branch ID's, representing the path through the
			GMPE logic tree

		:return:
			instance of :class:`GroundMotionModel', mapping tectonic
			region type to GMPE name
		"""
		trts = self.gmpe_lt.tectonicRegionTypes
		trt_gmpe_dict = {}
		for l, branch_id in enumerate(path):
			branch = self.gmpe_lt.get_branch_by_id(branch_id)
			trt = trts[l]
			trt_gmpe_dict[trt] = branch.value
		name = " -- ".join(path)
		return GroundMotionModel(name, trt_gmpe_dict)

	def _get_implicit_openquake_params(self):
		"""
		Return a dictionary of implicit openquake parameters that are
		defined in source objects
		(rupture_mesh_spacing, area_source_discretization, mfd_bin_width).
		Warnings will be generated if one or more sources have different
		parameters than the first source.
		"""
		all_sources = []
		for sm in self.source_models:
			all_sources.extend(sm.sources)
		rupture_mesh_spacing = all_sources[0].rupture_mesh_spacing
		mfd_bin_width = all_sources[0].mfd.bin_width
		for src in all_sources[1:]:
			if src.rupture_mesh_spacing != rupture_mesh_spacing:
				print("Warning: rupture mesh spacing of src %s different from that of 1st source!" % src.source_id)
			if src.mfd.bin_width != mfd_bin_width:
				print("Warning: mfd bin width of src %s different from that of 1st source!" % src.source_id)

		area_sources = []
		for sm in self.source_models:
			area_sources.extend(sm.get_area_sources())
		if len(area_sources) > 0:
			area_source_discretization = area_sources[0].area_discretization
			for src in area_sources[1:]:
				if src.area_discretization != area_source_discretization:
					print("Warning: area discretization of src %s different from that of 1st source!" % src.source_id)
		else:
			area_source_discretization = 5.

		params = {}
		params['rupture_mesh_spacing'] = rupture_mesh_spacing
		params['width_of_mfd_bin'] = mfd_bin_width
		params['area_source_discretization'] = area_source_discretization

		return params

	def write_openquake(self, calculation_mode='classical', user_params=None):
		"""
		Write PSHA model tree input for OpenQuake.

		:param calculation_mode:
			str, calculation mode of OpenQuake (options: "classical" or
				"disaggregation") (default: "classical")
		:param user_params:
			{str, val} dict, defining respectively parameters and value for OpenQuake (default: None).
		"""
		#output_dir = os.path.join(self.output_dir, "openquake")
		output_dir = self.output_dir
		if not os.path.exists(output_dir):
			os.mkdir(output_dir)

		## set OQ_params object and override with params from user_params
		params = OQ_Params(calculation_mode=calculation_mode, description=self.name)
		implicit_params = self._get_implicit_openquake_params()
		for key in implicit_params:
			setattr(params, key, implicit_params[key])
		if user_params:
			for key in user_params:
				setattr(params, key, user_params[key])

		## set sites or grid_outline
		if self.sha_site_model and self.sha_site_model.grid_outline:
			grid_spacing_km = self._get_grid_spacing_km()
			params.set_grid_or_sites(grid_outline=self.sha_site_model.grid_outline, grid_spacing=grid_spacing_km)
		else:
			params.set_grid_or_sites(sites=self.get_sites())

		## write nrml files for source models
		for source_model in self.source_models:
			## make sure source id's are unique among source models
			## This is no longer necessary
			#for source in source_model.sources:
			#	source.source_id = source_model.name + '--' + source.source_id
			source_model.write_xml(os.path.join(output_dir, source_model.name + '.xml'))

		## write nrml file for soil site model if present and set file param, or set ref soil params
		self._soil_site_model_or_ref_soil_params(self.output_dir, params)

		## validate source model logic tree and write nrml file
		self.source_model_lt.validate()
		source_model_lt_file_name = 'source_model_lt.xml'
		self.source_model_lt.write_xml(os.path.join(output_dir, source_model_lt_file_name))
		params.source_model_logic_tree_file = source_model_lt_file_name

		## create ground motion model logic tree and write nrml file
		ground_motion_model_lt_file_name = 'ground_motion_model_lt.xml'
		self.gmpe_lt.write_xml(os.path.join(output_dir, ground_motion_model_lt_file_name))
		params.gsim_logic_tree_file = ground_motion_model_lt_file_name

		## convert return periods and time_span to poes
		if not self.return_periods in ([], None):
			if calculation_mode == "classical":
				params.poes = Poisson(life_time=self.time_span, return_period=self.return_periods)
			elif calculation_mode == "disaggregation":
				params.poes_disagg = Poisson(life_time=self.time_span, return_period=self.return_periods)

		## set other params
		params.intensity_measure_types_and_levels = self._get_openquake_imts()
		params.investigation_time = self.time_span
		params.truncation_level = self.truncation_level
		params.maximum_distance = self.integration_distance
		params.number_of_logic_tree_samples = self.num_lt_samples
		params.random_seed = self.random_seed

		## disaggregation params

		## write oq params to ini file
		params.write_config(os.path.join(output_dir, 'job.ini'))

	def run_nhlib(self, nrml_base_filespec=""):
		"""
		Run PSHA model with nhlib and store result in a SpectralHazardCurveFieldTree
		object.

		:param nrml_base_filespec:
			String, base file specification for NRML output file
			(default: "").
		"""
		# TODO: this method still needs to be updated
		if not nrml_base_filespec:
			os.path.join(self.output_dir, '%s' % self.name)
		else:
			nrml_base_filespec = os.path.splitext(nrml_base_filespec)[0]

		nhlib_params = self._get_nhlib_params()
		num_sites = len(nhlib_params["soil_site_model"])
		hazard_results = {}
		psha_models = self._get_psha_models()
		for imt, periods in self.imt_periods.items():
			hazard_results[imt] = np.zeros((num_sites, self.num_lts_samples, len(periods), self.num_intensities))
		psha_model_names, weights = [], []
		filespecs = ['']*len(psha_models)
		for j, psha_model in enumerate(psha_models):
			print psha_model.name
			psha_model_names.append(psha_model.name)
			weights.append(1./len(psha_models))
			hazard_result = psha_model.run_nhlib_poes(nhlib_params)
			for imt in self.imt_periods.keys():
				hazard_results[imt][:,j,:,:] = hazard_result[imt]
		imtls = self._get_imt_intensities()
		site_names = [site.name for site in self.get_sites()]
		for imt, periods in self.imt_periods.items():
			shcft = SpectralHazardCurveFieldTree(self.name, psha_model_names, filespecs, weights, self.get_sites(), periods, imt, imtls[imt], 'g', self.time_span, poes=hazard_results[imt], site_names=site_names)
			nrml_filespec = nrml_base_filespec + '_%s.xml' % imt
			shcft.write_nrml(nrml_filespec)
		return shcft

	def write_crisis(self, overwrite=True, enumerate_gmpe_lt=False, verbose=True):
		"""
		Write PSHA model tree input for Crisis.

		:param overwrite:
			Boolean, whether or not to overwrite existing input files
			(default: False)
		:param enumerate_gmpe_lt:
			bool, whether or not to enumerate the GMPE logic tree
			(default: False)
		:param verbose:
			bool, whether or not to print some information (default: True)
		"""
		if not os.path.exists(self.output_dir):
			os.mkdir(self.output_dir)
		site_filespec = os.path.join(self.output_dir, 'sites.ASC')
		gsims_dir = os.path.join(self.output_dir, 'gsims')
		if not os.path.exists(gsims_dir):
				os.mkdir(gsims_dir)

		## Create directory structure for logic tree:
		## only possible for source models
		sm_filespecs = {}
		all_filespecs = []
		for source_model in self.source_models:
			sm_filespecs[source_model.name] = []
			folder = os.path.join(self.output_dir, source_model.name)
			if not os.path.exists(folder):
				os.makedirs(folder)
			## If there is only one TRT, it is possible to make subdirectories for each GMPE
			trts = self.gmpe_lt.tectonicRegionTypes
			if len(trts) == 1:
				for gmpe_name in self.gmpe_lt.get_gmpe_names(trts[0]):
					subfolder = os.path.join(folder, gmpe_name)
					if not os.path.exists(subfolder):
						os.makedirs(subfolder)

		## Write CRISIS input files
		max_mag = self.source_model_lt.get_max_mag()
		for i, psha_model in enumerate(self.sample_logic_trees(self.num_lt_samples, enumerate_gmpe_lt=enumerate_gmpe_lt, verbose=verbose)):
			folder = os.path.join(self.output_dir, psha_model.source_model.name)
			if len(trts) == 1:
				folder = os.path.join(folder, psha_model.ground_motion_model[trts[0]])
			filespec = os.path.join(folder, 'lt-rlz-%04d.dat' % (i+1))
			if os.path.exists(filespec) and overwrite:
				os.unlink(filespec)
			## Write separate attenuation tables for different source models
			sm_gsims_dir = os.path.join(gsims_dir, psha_model.source_model.name)
			psha_model.write_crisis(filespec, sm_gsims_dir, site_filespec, atn_Mmax=max_mag)
			sm_filespecs[psha_model.source_model.name].append(filespec)
			all_filespecs.append(filespec)

		# Write CRISIS batch file(s)
		batch_filename = "lt_batch.dat"
		for sm_name in sm_filespecs.keys():
			folder = os.path.join(self.output_dir, sm_name)
			batch_filespec = os.path.join(folder, batch_filename)
			if os.path.exists(batch_filespec):
				if overwrite:
					os.unlink(batch_filespec)
				else:
					print("File %s exists! Set overwrite=True to overwrite." % filespec)
					continue
			of = open(batch_filespec, "w")
			weights = get_uniform_weights(len(sm_filespecs[sm_name]))
			for filespec, weight in zip(sm_filespecs[sm_name], weights):
				of.write("%s, %s\n" % (filespec, weight))
			of.close()

		batch_filespec = os.path.join(self.output_dir, batch_filename)
		if os.path.exists(batch_filespec):
			if overwrite:
				os.unlink(batch_filespec)
			else:
				print("File %s exists! Set overwrite=True to overwrite." % filespec)
				return
		of = open(batch_filespec, "w")
		weights = get_uniform_weights(len(all_filespecs))
		for filespec, weight in zip(all_filespecs, weights):
			of.write("%s, %s\n" % (filespec, weight))
		of.close()

	def _get_psha_models(self):
		"""
		Return list of :class:`PSHAModel` objects, defining sampled PSHA models from logic tree.
		"""
		psha_models = []
		for i in range(self.num_lts_samples):
			source_model, ground_motion_model = self._sample_lts()
			name = source_model.name + '_' + ground_motion_model.name
			psha_models.append(PSHAModel(name, source_model, ground_motion_model, self.output_dir, self.get_sites(), self.grid_outline, self.grid_spacing, self.soil_site_model, self.ref_soil_params, self.imt_periods, self.intensities, self.min_intensities, self.max_intensities, self.num_intensities, self.return_periods, self.time_span, self.truncation_level, self.integration_distance))
		return psha_models

	def _get_used_trts(self):
		"""
		Return list of strings, defining tectonic region types used in source models.
		"""
		used_trts = []
		for source_model in self.source_models:
			for source in source_model:
				trt = source.tectonic_region_type
				if trt not in used_trts:
					used_trts.append(trt)
		return used_trts

	def _get_openquake_trts_gsims_map_lt(self):
		"""
		Return {str: {str: float}} dict, defining respectively tectonic region types, gsims and gsim weight for OpenQuake.
		"""
		trts = self._get_used_trts()
		trts_gsims_map = {}
		for trt in trts:
			trts_gsims_map[trt] = {}
			for ground_motion_model in self.ground_motion_models:
				trts_gsims_map[trt][ground_motion_model[trt]] = 1./len(self.ground_motion_models)
		return trts_gsims_map

	def _enumerate_lts_samples(self):
		"""
		Enumerate logic tree samples.
		"""
		# TODO: this does not take into account the source_model_lt
		for source_model in self.source_models:
			for ground_motion_model in self.ground_motion_models:
				yield source_model, ground_motion_model

	def _sample_lts(self):
		"""
		Return logic tree sample.
		"""
		lts_sampling_methods = {'random': self._sample_lts_random, 'weighted': self._sample_lts_weighted, 'enumerated': self._sample_lts_enumerated}
		lts_sample = lts_sampling_methods[self.lts_sampling_method]()
		return lts_sample

	def _sample_lts_random(self):
		"""
		Return random logic tree sample.
		"""
		source_model = choice(self.source_models)
		ground_motion_model = choice(self.ground_motion_models)
		return source_model, ground_motion_model

	def _sample_lts_weighted(self):
		"""
		Return weighted logic tree sample.
		"""
		# TODO: complete
		pass

	def _sample_lts_enumerated(self):
		"""
		Return enumerated logic tree sample.
		"""
		return self.enumerated_lts_samples.next()


if __name__ == '__main__':
	"""
	"""
	pass

