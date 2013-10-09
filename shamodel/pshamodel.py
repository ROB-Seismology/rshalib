"""
:mod: `rhlib.pshamodel` defines :class:`rhlib.pshamodel.PSHAModelBase`, :class:`rhlib.pshamodel.PSHAModel` and :class:`rhlib.pshamodel.PSHAModelTree`.
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

from ..geo import *
from ..site import *
from ..result import SpectralHazardCurveField, SpectralHazardCurveFieldTree, Poisson, ProbabilityMatrix, DeaggregationSlice
from ..logictree import GroundMotionSystem, SeismicSourceSystem
from ..crisis.IO import writeCRISIS2007
from ..openquake.config import OQ_Params
from ..source import SourceModel



# TODO: make distinction between imt (PGA, SA) and im (SA(0.5, 5.0), SA(1.0, 5.0))


## Minimum and maximum values for random number generator
MIN_SINT_32 = -(2**31)
MAX_SINT_32 = (2**31) - 1


class PSHAModelBase(object):
	"""
	PSHAModelBase holds common attributes and methods for :class:`PSHAModel` and :class:`PSHAModelTree`.

	:param name:
		String, defining name for PSHAModel or PSHAModelTree.
	:param output_dir:
		String, defining full path to output directory.
	:param sites:
		List of (float, float) tuples, defining (longitude, latitude) of sites (default: []).
	:param grid_outline:
		List of (float, float) tuples, defining (longitude, latitude) of points defining grid outline (default: []).
		Minimum number of points is two: left lower corner and upper right corner.
	:param grid_spacing:
		Float, defining grid spacing in degrees (same for lon and lat), or
		Tuple of Floats, grid spacing in degrees (lon and lat separately), or
		String, defining grid spacing in km (same for lon and lat), e.g. '10km'
		(default: 0.5).
	:param soil_site_model:
		:class:`SoilSiteModel` object.
	:param ref_soil_params:
		{"vs30": float, "vs30measured": bool, "z1pt0": float, "z2pt5": float) dict, defining reference soil parameters (default: REF_SOIL_PARAMS).
	:param imt_periods:
		Dictionary mapping intensity measure types (e.g. "PGA", "SA", "PGV", "PGD") to lists or arrays of periods in seconds (float values).
		Periods must be monotonically increasing or decreasing.
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
		Float, defining truncation level in number of standard deviations (default: 3.).
	:param integration_distance:
		Float, defining integration distance in km (default: 200.).
	"""

	def __init__(self, name, output_dir, sites, grid_outline, grid_spacing, soil_site_model, ref_soil_params, imt_periods, intensities, min_intensities, max_intensities, num_intensities, return_periods, time_span, truncation_level, integration_distance):
		"""
		"""
		self.name = name
		self.output_dir = output_dir
		self.soil_site_model = soil_site_model
		self.sites = sites
		self._set_grid_outline(grid_outline)
		if isinstance(grid_spacing, (int, float)):
			## Grid spacing in degrees
			self.grid_spacing = (grid_spacing, grid_spacing)
		else:
			## Grid spacing as string or tuple
			self.grid_spacing = grid_spacing
		self.ref_soil_params = ref_soil_params
		self.imt_periods = imt_periods
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
		self.truncation_level = truncation_level
		self.integration_distance = integration_distance

	def _set_grid_outline(self, grid_outline):
		"""
		Set list of (float, float) tuples for grid outline, defining grid outline with at least three points.
		"""
		# TODO: move to validate method?
		if len(grid_outline) == 1:
			raise Exception("grid_outline must contain 2 points at least")
		## Do not convert to (ll, lr, ur, ul) here because we want to pass the
		## same grid_outline to crisis as well
		## This functionality is now in _get_sites()
		#if len(grid_outline) == 2:
		#	ll, ur = grid_outline
		#	lr = (ur[0], ll[1])
		#	ul = (ll[0], ur[1])
		#	grid_outline = [ll, lr, ur, ul]
		self.grid_outline = grid_outline

	def _get_intensities_limits(self, intensities_limits):
		"""
		Return dict, defining minimum or maximum intensity for intensity type and period.

		:param intensities_limits:
			dict or float
		"""
		if not isinstance(intensities_limits, dict):
			intensities_limits = {imt: [intensities_limits]*len(periods) for imt, periods in self.imt_periods.items()}
		return intensities_limits

	def _get_sites(self):
		"""
		Return list of sites or grid sites.
		"""
		if self.soil_site_model:
			sites = self.soil_site_model.sites
		elif self.sites:
			sites = self.sites
		elif self.grid_outline:
			if len(self.grid_outline) < 2:
				raise Exception("Grid outline should contain at least 2 points")
			elif len(self.grid_outline) == 2:
				## If only 2 points are given, they correspond to the lower left
				## and upper right corners of the grid
				ll, ur = self.grid_outline
				lr = (ur[0], ll[1])
				ul = (ll[0], ur[1])
				grid_outline = [ll, lr, ur, ul]
			else:
				grid_outline = self.grid_outline

			if isinstance(self.grid_spacing, (str, unicode)) and self.grid_spacing[-2:] == 'km':
				grid_spacing_km = float(self.grid_spacing[:-2])
				polygon = Polygon([Point(*site) for site in grid_outline])
				mesh = polygon.discretize(grid_spacing_km)
				sites = [PSHASite(site.longitude, site.latitude) for site in mesh]
			else:
				# TODO: make uniform with crisis
				grid_outline_lons = [pt[0] for pt in grid_outline]
				grid_outline_lats = [pt[1] for pt in grid_outline]
				lon_min, lon_max = min(grid_outline_lons), max(grid_outline_lons)
				lat_min, lat_max = min(grid_outline_lats), max(grid_outline_lats)
				longitudes = np.arange(lon_min, lon_max + float(self.grid_spacing[0]), float(self.grid_spacing[0]))
				latitudes = np.arange(lat_min, lat_max + float(self.grid_spacing[1]), float(self.grid_spacing[1]))
				sites = []
				for lon in longitudes:
					for lat in latitudes:
						sites.append(PSHASite(lon, lat))
				# TODO: remove grid points outside grid_outline !
		return sites

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
		nhlib_params['soil_site_model'] = self._get_nhlib_soil_site_model()
		nhlib_params['imts'] = self._get_nhlib_imts()
		nhlib_params['ssdf'] = nhlib.calc.filters.source_site_distance_filter(self.integration_distance)
		nhlib_params['rsdf'] = nhlib.calc.filters.rupture_site_distance_filter(self.integration_distance)
		return nhlib_params

	def _get_nhlib_soil_site_model(self):
		"""
		Return :class:`SoilSitemodel` object if present, and interpolate sites or grid sites if present, or reference site model.
		"""
		if self.soil_site_model:
			# TODO: implement site model filter for sites or grid (current method is incorrect)
#			if self._get_sites():
#				mesh = nhlib.geo.Mesh(zip(self.sites))
#				return self.soil_site_model.get_closest_points(mesh)
			return self.soil_site_model
		else:
			return self._get_nhlib_ref_soil_site_model()

	def _get_nhlib_ref_soil_site_model(self):
		"""
		Return SoilSiteModel object with ref_soil_params for sites.
		"""
		sites = []
		for site in self._get_sites():
			sites.append(nhlib.site.Site(Point(*site), **self.ref_soil_params))
		soil_site_model = SoilSiteModel('ref_soil_site_model', sites)
		return soil_site_model

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
		if isinstance(self.grid_spacing, (str, unicode)) and self.grid_spacing[-2:] == 'km':
			grid_spacing_km = float(self.grid_spacing[:-2])
		else:
			central_latitude = np.mean([site[1] for site in self.grid_outline])
			grid_spacing_km1 = self._degree_to_km(self.grid_spacing[0], central_latitude)
			grid_spacing_km2 = self._degree_to_km(self.grid_spacing[1])
			grid_spacing_km = min(grid_spacing_km1, grid_spacing_km2)

		return grid_spacing_km

	def _get_grid_spacing_degrees(self, adjust_lat=True):
		"""
		Return grid spacing in degrees as a tuple
		"""
		central_latitude = np.mean([site[1] for site in self.grid_outline])
		if isinstance(self.grid_spacing, (str, unicode)) and self.grid_spacing[-2:] == 'km':
			grid_spacing_km = float(self.grid_spacing[:-2])
			grid_spacing_lon = self._km_to_degree(grid_spacing_km, central_latitude)
			if adjust_lat:
				grid_spacing_lat = self._km_to_degree(grid_spacing_km)
				grid_spacing = (grid_spacing_lon, grid_spacing_lat)
			else:
				grid_spacing = (grid_spacing_lon, grid_spacing_lon)
		elif isinstance(self.grid_spacing, (int, float)):
			if adjust_lat:
				grid_spacing = (self.grid_spacing, self.grid_spacing * np.cos(np.radians(central_latitude)))
			else:
				grid_spacing = (self.grid_spacing, self.grid_spacing)
		else:
			grid_spacing = self.grid_spacing

		return grid_spacing


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
		site_names = [site.name for site in self._get_sites()]
		for imt, periods in self.imt_periods.items():
			# TODO: add method to PSHAModelBase to associate nhlib/OQ imt's with units
			shcf = SpectralHazardCurveField(self.name, [''], self._get_sites(), periods, imt, imtls[imt], 'g', self.time_span, poes=hazard_result[imt], site_names=site_names)
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
		site = self._get_sites()[site_index]
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

	def write_openquake(self, calculation_mode='classical', user_params=None, **kwargs):
		"""
		Write PSHA model input for OpenQuake.

		:param calculation_mode:
			str, calculation mode of OpenQuake (options: "classical" or
				"disaggregation") (default: "classical")
		:param user_params:
			{str, val} dict, defining respectively parameters and value for OpenQuake (default: None).
		"""
		## set OQ_params object and override with params from user_params
		params = OQ_Params(calculation_mode=calculation_mode, description=self.name)
		params.mean_hazard_curves = False
		params.quantile_hazard_curves = []
		if user_params:
			for key in user_params:
				setattr(params, key, user_params[key])

		params.number_of_logic_tree_samples = 1

		## set sites or grid_outline
		if self.grid_outline:
			grid_spacing_km = self._get_grid_spacing_km()
			params.set_grid_or_sites(grid_outline=self.grid_outline, grid_spacing=grid_spacing_km)
		else:
			params.set_grid_or_sites(sites=self.sites)

		## write nrml file for source model
		self.source_model.write_xml(os.path.join(self.output_dir, self.source_model.name + '.xml'))

		## write nrml file for site model if present and set site params
		if self.soil_site_model:
			self.soil_site_model.write_xml(os.path.join(self.output_dir, self.soil_site_model.name + '.xml'))
			params.set_soil_site_model_or_reference_params(soil_site_model_file=self.soil_site_model.name + '.xml')
		else:
			params.set_soil_site_model_or_reference_params(
				reference_vs30_value=self.ref_soil_params["vs30"],
				reference_vs30_type={True: 'measured', False:'inferred'}[self.ref_soil_params["vs30measured"]],
				reference_depth_to_1pt0km_per_sec=self.ref_soil_params["z1pt0"],
				reference_depth_to_2pt5km_per_sec=self.ref_soil_params["z2pt5"])

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
			params.poes = Poisson(life_time=self.time_span, return_period=self.return_periods)

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

	def write_crisis(self, filespec="", atn_folder="", site_filespec="", overwrite=False):
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
		:param overwrite:
			Boolean, whether or not to overwrite existing input files (default: False)

		:return:
			String, full path to CRISIS input .DAT file
		"""
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
		writeCRISIS2007(filespec, self.source_model, self.ground_motion_model, gsim_atn_map, self.return_periods, self.grid_outline, grid_spacing, self.sites, site_filespec, self.imt_periods, self.intensities, self.min_intensities, self.max_intensities, self.num_intensities, 'g', self.name, self.truncation_level, self.integration_distance, source_discretization=(1.0, 5.0), vs30=self.ref_soil_params["vs30"], output={"gra": True, "map": True, "fue": False, "des": False, "smx": True, "eps": False, "res_full": False}, map_filespec="", cities_filespec="", overwrite=overwrite)

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

	:param source_models:
		List of SourceModel objects.
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
	def __init__(self, name, source_models, source_model_lt, gmpe_lt, output_dir, sites=[], grid_outline=[], grid_spacing=0.5, soil_site_model=None, ref_soil_params=REF_SOIL_PARAMS, imt_periods={'PGA': [0]}, intensities=None, min_intensities=0.001, max_intensities=1., num_intensities=100, return_periods=[], time_span=50., truncation_level=3., integration_distance=200., num_lt_samples=1, random_seed=42):
		"""
		"""
		from openquake.engine.input.logictree import LogicTreeProcessor
		PSHAModelBase.__init__(self, name, output_dir, sites, grid_outline, grid_spacing, soil_site_model, ref_soil_params, imt_periods, intensities, min_intensities, max_intensities, num_intensities, return_periods, time_span, truncation_level, integration_distance)
		self.source_models = source_models
		self.source_model_lt = source_model_lt
		self.gmpe_lt = gmpe_lt.get_optimized_system(self.source_models)
		self.num_lt_samples = num_lt_samples
		#self.lts_sampling_method = lts_sampling_method
		#if self.lts_sampling_method == 'enumerated':
		#	self.enumerated_lts_samples = self._enumerate_lts_samples()
		self.random_seed = random_seed
		self.ltp = LogicTreeProcessor(None, source_model_lt=self.source_model_lt, gmpe_lt=self.gmpe_lt)
		self._init_rnd()

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

	def sample_logic_trees(self, num_samples=1):
		"""
		Sample both source-model and GMPE logic trees, in a way that is
		similar to :meth:`_initialize_realizations_montecarlo` of
		:class:`BaseHazardCalculator` in oq-engine

		:param num_samples:
			int, number of random samples
			If zero, :meth:`enumerate_logic_trees` will be called
			(default: 1)

		:return:
			list of instances of :class:`PSHAModel`
		"""
		if num_samples == 0:
			return self.enumerate_logic_trees()

		psha_models = []
		for i in xrange(num_samples):
			## Generate 2nd-order random seeds
			smlt_random_seed = self.rnd.randint(MIN_SINT_32, MAX_SINT_32)
			gmpelt_random_seed = self.rnd.randint(MIN_SINT_32, MAX_SINT_32)

			## Call OQ logictree processor
			sm_name, smlt_path = self.ltp.sample_source_model_logictree(smlt_random_seed)
			gmpelt_path = self.ltp.sample_gmpe_logictree(gmpelt_random_seed)

			## Convert to objects
			source_model = self._smlt_sample_to_source_model(sm_name, smlt_path, verbose=verbose)
			gmpe_model = self._gmpe_sample_to_gmpe_model(path)

			## Convert to PSHA model
			name = "%s : LT sample %04d (SM: %s; GMPE: %s)" % (self.name, i, " -- ".join(smlt_path), " -- ".join(gmpelt_path))
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
			name = "%s : LT enum %04d (SM: %s; GMPE: %s)" % (self.name, i, source_model.name, gmpe_model.name)
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
			sites=self.sites, grid_outline=self.grid_outline, grid_spacing=self.grid_spacing,
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
			bool, whether or not to print some information
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
			bool, whether or not to print some information
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
			bool, whether or not to print some information

		:return:
			instance of :class:`SourceModel`
		"""
		for sm in self.source_models:
			if sm.name == os.path.splitext(sm_name)[0]:
				modified_sources = []
				for src in sm:
					modified_src = copy.deepcopy(src)
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
		name = " -- ".join(path)
		return SourceModel(name, modified_sources)

	def sample_gmpe_lt(self, num_samples=1, verbose=False, show_plot=False):
		"""
		Sample GMPE logic tree

		:param num_samples:
			int, number of random samples
			If zero, :meth:`enumerate_gmpe_lt` will be called
			(default: 1)
		:param verbose:
			bool, whether or not to print some information
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
			bool, whether or not to print some information
		:param show_plot:
			bool, whether or not to plot a diagram of the sampled branch path
			(default: False)

		:return:
			tuple of:
			- list of instances of :class:`SourceModel`, one for each sample
			- list of corresponding weights
		"""
		gmpe_models, weights = [], []
		for gmpelt_path_weight, gmpelt_branches in self.gmpe_lt.root_branchset.enumerate_paths():
			gmpelt_path = [branch.branch_id for branch in gmpelt_branches]
			if verbose:
				print gmpelt_path_weight, gmpelt_path
			if show_plot:
				self.gmpe_lt.plot_diagram(highlight_path=gmpelt_path)
			gmpe_model = self._gmpe_sample_to_gmpe_model(path)
			gmpe_models.append(gmpe_model)
			weights.append(weight)
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

	def write_openquake(self, user_params=None):
		"""
		Write PSHA model tree input for OpenQuake.

		:param user_params:
			{str, val} dict, defining respectively parameters and value for OpenQuake (default: None).
		"""
		## set OQ_params object and override with params from user_params
		params = OQ_Params(calculation_mode='classical', description=self.name)
		if user_params:
			for key in user_params:
				setattr(params, key, user_params[key])

		## set sites or grid_outline
		if self.grid_outline:
			grid_spacing_km = self._get_grid_spacing_km()
			params.set_grid_or_sites(grid_outline=self.grid_outline, grid_spacing=grid_spacing_km)
		else:
			params.set_grid_or_sites(sites=self.sites)

		## write nrml files for source models
		for source_model in self.source_models:
			## make sure source id's are unique among source models
			for source in source_model.sources:
				source.source_id = source_model.name + '--' + source.source_id
			source_model.write_xml(os.path.join(self.output_dir, source_model.name + '.xml'))

		## write nrml file for site model if present and set site params
		if self.soil_site_model:
			self.soil_site_model.write_xml(os.path.join(self.output_dir, self.soil_site_model.name + '.xml'))
			params.set_soil_site_model_or_reference_params(soil_site_model_file=self.soil_site_model.name + '.xml')
		else:
			params.set_soil_site_model_or_reference_params(
				reference_vs30_value=self.ref_soil_params["vs30"],
				reference_vs30_type={True: 'measured', False:'inferred'}[self.ref_soil_params["vs30measured"]],
				reference_depth_to_1pt0km_per_sec=self.ref_soil_params["z1pt0"],
				reference_depth_to_2pt5km_per_sec=self.ref_soil_params["z2pt5"])

		## validate source model logic tree and write nrml file
		self.source_model_lt.validate()
		source_model_lt_file_name = 'source_model_lt.xml'
		self.source_model_lt.write_xml(os.path.join(self.output_dir, source_model_lt_file_name))
		params.source_model_logic_tree_file = source_model_lt_file_name

		## create ground motion model logic tree and write nrml file
		ground_motion_model_lt_file_name = 'ground_motion_model_lt.xml'
		self.ground_motion_model_lt.write_xml(os.path.join(self.output_dir, ground_motion_model_lt_file_name))
		params.gsim_logic_tree_file = ground_motion_model_lt_file_name

		## convert return periods and time_span to poes
		params.poes = Poisson(life_time=self.time_span, return_period=self.return_periods)

		## set other params
		params.intensity_measure_types_and_levels = self._get_openquake_imts()
		params.investigation_time = self.time_span
		params.truncation_level = self.truncation_level
		params.maximum_distance = self.integration_distance

		## write oq params to ini file
		params.write_config(os.path.join(self.output_dir, 'job.ini'))

	def run_nhlib(self, nrml_base_filespec=""):
		"""
		Run PSHA model with nhlib and store result in a SpectralHazardCurveFieldTree
		object.

		:param nrml_base_filespec:
			String, base file specification for NRML output file
			(default: "").
		"""
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
		site_names = [site.name for site in self._get_sites()]
		for imt, periods in self.imt_periods.items():
			shcft = SpectralHazardCurveFieldTree(self.name, psha_model_names, filespecs, weights, self._get_sites(), periods, imt, imtls[imt], 'g', self.time_span, poes=hazard_results[imt], site_names=site_names)
			nrml_filespec = nrml_base_filespec + '_%s.xml' % imt
			shcft.write_nrml(nrml_filespec)
		return shcft

	def write_crisis(self):
		"""
		Write PSHA model tree input for Crisis.
		"""
		# TODO: Needs further work.
		site_filespec = os.path.join(self.output_dir, 'sites.ASC')
		gsims_dir = os.path.join(self.output_dir, 'gsims')
		if not os.path.exists(gsims_dir):
				os.mkdir(gsims_dir)

		## create directory structure for logic tree: not sure this is possible
		for source_model in self.source_models:
			for ground_motion_model in self.ground_motion_models:
				dir = os.path.join(os.path.join(self.output_dir, source_model.name), ground_motion_model.name)
				if not os.path.exists(dir):
					os.makedirs(dir)

		for psha_model in self.sample_logic_trees(self.num_lt_samples):
			filespec = os.path.join(os.path.join(os.path.join(self.output_dir, psha_model.source_model.name), psha_model.ground_motion_model.name), psha_model.name + '.dat')
			psha_model.write_crisis(filespec, gsims_dir, site_filespec)

		# TODO: write CRISIS batch file too

	def _get_psha_models(self):
		"""
		Return list of :class:`PSHAModel` objects, defining sampled PSHA models from logic tree.
		"""
		psha_models = []
		for i in range(self.num_lts_samples):
			source_model, ground_motion_model = self._sample_lts()
			name = source_model.name + '_' + ground_motion_model.name
			psha_models.append(PSHAModel(name, source_model, ground_motion_model, self.output_dir, self.sites, self.grid_outline, self.grid_spacing, self.soil_site_model, self.ref_soil_params, self.imt_periods, self.intensities, self.min_intensities, self.max_intensities, self.num_intensities, self.return_periods, self.time_span, self.truncation_level, self.integration_distance))
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

