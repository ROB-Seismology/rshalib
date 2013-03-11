"""
:mod: `rhlib.pshamodel` defines :class:`rhlib.pshamodel.PSHAModelBase`, :class:`rhlib.pshamodel.PSHAModel` and :class:`rhlib.pshamodel.PSHAModelTree`.
"""

# TODO: check if documentation is compatibele with Sphinx
# NOTE: damping for spectral periods is fixed at 5.

### imports
import numpy as np
import os
from collections import OrderedDict
from random import choice

import openquake.hazardlib as nhlib
from openquake.hazardlib.imt import PGA, SA, PGV, PGD, MMI

from ..geo import *
from ..site import *
from ..result import SpectralHazardCurveField, SpectralHazardCurveFieldTree, Poisson, DeaggregationResult
from ..logictree import GroundMotionSystem, create_basic_seismicSourceSystem
from ..crisis.IO import writeCRISIS2007
from ..openquake.config import OQ_Params



# TODO: make distinction between imt (PGA, SA) and im (SA(0.5, 5.0), SA(1.0, 5.0))

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
	:param site_model:
		:class:`SiteModel` object.
	:param ref_site_params:
		(float, boolean, float, float) tuple, defining (vs30, vs30measured, z1pt0, z2pt5) as reference site parameters (default: (800., False, 100., 1.)).
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
		List of floats, defing return periods.
	:param time_span:
		Float, defining time span in years (default 50.).
	:param truncation_level:
		Float, defining truncation level in number of standard deviations (default: 3.).
	:param integration_distance:
		Float, defining integration distance in km (default: 200.).
	"""
	# TODO: consider grid_outline to be tuple

	def __init__(self, name, output_dir, sites, grid_outline, grid_spacing, site_model, ref_site_params, imt_periods, intensities, min_intensities, max_intensities, num_intensities, return_periods, time_span, truncation_level, integration_distance):
		"""
		"""
		self.name = name
		self.output_dir = output_dir
		self.site_model = site_model
		self.sites = sites
		self._set_grid_outline(grid_outline)
		if isinstance(grid_spacing, (int, float)):
			## Grid spacing in degrees
			self.grid_spacing = (grid_spacing, grid_spacing)
		else:
			## Grid spacing as string or tuple
			self.grid_spacing = grid_spacing
		self.ref_site_params = ref_site_params
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
		if self.site_model:
			sites = self.site_model.sites
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
			'site_model': site model object
			'imts': dictionary mapping intensity measure type objects to intensities
			'ssdf': source_to_site distance filter
			'rsdf': rupture_to_site distance filter
		"""
		nhlib_params = {}
		nhlib_params['site_model'] = self._get_nhlib_site_model()
		nhlib_params['imts'] = self._get_nhlib_imts()
		nhlib_params['ssdf'] = nhlib.calc.filters.source_site_distance_filter(self.integration_distance)
		nhlib_params['rsdf'] = nhlib.calc.filters.rupture_site_distance_filter(self.integration_distance)
		return nhlib_params

	def _get_nhlib_site_model(self):
		"""
		Return :class:`Sitemodel` object if present, and interpolate sites or grid sites if present, or reference site model.
		"""
		if self.site_model:
			# TODO: implement site model filter for sites or grid (current method is incorrect)
#			if self._get_sites():
#				mesh = nhlib.geo.Mesh(zip(self.sites))
#				return self.site_model.get_closest_points(mesh)
			return self.site_model
		else:
			return self._get_nhlib_ref_site_model()

	def _get_nhlib_ref_site_model(self):
		"""
		Return SiteModel object with ref_site_params for sites.
		"""
		sites = []
		for site in self._get_sites():
			sites.append(nhlib.site.Site(Point(*site), *self.ref_site_params))
		site_model = SiteModel('ref_site_model', sites)
		return site_model

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

	def __init__(self, name, source_model, ground_motion_model, output_dir, sites=[], grid_outline=[], grid_spacing=0.5, site_model=None, ref_site_params=(800., False, 100., 2.), imt_periods={'PGA': [0]}, intensities=None, min_intensities=0.001, max_intensities=1., num_intensities=100, return_periods=[], time_span=50., truncation_level=3., integration_distance=200.):

		"""
		"""
		# TODO: consider moving 'name' parameter to third position, to be in accordance with order of parameters in docstring.
		PSHAModelBase.__init__(self, name, output_dir, sites, grid_outline, grid_spacing, site_model, ref_site_params, imt_periods, intensities, min_intensities, max_intensities, num_intensities, return_periods, time_span, truncation_level, integration_distance)
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
			dict containing parameters specific for nhlib, namely 'site_model',
			'imts', 'ssdf', and 'rsdf'. See :class:`PSHAModelBase`.`_get_nhlib_params`
			for an explanation of these keys.

		:return:
			dict {imt (string) : poes (2-D numpy array of poes)}
		"""
		if not nhlib_params:
			nhlib_params = self._get_nhlib_params()
		num_sites = len(nhlib_params['site_model'])
		hazard_curves = nhlib.calc.hazard_curves_poissonian(self.source_model, nhlib_params['site_model'], nhlib_params['imts'], self.time_span, self._get_nhlib_trts_gsims_map(), self.truncation_level, nhlib_params['ssdf'], nhlib_params['rsdf'])
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
			A tuple of two items. First is itself a tuple of bin edges information
			for (in specified order) magnitude, distance, longitude, latitude,
			epsilon and tectonic region types.

			Second item is 6d-array representing the full disaggregation matrix.
			Dimensions are in the same order as bin edges in the first item
			of the result tuple. The matrix can be used directly by pmf-extractor
			functions.
		"""
		if not n_epsilons:
			n_epsilons = 2 * int(np.ceil(self.truncation_level))
		if not mag_bin_width:
			mag_bin_width = self.source_model[0].mfd.bin_width
		site = self._get_sites()[site_index]
		nhlib_site = nhlib.site.Site(Point(*site), *self.ref_site_params)
		#imt = self._get_nhlib_imts()
		ssdf = nhlib.filters.source_site_distance_filter(self.integration_distance)
		rsdf = nhlib.filters.rupture_site_distance_filter(self.integration_distance)

		tom = nhlib.tom.PoissonTOM(self.time_span)
		bin_edges, deagg_matrix = nhlib.calc.disaggregation(self.source_model, nhlib_site, imt, iml, self._get_nhlib_trts_gsims_map(), tom, self.truncation_level, n_epsilons, mag_bin_width, dist_bin_width, coord_bin_width, ssdf, rsdf)
		return DeaggregationResult(bin_edges, deagg_matrix, site, imt, iml, self.time_span)

	def write_openquake(self, user_params=None):
		"""
		Write PSHA model input for OpenQuake.

		:param user_params:
			{str, val} dict, defining respectively parameters and value for OpenQuake (default: None).
		"""
		## set OQ_params object and override with params from user_params
		params = OQ_Params(calculation_mode='classical', description=self.name)
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
		if self.site_model:
			self.site_model.write_xml(os.path.join(self.output_dir, self.site_model.name + '.xml'))
			params.set_site_model_or_reference_params(site_model_file=self.site_model.name + '.xml')
		else:
			params.set_site_model_or_reference_params(
				reference_vs30_value=self.ref_site_params[0],
				reference_vs30_type={True: 'measured', False:'inferred'}[self.ref_site_params[1]],
				reference_depth_to_1pt0km_per_sec=self.ref_site_params[2],
				reference_depth_to_2pt5km_per_sec=self.ref_site_params[3])

		## validate source model logic tree and write nrml file
		source_model_lt = create_basic_seismicSourceSystem([self.source_model])
		source_model_lt.validate()
		source_model_lt_file_name = 'source_model_lt.xml'
		source_model_lt.write_xml(os.path.join(self.output_dir, source_model_lt_file_name))
		params.source_model_logic_tree_file = source_model_lt_file_name

		## create ground_motion_model logic tree and write nrml file
		ground_motion_model_lt = GroundMotionSystem(self.ground_motion_model.name, self._get_openquake_trts_gsims_map())
		ground_motion_model_lt_file_name = 'ground_motion_model_lt.xml'
		ground_motion_model_lt.write_xml(os.path.join(self.output_dir, ground_motion_model_lt_file_name))
		params.gsim_logic_tree_file = ground_motion_model_lt_file_name

		## convert return periods and time_span to poes
		if not self.return_periods in ([], None):
			params.poes_hazard_maps = Poisson(life_time=self.time_span, return_period=self.return_periods)

		## set other params
		params.intensity_measure_types_and_levels = self._get_openquake_imts()
		params.investigation_time = self.time_span
		params.truncation_level = self.truncation_level
		params.maximum_distance = self.integration_distance

		# write oq params to ini file
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

		## Write input file. This will also write the site file and attenuation
		## tables if necessary.
		writeCRISIS2007(filespec, self.source_model, self.ground_motion_model, gsim_atn_map, self.return_periods, self.grid_outline, self.grid_spacing, self.sites, site_filespec, self.imt_periods, self.intensities, self.min_intensities, self.max_intensities, self.num_intensities, 'g', self.name, self.truncation_level, self.integration_distance, source_discretization=(1.0, 5.0), vs30=self.ref_site_params[0], output={"gra": True, "map": True, "fue": False, "des": False, "smx": True, "eps": False, "res_full": False}, map_filespec="", cities_filespec="", overwrite=overwrite)

		## Return name of output file
		return filespec

	def _get_nhlib_trts_gsims_map(self):
		"""
		Return {str, GroundShakingIntensityModel object} dict, defining respectively tectonic region types and gsim for nhlib.
		"""
		#return {trt: NHLIB_GSIMS_MAP[self.ground_motion_model[trt]]() for trt in self._get_used_trts()}
		return {trt: nhlib.gsim.get_available_gsims()[self.ground_motion_model[trt]]() for trt in self._get_used_trts()}

	def _get_openquake_trts_gsims_map(self):
		"""
		Return {str: {str: 1.}} dict, defining respectively tectonic region types, gsim and gsim weight for OpenQuake.
		"""
		trts = self._get_used_trts()
		trts_gsims_map = {}
		for trt in trts:
			trts_gsims_map[trt] = {self.ground_motion_model[trt]: 1.}
		return trts_gsims_map

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
	:param site_model:
		SiteModel object
	:param lts_sampling_method:
		String, defining sampling method for logic trees (options: 'random' and 'enumerated') (default: 'random').
	:param num_lts_samples:
		Integer, defining times to sample logic trees (default: 1).

	See :class:`PSHAModelBase` for other arguments.
	"""
	def __init__(self, name, source_models, source_model_lt, ground_motion_models, output_dir, sites=[], grid_outline=[], grid_spacing=0.5, site_model=None, ref_site_params=(800., False, 100., 2.), imt_periods={'PGA': [0]}, intensities=None, min_intensities=0.001, max_intensities=1., num_intensities=100, return_periods=[], time_span=50., truncation_level=3., integration_distance=200., lts_sampling_method='random', num_lts_samples=1):
		"""
		"""
		PSHAModelBase.__init__(self, name, output_dir, sites, grid_outline, grid_spacing, site_model, ref_site_params, imt_periods, intensities, min_intensities, max_intensities, num_intensities, return_periods, time_span, truncation_level, integration_distance)
		self.source_models = source_models
		self.source_model_lt = source_model_lt
		self.ground_motion_models = ground_motion_models
		self.lts_sampling_method = lts_sampling_method
		self.num_lts_samples = num_lts_samples
		if self.lts_sampling_method == 'enumerated':
			self.enumerated_lts_samples = self._enumerate_lts_samples()

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
		num_sites = len(nhlib_params["site_model"])
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

	def write_openquake(self, user_params=None):
		"""
		Write PSHA model tree input for OpenQuake.

		:param user_params:
			{str, val} dict, defining respectively parameters and value for OpenQuake (default: None).
		"""
		## set OQ_params object and override with params from user_params
		params = OQ_Params(calculation_mode='classical', description = self.name)
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
		if self.site_model:
			self.site_model.write_xml(os.path.join(self.output_dir, self.site_model.name + '.xml'))
			params.set_site_model_or_reference_params(site_model_file=self.site_model.name + '.xml')
		else:
			params.set_site_model_or_reference_params(
				reference_vs30_value=self.ref_site_params[0],
				reference_vs30_type={True: 'measured', False:'inferred'}[self.ref_site_params[1]],
				reference_depth_to_1pt0km_per_sec=self.ref_site_params[2],
				reference_depth_to_2pt5km_per_sec=self.ref_site_params[3])

		## validate source model logic tree or create if not present and write nrml file
		if not self.source_model_lt:
			source_model_lt = create_basic_seismicSourceSystem(self.source_models)
		else:
			self.source_model_lt.validate()
			source_model_lt = self.source_model_lt
		source_model_lt_file_name = 'source_model_lt.xml'
		source_model_lt.write_xml(os.path.join(self.output_dir, source_model_lt_file_name))
		params.source_model_logic_tree_file = source_model_lt_file_name

		## create ground motion model logic tree and write nrml file
		ground_motion_model_lt = GroundMotionSystem('ground_motion_model_lt', self._get_openquake_trts_gsims_map_lt())
		ground_motion_model_lt_file_name = 'ground_motion_model_lt.xml'
		ground_motion_model_lt.write_xml(os.path.join(self.output_dir, ground_motion_model_lt_file_name))
		params.gsim_logic_tree_file = ground_motion_model_lt_file_name

		## convert return periods and time_span to poes
		params.poes_hazard_maps = Poisson(life_time=self.time_span, return_period=self.return_periods)

		## set other params
		params.intensity_measure_types_and_levels = self._get_openquake_imts()
		params.investigation_time = self.time_span
		params.truncation_level = self.truncation_level
		params.maximum_distance = self.integration_distance

		## write oq params to ini file
		params.write_config(os.path.join(self.output_dir, 'job.ini'))

	def write_crisis(self):
		"""
		Write PSHA model tree input for Crisis.
		"""
		# TODO: Needs further work.
		site_filespec = os.path.join(self.output_dir, 'sites.ASC')
		gsims_dir = os.path.join(self.output_dir, 'gsims')
		if not os.path.exists(gsims_dir):
				os.mkdir(gsims_dir)
		## create logic tree structure
		for source_model in self.source_models:
			for ground_motion_model in self.ground_motion_models:
				dir = os.path.join(os.path.join(self.output_dir, source_model.name), ground_motion_model.name)
				if not os.path.exists(dir):
					os.makedirs(dir)
		for psha_model in self._get_psha_models():
			filespec = os.path.join(os.path.join(os.path.join(self.output_dir, psha_model.source_model.name), psha_model.ground_motion_model.name), psha_model.name + '.dat')
			psha_model.write_crisis(filespec, gsims_dir, site_filespec)

	def _get_psha_models(self):
		"""
		Return list of :class:`PSHAModel` objects, defining sampled PSHA models from logic tree.
		"""
		psha_models = []
		for i in range(self.num_lts_samples):
			source_model, ground_motion_model = self._sample_lts()
			name = source_model.name + '_' + ground_motion_model.name
			psha_models.append(PSHAModel(name, source_model, ground_motion_model, self.output_dir, self.sites, self.grid_outline, self.grid_spacing, self.site_model, self.ref_site_params, self.imt_periods, self.intensities, self.min_intensities, self.max_intensities, self.num_intensities, self.return_periods, self.time_span, self.truncation_level, self.integration_distance))
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

