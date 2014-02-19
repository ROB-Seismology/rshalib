"""
:mod:`rhsalib.shamodel.pshamodel` exports :class:`rhlib.pshamodel.PSHAModel` and :class:`rhlib.pshamodel.PSHAModelTree`
"""

# TODO: check if documentation is compatibele with Sphinx
# NOTE: damping for spectral periods is fixed at 5.

# TODO: replace nhlib with oqhazlib

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
from ..result import SpectralHazardCurveField, SpectralHazardCurveFieldTree, Poisson, ProbabilityArray, ProbabilityMatrix, DeaggregationSlice, SpectralDeaggregationCurve
from ..logictree import GroundMotionSystem, SeismicSourceSystem
from ..crisis import write_DAT_2007
from ..openquake import OQ_Params
from ..source import SourceModel
from ..gsim import GroundMotionModel
from ..pmf import get_uniform_weights



# TODO: make distinction between imt (PGA, SA) and im (SA(0.5, 5.0), SA(1.0, 5.0))
# (perhaps put all these functions in rshalib.imt)

# im: intensity measure, e.g. "PGA", "SA"
# imt: IMT object, e.g. PGA(), SA(0.2, 5.0)
# imls: intensity measure levels, e.g. [0.1, 0.2, 0.3]
# im_periods: dict mapping im to spectral periods, e.g. {"PGA": [0], "SA": [0.1, 0.5, 1.]}
# imtls --> imt_imls: dict mapping IMT objects to imls (1-D arrays)
# im_imls: dict mapping im strings to imls (2-D arrays)


## Minimum and maximum values for random number generator
MIN_SINT_32 = -(2**31)
MAX_SINT_32 = (2**31) - 1



class PSHAModelBase(SHAModelBase):
	"""
	Base class for PSHA models, holding common attributes and methods.

	:param root_folder:
		String, defining full path to root folder.
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

	def __init__(self, name, root_folder, sites, grid_outline, grid_spacing, soil_site_model, ref_soil_params, imt_periods, intensities, min_intensities, max_intensities, num_intensities, return_periods, time_span, truncation_level, integration_distance):
		"""
		"""
		SHAModelBase.__init__(self, name, sites, grid_outline, grid_spacing, soil_site_model, ref_soil_params, imt_periods, truncation_level, integration_distance)

		self.root_folder = root_folder
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

	@property
	def source_site_filter(self):
		return nhlib.calc.filters.source_site_distance_filter(self.integration_distance)

	@property
	def rupture_site_filter(self):
		return nhlib.calc.filters.rupture_site_distance_filter(self.integration_distance)

	@property
	def poisson_tom(self):
		return nhlib.tom.PoissonTOM(self.time_span)

	@property
	def oq_root_folder(self):
		return os.path.join(self.root_folder, "openquake")

	@property
	def crisis_root_folder(self):
		return os.path.join(self.root_folder, "crisis")

	@property
	def oq_output_folder(self):
		return os.path.join(self.oq_root_folder, "computed_output")

	def get_oq_hc_folder(self, calc_id=None, multi=False):
		"""
		Return full path to OpenQuake hazard_curve folder

		:param calc_id:
			int or str, OpenQuake calculation ID (default: None, will
				be determined automatically)
		:param multi:
			bool, whether or not path to multi_folder should be returned
			(default: False)

		:return:
			str, path spec
		"""
		folder = os.path.join(self.oq_output_folder, "classical")
		if calc_id is None:
			calc_id = self._get_oq_calc_id(folder)
		hc_folder = os.path.join(folder, "calc_" + calc_id, "hazard_curve")
		if multi:
			hc_folder += "_multi"
		return hc_folder

	def get_oq_uhs_folder(self, calc_id=None):
		"""
		Return full path to OpenQuake uhs folder

		:param calc_id:
			int or str, OpenQuake calculation ID (default: None, will
				be determined automatically)

		:return:
			str, path spec
		"""
		folder = os.path.join(self.oq_output_folder, "classical")
		if calc_id is None:
			calc_id = self._get_oq_calc_id(folder)
		uhs_folder = os.path.join(folder, "calc_" + calc_id, "uh_spectra")
		return uhs_folder

	def get_oq_hm_folder(self, calc_id=None):
		"""
		Return full path to OpenQuake hazard-map folder

		:param calc_id:
			int or str, OpenQuake calculation ID (default: None, will
				be determined automatically)

		:return:
			str, path spec
		"""
		folder = os.path.join(self.oq_output_folder, "classical")
		if calc_id is None:
			calc_id = self._get_oq_calc_id(folder)
		hm_folder = os.path.join(folder, "calc_" + calc_id, "hazard_map")
		return hm_folder

	def get_oq_disagg_folder(self, calc_id=None, multi=False):
		"""
		Return full path to OpenQuake disaggregation folder

		:param calc_id:
			int or str, OpenQuake calculation ID (default: None, will
				be determined automatically)
		:param multi:
			bool, whether or not path to multi_folder should be returned
			(default: False)

		:return:
			str, path spec
		"""
		folder = os.path.join(self.oq_output_folder, "disaggregation")
		if calc_id is None:
			calc_id = self._get_oq_calc_id(folder)
		disagg_folder = os.path.join(folder, "calc_" + calc_id, "disagg_matrix")
		if multi:
			disagg_folder += "_multi"
		return disagg_folder

	def _get_oq_calc_id(self, folder):
		"""
		Get OpenQuake calculation ID. If there is more than one calc_
		subfolder, the first one will be returned.

		:param folder:
			str, parent folder where calc_ subfolders are stored

		:return:
			str or None, calc_id
		"""
		for entry in os.listdir(folder):
			if entry[:5] == "calc_":
				return entry.split('_')[1]

	def _get_intensities_limits(self, intensities_limits):
		"""
		Return dict, defining minimum or maximum intensity for intensity type and period.

		:param intensities_limits:
			dict or float
		"""
		if not isinstance(intensities_limits, dict):
			intensities_limits = {imt: [intensities_limits]*len(periods) for imt, periods in self.imt_periods.items()}
		return intensities_limits

	def _get_im_imls(self, combine_pga_and_sa=True):
		"""
		Construct a dictionary containing a 2-D array [k, l] of intensities for each IMT.

		:param combine_pga_and_sa:
			bool, whether or not to combine PGA and SA, if present
			(default: True)

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
		if combine_pga_and_sa and "PGA" in self.imt_periods.keys() and "SA" in self.imt_periods.keys():
			imtls["PGA"].shape = (1, self.num_intensities)
			imtls["SA"] = np.concatenate([imtls["PGA"], imtls["SA"]], axis=0)
			del imtls["PGA"]
		return imtls

	def _get_imtls(self):
		"""
		Construct a dictionary mapping nhlib intensity measure type objects
		to 1-D arrays of intensity measure levels. This dictionary can be passed
		as an argument to the nhlib.calc.hazard_curves_poissonian function.

		:return:
			dict {:mod:`nhlib.imt` object: 1-D numpy array of floats}
		"""
		imtls = OrderedDict()
		for im, periods in sorted(self.imt_periods.items()):
			if im == "SA":
				for k, period in enumerate(periods):
					imt = getattr(nhlib.imt, im)(period, damping=5.)
					if self.intensities:
						imtls[imt] = np.array(self.intensities)
					else:
						imtls[imt] = np.logspace(np.log10(self.min_intensities[im][k]), np.log10(self.max_intensities[im][k]), self.num_intensities)
			else:
				imt = getattr(nhlib.imt, im)()
				if self.intensities:
					imtls[imt] = np.array(self.intensities)
				else:
					imtls[imt] = np.logspace(np.log10(self.min_intensities[im][0]), np.log10(self.max_intensities[im][0]), self.num_intensities)
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

	def _handle_oq_soil_params(self, params):
		"""
		Write nrml file for soil site model if present and set file param,
		or set reference soil params

		:param params:
			instance of :class:`OQ_Params` where soil parameters will
			be added.
		"""
		if self.soil_site_model:
			file_name = (self.soil_site_model.name or "soil_site_model") + ".xml"
			self.soil_site_model.write_xml(os.path.join(self.oq_root_folder, file_name))
			params.set_soil_site_model_or_reference_params(soil_site_model_file=file_name)
		else:
			params.set_soil_site_model_or_reference_params(
				reference_vs30_value=self.ref_soil_params["vs30"],
				reference_vs30_type={True: 'measured', False:'inferred'}[self.ref_soil_params["vs30measured"]],
				reference_depth_to_1pt0km_per_sec=self.ref_soil_params["z1pt0"],
				reference_depth_to_2pt5km_per_sec=self.ref_soil_params["z2pt5"],
				reference_kappa=self.ref_soil_params.get("kappa", None))

	def read_oq_hcf(self, curve_name, im, T, calc_id=None):
		"""
		Read OpenQuake hazard curve field

		:param curve_name:
			str, identifying hazard curve (e.g., "rlz-01", "mean", "quantile_0.84")
		:param im:
			str, intensity measure
		:param T:
			float, spectral period
		:param calc_id:
			str, calculation ID. (default: None, will determine from folder structure)

		:return:
			instance of :class:`HazardCurveField`
		"""
		from ..openquake import parse_hazard_curves

		hc_folder = self.get_oq_hc_folder(calc_id=calc_id)
		if im == "SA":
			imt_subfolder = "SA-%s" % T
		else:
			imt_subfolder = im
		xml_filename = "hazard_curve-%s.xml" % curve_name
		#print xml_filename
		xml_filespec = os.path.join(hc_folder, imt_subfolder, xml_filename)
		hcf = parse_hazard_curves(xml_filespec)
		hcf.set_site_names(self.get_sha_sites())

		return hcf

	def read_oq_uhs_multi(self):
		# TODO
		pass

	def read_oq_uhs_field(self, curve_name, return_period, calc_id=None):
		"""
		Read OpenQuake hazard curve field

		:param curve_name:
			str, identifying hazard curve (e.g., "rlz-01", "mean", "quantile_0.84")
		:param return period:
			float, return period
		:param calc_id:
			str, calculation ID. (default: None, will determine from folder structure)

		:return:
			instance of :class:`UHSField`
		"""
		from ..result import Poisson
		from ..openquake import parse_uh_spectra

		poe = str(round(Poisson(life_time=self.life_time, return_period=return_period), 13))

		# TODO: site lon, lat in filename
		uhs_folder = self.get_oq_uhs_folder(calc_id=calc_id)
		xml_filename = "uh_spectra-poe_%s%s.xml" % (poe, curve_name)
		#print xml_filename
		xml_filespec = os.path.join(uhs_folder, xml_filename)
		uhsf = parse_uh_spectra(xml_filespec)
		uhsf.set_site_names(self.get_sha_sites())

		return uhsf

	def read_oq_disagg_matrix(self, curve_name, im, T, return_period, site, calc_id=None):
		"""
		Read OpenQuake deaggregation matrix for a particular im, spectral period,
		return period and site.

		:param curve_name:
			str, identifying hazard curve (e.g., "rlz-01", "mean", "quantile_0.84")
		:param im:
			str, intensity measure
		:param T:
			float, spectral period
		:param return period:
			float, return period
		:param site:
			instance of :class:`SHASite` or :class:`SoilSite`
		:param calc_id:
			str, calculation ID. (default: None, will determine from folder structure)

		:return:
			instance of :class:`DeaggregationSlice`
		"""
		poe = str(round(Poisson(life_time=self.life_time, return_period=return_period), 13))

		disagg_folder = self.get_oq_disagg_folder(calc_id=calc_id, multi=False)
		xml_filename = "disagg_matrix(%s)-lon_%s-lat_%s-%s.xml"
		xml_filename %= (poe, site.lon, site.lat, curve_name)
		xml_filespec = os.path.join(uhs_folder, xml_filename)
		ds = parse_disaggregation(xml_filespec, site.name)
		return ds

	def read_oq_disagg_matrix_full(self):
		# TODO
		pass

	def read_oq_disagg_matrix_multi(self, curve_name, site, calc_id=None):
		"""
		Read OpenQuake multi-deaggregation matrix for a particular site.

		:param curve_name:
			str, identifying hazard curve (e.g., "rlz-01", "mean", "quantile_0.84")
		:param site:
			instance of :class:`SHASite` or :class:`SoilSite`
		:param calc_id:
			str, calculation ID. (default: None, will determine from folder structure)

		:return:
			instance of :class:`SpectralDeaggregationCurve`
		"""
		from ..openquake import parse_spectral_deaggregation_curve

		disagg_folder = self.get_oq_disagg_folder(calc_id=calc_id, multi=True)
		xml_filename = "disagg_matrix_multi-lon_%s-lat_%s-%s.xml"
		xml_filename %= (site.lon, site.lat, curve_name)
		xml_filespec = os.path.join(disagg_folder, xml_filename)
		sdc = parse_spectral_deaggregation_curve(xml_filespec, site.name)
		return sdc

	def read_crisis_batch(self):
		"""
		Reach CRISIS batch file

		:return:
			list of gra_filespecs
		"""
		from ..crisis import read_batch

		batch_filename = "lt_batch.dat"
		batch_filespec = os.path.join(self.crisis_root_folder, batch_filename)
		#print batch_filespec
		return read_batch(batch_filespec)

	def read_crisis_shcf(self, curve_name):
		"""
		Read CRISIS spectral hazard curve field

		:param curve_name:
			str, identifying hazard curve (e.g., "rlz-01")

		:return:
			instance of :class:`SpectralHazardCurveField`
		"""
		from ..crisis import read_GRA

		gra_filespecs, weights = self.read_crisis_batch(test_case)
		for gra_filespec in gra_filespecs:
			gra_filename = os.path.split(gra_filespec)[1]
			if curve_name in gra_filename:
				break
		#print gra_filename

		shcf = read_GRA(gra_filespec)
		return shcf


class PSHAModel(PSHAModelBase):
	"""
	Class representing a single PSHA model.

	:param source_model:
		SourceModel object.
	:param ground_motion_model:
		GroundMotionModel object.

	See :class:`PSHAModelBase` for other arguments.
	"""

	def __init__(self, name, source_model, ground_motion_model, root_folder, sites=[], grid_outline=[], grid_spacing=0.5, soil_site_model=None, ref_soil_params=REF_SOIL_PARAMS, imt_periods={'PGA': [0]}, intensities=None, min_intensities=0.001, max_intensities=1., num_intensities=100, return_periods=[], time_span=50., truncation_level=3., integration_distance=200.):

		"""
		"""
		# TODO: consider moving 'name' parameter to third position, to be in accordance with order of parameters in docstring.
		PSHAModelBase.__init__(self, name, root_folder, sites, grid_outline, grid_spacing, soil_site_model, ref_soil_params, imt_periods, intensities, min_intensities, max_intensities, num_intensities, return_periods, time_span, truncation_level, integration_distance)
		self.source_model = source_model
		self.ground_motion_model = ground_motion_model

	@property
	def smlt_path(self):
		return self.source_model.description

	@property
	def gmpelt_path(self):
		return self.ground_motion_model.name

	def calc_shcf(self, cav_min=0., combine_pga_and_sa=True):
		"""
		Run PSHA model with nhlib, and store result in one or more
		SpectralHazardCurfeField objects.

		:param cav_min:
			float, CAV threshold in g.s (default: 0. = no CAV filtering).
		:param combine_pga_and_sa:
			bool, whether or not to combine PGA and SA, if present
			(default: True)

		:return:
			dict {imt (string) : SpectralHazardCurveField object}
		"""
		hazard_result = self.calc_poes(cav_min=cav_min, combine_pga_and_sa=combine_pga_and_sa)
		im_imls = self._get_im_imls(combine_pga_and_sa=combine_pga_and_sa)
		im_shcf_dict = {}
		site_names = [site.name for site in self.get_sha_sites()]
		for imt in hazard_result.keys():
			periods = self.imt_periods[imt]
			if imt == "SA" and combine_pga_and_sa and "PGA" in self.imt_periods.keys():
				periods = [0] + list(periods)
			# TODO: add method to PSHAModelBase to associate nhlib/OQ imt's with units
			poes = ProbabilityArray(hazard_result[imt])
			shcf = SpectralHazardCurveField(self.name, poes, [''], self.get_sha_sites(), periods, imt, im_imls[imt], 'g', self.time_span)
			im_shcf_dict[imt] = shcf
		return im_shcf_dict

	def calc_poes(self, cav_min=0., combine_pga_and_sa=True):
		"""
		Run PSHA model with nhlib. Output is a dictionary mapping intensity
		measure types to probabilities of exceedance (poes).

		:param cav_min:
			float, CAV threshold in g.s (default: 0. = no CAV filtering).
		:param combine_pga_and_sa:
			bool, whether or not to combine PGA and SA, if present
			(default: True)

		:return:
			dict {imt (string) : poes (2-D numpy array of poes)}
		"""
		num_sites = len(self.get_soil_site_model())
		hazard_curves = nhlib.calc.hazard_curves_poissonian(self.source_model, self.get_soil_site_model(), self._get_imtls(), self.time_span, self._get_trt_gsim_dict(), self.truncation_level, self.source_site_filter, self.rupture_site_filter, cav_min=cav_min)
		hazard_result = {}
		for imt, periods in self.imt_periods.items():
			if imt == "SA":
				poes = np.zeros((num_sites, len(periods), self.num_intensities))
				for k, period in enumerate(periods):
					poes[:,k,:] = hazard_curves[eval(imt)(period, 5.)]
				hazard_result[imt] = poes
			else:
				poes = hazard_curves[eval(imt)()].reshape(num_sites, 1, self.num_intensities)
				hazard_result[imt] = poes
		if combine_pga_and_sa and "PGA" in self.imt_periods.keys() and "SA" in self.imt_periods.keys():
			hazard_result["SA"] = np.concatenate([hazard_result["PGA"], hazard_result["SA"]], axis=1)
			del hazard_result["PGA"]
		return hazard_result

	def calc_shcf_mp(self, cav_min=0, decompose_area_sources=False, individual_sources=False, num_cores=None, verbose=True):
		"""
		Parallellized computation of spectral hazard curve field.

		Note: at least in Windows, this method has to be executed in
		a main section (i.e., behind if __name__ == "__main__":)

		:param cav_min:
			float, CAV threshold in g.s (default: 0)
		:param decompose_area_sources:
			bool, whether or not area sources should be decomposed into
			point sources for the computation (default: False)
		:param individual_sources:
			bool, whether or not hazard curves should be computed for each
			source individually (default: False)
		:param num_cores:
			int, number of CPUs to be used. Actual number of cores used
			may be lower depending on available cores and memory
			(default: None, will determine automatically)
		:param verbose:
			bool, whether or not to print some progress information
			(default: True)

		:return:
			instance of :class:`SpectralHazardCurveField` (if group_sources
			is True) or dict mapping source IDs to instances of
			:class:`SpectralHazardCurveField` (if group_sources is False)
		"""
		import mp

		if not num_cores:
			num_cores = mp.multiprocessing.cpu_count()

		if decompose_area_sources:
			source_model = self.source_model.decompose_area_sources()
		else:
			source_model = self.source_model

		## Create list with arguments for each job
		job_args = []
		for source in source_model:
			job_args.append((self, source, cav_min, verbose))

		## Launch multiprocessing
		curve_list = mp.run_parallel(mp.calc_shcf_by_source, job_args, num_cores)
		poes = ProbabilityArray(curve_list)

		## Recombine hazard curves computed for each source
		if not individual_sources:
			poes = np.prod(poes, axis=0)
		else:
			total_poes = np.prod(poes, axis=0)
			if decompose_area_sources:
				curve_list = []
				prev_num_pts = 0
				for src in self.source_model:
					num_pts = len(src.polygon.discretize(src.area_discretization))
					curve_list.append(np.prod(poes[prev_num_pts:prev_num_pts+num_pts], axis=0))
					prev_num_pts += num_pts
				poes = ProbabilityArray(curve_list)

		## Convert non-exceedance to exceedance probabilities
		poes -= 1
		poes *= -1
		if individual_sources:
			total_poes -= 1
			total_poes *= -1

		## Construct spectral hazard curve field
		# TODO: use _get_im_imls, and return shcf_dict, and correct order of periods !!
		sites = self.get_sites()
		imtls = self._get_imtls()
		ims = self.imt_periods.keys()
		periods = []
		for im in sorted(ims):
			periods.extend(self.imt_periods[im])
		intensities = []
		for im in sorted(ims):
			for T in self.imt_periods[im]:
				if T == 0:
					imt = getattr(nhlib.imt, im)()
				else:
					imt = getattr(nhlib.imt, im)(T, damping=5)
				intensities.append(imtls[imt])
		periods = np.array(periods)
		intensities = np.array(intensities)
		if len(ims) == 1:
			im = ims[0]
		else:
			im = "SA"

		if individual_sources:
			shcf_dict = OrderedDict()
			for i, src in enumerate(self.source_model):
				shcf_dict[src.source_id] = SpectralHazardCurveField(self.name,
												poes[i], [""]*len(periods), sites,
												periods, im, intensities, 'g',
												self.time_span)
				shcf_dict['Total'] = SpectralHazardCurveField(self.name, total_poes,
											[""]*len(periods), sites, periods, im,
											intensities, 'g', self.time_span)
			return shcf_dict
		else:
			shcf = SpectralHazardCurveField(self.name, poes, [""]*len(periods),
							sites, periods, im, intensities, 'g', self.time_span)
			return shcf

	def deagg_nhlib(self, site, imt, iml, mag_bin_width=None, dist_bin_width=10., n_epsilons=None, coord_bin_width=1.0):
		"""
		Run deaggregation with nhlib

		:param site:
			instance of :class:`SHASite` or :class:`SoilSite`
		:param imt:
			Instance of :class:`nhlib.imt._IMT`, intensity measure type
		:param iml:
			Float, intensity measure level
		:param mag_bin_width:
			Float, magnitude bin width (default: None, will take MFD bin width
				of first source)
		:param dist_bin_width:
			Float, distance bin width in km (default: 10.)
		:param n_epsilons:
			Int, number of epsilon bins (default: None, will result in bins
				corresponding to integer epsilon values)
		:param coord_bin_width:
			Float, lon/lat bin width in decimal degrees (default: 1.)

		:return:
			instance of :class:`DeaggregationSlice`
		"""
		if not n_epsilons:
			n_epsilons = 2 * int(np.ceil(self.truncation_level))
		if not mag_bin_width:
			mag_bin_width = self.source_model[0].mfd.bin_width
		if not isinstance(site, SoilSite):
			site = site.to_soil_site(self.ref_soil_params)
		#imt = self._get_imtls()
		ssdf = nhlib.calc.filters.source_site_distance_filter(self.integration_distance)
		rsdf = nhlib.calc.filters.rupture_site_distance_filter(self.integration_distance)

		#tom = nhlib.tom.PoissonTOM(self.time_span)
		#bin_edges, deagg_matrix = nhlib.calc.disaggregation(self.source_model, nhlib_site, imt, iml, self._get_trt_gsim_dict(), tom, self.truncation_level, n_epsilons, mag_bin_width, dist_bin_width, coord_bin_width, ssdf, rsdf)
		bin_edges, deagg_matrix = nhlib.calc.disaggregation_poissonian(self.source_model, site, imt, iml, self._get_trt_gsim_dict(), self.time_span, self.truncation_level, n_epsilons, mag_bin_width, dist_bin_width, coord_bin_width, ssdf, rsdf)
		deagg_matrix = ProbabilityMatrix(deagg_matrix)
		imt_name = str(imt).split('(')[0]
		if imt_name == "SA":
			period = imt.period
		else:
			period = 0
		return DeaggregationSlice(bin_edges, deagg_matrix, site, imt_name, iml, period, self.time_span)

	def deagg_nhlib_multi(self, site_imtls, mag_bin_width=None, dist_bin_width=10., n_epsilons=None, coord_bin_width=1.0):
		"""
		Run deaggregation with nhlib for multiple sites, multiple imt's
		per site, and multiple iml's per iml

		:param site_imtls:
			nested dictionary mapping (lon, lat) tuples to dictionaries
			mapping nhlib intensity measure type objects to 1-D arrays
			of intensity measure levels
		:param mag_bin_width:
			Float, magnitude bin width (default: None, will take MFD bin width
				of first source)
		:param dist_bin_width:
			Float, distance bin width in km (default: 10.)
		:param n_epsilons:
			Int, number of epsilon bins (default: None, will result in bins
				corresponding to integer epsilon values)
		:param coord_bin_width:
			Float, lon/lat bin width in decimal degrees (default: 1.)

		:return:
			instance of :class:`SpectralDeaggregationCurve` or None
		"""
		if not n_epsilons:
			n_epsilons = 2 * int(np.ceil(self.truncation_level))
		if not mag_bin_width:
			mag_bin_width = self.source_model[0].mfd.bin_width

		ssdf = nhlib.calc.filters.source_site_distance_filter(self.integration_distance)
		rsdf = nhlib.calc.filters.rupture_site_distance_filter(self.integration_distance)

		site_model = self.get_soil_site_model()
		all_sites = site_model.get_sha_sites()
		deagg_soil_sites = [site for site in site_model.get_sites() if (site.lon, site.lat) in site_imtls.keys()]
		deagg_site_model = SoilSiteModel("", deagg_soil_sites)
		for deagg_result in nhlib.calc.disaggregation_poissonian_multi(self.source_model, deagg_site_model, site_imtls, self._get_trt_gsim_dict(), self.time_span, self.truncation_level, n_epsilons, mag_bin_width, dist_bin_width, coord_bin_width, ssdf, rsdf):
			deagg_site, bin_edges, deagg_matrix = deagg_result
			if (bin_edges, deagg_matrix) == (None, None):
				## No deaggregation results for this site
				yield None
			else:
				for site in all_sites:
					if deagg_site.location.longitude == site.lon and deagg_site.location.latitude == site.lat:
						break
				imtls = site_imtls[(site.lon, site.lat)]
				imts = imtls.keys()
				periods = [getattr(imt, "period", 0) for imt in imts]
				intensities = np.array([imtls[imt] for imt in imts])
				deagg_matrix = ProbabilityMatrix(deagg_matrix)
				yield SpectralDeaggregationCurve(bin_edges, deagg_matrix, site, "SA", intensities, periods, self.time_span)

	def get_deagg_bin_edges(self, mag_bin_width, dist_bin_width, n_epsilons, coord_bin_width):
		"""
		Determine bin edges for deaggregation.
		Note: no default values!

		:param mag_bin_width:
			Float, magnitude bin width
		:param dist_bin_width:
			Float, distance bin width in km
		:param n_epsilons:
			Int, number of epsilon bins
			corresponding to integer epsilon values)
		:param coord_bin_width:
			Float, lon/lat bin width in decimal degrees

		:return:
			(mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, src_bins) tuple
			- mag_bins: magnitude bin edges
			- dist_bins: distance bin edges
			- lon_bins: longitude bin edges
			- lat_bins: latitude bin edges
			- eps_bins: epsilon bin edges
			- src_bins: source bins
		"""
		from openquake.hazardlib.geo.geodetic import npoints_between
		from openquake.hazardlib.geo.utils import get_longitudinal_extent

		## (copied from oqhazlib)
		min_mag, max_mag = self.source_model.min_mag, self.source_model.max_mag
		mag_bins = mag_bin_width * np.arange(
			int(np.floor(min_mag / mag_bin_width)),
			int(np.ceil(max_mag / mag_bin_width) + 1)
		)

		min_dist, max_dist = 0, self.integration_distance
		dist_bins = dist_bin_width * np.arange(
			int(np.floor(min_dist / dist_bin_width)),
			int(np.ceil(max_dist / dist_bin_width) + 1)
		)

		## Note that ruptures may extend beyond source limits!
		west, east, south, north = self.source_model.get_bounding_box()
		west -= coord_bin_width
		east += coord_bin_width
		west = np.floor(west / coord_bin_width) * coord_bin_width
		east = np.ceil(east / coord_bin_width) * coord_bin_width
		lon_extent = get_longitudinal_extent(west, east)
		lon_bins, _, _ = npoints_between(
			west, 0, 0, east, 0, 0,
			np.round(lon_extent / coord_bin_width) + 1
		)

		# Note: why is this different from lon_bins?
		south -= coord_bin_width
		north += coord_bin_width
		lat_bins = coord_bin_width * np.arange(
			int(np.floor(south / coord_bin_width)),
			int(np.ceil(north / coord_bin_width) + 1)
		)

		eps_bins = np.linspace(-self.truncation_level, self.truncation_level,
								  n_epsilons + 1)

		src_bins = [src.source_id for src in self.source_model]

		return (mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, src_bins)

	def deaggregate(self, site_imtls, mag_bin_width=None, dist_bin_width=10., n_epsilons=None, coord_bin_width=1.0, dtype='d', verbose=False):
		"""
		Hybrid rshalib/oqhazlib deaggregation for multiple sites, multiple
		imt's per site, and multiple iml's per iml, that is more speed- and
		memory-efficient than the standard deaggregation method in oqhazlib.
		Note that deaggregation by tectonic region type is replaced with
		deaggregation by source.

		:param site_imtls:
			nested dictionary mapping (lon, lat) tuples to dictionaries
			mapping oqhazlib IMT objects to 1-D arrays of intensity measure
			levels
		:param mag_bin_width:
			Float, magnitude bin width (default: None, will take MFD bin width
			of first source)
		:param dist_bin_width:
			Float, distance bin width in km (default: 10.)
		:param n_epsilons:
			Int, number of epsilon bins (default: None, will result in bins
			corresponding to integer epsilon values)
		:param coord_bin_width:
			Float, lon/lat bin width in decimal degrees (default: 1.)
		:param dtype:
			str, precision of deaggregation matrix (default: 'd')
		:param verbose:
			Bool, whether or not to print some progress information

		:return:
			dict, mapping site (lon, lat) tuples to instances of
			:class:`SpectralDeaggregationCurve`
		"""
		# TODO: determine site_imtls from self.return_periods (separate method)
		from openquake.hazardlib.site import SiteCollection

		if site_imtls in (None, {}):
			pass

		if not n_epsilons:
			n_epsilons = 2 * int(np.ceil(self.truncation_level))
		if not mag_bin_width:
			mag_bin_width = self.source_model[0].mfd.bin_width

		## Determine bin edges first
		bin_edges = self.get_deagg_bin_edges(mag_bin_width, dist_bin_width, n_epsilons, coord_bin_width)
		mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, src_bins = bin_edges

		## Create deaggregation matrices
		deagg_matrix_dict = {}
		for site_key in site_imtls.keys():
			deagg_matrix_dict[site_key] = {}
			imtls = site_imtls[site_key]
			imts = imtls.keys()
			num_imts = len(imts)
			num_imls = len(imtls[imts[0]])

			deagg_matrix_shape = (num_imts, num_imls, len(mag_bins) - 1, len(dist_bins) - 1, len(lon_bins) - 1,
						len(lat_bins) - 1, len(eps_bins) - 1, len(src_bins))

			## Initialize array with ones representing NON-exceedance probabilities !
			deagg_matrix = ProbabilityMatrix(np.ones(deagg_matrix_shape, dtype=dtype))
			deagg_matrix_dict[site_key] = deagg_matrix

		## Perform deaggregation
		tom = self.poisson_tom
		gsims = self._get_trt_gsim_dict()
		source_site_filter = self.source_site_filter
		rupture_site_filter = self.rupture_site_filter

		site_model = self.get_soil_site_model()
		deagg_soil_sites = [site for site in site_model.get_sites() if (site.lon, site.lat) in site_imtls.keys()]
		deagg_site_model = SoilSiteModel("", deagg_soil_sites)

		sources = self.source_model.sources
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
					for imt_idx, imt in enumerate(imts):
						imls = imtls[imt]
						## In contrast to what is stated in the documentation,
						## disaggregate_poe does handle more than one iml
						poes_given_rup_eps = gsim.disaggregate_poe(
							sctx2, rctx2, dctx2, imt, imls, self.truncation_level, n_epsilons
						)

						## Probability of non-exceedance
						pone = (1. - prob_one_or_more) ** poes_given_rup_eps

						try:
							deagg_matrix_dict[site_key][imt_idx, :, mag_idx, dist_idx, lon_idx, lat_idx, :, src_idx] *= pone
						except IndexError:
							## May fail if rupture extent is beyond (lon,lat) range
							pass

		## Create SpectralDeaggregationCurve for each site
		deagg_result = {}
		all_sites = site_model.get_sha_sites()
		for deagg_site in deagg_site_model:
			for site in all_sites:
				if deagg_site.location.longitude == site.lon and deagg_site.location.latitude == site.lat:
					break
			site_key = (site.lon, site.lat)
			imtls = site_imtls[site_key]
			imts = imtls.keys()
			periods = [getattr(imt, "period", 0) for imt in imts]
			intensities = np.array([imtls[imt] for imt in imts])
			deagg_matrix = deagg_matrix_dict[site_key]
			## Convert probabilities of non-exceedance back to poes
			#deagg_matrix = 1 - deagg_matrix
			## Modify matrix in-place to save memory
			deagg_matrix -= 1
			deagg_matrix *= -1
			deagg_result[site_key] = SpectralDeaggregationCurve(bin_edges,
										deagg_matrix, site, "SA", intensities,
										periods, self.time_span)

		return deagg_result

	def deaggregate_mp(self, site_imtls, mag_bin_width=None, dist_bin_width=10., n_epsilons=None, coord_bin_width=1.0, dtype='d', num_cores=None, verbose=False):
		"""
		Hybrid rshalib/oqhazlib deaggregation for multiple sites, multiple
		imt's per site, and multiple iml's per iml, using multiprocessing.
		Note that deaggregation by tectonic region type is replaced with
		deaggregation by source.

		Note: at least in Windows, this method has to be executed in
		a main section (i.e., behind if __name__ == "__main__":)

		:param site_imtls:
			nested dictionary mapping (lon, lat) tuples to dictionaries
			mapping oqhazlib IMT objects to 1-D arrays of intensity measure
			levels
		:param mag_bin_width:
			Float, magnitude bin width (default: None, will take MFD bin width
			of first source)
		:param dist_bin_width:
			Float, distance bin width in km (default: 10.)
		:param n_epsilons:
			Int, number of epsilon bins (default: None, will result in bins
			corresponding to integer epsilon values)
		:param coord_bin_width:
			Float, lon/lat bin width in decimal degrees (default: 1.)
		:param dtype:
			str, precision of deaggregation matrix (default: 'd')
		:param num_cores:
			int, number of CPUs to be used. Actual number of cores used
			may be lower depending on available cores
			(default: None, will determine automatically)
		:param verbose:
			Bool, whether or not to print some progress information

		:return:
			dict, mapping site (lon, lat) tuples to instances of
			:class:`SpectralDeaggregationCurve`
		"""
		import mp

		if not n_epsilons:
			n_epsilons = 2 * int(np.ceil(self.truncation_level))
		if not mag_bin_width:
			mag_bin_width = self.source_model[0].mfd.bin_width

		## Determine bin edges first
		bin_edges = self.get_deagg_bin_edges(mag_bin_width, dist_bin_width, n_epsilons, coord_bin_width)
		mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, src_bins = bin_edges

		## Create deaggregation matrices
		deagg_matrix_dict = {}
		for site_key in site_imtls.keys():
			deagg_matrix_dict[site_key] = {}
			imtls = site_imtls[site_key]
			imts = imtls.keys()
			num_imts = len(imts)
			num_imls = len(imtls[imts[0]])

			deagg_matrix_shape = (num_imts, num_imls, len(mag_bins) - 1, len(dist_bins) - 1, len(lon_bins) - 1,
						len(lat_bins) - 1, len(eps_bins) - 1, len(src_bins))

			## Initialize array with zeros representing exceedance probabilities !
			deagg_matrix = ProbabilityMatrix(np.zeros(deagg_matrix_shape, dtype=dtype))
			deagg_matrix_dict[site_key] = deagg_matrix

		site_model = self.get_soil_site_model()
		deagg_soil_sites = [site for site in site_model.get_sites() if (site.lon, site.lat) in site_imtls.keys()]
		deagg_site_model = SoilSiteModel("", deagg_soil_sites)

		## Convert imt's in site_imtls to tuples to avoid mangling up by mp
		copy_of_site_imtls = OrderedDict()
		for site_key in site_imtls.keys():
			copy_of_site_imtls[site_key] = OrderedDict()
			for imt in site_imtls[site_key]:
				copy_of_site_imtls[site_key][tuple(imt)] = site_imtls[site_key][imt]

		## Create list with arguments for each job
		job_args = []
		for source in self.source_model:
			job_args.append((self, source, copy_of_site_imtls, deagg_site_model, mag_bins, dist_bins, eps_bins, lon_bins, lat_bins, dtype, verbose))

		## Launch multiprocessing
		if not num_cores:
			num_cores = mp.multiprocessing.cpu_count()

		for src_idx, src_deagg_matrix_dict in enumerate(mp.run_parallel(mp.deaggregate_by_source, job_args, num_cores)):
			for site_key in deagg_matrix_dict.keys():
				deagg_matrix_dict[site_key][:,:,:,:,:,:,:,src_idx] = src_deagg_matrix_dict[site_key]

		## Create SpectralDeaggregationCurve for each site
		deagg_result = {}
		all_sites = site_model.get_sha_sites()
		for deagg_site in deagg_site_model:
			for site in all_sites:
				if deagg_site.location.longitude == site.lon and deagg_site.location.latitude == site.lat:
					break
			site_key = (site.lon, site.lat)
			imtls = site_imtls[site_key]
			imts = imtls.keys()
			periods = [getattr(imt, "period", 0) for imt in imts]
			intensities = np.array([imtls[imt] for imt in imts])
			deagg_matrix = deagg_matrix_dict[site_key]
			deagg_result[site_key] = SpectralDeaggregationCurve(bin_edges,
										deagg_matrix, site, "SA", intensities,
										periods, self.time_span)

		return deagg_result

	def _interpolate_oq_site_imtls(self, curve_name, sites, imt_periods, calc_id=None):
		"""
		Determine intensity levels corresponding to psha-model return periods
		from saved hazard curves. Mainly useful as helper function for
		deaggregation.

		:param curve_name:
			str, identifying hazard curve (e.g., "rlz-01", "mean", "quantile_0.84")
		:param sites:
			list with instances of :class:`SHASite` or instance of
			:class:`SHASiteModel`. Note that instances
			of class:`SoilSite` will not work with multiprocessing
		:param imt_periods:
			dictionary mapping intensity measure strings to lists of spectral
			periods.
		:param calc_id:
			str, calculation ID. (default: None, will determine from folder structure)

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

		for im in sorted(imt_periods.keys()):
			damping = 5.
			for T in sorted(imt_periods[im]):
				if im == "SA":
					imt = getattr(nhlib.imt, im)(T, damping)
				else:
					imt = getattr(nhlib.imt, im)()

				hcf = self.read_oq_hcf(curve_name, im, T, calc_id=calc_id)
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

		if not os.path.exists(self.oq_root_folder):
			os.mkdir(self.oq_root_folder)

		## write nrml file for source model
		self.source_model.write_xml(os.path.join(self.oq_root_folder, self.source_model.name + '.xml'))

		## write nrml file for soil site model if present and set file param, or set ref soil params
		self._handle_oq_soil_params(params)

		## validate source model logic tree and write nrml file
		source_model_lt = SeismicSourceSystem(self.source_model.name, self.source_model)
		source_model_lt.validate()
		source_model_lt_file_name = 'source_model_lt.xml'
		source_model_lt.write_xml(os.path.join(self.oq_root_folder, source_model_lt_file_name))
		params.source_model_logic_tree_file = source_model_lt_file_name

		## create ground_motion_model logic tree and write nrml file
		ground_motion_model_lt = self.ground_motion_model.get_optimized_model(self.source_model).to_ground_motion_system()
		ground_motion_model_lt_file_name = 'ground_motion_model_lt.xml'
		ground_motion_model_lt.write_xml(os.path.join(self.oq_root_folder, ground_motion_model_lt_file_name))
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
		params.write_config(os.path.join(self.oq_root_folder, 'job.ini'))

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

		if not os.path.exists(self.crisis_root_folder):
			os.mkdir(self.crisis_root_folder)

		## Construct default filenames and paths if none are specified
		if not filespec:
			filespec = os.path.join(self.crisis_root_folder, self.name + '.DAT')
		if not atn_folder:
			atn_folder = os.path.join(self.crisis_root_folder, 'gsims')
		if not os.path.exists(atn_folder):
			os.mkdir(atn_folder)
		if not site_filespec:
			site_filespec = os.path.join(self.crisis_root_folder, 'sites.ASC')

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

	def _get_trt_gsim_dict(self):
		"""
		:return:
			dict, mapping tectonic region types (str) to instances of
			:class:` GroundShakingIntensityModel`
		"""
		return {trt: nhlib.gsim.get_available_gsims()[self.ground_motion_model[trt]]() for trt in self._get_used_trts()}

	def _get_used_trts(self):
		"""
		:return:
			list of strings, defining tectonic region types used in source model.
		"""
		used_trts = set()
		for source in self.source_model:
			used_trts.add(source.tectonic_region_type)
		return list(used_trts)

	def _get_used_gsims(self):
		"""
		:return:
			list of strings, defining gsims of tectonic region types used in source model.
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
	def __init__(self, name, source_model_lt, gmpe_lt, root_folder, sites=[], grid_outline=[], grid_spacing=0.5, soil_site_model=None, ref_soil_params=REF_SOIL_PARAMS, imt_periods={'PGA': [0]}, intensities=None, min_intensities=0.001, max_intensities=1., num_intensities=100, return_periods=[], time_span=50., truncation_level=3., integration_distance=200., num_lt_samples=1, random_seed=42):
		"""
		"""
		from openquake.engine.input.logictree import LogicTreeProcessor
		PSHAModelBase.__init__(self, name, root_folder, sites, grid_outline, grid_spacing, soil_site_model, ref_soil_params, imt_periods, intensities, min_intensities, max_intensities, num_intensities, return_periods, time_span, truncation_level, integration_distance)
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

	def sample_logic_trees(self, num_samples=None, enumerate_gmpe_lt=False, verbose=False):
		"""
		Sample both source-model and GMPE logic trees, in a way that is
		similar to :meth:`_initialize_realizations_montecarlo` of
		:class:`BaseHazardCalculator` in oq-engine

		:param num_samples:
			int, number of random samples
			If zero, :meth:`enumerate_logic_trees` will be called
			(default: None, will use num_lt_samples)
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
		elif num_samples is None:
			num_samples = self.num_lt_samples

		psha_models = []

		if enumerate_gmpe_lt:
			gmpe_models, _ = self.enumerate_gmpe_lt(verbose=verbose)
			gmpelt_paths = self.gmpe_lt.root_branchset.enumerate_paths()

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
				gmpelt_paths = [gmpelt_path]

			for gmpe_model, gmpelt_path in zip(gmpe_models, gmpelt_paths):
				## Convert to PSHA model
				name = "%s, LT sample %04d (SM_LTP: %s; GMPE_LTP: %s)" % (self.name, i+1, " -- ".join(smlt_path), " -- ".join(gmpelt_path))
				psha_model = self._get_psha_model(source_model, gmpe_model, name)
				psha_models.append(psha_model)

			## Update the seed for the next realization
			seed = self.rnd.randint(MIN_SINT_32, MAX_SINT_32)
			self.rnd.seed(seed)

		# TODO: use yield instead?
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
		:param smlt_path:
			str, source-model logic-tree path
		:param gmpelt_path:
			str, GMPE logic-tree path
		"""
		root_folder = self.root_folder
		optimized_gmpe_model = gmpe_model.get_optimized_model(source_model)
		#if self.soil_site_model or self.grid_outline:
		#	sites = []
		#else:
		#	sites = self.get_sites()
		psha_model = PSHAModel(name, source_model, optimized_gmpe_model, root_folder,
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
		if not os.path.exists(self.oq_root_folder):
			os.mkdir(self.oq_root_folder)

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
			source_model.write_xml(os.path.join(self.oq_root_folder, source_model.name + '.xml'))

		## write nrml file for soil site model if present and set file param, or set ref soil params
		self._handle_oq_soil_params(params)

		## validate source model logic tree and write nrml file
		self.source_model_lt.validate()
		source_model_lt_file_name = 'source_model_lt.xml'
		self.source_model_lt.write_xml(os.path.join(self.oq_root_folder, source_model_lt_file_name))
		params.source_model_logic_tree_file = source_model_lt_file_name

		## create ground motion model logic tree and write nrml file
		ground_motion_model_lt_file_name = 'ground_motion_model_lt.xml'
		self.gmpe_lt.write_xml(os.path.join(self.oq_root_folder, ground_motion_model_lt_file_name))
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
		params.write_config(os.path.join(self.oq_root_folder, 'job.ini'))

	def run_nhlib(self, nrml_base_filespec=""):
		"""
		Run PSHA model with nhlib and store result in a SpectralHazardCurveFieldTree
		object.

		:param nrml_base_filespec:
			String, base file specification for NRML output file
			(default: "").
		"""
		# TODO: this method is probably obsolete
		if not nrml_base_filespec:
			os.path.join(self.output_dir, '%s' % self.name)
		else:
			nrml_base_filespec = os.path.splitext(nrml_base_filespec)[0]

		num_sites = len(self.get_soil_site_model())
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
			hazard_result = psha_model.run_nhlib_poes()
			for imt in self.imt_periods.keys():
				hazard_results[imt][:,j,:,:] = hazard_result[imt]
		im_imls = self._get_im_imls()
		site_names = [site.name for site in self.get_sites()]
		for imt, periods in self.imt_periods.items():
			shcft = SpectralHazardCurveFieldTree(self.name, psha_model_names, filespecs, weights, self.get_sha_sites(), periods, imt, im_imls[imt], 'g', self.time_span, poes=hazard_results[imt], site_names=site_names)
			nrml_filespec = nrml_base_filespec + '_%s.xml' % imt
			shcft.write_nrml(nrml_filespec)
		return shcft

	def calc_shcf_mp(self, cav_min=0, combine_pga_and_sa=True, num_cores=None, verbose=True):
		"""
		Compute spectral hazard curve fields using multiprocessing.
		The results are written to XML files.

		Note: at least in Windows, this method has to be executed in
		a main section (i.e., behind if __name__ == "__main__":)

		:param cav_min:
			float, CAV threshold in g.s (default: 0)
		:param combine_pga_and_sa:
			bool, whether or not to combine PGA and SA, if present
			(default: True)
		:param num_cores:
			int, number of CPUs to be used. Actual number of cores used
			may be lower depending on available cores and memory
			(default: None, will determine automatically)
		:param verbose:
			bool whether or not to print some progress information
			(default: True)

		:return:
			list of exit codes for each sample (0 for succesful execution,
			1 for error)
		"""
		import mp

		## Generate all PSHA models
		psha_models = self.sample_logic_trees(self.num_lt_samples, enumerate_gmpe_lt=False, verbose=False)

		## Determine number of simultaneous processes
		if not num_cores:
			num_cores = mp.multiprocessing.cpu_count()
		else:
			num_cores = min(mp.multiprocessing.cpu_count(), num_cores)

		## Create list with arguments for each job
		fmt = "%%0%dd" % len(str(self.num_lt_samples))
		job_args = [(psha_model, fmt % sample_idx, cav_min, combine_pga_and_sa, verbose) for (psha_model, sample_idx) in zip(psha_models, range(self.num_lt_samples))]

		## Launch multiprocessing
		return mp.run_parallel(mp.calc_shcf_psha_model, job_args, num_cores, verbose)

	def deaggregate_mp(self, sites, imt_periods, mag_bin_width=None, dist_bin_width=10., n_epsilons=None, coord_bin_width=1.0, num_cores=None, dtype='d', verbose=False):
		"""
		Deaggregate logic tree using multiprocessing.
		Intensity measure levels corresponding to psha_model.return_periods
		will be interpolated first, so the hazard curves must have been
		computed before.

		Note: at least in Windows, this method has to be executed in
		a main section (i.e., behind if __name__ == "__main__":)

		:param sites:
			list with instances of :class:`SHASite` for which deaggregation
			will be performed. Note that instances of class:`SoilSite` will
			not work with multiprocessing
		:param imt_periods:
			dictionary mapping intensity measure strings to lists of spectral
			periods.
		:param mag_bin_width:
			Float, magnitude bin width (default: None, will take MFD bin width
			of first source)
		:param dist_bin_width:
			Float, distance bin width in km (default: 10.)
		:param n_epsilons:
			Int, number of epsilon bins (default: None, will result in bins
			corresponding to integer epsilon values)
		:param coord_bin_width:
			Float, lon/lat bin width in decimal degrees (default: 1.)
		:param num_cores:
			Int, number of CPUs to be used. Actual number of cores used
			may be lower depending on available cores and memory
			(default: None, will determine automatically)
		:param dtype:
			str, precision of deaggregation matrix (default: 'd')
		:param verbose:
			Bool, whether or not to print some progress information

		:return:
			list of exit codes for each sample (0 for succesful execution,
			1 for error)
		"""
		import platform
		import psutil
		import mp

		## Generate all PSHA models
		psha_models = self.sample_logic_trees(self.num_lt_samples, enumerate_gmpe_lt=False, verbose=False)

		## Convert sites to SHASite objects if necessary, because SoilSites
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
		# TODO: check imts as well

		## Determine number of simultaneous processes based on estimated memory consumption
		bin_edges = psha_models[0].get_deagg_bin_edges(mag_bin_width, dist_bin_width, n_epsilons, coord_bin_width)
		mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, src_bins = bin_edges

		num_imls = len(self.return_periods)
		num_imts = np.sum([len(imt_periods[im]) for im in imt_periods.keys()])
		matrix_size = (len(sites) * num_imts * num_imls * (len(mag_bins) - 1)
						* (len(dist_bins) - 1) * (len(lon_bins) - 1) * (len(lat_bins) - 1)
						* len(eps_bins) * len(src_bins) * 4)

		if not num_cores:
			num_cores = mp.multiprocessing.cpu_count()
		else:
			num_cores = min(mp.multiprocessing.cpu_count(), num_cores)
		free_mem = psutil.phymem_usage()[2]
		if platform.uname()[0] == "Windows":
			## 32-bit limit
			# Note: is this limit valid for all subprocesses combined?
			free_mem = min(free_mem, 2E+9)
		#print free_mem, matrix_size
		num_processes = min(num_cores, np.floor(free_mem / matrix_size))

		## Create list with arguments for each job
		fmt = "%%0%dd" % len(str(self.num_lt_samples))
		job_args = [(psha_model, fmt % (sample_idx + 1), deagg_sites, imt_periods, mag_bin_width, dist_bin_width, n_epsilons, coord_bin_width, dtype, verbose) for (psha_model, sample_idx) in zip(psha_models, range(self.num_lt_samples))]

		## Launch multiprocessing
		return mp.run_parallel(mp.deaggregate_psha_model, job_args, num_processes, verbose)

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
		if not os.path.exists(self.crisis_root_folder):
			os.mkdir(self.crisis_root_folder)
		site_filespec = os.path.join(self.crisis_root_folder, 'sites.ASC')
		gsims_dir = os.path.join(self.crisis_root_folder, 'gsims')
		if not os.path.exists(gsims_dir):
				os.mkdir(gsims_dir)

		## Create directory structure for logic tree:
		## only possible for source models
		sm_filespecs = {}
		all_filespecs = []
		for source_model in self.source_models:
			sm_filespecs[source_model.name] = []
			folder = os.path.join(self.crisis_root_folder, source_model.name)
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
			folder = os.path.join(self.crisis_root_folder, psha_model.source_model.name)
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
			folder = os.path.join(self.crisis_root_folder, sm_name)
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

		batch_filespec = os.path.join(self.crisis_root_folder, batch_filename)
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

	def read_oq_shcft(self, calc_id=None, add_stats=False):
		"""
		Read OpenQuake spectral hazard curve field tree.
		Read from the folder 'hazard_curve_multi' if present, else read individual
		hazard curves from the folder 'hazard_curve'.

		:param calc_id:
			list of ints, calculation IDs.
			(default: None, will determine from folder structure)
		:param add_stats:
			bool indicating whether or not mean and quantiles have to be appended

		:return:
			instance of :class:`SpectralHazardCurveFieldTree`
		"""
		from ..openquake import read_shcft

		hc_folder = self.get_oq_hc_folder(calc_id=calc_id)
		## Go one level up, read_shcft will choose between hazard_curve and hazard_curve_multi
		hc_folder = os.path.split(hc_folder)[0]
		shcft = read_shcft(hc_folder, self.get_sha_sites(), add_stats=add_stats)
		return shcft

	def read_oq_uhsft(self, return_period, calc_id=None, add_stats=False):
		"""
		Read OpenQuake UHS field tree

		:param return period:
			float, return period
		:param add_stats:
			bool indicating whether or not mean and quantiles have to be appended

		:return:
			instance of :class:`UHSFieldTree`
		"""
		from ..openquake import read_uhsft

		uhs_folder = self.get_oq_uhs_folder(calc_id=calc_id)
		uhsft = read_uhsft(uhs_folder, return_period, self.get_sha_sites(), add_stats=add_stats)
		return uhsft

	def read_crisis_shcft(self):
		"""
		Read CRISIS spectral hazard curve field tree

		:return:
			instance of :class:`SpectralHazardCurveFieldTree`
		"""
		from ..crisis import read_GRA_multi

		gra_filespecs, weights = self.read_crisis_batch(test_case)
		shcft = read_GRA_multi(gra_filespecs, weights=weights)
		return shcft

	# TODO: the following methods are probably obsolete

	def _get_psha_models(self):
		"""
		Return list of :class:`PSHAModel` objects, defining sampled PSHA models from logic tree.
		"""
		psha_models = []
		for i in range(self.num_lts_samples):
			source_model, ground_motion_model = self._sample_lts()
			name = source_model.name + '_' + ground_motion_model.name
			psha_models.append(PSHAModel(name, source_model, ground_motion_model, self.root_folder, self.get_sites(), self.grid_outline, self.grid_spacing, self.soil_site_model, self.ref_soil_params, self.imt_periods, self.intensities, self.min_intensities, self.max_intensities, self.num_intensities, self.return_periods, self.time_span, self.truncation_level, self.integration_distance))
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

