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

# im (or imt_name?): intensity measure, e.g. "PGA", "SA"
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

	def create_folder_structure(self, path):
		"""
		Create folder structure if it does not exist yet.
		Note: Only folders below the root folder will be created.

		:param path:
			str, absolute path
		"""
		path = path.split(self.root_folder)[1]
		subfolders = path.split(os.path.sep)
		partial_path = self.root_folder
		for subfolder in subfolders:
			partial_path = os.path.join(partial_path, subfolder)
			if not os.path.exists(partial_path):
				os.mkdir(partial_path)

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
		hc_folder = os.path.join(folder, "calc_%s" % calc_id, "hazard_curve")
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
		uhs_folder = os.path.join(folder, "calc_%s" % calc_id, "uh_spectra")
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
		hm_folder = os.path.join(folder, "calc_%s" % calc_id, "hazard_map")
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
		disagg_folder = os.path.join(folder, "calc_%s" % calc_id, "disagg_matrix")
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
		if os.path.exists(folder):
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

	def _get_imts(self):
		"""
		Construct ordered list of IMT objects
		"""
		imts = []
		for im, periods in sorted(self.imt_periods.items()):
			for period in periods:
				imts.append(self._construct_imt(im, period))
		return imts

	def _construct_imt(self, im, period):
		"""
		Construct IMT object from intensity measure and period.

		:param im:
			str, intensity measure, e.g. "PGA", "SA"
		:param period:
			float, spectral period

		:return:
			instance of :class:`IMT`
		"""
		if im == "SA":
			imt = getattr(nhlib.imt, im)(period, damping=5.)
		else:
			imt = getattr(nhlib.imt, im)()
		return imt

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
			if imt == "SA":
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

	def _get_oq_imt_subfolder(self, im, T):
		"""
		Determine OpenQuake subfolder name for a particular IMT

		:param im:
			str, intensity measure, e.g. "SA", "PGA"
		:param T:
			float, spectral period in seconds

		:return:
			str, IMT subfolder
		"""
		if im == "SA":
			imt_subfolder = "SA-%s" % T
		else:
			imt_subfolder = im
		return imt_subfolder

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

		hc_folder = self.get_oq_hc_folder(calc_id=calc_id, multi=False)
		imt_subfolder = self._get_oq_imt_subfolder(im, T)
		xml_filename = "hazard_curve-%s.xml" % curve_name
		#print xml_filename
		xml_filespec = os.path.join(hc_folder, imt_subfolder, xml_filename)
		hcf = parse_hazard_curves(xml_filespec)
		hcf.set_site_names(self.get_sha_sites())

		return hcf

	def write_oq_hcf(self, hcf, curve_name, calc_id="oqhazlib"):
		"""
		Write OpenQuake hazard curve field. Folder structure will be
		created, if necessary.

		:param hcf:
			instance of :class:`HazardCurveField`
		:param curve_name:
			str, identifying hazard curve (e.g., "rlz-01", "mean", "quantile_0.84")
		:param calc_id:
			str, calculation ID. (default: "oqhazlib")
		"""
		hc_folder = self.get_oq_hc_folder(calc_id=calc_id, multi=False)
		imt_subfolder = self._get_oq_imt_subfolder(hcf.imt, hcf.T)
		imt_hc_folder = os.path.join(hc_folder, imt_subfolder)
		self.create_folder_structure(imt_hc_folder)
		xml_filename = "hazard_curve-%s.xml" % curve_name
		xml_filespec = os.path.join(imt_hc_folder, xml_filename)
		hcf.write_nrml(xml_filespec)

	def read_oq_shcf(self, curve_name, calc_id=None):
		"""
		Read OpenQuake spectral hazard curve field

		:param curve_name:
			str, identifying hazard curve (e.g., "rlz-01", "mean", "quantile_0.84")
		:param calc_id:
			str, calculation ID. (default: None, will determine from folder structure)

		:return:
			instance of :class:`SpectralHazardCurveField`
		"""
		from ..openquake import parse_hazard_curves, parse_spectral_hazard_curve_field

		hc_folder = self.get_oq_hc_folder(calc_id=calc_id, multi=True)
		xml_filename = "hazard_curve_multi-%s.xml" % curve_name
		#print xml_filename
		xml_filespec = os.path.join(hc_folder, xml_filename)
		try:
			shcf = parse_hazard_curves(xml_filespec)
		except:
			shcf = parse_spectral_hazard_curve_field(xml_filespec)
		shcf.set_site_names(self.get_sha_sites())

		return shcf

	def write_oq_shcf(self, shcf, curve_name, calc_id="oqhazlib"):
		"""
		Write OpenQuake spectral hazard curve field. Folder structure
		will be created, if necessary.

		:param shcf:
			instance of :class:`SpectralHazardCurveField`
		:param curve_name:
			str, identifying hazard curve (e.g., "rlz-01", "mean", "quantile_0.84")
		:param calc_id:
			str, calculation ID. (default: "oqhazlib")
		"""
		hc_folder = self.get_oq_hc_folder(calc_id=calc_id, multi=True)
		self.create_folder_structure(hc_folder)
		xml_filename = "hazard_curve_multi-%s.xml" % curve_name
		xml_filespec = os.path.join(hc_folder, xml_filename)
		shcf.write_nrml(xml_filespec)

	def read_oq_uhs_multi(self):
		# TODO
		pass

	def read_oq_uhs_field(self, curve_name, return_period, calc_id=None):
		"""
		Read OpenQuake hazard curve field.

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

		poe = str(round(Poisson(life_time=self.time_span, return_period=return_period), 13))

		uhs_folder = self.get_oq_uhs_folder(calc_id=calc_id)
		xml_filename = "uh_spectra-poe_%s-%s.xml" % (poe, curve_name)
		#print xml_filename
		xml_filespec = os.path.join(uhs_folder, xml_filename)
		uhsf = parse_uh_spectra(xml_filespec)
		uhsf.set_site_names(self.get_sha_sites())

		return uhsf

	def write_oq_uhs_field(self, uhsf):
		# TODO
		pass

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
		poe = str(round(Poisson(life_time=self.time_span, return_period=return_period), 13))

		disagg_folder = self.get_oq_disagg_folder(calc_id=calc_id, multi=False)
		imt_subfolder = self._get_oq_imt_subfolder(im, T)
		xml_filename = "disagg_matrix(%s)-lon_%s-lat_%s-%s.xml"
		xml_filename %= (poe, site.lon, site.lat, curve_name)
		xml_filespec = os.path.join(disagg_folder, imt_subfolder, xml_filename)
		ds = parse_disaggregation(xml_filespec, site.name)
		return ds

	def write_oq_disagg_matrix(self, ds, curve_name, calc_id="oqhazlib"):
		"""
		Write OpenQuake deaggregation matrix. Folder structure will be
		created, if necessary.

		:param ds:
			instance of :class:`DeaggregationSlice`
		:param curve_name:
			str, identifying hazard curve (e.g., "rlz-01", "mean", "quantile_0.84")
		:param calc_id:
			str, calculation ID. (default: "oqhazlib")
		"""
		poe = str(round(Poisson(life_time=ds.time_span, return_period=ds.return_period), 13))

		disagg_folder = self.get_oq_disagg_folder(calc_id=calc_id, multi=False)
		imt_subfolder = self._get_oq_imt_subfolder(ds.imt, ds.period)
		imt_disagg_folder = os.path.join(disagg_folder, imt_subfolder)
		self.create_folder_structure(imt_disagg_folder)
		xml_filename = "disagg_matrix(%s)-lon_%s-lat_%s-%s.xml"
		xml_filename %= (poe, ds.site.lon, ds.site.lat, curve_name)
		xml_filespec = os.path.join(imt_disagg_folder, xml_filename)
		ds.write_nrml(xml_filespec, self.smlt_path, self.gmpelt_path)

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

	def write_oq_disagg_matrix_multi(self, sdc, curve_name, calc_id="oqhazlib"):
		"""
		Write OpenQuake multi-deaggregation matrix. Folder structure
		will be created, if necessary.

		:param sdc:
			instance of :class:`SpectralDeaggregationCurve`
		:param curve_name:
			str, identifying hazard curve (e.g., "rlz-01", "mean", "quantile_0.84")
		:param calc_id:
			str, calculation ID. (default: "oqhazlib")
		"""
		disagg_folder = self.get_oq_disagg_folder(calc_id=calc_id, multi=True)
		self.create_folder_structure(disagg_folder)
		xml_filename = "disagg_matrix_multi-lon_%s-lat_%s-%s.xml"
		xml_filename %= (sdc.site.lon, sdc.site.lat, curve_name)
		xml_filespec = os.path.join(disagg_folder, xml_filename)
		sdc.write_nrml(xml_filespec, self.smlt_path, self.gmpelt_path)

	def read_crisis_batch(self, batch_filename = "lt_batch.dat"):
		"""
		Reach CRISIS batch file

		:param batch_filename:
			str, name of batch file (default: "lt_batch.dat")

		:return:
			list of gra_filespecs
		"""
		from ..crisis import read_batch

		batch_filespec = os.path.join(self.crisis_root_folder, batch_filename)
		#print batch_filespec
		return read_batch(batch_filespec)

	def read_crisis_shcf(self, curve_name, batch_filename="lt_batch.dat"):
		"""
		Read CRISIS spectral hazard curve field

		:param curve_name:
			str, identifying hazard curve (e.g., "rlz-01")
		:param batch_filename:
			str, name of batch file (default: "lt_batch.dat")

		:return:
			instance of :class:`SpectralHazardCurveField`
		"""
		from ..crisis import read_GRA

		gra_filespecs, weights = self.read_crisis_batch(batch_filename)
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

	def calc_shcf_mp(self, cav_min=0, decompose_area_sources=False, individual_sources=False, num_cores=None, combine_pga_and_sa=True, verbose=True):
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
		:param combine_pga_and_sa:
			bool, whether or not to combine PGA and SA, if present
			(default: True)
		:param verbose:
			bool, whether or not to print some progress information
			(default: True)

		:return:
			dictionary mapping intensity measure (str) to:
			- instance of :class:`SpectralHazardCurveField` (if group_sources
			is True) or
			- dict mapping source IDs to instances of
			:class:`SpectralHazardCurveField` (if group_sources is False)
		"""
		import mp

		if not num_cores:
			num_cores = mp.multiprocessing.cpu_count()

		if decompose_area_sources:
			source_model = self.source_model.decompose_area_sources()
			num_decomposed_sources = self.source_model.get_num_decomposed_sources()
			cum_num_decomposed_sources = np.concatenate([[0], np.add.accumulate(num_decomposed_sources)])
		else:
			source_model = self.source_model

		## Create list with arguments for each job
		job_args = []
		for source in source_model:
			job_args.append((self, source, cav_min, verbose))

		## Launch multiprocessing
		curve_list = mp.run_parallel(mp.calc_shcf_by_source, job_args, num_cores, verbose=verbose)
		poes = ProbabilityArray(curve_list)

		## Recombine hazard curves computed for each source
		if not individual_sources:
			poes = np.prod(poes, axis=0)
		else:
			total_poes = np.prod(poes, axis=0)
			if decompose_area_sources:
				curve_list = []
				for src_idx, src in enumerate(self.source_model):
					start = cum_num_decomposed_sources[src_idx]
					stop = cum_num_decomposed_sources[src_idx+1]
					curve_list.append(np.prod(poes[start:stop], axis=0))
				poes = ProbabilityArray(curve_list)

		## Convert non-exceedance to exceedance probabilities
		poes -= 1
		poes *= -1
		if individual_sources:
			total_poes -= 1
			total_poes *= -1

		## Construct spectral hazard curve field
		shcf_dict = {}
		sites = self.get_sha_sites()
		imts = self._get_imts()
		im_imls = self._get_im_imls(combine_pga_and_sa=combine_pga_and_sa)
		for im, intensities in im_imls.items():
			periods = self.imt_periods[im]
			## Determine period indexes in poes array
			period_idxs = []
			for T in periods:
				imt = self._construct_imt(im, T)
				period_idxs.append(imts.index(imt))
			if im == "SA" and combine_pga_and_sa and "PGA" in self.imt_periods.keys():
				periods = np.concatenate([[0], periods])
				imt = self._construct_imt("PGA", 0)
				period_idxs.insert(0, imts.index(imt))

			if individual_sources:
				src_shcf_dict = OrderedDict()
				for i, src in enumerate(self.source_model):
					src_shcf_dict[src.source_id] = SpectralHazardCurveField(self.name,
													poes[i][:,period_idxs,:], [""]*len(periods), sites,
													periods, im, intensities, 'g',
													self.time_span)
				src_shcf_dict['Total'] = SpectralHazardCurveField(self.name, total_poes[:,period_idxs,:],
											[""]*len(periods), sites, periods, im,
											intensities, 'g', self.time_span)
				shcf_dict[im] = src_shcf_dict
			else:
				shcf = SpectralHazardCurveField(self.name, poes[:,period_idxs,:], [""]*len(periods),
								sites, periods, im, intensities, 'g', self.time_span)
				shcf_dict[im] = shcf

		return shcf_dict

	def deagg_nhlib(self, site, imt, iml, return_period, mag_bin_width=None, dist_bin_width=10., n_epsilons=None, coord_bin_width=1.0):
		"""
		Run deaggregation with nhlib

		:param site:
			instance of :class:`SHASite` or :class:`SoilSite`
		:param imt:
			Instance of :class:`nhlib.imt._IMT`, intensity measure type
		:param iml:
			Float, intensity measure level
		:param return_period:
			Float, return period corresponding to iml
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
		ssdf = self.source_site_distance_filter
		rsdf = self.rupture_site_distance_filter

		bin_edges, deagg_matrix = nhlib.calc.disaggregation_poissonian(self.source_model, site, imt, iml, self._get_trt_gsim_dict(), self.time_span, self.truncation_level, n_epsilons, mag_bin_width, dist_bin_width, coord_bin_width, ssdf, rsdf)
		deagg_matrix = ProbabilityMatrix(deagg_matrix)
		imt_name = str(imt).split('(')[0]
		if imt_name == "SA":
			period = imt.period
		else:
			period = 0
		return DeaggregationSlice(bin_edges, deagg_matrix, site, imt_name, iml, period, return_period, self.time_span)

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

		ssdf = self.source_site_distance_filter
		rsdf = self.rupture_site_distance_filter

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
				yield SpectralDeaggregationCurve(bin_edges, deagg_matrix, site, "SA", intensities, periods, self.return_periods, self.time_span)

	def get_deagg_bin_edges(self, mag_bin_width, dist_bin_width, coord_bin_width, n_epsilons):
		"""
		Determine bin edges for deaggregation.
		Note: no default values!

		:param mag_bin_width:
			Float, magnitude bin width
		:param dist_bin_width:
			Float, distance bin width in km
		:param coord_bin_width:
			Float, lon/lat bin width in decimal degrees
		:param n_epsilons:
			Int, number of epsilon bins
			corresponding to integer epsilon values)

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
		print west, east, south, north
		west -= coord_bin_width
		east += coord_bin_width

		lon_bins = coord_bin_width * np.arange(
			int(np.floor(west / coord_bin_width)),
			int(np.ceil(east / coord_bin_width) + 1)
		)

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
		from openquake.hazardlib.site import SiteCollection

		# TODO: determine site_imtls from self.return_periods (separate method)
		if site_imtls in (None, {}):
			pass

		if not n_epsilons:
			n_epsilons = 2 * int(np.ceil(self.truncation_level))
		if not mag_bin_width:
			mag_bin_width = self.source_model[0].mfd.bin_width

		## Determine bin edges first
		bin_edges = self.get_deagg_bin_edges(mag_bin_width, dist_bin_width, coord_bin_width, n_epsilons)
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
										periods, self.return_periods, self.time_span)

		return deagg_result

	def deaggregate_mp(self, site_imtls, decompose_area_sources=False, mag_bin_width=None, dist_bin_width=10., n_epsilons=None, coord_bin_width=1.0, dtype='d', num_cores=None, verbose=False):
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
		:param decompose_area_sources:
			bool, whether or not area sources should be decomposed into
			point sources for the computation (default: False)
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
		bin_edges = self.get_deagg_bin_edges(mag_bin_width, dist_bin_width, coord_bin_width, n_epsilons)
		mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, src_bins = bin_edges

		## Create deaggregation matrices
		num_sites = len(site_imtls.keys())
		imtls = site_imtls[site_imtls.keys()[0]]
		num_imts = len(imtls.keys())
		imls = imtls[imtls.keys()[0]]
		num_imls = len(imls)

		deagg_matrix_shape = (num_sites, num_imts, num_imls, len(mag_bins) - 1,
					len(dist_bins) - 1, len(lon_bins) - 1, len(lat_bins) - 1,
					len(eps_bins) - 1, len(src_bins))
		deagg_matrix_len = np.prod(deagg_matrix_shape)

		## Create shared-memory array, and expose it as a numpy array
		shared_deagg_array = mp.multiprocessing.Array(dtype, deagg_matrix_len, lock=True)

		## Initialize array with ones representing non-exceedance probabilities !
		deagg_matrix = np.frombuffer(shared_deagg_array.get_obj())
		deagg_matrix = deagg_matrix.reshape(deagg_matrix_shape)
		deagg_matrix += 1


		## Create soil site model for deaggregation sites, making sure
		## order is same as sorted keys of site_imtls
		site_model = self.get_soil_site_model()
		deagg_soil_sites = []
		for (site_lon, site_lat) in sorted(site_imtls.keys()):
			for site in site_model.get_sites():
				if (site.lon, site.lat) == (site_lon, site_lat):
					deagg_soil_sites.append(site)
					break
		deagg_site_model = SoilSiteModel("", deagg_soil_sites)

		## Convert imt's in site_imtls to tuples to avoid mangling up by mp
		copy_of_site_imtls = OrderedDict()
		for site_key in site_imtls.keys():
			copy_of_site_imtls[site_key] = OrderedDict()
			for imt in site_imtls[site_key]:
				copy_of_site_imtls[site_key][tuple(imt)] = site_imtls[site_key][imt]

		if decompose_area_sources:
			source_model = self.source_model.decompose_area_sources()
			num_decomposed_sources = self.source_model.get_num_decomposed_sources()
			cum_num_decomposed_sources = np.add.accumulate(num_decomposed_sources)
		else:
			source_model = self.source_model

		## Create list with arguments for each job
		job_args = []
		for idx, source in enumerate(source_model.sources):
			if decompose_area_sources:
				src_idx = np.where(cum_num_decomposed_sources > idx)[0][0]
			else:
				src_idx = idx
			job_args.append((self, source, src_idx, deagg_matrix_shape, copy_of_site_imtls, deagg_site_model, mag_bins, dist_bins, eps_bins, lon_bins, lat_bins, dtype, verbose))

		## Launch multiprocessing
		if not num_cores:
			num_cores = mp.multiprocessing.cpu_count()

		mp.run_parallel(mp.deaggregate_by_source, job_args, num_cores, shared_arr=shared_deagg_array, verbose=verbose)

		## Convert to exceedance probabilities
		deagg_matrix -= 1
		deagg_matrix *= -1

		## Create SpectralDeaggregationCurve for each site
		deagg_result = {}
		all_sites = site_model.get_sha_sites()
		for site_idx, site_key in enumerate(sorted(site_imtls.keys())):
			site_lon, site_lat = site_key
			for site in all_sites:
				if site.lon == site_lon and site.lat == site_lat:
					break
			imtls = site_imtls[site_key]
			imts = imtls.keys()
			periods = [getattr(imt, "period", 0) for imt in imts]
			intensities = np.array([imtls[imt] for imt in imts])
			site_deagg_matrix = ProbabilityMatrix(deagg_matrix[site_idx])
			deagg_result[site_key] = SpectralDeaggregationCurve(bin_edges,
										site_deagg_matrix, site, "SA", intensities,
										periods, self.return_periods, self.time_span)

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
			for T in sorted(imt_periods[im]):
				imt = self._construct_imt(im, T)

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
			list with (instance of :class:`PSHAModel`, weight) tuples
		"""
		#from itertools import izip
		if num_samples is None:
			num_samples = self.num_lt_samples

		if num_samples == 0:
			return self.enumerate_logic_trees(verbose=verbose)

		psha_models_weights = []

		if enumerate_gmpe_lt:
			gmpe_models_weights = self.enumerate_gmpe_lt(verbose=verbose)
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
				gmpe_models_weights = [(self._gmpe_sample_to_gmpe_model(gmpelt_path), 1.)]
				gmpelt_paths = [gmpelt_path]

			for (gmpe_model, gmpelt_weight), gmpelt_path in zip(gmpe_models_weights, gmpelt_paths):
				## Convert to PSHA model
				name = "%s, LT sample %04d (SM_LTP: %s; GMPE_LTP: %s)" % (self.name, i+1, " -- ".join(smlt_path), " -- ".join(gmpelt_path))
				psha_model = self._get_psha_model(source_model, gmpe_model, name)
				psha_models_weights.append((psha_model, gmpelt_weight/num_samples))
				#yield (psha_model, gmpelt_weight)

			## Update the seed for the next realization
			seed = self.rnd.randint(MIN_SINT_32, MAX_SINT_32)
			self.rnd.seed(seed)

		# TODO: use yield instead?
		return psha_models_weights

	def enumerate_logic_trees(self, verbose=False):
		"""
		Enumerate both source-model and GMPE logic trees, in a way that is
		similar to :meth:`_initialize_realizations_enumeration` of
		:class:`BaseHazardCalculator` in oq-engine

		:param verbose:
			bool, whether or not to print some information (default: False)

		:return:
			list with (instance of :class:`PSHAModel`, weight) tuples
		"""
		psha_models_weights = []
		for i, path_info in enumerate(self.ltp.enumerate_paths()):
			sm_name, weight, smlt_path, gmpelt_path = path_info
			source_model = self._smlt_sample_to_source_model(sm_name, smlt_path, verbose=verbose)
			gmpe_model = self._gmpe_sample_to_gmpe_model(gmpelt_path)
			name = "%s, LT enum %04d (SM_LTP: %s; GMPE_LTP: %s)" % (self.name, i, source_model.description, gmpe_model.name)
			psha_model = self._get_psha_model(source_model, gmpe_model, name)
			psha_models_weights.append((psha_model, weight))
			#yield (psha_model, weight)
		return psha_models_weights

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

		:return:
			instance of :class:`PSHAModel`
		"""
		root_folder = self.root_folder
		optimized_gmpe_model = gmpe_model.get_optimized_model(source_model)
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
			list with (instance of :class:`SourceModel`, weight) tuples
		"""
		if num_samples == 0:
			return self.enumerate_source_model_lt(verbose=verbose, show_plot=show_plot)

		modified_source_models_weights = []
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
			weight = 1./num_samples
			modified_source_models_weights.append((source_model, weight))
			#yield (source_model, weight)
		return modified_source_models_weights

	def enumerate_source_model_lt(self, verbose=False, show_plot=False):
		"""
		Enumerate source-model logic tree

		:param verbose:
			bool, whether or not to print some information (default: False)
		:param show_plot:
			bool, whether or not to plot a diagram of the sampled branch path
			(default: False)

		:return:
			list with (instance of :class:`SourceModel`, weight) tuples
		"""
		modified_source_models_weights = []
		for smlt_path_weight, smlt_branches in self.source_model_lt.root_branchset.enumerate_paths():
			smlt_path = [branch.branch_id for branch in smlt_branches]
			sm_name = os.path.splitext(smlt_branches[0].value)[0]
			if verbose:
				print smlt_path_weight, sm_name, smlt_path
			if show_plot:
				self.source_model_lt.plot_diagram(highlight_path=smlt_path)
			## Apply uncertainties
			source_model = self._smlt_sample_to_source_model(sm_name, smlt_path, verbose=verbose)
			modified_source_models_weights.append((source_model, smlt_path_weight))
			#yield (source_model, smlt_path_weight)
		return modified_source_models_weights

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
			list with (instance of :class:`GroundMotionModel`, weight) tuples
		"""
		if num_samples == 0:
			return self.enumerate_gmpe_lt(verbose=verbose, show_plot=show_plot)

		gmpe_models_weights = []
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
			weight = 1./num_samples
			gmpe_models_weights.append((gmpe_model, weight))
			if verbose:
				print gmpe_model
			#yield (gmpe_model, weight)
		return gmpe_models_weights

	def enumerate_gmpe_lt(self, verbose=False, show_plot=False):
		"""
		Enumerate GMPE logic tree

		:param verbose:
			bool, whether or not to print some information (default: False)
		:param show_plot:
			bool, whether or not to plot a diagram of the sampled branch path
			(default: False)

		:return:
			list with (instance of :class:`GroundMotionModel`, weight) tuples
		"""
		gmpe_models_weights = []
		for gmpelt_path_weight, gmpelt_branches in self.gmpe_lt.root_branchset.enumerate_paths():
			gmpelt_path = [branch.branch_id for branch in gmpelt_branches]
			if verbose:
				print gmpelt_path_weight, gmpelt_path
			if show_plot:
				self.gmpe_lt.plot_diagram(highlight_path=gmpelt_path)
			gmpe_model = self._gmpe_sample_to_gmpe_model(gmpelt_path)
			gmpe_models_weights.append((gmpe_model, gmpelt_path_weight))
			#yield (gmpe_model, gmpelt_path_weight)
		return gmpe_models_weights

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
		psha_models_weights = self.sample_logic_trees(self.num_lt_samples, enumerate_gmpe_lt=False, verbose=False)

		## Determine number of simultaneous processes
		if not num_cores:
			num_cores = mp.multiprocessing.cpu_count()
		else:
			num_cores = min(mp.multiprocessing.cpu_count(), num_cores)

		## Create list with arguments for each job
		job_args = []
		num_lt_samples = self.num_lt_samples or self.get_num_paths()
		fmt = "%%0%dd" % len(str(num_lt_samples))
		for sample_idx, (psha_model, weight) in enumerate(psha_models_weights):
			job_args.append((psha_model, fmt % (sample_idx + 1), cav_min, combine_pga_and_sa, verbose))

		## Launch multiprocessing
		return mp.run_parallel(mp.calc_shcf_psha_model, job_args, num_cores, verbose=verbose)

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
		psha_models_weights = self.sample_logic_trees(self.num_lt_samples, enumerate_gmpe_lt=False, verbose=False)

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
		psha_model0 = psha_models_weights[0][0]
		bin_edges = psha_model0.get_deagg_bin_edges(mag_bin_width, dist_bin_width, coord_bin_width, n_epsilons)
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
		job_args = []
		num_lt_samples = self.num_lt_samples or self.get_num_paths()
		fmt = "%%0%dd" % len(str(num_lt_samples))
		for sample_idx, (psha_model, weight) in enumerate(psha_models_weights):
			job_args.append((psha_model, fmt % (sample_idx + 1), deagg_sites, imt_periods, mag_bin_width, dist_bin_width, n_epsilons, coord_bin_width, dtype, verbose))

		## Launch multiprocessing
		return mp.run_parallel(mp.deaggregate_psha_model, job_args, num_processes, verbose=verbose)

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
		for i, (psha_model, weight) in enumerate(self.sample_logic_trees(self.num_lt_samples, enumerate_gmpe_lt=enumerate_gmpe_lt, verbose=verbose)):
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

	def read_oq_shcft(self, add_stats=False, calc_id=None):
		"""
		Read OpenQuake spectral hazard curve field tree.
		Read from the folder 'hazard_curve_multi' if present, else read individual
		hazard curves from the folder 'hazard_curve'.

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

	def write_oq_shcft(self, shcft):
		# TODO
		pass

	def read_oq_uhsft(self, return_period, add_stats=False, calc_id=None):
		"""
		Read OpenQuake UHS field tree

		:param return period:
			float, return period
		:param add_stats:
			bool indicating whether or not mean and quantiles have to be appended
		:param calc_id:
			list of ints, calculation IDs.
			(default: None, will determine from folder structure)

		:return:
			instance of :class:`UHSFieldTree`
		"""
		from ..openquake import read_uhsft

		uhs_folder = self.get_oq_uhs_folder(calc_id=calc_id)
		uhsft = read_uhsft(uhs_folder, return_period, self.get_sha_sites(), add_stats=add_stats)
		return uhsft

	def write_oq_uhsft(self, uhsft):
		# TODO
		pass

	def read_crisis_shcft(self, batch_filename="lt_batch.dat"):
		"""
		Read CRISIS spectral hazard curve field tree

		:param batch_filename:
			str, name of batch file (default: "lt_batch.dat")

		:return:
			instance of :class:`SpectralHazardCurveFieldTree`
		"""
		from ..crisis import read_GRA_multi

		gra_filespecs, weights = self.read_crisis_batch(batch_filename)
		shcft = read_GRA_multi(gra_filespecs, weights=weights)
		return shcft

	def get_deagg_bin_edges(self, mag_bin_width, dist_bin_width, coord_bin_width, n_epsilons):
		"""
		Obtain overall deaggregation bin edges

		:param mag_bin_width:
			Float, magnitude bin width
		:param dist_bin_width:
			Float, distance bin width in km
		:param coord_bin_width:
			Float, lon/lat bin width in decimal degrees
		:param n_epsilons:
			Int, number of epsilon bins
			corresponding to integer epsilon values)

		:return:
			(mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, src_bins) tuple
			- mag_bins: magnitude bin edges
			- dist_bins: distance bin edges
			- lon_bins: longitude bin edges
			- lat_bins: latitude bin edges
			- eps_bins: epsilon bin edges
			- trt_bins: source or tectonic-region-type bins
		"""
		min_mag = 9
		max_mag = 0
		min_lon, max_lon = 180, -180
		min_lat, max_lat = 90, -90
		for i, (psha_model, weight) in enumerate(self.sample_logic_trees()):
			source_model = psha_model.source_model
			if source_model.max_mag > max_mag:
				max_mag = source_model.max_mag
			if source_model.min_mag < min_mag:
				min_mag = source_model.min_mag
			west, east, south, north = source_model.get_bounding_box()
			if west < min_lon:
				min_lon = west
			if east > max_lon:
				max_lon = east
			if south < min_lat:
				min_lat = south
			if north > max_lat:
				max_lat = north

		trt_bins = set()
		if len(self.source_models) > 1:
			## Collect tectonic region types
			for source_model in self.source_models:
				for src in source_model:
					trt_bins.add(src.tectonic_region_type)
		else:
			## Collect source IDs
			for src in self.source_models[0]:
				trt_bins.add(src.source_id)

		min_mag = np.floor(min_mag / mag_bin_width) * mag_bin_width
		max_mag = np.ceil(max_mag / mag_bin_width) * mag_bin_width
		min_dist = 0
		max_dist = self.integration_distance
		## Note that ruptures may extend beyond source limits
		min_lon = (np.floor(min_lon / coord_bin_width) - 1)* coord_bin_width
		min_lat = (np.floor(min_lat / coord_bin_width) + 1) * coord_bin_width
		max_lon = (np.ceil(max_lon / coord_bin_width) - 1) * coord_bin_width
		max_lat = (np.ceil(max_lat / coord_bin_width) + 1) * coord_bin_width

		nmags = int((max_mag - min_mag) / mag_bin_width)
		ndists = int(self.integration_distance / dist_bin_width)
		nlons = int((max_lon - min_lon) / coord_bin_width)
		nlats = int((max_lat - min_lat) / coord_bin_width)

		mag_bins = np.linspace(min_mag, max_mag, nmags + 1)
		dist_bins = np.linspace(min_dist, max_dist, ndists + 1)
		lon_bins = np.linspace(min_lon, max_lon, nlons + 1)
		lat_bins = np.linspace(min_lat, max_lat, nlats + 1)
		eps_bins = np.linspace(-self.truncation_level, self.truncation_level,
								  n_epsilons + 1)
		trt_bins = sorted(trt_bins)

		return (mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, trt_bins)

	def get_mean_deagg_sdc(self, site, calc_id="oqhazlib"):
		"""
		Compute mean spectral deaggregation curve from individual models.

		:param site:

		:return:
			instance of :class:`SpectralDeaggregationCurve`
		"""
		import gc

		## Read saved deaggregation files for given site
		num_lt_samples = self.num_lt_samples or self.get_num_paths()
		fmt = "%%0%dd" % len(str(num_lt_samples))
		for i, (psha_model, weight) in enumerate(self.sample_logic_trees()):
			curve_name = fmt % (i+1)
			print curve_name
			sdc = self.read_oq_disagg_matrix_multi(curve_name, site, calc_id=calc_id)
			## Apply weight
			sdc_matrix = sdc.deagg_matrix.to_fractional_contribution_matrix()
			sdc_matrix *= weight

			if i == 0:
				## Obtain overall bin edges
				mag_bin_width = sdc.mag_bin_width
				dist_bin_width = sdc.dist_bin_width
				coord_bin_width = sdc.lon_bin_wdith
				num_epsilon_bins = sdc.neps

				bin_edges = self.get_deagg_bin_edges(mag_bin_width, dist_bin_width, coord_bin_width, num_epsilon_bins)
				mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, trt_bins = bin_edges

				## Create empty matrix
				nmags = len(mag_bins) - 1
				ndists = len(dist_bins) - 1
				nlons = len(lon_bins) - 1
				nlats = len(lat_bins) - 1
				neps = num_epsilon_bins
				ntrts = len(trt_bins)
				num_periods = len(sdc.periods)

				shape = (num_periods, nmags, ndists, nlons, nlats, neps, ntrts)
				mean_deagg_matrix = rshalib.result.FractionalContributionMatrix(np.zeros(shape, sdc.dtype))

			## Sum deaggregation results of logic_tree samples
			## Assume min_mag, distance bins and eps bins are the same for all models
			max_mag_idx = sdc.nmags
			min_lon_idx = (sdc.min_lon - lon_bins[0]) / coord_bin_width
			max_lon_idx = min_lon_idx + sdc.nlons
			min_lat_idx = (sdc.min_lat - lat_bins[0]) / coord_bin_width
			max_lat_idx = min_lat_idx + sdc.nlats
			#print max_mag_idx
			#print sdc.min_lon, sdc.max_lon
			#print min_lon_idx, max_lon_idx
			#print sdc.min_lat, sdc.max_lat
			#print min_lat_idx, max_lat_idx

			if len(self.source_models) == 1:
				## trt bins correspond to source IDs
				mean_deagg_matrix[:,:max_mag_idx,:,min_lon_idx:max_lon_idx,min_lat_idx:max_lat_idx,:,:] += sdc_matrix
			else:
				## trt bins correspond to tectonic region types
				for trt_idx, trt in enumerate(trt_bins):
					src_idxs = []
					for src_id in sdc.trt_bins:
						src = psha_model.source_model[src_id]
						if src.tectonic_region_type == trt:
							src_idxs.append(src_id)
				mean_deagg_matrix[:,:max_mag_idx,:,min_lon_idx:max_lon_idx,min_lat_idx:max_lat_idx,:,trt_idx] += sdc_matrix[:,:,:,:,:,:,:,src_idxs]

			del sdc_matrix
			gc.collect()

		intensities = np.zeros_like(sdc.intensities)

		return SpectralDeaggregationCurve(bin_edges, mean_deagg_matrix, site, sdc.imt, intensities, sdc.periods, sdc.return_periods, sdc.timespan)



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

