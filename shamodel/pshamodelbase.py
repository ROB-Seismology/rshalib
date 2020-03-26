"""
PSHAModelBase class
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import int

try:
	## Python 2
	basestring
except:
	## Python 3
	basestring = str


# NOTE: damping for spectral periods is fixed at 5.


### imports
import os
from collections import OrderedDict

import numpy as np

import openquake.hazardlib as oqhazlib
from openquake.hazardlib.imt import PGA, SA, PGV, PGD, MMI

from ..poisson import poisson_conv
from ..geo import *
from ..site import *
from ..result import (SpectralHazardCurveField, DeaggregationSlice,
						SpectralDeaggregationCurve)
from .base import SHAModelBase


# TODO: make distinction between imt (PGA, SA) and im (SA(0.5, 5.0), SA(1.0, 5.0))
# (perhaps put all these functions in rshalib.imt)

# im (or imt_name?): intensity measure, e.g. "PGA", "SA"
# imt: IMT object, e.g. PGA(), SA(0.2, 5.0)
# imls: intensity measure levels, e.g. [0.1, 0.2, 0.3]
# im_periods: dict mapping im to spectral periods, e.g. {"PGA": [0], "SA": [0.1, 0.5, 1.]}
# imtls --> imt_imls: dict mapping IMT objects to imls (1-D arrays)
# im_imls: dict mapping im strings to imls (2-D arrays)



__all__ = ['PSHAModelBase']


class PSHAModelBase(SHAModelBase):
	"""
	Base class for PSHA models, holding common attributes and methods.

	:param root_folder:
		str, defining full path to root folder.
	:param site_model:
	:param ref_soil_params:
	:param imt_periods:
		see :class:`SHAModelBase`
	:param intensities:
		list of floats or array, defining equal intensities for all
		intensity measure types and periods
		When given, params min_intensities, max_intensities and
		num_intensities are ignored.
	:param min_intensities:
		dict mapping intensity measure types (e.g. "PGA", "SA",
		"PGV", "PGD") to lists or arrays (one for each period) of
		minimum intensities (float values).
	:param max_intensities:
		dict mapping intensity measure types (e.g. "PGA", "SA",
		"PGV", "PGD") to lists or arrays (one for each period) of
		maximum intensities (float values).
	:param num_intensities:
		float, defining number of intensities
	:param return_periods:
		list of floats, defining return periods.
	:param time_span:
		float, defining time span in years
	:param truncation_level:
	:param integration_distance:
		see :class:`SHAModelBase`
	"""

	def __init__(self, name, root_folder,
				site_model, ref_soil_params,
				imt_periods, intensities,
				min_intensities, max_intensities, num_intensities,
				return_periods, time_span,
				truncation_level, integration_distance):
		"""
		"""
		SHAModelBase.__init__(self, name, site_model, ref_soil_params, imt_periods,
							truncation_level, integration_distance)

		self.root_folder = root_folder
		self.intensities = intensities
		if self.intensities:
			self.min_intensities = None
			self.max_intensities = None
			if isinstance(self.intensities, dict):
				key = self.intensities.keys()[0]
				self.num_intensities = len(self.intensities[key])
			else:
				self.num_intensities = len(self.intensities)
		else:
			self.min_intensities = self._get_intensities_limits(min_intensities)
			self.max_intensities = self._get_intensities_limits(max_intensities)
			self.num_intensities = num_intensities
		self.return_periods = np.array(return_periods)
		self.time_span = time_span

	@property
	def poisson_tom(self):
		return oqhazlib.tom.PoissonTOM(self.time_span)

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
		import errno

		path = path.split(self.root_folder)[1]
		subfolders = path.split(os.path.sep)
		partial_path = self.root_folder
		for subfolder in subfolders:
			partial_path = os.path.join(partial_path, subfolder)
			if not os.path.exists(partial_path):
				try:
					os.mkdir(partial_path)
				except OSError, err:
					if err.errno == errno.EEXIST and os.path.isdir(partial_path):
						## Folder already created by another process
						pass
					else:
						## Our target dir exists as a file, or different error,
						## reraise the error!
						raise

	def get_oq_hc_folder(self, calc_id=None, multi=False):
		"""
		Return full path to OpenQuake hazard_curve folder

		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)
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
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)

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
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)

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
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)
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
		Return dict, defining minimum or maximum intensity
		for intensity type and period.

		:param intensities_limits:
			dict or float
		"""
		if not isinstance(intensities_limits, dict):
			intensities_limits = {imt: [intensities_limits]*len(periods)
								for imt, periods in self.imt_periods.items()}
		return intensities_limits

	def _get_im_imls(self, combine_pga_and_sa=True):
		"""
		Construct a dictionary containing a 2-D array [k, l] of
		intensities for each IMT.

		:param combine_pga_and_sa:
			bool, whether or not to combine PGA and SA, if present
			(default: True)

		:return:
			dict {IMT (string): intensities (2-D numpy array of floats)}
		"""
		from ..utils import logrange

		imtls = {}
		for imt, periods in self.imt_periods.items():
			if len(periods) > 1:
				imls = np.zeros((len(periods), self.num_intensities))
				for k, period in enumerate(periods):
					if self.intensities:
						if isinstance(self.intensities, dict):
							imls[k,:] = np.array(self.intensities[(imt, period)])
						else:
							imls[k,:] = np.array(self.intensities)
					else:
						imls[k,:] = logrange(self.min_intensities[imt][k],
											self.max_intensities[imt][k],
											self.num_intensities)
				imtls[imt] = imls
			else:
				if self.intensities:
					if isinstance(self.intensities, dict):
						imtls[imt] = np.array(self.intensities[(imt, periods[0])])
					else:
						imtls[imt] = np.array(self.intensities)
				else:
					imtls[imt] = logrange(self.min_intensities[imt][0],
										self.max_intensities[imt][0],
										self.num_intensities)
				imtls[imt] = imtls[imt].reshape(1, self.num_intensities)
		if (combine_pga_and_sa and "PGA" in self.imt_periods.keys()
			and "SA" in self.imt_periods.keys()):
			imtls["PGA"].shape = (1, self.num_intensities)
			imtls["SA"] = np.concatenate([imtls["PGA"], imtls["SA"]], axis=0)
			del imtls["PGA"]
		return imtls

	def _get_imtls(self):
		"""
		Construct a dictionary mapping oqhazlib intensity measure type
		objects to 1-D arrays of intensity measure levels.
		This dictionary can be passed as an argument to
		:func:`oqhazlib.calc.hazard_curves_poissonian``.

		:return:
			dict {:mod:`oqhazlib.imt` object: 1-D numpy array of floats}
		"""
		from ..utils import logrange

		imtls = OrderedDict()
		for im, periods in sorted(self.imt_periods.items()):
			for k, period in enumerate(periods):
				imt = self._construct_imt(im, period)
				if self.intensities:
					if isinstance(self.intensities, dict):
						imtls[imt] = np.array(self.intensities[(im, period)])
					else:
						imtls[imt] = np.array(self.intensities)
				else:
					imtls[imt] = logrange(self.min_intensities[im][k],
										self.max_intensities[im][k],
										self.num_intensities)

		return imtls

	def _get_deagg_site_imtls(self, deagg_sites, deagg_imt_periods):
		"""
		Construct imtls dictionary containing all available intensities
		for the spectral periods for which deaggregation will be performed.
		This dictionary can be passed as an argument to deaggregate methods.

		:param deagg_sites:
			list with instances of :class:`GenericSite` for which deaggregation
			will be performed. Note that instances of class:`SoilSite` will
			not work with multiprocessing
		:param deagg_imt_periods:
			dictionary mapping intensity measure strings to lists of spectral
			periods for which deaggregation will be performed.

		:return:
			dictionary mapping site (lon, lat) tuples to dictionaries
			mapping oqhazlib intensity measure type objects to 1-D arrays
			of intensity measure levels
		"""
		all_imtls = self._get_imtls()
		site_imtls = OrderedDict()
		for site in deagg_sites:
			try:
				lon, lat = site.lon, site.lat
			except AttributeError:
				lon, lat = site.location.longitude, site.location.latitude
			site_imtls[(lon, lat)] = OrderedDict()
			for im in deagg_imt_periods.keys():
				for T in deagg_imt_periods[im]:
					imt = self._construct_imt(im, T)
					site_imtls[(lon, lat)][imt] = all_imtls[imt]
		return site_imtls

	def _get_openquake_imts(self):
		"""
		Construct a dictionary mapping intensity measure type strings
		to 1-D arrays of intensity measure levels. This dictionary can be
		passed to :class:`OQParams`.`set_imts` function, which is used to
		generate the configuration file for OpenQuake.

		:return:
			dict {imt (string): 1-D numpy array of floats}
		"""
		from ..utils import logrange

		# TODO: probably better to move this into config.py, where we had a similar method
		imtls = {}
		for imt, periods in self.imt_periods.items():
			if imt == "SA":
				for k, period in enumerate(periods):
					if self.intensities:
						if isinstance(self.intensities, dict):
							intensities = self.intensities[(imt, period)]
						else:
							intensities = self.intensities
					else:
						intensities = logrange(self.min_intensities[imt][k],
												self.max_intensities[imt][k],
												self.num_intensities)
					key = imt + "(%s)" % period
					imtls[key] = list(map(float, intensities))
			else:
				if self.intensities:
					if isinstance(self.intensities, dict):
						imtls[imt] = self.intensities[(imt, periods[0])]
					else:
						imtls[imt] = self.intensities
				else:
					imtls[imt] = logrange(self.min_intensities[imt][0],
											self.max_intensities[imt][0],
											self.num_intensities)
				imtls[imt] = list(map(float, intensities))
		return imtls

	def _handle_oq_soil_params(self, params, calc_id=None):
		"""
		Write nrml file for soil site model if present and set file param,
		or set reference soil params

		:param params:
			instance of :class:`OQ_Params` where soil parameters will
			be added.
		:param calc_id:
			str, calculation ID correspoding to subfolder where xml files
			will be written.
			(default: None)
		"""
		if isinstance(self.site_model, SoilSiteModel):
			if calc_id:
				oq_folder = os.path.join(self.oq_root_folder, "calc_%s" % calc_id)
			else:
				oq_folder = self.oq_root_folder
			file_name = (self.site_model.name or "soil_site_model") + ".xml"
			self.site_model.write_xml(os.path.join(oq_folder, file_name))
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

	def read_oq_hcf(self, curve_name, im, T, curve_path="", calc_id=None):
		"""
		Read OpenQuake hazard curve field

		:param curve_name:
			str, identifying hazard curve
			(e.g., "rlz-01", "mean", "quantile_0.84")
		:param im:
			str, intensity measure
		:param T:
			float, spectral period
		:param curve_path:
			str, path to hazard curve relative to main hazard-curve folder
			(default: "")
		:param calc_id:
			str, calculation ID.
			(default: None, will determine from folder structure)

		:return:
			instance of :class:`HazardCurveField`
		"""
		from ..openquake import parse_hazard_curves

		hc_folder = self.get_oq_hc_folder(calc_id=calc_id, multi=False)
		hc_folder = os.path.join(hc_folder, curve_path)
		imt_subfolder = self._get_oq_imt_subfolder(im, T)
		xml_filename = "hazard_curve-%s.xml" % curve_name
		#print(xml_filename)
		xml_filespec = os.path.join(hc_folder, imt_subfolder, xml_filename)
		hcf = parse_hazard_curves(xml_filespec)
		hcf.set_site_names(self.get_generic_sites())

		return hcf

	def write_oq_hcf(self, hcf, curve_name, curve_path="", calc_id="oqhazlib"):
		"""
		Write OpenQuake hazard curve field. Folder structure will be
		created, if necessary.

		:param hcf:
			instance of :class:`HazardCurveField`
		:param curve_name:
			str, identifying hazard curve
			(e.g., "rlz-01", "mean", "quantile_0.84")
		:param curve_path:
			str, path to hazard curve relative to main hazard-curve folder
			(default: "")
		:param calc_id:
			str, calculation ID. (default: "oqhazlib")
		"""
		hc_folder = self.get_oq_hc_folder(calc_id=calc_id, multi=False)
		hc_folder = os.path.join(hc_folder, curve_path)
		imt_subfolder = self._get_oq_imt_subfolder(hcf.IMT, hcf.period)
		imt_hc_folder = os.path.join(hc_folder, imt_subfolder)
		self.create_folder_structure(imt_hc_folder)
		xml_filename = "hazard_curve-%s.xml" % curve_name
		xml_filespec = os.path.join(imt_hc_folder, xml_filename)
		hcf.write_nrml(xml_filespec)

	def get_oq_shcf_filespec(self, curve_name, curve_path="", calc_id=None):
		"""
		Get full path to OpenQuake spectral hazard curve xml file

		:param curve_name:
			str, identifying hazard curve
			(e.g., "rlz-01", "mean", "quantile_0.84")
		:param curve_path:
			str, path to hazard curve relative to main hazard-curve folder
			(default: "")
		:param calc_id:
			str, calculation ID.
			(default: None, will determine from folder structure)

		:return:
			str, full path to spectral hazard curve file
		"""
		# TODO: we should add im parameter
		hc_folder = self.get_oq_hc_folder(calc_id=calc_id, multi=True)
		hc_folder = os.path.join(hc_folder, curve_path)
		xml_filename = "hazard_curve_multi-%s.xml" % curve_name
		xml_filespec = os.path.join(hc_folder, xml_filename)
		return xml_filespec

	def read_oq_shcf(self, curve_name, curve_path="", calc_id=None, verbose=False):
		"""
		Read OpenQuake spectral hazard curve field

		:param curve_name:
			str, identifying hazard curve
			(e.g., "rlz-01", "mean", "quantile_0.84")
		:param curve_path:
			str, path to hazard curve relative to main hazard-curve folder
			(default: "")
		:param calc_id:
			str, calculation ID.
			(default: None, will determine from folder structure)
		:param verbose:
			bool, whether or not to print additional information
			(default: False)

		:return:
			instance of :class:`SpectralHazardCurveField`
		"""
		from ..openquake import parse_hazard_curves, parse_spectral_hazard_curve_field

		xml_filespec = self.get_oq_shcf_filespec(curve_name, curve_path=curve_path,
												calc_id=calc_id)
		if verbose:
			print("Reading hazard curve file %s" % xml_filespec)
		try:
			shcf = parse_hazard_curves(xml_filespec)
		except:
			shcf = parse_spectral_hazard_curve_field(xml_filespec)
		shcf.set_site_names(self.get_generic_sites())

		return shcf

	def write_oq_shcf(self, shcf, curve_name, curve_path="", calc_id="oqhazlib"):
		"""
		Write OpenQuake spectral hazard curve field. Folder structure
		will be created, if necessary.

		:param shcf:
			instance of :class:`SpectralHazardCurveField`
		:param curve_name:
			str, identifying hazard curve
			(e.g., "rlz-01", "mean", "quantile_0.84")
		:param curve_path:
			str, path to hazard curve relative to main hazard-curve folder
			(default: "")
		:param calc_id:
			str, calculation ID.
			(default: "oqhazlib")
		"""
		xml_filespec = self.get_oq_shcf_filespec(curve_name, curve_path=curve_path,
												calc_id=calc_id)
		hc_folder = os.path.split(xml_filespec)[0]
		self.create_folder_structure(hc_folder)
		shcf.write_nrml(xml_filespec, smlt_path=self.smlt_path,
						gmpelt_path=self.gmpelt_path)

	def read_oq_uhs_multi(self):
		# TODO
		pass

	def read_oq_uhs_field(self, curve_name, return_period, curve_path="",
						calc_id=None):
		"""
		Read OpenQuake hazard curve field.

		:param curve_name:
			str, identifying hazard curve
			(e.g., "rlz-01", "mean", "quantile_0.84")
		:param return period:
			float, return period
		:param curve_path:
			str, path to hazard curve relative to main uhs folder
			(default: "")
		:param calc_id:
			str, calculation ID.
			(default: None, will determine from folder structure)

		:return:
			instance of :class:`UHSField`
		"""
		from ..openquake import parse_uh_spectra

		poe = str(round(poisson_conv(t=self.time_span, tau=return_period), 13))

		uhs_folder = self.get_oq_uhs_folder(calc_id=calc_id)
		uhs_folder = os.path.join(uhs_folder, curve_path)
		xml_filename = "uh_spectra-poe_%s-%s.xml" % (poe, curve_name)
		#print(xml_filename)
		xml_filespec = os.path.join(uhs_folder, xml_filename)
		uhsf = parse_uh_spectra(xml_filespec)
		uhsf.set_site_names(self.get_generic_sites())

		return uhsf

	def write_oq_uhs_field(self, uhsf):
		# TODO
		pass

	def read_oq_disagg_matrix(self, curve_name, im, T, return_period, site,
							curve_path="", calc_id=None):
		"""
		Read OpenQuake deaggregation matrix for a particular im,
		spectral period, return period and site.

		:param curve_name:
			str, identifying hazard curve
			(e.g., "rlz-01", "mean", "quantile_0.84")
		:param im:
			str, intensity measure
		:param T:
			float, spectral period
		:param return period:
			float, return period
		:param site:
			instance of :class:`GenericSite` or :class:`SoilSite`
		:param curve_path:
			str, path to hazard curve relative to main deaggregation folder
			(default: "")
		:param calc_id:
			str, calculation ID.
			(default: None, will determine from folder structure)

		:return:
			instance of :class:`DeaggregationSlice`
		"""
		from ..openquake import parse_disaggregation

		poe = str(round(poisson_conv(t=self.time_span, tau=return_period), 13))

		disagg_folder = self.get_oq_disagg_folder(calc_id=calc_id, multi=False)
		disagg_folder = os.path.join(disagg_folder, curve_path)
		imt_subfolder = self._get_oq_imt_subfolder(im, T)
		xml_filename = "disagg_matrix(%s)-lon_%s-lat_%s-%s.xml"
		xml_filename %= (poe, site.lon, site.lat, curve_name)
		xml_filespec = os.path.join(disagg_folder, imt_subfolder, xml_filename)
		ds = parse_disaggregation(xml_filespec, site.name)
		return ds

	def write_oq_disagg_matrix(self, ds, curve_name, curve_path="",
								calc_id="oqhazlib"):
		"""
		Write OpenQuake deaggregation matrix. Folder structure will be
		created, if necessary.

		:param ds:
			instance of :class:`DeaggregationSlice`
		:param curve_name:
			str, identifying hazard curve
			(e.g., "rlz-01", "mean", "quantile_0.84")
		:param curve_path:
			str, path to hazard curve relative to main deaggregation folder
			(default: "")
		:param calc_id:
			str, calculation ID. (default: "oqhazlib")
		"""
		poe = str(round(poisson_conv(t=ds.time_span, tau=ds.return_period), 13))

		disagg_folder = self.get_oq_disagg_folder(calc_id=calc_id, multi=False)
		disagg_folder = os.path.join(disagg_folder, curve_path)
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

	def get_oq_sdc_filespec(self,  curve_name, site, curve_path="", calc_id=None):
		"""
		Get full path to OpenQuake spectral deaggregation curve xml file

		:param curve_name:
			str, identifying hazard curve
			(e.g., "rlz-01", "mean", "quantile_0.84")
		:param site:
			instance of :class:`GenericSite` or :class:`SoilSite`
		:param curve_path:
			str, path to hazard curve relative to main deaggregation folder
			(default: "")
		:param calc_id:
			str, calculation ID.
			(default: None, will determine from folder structure)

		:return:
			str, full path to spectral deaggregation curve file
		"""
		disagg_folder = self.get_oq_disagg_folder(calc_id=calc_id, multi=True)
		disagg_folder = os.path.join(disagg_folder, curve_path)
		xml_filename = "disagg_matrix_multi-lon_%s-lat_%s-%s.xml"
		xml_filename %= (site.lon, site.lat, curve_name)
		xml_filespec = os.path.join(disagg_folder, xml_filename)
		return xml_filespec

	def read_oq_disagg_matrix_multi(self, curve_name, site, curve_path="",
									calc_id=None, dtype='f', verbose=False):
		"""
		Read OpenQuake multi-deaggregation matrix for a particular site.

		:param curve_name:
			str, identifying hazard curve
			(e.g., "rlz-01", "mean", "quantile_0.84")
		:param site:
			instance of :class:`GenericSite` or :class:`SoilSite`
		:param curve_path:
			str, path to hazard curve relative to main deaggregation folder
			(default: "")
		:param calc_id:
			str, calculation ID.
			(default: None, will determine from folder structure)
		:param dtype:
			str, precision of deaggregation matrix
			(default: 'f')
		:param verbose:
			bool, whether or not to print additional information
			(default: False)

		:return:
			instance of :class:`SpectralDeaggregationCurve`
		"""
		from ..openquake import parse_spectral_deaggregation_curve

		xml_filespec = self.get_oq_sdc_filespec(curve_name, site,
										curve_path=curve_path, calc_id=calc_id)
		if verbose:
			print("Reading deaggregation file %s" % xml_filespec)
		sdc = parse_spectral_deaggregation_curve(xml_filespec, site.name, dtype=dtype)
		return sdc

	def write_oq_disagg_matrix_multi(self, sdc, curve_name, curve_path="",
									calc_id="oqhazlib"):
		"""
		Write OpenQuake multi-deaggregation matrix. Folder structure
		will be created, if necessary.

		:param sdc:
			instance of :class:`SpectralDeaggregationCurve`
		:param curve_name:
			str, identifying hazard curve
			(e.g., "rlz-01", "mean", "quantile_0.84")
		:param curve_path:
			str, path to hazard curve relative to main deaggregation folder
			(default: "")
		:param calc_id:
			str, calculation ID.
			(default: "oqhazlib")
		"""
		xml_filespec = self.get_oq_sdc_filespec(curve_name, sdc.site,
										curve_path=curve_path, calc_id=calc_id)
		disagg_folder = os.path.split(xml_filespec)[0]
		self.create_folder_structure(disagg_folder)
		sdc.write_nrml(xml_filespec, self.smlt_path, self.gmpelt_path)

	def read_crisis_batch(self, batch_filename = "lt_batch.dat"):
		"""
		Reach CRISIS batch file

		:param batch_filename:
			str, name of batch file
			(default: "lt_batch.dat")

		:return:
			list of gra_filespecs
		"""
		from ..crisis import read_batch

		batch_filespec = os.path.join(self.crisis_root_folder, batch_filename)
		#print(batch_filespec)
		return read_batch(batch_filespec)

	def read_crisis_shcf(self, curve_name, batch_filename="lt_batch.dat"):
		"""
		Read CRISIS spectral hazard curve field

		:param curve_name:
			str, identifying hazard curve (e.g., "rlz-01")
		:param batch_filename:
			str, name of batch file
			(default: "lt_batch.dat")

		:return:
			instance of :class:`SpectralHazardCurveField`
		"""
		from ..crisis import read_GRA

		gra_filespecs, weights = self.read_crisis_batch(batch_filename)
		for gra_filespec in gra_filespecs:
			gra_filename = os.path.split(gra_filespec)[1]
			if curve_name in gra_filename:
				break
		#print(gra_filename)

		shcf = read_GRA(gra_filespec)
		return shcf



if __name__ == '__main__':
	"""
	"""
	pass

