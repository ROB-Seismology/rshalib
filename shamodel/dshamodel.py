"""
:mod:`rshalib.shamodel.dshamodel` exports :class:`rshalib.shamodel.dshamodel.DSHAModel`
"""


import numpy as np

from openquake.hazardlib.calc import ground_motion_fields
from openquake.hazardlib.calc.filters import rupture_site_distance_filter, rupture_site_noop_filter
from openquake.hazardlib.gsim import get_available_gsims
from openquake.hazardlib.imt import *

from ..result import HazardMap, HazardMapSet
from base import SHAModelBase
from ..site.ref_soil_params import REF_SOIL_PARAMS


# NOTE: for each IMT a hazard map set is returned with for each realization a max hazard map.
# But realizations are calculated per rupture, not per set of ruptures, so is this correct?


class RuptureDSHAModel(SHAModelBase):
	"""
	Class representing a single DSHA model.
	"""

	def __init__(self, name, ruptures, gsim_name, sites=None, grid_outline=None, grid_spacing=None, soil_site_model=None, ref_soil_params=REF_SOIL_PARAMS, imt_periods={'PGA': [0]}, truncation_level=0., realizations=1., correlation_model=None, integration_distance=200.):
		"""
		:param name:
			see :class:`..shamodel.SHAModelBase`
		:param ruptures:
			list of instances of :class:`..source.Rupture`
		:param gsim_name:
			str, name of supported GMPE
		:param sites:
			see :class:`..shamodel.SHAModelBase` (default: None)
		:param grid_outline:
			see :class:`..shamodel.SHAModelBase` (default: None)
		:param grid_spacing:
			see :class:`..shamodel.SHAModelBase` (default: None)
		:param soil_site_model:
			see :class:`..shamodel.SHAModelBase` (default: None)
		:param ref_soil_params:
			see :class:`..shamodel.SHAModelBase` (default: REF_SOIL_PARAMS)
		:param imt_periods:
			see :class:`..shamodel.SHAModelBase` (default: {'PGA': [0]})
		:param truncation_level:
			see :class:`..shamodel.SHAModelBase` (default: 0.)
		:param realizations:
			int, number of realizations (default: 1)
		:param correlation_model:
			instance of subclass of :class:`openquake.hazardlib.correlation.BaseCorrelationModel` (default: None)
		:param integration_distance:
			see :class:`..shamodel.SHAModelBase` (default: 200.)
		"""
		SHAModelBase.__init__(self, name, sites, grid_outline, grid_spacing, soil_site_model, ref_soil_params, imt_periods, truncation_level, integration_distance)

		self.ruptures = ruptures
		self.gsim_name = gsim_name
		self.realizations = realizations
		self.correlation_model = correlation_model

	def _get_hazardlib_imts(self):
		"""
		:returns:
			list of instances of subclasses of :class:`openquake.hazardlib._IMT`
		"""
		imts = []
		for imt in self.imt_periods:
			if imt == "SA":
				for period in self.imt_periods[imt]:
					imts.append(eval(imt)(period, 5.))
			else:
				imts.append(eval(imt)())
		return imts

	def _get_hazardlib_rsdf(self):
		"""
		:returns:
			function, filter for sites further away from rupture than integration_distance
		"""
		if self.integration_distance:
			return rupture_site_distance_filter(self.integration_distance)
		else:
			return rupture_site_noop_filter

	def run_hazardlib(self):
		"""
		:returns:
			instance of :class:`..results.HazardMapSet`
		"""
		soil_site_model = self.get_soil_site_model()
		gsim = get_available_gsims()[self.gsim_name]()
		imts = self._get_imts()
		rsdf = self.rupture_site_filter
		intensities = {imt: np.zeros((len(soil_site_model), len(self.ruptures), self.realizations)) for imt in imts}
		for i, rupture in enumerate(self.ruptures):
			gmfs = ground_motion_fields(rupture, soil_site_model, imts, gsim, self.truncation_level, self.realizations, self.correlation_model, rsdf)
			for imt, gmf in gmfs.items():
				if self.correlation_model:
					intensities[imt][:,i,:] = gmf.getA()
				else:
					intensities[imt][:,i,:] = gmf
		hazard_map_sets = {}
		for imt in imts:
			if isinstance(imt, SA):
				period = imt.period
				IMT = "SA"
				key = period
			else:
				period = 0
				# TODO: need a more elegant solution
				IMT = {PGV(): "PGV", PGD(): "PGD", PGA(): "PGA", MMI(): "MMI"}[imt]
				key = IMT
			hazard_map_sets[key] = HazardMapSet("", [""]*self.realizations, self.sha_site_model, period, IMT, np.amax(intensities[imt], axis=1).T, intensity_unit="g", timespan=1, poes=None, return_periods=np.ones(self.realizations), vs30s=None)
		return hazard_map_sets


class DSHAModel(SHAModelBase):
	def __init__(self, name, source_model, gmpe_system_def, sites=None,
				grid_outline=None, grid_spacing=None, soil_site_model=None,
				ref_soil_params=REF_SOIL_PARAMS, imt_periods={'PGA': [0]},
				truncation_level=0., integration_distance=200.,
				correlation_model=None):

		# TODO: evaluate whether truncation_level and correlation_model have to be class properties

		SHAModelBase.__init__(self, name, sites, grid_outline, grid_spacing, soil_site_model, ref_soil_params, imt_periods, truncation_level, integration_distance)
		self.source_model = source_model
		self.gmpe_system_def = gmpe_system_def
		self.correlation_model = correlation_model

	def calc_gmf(self, num_realizations=1, correlation_model=None):
		pass

	def calc_gmf_envelope(self, total_residual_epsilon=0):
		"""
		Historical ground motion check
		"""
		fake_tom = oqhazlib.tom.PoissonTOM(1)
		total_residual_epsilons = np.ones_like(self.get_sha_sites())
		total_residual_epsilons *= total_residual_epsilon

		soil_site_model = self.get_soil_site_model()
		imt_list = self._get_imts()
		gmf = np.zeros((len(soil_site_model), len(imt_list)))

		for k, imt in enumerate(imt_list):
			for src in self.source_model:
				src_gmf = np.zeros(len(soil_site_model))
				trt = src.tectonic_region_type
				gmpe_pmf = self.gmpe_system_def[trt]
				for (gmpe_name, weight) in gmpe_pmf:
					gsim = oqhazlib.gsim.get_available_gsims()[gmpe_name]
					gsim_gmf = np.zeros(len(soil_site_model))
					for rup in src.iter_ruptures(fake_tom):
						# TODO: check if CharacteristicFaultSources return only 1 rupture
						# Weighted average using rup.occurrence_rate or envelope?
						gmf = oqhazlib.ground_motion_field_with_residuals(
							rup, soil_site_model, imt, gsim, 1,
							total_residual_epsilons=total_residual_epsilons,
							intra_residual_epsilons=None,
							inter_residual_epsilons=None)

		return UHSField(self.name, "", sites, periods, IMT, gmf, intensity_unit="g", timespan=1, poe=None, return_period=None, vs30s=None)



if __name__ == "__main__":
	"""
	Test
	"""
	from openquake.hazardlib.scalerel import PointMSR

	from hazard.rshalib.source.rupture import Rupture

	name = "TestDSHAModel"

	strike, dip, rake, trt, rms, rar, usd, lsd, msr = 0., 45., 0., "", 1., 1., 0., 30., PointMSR()

	## earthquake
	earthquakes = [(4.5, 50.5, 10., 6.5)]

	## catalog
	from hazard.psha.Projects.SHRE_NPP.catalog import cc_catalog
	earthquakes = [(e.lon, e.lat, e.depth or 10., e.get_MW())for e in cc_catalog]

	ruptures = [Rupture.from_hypocenter(lon, lat, depth, mag, strike, dip, rake, trt, rms, rar, usd, lsd, msr) for lon, lat, depth, mag in earthquakes]

	gsim = "CauzziFaccioli2008"
	sites = None
	grid_outline = (1, 8, 49, 52)
	grid_spacing = 0.1
	soil_site_model = None
	ref_soil_params = REF_SOIL_PARAMS
	imt_periods = {'PGA': [0]}
	truncation_level = 0.
	realizations = 2
	correlation_model = None
	integration_distance = None

	dhsa_model = DSHAModel(name, ruptures, gsim, sites, grid_outline, grid_spacing, soil_site_model, ref_soil_params, imt_periods, truncation_level, realizations, correlation_model, integration_distance)

	hazard_map_sets = dhsa_model.run_hazardlib()
	print hazard_map_sets["PGA"].getHazardMap(0)
	print hazard_map_sets["PGA"].getHazardMap(1)

