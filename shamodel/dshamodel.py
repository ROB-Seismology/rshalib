"""
:mod:`rshalib.shamodel.dshamodel` exports :class:`rshalib.shamodel.dshamodel.DSHAModel`
"""


import numpy as np

import openquake.hazardlib as oqhazlib
#from openquake.hazardlib.calc import ground_motion_fields
#from openquake.hazardlib.calc.filters import rupture_site_distance_filter, rupture_site_noop_filter
from openquake.hazardlib.gsim import get_available_gsims
from openquake.hazardlib.imt import *

from ..result import HazardMap, HazardMapSet, UHSField
from base import SHAModelBase
from ..site.ref_soil_params import REF_SOIL_PARAMS
from ..calc import mp


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
		# TODO!
		pass

	def calc_gmf_envelope(self, stddev_type="total", np_aggregation="avg"):
		"""
		Historical ground motion check

		Fixed epsilon!

		:param stddev_type:
			str, standard deviation type, one of "total", "inter-event" or "intra-event"
			Note: GMPE must support inter-event / intra-event standard deviations
			if one of these options is chosen.
			(default: "total")
		:param np_aggregation:
			str, how to aggregate nodal planes for a given point rupture,
			either "avg" (weighted average) or "max" (maximum)
			(default: "avg")
		"""
		soil_site_model = self.get_soil_site_model()
		num_sites = len(soil_site_model)
		imt_list = self._get_imts()
		fake_tom = oqhazlib.tom.PoissonTOM(1)
		if stddev_type == "total":
			total_residual_epsilons = np.ones_like(num_sites)
			total_residual_epsilons *= self.truncation_level
			inter_residual_epsilons = total_residual_epsilons
			intra_residual_epsilons = total_residual_epsilons
		elif stddev_type == "inter-event":
			total_residual_epsilons = None
			inter_residual_epsilons = np.ones_like(num_sites)
			inter_residual_epsilons *= self.truncation_level
			intra_residual_epsilons = np.ones_like(num_sites)
		elif stddev_type == "intra-event":
			total_residual_epsilons = None
			inter_residual_epsilons = np.ones_like(num_sites)
			intra_residual_epsilons = np.ones_like(num_sites)
			intra_residual_epsilons *= self.truncation_level

		gmf_envelope = np.zeros((num_sites, len(imt_list)))
		amax = 0

		for k, imt in enumerate(imt_list):
			for src in self.source_model:
				src_gmf = np.zeros(len(soil_site_model))
				trt = src.tectonic_region_type
				gmpe_pmf = self.gmpe_system_def[trt]
				for (gmpe_name, gmpe_weight) in gmpe_pmf:
					gsim = oqhazlib.gsim.get_available_gsims()[gmpe_name]()
					gmpe_gmf = np.zeros(len(soil_site_model))
					total_rupture_probability = 0
					for rup in src.iter_ruptures(fake_tom):
						total_rupture_probability += rup.occurrence_rate
						# TODO: check if CharacteristicFaultSources return only 1 rupture
						gmf = mp.calc_gmf_with_fixed_epsilon(
							rup, soil_site_model, tuple(imt), gsim, self.truncation_level,
							total_residual_epsilons=total_residual_epsilons,
							intra_residual_epsilons=intra_residual_epsilons,
							inter_residual_epsilons=inter_residual_epsilons,
							integration_distance=self.integration_distance)
						## Aggregate gmf's corresponding to different nodal planes
						if np_aggregation == "avg":
							gmpe_gmf += (gmf * rup.occurrence_rate)
						elif np_aggregation == "max":
							gmpe_gmf = np.max([gmpe_gmf, gmf], axis=0)
					## Weighted average of gmf's due to different GMPEs
					assert np.allclose(total_rupture_probability, 1)
					src_gmf += (gmpe_gmf * float(gmpe_weight))
				amax = max(amax, src_gmf.max())
				## Take envelope of different source gmf's
				gmf_envelope[:,k] = np.max([gmf_envelope[:,k], src_gmf], axis=0)
		print gmf_envelope.min(), gmf_envelope.max()
		print amax

		periods = self._get_periods()
		sites = soil_site_model.get_sha_sites()
		return UHSField(self.name, "", sites, periods, "SA", gmf_envelope, intensity_unit="g", timespan=1, return_period=1)

	def calc_gmf_envelope_mp(self, num_cores=None, stddev_type="total",
							np_aggregation="avg", verbose=False):
		"""
		:param num_cores:
			int, number of CPUs to be used. Actual number of cores used
			may be lower depending on available cores and memory
			(default: None, will determine automatically)
		"""
		soil_site_model = self.get_soil_site_model()
		num_sites = len(soil_site_model)
		imt_list = self._get_imts()
		fake_tom = oqhazlib.tom.PoissonTOM(1)
		if stddev_type == "total":
			total_residual_epsilons = np.ones_like(num_sites)
			total_residual_epsilons *= self.truncation_level
			inter_residual_epsilons = total_residual_epsilons
			intra_residual_epsilons = total_residual_epsilons
		elif stddev_type == "inter-event":
			total_residual_epsilons = None
			inter_residual_epsilons = np.ones_like(num_sites)
			inter_residual_epsilons *= self.truncation_level
			intra_residual_epsilons = np.ones_like(num_sites)
		elif stddev_type == "intra-event":
			total_residual_epsilons = None
			inter_residual_epsilons = np.ones_like(num_sites)
			intra_residual_epsilons = np.ones_like(num_sites)
			intra_residual_epsilons *= self.truncation_level

		gmf_envelope = np.zeros((num_sites, len(imt_list)))

		## Create list with arguments for each job
		job_args = []
		for k, imt in enumerate(imt_list):
			for src in self.source_model:
				trt = src.tectonic_region_type
				gmpe_pmf = self.gmpe_system_def[trt]
				for (gmpe_name, gmpe_weight) in gmpe_pmf:
					gsim = oqhazlib.gsim.get_available_gsims()[gmpe_name]()
					gmpe_gmf = np.zeros(len(soil_site_model))
					for rup in src.iter_ruptures(fake_tom):
						# TODO: check if CharacteristicFaultSources return only 1 rupture
						job_args.append((rup, soil_site_model, tuple(imt), gsim, self.truncation_level,
										total_residual_epsilons, intra_residual_epsilons,
										inter_residual_epsilons, self.integration_distance))

		## Launch multiprocessing
		if len(job_args) > 0:
			gmf_list = mp.run_parallel(mp.calc_gmf_with_fixed_epsilon, job_args, num_cores, verbose=verbose)

		i = 0
		for k, imt in enumerate(imt_list):
			for src in self.source_model:
				src_gmf = np.zeros(len(soil_site_model))
				trt = src.tectonic_region_type
				gmpe_pmf = self.gmpe_system_def[trt]
				for (gmpe_name, gmpe_weight) in gmpe_pmf:
					gsim = oqhazlib.gsim.get_available_gsims()[gmpe_name]()
					gmpe_gmf = np.zeros(len(soil_site_model))
					total_rupture_probability = 0
					for rup in src.iter_ruptures(fake_tom):
						total_rupture_probability += rup.occurrence_rate
						gmf = gmf_list[i]
						i += 1
						## Aggregate gmf's corresponding to different nodal planes
						if np_aggregation == "avg":
							gmpe_gmf += (gmf * rup.occurrence_rate)
						elif np_aggregation == "max":
							gmpe_gmf = np.max([gmpe_gmf, gmf], axis=0)
					## Weighted average of gmf's due to different GMPEs
					assert np.allclose(total_rupture_probability, 1)
					src_gmf += (gmpe_gmf * float(gmpe_weight))
				## Take envelope of different source gmf's
				gmf_envelope[:,k] = np.max([gmf_envelope[:,k], src_gmf], axis=0)

		periods = self._get_periods()
		sites = soil_site_model.get_sha_sites()
		return UHSField(self.name, "", sites, periods, "SA", gmf_envelope, intensity_unit="g", timespan=1, return_period=1)



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

