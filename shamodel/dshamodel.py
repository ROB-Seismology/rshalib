"""
:mod:`rshalib.shamodel.dshamodel` exports :class:`rshalib.shamodel.dshamodel.DSHAModel`
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np

from .. import oqhazlib
from openquake.hazardlib.imt import *

from ..result import HazardMapSet, UHSField, UHSFieldTree
from ..site.ref_soil_params import REF_SOIL_PARAMS
from ..calc import mp
from .base import SHAModelBase


# NOTE: for each IMT a hazard map set is returned with for each realization a max hazard map.
# But realizations are calculated per rupture, not per set of ruptures, so is this correct?



__all__ = ['DSHAModel']


## Probably obsolete

class DSHAModel(SHAModelBase):
	"""
	Class defining deterministic seismic hazard model.
	Note that a DSHAModel should only consist of point sources or
	characteristic fault sources

	:param name:
		str, model name
	:param source_model:
		instance of :class:`rshalib.source.SourceModel`
	:param gmpe_system_def:
		dict, mapping tectonic region types to GMPEPMF objects
	:param site_model:
		instance of :class:`rshalib.site.GenericSiteModel`
		or :class:`rshalib.site.SoilSiteModel`,
		sites where ground motions will be computed
	:param ref_soil_params:
		dict, value for each soil parameter of :class:`rshalib.site.SoilSite`
		Required if :param:`site_model` is generic, ignored otherwise
		(default: REF_SOIL_PARAMS)
	:param imt_periods:
		dict, mapping IMTs to lists of periods
		(default: {'PGA': [0]})
	:param truncation_level:
		float, number of standard deviations to consider on GMPE uncertainty
		(default: 0 = mean ground motion)
	:param integration_distance:
		float, maximum distance with respect to source to compute ground
		motion
		(default: 200)
	"""
	def __init__(self, name, source_model, gmpe_system_def, site_model,
				ref_soil_params=REF_SOIL_PARAMS, imt_periods={'PGA': [0]},
				truncation_level=0., integration_distance=200.):

		#TODO: gmpe_system_def or ground_motion_model?

		SHAModelBase.__init__(self, name, site_model, ref_soil_params,
							imt_periods, truncation_level, integration_distance)
		self.source_model = source_model
		self.gmpe_system_def = gmpe_system_def
		self._check_sources()

	def _check_sources(self):
		"""
		Make sure that DSHAModel only contains point sources or
		characteristic fault sources
		"""
		from ..source import PointSource, CharacteristicFaultSource

		for src in self.source_model.sources:
			if not isinstance(src, (PointSource, CharacteristicFaultSource)):
				msg = "Source type (%s) of source %s not allowed!"
				msg %= (src.__class__, src.source_id)
				raise Exception(msg)

	def calc_random_gmf(self, num_realizations=1, correlation_model=None,
				np_aggregation="avg", gmpe_aggregation="avg", src_aggregation="max",
				random_seed=None):
		"""
		Compute random ground-motion fields.
		Note: uncertainty range depends on class property `truncation_level`
		Note: uncertainties between different IMTs are not correlated,
			at least not in currently used version of OQ.

		:param num_realizations:
			int, number of random fields to generate
			(default: 1)
		:param correlation_model:
			str or instance of :class:`oqhazlib.correlation.BaseCorrelationModel`:
			spatial correlation model for intra-event residuals.
			For this to work, the GMPEs used should have inter-event and
			intra-event uncertainties defined!
			(default: None)
		:param np_aggregation:
			str, how to aggregate nodal planes for a given point rupture,
			either "avg" (weighted average), "min" (minimum) or "max" (maximum)
			(default: "avg")
		:param gmpe_aggregation:
			str, how to aggregate different GMPEs, either "avg", "min" or "max"
			(default: "avg")
		:param src_aggregation:
			str, how to aggregate different sources, either "sum", "min",
			"max" or "avg"
			or list or array containing weights
			(default: "max")
		:param random_seed:
			int, seed for the random number generator
			(default: None)

		:return:
			instance of :class:`UHSFieldTree`
		"""
		# TODO: should uncertainties for different IMTs be correlated?
		from openquake.hazardlib.calc import ground_motion_fields

		assert self.truncation_level >= 0
		if self.truncation_level > 0:
			np.random.seed(seed=random_seed)

		soil_site_model = self.get_soil_site_model()
		num_sites = len(soil_site_model)
		imt_list = self._get_imts()
		fake_tom = oqhazlib.tom.PoissonTOM(1)

		if self.truncation_level == 0:
			print("Correlation model ignored because truncation_level = 0!")
			correlation_model = None

		if isinstance(correlation_model, str):
			if not "CorrelationModel" in correlation_model:
				correlation_model += "CorrelationModel"
			correlation_model = getattr(oqhazlib.correlation, correlation_model)()

		gmf_shape = (num_sites, num_realizations, len(imt_list))
		GMF = np.zeros(gmf_shape)
		for s, src in enumerate(self.source_model):
			src_gmf = np.zeros(gmf_shape)
			trt = src.tectonic_region_type
			gmpe_pmf = self.gmpe_system_def[trt]
			for (gmpe_name, gmpe_weight) in gmpe_pmf:
				gsim = self._get_gsim(gmpe_name)
				gmpe_gmf = np.zeros(gmf_shape)
				total_rupture_probability = 0
				for r, rup in enumerate(src.iter_ruptures(fake_tom)):
					total_rupture_probability += rup.occurrence_rate
					gmf_dict = ground_motion_fields(rup, soil_site_model, imt_list,
									gsim, self.truncation_level, num_realizations,
									correlation_model, self.rupture_site_filter)

					for imt, rup_gmf in gmf_dict.items():
						k = imt_list.index(imt)
						## Aggregate gmf's corresponding to different nodal planes
						if np_aggregation == "avg":
							gmpe_gmf[:,:,k] += (rup_gmf * rup.occurrence_rate)
						elif np_aggregation == "min":
							if r == 0:
								gmpe_gmf[:,:,k] = rup_gmf
							else:
								gmpe_gmf[:,:,k] = np.minimum(gmpe_gmf[:,:,k], rup_gmf)
						elif np_aggregation == "max":
							gmpe_gmf[:,:,k] = np.maximum(gmpe_gmf[:,:,k], rup_gmf)
						else:
							raise Exception("aggregation:%s not supported for nodal planes!"
											% np_aggregation)

				## Aggregate gmf's corresponding to different GMPEs
				if gmpe_aggregation == "avg":
					assert np.allclose(total_rupture_probability, 1)
					src_gmf += (gmpe_gmf * float(gmpe_weight))
				elif gmpe_aggregation == "min":
					if np.allclose(src_gmf, 0):
						src_gmf += gmpe_pmf
					else:
						src_gmf = np.minimum(src_gmf, gmpe_gmf)
				elif gmpe_aggregation == "max":
					src_gmf = np.maximum(src_gmf, gmpe_gmf)
				else:
					raise Exception("aggregation:%s not supported for GMPEs!"
									% gmpe_aggregation)

			## Aggregate gmf's corresponding to different sources
			if isinstance(src_aggregation, (list, np.ndarray)):
				GMF += (src_gmf * float(src_aggregation[s]))
			elif src_aggregation == "sum":
				GMF += src_gmf
			elif src_aggregation == "min":
				if s == 0:
					GMF += src_gmf
				else:
					GMF = np.minimum(GMF, src_gmf)
			elif src_aggregation == "max":
				GMF = np.maximum(GMF, src_gmf)
			elif src_aggregation == "avg":
				GMF += (src_gmf / float(len(self.source_model)))
			else:
				raise Exception("aggregation:%s not supported for sources!"
								% src_aggregation)

		periods = self._get_periods()
		sites = soil_site_model.get_generic_sites()
		vs30s = soil_site_model.vs30
		branch_names = ["Realization #%d" % (i+1) for i in range(num_realizations)]
		filespecs = ["" for i in range(num_realizations)]
		weights = []
		imt = self.get_imt_families()[0]
		damping = 0 if imt in ('PGA', 'PGV', 'PGD') else 0.05

		return UHSFieldTree(branch_names, weights, sites, periods,
							GMF, intensity_unit="g", imt=imt,
							model_name=self.name, filespecs=filespecs,
							timespan=1, return_period=1,
							damping=damping, vs30s=vs30s)

	def calc_random_gmf_mp(self, num_realizations=1, correlation_model=None,
				np_aggregation="avg", gmpe_aggregation="avg", src_aggregation="max",
				random_seed=None, correlate_imt_uncertainties=False, num_cores=None,
				verbose=False):
		"""
		Compute random ground-motion fields on multiple cores.

		:param num_realizations:
		:param correlation_model:
		:param np_aggregation:
		:param gmpe_aggregation:
		:param src_aggregation:
		:param random_seed:
			see :meth:`calc_random_gmf`
		:param correlate_imt_uncertainties:
			bool, whether or not to correlate uncertainties between
			different IMTs for given rupture and GMPE
			(default: False)
		:param num_cores:
			int, number of CPUs to be used. Actual number of cores used
			may be lower depending on available cores and memory
			(default: None, will determine automatically)
		:param verbose:
			bool, whether or not to print some progress information
			(default: True)

		See :meth:`calc_random_gmf` for remaining parameters

		:return:
			instance of :class:`UHSFieldTree`
		"""
		import random
		from ..calc import mp

		MAX_SINT_32 = (2**31) - 1
		rnd = random.Random()
		rnd.seed(random_seed)

		assert self.truncation_level >= 0

		soil_site_model = self.get_soil_site_model()
		num_sites = len(soil_site_model)
		num_gmpes = max([len(gmpe_pmf) for gmpe_pmf in self.gmpe_system_def.values()])
		imt_list = self._get_imts()
		num_periods = len(imt_list)
		fake_tom = oqhazlib.tom.PoissonTOM(1)

		if self.truncation_level == 0:
			print("Correlation model ignored because truncation_level = 0!")
			correlation_model = None

		if isinstance(correlation_model, str):
			if not "CorrelationModel" in correlation_model:
				correlation_model += "CorrelationModel"
			correlation_model = getattr(oqhazlib.correlation, correlation_model)()

		## Create list with arguments for each job
		job_args = []
		num_ruptures_by_source = []
		for src in self.source_model:
			src_num_ruptures = 0
			trt = src.tectonic_region_type
			gmpe_pmf = self.gmpe_system_def[trt]
			total_rupture_probability = 0
			for r, rup in enumerate(src.iter_ruptures(fake_tom)):
				total_rupture_probability += rup.occurrence_rate
				for g, gmpe_name in enumerate(gmpe_pmf.gmpe_names):
					gsim = self._get_gsim(gmpe_name)
					for k, imt in enumerate(imt_list):
						if k == 0 or not correlate_imt_uncertainties:
							random_seed2 = rnd.randint(0, MAX_SINT_32)
						r = int(np.sum(num_ruptures_by_source) + src_num_ruptures)
						shared_arr_idx = (r, g, k)
						shared_arr_shape = []
						job_args.append([rup, soil_site_model, tuple(imt), gsim,
							self.truncation_level, num_realizations, shared_arr_idx,
							shared_arr_shape, correlation_model, self.integration_distance,
							random_seed2])
				src_num_ruptures += 1
			num_ruptures_by_source.append(src_num_ruptures)
			assert np.allclose(total_rupture_probability, 1)
		tot_num_ruptures = np.sum(num_ruptures_by_source)
		shared_arr_shape = (tot_num_ruptures, num_gmpes, num_sites,
							num_realizations, num_periods)
		shared_arr_len = int(np.prod(shared_arr_shape))

		## Create shared-memory array, and expose it as a numpy array
		shared_gmf_array = mp.multiprocessing.Array('d', shared_arr_len, lock=True)

		## Launch multiprocessing
		if len(job_args) > 0:
			for i in range(len(job_args)):
				job_args[i][7] = shared_arr_shape

			mp.run_parallel(mp.calc_random_gmf, job_args, num_cores,
							shared_arr=shared_gmf_array, verbose=verbose)

		## Compute aggregated ground-motion fields
		gmf_matrix = np.frombuffer(shared_gmf_array.get_obj())
		gmf_matrix = gmf_matrix.reshape(shared_arr_shape)

		gmf_shape = (num_sites, num_realizations, len(imt_list))
		GMF = np.zeros(gmf_shape)
		rup_idx = 0
		for s, src in enumerate(self.source_model):
			src_gmf = np.zeros(gmf_shape)
			trt = src.tectonic_region_type
			gmpe_pmf = self.gmpe_system_def[trt]
			for g, gmpe_weight in enumerate(gmpe_pmf.weights):
				gmpe_gmf = np.zeros(gmf_shape)
				for r in range(num_ruptures_by_source[s]):
					for k in range(num_periods):
						rup_gmf = gmf_matrix[rup_idx, g, :, :, k]
						## Aggregate gmf's corresponding to different nodal planes
						if np_aggregation == "avg":
							gmpe_gmf[:,:,k] += (rup_gmf * rup.occurrence_rate)
						elif np_aggregation == "min":
							if r == 0:
								gmpe_gmf[:,:,k] = rup_gmf
							else:
								gmpe_gmf[:,:,k] = np.minimum(gmpe_gmf[:,:,k], rup_gmf)
						elif np_aggregation == "max":
							gmpe_gmf[:,:,k] = np.maximum(gmpe_gmf[:,:,k], rup_gmf)
						else:
							raise Exception("aggregation:%s not supported for nodal planes!"
											% np_aggregation)
					rup_idx += 1

				## Aggregate gmf's corresponding to different GMPEs
				if gmpe_aggregation == "avg":
					src_gmf += (gmpe_gmf * float(gmpe_weight))
				elif gmpe_aggregation == "min":
					if np.allclose(src_gmf, 0):
						src_gmf += gmpe_pmf
					else:
						src_gmf = np.minimum(src_gmf, gmpe_gmf)
				elif gmpe_aggregation == "max":
					src_gmf = np.maximum(src_gmf, gmpe_gmf)
				else:
					raise Exception("aggregation:%s not supported for GMPEs!"
									% gmpe_aggregation)

			## Aggregate gmf's corresponding to different sources
			if isinstance(src_aggregation, (list, np.ndarray)):
				GMF += (src_gmf * float(src_aggregation[s]))
			elif src_aggregation == "sum":
				GMF += src_gmf
			elif src_aggregation == "min":
				if s == 0:
					GMF += src_gmf
				else:
					GMF = np.minimum(GMF, src_gmf)
			elif src_aggregation == "max":
				GMF = np.maximum(GMF, src_gmf)
			elif src_aggregation == "avg":
				GMF += (src_gmf / float(len(self.source_model)))
			else:
				raise Exception("aggregation:%s not supported for sources!"
								% src_aggregation)

		periods = self._get_periods()
		sites = soil_site_model.get_generic_sites()
		vs30s = soil_site_model.vs30
		branch_names = ["Realization #%d" % (i+1) for i in range(num_realizations)]
		filespecs = ["" for i in range(num_realizations)]
		weights = []
		imt = self.get_imt_families()[0]
		damping = 0 if imt in ('PGA', 'PGV', 'PGD') else 0.05

		return UHSFieldTree(branch_names, weights, sites, periods,
							GMF, intensity_unit="g", imt=imt,
							model_name=self.name, filespecs=filespecs,
							timespan=1, return_period=1,
							damping=damping, vs30s=vs30s)

	def calc_gmf_fixed_epsilon(self, stddev_type="total", np_aggregation="avg",
								gmpe_aggregation="avg", src_aggregation="max"):
		"""
		Compute ground-motion field considering a fixed epsilon of the
		GMPE uncertainty.
		Note: epsilon value corresponds to class property `truncation_level`

		:param stddev_type:
			str, standard deviation type, one of "total", "inter-event"
			or "intra-event"
			Note: GMPE must support inter-event / intra-event standard
			deviations if one of these options is chosen.
			(default: "total")
		:param np_aggregation:
			str, how to aggregate nodal planes for a given point rupture,
			either "avg" (weighted average) or "max" (maximum)
			(default: "avg")
		:param gmpe_aggregation:
			str, how to aggregate different GMPEs, either "avg", "min" or "max"
			(default: "avg")
		:param src_aggregation:
			str, how to aggregate different sources, either "sum", "min",
			"max" or "avg"
			or list or array containing weights
			(default: "max")

		:return:
			instance of :class:`UHSField`
		"""
		from ..calc import mp

		soil_site_model = self.get_soil_site_model()
		num_sites = len(soil_site_model)
		imt_list = self._get_imts()
		fake_tom = oqhazlib.tom.PoissonTOM(1)
		if stddev_type == "total":
			total_residual_epsilons = np.ones(num_sites)
			total_residual_epsilons *= self.truncation_level
			inter_residual_epsilons = total_residual_epsilons
			intra_residual_epsilons = total_residual_epsilons
		elif stddev_type == "inter-event":
			total_residual_epsilons = None
			inter_residual_epsilons = np.ones(num_sites)
			inter_residual_epsilons *= self.truncation_level
			intra_residual_epsilons = np.ones(num_sites)
		elif stddev_type == "intra-event":
			total_residual_epsilons = None
			inter_residual_epsilons = np.ones(num_sites)
			intra_residual_epsilons = np.ones(num_sites)
			intra_residual_epsilons *= self.truncation_level

		gmf_envelope = np.zeros((num_sites, len(imt_list)))
		amax = 0

		for k, imt in enumerate(imt_list):
			for s, src in enumerate(self.source_model):
				src_gmf = np.zeros(len(soil_site_model))
				trt = src.tectonic_region_type
				gmpe_pmf = self.gmpe_system_def[trt]
				for (gmpe_name, gmpe_weight) in gmpe_pmf:
					gsim = self._get_gsim(gmpe_name)
					gmpe_gmf = np.zeros(len(soil_site_model))
					total_rupture_probability = 0
					for r, rup in enumerate(src.iter_ruptures(fake_tom)):
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
						elif np_aggregation == "min":
							if r == 0:
								gmpe_gmf += gmf
							else:
								gmpe_gmf = np.minimum(gmpe_gmf, gmf)
						elif np_aggregation == "max":
							gmpe_gmf = np.maximum(gmpe_gmf, gmf)
						else:
							raise Exception("aggregation:%s not supported for nodal planes!"
											% np_aggregation)

					## Aggregate gmf's corresponding to different GMPEs
					if gmpe_aggregation == "avg":
						assert np.allclose(total_rupture_probability, 1)
						src_gmf += (gmpe_gmf * float(gmpe_weight))
					elif gmpe_aggregation == "min":
						if np.allclose(src_gmf, 0):
							src_gmf += gmpe_gmf
						else:
							src_gmf = np.minimum(src_gmf, gmpe_gmf)
					elif gmpe_aggregation == "max":
						src_gmf = np.maximum(src_gmf, gmpe_gmf)
					else:
						raise Exception("aggregation:%s not supported for GMPEs!"
										% gmpe_aggregation)

				#amax = max(amax, src_gmf.max())

				## Aggregate gmf's corresponding to different sources
				if isinstance(src_aggregation, (list, np.ndarray)):
					gmf_envelope[:,k] += (src_gmf * float(src_aggregation[s]))
				elif src_aggregation == "sum":
					gmf_envelope[:,k] += src_gmf
				elif src_aggregation == "min":
					if s == 0:
						gmf_envelope[:,k] += src_gmf
					else:
						gmf_envelope[:,k] = np.minimum(gmf_envelope[:,k], src_gmf)
				elif src_aggregation == "max":
					gmf_envelope[:,k] = np.maximum(gmf_envelope[:,k], src_gmf)
				elif src_aggregation == "avg":
					gmf_envelope[:,k] += (src_gmf / float(len(self.source_model)))
				else:
					raise Exception("aggregation:%s not supported for sources!"
									% src_aggregation)

		periods = self._get_periods()
		sites = soil_site_model.get_generic_sites()
		vs30s = soil_site_model.vs30
		imt = self.get_imt_families()[0]
		damping = 0 if imt in ('PGA', 'PGV', 'PGD') else 0.05

		return UHSField(sites, periods, gmf_envelope, intensity_unit="g", imt=imt,
						model_name=self.name, filespec="",
						timespan=1, return_period=1,
						damping=damping, vs30s=vs30s)

	def calc_gmf_fixed_epsilon_mp(self, stddev_type="total", np_aggregation="avg",
								gmpe_aggregation="avg", src_aggregation="max",
								num_cores=None, verbose=False):
		"""
		Compute ground-motion field using fixed epsilon on multiple cores

		:param stddev_type:
		:param np_aggregation:
		:param gmpe_aggregation:
		:param src_aggregation:
			see :meth:`calc_gmf_fixed_epsilon`
		:param num_cores:
			int, number of CPUs to be used. Actual number of cores used
			may be lower depending on available cores and memory
			(default: None, will determine automatically)
		:param verbose:
			bool, whether or not to print some progress information
			(default: True)

		See :meth:`calc_gmf_fixed_epsilon` for remaining parameters

		:return:
			instance of :class:`UHSField`
		"""
		soil_site_model = self.get_soil_site_model()
		num_sites = len(soil_site_model)
		imt_list = self._get_imts()
		fake_tom = oqhazlib.tom.PoissonTOM(1)
		if stddev_type == "total":
			total_residual_epsilons = np.ones(num_sites)
			total_residual_epsilons *= self.truncation_level
			inter_residual_epsilons = total_residual_epsilons
			intra_residual_epsilons = total_residual_epsilons
		elif stddev_type == "inter-event":
			total_residual_epsilons = None
			inter_residual_epsilons = np.ones(num_sites)
			inter_residual_epsilons *= self.truncation_level
			intra_residual_epsilons = np.ones(num_sites)
		elif stddev_type == "intra-event":
			total_residual_epsilons = None
			inter_residual_epsilons = np.ones(num_sites)
			intra_residual_epsilons = np.ones(num_sites)
			intra_residual_epsilons *= self.truncation_level

		gmf_envelope = np.zeros((num_sites, len(imt_list)))

		## Create list with arguments for each job
		job_args = []
		for k, imt in enumerate(imt_list):
			for src in self.source_model:
				trt = src.tectonic_region_type
				gmpe_pmf = self.gmpe_system_def[trt]
				for (gmpe_name, gmpe_weight) in gmpe_pmf:
					gsim = self._get_gsim(gmpe_name)
					gmpe_gmf = np.zeros(len(soil_site_model))
					for rup in src.iter_ruptures(fake_tom):
						# TODO: check if CharacteristicFaultSources return only 1 rupture
						job_args.append((rup, soil_site_model, tuple(imt), gsim,
										self.truncation_level,
										total_residual_epsilons, intra_residual_epsilons,
										inter_residual_epsilons, self.integration_distance))

		## Launch multiprocessing
		if len(job_args) > 0:
			gmf_list = mp.run_parallel(mp.calc_gmf_with_fixed_epsilon, job_args,
										num_cores, verbose=verbose)

		i = 0
		for k, imt in enumerate(imt_list):
			for s, src in enumerate(self.source_model):
				src_gmf = np.zeros(len(soil_site_model))
				trt = src.tectonic_region_type
				gmpe_pmf = self.gmpe_system_def[trt]
				for (gmpe_name, gmpe_weight) in gmpe_pmf:
					gsim = self._get_gsim(gmpe_name)
					gmpe_gmf = np.zeros(len(soil_site_model))
					total_rupture_probability = 0
					for r, rup in enumerate(src.iter_ruptures(fake_tom)):
						total_rupture_probability += rup.occurrence_rate
						gmf = gmf_list[i]
						i += 1

						## Aggregate gmf's corresponding to different nodal planes
						if np_aggregation == "avg":
							gmpe_gmf += (gmf * rup.occurrence_rate)
						elif np_aggregation == "min":
							if r == 0:
								gmpe_gmf += gmf
							else:
								gmpe_gmf = np.minimum(gmpe_gmf, gmf)
						elif np_aggregation == "max":
							gmpe_gmf = np.maximum(gmpe_gmf, gmf)
						else:
							raise Exception("aggregation:%s not supported for nodal planes!"
											% np_aggregation)

					## Aggregate gmf's corresponding to different GMPEs
					if gmpe_aggregation == "avg":
						assert np.allclose(total_rupture_probability, 1)
						src_gmf += (gmpe_gmf * float(gmpe_weight))
					elif gmpe_aggregation == "min":
						if np.allclose(src_gmf, 0):
							src_gmf += gmpe_gmf
						else:
							src_gmf = np.minimum(src_gmf, gmpe_gmf)
					elif gmpe_aggregation == "max":
						src_gmf = np.maximum(src_gmf, gmpe_gmf)
					else:
						raise Exception("aggregation:%s not supported for GMPEs!"
										% gmpe_aggregation)

				## Aggregate gmf's corresponding to different sources
				if isinstance(src_aggregation, (list, np.ndarray)):
					gmf_envelope[:,k] += (src_gmf * float(src_aggregation[s]))
				elif src_aggregation == "sum":
					gmf_envelope[:,k] += src_gmf
				elif src_aggregation == "min":
					if s == 0:
						gmf_envelope[:,k] += src_gmf
					else:
						gmf_envelope[:,k] = np.minimum(gmf_envelope[:,k], src_gmf)
				elif src_aggregation == "max":
					gmf_envelope[:,k] = np.maximum(gmf_envelope[:,k], src_gmf)
				elif src_aggregation == "avg":
					gmf_envelope[:,k] += (src_gmf / float(len(self.source_model)))
				else:
					raise Exception("aggregation:%s not supported for sources!"
									% src_aggregation)

		periods = self._get_periods()
		sites = soil_site_model.get_generic_sites()
		vs30s = soil_site_model.vs30
		imt = self.get_imt_families()[0]
		damping = 0 if imt in ('PGA', 'PGV', 'PGD') else 0.05

		return UHSField(sites, periods, gmf_envelope, intensity_unit="g", imt=imt,
						model_name=self.name, filespec="",
						timespan=1, return_period=1,
						damping=damping, vs30s=vs30s)


class RuptureDSHAModel(SHAModelBase):
	"""
	Class representing a single DSHA model.

	:param name:
		str, name for DSHA model
	:param ruptures:
		list of instances of :class:`rshalib.source.Rupture`
	:param gsim_name:
		str, name of supported GMPE
	:param site_model:
		instance of :class:`rshalib.site.GenericSiteModel`
		or :class:`rshalib.site.SoilSiteModel`,
		sites where ground motions will be computed
	:param ref_soil_params:
		dict, value for each soil parameter of :class:`rshalib.site.SoilSite`
		Required if :param:`site_model` is generic, ignored otherwise
		(default: REF_SOIL_PARAMS)
	:param imt_periods:
		see :class:`..shamodel.SHAModelBase`
		(default: {'PGA': [0]})
	:param truncation_level:
		see :class:`..shamodel.SHAModelBase`
		(default: 0.)
	:param realizations:
		int, number of realizations
		(default: 1)
	:param correlation_model:
		instance of subclass of :class:`openquake.hazardlib.correlation.BaseCorrelationModel`
		(default: None)
	:param integration_distance:
		see :class:`..shamodel.SHAModelBase`
		(default: 200.)
	"""
	def __init__(self, name, ruptures, gsim_name,
				site_model, ref_soil_params=REF_SOIL_PARAMS,
				imt_periods={'PGA': [0]},
				truncation_level=0., realizations=1.,
				correlation_model=None, integration_distance=200.):
		"""
		"""
		SHAModelBase.__init__(self, name, site_model, ref_soil_params,
							imt_periods, truncation_level, integration_distance)

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
			function, filter for sites further away from rupture than
			integration_distance
		"""
		from openquake.hazardlib.calc.filters import (rupture_site_distance_filter,
														rupture_site_noop_filter)

		if self.integration_distance:
			return rupture_site_distance_filter(self.integration_distance)
		else:
			return rupture_site_noop_filter

	def _get_gsim(self):
		"""
		Fetch gsim

		:return:
			instance of :class:`openquake.hazardlib.gsim.GroundShakingIntensityModel
		"""
		return super(RuptureDSHAModel, self)._get_gsim(self.gsim_name)

	def run_hazardlib(self):
		"""
		:returns:
			instance of :class:`rshalib.result.HazardMapSet`
		"""
		from openquake.hazardlib.calc import ground_motion_fields

		soil_site_model = self.get_soil_site_model()
		gsim = self._get_gsim()
		imts = self._get_imts()
		rsdf = self.rupture_site_filter
		array_shape = (len(soil_site_model), len(self.ruptures), self.realizations)
		intensities = {imt: np.zeros(array_shape) for imt in imts}
		for i, rupture in enumerate(self.ruptures):
			gmfs = ground_motion_fields(rupture, soil_site_model, imts, gsim,
										self.truncation_level, self.realizations,
										self.correlation_model, rsdf)
			for imt, gmf in gmfs.items():
				if self.correlation_model:
					intensities[imt][:,i,:] = gmf.getA()
				else:
					intensities[imt][:,i,:] = gmf

		hazard_map_sets = {}
		sites = self.get_generic_sites()
		return_periods = np.ones(self.realizations)
		model_name = ""
		filespecs = [""]*self.realizations
		intensity_unit = "g"
		for imt in imts:
			if isinstance(imt, SA):
				period = imt.period
				IMT = "SA"
				key = period
				damping = imt.damping / 100.
			else:
				period = 0
				# TODO: need a more elegant solution
				IMT = {PGV(): "PGV", PGD(): "PGD", PGA(): "PGA", MMI(): "MMI"}[imt]
				key = IMT
				damping = 0.
			hm_intensities = np.amax(intensities[imt], axis=1).T
			hms = HazardMapSet(sites, period, hms_intensities, intensity_unit, IMT,
								model_name=model_name, filespecs=filespecs,
								timespan=1, return_periods=return_periods,
								damping=damping, vs30s=None)
			hazard_map_sets[key] = hms
		return hazard_map_sets



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
	print(hazard_map_sets["PGA"].get_hazard_map(0))
	print(hazard_map_sets["PGA"].get_hazard_map(1))

