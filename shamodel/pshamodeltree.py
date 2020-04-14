"""
PSHAModelTree class
"""

from __future__ import absolute_import, division, print_function, unicode_literals


### imports
import os
import copy
import random

import numpy as np

from .. import oqhazlib
from openquake.hazardlib.imt import PGA, SA, PGV, PGD, MMI

from ..calc import mp
from ..geo import *
from ..site import *
from ..result import (SpectralHazardCurveField, SpectralHazardCurveFieldTree,
						SpectralDeaggregationCurve)
from ..source import SourceModel
from ..gsim import GroundMotionModel
from .pshamodelbase import PSHAModelBase
from .pshamodel import PSHAModel


## Minimum and maximum values for random number generator
MIN_SINT_32 = -(2**31)
MAX_SINT_32 = (2**31) - 1



__all__ = ['PSHAModelTree']


class PSHAModelTree(PSHAModelBase):
	"""
	Class representing a PSHA model logic tree.

	:param name:
		str, model name
	:param source_model_lt:
		instance of :class:`rshalib.logictree.SeismicSourceSystem`,
		defining source model logic tree.
	:param gmpe_lt:
		instance of :class:`rshalib.logictree.GroundMotionSystem`,
		defining ground-motionlogic tree.
	:param root_folder:
	:param site_model:
	:param ref_soil_params:
	:param imt_periods:
	:param intensities:
	:param min_intensities:
	:param max_intensities:
	:param num_intensities:
	:param return_periods:
	:param time_span:
	:param truncation_level:
	:param integration_distance:
	:param damping:
	:param intensity_unit:
		see :class:`PSHAModelBase`
	:param num_lt_samples:
		int, defining times to sample logic trees
		(default: 1).
	:param random_seed:
		int, seed for random number generator
		(default: 42)
	"""
	def __init__(self, name, source_model_lt, gmpe_lt, root_folder,
				site_model, ref_soil_params=REF_SOIL_PARAMS,
				imt_periods={'PGA': [0]}, intensities=None,
				min_intensities=0.001, max_intensities=1., num_intensities=100,
				return_periods=[], time_span=50.,
				truncation_level=3., integration_distance=200.,
				damping=0.05, intensity_unit=None,
				num_lt_samples=1, random_seed=42):
		"""
		"""
		from openquake.engine.input.logictree import LogicTreeProcessor
		PSHAModelBase.__init__(self, name, root_folder, site_model, ref_soil_params,
								imt_periods, intensities, min_intensities,
								max_intensities, num_intensities, return_periods,
								time_span, truncation_level, integration_distance,
								damping, intensity_unit)
		self.source_model_lt = source_model_lt
		self.gmpe_lt = gmpe_lt.get_optimized_system(self.source_models)
		self.num_lt_samples = num_lt_samples
		#self.lts_sampling_method = lts_sampling_method
		#if self.lts_sampling_method == 'enumerated':
		#	self.enumerated_lts_samples = self._enumerate_lts_samples()
		self.random_seed = random_seed
		self.ltp = LogicTreeProcessor(self.source_model_lt, self.gmpe_lt)
		self._init_rnd()

	def __repr__(self):
		txt = '<PSHAModelTree "%s">' % self.name

	@property
	def source_models(self):
		return self.source_model_lt.source_models

	def get_source_model_by_name(self, source_model_name):
		"""
		Get source model by name

		:param source_model_name:
			str, name of source model

		:return:
			instance of :class:`rshalib.source.SourceModel`
		"""
		return self.source_model_lt.get_source_model_by_name(source_model_name)

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
		num_gmpelt_paths = self.gmpe_lt.get_num_paths()
		return num_smlt_paths * num_gmpelt_paths

	def sample_logic_tree_paths(self, num_samples, enumerate_gmpe_lt=False,
								skip_samples=0):
		"""
		Sample paths from logic tree

		:param num_samples:
			int, number of random samples
			If zero, :meth:`enumerate_logic_tree_paths` will be called
			(default: None, will use num_lt_samples)
		:param enumerate_gmpe_lt:
			bool, whether or not to enumerate the GMPE logic tree
			(default: False)
		:param skip_samples:
			int, number of samples to skip
			(default: 0)

		:return:
			list of (source-model name, source-model logic tree path,
			ground-motion logic tree path, weight) tuples
		"""
		if num_samples is None:
			num_samples = self.num_lt_samples

		if num_samples == 0:
			return self.enumerate_logic_tree_paths()

		lt_paths_weights = []

		if enumerate_gmpe_lt:
			gmpelt_paths_weights = self.enumerate_gmpe_lt_paths()

		for i in range(num_samples + skip_samples):
			## Generate 2nd-order random seeds
			smlt_random_seed = self.rnd.randint(MIN_SINT_32, MAX_SINT_32)
			gmpelt_random_seed = self.rnd.randint(MIN_SINT_32, MAX_SINT_32)

			## Call OQ logictree processor
			sm_name, smlt_path = self.ltp.sample_source_model_logictree(smlt_random_seed)
			gmpelt_path = self.ltp.sample_gmpe_logictree(gmpelt_random_seed)

			if i >= skip_samples:
				if not enumerate_gmpe_lt:
					gmpelt_paths_weights = [(gmpelt_path, 1.)]

				for gmpelt_path, gmpelt_weight in gmpelt_paths_weights:
					weight = gmpelt_weight / num_samples
					lt_paths_weights.append((sm_name, smlt_path, gmpelt_path, weight))

			## Update the seed for the next realization
			seed = self.rnd.randint(MIN_SINT_32, MAX_SINT_32)
			self.rnd.seed(seed)

		return lt_paths_weights

	def sample_logic_trees(self, num_samples=None, enumerate_gmpe_lt=False,
							skip_samples=0, verbose=False):
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
		:param skip_samples:
			int, number of samples to skip
			(default: 0)
		:param verbose:
			bool, whether or not to print some information
			(default: False)

		:return:
			list with (instance of :class:`PSHAModel`, weight) tuples
		"""
		psha_models_weights = []

		for i, (sm_name, smlt_path, gmpelt_path, weight) in \
			enumerate(self.sample_logic_tree_paths(num_samples,
											enumerate_gmpe_lt=enumerate_gmpe_lt,
											skip_samples=skip_samples)):
			## Convert to objects
			source_model = self._smlt_sample_to_source_model(sm_name, smlt_path,
															verbose=verbose)
			gmpe_model = self._gmpe_sample_to_gmpe_model(gmpelt_path)
			## Convert to PSHA model
			sample_num = i + skip_samples + 1
			name = "%s, LT sample %04d (SM_LTP: %s; GMPE_LTP: %s)"
			name %= (self.name, sample_num, "--".join(smlt_path),
					"--".join(gmpelt_path))
			psha_model = self._get_psha_model(source_model, gmpe_model, name)
			psha_models_weights.append((psha_model, weight))

		return psha_models_weights

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

		for i in range(num_samples + skip_samples):
			## Generate 2nd-order random seeds
			smlt_random_seed = self.rnd.randint(MIN_SINT_32, MAX_SINT_32)
			gmpelt_random_seed = self.rnd.randint(MIN_SINT_32, MAX_SINT_32)

			## Call OQ logictree processor
			sm_name, smlt_path = self.ltp.sample_source_model_logictree(smlt_random_seed)
			gmpelt_path = self.ltp.sample_gmpe_logictree(gmpelt_random_seed)

			if i >= skip_samples:
				## Convert to objects
				source_model = self._smlt_sample_to_source_model(sm_name, smlt_path, verbose=verbose)
				if not enumerate_gmpe_lt:
					gmpe_models_weights = [(self._gmpe_sample_to_gmpe_model(gmpelt_path), 1.)]
					gmpelt_paths = [gmpelt_path]

				for (gmpe_model, gmpelt_weight), gmpelt_path in zip(gmpe_models_weights, gmpelt_paths):
					## Convert to PSHA model
					name = "%s, LT sample %04d (SM_LTP: %s; GMPE_LTP: %s)" % (self.name, i+1, "--".join(smlt_path), "--".join(gmpelt_path))
					psha_model = self._get_psha_model(source_model, gmpe_model, name)
					psha_models_weights.append((psha_model, gmpelt_weight/num_samples))
					#yield (psha_model, gmpelt_weight)

			## Update the seed for the next realization
			seed = self.rnd.randint(MIN_SINT_32, MAX_SINT_32)
			self.rnd.seed(seed)

		# TODO: use yield instead?
		return psha_models_weights
		"""

	def enumerate_logic_trees(self, verbose=False):
		"""
		Enumerate both source-model and GMPE logic trees, in a way that is
		similar to :meth:`_initialize_realizations_enumeration` of
		:class:`BaseHazardCalculator` in oq-engine

		:param verbose:
			bool, whether or not to print some information
			(default: False)

		:return:
			list with (instance of :class:`PSHAModel`, weight) tuples
		"""
		psha_models_weights = []
		for i, path_info in enumerate(self.ltp.enumerate_paths()):
			sm_name, weight, smlt_path, gmpelt_path = path_info
			source_model = self._smlt_sample_to_source_model(sm_name, smlt_path,
															verbose=verbose)
			gmpe_model = self._gmpe_sample_to_gmpe_model(gmpelt_path)
			name = "%s, LT enum %04d (SM_LTP: %s; GMPE_LTP: %s)"
			name %= (self.name, i, source_model.description, gmpe_model.name)
			psha_model = self._get_psha_model(source_model, gmpe_model, name)
			psha_models_weights.append((psha_model, weight))
			#yield (psha_model, weight)
		return psha_models_weights

	def enumerate_logic_tree_paths(self):
		"""
		Enumerate all paths in logic tree

		:return:
			list of (source-model name, source-model logic tree path,
			ground-motion logic tree path, weight) tuples
		"""
		lt_paths_weights = []
		for i, path_info in enumerate(self.ltp.enumerate_paths()):
			sm_name, weight, smlt_path, gmpelt_path = path_info
			lt_paths_weights.append((sm_name, smlt_path, gmpelt_path, weight))
		return lt_paths_weights

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
			self.site_model, ref_soil_params=self.ref_soil_params,
			imt_periods=self.imt_periods, intensities=self.intensities,
			min_intensities=self.min_intensities, max_intensities=self.max_intensities,
			num_intensities=self.num_intensities, return_periods=self.return_periods,
			time_span=self.time_span, truncation_level=self.truncation_level,
			integration_distance=self.integration_distance)
		return psha_model

	def sample_source_model_lt_paths(self, num_samples=1):
		"""
		Sample source-model logic-tree paths

		:param num_samples:
			int, number of random samples.
			In contrast to :meth:`sample_source_model_lt`, no enumeration
			occurs if num_samples is zero!
			(default: 1)

		:return:
			generator object yielding
			(source_model_name, branch_path, weight) tuple
		"""
		for i in range(num_samples):
			## Generate 2nd-order random seed
			random_seed = self.rnd.randint(MIN_SINT_32, MAX_SINT_32)
			## Call OQ logictree processor
			sm_name, path = self.ltp.sample_source_model_logictree(random_seed)
			weight = 1./num_samples
			yield (sm_name, path, weight)

	def sample_source_model_lt(self, num_samples=1, verbose=False, show_plot=False):
		"""
		Sample source-model logic tree

		:param num_samples:
			int, number of random samples.
			If zero, :meth:`enumerate_source_model_lt` will be called
			(default: 1)
		:param verbose:
			bool, whether or not to print some information
			(default: False)
		:param show_plot:
			bool, whether or not to plot a diagram of the sampled branch path
			(default: False)

		:return:
			list with (instance of :class:`SourceModel`, weight) tuples
		"""
		if num_samples == 0:
			return self.enumerate_source_model_lt(verbose=verbose, show_plot=show_plot)

		modified_source_models_weights = []
		for (sm_name, path, weight) in self.sample_source_model_lt_paths(num_samples):
			if verbose:
				print(sm_name, path)
			if show_plot:
				self.source_model_lt.plot_diagram(highlight_path=path)
			## Apply uncertainties
			source_model = self._smlt_sample_to_source_model(sm_name, path,
															verbose=verbose)
			modified_source_models_weights.append((source_model, weight))
			#yield (source_model, weight)
		return modified_source_models_weights

	def enumerate_source_model_lt_paths(self):
		"""
		Enumerate source-model logic-tree paths

		:return:
			generator object yielding
			(source_model_name, branch_path, weight) tuple
		"""
		for weight, smlt_branches in self.source_model_lt.root_branchset.enumerate_paths():
			smlt_path = [branch.branch_id for branch in smlt_branches]
			sm_name = os.path.splitext(smlt_branches[0].value)[0]
			yield (sm_name, smlt_path, weight)

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
		for (sm_name, smlt_path, weight) in self.enumerate_source_model_lt_paths():
			if verbose:
				print(smlt_path_weight, sm_name, smlt_path)
			if show_plot:
				self.source_model_lt.plot_diagram(highlight_path=smlt_path)
			## Apply uncertainties
			source_model = self._smlt_sample_to_source_model(sm_name, smlt_path,
															verbose=verbose)
			modified_source_models_weights.append((source_model, weight))
			#yield (source_model, weight)
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
			bool, whether or not to print some information
			(default: False)

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
					modified_src.mfd = src.mfd.copy()
					apply_uncertainties = self.ltp.parse_source_model_logictree_path(path)
					apply_uncertainties(modified_src)
					if verbose:
						print("  %s" % src.source_id)
						if hasattr(src.mfd, 'a_val'):
							print("    %.2f %.3f %.3f  -->  %.2f %.3f %.3f"
								% (src.mfd.max_mag, src.mfd.a_val, src.mfd.b_val,
								modified_src.mfd.max_mag, modified_src.mfd.a_val,
								modified_src.mfd.b_val))
						elif hasattr(src.mfd, 'occurrence_rates'):
							print("    %s  -->  %s"
								% (src.mfd.occurrence_rates,
								modified_src.mfd.occurrence_rates))
					modified_sources.append(modified_src)
				break
		description = "--".join(path)
		return SourceModel(sm.name, modified_sources, description)

	def sample_gmpe_lt_paths(self, num_samples=1):
		"""
		Sample GMPE logic-tree paths

		:param num_samples:
			int, number of random samples.
			In contrast to :meth:`sample_gmpe_lt`, no enumeration
			occurs if num_samples is zero!
			(default: 1)

		:return:
			generator object yielding (branch_path, weight) tuple
		"""
		for i in range(num_samples):
			## Generate 2nd-order random seed
			random_seed = self.rnd.randint(MIN_SINT_32, MAX_SINT_32)
			## Call OQ logictree processor
			gmpe_lt_path = self.ltp.sample_gmpe_logictree(random_seed)
			weight = 1./num_samples
			yield (gmpe_lt_path, weight)

	def sample_gmpe_lt(self, num_samples=1, verbose=False, show_plot=False):
		"""
		Sample GMPE logic tree

		:param num_samples:
			int, number of random samples
			If zero, :meth:`enumerate_gmpe_lt` will be called
			(default: 1)
		:param verbose:
			bool, whether or not to print some information
			(default: False)
		:param show_plot:
			bool, whether or not to plot a diagram of the sampled branch path
			(default: False)

		:return:
			list with (instance of :class:`GroundMotionModel`, weight) tuples
		"""
		if num_samples == 0:
			return self.enumerate_gmpe_lt(verbose=verbose, show_plot=show_plot)

		gmpe_models_weights = []
		for gmpe_lt_path, weight in self.sample_gmpe_lt_paths(num_samples):
			if verbose:
				print(gmpe_lt_path)
			if show_plot:
				self.gmpe_lt.plot_diagram(highlight_path=gmpe_lt_path)
			## Convert to GMPE model
			gmpe_model = self._gmpe_sample_to_gmpe_model(gmpe_lt_path)
			gmpe_models_weights.append((gmpe_model, weight))
			if verbose:
				print(gmpe_model)
			#yield (gmpe_model, weight)
		return gmpe_models_weights

	def enumerate_gmpe_lt_paths(self):
		"""
		Enumerate GMPE logic-tree paths

		:return:
			generator object yielding (branch_path, weight) tuple
		"""
		for weight, gmpelt_branches in self.gmpe_lt.root_branchset.enumerate_paths():
			gmpelt_path = [branch.branch_id for branch in gmpelt_branches]
			yield (gmpelt_path, weight)

	def enumerate_gmpe_lt(self, verbose=False, show_plot=False):
		"""
		Enumerate GMPE logic tree

		:param verbose:
			bool, whether or not to print some information
			(default: False)
		:param show_plot:
			bool, whether or not to plot a diagram of the sampled branch path
			(default: False)

		:return:
			list with (instance of :class:`GroundMotionModel`, weight) tuples
		"""
		gmpe_models_weights = []
		for (gmpelt_path, gmpelt_path_weight) in self.enumerate_gmpe_lt_paths(num_samples):
			if verbose:
				print(gmpelt_path_weight, gmpelt_path)
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
		trts = self.gmpe_lt.tectonic_region_types
		trt_gmpe_dict = {}
		for l, branch_id in enumerate(path):
			branch = self.gmpe_lt.get_branch_by_id(branch_id)
			trt = trts[l]
			trt_gmpe_dict[trt] = branch.value
		name = "--".join(path)
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
				print("Warning: rupture mesh spacing of src %s different "
					"from that of 1st source!" % src.source_id)
			if src.mfd.bin_width != mfd_bin_width:
				print("Warning: mfd bin width of src %s different "
					"from that of 1st source!" % src.source_id)

		area_sources = []
		for sm in self.source_models:
			area_sources.extend(sm.get_area_sources())
		if len(area_sources) > 0:
			area_source_discretization = area_sources[0].area_discretization
			for src in area_sources[1:]:
				if src.area_discretization != area_source_discretization:
					print("Warning: area discretization of src %s different "
						"from that of 1st source!" % src.source_id)
		else:
			area_source_discretization = 5.

		params = {}
		params['rupture_mesh_spacing'] = rupture_mesh_spacing
		params['width_of_mfd_bin'] = mfd_bin_width
		params['area_source_discretization'] = area_source_discretization

		return params

	def write_openquake(self, calculation_mode='classical', user_params=None,
						calc_id=None):
		"""
		Write PSHA model tree input for OpenQuake.

		:param calculation_mode:
			str, calculation mode of OpenQuake (options: "classical" or
				"disaggregation")
			(default: "classical")
		:param user_params:
			{str, val} dict, defining respectively parameters and value
			for OpenQuake
			(default: None).
		:param calc_id:
			str, calculation ID correspoding to subfolder where xml files
			will be written.
			(default: None)
		"""
		from ..poisson import poisson_conv
		from ..openquake import OQ_Params

		if not os.path.exists(self.oq_root_folder):
			os.mkdir(self.oq_root_folder)

		if calc_id:
			oq_folder = os.path.join(self.oq_root_folder, "calc_%s" % calc_id)
		else:
			oq_folder = self.oq_root_folder

		if not os.path.exists(oq_folder):
			os.mkdir(oq_folder)

		## set OQ_params object and override with params from user_params
		params = OQ_Params(calculation_mode=calculation_mode, description=self.name)
		implicit_params = self._get_implicit_openquake_params()
		for key in implicit_params:
			setattr(params, key, implicit_params[key])
		if user_params:
			for key in user_params:
				setattr(params, key, user_params[key])

		## set sites or grid_outline
		if (isinstance(self.site_model, GenericSiteModel)
			and self.site_model.grid_outline):
			grid_spacing_km = self.site_model._get_grid_spacing_km()
			params.set_grid_or_sites(grid_outline=self.site_model.grid_outline,
									grid_spacing=grid_spacing_km)
		else:
			params.set_grid_or_sites(sites=self.get_sites())

		## write nrml files for source models
		for source_model in self.source_models:
			## make sure source id's are unique among source models
			## This is no longer necessary
			#for source in source_model.sources:
			#	source.source_id = source_model.name + '--' + source.source_id
			source_model.write_xml(os.path.join(oq_folder, source_model.name + '.xml'))

		## write nrml file for soil site model if present and set file param
		## or set ref soil params
		self._handle_oq_soil_params(params, calc_id=calc_id)

		## validate source model logic tree and write nrml file
		self.source_model_lt.validate()
		source_model_lt_file_name = 'source_model_lt.xml'
		self.source_model_lt.write_xml(os.path.join(oq_folder, source_model_lt_file_name))
		params.source_model_logic_tree_file = source_model_lt_file_name

		## create ground motion model logic tree and write nrml file
		ground_motion_model_lt_file_name = 'ground_motion_model_lt.xml'
		self.gmpe_lt.write_xml(os.path.join(oq_folder, ground_motion_model_lt_file_name))
		params.gsim_logic_tree_file = ground_motion_model_lt_file_name

		## convert return periods and time_span to poes
		if not (self.return_periods is None or len(self.return_periods) == 0):
			if calculation_mode == "classical":
				params.poes = poisson_conv(t=self.time_span, tau=self.return_periods)
			elif calculation_mode == "disaggregation":
				params.poes_disagg = poisson_conv(t=self.time_span, tau=self.return_periods)

		## set other params
		params.intensity_measure_types_and_levels = self._get_openquake_imts()
		params.investigation_time = self.time_span
		params.truncation_level = self.truncation_level
		params.maximum_distance = self.integration_distance
		params.number_of_logic_tree_samples = self.num_lt_samples
		params.random_seed = self.random_seed

		## disaggregation params

		## write oq params to ini file
		params.write_config(os.path.join(oq_folder, 'job.ini'))

	def calc_shcf_mp(self, cav_min=0, combine_pga_and_sa=True, num_cores=None,
					calc_id="oqhazlib", verbose=True):
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
		:param calc_id:
			int or str, OpenQuake calculation ID (default: "oqhazlib")
		:param verbose:
			bool whether or not to print some progress information
			(default: True)

		:return:
			list of exit codes for each sample (0 for succesful execution,
			1 for error)
		"""
		## Generate all PSHA models
		psha_models_weights = self.sample_logic_trees(self.num_lt_samples,
										enumerate_gmpe_lt=False, verbose=False)

		## Determine number of simultaneous processes
		if not num_cores:
			num_cores = mp.multiprocessing.cpu_count()
		else:
			num_cores = min(mp.multiprocessing.cpu_count(), num_cores)

		## Create list with arguments for each job
		job_args = []
		num_lt_samples = self.num_lt_samples or self.get_num_paths()
		fmt = "%%0%dd" % len(str(num_lt_samples))
#		curve_name = "rlz-%s" % (fmt % (sample_idx + 1))
		curve_path = ""
		for sample_idx, (psha_model, weight) in enumerate(psha_models_weights):
			curve_name = "rlz-%s" % (fmt % (sample_idx + 1))
			job_args.append((psha_model, curve_name, curve_path, cav_min,
							combine_pga_and_sa, calc_id, verbose))

		## Launch multiprocessing
		return mp.run_parallel(mp.calc_shcf_psha_model, job_args, num_cores,
								verbose=verbose)

	def deaggregate_mp(self, sites, imt_periods,
						mag_bin_width=None, dist_bin_width=10.,
						n_epsilons=None, coord_bin_width=1.0,
						num_cores=None, dtype='d', calc_id="oqhazlib",
						interpolate_rp=True, verbose=False):
		"""
		Deaggregate logic tree using multiprocessing.
		Intensity measure levels corresponding to psha_model.return_periods
		will be interpolated first, so the hazard curves must have been
		computed before.

		Note: at least in Windows, this method has to be executed in
		a main section (i.e., behind if __name__ == "__main__":)

		:param sites:
			list with instances of :class:`GenericSite` for which deaggregation
			will be performed. Note that instances of class:`SoilSite` will
			not work with multiprocessing
		:param imt_periods:
			dictionary mapping intensity measure strings to lists of spectral
			periods.
		:param mag_bin_width:
			Float, magnitude bin width (default: None, will take MFD bin width
			of first source)
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
		:param num_cores:
			Int, number of CPUs to be used. Actual number of cores used
			may be lower depending on available cores and memory
			(default: None, will determine automatically)
		:param dtype:
			str, precision of deaggregation matrix
			(default: 'd')
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: "oqhazlib")
		:param interpolate_rp:
			bool, whether or not to interpolate intensity levels corresponding
			to return periods from the hazard curve of the corresponding
			realization first. If False, deaggregation will be performed for all
			intensity levels available for a given spectral period.
			(default: True).
		:param verbose:
			Bool, whether or not to print some progress information

		:return:
			list of exit codes for each sample (0 for succesful execution,
			1 for error)
		"""
		import platform
		import psutil

		## Generate all PSHA models
		psha_models_weights = self.sample_logic_trees(self.num_lt_samples,
										enumerate_gmpe_lt=False, verbose=False)

		## Convert sites to GenericSite objects if necessary, because SoilSites
		## cause problems when used in conjunction with multiprocessing
		## (notably the name attribute cannot be accessed, probably due to
		## the use of __slots__ in parent class)
		## Note that this is similar to the deepcopy problem with MFD objects.
		deagg_sites = []
		site_model = self.get_soil_site_model()
		for site in sites:
			if isinstance(site, SoilSite):
				site = site.to_generic_site()
			if site in site_model:
				deagg_sites.append(site)
		# TODO: check imts as well

		## Determine number of simultaneous processes based on estimated
		## memory consumption
		psha_model0 = psha_models_weights[0][0]
		bin_edges = psha_model0.get_deagg_bin_edges(mag_bin_width, dist_bin_width,
													coord_bin_width, n_epsilons)
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
		#print(free_mem, matrix_size)
		num_processes = min(num_cores, np.floor(free_mem / matrix_size))

		## Create list with arguments for each job
		job_args = []
		num_lt_samples = self.num_lt_samples or self.get_num_paths()
		fmt = "%%0%dd" % len(str(num_lt_samples))
		curve_name = "rlz-%s" % (fmt % (sample_idx + 1))
		curve_path = ""
		for sample_idx, (psha_model, weight) in enumerate(psha_models_weights):
			job_args.append((psha_model, curve_name, curve_path, deagg_sites,
							imt_periods, mag_bin_width, dist_bin_width,
							n_epsilons, coord_bin_width, dtype, calc_id,
							interpolate_rp, verbose))

		## Launch multiprocessing
		return mp.run_parallel(mp.deaggregate_psha_model, job_args, num_processes,
								verbose=verbose)

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
			bool, whether or not to print some information
			(default: True)
		"""
		from ..pmf import get_uniform_weights

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
			## If there is only one TRT, it is possible to make subdirectories
			## for each GMPE
			trts = self.gmpe_lt.tectonic_region_types
			if len(trts) == 1:
				for gmpe_name in self.gmpe_lt.get_gmpe_names(trts[0]):
					subfolder = os.path.join(folder, gmpe_name)
					if not os.path.exists(subfolder):
						os.makedirs(subfolder)

		## Write CRISIS input files
		max_mag = self.source_model_lt.get_max_mag()
		for i, (psha_model, weight) in \
					enumerate(self.sample_logic_trees(self.num_lt_samples,
					enumerate_gmpe_lt=enumerate_gmpe_lt, verbose=verbose)):
			folder = os.path.join(self.crisis_root_folder, psha_model.source_model.name)
			if len(trts) == 1:
				folder = os.path.join(folder, psha_model.ground_motion_model[trts[0]])
			filespec = os.path.join(folder, 'lt-rlz-%04d.dat' % (i+1))
			if os.path.exists(filespec) and overwrite:
				os.unlink(filespec)
			## Write separate attenuation tables for different source models
			sm_gsims_dir = os.path.join(gsims_dir, psha_model.source_model.name)
			psha_model.write_crisis(filespec, sm_gsims_dir, site_filespec,
									atn_Mmax=max_mag)
			sm_filespecs[psha_model.source_model.name].append(filespec)
			all_filespecs.append(filespec)

		## Write CRISIS batch file(s)
		batch_filename = "lt_batch.dat"
		for sm_name in sm_filespecs.keys():
			folder = os.path.join(self.crisis_root_folder, sm_name)
			batch_filespec = os.path.join(folder, batch_filename)
			if os.path.exists(batch_filespec):
				if overwrite:
					os.unlink(batch_filespec)
				else:
					print("File %s exists! Set overwrite=True to overwrite."
							% filespec)
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
				print("File %s exists! Set overwrite=True to overwrite."
						% filespec)
				return
		of = open(batch_filespec, "w")
		weights = get_uniform_weights(len(all_filespecs))
		for filespec, weight in zip(all_filespecs, weights):
			of.write("%s, %s\n" % (filespec, weight))
		of.close()

	def read_oq_shcft(self, add_stats=False, calc_id=None):
		"""
		Read OpenQuake spectral hazard curve field tree.
		Read from the folder 'hazard_curve_multi' if present,
		else read individual hazard curves from the folder 'hazard_curve'.

		:param add_stats:
			bool indicating whether or not mean and quantiles have to be
			appended
			(default: False)
		:param calc_id:
			list of ints, calculation IDs.
			(default: None, will determine from folder structure)

		:return:
			instance of :class:`SpectralHazardCurveFieldTree`
		"""
		from ..openquake import read_shcft

		hc_folder = self.get_oq_hc_folder(calc_id=calc_id)
		## Go one level up, read_shcft will choose between hazard_curve
		## and hazard_curve_multi
		hc_folder = os.path.split(hc_folder)[0]
		shcft = read_shcft(hc_folder, self.get_generic_sites(), add_stats=add_stats)
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
			bool indicating whether or not mean and quantiles have to be
			appended
			(default: False)
		:param calc_id:
			list of ints, calculation IDs.
			(default: None, will determine from folder structure)

		:return:
			instance of :class:`UHSFieldTree`
		"""
		from ..openquake import read_uhsft

		uhs_folder = self.get_oq_uhs_folder(calc_id=calc_id)
		uhsft = read_uhsft(uhs_folder, return_period, self.get_generic_sites(),
							add_stats=add_stats)
		return uhsft

	def write_oq_uhsft(self, uhsft):
		# TODO
		pass

	def get_oq_shcf_percentiles(self, percentile_levels, write_xml=False,
								calc_id=None):
		"""
		Compute or read percentiles of OpenQuake spectral hazard curve
		fields

		:param percentile_levels:
			list or array with percentile levels in the range 0 - 100
		:param write_xml:
			bool, whether or not to write percentile curves to xml files
			(default: False)
		:param calc_id:
			int or str, OpenQuake calculation ID
			(default: None, will be determined automatically)

		:return:
			list with instances of
			:class:`rshalib.result.SpectralHazardCurveField`
		"""
		shcft = self.read_oq_shcft(calc_id=calc_id)
		perc_intercepts = shcft.calc_percentiles_epistemic(percentile_levels,
															weighted=True)

		perc_shcf_list = []
		for p, perc_level in enumerate(percentile_levels):
			curve_name = "quantile-%.2f" % (perc_level / 100.)
			xml_filespec = self.get_oq_shcf_filespec(curve_name, calc_id=calc_id)
			if not write_xml and os.path.exists(xml_filespec):
				shcf = self.read_oq_shcf(curve_name, calc_id=calc_id)
			else:
				model_name = "P%02d(%s)" % (perc_level, self.name)
				hazard_values = perc_intercepts[:,:,:,p]
				filespecs = ['']*len(shcft.sites)
				sites = shcft.sites
				periods = shcft.periods
				IMT = shcft.IMT
				intensities = shcft.intensities
				intensity_unit = shcft.intensity_unit
				timespan = self.time_span
				# TODO: get damping from shcft or from self?
				damping = self.damping
				shcf = SpectralHazardCurveField(hazard_values, sites, periods,
										intensities, intensity_unit, IMT,
										model_name=model_name, filespecs=filespecs,
										timespan=timespan, damping=damping)
				if isinstance(self, DecomposedPSHAModelTree):
					self.write_oq_shcf(shcf, "", "", "", "", curve_name,
										calc_id=calc_id)
				else:
					self.write_oq_shcf(shcf, curve_name, calc_id=calc_id)
			perc_shcf_list.append(shcf)
		return perc_shcf_list

	def read_crisis_shcft(self, batch_filename="lt_batch.dat"):
		"""
		Read CRISIS spectral hazard curve field tree

		:param batch_filename:
			str, name of batch file
			(default: "lt_batch.dat")

		:return:
			instance of :class:`SpectralHazardCurveFieldTree`
		"""
		from ..crisis import read_GRA_multi

		gra_filespecs, weights = self.read_crisis_batch(batch_filename)
		shcft = read_GRA_multi(gra_filespecs, weights=weights)
		return shcft

	def get_deagg_bin_edges(self, mag_bin_width, dist_bin_width, coord_bin_width,
							n_epsilons):
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
			west -= coord_bin_width
			east += coord_bin_width
			south -= coord_bin_width
			north += coord_bin_width
			if west < min_lon:
				min_lon = west
			if east > max_lon:
				max_lon = east
			if south < min_lat:
				min_lat = south
			if north > max_lat:
				max_lat = north

		if len(self.source_models) > 1:
			## Collect tectonic region types
			trt_bins = set()
			for source_model in self.source_models:
				for src in source_model:
					trt_bins.add(src.tectonic_region_type)
			trt_bins = sorted(trt_bins)
		else:
			## Collect source IDs
			trt_bins = [src.source_id for src in self.source_models[0]]

		#min_mag = np.floor(min_mag / mag_bin_width) * mag_bin_width
		dmag = np.ceil((max_mag - min_mag) / mag_bin_width) * mag_bin_width
		max_mag = min_mag + dmag

		min_dist = 0
		max_dist = np.ceil(self.integration_distance / dist_bin_width) * dist_bin_width

		## Note that ruptures may extend beyond source limits
		min_lon = np.floor(min_lon / coord_bin_width) * coord_bin_width
		min_lat = np.floor(min_lat / coord_bin_width) * coord_bin_width
		max_lon = np.ceil(max_lon / coord_bin_width) * coord_bin_width
		max_lat = np.ceil(max_lat / coord_bin_width) * coord_bin_width

		nmags = int(round(dmag / mag_bin_width))
		ndists = int(round(max_dist / dist_bin_width))
		nlons = int((max_lon - min_lon) / coord_bin_width)
		nlats = int((max_lat - min_lat) / coord_bin_width)

		mag_bins = min_mag + mag_bin_width * np.arange(nmags + 1)
		dist_bins = np.linspace(min_dist, max_dist, ndists + 1)
		lon_bins = np.linspace(min_lon, max_lon, nlons + 1)
		lat_bins = np.linspace(min_lat, max_lat, nlats + 1)
		eps_bins = np.linspace(-self.truncation_level, self.truncation_level,
								  n_epsilons + 1)

		return (mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, trt_bins)

	def get_oq_mean_sdc(self, site, calc_id=None, dtype='d'):
		"""
		Compute mean spectral deaggregation curve from individual models.

		:param site:
			instance of :class:`GenericSite` or :class:`SoilSite`
		:param calc_id:
			list of ints, calculation IDs.
			(default: None, will determine from folder structure)
		:param dtype:
			str, precision of deaggregation matrix
			(default: 'd')

		:return:
			instance of :class:`SpectralDeaggregationCurve`
		"""
		import gc
		from ..result import FractionalContributionMatrix, SpectralDeaggregationCurve

		## Read saved deaggregation files for given site
		num_lt_samples = self.num_lt_samples or self.get_num_paths()
		fmt = "rlz-%%0%dd" % len(str(num_lt_samples))
		for i, (psha_model, weight) in enumerate(self.sample_logic_trees()):
			curve_name = fmt % (i+1)
			print(curve_name)
			sdc = self.read_oq_disagg_matrix_multi(curve_name, site,
													calc_id=calc_id, dtype=dtype)
			## Apply weight
			sdc_matrix = sdc.deagg_matrix.to_fractional_contribution_matrix()
			sdc_matrix *= float(weight)

			if i == 0:
				## Obtain overall bin edges
				mag_bin_width = sdc.mag_bin_width
				dist_bin_width = sdc.dist_bin_width
				coord_bin_width = sdc.lon_bin_width
				num_epsilon_bins = sdc.neps

				## Create empty matrix
				bin_edges = self.get_deagg_bin_edges(mag_bin_width, dist_bin_width,
												coord_bin_width, num_epsilon_bins)
				mean_deagg_matrix = SpectralDeaggregationCurve.construct_empty_deagg_matrix(
										num_periods, num_intensities, bin_edges,
										FractionalContributionMatrix, sdc.deagg_matrix.dtype)

			## Sum deaggregation results of logic_tree samples
			## Assume min_mag, distance bins and eps bins are the same for all models
			max_mag_idx = sdc.nmags
			min_lon_idx = int((sdc.min_lon - lon_bins[0]) / coord_bin_width)
			max_lon_idx = min_lon_idx + sdc.nlons
			min_lat_idx = int((sdc.min_lat - lat_bins[0]) / coord_bin_width)
			max_lat_idx = min_lat_idx + sdc.nlats
			#print(max_mag_idx)
			#print(sdc.min_lon, sdc.max_lon)
			#print(min_lon_idx, max_lon_idx)
			#print(sdc.min_lat, sdc.max_lat)
			#print(min_lat_idx, max_lat_idx)

			if sdc.trt_bins == trt_bins:
				## trt bins correspond to source IDs
				mean_deagg_matrix[:,:,:max_mag_idx,:,min_lon_idx:max_lon_idx,min_lat_idx:max_lat_idx,:,:] += sdc_matrix
			else:
				## trt bins correspond to tectonic region types
				for trt_idx, trt in enumerate(trt_bins):
					src_idxs = []
					for src_idx, src_id in enumerate(sdc.trt_bins):
						src = psha_model.source_model[src_id]
						if src.tectonic_region_type == trt:
							src_idxs.append(src_idx)
				mean_deagg_matrix[:,:,:max_mag_idx,:,min_lon_idx:max_lon_idx,min_lat_idx:max_lat_idx,:,trt_idx] += sdc_matrix[:,:,:,:,:,:,:,src_idxs].fold_axis(-1)

			del sdc_matrix
			gc.collect()

		intensities = np.zeros_like(sdc.intensities)

		return SpectralDeaggregationCurve(bin_edges, mean_deagg_matrix, site,
											intensities, sdc.intensity_unit, sdc.imt,
											sdc.periods, sdc.return_periods,
											sdc.timespan, sdc.damping)

	def to_decomposed_psha_model_tree(self):
		"""
		Convert to decomposed PSHA model tree

		:return:
			instance of :class:`DecomposedPSHAModelTree`
		"""
		from .decomposed_pshamodeltree import DecomposedPSHAModelTree

		return DecomposedPSHAModelTree(self.name, self.source_model_lt,
								self.gmpe_lt, self.root_folder, self.site_model,
								self.ref_soil_params, self.imt_periods,
								self.intensities, self.min_intensities,
								self.max_intensities, self.num_intensities,
								self.return_periods, self.time_span,
								self.truncation_level, self.integration_distance,
								self.num_lt_samples, self.random_seed)

	def get_max_return_period(self, site_idx=0, calc_id="decomposed",
							verbose=True):
		"""
		Determine maximum return period covered by hazard curves

		:return:
			float, max. return period
		"""
		periods = self._get_periods()
		imt_exceedance_rates = np.zeros_like(periods)
		if verbose:
			print("Reading hazard curves")
		for source_model in self.source_models:
			print("  %s" % source_model.name)
			sm_imt_exceedance_rates = np.zeros_like(periods)
			for src in source_model.sources:
				if verbose:
					print("    %s" % src.source_id)
				src_shcft = self.read_oq_source_shcft(source_model.name, src,
													calc_id=calc_id)
				src_imt_exceedance_rates = src_shcft.exceedance_rates[site_idx,:,:,-1].max(axis=0)
				sm_imt_exceedance_rates += src_imt_exceedance_rates
			imt_exceedance_rates = np.max([imt_exceedance_rates, sm_imt_exceedance_rates],
											axis=0)

		if verbose:
			print
		for i in range(len(periods)):
			T = src_shcft.periods[i]
			TR = 1./imt_exceedance_rates[i]
			if verbose:
				print("T = %s s: TR = %.1f yr" % (T, TR))
		if verbose:
			print("Max TR: %.1f yr" % (1./imt_exceedance_rates.max()))

		return 1./imt_exceedance_rates

	def _get_used_trts(self):
		"""
		Return list of strings, defining tectonic region types
		used in source models.
		"""
		used_trts = []
		for source_model in self.source_models:
			for source in source_model:
				trt = source.tectonic_region_type
				if trt not in used_trts:
					used_trts.append(trt)
		return used_trts


	# TODO: the following methods are probably obsolete

	def run_oqhazlib(self, nrml_base_filespec=""):
		"""
		Run PSHA model with oqhazlib and store result in a
		SpectralHazardCurveFieldTree object.

		:param nrml_base_filespec:
			str, base file specification for NRML output file
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
			hazard_results[imt] = np.zeros((num_sites, self.num_lts_samples,
											len(periods), self.num_intensities))
		psha_model_names, weights = [], []
		filespecs = ['']*len(psha_models)
		for j, psha_model in enumerate(psha_models):
			print(psha_model.name)
			psha_model_names.append(psha_model.name)
			weights.append(1./len(psha_models))
			hazard_result = psha_model.run_oqhazlib_poes()
			for imt in self.imt_periods.keys():
				hazard_results[imt][:,j,:,:] = hazard_result[imt]
		im_imls = self._get_im_imls()
		sites = self.get_generic_sites()
		site_names = [site.name for site in sites]
		for imt, periods in self.imt_periods.items():
			shcft = SpectralHazardCurveFieldTree(hazard_results[imt],
									psha_model_names, weights, sites, periods,
									im_imls[imt], self.intensity_unit, imt,
									model_name=self.name, filespecs=filespecs,
									timespan=self.time_span, damping=self.damping)
			nrml_filespec = nrml_base_filespec + '_%s.xml' % imt
			shcft.write_nrml(nrml_filespec)
		return shcft

	def _get_psha_models(self):
		"""
		Return list of :class:`PSHAModel` objects, defining sampled
		PSHA models from logic tree.
		"""
		psha_models = []
		for i in range(self.num_lts_samples):
			source_model, ground_motion_model = self._sample_lts()
			name = source_model.name + '_' + ground_motion_model.name
			psha_models.append(PSHAModel(name, source_model, ground_motion_model,
										self.root_folder, self.site_model,
										self.ref_soil_params, self.imt_periods,
										self.intensities, self.min_intensities,
										self.max_intensities, self.num_intensities,
										self.return_periods, self.time_span,
										self.truncation_level, self.integration_distance))
		return psha_models

	def _get_openquake_trts_gsims_map_lt(self):
		"""
		Return {str: {str: float}} dict, defining respectively
		tectonic region types, gsims and gsim weight for OpenQuake.
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
		lts_sampling_methods = {'random': self._sample_lts_random,
								'weighted': self._sample_lts_weighted,
								'enumerated': self._sample_lts_enumerated}
		lts_sample = lts_sampling_methods[self.lts_sampling_method]()
		return lts_sample

	def _sample_lts_random(self):
		"""
		Return random logic tree sample.
		"""
		from random import choice

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

