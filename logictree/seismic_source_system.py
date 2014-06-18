import os
import copy
import pprint

try:
    import simplejson as json
except ImportError:
    import json

import numpy as np

from logictree import LogicTreeBranch, LogicTreeBranchSet, LogicTreeBranchingLevel, LogicTree
from configobj import ConfigObj
from ..pmf import SourceModelPMF, MmaxPMF, MFDPMF, get_uniform_weights
from ..source import SourceModel


class SeismicSourceSystem(LogicTree):
	"""
	Class representing Seismic Source System
	Inherits from LogicTree class

	:param ID:
		string, identifier
	:param source_model_pmf:
		instance of :class:`SourceModel`or of :class:`SourceModelPMF`

	After initialization, additional uncertainty levels can be
	appended using :meth:`append_independent_uncertainty_level`
	"""
	def __init__(self, ID, source_model_pmf=None):
		super(SeismicSourceSystem, self).__init__(ID, [])
		if source_model_pmf:
			if isinstance(source_model_pmf, SourceModel):
				source_model = source_model_pmf
				source_model_pmf = SourceModelPMF([source_model], [1])
			self.set_root_uncertainty_level(source_model_pmf)

	@property
	def source_models(self):
		return self.source_model_pmf.source_models

	def get_source_model_by_name(self, source_model_name):
		"""
		Get source model by name

		:param source_model_name:
			str, name of source model

		:return:
			instance ov :class:`rshalib.source.SourceModel`
		"""
		[source_model] = [somo for somo in self.source_models if somo.name == source_model_name]
		return source_model

	def get_max_mag(self):
		"""
		Determine maximum absolute magnitude (maxMagGRAbsolute)
		in the model.

		:return:
			float, maximum magnitude
		"""
		max_mag = 0
		for bs in self.get_branchsets(unc_class="mmax"):
			if bs.uncertainty_type == "maxMagGRAbsolute":
				for branch in bs:
					if isinstance(branch.value, dict):
						value = max(branch.value.values())
					else:
						value = branch.value
					max_mag = max(max_mag, value)
		return max_mag

	def get_source_max_mag(self, source_model_name, src):
		"""
		Determine maximum absolute magnitude for a particular source

		:param source_model_name:
			str, name of source model
		:param src:
			instance of :class:`rshalib.source.PointSource`,
			`rshalib.source.AreaSource`, `rshalib.source.SimpleFaultSource`
			or `rshalib.source.ComplexFaultSource`

		:return:
			float, maximum magnitude
		"""
		max_mag = 0
		for bs in self.get_source_branch_sets(source_model_name, src):
			if bs.uncertainty_type == "maxMagGRAbsolute":
				for branch in bs:
					if isinstance(branch.value, dict):
						value = max(branch.value.values())
					else:
						value = branch.value
					max_mag = max(max_mag, value)
		## If max_mag is still zero, it means there were no branches where
		## it was altered; in that case, use max_mag from the source MFD
		if max_mag == 0:
			max_mag = src.mfd.max_mag
		return max_mag

	@classmethod
	def parse_from_xml(cls, xml_filespec, validate=False):
		"""
		Read source-model logic tree from XML file

		:param xml_filespec:
			string, full path to XML file
		:param validate:
			Bool, whether or not parsed XML should be validated
			(default: False)

		:return:
			instance of :class:`SeismicSourceSystem`

		Note: branching level ID's and branch set ID's cannot be recovered
		"""
		from openquake.engine.input.logictree import SourceModelLogicTree

		## Parse using oq-engine's logictree
		basepath, filename = os.path.split(xml_filespec)
		smlt = SourceModelLogicTree(None, basepath=basepath, filename=filename, calc_id=None, validate=validate)

		def convert_branchset(branchset):
			## Convert oq-engine BranchSet to rshalib LogicTreeBranchSet
			filters = branchset.filters
			applyToSources = filters.get("applyToSources", [])
			applyToSourceType = filters.get("applyToSourceType", [])
			applyToTectonicRegionType = filters.get("applyToTectonicRegionType", "")
			if hasattr(branchset, "applyToBranches"):
				applyToBranches = branchset.applyToBranches
			else:
				applyToBranches = []
			new_bs = LogicTreeBranchSet(branchset.id, branchset.uncertainty_type, branchset.branches, applyToBranches=applyToBranches, applyToSources=applyToSources, applyToSourceType=applyToSourceType, applyToTectonicRegionType=applyToTectonicRegionType)
			for i, branch in enumerate(new_bs):
				new_bs.branches[i] = LogicTreeBranch(branch.branch_id, branch.weight, branch.value, parent_branchset=new_bs)
			return new_bs

		## Reconstruct branching levels from branches
		branching_levels = []
		branching_level_nr = 0
		branching_level_id = "bl%02d" % branching_level_nr
		branchset_id = "%s_bs00" % branching_level_id
		smlt.root_branchset.id = branchset_id
		branching_levels.append(LogicTreeBranchingLevel(branching_level_id, [smlt.root_branchset]))
		branchsets = [smlt.root_branchset]
		while branchsets:
			branching_level_nr += 1
			branching_level_id = "bl%02d" % branching_level_nr
			branching_level = LogicTreeBranchingLevel(branching_level_id, [])
			branchset_nr = 0
			for branchset in branchsets:
				for branch in branchset.branches:
					child_branchset = branch.child_branchset
					if child_branchset:
						try:
							index = branching_level.branch_sets.index(child_branchset)
						except ValueError:
							branchset_id = "%s_bs%02d" % (branching_level_id, branchset_nr)
							child_branchset.id = branchset_id
							child_branchset.applyToBranches = [branch.branch_id]
							branching_level.branch_sets.append(child_branchset)
							branchset_nr += 1
						else:
							branching_level.branch_sets[index].applyToBranches.append(branch.branch_id)

			if len(branching_level.branch_sets) > 0:
				branching_levels.append(branching_level)
				branchsets = branching_level.branch_sets
			else:
				branchsets = []

		for branching_level in branching_levels:
			converted_branchsets = [convert_branchset(bs) for bs in branching_level.branch_sets]
			branching_level.branch_sets = converted_branchsets
		sss = SeismicSourceSystem(filename, None)
		sss.branching_levels = branching_levels
		sss.connect_branches()
		return sss

	def get_bl_branch_ids(self):
		"""
		Construct lists of branch ID's in each branching level for each
		root branch.
		Requires branches to be connected (see :meth:`connect_branches`)

		:return:
			dict, mapping root branch ID to a list of sets containing
			branch ID's in each branching level
		"""
		bl_branch_ids = {}
		for root_branch in self.root_branchset:
			sm_name = os.path.splitext(root_branch.value)[0]
			bl_branch_ids[sm_name] = [set([root_branch.branch_id])]

		for prev_bl_index, branching_level in enumerate(self.branching_levels[1:]):
			bl_index = prev_bl_index + 1
			for branchset in branching_level.branch_sets:
				branch_ids = [branch.branch_id for branch in branchset.branches]
				parent_branches = self.get_parent_branches(branchset)
				for sm_name in bl_branch_ids.keys():
					if len(bl_branch_ids[sm_name]) == bl_index:
						for parent_branch in parent_branches:
							if parent_branch.branch_id in bl_branch_ids[sm_name][prev_bl_index]:
								try:
									bl_branch_ids[sm_name][bl_index].update(branch_ids)
								except IndexError:
									bl_branch_ids[sm_name].append(set(branch_ids))

		return bl_branch_ids

	def get_bl_branchsets(self):
		"""
		Construct lists of branch sets in each branching level for each
		root branch.
		Requires branches to be connected (see :meth:`connect_branches`)
		Present implementation also assumes that uncertainties are independent,
		i.e. there is only one branch set in each branching level that is linked
		to a given root branch.

		:return:
			dict, mapping root branch ID to a list of branch sets in each
			branching level
		"""
		bl_branch_sets = {}
		for root_branch in self.root_branchset:
			sm_name = os.path.splitext(root_branch.value)[0]
			bl_branch_sets[sm_name] = []

		for prev_bl_index, branching_level in enumerate(self.branching_levels[1:]):
			bl_index = prev_bl_index + 1
			for branchset in branching_level.branch_sets:
				parent_branches = self.get_parent_branches(branchset)
				for sm_name in bl_branch_sets.keys():
					if prev_bl_index == 0:
						for parent_branch in parent_branches:
							if os.path.splitext(parent_branch.value)[0] == sm_name:
								bl_branch_sets[sm_name].append(branchset)
					else:
						if len(bl_branch_sets[sm_name]) == prev_bl_index:
							for parent_branch in parent_branches:
								if parent_branch.branch_id in [b.branch_id for b in bl_branch_sets[sm_name][prev_bl_index-1].branches]:
									if len(bl_branch_sets[sm_name]) == prev_bl_index:
										bl_branch_sets[sm_name].append(branchset)

		return bl_branch_sets

	def list_correlated_sources(self, source_model):
		"""
		Obtain lists of correlated source ids. Note that uncorrelated
		sources are listed as 1-element lists.

		:param:
			instance of :class:`SourceModel`

		:return:
			list containing lists of correlated source ids
		"""
		branch_sets = self.get_bl_branchsets()[source_model.name]

		correlated_source_ids = []
		for bs in branch_sets:
			src_ids = bs.get_filtered_source_ids(source_model)
			if not src_ids in correlated_source_ids:
				correlated_source_ids.append(src_ids)
		return correlated_source_ids

	def get_source_branch_sets(self, source_model_name, src):
		"""
		Get list of logic-tree branch sets applying to a particular source

		:param source_model_name:
			str, name of source model
		:param src:
			instance of :class:`rshalib.source.PointSource`,
			`rshalib.source.AreaSource`, `rshalib.source.SimpleFaultSource`
			or `rshalib.source.ComplexFaultSource`

		:return:
			list with instances of :class:`LogicTreeBranchSet`
		"""
		branch_sets = self.get_bl_branchsets()[source_model_name]

		src_branch_sets = []
		for bs in branch_sets:
			if bs.filter_source(src):
				src_branch_sets.append(bs)

		return src_branch_sets

	def enumerate_branch_paths_by_source(self, source_model_name, src):
		"""
		Enumerate branch paths for a particular source

		:param source_model_name:
			str, name of source model
		:param src:
			instance of :class:`rshalib.source.PointSource`,
			`rshalib.source.AreaSource`, `rshalib.source.SimpleFaultSource`
			or `rshalib.source.ComplexFaultSource`

		:return:
			generator object yielding (branch_path, weight) tuple
			branch_path is a list with instances of :class:`LogicTreeBranch`
		"""
		src_branch_sets = self.get_source_branch_sets(source_model_name, src)
		if len(src_branch_sets):
			for branch_indices in np.ndindex(*[len(bs) for bs in src_branch_sets]):
				branch_path = []
				weight = 1
				for bs_index, br_index in enumerate(branch_indices):
					branch = src_branch_sets[bs_index].branches[br_index]
					branch_path.append(branch)
					weight *= branch.weight
				yield (branch_path, weight)

	def enumerate_source_realizations(self, source_model_name, src):
		"""
		Loop over all possible branch paths for a particular source,
		simultaneously applying logic-tree choices to the source

		:param source_model_name:
			str, name of source model
		:param src:
			instance of :class:`rshalib.source.PointSource`,
			`rshalib.source.AreaSource`, `rshalib.source.SimpleFaultSource`
			or `rshalib.source.ComplexFaultSource`

		:return:
			generator object yielding (modified_src, branch_path, weight)
			tuple:
			- modified_src: source object with logic-tree path applied
			- branch_path: list with branch IDs of logic_tree path
			- weight: decimal, corresponding weight
		"""
		import copy

		for (branch_path, weight) in self.enumerate_branch_paths_by_source(source_model_name, src):
			for i, branch in enumerate(branch_path):
				if i == 0:
					## Note: copy MFD explicitly, as not all source attributes are
					## instantiated properly when deepcopy is used!
					modified_src = copy.copy(src)
					modified_src.mfd = src.mfd.get_copy()
				bs = branch.parent_branchset
				bs.apply_uncertainty(branch.value, modified_src)
			branch_path = [b.branch_id for b in branch_path]
			yield (modified_src, branch_path, weight)

		"""
		src_branch_sets = self.get_source_branch_sets(source_model, src)
		bs0 = src_branch_sets[0]
		branch_path = [""] * len(src_branch_sets)
		for branch in bs0.branches:
			## Note: copy MFD explicitly, as not all source attributes are
			## instantiated properly when deepcopy is used!
			modified_src = copy.copy(src)
			modified_src.mfd = src.mfd.get_copy()
			bs0.apply_uncertainty(branch.value, modified_src)
			branch_path[0] = branch.branch_id

			for bl_index in range(1, len(src_branch_sets)):
				bs = src_branch_sets[bl_index]
				for branch in bs.branches:
					bs.apply_uncertainty(branch.value, modified_src)
					branch_path[bl_index] = branch.branch_id
		"""

	def set_root_uncertainty_level(self, source_model_pmf, overwrite=False):
		"""
		Set root uncertainty level (corresponding to source-model uncertainties)
		of the logic tree.

		:param source_model_pmf:
			instance of :class:`SourceModelPMF`
		:param overwrite:
			Bool, indicating whether or not current branching levels should
			be overwritten (default: False)
		"""
		if len(self.branching_levels) > 0 and not overwrite:
			raise Exception("Logic tree already has root branchset")
		else:
			self.branching_levels = []
		if not isinstance(source_model_pmf, SourceModelPMF):
			raise Exception("Argument must be SourceModelPMF instance")
		branching_level_id = "bl00"
		#branch_set_id = "%s_bs00" % branching_level_id
		branch_set_id = "SM"
		branch_set = LogicTreeBranchSet.from_PMF(branch_set_id, source_model_pmf)
		branching_level = LogicTreeBranchingLevel(branching_level_id, [branch_set])
		self.branching_levels.append(branching_level)
		self.connect_branches()
		self.source_model_pmf = source_model_pmf

	def append_independent_uncertainty_level(self, unc_pmf_dict, correlated=False):
		"""
		Append an independent uncertainty level to the logic tree,
		converting to the branching-level structure used by OpenQuake.
		Independent means that the uncertainty level does not depend
		on the previous uncertainty level (except for the source-model
		uncertainty)
		This method requires that the root level representing source-model
		uncertainties is present.

		:param unc_pmf_dict:
			dictionary specifying uncertainty level, taking the following form:
			{sm_nameN: {src_id_N: instance of :class:`MmaxPMF` or :class:`MFDPMF`}}
		:param correlated:
			Bool, indicating whether uncertainties in level are correlated for
			the different sources in each source model (default: False)
		"""
		if self.root_branchset is None:
			raise Exception("Root uncertainty level must be set first!")
		source_model_names = [os.path.splitext(branch.value)[0] for branch in self.root_branchset]

		## This dictionary will contain lists of branch id's in each branching
		## level for the different source models
		bl_branch_ids = self.get_bl_branch_ids()

		## Loop over source models
		for sm_name, src_unc_pmf_dict in unc_pmf_dict.items():
			## Determine uncertainty type
			first_pmf = src_unc_pmf_dict[src_unc_pmf_dict.keys()[0]]
			if isinstance(first_pmf, MmaxPMF):
				unc_type = "Mmax"
			else:
				unc_type = "MFD"

			## Sanity check
			if sm_name is None:
				## Source model unspecified, sources must be unspecified too
				if src_unc_pmf_dict.keys() != [None]:
					raise Exception("If source model is unspecified, then sources must be unspecified too")
			if sm_name:
				## Source model specified, check if source ID's are in source model
				source_model = [sm for sm in self.source_models if sm.name == sm_name][0]
				sm_src_ids = [src.source_id for src in source_model]
				for src_id in src_unc_pmf_dict.keys():
					if src_id != None and src_id not in sm_src_ids:
						raise Exception("Source %s not in source model %s!" % (src_id, sm_name))

			if sm_name is None:
				## Source model, and hence sources unspecified
				## Only one branching level will be added
				applyToSources = []
				num_branching_levels = [len(bl_branch_ids[key]) for key in bl_branch_ids.keys()]
				if len(bl_branch_ids) == 1 or len(set(num_branching_levels)) == 1:
					## Same branching level for different source models
					applyToBranches = []
					applyToSources = []
					branch_set_id = "None--None--%s" % (unc_type)
					branching_level_nr = len(bl_branch_ids[bl_branch_ids.keys()[0]])
					branch_set = LogicTreeBranchSet.from_PMF(branch_set_id, src_unc_pmf_dict[None], applyToBranches=applyToBranches, applyToSources=applyToSources)
					self._append_branchset(branch_set, branching_level_nr)
					for sm_name in source_model_names:
						bl_branch_ids[sm_name].append(set([branch.branch_id for branch in branch_set]))
				else:
					## Branching level different for different source models
					for prev_level_sm_name in bl_branch_ids.keys():
						applyToBranches = bl_branch_ids[prev_level_sm_name][-1]
						branch_set_id = "%s--None--%s" % (prev_level_sm_name, unc_type)
						#branch_set_id = "%s_bs%02d" % (branching_level_id, branch_set_nr)
						branch_set = LogicTreeBranchSet.from_PMF(branch_set_id, src_unc_pmf_dict[None], applyToBranches=applyToBranches, applyToSources=applyToSources)
						branching_level_nr = len(bl_branch_ids[prev_level_sm_name])
						self._append_branchset(branch_set, branching_level_nr)
						bl_branch_ids[prev_level_sm_name].append(set([branch.branch_id for branch in branch_set]))
			else:
				## Source model specified
				last_branching_level_nr = len(bl_branch_ids[sm_name])
				if last_branching_level_nr > 1:
					## If other source models have same branch ID's in last
					## branching level, then it means source model was None
					## in previous branching level
					bl_other_end_branch_ids = [bl_branch_ids[key][-1] for key in source_model_names if not key == sm_name]
					bl_end_branch_ids = bl_branch_ids[sm_name][-1]
					if len(bl_end_branch_ids.difference(*bl_other_end_branch_ids)) < len(bl_end_branch_ids):
						raise Exception("Not possible to connect source model %s" % sm_name)
				sm_index = source_model_names.index(sm_name)
				if correlated == True:
					## Correlated uncertainties
					applyToBranches = bl_branch_ids[sm_name][-1]
					applyToSources = []
					branching_level_nr = last_branching_level_nr
					branch_set_id = "%s--CORR--%s" % (sm_name, unc_type)
					branch_set = LogicTreeBranchSet.from_PMF_dict(branch_set_id, src_unc_pmf_dict, applyToBranches=applyToBranches)
					self._append_branchset(branch_set, branching_level_nr)
					bl_branch_ids[sm_name].append(set([branch.branch_id for branch in branch_set]))
				else:
					## Uncorrelated uncertainties
					for i, src_id in enumerate(src_unc_pmf_dict.keys()):
						applyToBranches = bl_branch_ids[sm_name][-1]
						if src_id is None:
							## Sources unspecified
							## Only one branching level required for this source model
							applyToSources = []
						else:
							applyToSources = [src_id]
						branching_level_nr = last_branching_level_nr + i
						#branch_set_id = "%s_bs%02d" % (branching_level_id, branch_set_nr)
						branch_set_id = "%s--%s--%s" % (sm_name, src_id, unc_type)
						branch_set = LogicTreeBranchSet.from_PMF(branch_set_id, src_unc_pmf_dict[src_id], applyToBranches=applyToBranches, applyToSources=applyToSources)
						self._append_branchset(branch_set, branching_level_nr)
						bl_branch_ids[sm_name].append(set([branch.branch_id for branch in branch_set]))
						#print sm_name, src_id, branching_level_nr, len(bl_branch_ids[sm_name]), len(branching_levels)

		self.connect_branches()

	def _append_branchset(self, branch_set, branching_level_nr):
		"""
		Helper function for :meth:`append_independent_uncertainty_level`

		:param branch_set:
			instance of :class:`LogicTreeBranchSet
		:param branching_level_nr:
			int, index of branching level where branchset has to be appended
		"""
		try:
			branch_set_nr = len(self.branching_levels[branching_level_nr])
		except IndexError:
			branch_set_nr = 0
		if branch_set_nr:
			branching_level = self.branching_levels[branching_level_nr]
			branching_level.branch_sets.append(branch_set)
		else:
			branching_level_id = "bl%02d" % branching_level_nr
			branching_level = LogicTreeBranchingLevel(branching_level_id, [branch_set])
			self.branching_levels.append(branching_level)

	@classmethod
	def from_independent_uncertainty_levels(cls, sss_id, source_model_pmf, unc2_pmf_dict, unc3_pmf_dict, unc2_correlated=False, unc3_correlated=False):
		source_model_lt = SeismicSourceSystem(sss_id, None)
		print("Setting root uncertainty level")
		source_model_lt.set_root_uncertainty_level(source_model_pmf)
		if unc2_pmf_dict:
			print("Appending second uncertainty level")
			source_model_lt.append_independent_uncertainty_level(unc2_pmf_dict, correlated=unc2_correlated)
		if unc3_pmf_dict:
			print("Appending third uncertainty level")
			#assert unc2_pmf_dict.keys() != [None], "If sources are specified, then source model must be specified in the previous uncertainty level"
			source_model_lt.append_independent_uncertainty_level(unc3_pmf_dict, correlated=unc3_correlated)
		return source_model_lt

	## Note: in the case of dependent uncertainties (e.g., MFD depends on Mmax),
	## we need to alternate the two uncertainties for each source

	def plot_uncertainty_levels_diagram(self, source_model_pmf, unc_pmf_dicts, branch_labels=True, fig_filespec=None, dpi=300):
		import networkx as nx
		import pylab

		graph = nx.Graph()

		pos = {}
		edge_labels = {}
		root_node = "ROOT"
		pos[root_node] = (0, 0.5)
		source_nodes, num_sources = {}, 0
		for sm in source_model_pmf.source_models:
			source_nodes[sm.name] = [src.source_id for src in sm]
			num_sources += len(source_nodes[sm.name])
		source_model_nodes = source_nodes.keys()
		all_source_nodes = sum([source_nodes[sm_name] for sm_name in source_model_nodes], [])
		dy = 2./num_sources
		y = dy / 2
		sm_src_y = {}
		for sm_name in source_nodes.keys():
			sm_src_y[sm_name] = {}
			ymin = y
			for src_id in source_nodes[sm_name]:
				pos[src_id] = (0.67, y)
				sm_src_y[sm_name][src_id] = y
				y += dy
			ymax = y - dy
			pos[sm_name] = (0.33, np.mean([ymin, ymax]))

		x = 1
		prev_sm_nodes = {sm_name: {src_id: [src_id] for src_id in source_nodes[sm_name]} for sm_name in source_model_nodes}
		branchset_nodes = {"Mmax": [], "MFD": []}
		branch_nodes = []
		for unc_pmf_dict in unc_pmf_dicts:
			current_sm_nodes = {}
			for sm_name in unc_pmf_dict.keys():
				current_sm_nodes[sm_name] = {}
				src_unc_pmf_dict = unc_pmf_dict[sm_name]

				## Determine uncertainty type
				first_pmf = src_unc_pmf_dict[src_unc_pmf_dict.keys()[0]]
				if isinstance(first_pmf, MmaxPMF):
					unc_type = "Mmax"
				else:
					unc_type = "MFD"

				if sm_name is None:
					pass
				else:
					for src_id in src_unc_pmf_dict.keys():
						if src_id is None:
							pass
						else:
							current_sm_nodes[sm_name][src_id] = []
							unc_pmf = src_unc_pmf_dict[src_id]
							branchset = "%s_%s_%s" % (sm_name, src_id, unc_type)
							y = sm_src_y[sm_name][src_id]
							pos[branchset] = (x, y)
							branchset_nodes[unc_type].append(branchset)
							pmf_len = len(unc_pmf)
							for i in range(pmf_len):
								branch = "%s%02d" % (branchset, i+1)
								branch_nodes.append(branch)
								if pmf_len > 1:
									pos[branch] = (x+0.5, y-dy/2+(dy/pmf_len)*(i+0.5))
								graph.add_edge(branchset, branch)
								weight_label = str(unc_pmf.weights[i]).rstrip('0')
								edge_labels[(branchset, branch)] = weight_label
								current_sm_nodes[sm_name][src_id].append(branch)
							for prev_branch in prev_sm_nodes[sm_name][src_id]:
								graph.add_edge(prev_branch, branchset)
			x += 1
			prev_sm_nodes = current_sm_nodes

		## Add nodes
		graph.add_node(root_node)
		graph.add_nodes_from(source_model_nodes)
		for sm_name in source_nodes.keys():
			graph.add_nodes_from(source_nodes[sm_name])
		for unc_type in branchset_nodes.keys():
			graph.add_nodes_from(branchset_nodes[unc_type])
		graph.add_nodes_from(branch_nodes)

		## Add edges
		for sm_name in source_model_nodes:
			graph.add_edge(root_node, sm_name)
			for src in source_nodes[sm_name]:
				graph.add_edge(sm_name, src)

		## Draw nodes
		nx.draw_networkx_nodes(graph, pos, nodelist=[root_node], node_shape='>', node_color='red', node_size=300, label="ROOT")
		nx.draw_networkx_nodes(graph, pos, nodelist=source_model_nodes, node_shape='o', node_color='white', node_size=300, label="_nolegend_")
		nx.draw_networkx_nodes(graph, pos, nodelist=all_source_nodes, node_shape='s', node_color='red', node_size=300, label="sources")
		nx.draw_networkx_nodes(graph, pos, nodelist=branchset_nodes["Mmax"], node_shape=">", node_color="green", node_size=300, label="Mmax")
		nx.draw_networkx_nodes(graph, pos, nodelist=branchset_nodes["MFD"], node_shape=">", node_color="yellow", node_size=300, label="MFD")
		nx.draw_networkx_nodes(graph, pos, nodelist=branch_nodes, node_shape="o", node_color="white", node_size=300, label="branches")

		## Draw edges
		nx.draw_networkx_edges(graph, pos)

		## Draw labels
		if branch_labels:
			nx.draw_networkx_labels(graph, pos, font_size=10, horizontalalignment="center", verticalalignment="bottom", xytext=(0,20), textcoords="offset points")
		nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10, horizontalalignment="center", verticalalignment="bottom", xytext=(0,20), textcoords="offset points")

		## Plot decoration
		pylab.legend(loc=2, scatterpoints=1, markerscale=0.75, prop={'size': 12})
		#pylab.axis((-0.5, len(self.branching_levels), 0, 1.2))
		pylab.xticks(range(len(unc_pmf_dicts) + 2))
		pylab.yticks([])
		pylab.xlabel("Uncertainty level", fontsize=14)
		ax = pylab.gca()
		for tick in ax.xaxis.get_major_ticks():
			tick.label.set_fontsize(14)
		pylab.grid(True, axis='x')
		pylab.title("Logic-tree diagram", fontsize=16)

		if fig_filespec:
			pylab.savefig(fig_filespec, dpi=dpi)
			pylab.clf()
		else:
			pylab.show()

	def get_source_logic_tree(self, source_model_name, src_id):
		"""
		Construct reduced logic tree for a particular source

		:param source_model_name:
			str, name of source model
		:param src_id:
			str, source ID

		:return:
			instance of :class:`SeismicSourceSystem`
		"""
		source_model = self.get_source_model_by_name(source_model_name)
		src = source_model[src_id]
		sss = SeismicSourceSystem("%s--%s" % (source_model.name, src.source_id), source_model)
		branch_sets = self.get_source_branch_sets(source_model.name, src)
		for bl_index, bs in enumerate(branch_sets):
			pmf_dict = {source_model.name: bs.to_pmf_dict(source_model)}
			sss.append_independent_uncertainty_level(pmf_dict, correlated=True)
		sss.connect_branches()
		return sss


class SeismicSourceSystem_v1(LogicTree):
	def __init__(self, ID, source_system_dict={}, sourceModelObjs=[]):
		"""
		source_system_dict:
		Mmax_branches:
		unc_type = "absolute": Mmax, weight
		unc_type = "relative": Mmax_increment, weight
		MFD_branches:
		unc_type = "absolute": a, b, Mmin, weight
		unc_type = "relative": b_increment, weight
		unc_type = "discretized": Mmin, binSize, Nvalues, weight = 1.0 (i.e., only 1 branch allowed)

		{'SourceModel01': {'sources': {'Source01': {'Mmax_models': {'Mmax01': {'MFD_models': {'mfd01': {'Mmin': 4.0,
																										'a': 2.4,
																										'b': 0.89,
																										'weight': 1.0}},
																				'MFD_unc_type': 'absolute',
																				'Mmax': 6.7,
																				'weight': 1.0}},
													'Mmax_unc_type': 'absolute'}},
							'weight': 1.0,
							'xml_file': 'TwoZoneModel.xml'}}

		If SourceModelObjs is specified, we will update the MFD in the models directly
		(this would typicaly occur for sources that have only 1 logic-tree branch,
		and only if Mmax unc_type is absolute and MFD unc_type is absolute or discretized)
		Source Model IDs and Source  IDs must of course correspond to those in the SourceModelObjs
		"""
		LogicTree.__init__(self, ID, [])
		self.source_system_dict = source_system_dict
		self.sourceModelObjs = sourceModelObjs
		self._validate_SourceModelObjs()
		self._construct_lt()

	def getSourceModelIDs(self):
		return self.source_system_dict.keys()

	def getSourceIDs(self, sourceModelID):
		return self.source_system_dict[sourceModelID]["sources"].keys()

	def getMmaxModels(self, sourceModelID, sourceID):
		return self.source_system_dict[sourceModelID]["sources"][sourceID]["Mmax_models"].items()

	def getMmaxUncType(self, sourceModelID, sourceID):
		return self.source_system_dict[sourceModelID]["sources"][sourceID]["Mmax_unc_type"]

	def getMFDModels(self, sourceModelID, sourceID, MmaxID):
		return self.source_system_dict[sourceModelID]["sources"][sourceID]["Mmax_models"][MmaxID]["MFD_models"].items()

	def getMFDUncType(self, sourceModelID, sourceID, MmaxID):
		return self.source_system_dict[sourceModelID]["sources"][sourceID]["Mmax_models"][MmaxID]["MFD_unc_type"]

	def _construct_lt(self):
		## First branching level: source models
		branchingLevelNum = 1
		branchSetNum = 1
		branches = []
		for sourceModelID in self.getSourceModelIDs():
			uncertaintyModel = self.source_system_dict[sourceModelID]["xml_file"]
			uncertaintyWeight = self.source_system_dict[sourceModelID]["weight"]
			branch = LogicTreeBranch(sourceModelID, uncertaintyModel, uncertaintyWeight)
			branches.append(branch)
		branchingLevelID = "lbl%03d" % branchingLevelNum
		branchSetID = branchingLevelID + "_lbs%03d" % branchSetNum
		branchSet = LogicTreeBranchSet(branchSetID, "sourceModel", branches)
		branchingLevel = LogicTreeBranchingLevel(branchingLevelID, [branchSet])
		self.branchingLevels.append(branchingLevel)

		## Second branching level: Mmax models
		branchingLevelNum = 2
		branchSetNum = 1
		branchingLevelID = "lbl%03d" % branchingLevelNum
		branchSets = []
		for i, sourceModelID in enumerate(self.getSourceModelIDs()):
			for sourceID in self.getSourceIDs(sourceModelID):
				# If source has only 1 Mmax model and 1 MFD model, update corresponding SourceObj
				if self.sourceModelObjs and self._is_single_branch_source(sourceModelID, sourceID):
					self._update_SourceObj(sourceModelID, sourceID)
				else:
					branches = []
					unc_type = self.getMmaxUncType(sourceModelID, sourceID)
					for Mmax_label, Mmax_model in self.getMmaxModels(sourceModelID, sourceID):
						branchID = "%s:%s:%s" % (sourceModelID, sourceID, Mmax_label)
						if unc_type == "absolute":
							uncertaintyModel = Mmax_model["Mmax"]
						elif unc_type == "relative":
							uncertaintyModel = Mmax_model["Mmax_increment"]
						branch = LogicTreeBranch(branchID, uncertaintyModel, Mmax_model["weight"])
						branches.append(branch)
					uncertaintyType = {"absolute": "maxMagGRAbsolute", "relative": "maxMagGRRelative"}[unc_type]
					branchSetID = branchingLevelID + ":%03d" % branchSetNum
					branchSet = LogicTreeBranchSet(branchSetID, uncertaintyType, branches, applyToBranches=str(sourceModelID), applyToSources=str(sourceID))
					branchSets.append(branchSet)
					branchSetNum += 1
		if len(branchSets):
			branchingLevel = LogicTreeBranchingLevel(branchingLevelID, branchSets)
			self.branchingLevels.append(branchingLevel)

		## Third branching level: MFD models
		branchingLevelNum = 3
		branchSetNum = 1
		branchingLevelID = "lbl%03d" % branchingLevelNum
		branchSets = []
		for i, sourceModelID in enumerate(self.getSourceModelIDs()):
			for sourceID in self.getSourceIDs(sourceModelID):
				if self.sourceModelObjs and self._is_single_branch_source(sourceModelID, sourceID):
					## We don't need to do anything here, the source object has already
					## been updated in the second branching level
					pass
				else:
					for Mmax_label, Mmax_model in self.getMmaxModels(sourceModelID, sourceID):
						branches = []
						applyToBranchID = "%s:%s:%s" % (sourceModelID, sourceID, Mmax_label)
						unc_type = self.getMFDUncType(sourceModelID, sourceID, Mmax_label)
						for mfd_label, mfd_model in self.getMFDModels(sourceModelID, sourceID, Mmax_label):
							branchID = "%s:%s:%s:%s" % (sourceModelID, sourceID, Mmax_label, mfd_label)
							if unc_type == "absolute":
								uncertaintyModel = "%s %s" % (mfd_model["a"], mfd_model["b"])
							elif unc_type == "relative":
								uncertaintyModel = str(mfd_model["b_increment"])
							branch = LogicTreeBranch(branchID, uncertaintyModel, mfd_model["weight"])
							branches.append(branch)
						uncertaintyType = {"absolute": "abGRAbsolute", "relative": "bGRRelative"}[unc_type]
						branchSetID = branchingLevelID + ":%03d" % branchSetNum
						branchSet = LogicTreeBranchSet(branchSetID, uncertaintyType, branches, applyToBranches=applyToBranchID)
						branchSets.append(branchSet)
						branchSetNum += 1
		if len(branchSets):
			branchingLevel = LogicTreeBranchingLevel(branchingLevelID, branchSets)
			self.branchingLevels.append(branchingLevel)

	def _is_single_branch_source(self, sourceModelID, sourceID):
		answer = False
		if len(self.getMmaxModels(sourceModelID, sourceID)) == 1:
			for Mmax_label, Mmax_model in self.getMmaxModels(sourceModelID, sourceID):
				Mmax_unc_type = self.getMmaxUncType(sourceModelID, sourceID)
				if Mmax_unc_type == "absolute" and len(self.getMFDModels(sourceModelID, sourceID, Mmax_label)) == 1:
						mfd_unc_type = self.getMFDUncType(sourceModelID, sourceID, Mmax_label)
						if mfd_unc_type in ("absolute", "discretized"):
							answer = True
		return answer

	def _update_SourceObj(self, sourceModelID, sourceID):
		sourceModelObj = [obj for obj in self.sourceModelObjs if obj.name == sourceModelID][0]
		sourceObj = [obj for obj in sourceModelObj.sourceObjs if obj.name == sourceID][0]
		for Mmax_label, Mmax_model in self.getMmaxModels(sourceModelID, sourceID):
			mfd_unc_type = self.getMFDUncType(sourceModelID, sourceID, Mmax_label)
			mfd_model = self.getMFDModels(sourceModelID, sourceID, Mmax_label)[0]
			if mfd_unc_type == "absolute":
				## TO DO: distribute a values over different rupture rate models if there is more than 1
				Mmax = Mmax_model["Mmax"]
				a, b = mfd_model["a"], mfd_model["b"]
				Mmin = mfd_model["Mmin"]
				mfdObj = TruncatedGutenbergRichter(a, b, Mmin, Mmax)
				sourceObj.ruptureRateModelObjs[0].mfdObj = mfdObj
			elif mfd_unc_type == "discretized":
				Mmin = mfd_model["Mmin"]
				binSize = mfd_model["binSize"]
				Nvalues = mfd_model["Nvalues"]
				mfdObj = EvenlyDiscretizedIncrementalMFD(binSize, Mmin, Nvalues)
				sourceObj.ruptureRateModelObjs[0].mfdObj = mfdObj

	def _validate_SourceModelObjs(self):
		if self.sourceModelObjs:
			if len(self.sourceModelObjs) != len(self.getSourceModelIDs):
				raise NRMLError("Wrong number of source model objects!")
			for sourceModelID in self.getSourceModelIDs():
				## Note that order in dictionary is not guaranteed
				sourceModelObj = [obj for obj in self.sourceModelObjs if obj.name == sourceModelID][0]
				sourceNames = self.source_system_dict[sourceModelID]["sources"].keys()
				if len(sourceNames) != len(sourceModelObj.sourceObjs):
					raise NRMLError("Wrong number of sources in source model %s!" % sourceModelID)

	def import_cfg(self, cfg_filespec):
		self.source_system_dict = ConfigObj(cfg_filespec)
		self._construct_lt()

	def import_json(self, json_filespec):
		f = open(json_filespec)
		self.source_system_dict = json.loads(f.read())
		f.close()
		self._construct_lt()

	def export_cfg(self, out_filespec):
		if type(self.source_system_dict) is dict:
			self.source_system_dict = ConfigObj(self.source_system_dict, indent_type="	")

		self.source_system_dict.filename = out_filespec
		self.source_system_dict.write()

	def export_json(self, out_filespec):
		f = open(out_filespec, "w")
		f.write(json.dumps(self.source_system_dict))
		f.close()

	def print_system_dict(self):
		pp = pprint.PrettyPrinter()
		if isinstance(self.source_system_dict, ConfigObj):
			pp.pprint(self.source_system_dict.dict())
		else:
			pp.pprint(self.source_system_dict)


class SeismicSourceSystem_v2(LogicTree):
	"""
	Class for constructing a seismic source system based on a dictionary that is
	nested to an arbitrary level
	Arguemnts:
		ID: identifier
		source_system_dict: source system dictionary, taking the following form:
	{'SourceModel01': {'sources': {'Source01': {'models': {'Mmax01': {'Mmax': 6.7,
																	'models': {'MFD01': {'Mmin': 4.0,
																						'a': 2.4,
																						'b': 0.89,
																						'weight': 0.5},
																				'MFD02': {'Mmin': 4.0,
																						'a': 2.4,
																						'b': 0.89,
																						'weight': 0.5}},
																	'unc_type': 'absolute',
																	'weight': 0.5},
															'Mmax02': {'Mmax': 6.7,
																	'models': {'MFD01': {'Mmin': 4.0,
																						'a': 2.4,
																						'b': 0.89,
																						'weight': 0.5},
																				'MFD02': {'Mmin': 4.0,
																						'a': 2.4,
																						'b': 0.89,
																						'weight': 0.5}},
																	'unc_type': 'absolute',
																	'weight': 0.5}},
												'unc_type': 'absolute',
												'uncertainty_levels': ['Mmax', 'MFD', 'Dip', 'Depth']},
									'Source02': ...
					'weight': 1.0,
					'xml_file': 'TwoZoneModel.xml'}
	'SourceModel02': ...}
	"""
	def __init__(self, ID, source_system_dict={}):
		## Initialize LogicTree with empty list of branching levels
		## We will add branching levels, branch sets, and branches with _construct_lt()
		LogicTree.__init__(self, ID, [])
		self._set_source_system_dict(source_system_dict)

	def _set_source_system_dict(self, source_system_dict):
		"""
		Set source_system_dict as the new source system dictionary,
		and reconstruct the logic tree. This method is called during initialization
		Argument:
			source_system_dict: source system dictionary
		"""
		if not isinstance(source_system_dict, ConfigObj):
			self.source_system_dict = ConfigObj(source_system_dict, indent_type="\t")
		else:
			self.source_system_dict = source_system_dict
		if len(source_system_dict) != 0:
			self._construct_lt()
			self.validate()

	def getNumberOfBranches(self, sourceModelID, sourceID):
		"""
		Return number of end branches for a given source
		Arguments:
			sourceModelID
			sourceID
		"""
		numBranches = 0
		source = self.source_system_dict[sourceModelID]["sources"][sourceID]
		for topLevelIDs in self.walk_ssd_end_branches(source["models"]):
			numBranches += 1
		return numBranches

	def getSourceModelIDs(self):
		"""
		Return list of source model IDs
		"""
		return self.source_system_dict.keys()

	def getSourceIDs(self, sourceModelID):
		"""
		Return list of source IDs in a given source model
		Arguments:
			sourceModelID
		"""
		return self.source_system_dict[sourceModelID]["sources"].keys()

	def getBranchByPath(self, sourceModelID, sourceID, branch_path):
		"""
		Return branch (dictionary) from a branch path
		Arguments:
			sourceModelID
			sourceID
			branch_path: list of IDs of consecutive uncertainty levels above
				and including the wanted branch
		"""
		branch = self.source_system_dict[sourceModelID]["sources"][sourceID]
		for ID in branch_path:
			branch = branch["models"][ID]
		return branch

	def getBranchModels(self, sourceModelID, sourceID, branch_path=[]):
		"""
		Return a list of models in a particular branch
		Arguments:
			sourceModelID
			sourceID
			branch_path: list of IDs of consecutive uncertainty levels above
				and including the wanted branch (default: [], corresponds to
				first uncertainty level or branch below a source)
		Return value: list of (key, value (= dictionary)) tuples
		"""
		branch = self.getBranchByPath(sourceModelID, sourceID, branch_path)
		return branch.get("models", {}).items()

	def getBranchUncertaintyType(self, sourceModelID, sourceID, branch_path=[]):
		"""
		Return uncertainty type of a particular branch
		Arguments:
			sourceModelID
			sourceID
			branch_path: list of IDs of consecutive uncertainty levels above
				and including the wanted branch (default: [], corresponds to
				first uncertainty level or branch below a source). Use
				self.walk_ssd_branches() to generate a list of topLevelIDs
		Return value: string
		"""
		branch = self.source_system_dict[sourceModelID]["sources"][sourceID]
		for ID in branch_path:
			branch = branch["models"][ID]
		return branch.get("unc_type", "")

	def getMaxNumUncertaintyLevels(self):
		"""
		Return maximum number of uncertainty levels in the source system
		"""
		numBranchingLevels = 0
		for sourceModelID in self.getSourceModelIDs():
			for sourceID in self.getSourceIDs(sourceModelID):
				source = self.source_system_dict[sourceModelID]["sources"][sourceID]
				if len(source["uncertainty_levels"]) > numBranchingLevels:
					numBranchingLevels = len(source["uncertainty_levels"])
		return numBranchingLevels

	def walk_ssd_branches(self, topLevel, max_depth=2, current_depth=0):
		"""
		Recursively walk along the uncertainty levels and their associated models,
		from a given top level down to the maximum depth specified, and generate a list
		of consecutive top-level IDs, e.g.,
		max_depth=0: []
		max_depth=1: ["Mmax01"], ["Mmax02"], ...
		max_depth=2: ["Mmax01", "MFD01"], ["Mmax01", "MFD02"], ["Mmax02", "MFD01"], ...
		max_depth=3: ...

		Note that levels that do not contain a "models" keyword are not considered
		as top levels!

		Arguments:
			topLevel: any uncertainty level that has a "models" subdictionary,
				usually one starts from a particular source
			max_depth: maximum depth to drill down to (default: 2)
			current_depth: this parameter is only used for recursion, so that the
				function can keep track of its current depth (default: 0)
		"""
		if max_depth == 0:
			yield []
		else:
			for key, value in topLevel.items():
				if value.has_key("models"):
					if current_depth < (max_depth - 1):
						for subkey in self.walk_ssd_branches(value["models"], max_depth=max_depth, current_depth=current_depth+1):
							yield [key] + subkey
					elif current_depth < max_depth:
						yield [key]
				## Uncommenting the following 2 lines will result in levels that
				## do not have a "models" key to be returned as well
				#else:
				#	yield [key]

	def walk_ssd_end_branches(self, topLevel):
		"""
		Recursively walk along the uncertainty levels and their associated models,
		down from a given top level, and generate a list of consecutive top-level
		IDs, but only for end branches
		Arguments:
			topLevel: any uncertainty level that has a "models" subdictionary,
		"""
		for key, value in topLevel.items():
			if value.has_key("models"):
				for subkey in self.walk_ssd_end_branches(value["models"]):
					yield [key] + subkey
			else:
				yield [key]

	def _construct_lt(self):
		"""
		Construct logic tree consisting of branching levels, branch sets, and branches
		"""
		## First branching level: source models
		branchingLevelNum = 1
		branchSetNum = 1
		branches = []
		for sourceModelID in self.getSourceModelIDs():
			uncertaintyModel = self.source_system_dict[sourceModelID]["xml_file"]
			uncertaintyWeight = self.source_system_dict[sourceModelID]["weight"]
			branchID = sourceModelID
			branch = LogicTreeBranch(branchID, uncertaintyModel, uncertaintyWeight)
			branches.append(branch)
		branchingLevelID = "lbl%03d" % branchingLevelNum
		branchSetID = branchingLevelID + "--lbs%03d" % branchSetNum
		branchSet = LogicTreeBranchSet(branchSetID, "sourceModel", branches)
		branchingLevel = LogicTreeBranchingLevel(branchingLevelID, [branchSet])
		self.branchingLevels.append(branchingLevel)

		## Create empty branching levels for each uncertainty level
		for i in range(self.getMaxNumUncertaintyLevels()):
			branchingLevelID = "lbl%03d" % (i + 2)
			branchingLevel = LogicTreeBranchingLevel(branchingLevelID, [])
			self.branchingLevels.append(branchingLevel)

		## List holding number of branch sets for each branching level
		branchSetNum = [1] * self.getMaxNumUncertaintyLevels()

		## Loop over source models
		for sourceModelID in self.getSourceModelIDs():
			## Loop over sources
			for sourceID in self.getSourceIDs(sourceModelID):
				source = self.source_system_dict[sourceModelID]["sources"][sourceID]
				## Loop over uncertainty levels (may be different for each source)
				for uncertainty_level_depth, branching_level_type in enumerate(source["uncertainty_levels"]):
					branchingLevelNum = uncertainty_level_depth + 2
					branchingLevelID = "lbl%03d" % branchingLevelNum
					#print branchingLevelID
					## Loop over topLevelIDs above this uncertainty level
					for topLevelIDs in self.walk_ssd_branches(source["models"], max_depth=uncertainty_level_depth):
						#print uncertainty_level_depth, topLevelIDs
						if len(topLevelIDs) == uncertainty_level_depth:
							## Create a new branch set for this uncertainty level
							branchSetID = branchingLevelID + "--lbs%03d" % branchSetNum[uncertainty_level_depth]
							#print branchSetID
							unc_type = self.getBranchUncertaintyType(sourceModelID, sourceID, topLevelIDs)
							if branching_level_type == "Mmax":
								uncertaintyType = {"absolute": "maxMagGRAbsolute", "relative": "maxMagGRRelative"}[unc_type]
							elif branching_level_type == "MFD":
								uncertaintyType = {"absolute": "abGRAbsolute", "relative": "bGRRelative"}[unc_type]

							if uncertainty_level_depth == 0:
								applyToBranches = str(sourceModelID)
								applyToSources = str(sourceModelID) + "--" + str(sourceID)
							else:
								applyToBranches = "%s--%s--" % (sourceModelID, sourceID) + "--".join(topLevelIDs)
								applyToSources = str(sourceModelID) + "--" + str(sourceID)
							branchSet = LogicTreeBranchSet(branchSetID, uncertaintyType, [], applyToBranches=applyToBranches, applyToSources=applyToSources)
							self.branchingLevels[branchingLevelNum - 1].branchSets.append(branchSet)

							## Loop over the different uncertainty models in this uncertainty level
							## and add them as branches to the current branching set
							for label, model in self.getBranchModels(sourceModelID, sourceID, topLevelIDs):
								#print "\t%s" % label
								branchID = "%s--%s" % (sourceModelID, sourceID) + "".join(["--" + ID for ID in topLevelIDs]) + "--%s" % label
								if branching_level_type == "Mmax":
									if unc_type == "absolute":
										uncertaintyModel = model["Mmax"]
									elif unc_type == "relative":
										uncertaintyModel = model["Mmax_increment"]
								elif branching_level_type == "MFD":
									if unc_type == "absolute":
										## TO DO: take into account possibility of multiple values for 1 source
										uncertaintyModel = "%s %s" % (model["a"], model["b"])
									elif unc_type == "relative":
										uncertaintyModel = str(model["b_increment"])
								branch = LogicTreeBranch(branchID, uncertaintyModel, model["weight"])
								branchSet.branches.append(branch)

							## Increment branch-set number for this uncertainty level
							branchSetNum[uncertainty_level_depth] += 1

	def import_cfg(self, cfg_filespec):
		"""
		Import source system dictionary from config file
		Arguments:
			cfg_filespec: full path to config file
		"""
		source_system_dict = ConfigObj(cfg_filespec)
		self._set_source_system_dict(source_system_dict)

	def import_json(self, json_filespec):
		"""
		Import source system dictionary from json file
		Arguments:
			cfg_filespec: full path to json file
		"""
		f = open(json_filespec)
		source_system_dict = json.loads(f.read())
		f.close()
		self._set_source_system_dict(source_system_dict)

	def export_cfg(self, out_filespec):
		"""
		Export source system dictionary in config format
		Arguments:
			out_filespec: full path to output file
		"""
		self.source_system_dict.filename = out_filespec
		self.source_system_dict.write()

	def export_json(self, out_filespec):
		"""
		Export source system dictionary in json format
		Arguments:
			out_filespec: full path to output file
		"""
		f = open(out_filespec, "w")
		f.write(json.dumps(self.source_system_dict))
		f.close()

	def print_system_dict(self):
		"""
		Pretty print the source system dictionary
		"""
		pp = pprint.PrettyPrinter()
		if isinstance(self.source_system_dict, ConfigObj):
			pp.pprint(self.source_system_dict.dict())
		else:
			pp.pprint(self.source_system_dict)


#SeismicSourceSystem = SeismicSourceSystem_v2


class SymmetricRelativeSeismicSourceSystem(LogicTree):
	"""
	Class for constructing a symmetric seismic source system containing only
	relative uncertainties

	Arguments:
		ID: identifier
		branching_level_dicts: list of dictionaries for each branching level,
			each with the following keys: "uncertainty_type", "models", "weights",
			and "labels"
		explicit: optional argument indicating whether the logic tree will
			be written explicitly (each branchset repeated for each branch in
			the previous branching level) or not (default: False)
	"""
	def __init__(self, ID, branching_level_dicts, explicit=False):
		LogicTree.__init__(self, ID, [])
		self.branching_level_dicts = branching_level_dicts
		if explicit:
			self._construct_lt_explicit()
		else:
			self._construct_lt()

	def _construct_lt(self):
		for i in range(len(self.branching_level_dicts)):
			branches = []
			for j in range(len(self.branching_level_dicts[i]["models"])):
				uncertaintyModel = self.branching_level_dicts[i]["models"][j]
				uncertaintyWeight = self.branching_level_dicts[i]["weights"][j]
				branchID = self.branching_level_dicts[i]["labels"][j]
				branch = LogicTreeBranch(branchID, uncertaintyModel, uncertaintyWeight)
				branches.append(branch)
			branchingLevelID = "bl%03d" % i
			branchSetID = branchingLevelID + "--bs%03d" % i
			uncertainty_type = self.branching_level_dicts[i]["uncertainty_type"]
			if (i == 0 and uncertainty_type == "sourceModel") or (i > 0 and uncertainty_type in("maxMagGRRelative", "bGRRelative")):
				branchSet = LogicTreeBranchSet(branchSetID, uncertainty_type, branches)
				branchingLevel = LogicTreeBranchingLevel(branchingLevelID, [branchSet])
				self.branchingLevels.append(branchingLevel)
			else:
				raise Exception("uncertainty %s not allowed in branching level %d" % (uncertainty_type, i))

	def _construct_lt_explicit(self):
		num_models_per_branching_level = []
		branchIDs_per_branching_level = []
		## Loop over branching levels
		for i in range(len(self.branching_level_dicts)):
			branches = []
			branchIDs = []
			branchIDs_per_branching_level.append([])
			## Collect information for branches in this level
			for j in range(len(self.branching_level_dicts[i]["models"])):
				uncertaintyModel = self.branching_level_dicts[i]["models"][j]
				uncertaintyWeight = self.branching_level_dicts[i]["weights"][j]
				branchID = self.branching_level_dicts[i]["labels"][j]
				branchIDs.append(branchID)
				branch = LogicTreeBranch(branchID, uncertaintyModel, uncertaintyWeight)
				branches.append(branch)
			## Concatenate branch ID's with those of the previous branching level
			if i == 0:
				branchIDs_per_branching_level[-1].extend(branchIDs)
			else:
				for toplevel_branchID in branchIDs_per_branching_level[-2]:
					for branchID in branchIDs:
						branchIDs_per_branching_level[-1].append(toplevel_branchID + "--" + branchID)

			branchingLevelID = "bl%03d" % i
			uncertainty_type = self.branching_level_dicts[i]["uncertainty_type"]
			if (i == 0 and uncertainty_type == "sourceModel") or (i > 0 and uncertainty_type in("maxMagGRRelative", "bGRRelative")):
				## Build branch sets for all branches in the previous branching level
				branchSets = []
				for k in range(int(numpy.multiply.reduce(num_models_per_branching_level))):
					branchSetID = branchingLevelID + "--bs%03d" % k
					## Pick the correct branch ID's and apply_to_branch ID's
					if i == 0:
						apply_to_branches = ""
					else:
						apply_to_branches = branchIDs_per_branching_level[-2][k]
						for l, branch in enumerate(branches):
							branch.ID = branchIDs_per_branching_level[-1][k*len(branches)+l]
					branchSet = LogicTreeBranchSet(branchSetID, uncertainty_type, copy.deepcopy(branches), applyToBranches=apply_to_branches)
					branchSets.append(branchSet)
				branchingLevel = LogicTreeBranchingLevel(branchingLevelID, branchSets)
				self.branchingLevels.append(branchingLevel)
			else:
				raise Exception("uncertainty %s not allowed in branching level %d" % (uncertainty_type, i))
			num_models_per_branching_level.append(j+1)


def create_basic_seismicSourceSystem(sourceModels, weights=[]):
	"""
	Creeate a basic SeismicSourceSystem with a single branching level
	corresponding to the specified sourceModels, without logic-tree
	uncertainties
	Arguments:
		sourceModels: list of SourceModel objects
		weights: list of weights for each source model
		(default: [], will attribute equal weight to each model)
	Return value:
		source_system_lt: SeismicSourceSystem object
	"""
	ssd = ConfigObj(indent_type="	")

	if len(weights) == 0:
#		weights = numpy.ones(len(sourceModels)) / len(sourceModels)
		weights = get_uniform_weights(len(sourceModels))
	for sourceModel, weight in zip(sourceModels, weights):
		sourceModelID = sourceModel.name
		ssd[sourceModelID] = {}
		ssd[sourceModelID]["weight"] = weight
		ssd[sourceModelID]["xml_file"] = sourceModel.name + ".xml"
		ssd[sourceModelID]["sources"] = {}
		for source in sourceModel.sources:
			sourceID = source.source_id
			ssd[sourceModelID]["sources"][sourceID] = {}
			ssd[sourceModelID]["sources"][sourceID]["uncertainty_levels"] = []
			ssd[sourceModelID]["sources"][sourceID]["models"] = {}

	source_system_lt = SeismicSourceSystem_v2("lt0", ssd)
	return source_system_lt


