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


class SeismicSourceSystem(LogicTree):
	def __init__(self, id, branching_levels):
		super(SeismicSourceSystem, self).__init__(id, branching_levels)

	@classmethod
	def from_independent_uncertainty_levels(cls, sss_id, source_model_pmf, unc2_pmf_dict, unc3_pmf_dict):
		"""
		Construct seismic source system according to the original
		OpenQuake specification from parameters specifying the three
		different uncertainty levels (source model, Mmax, and MFD), where it
		is assumed that the Mmax and MFD uncertainties do not depend on each other.

		:param source_model_pmf:
			instance of :class:`SourceModelPMF`
		:param unc2_pmf_dict:
			dictionary specifying 2nd uncertainty class, taking the following form:
			{sm_nameN: {src_id_N: instance of :class:`MmaxPMF` or :class:`MFDPMF`}}
		:param unc3_pmf_dict:
			dictionary specifying 3rd uncertainty class, taking the following form:
			{sm_nameN: {src_id_N: instance of :class:`MFDPMF` or :class:`MmaxPMF`}}

		Note: if sm_name or src_id are None, it is assumed that the same
		PMF's are assigned to each source model or source, respectively
		"""
		branching_levels = []
		## This dictionary will contain lists of branch id's in each branching
		## level for the different source models
		bl_branch_ids = {}

		## Root branch set corresponds to source models
		branching_level_id = "bl00"
		#branch_set_id = "%s_bs00" % branching_level_id
		branch_set_id = "SM"
		branch_set = LogicTreeBranchSet.from_PMF(branch_set_id, source_model_pmf)
		branching_level = LogicTreeBranchingLevel(branching_level_id, [branch_set])
		branching_levels.append(branching_level)
		root_branch_ids = [branch.branch_id for branch in branch_set]
		source_model_names = [sm.name for sm in source_model_pmf.source_models]

		## Second uncertainty class
		for sm_name, src_unc2_pmf_dict in unc2_pmf_dict.items():
			first_pmf = src_unc2_pmf_dict[src_unc2_pmf_dict.keys()[0]]
			if isinstance(first_pmf, MmaxPMF):
				unc_type = "Mmax"
			else:
				unc_type = "MFD"
			bl_branch_ids[sm_name] = []
			if sm_name is None:
				## Source model unspecified, sources must be unspecified too
				## Only one branching level required
				applyToBranches = []
				applyToSources = []
				assert src_unc2_pmf_dict.keys() == [None], "If source model is unspecified, then sources must be unspecified too"
				branching_level_id = "bl01"
				# TODO: check branching_level_id
				#branch_set_id = "%s_bs00" % branching_level_id
				branch_set_id = "None_None_%s" % unc_type
				branch_set = LogicTreeBranchSet.from_PMF(branch_set_id, src_unc2_pmf_dict[None], applyToBranches=applyToBranches, applyToSources=applyToSources)
				branching_level = LogicTreeBranchingLevel(branching_level_id, [branch_set])
				branching_levels.append(branching_level)
				bl_branch_ids[sm_name].append([branch.branch_id for branch in branch_set])
			else:
				## Source model specified
				sm_index = source_model_names.index(sm_name)
				sm = source_model_pmf.source_models[sm_index]
				sm_src_ids = [src.source_id for src in sm]
				for i, src_id in enumerate(src_unc2_pmf_dict.keys()):
					if i == 0:
						applyToBranches = [root_branch_ids[sm_index]]
					else:
						applyToBranches = bl_branch_ids[sm_name][-1]
					if src_id is None:
						## Sources unspecified
						## Only one branching level required for this source model
						applyToSources = []
					else:
						applyToSources = [src_id]
					branching_level_nr = 1 + i
					branching_level_id = "bl%02d" % branching_level_nr
					try:
						branch_set_nr = len(branching_levels[branching_level_nr])
					except:
						branch_set_nr = 0
					#branch_set_id = "%s_bs%02d" % (branching_level_id, branch_set_nr)
					branch_set_id = "%s_%s_%s" % (sm_name, src_id, unc_type)
					branch_set = LogicTreeBranchSet.from_PMF(branch_set_id, src_unc2_pmf_dict[src_id], applyToBranches=applyToBranches, applyToSources=applyToSources)
					try:
						branching_level = branching_levels[branching_level_nr]
					except IndexError:
						branching_level = LogicTreeBranchingLevel(branching_level_id, [branch_set])
						branching_levels.append(branching_level)
					else:
						branching_level.branch_sets.append(branch_set)
					bl_branch_ids[sm_name].append([branch.branch_id for branch in branch_set])
					#print sm_name, src_id, branching_level_nr, len(bl_branch_ids[sm_name]), len(branching_levels)

		## Third uncertainty class
		for sm_name, src_unc3_pmf_dict in unc3_pmf_dict.items():
			first_pmf = src_unc3_pmf_dict[src_unc3_pmf_dict.keys()[0]]
			if isinstance(first_pmf, MmaxPMF):
				unc_type = "Mmax"
			else:
				unc_type = "MFD"
			if sm_name is None:
				## Source model unspecified, sources must be unspecified too
				## Only one branching level required
				applyToSources = []
				assert src_unc3_pmf_dict.keys() == [None], "If source model is unspecified, then sources must be unspecified too"
				for prev_level_sm_name in bl_branch_ids.keys():
					if len(bl_branch_ids) == 1:
						applyToBranches = []
						branch_set_id = "%s_None_%s" % (prev_level_sm_name, unc_type)
					else:
						other_sm_names = bl_branch_ids.keys()
						other_sm_names.remove(prev_level_sm_name)
						num_branching_levels = np.array([len(bl_branch_ids[other_sm_name]) for other_sm_name in other_sm_names])
						if (num_branching_levels == len(bl_branch_ids[prev_level_sm_name])).all():
							## Same branching level for different source models
							applyToBranches = []
							branch_set_id = "None_None_%s" % (unc_type)
						else:
							## Branching level different for different source models
							applyToBranches = bl_branch_ids[prev_level_sm_name][-1]
							branch_set_id = "%s_None_%s" % (prev_level_sm_name, unc_type)

					branching_level_nr = len(bl_branch_ids[prev_level_sm_name]) + 1
					branching_level_id = "bl%02d" % branching_level_nr
					try:
						branch_set_nr = len(branching_levels[branching_level_nr])
					except:
						branch_set_nr = 0
					#branch_set_id = "%s_bs%02d" % (branching_level_id, branch_set_nr)
					branch_set = LogicTreeBranchSet.from_PMF(branch_set_id, src_unc3_pmf_dict[None], applyToBranches=applyToBranches, applyToSources=applyToSources)
					try:
						branching_level = branching_levels[branching_level_nr]
					except IndexError:
						branching_level = LogicTreeBranchingLevel(branching_level_id, [branch_set])
						branching_levels.append(branching_level)
					else:
						branching_level.branch_sets.append(branch_set)
					if applyToBranches == [] and branch_set_nr == 0:
						break
					else:
						bl_branch_ids[prev_level_sm_name].append([branch.branch_id for branch in branch_set])
			else:
				assert unc2_pmf_dict.keys() != [None], "If sources are specified, then source model must be specified in the previous uncertainty level"
				## Source model specified
				last_branching_level_nr = len(bl_branch_ids[sm_name])
				sm_index = source_model_names.index(sm_name)
				sm = source_model_pmf.source_models[sm_index]
				sm_src_ids = [src.source_id for src in sm]
				for i, src_id in enumerate(src_unc3_pmf_dict.keys()):
					applyToBranches = bl_branch_ids[sm_name][-1]
					if src_id is None:
						## Sources unspecified
						## Only one branching level required for this source model
						applyToSources = []
					else:
						applyToSources = [src_id]
					branching_level_nr = last_branching_level_nr + i + 1
					branching_level_id = "bl%02d" % branching_level_nr
					try:
						branch_set_nr = len(branching_levels[branching_level_nr])
					except:
						branch_set_nr = 0
					#branch_set_id = "%s_bs%02d" % (branching_level_id, branch_set_nr)
					branch_set_id = "%s_%s_%s" % (sm_name, src_id, unc_type)
					branch_set = LogicTreeBranchSet.from_PMF(branch_set_id, src_unc3_pmf_dict[src_id], applyToBranches=applyToBranches, applyToSources=applyToSources)
					try:
						branching_level = branching_levels[branching_level_nr]
					except IndexError:
						branching_level = LogicTreeBranchingLevel(branching_level_id, [branch_set])
						branching_levels.append(branching_level)
						#bl_branch_ids[sm_name].append([branch.branch_id for branch in branch_set])
					else:
						branching_level.branch_sets.append(branch_set)
						#bl_branch_ids[sm_name][branching_level_nr].extend([branch.branch_id for branch in branch_set])
					bl_branch_ids[sm_name].append([branch.branch_id for branch in branch_set])
					#print sm_name, src_id, branching_level_nr, len(bl_branch_ids[sm_name]), len(branching_levels)

		source_model_lt = SeismicSourceSystem(sss_id, branching_levels)
		source_model_lt.connect_branches()
		return source_model_lt

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


