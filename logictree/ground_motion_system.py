import openquake.hazardlib as nhlib

from logictree import LogicTreeBranch, LogicTreeBranchSet, LogicTreeBranchingLevel, LogicTree



class GroundMotionSystem(LogicTree):
	"""
	Class representing Ground Motion System
	Inherits from LogicTree class

	:param ID:
		string, identifier
	:param gmpe_system_def:
		dictionary with tectonic region types as keys and instances of
		:class:`PMF` as values, containing GMPE models and their weights
	"""
	def __init__(self, ID, gmpe_system_def):
		LogicTree.__init__(self, ID, [])
		self.gmpe_system_def = gmpe_system_def
		self._construct_lt()

	def _construct_lt(self):
		"""
		Construct logic tree
		This method is called during initialization
		"""
		for i, tectonicRegionType in enumerate(self.tectonicRegionTypes):
			branchSetID = "lbs01"
			branchSet = LogicTreeBranchSet(branchSetID, "gmpeModel", [], applyToTectonicRegionType=tectonicRegionType)
			branchingLevelID = "lbl%02d" % i
			branchingLevel = LogicTreeBranchingLevel(branchingLevelID, [branchSet])
			for j, (weight, gmpe) in enumerate(self.gmpe_system_def[tectonicRegionType].data):
				branchID = "lbl%02d_lb%02d" % (i, j)
				#branch = LogicTreeBranch(branchID, nhlib.gsim.get_available_gsims()[gmpe], weight)
				branch = LogicTreeBranch(branchID, gmpe, weight)
				branchSet.branches.append(branch)
			self.branchingLevels.append(branchingLevel)

	def validate(self):
		"""
		Validate logic-tree structure
		"""
		LogicTree.validate(self)
		for branchingLevel in self.branchingLevels:
			if len(branchingLevel.branchSets) > 1:
				raise NRMLError("Branching Level %s: GroundMotionSystem can contain only one branchSet per branchingLevel!" % branchingLevel.ID)
			if branchingLevel.branchSets[0].uncertaintyType != "gmpeModel":
				raise NRMLError("Branching Level %s: GroundMotionSystem only allows uncertainties of type gmpeModel" % branchingLevel.ID)

	@property
	def tectonicRegionTypes(self):
		"""
		Return list of tectonic region types in the ground motion system
		"""
		return self.gmpe_system_def.keys()

	def get_gmpe_names(self, tectonicRegionType):
		"""
		Return list of GMPE model names for a specific tectonic region type

		:param tectonicRegionType:
			string, tectonic region type
		"""
		return self.gmpe_system_def[tectonicRegionType].values

	def get_gmpe_weights(self, tectonicRegionType):
		"""
		Return list of GMPE model weights for a specific tectonic region type

		:param tectonicRegionType:
			string, tectonic region type
		"""
		return self.gmpe_system_def[tectonicRegionType].weights

	def get_unique_gmpe_names(self):
		"""
		Return sorted list of all GMPE models in the ground motion system
		"""
		gmpe_models = set()
		for trt in self.gmpe_system_def.keys():
			for gmpe in self.get_gmpe_names(trt):
				gmpe_models.add(gmpe)
		return sorted(gmpe_models)

	def get_optimized_system(self, source_models):
		"""
		Return an optimized ground motion system where unused tectonic
		region types are removed

		:param source_models:
			list with instances of :class:`SourceModel`

		:return:
			instance of :class:`GroundMotionSystem`
		"""
		optimized_gmpe_system_def = {}
		used_trts = set()
		for source_model in source_models:
			for src in source_model.sources:
				used_trts.add(src.tectonic_region_type)
		for trt in used_trts:
			optimized_gmpe_system_def[trt] = self.gmpe_system_def[trt]
		return GroundMotionSystem(self.ID, optimized_gmpe_system_def)

