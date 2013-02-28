import openquake.hazardlib as nhlib

from logictree import LogicTreeBranch, LogicTreeBranchSet, LogicTreeBranchingLevel, LogicTree


class GroundMotionSystem(LogicTree):
	"""
	Class representing Ground Motion System
	Inherits from LogicTree class
	Arguments:
		ID: identifier
		gmpe_system_dict: dictionary with tectonic region types as keys and
			lists of GMPE models known by OQ as values
	"""
	def __init__(self, ID, gmpe_system_dict):
		LogicTree.__init__(self, ID, [])
		self.gmpe_system_dict = gmpe_system_dict
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
			for j, (gmpe, weight) in enumerate(self.gmpe_system_dict[tectonicRegionType].items()):
				branchID = "lbl%02d_lb%02d" % (i, j)
				branch = LogicTreeBranch(branchID, nhlib.gsim.get_available_gsims()[gmpe], weight)
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
		return self.gmpe_system_dict.keys()

	@property
	def GMPEModels(self, tectonicRegionType):
		"""
		Return list of GMPE models for a specific tectonic region type
		Argument:
			tectonicRegionType: tectonic region type (string)
		"""
		return self.gmpe_system_dict[tectonicRegionType].keys()

	@property
	def GMPEModelWeights(self, tectonicRegionType):
		"""
		Return list of GMPE model weights for a specific tectonic region type
		Argument:
			tectonicRegionType: tectonic region type (string)
		"""
		return self.gmpe_system_dict[tectonicRegionType].values()

	@property
	def AllGMPEModels(self):
		"""
		Return sorted list of all GMPE models in the ground motion system
		"""
		gmpe_models = set()
		for trt in self.gmpe_system_dict.keys():
			for gmpe in self.gmpe_system_dict[trt]:
				gmpe_models.add(gmpe)
		gmpe_models = list(gmpe_models)
		gmpe_models.sort()
		return gmpe_models


def optimize_gmpeSystemDict(gmpeSystemDict, sourceModels):
	"""
	Remove unused tectonicRegionTypes from gmpeSystemDict based on sourceModels
	"""
	optimized_gmpeSystemDict = {}
	usedTectonicRegionTypes = []
	for sourceModel in sourceModels:
		for sourceObj in sourceModel.sources:
			if not sourceObj.tectonic_region_type in usedTectonicRegionTypes:
				usedTectonicRegionTypes.append(sourceObj.tectonic_region_type)
	for tectonicRegionType in usedTectonicRegionTypes:
			optimized_gmpeSystemDict[tectonicRegionType] = gmpeSystemDict[tectonicRegionType]

	return optimized_gmpeSystemDict


def optimize_GroundMotionSystem(gmpeSystem, sourceModels):
	"""
	Remove unused tectonicRegionTypes from gmpeSystem based on sourceModels
	"""
	optimized_gmpeSystemDict = optimize_gmpeSystemDict(gmpeSystem.gmpe_system_dict, sourceModels)
	return GroundMotionSystem(gmpeSystem.ID, optimized_gmpeSystemDict)


