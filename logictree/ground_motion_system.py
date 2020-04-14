"""
Ground-motion logic tree
"""

from __future__ import absolute_import, division, print_function, unicode_literals


from .logictree import *



class GroundMotionSystem(LogicTree):
	"""
	Class representing Ground Motion System
	Inherits from LogicTree class

	A ground motion system has the following constraints:
	- Only one branch set can be defined per branching level.
	- Each branch set must define uncertainties of type gmpeModel.
	- All branch sets must define the applyToTectonicRegionType attribute.
		This is the only attribute allowed.
	- Each branch set must refer to a different tectonic region type.

	:param id:
		string, identifier
	:param gmpe_system_def:
		dictionary with tectonic region types as keys and instances of
		:class:`PMF` as values, containing GMPE models and their weights
	:param use_short_names:
		bool, whether to label branches using short names or full
		names of GMPEs (default: True)
	"""
	def __init__(self, id, gmpe_system_def, use_short_names=True):
		LogicTree.__init__(self, id, [])
		self.gmpe_system_def = gmpe_system_def
		self.use_short_names = use_short_names
		self._construct_lt()
		self.connect_branches()

	def __repr__(self):
		return '<GroundMotionSystem "%s">' % self.id

	def _construct_lt(self):
		"""
		Construct logic tree
		This method is called during initialization

		:param use_short_names:
			bool, whether to label branches using short names or full
			names of GMPEs (default: True)
		"""
		from .. import gsim
		for i, trt in enumerate(self.tectonic_region_types):
			branchingLevelID = "bl%02d" % i
			#branchSetID = "%s_bs01" % branchingLevelID
			branchSetID = "".join([word[0].upper() for word in trt.split()])
			branch_set = LogicTreeBranchSet.from_pmf(branchSetID,
												self.gmpe_system_def[trt],
												applyToTectonicRegionType=trt)
			## Rename branch ID's:
			for branch, gmpe_name in zip(branch_set.branches,
										self.gmpe_system_def[trt].gmpe_names):
				try:
					gmpe = getattr(gsim, gmpe_name)()
				except AttributeError:
					## OQ GMPEs that are not explicitly defined
					## in rshalig.gsim.gmpe.oqhazlib_gmpe
					gmpe = gsim.get_oq_gsim(gmpe_name)
				else:
					gmpe_name = {True: gmpe.short_name,
								False: gmpe.name}[self.use_short_names]
				branch.branch_id = "%s--%s" % (branchSetID, gmpe_name)
			branching_level = LogicTreeBranchingLevel(branchingLevelID, [branch_set])
			self.branching_levels.append(branching_level)

	def validate(self):
		"""
		Validate logic-tree structure
		"""
		LogicTree.validate(self)
		for branchingLevel in self.branchingLevels:
			if len(branchingLevel.branchSets) > 1:
				raise NRMLError("Branching Level %s: GroundMotionSystem can contain only one branchSet per branchingLevel!"
								% branchingLevel.ID)
			if branchingLevel.branchSets[0].uncertaintyType != "gmpeModel":
				raise NRMLError("Branching Level %s: GroundMotionSystem only allows uncertainties of type gmpeModel"
								% branchingLevel.ID)

	@property
	def tectonic_region_types(self):
		"""
		Return list of tectonic region types in the ground motion system
		"""
		return self.gmpe_system_def.keys()

	def get_gmpe_names(self, trt):
		"""
		Return list of GMPE model names for a specific tectonic region type

		:param trt:
			string, tectonic region type
		"""
		return self.gmpe_system_def[trt].values

	def get_gmpe_weights(self, trt):
		"""
		Return list of GMPE model weights for a specific tectonic region type

		:param trt:
			string, tectonic region type
		"""
		return self.gmpe_system_def[trt].weights

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
		return GroundMotionSystem(self.id, optimized_gmpe_system_def,
									use_short_names=self.use_short_names)

