# -*- coding: utf-8 -*-

"""
Classes representing NRML elements used in OpenQuake logic trees
(seismic source systems and ground motion systems). Every class has methods
to generate xml elements that can be used to build a NRML document
"""

import numpy as np
import pprint
from lxml import etree
from decimal import Decimal

import openquake.engine.input.logictree as oqlt
from ..nrml import ns
from ..nrml.common import *
from ..pmf import *


### Logic Tree elements

# TODO: elaborate validation

class LogicTreeBranch(oqlt.Branch):
	"""
	Class representing a logic-tree branch

	:param branch_id:
		string, identifier
	:param value:
		single value (string, float, tuple) or dictionary {src_id: value}
		identifying an uncertainty model; the content varies with the
		uncertainty_type attribute value of the parent branch_set.
		If value is a dictionary, this implies that uncertainties for
		the different sources (keys) are correlated.
		Evidently, this is also the case if value is a single value, and
		the applyToSources property of the parent branchset contains more
		than 1 source ID (or is an empty list).
	:param weight:
		Decimal, specifying the probability/weight associated to the value
	"""
	def __init__(self, branch_id, weight, value, parent_branchset=None):
		super(LogicTreeBranch, self).__init__(branch_id, weight, value)
		self.parent_branchset = parent_branchset

	def validate(self):
		pass

	def create_xml_element(self):
		"""
		Create xml element (NRML logicTreeBranch element)
		"""
		lb_elem = etree.Element(ns.LOGICTREE_BRANCH, branchID=self.branch_id)
		um_elem = etree.SubElement(lb_elem, ns.UNCERTAINTY_MODEL)
		if hasattr(self.value, '__iter__'):
			value = "  ".join(map(str, self.value))
		else:
			value = self.value
		um_elem.text = xmlstr(value)
		uw_elem = etree.SubElement(lb_elem, ns.UNCERTAINTY_WEIGHT)
		uw_elem.text = str(self.weight)
		return lb_elem


class LogicTreeBranchSet(oqlt.BranchSet):
	"""
	Class representing a logic-tree branch set.
	A LogicTreeBranchSet defines a particular epistemic uncertainty inside
	a branching level. It identifies a collection of LogicTreeBranch objects.
	There is no restriction on the number of branches in a branch set, as
	long as their uncertainty weights sum to 1.0
	Arguments:
		id: identifier
		uncertainty_type: required attribute
			Possible values for the uncertainty_type attribute are:
			- gmpeModel: identifying epistemic uncertainties on ground motion
				prediction equations
			- sourceModel: identifying epistemic uncertainties on source models
			- maxMagGRRelative: identifying epistemic uncertainties (relative:
				that is increments) to be added (or subtracted, depending on the
				sign of the increment) to the Gutenberg-Richter maximum magnitude value.
			- bGRRelative: identifying epistemic uncertainties (relative) to be
				applied to the Gutenberg-Richter b value.
			- abGRAbsolute:identifying epistemic uncertainties (absolute: that is
				new values used to replace original values) on the Gutenberg-Richter
				a and b values.
			- maxMagGRAbsolute: identifying epistemic uncertainties (absolute)
				on the Gutenberg-Richter maximum magnitude.
		branches: list of LogicTreeBranch objects
		applyToBranches: optional attribute, specifying to which logicTreeBranch
			elements (one or more) in the previous branching level, the branch set
			is linked to. Given as a string of space-separated branch id's, or
			the keyword "ALL", which means that the branch set is linked to all
			branches in the previous branching level (default: "")
		applyToSources: optional attribute, specifying to which source in a
			source model the uncertainty applies to. Specified as a string of
			space-separated source ids (default: "")
		applyToSourceType: optional attribute, specifying to which source type
			the uncertainty applies to. Only one source typology can be defined:
			area, point, simpleFault or complexFault (default: "")
		applyToTectonicRegionType: optional attribute, specifying to which
			tectonic region type the uncertainty applies to. Only one tectonic
			region type can be defined: Active Shallow Crust, Stable Shallow Crust,
			Subduction Interface, Subduction IntraSlab or Volcanic (default: "")
	"""
	def __init__(self, id, uncertainty_type, branches, applyToBranches=[], applyToSources=[], applyToSourceType="", applyToTectonicRegionType=""):
		# TODO: are applyToBranches and applyToSources strings or lists?
		filters = {}
		self.applyToBranches = applyToBranches
		filters["applyToSources"] = self.applyToSources = applyToSources
		filters["applyToSourceType"] = self.applyToSourceType = applyToSourceType
		filters["applyToTectonicRegionType"] = self.applyToTectonicRegionType = applyToTectonicRegionType
		super(LogicTreeBranchSet, self).__init__(uncertainty_type, filters)
		self.id = id
		self.branches = branches

	def __iter__(self):
		return iter(self.branches)

	def __len__(self):
		return len(self.branches)

	@classmethod
	def from_PMF_dict(cls, id, pmf_dict, applyToBranches=[], applyToSourceType="", applyToTectonicRegionType=""):
		"""
		Construct branch set from a PMF dictionary, defining correlated
		uncertainties for a number of sources.
		The applyToSources attribute will be automatically set from the
		keys of pmf_dict

		:param id:
			String, branch set identifier
		:param pmf_dict:
			dict, mapping source ID's to instances of :class:`PMF`
			Note that the number of uncertainties and their weights must
			be identical for the different sources.
		:param applyToBranches:
		:param applyToSourceType:
		:param applyToTectonicRegionType:
			See class constructor
		"""
		pmf = pmf_dict[pmf_dict.keys()[0]]
		if isinstance(pmf, (GMPEPMF, SourceModelPMF)):
			raise Exception("Not supported for this uncertainty type!")
		elif isinstance(pmf, MmaxPMF):
			if pmf.absolute:
				uncertainty_type = "maxMagGRAbsolute"
			else:
				uncertainty_type = "maxMagGRRelative"
		elif isinstance(pmf, MFDPMF):
			if pmf.is_bGRRelative():
				uncertainty_type = "bGRRelative"
			elif pmf.is_abGRAbsolute():
				uncertainty_type = "abGRAbsolute"
			elif pmf.is_incrementalMFDRates():
				# TODO: not yet implemented in OQ
				uncertainty_type = "incrementalMFDRates"

		applyToSources = pmf_dict.keys()
		branches = []
		# TODO: add check to assure that number of uncertainties is the same for each source
		for s, src_id in enumerate(applyToSources):
			for i, (weight, value) in enumerate(pmf.data):
				if s == 0:
					value = {src_id: value}
					branch_id = "%s%02d" % (id, i+1)
					branch = LogicTreeBranch(branch_id, weight, value, parent_branchset=None)
					branches.append(branch)
				else:
					branch = branches[i]
					assert weight == branch.weight, "Weights for correlated uncertainties must be identical!"
					branch.value[src_id] = value

		branchset = LogicTreeBranchSet(id, uncertainty_type, branches, applyToBranches=applyToBranches, applyToSources=applyToSources, applyToSourceType=applyToSourceType, applyToTectonicRegionType=applyToTectonicRegionType)
		for branch in branchset:
			branch.parent_branchset = branchset
		return branchset

	@classmethod
	def from_PMF(cls, id, pmf, applyToBranches=[], applyToSources=[], applyToSourceType="", applyToTectonicRegionType=""):
		"""
		Construct branch set from a PMF

		:param id:
			String, branch set identifier
		:param pmf:
			instance of :class:`PMF`
		:param applyToBranches:
		:param applyToSources:
		:param applyToSourceType:
		:param applyToTectonicRegionType:
			See class constructor
		"""
		if isinstance(pmf, GMPEPMF):
			uncertainty_type = "gmpeModel"
		elif isinstance(pmf, SourceModelPMF):
			uncertainty_type = "sourceModel"
		elif isinstance(pmf, MmaxPMF):
			if pmf.absolute:
				uncertainty_type = "maxMagGRAbsolute"
			else:
				uncertainty_type = "maxMagGRRelative"
		elif isinstance(pmf, MFDPMF):
			if pmf.is_bGRRelative():
				uncertainty_type = "bGRRelative"
			elif pmf.is_abGRAbsolute():
				uncertainty_type = "abGRAbsolute"
			elif pmf.is_incrementalMFDRates():
				# TODO: not yet implemented in OQ
				uncertainty_type = "incrementalMFDRates"

		branches = []
		for i, (weight, model) in enumerate(pmf.data):
			if isinstance(pmf, SourceModelPMF):
				## Models can be SourceModel objects or source model names
				if not isinstance(model, (str, unicode)):
					model = model.name
				## Add .xml extension if necessary
				if not model[-4:] == ".xml":
					model += ".xml"
			branch_id = "%s%02d" % (id, i+1)
			branch = LogicTreeBranch(branch_id, weight, model, parent_branchset=None)
			branches.append(branch)
		branchset = LogicTreeBranchSet(id, uncertainty_type, branches, applyToBranches=applyToBranches, applyToSources=applyToSources, applyToSourceType=applyToSourceType, applyToTectonicRegionType=applyToTectonicRegionType)
		for branch in branchset:
			branch.parent_branchset = branchset
		return branchset

	def validate_weights(self):
		"""
		Check if weights of child branches sums up to 1.0
		"""
		weights = [branch.weight for branch in self.branches]
		if abs(Decimal(1.0) - np.add.reduce(weights)) > 1E-3:
			raise NRMLError("BranchSet %s: branches do not sum to 1.0" % self.id)

	def validate_unc_type(self):
		"""
		Validate uncertainty type
		"""
		if self.uncertainty_type not in ENUM_OQ_UNCERTAINTYTYPES:
			raise NRMLError("BranchSet %s: uncertainty type %s not supported!" % (self.id, self.uncertainty_type))
		if self.uncertainty_type == "gmpeModel" and not self.applyToTectonicRegionType:
			raise NRMLError("BranchSet %s: applyToTectonicRegionType must be defined if uncertainty_type is gmpeModel" % self.id)

	def validate(self):
		"""
		Validate
		"""
		self.validate_weights()
		self.validate_unc_type()
		for branch in self.branches:
			branch.validate()

	def create_xml_element(self):
		"""
		Create xml element (NRML logicTreeBranchSet element)
		"""
		lbs_elem = etree.Element(ns.LOGICTREE_BRANCHSET, branchSetID=self.id, uncertaintyType=self.uncertainty_type)
		if self.applyToBranches:
			lbs_elem.set("applyToBranches", " ".join(map(str, self.applyToBranches)))
		if self.filters["applyToSources"]:
			lbs_elem.set("applyToSources", " ".join(map(str, self.filters["applyToSources"])))
		if self.filters["applyToSourceType"]:
			lbs_elem.set("applyToSourceType", self.filters["applyToSourceType"])
		if self.filters["applyToTectonicRegionType"]:
			lbs_elem.set("applyToTectonicRegionType", self.filters["applyToTectonicRegionType"])
		for branch in self.branches:
			lb_elem = branch.create_xml_element()
			lbs_elem.append(lb_elem)
		return lbs_elem


class LogicTreeBranchingLevel():
	"""
	Class representing a logic-tree branching level
	A branching level identifies the position in a tree where branching occurs.
	It contains a sequence of logicTreeBranchSet elements.
	There are no restrictions on the number of branch sets that can be defined
	inside a branching level.
	Arguments:
		id: identifier
		branch_sets: list of LogicTreeBranchSet objects
	"""
	def __init__(self, id, branch_sets):
		self.id = id
		self.branch_sets = branch_sets

	def __iter__(self):
		return iter(self.branch_sets)

	def __len__(self):
		return len(self.branch_sets)

	def validate(self):
		"""
		Validate
		"""
		for branch_set in self.branch_sets:
			branch_set.validate()

	def create_xml_element(self):
		"""
		Create xml element (NRML logicTreeBranchingLevel element)
		"""
		lbl_elem = etree.Element(ns.LOGICTREE_BRANCHINGLEVEL, branchingLevelID=self.id)
		for branch_set in self.branch_sets:
			lbs_elem = branch_set.create_xml_element()
			lbl_elem.append(lbs_elem)
		return lbl_elem


class LogicTree(object):
	"""
	Class representing a logic tree
	A LogicTree is defined as a sequence of LogicTreeBranchingLevel objects.
	The position in the sequence specifies in which level of the tree the
	branching level is located. That is, the first LogicTreeBranchingLevel object
	in the sequence represents the first level in the tree, the second object
	the second level in the tree, and so on.
	Arguments:
		id: identifier
		branching_levels: list of LogicTreeBranchingLevel objects
	"""
	def __init__(self, id, branching_levels):
		self.id = id
		self.branching_levels = branching_levels

	def __iter__(self):
		return iter(self.branching_levels)

	@property
	def root_branchset(self):
		return self.branching_levels[0].branch_sets[0]

	def validate(self):
		"""
		Validate
		"""
		for branching_level in self.branching_levels:
			branching_level.validate()

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML root element)
		"""
		lt_elem = etree.Element(ns.LOGICTREE, logicTreeID=self.id)
		for branching_level in self.branching_levels:
			lbl_elem = branching_level.create_xml_element()
			lt_elem.append(lbl_elem)
		return lt_elem

	def get_xml_string(self, encoding='latin1', pretty_print=True):
		"""
		Return XML string representation of logic tree
		This function is called by write_xml() and print_xml()
		Arguments:
			encoding: unicode encoding (default: 'latin1')
			pretty_print: boolean indicating whether or not to indent each
				element (default: True)
		"""
		tree = create_nrml_root(self, encoding=encoding)
		return etree.tostring(tree, xml_declaration=True, encoding=encoding, pretty_print=pretty_print)

	def write_xml(self, filespec, encoding='latin1', pretty_print=True):
		"""
		Write logic tree to XML file
		Arguments:
			filespec: full path to XML output file
			encoding: unicode encoding (default: 'latin1')
			pretty_print: boolean indicating whether or not to indent each
				element (default: True)
		"""
		try:
			of = open(filespec, 'w')
		except:
			raise IOError("Can't write to file %s" % filespec)
		else:
			tree = create_nrml_root(self, encoding=encoding)
			tree.write(of, xml_declaration=True, encoding=encoding, pretty_print=pretty_print)

	def print_xml(self, encoding='latin1', pretty_print=True):
		"""
		Print XML to screen
		"""
		print self.get_xml_string(encoding=encoding, pretty_print=pretty_print)

	def connect_branches(self):
		"""
		Connect all branches and branch sets of the logic tree in a chain.
		The child_branchset property of each branch is set to the branch set
		in the subsequent branching level that points to this branch through
		its "applyToBranches" filter.
		All branches are collected in a flat list that is set as the branches
		property of the logic tree.
		Connected branches are required for the logic tree processor.
		"""
		## Store all branches in self.branches,
		## and set child_branchset of every branch
		all_branches = [self.root_branchset.branches]
		for branching_level in self.branching_levels[1:]:
			level_branches = []
			for branch_set in branching_level.branch_sets:
				connecting_branch_ids = branch_set.applyToBranches
				if not connecting_branch_ids:
					connecting_branch_ids = [branch.branch_id for branch in all_branches[-1]]
				for branch in all_branches[-1]:
					if branch.branch_id in connecting_branch_ids:
						branch.child_branchset = branch_set
				level_branches.extend(branch_set.branches)
			all_branches.append(level_branches)

		## Flatten branches list
		self.branches = sum(all_branches, [])

		# TODO: write alternative version, with child_branchset as dictionary
		# TODO: adjust oq-engine logictree to parse xml files, not necessarily
		# read from dbase

	def are_branch_ids_unique(self):
		"""
		Determine whether or not all branch ID's are unique

		:return:
			Bool
		"""
		branch_ids = []
		for branching_level in self:
			for branch_set in branching_level:
				for branch in branch_set:
					if branch.branch_id in branch_ids:
						return False
					else:
						branch_ids.append(branch.branch_id)
		return True

	def get_branching_level_index(self, branchset):
		"""
		Determine which branching level the given branchset belongs to.

		:param branchset:
			instance of :class:`LogicTreeBranchSet`

		:return:
			Int
		"""
		for i, branching_level in enumerate(self):
			for bs in branching_level:
				if bs.id == branchset.id:
					return i

	def get_branch_by_id(self, branch_id):
		"""
		Return branch with given ID.

		:param branch_id:
			string, branch ID

		:return:
			instance of :class:`LogicTreeBranch`
		"""
		for branching_level in self:
			for branchset in branching_level:
				for branch in branchset:
					if branch.branch_id == branch_id:
						return branch

	def get_parent_branches(self, branchset):
		"""
		Return list with parent branches of given branchset
		:param branchset:
			instance of :class:`LogicTreeBranchSet`

		:return:
			list with instances of :class:`LogicTreeBranch`
		"""
		applyToBranches = branchset.applyToBranches
		if applyToBranches:
			branches = [self.get_branch_by_id(branch_id) for branch_id in applyToBranches]
		else:
			branches = []
			if branchset != self.root_branchset:
				parent_bl_index = self.get_branching_level_index(branchset) - 1
				parent_bl = self.branching_levels[parent_bl_index]
				for bs in parent_bl:
					branches.extend(bs.branches)
		return branches

	def get_branchsets(self, unc_class=None):
		"""
		Return a list of all branch sets in the logic tree

		:param unc_class:
			string, uncertainty class: "gmpe", "source_model", "mfd", "mmax"
			(default: None)

		:return:
			list with instances of :class:`LogicTreeBranchSet`
		"""
		branchsets = []
		for branching_level in self:
			if not unc_class:
				branchsets.extend(branching_level.branch_sets)
			else:
				unc_types = {"gmpe": ["gmpeModel"], "source_model": ["sourceModel"], "mfd": ["abGRAbsolute", "bGRRelative", "incrementalMFDRates"], "mmax": ["maxMagGRRelative", "maxMagGRAbsolute"]}[unc_class]
				for branchset in branching_level:
					if branchset.uncertainty_type in unc_types:
						branchsets.append(branchset)
		return branchsets

	def _calc_diagram_positions(self):
		"""
		Compute x and y positions of branchsets and branches in networkx diagram.
		Helper function for func:`plot_diagram`

		:return:
			dict {id: (x,y)}
		"""
		pos = {}
		pos[self.root_branchset.id] = (0, 0.5)
		num_branches = len(self.root_branchset)
		for b, branch in enumerate(self.root_branchset):
			pos[branch.branch_id] = (0.5, 1./num_branches/2 + b*(1./num_branches))
		for l, branching_level in enumerate(self.branching_levels[1:]):
			num_branchsets = len(branching_level)
			for s, branchset in enumerate(branching_level):
				num_branches = len(branchset)
				parent_branches = self.get_parent_branches(branchset)
				parent_branch_y = np.array([pos[pb.branch_id][1] for pb in parent_branches], dtype='f')
				y = parent_branch_y.mean()
				if len(parent_branches) > 1:
					ymin, ymax = parent_branch_y.min(), parent_branch_y.max()
				else:
					ymin = y - (1. / num_branchsets) + 1./num_branchsets/4
					ymax = y + (1. / num_branchsets) - 1./num_branchsets/4
					dy = float(ymax - ymin)
					ymin, ymax = ymin + dy/(num_branches-1)/2, ymax-dy/(num_branches-1)/2
				pos[branchset.id] = (l+1, y)
				for b, branch in enumerate(branchset):
					dy = float(ymax - ymin)
					#pos[branch.branch_id] = (l+1+0.5, ymin + dy/num_branches/2 + b*(dy/num_branches))
					pos[branch.branch_id] = (l+1+0.5, ymin + b*(dy/(num_branches-1)))
		return pos

	def plot_diagram(self, highlight_path=[]):
		"""
		Plot diagram of logic tree using networkx or pygraphviz.
		Requires branches to be connected.

		:param highlight_path:
			list of strings: branch ID's of path to highlight
		"""
		import networkx as nx
		import matplotlib.pyplot as plt
		all_branches = self.branches
		all_branchsets = self.get_branchsets()
		mfd_branchsets = self.get_branchsets("mfd")
		mmax_branchsets = self.get_branchsets("mmax")
		gmpe_branchsets = self.get_branchsets("gmpe")

		graph = nx.Graph()
		branchset_nodes = [branchset.id for branchset in all_branchsets]
		mfd_branchset_nodes = [branchset.id for branchset in mfd_branchsets]
		mmax_branchset_nodes = [branchset.id for branchset in mmax_branchsets]
		gmpe_branchset_nodes = [branchset.id for branchset in gmpe_branchsets]
		branch_nodes = [branch.branch_id for branch in all_branches]
		graph.add_nodes_from(branchset_nodes)
		graph.add_nodes_from(branch_nodes)
		for branch in self.branches:
			graph.add_edge(branch.parent_branchset.id, branch.branch_id)
			if branch.child_branchset:
				graph.add_edge(branch.branch_id, branch.child_branchset.id)

		pos = self._calc_diagram_positions()
		nx.draw_networkx_nodes(graph, pos, nodelist=[self.root_branchset.id], node_shape='>', node_color='red', node_size=500)
		nx.draw_networkx_nodes(graph, pos, nodelist=mmax_branchset_nodes, node_shape='>', node_color='yellow', node_size=500)
		nx.draw_networkx_nodes(graph, pos, nodelist=mfd_branchset_nodes, node_shape='>', node_color='green', node_size=500)
		nx.draw_networkx_nodes(graph, pos, nodelist=gmpe_branchset_nodes, node_shape='>', node_color='blue', node_size=500)
		nx.draw_networkx_nodes(graph, pos, nodelist=branch_nodes, node_color='white', node_size=250)
		nx.draw_networkx_edges(graph, pos)
		if highlight_path:
			for branch_id in highlight_path:
				branch = self.get_branch_by_id(branch_id)
				parent_branchset_id = branch.parent_branchset.id
				nx.draw_networkx_edges(graph, pos, edgelist=[(parent_branchset_id, branch_id)], edge_color='r', width=3)
				if branch.child_branchset:
					child_branchset_id = branch.child_branchset.id
					nx.draw_networkx_edges(graph, pos, edgelist=[(branch_id, child_branchset_id)], edge_color='r', width=3)
		labels = {}
		for branch_id in branch_nodes:
			labels[branch_id] = branch_id
		nx.draw_networkx_labels(graph, pos, labels=labels, size=8, verticalalignment="bottom")
		plt.show()



if __name__ == '__main__':
	from configobj import ConfigObj
	## Test construction of GroundMotionSystem
	from SHARE_GMPE import SHARE_gmpe_system_dict

	gmpe_lt = GroundMotionSystem("lt0", SHARE_gmpe_system_dict)
	gmpe_lt.validate()
	#gmpe_lt.print_xml()


	"""
	## Test construction of SeismicSourceSystem
	#ssd = {}
	ssd = ConfigObj(indent_type="	")

	sourceModelid = "SourceModel01"
	ssd[sourceModelid] = {}
	ssd[sourceModelid]["weight"] = 1.0
	ssd[sourceModelid]["xml_file"] = "TwoZoneModel.xml"
	ssd[sourceModelid]["sources"] = {}
	I = 2
	for i in range(I):
		sourceid = "Source%02d" % (i+1)
		ssd[sourceModelid]["sources"][sourceid] = {}
		ssd[sourceModelid]["sources"][sourceid]["uncertainty_levels"] = ["Mmax", "MFD", "Dip", "Depth"]
		ssd[sourceModelid]["sources"][sourceid]["unc_type"] = "absolute"
		ssd[sourceModelid]["sources"][sourceid]["models"] = {}
		J = 2
		for j in range(J):
			Mmaxid = "Mmax%02d" % (j+1)
			ssd[sourceModelid]["sources"][sourceid]["models"][Mmaxid] = {}
			ssd[sourceModelid]["sources"][sourceid]["models"][Mmaxid]["Mmax"] = 6.7
			ssd[sourceModelid]["sources"][sourceid]["models"][Mmaxid]["weight"] = 1.0 / J
			ssd[sourceModelid]["sources"][sourceid]["models"][Mmaxid]["unc_type"] = "absolute"
			ssd[sourceModelid]["sources"][sourceid]["models"][Mmaxid]["models"] = {}
			K = 2
			for k in range(K):
				mfdid = "MFD%02d" % (k+1)
				ssd[sourceModelid]["sources"][sourceid]["models"][Mmaxid]["models"][mfdid] = {}
				ssd[sourceModelid]["sources"][sourceid]["models"][Mmaxid]["models"][mfdid]["a"] = 2.4
				ssd[sourceModelid]["sources"][sourceid]["models"][Mmaxid]["models"][mfdid]["b"] = 0.89
				ssd[sourceModelid]["sources"][sourceid]["models"][Mmaxid]["models"][mfdid]["Mmin"] = 4.0
				ssd[sourceModelid]["sources"][sourceid]["models"][Mmaxid]["models"][mfdid]["weight"] = 1.0 / K

	ssd[sourceModelid]["sources"][sourceid]["models"][Mmaxid]["models"][mfdid]["models"] = {}
	L = 2
	for l in range(L):
		Dipid = "Dip%02d" % (l+1)
		ssd[sourceModelid]["sources"][sourceid]["models"][Mmaxid]["models"][mfdid]["models"][Dipid] = {}
		ssd[sourceModelid]["sources"][sourceid]["models"][Mmaxid]["models"][mfdid]["models"][Dipid]["dip"] = 2.4
		ssd[sourceModelid]["sources"][sourceid]["models"][Mmaxid]["models"][mfdid]["models"][Dipid]["weight"] = 1.0 / L

	ssd[sourceModelid]["sources"][sourceid]["models"][Mmaxid]["models"][mfdid]["models"][Dipid]["models"] = {}
	M = 2
	for m  in range(M):
		Depthid = "Depth%02d" % (m+1)
		ssd[sourceModelid]["sources"][sourceid]["models"][Mmaxid]["models"][mfdid]["models"][Dipid]["models"][Depthid] = {}
		ssd[sourceModelid]["sources"][sourceid]["models"][Mmaxid]["models"][mfdid]["models"][Dipid]["models"][Depthid]["depth"] = 20
		ssd[sourceModelid]["sources"][sourceid]["models"][Mmaxid]["models"][mfdid]["models"][Dipid]["models"][Depthid]["weight"] = 1.0 / M

	source_system_lt = SeismicSourceSystem_v2("lt0", ssd)
	#source_system_lt.print_xml()
	source_system_lt.write_xml(r"C:\Temp\source_model_system.xml")
	source_system_lt.export_cfg(r"C:\Temp\source_model_system.cfg")
	print
	source_system_lt.print_system_dict()
	print source_system_lt.getNumberOfBranches(sourceModelid, sourceid)
	"""

	## Test walk_ssd_branches function
	"""
	topLevel = source_system_lt.source_system_dict[sourceModelid]["sources"][sourceid]["models"]
	for info in source_system_lt.walk_ssd_end_branches(topLevel):
		print info
	"""

	## Creation of basic source_system_dict
	## To be tested
	"""
	from rob_sourceModels import create_rob_sourceModel
	sm1 = create_rob_sourceModel("Seismotectonic", Mmin=4.0)
	sm2 = create_rob_sourceModel("TwoZone", Mmin=4.0)
	source_system_lt = create_basic_seismicSourceSystem([sm1, sm2])
	source_system_lt.print_xml()
	"""

	## Test SymmetricRelativeSeismicSourceSystem
	branching_level_dicts = []
	branching_level = {}
	branching_level["uncertainty_type"] = "sourceModel"
	branching_level["models"] = ["Seismotectonic.xml", "TwoZone_split.xml"]
	branching_level["weights"] = [0.6, 0.4]
	branching_level["labels"] = ["Seismotectonic", "TwoZone"]
	branching_level_dicts.append(branching_level)
	branching_level = {}
	branching_level["uncertainty_type"] = "maxMagGRRelative"
	branching_level["models"] = [-0.25, 0, 0.25]
	branching_level["weights"] = [0.25, 0.5, 0.25]
	branching_level["labels"] = ["Mmax-0.25", "Mmax+0", "Mmax+0.25"]
	branching_level_dicts.append(branching_level)
	branching_level = {}
	branching_level["uncertainty_type"] = "bGRRelative"
	branching_level["models"] = [-0.15, 0, 0.15]
	branching_level["weights"] = [1./3, 1./3, 1./3]
	branching_level["labels"] = ["bGR-0.15", "bGR+0", "bGR+0.15"]
	branching_level_dicts.append(branching_level)
	source_system_lt = SymmetricRelativeSeismicSourceSystem("SymRelSSD", branching_level_dicts, explicit=True)
	source_system_lt.print_xml()
