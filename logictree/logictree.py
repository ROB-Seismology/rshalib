# -*- coding: utf-8 -*-

"""
Classes representing NRML elements used in OpenQuake logic trees
(seismic source systems and ground motion systems). Every class has methods
to generate xml elements that can be used to build a NRML document
"""

import numpy
import pprint
from lxml import etree
from decimal import Decimal

from ..nrml import ns
from ..nrml.common import *
from ..pmf.distributions import get_uniform_weights


### Logic Tree elements

# TODO: elaborate validation

class LogicTreeBranch():
	"""
	Class representing a logic-tree branch
	Arguments:
		ID: identifier
		uncertaintyModel: string identifying an uncertainty model; the content
			of the string varies with the uncertaintyType attribute value of the
			parent branchSet
		uncertaintyWeight: specifying the probability/weight associated to
			the uncertaintyModel
	"""
	def __init__(self, ID, uncertaintyModel, uncertaintyWeight):
		self.ID = ID
		self.uncertaintyModel = uncertaintyModel
		self.uncertaintyWeight = uncertaintyWeight

	def validate(self):
		pass

	def create_xml_element(self):
		"""
		Create xml element (NRML logicTreeBranch element)
		"""
		lb_elem = etree.Element(ns.LOGICTREE_BRANCH, branchID=self.ID)
		um_elem = etree.SubElement(lb_elem, ns.UNCERTAINTY_MODEL)
		um_elem.text = xmlstr(self.uncertaintyModel)
		uw_elem = etree.SubElement(lb_elem, ns.UNCERTAINTY_WEIGHT)
		uw_elem.text = str(self.uncertaintyWeight)
		return lb_elem


class LogicTreeBranchSet():
	"""
	Class representing a logic-tree branch set.
	A LogicTreeBranchSet defines a particular epistemic uncertainty inside
	a branching level. It identifies a collection of LogicTreeBranch objects.
	There is no restriction on the number of branches in a branch set, as
	long as their uncertainty weights sum to 1.0
	Arguments:
		ID: identifier
		uncertaintyType: required attribute
			Possible values for the uncertaintyType attribute are:
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
			is linked to. Given as a string of space-separated branch ID's, or
			the keyword "ALL", which means that the branch set is linked to all
			branches in the previous branching level (default: "")
		applyToSources: optional attribute, specifying to which source in a
			source model the uncertainty applies to. Specified as a string of
			space-separated source IDs (default: "")
		applyToSourceType: optional attribute, specifying to which source type
			the uncertainty applies to. Only one source typology can be defined:
			area, point, simpleFault or complexFault (default: "")
		applyToTectonicRegionType: optional attribute, specifying to which
			tectonic region type the uncertainty applies to. Only one tectonic
			region type can be defined: Active Shallow Crust, Stable Shallow Crust,
			Subduction Interface, Subduction IntraSlab or Volcanic (default: "")
	"""
	def __init__(self, ID, uncertaintyType, branches, applyToBranches="", applyToSources="", applyToSourceType="", applyToTectonicRegionType=""):
		self.ID = ID
		self.uncertaintyType = uncertaintyType
		self.branches = branches
		self.applyToBranches = applyToBranches
		self.applyToSources = applyToSources
		self.applyToSourceType = applyToSourceType
		self.applyToTectonicRegionType = applyToTectonicRegionType

	def validate_weights(self):
		"""
		Check if weights of child branches sums up to 1.0
		"""
		weights = [branch.uncertaintyWeight for branch in self.branches]
		if abs(Decimal(1.0) - numpy.add.reduce(weights)) > 1E-3:
			raise NRMLError("BranchSet %s: branches do not sum to 1.0" % self.ID)

	def validate_unc_type(self):
		"""
		Validate uncertainty type
		"""
		if self.uncertaintyType not in ENUM_OQ_UNCERTAINTYTYPES:
			raise NRMLError("BranchSet %s: uncertainty type %s not supported!" % (self.ID, self.uncertaintyType))
		if self.uncertaintyType == "gmpeModel" and not self.applyToTectonicRegionType:
			raise NRMLError("BranchSet %s: applyToTectonicRegionType must be defined if uncertaintyType is gmpeModel" % self.ID)

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
		lbs_elem = etree.Element(ns.LOGICTREE_BRANCHSET, branchSetID=self.ID, uncertaintyType=self.uncertaintyType)
		if self.applyToBranches:
			lbs_elem.set("applyToBranches", self.applyToBranches)
		if self.applyToSources:
			lbs_elem.set("applyToSources", self.applyToSources)
		if self.applyToSourceType:
			lbs_elem.set("applyToSourceType", self.applyToSourceType)
		if self.applyToTectonicRegionType:
			lbs_elem.set("applyToTectonicRegionType", self.applyToTectonicRegionType)
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
		ID: identifier
		branchSets: list of LogicTreeBranchSet objects
	"""
	def __init__(self, ID, branchSets):
		self.ID = ID
		self.branchSets = branchSets

	def validate(self):
		"""
		Validate
		"""
		for branchSet in self.branchSets:
			branchSet.validate()

	def create_xml_element(self):
		"""
		Create xml element (NRML logicTreeBranchingLevel element)
		"""
		lbl_elem = etree.Element(ns.LOGICTREE_BRANCHINGLEVEL, branchingLevelID=self.ID)
		for branchSet in self.branchSets:
			lbs_elem = branchSet.create_xml_element()
			lbl_elem.append(lbs_elem)
		return lbl_elem


class LogicTree():
	"""
	Class representing a logic tree
	A LogicTree is defined as a sequence of LogicTreeBranchingLevel objects.
	The position in the sequence specifies in which level of the tree the
	branching level is located. That is, the first LogicTreeBranchingLevel object
	in the sequence represents the first level in the tree, the second object
	the second level in the tree, and so on.
	Arguments:
		ID: identifier
		branchingLevels: list of LogicTreeBranchingLevel objects
	"""
	def __init__(self, ID, branchingLevels):
		self.ID = ID
		self.branchingLevels = branchingLevels

	def validate(self):
		"""
		Validate
		"""
		for branchingLevel in self.branchingLevels:
			branchingLevel.validate()

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML root element)
		"""
		lt_elem = etree.Element(ns.LOGICTREE, logicTreeID=self.ID)
		for branchingLevel in self.branchingLevels:
			lbl_elem = branchingLevel.create_xml_element()
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

	sourceModelID = "SourceModel01"
	ssd[sourceModelID] = {}
	ssd[sourceModelID]["weight"] = 1.0
	ssd[sourceModelID]["xml_file"] = "TwoZoneModel.xml"
	ssd[sourceModelID]["sources"] = {}
	I = 2
	for i in range(I):
		sourceID = "Source%02d" % (i+1)
		ssd[sourceModelID]["sources"][sourceID] = {}
		ssd[sourceModelID]["sources"][sourceID]["uncertainty_levels"] = ["Mmax", "MFD", "Dip", "Depth"]
		ssd[sourceModelID]["sources"][sourceID]["unc_type"] = "absolute"
		ssd[sourceModelID]["sources"][sourceID]["models"] = {}
		J = 2
		for j in range(J):
			MmaxID = "Mmax%02d" % (j+1)
			ssd[sourceModelID]["sources"][sourceID]["models"][MmaxID] = {}
			ssd[sourceModelID]["sources"][sourceID]["models"][MmaxID]["Mmax"] = 6.7
			ssd[sourceModelID]["sources"][sourceID]["models"][MmaxID]["weight"] = 1.0 / J
			ssd[sourceModelID]["sources"][sourceID]["models"][MmaxID]["unc_type"] = "absolute"
			ssd[sourceModelID]["sources"][sourceID]["models"][MmaxID]["models"] = {}
			K = 2
			for k in range(K):
				mfdID = "MFD%02d" % (k+1)
				ssd[sourceModelID]["sources"][sourceID]["models"][MmaxID]["models"][mfdID] = {}
				ssd[sourceModelID]["sources"][sourceID]["models"][MmaxID]["models"][mfdID]["a"] = 2.4
				ssd[sourceModelID]["sources"][sourceID]["models"][MmaxID]["models"][mfdID]["b"] = 0.89
				ssd[sourceModelID]["sources"][sourceID]["models"][MmaxID]["models"][mfdID]["Mmin"] = 4.0
				ssd[sourceModelID]["sources"][sourceID]["models"][MmaxID]["models"][mfdID]["weight"] = 1.0 / K

	ssd[sourceModelID]["sources"][sourceID]["models"][MmaxID]["models"][mfdID]["models"] = {}
	L = 2
	for l in range(L):
		DipID = "Dip%02d" % (l+1)
		ssd[sourceModelID]["sources"][sourceID]["models"][MmaxID]["models"][mfdID]["models"][DipID] = {}
		ssd[sourceModelID]["sources"][sourceID]["models"][MmaxID]["models"][mfdID]["models"][DipID]["dip"] = 2.4
		ssd[sourceModelID]["sources"][sourceID]["models"][MmaxID]["models"][mfdID]["models"][DipID]["weight"] = 1.0 / L

	ssd[sourceModelID]["sources"][sourceID]["models"][MmaxID]["models"][mfdID]["models"][DipID]["models"] = {}
	M = 2
	for m  in range(M):
		DepthID = "Depth%02d" % (m+1)
		ssd[sourceModelID]["sources"][sourceID]["models"][MmaxID]["models"][mfdID]["models"][DipID]["models"][DepthID] = {}
		ssd[sourceModelID]["sources"][sourceID]["models"][MmaxID]["models"][mfdID]["models"][DipID]["models"][DepthID]["depth"] = 20
		ssd[sourceModelID]["sources"][sourceID]["models"][MmaxID]["models"][mfdID]["models"][DipID]["models"][DepthID]["weight"] = 1.0 / M

	source_system_lt = SeismicSourceSystem_v2("lt0", ssd)
	#source_system_lt.print_xml()
	source_system_lt.write_xml(r"C:\Temp\source_model_system.xml")
	source_system_lt.export_cfg(r"C:\Temp\source_model_system.cfg")
	print
	source_system_lt.print_system_dict()
	print source_system_lt.getNumberOfBranches(sourceModelID, sourceID)
	"""

	## Test walk_ssd_branches function
	"""
	topLevel = source_system_lt.source_system_dict[sourceModelID]["sources"][sourceID]["models"]
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
