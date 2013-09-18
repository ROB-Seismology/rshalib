"""
Test logic tree construction
"""
import os
import random
import copy

import numpy as np

from openquake.engine.input.logictree import LogicTreeProcessor, SourceModelLogicTree
import hazard.rshalib as rshalib


## Full logic tree, with three uncertainty levels
## Source models
sm1 = rshalib.rob.create_rob_source_model("Leynaud")
sm1.name = "SM1"
sm1.sources = sm1.sources[:3]
for i, src in enumerate(sm1.sources):
	src.source_id = "SRC1%d" % i
sm2 = rshalib.rob.create_rob_source_model("TwoZone")
sm2.name = "SM2"
for i, src in enumerate(sm2.sources):
	src.source_id = "SRC2%d" % i
	#src.mfd = src.mfd.to_evenly_discretized_mfd()
source_models = [sm1, sm2]
source_model_pmf = rshalib.pmf.SourceModelPMF(source_models, [0.5, 0.5])

## Mmax
Mmax_pmf_dict = {}
for sm in source_models:
	Mmax_pmf_dict[sm.name] = {}
	for i, src in enumerate(sm):
		Mmax_pmf_dict[sm.name][src.source_id] = rshalib.pmf.MmaxPMF(np.array([6.0, 6.5, 7.0]) + i*0.01, [0.5, 0.3, 0.2], absolute=True)

## MFD
MFD_pmf_dict = {}
for sm in source_models:
	MFD_pmf_dict[sm.name] = {}
	for i, src in enumerate(sm):
		MFD_pmf_dict[sm.name][src.source_id] = rshalib.pmf.MFDPMF([np.array((1.3, 1.0)) + i*0.01, np.array((1.2, 0.9)) + i*0.01], [0.4, 0.6])

## incremental MFD
#MFD_pmf_dict[sm2.name] = {"SRC20": rshalib.pmf.MFDPMF([np.arange(1,10)[::-1]], [1]), "SRC21": rshalib.pmf.MFDPMF([np.arange(1,5)[::-1]], [1])}


## Basic logic tree with different source models only
#Mmax_pmf_dict = {}
#MFD_pmf_dict = {}

## Logic tree with relative uncertainties for all source models in both Mmax and MFD
#Mmax_pmf_dict = {None: {None: rshalib.pmf.MmaxPMF([-0.2, 0, 0.2], [0.5, 0.3, 0.2], absolute=False)}}
#MFD_pmf_dict = {None: {None: rshalib.pmf.MFDPMF([-0.1, 0.1], [0.4, 0.6])}}

## Logic tree with relative uncertainties for all sources in MFD
#Mmax_pmf_dict = {}
#for sm in source_models:
#	Mmax_pmf_dict[sm.name] = {None: rshalib.pmf.MmaxPMF([-0.2, 0, 0.2], [0.5, 0.3, 0.2], absolute=False)}
#MFD_pmf_dict = {}
#for sm in source_models:
#	MFD_pmf_dict[sm.name] = {None: rshalib.pmf.MFDPMF([-0.1, 0.1], [0.4, 0.6])}

## Mix of the above:
## - relative uncertainties for all sources in a source model in first level
## - relative uncertainties for all source models in second level
#Mmax_pmf_dict = {}
#for sm in source_models:
#	Mmax_pmf_dict[sm.name] = {None: rshalib.pmf.MmaxPMF([-0.2, 0, 0.2], [0.5, 0.3, 0.2], absolute=False)}
#MFD_pmf_dict = {None: {None: rshalib.pmf.MFDPMF([-0.1, 0.1], [0.4, 0.6])}}
## This doesn't / shouldn't work
## - relative uncertainties for all source models in a source model in first level
## - relative uncertainties for all sources in second level
#Mmax_pmf_dict = {None: {None: rshalib.pmf.MmaxPMF([-0.2, 0, 0.2], [0.5, 0.3, 0.2], absolute=False)}}
#MFD_pmf_dict = {}
#for sm in source_models:
#	MFD_pmf_dict[sm.name] = {None: rshalib.pmf.MFDPMF([-0.1, 0.1], [0.4, 0.6])}



## Construct logic tree
source_model_lt = rshalib.logictree.SeismicSourceSystem.from_independent_uncertainty_levels("lt", source_model_pmf, Mmax_pmf_dict, MFD_pmf_dict, unc2_correlated=False, unc3_correlated=False)
#print source_model_lt.are_branch_ids_unique()
source_model_lt.print_xml()
xml_filespec = r"C:\Temp\seismic_source_system.xml"
#source_model_lt.write_xml(xml_filespec)
print "Number of paths in logic tree: %d" % source_model_lt.get_num_paths()
#source_model_lt.plot_diagram(branch_label="branch_id")
source_model_lt.plot_uncertainty_levels_diagram(source_model_pmf, [Mmax_pmf_dict, MFD_pmf_dict])


## Parse logic tree from NRML
#xml_filespec = "E:\Home\_kris\Python\GEM\oq-engine\demos\hazard\LogicTreeCase3ClassicalPSHA\source_model_logic_tree.xml"
source_model_lt2 = rshalib.logictree.SeismicSourceSystem.parse_from_xml(xml_filespec, validate=False)
source_model_lt2.plot_diagram()
#source_model_lt2.write_xml(r"C:\Temp\seismic_source_system2.xml")


## Sample logic tree
from hazard.psha.Projects.SHRE_NPP.params.gmpe import gmpe_lt
#gmpe_lt.plot_diagram()

random_seed = 1
num_samples = 3
verbose=True
show_plot=False
psha_model_tree = rshalib.shamodel.PSHAModelTree("Test", source_models, source_model_lt, gmpe_lt, "", random_seed=random_seed)
#psha_models = psha_model_tree.sample_source_model_lt(num_samples, verbose=verbose, show_plot=show_plot)
#psha_model_tree.sample_gmpe_lt(num_samples, verbose=verbose)
psha_models, weights = psha_model_tree.enumerate_source_model_lt(verbose=verbose)
