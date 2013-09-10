"""
Test logic tree construction
"""
import os
import random
import copy

import hazard.rshalib as rshalib
from openquake.engine.input.logictree import LogicTreeProcessor, SourceModelLogicTree


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
source_models = [sm1, sm2]
source_model_pmf = rshalib.pmf.SourceModelPMF(source_models, [0.5, 0.5])

## Mmax
Mmax_pmf_dict = {}
for sm in source_models:
	Mmax_pmf_dict[sm.name] = {}
	for src in sm:
		Mmax_pmf_dict[sm.name][src.source_id] = rshalib.pmf.MmaxPMF([6.0, 6.5, 7.0], [0.5, 0.3, 0.2], absolute=True)

## MFD
MFD_pmf_dict = {}
for sm in source_models:
	MFD_pmf_dict[sm.name] = {}
	for src in sm:
		MFD_pmf_dict[sm.name][src.source_id] = rshalib.pmf.MFDPMF([(1.3, 1.0), (1.2, 0.9)], [0.4, 0.6])

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
source_model_lt = rshalib.logictree.SeismicSourceSystem.from_independent_uncertainty_levels("lt", source_model_pmf, Mmax_pmf_dict, MFD_pmf_dict, unc2_correlated=True, unc3_correlated=True)
#print source_model_lt.are_branch_ids_unique()
source_model_lt.print_xml()
#source_model_lt.write_xml(r"C:\Temp\seismic_source_system.xml")
source_model_lt.plot_diagram()

## Parse logic tree from NRML
source_model_lt2 = SourceModelLogicTree(None, basepath=r"C:\Temp", filename="seismic_source_system.xml", calc_id=None, validate=False)


## Sample logic tree
from hazard.psha.Projects.SHRE_NPP.params.gmpe import gmpe_system
#gmpe_system.plot_diagram()

ltp = LogicTreeProcessor(None, source_model_lt=source_model_lt, gmpe_lt=gmpe_system)
rnd = random.Random()
rnd.seed(42)
for i in range(10):
	sm_name, path = ltp.sample_source_model_logictree(rnd.randint(-(2 ** 31), (2 ** 31) - 1))
	print sm_name, path
	#source_model_lt.plot_diagram(highlight_path=path)

	for sm in source_models:
		if sm.name == os.path.splitext(sm_name)[0]:
			for src in sm:
				src = copy.deepcopy(src)
				apply_uncertainties = ltp.parse_source_model_logictree_path(path)
				print "  %s" % src.source_id
				print "    %.1f %.3f %.3f" % (src.mfd.max_mag, src.mfd.a_val, src.mfd.b_val)
				apply_uncertainties(src)
				print "    %.1f %.3f %.3f" % (src.mfd.max_mag, src.mfd.a_val, src.mfd.b_val)


## Enumerate logic tree
#for i, (smlt_path_weight, smlt_path) in enumerate(ltp.source_model_lt.root_branchset.enumerate_paths()):
#	print i, smlt_path_weight, [branch.branch_id for branch in smlt_path]
