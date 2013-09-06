"""
Test logic tree
"""

import hazard.rshalib as rshalib

## Source models
sm1 = rshalib.rob.create_rob_source_model("Leynaud")
sm1.sources = sm1.sources[-3:]
sm2 = rshalib.rob.create_rob_source_model("TwoZone")
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
		MFD_pmf_dict[sm.name][src.source_id] = rshalib.pmf.MFDPMF([(1.3, 1.0), (1.2, 0.9), (1.4, 1.1)], [0.4, 0.2, 0.4])

## Construct logic tree
source_model_lt = rshalib.logictree.SeismicSourceSystem.from_independent_uncertainty_levels("lt", source_model_pmf, Mmax_pmf_dict, MFD_pmf_dict)
source_model_lt.write_xml(r"C:\Temp\seismic_source_system.xml")

#for branching_level in source_model_lt:
#	for branch_set in branching_level:
#		print branch_set.applyToSources
