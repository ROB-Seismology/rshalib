"""
Template for building a generic PSHA model that can be run in
OpenQuake, nhlib or CRISIS
"""

import os
import numpy as np

from openquake.hazardlib.imt import PGA, SA
import hazard.rshalib as rshalib
#from hazard.psha.openquake.nhlib_nrml import *
#from hazard.psha.openquake.nhlib_rob import *
#from hazard.psha.openquake.lt_nrml import create_basic_seismicSourceSystem, GroundMotionSystem
#from hazard.psha.openquake.rob_sourceModels import *


### Model name
psha_model_name = "ROB_PSHA_model_example"


### Folder names
psha_root_folder = r"D:\PSHA\Test\OpenQuake\Tests"
psha_model_folder = os.path.join(psha_root_folder, psha_model_name)
if not os.path.exists(psha_model_folder):
	os.mkdir(psha_model_folder)

### PSHA parameters
## Threshold magnitude
Mmin = 4.5

## Define site model
sites = [rshalib.site.GenericSite(*(3.71704, 51.43233), name='Borssele_site')]
site_model = rshalib.site.GenericSiteModel(sites)

ref_soil_params = (800., True, 2., 1.)
#site_model = rshalib.site.SoilSiteModel('test_soil_site_model', [(3.71704, 51.43233)])

## Intensity measure type, spectral periods, and intensity levels
imt = 'SA'
imt_periods = {'SA': [0.5, 1.]}
Imin = 1E-3
Imax = 1.0
num_intensities = 15

## Return periods and investigation time
return_periods = [1E+3, 1E+4, 1E+5]
investigation_time = 50.

## Additional OpenQuake params
OQparams = {}
OQparams['random_seed'] = 42
OQparams['number_of_logic_tree_samples'] = 1
OQparams['rupture_mesh_spacing'] = 5.
OQparams['area_source_discretization'] = 5.
OQparams['width_of_mfd_bin'] = 0.2
OQparams['export_dir'] = 'output'
OQparams['mean_hazard_curves'] = True
OQparams['quantile_hazard_curves'] = [0.05, 0.16, 0.50, 0.84, 0.95]
OQparams['poes_hazard_maps'] = [0.1]


### Create source models
## Read from GIS tables
trt = 'Stable Shallow Crust'
source_model_names = ["Seismotectonic", "TwoZone_split"]
source_models = [rshalib.rob.read_source_model(name, min_mag=Mmin)
				for name in source_model_names]

## Example of how parameters of individual sources can be overridden
source = source_models[0]['RVG']
source.tectonic_region_type = trt
source = source_models[1]['RVG']
source.tectonic_region_type = trt

## Create ad hoc source model
mfd = rshalib.mfd.TruncatedGRMFD(4.0, 6.7, 0.1, 1.27, 0.87)
mfd = mfd.to_evenly_discretized_mfd()
mfd.min_mag = 4.0
strikes, weights = rshalib.pmf.get_uniform_distribution(0., 360., 0)
npd = rshalib.pmf.NodalPlaneDistribution([rshalib.geo.NodalPlane(strike, 45., 0)
											for strike in strikes], weights)
hypo_depths, weights = rshalib.pmf.get_normal_distribution(5., 15., 1)
hdd = rshalib.pmf.HypocentralDepthDistribution(hypo_depths, weights)
polygon = rshalib.geo.Polygon([rshalib.geo.Point(lon, lat)
							for lon, lat in zip([3.25, 4.50, 4.50, 3.25],
							[50.9, 50.9, 51.8, 51.8])])
source = rshalib.source.AreaSource('bg', 'background', trt, mfd, 1., 'WC1994',
									1., 0., 20., npd, hdd, polygon, 5.)
source_model = rshalib.source.SourceModel('Borssele_source_model1', [source])
#source_models.append(source_model)
source_models = [source_model]

source_model_weights = np.ones(len(source_models), 'f') / len(source_models)
source_model_pmf = rshalib.pmf.SourceModelPMF(source_models, source_model_weights)
source_model_lt = rshalib.logictree.SeismicSourceSystem('', source_model_pmf)

### Construct ground-motion model
gmpe_list = ['BooreAtkinson2008', 'BergeThierry2003', 'AmbraseysEtAl1996']
gmpe_weights = np.ones(len(gmpe_list), 'f') / len(gmpe_list)
ground_motion_pmf = rshalib.pmf.GMPEPMF(gmpe_list, gmpe_weights)
groud_motion_lt = rshalib.logictree.GroundMotionSystem('', {trt: ground_motion_pmf},
														use_short_names=False)



if __name__ == '__main__':
	"""
	"""
	## oqhazlib
	output_folder = os.path.join(psha_model_folder, "oqhazlib")
	if not os.path.exists(output_folder):
		os.mkdir(output_folder)
	psha_model_tree = rshalib.shamodel.PSHAModelTree(psha_model_name, source_model_lt,
											groud_motion_lt, root_folder=output_folder,
											site_model=site_model,
											ref_soil_params=ref_soil_params,
											imt_periods=imt_periods, time_span=50.,
											truncation_level=3., integration_distance=200.)
	for psha_model, weight in psha_model_tree.sample_logic_trees(5):
		print(psha_model.source_model.name, psha_model.ground_motion_model.name, weight)
	#psha_model.calc_shcf_mp(calc_id='test')

	## OpenQuake
#	output_folder = os.path.join(psha_model_folder, "OQ")
#	if not os.path.exists(output_folder):
#		os.mkdir(output_folder)
#	psha_model.write_openquake(OQparams)
#	psha_model.write_crisis()
#	imts = psha_model._get_openquake_imts()
#	print imts.keys()

	## CRISIS
#	output_folder = os.path.join(psha_model_folder, "CRISIS")
#	if not os.path.exists(output_folder):
#		os.mkdir(output_folder)
#	psha_model_tree = PSHAModelTree(name, source_models, source_model_lt, ground_motion_models, output_dir=output_folder, sites=sites, soil_site_model=soil_site_model, imt_periods=imt_periods, time_span=50., truncation_level=3., integration_distance=200., lts_sampling_method='enumerated', num_lts_samples=2)
#	psha_model_tree.run_nhlib()
#	psha_model_tree.write_openquake(params)
#	psha_model_tree.write_crisis()

