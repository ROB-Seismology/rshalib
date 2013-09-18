"""
Template for building a generic PSHA model that can be run in
OpenQuake, nhlib or CRISIS
"""

import os
import numpy as np

from hazard.psha.openquake.nhlib_nrml import *
from hazard.psha.openquake.nhlib_rob import *
from hazard.psha.openquake.lt_nrml import create_basic_seismicSourceSystem, GroundMotionSystem
from hazard.psha.openquake.rob_sourceModels import *
from hazard.psha.shamodel import PSHAModelTree, PSHAModel

from nhlib.imt import PGA, SA


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

## Grid, sites or site model
grid_outline = [(2.15, 49.15), (6.95, 51.95)]
#grid_spacing = 0.1
grid_spacing = 1.0

sites = []
sites = [SHASite(*(3.71704, 51.43233), name='Borssele_site')]

ref_site_params = (800., True, 2., 1.)
soil_site_model = None
#soil_site_model = create_soil_site_model('test_soil_site_model', [(3.71704, 51.43233)])

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
source_models = [create_rob_source_model(name, min_mag=Mmin) for name in source_model_names]

## Example of how parameters of individual sources can be overridden
source = source_models[0].RVG
source.tectonic_region_type = trt

## Create ad hoc source model
mfd = TruncatedGRMFD(4.0, 6.7, 0.1, 1.27, 0.87)
mfd = mfd.to_evenly_discretized_mfd()
mfd.min_mag = 4.0
strikes, weights = get_uniform_distribution(0., 360., 0)
npd = NodalPlaneDistribution([NodalPlane(strike, 45., 0) for strike in strikes], weights)
hypo_depths, weights = get_normal_distribution(5., 15., 1)
hdd = HypocentralDepthDistribution(hypo_depths, weights)
polygon = Polygon([Point(lon, lat) for lon, lat in zip([3.25, 4.50, 4.50, 3.25], [50.9, 50.9, 51.8, 51.8])])
source = AreaSource('bg', 'background', trt, mfd, 1., WC1994(), 1., 0., 20., npd, hdd, polygon, 5.)
source_model = SourceModel('Borssele_source_model1', [source])
source_models.append(source)
source_model_weights = numpy.ones(len(source_models), 'f') / len(source_models)

### Construct ground-motion model
ground_motion_model1 = GroundMotionModel('BooreAtkinson2008', {trt: 'BooreAtkinson2008'})
ground_motion_model2 = GroundMotionModel('BergeThierry2003', {trt: 'BergeThierry2003'})
ground_motion_model3 = GroundMotionModel('AmbraseysEtAl1996', {trt: 'AmbraseysEtAl1996'})

ground_motion_models = [ground_motion_model1, ground_motion_model2, ground_motion_model3]


source_model_lt = None
ground_motion_model_lt = None
soil_site_model_lt = None



if __name__ == '__main__':
	"""
	"""
	## nhlib
	output_folder = os.path.join(psha_model_folder, "nhlib")
	if not os.path.exists(output_folder):
		os.mkdir(output_folder)
	psha_model = PSHAModel(psha_model_name, source_model, ground_motion_model1, output_dir=output_folder, sites=sites, grid_outline=grid_outline, grid_spacing=grid_spacing, soil_site_model=soil_site_model, imt_periods=imt_periods, time_span=50., truncation_level=3., integration_distance=200.)
	shcfs = psha_model.run_nhlib_shcf(write=True)
	shcfs['SA'].plot()

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

