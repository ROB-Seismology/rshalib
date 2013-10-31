"""
Template for building a generic simple PSHA model (no logic tree)
that can be run in OpenQuake, hazardlib or CRISIS

Don't modify this example to test something, make a copy instead!
"""


import os
import numpy as np

from openquake.hazardlib.imt import PGA, SA
from openquake.hazardlib.scalerel import WC1994

import hazard.rshalib as rshalib


### Model name
psha_model_name = "ROB_PSHA_model_example"


### Folder names
psha_root_folder = r"D:\PSHA\Test\OpenQuake\Tests"
psha_model_folder = os.path.join(psha_root_folder, psha_model_name)
if not os.path.exists(psha_model_folder):
	os.mkdir(psha_model_folder)


### PSHA parameters

# TODO: parameter to turn on or off deaggregation

## Threshold magnitude and MFD bin width
Mmin = 4.0
mfd_bin_width = 0.1

## GMPE truncation level (in number of standard deviations)
gmpe_truncation_level = 3

## Discretization parameters (in km)
rupture_mesh_spacing = 5.
#area_discretization = 5.
area_discretization = 15.
integration_distance = 150.

## Grid, list of sites or site model
#grid_outline = []
grid_outline = [(2.15, 49.15), (6.95, 51.95)]
#grid_spacing = 0.1
#grid_spacing = 1.0
grid_spacing = '5km'

sites = []
#sites = [rshalib.site.SHASite(3.71704, 51.43233, name='Borssele')]


ref_soil_params = {"vs30": 800, "vs30measured": True, "z1pt0": 100., "z2pt5": 2.}
soil_site_model = None

## Intensity measure type, spectral periods, and intensity levels
imt_periods = {'PGA': [0], 'SA': [0.5, 1.]}
#imt_periods = {'PGA': [0]}
Imin = 1E-3
Imax = 1.0
num_intensities = 15

## Return periods and investigation time
return_periods = [1E+3, 1E+4, 1E+5]
investigation_time = 50.




### Create source model

## Nodal-plane distribution
min_strike, max_strike, delta_strike = 0, 270, 90
dip = 45.
min_rake, max_rake, delta_rake = -90, 90, 90
#min_rake, max_rake, delta_rake = -90, -90, 0
npd = rshalib.pmf.create_nodal_plane_distribution((min_strike, max_strike, delta_strike), dip, (min_rake, max_rake, delta_rake))

## Hypocentral depth distribution
#hypo_depths, hypo_weights = rshalib.pmf.get_normal_distribution(5., 15., 3)
hypo_depths, hypo_weights = rshalib.pmf.get_normal_distribution(10., 10., 0)
print hypo_depths, hypo_weights
hdd = rshalib.pmf.HypocentralDepthDistribution(hypo_depths, hypo_weights)

## Read from GIS tables
trt = 'Stable Shallow Crust'
source_model = rshalib.rob.create_rob_source_model("Leynaud_extended", min_mag=Mmin, mfd_bin_width=mfd_bin_width, rupture_mesh_spacing=rupture_mesh_spacing, area_discretization=area_discretization, hypocentral_distribution=hdd, nodal_plane_distribution=npd)
for source in source_model:
	print source.mfd.a_val

## Example of how parameters of individual sources can be overridden
source = source_model.RVG
#source.tectonic_region_type = trt

## Create ad hoc source model
Mmax, a_val, b_val = 6.7, 1.27, 0.87
mfd = rshalib.mfd.TruncatedGRMFD(Mmin, Mmax, mfd_bin_width, a_val, b_val)
#print mfd.get_center_magnitudes()
## Decrease minimum magnitude in MFD by half the bin width
#mfd2 = mfd.to_evenly_discretized_mfd()
#mfd2.min_mag = Mmin
#mfd2.occurrence_rates = np.concatenate([[0.], mfd2.occurrence_rates])
#mfd._get_rate(mfd2.get_center_magnitudes())
#mfd = mfd2
#print mfd.get_center_magnitudes()

polygon = rshalib.geo.Polygon([rshalib.geo.Point(lon, lat) for lon, lat in zip([3.25, 4.50, 4.50, 3.25], [50.9, 50.9, 51.8, 51.8])])
rupture_aspect_ratio = 1.
upper_seismogenic_depth, lower_seismogenic_depth = 0., 20.
source = rshalib.source.AreaSource('bg', 'background', trt, mfd, rupture_mesh_spacing, WC1994(), rupture_aspect_ratio, upper_seismogenic_depth, lower_seismogenic_depth, npd, hdd, polygon, area_discretization)
#source_model = rshalib.source.SourceModel('Borssele_source_model1', [source])

### Test export to/import from json
#source_model = SourceModel.from_json(source_model.to_json())
#print source_model.sources[0].hypocenter_distribution.hypo_depths
#print source_model.sources[0].hypocenter_distribution.data


### Construct ground-motion model
#ground_motion_model = rshalib.gsim.GroundMotionModel('SCR_BA2008', {trt: 'BooreAtkinson2008'})
#ground_motion_model = rshalib.gsim.GroundMotionModel('SCR_BT2003', {trt: 'BergeThierry2003'})
#ground_motion_model = rshalib.gsim.GroundMotionModel('SCR_A1996', {trt: 'AmbraseysEtAl1996'})
ground_motion_model = rshalib.gsim.GroundMotionModel('Mixed', {'Active Shallow Crust': 'BindiEtAl2011', 'Stable Shallow Crust': 'Campbell2003'})


def create_oq_params():
	## Additional OpenQuake params not automatically handled by pshamodel
	OQparams = {}
	OQparams['rupture_mesh_spacing'] = rupture_mesh_spacing
	OQparams['area_source_discretization'] = area_discretization
	OQparams['width_of_mfd_bin'] = mfd_bin_width
	OQparams['export_dir'] = 'output'
	return OQparams


def create_psha_model(engine="nhlib"):
	"""
	:param engine:
		String, name of hazard engine: "nhlib", "openquake" or "crisis"
		(default: "nhlib")
	"""
	output_folder = os.path.join(psha_model_folder, engine)
	if not os.path.exists(output_folder):
		os.mkdir(output_folder)
	return rshalib.shamodel.PSHAModel(psha_model_name, source_model, ground_motion_model,
					output_dir=output_folder,
					sites=sites,
					grid_outline=grid_outline,
					grid_spacing=grid_spacing,
					soil_site_model=soil_site_model,
					ref_soil_params=ref_soil_params,
					imt_periods=imt_periods,
					min_intensities=Imin,
					max_intensities=Imax,
					num_intensities=num_intensities,
					return_periods=return_periods,
					time_span=investigation_time,
					truncation_level=gmpe_truncation_level,
					integration_distance=integration_distance)



if __name__ == '__main__':
	"""
	"""
	import time

	## nhlib
	"""
	psha_model = create_psha_model("nhlib")
	start_time = time.time()
	shcfs = psha_model.run_nhlib_shcf(write=False)
	end_time = time.time()
	print end_time - start_time
	shcfs['PGA'].plot()
	"""

	## nhlib deaggregation
	"""
	psha_model = create_psha_model("nhlib")
	#import jsonpickle
	#json = jsonpickle.encode(psha_model)
	#psha_model = jsonpickle.decode(json)
	site_index = 0
	imt = PGA()
	#imt = eval(imt)(imt_periods[imt][0], 5.)
	print imt
	iml = 0.2
	deagg_result = psha_model.deagg_nhlib(site_index, imt, iml)
	print deagg_result.mag_bin_edges
	print deagg_result.dist_bin_edges
	print deagg_result.lon_bin_edges
	print deagg_result.lat_bin_edges
	print deagg_result.eps_bin_edges
	print deagg_result.trt_bins
	print deagg_result.matrix.shape
	print deagg_result.matrix.max()
	deagg_result.plot_mag_dist_pmf()
	"""

	## OpenQuake
	psha_model = create_psha_model("openquake")
	OQparams = create_oq_params()
	psha_model.write_openquake(user_params=OQparams)

	## Crisis
	#import rshalib.crisis.IO as IO
	#psha_model = create_psha_model("crisis")
	#gra_filespec = psha_model.write_crisis(overwrite=True)
	#shcf = IO.readCRISIS_GRA(os.path.splitext(gra_filespec)[0])
	#shcf.plot()
