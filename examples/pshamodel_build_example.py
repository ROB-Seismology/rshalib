"""
Template for building a generic simple PSHA model (no logic tree)
that can be run in OpenQuake, hazardlib or CRISIS

Don't modify this example to test something, make a copy instead!
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import os
import numpy as np

from openquake.hazardlib.imt import PGA, SA
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


## Define site model, either generic site model and reference soil parameters
## or a full-fledged soil site model

## Reference soil parameters for generic site model
ref_soil_params = {"vs30": 800, "vs30measured": True, "z1pt0": 100.,
					"z2pt5": 2., "kappa": 0.03}

## Grid, list of sites or soil site model
## Grid
#grid_outline = [(2.15, 49.15), (6.95, 51.95)]
#grid_spacing = 0.1
#grid_spacing = 1.0
#grid_spacing = '5km'
#site_model = rshalib.site.GenericSiteModel.from_grid_spec(grid_outline, grid_spacing

## List of sites
#sites = [rshalib.site.GenericSite(3.71704, 51.43233, name='Borssele')]
#site_model = rshalib.site.GenericSiteModel(sites)

## Soil site model, with different vs30 and kappa for each site
doel_soil_params, tihange_soil_params = {}, {}
doel_soil_params.update(ref_soil_params)
doel_soil_params['vs30'] = 2000
doel_soil_params['kappa'] = 0.01
tihange_soil_params.update(ref_soil_params)
doel = rshalib.site.SoilSite(4.259, 51.325, soil_params=doel_soil_params, name="Doel")
tihange = rshalib.site.SoilSite(5.274, 50.534, soil_params=tihange_soil_params,
								name="Tihange")
site_model = rshalib.site.SoilSiteModel([doel, tihange], "")


## Intensity measure type, spectral periods, and intensity levels
#imt_periods = {'PGA': [0], 'SA': [0.5, 1.]}
#imt_periods = {'SA': [1.]}
imt_periods = {'PGA': [0]}
Imin = 1E-3
Imax = 1.0
num_intensities = 15

## Return periods and investigation time
return_periods = [1E+3, 1E+4, 1E+5]
investigation_time = 50.


### Create source model

## Nodal-plane distribution
#min_strike, max_strike, delta_strike = 0, 270, 90
min_strike, max_strike, delta_strike = 45, 45, 0
dip = 45.
#min_rake, max_rake, delta_rake = -90, 90, 90
min_rake, max_rake, delta_rake = -90, -90, 0
npd = rshalib.pmf.create_nodal_plane_distribution((min_strike, max_strike, delta_strike),
												dip, (min_rake, max_rake, delta_rake))
npd.print_distribution()

## Hypocentral depth distribution
#hypo_depths, hypo_weights = rshalib.pmf.get_normal_distribution(5., 15., 3)
hypo_depths, hypo_weights = rshalib.pmf.get_normal_distribution(10., 10., 0)
hdd = rshalib.pmf.HypocentralDepthDistribution(hypo_depths, hypo_weights)
hdd.print_distribution()


msr = 'WC1994'
#msr = 'PointMSR'

## Read from GIS tables
trt = 'Stable Shallow Crust'
source_model = rshalib.rob.read_source_model("Leynaud_extended",
											source_ids=['NBM', 'SBM', 'RVG',
											'LIE', 'HFAG', 'HAIN', 'PDC',
											'ART', 'ARD', 'LIM'],
											min_mag=Mmin,
											mfd_bin_width=mfd_bin_width,
											rupture_mesh_spacing=rupture_mesh_spacing,
											area_discretization=area_discretization,
											hypocentral_distribution=hdd,
											nodal_plane_distribution=npd,
											magnitude_scaling_relationship=msr)
for source in source_model:
	print("%s: a=%.3f, b=%.3f"
			% (source.source_id, source.mfd.a_val, source.mfd.b_val))


## Example of how parameters of individual sources can be overridden
#source = source_model["RVG"]
#source.tectonic_region_type = trt

## Create ad hoc source model
Mmax, a_val, b_val = 6.7, 1.27, 0.87
mfd = rshalib.mfd.TruncatedGRMFD(Mmin, Mmax, mfd_bin_width, a_val, b_val)
#print(mfd.get_center_magnitudes())
## Decrease minimum magnitude in MFD by half the bin width
#mfd2 = mfd.to_evenly_discretized_mfd()
#mfd2.min_mag = Mmin
#mfd2.occurrence_rates = np.concatenate([[0.], mfd2.occurrence_rates])
#mfd._get_rate(mfd2.get_center_magnitudes())
#mfd = mfd2
#print(mfd.get_center_magnitudes())

polygon = rshalib.geo.Polygon([rshalib.geo.Point(lon, lat)
							for lon, lat in zip([3.25, 4.50, 4.50, 3.25],
												[50.9, 50.9, 51.8, 51.8])])
rupture_aspect_ratio = 1.
upper_seismogenic_depth, lower_seismogenic_depth = 0., 20.
source = rshalib.source.AreaSource('bg', 'background', trt, mfd,
									rupture_mesh_spacing, msr,
									rupture_aspect_ratio, upper_seismogenic_depth,
									lower_seismogenic_depth, npd, hdd, polygon,
									area_discretization)
#source_model = rshalib.source.SourceModel('Borssele_source_model1', [source])

### Test export to/import from json
#source_model = SourceModel.from_json(source_model.to_json())
#print(source_model.sources[0].hypocenter_distribution.hypo_depths)
#print(source_model.sources[0].hypocenter_distribution.data)


### Construct ground-motion model
#ground_motion_model = rshalib.gsim.GroundMotionModel('SCR_BA2008', {trt: 'BooreAtkinson2008'})
#ground_motion_model = rshalib.gsim.GroundMotionModel('SCR_BT2003', {trt: 'BergeThierry2003'})
#ground_motion_model = rshalib.gsim.GroundMotionModel('SCR_A1996', {trt: 'AmbraseysEtAl1996'})
ground_motion_model = rshalib.gsim.GroundMotionModel('Mixed',
									{'Active Shallow Crust': 'BindiEtAl2011',
									'Stable Shallow Crust': 'BooreAtkinson2008'})


def create_oq_params():
	## Additional OpenQuake params not automatically handled by pshamodel
	OQparams = {}
	OQparams['export_dir'] = 'output'
	return OQparams


def create_psha_model(engine="oqhazlib"):
	"""
	:param engine:
		String, name of hazard engine: "oqhazlib", "openquake" or "crisis"
		(default: "oqhazlib")
	"""
	root_folder = os.path.join(psha_model_folder, engine)
	if not os.path.exists(root_folder):
		os.mkdir(root_folder)
	return rshalib.shamodel.PSHAModel(psha_model_name, source_model, ground_motion_model,
					root_folder=root_folder,
					site_model=site_model,
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

	## oqhazlib
	psha_model = create_psha_model("oqhazlib")
	#print(psha_model._get_trt_gsim_dict())
	start_time = time.time()
	#shcf_dict = psha_model.calc_shcf()
	shcf_dict = psha_model.calc_shcf_mp(decompose_area_sources=False, num_cores=3)
	shcf = shcf_dict['PGA']
	end_time = time.time()
	print(end_time - start_time)
	tree = rshalib.nrml.create_nrml_root(shcf)
	from lxml import etree
	print(etree.tostring(tree, pretty_print=True))
	#shcf.write_nrml(r"C:\Temp\shcf.xml")
	shcf.plot()
	exit()


	## oqhazlib deaggregation
	site_index = 0
	site = psha_model.get_soil_site_model()[site_index]
	imt = PGA()
	#imt = eval(imt)(imt_periods[imt][0], 5.)
	print(imt)
	iml = 0.2
	#deagg_result = psha_model.deagg_oqhazlib(site, imt, iml, return_periods[0])
	site_imtls = {(site.lon, site.lat): {imt: [iml]}}
	#sdc = psha_model.deaggregate(site_imtls)[(site.lon, site.lat)]
	sdc = psha_model.deaggregate_mp(site_imtls, num_cores=3)[(site.lon, site.lat)]
	dc = sdc[0]
	ds = dc[0]
	print(ds.mag_bin_edges)
	print(ds.dist_bin_edges)
	print(ds.lon_bin_edges)
	print(ds.lat_bin_edges)
	print(ds.eps_bin_edges)
	print(ds.trt_bins)
	print(ds.matrix.shape)
	print(ds.matrix.max())
	ds.plot_mag_dist_pmf()


	## OpenQuake
	"""
	psha_model = create_psha_model("openquake")
	OQparams = create_oq_params()
	psha_model.write_openquake(user_params=OQparams)
	"""


	## Crisis
	"""
	psha_model = create_psha_model("crisis")
	gra_filespec = psha_model.write_crisis(overwrite=False)
	#shcf = rshalib.crisis.read_GRA(os.path.splitext(gra_filespec)[0])
	#shcf.plot()
	"""
