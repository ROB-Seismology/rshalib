"""
Template for computing a ground-motion field for a stochastic event set

Don't modify this example to test something, make a copy instead!
"""



from openquake.hazardlib.imt import PGA, SA

import hazard.rshalib as rshalib


### Model name
dsha_model_name = "ROB_stochastic_event_set_example"


### SHA parameters

## Threshold magnitude and MFD bin width
Mmin = 4.5
mfd_bin_width = 0.1

## GMPE and truncation level (in number of standard deviations)
gmpe = 'FaccioliEtAl2010'
#gmpe = 'AkkarEtAl2013'
gmpe_truncation_level = 0

## Discretization parameters (in km)
rupture_mesh_spacing = 2.5
area_discretization = 10.
integration_distance = 150.


## Grid, list of sites or soil site model
## Grid
grid_outline = [(2.15, 49.15), (6.95, 51.95)]
grid_spacing = 0.25
ref_soil_params = {"vs30": 800, "vs30measured": True, "z1pt0": 100., "z2pt5": 2., "kappa": None}


## Intensity measure type, spectral periods, and intensity levels
#imt_periods = {'PGA': [0], 'SA': [0.5, 1.]}
imt_periods = {'PGA': [0]}

## Time span for Poisson temporal occurrence model
time_span = 15000.


### Create source model

## Read from GIS tables
source_model_name = "Seismotectonic_Hybrid"
source_model = rshalib.rob.create_rob_source_model(source_model_name, min_mag=Mmin,
												   mfd_bin_width=mfd_bin_width, rupture_mesh_spacing=rupture_mesh_spacing,
												   area_discretization=area_discretization, hypocentral_distribution=None,
												   nodal_plane_distribution=None)
src = source_model["GeHeF"]
src = source_model["LIE"]

## Generate stochastic event set
ruptures = src.get_stochastic_event_set_Poisson(time_span)
src.plot_rupture_bounds_3d(ruptures)


lons, lats, depths, mags = [], [], [], []
dsha_model = rshalib.shamodel.DSHAModel(dsha_model_name, lons, lats, depths, mags,
										gmpe, grid_outline=grid_outline, grid_spacing=grid_spacing,
										ref_soil_params=ref_soil_params, imts=imt_periods.keys(),
										periods=imt_periods.values()[0], correlation_model=None,
										truncation_level=gmpe_truncation_level, maximum_distance=integration_distance)
dsha_model.ruptures = ruptures

hazard_field_sets = dsha_model.run_hazardlib()
hazard_field = hazard_field_sets['PGA'].get_max_hazard_map()
hazard_field.return_period = time_span
map = hazard_field.get_plot(region=(5,7,50.5,51.5), title=dsha_model_name, contour_interval=0.05, source_model=source_model_name)
map.plot()

