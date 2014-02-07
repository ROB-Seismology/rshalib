"""
Test CAV filtering in openquake.hazardlib
"""

import openquake.hazardlib as oqhazlib
import hazard.rshalib as rshalib


### Model name
psha_model_name = "CAV Test"


### PSHA parameters

## MFD
Mmin = 4.5
Mmax = 7.2
mfd_bin_width = 0.2

## GMPE truncation level
gmpe_truncation_level = 3

## Rupture parameters
rupture_mesh_spacing = 5.
rupture_aspect_ratio = 1.
upper_seismogenic_depth = 0.
lower_seismogenic_depth = 25.

## Integration distance
integration_distance = 100.

## Sites
ref_soil_params = {"vs30": 800, "vs30measured": True, "z1pt0": 100., "z2pt5": 2., "kappa": None}
sites = [rshalib.site.SHASite(5.25, 50.50, name='Near Tihange')]
grid_outline = []
grid_spacing = '10km'
soil_site_model = None

## Intensity measure type, spectral periods, and intensity levels
imt_periods = {'PGA': [0]}
Imin = 1E-3
Imax = 1.0
num_intensities = 25

## Return periods and investigation time
return_periods = [1E+3, 1E+4, 1E+5]
investigation_time = 50.


### Create source model

## Nodal-plane distribution
strike, dip, rake = 0, 45, -90
nodal_plane = rshalib.geo.NodalPlane(strike, dip, rake)
npd = rshalib.pmf.NodalPlaneDistribution([nodal_plane], [1])

## Hypocentral depth distribution
hdd = rshalib.pmf.HypocentralDepthDistribution([10], [1])

## MFD
a_val, b_val = 1., 0.95
mfd = rshalib.mfd.TruncatedGRMFD(Mmin, Mmax, mfd_bin_width, a_val, b_val)

## TRT
trt = "Stable Shallow Crust"

## MSR
msr = oqhazlib.scalerel.WC1994()

## Point source
location = sites[0]
src = rshalib.source.PointSource("PT", "Near Tihange", trt, mfd, rupture_mesh_spacing, msr, rupture_aspect_ratio, upper_seismogenic_depth, lower_seismogenic_depth, location, npd, hdd)

## Source model
source_model = rshalib.source.SourceModel("Test", [src])


### Ground-motion model
ground_motion_model = rshalib.gsim.GroundMotionModel('SCR_AkB2010', {trt: 'AkkarBommer2010'})


### PSHA model
output_folder = r"D:\PSHA\SHRE_NPP\cav_test"
psha_model = rshalib.shamodel.PSHAModel(psha_model_name, source_model, ground_motion_model,
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


### Run
shcf = psha_model.run_nhlib_shcf(cav_min=0.)["PGA"]
shcf_cav = psha_model.run_nhlib_shcf(cav_min=0.16)["PGA"]

hc = shcf.getHazardCurve()
hc_cav = shcf_cav.getHazardCurve()

hcc = rshalib.result.HazardCurveCollection([hc, hc_cav], labels=["No CAV", "CAV-filtered"])
hcc.plot()
