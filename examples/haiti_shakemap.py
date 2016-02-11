"""
Ground-motion field due to fault source
Haiti example from Eric Calais
"""


if __name__ == "__main__":
	import os
	import numpy as np
	import openquake.hazardlib as oqhazlib
	import hazard.rshalib as rshalib
	import mapping.Basemap as lbm
	from mapping.Basemap.cm.norm import PiecewiseLinearNorm


	## Output folder
	out_folder = r"D:\PSHA\Haiti_Calais"


	## Common parameters
	trt = "ASC"
	rms = 2.5
	msr = "WC1994"
	rar = 1.0

	## Fault 1
	id1 = "FLT1"
	name1 = "Fault 1"
	M1 = 7.
	mfd1 = rshalib.mfd.CharacteristicMFD(M1, 1, 0.1)
	dip1, rake1 = 37, 56
	usd1, lsd1 = 0, 5
	lons1 = [-72.35, -72.05]
	lats1 = [18.56, 18.53]
	trace1 = rshalib.geo.Line([rshalib.geo.Point(lon, lat) for lon, lat in zip(lons1, lats1)])
	fault1 = rshalib.source.SimpleFaultSource(id1, name1, trt, mfd1, rms, msr, rar,
											  usd1, lsd1, trace1, dip1, rake1)

	## Fault 2
	id2 = "FLT2"
	name2 = "Fault 2"
	M2 = 7.
	mfd2 = rshalib.mfd.CharacteristicMFD(M2, 1, 0.1)
	dip2, rake2 = 89.5, 90
	usd2, lsd2 = 0, 10
	lons2 = [-72.35, -72.05]
	lats2 = [18.51, 18.51]
	trace2 = rshalib.geo.Line([rshalib.geo.Point(lon, lat) for lon, lat in zip(lons2, lats2)])
	fault2 = rshalib.source.SimpleFaultSource(id2, name2, trt, mfd2, rms, msr, rar,
											  usd2, lsd2, trace2, dip2, rake2)
	# TODO: to_characteristic_source should adjust MFD too, to make sure rupture probability = 1


	# Define GMPEs
	gmpe_names = ["BooreAtkinson2008", "CampbellBozorgnia2008", "ChiouYoungs2008"]
	#gmpe_names = ["BooreAtkinson2008"]
	gmpe_system_defs = []
	for gmpe_name in gmpe_names:
		gmpe_system_def = {}
		gmpe_pmf = rshalib.pmf.GMPEPMF([gmpe_name], [1])
		gmpe_system_def[trt] = gmpe_pmf
		gmpe_system_defs.append(gmpe_system_def)
	gmpe_system_def = {}
	gmpe_pmf = rshalib.pmf.GMPEPMF(gmpe_names, rshalib.pmf.get_uniform_weights(len(gmpe_names)))
	gmpe_system_def[trt] = gmpe_pmf
	gmpe_system_defs.append(gmpe_system_def)


	## Define site model
	grid_outline = [-73.75, -70.75, 17.5, 20]
	grid_spacing = (0.1, 0.1)
	soil_site_model = None

	imt_periods = {'PGA': [0], 'SA': [0.25, 1.]}
	period_list = sorted(np.sum(imt_periods.values()))

	truncation_level = 0
	integration_distance = 300
	model_name = "Haiti"


	## Compute ground_motion field
	print("Computing ground-motion fields...")

	for flt in [fault1, fault2]:
		src_model = rshalib.source.SourceModel(flt.name, [flt])
		char_src_model = rshalib.source.SourceModel(flt.name, [flt.to_characteristic_source()])

		for gmpe_system_def in gmpe_system_defs:
			if len(gmpe_system_def[trt]) == 1:
				gmpe_name = gmpe_system_def[trt].gmpe_names[0]
			else:
				gmpe_name = "AverageGMPE"

			dsha_model = rshalib.shamodel.DSHAModel(model_name, char_src_model, gmpe_system_def,
							grid_outline=grid_outline, grid_spacing=grid_spacing,
							soil_site_model=soil_site_model, imt_periods=imt_periods,
							truncation_level=truncation_level, integration_distance=integration_distance)

			#uhs_field = dsha_model.calc_gmf_fixed_epsilon_mp(num_cores=4, stddev_type="total")
			#correlation_model = oqhazlib.correlation.JB2009CorrelationModel(vs30_clustering=True)
			correlation_model = None
			uhs_field = dsha_model.calc_random_gmf(correlation_model=correlation_model, random_seed=42)[0]
			num_sites = uhs_field.num_sites


			## Plot map
			for T in period_list:
				#contour_interval = 0.05
				#norm = None
				contour_interval = {0: 0.08, 0.25: 0.15, 1: 0.05}[T]
				breakpoints = [0., 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6]
				norm = PiecewiseLinearNorm(breakpoints)
				title = "%s, %s (mean + %s sigma)" % (src_model.name, gmpe_name, truncation_level)
				hm = uhs_field.getHazardMap(period_spec=T)
				site_style = lbm.PointStyle(shape=".", line_color="k", size=0.5)
				map = hm.get_plot(graticule_interval=(1,1), cmap="usgs", norm=norm,
								contour_interval=contour_interval, num_grid_cells=num_sites,
								title=title, projection="merc", site_style=site_style,
								source_model=src_model)
				#map.legend_style = None
				fig_filespec = None
				#fig_filename = "%s_%s_T=%ss.PNG" % (src_model.name, gmpe_name, T)
				#fig_filespec = os.path.join(out_folder, fig_filename)
				map.plot(fig_filespec=fig_filespec, dpi=150)
				#exit()
