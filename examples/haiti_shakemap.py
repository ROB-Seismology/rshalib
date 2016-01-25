"""
Ground-motion field due to fault source
Haiti example from Eric Calais
"""


if __name__ == "__main__":
	import hazard.rshalib as rshalib
	import mapping.Basemap as lbm


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
	fault1 = fault1.to_characteristic_source()

	## Fault 2
	id2 = "FLT2"
	name2 = "Fault 2"
	M2 = 7.
	mfd2 = rshalib.mfd.CharacteristicMFD(M2, 1, 0.1)
	dip2, rake2 = 90, 90
	usd2, lsd2 = 0, 10
	lons2 = [-72.35, -72.05]
	lats2 = [18.51, 18.51]
	trace2 = rshalib.geo.Line([rshalib.geo.Point(lon, lat) for lon, lat in zip(lons2, lats2)])
	fault2 = rshalib.source.SimpleFaultSource(id2, name2, trt, mfd2, rms, msr, rar,
											  usd2, lsd2, trace2, dip2, rake2)
	fault2 = fault2.to_characteristic_source()
	# TODO: to_characteristic_source should adjust MFD too, to make sure rupture probability = 1

	src_model = rshalib.source.SourceModel("Fault 2", [fault2])

	# Define GMPEs
	gmpe_system_def = {}
	#gmpe_names = ["BooreAtkinson2008", "CampbellBozorgnia2008", "ChiouYoungs2008"]
	gmpe_names = ["BooreAtkinson2008"]
	gmpe_pmf = rshalib.pmf.GMPEPMF(gmpe_names, rshalib.pmf.get_uniform_weights(len(gmpe_names)))
	gmpe_system_def[trt] = gmpe_pmf


	## Define site model
	grid_outline = [-73.75, -70.75, 17.5, 20]
	grid_spacing = (0.1, 0.1)
	soil_site_model = None

	imt_periods = {'PGA': [0], 'SA': [0.25, 1.]}
	truncation_level = 1
	integration_distance = 300
	model_name = "Haiti"

	dsha_model = rshalib.shamodel.DSHAModel(model_name, src_model, gmpe_system_def,
					grid_outline=grid_outline, grid_spacing=grid_spacing,
					soil_site_model=soil_site_model, imt_periods=imt_periods,
					truncation_level=truncation_level, integration_distance=integration_distance)


	## Compute ground_motion field
	print("Computing ground-motion field...")

	uhs_field = dsha_model.calc_gmf_envelope_mp(num_cores=4, stddev_type="total")
	num_sites = uhs_field.num_sites

	## Plot map
	T = 0.
	contour_interval = 0.05
	norm = None
	#contour_interval = None
	#breakpoints = [0., 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.]
	#norm = PiecewiseLinearNorm(breakpoints)
	title = "%s Ground-motion field" % (src_model.name.title())
	hm = uhs_field.getHazardMap(period_spec=T)
	site_style = lbm.PointStyle(shape=".", line_color="k", size=0.5)
	map = hm.get_plot(grid_interval=(1,1), cmap="usgs", norm=norm,
					contour_interval=contour_interval, num_grid_cells=num_sites,
					title=title, projection="merc", site_style=site_style)
	fig_filespec = None
	map.plot(fig_filespec=fig_filespec)
