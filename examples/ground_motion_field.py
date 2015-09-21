"""
Example of ground-motion field due to an earthquake
"""

# TODO, to ensure operation on web server:
# - Define selection for standard-rock and hard-rock
# - Move source models to seismo-gis


if __name__ == "__main__":
	import numpy as np
	import eqcatalog
	import hazard.rshalib as rshalib
	import mapping.Basemap as lbm
	from mapping.Basemap.cm.norm import PiecewiseLinearNorm


	## Read from database
	#eq_id = 1306  ## Alsdorf
	eq_id = 987  ## Roermond
	#eq_id = 5329  ## Ramsgate
	eq = eqcatalog.seismodb.query_ROB_LocalEQCatalogByID(eq_id)
	catalog = eqcatalog.EQCatalog([eq])

	## Create point-source model
	Mtype = "MW"
	Mrelation = {"ML": "Ahorner1983", "MS": "Utsu2002"}
	area_source_model_name = "Seismotectonic_Hybrid_v2"
	pt_src_model = rshalib.source.SourceModel.from_eq_catalog(catalog, Mtype,
						Mrelation, area_source_model_name=area_source_model_name)
	[src] = pt_src_model.sources
	## Override TRT
	trt = "Stable Shallow Crust"
	src.tectonic_region_type = trt

	## Override nodal plane if known
	strike, dip, rake = 125, 55, -90
	npl = rshalib.geo.NodalPlane(strike, dip, rake)
	npd = rshalib.pmf.NodalPlaneDistribution([npl], [1])
	#src.nodal_plane_distribution = npd

	## Import GMPEs defined in a recent hazard project
	#from hazard.psha.Projects.cAt_Rev.September2014.logictree.gmpe_lt import construct_gmpe_lt
	#rock_type = "soft"
	#gmpe_system_def = construct_gmpe_lt(rock_type).gmpe_system_def
	#gmpe_spec = "GMPE logic tree (%s rock)" % rock_type

	## Or define a single GMPE
	gmpe_system_def = {}
	gmpe_name = "RietbrockEtAl2013MD"
	gmpe_pmf = rshalib.pmf.GMPEPMF([gmpe_name], [1])
	gmpe_system_def[trt] = gmpe_pmf
	gmpe_spec = gmpe_name + " GMPE"

	## Compute ground_motion field
	print("Computing ground-motion field...")

	grid_outline = [(2, 49), (7,52)]
	grid_spacing = (0.2, 0.1)
	soil_site_model = None
	imt_periods = {'PGA': [0], 'SA': [0.05, 0.067, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.75, 1., 2.]}
	model_name = ""
	truncation_level = 1
	integration_distance = 300
	np_aggregation = "avg"

	dsha_model = rshalib.shamodel.DSHAModel(model_name, pt_src_model, gmpe_system_def,
					grid_outline=grid_outline, grid_spacing=grid_spacing,
					soil_site_model=soil_site_model, imt_periods=imt_periods,
					truncation_level=truncation_level, integration_distance=integration_distance)

	uhs_field = dsha_model.calc_gmf_envelope_mp(num_cores=4, stddev_type="total",
											np_aggregation=np_aggregation)
	num_sites = uhs_field.num_sites

	## Plot map
	T = 0.2
	#contour_interval = 0.02
	#norm = None
	contour_interval = None
	breakpoints = [0., 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.]
	norm = PiecewiseLinearNorm(breakpoints)
	title = "%s (%s)\nGround-motion field, %s" % (eq.name.title(), eq.date.isoformat(), gmpe_spec)
	hm = uhs_field.getHazardMap(period_spec=T)
	map = hm.get_plot(grid_interval=(2,1), cmap="jet", norm=norm, contour_interval=contour_interval, num_grid_cells=num_sites, title=title, projection="tmerc")
	map.plot()

	## Export grid layer to geotiff
	## Note: do not plot map before or map area will be shifted due to colorbar!
	layers = [lyr for lyr in map.layers if isinstance(lyr.data, lbm.GridData)]
	layers[0].style.color_map_theme.colorbar_style = None
	map.layers = layers
	#map.export_geotiff(r"C:\Temp\gmf.tif", dpi=300, verbose=True)

	## Plot UHS
	lon, lat = 4.367777777777778, 50.79499999999999
	site = rshalib.site.SHASite(lon, lat)
	idx = uhs_field.get_nearest_site_index(site)
	title = "%s (%s)\nResponse spectrum, %s" % (eq.name.title(), eq.date.isoformat(), gmpe_spec)
	uhs = uhs_field.getUHS(idx)
	uhs.plot(title=title)
