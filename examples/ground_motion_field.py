"""
Example to compute ground-motion field due to a local earthquake
using rshalib
"""

# TODO, to ensure operation on web server:
# - Move source models to seismo-gis

# Advantages:
# - runs on server (but lot of requirements)
# - export to GeoTiff (image or single-band), which can be manipulated in web app


#out_folder = "D:\\Earthquake Reports\\20180525\\accelero"
#out_folder = "E:\\Home\\_kris\\Publications\\2018 - AcceleROB"
out_folder = "C:\Temp"


if __name__ == "__main__":
	import os
	from collections import OrderedDict
	import numpy as np
	import eqcatalog
	import hazard.rshalib as rshalib


	## Read earthquake from database
	#eq_id = 1306  ## Alsdorf
	#eq_id = 987  ## Roermond
	#eq_id = 5329  ## Ramsgate
	eq_id = 6625  ## Kinrooi
	catalog = eqcatalog.rob.query_local_eq_catalog_by_id(eq_id)
	[eq] = catalog.eq_list

	## Create point-source model
	Mtype = "MW"
	Mrelation = OrderedDict(MS="Utsu2002", ML="Ahorner1983")
	area_source_model_name = "Seismotectonic_Hybrid_v2"
	pt_src_model = rshalib.source.SourceModel.from_eq_catalog(catalog, Mtype,
						Mrelation, area_source_model_name=area_source_model_name)
	[src] = pt_src_model.sources
	print(src.mfd.max_mag)
	src.hypocenter_distribution.print_distribution()
	## Override TRT
	trt = "Stable Shallow Crust"
	src.tectonic_region_type = trt

	## Override nodal plane if known
	# TODO: read from database
	strike, dip, rake = 125, 55, -90
	npl = rshalib.geo.NodalPlane(strike, dip, rake)
	npd = rshalib.pmf.NodalPlaneDistribution([npl], [1])
	#src.nodal_plane_distribution = npd

	## Import GMPE logic tree
	version = 2015
	site_conditions = "rock"
	gmpe_system_def_lt = rshalib.rob.construct_gmpe_lt(version, site_conditions)
	gmpe_system_def = gmpe_system_def_lt.gmpe_system_def
	gmpe_spec = "GMPE logic tree (%s)" % site_conditions

	## Or define a single GMPE
	"""
	gmpe_system_def = {}
	#gmpe_name = "RietbrockEtAl2013MD"
	gmpe_name = "AtkinsonBoore2006Prime"
	#gmpe_name = "Atkinson2015"
	gmpe_pmf = rshalib.pmf.GMPEPMF([gmpe_name], [1])
	gmpe_system_def[trt] = gmpe_pmf
	gmpe_spec = gmpe_name + " GMPE"
	"""

	## Compute ground_motion field for single IMT or UHS for single point
	print("Computing ground-motion field...")

	## Common parameters for map / UHS
	model_name = ""
	truncation_level = 0
	integration_distance = 300
	np_aggregation = "max"
	intensity_unit = "m/s2"

	plot_map = True
	plot_uhs = False

	if plot_map:
		import mapping.layeredbasemap as lbm
		from mapping.layeredbasemap.cm.norm import PiecewiseLinearNorm

		#grid_outline = [(2, 49), (7, 52)]
		#grid_spacing = (0.2, 0.1)
		grid_outline = [(5, 50.75), (6, 51.5)]
		grid_spacing = (0.05, 0.025)

		site_model = rshalib.site.GenericSiteModel.from_grid_spec(grid_outline,
																grid_spacing)

		#graticule_interval = (2, 1)
		graticule_interval = (1, 0.5)

		resolution = 'h'

		#imt_periods = {'PGA': [0]}
		imt_periods = {'SA': [0.125]}

		dsha_model = rshalib.shamodel.DSHAModel(model_name, pt_src_model,
						gmpe_system_def, site_model, imt_periods=imt_periods,
						truncation_level=truncation_level,
						integration_distance=integration_distance)

		uhs_field = dsha_model.calc_gmf_fixed_epsilon_mp(num_cores=3,
							stddev_type="total", np_aggregation=np_aggregation)
		#[uhs_field] = dsha_model.calc_random_gmf_mp(num_cores=3, correlate_imt_uncertainties=True)
		num_sites = uhs_field.num_sites

		## Plot map
		#contour_interval = 0.02
		#contour_interval = 0.05
		contour_interval = 0.1
		#norm = None
		#contour_interval = None
		breakpoints = np.array([0., 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.]) * 2
		norm = PiecewiseLinearNorm(breakpoints)
		#title = "%s (%s)\nGround-motion field, %s"
		#title %= (eq.name.title(), eq.date, gmpe_spec)
		title = ""
		site_style = lbm.PointStyle('.', size=1, line_width=0, line_color=None,
									fill_color='k')

		T = imt_periods.values()[0][0]
		hm = uhs_field.get_hazard_map(period_spec=T)
		map = hm.get_plot(graticule_interval=graticule_interval,
						cmap="jet", norm=norm, contour_interval=contour_interval,
						num_grid_cells=num_sites, site_style=site_style,
						intensity_unit=intensity_unit,
						resolution=resolution, title=title, projection="merc")

		## Add topographic hillshading
		#layer = map.get_layer_by_name("intensity_grid")
		#elevation_grid = lbm.WCSData("http://seishaz.oma.be:8080/geoserver/wcs", "ngi:DTM10k", region=map.region, down_sampling=25)
		#blend_mode = "soft"
		#hillshade_style = lbm.HillshadeStyle(45, 45, 1, blend_mode=blend_mode,
		#										elevation_grid=elevation_grid)
		#layer.style.hillshade_style = hillshade_style

		## Add epicenter
		data = lbm.PointData(eq.lon, eq.lat)
		style = lbm.PointStyle('*', size=12, fill_color='m')
		layer = lbm.MapLayer(data, style)
		map.layers.append(layer)

		## Add stations
		label_style = lbm.TextStyle(font_size=8, vertical_alignment='top', offset=(0,-5))
		accelerob_stations = ['A029', 'A047', 'A066']
		accelerometers = ['KIN', 'MAS', 'BRE']
		seismometers = ['OPT']
		#all_stations = [accelerob_stations, accelerometers, seismometers]
		all_stations = [accelerometers, seismometers]
		colors = ['b', 'r', 'g']

		for stations, color in zip(all_stations, colors):
			lons, lats = eqcatalog.rob.get_station_coordinates(stations)
			data = lbm.MultiPointData(lons, lats, labels=stations)
			style = lbm.PointStyle('^', size=8, fill_color=color, label_style=label_style)
			layer = lbm.MapLayer(data, style)
			map.layers.append(layer)

			## Interpolate map value for each station
			accs = hm.get_site_intensities(lons, lats, intensity_unit=intensity_unit)
			imt = imt_periods.keys()[0]
			for s, station in enumerate(stations):
				acc = accs[s]
				lon, lat = lons[s], lats[s]
				d = eq.epicentral_distance((lon, lat))
				msg = '%s: %s=%.3f %s, d=%.1f km'
				msg %= (station, imt, acc, intensity_unit, d)
				print(msg)

		label = "GMPE: %s" % gmpe_spec
		pos = (0.035, 0.035)
		text_style = lbm.TextStyle(font_size=10, horizontal_alignment='left',
							vertical_alignment='bottom', multi_alignment='left',
							background_color='w', border_color='k', border_pad=0.5)
		map.draw_text_box(pos, label, text_style, zorder=10000)

		#fig_filespec = os.path.join(out_folder, "Kinrooi_gmf_T=%ss.png" % T)
		fig_filespec = None
		dpi = {None: 96}.get(fig_filespec, 300)
		map.plot(fig_filespec=fig_filespec, dpi=dpi)

		## Export grid layer to geotiff
		## Note: do not plot map before or map area will be shifted due to colorbar!
		layers = [lyr for lyr in map.layers if isinstance(lyr.data, lbm.GridData)]
		layers[0].style.color_map_theme.colorbar_style = None
		map.layers = layers
		#map.export_geotiff(r"C:\Temp\gmf.tif", dpi=300)
		#hm.export_geotiff(r"C:\Temp\gmf2", num_cells=num_sites)


	if plot_uhs:
		## Compute UHS for a particular point
		print("Computing UHS...")
		#station = 'A029'
		#station = 'MAS'
		#lon, lat = 5.790812, 51.095634
		station = 'BRE'
		#lon, lat = 5.595862, 51.138866
		[lon], [lat] = lons, lats = eqcatalog.rob.get_station_coordinates([station])
		print("d = %.1f km" % eq.epicentral_distance((lon, lat)))

		sites = [rshalib.site.GenericSite(lon, lat)]
		site_model = rshalib.site.GenericSiteModel(sites)

		imt_periods = {'PGA': [0],
					'SA': [0.05, 0.067, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.75, 1., 2.]}
					#'SA': [0.0303, 0.05, 0.1, 0.2, 0.3, 0.5, 1., 2.]}

		## Read transfer function
		ctfs_file = r"E:\Home\_kris\Projects\2015 - Belgoprocess\Results\TF\CTF_ModelB_MC.csv"
		ctfs = rshalib.siteresponse.read_TF_transfer1D(ctfs_file)
		tf = ctfs.percentile(50)

		## Separate computation for each GMPE in the logic tree
		gmpe_system_defs = [gmpe_system_def]
		#gmpe_system_defs = [gmpe_system_def_lt]
		#gmpe_pmf = gmpe_system_def_lt[trt]
		#gmpe_names = gmpe_pmf.gmpe_names
		#for gmpe_name in gmpe_names:
		#	gmpe_system_def = {trt: rshalib.pmf.GMPEPMF([gmpe_name], [1])}
		#	gmpe_system_defs.append(gmpe_system_def)

		rs_list = []
		labels = []
		for gmpe_system_def in gmpe_system_defs:
			dsha_model = rshalib.shamodel.DSHAModel(model_name, pt_src_model,
						gmpe_system_def, site_model=site_model,
						imt_periods=imt_periods, truncation_level=truncation_level,
						integration_distance=integration_distance)

			uhs_field = dsha_model.calc_gmf_fixed_epsilon_mp(num_cores=3,
							stddev_type="total", np_aggregation=np_aggregation)

			#idx = uhs_field.get_nearest_site_index(site)
			idx = 0
			brs = uhs_field.get_uhs(idx)

			distance = eq.epicentral_distance((lon, lat))
			title = "%s (%s) @ %s (d=%.1f km)"
			title %= (eq.name.title(), eq.date, station, distance)
			print("PGA: %.3f m/s2" % (brs[0.] * 9.80665))

			## Compute surface response spectrum
			pgm_freq = 50
			mag = eq.get_MW()
			#srs = brs.to_srs(tf, pgm_freq=pgm_freq, mag=mag, distance=max(10, distance))
			#vsrs = srs.get_vertical_spectrum(guidance='ASCE4-98')

			#rs_list.extend([brs, srs])
			rs_list.append(brs)

			gmpe_pmf = gmpe_system_def[trt]
			if len(gmpe_pmf) > 1:
				gmpe_name = "Logic tree (%s)" % site_conditions
				labels.extend(["%s (bedrock)" % gmpe_name,
								"%s (surface)" % gmpe_name])
			else:
				gmpe_name = gmpe_pmf.gmpe_names[0]
				labels.extend(["_nolegend_", "%s (surface)" % gmpe_name])
			#labels.append(gmpe_name)

		#spec_list = [brs, srs, vsrs]
		#labels = ["Bedrock RS (5% damping)", "Surface RS (hor.)", "Surface RS (vert.)"]
		#colors = ["r", "b", "b"]
		#linestyles = ['-', '-', '--']

		colors = ['r', 'r', 'g', 'g', 'b', 'b', 'c', 'c', 'm', 'm', 'k', 'k']
		#colors = ['r', 'g', 'b', 'c', 'm', 'k']
		linestyles = ['--', '-'] * 6
		#linestyles = ['-'] * 6
		uhsc = rshalib.result.UHSCollection(rs_list, labels=labels, colors=colors,
											linestyles=linestyles)

		#fig_filespec = os.path.join(out_folder, "Kinrooi_%s_rs_GMPE_%s.png" % (station, site_conditions))
		#fig_filespec = r"C:\Temp\RS_Atkinson2015_original_coeffs.PNG"
		fig_filespec = None
		uhsc.plot(title=title, intensity_unit='m/s2', plot_freq=True,
				legend_location=2, fig_filespec=fig_filespec)
