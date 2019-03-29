"""
Functions to download SHARE results from EFEHR website
"""

import numpy as np
import os
import urllib
from lxml import etree
from collections import OrderedDict

import hazard.rshalib as rshalib


def query_efehr(params, endpoint='map', verbose=False):
	"""
	Query EFEHR website

	:params:
		dict, mapping EFEHR URL parameters to values
	:param verbose:
		bool, prints url if set to True (default: False)

	:return:
		list with lines of data returned by the server
	"""
	## Construct URL
	base_url = "http://appsrvr.share-eu.org:8080/share"
	url = "%s/%s?%s" % (base_url, endpoint, urllib.urlencode(params))
	if verbose:
		print url

	f = urllib.urlopen(url)
	data = f.readlines()
	f.close()

	return data


def download_data(url, out_filespec):
	with open(out_filespec, 'wb') as fd:
		urlfd = urllib.urlopen(url)
		fd.writelines(urlfd.readlines())
		urlfd.close()


def get_map_models(lon, lat, verbose=False):
	"""
	Provide an index of all hazard models that are defined for a
	specific point of interest.

	:param lon:
		float, longitude
	:param lat:
		float, latitude
	:param verbose:
		bool, prints url if set to True (default: False)

	:return:
		dict mapping map id's to map names
	"""
	efehr_params = OrderedDict()

	efehr_params['lon'] = lon
	efehr_params['lat'] = lat

	xml_data = query_efehr(efehr_params, verbose=verbose)
	xml = etree.fromstring(xml_data[0])

	map_models = OrderedDict()
	for mdl_elem in xml.findall('model'):
		id = int(mdl_elem.find('id').text)
		name = mdl_elem.find('name').text
		map_models[id] = name

	return map_models


def get_map_imts(map_id, verbose=False):
	"""
	Provide all intensity measurement types (IMT) for which a specified
	model provides hazard information.

	:param map_id:
		int, map id
	:param verbose:
		bool, prints url if set to True (default: False)

	:return:
		list with (imt string, imt unit) tuples
	"""
	efehr_params = OrderedDict()

	efehr_params['id'] = map_id

	xml_data = query_efehr(efehr_params, verbose=verbose)
	xml = etree.fromstring(xml_data[0])

	imts = []
	for imt_elem in xml.findall('imtcode'):
		imt = imt_elem.find('code').text
		imt_unit = imt_elem.find('imunitdescr').text
		imts.append((imt, imt_unit))

	return imts


def get_map_poes(map_id, imt_string, verbose=False):
	"""
	Provide probability of exceedance and investigation timespan's with
	maps available, given model ID and intensity measurement types (IMT)
	for which a specified model provides hazard information.

	:param map_id:
		int, map id
	:param imt_string:
		str, imt string
	:param verbose:
		bool, prints url if set to True (default: False)

	:return:
		list with (poe, timespan) tuples
	"""
	efehr_params = OrderedDict()

	efehr_params['id'] = map_id
	efehr_params['imt'] = imt_string

	xml_data = query_efehr(efehr_params, verbose=verbose)
	xml = etree.fromstring(xml_data[0])

	poes = []
	for poe_elem in xml.findall('exceedance'):
		poe = float(poe_elem.find('hmapexceedprob').text)
		timespan = int(poe_elem.find('hmapexceedyears').text)
		poes.append((poe, timespan))

	return poes


def get_map_soil_classes(map_id, imt_string, poe, timespan, verbose=False):
	"""
	Provide site class with maps available, given model ID, IMT, POE,
	time span for which a specified model provides hazard information.

	:param map_id:
		int, map id
	:param imt_string:
		str, imt string
	:param poe:
		float, probability of exceedance
	:param timespan:
		int, investigation timespan
	:param verbose:
		bool, prints url if set to True (default: False)

	:return:
		list with soil classes
	"""
	efehr_params = OrderedDict()

	efehr_params['id'] = map_id
	efehr_params['imt'] = imt_string
	efehr_params['hmapexceedprob'] = poe
	efehr_params['hmapexceedyears'] = timespan

	xml_data = query_efehr(efehr_params, verbose=verbose)
	#xml = etree.fromstring('<soiltypes>' + xml_data[0] + '</soiltypes>')
	xml = etree.fromstring(xml_data[0])

	soil_classes = []
	for soil_elem in xml.findall('soiltype'):
		soil_class = soil_elem.find('type').text
		soil_classes.append(soil_class)

	return soil_classes


def get_map_aggregation_types(map_id, imt_string, poe, timespan, soil_class,
								verbose=False):
	"""
	Provide aggregation types with maps available, given model ID, IMT,
	POE, time span, site class for which a specified model provides
	hazard information.

	:param map_id:
		int, map id
	:param imt_string:
		str, imt string
	:param poe:
		float, probability of exceedance
	:param timespan:
		int, investigation timespan
	:param soil_class:
		str, soil class
	:param verbose:
		bool, prints url if set to True (default: False)

	:return:
		list with (aggregation type, aggregation level) tuples
	"""
	efehr_params = OrderedDict()

	efehr_params['id'] = map_id
	efehr_params['imt'] = imt_string
	efehr_params['hmapexceedprob'] = poe
	efehr_params['hmapexceedyears'] = timespan
	efehr_params['soiltype'] = soil_class

	xml_data = query_efehr(efehr_params, verbose=verbose)
	xml = etree.fromstring(xml_data[0])

	aggregation_types = []
	for fractile_elem in xml.findall('fractile'):
		agg_type = fractile_elem.find('aggregationtype').text
		agg_level = float(fractile_elem.find('aggregationlevel').text)
		aggregation_types.append((agg_type, agg_level))

	return aggregation_types


def get_map_wms_id(map_id, imt_string, poe, timespan, soil_class, agg_type,
					agg_level, verbose=False):
	"""
	Provide the map identity, and the layer reference for the web map
	service of the hazard map for given model ID, IMT, POE, time span,
	site class and aggregation type.

	:param map_id:
		int, map id
	:param imt_string:
		str, imt string
	:param poe:
		float, probability of exceedance
	:param timespan:
		int, investigation timespan
	:param soil_class:
		str, soil class
	:param agg_type:
		str, aggregation type
	:param agg_level:
		float, aggregation level
	:param verbose:
		bool, prints url if set to True (default: False)

	:return:
		tuple of (WMS id, WMS name)
	"""
	efehr_params = OrderedDict()

	efehr_params['id'] = map_id
	efehr_params['imt'] = imt_string
	efehr_params['hmapexceedprob'] = poe
	efehr_params['hmapexceedyears'] = timespan
	efehr_params['soiltype'] = soil_class
	efehr_params['aggregationtype'] = agg_type
	efehr_params['aggregationlevel'] = agg_level

	xml_data = query_efehr(efehr_params, verbose=verbose)
	xml = etree.fromstring(xml_data[0])

	wms_id = int(xml.find('hmapid').text)
	wms_name = xml.find('hmapwms').text

	return (wms_id, wms_name)


def get_map_wms_url(map_id, imt_string, poe, timespan, soil_class, agg_type,
					agg_level, verbose=False):
	"""
	Provide the URL for the web map service for given model ID, IMT, POE,
	time span, site class and aggregation type.

	:param map_id:
		int, map id
	:param imt_string:
		str, imt string
	:param poe:
		float, probability of exceedance
	:param timespan:
		int, investigation timespan
	:param soil_class:
		str, soil class
	:param agg_type:
		str, aggregation type
	:param agg_level:
		float, aggregation level
	:param verbose:
		bool, prints url if set to True (default: False)

	:return:
		int, WMS id
	"""
	wms_id, wms_name = get_map_wms_id(map_id, imt_string, poe, timespan, soil_class,
									agg_type, agg_level, verbose=verbose)

	#url = 'http://efehrmaps.ethz.ch/cgi-bin/mapserv?MAP=/var/www/mapfile/sharehazard.01.map&LAYERS=%s&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap' % wms_name
	url = ('http://gemmsrvr.ethz.ch/cgi-bin/mapserv?'
			'MAP=/var/www/mapfile/sharehazard.01.map'
			'&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&srs=epsg:4326'
			'&LAYERS=%s&bbox=-25.27,34.45,44.83,71.55'
			'&width=3505&height=1855&format=png')
	url %= wms_name
	return url


def get_hazard_map(region, period=0, poe=0.1, perc=None, verbose=False):
	"""
	Fetch hazard map

	:param region:
		(west, east, south, north) tuple
	:param period:
		float, spectral period (0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 0.75, 1.0, 2.0, 3.0, 4.0)
		(default: 0)
	:param poe:
		float, probability of exceedance in 50 years (0.01, 0.02, 0.05, 0.1, 0.39, 0.5)
		(default: 0.1)
	:param perc:
		float, percentile (None, 5, 15, 50, 85, 95)
		If None, arithmetic mean will be selected
	:param verbose:
		bool, prints url if set to True (default: False)

	:return:
		instance of :class:`rshalib.result.HazardMap`
	"""
	## Convert parameters
	efehr_params = OrderedDict()

	efehr_params['lon1'] = region[0]  ## left
	efehr_params['lon2'] = region[1]  ## right
	efehr_params['lat1'] = region[3]  ## upper
	efehr_params['lat2'] = region[2]  ## lower
	#W, E, S, N = region
	#efehr_params['coordinates'] = str([[W, S], [W, N], [E, N], [E, S]])
	#efehr_params['coordinates'] = efehr_params['coordinates'].replace(' ', '')

	efehr_params['id'] = 68
	hazard_model = "SHARE Mean Hazard Model"

	## What is difference with the above??
	#efehr_params['id'] = 71
	#hazard_model = "SHARE Preferred Mean Hazard Model"

	#efehr_params['id'] = 74
	#hazard_model = "GSHAP"

	if period == 0:
		efehr_params['imt'] = "PGA"
		imt = "PGA"
	else:
		efehr_params['imt'] = "SA[%0.2fs]" % period
		imt = "SA"

	efehr_params['hmapexceedprob'] = poe
	efehr_params['hmapexceedyears'] = timespan = 50

	efehr_params['soiltype'] = soil_type = "rock_vs30_800ms-1"
	#efehr_params['soiltype'] = soil_type = "rock"

	if perc is None:
		efehr_params['aggregationtype'] = "arithmetic"
		efehr_params['aggregationlevel'] = 0.5
	else:
		efehr_params['aggregationtype'] = "ordinal"
		efehr_params['aggregationlevel'] = perc / 100.

	data = query_efehr(efehr_params, verbose=verbose)

	model_name = "%s - %s - %s" % (hazard_model, soil_type, efehr_params['imt'])
	if perc is None:
		model_name += " - Mean"
	else:
		model_name += " - P%02d" % perc
	filespec = ""
	sites = []
	intensities = []
	for i, line in enumerate(data):
		if i >= 1 and line:
			cols = line.split(';')
			try:
				lon, lat, sa = [float(col.strip()) for col in cols]
			except:
				continue
			sites.append(rshalib.site.SHASite(lon, lat))
			intensities.append(sa)

	if len(intensities):
		hm = rshalib.result.HazardMap(model_name, filespec, sites, period, imt,
							intensities, intensity_unit="g", timespan=timespan,
							poe=poe, return_period=None, vs30s=None)
		return hm
	else:
		print("Request did not yield any data!")


def download_hazard_map(out_folder, period=0, poe=0.1, perc=None, format='PNG',
						verbose=False):
	"""
	Download given hazard map.

	:param out_folder:
		str, folder where to save the downloaded map
	:param period:
	:param poe:
	:param perc:
		see :func:`get_hazard_map`
	:param format:
		str, 'PNG' or 'SHP'
	"""
	map_id = 68

	if period == 0:
		imt_string = "PGA"
	else:
		imt_string = "SA[%0.2fs]" % period

	timespan = 50
	pt = rshalib.poisson.PoissonT(timespan)
	return_period = int(round(pt.get_tau(poe)))

	soil_class = "rock_vs30_800ms-1"

	if perc is None:
		agg_type = "arithmetic"
		agg_level = 0.5
		agg_string = 'mean'
	else:
		agg_type = "ordinal"
		agg_level = perc / 100.
		agg_string = 'perc%02d' % perc

	if format == 'SHP':
		(wms_id, wms_name) = get_map_wms_id(map_id, imt_string, poe, timespan,
								soil_class, agg_type, agg_level, verbose=verbose)
		url = 'http://appsrvr.share-eu.org/share/staticdownload/%s.zip'
		url %= wms_name
		format = 'ZIP'

	elif format == 'PNG':
		url = get_map_wms_url(map_id, imt_string, poe, timespan, soil_class,
								agg_type, agg_level, verbose=verbose)

	out_filename = 'SHARE_Tr=%04dyr_rock_T=%s_%s.%s'
	out_filename %= (return_period, imt_string, agg_string, format)
	out_filespec = os.path.join(out_folder, out_filename)

	if not os.path.exists(out_filespec):
		download_data(url, out_filespec)
	else:
		print("Map %s already downloaded!" % out_filename)


def download_all_hazard_maps(out_folder):
	"""
	"""
	import time

	periods = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75, 1., 2., 3., 4.]
	poes = [0.01, 0.02, 0.05, 0.1, 0.39, 0.5]
	percentiles = [None, 5, 15, 50, 85, 95]

	for perc in percentiles:
		for poe in poes:
			for period in periods:
				print("perc=%s, poe=%s, period=%s" % (perc, poe, period))
				download_hazard_map(out_folder, period=period, poe=poe, perc=perc,
									format='SHP')
				time.sleep(5)


def convert_shp_to_geotiff(shp_file):
	"""
	"""
	import mapping.layeredbasemap as lbm

	gis_data = lbm.GisData(shp_file)
	_, _, pg_data = gis_data.get_data()
	pt_data = lbm.MultiPointData([], [])
	for pg in pg_data:
		pt = pg.get_centroid()
		pt_data.append(pt)
	pt_data.values = pg_data.values['HPVALUE']

	dlon = dlat = 0.1
	lons = np.arange(-25.27 + dlon/2, 44.83, dlon)
	lats = np.arange(34.45 + dlat/2, 71.55, dlat)
	mesh_lons, mesh_lats = np.meshgrid(lons, lats)
	mesh_values = np.ones_like(mesh_lons) * np.nan
	lon_idxs = np.round((pt_data.lons - lons[0]) / dlon)
	lat_idxs = np.round((pt_data.lats - lats[0]) / dlat)
	for i in range(len(pt_data)):
		lon_idx, lat_idx = int(lon_idxs[i]), int(lat_idxs[i])
		mesh_values[lat_idx, lon_idx] = pt_data.values[i]

	grid = lbm.MeshGridData(mesh_lons, mesh_lats, mesh_values, unit='g')
	out_file = os.path.splitext(shp_file)[0] + '.TIF'
	grid.export_gdal('GTiff', out_file)


def get_hazard_curve():
	pass


def get_hazard_spectrum(lon, lat, poe=0.1, verbose=False):
	"""
	Get mean rock UHS for poe in 50 years from SHARE Mean Hazard Model
	(id=68)

	:param lon:
		float, longitude
	:param lat:
		float, latitude
	:param poe:
		float, probability of exceedance in 50 years (0.01, 0.02, 0.05, 0.1, 0.39, 0.5)
		(default: 0.1)
	:param verbose:
		bool, prints url if set to True (default: False)

	:return:
		instance of :class:`rshalib.result.UHS`
	"""
	efehr_params = OrderedDict()
	efehr_params['lon'] = lon
	efehr_params['lat'] = lat
	efehr_params['id'] = 68
	efehr_params['imt'] = 'SA'
	efehr_params['poe'] = poe
	efehr_params['timespanpoe'] = 50
	efehr_params['soiltype'] = soil_type = 'rock_vs30_800ms-1'
	efehr_params['aggregationtype'] = 'arithmetic'
	efehr_params['aggregationlevel'] = 0.5

	xml_data = query_efehr(efehr_params, endpoint='spectra', verbose=verbose)
	tree = etree.fromstring(''.join(xml_data))

	for e in tree.iter():
		if e.tag.endswith("spectraPeriodList"):
			periods = np.array(e.text.split(" "), dtype=float)
		if e.tag.endswith("IML"):
			intensities = np.array(e.text.split(" "), dtype=float)
	model_name = "SHARE Mean Hazard Model - %s" % soil_type
	site = rshalib.site.SHASite(lon, lat)
	return rshalib.result.UHS(model_name, "", site, periods, 'SA', intensities,
							intensity_unit="g", timespan=50, poe=poe)



if __name__ == "__main__":
	"""
	lon, lat = 4, 51

	uhs = get_hazard_spectrum(lon, lat, verbose=True)
	uhs.plot()
	#exit()

	verbose = False
	map_models = get_map_models(lon, lat, verbose=verbose)
	print(map_models)
	for map_id, map_name in map_models.items():
		print(map_id, map_name)
		for imt, imt_unit in get_map_imts(map_id, verbose=verbose):
			print(imt, imt_unit)
			for poe, timespan in get_map_poes(map_id, imt, verbose=verbose):
				print(poe, timespan)
				for soil_class in get_map_soil_classes(map_id, imt, poe, timespan, verbose=verbose):
					print(soil_class)
					for agg_type, agg_level in get_map_aggregation_types(map_id, imt, poe, timespan, soil_class, verbose=verbose):
						print(agg_type, agg_level)
						#print(get_map_wms_id(map_id, imt, poe, timespan, soil_class, agg_type, agg_level, verbose=verbose))
						print(get_map_wms_url(map_id, imt, poe, timespan, soil_class, agg_type, agg_level, verbose=verbose))

	"""

	lon, lat = 4, 51
	print(get_map_models(lon, lat, verbose=True))
	print(get_map_imts(68, verbose=True))
	print(get_map_poes(68, 'SA[0.10s]', verbose=True))
	print(get_map_soil_classes(68, 'SA[0.10s]', 0.1, 50, verbose=True))
	print(get_map_aggregation_types(68, 'SA[0.10s]', 0.1, 50, 'rock_vs30_800ms-1', verbose=True))
	print(get_map_wms_id(68, 'SA[0.10s]', 0.1, 50, 'rock_vs30_800ms-1', 'arithmetic', 0.5, verbose=True))

	#out_folder = "C:\\Temp\\efehr"
	out_folder = "D:\\seismo-gis\\collections\\SHARE_ESHM13\\HazardMaps\\SHP"
	download_all_hazard_maps(out_folder)
	#download_hazard_map(out_folder, format='SHP')
	#convert_shp_to_geotiff('C:\\Temp\\efehr\\hmap491.shp')
	exit()

	#region = (2,7,49.25,51.75)
	region = (4,5,50,51)
	period = 0
	poe = 0.1
	hm = get_hazard_map(region, period=period, poe=poe, verbose=True)
	map = hm.get_plot()
	map.plot()
