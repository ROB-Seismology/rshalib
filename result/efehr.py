"""
Functions to download SHARE results from EFEHR website
"""

import os
import urllib
from lxml import etree
from collections import OrderedDict

import hazard.rshalib as rshalib


def query_efehr(params, verbose=False):
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
	base_url = "http://appsrvr.share-eu.org:8080/share/map"
	url = "%s?%s" % (base_url, urllib.urlencode(params))
	if verbose:
		print url

	f = urllib.urlopen(url)
	data = f.readlines()
	f.close()

	return data


def get_map_models(lon, lat, verbose=False):
	"""
	Provide an index of all hazard models that are defined for a specific point of interest.

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
	Provide all intensity measurement types (IMT) for which a specified model
	provides hazard information.

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
	Provide probability of exceedance and investigation timespan's with maps
	available, given model ID and intensity measurement types (IMT) for which
	a specified model provides hazard information.

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
	Provide site class with maps available, given model ID, IMT, POE, time span
	for which a specified model provides hazard information.

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
	xml = etree.fromstring('<soiltypes>' + xml_data[0] + '</soiltypes>')

	soil_classes = []
	for soil_elem in xml.findall('soiltype'):
		soil_class = soil_elem.find('type').text
		soil_classes.append(soil_class)

	return soil_classes


def get_map_aggregation_types(map_id, imt_string, poe, timespan, soil_class, verbose=False):
	"""
	Provide aggregation types with maps available, given model ID, IMT, POE,
	time span, site class for which a specified model provides hazard information.

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


def get_map_wms_id(map_id, imt_string, poe, timespan, soil_class, agg_type, agg_level, verbose=False):
	"""
	Provide the map identity, and the layer reference for the web map service,
	of the hazard map for given model ID, IMT, POE, time span, site class and
	aggregation type.

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


def get_map_wms_url(map_id, imt_string, poe, timespan, soil_class, agg_type, agg_level, verbose=False):
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
	wms_id, wms_name = get_map_wms_id(map_id, imt_string, poe, timespan, soil_class, agg_type, agg_level, verbose=verbose)

	#url = 'http://efehrmaps.ethz.ch/cgi-bin/mapserv?MAP=/var/www/mapfile/sharehazard.01.map&LAYERS=%s&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap' % wms_name
	url = 'http://gemmsrvr.ethz.ch/cgi-bin/mapserv?MAP=/var/www/mapfile/sharehazard.01.map&LAYERS=%s&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap' % wms_name
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

	efehr_params['id'] = 68
	hazard_model = "SHARE Mean Hazard Model"

	if period == 0:
		efehr_params['imt'] = "PGA"
		imt = "PGA"
	else:
		efehr_params['imt'] = "SA[%0.2d]" % period
		imt = "SA"

	efehr_params['hmapexceedprob'] = poe
	efehr_params['hmapexceedyears'] = timespan = 50

	efehr_params['soiltype'] = soil_type = "rock"

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

	hm = rshalib.result.HazardMap(model_name, filespec, sites, period, imt, intensities, intensity_unit="g", timespan=timespan, poe=poe, return_period=None, vs30s=None)

	return hm


def get_hazard_curve():
	pass


def get_hazard_spectrum():
	pass



if __name__ == "__main__":
	lon, lat = 4, 51
	map_models = get_map_models(lon, lat)
	for map_id, map_name in map_models.items():
		print map_id,map_name
		for imt, imt_unit in get_map_imts(map_id):
			print imt, imt_unit
			for poe, timespan in get_map_poes(map_id, imt):
				print poe, timespan
				for soil_class in get_map_soil_classes(map_id, imt, poe, timespan):
					print soil_class
					for agg_type, agg_level in get_map_aggregation_types(map_id, imt, poe, timespan, soil_class):
						print agg_type, agg_level
						#print get_map_wms_id(map_id, imt, poe, timespan, soil_class, agg_type, agg_level)
						print get_map_wms_url(map_id, imt, poe, timespan, soil_class, agg_type, agg_level)


	"""
	region = (2,7,49.25,51.75)
	#region = (4,5,50,51)
	period = 0
	poe = 0.1
	hm = get_hazard_map(region, period=period, poe=poe, verbose=True)
	map = hm.get_plot()
	map.plot()
	"""
