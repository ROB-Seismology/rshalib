"""
Functions to download SHARE results from EFEHR website
"""

import os
import urllib
from collections import OrderedDict

import hazard.rshalib as rshalib



def get_hazard_map(region, period=0, poe=0.1, perc=None, verbose=True):
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
		bool, prints url if set to True (default: True)

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

	## Construct URL
	base_url = "http://appsrvr.share-eu.org:8080/share/map"
	url = "%s?%s" % (base_url, urllib.urlencode(efehr_params))
	print url

	f = urllib.urlopen(url)
	data = f.readlines()
	f.close()


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
	region = (2,7,49.25,51.75)
	#region = (4,5,50,51)
	period = 0
	poe = 0.1
	hm = get_hazard_map(region, period=period, poe=poe, verbose=False)
	map = hm.get_plot()
	map.plot()
