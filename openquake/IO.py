# -*- coding: utf-8 -*-
"""
I/O classes and functions for OpenQuake (tested for version 1.0.0).
"""


import numpy as np
import os

from lxml import etree

from ..nrml import ns
from ..result import DeaggregationSlice, DeaggregationCurve, SpectralDeaggregationCurve, HazardCurveField, HazardMap, Poisson, ProbabilityArray, ProbabilityMatrix, SpectralHazardCurveField, SpectralHazardCurveFieldTree, UHSField, UHSFieldTree
from ..site import SHASite


NRML = ns.NRML_NS
GML = ns.GML_NS
intensity_unit = {'PGD': 'cm', 'PGV': 'cms', 'PGA': 'g', 'SA': 'g'}


def _get_model_name(e):
	"""
	"""
	model_name = e.get(ns.STATISTICS, None)
	model_name = model_name or e.get(ns.SMLT_PATH) + "_" + e.get(ns.GMPELT_PATH)
	if model_name == "quantile":
		model_name += "_%s" % float(e.get(ns.QUANTILE_VALUE))
	return model_name

def _parse_hazard_curve(hazard_curve, site_names={}):
	"""
	Parse OpenQuake nrml element of type "hazardCurve"
	"""
	lon, lat = map(float, hazard_curve.findtext(".//" + ns.POSITION).split())
	site = SHASite(lon, lat, name=site_names.get((lon, lat), None))
	try:
		poes = np.array(hazard_curve.findtext(".//" + ns.POES).split(), float)
	except:
		poes = np.array(hazard_curve.findtext(".//{%s}poes" % NRML).split(), float)
	return site, poes.clip(1E-15)


def _parse_hazard_curves(hazard_curves, site_names={}):
	"""
	Parse OpenQuake nrml element of type "hazardCurves"
	"""
	model_name = _get_model_name(hazard_curves)
	imt = hazard_curves.get(ns.IMT)
	period = float(hazard_curves.get(ns.PERIOD, 0))
	timespan = float(hazard_curves.get(ns.INVESTIGATION_TIME))
	intensities = map(float, hazard_curves.findtext(ns.IMLS).split())
	sites = []
	poess = []
	for hazard_curve in hazard_curves.findall(ns.HAZARD_CURVE):
		site, poes = _parse_hazard_curve(hazard_curve, site_names)
		sites.append(site)
		poess.append(poes)
	return model_name, sites, period, imt, intensities, timespan, ProbabilityArray(poess)


def parse_hazard_curves(xml_filespec, site_names={}):
	"""
	Parse OpenQuake nrml output file of type "hazard curves"

	:param xml_filespec:
		String, filespec of file to parse.

	:return:
		instance of :class:`..result.HazardCurveField`
	"""
	nrml = etree.parse(xml_filespec)
	model_name, sites, period, imt, intensities, timespan, poess = _parse_hazard_curves(nrml.find(ns.HAZARD_CURVES), site_names)
	hcf = HazardCurveField(model_name, poess, xml_filespec, sites, period, imt, intensities, intensity_unit=intensity_unit[imt], timespan=timespan)
	return hcf


def parse_hazard_curves_multi(xml_filespec, site_names={}):
	"""
	Parse OpenQuake nrml output file of type "hazard curves multi"

	:param xml_filespec:
		String, filespec of file to parse.

	:return:
		instance of :class:`..result.SpectralHazardCurveField` or :class:`..result.SpectralHazardCurveFieldTree`
	"""
	nrml = etree.parse(xml_filespec).getroot()
	branch_names = []
	periods = []
	intensities = []
	poes = []
	for hazard_curves in nrml.findall(ns.HAZARD_CURVES):
		model_name, sites, period, imt, intensities_, timespan, poes_ = _parse_hazard_curves(hazard_curves, site_names)
		assert imt in ("PGA", "SA")
		if period not in periods:
			periods.append(period)
			intensities.append(intensities_)
		if model_name not in branch_names:
			branch_names.append(model_name)
		poes.extend(poes_)
	intensities = np.array(intensities)
	poes_ = ProbabilityArray(np.zeros((len(sites), len(branch_names), len(periods), intensities.shape[1])))
	m = 0
	for j in range(len(branch_names)):
		for k in range(len(periods)):
			for i in range(len(sites)):
				poes_[i, j, k] = poes[m]
				m += 1

	if len(set(branch_names)) == 1:
		filespecs = [xml_filespec] * len(periods)
		result = SpectralHazardCurveField(model_name, poes_[:,0,:,:], filespecs, sites, periods, "SA", intensities, timespan=timespan)
	else:
		filespecs = [xml_filespec] * len(branch_names)
		weights = np.array([1.] * len(branch_names))
		weights /= weights.sum()
		result = SpectralHazardCurveFieldTree(model_name, poes_, branch_names, filespecs, weights, sites, periods, "SA", intensities, timespan=timespan)

	## Order spectral periods
	result.reorder_periods()
	return result


def parse_spectral_hazard_curve_field(xml_filespec, site_names={}, timespan=50):
	"""
	Parse spectralHazardCurveField as written by :meth:`write_nrml` of
	:class:`..result.SpectralHazardCurveField`

	:param xml_filespec:
		String, filespec of file to parse.
	:param site_names:
		Dict, mapping (lon, lat) tuples to site names (default: {})
	:param timespan:
		Float, time span in years (default: 50)

	:return:
		instance of :class:`..result.SpectralHazardCurveField`
	"""
	nrml = etree.parse(xml_filespec).getroot()
	shcf_elem = nrml.find(ns.SPECTRAL_HAZARD_CURVE_FIELD)
	imt = shcf_elem.get(ns.IMT)
	timespan = float(shcf_elem.get(ns.INVESTIGATION_TIME, timespan))
	model_name = shcf_elem.get(ns.NAME)
	smlt_path = shcf_elem.get(ns.SMLT_PATH, '')
	gmpelt_path = shcf_elem.get(ns.GMPELT_PATH, '')

	periods = []
	imls = []
	sites = []
	poes = []

	for hcf_elem in shcf_elem.findall(ns.HAZARD_CURVE_FIELD):
		period_poes = []
		# Note: 'saPeriod'
		period = float(hcf_elem.get(ns.PERIOD, 0.))
		periods.append(period)
		# Note: 'IMLs'
		period_imls = map(float, hcf_elem.findtext(ns.IMLS).split())
		imls.append(period_imls)
		for hc_elem in hcf_elem.findall(ns.HAZARD_CURVE):
			# Note: poes instead of poEs
			site, site_poes = _parse_hazard_curve(hc_elem, site_names=site_names)
			if len(periods) == 1:
				sites.append(site)
			period_poes.append(site_poes)
		poes.append(period_poes)

	periods = np.array(periods)
	imls = np.array(imls)
	poes = ProbabilityArray(poes)
	poes = poes.swapaxes(0, 1)

	filespecs = [xml_filespec] * len(periods)
	shcf = SpectralHazardCurveField(model_name, poes, filespecs, sites, periods, imt, imls, timespan=timespan)

	## Order spectral periods
	shcf.reorder_periods()
	return shcf


def parse_hazard_map(xml_filespec):
	"""
	Parse OpenQuake nrml output file of type "hazard map"

	:param xml_filespec:
		String, filespec of file to parse.

	:return:
		instance of :class:`..result.HazardMap`
	"""
	nrml = etree.parse(xml_filespec)
	sites, intensities = [], []
	for e in nrml.iter():
		if e.tag == ns.HAZARD_MAP:
			model_name = _get_model_name(e)
			IMT = e.get(ns.IMT)
			if e.attrib.has_key(ns.PERIOD):
				period = e.get(ns.PERIOD)
			else:
				period = 0
			timespan = float(e.get(ns.INVESTIGATION_TIME))
			poe = float(e.get(ns.POE))
		if e.tag == ns.NODE:
			lon = float(e.get(ns.LON))
			lat = float(e.get(ns.LAT))
			sites.append(SHASite(lon, lat))
			iml = float(e.get(ns.IML))
			intensities.append(iml)
	hm = HazardMap(model_name, xml_filespec, sites, period, IMT,
		np.array(intensities), intensity_unit=intensity_unit[IMT],
		timespan=timespan, poe=poe)
	return hm


def parse_uh_spectra(xml_filespec, sites=[]):
	"""
	Parse OpenQuake nrml output file of type "uniform hazard spectra"

	:param xml_filespec:
		String, filespec of file to parse.
	:param sites:
		list with instances of :class:`..site.SHASite` (default: {})

	:return:
		instance of :class:`..result.UHSField`
	"""
	nrml = etree.parse(xml_filespec).getroot()
	uh_spectra = nrml.find(ns.UNIFORM_HAZARD_SPECTRA)
	model_name = _get_model_name(uh_spectra)
	periods = uh_spectra.find(ns.PERIODS)
	periods = map(float, str(periods.text).split())
	IMT = 'SA'
	timespan = float(uh_spectra.get(ns.INVESTIGATION_TIME))
	poe = float(uh_spectra.get(ns.POE))
	uh_sites, intensities = [], []
	for uh_spectrum in uh_spectra.findall(ns.UHS):
		pos = uh_spectrum.find(ns.POINT).find(ns.POSITION)
		lon, lat = map(float, pos.text.split())
		uh_sites.append(SHASite(lon, lat))
		imls = uh_spectrum.find(ns.IMLS)
		intensities.append(map(float, imls.text.split()))
	uhs_field = UHSField(model_name, xml_filespec, uh_sites, periods, IMT,
		intensities=np.array(intensities), intensity_unit=intensity_unit[IMT],
		timespan=timespan, poe=poe)
	uhs_field.set_site_names(sites)
	return uhs_field


def parse_disaggregation(xml_filespec, site_name=None):
	"""
	Parse OpenQuake nrml output file of type "disaggregation"

	:param xml_filespec:
		String, filespec of file to parse.
	:param site_name:
		String, name of site (default: None)

	:return:
		dict {disaggregation type: instance of :class:`..result.DeaggregationSlice`}
		Available disaggregation types:
			- 'Mag'
			- 'Dist'
			- 'Lon,Lat'
			- 'TRT'
			- 'Mag,Dist'
			- 'Mag,Dist,Eps'
			- 'Mag,Lon,Lat'
			- 'Lon,Lat,TRT'
	"""
	nrml = etree.parse(xml_filespec).getroot()
	disagg_matrices = nrml.find(ns.DISAGG_MATRICES)
	mag_bin_edges = np.array(disagg_matrices.get(ns.MAG_BIN_EDGES).split(', '),
		dtype=float)
	dist_bin_edges = np.array(disagg_matrices.get(ns.DIST_BIN_EDGES).split(', '),
		dtype=float)
	lon_bin_edges = np.array(disagg_matrices.get(ns.LON_BIN_EDGES).split(', '),
		dtype=float)
	lat_bin_edges = np.array(disagg_matrices.get(ns.LAT_BIN_EDGES).split(', '),
		dtype=float)
	eps_bin_edges = np.array(disagg_matrices.get(ns.EPS_BIN_EDGES).split(', '),
		dtype=float)
	tectonic_region_types = disagg_matrices.get(
		ns.TECTONIC_REGION_TYPES).split(', ')
	lon = float(disagg_matrices.get(ns.LON))
	lat = float(disagg_matrices.get(ns.LAT))
	site = SHASite(lon, lat, name=site_name)
	imt = disagg_matrices.get(ns.IMT)
	if disagg_matrices.attrib.has_key(ns.PERIOD):
		period = float(disagg_matrices.get(ns.PERIOD))
	else:
		period = 0.
	timespan = float(disagg_matrices.get(ns.INVESTIGATION_TIME))
	deaggregation_slices = {}
	for disagg_matrix in disagg_matrices.findall(ns.DISAGG_MATRIX):
		dims = np.array(disagg_matrix.get(ns.DIMS).split(','), dtype=float)
		probs = []
		for prob in disagg_matrix.findall(ns.PROB):
			probs.append(float(prob.get(ns.VALUE)))
		probs = np.reshape(probs, dims)
		type = disagg_matrix.get(ns.TYPE)
		if type == 'Mag':
			probs = probs[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
			bin_edges = (mag_bin_edges, [], [], [], [], [])
		if type == 'Dist':
			probs = probs[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
			bin_edges = ([], dist_bin_edges, [], [], [], [])
		if type == 'TRT':
			probs = probs[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
			bin_edges = ([], [], [], [], [], tectonic_region_types)
		if type == 'Mag,Dist':
			probs = probs[:, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
			bin_edges = (mag_bin_edges, dist_bin_edges, [], [], [], [])
		if type == 'Mag,Dist,Eps':
			probs = probs[:, :, np.newaxis, np.newaxis, :, np.newaxis]
			bin_edges = (mag_bin_edges, dist_bin_edges, [], [], eps_bin_edges, [])
		if type == 'Lon,Lat':
			probs = probs[np.newaxis, np.newaxis, :, :, np.newaxis, np.newaxis]
			bin_edges = ([], [], lon_bin_edges, lat_bin_edges, [], [])
		if type == 'Mag,Lon,Lat':
			probs = probs[:, np.newaxis, :, :, np.newaxis, np.newaxis]
			bin_edges = (mag_bin_edges, [], lon_bin_edges, lat_bin_edges, [], [])
		if type == 'Lon,Lat,TRT':
			probs = probs[np.newaxis, np.newaxis, :, :, np.newaxis, :]
			bin_edges = ([], [], lon_bin_edges, lat_bin_edges, [], tectonic_region_types)
		deagg_matrix = ProbabilityMatrix(probs)
		iml = disagg_matrix.get(ns.IML)
		poe = float(disagg_matrix.get(ns.POE))
		return_period = Poisson(life_time=timespan, prob=poe)
		deaggregation_slices[type] = DeaggregationSlice(bin_edges, deagg_matrix, site, imt, iml, period, return_period, timespan)
	return deaggregation_slices


def parse_disaggregation_full(xml_filespec, site_name=None):
	"""
	Parse OpenQuake nrml output file containing full 6-D disaggregation matrix

	:param xml_filespec:
		String, filespec of file to parse.
	:param site_name:
		String, name of site (default: None)

	:return:
		instance of :class:`..result.DeaggregationSlice`
	"""
	nrml = etree.parse(xml_filespec).getroot()
	disagg_matrix = nrml.find(ns.DISAGG_MATRIX)
	shape = tuple(map(int, disagg_matrix.get(ns.DIMS).split(',')))
	mag_bin_edges = np.array(disagg_matrix.get(ns.MAG_BIN_EDGES).split(', '),
		dtype=float)
	dist_bin_edges = np.array(disagg_matrix.get(ns.DIST_BIN_EDGES).split(', '),
		dtype=float)
	lon_bin_edges = np.array(disagg_matrix.get(ns.LON_BIN_EDGES).split(', '),
		dtype=float)
	lat_bin_edges = np.array(disagg_matrix.get(ns.LAT_BIN_EDGES).split(', '),
		dtype=float)
	eps_bin_edges = np.array(disagg_matrix.get(ns.EPS_BIN_EDGES).split(', '),
		dtype=float)
	tectonic_region_types = disagg_matrix.get(
		ns.TECTONIC_REGION_TYPES).split(', ')
	bin_edges = (mag_bin_edges, dist_bin_edges, lon_bin_edges, lat_bin_edges, eps_bin_edges, tectonic_region_types)
	lon = float(disagg_matrix.get(ns.LON))
	lat = float(disagg_matrix.get(ns.LAT))
	site = SHASite(lon, lat, name=site_name)
	imt = disagg_matrix.get(ns.IMT)
	period = float(disagg_matrix.get(ns.PERIOD, 0.))
	timespan = float(disagg_matrix.get(ns.INVESTIGATION_TIME))
	iml = float(disagg_matrix.get(ns.IML))
	poe = float(disagg_matrix.get(ns.POE))
	return_period = Poisson(life_time=timespan, prob=poe)
	prob_matrix = ProbabilityMatrix(np.zeros(shape))
	for prob in disagg_matrix.findall(ns.PROB):
		index = prob.get(ns.INDEX)
		value = prob.get(ns.VALUE)
		prob_matrix[tuple(map(int, index.split(",")))] = value
	deaggregation_slice = DeaggregationSlice(bin_edges, prob_matrix, site, imt, iml, period, return_period, timespan)
	return deaggregation_slice


def parse_spectral_deaggregation_curve(xml_filespec, site_name=None, ignore_coords=False, dtype='f'):
	"""
	Parse spectral deaggregation curve

	:param xml_filespec:
		String, filespec of file to parse.
	:param site_name:
		String, name of site (default: None)
	:param ignore_coords:
		Bool, whether or not to ignore coordinate bins
		(default: False)
	:param dtype:
		String, precision of deaggregation matrix (default: 'f')

	:return:
		instance of :class:`SpectralDeaggregationCurve`
	"""
	nrml = etree.parse(xml_filespec).getroot()
	sdc_elem = nrml.find(ns.SPECTRAL_DEAGGREGATION_CURVE)
	shape = list(map(int, sdc_elem.get(ns.DIMS).split(',')))
	if ignore_coords:
		shape[-3] = shape[-4] = 1
	mag_bin_edges = np.array(sdc_elem.get(ns.MAG_BIN_EDGES).split(', '),
		dtype=float)
	dist_bin_edges = np.array(sdc_elem.get(ns.DIST_BIN_EDGES).split(', '),
		dtype=float)
	lon_bin_edges = np.array(sdc_elem.get(ns.LON_BIN_EDGES).split(', '),
		dtype=float)
	lat_bin_edges = np.array(sdc_elem.get(ns.LAT_BIN_EDGES).split(', '),
		dtype=float)
	if ignore_coords:
		lon_bin_edges = lon_bin_edges[::len(lon_bin_edges)-1]
		lat_bin_edges = lat_bin_edges[::len(lat_bin_edges)-1]
	eps_bin_edges = np.array(sdc_elem.get(ns.EPS_BIN_EDGES).split(', '),
		dtype=float)
	tectonic_region_types = sdc_elem.get(
		ns.TECTONIC_REGION_TYPES).split(', ')
	bin_edges = (mag_bin_edges, dist_bin_edges, lon_bin_edges, lat_bin_edges, eps_bin_edges, tectonic_region_types)
	lon = float(sdc_elem.get(ns.LON))
	lat = float(sdc_elem.get(ns.LAT))
	site = SHASite(lon, lat, name=site_name)
	timespan = float(sdc_elem.get(ns.INVESTIGATION_TIME))
	dcs = []
	for dc_elem in sdc_elem.findall(ns.DEAGGREGATION_CURVE):
		imt = dc_elem.get(ns.IMT)
		period = float(dc_elem.get(ns.PERIOD, 0.))
		dss = []
		for iml_idx, ds_elem in enumerate(dc_elem.findall(ns.DEAGGREGATION_SLICE)):
			iml = float(ds_elem.get(ns.IML))
			poe = float(ds_elem.get(ns.POE))
			return_period = Poisson(life_time=timespan, prob=poe)
			prob_matrix = ProbabilityMatrix(np.zeros(shape, dtype=dtype))
			for prob in ds_elem.findall(ns.PROB):
				index = prob.get(ns.INDEX)
				value = float(prob.get(ns.VALUE))
				idx = map(int, index.split(','))
				if ignore_coords:
					idx[-3] = idx[-4] = 0
					idx = tuple(idx)
					if prob_matrix[idx] == 0:
						prob_matrix[idx] = value
					else:
						prob_matrix[idx] = 1 - ((1 - prob_matrix[idx]) * (1 - value))
				else:
					idx = tuple(idx)
					prob_matrix[idx] = value
			ds = DeaggregationSlice(bin_edges, prob_matrix, site, imt, iml, period, return_period, timespan)
			dss.append(ds)
		dc = DeaggregationCurve.from_deaggregation_slices(dss)
		dcs.append(dc)
	sdc = SpectralDeaggregationCurve.from_deaggregation_curves(dcs)
	return sdc


def parse_any_output(xml_filespec):
	"""
	Parse OpenQuake nrml output file of any type ("hazard curves", "hazard curves multi", "hazard map", "uniform hazard spectra" or "disaggregation").

	:param xml_filespec:
		String, filespec of file to parse
	"""
	nrml = etree.parse(xml_filespec)
	hazard_curves = nrml.findall(ns.HAZARD_CURVES)
	if len(hazard_curves) == 1:
		return parse_hazard_curves(xml_filespec)
	if len(hazard_curves) >= 2:
		return parse_hazard_curves_multi(xml_filespec)
	if nrml.findall(ns.HAZARD_MAP):
		return parse_hazard_map(xml_filespec)
	if nrml.findall(ns.UNIFORM_HAZARD_SPECTRA):
		return parse_uh_spectra(xml_filespec)
	if nrml.findall(ns.DISAGG_MATRICES):
		return parse_disaggregation(xml_filespec)
	raise "File is not an output of OpenQuake"


def read_multi_folder(directory, sites=[], add_stats=False, model_name=""):
	"""
	Read OpenQuake output folder with 'hazard_curve_multi' file(s)

	:param directory:
		str, path to folder
	:param sites:
		list with instances of :class:`..site.SHASite` (default: {})
	:param add_stats:
		bool, add mean and quantiles if present (default: False)
	:param model_name:
		str, name of logictree

	:returns:
		instance of :class:`..result.SpectralHazardCurveFieldTree`
	"""
	for filename in sorted(os.listdir(directory)):
		if filename[:23] == "hazard_curve_multi-rlz-" and filename[-6:] == "01.xml":
			break
	xml_filespec = os.path.join(directory, filename)
	try:
		shcft = parse_hazard_curves_multi(xml_filespec)
	except:
		shcf_list = []
		for filename in sorted(os.listdir(directory)):
			if filename[:23] == "hazard_curve_multi-rlz-":
				xml_filespec = os.path.join(directory, filename)
				shcf = parse_spectral_hazard_curve_field(xml_filespec)
				shcf_list.append(shcf)
		print("Read %d spectral hazard curves" % len(shcf_list))
		shcft = SpectralHazardCurveFieldTree.from_branches(shcf_list, "")

	if add_stats:
		mean_xml_filespec = os.path.join(directory, "hazard_curve_multi-mean.xml")
		assert os.path.exists(mean_xml_filespec)
		mean_shcf = parse_hazard_curves_multi(mean_xml_filespec)
		perc_shcf_list, perc_levels = [], []
		for multi_filename in sorted(os.listdir(directory)):
			if "quantile" in multi_filename:
				xml_filespec = os.path.join(directory, multi_filename)
				perc_shcf_list.append(parse_hazard_curves_multi(xml_filespec))
				perc = os.path.splitext(multi_filename)[0].split("quantile_")[1]
				perc_levels.append(int(float(perc) * 100))
		shcft.set_mean(mean_shcf)
		shcft.set_percentiles(perc_shcf_list, perc_levels)

	shcft.model_name = model_name
	shcft.set_site_names(sites)
	return shcft


def read_curve_folder(directory, sites=[], add_stats=False, verbose=True):
	"""
	Read OpenQuake output folder with subfolder for each imt with 'hazard_curve' file(s)

	:param directory:
		str, path to folder
	:param sites:
		list with instances of :class:`..site.SHASite` (default: {})
	:param add_stats:
		bool, add mean and quantiles if present (default: False)
	:param verbose:
		bool, print information (default: True)

	:returns:
		instance of :class:`..result.SpectralHazardCurveFieldTree`
	"""
	imt_subfolders = sorted(os.listdir(directory))
	hc_filenames = sorted(os.listdir(os.path.join(directory, imt_subfolders[0])))
	hc_rlz_filenames = []
	hc_quantile_filenames = []
	for hc_filename in hc_filenames:
		if "rlz" in hc_filename:
			hc_rlz_filenames.append(hc_filename)
		if "quantile" in hc_filename:
			hc_quantile_filenames.append(hc_filename)
	shcf_list = []
	for hc_rlz_filename in hc_rlz_filenames:
		if verbose:
			print "reading %s" % hc_rlz_filename
		hcf_list = []
		for imt_subfolder in imt_subfolders:
			xml_filespec = os.path.join(directory, imt_subfolder, hc_rlz_filename)
			hcf_list.append(parse_hazard_curves(xml_filespec))
		shcf_list.append(SpectralHazardCurveField.from_hazard_curve_fields(hcf_list, hcf_list[0].model_name))
	if add_stats:
		assert os.path.exists(os.path.join(directory, imt_subfolders[0], "hazard_curve-mean.xml"))
		mean_hcf_list = []
		for imt_subfolder in imt_subfolders:
			xml_filespec = os.path.join(directory, imt_subfolder, "hazard_curve-mean.xml")
			mean_hcf_list.append(parse_hazard_curves(xml_filespec))
		mean_shcf = SpectralHazardCurveField.from_hazard_curve_fields(mean_hcf_list, mean_hcf_list[0].model_name)
		perc_shcf_list, perc_levels = [], []
		for hc_quantile_filename in hc_quantile_filenames:
			perc_hcf_list = []
			for imt_subfolder in imt_subfolders:
				xml_filespec = os.path.join(directory, imt_subfolder, hc_quantile_filename)
				perc_hcf_list.append(parse_hazard_curves(xml_filespec))
			perc_shcf_list.append(SpectralHazardCurveField.from_hazard_curve_fields(perc_hcf_list, perc_hcf_list[0].model_name))
			perc = os.path.splitext(hc_quantile_filename)[0].split("quantile_")[1]
			perc_levels.append(int(float(perc) * 100))
	else:
		mean_shcf, perc_levels, perc_shcf_list = None, None, None
	shcft = SpectralHazardCurveFieldTree.from_branches(shcf_list, shcf_list[0].model_name, mean=mean_shcf, percentile_levels=perc_levels, percentiles=perc_shcf_list)
	shcft.set_site_names(sites)
	return shcft


def read_shcft(directory, sites=[], add_stats=False):
	"""
	Read OpenQuake output folder with 'hazard_curve_multi' and/or 'hazard_curve' subfolders.
	Read from the folder 'hazard_curve_multi' if present, else read individual hazard curves from the folder 'hazard_curve'.

	:param directory:
		str, path to folder
	:param sites:
		list with instances of :class:`..site.SHASite` (default: {})
	:param add_stats:
		bool, add mean and quantiles if present (default: False)

	:returns:
		instance of :class:`..result.SpectralHazardCurveFieldTree`
	"""
	multi_folder = os.path.join(directory, "hazard_curve_multi")
	print multi_folder
	if os.path.exists(multi_folder):
		shcft = read_multi_folder(multi_folder, sites, add_stats)
	else:
		curve_folder = os.path.join(directory, "hazard_curve")
		shcft = read_curve_folder(curve_folder, sites, add_stats)
	return shcft


def read_uhsft(directory, return_period, sites=[], add_stats=False):
	"""
	Read OpenQuake output folder with 'uh_spectra' file(s) or 'uh_spectra' subfolder with such files

	:param directory:
		str, path to folder
	:param return period:
		float, return period
	:param sites:
		list with instances of :class:`..site.SHASite` (default: {})
	:param add_stats:
		bool, add mean and quantiles if present (default: False)

	:returns:
		instance of :class:`..result.UHSFieldTree`
	"""
	uhs_folder = os.path.join(directory, "uh_spectra")
	if os.path.exists(uhs_folder):
		directory = uhs_folder
	poe = str(round(Poisson(life_time=50, return_period=return_period), 13))
	uhs_filenames = sorted(os.listdir(directory))
	uhs_rlz_filenames, uhs_quantile_filenames = [], []
	for uhs_filename in uhs_filenames:
		if poe in uhs_filename:
			if "mean" in uhs_filename:
				uhs_mean_filename = uhs_filename
			elif "quantile" in uhs_filename:
				uhs_quantile_filenames.append(uhs_filename)
			else:
				uhs_rlz_filenames.append(uhs_filename)
	uhsf_list = []
	for uhs_rlz_filename in uhs_rlz_filenames:
		xml_filespec = os.path.join(directory, uhs_rlz_filename)
		uhsf_list.append(parse_uh_spectra(xml_filespec))
	if add_stats:
		mean = parse_uh_spectra(os.path.join(directory, uhs_mean_filename))
		perc_list, perc_levels = [], []
		for uhs_quantile_filename in uhs_quantile_filenames:
			xml_filespec = os.path.join(directory, uhs_quantile_filename)
			perc_list.append(parse_uh_spectra(xml_filespec))
			perc = os.path.splitext(uhs_quantile_filename)[0].split("quantile_")[1]
			perc_levels.append(int(float(perc) * 100))
	else:
		mean, perc_list, perc_levels = None, None, None
	uhsft = UHSFieldTree.from_branches(uhsf_list, uhsf_list[0].model_name, mean=mean, percentile_levels=perc_levels, percentiles=perc_list)
	uhsft.set_site_names(sites)
	return uhsft


def write_disaggregation_slice(site, imt, period, iml, poe, timespan, bin_edges, matrix, nrml_filespec, sourceModelTreePath=None, gsimTreePath=None):
	"""
	Write disaggregation slice to nrml file.

	:param site:
		tuple or instance of :class:`rshalib.site.SHASite`
	:param imt:
		str, intensity measure type
	:param period:
		float, period of imt
	:param iml:
		float, intensity measure level
	:param poe:
		float, probability of exceedance
	:param timespan:
		float, timespan
	:param bin_edges:
		list of tuples, defining edges for magnitudes, distances, lons, lats,
		epsilons and tectonic region types
	:param matrix:
		6d array, probability of exceedances
	:nrml_filespec:
		str, filespec of nrml file to write to
	"""
	with open(nrml_filespec, "w") as nrml_file:
		root = etree.Element("nrml", nsmap=ns.NSMAP)
		diss = etree.SubElement(root, "disaggMatrix")
		if sourceModelTreePath:
			diss.set(ns.SMLT_PATH, sourceModelTreePath)
		if gsimTreePath:
			diss.set(ns.GMPELT_PATH, gsimTreePath)
		lon, lat = site[0], site[1]
		diss.set(ns.LON, str(lon))
		diss.set(ns.LAT, str(lat))
		diss.set(ns.IMT, str(iml))
		diss.set(ns.PERIOD, str(period))
		diss.set(ns.IML, str(iml))
		diss.set(ns.POE, str(poe))
		diss.set(ns.INVESTIGATION_TIME, str(timespan))
		mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, trts = bin_edges
		diss.set(ns.MAG_BIN_EDGES, ", ".join(map(str, mag_bins)))
		diss.set(ns.DIST_BIN_EDGES, ", ".join(map(str, dist_bins)))
		diss.set(ns.LON_BIN_EDGES, ", ".join(map(str, lon_bins)))
		diss.set(ns.LAT_BIN_EDGES, ", ".join(map(str, lat_bins)))
		diss.set(ns.EPS_BIN_EDGES, ", ".join(map(str, eps_bins)))
		diss.set(ns.TECTONIC_REGION_TYPES, ", ".join(trts))
		dims = ",".join(map(str, matrix.shape))
		diss.set(ns.DIMS, dims)
		for index, value in np.ndenumerate(matrix):
			if not np.allclose(value, 0.):
				index = ",".join(map(str, index))
				value = str(value)
				prob = etree.SubElement(diss, "prob")
				prob.set(ns.INDEX, index)
				prob.set(ns.VALUE, value)
		nrml_file.write(etree.tostring(root, pretty_print=True, xml_declaration=True, encoding="UTF-8"))

