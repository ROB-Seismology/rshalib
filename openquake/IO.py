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
	model_name = e.get("statistics", None)
	model_name = model_name or e.get("sourceModelTreePath") + "_" + e.get("gsimTreePath")
	if model_name == "quantile":
		model_name += "_%s" % float(e.get("quantileValue"))
	return model_name

def _parse_hazard_curve(hazard_curve, site_names={}):
	"""
	Parse OpenQuake nrml element of type "hazardCurve"
	"""
	lon, lat = map(float, hazard_curve.findtext(".//{%s}pos" % GML).split())
	site = SHASite(lon, lat, name=site_names.get((lon, lat), None))
	poes = np.array(hazard_curve.findtext(".//{%s}poEs" % NRML).split(), float)
	return site, poes.clip(1E-15)


def _parse_hazard_curves(hazard_curves, site_names={}):
	"""
	Parse OpenQuake nrml element of type "hazardCurves"
	"""
	model_name = _get_model_name(hazard_curves)
	imt = hazard_curves.get("IMT")
	period = float(hazard_curves.get("saPeriod", 0))
	timespan = float(hazard_curves.get("investigationTime"))
	intensities = map(float, hazard_curves.findtext("{%s}IMLs" % NRML).split())
	sites = []
	poess = []
	for hazard_curve in hazard_curves.findall("{%s}hazardCurve" % NRML):
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
	model_name, sites, period, imt, intensities, timespan, poess = _parse_hazard_curves(nrml.find("{%s}hazardCurves" % NRML), site_names)
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
	for hazard_curves in nrml.findall("{%s}hazardCurves" % NRML):
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
		if e.tag == '{%s}hazardMap' % NRML:
			model_name = _get_model_name(e)
			IMT = e.get('IMT')
			if e.attrib.has_key('saPeriod'):
				period = e.get('saPeriod')
			else:
				period = 0
			timespan = float(e.get('investigationTime'))
			poe = float(e.get('poE'))
		if e.tag == '{%s}node' % NRML:
			lon = float(e.get('lon'))
			lat = float(e.get('lat'))
			sites.append(SHASite(lon, lat))
			iml = float(e.get('iml'))
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
	uh_spectra = nrml.find('{%s}uniformHazardSpectra' % NRML)
	model_name = _get_model_name(uh_spectra)
	periods = uh_spectra.find('{%s}periods' % NRML)
	periods = map(float, str(periods.text).split())
	IMT = 'SA'
	timespan = float(uh_spectra.get('investigationTime'))
	poe = float(uh_spectra.get('poE'))
	uh_sites, intensities = [], []
	for uh_spectrum in uh_spectra.findall('{%s}uhs' % NRML):
		pos = uh_spectrum.find('{%s}Point' % GML).find('{%s}pos' % GML)
		lon, lat = map(float, pos.text.split())
		uh_sites.append(SHASite(lon, lat))
		imls = uh_spectrum.find('{%s}IMLs' % NRML)
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
	disagg_matrices = nrml.find('{%s}disaggMatrices' % NRML)
	mag_bin_edges = np.array(disagg_matrices.get('magBinEdges').split(', '),
		dtype=float)
	dist_bin_edges = np.array(disagg_matrices.get('distBinEdges').split(', '),
		dtype=float)
	lon_bin_edges = np.array(disagg_matrices.get('lonBinEdges').split(', '),
		dtype=float)
	lat_bin_edges = np.array(disagg_matrices.get('latBinEdges').split(', '),
		dtype=float)
	eps_bin_edges = np.array(disagg_matrices.get('epsBinEdges').split(', '),
		dtype=float)
	tectonic_region_types = disagg_matrices.get(
		'tectonicRegionTypes').split(', ')
	lon = float(disagg_matrices.get('lon'))
	lat = float(disagg_matrices.get('lat'))
	site = SHASite(lon, lat, name=site_name)
	imt = disagg_matrices.get('IMT')
	if disagg_matrices.attrib.has_key('saPeriod'):
		period = float(disagg_matrices.get('saPeriod'))
	else:
		period = 0.
	timespan = float(disagg_matrices.get('investigationTime'))
	deaggregation_slices = {}
	for disagg_matrix in disagg_matrices.findall('{%s}disaggMatrix' % NRML):
		dims = np.array(disagg_matrix.get('dims').split(','), dtype=float)
		probs = []
		for prob in disagg_matrix.findall('{%s}prob' % NRML):
			probs.append(float(prob.get('value')))
		probs = np.reshape(probs, dims)
		type = disagg_matrix.get('type')
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
		iml = disagg_matrix.get('iml')
		deaggregation_slices[type] = DeaggregationSlice(bin_edges, deagg_matrix, site, imt, iml, period, timespan)
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
	disagg_matrix = nrml.find('{%s}disaggMatrix' % NRML)
	shape = tuple(map(int, disagg_matrix.get('dims').split(',')))
	mag_bin_edges = np.array(disagg_matrix.get('magBinEdges').split(', '),
		dtype=float)
	dist_bin_edges = np.array(disagg_matrix.get('distBinEdges').split(', '),
		dtype=float)
	lon_bin_edges = np.array(disagg_matrix.get('lonBinEdges').split(', '),
		dtype=float)
	lat_bin_edges = np.array(disagg_matrix.get('latBinEdges').split(', '),
		dtype=float)
	eps_bin_edges = np.array(disagg_matrix.get('epsBinEdges').split(', '),
		dtype=float)
	tectonic_region_types = disagg_matrix.get(
		'tectonicRegionTypes').split(', ')
	bin_edges = (mag_bin_edges, dist_bin_edges, lon_bin_edges, lat_bin_edges, eps_bin_edges, tectonic_region_types)
	lon = float(disagg_matrix.get('lon'))
	lat = float(disagg_matrix.get('lat'))
	site = SHASite(lon, lat, name=site_name)
	imt = disagg_matrix.get('IMT')
	period = float(disagg_matrix.get('saPeriod', 0.))
	timespan = float(disagg_matrix.get('investigationTime'))
	iml = disagg_matrix.get('iml')
	prob_matrix = ProbabilityMatrix(np.zeros(shape))
	for prob in disagg_matrix.findall('{%s}prob' % NRML):
		index = prob.get("index")
		value = prob.get("value")
		prob_matrix[tuple(map(int, index.split(",")))] = value
	deaggregation_slice = DeaggregationSlice(bin_edges, prob_matrix, site, imt, iml, period, timespan)
	return deaggregation_slice


def parse_spectral_deaggregation_curve(xml_filespec, site_name=None):
	"""
	"""
	nrml = etree.parse(xml_filespec).getroot()
	sdc_elem = nrml.find('{%s}spectralDeaggregationCurve' % NRML)
	shape = tuple(map(int, sdc_elem .get('dims').split(',')))
	mag_bin_edges = np.array(sdc_elem.get('magBinEdges').split(', '),
		dtype=float)
	dist_bin_edges = np.array(sdc_elem.get('distBinEdges').split(', '),
		dtype=float)
	lon_bin_edges = np.array(sdc_elem.get('lonBinEdges').split(', '),
		dtype=float)
	lat_bin_edges = np.array(sdc_elem.get('latBinEdges').split(', '),
		dtype=float)
	eps_bin_edges = np.array(sdc_elem.get('epsBinEdges').split(', '),
		dtype=float)
	tectonic_region_types = sdc_elem.get(
		'tectonicRegionTypes').split(', ')
	bin_edges = (mag_bin_edges, dist_bin_edges, lon_bin_edges, lat_bin_edges, eps_bin_edges, tectonic_region_types)
	lon = float(sdc_elem.get('lon'))
	lat = float(sdc_elem.get('lat'))
	site = SHASite(lon, lat, name=site_name)
	timespan = float(sdc_elem.get('investigationTime'))
	dcs = []
	for dc_elem in sdc_elem.findall('{%s}deaggregationCurve' % NRML):
		imt = dc_elem.get('imt')
		period = float(dc_elem.get('saPeriod', 0.))
		dss = []
		for ds_elem in dc_elem.findall('{%s}deaggregationSlice' % NRML):
			iml = float(ds_elem.get('iml'))
			matrix = ProbabilityMatrix(np.zeros(shape))
			for prob in ds_elem.findall('{%s}prob' % NRML):
				index = prob.get('index')
				value = prob.get('value')
				prob_matrix[tuple(map(int, index.split(',')))] = value
			ds = DeaggregationSlice(bin_edges, matrix, site, imt, iml, period, timespan)
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
	hazard_curves = nrml.findall("{%s}hazardCurves" % NRML)
	if len(hazard_curves) == 1:
		return parse_hazard_curves(xml_filespec)
	if len(hazard_curves) >= 2:
		return parse_hazard_curves_multi(xml_filespec)
	if nrml.findall("{%s}hazardMap" % NRML):
		return parse_hazard_map(xml_filespec)
	if nrml.findall("{%s}uniformHazardSpectra" % NRML):
		return parse_uh_spectra(xml_filespec)
	if nrml.findall("{%s}disaggMatrices" % NRML):
		return parse_disaggregation(xml_filespec)
	raise "File is not an output of OpenQuake"


def read_multi_folder(directory, sites=[], add_stats=False):
	"""
	Read OpenQuake output folder with 'hazard_curve_multi' file(s)

	:param directory:
		str, path to folder
	:param sites:
		list with instances of :class:`..site.SHASite` (default: {})
	:param add_stats:
		bool, add mean and quantiles if present (default: False)

	:returns:
		instance of :class:`..result.SpectralHazardCurveFieldTree`
	"""
	for filename in sorted(os.listdir(directory)):
		if filename[:23] == "hazard_curve_multi-rlz-" and filename[-6:] == "01.xml":
			break
	xml_filespec = os.path.join(directory, filename)
	shcft = parse_hazard_curves_multi(xml_filespec)
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
		shcf_list = [shcf for shcf in shcft]
		shcft = SpectralHazardCurveFieldTree.from_branches(shcf_list, shcft.model_name, mean=mean_shcf, percentile_levels=perc_levels, percentiles=perc_shcf_list)
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
			diss.set("sourceModelTreePath", sourceModelTreePath)
		if gsimTreePath:
			diss.set("gsimTreePath", gsimTreePath)
		lon, lat = site[0], site[1]
		diss.set("lon", str(lon))
		diss.set("lat", str(lat))
		diss.set("imt", str(iml))
		diss.set("saPeriod", str(period))
		diss.set("iml", str(iml))
		diss.set("poE", str(poe))
		diss.set("investigationTime", str(timespan))
		mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, trts = bin_edges
		diss.set("magBinEdges", ", ".join(map(str, mag_bins)))
		diss.set("distBinEdges", ", ".join(map(str, dist_bins)))
		diss.set("lonBinEdges", ", ".join(map(str, lon_bins)))
		diss.set("latBinEdges", ", ".join(map(str, lat_bins)))
		diss.set("epsBinEdges", ", ".join(map(str, eps_bins)))
		diss.set("tectonicRegionTypes", ", ".join(trts))
		dims = ",".join(map(str, matrix.shape))
		diss.set("dims", dims)
		for index, value in np.ndenumerate(matrix):
			if not np.allclose(value, 0.):
				index = ",".join(map(str, index))
				value = str(value)
				prob = etree.SubElement(diss, "prob")
				prob.set("index", index)
				prob.set("value", value)
		nrml_file.write(etree.tostring(root, pretty_print=True, xml_declaration=True, encoding="UTF-8"))

