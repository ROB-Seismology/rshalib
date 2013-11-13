# -*- coding: utf-8 -*-
"""
I/O classes and functions for OpenQuake (tested for version 1.0.0).
"""


import numpy as np
import os

from lxml import etree

#from ..nrml import ns
#from ..result import DeaggregationSlice, HazardCurveField, HazardMap, ProbabilityMatrix, SpectralHazardCurveField, SpectralHazardCurveFieldTree, UHSField
#from ..site import SHASite

from hazard.rshalib.nrml import ns
from hazard.rshalib.result import DeaggregationSlice, HazardCurveField, HazardMap, ProbabilityMatrix, SpectralHazardCurveField, SpectralHazardCurveFieldTree, UHSField
from hazard.rshalib.site import SHASite

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

def _parse_hazard_curve(hazard_curve):
	"""
	Parse OpenQuake nrml element of type "hazardCurve"
	"""
	site = tuple(map(float, hazard_curve.findtext(".//{%s}pos" % GML).split()))
	poes = np.array(hazard_curve.findtext(".//{%s}poEs" % NRML).split(), float)
	return site, poes.clip(10**-15)


def _parse_hazard_curves(hazard_curves):
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
		site, poes = _parse_hazard_curve(hazard_curve)
		sites.append(site)
		poess.append(poes)
	return model_name, sites, period, imt, intensities, timespan, np.array(poess)


def parse_hazard_curves(xml_filespec):
	"""
	Parse OpenQuake nrml output file of type "hazard curves"
	
	:param xml_filespec:
		String, filespec of file to parse.

	:return:
		instance of :class:`..result.HazardCurveField`
	"""
	nrml = etree.parse(xml_filespec)
	model_name, sites, period, imt, intensities, timespan, poess = _parse_hazard_curves(nrml.find("{%s}hazardCurves" % NRML))
	hcf = HazardCurveField(model_name, xml_filespec, sites, period, imt, intensities, intensity_unit=intensity_unit[imt], timespan=timespan, poes=poess)
	return hcf


def parse_hazard_curves_multi(xml_filespec):
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
		model_name, sites, period, imt, intensities_, timespan, poes_ = _parse_hazard_curves(hazard_curves)
		assert imt in ("PGA", "SA")
		if period not in periods:
			periods.append(period)
			intensities.append(intensities_)
		if model_name not in branch_names:
			branch_names.append(model_name)
		poes.extend(poes_)
	intensities = np.array(intensities)
	poes_ = np.zeros((len(sites), len(branch_names), len(periods), intensities.shape[1]))
	m = 0
	for j in range(len(branch_names)):
		for k in range(len(periods)):
			for i in range(len(sites)):
				poes_[i, j, k] = poes[m]
				m += 1
	print poes_[1, 1, 1]
	filespecs = [xml_filespec] * len(branch_names)
	if len(set(branch_names)) == 1:
		return SpectralHazardCurveField(model_name, filespecs, sites, periods, "SA", intensities, timespan=timespan, poes=poes_)
	else:
		weights = np.array([1.] * len(branch_names))
		weights /= weights.sum()
		return SpectralHazardCurveFieldTree(model_name, branch_names, filespecs, weights, sites, periods, "SA", intensities, timespan=timespan, poes=poes_)


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


def parse_uh_spectra(xml_filespec):
	"""
	Parse OpenQuake nrml output file of type "uniform hazard spectra"

	:param xml_filespec:
		String, filespec of file to parse.

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
	sites, intensities = [], []
	for uh_spectrum in uh_spectra.findall('{%s}uhs' % NRML):
		pos = uh_spectrum.find('{%s}Point' % GML).find('{%s}pos' % GML)
		lon, lat = map(float, pos.text.split())
		sites.append(SHASite(lon, lat))
		imls = uh_spectrum.find('{%s}IMLs' % NRML)
		intensities.append(map(float, imls.text.split()))
	uhs_field = UHSField(model_name, xml_filespec, sites, periods, IMT,
		intensities=np.array(intensities), intensity_unit=intensity_unit[IMT],
		timespan=timespan, poe=poe)
	return uhs_field


def parse_disaggregation(xml_filespec):
	"""
	Parse OpenQuake nrml output file of type "disaggregation"

	:param xml_filespec:
		String, filespec of file to parse.

	:return:
		dict {disaggregation type: instance of :class:`..result.DeaggregationSlice`}
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
	site = SHASite(lon, lat)
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


#def output_parser(directory):
#	"""
#	"""
#	output = []
#	for file in os.listdir(directory):
#		if file.startswith("hazard_map"):
#			output.append(parse_hazard_map(os.path.join(directory, file)))
#	return output


if __name__ == "__main__":
	"""
	Test files are located in ../test/nrml.
	"""
#	xml_filespec = r"D:\Python\hazard\rshalib\test\nrml\hazard_curve_v1.0.0.xml"
#	xml_filespec = r"D:\Python\hazard\rshalib\test\nrml\hazard_curve\PGA\hazard_curve_0.xml"
#	xml_filespec = r"D:\Python\hazard\rshalib\test\nrml\hazard_curve\PGA\hazard_curve-mean.xml"
#	xml_filespec = r"D:\Python\hazard\rshalib\test\nrml\hazard_curve\PGA\hazard_curve-quantile_0.05.xml"
#	hcf = parse_any_output(xml_filespec)
#	hcf.plot([0], title=hcf.model_name)

	xml_filespec = r"D:\Python\hazard\rshalib\test\nrml\hazard_curve_multi\hazard_curve_multi_0.xml"
	shcf = parse_any_output(xml_filespec)
	shcf.plot()

#	xml_filespec = r"D:\Python\hazard\rshalib\test\nrml\hazard_map_v1.0.0.xml"
#	hm = parse_any_output(xml_filespec)
#	hm.plot(title=hm.model_name)

#	xml_filespec = r"D:\Python\hazard\rshalib\test\nrml\uh_spectra_v1.0.0.xml"
#	uhs_field = parse_any_output(xml_filespec)
#	uhs_field.plot([0], title=uhs_field.model_name)

#	xml_filespec = r"D:\Python\hazard\rshalib\test\nrml\disagg_matrix_v1.0.0.xml"
#	dis_slices = parse_any_output(xml_filespec)
#	dis_slices['Mag,Dist'].plot_mag_dist_pmf()

