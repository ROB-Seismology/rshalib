# -*- coding: utf-8 -*-
"""
I/O classes and functions for OpenQuake (tested for version 1.0.0). 
"""


import numpy as np
import os

from lxml import etree

from ..nrml import ns
from ..result import HazardCurveField, HazardMap, UHSField, DeaggregationSlice, ProbabilityMatrix
from ..site import SHASite


NRML = ns.NRML_NS
GML = ns.GML_NS
intensity_unit = {'PGD': 'cm', 'PGV': 'cms', 'PGA': 'g', 'SA': 'g'}


def parse_hazard_curves(xml_filespec):
	"""
	Parse OpenQuake nrml xml hazard curve file (= hazard curve field).
	
	:param xml_filespec:
		String, filespec of file to parse.
	
	:return:
		instance of :class:`HazardCurveField`
	"""
	nrml = etree.parse(xml_filespec)
	sites, poess = [], []
	for e in nrml.iter():
		if e.tag == '{%s}hazardCurves' % NRML:
			model_name = (e.get('sourceModelTreePath') + ' - '
				+ e.get('gsimTreePath'))
			if e.attrib.has_key('saPeriod'):
				period = e.get('saPeriod')
			else:
				period = 0
			IMT = e.get('IMT')
			timespan = float(e.get('investigationTime'))
		if e.tag == '{%s}IMLs' % NRML:
			intensities = np.array(e.text.split(), dtype=float)
		if e.tag == '{%s}hazardCurve' % NRML:
			lon_lat, poes = parse_hazard_curve(e)
			sites.append(SHASite(*lon_lat))
			poess.append(poes)
	hcf = HazardCurveField(model_name, xml_filespec, sites, period, IMT,
		intensities, intensity_unit=intensity_unit[IMT], timespan=timespan,
		poes=np.array(poess))
	return hcf


def parse_hazard_curve(hazard_curve):
	"""
	Subroutine for parse_hazard_curves.
	"""
	for e in hazard_curve.iter():
		if e.tag == '{%s}pos' % GML:
			lon, lat = map(float, str(e.text).split())
		if e.tag == '{%s}poEs' % NRML:
			poes = np.array(e.text.split(), dtype=float)
			poes[np.where(poes==0)] = 10**-15
	return (lon, lat), poes


def parse_hazard_map(xml_filespec):
	"""
	Parse OpenQuake nrml xml hazard map file.
	
	:param xml_filespec:
		String, filespec of file to parse.
	
	:return:
		instance of :class:`HazardMap`
	"""
	nrml = etree.parse(xml_filespec)
	sites, intensities = [], []
	for e in nrml.iter():
		if e.tag == '{%s}hazardMap' % NRML:
			model_name = (e.get('sourceModelTreePath') + ' - '
				+ e.get('gsimTreePath'))
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
	Parse OpenQuake nrml xml uniform hazard spectra file.
	
	:param xml_filespec:
		String, filespec of file to parse.
	
	:return:
		instance of :class:`UHSField`
	"""
	nrml = etree.parse(xml_filespec).getroot()
	uh_spectra = nrml.find('{%s}uniformHazardSpectra' % NRML)
	model_name = (uh_spectra.get('sourceModelTreePath') + ' - '
		+ uh_spectra.get('gsimTreePath'))
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
	Parse OpenQuake nrml xml disaggregation file.
	
	:param xml_filespec:
		String, filespec of file to parse.
	
	:return:
		dict {disaggregatuin type: instance of :class:`DeaggregationSlice`}
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


#def rename_output_files(dir):
#	"""
#	"""
#	for name in os.listdir(dir):
#		filespec = os.path.join(dir, name)
#		if os.path.splitext(name)[-1] == ".xml":
#			if name.startswith('hazard-map'):
#				hm = parse_hazard_map(filespec)
#				new_filespec = os.path.join(dir, 'hazard-map_%.fyr_%s(%s).xml' % (hm.return_period, hm.IMT, hm.period))
#				if new_filespec != filespec:
#					os.rename(filespec, new_filespec)
#			if name.startswith('hazard-curve'):
#				hc = parse_hazard_curve(filespec)
#				new_filespec = os.path.join(dir, 'hazard-curve_%s(%s).xml' % (hc.IMT, hc.period))
#				if new_filespec != filespec:
#					os.rename(filespec, new_filespec)


if __name__ == "__main__":
	"""
	Test files are located in ../test/nrml.
	"""
#	xml_filespec = r'D:\Python\hazard\rshalib\test\nrml\hazard_curve_v1.0.0.xml'
#	hcf = parse_hazard_curves(xml_filespec)
#	hcf.plot([0])

#	xml_filespec = r'D:\Python\hazard\rshalib\test\nrml\hazard_map_v1.0.0.xml'
#	hm = parse_hazard_map(xml_filespec)
#	hm.plot()

#	xml_filespec = r'D:\Python\hazard\rshalib\test\nrml\uh_spectra_v1.0.0.xml'
#	uhs_field = parse_uh_spectra(xml_filespec)
#	uhs_field.plot([0])

	xml_filespec = r'D:\Python\hazard\rshalib\test\nrml\disagg_matrix_v1.0.0.xml'
	dis_slices = parse_disaggregation(xml_filespec)
	dis_slices['Mag,Dist'].plot_mag_dist_pmf()

