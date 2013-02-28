# -*- coding: utf-8 -*-
"""
I/O classes and functions for openquake
"""

### imports
import numpy
import os
import h5py
from lxml import etree

from ..nrml import ns
from ..nrml.common import *
from ..result import *


### constants
NRML = ns.NRML_NS
GML = ns.GML_NS


class NrmlParser():
	"""
	class to parse nrml file
	"""
	# TODO: check if same names are used as in hazardresult.py

	def __init__(self, xml_filespec):
		"""
		initiator to create xml object from xml file
		arguments:
			xml_filespec
		initiates:
			nrmlObj
		"""
		self.xml_filespec = xml_filespec
		self.nrmlObj = etree.parse(self.xml_filespec)

	def hazardCurveField(self):
		"""
		method to parse hazard curve object from xml object
		returns:
			HazardCurve object
		"""
		## test if nrmlObj is hazard curve field
		assert self.nrmlObj.find("//{%s}hazardCurveField" % NRML) is not None, \
			"nrmlObj is not a hazard curve field"
		## get model
		if self.nrmlObj.find("//*[@statistics]") != None:
			model_nrmlObj = self.nrmlObj.find("//*[@statistics]")
			model = model_nrmlObj.attrib["statistics"]
		else:
			model_nrmlObj = self.nrmlObj.find("//*[@endBranchLabel]")
			model = model_nrmlObj.attrib["endBranchLabel"]
		## get sites
		sites = []
		for site_nrmlObj in self.nrmlObj.findall("//{%s}pos" % GML):
			site_string = site_nrmlObj.text
			site_stringList = site_string.split(" ")
			site = map(float, site_stringList)
			sites.append(tuple(site))
		## get period
		period_nrmlObj = self.nrmlObj.find("//*[@saPeriod]")
		period_string = period_nrmlObj.attrib["saPeriod"]
		period = float(period_string)
		## get iMT
		iMT_nrmlObj = self.nrmlObj.find("//*[@IMT]")
		iMT = iMT_nrmlObj.attrib["IMT"]
		## get intensities
		intensities_nrmlObj = self.nrmlObj.find("//{%s}IML" % NRML)
		intensities_string = intensities_nrmlObj.text
		intensities_stringList = intensities_string.split(" ")
		intensities = map(float, intensities_stringList)
		intensities = numpy.array(intensities)
		## get timespan
		timespan_nrmlObj = self.nrmlObj.find("//*[@investigationTimeSpan]")
		timespan_string = timespan_nrmlObj.attrib["investigationTimeSpan"]
		timespan = float(timespan_string)
		## get poes
		poes = []
		for site_poes_nrmlObj in self.nrmlObj.findall("//{%s}poE" % NRML):
			site_poes_string = site_poes_nrmlObj.text
			site_poes_stringList = site_poes_string.split(" ")
			site_poes = map(float, site_poes_stringList)
			poes.append(site_poes)
		poes = numpy.array(poes)
		## set zero values to very small value (10^-15)
		poes[numpy.where(poes==0)] = 10**-15
		## return HazardCurve Object
		hazardCurveField = HazardCurveField(model, [self.xml_filespec],
			sites, period, iMT, intensities, timespan=timespan, poes=poes)
		return hazardCurveField

	def hazardMap(self):
		"""
		method to parse hazard map object from xml object
		returns:
			HazardMap object
		"""
		## test if nrmlObj is hazard map
		assert self.nrmlObj.find("//{%s}hazardMap" % NRML) is not None, \
			"nrmlObj is not a hazard map"
		## get model
		if self.nrmlObj.find("//*[@statistics]") != None:
			model_nrmlObj = self.nrmlObj.find("//*[@statistics]")
			model = model_nrmlObj.attrib["statistics"]
		else:
			model_nrmlObj = self.nrmlObj.find("//*[@endBranchLabel]")
			model = model_nrmlObj.attrib["endBranchLabel"]
		## get sites
		sites = []
		for site_nrmlObj in self.nrmlObj.findall("//{%s}pos" % GML):
			site_string = site_nrmlObj.text
			site_stringList = site_string.split(" ")
			site = map(float, site_stringList)
			sites.append(tuple(site))
		## get periods
		# TODO: add automatic periods attribute (when added to openquake output)
		period = 0.
		## get iMT
		iMT_nrmlObj = self.nrmlObj.find("//*[@IMT]")
		iMT = iMT_nrmlObj.attrib["IMT"]
		## get intensities
		intensities = []
		for intensity_nrmlObj in self.nrmlObj.findall("//{%s}IML" % NRML):
			intensity_string = intensity_nrmlObj.text
			intensity = float(intensity_string)
			intensities.append(intensity)
		intensities = numpy.array(intensities)
		## get timespan
		timespan_nrmlObj = self.nrmlObj.find("//*[@investigationTimeSpan]")
		timespan_string = timespan_nrmlObj.attrib["investigationTimeSpan"]
		timespan = float(timespan_string)
		## get poe
		poe_nrmlObj = self.nrmlObj.find("//*[@poE]")
		poe_string = poe_nrmlObj.attrib["poE"]
		poe = float(poe_string)
		## get vs30s
		vs30s = []
		for vs30_nrmlObj in self.nrmlObj.findall("//{%s}vs30" % NRML):
			vs30_string = vs30_nrmlObj.text
			vs30 = float(vs30_string)
			vs30s.append(vs30)
		vs30s = numpy.array(vs30s)
		## return HazardMap object
		hazardMap = HazardMap(model, [self.xml_filespec], sites, period,
			iMT, intensities, intensity_unit="g", timespan=timespan, poe=poe, vs30s=vs30s)
		return hazardMap

	def uhsResultSet(self):
		"""
		method to parse hazard spectrum object from xml object
		returns:
			HazardSpectrum object
		"""
		# TODO: add automatic model_name and imt attributes (when added to openquake output)
		## test if nrmlObj is uhs result set
		assert self.nrmlObj.find("//{%s}uhsResultSet" % NRML), \
			"nrmlObj is not a hazard result set"
		## get model
		model = ""
		## get periods
		periods_nrmlObj = self.nrmlObj.find("//{%s}uhsPeriods" % NRML)
		periods_string = periods_nrmlObj.text
		periods_stringList = periods_string.split(" ")
		periods = map(float, periods_stringList)
		## get iMT
		iMT = ""
		## get timespan
		timespan_nrmlObj = self.nrmlObj.find("//{%s}timeSpan" % NRML)
		timespan_string = timespan_nrmlObj.text
		timespan = float(timespan_string)
		## get poes, sites and intensities
		poes, sites, intensities = [], None, []
		for poe_nrmlObj in self.nrmlObj.findall("//*[@poE]"):
			poe_string = poe_nrmlObj.attrib["poE"]
			poe = float(poe_string)
			poes.append(poe)
			# get sites and intensities from hdf5 files
			hdf5_filespec = poe_nrmlObj.attrib["path"]
			poe_sites, poe_intensities = Hdf5Parser(hdf5_filespec).hazardSpectrumResult()
			if not sites:
				sites = poe_sites
			intensities.append(poe_intensities)
		intensities = numpy.array(intensities)
		## return HazardSpectrum object
		uhsResultSet = UHSFieldSet(model, [self.xml_filespec], sites,
			periods, iMT, intensities, timespan=timespan, poes=poes)
		return uhsResultSet


class Hdf5Parser():
	"""
	class to parse hdf5 file
	"""
	# TODO: check if same names are used as in hazardresult.py

	def __init__(self, hdf5_filespec):
		"""
		initiator to create hdf5 object from hdf5 file
		arguments:
			hdf5_filespec
		initiates:
			hdf5Obj
		"""
		self.hdf5Obj = h5py.File(hdf5_filespec, "r").items()

	def hazardSpectrumResult(self):
		"""
		method to parse hazard spectrum result from hdf5 object
		returns:
			sites
			intensities
		"""
		sites, intensities = [], []
		for item in self.hdf5Obj:
			site_unicode = item[0]
			site_string = str(site_unicode)
			site_stringList = site_string.split("-")
			site = [float(coord[4:]) for coord in site_stringList]
			sites.append(tuple(site))
			intensities.append(list(item[1].value[0]))
		return sites, intensities

	def hazardDisaggregationResult(self):
		"""
		"""
		pass


def createSpectralHazardCurveFieldTree(root_dir, num_models):
	"""
	"""
	# TODO: complete
	filespecs = []
	shcfs = []
	for i in range(num_models):
		print "processing model %s/%s" % (i+1, num_models)
		filespec = os.path.join(root_dir, "hazardcurve-%s.xml" % i)
		filespecs.append(filespec)
		shcfs.append(NrmlParser(filespec).hazardCurveField().toSpectral())
	mean = NrmlParser(os.path.join(root_dir, "hazardcurve-mean.xml")).hazardCurveField().toSpectral()
	percentile_levels = []
	for filespec in os.listdir(root_dir):
		if "hazardcurve-quantile" in os.path.basename(filespec):
			percentile_levels.append(float(os.path.splitext(os.path.basename(filespec))[0][-3:])*100)
			percentile_levels.sort()
	percentiles = []
	for i, percentile_level in enumerate(percentile_levels):
		percentiles.append(NrmlParser(os.path.join(root_dir, "hazardcurve-quantile-%.2f.xml" % (percentile_level/100.))).hazardCurveField().toSpectral())
	return SpectralHazardCurveFieldTree.from_branches(shcfs, "", mean=mean, percentiles=percentiles, percentile_levels=percentile_levels)


if __name__ == "__main__":
	"""
	root_dir = r"X:\PSHA\CRISISvsOQ\openquake\Doel\PGA"
	num_models = 2400
	shctf = createSpectralHazardCurveFieldTree(root_dir, num_models)
	shcft.plot(title="OpenQuake Results for Doel (PGA)")
	"""

	filespec = r"X:\PSHA\CRISISvsOQ\openquake\Grid\hazardmap-0.0999123737477-mean.xml"
	#filespec = r"X:\PSHA\CRISISvsOQ\openquake\Grid\hazardmap-0.0208325719145-mean.xml"
	hm = NrmlParser(filespec).hazardMap()
	hm.model_name = "OpenQuake"
	print hm.num_sites
	hm = hm.trim(lonmax=6.85, latmax=51.85)
	print hm.num_sites
	hm.plot(amax=0.14)
