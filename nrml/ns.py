# -*- coding: utf-8 -*-

"""
nrml namespace
"""


## Namespace declaration
NRML_NS = 'http://openquake.org/xmlns/nrml/0.4'
GML_NS = 'http://www.opengis.net/gml'

NRML = "{%s}" % NRML_NS
GML = "{%s}" % GML_NS

NSMAP = {None: NRML_NS, "gml": GML_NS}

ROOT = "%snrml" % NRML

## Main elements
SOURCE_MODEL = "%ssourceModel" % NRML
AREA_SOURCE = "%sareaSource" % NRML
POINT_SOURCE = "%spointSource" % NRML
COMPLEX_FAULT_SOURCE = "%scomplexFaultSource" % NRML
SIMPLE_FAULT_SOURCE = "%ssimpleFaultSource" % NRML

## Source identifiers
ID = "%sid" % NRML
NAME = "%sname" % NRML
TECTONIC_REGION_TYPE = "%stectonicRegion" % NRML

## MFD's
TRUNCATED_GUTENBERG_RICHTER_MFD = "%struncGutenbergRichterMFD" % NRML
A_VALUE = "%saValue" % NRML
B_VALUE = "%sbValue" % NRML
MINIMUM_MAGNITUDE = "%sminMag" % NRML
MAXIMUM_MAGNITUDE = "%smaxMag" % NRML

EVENLY_DISCRETIZED_INCREMENTAL_MFD = "%sincrementalMFD" % NRML
BIN_WIDTH = "%sbinWidth" % NRML
OCCURRENCE_RATES = "%soccurRates" %NRML

## Probability distributions
PROBABILITY = "%sprobability" % NRML

NODAL_PLANE_DISTRIBUTION = "%snodalPlaneDist" % NRML
NODAL_PLANE = "%snodalPlane" % NRML
STRIKE = "%sstrike" % NRML
DIP = "%sdip" % NRML
RAKE = "%srake" % NRML

HYPOCENTRAL_DEPTH_DISTRIBUTION = "%shypoDepthDist" % NRML
HYPOCENTRAL_DEPTH = "%shypoDepth" % NRML
DEPTH = "%sdepth" % NRML

## Source geometry
POINT_GEOMETRY = "%spointGeometry" % NRML
AREA_GEOMETRY = "%sareaGeometry" % NRML
COMPLEX_FAULT_GEOMETRY = "%scomplexFaultGeometry" % NRML
SIMPLE_FAULT_GEOMETRY = "%ssimpleFaultGeometry" % NRML

POSITION = "%spos" % GML
POSITION_LIST = "%sposList" % GML
POINT = "%sPoint" % GML
LINE_STRING = "%sLineString" % GML

POLYGON = "%sPolygon" % GML
EXTERIOR = "%sexterior" % GML
LINEAR_RING = "%sLinearRing" % GML

FAULT_TOP_EDGE = "%sfaultTopEdge" % NRML
FAULT_BOTTOM_EDGE = "%sfaultBottomEdge" % NRML
INTERMEDIATE_EDGE = "%sintermediateEdge" % NRML

## Other source parameters
UPPER_SEISMOGENIC_DEPTH = "%supperSeismoDepth" % NRML
LOWER_SEISMOGENIC_DEPTH = "%slowerSeismoDepth" % NRML

MAGNITUDE_SCALING_RELATIONSHIP = "%smagScaleRel" % NRML
RUPTURE_ASPECT_RATIO = "%sruptAspectRatio" % NRML


## Site model
# TODO: see which names belong to GML namespace

SITE_MODEL = "%ssiteModel" % NRML

SITE = "%ssite" % NRML

LON = "%slon" % NRML
LAT = "%slat" % NRML
VS30 = "%svs30" % NRML
VS30TYPE = "%svs30Type" % NRML
Z1PT0 = "%sz1pt0" % NRML
Z2PT5 = "%sz2pt5" % NRML


## Logic-Tree elements
LOGICTREE = "%slogicTree" % NRML

LOGICTREE_BRANCHINGLEVEL = "%slogicTreeBranchingLevel" % NRML
LOGICTREE_BRANCHSET = "%slogicTreeBranchSet" % NRML
LOGICTREE_BRANCH = "%slogicTreeBranch" % NRML

UNCERTAINTY_WEIGHT = "%suncertaintyWeight" % NRML
UNCERTAINTY_MODEL = "%suncertaintyModel" % NRML


## Hazard result
SPECTRAL_HAZARD_CURVE_FIELD_TREE = "%sspectralHazardCurveFieldTree" % NRML
SPECTRAL_HAZARD_CURVE_FIELD = "%sspectralHazardCurveField" % NRML
HAZARD_CURVE_FIELD = "%shazardCurveField" % NRML
HAZARD_CURVE = "%shazardCurve" % NRML
IMT = "%sIMT" % NRML
PERIOD = "%speriod" % NRML
IMLS = "%sIMLS" % NRML
POES = "%spoes" % NRML
