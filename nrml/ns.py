# -*- coding: utf-8 -*-

"""
nrml (and gml) namespace
"""

from __future__ import absolute_import, division, print_function, unicode_literals


## Namespace declaration
NRML_NS = "http://openquake.org/xmlns/nrml/0.4"
GML_NS = "http://www.opengis.net/gml"

NRML = "{%s}" % NRML_NS
GML = "{%s}" % GML_NS

NSMAP = {None: NRML_NS, "gml": GML_NS}

ROOT = "%snrml" % NRML


## Main source elements
SOURCE_MODEL = "%ssourceModel" % NRML
AREA_SOURCE = "%sareaSource" % NRML
POINT_SOURCE = "%spointSource" % NRML
COMPLEX_FAULT_SOURCE = "%scomplexFaultSource" % NRML
SIMPLE_FAULT_SOURCE = "%ssimpleFaultSource" % NRML
CHARACTERISTIC_FAULT_SOURCE = "%scharacteristicFaultSource" % NRML

## Source identifiers
NAME = "name"
ID = "id"
TECTONIC_REGION_TYPE = "tectonicRegion"

## MFDs
TRUNCATED_GUTENBERG_RICHTER_MFD = "%struncGutenbergRichterMFD" % NRML
A_VALUE = "aValue"
B_VALUE = "bValue"
MINIMUM_MAGNITUDE = "minMag"
MAXIMUM_MAGNITUDE = "maxMag"

EVENLY_DISCRETIZED_INCREMENTAL_MFD = "%sincrementalMFD" % NRML
BIN_WIDTH = "%sbinWidth" % NRML
OCCURRENCE_RATES = "%soccurRates" %NRML

## Probability distributions
PROBABILITY = "probability"

NODAL_PLANE_DISTRIBUTION = "%snodalPlaneDist" % NRML
NODAL_PLANE = "%snodalPlane" % NRML
STRIKE = "strike"
DIP = "dip"
RAKE = "rake"

HYPOCENTRAL_DEPTH_DISTRIBUTION = "%shypoDepthDist" % NRML
HYPOCENTRAL_DEPTH = "%shypoDepth" % NRML
DEPTH = "depth"

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

SURFACE = "%ssurface" % NRML

## Other source parameters
UPPER_SEISMOGENIC_DEPTH = "%supperSeismoDepth" % NRML
LOWER_SEISMOGENIC_DEPTH = "%slowerSeismoDepth" % NRML

MAGNITUDE_SCALING_RELATIONSHIP = "%smagScaleRel" % NRML
RUPTURE_ASPECT_RATIO = "%sruptAspectRatio" % NRML


## Site model
SITE_MODEL = "%ssiteModel" % NRML
SITE = "%ssite" % NRML
LON = "lon"
LAT = "lat"
VS30 = "vs30"
VS30TYPE = "vs30Type"
Z1PT0 = "z1pt0"
Z2PT5 = "z2pt5"
KAPPA = "kappa"


## Logic-Tree elements
LOGICTREE = "%slogicTree" % NRML

LOGICTREE_BRANCHINGLEVEL = "%slogicTreeBranchingLevel" % NRML
LOGICTREE_BRANCHSET = "%slogicTreeBranchSet" % NRML
LOGICTREE_BRANCH = "%slogicTreeBranch" % NRML

UNCERTAINTY_WEIGHT = "%suncertaintyWeight" % NRML
UNCERTAINTY_MODEL = "%suncertaintyModel" % NRML


## Hazard result
HAZARD_CURVES = "%shazardCurves" % NRML
HAZARD_CURVE = "%shazardCurve" % NRML
UNIFORM_HAZARD_SPECTRA = "%suniformHazardSpectra" % NRML
UHS = "%suhs" % NRML
HAZARD_MAP = "%shazardMap" % NRML
DISAGG_MATRICES = "%sdisaggMatrices" % NRML
DISAGG_MATRIX = "%sdisaggMatrix" % NRML
NODE = "%snode" % NRML
PROB = "%sprob" % NRML
IMLS = "%sIMLs" % NRML
POES = "%spoEs" % NRML
PERIODS = "%speriods" % NRML
IMT = "IMT"
INVESTIGATION_TIME = "investigationTime"
STATISTICS = "statistics"
QUANTILE_VALUE = "quantileValue"
SMLT_PATH = "sourceModelTreePath"
GMPELT_PATH = "gsimTreePath"
PERIOD = "saPeriod"
DAMPING = "saDamping"
POE = "poE"
IML = "iml"
TYPE = "type"
DIMS = "dims"
INDEX = "index"
VALUE = "value"
MAG_BIN_EDGES = "magBinEdges"
DIST_BIN_EDGES = "distBinEdges"
LON_BIN_EDGES = "lonBinEdges"
LAT_BIN_EDGES = "latBinEdges"
EPS_BIN_EDGES = "epsBinEdges"
TECTONIC_REGION_TYPES = "tectonicRegionTypes"


## additions for rshalib
SPECTRAL_HAZARD_CURVE_FIELD_TREE = "%sspectralHazardCurveFieldTree" % NRML
SPECTRAL_HAZARD_CURVE_FIELD = "%sspectralHazardCurveField" % NRML
HAZARD_CURVE_FIELD = "%shazardCurveField" % NRML
SPECTRAL_DEAGGREGATION_CURVE = "%sspectralDeaggregationCurve" % NRML
DEAGGREGATION_CURVE = "%sdeaggregationCurve" % NRML
DEAGGREGATION_SLICE = "%sdeaggregationSlice" % NRML

