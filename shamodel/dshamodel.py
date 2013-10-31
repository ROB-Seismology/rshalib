"""
:mod:`rshalib.shamodel.dshamodel` exports :class:`rshalib.shamodel.dshamodel.DSHAModel`
"""


import numpy as np

from openquake.hazardlib.calc import ground_motion_fields
from openquake.hazardlib.calc.filters import rupture_site_distance_filter, rupture_site_noop_filter
from openquake.hazardlib.gsim import get_available_gsims
from openquake.hazardlib.imt import PGD, PGV, PGA, SA
from openquake.hazardlib.scalerel import PointMSR
from openquake.hazardlib.source import Rupture

from hazard.rshalib.geo import NodalPlane, Point
from hazard.rshalib.pmf import HypocentralDepthDistribution, NodalPlaneDistribution
from hazard.rshalib.result import HazardMap, HazardMapSet
from hazard.rshalib.shamodel.base import SHAModelBase
from hazard.rshalib.site import SHASiteModel
from hazard.rshalib.site.ref_soil_params import REF_SOIL_PARAMS
from hazard.rshalib.source import PointSource


# TODO: add documentation


class DSHAModel(SHAModelBase):
	"""
	"""
	
	def __init__(self, name, lons, lats, depths, mags, gsim, sites=None, grid_outline=None, grid_spacing=None, soil_site_model=None, ref_soil_params=REF_SOIL_PARAMS, usd=0, lsd=30., imts=["PGA"], periods=[], correlation_model=None, truncation_level=0., maximum_distance=None):
		"""
		"""
		SHAModelBase.__init__(self, name, sites, grid_outline, grid_spacing, soil_site_model, ref_soil_params, truncation_level)
		assert len(lons) == len(lats) == len(depths) == len(mags)
		self.lons = lons
		self.lats = lats
		self.depths = depths
		self.mags = mags
		self.gsim = gsim
		self.usd = usd
		self.lsd = lsd
		self.imts = imts
		self.periods = periods
		self.correlation_model = correlation_model
		self.maximum_distance = maximum_distance
		self.realizations = 1. ## NOTE: only one realization possible

	def get_rupture(self, i): # NOTE: rupture strike/dip/rake but zero area
		"""
		:returns:
			instance of :class:`openquake.hazardlib.source.Rupture`
		"""
		lon = self.lons[i]
		lat = self.lats[i]
		depth = self.depths[i]
		mag = self.mags[i]
		nodal_plane = NodalPlane(0., 90, 0.)
		hypocenter = Point(lon, lat, depth)
		npd = NodalPlaneDistribution([nodal_plane], [1.])
		hdd = HypocentralDepthDistribution([depth], [1.])
		point_source = PointSource("", "", "", "", "",
			PointMSR(), 1, self.usd, self.lsd, hypocenter, npd, hdd)
		surface = point_source._get_rupture_surface(mag, nodal_plane, hypocenter)
		rupture = Rupture(mag, 0., "", hypocenter, surface, "")
		return rupture
	
	def _get_hazardlib_imts(self):
		"""
		:returns:
			list of instances of subclasses of :class:`openquake.hazardlib._IMT`
		"""
		imts = []
		for imt in self.imts:
			if imt[0] == "S":
				for period in self.periods:
					imts.append(eval(imt)(period, 5.))
			else:
				imts.append(eval(imt)())
		return imts

	def _get_hazardlib_rsdf(self):
		"""
		"""
		if self.maximum_distance:
			return rupture_site_distance_filter(self.maximum_distance)
		else:
			return rupture_site_noop_filter
	
	def run_hazardlib(self):
		"""
		"""
		soil_site_model = self.get_soil_site_model()
		imts = self._get_hazardlib_imts()
		gsim = get_available_gsims()[self.gsim]()
		rsdf = self._get_hazardlib_rsdf()
		hazard_maps = {imt: [] for imt in imts}
		for i in xrange(len(self.lons)):
			gmfs = ground_motion_fields(
				self.get_rupture(i),
				soil_site_model,
				imts,
				gsim,
				self.truncation_level,
				self.realizations,
				self.correlation_model,
				rsdf
				)
			for imt, gmf in gmfs.items():
				if isinstance(imt, SA):
					period = imt.period
					IMT = "SA"
				else:
					period = 0
					IMT = {PGV(): "PGV", PGD(): "PGD", PGA(): "PGA"}[imt]
				if self.correlation_model:
					intensities = gmf.getA()[:,0]
				else:
					intensities = gmf[:,0]
				hazard_maps[imt].append(HazardMap("", "", sites=self.sha_site_model, period=period, IMT=IMT, intensities=intensities, intensity_unit="g", timespan=1, poe=[], return_period=[], site_names=None, vs30s=None))
		hazard_map_sets = {}
		for imt in imts:
			if isinstance(imt, SA):
				period = imt.period
				IMT = "SA"
				key = period
			else:
				period = 0
				IMT = {PGV(): "PGV", PGD(): "PGD", PGA(): "PGA"}[imt]
				key = IMT
			intensities = np.array([hm.intensities for hm in hazard_maps[imt]])
			hazard_map_sets[key] = HazardMapSet("", "", self.sha_site_model, period, IMT, intensities, intensity_unit="g", timespan=1, poes=[1], return_periods=[], site_names=None, vs30s=None)
		return hazard_map_sets


if __name__ == "__main__":
	"""
	Test
	"""
	
	name = "TestDSHAModel"
	
	## earthquake
	lons =[4.5]
	lats=[50.5]
	depths=[10.]
	mags=[6.5]
	
	## catalog
#	from hazard.psha.Projects.SHRE_NPP.catalog import cc_catalog
#	lons = cc_catalog.lons
#	lats = cc_catalog.lats
#	depths = [d or 10 for d in cc_catalog.get_depths()]
#	mags = cc_catalog.mags
	
	sites = None
	grid_outline = (1, 8, 49, 52)
	grid_spacing = 0.1
	gsim = "CauzziFaccioli2008"
	usd = 0
	lsd = 30.
	imts = ["PGA"]
	periods = []
	correlation_model = None
	truncation_level = 0.
	maximum_distance = None
	
	dhsa_model = DSHAModel(
		name=name,
		lons=lons,
		lats=lats,
		depths=depths,
		mags=mags,
		sites=sites,
		grid_outline=grid_outline,
		grid_spacing=grid_spacing,
		gsim=gsim,
		usd=usd,
		lsd=lsd,
		imts=imts,
		periods=periods,
		correlation_model=correlation_model,
		truncation_level=truncation_level,
		maximum_distance=maximum_distance,
		)
	
	hazard_map_sets = dhsa_model.run_hazardlib()
	hm = hazard_map_sets["PGA"].get_max_hazard_map()
	hm.plot(title="", site_symbol="", intensity_levels=[], num_grid_cells=500) # TODO: fix colorbar ticks bug

