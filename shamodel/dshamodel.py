"""
:mod:`rshalib.shamodel.dshamodel` defines :class:`rshalib.shamodel.dshamodel.DSHAModel`.
"""


from openquake.hazardlib.calc import ground_motion_fields
from openquake.hazardlib.calc.filters import rupture_site_distance_filter, rupture_site_noop_filter
from openquake.hazardlib.gsim import get_available_gsims
from openquake.hazardlib.imt import PGA, PGD, PGV, SA
from openquake.hazardlib.scalerel import WC1994 
from openquake.hazardlib.source import Rupture

from base import SHAModelBase
from hazard.rshalib.geo import NodalPlane, Point, Polygon
from hazard.rshalib.pmf import HypocentralDepthDistribution, NodalPlaneDistribution
from hazard.rshalib.result import HazardMap
from hazard.rshalib.site import SHASite, SoilSiteModel, SoilSite
from hazard.rshalib.source import PointSource


# TODO: extract common methods with pshamodels to superclass
# TODO: make GMF set class
# TODO: support multiple nodal planes to account for unknown nodal plane
# TODO: support correlation models
# TODO: complete write_openquake


class DSHAModel(SHAModelBase):
	"""
	"""
	
	def __init__(self, rupture, gsim, truncation_level, imts, periods, sites=None, grid_outline=None, grid_spacing=None, realizations=1, maximum_distance=None):
		"""
		"""
		self.rupture = rupture
		self.grid_outline = grid_outline
		self.grid_spacing = grid_spacing
		self.imts = imts
		self.periods = periods
		self.gsim = gsim
		self.truncation_level = truncation_level
		self.realizations = realizations
		self.correlation_model = None
		if maximum_distance:
			self.rupture_site_filter = rupture_site_distance_filter(maximum_distance)
		else:
			self.rupture_site_filter = rupture_site_noop_filter
		
		super(DSHAModel, self).__init__(sites=sites, grid_outline=grid_outline, grid_spacing=grid_spacing)
	
	def _get_hazardlib_imts(self):
		"""
		"""
		imts = []
		for imt in self.imts:
			if imt == "SA":
				for period in self.periods:
					imts.append(eval(imt)(period, 5.)) # TODO: other damping than 5 available?
			else:
				imts.append(eval(imt)())
		return imts
	
	def run_hazardlib(self):
		"""
		"""
		sites = self.get_soil_site_model()
		gmfs = ground_motion_fields(self.rupture, sites, self._get_hazardlib_imts(), get_available_gsims()[self.gsim](), self.truncation_level, self.realizations, self.correlation_model, self.rupture_site_filter)
		hms = {}
		for imt, gmf in gmfs.items(): # TODO: better implementation to handle imt as string/object
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
			hms[IMT] = HazardMap("", "", sites=[SHASite(site.location.longitude, site.location.latitude) for site in sites.__iter__()], period=period, IMT="PGA", intensities=intensities, intensity_unit="g", timespan=50, poe=[], return_period=[], site_names=None, vs30s=None)
		return hms
	
#	def write_openquake(self):
#		"""
#		"""
#		pass


def get_gmf(lon, lat, mag, depth, gmm, truncation_level=1., strike=0., dip=45., rake=0., usd=0., lsd=30., rar=1., msr=WC1994(), grid_outline=(1., 8., 49., 52.), grid_spacing="10km", imts=["PGA"], periods=[], maximum_distance=None):
	"""
	"""
	nodal_plane = NodalPlane(strike, dip, rake)
	hypocenter = Point(lon, lat, depth)
	point_source = PointSource("", "", "", "", "", msr, rar, usd, lsd, hypocenter, NodalPlaneDistribution([nodal_plane], [1.]), HypocentralDepthDistribution([depth], [1.]))
	surface = point_source._get_rupture_surface(mag, nodal_plane, hypocenter)
	rupture = Rupture(mag, rake, "", hypocenter, surface, "")
	dsha_model = DSHAModel(rupture, gmm, truncation_level, imts, periods, grid_outline=grid_outline, grid_spacing=grid_spacing, realizations=1., maximum_distance=maximum_distance)
	return dsha_model.run_hazardlib()


if __name__ == "__main__":
	"""
	"""
	hm = get_gmf(lon=4.5, lat=50.5, mag=6.5, depth=10., gmm="AtkinsonBoore2006", truncation_level=1.)
	hm["PGA"].plot(title="") # TODO: fix colorbar ticks bug

