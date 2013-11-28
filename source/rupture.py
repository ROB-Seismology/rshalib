"""
"""


import openquake.hazardlib as oqhazlib

from ..geo import NodalPlane, Point
from ..pmf import HypocentralDepthDistribution, NodalPlaneDistribution


class Rupture(oqhazlib.source.Rupture):
	"""
	"""

	@classmethod
	def from_hypocenter(cls, lon, lat, depth, mag, strike, dip, rake, trt, rms, rar, usd, lsd, msr):
		"""
		Create rupture by extending hypocenter with magnitude scaling relationship.

		:params lon:
			float, lon of rupture hypocenter
		:params lat:
			float, lat of rupture hypocenter
		:params depth:
			float, depth of rupture hypocenter
		:params mag:
			float, mag of rupture
		:params strike:
			float, strike of rupture
		:params dip:
			float, dip of rupture
		:param rake:
			float, rake of rupture
		:param trt:
			str, tectonic region type
		:param rms:
			float, rupture mesh spacing (km)
		:param rar:
			float, rupture aspect ratio
		:param usd:
			float, upper seismogenic depth
		:param lsd:
			float, lower seismogenic depth
		:param msr:
			instance of subclass of :class:`openquake.hazardlib.scaler.BaseMSR`

		:return:
			instance of :class:`Rupture`
		"""
		hypocenter = Point(lon, lat, depth)
		nodal_plane = NodalPlane(strike, dip, rake)
		npd = NodalPlaneDistribution([nodal_plane], [1.])
		hdd = HypocentralDepthDistribution([depth], [1.])
		point_source = oqhazlib.source.PointSource("", "", trt, "", rms, msr, rar, usd, lsd, hypocenter, npd, hdd)
		surface = point_source._get_rupture_surface(mag, nodal_plane, hypocenter)
		rupture = cls(mag, rake, trt, hypocenter, surface, source_typology=oqhazlib.source.PointSource)
		return rupture

