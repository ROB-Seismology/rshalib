"""
Ruptures
"""

from __future__ import absolute_import, division, print_function, unicode_literals

try:
	## Python 2
	basestring
except:
	## Python 3
	basestring = str


from .. import oqhazlib, OQ_VERSION

from ..geo import NodalPlane, Point
from ..pmf import HypocentralDepthDistribution, NodalPlaneDistribution



__all__ = ['Rupture']


if OQ_VERSION >= '2.9.0':
	_base_class = oqhazlib.source.BaseRupture
else:
	_base_class = oqhazlib.source.Rupture


class Rupture(_base_class):
	"""
	"""
	def __repr__(self):
		txt = '<%s Rupture M=%.2f, rake=%.1f>'
		txt %= (self.source_type, self.mag, self.rake)
		return txt

	@property
	def source_type(self):
		return self.source_typology.__name__[:-6]

	@classmethod
	def from_hypocenter(cls, lon, lat, depth, mag, strike, dip, rake, trt,
						rms, rar, usd, lsd, msr, slip_direction=0.):
		"""
		Create rupture by extending hypocenter with magnitude scaling
		relationship.

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
			str or instance of subclass of :class:`oqhazlib.scalerel.BaseMSR`
		:param slip_direction:
			float, angle describing rupture propagation (in degrees)
			Not supported by early versions of OpenQuake
			(default: 0.)

		:return:
			instance of :class:`Rupture`
		"""
		hypocenter = Point(lon, lat, depth)
		nodal_plane = NodalPlane(strike, dip, rake)
		npd = NodalPlaneDistribution([nodal_plane], [1.])
		hdd = HypocentralDepthDistribution([depth], [1.])
		if (not isinstance(msr, oqhazlib.scalerel.BaseMSR)
			and isinstance(msr, basestring)):
			msr = getattr(oqhazlib.scalerel, msr)()
		point_source = oqhazlib.source.PointSource("", "", trt, "", rms, msr,
											rar, usd, lsd, hypocenter, npd, hdd)
		surface = point_source._get_rupture_surface(mag, nodal_plane, hypocenter)

		extra_kwargs = {}
		if oqhazlib_version >= '2.9.0':
			extra_kwargs = {'rupture_slip_direction': slip_direction}
		rupture = cls(mag, rake, trt, hypocenter, surface,
						source_typology=oqhazlib.source.PointSource, **extra_kwargs)

		return rupture
