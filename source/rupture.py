"""
Ruptures
"""

from __future__ import absolute_import, division, print_function, unicode_literals


from .. import oqhazlib, OQ_VERSION

from ..msr import get_oq_msr
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
		msr = get_oq_msr(msr)

		if OQ_VERSION >= '2.9.0':
			tom = oqhazlib.tom.PoissonTOM(1)
			point_source = oqhazlib.source.PointSource("", "", trt, "", rms, msr,
										rar, tom, usd, lsd, hypocenter, npd, hdd)
		else:
			point_source = oqhazlib.source.PointSource("", "", trt, "", rms, msr,
											rar, usd, lsd, hypocenter, npd, hdd)
		surface = point_source._get_rupture_surface(mag, nodal_plane, hypocenter)

		## OpenQuake version dependent keyword arguments
		oqver_kwargs = {}
		if OQ_VERSION >= '2.9.0':
			oqver_kwargs['rupture_slip_direction'] = slip_direction
		if OQ_VERSION < '3.2.0':
			oqver_kwargs['source_typology'] = oqhazlib.source.PointSource
		rupture = cls(mag, rake, trt, hypocenter, surface, **oqver_kwargs)

		return rupture
