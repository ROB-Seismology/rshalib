# -*- coding: utf-8 -*-

"""
RuptureSource class
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from .. import (oqhazlib, OQ_VERSION)
from openquake.hazardlib.source import (PointSource, AreaSource, SimpleFaultSource,
									ComplexFaultSource, CharacteristicFaultSource)


__all__ = ['RuptureSource']


class RuptureSource():
	"""
	Mixin class containing methods to explore rupture data
	independent of source type.

	The number of ruptures generated in an area source is equal to
	the number of area discretization points x number of magnitude bins
	x number of hypocentral depths x number of nodal planes
	"""
	def iter_ruptures(self, tom=None):
		"""
		OQ version-independent wrapper for :meth:`iter_ruptures`

		:param tom:
			instance of :class:`oqhazlib.tom.PoissonTOM` or None
			(default: None)
		"""
		if tom:
			timespan = self.timespan
			self.timespan = tom.time_span

		## OpenQuake version dependent arguments
		oqver_args = []
		if OQ_VERSION < '2.9.0':
			oqver_args = [self.tom]
		for rup in super(RuptureSource, self).iter_ruptures(*oqver_args):
			yield rup

		if tom:
			self.timespan = timespan

	def get_ruptures_poisson(self, mag=None, strike=None, dip=None, rake=None,
							depth=None, timespan=None):
		"""
		Generate ruptures according to Poissonian temporal occurrence model,
		optionally filtering by magnitude, strike, dip and rake.

		:param timespan:
			float, time interval for Poisson distribution, in years
			(default: None, will use :prop:`timespan`)
		:param mag:
			float, magnitude value (center of bin) of ruptures to plot
			(default: None)
		:param strike:
			float, strike in degrees of ruptures to plot (default: None)
		:param dip:
			float, dip in degrees of ruptures to plot (default: None)
		:param rake:
			float, rake in degrees of ruptures to plot (default: None)
		:param depth:
			float, depth in km of ruptures to plot (default: None)

		:return:
			list of instances of :class:`ProbabilisticRupture`
		"""
		if timespan:
			tom = oqhazlib.tom.PoissonTOM(timespan)
		else:
			tom = None
		ruptures = list(self.iter_ruptures(tom))

		## Filter by magnitude, strike, dip, and rake
		if mag is not None:
			ruptures = [rup for rup in ruptures if np.allclose(rup.mag, mag)]
		if strike is not None:
			ruptures = [rup for rup in ruptures
						if np.allclose(rup.surface.get_strike(), strike)]
		if dip is not None:
			ruptures = [rup for rup in ruptures
						if np.allclose(rup.surface.get_dip(), dip)]
		if rake is not None:
			ruptures = [rup for rup in ruptures if np.allclose(rup.rake, rake)]
		if depth is not None:
			ruptures = [rup for rup in ruptures
						if np.allclose(rup.hypocenter.depth, depth)]
		return ruptures

	def get_stochastic_event_set_poisson(self, timespan=None):
		"""
		Generate a random set of ruptures following a Poissonian temporal
		occurrence model. The stochastic event set represents a possible
		realization of the seismicity as described by the source,
		in the given time span.

		:param timespan:
			float, time interval for Poisson distribution, in years
			(default: None, will use :prop:`timespan`)

		:return:
			list of instances of :class:`ProbabilisticRupture`
		"""
		## OpenQuake version dependent call
		if OQ_VERSION >= '2.9.0':
			event_set = list(oqhazlib.calc.stochastic_event_set([self]))
		else:
			timespan = timespan or self.timespan
			event_set = list(oqhazlib.calc.stochastic_event_set_poissonian([self],
																		timespan))
		return event_set

	def get_rupture_bounds(self, rupture):
		"""
		Determine 3-D coordinates of rupture bounds

		:param rupture:
			instance of :class:`ProbabilisticRupture`

		:return:
			(lons, lats, depths) tuple of numpy float arrays
			depths are in km
		"""
		if isinstance(self, (PointSource, AreaSource)):
			corner_indexes = [0,1,3,2,0]
			lons = rupture.surface.corner_lons[corner_indexes]
			lats = rupture.surface.corner_lats[corner_indexes]
			depths = rupture.surface.corner_depths[corner_indexes]
		elif isinstance(self, (SimpleFaultSource, CharacteristicFaultSource)):
			mesh = rupture.surface.mesh
			all_lons, all_lats = mesh.lons.flatten(), mesh.lats.flatten()
			all_depths = mesh.depths.flatten()
			top_edge_depth = rupture.surface.get_top_edge_depth()
			top_edge_indexes = np.where(all_depths == top_edge_depth)[0]
			bottom_edge_depth = all_depths.max()
			bottom_edge_indexes = np.where(all_depths == bottom_edge_depth)[0]
			edge_indexes = np.concatenate([top_edge_indexes,
											bottom_edge_indexes[::-1],
											top_edge_indexes[:1]])
			lons = all_lons[edge_indexes]
			lats = all_lats[edge_indexes]
			depths = all_depths[edge_indexes]
			#polygon = rupture.surface.mesh.get_convex_hull()
			#lons.append(polygon.lons)
			#lats.append(polygon.lats)
		# TODO: ComplexFaultRupture
		return lons, lats, depths

	def get_rupture_centers(self, ruptures):
		"""
		Determine 3-D coordinates of rupture centers

		:param ruptures:
			list of instances of :class:`ProbabilisticRupture`
			(which should be generated by this rupturesource)

		:return:
			(lons, lats, depths) tuple of numpy float arrays
			depths are in km
		"""

		lons = np.array([rup.hypocenter.longitude for rup in ruptures])
		lats = np.array([rup.hypocenter.latitude for rup in ruptures])
		if isinstance(self, (PointSource, AreaSource)):
			## rup.hypocenter does not appear to have correct depth information
			depths = np.array([rup.surface.get_middle_point().depth for rup in ruptures])
		if isinstance(self, (SimpleFaultSource, ComplexFaultSource,
							CharacteristicFaultSource)):
			depths = np.array([rup.hypocenter.depth for rup in ruptures])
		return lons, lats, depths

	def plot_rupture_mags_vs_occurrence_rates(self, color_param=None,
										timespan=None,
										marker='o', cmap='jet', **kwargs):
		"""
		Plot rupture magnitudes with respect to occurrence rates

		:param color_param:
			str, which parameter to plot with varying color: one of
			"depth", "strike", "dip" or "rake"
			(default: None)
		:param timespan:
			float, time interval for Poisson distribution, in years
			(default: None, will use :prop:`timespan`)
		:param marker:
			char, matplotlib marker style
			(default: 'o')
		:param cmap:
			str, name of color map to use if :param:`color_param` is set
			(default: 'jet')
		:kwargs:
			optional keyword arguments understood by
			:func:`generic_mpl.plot_xy`

		:return:
			matplotlib Axes instance
		"""
		import matplotlib
		from plotting.generic_mpl import plot_xy

		ruptures = self.get_ruptures_poisson(timespan=timespan)
		mags = np.array([rup.mag for rup in ruptures])
		occurrences = np.array([rup.occurrence_rate for rup in ruptures])
		if color_param == "depth":
			c = self.get_rupture_centers(ruptures)[2]
			vmin, vmax = self.upper_seismogenic_depth, self.lower_seismogenic_depth
		elif color_param == "strike":
			c = np.array([rup.surface.get_strike() for rup in ruptures])
			vmin, vmax = 0, 360
		elif color_param == "dip":
			c = np.array([rup.surface.get_dip() for rup in ruptures])
			vmin, vmax = 0, 90
		elif color_param == "rake":
			c = np.array([rup.rake for rup in ruptures])
			vmin, vmax = -90, 90
		else:
			c = None
			colors = None

		if c is not None:
			norm  = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
			sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
			colors = sm.to_rgba(c)

		datasets = [(mags, occurrences)]
		kwargs['yscaling'] = kwargs.get('yscaling', 'log')
		kwargs['xlabel'] = kwargs.get('xlabel', 'Magnitude')
		kwargs['ylabel'] = kwargs.get('ylabel', 'Occurrence rate (1/yr)')
		kwargs['ymin'] = occurrences.min()
		kwargs['ymax'] = occurrences.max()

		return plot_xy(datasets, markers=[marker], colors=cmap,
						linestyles=None, linewidths=[0], **kwargs)

	def plot_rupture_bounds_3d(self, ruptures, fill=False):
		"""
		Plot rupture bounds in 3 dimensions.
		Note that lon, lat coordinates are transformed to UTM coordinates
		in order to obtain metric coordinates

		:param ruptures:
			list of instances of :class:`ProbabilisticRupture`.
		:param fill:
			bool, whether or not to plot ruptures with a transparent fill
			(default: False)
		"""
		import pylab
		import mpl_toolkits.mplot3d.axes3d as p3
		import mapping.geotools.coordtrans as coordtrans

		## Determine UTM zone and hemisphere
		utm_spec = coordtrans.get_utm_spec(*self.get_centroid())

		fig = pylab.figure()
		ax = p3.Axes3D(fig)
		for rup in ruptures:
			if -135 <= rup.rake <= -45:
				color = (0.,1.,0.)
			elif 45 <= rup.rake <= 135:
				color = (1.,0.,0.)
			else:
				color = (0.,0.,1.)
			lons, lats, depths = self.get_rupture_bounds(rup)
			x, y = coordtrans.lonlat_to_utm(lons, lats, utm_spec=utm_spec)
			x, y = x / 1000., y / 1000.
			ax.plot3D(x, y, -depths, color=color)
			if fill:
				tri = p3.art3d.Poly3DCollection([zip(x, y, -depths)])
				tri.set_color(color + (0.05,))
				tri.set_edgecolor('k')
				ax.add_collection3d(tri)

		## Plot source outline
		if isinstance(self, PointSource):
			origin = (self.location.longitude, self.location.latitude)
			src_longitudes, src_latitudes = [], []
			dist = 10
			num_azimuths = 37
			for azimuth in np.linspace(0, 360, num_azimuths):
				lon, lat = oqhazlib.geo.geodetic.point_at(origin[0], origin[1],
															azimuth, dist)
				src_longitudes.append(lon)
				src_latitudes.append(lat)
			src_depths = np.zeros(num_azimuths)
		elif isinstance(self, AreaSource):
			src_longitudes = self.polygon.lons
			src_latitudes = self.polygon.lats
			src_depths = np.zeros_like(src_longitudes)
		elif isinstance(self, SimpleFaultSource):
			polygon = self.get_polygon()
			src_longitudes = polygon.lons
			src_latitudes = polygon.lats
			src_depths = np.array([pt.depth for pt in polygon])

		x, y = coordtrans.lonlat_to_utm(src_longitudes, src_latitudes,
										utm_spec=utm_spec)
		x, y = x / 1000., y / 1000.
		ax.plot3D(x, y, -src_depths, 'k', lw=3)

		## Upper and lower seismogenic depth
		ax.plot3D(x, y, np.ones_like(x) * -self.upper_seismogenic_depth, 'k--', lw=3)
		ax.plot3D(x, y, np.ones_like(x) * -self.lower_seismogenic_depth, 'k--', lw=3)

		## Highlight first rupture if source is a fault
		if isinstance(self, (SimpleFaultSource, ComplexFaultSource)):
			lons, lats, depths = self.get_rupture_bounds(ruptures[0])
			x, y = coordtrans.lonlat_to_utm(lons, lats, utm_spec=utm_spec)
			x, y = x / 1000., y / 1000.
			ax.plot3D(x, y, -depths, 'm--', lw=3)

		## Plot decoration
		ax.set_xlabel("Easting (km)")
		ax.set_ylabel("Northing (km)")
		ax.set_zlabel("Depth (km)")
		pylab.title(self.name)
		pylab.show()

	def plot_rupture_map(self, mag, bounds=True, strike=None, dip=None, rake=None,
						timespan=None):
		"""
		Plot map showing rupture bounds or rupture centers.

		:param mag:
			float, magnitude value (center of bin) of ruptures to plot
		:param bounds:
			bool, whether rupture bounds (True) or centers (False) should
			be plotted (default: True)
		:param strike:
			float, strike in degrees of ruptures to plot (default: None)
		:param dip:
			float, dip in degrees of ruptures to plot (default: None)
		:param rake:
			float, rake in degrees of ruptures to plot (default: None)
		:param timespan:
			float, time interval for Poisson distribution, in years
			(default: None, will use :prop:`timespan`)
		"""
		from mpl_toolkits.basemap import Basemap

		## Generate ruptures, and filter according to magnitude bin, strike and rake
		ruptures = self.get_ruptures_poisson(timespan=timespan)
		ruptures = [rup for rup in ruptures if np.allclose(rup.mag, mag)]
		if strike is not None:
			ruptures = [rup for rup in ruptures
						if np.allclose(rup.surface.strike, strike)]
		if dip is not None:
			ruptures = [rup for rup in ruptures
						if np.allclose(rup.surface.get_dip(), dip)]
		if rake is not None:
			ruptures = [rup for rup in ruptures if np.allclose(rup.rake, rake)]

		## Get source coordinates
		if isinstance(self, PointSource):
			src_longitudes = [self.location.longitude]
			src_latitudes = [self.location.latitude]
		elif isinstance(self, AreaSource):
			src_longitudes = self.polygon.lons
			src_latitudes = self.polygon.lats
		elif isinstance(self, SimpleFaultSource):
			polygon = self.get_polygon()
			src_longitudes = polygon.lons
			src_latitudes = polygon.lats

		## Determine map bounds and central longitude and latitude
		llcrnrlon, llcrnrlat = min(src_longitudes), min(src_latitudes)
		urcrnrlon, urcrnrlat = max(src_longitudes), max(src_latitudes)
		if np.allclose(llcrnrlon, urcrnrlon):
			llcrnrlon, urcrnrlon = llcrnrlon - 0.1, urcrnrlon + 0.1
		if np.allclose(llcrnrlat, urcrnrlat):
			llcrnrlat, urcrnrlat = llcrnrlat - 0.1, urcrnrlat + 0.1
		lon_0 = (llcrnrlon + urcrnrlon) / 2.
		lat_0 = (llcrnrlat + urcrnrlat) / 2.

		## Initiate map
		map = Basemap(projection="aea", resolution='h', llcrnrlon=llcrnrlon,
					llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat,
					lon_0=lon_0, lat_0=lat_0)
		map.drawcoastlines()
		map.drawcountries()

		## Plot ruptures
		for rup in ruptures:
			if -135 <= rup.rake <= -45:
				color = 'g'
			elif 45 <= rup.rake <= 135:
				color = 'r'
			else:
				color = 'b'
			if bounds == True:
				lons, lats, depths = self.get_rupture_bounds(rup)
				x, y = map(lons, lats)
				map.plot(x, y, color)
		if bounds == False:
			lons, lats, depths = self.get_rupture_centers(ruptures)
			x, y = map(lons, lats)
			map.plot(x, y, 'o', color=color)

		## Highlight first rupture if source is a fault
		if isinstance(self, (SimpleFaultSource, ComplexFaultSource)):
			lons, lats, depths = self.get_rupture_bounds(ruptures[0])
			x, y = map(lons, lats)
			map.plot(x, y, 'm--', lw=3)

		## Plot source outline
		if isinstance(self, PointSource):
			pass
		elif isinstance(self, AreaSource):
			x, y = map(src_longitudes, src_latitudes)
			map.plot(x, y, 'k', lw=3)
		if isinstance(self, SimpleFaultSource):
			x, y = map(src_longitudes, src_latitudes)
			map.plot(x, y, 'k--', lw=2)
			x, y = map(self.longitudes, self.latitudes)
			map.plot(x, y, 'k', lw=3)

		map.drawmapboundary()
		pylab.title(self.name)
		pylab.show()


if __name__ == '__main__':
	pass
