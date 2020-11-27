"""
Smoothed seismicity model from which a source model can be made.
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import sys
if sys.version[0] == '3':
	basestring = str

import numpy as np
import ogr

import scipy.stats

from mapping.geotools.geodetic import (spherical_distance, meshed_spherical_distance)

from ..mfd import (EvenlyDiscretizedMFD, TruncatedGRMFD)


__all__ = ['SmoothedSeismicity', 'VereJonesKernel', 'PowerlawKernel',
			'AdaptiveBandwidthFunc']


class SmoothingKernel(object):
	def __init__(self, bandwidths):
		self.bandwidths = bandwidths

	def __call__(self, bandwidths):
		"""
		Spawn new instance with given bandwidths

		:param bandwidths:
			1-D float array, with length either 1 (single value)
			or equal to the number of earthquakes (different value
			for each earthquake)

		:return:
			instance of :class:`VereJonesKernel`
		"""
		return self.__class__(bandwidths)


class VereJonesKernel(SmoothingKernel):
	"""
	Isotropic power-law kernel proposed by Vere-Jones (1992)

	:param bandwidths:
		1-D float array, with length either 1 (single value)
		or equal to the number of earthquakes (different value
		for each earthquake)
	:param power:
		float, power-law index. Recommended values for this parameter
		are between 1.5 and 2, corresponding to a cubic or quadratic
		decay of the probability density function with distance
	"""
	def __init__(self, bandwidths, power=2.):
		super(VereJonesKernel, self).__init__(bandwidths)
		self.power = power

	def __call__(self, bandwidths):
		"""
		Spawn new instance with given bandwidths

		:param bandwidths:
			1-D float array, with length either 1 (single value)
			or equal to the number of earthquakes (different value
			for each earthquake)

		:return:
			instance of :class:`VereJonesKernel`
		"""
		return self.__class__(bandwidths, power=self.power)

	def pdf(self, distances):
		"""
		Compute probability density as a function of distance
		"""
		H2 = self.bandwidths**2
		pow = self.power
		return ((pow - 1) / (np.pi * H2)) * (1 + (distances**2 / H2))**-pow


class PowerlawKernel(SmoothingKernel):
	"""
	Isotropic adaptive power-law kernel (Helmstetter et al., 2007)
	Also used in Hiemer et al. (2014)

	:param bandwidths:
		1-D float array, with length either 1 (single value)
		or equal to the number of earthquakes (different value
		for each earthquake)
	:param power:
		float, power-law exponent.
		Hiemer et al. (2014) and Hochstetter et al. (2007) use a value of 1.5,
		ESHM20 used 1.66637
		(default: 1.5)
	"""
	def __init__(self, bandwidths, power=1.5):
		super(PowerlawKernel, self).__init__(bandwidths)
		self.power = power

	def pdf(self, distances):
		"""
		Compute probability density as a function of distance
		"""
		H2 = self.bandwidths**2
		return 1 / (distances**2 + H2)**self.power


class AdaptiveBandwidthFunc(object):
	"""
	Adaptive bandwidth function, growing exponentially with magnitude,
	possibly truncated to a maximum value

	:param c1:
		float, c1 parameter (e.g., 0.0041, cf. Molina et al., 2001)
	:param c2:
		float, c2 parameter (e.g., 1.58, cf. Molina et al., 2001)
	:param min:
		float, lower truncation value (in km)
		(default: 0.)
	:param max:
		float, upper truncation value (in km)
		(default: np.inf = untruncated)
	"""
	def __init__(self, c1, c2, min=0., max=np.inf):
		self.c1 = c1
		self.c2 = c2
		self.min = min
		self.max = max

	def __call__(self, mags):
		"""
		Compute magnitude-dependent bandwidths

		:param mags:
			1D array, earthquake magnitudes

		:return:
			1D array, bandwidths (in km)
		"""
		c1, c2 = self.c1, self.c2
		return np.maximum(self.min, np.minimum(self.max, c1 * np.exp(c2 * mags)))


class MeshGrid(object):
	"""
	"""
	def __init__(self, lons, lats):
		assert lons.shape == lats.shape
		assert len(lons.shape) == 2
		self.lons = lons
		self.lats = lats

	@property
	def shape(self):
		return self.lons.shape

	@classmethod
	def from_gridspec(cls, grid_outline, grid_spacing):
		"""
		Construct from outline and spacing

		:param grid_outline:
			(lonmin, lonmax, latmin, latmax) tuple
		:param grid_spacing:
			int or float, lon/lat spacing
			or tuple of ints or floats, lon and lat spacing
			or string ending in 'km', spacing in km
		"""
		lonmin, lonmax, latmin, latmax = grid_outline

		if isinstance(grid_spacing, (int, float)):
			grid_spacing = (grid_spacing, grid_spacing)
		elif isinstance(grid_spacing, basestring):
			assert grid_spacing.endswith("km")
			grid_spacing = float(grid_spacing[:-2])
			center_lat = np.mean([latmin, latmax])
			dlat = grid_spacing / 111.
			## Approximate longitude spacing
			dlon = grid_spacing / (111. * np.cos(np.radians(center_lat)))
			grid_spacing = (dlon, dlat)

		dlon, dlat = grid_spacing
		lons = np.arange(lonmin, lonmax+dlon/10., dlon)
		lats = np.arange(latmin, latmax+dlat/10., dlat)
		## In case of kilometric spacing, center grid nodes
		lons += (lonmax - lons[-1]) / 2.
		lats += (latmax - lats[-1]) / 2.
		grid_lons, grid_lats = np.meshgrid(lons, lats)

		return cls(grid_lons, grid_lats)

	def get_node_areas(self):
		"""
		Calculate areas corresponding to each grid node

		:return:
				2-D float array, node areas (in square km)
		"""
		from plotting.generic_mpl import grid_center_to_edge_coordinates

		edge_lons, edge_lats = grid_center_to_edge_coordinates(self.lons, self.lats)
		xdistances = spherical_distance(edge_lons[:,:-1], edge_lats[:,:-1],
											  edge_lons[:,1:], edge_lats[:,1:])
		ydistances = spherical_distance(edge_lons[:-1], edge_lats[:-1],
											  edge_lons[1:], edge_lats[1:])
		xdistances = (xdistances[:-1] + xdistances[1:]) / 2
		ydistances = (ydistances[:,:-1] + ydistances[:,1:]) / 2

		return ((xdistances / 1000) * (ydistances / 1000))


class SmoothedSeismicity(object):
	"""
	Smoothed seismicity over a rectangular geographic grid

	:param grid_outline:
		(lonmin, lonmax, latmin, latmax) tuple
	:param grid_spacing:
		int or float, lon/lat spacing
		or tuple of ints or floats, lon and lat spacing
		or string ending in 'km', spacing in km
	:param eq_catalog:
		instance of :class:`eqcatalog.EQCatalog`, earthquake catalog
		(declustered, not necessarily completeness-constrained) for the region
	:param completeness:
		instance of :class:`eqcatalog.Completeness`, defining catalog
		completeness
	:param Mtype:
		str, magnitude type: "ML", "MS" or "MW"
		(default: "MW")
	:param Mrelation:
		{str: str} dict, mapping name of magnitude conversion relation
		to magnitude type ("MW", "MS" or "ML")
		(default: {}, will use default Mrelation of catalog for Mtype)
	:param mag_bin_width:
		float, magnitude bin width
		(default: 0.1)
	:param min_mag:
		float, minimum magnitude for smoothing.
		This has implications for calculation of smoothing distances if
		:param:`nth_neighbour`: is set!
		In SHARE and ESHM20, a value of 4.5 was used.
		(default: None, will use minimum completeness magnitude
	:param kernel_shape:
		str, name of distribution supported by scipy.stats, defining
		shape of smoothing kernel, e.g., 'norm', 'uniform'
		or instance of :class:`SmoothingKernel`
		(default: 'norm')
	:param bandwidth:
		positive float, smoothing bandwidth (in km)
		(interpreted as minimum smoothing bandwidth
		if :param:`nth_neighbour` is > zero)
		or function returning bandwidth depending on earthquake magnitude
		(default: 15)
	:param nth_neighbour:
		positive integer, n-th closest earthquake to take distance
		as spatially adaptive bandwidth
		(default: 0)
	"""
	def __init__(self, grid_outline, grid_spacing,
				eq_catalog, completeness, Mtype='MW', Mrelation={},
				mag_bin_width=0.1, min_mag=None,
				kernel_shape='norm', bandwidth=15., nth_neighbour=0,
				max_smoothing_distance=np.inf):
		self.grid_outline = grid_outline
		self.grid_spacing = grid_spacing
		self.eq_catalog = eq_catalog
		self.completeness = completeness
		self.Mtype = Mtype
		self.Mrelation = Mrelation
		self.mag_bin_width = mag_bin_width
		self.kernel_shape = kernel_shape
		self.bandwidth = bandwidth
		self.nth_neighbour = nth_neighbour
		self.min_smoothing_distance = bandwidth
		self.max_smoothing_distance = max_smoothing_distance

		## Set up grid
		self.init_grid()
		## Initialize catalog
		self.init_eq_catalog()
		## Set minimum magnitude and initialize earthquakes and smoothing kernel
		self.set_min_mag(min_mag or self.completeness.min_mag)
		## Initialize smoothing kernel
		#self.init_kernel()

		## Property containing earthquake-grid distances
		self._eq_grid_distances = None

	def __len__(self):
		return len(self.grid_lons)

	def init_grid(self):
		"""
		Initialize grid from outline and spacing

		:return:
			None, sets :prop:`grid`
		"""
		self.grid = MeshGrid.from_gridspec(self.grid_outline, self.grid_spacing)

	def set_grid_spacing(self, grid_spacing):
		"""
		Change grid spacing to new value

		:param grid_spacing:
			see :meth:`__init__`

		:return:
			None, :prop:`grid_spacing` and :prop:`grid` are modified
			in place
		"""
		self.grid_spacing = grid_spacing
		self.init_grid()

	@property
	def grid_lons(self):
		return self.grid.lons.flatten()

	@property
	def grid_lats(self):
		return self.grid.lats.flatten()

	def get_node_areas(self):
		"""
		Calculate areas corresponding to each grid node

		:return:
				2-D float array, node areas (in square km)
		"""
		return self.grid.get_node_areas()

	def init_eq_catalog(self):
		"""
		Initialize earthquake catalog by applying completeness constraint

		:return:
			None, :prop:`eq_catalog` is modified in place
		"""
		self.eq_catalog = self.eq_catalog.subselect_completeness(self.completeness,
										Mtype=self.Mtype, Mrelation=self.Mrelation)

	def init_earthquakes(self):
		"""
		Initialize earthquakes depending on :prop:`min_mag`

		:return:
			None, :prop:`eq_lons`, :prop:`eq_lats`, :prop:`eq_mags` and
			:prop:`eq_datetimes` are set
		"""
		subcatalog = self.eq_catalog.subselect(Mmin=self.min_mag, Mtype=self.Mtype,
												Mrelation=self.Mrelation)
		eq_lons = subcatalog.lons
		eq_lats = subcatalog.lats
		eq_mags = subcatalog.get_magnitudes(Mtype=self.Mtype,
											Mrelation=self.Mrelation)
		eq_datetimes = subcatalog.get_datetimes()
		nan_idxs = np.isnan(eq_lons) | np.isnan(eq_mags)
		self.eq_lons = eq_lons[~nan_idxs]
		self.eq_lats = eq_lats[~nan_idxs]
		self.eq_mags = eq_mags[~nan_idxs]
		self.eq_datetimes = eq_datetimes[~nan_idxs]

	def set_min_mag(self, min_mag):
		"""
		Change minimum magnitude.
		This will reinitialize the smoothing kernel.

		:param min_mag:
			float, minimum magnitude considered for smoothing

		:return:
			None, :prop:`min_mag`, :prop:`eq_lons`, :prop:`eq_lats`
			and :prop:`eq_mags` are modified in place
		"""
		assert min_mag >= self.completeness.min_mag
		self.min_mag = min_mag
		self.init_earthquakes()
		self.init_kernel()

	def calc_inter_eq_distances(self):
		"""
		Compute distances between each earthquake and each other
		earthquake

		:return:
			2-D [num_eq, num_eq] float array, distances (in km)
		"""
		distances = meshed_spherical_distance(self.eq_lons, self.eq_lats,
											self.eq_lons, self.eq_lats)
		distances /= 1000
		return distances

	def calc_eq_grid_distances(self):
		"""
		Compute distances between each earthquake and each grid node,
		if necessary, and set :prop:`_eq_grid_distances`

		:return:
			2-D [num_eq, num_grid_nodes] float array, distances in km
		"""
		if (self._eq_grid_distances is None
			or self._eq_grid_distances.shape[1] != len(self.eq_lons)):
			distances = meshed_spherical_distance(self.eq_lons, self.eq_lats,
												self.grid_lons, self.grid_lats)
			distances /= 1000
			self._eq_grid_distances = distances
		else:
			distances = self._eq_grid_distances

		return distances

	def get_eq_bandwidths(self):
		"""
		Define smoothing bandwidths for each earthquake

		:return:
			1-D float array, with length either 1 (single value)
			or equal to the number of earthquakes (different value
			for each earthquake)
		"""
		if not self.nth_neighbour:
			if callable(self.bandwidth):
				return np.asarray(self.bandwidth(self.eq_mags))
			else:
				return np.array([self.bandwidth])
		else:
			## Note: should distance to nth neighbour be computed using all
			## earthquakes within the completeness limits or using only
			## earthquakes with magnitude >= min_mag (of each bin!)?
			## The latter would imply that smoothing distance for earthquakes
			## with a particular magnitude would not change if the completeness
			## of the catalogue is improved (i.e., lower magnitudes added),
			## which may be more correct...
			## On the other hand, if using the latter, a problem occurs for
			## the highest magnitudes, which don't have as many neighbours
			## (the absolute highest magnitude has none)!
			## Another thought: distance to nth neighbour could be different
			## depending on completeness of different magnitude intervals...
			assert np.isscalar(self.bandwidth)
			distances = self.calc_inter_eq_distances()
			distances.sort(axis=0)
			nth_neighbour = min(self.nth_neighbour, len(self.eq_lons)-1)
			distances = distances[nth_neighbour]
			distances = np.maximum(self.min_smoothing_distance, distances)
			distances = np.minimum(self.max_smoothing_distance, distances)
			#print(distances.max())
			return distances

	def init_kernel(self):
		"""
		Initialize smoothing kernel based on bandwidths and kernel shape

		:return:
			None, :prop:`kernel` is set
		"""
		bandwidths = self.get_eq_bandwidths()[np.newaxis].T
		if isinstance(self.kernel_shape, SmoothingKernel):
			self.kernel = self.kernel_shape(bandwidths)
		else:
			kernel_func = getattr(scipy.stats, self.kernel_shape)
			self.kernel = kernel_func(0, bandwidths)

	def set_nth_neighbour(self, value):
		"""
		Change nth neighbour setting. This will reinitialize the
		smoothing kernel

		:param value:
			int, new value for :prop:`nth_neighbour`

		:return:
			None, :prop:`nth_neighbour` and :prop:`kernel` are modified
		"""
		self.nth_neighbour = value
		self.init_kernel()

	def set_bandwidth(self, bandwidth):
		"""
		Change smoothing bandwidth. This will reinitialize the
		smoothing kernel

		:param bandwidth:
			float, new value for :prop:`bandwidth`

		:return:
			None, :prop:`bandwidth` and :prop:`kernel` are modified
		"""
		self.bandwidth = bandwidth
		self.init_kernel()

	def get_grid_center(self):
		"""
		Get coordinates of center of grid

		:return:
			(lon, lat) tuple of floats
		"""
		lonmin, lonmax, latmin, latmax = self.grid_outline
		center_lon = np.mean([lonmin, lonmax])
		center_lat = np.mean([latmin, latmax])
		return (center_lon, center_lat)

	def get_lonlat_index(self, lon, lat, flat=True):
		"""
		Determine index of grid node closest to given longitude and latitude

		:param lon:
			float, longitude
		:param lat:
			float, latitude
		:param flat:
			bool, whether index should be in flattened array (True)
			or in 2D array (False)

		:return:
			(idx0, idx1) tuple of ints (if :param:`flat` is false)
			or int (if :param:`flat` is True)
		"""
		lon_idx = np.argmin(np.abs(self.grid.lons[0] - lon))
		lat_idx = np.argmin(np.abs(self.grid.lats[:,0] - lat))
		if not flat:
			return (lat_idx, lon_idx)
		else:
			return np.ravel_multi_index((lat_idx, lon_idx), self.grid.shape)

	def calc_norm_factor(self):
		"""
		Compute factor to normalize earthquake densities as the sum
		of the densities obtained in all grid nodes for an earthquake
		situated at the center of the grid. In theory, it should normalize
		to unity over an infinite area.
		This normalization factor depends on the bandwidth(s).

		This should ensure that:
		- earthquakes that are well inside the grid should have a
		total probability of 1 over all grid nodes
		- earthquakes that are close to the edges of the grid (either
		inside or outside) should have a total probability less than 1
		- earthquakes that are far away from the grid should have
		a total probability close to 0

		:return:
			1-D float array, with length corresponding to number of
			earthquakes
		"""
		center_lon, center_lat = self.get_grid_center()
		idx = self.get_lonlat_index(center_lon, center_lat)
		center_lon = self.grid_lons[idx]
		center_lat = self.grid_lats[idx]
		distances = spherical_distance(center_lon, center_lat,
										self.grid_lons, self.grid_lats)
		distances /= 1000.
		weights = self.kernel.pdf(distances)
		return 1. / np.sum(weights, axis=1)

	def _parse_min_mag(self, min_mag):
		"""
		Parse minimum magnitude value, making sure it is not lower
		than :prop:`min_mag`

		:param min_mag:
			float or None

		:return:
			float
		"""
		min_mag = min_mag or self.min_mag
		assert min_mag >= self.min_mag
		return min_mag

	def _parse_max_mag(self, max_mag):
		"""
		Parse maximum magnitude value, making sure it is not lower
		than :prop:`min_mag`

		:param max_mag:
			float or None

		:return:
			float
		"""
		dM = self.mag_bin_width
		max_mag = max_mag or np.ceil(self.eq_mags.max()/dM) * dM
		assert max_mag > self.min_mag
		return max_mag

	def _parse_min_max_mag(self, min_mag, max_mag):
		"""
		Parse both minimum and maximum magnitude values

		:param min_mag:
			float or None
		:param max_mag:
			float or None

		:return:
			(min_mag, max_mag) tuple of floats
		"""
		min_mag = self._parse_min_mag(min_mag)
		max_mag = self._parse_max_mag(max_mag)
		assert max_mag > min_mag
		return (min_mag, max_mag)

	def calc_eq_densities(self, min_mag=None, max_mag=None, norm_by_area=False):
		"""
		Compute probability density of each earthquake at each grid node
		in a particular magnitude range

		:param min_mag:
			float, minimum magnitude to compute density for
			(default: None, will use :prop:`min_mag`)
		:param max_mag:
			float, maximum magnitude to compute density for
			(default: None, will use largest magnitude in catalog)
		:param norm_by_area:
			bool, whether or not to normalize densities by area
			(default: False)

		:return:
			2-D [num earthquakes, num grid_nodes] float array
		"""
		min_mag, max_mag = self._parse_min_max_mag(min_mag, max_mag)

		norm_factor = self.calc_norm_factor()[np.newaxis].T
		distances = self.calc_eq_grid_distances()
		densities = self.kernel.pdf(distances)
		densities *= norm_factor

		idxs = (self.eq_mags >= min_mag) & (self.eq_mags < max_mag)
		densities = densities[idxs]

		if norm_by_area:
			densities /= (self.get_node_areas().flatten()[np.newaxis])

		return densities

	def calc_grid_densities(self, min_mag=None, max_mag=None, norm_by_area=False):
		"""
		Compute total density due to all earthquakes in each grid node

		:param min_mag:
		:param max_mag:
		:param norm_by_area:
			see :meth:`calc_eq_densities`

		:return:
			1-D float array
		"""
		densities = self.calc_eq_densities(min_mag=min_mag, max_mag=max_mag,
													norm_by_area=norm_by_area)
		return np.sum(densities, axis=0)

	def calc_total_eq_densities(self, min_mag=None, max_mag=None, norm_by_area=False):
		"""
		Compute total density in the grid for each earthquake.
		This is mainly useful to check that the total density of each
		earthquake does not exceed 1

		:param min_mag:
		:param max_mag:
		:param norm_by_area:
			see :meth:`calc_eq_densities`

		:return:
			1-D float array
		"""
		densities = self.calc_eq_densities(min_mag=min_mag, max_mag=max_mag,
													norm_by_area=norm_by_area)
		return np.sum(densities, axis=1)

	def calc_moment_densities(self, min_mag=None, max_mag=None, unit='N.m',
									norm_by_area=False):
		"""
		Compute moment density for each earthquake at each grid node

		:param min_mag:
		:param max_mag:
		:param norm_by_area:
			see :meth:`calc_eq_densities`
		:param unit:
			str, moment unit, either 'dyn.cm' or 'N.m'
			(default: 'N.m')

		:return:
			2-D [num earthquakes, num grid_nodes] float array
		"""
		from eqcatalog.moment import mag_to_moment

		min_mag, max_mag = self._parse_min_max_mag(min_mag, max_mag)

		densities = self.calc_eq_densities(min_mag=min_mag, max_mag=max_mag,
													norm_by_area=norm_by_area)
		eq_mags = self.eq_mags[(self.eq_mags >= min_mag) & (self.eq_mags < max_mag)]
		eq_moments = mag_to_moment(eq_mags, unit=unit)
		eq_moments = eq_moments[np.newaxis].T

		return densities * eq_moments

	def calc_grid_moment_densities(self, min_mag=None, max_mag=None, unit='N.m',
											norm_by_area=False):
		"""
		Compute moment density in each grid node due to all earthquakes.
		Note that this does not correspond to the total moment that would
		have been released during the entire duration of the catalog,
		if completeness for the considered magnitude interval is not
		uniform.

		:param min_mag:
		:param max_mag:
		:param unit:
		:param norm_by_area:
			see :meth:`calc_moment_densities`

		:return:
			1-D float array
		"""
		moment_densities = self.calc_moment_densities(min_mag=min_mag,
													max_mag=max_mag, unit=unit,
													norm_by_area=norm_by_area)
		return np.sum(moment_densities, axis=0)

	def calc_grid_moment_rates(self, end_date=None, min_mag=None, max_mag=None,
								unit='N.m', norm_by_area=False):
		"""
		Compute moment rate in each grid node.
		Note that this does take into account possibly non-uniform
		completeness for the considered magnitude interval

		:param end_date:
			datetime.date object or int, end date or end year of the
			catalog
			(default: None, will use end date of :prop:`eq_catalog`
		:param min_mag:
		:param max_mag:
		:param unit:
		:param norm_by_area:
			see :meth:`calc_moment_densities`

		:return:
			1-D float array
		"""
		end_date = end_date or self.eq_catalog.end_date
		min_mag, max_mag = self._parse_min_max_mag(min_mag, max_mag)

		moment_densities = self.calc_moment_densities(min_mag=min_mag,
													max_mag=max_mag, unit=unit,
													norm_by_area=norm_by_area)
		eq_mags = self.eq_mags[(self.eq_mags >= min_mag) & (self.eq_mags < max_mag)]
		time_spans = self.completeness.get_completeness_timespans(eq_mags, end_date)
		time_spans = time_spans[np.newaxis].T
		moment_rate_densities = moment_densities / time_spans

		return np.sum(moment_rate_densities, axis=0)

	def extrapolate_grid_moments(self, end_date=None, min_mag=None, max_mag=None,
								unit='N.m', norm_by_area=False):
		"""
		Compute total (extrapolated) seismic moment in each grid node
		since the beginning of the catalog, assuming activity has been
		constant

		:param end_date:
		:param min_mag:
		:param max_mag:
		:param unit:
		:param norm_by_area:
			see :meth:`calc_grid_moment_rates`
		"""
		end_date = end_date or self.eq_catalog.end_date
		max_mag = self._parse_max_mag(max_mag)

		moment_rates = self.calc_grid_moment_rates(end_date, min_mag=min_mag,
													max_mag=max_mag, unit=unit,
													norm_by_area=norm_by_area)
		[time_span] = self.completeness.get_completeness_timespans([max_mag],
																	end_date)

		return moment_rates * time_span

	def get_mfd_bins(self, min_mag=None, max_mag=None):
		"""
		Get magnitude bins for magnitude-frequency distribution (MFD)

		:param min_mag:
			float, minimum magnitude to use for MFD
		:param max_mag:
			float, maximum magnitude to use for MFD

		:return:
			1-D float array
		"""
		from ..utils import seq

		min_mag, max_mag = self._parse_min_max_mag(min_mag, max_mag)
		return seq(min_mag, max_mag, self.mag_bin_width)

	def calc_grid_occurrence_rates(self, end_date=None, min_mag=None, max_mag=None,
											norm_by_area=False):
		"""
		Compute cumulative annual occurrence rates for a particular
		magnitude range in each grid node

		:param end_date:
			datetime.date object or int, end date or end year of the
			catalog
			(default: None, will use end date of :prop:`eq_catalog`
		:param min_mag:
		:param max_mag:
		:param norm_by_area:
			see :meth:`calc_eq_densities`

		:return:
			1-D float array
		"""
		end_date = end_date or self.eq_catalog.end_date
		min_mag, max_mag = self._parse_min_max_mag(min_mag, max_mag)

		densities = self.calc_eq_densities(min_mag=min_mag, max_mag=max_mag,
													norm_by_area=norm_by_area)
		eq_mags = self.eq_mags[(self.eq_mags >= min_mag) & (self.eq_mags < max_mag)]
		if len(eq_mags):
			time_spans = self.completeness.get_completeness_timespans(eq_mags, end_date)
			time_spans = time_spans[np.newaxis].T

			#[time_span] = self.completeness.get_completeness_timespans([min_mag], end_date)
			#return densities / time_span

			occurrence_rates = densities / time_spans

			## Note: in Hiemer et al. (2014) and Helmstetter et al. (2007),
			## occurrence rates are corrected for spatial (not temporal!)
			## variations of the completeness magnitude (Mc > min_mag) in each cell
			#occurrence_rates *= 10**(b_value * (Mc - self.min_mag))

			return np.sum(occurrence_rates, axis=0)
		else:
			return np.zeros(len(self.grid_lons))

	def calc_mfd_occurrence_rates(self, end_date=None, min_mag=None, max_mag=None):
		"""
		Compute annual occurrence rates for each magnitude bin of the MFD
		in each grid node

		:param end_date:
			see :meth:`calc_grid_occurrence_rates`
		:param min_mag:
		:param max_mag:
			see :meth:`get_mfd_bins`

		:return:
			2-D [num mag bins, num grid nodes] float array
		"""
		mag_bins = self.get_mfd_bins(min_mag, max_mag)

		occ_rates = np.zeros((len(mag_bins), len(self.grid_lons)))
		for i, mag_bin in enumerate(mag_bins):
			occ_rates[i] = self.calc_grid_occurrence_rates(end_date, mag_bin,
													mag_bin + self.mag_bin_width)

		return occ_rates

	def calc_incremental_mfds(self, end_date=None, min_mag=None, max_mag=None):
		"""
		Compute discretized MFD in each grid node

		:param end_date:
		:param min_mag:
		:param max_mag:
			see :meth:`calc_mfd_occurrence_rates`

		:return:
			list with instances of :class:`EvenlyDiscretizedMFD`
		"""
		occurrence_rates = self.calc_mfd_occurrence_rates(end_date, min_mag=min_mag,
														max_mag=max_mag)
		mag_bins = self.get_mfd_bins(min_mag, max_mag)
		num_grid_nodes = occurrence_rates.shape[1]

		mfd_list = []
		for i in range(num_grid_nodes):
			mfd = EvenlyDiscretizedMFD(mag_bins[0] + self.mag_bin_width / 2.,
									self.mag_bin_width, occurrence_rates[:,i],
									Mtype=self.Mtype)
			mfd_list.append(mfd)

		return mfd_list

	def calc_total_incremental_mfd(self, end_date=None, min_mag=None, max_mag=None):
		"""
		Compute total discretized MFD for the entire grid

		:param end_date:
		:param min_mag:
		:param max_mag:
			see :meth:`calc_mfd_occurrence_rates`

		:return:
			instance of :class:`EvenlyDiscretizedMFD`
		"""
		occurrence_rates = self.calc_mfd_occurrence_rates(end_date, min_mag=min_mag,
														max_mag=max_mag)
		occurrence_rates = occurrence_rates.sum(axis=1)
		mag_bins = self.get_mfd_bins(min_mag=min_mag, max_mag=max_mag)
		mfd = EvenlyDiscretizedMFD(mag_bins[0] + self.mag_bin_width / 2.,
									self.mag_bin_width, occurrence_rates,
									Mtype=self.Mtype)

		return mfd

	def calc_grid_a_values(self, b_value, end_date=None, min_mag=None):
		"""
		Compute a values based on the cumulative occurrence rates
		for the given minimum magnitude and the given b-value(s)

		:param b_value:
			float, uniform b-value
			or 1D array, b-values for each grid node
		:param end_date:
		:param min_mag:
			see :meth:`calc_grid_occurrence_rates`

		:return:
			1D float array
		"""
		min_mag = self._parse_min_mag(min_mag)

		cumul_rates = self.calc_grid_occurrence_rates(end_date, min_mag=min_mag)
		num_grid_nodes = len(cumul_rates)

		if np.isscalar(b_value):
			b_values = np.array([b_value] * num_grid_nodes)
		else:
			assert len(b_value) == num_grid_nodes
			b_values = np.asarray(b_value)

		a_values = np.log10(cumul_rates) + b_values * min_mag

		return a_values

	def calc_gr_mfds(self, b_value, end_date=None, min_mag=None, max_mag=None,
					  method='extrapolate_a', prior_weight=0.5):
		"""
		Compute Gutenberg-Richter MFD in each grid node based on the
		cumulative occurrence rates for the given magnitude range
		and the given b-value(s)

		:param b_value:
		:param end_date:
		:param min_mag:
		:param max_mag:
			see :meth:`calc_grid_a_values`
		:param method:
			str, GR calculation method
			- 'extrapolate_a': simple extrapolation of a-values,
			  requires :param:`b-balue` to be set, :param:`max_mag` is ignored
			- 'fit_incremental_multi': MLE fit GR to incremental MFD in each node
			  simultaneously (common b-value),
			  :param:`b-balue` is ignored
			- 'fit_incremental_single': MLE fit GR to incremental MFD in each node
			  separately (different b-values),
			  if :param:`b-balue` is set, it is used as prior value(s), combined
			  with :param:`prior_weight`
		:param prior_weight:
			float, weight for prior b-value(s) in fit_incremental_single method
			(default: 0.5)

		:return:
			list with instances of :class:`TruncatedGRMFD`
			or :class:`NatLogTruncatedGRMFD`
		"""
		min_mag, max_mag = self._parse_min_max_mag(min_mag, max_mag)

		if method == 'extrapolate_a':
			a_values = self.calc_grid_a_values(b_value, end_date=end_date,
														min_mag=min_mag)

			num_grid_nodes = len(a_values)

			if np.isscalar(b_value):
				b_values = np.array([b_value] * num_grid_nodes)
			else:
				assert len(b_value) == num_grid_nodes
				b_values = np.asarray(b_value)

			mfd_list = []
			for i in range(num_grid_nodes):
				mfd = TruncatedGRMFD(min_mag, max_mag, self.mag_bin_width,
										a_values[i], b_values[i], Mtype=self.Mtype)
				mfd_list.append(mfd)

		elif method[:15] == 'fit_incremental':
			from ..mfd import NatLogTruncatedGRMFD

			imfd_list = self.calc_incremental_mfds(end_date=end_date,
														min_mag=min_mag, max_mag=max_mag)

			dMi = self.mag_bin_width
			Mi = self.get_mfd_bins(min_mag, max_mag) + dMi / 2.
			num_bins = len(Mi)
			num_grid_nodes = len(imfd_list)
			nij = np.zeros((num_grid_nodes, num_bins))
			for i, imfd in enumerate(imfd_list):
				nij[i] = imfd.get_num_earthquakes(self.completeness, end_date)
			if method == 'fit_incremental_multi':
				from eqcatalog.calcGR_MLE import estimate_gr_params_multi
				alphas, beta, covs = estimate_gr_params_multi(nij, Mi, dMi,
															self.completeness,
															end_date, precise=False)
				betas = [beta] * num_grid_nodes
			elif method == 'fit_incremental_single':
				from eqcatalog.calcGR_MLE import estimate_gr_params
				alphas, betas, covs = [], [], []
				for i, ni in enumerate(nij):
					if not b_value:
						prior_b = 1.
						prior_weight = 0
					else:
						if np.isscalar(b_value):
							prior_b = b_value
						else:
							prior_b = b_value[i]
					alpha, beta, cov = estimate_gr_params(ni, Mi, dMi, self.completeness,
																	end_date, prior_b=prior_b,
																	prior_weight=prior_weight,
																	precise=False)
					alphas.append(alpha)
					betas.append(beta)
					covs.append(cov)

			mfd_list = []
			for i in range(num_grid_nodes):
				mfd = NatLogTruncatedGRMFD(min_mag, max_mag, self.mag_bin_width,
										alphas[i], betas[i], cov=covs[i], Mtype=self.Mtype)
				mfd_list.append(mfd)

		return mfd_list

	def calc_total_gr_mfd(self, b_value, end_date=None, min_mag=None,
								max_mag=None, method='extrapolate_a', prior_weight=0.5):
		"""
		Compute total Gutenberg-Richter MFD for the entire grid

		:param b_value:
		:param end_date:
		:param min_mag:
		:param max_mag:
		:param method:
		:param prior_weight:
			see :meth:`calc_gr_mfds`

		:return:
			instance of :class:`TruncatedGRMFD`
		"""
		if method == 'extrapolate_a' and np.isscalar(b_value):
			min_mag, max_mag = self._parse_min_max_mag(min_mag, max_mag)
			cumul_rates = self.calc_grid_occurrence_rates(end_date,
											min_mag=min_mag, max_mag=None)
			cumul_rate = np.sum(cumul_rates)
			a_value = np.log10(cumul_rate) + b_value * min_mag
			mfd = TruncatedGRMFD(min_mag, max_mag, self.mag_bin_width,
									a_value, b_value, Mtype=self.Mtype)

		else:
			from ..mfd import sum_mfds
			mfd_list = self.calc_gr_mfds(b_value, end_date=end_date,
										min_mag=min_mag, max_mag=max_mag,
										method=method, prior_weight=prior_weight)
			mfd = sum_mfds(mfd_list)

		return mfd

	def to_source_model(self, mfd_type, end_date=None, min_mag=None, max_mag=None,
						b_value=None, gr_method='extrapolate_a', prior_weight=0.5,
						area_src_model=None, tectonic_region_type='',
						rupture_mesh_spacing=2.5, magnitude_scaling_relationship='WC1994',
						rupture_aspect_ratio=1., upper_seismogenic_depth=0.,
						lower_seismogenic_depth=25., nodal_plane_distribution=None,
						hypocenter_distribution=None, timespan=1.):
		"""
		Generate point-source model from smoothed seismicity grid

		:param mfd_type:
			str, type of MFD to calculate for each grid node:
			'gr' (Gutenberg-Richter), 'incremental' (EvenlyDiscretized)
			or 'incremental+gr' (= incremental + GR from Mmax_obs onward)
		:param end_date:
		:param min_mag:
		:param max_mag:
			end date, minimum and maximum magnitude to compute MFDs
			see :meth:`calc_gr_mfds` or :meth:`calc_incremental_mfds`
		:param b_value:
			float, uniform b-value for grid nodes not covered by any
			source in :param:`area_src_model`.
			Only needed if :param:`mfd_type` is 'gr' or 'incremental+gr'
			(default: None)
		:param gr_method:
			str, GR calculation method (see :meth:`calc_gr_mfds`)
			Only needed if :param:`mfd_type` is 'gr' or 'incremental+gr'
			(default: 'extrapolate_a')
		:param prior_weight:
			float, weight for prior b-value(s) in fit_incremental_single gr_method
			(default: 0.5)
		:param area_src_model:
			instance of :class:`rshalib.source.SourceModel`
			background area source model to use to derive b-value
			(if :param:`mfd_type` == 'gr') and point-source
			properties. If not specified or if source model does not
			cover the entire grid, all individual properties need to be
			specified
			(default: None)
		:param tectonic_region_type:
			str, uniform tectonic region type
			(default: '')
		:param rupture_mesh_spacing:
			float, uniform rupture mesh spacing
			(default: 2.5)
		:param magnitude_scaling_relationship:
			str or object, uniform magnitude scaling relationship
			(default: 'WC1994')
		:param rupture_aspect_ratio:
			float, uniform rupture aspect ratio
			(default: 1.)
		:param upper_seismogenic_depth:
			float, uniform upper seismogenic depth
			(default: 0.)
		:param lower_seismogenic_depth:
			float, uniform lower seismogenic depth
			(default: 25.)
		:param nodal_plane_distribution:
			instance of :class:`rshalib.pmf.NodalPlaneDistribution`
			(default: None)
		:param hypocenter_distribution:
			instance of :class:`rshalib.pmf.HypocenterDistribution`
		:param timespan:
			float, uniform timespan for Poisson temporal occurrence model
			(default: 1.)

		:return:
			instance of :class:`rshalib.source.SourceModel`
		"""
		from mapping.geotools.pt_in_polygon import filter_points_by_polygon
		from ..geo import Point
		from .point import PointSource
		from .source_model import SourceModel

		num_grid_nodes = len(self.grid_lons)

		## Create dummy source for grid nodes that are not inside an area source
		dummy_mfd = type(str('DummyMFD'), (), {})()
		dummy_mfd.b_val = b_value
		dummy_src = type(str('DummySource'), (), {})()
		dummy_src.mfd = dummy_mfd
		dummy_src.tectonic_region_type = tectonic_region_type
		dummy_src.rupture_mesh_spacing = rupture_mesh_spacing
		dummy_src.magnitude_scaling_relationship = magnitude_scaling_relationship
		dummy_src.rupture_aspect_ratio = rupture_aspect_ratio
		dummy_src.upper_seismogenic_depth = upper_seismogenic_depth
		dummy_src.lower_seismogenic_depth = lower_seismogenic_depth
		dummy_src.nodal_plane_distribution = nodal_plane_distribution
		dummy_src.hypocenter_distribution = hypocenter_distribution
		dummy_src.timespan = timespan

		## Determine to which area source each grid node belongs
		node_src_dict = {}
		if area_src_model:
			for source in area_src_model.get_area_sources():
				idxs_inside, _ = filter_points_by_polygon(self.grid_lons,
														self.grid_lats, source)
				for idx in idxs_inside:
					node_src_dict[idx] = source

		## Compute MFDs
		if mfd_type in ('incremental', 'incremental+gr'):
			mfd_list = self.calc_incremental_mfds(end_date=end_date,
												min_mag=min_mag, max_mag=max_mag)
			if mfd_type == 'incremental':
				## Remove trailing zeros to speed up hazard computations
				for mfd in mfd_list:
					mfd.rstrip()
			else:
				imfd_list = mfd_list

		if mfd_type in ('gr', 'incremental+gr'):
			if gr_method in ('extrapolate_a', 'fit_incremental_single'):
				b_values = [node_src_dict.get(i, dummy_src).mfd.b_val
							for i in range(num_grid_nodes)]
				mfd_list = self.calc_gr_mfds(b_values, end_date=end_date,
											min_mag=min_mag, max_mag=max_mag,
											method=gr_method, prior_weight=prior_weight)
			elif gr_method == 'fit_incremental_multi':
				## Joint MFD fitting of all nodes in each area source
				from eqcatalog.calcGR_MLE import estimate_gr_params_multi
				from ..mfd import NatLogTruncatedGRMFD

				node_mfd_dict = {}
				end_date = end_date or self.eq_catalog.end_date
				imfd_list = self.calc_incremental_mfds(end_date=end_date,
															min_mag=min_mag, max_mag=max_mag)
				Mi = self.get_mfd_bins(min_mag, max_mag)
				dMi = self.mag_bin_width
				num_bins = len(Mi)

				for source in set(list(node_src_dict.values()) + [None]):
					if source is None:
						## Nodes outside source model
						node_idxs = [idx for idx in range(num_grid_nodes)
										if not idx in node_src_dict]
					else:
						node_idxs = [key for (key, val) in node_src_dict.items()
										if val is source]
					num_source_nodes = len(node_idxs)
					nij = np.zeros((num_source_nodes, num_bins))
					for i, idx in enumerate(node_idxs):
						imfd = imfd_list[idx]
						nij[i] = imfd.get_num_earthquakes(self.completeness, end_date)
					alphas, beta, covs = estimate_gr_params_multi(nij, Mi, dMi,
																self.completeness,
																end_date, precise=False)
					for i, idx in enumerate(node_idxs):
						mfd = NatLogTruncatedGRMFD(min_mag, max_mag, dMi, alphas[i],
													beta, cov=covs[i], Mtype=self.Mtype)
						node_mfd_dict[idx] = mfd

				mfd_list = [node_mfd_dict[idx] for idx in range(num_grid_nodes)]

			if mfd_type == 'incremental+gr':
				## Incremental (observed), extended with Gutenberg-Richter fit
				## for magnitudes above Mmax_obs
				mmax_obs = self._parse_max_mag(None)
				for idx in range(num_grid_nodes):
					imfd = imfd_list[idx]
					tmfd = mfd_list[idx]
					tmfd.min_mag = mmax_obs
					mfd_list[idx] = imfd + tmfd
		try:
			mfd_list
		except:
			raise Exception('MFD type %s not supported!' % mfd_type)

		## Create point sources
		point_sources = []
		grid_lons, grid_lats = self.grid_lons, self.grid_lats
		for i in range(num_grid_nodes):
			source_id = 'PT%04d' % i
			lon, lat = grid_lons[i], grid_lats[i]
			name = '(%s, %s)' % (lon, lat)
			point = Point(lon, lat)
			mfd = mfd_list[i]
			bg_source = node_src_dict.get(i, dummy_src)
			trt = bg_source.tectonic_region_type
			rms = bg_source.rupture_mesh_spacing
			msr = bg_source.magnitude_scaling_relationship
			rar = bg_source.rupture_aspect_ratio
			usd = bg_source.upper_seismogenic_depth
			lsd = bg_source.lower_seismogenic_depth
			npd = bg_source.nodal_plane_distribution
			hdd = bg_source.hypocenter_distribution
			pt_src = PointSource(source_id, name, trt, mfd, rms, msr,
								rar, usd, lsd, point, npd, hdd)
			point_sources.append(pt_src)

		if area_src_model:
			src_model_name = area_src_model.name + ' (smoothed)'
		else:
			src_model_name = 'Smoothed Seismicity'

		return SourceModel(src_model_name, point_sources)

	def get_grid_values(self, quantity, min_mag=None, max_mag=None,
							norm_by_area=False, **kwargs):
		"""
		Compute gridded values for a particular quantity, and return
		them as a 2-D array

		:param quantity:
			str, name of quantity to compute: 'density', 'occurrence_rate',
			'a_value', 'moment_density', 'moment_rate' or 'moment'
		:param min_mag:
			float, minimum magnitude to compute quantity for
			(default: None, will use :prop:`min_mag`)
		:param max_mag:
			float, maximum magnitude to compute quantity for
			(default: None, will use largest magnitude in catalog)
		:param norm_by_area:
			bool, whether or not to normalize densities by area
			(except if :param:`quantity` = 'a_value')
			(default: False)
		:kwargs:
			additional keyword-arguments for calc_grid_* functions:
			'end_date', 'b_value', 'unit'

		:return:
			2-D float array
		"""
		min_mag, max_mag = self._parse_min_max_mag(min_mag, max_mag)

		if quantity == 'density':
			values = self.calc_grid_densities(min_mag=min_mag, max_mag=max_mag,
														norm_by_area=norm_by_area)
		elif quantity == 'occurrence_rate':
			end_date = kwargs.pop('end_date', self.eq_catalog.end_date)
			values = self.calc_grid_occurrence_rates(end_date=end_date,
												min_mag=min_mag, max_mag=max_mag,
												norm_by_area=norm_by_area)
		elif quantity == 'moment_density':
			unit = kwargs.pop('unit', 'N.m')
			values = self.calc_grid_moment_densities(min_mag=min_mag,
													max_mag=max_mag, unit=unit,
													norm_by_area=norm_by_area)
		elif quantity == 'moment_rate':
			end_date = kwargs.pop('end_date', self.eq_catalog.end_date)
			unit = kwargs.pop('unit', 'N.m')
			values = self.calc_grid_moment_rates(end_date=end_date, min_mag=min_mag,
												max_mag=max_mag, unit=unit,
												norm_by_area=norm_by_area)
		elif quantity == 'moment':
			end_date = kwargs.pop('end_date', self.eq_catalog.end_date)
			unit = kwargs.pop('unit', 'N.m')
			values = self.extrapolate_grid_moments(end_date=end_date, min_mag=min_mag,
													max_mag=max_mag, unit=unit,
													norm_by_area=norm_by_area)
		elif quantity == 'a_value':
			end_date = kwargs.pop('end_date', self.eq_catalog.end_date)
			b_value = kwargs.pop('b_value')
			values = self.calc_grid_a_values(b_value, end_date=end_date,
											min_mag=min_mag)

		values = values.reshape(self.grid.shape)

		return values

	def plot_grid(self, quantity, min_mag=None, max_mag=None,
					norm_by_area=False, **kwargs):
		"""
		Plot grid for a particular quantity

		:param quantity:
		:param min_mag:
		:param max_mag:
		:param norm_by_area:
			see :meth:`get_grid_values`
		:kwargs:
			additional keyword-arguments for calc_grid_* functions
			('end_date', 'b_value', 'unit') or to pass to
			:func:`generic_mpl.plot_grid`

		:return:
			matplotlib axes instance
		"""
		from plotting.generic_mpl import plot_grid

		min_mag, max_mag = self._parse_min_max_mag(min_mag, max_mag)

		values = self.get_grid_values(quantity, min_mag=min_mag, max_mag=max_mag,
											norm_by_area=norm_by_area, **kwargs)
		kwargs.pop('end_date', None)
		kwargs.pop('b_value', None)
		unit = kwargs.pop('unit', 'N.m')

		if quantity == 'density':
			cbar_title = kwargs.pop('cbar_title', 'Earthquake density')
		elif quantity == 'occurrence_rate':
			cbar_title = kwargs.pop('cbar_title', 'Annual frequency')
		elif quantity == 'moment_density':
			cbar_title = kwargs.pop('cbar_title', 'Moment density (%s)' % unit)
		elif quantity == 'moment_rate':
			cbar_title = kwargs.pop('cbar_title', 'Moment rate (%s/yr)' % unit)
		elif quantity == 'moment':
			cbar_title = kwargs.pop('cbar_title', 'Extrapolated moment (%s)' % unit)
		elif quantity == 'a_value':
			cbar_title = kwargs.pop('cbar_title', 'GR a value')

		cbar_title += (' (M=%.1f - %.1f)' % (min_mag, max_mag))

		region = kwargs.pop('region', self.grid_outline)
		xmin = kwargs.pop('xmin', region[0])
		xmax = kwargs.pop('xmax', region[1])
		ymin = kwargs.pop('ymin', region[2])
		ymax = kwargs.pop('ymax', region[3])

		return plot_grid(values, self.grid.lons, self.grid.lats,
						xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
						cbar_title=cbar_title, **kwargs)

	def to_folium_layer(self, quantity, min_mag=None, max_mag=None,
						norm_by_area=False,
						cmap='jet', opacity=0.7, pixelated=False,
						vmin=None, vmax=None, **kwargs):
		"""
		Generate folium map layer

		:param quantity:
			str, name of quantity to compute: 'density', 'occurrence_rate',
			'a_value', 'moment_density', 'moment_rate' or 'moment'
		:param min_mag:
			float, minimum magnitude to compute quantity for
			(default: None, will use :prop:`min_mag`)
		:param max_mag:
			float, maximum magnitude to compute quantity for
			(default: None, will use largest magnitude in catalog)
		:param norm_by_area:
			bool, whether or not to normalize densities by area
			(except if :param:`quantity` = 'a_value')
			(default: False)
		:param cmap:
			str, name of matplotlib or branca colormap
			(default: 'jet')
		:param opacity:
			float in the range 0-1, grid opacity
			(default: 0.7)
		:param pixelated:
			bool, whether or not color should be uniform in each
			grid cell
			(default: False)
		:param vmin:
			float, value corresponding to first color in colormap
			(default: None)
		:param vmax:
			float, value corresponding to last color in colormap
			(default: None)
		:kwargs:
			additional keyword-arguments for calc_grid_* functions:
			'end_date', 'b_value', 'unit'

		:return:
			(image_overlay, colormap) tuple of elements, which can
			be added to a folium map
		"""
		from matplotlib.cm import get_cmap
		from matplotlib.colors import rgb2hex, hex2color
		import branca.colormap as cm
		import folium

		values = self.get_grid_values(quantity, min_mag=min_mag, max_mag=max_mag,
											norm_by_area=norm_by_area, **kwargs)
		unit = kwargs.get('unit', 'N.m')

		if quantity == 'density':
			title = 'Earthquake density'
		elif quantity == 'occurrence_rate':
			title = 'Annual frequency'
		elif quantity == 'moment_density':
			title = 'Moment density (%s)' % unit
		elif quantity == 'moment_rate':
			title = 'Moment rate (%s/yr)' % unit
		elif quantity == 'moment':
			title = 'Extrapolated moment (%s)' % unit
		elif quantity == 'a_value':
			title = 'GR a value'

		lonmin, lonmax, latmin, latmax = self.grid_outline
		bounds = [(latmin, lonmin), (latmax, lonmax)]

		if isinstance(cmap, basestring):
			try:
				cmap = get_cmap(cmap)
			except:
				cmap = getattr(cm.linear, cmap)
				colorfunc = lambda val: hex2color(cmap(val))
			else:
				colorfunc = cmap
				dc = 1. / cmap.N / 2.
				colors = cmap(np.linspace(dc, 1-dc, cmap.N))
				hex_colors = [rgb2hex(color) for color in colors]
				cmap = cm.LinearColormap(colors=hex_colors)

		vmin = vmin or values.min()
		vmax = vmax or values.max()
		cmap = cmap.scale(vmin, vmax)
		cmap.caption = title

		layer = folium.raster_layers.ImageOverlay(values, bounds, origin='lower',
												opacity=opacity, pixelated=pixelated,
												colormap=colorfunc, mercator_project=True,
												name=title)

		return (layer, cmap)

	def get_folium_map(self, quantity, min_mag=None, max_mag=None,
						norm_by_area=False,
						cmap='jet', opacity=0.7, pixelated=False,
						vmin=None, vmax=None, bgmap='OpenStreetMap',
						**kwargs):
		"""
		Generate folium map

		:param quantity:
		:param min_mag:
		:param max_mag:
		:param norm_by_area:
		:param cmap:
		:param opacity:
		:param pixelated:
		:param vmin:
		:param vmax:
		:kwargs:
			see :meth:`to_folium_layer`
		:param bgmap:
			str, background tile map
			(default: 'OpenStreetMap')

		:return:
			instance of :class:`folium.Map`
		"""
		import folium

		layer, cmap = self.to_folium_layer(quantity, min_mag=min_mag,
							max_mag=max_mag, norm_by_area=norm_by_area,
							cmap=cmap, opacity=opacity, pixelated=pixelated,
							vmin=vmin, vmax=vmax, **kwargs)

		map = folium.Map(tiles=bgmap, control_scale=True)
		lonmin, lonmax, latmin, latmax = self.grid_outline
		bounds = [(latmin, lonmin), (latmax, lonmax)]

		layer.add_to(map)
		cmap.add_to(map)

		map.fit_bounds(bounds)
		folium.LayerControl().add_to(map)

		return map


class LegacySmoothedSeismicity(object):
	"""
	Older implementation by Bart Vleminckx.
	Left here for illustration

	:param e_lons:
		1d np.array of floats, lons of earthquakes
	:param e_lats:
		1d np.array of floats, lats of earthquakes
	:param e_mags:
		1d np.array of floats, mags of earthquakes
	:param s_lons:
		1d np.array of floats, lons of source sites
	:param s_lats:
		1d np.array of floats, lats of source sites
	:param completeness:
		instance of eqcatalog.Completeness
	:param end_date:
		instance of datetime.date
	:param bin_width:
		positve float, width of magnitude bins
	:param bandwidth:
		positive float, bandwidth of smoothing (in km)
		or minimum bandwidth if :param:`number` is > zero
	:param number:
		positive integer, distance n-th closest earthquake as bandwidth
		(default: 0)
	"""
	def __init__(self, e_lons, e_lats, e_mags, s_lons, s_lats,
				completeness, end_date,
				bin_width, bandwidth, number=0):
		"""
		"""
		self.e_lons = e_lons
		self.e_lats = e_lats
		self.e_mags = e_mags
		self.s_lons = s_lons
		self.s_lats = s_lats
		self.completeness = completeness
		self.end_date = end_date
		self.bin_width = bin_width
		self.bandwidth = bandwidth
		self.number = number
		self._smooth()

	@property
	def region(self):
		"""
		Get bounding box of source sites

		:returns:
			tuple of four floats, (min_lon, max_lon, min_lat, max_lat)
		"""
		min_lon = self.s_lons.min()
		max_lon = self.s_lons.max()
		min_lat = self.s_lats.min()
		max_lat = self.s_lats.max()
		return (min_lon, max_lon, min_lat, max_lat)

	def _get_mag_bins(self, min_mag, max_mag):
		"""
		Give lower edges of magnitude bins.

		:param min_mag:
			float, minimum magnitude
		:param max_mag:
			float, maximum magnitude
		"""
		mag = min_mag
		mag_bins = []
		while mag <= max_mag:
			mag_bins.append(mag)
			mag += self.bin_width
		return np.array(mag_bins)

	def _get_bandwidths(self):
		"""
		"""
		distances = haversine(
			self.e_lons, self.e_lats,
			self.e_lons, self.e_lats,
			)
		distances.sort(axis=1)
		distances = distances[:,self.number]
		distances[distances < self.bandwidth] = self.bandwidth
		return distances

	def _smooth(self):
		"""
		"""
		from scipy.stats import norm

		distances = haversine(
			self.e_lons, self.e_lats,
			self.s_lons, self.s_lats,
			)
		if not self.number:
			rv = norm(0, self.bandwidth)
			weights = rv.pdf(distances)
		else:
			rv = norm(0, self._get_bandwidths()[np.newaxis].T)
			weights = rv.pdf(distances)

		e_sums = weights.sum(axis=1)
		weights /= e_sums[np.newaxis].T
		## Normalize weights such that each earthquake contributes 1 in total?
		## But: earthquakes that are too far away should contribute less...
		#e_sums = weights.sum(axis=0)
		#e_sums = np.minimum(e_sums, rv.pdf(0))
		#weights /= e_sums
		min_mag = self.completeness.min_mag
		max_mag = self.e_mags.max()
		mag_bins = self._get_mag_bins(min_mag, max_mag)
		values = np.zeros((mag_bins.shape[0], weights.shape[1]))
		for i, mag_bin in enumerate(mag_bins):
			indices = np.where(np.logical_and(
				self.e_mags >= mag_bin,
				self.e_mags < mag_bin+self.bin_width,
				))
			values[i] = weights[indices].sum(axis=0)
		time_spans = np.array(self.completeness.get_completeness_timespans(mag_bins,
																	self.end_date))
		values /= time_spans[np.newaxis].T
		self.values = values

	def _get_mfd_obs(self, i, max_mag=None):
		"""
		"""
		min_mag = self.completeness.min_mag + self.bin_width / 2.
		if max_mag:
			mag_bins = self._get_mag_bins(min_mag, max_mag)
		inc_occ_rates = np.trim_zeros(self.values[:,i].T[0], trim="b")
		if not np.allclose(inc_occ_rates, 0.):
			if max_mag:
				zeros = np.zeros(len(mag_bins) - len(inc_occ_rates))
				inc_occ_rates = np.concatenate((inc_occ_rates, zeros))
			return EvenlyDiscretizedMFD(min_mag, self.bin_width, inc_occ_rates)
		else:
			return None

	def _get_mfd_est(self, i, max_mag=None, b_val=None, method="Weichert"):
		"""
		"""
		mfd_obs = self._get_mfd_obs(i, max_mag)
		if mfd_obs:
			return mfd_obs.to_truncated_gr_mfd(self.completeness, self.end_date,
										b_val=b_val, method=method, verbose=False)
		else:
			return None

	def to_source_model(self, source_model, mfd_est_method="Weichert"):
		"""
		Get smoothed version of an area source_model.

		:param source_model:
			instance of :class:`rshalib.source.SourceModel`
		:param mfd_est_method:
			str, method to estimate mfds by

		:returns:
			instance of :class:`rshalib.source.SourceModel`
		"""
		from ..geo import Point
		from .point import PointSource
		from .source_model import SourceModel

		point_sources = []
		for i in np.ndindex(self.values.shape[1:2]):
			lon = self.s_lons[i]
			lat = self.s_lats[i]
			ogr_point = ogr.Geometry(ogr.wkbPoint)
			ogr_point.SetPoint(0, lon, lat)
			for source in source_model.get_area_sources():
				if source.to_ogr_geometry().Contains(ogr_point):
					Mmax = source.mfd.max_mag
					b_val = source.mfd.b_val
					mfd_est = self._get_mfd_est(i, max_mag=Mmax, b_val=b_val,
												method=mfd_est_method)
					if mfd_est:
						id = '%s' % i[0]
						name = '%.2f %.2f' % (lon, lat)
						point = Point(lon, lat)
						mfd_est.min_mag = source.mfd.min_mag
						trt = source.tectonic_region_type
						rms = source.rupture_mesh_spacing
						msr = source.magnitude_scaling_relationship
						rar = source.rupture_aspect_ratio
						usd = source.upper_seismogenic_depth
						lsd = source.lower_seismogenic_depth
						npd = source.nodal_plane_distribution
						hdd = source.hypocenter_distribution
						pt_src = PointSource(id, name, trt, mfd_est, rms, msr,
											rar, usd, lsd, point, npd, hdd)
						point_sources.append(pt_src)
		return SourceModel('Smoothed_' + source_model.name, point_sources)


def haversine(lon1, lat1, lon2, lat2, earth_radius=6371.227):
	"""
	Calculate geographical distance using the haversine formula.

	:param lon1:
		1d np array or float, lons of the first set of locations
	:param lat1:
		1d np array or float, lats of the frist set of locations
	:param lon2:
		1d np array or float, lons of the second set of locations
	:param lat2:
		1d np array or float, lats of the second set of locations
	:param earth_radius:
		radius of the earth in km, float

	:returns:
		np array, geographical distance in km
	"""
	cfact = np.pi / 180.
	lon1 = cfact * lon1
	lat1 = cfact * lat1
	lon2 = cfact * lon2
	lat2 = cfact * lat2

	## Number of locations in each set of points
	if not np.shape(lon1):
		nlocs1 = 1
		lon1 = np.array([lon1])
		lat1 = np.array([lat1])
	else:
		nlocs1 = np.max(np.shape(lon1))
	if not np.shape(lon2):
		nlocs2 = 1
		lon2 = np.array([lon2])
		lat2 = np.array([lat2])
	else:
		nlocs2 = np.max(np.shape(lon2))
	## Pre-allocate array
	distances = np.zeros((nlocs1, nlocs2))
	i = 0
	while i < nlocs2:
		## Perform distance calculation
		dlat = lat1 - lat2[i]
		dlon = lon1 - lon2[i]
		aval = (np.sin(dlat / 2.) ** 2.) + (np.cos(lat1) * np.cos(lat2[i]) *
											(np.sin(dlon / 2.) ** 2.))
		distances[:, i] = (2. * earth_radius * np.arctan2(np.sqrt(aval),
													  np.sqrt(1 - aval))).T
		i += 1
	return distances
