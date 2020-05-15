"""
Smoothed seismicity model from which a source model can be made.
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import sys
if sys.version[1] == '3':
	basestring = str

import numpy as np
import ogr

import scipy.stats

from mapping.geotools.geodetic import (spherical_distance, meshed_spherical_distance)

from ..geo import Point
from ..mfd import EvenlyDiscretizedMFD
from .point import PointSource
from .source_model import SourceModel

# TODO: to be completed


__all__ = ['SmoothedSeismicity']


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


class SmoothedSeismicity(object):
	"""
	Smoothed Seismicity over a rectangular geographic grid

	:param grid_outline:
		(lonmin, lonmax, latmin, latmax) tuple
	:param grid_spacing:
		int or float, lon/lat spacing
		or tuple of ints or floats, lon and lat spacing
		or string ending in 'km', spacing in km
	:param eq_catalog:
		instance of :class:`eqcatalog.EQCatalog`, earthquake catalog
		for the region
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
		float, minimum magnitude for smoothing
		(default: None, will use minimum completeness magnitude
	:param bandwidth:
		positive float, smoothing bandwidth (in km)
		(interpreted as minimum smoothing bandwidth
		if :param:`nth_neighbour` is > zero)
		or function returning bandwidth depending on earthquake magnitude
		(default: 15)
	:param nth_neighbour:
		positive integer, distance n-th closest earthquake as bandwidth
		(default: 0)
	:param kernel_shape:
		str, name of distribution supported by scipy.stats, defining
		shape of smoothing kernel
		e.g., 'norm', 'uniform'
		(default: 'norm')
	"""
	def __init__(self, grid_outline, grid_spacing,
				eq_catalog, completeness, Mtype='MW', Mrelation={},
				mag_bin_width=0.1, min_mag=None,
				bandwidth=15., nth_neighbour=0, kernel_shape='norm'):
		self.eq_catalog = eq_catalog
		self.grid_outline = grid_outline
		self.grid_spacing = grid_spacing
		self.completeness = completeness
		self.Mtype = Mtype
		self.Mrelation = Mrelation
		self.mag_bin_width = mag_bin_width
		self.set_min_mag(min_mag or self.completeness.min_mag)
		self.bandwidth = bandwidth
		self.nth_neighbour = nth_neighbour
		self.kernel_shape = kernel_shape

		self.init_grid()
		self.init_kernel()

	def init_grid(self):
		"""
		"""
		#from ..site import GenericSiteModel

		#self.grid = GenericSiteModel.from_grid_spec(self.grid_outline, self.grid_spacing)
		self.grid = MeshGrid.from_gridspec(self.grid_outline, self.grid_spacing)

	def set_grid_spacing(self, grid_spacing):
		"""
		"""
		self.grid_spacing = grid_spacing
		self.init_grid()

	@property
	def grid_lons(self):
		return self.grid.lons.flatten()

	@property
	def grid_lats(self):
		return self.grid.lats.flatten()

	def init_earthquakes(self):
		"""
		"""
		subcatalog = self.eq_catalog.subselect(Mmin=self.min_mag, Mtype=self.Mtype,
												Mrelation=self.Mrelation)
		eq_lons = subcatalog.lons
		eq_lats = subcatalog.lats
		eq_mags = subcatalog.get_magnitudes(Mtype=self.Mtype,
											Mrelation=self.Mrelation)
		nan_idxs = np.isnan(eq_lons) | np.isnan(eq_mags)
		self.eq_lons = eq_lons[~nan_idxs]
		self.eq_lats = eq_lats[~nan_idxs]
		self.eq_mags = eq_mags[~nan_idxs]

	def set_min_mag(self, min_mag):
		"""
		"""
		assert min_mag >= self.completeness.min_mag
		self.min_mag = min_mag
		self.init_earthquakes()

	def calc_inter_eq_distances(self):
		"""
		"""
		distances = meshed_spherical_distance(self.eq_lons, self.eq_lats,
											self.eq_lons, self.eq_lats)
		distances /= 1000
		return distances

	def calc_eq_grid_distances(self):
		"""
		"""
		distances = meshed_spherical_distance(self.eq_lons, self.eq_lats,
											self.grid_lons, self.grid_lats)
		distances /= 1000
		return distances

	def get_bandwidths(self):
		"""
		"""
		# TODO: magnitude-dependent bandwith?
		if not self.nth_neighbour:
			if callable(self.bandwidth):
				return np.asarray(self.bandwidth(self.eq_mags))
			else:
				return np.array([self.bandwidth])
		else:
			assert np.isscalar(self.bandwidth)
			distances = self.calc_inter_eq_distances()
			distances.sort(axis=0)
			distances = distances[self.nth_neighbour]
			distances = np.maximum(self.bandwidth, distances)
			return distances

	def init_kernel(self):
		"""
		"""
		kernel_func = getattr(scipy.stats, self.kernel_shape)
		bandwidths = self.get_bandwidths()[np.newaxis].T
		self.kernel = kernel_func(0, bandwidths)

	def set_nth_neighbour(self, value):
		self.nth_neighbour = value
		self.init_kernel()

	def get_grid_center(self):
		"""
		"""
		lonmin, lonmax, latmin, latmax = self.grid_outline
		center_lon = np.mean([lonmin, lonmax])
		center_lat = np.mean([latmin, latmax])
		return (center_lon, center_lat)

	def calc_norm_factor(self):
		"""
		Compute factor to normalize earthquake densities as the sum
		of the densities obtained in all grid nodes for an earthquake
		situated at the center of the grid
		"""
		center_lon, center_lat = self.get_grid_center()
		distances = spherical_distance(center_lon, center_lat,
										self.grid_lons, self.grid_lats)
		distances /= 1000.
		weights = self.kernel.pdf(distances)
		return 1. / np.sum(weights, axis=1)

	def calc_grid_densities(self, min_mag=None, max_mag=None):
		"""
		"""
		min_mag = min_mag or self.min_mag
		assert min_mag >= self.min_mag
		max_mag = max_mag or self.eq_mags.max() + self.mag_bin_width
		assert max_mag > min_mag

		norm_factor = self.calc_norm_factor()[np.newaxis].T
		distances = self.calc_eq_grid_distances()
		densities = self.kernel.pdf(distances)
		densities *= norm_factor

		idxs = (self.eq_mags >= min_mag) & (self.eq_mags < max_mag)
		return np.sum(densities[idxs], axis=0)

	def calc_total_eq_densities(self, min_mag=None, max_mag=None):
		"""
		This is to check that total density of each earthquake does not
		exceed 1
		"""
		min_mag = min_mag or self.min_mag
		assert min_mag >= self.min_mag
		max_mag = max_mag or self.eq_mags.max() + self.mag_bin_width
		assert max_mag > min_mag

		norm_factor = self.calc_norm_factor()[np.newaxis].T
		distances = self.calc_eq_grid_distances()
		densities = self.kernel.pdf(distances)
		densities *= norm_factor

		idxs = (self.eq_mags >= min_mag) & (self.eq_mags < max_mag)
		return np.sum(densities[idxs], axis=1)

	def plot_grid_densities(self, min_mag=None, max_mag=None, **kwargs):
		"""
		"""
		from plotting.generic_mpl import plot_grid

		densities = self.calc_grid_densities(min_mag=min_mag, max_mag=max_mag)
		densities = densities.reshape(self.grid.shape)

		return plot_grid(densities, self.grid.lons, self.grid.lats, **kwargs)

	def get_mag_bins(self, min_mag=None, max_mag=None):
		"""
		"""
		from ..utils import seq

		min_mag = min_mag or self.min_mag
		max_mag = max_mag or self.eq_mags.max() + self.mag_bin_width
		return seq(min_mag, max_mag, self.mag_bin_width)

	def calc_occurrence_rates(self, end_date, min_mag=None):
		"""
		"""
		min_mag = min_mag or self.min_mag
		mag_bins = self.get_mag_bins(min_mag)

		densities = np.zeros((len(mag_bins), len(self.grid_lons)))
		for i, mag_bin in enumerate(mag_bins):
			densities[i] = self.calc_grid_densities(mag_bin, mag_bin + self.mag_bin_width)
		time_spans = self.completeness.get_completeness_timespans(mag_bins, end_date)
		occurrence_rates = densities / time_spans[np.newaxis].T

		return occurrence_rates

	def calc_mfds(self, end_date, min_mag=None):
		"""
		"""
		occurrence_rates = self.calc_occurrence_rates(end_date, min_mag=min_mag)
		mag_bins = self.get_mag_bins(min_mag=min_mag)
		num_grid_cells = occurrence_rates.shape[1]
		mfd_list = []
		for i in range(num_grid_cells):
			mfd = EvenlyDiscretizedMFD(mag_bins[0] + self.mag_bin_width / 2.,
									self.mag_bin_width, occurrence_rates[:,i],
									Mtype=self.Mtype)
			mfd_list.append(mfd)

		return mfd_list

	def calc_total_mfd(self, end_date, min_mag=None):
		"""
		"""
		occurrence_rates = self.calc_occurrence_rates(end_date, min_mag=min_mag)
		occurrence_rates = occurrence_rates.sum(axis=1)
		mag_bins = self.get_mag_bins(min_mag=min_mag)
		mfd = EvenlyDiscretizedMFD(mag_bins[0] + self.mag_bin_width / 2.,
									self.mag_bin_width, occurrence_rates,
									Mtype=self.Mtype)

		return mfd


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

