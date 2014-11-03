"""
Smoothed seismicity model from which a source model can be made.
"""


import numpy as np
import ogr

from scipy.stats import norm

from ..geo import Point
from ..mfd import EvenlyDiscretizedMFD
from source import PointSource
from source_model import SourceModel


class SmoothedSeismicity(object):
	"""
	"""
	
	def __init__(self, e_lons, e_lats, e_mags, s_lons, s_lats, completeness, end_date, bin_width, bandwidth, number=None):
		"""
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
		:param number:
			positive integer, distance n-th closest earthquake as bandwidth
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
		time_spans = np.array(self.completeness.get_completeness_timespans(mag_bins, self.end_date))
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
			return mfd_obs.to_truncated_GR_mfd(self.completeness, self.end_date, b_val=b_val, method=method, verbose=False)
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
					mfd_est = self._get_mfd_est(i, max_mag=Mmax, b_val=b_val, method=mfd_est_method)
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
						point_sources.append(PointSource(id, name, trt, mfd_est, rms, msr, rar, usd, lsd, point, npd, hdd))
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

