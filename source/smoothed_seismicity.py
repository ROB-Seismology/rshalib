"""
Classes to smooth seismicity.
"""


import numpy as np

from openquake.hazardlib.geo.geodetic import geodetic_distance
from scipy.stats import norm

from ..geo import Point
from ..mfd import EvenlyDiscretizedMFD
from source import PointSource
from source_model import SourceModel


class SmoothedSeismicity(object):
	"""
	Each earthquake of a catalog is smoothed over a grid by a smoothing kernel
	with certain smoothing bandwidth. This bandwidth can be fixed for all
	earthquakes or variable, calculated seperatly for each earthquake by a
	smoothing number n, as the distance to the n-th closest earthquake. With a
	smoothing number, the smoothing distance is a minumum. It is possible to
	limit the smoothing to a certain distance. All smoothed values are summed
	per magnitude bin and divided by the completeness timespan for the bin.
	Estimated mfds can be calculated by providing maximum magnitude, b value and
	estimation method.
	"""
	
	def __init__(self, catalog, grid, completeness, mag_type, mag_relation, mag_bin_width, smoothing_kernel, smoothing_number, smoothing_bandwidth, smoothing_limit):
		"""
		:param catalog:
			instance of :class:`eqcatalog.EQCatalog`, cc catalog
		:param grid:
			instance of :class:`rshalib.site.SHASiteModel, grid of lons and lats
		:param completeness:
			instance of :class:`eqcatalog.Completeness`, completeness of catalog
		:param mag_type:
			see :method:`eqcatalog.EQCatalog.get_magnitudes`
		:param mag_relation:
			see :method:`eqcatalog.EQCatalog.get_magnitudes`
		:param mag_bin_width:
			float, magnitude bin width for smoothing
		:param smoothing_kernel:
			function, taking array of distances and bandwidth, and returning weights
		:param smoothing_number:
			int, number n to take distance to n-th closest earthquake is bandwidth
		:param smoothing_bandwidth:
			float, (minimum) bandwidth in km for smoothing kernel
		:param smoothing_limit:
			float, maximum distance in km for smoothing
		"""
		catalog.lons = catalog.lons
		catalog.lats = catalog.lats
		catalog.mags = catalog.get_magnitudes(mag_type, mag_relation)
		self.catalog = catalog
		self.grid = grid
		self.completeness = completeness
		self.mag_bin_width = mag_bin_width
		self.mag_bins = self._get_mag_bins()
		self.smoothing_kernel = smoothing_kernel
		self.smoothing_number = smoothing_number
		self.smoothing_bandwidth = smoothing_bandwidth
		self.smoothing_limit = smoothing_limit
		self.smoothing_bandwidths = self._get_smoothing_bandwidths()
		self.inc_occ_count = self._get_inc_occ_count()
		self.inc_occ_rates = self._get_inc_occ_rates()
		assert np.allclose(self.inc_occ_count.sum(), len(self.catalog.mags))
	
	@property
	def shape(self):
		"""
		:return:
			tuple, shape of grid + mag bins
		"""
		return self.grid.shape + self.mag_bins.shape
	
	def _get_mag_bins(self):
		"""
		:return:
			1d array, left edge of mag bins
		""" 
		return np.arange(self.completeness.min_mag, self.catalog.mags.max()+self.mag_bin_width, self.mag_bin_width)
	
	def _get_mag_bin_index(self, mag):
		"""
		:return:
			int, index of mag bin of mag
		"""
		for i in range(len(self.mag_bins))[:-1]:
			next_bin = self.mag_bins[i+1]
			if mag < next_bin and not np.allclose(mag, next_bin):
				return i
		return i
	
	def _get_smoothing_bandwidths(self): # TODO: include in inc_occ_count loop?
		"""
		:return:
			1d array (~ catalog) or None, smoothing bandwidths
		"""
		if self.smoothing_number:
			smoothing_bandwidths = np.zeros(len(self.catalog))
			for i in range(len(self.catalog)):
				smoothing_bandwidths[i] = sorted(geodetic_distance(
					self.catalog.lons,
					self.catalog.lats,
					self.catalog.lons[i],
					self.catalog.lats[i],
					))[self.smoothing_number]
			smoothing_bandwidths = smoothing_bandwidths.clip(self.smoothing_bandwidth)
			return smoothing_bandwidths
	
	def _get_inc_occ_count(self):
		"""
		:return:
			3d array (~ shape), smoothed incremental occurrence count
		"""
		inc_occ_count = np.zeros(self.shape)
		for i in range(len(self.catalog)):
			grid_distances = geodetic_distance(self.grid.lons, self.grid.lats, self.catalog.lons[i], self.catalog.lats[i])
			if self.smoothing_limit:
				grid_indices = np.where(grid_distances <= self.smoothing_limit)
			else:
				grid_indices = np.where(grid_distances)
			if self.smoothing_number:
				smoothing_bandwidth = self.smoothing_bandwidths[i]
			else:
				smoothing_bandwidth = self.smoothing_bandwidth
			smoothed_vals = self.smoothing_kernel(grid_distances[grid_indices], smoothing_bandwidth)
			smoothed_vals /= smoothed_vals.sum()
			j = self._get_mag_bin_index(self.catalog.mags[i])
			inc_occ_count[:,:,j][grid_indices] += smoothed_vals
		return inc_occ_count
	
	def _get_inc_occ_rates(self):
		"""
		:return:
			3d array (~ shape), smoothed incremental occurrence rates
		"""
		return np.apply_along_axis(np.divide, 2, self.inc_occ_count, self.completeness.get_completeness_timespans(self.mag_bins, self.catalog.end_date))
	
	def set_mfds_obs(self, max_mag=None):
		"""
		Set obs mfd for each grid cell.
		
		:param max_mag:
			float, left edge of max mag bin
		
		:set:
			"mfds_obs", np array (~ grid.shape) of None's or instances of :class:`rshalib.mfd.EvenlyDiscretizedMFD`
		"""
		min_mag = self.completeness.min_mag + self.mag_bin_width / 2.
		if max_mag:
			mag_bins = np.arange(min_mag, max_mag+self.mag_bin_width, self.mag_bin_width)
		mfds_obs = np.zeros(self.grid.shape, dtype=np.object)
		for i in np.ndindex(self.grid.shape):
			inc_occ_rates = np.trim_zeros(self.inc_occ_rates[i], trim="b")
			if not np.allclose(inc_occ_rates, 0.):
				if max_mag:
					zeros = np.zeros(len(mag_bins) - len(inc_occ_rates))
					inc_occ_rates = np.concatenate((inc_occ_rates, zeros))
				mfd_obs = EvenlyDiscretizedMFD(min_mag, self.mag_bin_width, inc_occ_rates)
			else:
				mfd_obs = None
			mfds_obs[i] = mfd_obs
		self.mfds_obs = mfds_obs
	
	def set_mfds_est(self, max_mag, b_val, method):
		"""
		Set est mfd for each grid cell.
		
		:param max_mag:
			float, left edge of max mag bin
		:param b_val
			float, b value
		:param method
			string, estimation method ("Weichert", "LSQi", or "LSQc")
		
		:set:
			"mfds_est", np array (~ grid.shape) of None's or instances of :class:`rshalib.mfd.TruncatedGRMFD`
		"""
		self.set_mfds_obs(max_mag)
		mfds_est = np.zeros(self.grid.shape, dtype=np.object)
		for i in np.ndindex(self.grid.shape):
			mfds_est[i] = self.mfds_obs[i].to_truncated_GR_mfd(self.completeness, self.catalog.end_date, b_val=b_val, method=method, verbose=False) if self.mfds_obs[i] else None
		self.mfds_est = mfds_est
	
	def get_cum_occ_count(self):
		"""
		Get smoothed cumulative occurrence count.
		
		:return:
			3d array (~ shape)
		"""
		return np.add.accumulate(self.inc_occ_count[:,:,::-1], 2)[:,:,::-1]
	
	def get_cum_occ_rates(self):
		"""
		Get smoothed cumulative occurrence rates.
		
		:return:
			3d array (~ shape)
		"""
		return np.add.accumulate(self.inc_occ_rates[:,:,::-1], 2)[:,:,::-1]
	
	def get_a_vals(self):
		"""
		Get a values.
		
		:return:
			2d array (~ grid.shape)
		"""
		a_vals = np.zeros(self.grid.shape)
		for i in np.ndindex(self.grid.shape):
			mfd_est = self.mfds_est[i]
			a_vals[i] = mfd_est.a_val if mfd_est else np.NAN
		return a_vals
	
	def get_b_vals(self):
		"""
		Get b values.
		
		:return:
			2d array (~ grid.shape)
		"""
		b_vals = np.zeros(self.grid.shape)
		for i in np.ndindex(self.grid.shape):
			mfd_est = self.mfds_est[i]
			b_vals[i] = mfd_est.b_val if mfd_est else np.NAN
		return b_vals

	def get_norm_a_vals(self, norm_area):
		"""
		Get normalized a values.
		
		:param norm_area:
			float, normalization area in km2
		:return:
			2d array (~ grid.shape)
		"""
		return np.log10((10**self.a_vals) * (norm_area / self.grid.areas))
	
	def get_smrs_obs(self):
		"""
		Get seismic moment rates obs.
		
		:return:
			2d array (~ grid.shape)
		"""
		smrs_obs = np.zeros(self.grid.shape)
		for i in np.ndindex(self.grid.shape):
			mfd_obs = self.mfds_obs[i]
			if mfd_obs:
				smrs_obs[i] = mfd_obs._get_total_moment_rate()
		return smrs_obs
	
	def get_smrs_est(self):
		"""
		Get seismic moment rates est.
		
		:return:
			2d array (~ grid.shape)
		"""
		smrs_est = np.zeros(self.grid.shape)
		for i in np.ndindex(self.grid.shape):
			mfd_est = self.mfds_est[i]
			if mfd_est:
				smrs_est[i] = mfd_est._get_total_moment_rate()
		return smrs_est
	
	def get_area_sources_mfd_obs(self, source_model_name):
		"""
		Sum obs mfds for each area source of area source model
		"""
		import ogr
		from eqcatalog.source_models import read_source_model
		source_model = read_source_model(source_model_name, verbose=False)
		mfds = {source_id: None for source_id in source_model if source_model[source_id]["obj"].GetGeometryName() == "POLYGON"}
		for i in np.ndindex(self.grid.shape):
			mfd_obs = self.mfds_obs[i]
			if mfd_obs:
				point = ogr.Geometry(ogr.wkbPoint)
				point.SetPoint(0, self.grid.lons[i], self.grid.lats[i])
				for source_id in mfds:
					if source_model[source_id]["obj"].Contains(point):
						if mfds[source_id]:
							mfds[source_id] = mfd_obs + mfds[source_id]
						else:
							mfds[source_id] = mfd_obs
		return mfds
	
	def get_area_sources_mfd_est(self, source_model_name):
		"""
		Sum est mfds for each area source of area source model
		"""
		import ogr
		from eqcatalog.source_models import read_source_model
		source_model = read_source_model(source_model_name, verbose=False)
		mfds = {source_id: None for source_id in source_model if source_model[source_id]["obj"].GetGeometryName() == "POLYGON"}
		for i in np.ndindex(self.grid.shape):
			mfd_est = self.mfds_est[i]
			if mfd_est:
				point = ogr.Geometry(ogr.wkbPoint)
				point.SetPoint(0, self.grid.lons[i], self.grid.lats[i])
				for source_id in mfds:
					if source_model[source_id]["obj"].Contains(point):
						if mfds[source_id]:
							mfds[source_id] = mfd_est + mfds[source_id]
						else:
							mfds[source_id] = mfd_est
		return mfds
	
	def get_total_mfd_obs(self):
		"""
		Get total obs mfd (sum of all obs mfds).
		
		:return:
			instance of :class:`rshalib.mfd.EvenlyDiscretizedMFD` or None
		"""
		total_mfd = None
		for i in np.ndindex(self.grid.shape):
			mfd_obs = self.mfds_obs[i]
			if mfd_obs:
				total_mfd = total_mfd + mfd_obs if total_mfd else mfd_obs
		return total_mfd
	
	def get_total_mfd_est(self):
		"""
		Get total est mfd (sum of all est mfds).
		
		:return:
			instance of :class:`rshalib.mfd.EvenlyDiscretizedMFD` or None
		"""
		total_mfd = None
		for i in np.ndindex(self.grid.shape):
			mfd_est = self.mfds_est[i]
			if mfd_est:
				total_mfd = total_mfd + mfd_est if total_mfd else mfd_est
		return total_mfd
	
	def to_source_model(self, smn, trt, rms, msr, rar, usd, lsd, npd, hdd):
		"""
		To source model.
		
		:param smn:
			see :class:`rshalib.source.SourceModel`
		:param trt:
			see :class:`rshalib.source.PointSource`
		:param rms:
			see :class:`rshalib.source.PointSource`
		:param msr:
			see :class:`rshalib.source.PointSource`
		:param rar:
			see :class:`rshalib.source.PointSource`
		:param usd:
			see :class:`rshalib.source.PointSource`
		:param lsd:
			see :class:`rshalib.source.PointSource`
		:param npd:
			see :class:`rshalib.source.PointSource`
		:param hdd:
			see :class:`rshalib.source.PointSource`
		
		:return:
			instance of :class:`rshalib.source.SourceModel`
		"""
		point_sources = []
		for i in np.ndindex(self.grid.shape):
			mfd_est = self.mfds_est[i]
			if mfd_est:
				lon = self.grid.lons[i]
				lat = self.grid.lats[i]
				id, name = str(i), str((lon, lat))
				point = Point(lon, lat)
				point_sources.append(PointSource(id, name, trt, mfd_est, rms, msr, rar, usd, lsd, point, npd, hdd))
		return SourceModel(smn, point_sources)
	
	def _plot_map(self, data, dlon=None, dlat=None, vmin=None, vmax=None, fig_title="", clb_title="", fig_filespec=None, fig_width=0, dpi=300):
		"""
		"""
		import mapping.Basemap.LayeredBasemap as lbmap
		map_layers = []
		cmt = lbmap.ThematicStyleColormap("jet", vmin=vmin, vmax=vmax)
		cbs = lbmap.ColorbarStyle(title=clb_title)
		gd = lbmap.GridData(self.grid.lons, self.grid.lats, data)
		gs = lbmap.GridStyle(color_map_theme=cmt, colorbar_style=cbs)
		map_layers.append(lbmap.MapLayer(gd, gs))
		for built_in in ("coastlines", "countries"):
			map_layers.append(lbmap.MapLayer(style=lbmap.LineStyle(), data=lbmap.BuiltinData(built_in)))
		map = lbmap.LayeredBasemap(layers=map_layers, region=self.grid.region, projection="merc", resolution="i", grid_interval=(dlon, dlat), title=fig_title)
		map.plot(fig_filespec=fig_filespec, fig_width=fig_width, dpi=dpi)


def normal_smoothing_kernel(distances, bandwidth):
	"""
	Normal smoothing kernel.
	
	:param distances:
		array
	:param bandwidth:
		float
	
	:return:
		array (~ distances), weights
	"""
	return norm.pdf(distances, 0., bandwidth)


def uniform_smoothing_kernel(distances, bandwidth):
	"""
	Uniform smoothing kernel.
	
	:param distances:
		array
	:param bandwidth:
		float
	
	:return:
		array (~ distances), weights
	"""
	return np.ones_like(distances)

