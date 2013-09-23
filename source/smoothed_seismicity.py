# -*- coding: utf-8 -*-
"""
"""


##import matplotlib.pyplot as plt


##from openquake.hazardlib.scalerel import WC1994

##from hazard.rshalib.geo import NodalPlane, Point

##from hazard.rshalib.pmf import NodalPlaneDistribution, HypocentralDepthDistribution
##from hazard.rshalib.shamodel import PSHAModel
##from hazard.rshalib.site import SHASite
#from hazard.rshalib.source import PointSource, SourceModel
##from mapping.Basemap.LayeredBasemap import BuiltinData, ColorbarStyle, GridData, GridStyle, LayeredBasemap, LegendStyle, LineStyle, MapLayer, ThematicStyleColormap



import numpy as np

from hazard.rshalib.mfd import EvenlyDiscretizedMFD, TruncatedGRMFD
from hazard.rshalib.site import SHASiteModel as SiteModel


class SmoothedSeismicity(object):
	"""
	Class for smoothing seismicity.
	"""
	
	# TODO: nonlinear spacing closer to certain sites
	# TODO: what with earthquakes outside grid or close to grid boundary?
	# TODO: what with obs None and to est None
	# TODO: for smsr: what with two above todo's
	# TODO: check if to use left or right edge of mag bin for max mag
	# TODO: test smoothing number (account for earthquake itself)
	
	def __init__(self, grid_outline, grid_spacing, catalog, mag_bin_width, smoothing_kernel, smoothing_number=None, smoothing_bandwidth=None, smoothing_limit=1.):
		"""
		:param grid_outline:
			see :class:`rshalib.site.SHASiteModel`
		:param grid_spacing:
			see :class:`rshalib.site.SHASiteModel`
		:param catalog:
			instance of :class:`eqcatalog.EQCatalog`
		:param mag_bin_width:
			float, width of mag bins
		:param smoothing_kernel:
			pdf method of scipy.stats
		:param smoothing_number:
			int, use distance of n-th closest earthquake as smoothing bandwidth (default: None)
		:param smoothing_bandwidth:
			float, if smoothing number given this is minimum smoothing bandwidth (default: None)
		:param smoothing_limit:
			float, limit of smoothing in times smoothing bandwidth (default: 1.)
		"""
		self.site_model = SiteModel(
			grid_outline=grid_outline,
			grid_spacing=grid_spacing,
			)
		self.catalog = catalog
		self.mag_bin_width = mag_bin_width
		self.mag_bins = self._get_mag_bins()
		self.shape = (
			self.site_model.shape[0],
			self.site_model.shape[1],
			len(self.mag_bins),
			)
		self.smoothing_kernel = smoothing_kernel
		self.smoothing_number = smoothing_number
		self.smoothing_bandwidth = smoothing_bandwidth
		self.smoothing_limit = smoothing_limit
		
		self._smooth()
	
	def _get_mag_bins(self, min_mag=None, max_mag=None):
		"""
		:param min_mag:
			float, minimum magnitude (default: None, min mag of completeness)
		:param max_mag:
			float, maximim magnitude (default: None, max mag of catalog)
		
		:return:
			1d np array, lower edges of mag bins
		"""
		min_mag = min_mag or self.catalog.completeness.min_mag
		max_mag = max_mag or self.catalog.mags.max()
		mag_bins = []
		while min_mag < max_mag or np.allclose([min_mag], max_mag):
			mag_bins.append(min_mag)
			min_mag += self.mag_bin_width
		return np.array(mag_bins)
	
	def _get_mag_bin_i(self, mag):
		"""
		:param mag:
			float, mag to get index of bin for in self.mag_bins
		
		:return:
			int, index of mag
		"""
		for i, mag_bin in enumerate(self._get_mag_bins(max_mag=mag)):
			if mag > mag_bin or np.allclose([mag], mag_bin):
				mag_bin_i = i
		return mag_bin_i
	
	def _smooth(self):
		"""
		:set inc_occs:
			3d np array (~ self.shape)
		:set cum_occs:
			3d np array (~ self.shape)
		:set inc_occ_rates:
			3d np array (~ self.shape)
		:set cum_occ_rates:
			3d np array (~ self.shape)
		:set smoothing_bandwidths:
			2d np array (~ self.site_model.shape) or None
		"""
		if self.smoothing_number:
			smoothing_bandwidths = np.zeros(len(self.catalog))
		else:
			smoothing_bandwidths = None
		smoothing_bandwidth = self.smoothing_bandwidth
		inc_occs = np.zeros(self.shape)
		for i in xrange(len(self.catalog)):
			lon = self.catalog.lons[i]
			lat = self.catalog.lats[i]
			mag = self.catalog.mags[i]
			site_distances = self.site_model.get_geographic_distance(lon, lat)
			if self.smoothing_number:
				catalog_distances = self.catalog.get_epicentral_distances(lon, lat)
				smoothing_bandwidth = sorted(catalog_distances)[self.smoothing_number - 1]
				smoothing_bandwidth = max([smoothing_bandwidth, self.smoothing_bandwidth])
				smoothing_bandwidths[i] = smoothing_bandwidth
			i = np.where(site_distances <= smoothing_bandwidth * self.smoothing_limit)
			inc_occ = self.smoothing_kernel(site_distances[i], scale=smoothing_bandwidth)
			inc_occ /= inc_occ.sum()
			inc_occs[:,:,self._get_mag_bin_i(mag)][i] += inc_occ
		self.inc_occs = inc_occs
		self.cum_occs = np.add.accumulate(self.inc_occs[:,:,::-1], 2)[:,:,::-1]
		time_spans = self.catalog.completeness.get_completeness_timespans(
			self.mag_bins, self.catalog.end_date
			)
		self.inc_occ_rates = np.apply_along_axis(np.divide, 2, self.inc_occs, time_spans)
		self.cum_occ_rates = np.add.accumulate(self.inc_occ_rates[:,:,::-1], 2)[:,:,::-1]
		self.smoothing_bandwidths = smoothing_bandwidths
	
	def get_mfd_obs(self, lon, lat):
		"""
		Get observed (evenly discretized) mfd for site.
		
		:param lon:
			float, lon of site
		:param lat:
			float, lat of site
		
		:return:
			instance of :class:`rshalib.mfd.EvenlyDiscretizedMFD` or None
		"""
		inc_occ_rates = np.trim_zeros(
			self.inc_occ_rates[self.site_model.get_site(lon, lat, True)], trim='b'
			)
		if np.allclose(inc_occ_rates, 0.):
			return None
		else:
			return EvenlyDiscretizedMFD(
				self.mag_bins[0] + self.mag_bin_width / 2., self.mag_bin_width, inc_occ_rates
				)
	
	def get_mfd_est(self, lon, lat):
		"""
		Get estimated (truncated gr) mfd for site.
		
		:param lon:
			float, lon of site
		:param lat:
			float, lat of site
		
		:return:
			instance of :class:`rshalib.mfd.TruncatedGRMFD` or None
		"""
		mfd_obs = self.get_mfd_obs(lon, lat)
		if mfd_obs and len(np.where(mfd_obs.occurrence_rates != 0.)[0]) >= 2:
			max_mag_bay = mfd_obs.get_Bayesian_Mmax_pdf(completeness=self.catalog.completeness, end_date=self.catalog.end_date, verbose=False)
			max_mag = max_mag_bay[2].get_percentiles([0.5])[0]
			a_val = max_mag_bay[3][2]
			b_val = max_mag_bay[3][3]
			return TruncatedGRMFD(self.mag_bins[0], max_mag, self.mag_bin_width, a_val, b_val)
		else:
			return None
	
	def get_smrs_obs(self):
		"""
		Get seismic moment rates observed.
		
		2d np array (~ self.site_model.shape)
		"""
		smrs_obs = np.zeros(self.site_model.shape)
		for i in np.ndindex(self.site_model.shape):
			lon = self.site_model.lons[i]
			lat = self.site_model.lats[i]
			mfd_obs = self.get_mfd_obs(lon, lat)
			if mfd_obs:
				smrs_obs[i] = mfd_obs._get_total_moment_rate()
		return smrs_obs
	
	def get_tsmr_catalog_obs(self):
		"""
		Get total seismic moment rate observed of catalog.
		
		:returns:
			float
		"""
		min_mag = self.catalog.mags.min()
		max_mag = self.catalog.mags.max()
		mfd_obs = catalog.get_incremental_MFD(
			min_mag, max_mag, self.mag_bin_width, completeness=self.catalog.completeness, verbose=False
			)
		return mfd_obs._get_total_moment_rate()
	
	def get_smrs_est(self):
		"""
		Get seismic moment rates estimated.
		
		2d np array (~ self.site_model.shape)
		"""
		smrs_est = np.zeros(self.site_model.shape)
		for i in np.ndindex(self.site_model.shape):
			lon = self.site_model.lons[i]
			lat = self.site_model.lats[i]
			mfd_est = self.get_mfd_est(lon, lat)
			if mfd_est:
				smrs_est[i] = mfd_est._get_total_moment_rate()
		return smrs_est
	
	def get_tsmr_catalog_est(self, max_mag=None):
		"""
		Get total seismic moment rate estimated of catalog.
		
		:returns:
			float
		"""
		min_mag = self.catalog.mags.min()
		max_mag = max_mag or self.catalog.get_Bayesian_Mmax_pdf(completeness=self.catalog.completeness, verbose=False)[2].get_percentiles([0.5])[0]
		mfd_est = catalog.get_estimated_MFD(
			Mmin=min_mag,
			Mmax=max_mag,
			dM=self.mag_bin_width,
			completeness=self.catalog.completeness,
			verbose=False,
			)
		return mfd_est._get_total_moment_rate()
	
	def get_max_mags_obs(self):
		"""
		Get max mag observed for each site.
		
		:return:
			2d np array (~ self.site_model.shape)
		"""
		max_mags = np.zeros(self.site_model.shape)
		for i in np.ndindex(self.site_model.shape):
			lon = self.site_model.lons[i]
			lat = self.site_model.lats[i]
			mfd_obs = self._get_mfd_obs(lon, lat)
			if hasattr(mfd_obs, "max_mag"):
				max_mags[i] = mfd_obs.max_mag - self.dmag
		max_mags[max_mags == 0] = np.NaN
		return max_mags

	def get_max_mags_bay(self):
		"""
		Get max mag Bayesian for each site.
		
		:return:
			2d np array (~ self.site_model.shape)
		"""
		max_mags = np.zeros(self.site_model.shape)
		for i in np.ndindex(self.site_model.shape):
			lon = self.site_model.lons[i]
			lat = self.site_model.lats[i]
			mfd_est = self._get_mfd_est(lon, lat)
			if hasattr(mfd_est, "max_mag"):
				max_mags[i] = mfd_est.max_mag
		max_mags[max_mags == 0] = np.NaN
		return max_mags

	def get_max_mags_area_sources(self, area_source_model_name):
		"""
		Get max mag for each site from area source model.
		
		:param area_source_model_name:
			str, name of area_source_model_name (options: ??)

		:return:
			2d np array (~ self.site_model.shape)
		"""
		area_source_model = read_source_model(source_model_name, verbose=False)
		max_mags = np.zeros(self.site_model.shape)
		for i in np.ndindex(self.site_model.shape):
			point = ogr.Geometry(ogr.wkbPoint)
			point.SetPoint(0, self.site_model.lons[i], self.site_model.lats[i])
			for source in source_model:
				if (source_model[source]['obj'].GetGeometryType() == 3 and
					source_model[source]['obj'].Contains(point)):
					column = rob_source_models_dict[source_model_name].column_map['max_mag']
					max_mags[i] = source_model[source][column]
		max_mags[max_mags == 0] = np.NaN
		return max_mags

	def get_a_vals(self):
		"""
		Get a value for each site.
		
		:return:
			2d np array (~ self.site_model.shape)
		"""
		a_vals = np.zeros(self.site_model_shape)
		for i in np.ndindex(self.site_model_shape):
			lon = self.site_model.lons[i]
			lat = self.site_model.lats[i]
			mfd_est = self.get_mfd_est(lon, lat)
			if hasattr(mfd_est, "a_val"):
				a_vals[i] = mfd_est.a_val
		a_vals[a_vals == 0] = np.NaN
		return a_vals
	
	def get_b_vals(self):
		"""
		Get b value for each site.
		
		:return:
			2d np array (~ self.site_model.shape)
		"""
		b_vals = np.zeros(self.site_model_shape)
		for i in np.ndindex(self.site_model_shape):
			lon = self.site_model.lons[i]
			lat = self.site_model.lats[i]
			mfd_est = self._get_mfd_est(lon, lat)
			if hasattr(mfd_est, "b_val"):
				b_vals[i] = mfd_est.b_val
		b_vals[b_vals == 0] = np.NaN
		return b_vals

	def to_source_model(self, name):
		"""
		"""
		point_sources = []
		for i in np.ndindex(self.site_model.shape):
			lon = self.site_model.lons[i]
			lat = self.site_model.lats[i]
			mfd = self.get_mfd_est(lon, lat)
			point = Point(lon, lat)
			point_sources.append(PointSource(str(i), str((self.lons[i], self.lats[i])), trt, mfd, rms, msr, rar, usd, lsd, point, npd, hdd))
		return SourceModel(name, point_sources)


if __name__ == "__main__":
	"""
	"""
	grid_outline = (1., 8., 49., 52.)
#	grid_outline = [(1, 49), (8, 52)]
#	grid_outline = [(1, 49), (8, 49), (8, 52)]
#	grid_spacing = "5km"
#	grid_spacing = (0.05, 0.1)
	grid_spacing = 0.1
	from hazard.psha.Projects.SHRE_NPP.catalog import catalog
	dmag = 0.1
	from scipy.stats import norm
	smoothing_kernel = norm.pdf
	smoothing_number = None
	smoothing_bandwidth = 20.
	smoothing_limit = 1.
	
	ss = SmoothedSeismicity(grid_outline, grid_spacing, catalog, dmag, smoothing_kernel, smoothing_number, smoothing_bandwidth, smoothing_limit)

#	print ss.site_model.shape
#	print len(ss.site_model)
#	print ss.site_model.slons
#	print ss.site_model.slats

#	print len(ss.catalog)
#	print ss.catalog.mags.max()
#	print ss.mag_bins

#	print ss.inc_occs.max()
#	print ss.smoothing_bandwidths
	
#	print ss.get_smrs_obs().sum()
#	print ss.get_tsmr_catalog_obs()
	
#	print ss.get_smrs_est().sum()
#	print ss.get_tsmr_catalog_est()

#	ss.get_mfd_obs(4.5, 50.5).plot()
#	ss.get_mfd_est(4.5, 50.5).plot()





