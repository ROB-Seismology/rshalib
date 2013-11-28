"""
:mod:`rshalib.source.smoothed_seismicity` exports :class:`SmoothedSeismicity`
"""


#import matplotlib
import numpy as np
import ogr
#import osr

from scipy.stats import norm

#from openquake.hazardlib.geo.geodetic import geodetic_distance, point_at

from eqcatalog.calcGR import calcGR_Weichert
from eqcatalog.source_models import read_source_model, rob_source_models_dict
from ..geo import Point
from ..mfd import EvenlyDiscretizedMFD, TruncatedGRMFD
from ..source import PointSource, SourceModel
#from mapping.Basemap.LayeredBasemap import *
#from mapping.Basemap.cm.norm import PiecewiseLinearNorm
#from mapping.geo.coordtrans import wgs84, get_utm_spec, get_utm_srs


from ..site import SHASiteModel


#class Grid(object):
#	"""
#	"""
	
#	def __init__(self, region, dlon, dlat):
#		"""
#		"""
#		assert (region[1] - region[0]) / dlon == round((region[1] - region[0]) / dlon)
#		assert (region[3] - region[2]) / dlat == round((region[3] - region[2]) / dlat)
#		self.dlon = dlon
#		self.dlat = dlat
#		self.slons = sequence(region[0] + self.dlon / 2., region[1], self.dlon)
#		self.slats = sequence(region[2] + self.dlat / 2., region[3], self.dlat)
#		self.region = (
#			region[0],
#			self.slons[-1] + self.dlon / 2.,
#			region[2],
#			self.slats[-1] + self.dlat / 2.
#			)
#		self.shape = (len(self.slats), len(self.slons))
#		grid = np.dstack(np.meshgrid(self.slons, self.slats[::-1]))
#		self.lons = grid[:,:,0]
#		self.lats = grid[:,:,1]
	
#	def __len__(self):
#		"""
#		:return:
#			int, len of grid sites
#		"""
#		return np.prod(self.shape)
	
#	def __iter__(self):
#		"""
#		:return:
#			iterable over (lon, lat) of grid sites
#		"""
#		for i in np.ndindex(self.shape):
#			yield (self.lons[i], self.lats[i])
	
#	def __getitem__(self, i):
#		"""
#		:param i:
#			index of grid site
		
#		:return:
#			(float, float) tuple, (lon, lat) of site
#		"""
#		return self.lons[i], self.lats[i]
	
#	def get_distances(self, lon, lat):
#		"""
#		Get distances of grid sites to site.
		
#		:param lon:
#			float, lon of site
#		:param lat:
#			float, lat of site
		
#		:return:
#			2d np.array (~ self.shape), distances in km
#		"""
#		return geodetic_distance(self.lons, self.lats, lon, lat)
	
#	def get_index(self, lon, lat):
#		"""
#		Get index of grid site.
		
#		:param lon:
#			float, lon of site
#		:param lat:
#			float, lat of site
		
#		:return:
#			(int, int) tuple
#		"""
#		return np.unravel_index(self.get_distances(lon, lat).argmin(), self.shape)
	
#	def get_area(self):
#		"""
#		Get area of region.
		
#		:return:
#			float, area in square km
#		"""
#		wgs84 = osr.SpatialReference()
#		wgs84.SetWellKnownGeogCS("WGS84")
#		ring = ogr.Geometry(ogr.wkbLinearRing)
#		poly = ogr.Geometry(ogr.wkbPolygon)
#		w, e, s, n = self.region
#		ring.AddPoint(w, s)
#		ring.AddPoint(e, s)
#		ring.AddPoint(e, n)
#		ring.AddPoint(w, n)
#		ring.AddPoint(w, s)
#		poly.AssignSpatialReference(wgs84)
#		poly.AddGeometry(ring)
#		centroid = poly.Centroid()
#		coordTrans = osr.CoordinateTransformation(wgs84, get_utm_srs(get_utm_spec(centroid.GetX(), centroid.GetY())))
#		poly.Transform(coordTrans)
#		area = poly.GetArea() / 1E6
#		return area


class SmoothedSeismicity(object):
	"""
	"""
	
	def __init__(self, grid_outline, grid_spacing, catalog, min_mag, dmag, smoothing_kernel, smoothing_number, smoothing_bandwidth, smoothing_limit):
		"""
		"""
		self.grid = SHASiteModel(
			grid_outline=grid_outline,
			grid_spacing=grid_spacing,
			)
		self.catalog = catalog
		self.min_mag = min_mag or self.catalog.completeness.min_mag
		self.dmag = dmag
		self._set_mag_bins()
		self.shape = (
			len(self.grid.slats),
			len(self.grid.slons),
			len(self.mag_bins)
			)
		self.smoothing_kernel = smoothing_kernel
		self.smoothing_number = smoothing_number
		self.smoothing_bandwidth = smoothing_bandwidth
		self.smoothing_limit = smoothing_limit
		
		self._set_smoothing_bandwidths()
#		self._validate_smoothed_region()
		self._smooth()
		self._set_cum_occs()
		self._set_inc_occ_rates()
		self._set_cum_occ_rates()
		self._set_trt_data()
	
	def _set_mag_bins(self):
		"""
		"""
		self.mag_bins = sequence(self.min_mag, self.catalog.mags.max(), self.dmag)
	
	def _set_smoothing_bandwidths(self):
		"""
		"""
		if self.smoothing_number:
			self.smoothing_bandwidths = np.zeros(len(self.catalog))
			for i in xrange(len(self.catalog)):
				lon = self.catalog.lons[i]
				lat = self.catalog.lats[i]
				distances = sorted(self.catalog.get_epicentral_distances(lon, lat))
				smoothing_bandwidth = distances[self.smoothing_number]
				smoothing_bandwidth = max([smoothing_bandwidth, self.smoothing_bandwidth])
				self.smoothing_bandwidths[i] = smoothing_bandwidth
		else:
			self.smoothing_bandwidths = None
	
##	def _validate_smoothed_region(self):
##		"""
##		"""
##		for i in xrange(len(self.catalog)):
##			lon = self.catalog.lons[i]
##			lat = self.catalog.lats[i]
##			if self.smoothing_number:
##				smoothing_distance = self.smoothing_distances[i]
##			else:
##				smoothing_distance = self.smoothing_distance
##			w = point_at(lon, lat, 270., smoothing_distance)[0]
##			e = point_at(lon, lat, 090., smoothing_distance)[0]
##			s = point_at(lon, lat, 180., smoothing_distance)[1]
##			n = point_at(lon, lat, 000., smoothing_distance)[1]
##			assert self.grid.region[0] < w or np.allclose(self.grid.region[0], w)
##			assert self.grid.region[1] > e or np.allclose(self.grid.region[1], e)
##			assert self.grid.region[2] < s or np.allclose(self.grid.region[2], s)
##			assert self.grid.region[3] > n or np.allclose(self.grid.region[3], n)
	
	def _smooth(self):
		"""
		"""
		if not self.smoothing_number:
			smoothing_bandwidth = self.smoothing_bandwidth
		self.inc_occs = np.zeros(self.shape)
		for i in xrange(len(self.catalog)):
			lon = self.catalog.lons[i]
			lat = self.catalog.lats[i]
			mag = self.catalog.mags[i]
			distances = self.grid.get_geographic_distances(lon, lat)
			if self.smoothing_number:
				smoothing_bandwidth = self.smoothing_bandwidths[i]
			if self.smoothing_limit:
				indices = np.where(distances <= smoothing_bandwidth * self.smoothing_limit)
			else:
				indices = np.where(distances)
			if self.smoothing_kernel == "uniform":
				inc_occ = np.ones_like(distances[indices])
			if self.smoothing_kernel == "reciprocal":
				inc_occ = 1. / distances[indices]
			if self.smoothing_kernel == "normal":
				inc_occ = norm.pdf(distances[indices], 0, smoothing_bandwidth)
			inc_occ /= inc_occ.sum()
			k = np.abs(self.mag_bins - mag).argmin()
			self.inc_occs[:,:,k][indices] += inc_occ
	
	def _set_cum_occs(self):
		"""
		:set cum_occs:
			3d np array (~ self.shape)
		"""
		self.cum_occs = np.add.accumulate(self.inc_occs[:,:,::-1], 2)[:,:,::-1]
		self.cum_occs[np.where(self.cum_occs[:,:,0] == 0)] = [np.nan] * len(self.mag_bins)
	
	def _set_inc_occ_rates(self):
		"""
		:set inc_occ_rates:
			3d np array (~ self.shape)
		"""
		self.inc_occ_rates = np.apply_along_axis(np.divide, 2, self.inc_occs, self.catalog.completeness.get_completeness_timespans(self.mag_bins, self.catalog.end_date))
	
	def _set_cum_occ_rates(self):
		"""
		:set cum_occ_rates:
			3d np array (~ self.shape)
		"""
		self.cum_occ_rates = np.add.accumulate(self.inc_occ_rates[:,:,::-1], 2)[:,:,::-1]
		self.cum_occ_rates[np.where(self.cum_occ_rates[:,:,0] == 0)] = [np.nan] * len(self.mag_bins)
	
#	def _set_trts(self):
#		"""
#		"""
#		self.trts = np.zeros(self.grid.shape)
#		self.trts.fill(np.nan)
#		for i in np.ndindex(self.grid.shape):
#			point = ogr.Geometry(ogr.wkbPoint)
#			point.SetPoint(0, self.grid.lons[i], self.grid.lats[i])
#			if area_source_model["RVRS"]['obj'].Contains(point):
#				self.trts[i] = 1
#			else:
#				self.trts[i] = 0
	
	def _set_trt_data(self):
		"""
		"""
		area_source_model = read_source_model("TwoZonev2", verbose=False)
		self.max_mags, self.b_vals = np.zeros(self.grid.shape), np.zeros(self.grid.shape)
		self.max_mags.fill(np.nan), self.b_vals.fill(np.nan)
		for i in np.ndindex(self.grid.shape):
			point = ogr.Geometry(ogr.wkbPoint)
			point.SetPoint(0, self.grid.lons[i], self.grid.lats[i])
			if area_source_model["RVRS"]['obj'].Contains(point):
				self.max_mags[i], self.b_vals[i] = 7.2, 0.942
			else:
				self.max_mags[i], self.b_vals[i] = 6.7, 0.948
	
	def get_a_val(self, i):
		"""
		"""
		inc_occs = np.trim_zeros(self.inc_occs[i], trim='b')
		if not np.allclose(inc_occs, 0.):
			completeness = self.catalog.completeness
			end_date = self.catalog.end_date
			mag_bins = sequence(self.min_mag, self.max_mags[i], self.dmag)
			zeros = np.zeros((len(mag_bins) - len(inc_occs)))
			inc_occs = np.concatenate((inc_occs, zeros))
			a_val, _, _, _ = calcGR_Weichert(mag_bins, inc_occs, completeness, end_date, self.b_vals[i], False)
			return a_val
		else:
			return np.nan
	
	def get_a_vals(self):
		"""
		Get a value for each grid site.
		
		:return:
			2d np array (~ self.grid.shape)
		"""
		a_vals = np.zeros(self.grid.shape)
		a_vals.fill(np.nan)
		for i in np.ndindex(self.grid.shape):
			a_vals[i] = self.get_a_val(i)
		return a_vals
	
	def get_mfd_obs(self, i):
		"""
		Get observed (evenly discretized) mfd for site.
		
		:param i:
			??, index of site
		
		:return:
			instance of :class:`rshalib.mfd.EvenlyDiscretizedMFD` or None
		"""
		inc_occ_rates = np.trim_zeros(self.inc_occ_rates[i], trim='b')
		if not np.allclose(inc_occ_rates, 0.):
			return EvenlyDiscretizedMFD(self.min_mag + self.dmag / 2., self.dmag, inc_occ_rates)
		else:
			return None
	
	def get_mfd_est(self, i):
		"""
		Get estimated (truncated gr) mfd for site.
		
		:param i:
			??, index of site
		
		:return:
			instance of :class:`rshalib.mfd.TruncatedGRMFD` or None
		"""
		a_val = self.get_a_val(i)
		if not np.isnan(a_val):
			return TruncatedGRMFD(self.min_mag, self.max_mags[i], self.dmag, a_val, self.b_vals[i])
		else:
			return None
	
#	def get_mfd_johnston(self, max_mag, region="europe"):
#		"""
#		"""
#		return TruncatedGRMFD.construct_Johnston1994MFD(self.min_mag, max_mag, self.dmag, self.grid.get_area(), region)
	
	def get_total_mfd_obs(self):
		"""
		Get total obs mfd (sum of all obs mfds).
		
		:return:
			instance of :class:`rshalib.mfd.EvenlyDiscretizedMFD` or None
		"""
		total_mfd = None
		for i in np.ndindex(self.grid.shape):
			mfd_obs = self.get_mfd_obs(i)
			if mfd_obs:
				if total_mfd:
					total_mfd = mfd_obs + total_mfd
				else:
					total_mfd = mfd_obs
		return total_mfd
	
	def get_total_mfd_est(self):
		"""
		Get total est mfd (sum of all est mfds).
		
		:return:
			instance of :class:`rshalib.mfd.EvenlyDiscretizedMFD` or None
		"""
		total_mfd = None
		for i in np.ndindex(self.grid.shape):
			mfd_est = self.get_mfd_est(i)
			if mfd_est:
				if total_mfd:
					total_mfd = mfd_est + total_mfd
				else:
					total_mfd = mfd_est
		return total_mfd
	
	def get_smrs_obs(self):
		"""
		Get seismic moment rates obs.
		
		:return:
			2d np array (~ self.grid.shape)
		"""
		smrs_obs = np.zeros(self.grid.shape)
		for i in np.ndindex(self.grid.shape):
			mfd_obs = self.get_mfd_obs(i)
			if mfd_obs:
				smrs_obs[i] = mfd_obs._get_total_moment_rate()
		return smrs_obs
	
	def get_smrs_est(self):
		"""
		Get seismic moment rates est.
		
		2d np array (~ self.grid.shape)
		"""
		smrs_est = np.zeros(self.grid.shape)
		for i in np.ndindex(self.grid.shape):
			mfd_est = self.get_mfd_est(i)
			if mfd_est:
				smrs_est[i] = mfd_est._get_total_moment_rate()
		return smrs_est
	
	def to_source_model(self, source_model_name, trt, rms, msr, rar, usd, lsd, npd, hdd):
		"""
		Get source model from smoothed seismicity.
		
		:return:
			instance of :class:`rshalib.source.SourceModel`
		"""
		point_sources = []
		for i in np.ndindex(self.grid.shape):
			mfd = self.get_mfd_est(i)
			if mfd:
				lon = self.grid.lons[i]
				lat = self.grid.lats[i]
				id = str(i)
				name = str((lon, lat))
				point = Point(lon, lat)
				point_sources.append(PointSource(id, name, trt, mfd, rms, msr, rar, usd, lsd, point, npd, hdd))
		return SourceModel(source_model_name, point_sources)
	
#	def plot_map(self, data=None, grid=None, catalog=None, builtin=True, region=None, title="", label="", fig_filespec=""):
#		"""
#		"""
#		map_layers = []
#		if data != None:
#			dd = GridData(self.grid.lons, self.grid.lats[::-1], data[::-1])
##			vmin = np.nanmin(data)
##			vmax = np.nanmax(data)
##			norm = PiecewiseLinearNorm([-2.0, +2.0])
#			ds = GridStyle(
##				color_map_theme=ThematicStyleColormap(colorbar_style=ColorbarStyle(title=label, ticks=[-2.0, -1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0])),
#				color_map_theme=ThematicStyleColormap(colorbar_style=ColorbarStyle(title=label, ticks=sequence(-6.2, -0.6, 0.4))),
##				color_map_theme=ThematicStyleColormap(colorbar_style=ColorbarStyle(title=label)),
#				color_gradient="discontinuous",
##				contour_levels=[-2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
##				contour_levels=[-3.0, -2.8, -2.6, -2.4, -2.2, -2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
#				contour_levels=sequence(-6.2, -0.6, 0.2)
##				contour_levels=sequence(-4., 1.6, 0.2)
#				)
#			map_layers.append(MapLayer(dd, ds))
#		if grid:
#			gd = MultiPointData(self.grid.lons, self.grid.lats)
#			gs = PointStyle(shape='.', size=5, line_color='k', fill_color='k')
#			map_layers.append(MapLayer(gd, gs))
#		if catalog:
#			cd = MultiPointData(self.catalog.lons, self.catalog.lats)
#			cs = PointStyle(shape='.', size=5, line_color='r', fill_color='r')
#			map_layers.append(MapLayer(cd, cs))
#		if builtin:
#			coastline_layer = MapLayer(style=LineStyle(), data=BuiltinData("coastlines"))
#			countries_layer = MapLayer(style=LineStyle(), data=BuiltinData("countries" ))
#			map_layers.extend([coastline_layer, countries_layer])
#		region = region or self.grid.region
#		map = LayeredBasemap(layers=map_layers, region=region, projection="merc", resolution="i", title=title, legend_style=LegendStyle())
#		map.plot(
#			fig_filespec=fig_filespec
#			)


def sequence(min, max, step):
	"""
	Get sequence. sequence[-1] <= max.
	
	:param min:
		float, minimum of sequence
	:param max:
		float, maximum of sequence
	:param step:
		float, step of sequence
	
	:return:
		1d np array
	"""
	sequence = []
	while min < max or np.allclose(min, max):
		sequence.append(min)
		min += step
		if np.allclose(min, 0.):
			min = 0.
	return np.array(sequence)


#if __name__ == "__main__":
#	"""
#	"""
#	pass

