# -*- coding: utf-8 -*-
# pylint: disable=W0142, W0312, C0103, R0913
"""
Blueprint for classes representing hazard results of both OpenQuake and CRISIS
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import int

try:
	## Python 2
	basestring
except:
	## Python 3
	basestring = str


### imports
import os, sys
from decimal import Decimal
import numpy as np

from scipy.stats import mstats, scoreatpercentile
import matplotlib
import pylab

from ..poisson import poisson_conv
from ..nrml import ns
from ..nrml.common import *
from ..site import GenericSite
from ..utils import interpolate, logrange, wquantiles
from ..pmf import NumericPMF

from .plot import plot_hazard_curve, plot_hazard_spectrum, plot_histogram
from .base_array import *
from .hc_base import *

from .response_spectrum import ResponseSpectrum


# TODO: unit names should be the same as robspy !!!
# TODO: change order IMT, intensities, intensity_unit --> intensities, intensity_unit, IMT


common_plot_docstring = """
			fig_filespec: full path to ouptut image. If None, graph will be plotted on screen
				(default: None)
			title: title to appear above the graph (default: None, will generate title)
			want_recurrence: boolean indicating whether or not to plot recurrence interval
				instead of exceedance rate in the Y axis (default: False)
			want_poe: boolean indicating whether or not to plot probability of exceedance
				instead of exceedance rate in the Y axis (default: False)
			interpol_rp: return period for which to interpolate intensity
				(one value or a list of values for each dataset). Will be plotted
				with a dashed line for each dataset (default: None, i.e. no interpolation)
			interpol_prob: exceedance probability for which to interpolate intensity
				(one value or list of values for each dataset). Will be plotted
				with a dashed line for each dataset  (default: None, i.e. no interpolation)
			interpol_rp_range: return period range for which to interpolate intensity
				([min return period, max return period] list). Will be plotted
				with a grey area for first dataset only (default: None, i.e. no interpolation)
			amax: maximum intensity to plot in X axis (default: None)
			rp_max: maximum return period to plot in Y axis (default: 1E+07)
			legend_location: location of legend (matplotlib location code) (default=0):
				"best" 	0
				"upper right" 	1
				"upper left" 	2
				"lower left" 	3
				"lower right" 	4
				"right" 	5
				"center left" 	6
				"center right" 	7
				"lower center" 	8
				"upper center" 	9
				"center" 	10
			lang: language to use for labels: en=English, nl=Dutch (default: en)
            dpi: Int, image resolution in dots per inch (default: 300)
"""



class HazardMap(HazardResult, HazardField):
	"""
	Class representing a hazard map or a ground-motion field
	One hazard map, corresponds to 1 OQ file
	sites: 1-D list [i] with (lon, lat) tuples of all sites
	intensities: 1-D array [i]
	"""
	def __init__(self, model_name, filespec, sites, period, IMT, intensities, intensity_unit="", timespan=50, poe=None, return_period=None, vs30s=None):
		if return_period:
			hazard_values = ExceedanceRateArray([1./return_period])
		elif poe:
			hazard_values = ProbabilityArray([poe])
		HazardResult.__init__(self, hazard_values, timespan=timespan, IMT=IMT, intensities=intensities, intensity_unit=intensity_unit)
		HazardField.__init__(self, sites)
		self.model_name = model_name
		self.filespec = filespec
		self.period = period
		self.vs30s = as_array(vs30s)

	def __repr__(self):
		return "<HazardMap: %d sites, %d intensities, period=%s s>" % (self.num_sites, self.num_intensities, self.period)

	def __iter__(self):
		self._current_index = 0
		return self

	def next(self):
		try:
			site = self.sites[self._current_index]
		except:
			raise StopIteration
		else:
			self._current_index += 1
			return self.getHazardValue(self._current_index-1)

	def __getitem__(self, site_spec):
		return self.getHazardValue(site_spec)

	def getHazardValue(self, site_spec=0, intensity_unit="g"):
		site_index = self.site_index(site_spec)
		try:
			site = self.sites[site_index]
		except:
			raise IndexError("Site index %s out of range" % site_index)
		else:
			return self.get_intensities(intensity_unit)[site_index]

	def min(self, intensity_unit="g"):
		"""
		Return minimum intensity

		:param intensity_unit:
			string, intensity unit to scale result,
			either "g", "mg", "ms2", "gal" or "cms2" (default: "g")
		"""
		return self.get_intensities(intensity_unit).min()

	def max(self, intensity_unit="g"):
		"""
		Return maximum intensity

		:param intensity_unit:
			string, intensity unit to scale result,
			either "g", "mg", "ms2", "gal" or "cms2" (default: "g")
		"""
		return self.get_intensities(intensity_unit).max()

	def mean(self, intensity_unit="g"):
		"""
		Return mean intensity in the map

		:param intensity_unit:
			string, intensity unit to scale result,
			either "g", "mg", "ms2", "gal" or "cms2" (default: "g")
		"""
		return self.get_intensities(intensity_unit).mean()

	def median(self, intensity_unit="g"):
		"""
		Return median intensity in the map

		:param intensity_unit:
			string, intensity unit to scale result,
			either "g", "mg", "ms2", "gal" or "cms2" (default: "g")
		"""
		return np.median(self.get_intensities(intensity_unit))

	def scoreatpercentile(self, perc, intensity_unit="g"):
		"""
		Return intensity corresponding to given percentile

		:param perc:
			float, percentile in range [0, 100]

		:param intensity_unit:
			string, intensity unit to scale result,
			either "g", "mg", "ms2", "gal" or "cms2" (default: "g")
		"""
		return scoreatpercentile(self.get_intensities(intensity_unit), perc)

	def argmin(self):
		"""
		Return site index corresponding to minimum intensity value
		"""
		return self.intensities.argmin()

	def argmax(self):
		"""
		Return site index corresponding to maximum intensity value
		"""
		return self.intensities.argmax()

	@property
	def poe(self):
		return self.poes[0]

	@property
	def return_period(self):
		return self.return_periods[0]

	@property
	def exceedance_rate(self):
		return self.exceedance_rates[0]

	def trim(self, lonmin=None, lonmax=None, latmin=None, latmax=None):
		if lonmin is None:
			lonmin = self.lonmin()
		if lonmax is None:
			lonmax = self.lonmax()
		if latmin is None:
			latmin = self.latmin()
		if latmax is None:
			latmax = self.latmax()
		site_indexes, sites, vs30s = [], [], []
		longitudes, latitudes = self.longitudes, self.latitudes
		for i in range(self.num_sites):
			if lonmin <= longitudes[i] <= lonmax and latmin <= latitudes[i] <= latmax:
				site_indexes.append(i)
				sites.append(self.sites[i])
				if self.vs30s is not None:
					vs30s.append(self.vs30s[i])

		model_name = self.model_name
		filespec = self.filespec
		period = self.period
		IMT = self.IMT
		intensities = self.intensities[site_indexes]
		intensity_unit = self.intensity_unit
		timespan = self.timespan
		poe = self.poe
		return_period = self.return_period

		hm = HazardMap(model_name, filespec, sites, period, IMT, intensities, intensity_unit, timespan, poe, return_period, vs30s)
		return hm

	def interpolate_map(self, grid_extent=(None, None, None, None), num_grid_cells=50, method="cubic"):
		"""
		Interpolate hazard map on a regular (lon, lat) grid

		:param grid_extent:
			(lonmin, lonmax, latmin, latmax) tuple of floats
		:param num_grid_cells:
			Integer or tuple, number of grid cells in X and Y direction
		:param method:
			Str, interpolation method supported by griddata (either
			"linear", "nearest" or "cubic") (default: "cubic")

		:return:
			instance of :class:`HazardMap`
		"""
		grid_lons, grid_lats = self.meshgrid(grid_extent, num_grid_cells)
		grid_intensities = self.get_site_intensities(grid_lons, grid_lats, method)
		intensities = grid_intensities.flatten()

		model_name = self.model_name + " (interpolated)"
		filespec = self.filespec
		sites = list(zip(grid_lons.flatten(), grid_lats.flatten()))
		period = self.period
		IMT = self.IMT
		intensity_unit = self.intensity_unit
		timespan = self.timespan
		poe = self.poe
		return_period = self.return_period
		vs30s = None

		return HazardMap(model_name, filespec, sites, period, IMT, intensities, intensity_unit, timespan, poe, return_period, vs30s)

	def get_residual_map(self, other_map, grid_extent=(None, None, None, None), num_grid_cells=50, interpol_method="linear", abs=True):
		"""
		Compute difference with another hazard map. If sites are different,
		the maps will be interpolated on a regular (lon, lat) grid

		:param other_map:
			instance of :class:`HazardMap`
		:param grid_extent:
			(lonmin, lonmax, latmin, latmax) tuple of floats
		:param num_grid_cells:
			Integer or tuple, number of grid cells in X and Y direction
		:param interpol_method:
			Str, interpolation method supported by griddata (either
			"linear", "nearest" or "cubic") (default: "linear")
		:param abs:
			Bool, whether or not residual map values should be absolute (True)
			or in percentage relative to current map (False)
			(default: True)

		:return:
			instance of :class:`HazardMap`
		"""
		if self.sites == other_map.sites:
			residuals = self.intensities - other_map.intensities
			if not abs:
				residuals /= self.intensities
				residuals *= 100
			sites = self.sites
			vs30s = self.vs30s
		else:
			lonmin, lonmax, latmin, latmax = grid_extent
			lonmin = max(self.lonmin(), other_map.lonmin())
			lonmax = min(self.lonmax(), other_map.lonmax())
			latmin = max(self.latmin(), other_map.latmin())
			latmax = min(self.latmax(), other_map.latmax())
			grid_extent = (lonmin, lonmax, latmin, latmax)
			grid_lons, grid_lats = self.meshgrid(grid_extent, num_grid_cells)
			grid_intensities1 = self.get_site_intensities(grid_lons, grid_lats, interpol_method)
			grid_intensities2 = other_map.get_site_intensities(grid_lons, grid_lats, interpol_method)
			residuals = (grid_intensities1 - grid_intensities2).flatten()
			residuals[np.isnan(residuals)] = 0.
			if not abs:
				residuals /= grid_intensities1.flatten()
				residuals *= 100
			sites = list(zip(grid_lons.flatten(), grid_lats.flatten()))
			vs30s = None

		model_name = "Residuals (%s - %s)" % (self.model_name, other_map.model_name)
		filespec = None
		period = self.period
		IMT = self.IMT
		intensity_unit = self.intensity_unit

		if self.timespan == other_map.timespan:
			timespan = self.timespan
		else:
			timespan = np.nan
		if self.poe == other_map.poe:
			poe = self.poe
		else:
			poe = np.nan
		if round(self.return_period) == round(other_map.return_period):
			return_period = self.return_period
		else:
			return_period = np.nan

		return HazardMap(model_name, filespec, sites, period, IMT, residuals, intensity_unit, timespan, poe, return_period, vs30s)

	def extract_partial_map(self, sites, interpolate=False):
		"""
		Extract partial map

		:param sites:
			list with instances of GenericSite or (lon, lat) tuples
		:param interpolate:
			bool or string
			If False, sites must match exactly, otherwise
			the intensities will be interpolated
			If string, should correspond to interpolation method
			("nearest", "linear" or "cubic")
			If True, cubic interpolation is applied
			(default: False)

		:return:
			instance of :class:`HazardMap`
		"""
		if interpolate:
			if isinstance(sites[0], GenericSite):
				lons = [site.lon for site in sites]
				lats = [site.lat for site in sites]
			else:
				## (lon, lat) tuples
				lons, lats = zip(*sites)
			if isinstance(interpolate, basestring):
				method = interpolate
			else:
				method = "cubic"
			intensities = self.get_site_intensities(lons, lats, method=method)
			vs30s = None
		else:
			site_idxs = self.get_site_indexes(sites)
			#site_idxs = [self.site_index(site) for site in sites]
			intensities = self.intensities[site_idxs]
			if self.vs30s is not None:
				vs30s = self.vs30s[site_idxs]
			else:
				vs30s = self.vs30s

		model_name = self.model_name + " (partial)"
		filespec = self.filespec
		period = self.period
		IMT = self.IMT
		intensity_unit = self.intensity_unit
		timespan = self.timespan
		poe = self.poe
		return_period = self.return_period

		return HazardMap(model_name, filespec, sites, period, IMT, intensities,
						intensity_unit, timespan, poe, return_period, vs30s)

	def export_VM(self, base_filespec, num_cells=100):
		"""
		Export hazard map to a Vertical Mapper grid
		"""
		import mapping.VMPython as vm
		if self.IMT in ("PGA", "PGV", "PGV"):
			imt_label = self.IMT
		else:
			imt_label = "T=%.3fs" % self.period
		rp_label = "%.3Gyr" % self.return_period
		grd_filespec = os.path.splitext(base_filespec)[0] + "_%s_%s" % (imt_label, rp_label)

		(lonmin, lonmax, dlon), (latmin, latmax, dlat) = self.get_grid_properties()
		assert abs(dlon - dlat) < 1E-12
		cell_size = dlon
		zmin, zmax = self.min(), self.max()
		vmgrd = vm.CreateNumericGrid(grd_filespec, lonmin, lonmax, latmin, latmax, cell_size, zmin, zmax, Zdescription=self.intensity_unit)
		print("Created VM grid %s" % grd_filespec)

		intensity_grid = self.get_grid_intensities(num_cells)
		nrows = intensity_grid.shape[0]
		for rownr, row in enumerate(intensity_grid):
			vmgrd.WriteRow(row, (nrows-1)-rownr)
		vmgrd.Close()

	def export_GeoTiff(self, base_filespec, num_cells=None, cell_size=None, interpol_method='cubic', intensity_unit='g', nodata_value=np.nan):
		"""
		Export hazard map to GeoTiff raster

		:param base_filespec:
			str, base output file specification, spectral period,
			return period and TIF extension will be appended to filename
		:param num_cells:
			int or tuple of ints, number of grid cells in X and Y direction
			If None, :param:`cell_size` must be set
			(default: None)
		:param cell_size:
			float or tuple of floats, cell size (in decimal degrees)
			in X and Y direction.
			If None, :param:`num_cells` must be set
			(default: None)
		:param interpol_method:
			str, interpolation method supported by griddata, one of
			"nearest", "linear" or "cubic"
			(default: "cubic")
		:param intensity_unit:
			string, intensity unit to scale result,
			either "g", "mg", "ms2", "gal" or "cms2" (default: "g")
		:param nodata_value:
			float, value to use for "no data"
			(default: np.nan)
		"""
		import gdal, osr

		assert num_cells or cell_size

		if self.IMT in ("PGA", "PGV", "PGV"):
			imt_label = self.IMT
		else:
			imt_label = "T=%.3fs" % self.period
		rp_label = "%.3Gyr" % self.return_period
		grd_filespec = os.path.splitext(base_filespec)[0] + "_%s_%s.TIF" % (imt_label, rp_label)

		if isinstance(num_cells, int):
			num_cells = (num_cells, num_cells)
		if isinstance(cell_size, int):
			cell_size = (cell_size, cell_size)

		lonmin, lonmax = self.lonmin(), self.lonmax()
		latmin, latmax = self.latmin(), self.latmax()

		if num_cells:
			nlons, nlats = num_cells
			cell_size_x = (lonmax - lonmin) / (num_cells[0] - 1)
			cell_size_y = (latmax - latmin) / (num_cells[1] - 1)
			cell_size = (cell_size_x, cell_size_y)
		else:
			lonmin = np.floor(lonmin / cell_size[0]) * cell_size[0]
			lonmax = np.ceil(lonmax / cell_size[0]) * cell_size[0]
			latmin = np.floor(latmin / cell_size[1]) * cell_size[1]
			latmax = np.ceil(latmax / cell_size[1]) * cell_size[1]
			nlons = int(round((lonmax - lonmin) / cell_size[0] + 1))
			nlats = int(round((latmax - latmin) / cell_size[1] + 1))
			num_cells = (nlons, nlats)

		extent = (lonmin, lonmax, latmin, latmax)
		#print(extent, cell_size, num_cells)

		intensities = self.get_grid_intensities(extent=extent, num_cells=num_cells,
							method=interpol_method, intensity_unit=intensity_unit,
							nodata_value=nodata_value)
		## Order of rows should be north to south, otherwise image is upside down
		intensities = intensities[::-1,:]

		driver = gdal.GetDriverByName("Gtiff")
		ds = driver.Create(grd_filespec, nlons, nlats, 1, gdal.GDT_Float32)
		## Affine transform takes 6 parameters:
		## top left x, cell size x, rotation, top left y, rotation, cell size y
		## Note that x, y coordinates refer to top left corner of top left pixel!
		## For north-up images, rotation coefficients are zero
		ds.SetGeoTransform((lonmin-cell_size[0]/2., cell_size[0], 0, latmax+cell_size[1]/2., 0, -cell_size[1]))
		srs = osr.SpatialReference()
		srs.SetWellKnownGeogCS("WGS84")
		ds.SetProjection(srs.ExportToWkt())
		band = ds.GetRasterBand(1)
		band.WriteArray(intensities.astype(np.float32))
		band.SetNoDataValue(nodata_value)
		band.ComputeStatistics()
		#band.SetStatistics(np.min(mag_grid), np.max(mag_grid), np.average(mag_grid), np.std(mag_grid))
		ds.FlushCache()

	def export_kml(self):
		# TODO!
		pass

	def get_plot(self, region=None, projection="merc", resolution="i", graticule_interval=(1., 1.),
				cmap="usgs", norm=None, contour_interval=None, amin=None, amax=None,
				num_grid_cells=100, plot_style="cont", contour_line_style="default",
				site_style="default", source_model="", source_model_style="default",
				countries_style="default", coastline_style="default", intensity_unit="",
				hide_sea=False, colorbar_style="default", show_legend=True,
				title=None, ax=None, **kwargs):
		"""
		Plot hazard map

		:param region:
			(west, east, south, north) tuple specifying rectangular region
			to plot in geographic coordinates
			(default: None)
		:param projection:
			string, map projection. See Basemap documentation
			(default: "merc")
		:param resolution:
			char, resolution of builtin shorelines / country borders:
			'c' (crude), 'l' (low), 'i' (intermediate), 'h' (high), 'f' (full)
			(default: 'i')
		:param graticule_interval:
			(dlon, dlat) tuple of floats, spacing of grid lines (meridians,
			parallels) to draw over the map
			(default: (1., 1.)
		:param cmap:
			str of matplotlib colormap specification: color map to be used
			for ground-motion values
			Some nice color maps are: jet, spectral, gist_rainbow_r,
				Spectral_r, usgs
			(default: "usgs")
		:param norm:
			instance of :class:`matplotlib.colors.Normalize` used to
			map a range of values to the range of colors in :param:`cmap`
			(default: None, will use norm corresponding to named color map
			or linear normalization between the minimum and maximum
			ground-motion values)
		:param contour_interval:
			float, ground-motion contour interval (default: None = auto)
		:param amin:
			float, minimum ground-motion level to color/contour
			(default: None)
		:param amax:
			float, maximum ground-motion level to color/contour
			(default: None)
		:param num_grid_cells:
			int or tuple, number of grid cells for interpolating
			intensity grid in X and Y direction
			(default: 100)
		:param plot_style:
			String, either "disc" for discrete or "cont" for continuous
			(default: "cont")
		:param contour_line_style:
			instance of :class:`LayeredBasemap.LineStyle`, defining style
			for plotting contour lines
			(default: "default")
		:param site_style:
			instance of :class:`LayeredBasemap.PointStyle`, defining style
			for plotting sites where hazard was computed
			(default: "default")
		:param source_model:
			str or instance of :class:`SourceModel`: name of known source
			model or SourceModel object to plot on top of hazard map
			(default: "")
		:param source_model_style:
			instance of :class:`LayeredBasemap.CompositeStyle`, defining
			style for plotting source model
			(default: "default")
		:param countries_style:
			instance of :class:`LayeredBasemap.LineStyle`, defining
			style for plotting country borders
			(default: "default")
		:param coastline_style:
			instance of :class:`LayeredBasemap.LineStyle`, defining
			style for plotting coastlines
			(default: "default")
		:param intensity_unit:
			str, unit in which ground-motion values need to be expressed
			(default: "")
		:param hide_sea:
			bool, whether or not hazard map should be masked over seas
			and oceans
			(default: False)
		:param colorbar_style:
			instance of :class:`LayeredBasemap.ColorbarStyle`, defining
			style for plotting color bar
			(default: "default")
		:param show_legend:
			bool, whether or not to show the legend for sources
			(default: True)
		:param title:
			str, map title. If empty string, no title will be plotted.
			If None, default title will be used
			(default: None)
		:param ax:
			matplotlib Axes instance
			(default: None)
		:param kwargs:
			additional keyword arguments to be passed to LayeredBasemap
			constructor

		:return:
			instance of :class:`LayeredBasemap.LayeredBasemap`, where
			additional layers may be added before final plotting.
		"""
		import mapping.layeredbasemap as lbm
		from .plot import get_intensity_unit_label

		## Construct default styles:
		if site_style == "default":
			site_style = lbm.PointStyle(shape="x", line_color="w", size=2.5)
		if source_model_style == "default":
			polygon_style = lbm.PolygonStyle(line_width=2, fill_color="none")
			line_style = lbm.LineStyle(line_width=3, line_color='purple')
			point_style = lbm.PointStyle(shape='*', fill_color='yellow', size=12)
			source_model_style = lbm.CompositeStyle(line_style=line_style, polygon_style=polygon_style, point_style=point_style)
		elif isinstance(source_model_style, lbm.PolygonStyle):
			polygon_style = source_model_style
			source_model_style = lbm.CompositeStyle(polygon_style=polygon_style, line_style=polygon_style.to_line_style())
		if countries_style == "default":
			countries_style = lbm.LineStyle(line_width=2, line_color="w")
		if coastline_style == "default":
			coastline_style = lbm.LineStyle(line_width=2, line_color="w")
		if contour_line_style == "default":
			contour_label_style = lbm.TextStyle(font_size=10, background_color=(1,1,1,0.5))
			contour_line_style = lbm.LineStyle(label_style=contour_label_style)

		## Prepare intensity grid and contour levels
		longitudes, latitudes = self.longitudes, self.latitudes
		if num_grid_cells is None:
			## Assume site lons and lats already define a meshed grid
			lons = np.sort(np.unique(self.longitudes))
			lats = np.sort(np.unique(self.latitudes))
			grid_lons, grid_lats = np.meshgrid(lons, lats, copy=False)
			interpol_method = "nearest"
		else:
			grid_lons, grid_lats = self.meshgrid(num_cells=num_grid_cells)
			interpol_method = "cubic"
		if not intensity_unit:
			intensity_unit = self.intensity_unit
		intensity_grid = self.get_site_intensities(grid_lons, grid_lats,
						method=interpol_method, intensity_unit=intensity_unit)

		# TODO: option to use contour levels as defined in norm
		if not contour_interval:
			arange = self.max() - self.min()
			candidates = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 1.0])
			try:
				index = np.where(arange / candidates <= 10)[0][0]
			except IndexError:
				index = 0
			contour_interval = candidates[index]
		else:
			contour_interval = float(contour_interval)

		if amin is None:
			amin = np.floor(self.min(intensity_unit) / contour_interval) * contour_interval
		if amax is None:
			amax = np.ceil(self.max(intensity_unit) / contour_interval) * contour_interval

		if contour_interval is not None:
			contour_levels = np.arange(amin, amax+contour_interval, contour_interval)
			## Sometimes, there is an empty contour interval at the end
			if len(contour_levels) > 1 and contour_levels[-2] > self.max():
				contour_levels = contour_levels[:-1]
		elif contour_interval == 0:
			contour_levels = []
		else:
			contour_levels = None

		## Compute map limits
		if not region:
			llcrnrlon, llcrnrlat = min(longitudes), min(latitudes)
			urcrnrlon, urcrnrlat = max(longitudes), max(latitudes)
			region = (llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat)

		## Color map and norm
		if isinstance(cmap, basestring):
			if cmap.lower() in ("usgs", "share", "gshap"):
				cmap_name = cmap
				cmap = lbm.cm.get_cmap("hazard", cmap_name)
				if norm is None:
					norm = lbm.cm.get_norm("hazard", cmap_name)
			else:
				cmap = matplotlib.cm.get_cmap(cmap)

		if isinstance(norm, basestring):
			if norm.lower() in ("usgs", "share", "gshap"):
				norm = lbm.cm.get_norm("hazard", norm)

		## Intensity grid
		if self.IMT in ("SA", "PGA"):
			label_format="%.2f"
			if self.period == 0:
				imt_label = "PGA"
			else:
				imt_label = "%s (%s s)" % (self.IMT, self.period)
		else:
			imt_label = self.IMT
			label_format="%s"
		intensity_unit_label = get_intensity_unit_label(intensity_unit)
		cbar_label = imt_label
		if intensity_unit:
			cbar_label += ' (%s)' % intensity_unit_label

		map_layers = []
		ticks = contour_levels
		if not (ticks is None or ticks == []):
			ticks = ticks[ticks <= norm.vmax]
		if colorbar_style == "default":
			colorbar_style = lbm.ColorbarStyle(location="bottom", format=label_format, ticks=ticks, title=cbar_label)
		color_map_theme = lbm.ThematicStyleColormap(color_map=cmap, norm=norm, vmin=amin, vmax=amax, colorbar_style=colorbar_style)
		color_gradient = {"cont": "continuous", "disc": "discontinuous"}[plot_style]
		grid_style = lbm.GridStyle(color_map_theme=color_map_theme, color_gradient=color_gradient, line_style=contour_line_style, contour_levels=contour_levels)
		grid_data = lbm.MeshGridData(grid_lons, grid_lats, intensity_grid)
		layer = lbm.MapLayer(grid_data, grid_style, name="intensity_grid")
		map_layers.append(layer)

		## Intensity data points
		if site_style:
			site_data = lbm.MultiPointData(longitudes, latitudes)
			map_layers.append(lbm.MapLayer(site_data, site_style, name="intensity_points"))

		if hide_sea:
			continent_style = lbm.FocmecStyle(fill_color=(1, 1, 1, 0), bg_color=(1, 1, 1, 1), line_width=0, line_color="none")
			data = lbm.BuiltinData("continents")
			map_layers.append(lbm.MapLayer(data, continent_style, name="ocean"))

		## Coastlines and national boundaries
		if coastline_style:
			map_layers.append(lbm.MapLayer(lbm.BuiltinData("coastlines"), coastline_style, name="coastlines"))
		if countries_style:
			map_layers.append(lbm.MapLayer(lbm.BuiltinData("countries"), countries_style, name="countries"))

		## Source model
		if source_model and source_model_style:
			from ..source import (SourceModel, PointSource, AreaSource,
							SimpleFaultSource, CharacteristicFaultSource)
			legend_label = {}
			#legend_label["polygons"] = "Area sources"
			#legend_label["lines"] = "Fault sources"
			#legend_label["points"] = "Point sources"

			if isinstance(source_model, basestring):
				from eqcatalog.source_models import rob_source_models_dict
				gis_filespec = rob_source_models_dict[source_model].gis_filespec
				sm_data = lbm.GisData(gis_filespec)
			elif isinstance(source_model, SourceModel):
				# TODO: add ComplexFaultSource
				# TODO: how to handle CharacteristicSource (doesn't have get_polygon method)?
				polygon_data = lbm.MultiPolygonData([], [])
				line_data = lbm.MultiLineData([], [])
				point_data = lbm.MultiPointData([], [])
				for source in source_model:
					if isinstance(source, AreaSource):
						polygon_data.append(lbm.PolygonData(source.longitudes, source.latitudes))
						if not "polygons" in legend_label and show_legend:
							legend_label["polygons"] = "Area sources"
					elif isinstance(source, (SimpleFaultSource, CharacteristicFaultSource)):
						pg = source.get_polygon()
						polygon_data.append(lbm.PolygonData(pg.lons, pg.lats))
						fault_trace = source.fault_trace
						line_data.append(lbm.LineData(fault_trace.lons, fault_trace.lats))
						if not "lines" in legend_label and show_legend:
							legend_label["lines"] = "Fault sources"
					elif isinstance(source, PointSource):
						point_data.append(lbm.PointData(source.location.longitude, source.location.latitude))
						if not "points" in legend_label and show_legend:
							legend_label["points"] = "Point sources"
					else:
						print("Warning: Skipped plotting source %s, source type not supported" % source.source_id)
				sm_data = lbm.CompositeData(lines=line_data, polygons=polygon_data,
											points=point_data)
			sm_style = source_model_style
			map_layers.append(lbm.MapLayer(sm_data, sm_style, legend_label=legend_label, name="source_model"))

		## Title
		if title is None:
			title = "%s\n%.4G yr return period" % (self.model_name, self.return_period)

		if source_model:
			legend_style = lbm.LegendStyle(location=0)
		else:
			legend_style = None

		graticule_style = lbm.GraticuleStyle(annot_axes="SE")
		map = lbm.LayeredBasemap(map_layers, title, projection, region=region, graticule_interval=graticule_interval, resolution=resolution, graticule_style=graticule_style, legend_style=legend_style, ax=ax, **kwargs)
		return map


class HazardMapSet(HazardResult, HazardField):
	"""
	Class representing a set of hazard maps or ground-motion fields for different
	return periods.
	Corresponds to 1 CRISIS MAP file containing 1 spectral period.
	sites: 1-D list [i] with (lon, lat) tuples of all sites
	intensities: 2-D array [p, i]
	"""
	def __init__(self, model_name, filespecs, sites, period, IMT, intensities, intensity_unit="g", timespan=50, poes=None, return_periods=None, vs30s=None):
		if not return_periods in (None, []):
			hazard_values = ExceedanceRateArray(1./as_array(return_periods))
		elif poes:
			hazard_values = ProbabilityArray(as_array(poes))
		HazardResult.__init__(self, hazard_values, timespan=timespan, IMT=IMT, intensities=intensities, intensity_unit=intensity_unit)
		HazardField.__init__(self, sites)
		self.model_name = model_name
		if len(filespecs) == 1:
			filespecs *= len(return_periods)
		self.filespecs = filespecs
		self.period = period
		self.vs30s = as_array(vs30s)

	def __iter__(self):
		self._current_index = 0
		return self

	def next(self):
		try:
			return_period = self.return_periods[self._current_index]
		except:
			raise StopIteration
		else:
			self._current_index += 1
			return self.getHazardMap(index=self._current_index-1)

	def __getitem__(self, index):
		return self.getHazardMap(index=index)

	def __len__(self):
		return len(self.return_periods)

	@classmethod
	def from_hazard_maps(self, hazard_maps, model_name=""):
		"""
		Construct from a list of hazard maps

		:param hazard_maps:
			list with instances of :class:`HazardMap`

		:return:
			instance of :class:`HazardMapSet`
		"""
		filespecs = [map.filespec for map in hazard_maps]
		hm0 = hazard_maps[0]
		sites = hm0.sites
		period = hm0.period
		IMT = hm0.IMT
		intensities = np.zeros((len(hazard_maps), len(sites)))
		intensity_unit = hm0.intensity_unit
		timespan = hm0.timespan
		poes = []
		return_periods = []
		vs30s = hm0.vs30s
		for i, hm in enumerate(hazard_maps):
			assert hm.sites == hm0.sites
			assert hm.intensity_unit == hm0.intensity_unit
			#assert (hm.vs30s == hm0.vs30s).all()
			intensities[i] = hm.intensities
			poes.append(hm.poe)
			return_periods.append(hm.return_period)

		return HazardMapSet(model_name, filespecs, sites, period, IMT, intensities, intensity_unit=intensity_unit, timespan=timespan, poes=poes, return_periods=return_periods, vs30s=vs30s)

	def getHazardMap(self, index=None, poe=None, return_period=None):
		"""
		Return a particular hazard map
		Parameters:
			index: index of hazard map in set (default: None)
			poe: probability of exceedance of hazard map (default: None)
			return_period: return period of hazard map (default: None)
		Return value:
			HazardMap object
		Notes:
			One of index, poe or return_period must be specified.
		"""
		if (index, poe, return_period) == (None, None, None):
			raise Exception("One of index, poe or return_period must be specified!")
		if index is None:
			if poe is not None:
				index = np.where(np.abs(self.poes - poe) < 1E-6)[0]
				if len(index) == 0:
					raise ValueError("No hazard map for poE=%s" % poe)
				else:
					index = index[0]
			elif return_period is not None:
				index = np.where(np.abs(self.return_periods - return_period) < 1)[0]
				if len(index) == 0:
					raise ValueError("No hazard map for return period=%s" % return_period)
				else:
					index = index[0]

		try:
			return_period = self.return_periods[index]
		except:
			raise IndexError("Index %s out of range" % index)
		else:
			filespec = self.filespecs[index]
			intensities = self.intensities[index]
			return HazardMap(self.model_name, filespec, self.sites, self.period, self.IMT, intensities, self.intensity_unit, self.timespan, return_period=return_period, vs30s=self.vs30s)

	# TODO: the following methods are perhaps better suited in a HazardMapTree class
	def get_max_hazard_map(self):
		"""
		Get hazard map with for each site the maximum value of all hazard maps in the set.

		:returns:
			instance of :class:`HazardMap`
		"""
		intensities = np.amax(self.intensities, axis=0)
		if len(set(self.return_periods)) == 1:
			return_period = self.return_periods[0]
		else:
			return_period = 1
		model_name = "Max(%s)" % self.model_name
		return HazardMap(model_name, "", self.sites, self.period, self.IMT, intensities, self.intensity_unit, self.timespan, return_period=return_period, vs30s=self.vs30s)

	def get_min_hazard_map(self):
		"""
		Get hazard map with for each site the minimum value of all hazard maps in the set.

		:returns:
			instance of :class:`HazardMap`
		"""
		intensities = np.amin(self.intensities, axis=0)
		if len(set(self.return_periods)) == 1:
			return_period = self.return_periods[0]
		else:
			return_period = 1
		model_name = "Min(%s)" % self.model_name
		return HazardMap(model_name, "", self.sites, self.period, self.IMT, intensities, self.intensity_unit, self.timespan, return_period=return_period, vs30s=self.vs30s)

	def get_mean_hazard_map(self):
		"""
		Get mean hazard map

		:return:
			instance of :class:`HazardMap`
		"""
		intensities = np.mean(self.intensities, axis=0)
		if len(set(self.return_periods)) == 1:
			return_period = self.return_periods[0]
		else:
			return_period = 1
		model_name = "Mean(%s)" % self.model_name
		return HazardMap(model_name, "", self.sites, self.period, self.IMT, intensities, self.intensity_unit, self.timespan, return_period=return_period, vs30s=self.vs30s)

	def get_median_hazard_map(self):
		"""
		Get median hazard map

		:return:
			instance of :class:`HazardMap`
		"""
		intensities = np.median(self.intensities, axis=0)
		if len(set(self.return_periods)) == 1:
			return_period = self.return_periods[0]
		else:
			return_period = 1
		model_name = "Median(%s)" % self.model_name
		return HazardMap(model_name, "", self.sites, self.period, self.IMT, intensities, self.intensity_unit, self.timespan, return_period=return_period, vs30s=self.vs30s)

	def get_percentile_hazard_map(self, perc):
		"""
		Get hazard map corresponding to percentile level

		:param perc:
			int, percentile in range [0, 100]

		:return:
			instance of :class:`HazardMap`
		"""
		intensities = np.percentile(self.intensities, perc, axis=0)
		if len(set(self.return_periods)) == 1:
			return_period = self.return_periods[0]
		else:
			return_period = 1
		model_name = "Perc%d%s)" % (perc, self.model_name)
		return HazardMap(model_name, "", self.sites, self.period, self.IMT, intensities, self.intensity_unit, self.timespan, return_period=return_period, vs30s=self.vs30s)

	def get_variance_hazard_map(self):
		"""
		Get hazard map of variance at each site.

		:return:
			instance of :class:`HazardMap`
		"""
		intensities = np.var(self.intensities, axis=0)
		if len(set(self.return_periods)) == 1:
			return_period = self.return_periods[0]
		else:
			return_period = 1
		model_name = "Var(%s)" % self.model_name
		return HazardMap(model_name, "", self.sites, self.period, self.IMT, intensities, self.intensity_unit, self.timespan, return_period=return_period, vs30s=self.vs30s)

	def get_std_hazard_map(self):
		"""
		Get hazard map of standard deviation at each site.

		:return:
			instance of :class:`HazardMap`
		"""
		intensities = np.std(self.intensities, axis=0)
		if len(set(self.return_periods)) == 1:
			return_period = self.return_periods[0]
		else:
			return_period = 1
		model_name = "Std(%s)" % self.model_name
		return HazardMap(model_name, "", self.sites, self.period, self.IMT, intensities, self.intensity_unit, self.timespan, return_period=return_period, vs30s=self.vs30s)

	def export_VM(self, base_filespec):
		for hazardmap in self:
			hazardmap.export_VM(self, base_filespec)


## Aliases
GroundMotionField = HazardMap

# TODO: implement GroundMotionField without probabilistic properties
# and subclass HazardMap from that


if __name__ == "__main__":
	pass
