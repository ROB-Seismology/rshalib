# -*- coding: utf-8 -*-
# pylint: disable=W0142, W0312, C0103, R0913
"""
GroundMotionField and derived classes
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
import os

import numpy as np
from scipy.stats import scoreatpercentile
import matplotlib

from ..site import GenericSite
from ..utils import interpolate

from .base_array import *
from .hc_base import *


# TODO: instead of vs30s property, we could use SoilSite objects


__all__ = ['GroundMotionField', 'HazardMap', 'HazardMapSet']


class GroundMotionField(IntensityResult, HazardField):
	"""
	Class representing a simple ground-motion field without probabilistic
	attributes

	:param sites:
		1-D list [i] with instances of :class:`GenericSite`
		or (lon, lat, [z]) tuples of all sites
	:param period:
		float, spectral period (in s)
	:param intensities:
		1-D array [i], ground-motion levels
	:param intensity_unit:
		str, intensity unit
		If not specified, default intensity unit for given imt will be used
	:param imt:
		str, intensity measure type ('PGA', 'PGV', 'PGD', 'SA', 'SV', 'SD')
	:param model_name:
		str, name of this model
		(default: "")
	:param filespec:
		str, full path to file containing this ground-motion map
		(default: None)
	:param damping:
		float, damping corresponding to response spectrum
		(expressed as fraction of critical damping)
		(default: 0.05)
	:param vs30s:
		1-D array [i] of VS30 values for each site
		(default: None)
	"""
	def __init__(self, sites, period,
				intensities, intensity_unit, imt,
				model_name='', filespec=None,
				damping=0.05, vs30s=None):
		IntensityResult.__init__(self, intensities=intensities,
								intensity_unit=intensity_unit, imt=imt,
								damping=damping)
		#if np.isnan(self.intensities).any():
		#	self.intensities = np.ma.array(self.intensities,
		#									mask=np.isnan(self.intensities))
		HazardField.__init__(self, sites)
		self.model_name = model_name
		self.filespec = filespec
		self.period = period
		self.vs30s = as_array(vs30s)

	def __repr__(self):
		txt = '<GroundMotionField "%s" | %s T=%s s | %d sites | region=%s>'
		txt %= (self.model_name, self.imt, self.period, self.num_sites,
				self.get_region())
		return txt

	def __iter__(self):
		for i in range(self.num_sites):
			yield self.get_site_intensity(i)

	def __getitem__(self, site_spec):
		return self.get_site_intensity(site_spec)

	def get_site_intensity(self, site_spec, intensity_unit="g"):
		"""
		Extract intensity value for a particular site

		:param site_spec:
			site specification:
			- int: site index
			- str: site name
			- instance of :class:`rshalib.site.GenericSite`: site
			- (lon, lat) tuple
		:param intensity_unit:
			string, intensity unit to scale result,
			either "g", "mg", "m/s2", "gal" or "cm/s2"
			(default: "")

		:return:
			float
		"""
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
			see :meth:`get_intensities`

		:return:
			float
		"""
		return np.nanmin(self.get_intensities(intensity_unit))

	def max(self, intensity_unit="g"):
		"""
		Return maximum intensity

		:param intensity_unit:
			see :meth:`get_intensities`

		:return:
			float
		"""
		return np.nanmax(self.get_intensities(intensity_unit))

	def mean(self, intensity_unit="g"):
		"""
		Return mean intensity in the map

		:param intensity_unit:
			see :meth:`get_intensities`

		:return:
			float
		"""
		return np.nanmean(self.get_intensities(intensity_unit))

	def median(self, intensity_unit="g"):
		"""
		Return median intensity in the map

		:param intensity_unit:
			see :meth:`get_intensities`

		:return:
			float
		"""
		return np.nanmedian(self.get_intensities(intensity_unit))

	def scoreatpercentile(self, perc, intensity_unit="g"):
		"""
		Return intensity corresponding to given percentile

		:param perc:
			float, percentile in range [0, 100]

		:param intensity_unit:
			see :meth:`get_intensities`

		:return:
			float
		"""
		return scoreatpercentile(self.get_intensities(intensity_unit), perc)

	def argmin(self):
		"""
		Return site index corresponding to minimum intensity value

		:return:
			int
		"""
		return self.intensities.argmin()

	def argmax(self):
		"""
		Return site index corresponding to maximum intensity value

		:return:
			int
		"""
		return self.intensities.argmax()

	def trim(self, lonmin=None, lonmax=None, latmin=None, latmax=None):
		"""
		Trim map to given lon/lat bounds

		:param lonmin:
			float, minimum longitude
			(default: None)
		:param lonmax:
			float, maximum longitude
			(default: None)
		:param latmin:
			float, minimum latitude
			(default: None)
		:param latmax:
			float, maximum latitude
			(default: None)

		:result:
			instance of :class:`GroundMotionField` or :class:`HazardMap`
		"""
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
		imt = self.imt
		intensities = self.intensities[site_indexes]
		intensity_unit = self.intensity_unit
		damping = self.damping

		opt_kwargs = {}
		if hasattr(self, 'return_period'):
			## HazardMap
			opt_kwargs = dict(timespan=self.timespan, poe=self.poe,
							return_period=self.return_period)

		gmf = self.__class__(sites, period, intensities, intensity_unit, imt,
							model_name=model_name, filespec=filespec,
							damping=damping, vs30s=vs30s, **opt_kwargs)
		return gmf

	def interpolate_map(self, grid_extent=(None, None, None, None),
						num_grid_cells=50, method="cubic"):
		"""
		Interpolate ground-motion map on a regular (lon, lat) grid

		:param grid_extent:
			(lonmin, lonmax, latmin, latmax) tuple of floats, grid bounds
			(default: (None, None, None, None))
		:param num_grid_cells:
			int or tuple, number of grid cells in X and Y direction
			(default: 50)
		:param method:
			str, interpolation method supported by griddata (either
			"linear", "nearest" or "cubic")
			(default: "cubic")

		:return:
			instance of :class:`GroundMotionField` or :class:`HazardMap`
		"""
		grid_lons, grid_lats = self.meshgrid(grid_extent, num_grid_cells)
		grid_intensities = self.get_site_intensities(grid_lons, grid_lats, method)
		intensities = grid_intensities.flatten()

		model_name = self.model_name + " (interpolated)"
		filespec = self.filespec
		sites = list(zip(grid_lons.flatten(), grid_lats.flatten()))
		period = self.period
		imt = self.imt
		intensity_unit = self.intensity_unit
		damping = self.damping
		vs30s = None

		opt_kwargs = {}
		if hasattr(self, 'return_period'):
			## HazardMap
			opt_kwargs = dict(timespan=self.timespan, poe=self.poe,
							return_period=self.return_period)

		gmf = self.__class__(sites, period, intensities, intensity_unit, imt,
							model_name=model_name, filespec=filespec,
							damping=damping, vs30s=vs30s, **opt_kwargs)
		return gmf

	def get_residual_map(self, other_map, grid_extent=(None, None, None, None),
						num_grid_cells=50, interpol_method="linear", abs=True):
		"""
		Compute difference with another ground-motion map. If sites are
		different, the maps will be interpolated on a regular (lon, lat)
		grid

		:param other_map:
			instance of :class:`GroundMotionField` or :class:`HazardMap`
		:param grid_extent:
			(lonmin, lonmax, latmin, latmax) tuple of floats,
			grid extent for interpolation if map sites are different
			(default: (None, None, None, None))
		:param num_grid_cells:
		:param interpol_method:
			see :meth:`interpolate_map`
		:param abs:
			bool, whether or not residual map values should be absolute
			(True) or in percentage relative to current map (False)
			(default: True)

		:return:
			instance of :class:`GroundMotionField` or :class:`HazardMap`
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
		imt = self.imt
		intensity_unit = self.intensity_unit
		damping = self.damping

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

		opt_kwargs = {}
		if (hasattr(self, 'return_period')
			and np.isclose(self.return_period, other_map.return_period)):
			## return HazardMap if return periods are the same,
			## otherwise a GroundMotionField
			opt_kwargs = dict(timespan=self.timespan, poe=self.poe,
							return_period=self.return_period)

		gmf = self.__class__(sites, period, intensities, intensity_unit, imt,
							model_name=model_name, filespec=filespec,
							damping=damping, vs30s=vs30s, **opt_kwargs)
		return gmf

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
			instance of :class:`GroundMotionField` or :class:`HazardMap`
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
		imt = self.imt
		intensity_unit = self.intensity_unit
		damping = self.damping

		opt_kwargs = {}
		if hasattr(self, 'return_period'):
			## HazardMap
			opt_kwargs = dict(timespan=self.timespan, poe=self.poe,
							return_period=self.return_period)

		gmf = self.__class__(sites, period, intensities, intensity_unit, imt,
							model_name=model_name, filespec=filespec,
							damping=damping, vs30s=vs30s, **opt_kwargs)
		return gmf

	def export_VM(self, base_filespec, num_cells=100):
		"""
		Export hazard map to a Vertical Mapper grid
		"""
		import mapping.VMPython as vm
		if self.imt in ("PGA", "PGV", "PGV"):
			imt_label = self.imt
		else:
			imt_label = "T=%.3fs" % self.period
		rp_label = "%.3Gyr" % self.return_period
		grd_filespec = (os.path.splitext(base_filespec)[0]
						+ "_%s_%s" % (imt_label, rp_label))

		(lonmin, lonmax, dlon), (latmin, latmax, dlat) = self.get_grid_properties()
		assert abs(dlon - dlat) < 1E-12
		cell_size = dlon
		zmin, zmax = self.min(), self.max()
		vmgrd = vm.CreateNumericGrid(grd_filespec, lonmin, lonmax, latmin, latmax,
							cell_size, zmin, zmax, Zdescription=self.intensity_unit)
		print("Created VM grid %s" % grd_filespec)

		intensity_grid = self.get_grid_intensities(num_cells)
		nrows = intensity_grid.shape[0]
		for rownr, row in enumerate(intensity_grid):
			vmgrd.WriteRow(row, (nrows-1)-rownr)
		vmgrd.Close()

	def export_geotiff(self, base_filespec, num_cells=None, cell_size=None,
						interpol_method='cubic', intensity_unit='g',
						nodata_value=np.nan):
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

		if self.imt in ("PGA", "PGV", "PGV"):
			imt_label = self.imt
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

	def to_lbm_layer(self, cmap="usgs", norm=None, contour_interval=None,
				amin=None, amax=None, intensity_unit="",
				num_grid_cells=100, color_gradient="cont",
				contour_line_style="default", colorbar_style="default"):
		"""
		Construct map layer to be plotted with layeredbasemap

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
		:param intensity_unit:
			str, unit in which ground-motion values need to be expressed
			(default: "")
		:param num_grid_cells:
			int or tuple, number of grid cells for interpolating
			intensity grid in X and Y direction
			(default: 100)
		:param color_gradient:
			String, either "disc" for discrete or "cont" for continuous
			(default: "cont")
		:param contour_line_style:
			instance of :class:`LayeredBasemap.LineStyle`, defining style
			for plotting contour lines
			(default: "default")
		:param colorbar_style:
			instance of :class:`LayeredBasemap.ColorbarStyle`, defining
			style for plotting color bar
			(default: "default")

		:return:
			instance of :class:`layeredbasemap.MapLayer`
		"""
		import mapping.layeredbasemap as lbm
		from .plot import get_intensity_unit_label

		## Construct default styles
		if contour_line_style == "default":
			contour_label_style = lbm.TextStyle(font_size=10,
												background_color=(1,1,1,0.5))
			contour_line_style = lbm.LineStyle(label_style=contour_label_style)

		## Prepare intensity grid and contour levels
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
			## If intensity_unit is 'g'
			candidates = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08,
								0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 1.0])
			try:
				index = np.where(arange / candidates <= 10)[0][0]
			except IndexError:
				index = 0
			contour_interval = candidates[index]
		else:
			contour_interval = float(contour_interval)

		if amin is None:
			amin = (np.floor(self.min(intensity_unit) / contour_interval)
					* contour_interval)
		if amax is None:
			amax = (np.ceil(self.max(intensity_unit) / contour_interval)
					* contour_interval)

		if contour_interval is not None:
			contour_levels = np.arange(amin, amax+contour_interval, contour_interval)
			## Sometimes, there is an empty contour interval at the end
			if len(contour_levels) > 1 and contour_levels[-2] > self.max():
				contour_levels = contour_levels[:-1]
		elif contour_interval == 0:
			contour_levels = []
		else:
			contour_levels = None

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
		if self.imt in ("SA", "PGA"):
			label_format="%.2f"
			if self.period == 0:
				imt_label = "PGA"
			else:
				imt_label = "%s (%s s)" % (self.imt, self.period)
		else:
			imt_label = self.imt
			label_format="%s"
		intensity_unit_label = get_intensity_unit_label(intensity_unit)
		cbar_label = imt_label
		if intensity_unit:
			cbar_label += ' (%s)' % intensity_unit_label

		ticks = contour_levels
		if not (ticks is None or ticks == []):
			ticks = ticks[ticks <= norm.vmax]
		if colorbar_style == "default":
			colorbar_style = lbm.ColorbarStyle(location="bottom",
												format=label_format, ticks=ticks,
												title=cbar_label)
		color_map_theme = lbm.ThematicStyleColormap(color_map=cmap, norm=norm,
													vmin=amin, vmax=amax,
													colorbar_style=colorbar_style)
		color_gradient = {"cont": "continuous",
						"disc": "discontinuous"}[color_gradient]
		grid_style = lbm.GridStyle(color_map_theme=color_map_theme,
									color_gradient=color_gradient,
									line_style=contour_line_style,
									contour_levels=contour_levels)
		grid_data = lbm.MeshGridData(grid_lons, grid_lats, intensity_grid)
		layer = lbm.MapLayer(grid_data, grid_style, name="intensity_grid")

		return layer

	def get_plot(self, region=None, projection="merc", resolution="i",
				graticule_interval=(1., 1.),
				cmap="usgs", norm=None, contour_interval=None,
				amin=None, amax=None, intensity_unit="",
				num_grid_cells=100, color_gradient="cont", contour_line_style="default",
				colorbar_style="default", site_style="default",
				source_model="", source_model_style="default",
				countries_style="default", coastline_style="default",
				hide_sea=False,
				title=None, show_legend=True, ax=None, **kwargs):
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
		:param norm:
		:param contour_interval:
		:param amin:
		:param amax:
		:param intensity_unit:
		:param num_grid_cells:
		:param color_gradient:
		:param contour_line_style:
		:param colorbar_style:
			see :meth:`to_lbm_layer`
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
		:param hide_sea:
			bool, whether or not hazard map should be masked over seas
			and oceans
			(default: False)
		:param title:
			str, map title. If empty string, no title will be plotted.
			If None, default title will be used
			(default: None)
		:param show_legend:
			bool, whether or not to show the legend for sources
			(default: True)
		:param ax:
			matplotlib Axes instance
			(default: None)
		:kwargs:
			additional keyword arguments to be passed to LayeredBasemap
			constructor

		:return:
			instance of :class:`LayeredBasemap.LayeredBasemap`, where
			additional layers may be added before final plotting.
		"""
		import mapping.layeredbasemap as lbm
		from .plot import get_intensity_unit_label

		## Construct default styles
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

		## Compute map limits
		if not region:
			region = self.get_region()

		map_layers = []

		## Intensity grid
		grid_layer = self.to_lbm_layer(cmap=cmap, norm=norm,
						contour_interval=contour_interval,
						amin=amin, amax=amax, intensity_unit=intensity_unit,
						num_grid_cells=num_grid_cells, color_gradient=color_gradient,
						contour_line_style=contour_line_style,
						colorbar_style=colorbar_style)
		map_layers.append(grid_layer)

		## Intensity data points
		if site_style:
			site_data = lbm.MultiPointData(self.longitudes, self.latitudes)
			map_layers.append(lbm.MapLayer(site_data, site_style,
											name="intensity_points"))

		if hide_sea:
			continent_style = lbm.FocmecStyle(fill_color=(1, 1, 1, 0),
												bg_color=(1, 1, 1, 1),
												line_width=0, line_color="none")
			data = lbm.BuiltinData("continents")
			map_layers.append(lbm.MapLayer(data, continent_style, name="ocean"))

		## Coastlines and national boundaries
		if coastline_style:
			map_layers.append(lbm.MapLayer(lbm.BuiltinData("coastlines"),
											coastline_style, name="coastlines"))
		if countries_style:
			map_layers.append(lbm.MapLayer(lbm.BuiltinData("countries"),
											countries_style, name="countries"))

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
						polygon_data.append(lbm.PolygonData(source.longitudes,
															source.latitudes))
						if not "polygons" in legend_label and show_legend:
							legend_label["polygons"] = "Area sources"
					elif isinstance(source, (SimpleFaultSource, CharacteristicFaultSource)):
						pg = source.get_polygon()
						polygon_data.append(lbm.PolygonData(pg.lons, pg.lats))
						fault_trace = source.fault_trace
						line_data.append(lbm.LineData(fault_trace.lons,
													fault_trace.lats))
						if not "lines" in legend_label and show_legend:
							legend_label["lines"] = "Fault sources"
					elif isinstance(source, PointSource):
						point_data.append(lbm.PointData(source.location.longitude,
														source.location.latitude))
						if not "points" in legend_label and show_legend:
							legend_label["points"] = "Point sources"
					else:
						print("Warning: Skipped plotting source %s, "
							"source type not supported" % source.source_id)
				sm_data = lbm.CompositeData(lines=line_data, polygons=polygon_data,
											points=point_data)
			sm_style = source_model_style
			sm_layer = lbm.MapLayer(sm_data, sm_style, legend_label=legend_label,
									name="source_model")
			map_layers.append(sm_layer)

		## Title
		if title is None:
			title = self.model_name
			if hasattr(self, "return_period"):
				title += "\nTr=%.4G yr" % self.return_period

		if source_model:
			legend_style = lbm.LegendStyle(location=0)
		else:
			legend_style = None

		graticule_style = lbm.GraticuleStyle(annot_axes="SE")
		map = lbm.LayeredBasemap(map_layers, title, projection, region=region,
					resolution=resolution, graticule_interval=graticule_interval,
					graticule_style=graticule_style, legend_style=legend_style,
					ax=ax, **kwargs)
		return map


class HazardMap(HazardResult, GroundMotionField):
	"""
	Class representing a hazard map, i.e. a ground-motion field for
	a particular return period.
	One hazard map, corresponds to 1 OQ file

	:param sites:
	:param period:
	:param intensities:
	:param intensity_unit:
	:param imt:
	:param model_name:
	:param filespec:
		see :class:`GroundMotionField`
	:param timespan:
		float, time span corresponding to probability of exceedance
		(default: 50)
	:param poe:
		float, probability of exceedance
		(default: None)
	:param return_period:
		float, return period
		(default: None)
		Note: either return period or poe must be specified!
	:param damping:
	:param vs30s:
		see :class:`GroundMotionField`
	"""
	def __init__(self, sites, period,
				intensities, intensity_unit, imt,
				model_name='', filespec=None,
				timespan=50, poe=None, return_period=None,
				damping=0.05, vs30s=None):
		if return_period:
			hazard_values = ExceedanceRateArray([1./return_period])
		elif poe:
			hazard_values = ProbabilityArray([poe])
		HazardResult.__init__(self, hazard_values, timespan=timespan, imt=imt,
							intensities=intensities, intensity_unit=intensity_unit,
							damping=damping)
		GroundMotionField.__init__(self, sites, period,
									intensities, intensity_unit, imt,
									model_name=model_name, filespec=filespec,
									damping=damping, vs30s=vs30s)

	def __repr__(self):
		txt = '<HazardMap "%s" | %s T=%s s | Tr=%G yr | %d sites | region=%s>'
		txt %= (self.model_name, self.imt, self.period, self.return_period,
				self.num_sites, self.get_region())
		return txt

	@property
	def poe(self):
		return self.poes[0]

	@property
	def return_period(self):
		return self.return_periods[0]

	@property
	def exceedance_rate(self):
		return self.exceedance_rates[0]


class HazardMapSet(HazardResult, HazardField):
	"""
	Class representing a set of hazard maps or ground-motion fields
	for different return periods.
	Corresponds to 1 CRISIS MAP file containing 1 spectral period.

	:param sites:
	:param period:
		see :class:`HazardMap`
	:param intensities:
		2-D array [p, i], ground-motion values for different return
		periods p and sites i
	:param intensity_unit:
	:param imt:
	:param model_name:
		see :class:`HazardMap`
	:param filespecs:
		list of strings [p], paths to files corresponding to hazard maps
		(default: [])
	:param timespan:
		float, time span corresponding to exceedance probabilities
		(default: 50)
	:param poes:
		1-D array or probabilities of exceedance [p]
		(default: None)
	:param return_periods:
		1-D [p] array of return periods
		(default: None)
	:param damping:
	:param vs30s:
		see :class:`HazardMap`
	"""
	def __init__(self, sites, period,
				intensities, intensity_unit, imt,
				model_name="", filespecs=[],
				timespan=50, poes=None, return_periods=None,
				damping=0.05, vs30s=None):
		if not return_periods in (None, []):
			hazard_values = ExceedanceRateArray(1./as_array(return_periods))
		elif poes:
			hazard_values = ProbabilityArray(as_array(poes))

		HazardResult.__init__(self, hazard_values, timespan=timespan, imt=imt,
							intensities=intensities, intensity_unit=intensity_unit,
							damping=damping)
		HazardField.__init__(self, sites)

		self.period = period
		self.model_name = model_name
		if len(filespecs) == 1:
			filespecs *= len(return_periods)
		self.filespecs = filespecs
		self.vs30s = as_array(vs30s)

	def __repr__(self):
		txt = '<HazardMapSet "%s" | %s T=%s s  | %d sites | nTr=%d>'
		txt %= (self.model_name, self.imt, self.period, self.num_sites,
				len(self))
		return txt

	def __iter__(self):
		for i in range(len(self)):
			yield self.get_hazard_map(index=i)

	def __getitem__(self, index):
		return self.get_hazard_map(index=index)

	def __len__(self):
		return len(self.return_periods)

	@classmethod
	def from_hazard_maps(cls, hazard_maps, model_name=""):
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
		imt = hm0.imt
		intensities = np.zeros((len(hazard_maps), len(sites)))
		intensity_unit = hm0.intensity_unit
		timespan = hm0.timespan
		poes = []
		return_periods = []
		damping = hm0.damping
		vs30s = hm0.vs30s
		for i, hm in enumerate(hazard_maps):
			assert hm.sites == hm0.sites
			assert hm.intensity_unit == hm0.intensity_unit
			#assert (hm.vs30s == hm0.vs30s).all()
			intensities[i] = hm.intensities
			poes.append(hm.poe)
			return_periods.append(hm.return_period)

		return cls(sites, period, intensities, intensity_unit, imt,
				model_name=model_name, filespecs=filespecs,
				timespan=timespan, return_periods=return_periods,
				damping=damping, vs30s=vs30s)

	def get_hazard_map(self, index=None, poe=None, return_period=None):
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
			raise Exception("One of index, poe or return_period "
							"must be specified!")
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
					raise ValueError("No hazard map for return period=%s"
									% return_period)
				else:
					index = index[0]

		try:
			return_period = self.return_periods[index]
		except:
			raise IndexError("Index %s out of range" % index)
		else:
			filespec = self.filespecs[index]
			intensities = self.intensities[index]
			return HazardMap(self.sites, self.period,
							intensities, self.intensity_unit, self.imt,
							model_name=self.model_name, filespec=filespec,
							timespan=self.timespan, return_period=return_period,
							damping=self.damping, vs30s=self.vs30s)

	# TODO: the following methods are perhaps better suited in a HazardMapTree class
	def get_max_hazard_map(self):
		"""
		Get hazard map with for each site the maximum value of
		all hazard maps in the set.

		:returns:
			instance of :class:`HazardMap`
		"""
		intensities = np.amax(self.intensities, axis=0)
		if len(set(self.return_periods)) == 1:
			return_period = self.return_periods[0]
		else:
			return_period = 1
		model_name = "Max(%s)" % self.model_name

		return HazardMap(self.sites, self.period,
						intensities, self.intensity_unit, self.imt,
						model_name=model_name, filespec=None,
						timespan=self.timespan, return_period=return_period,
						damping=self.damping, vs30s=self.vs30s)

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

		return HazardMap(self.sites, self.period,
						intensities, self.intensity_unit, self.imt,
						model_name=model_name, filespec=None,
						timespan=self.timespan, return_period=return_period,
						damping=self.damping, vs30s=self.vs30s)

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

		return HazardMap(self.sites, self.period,
						intensities, self.intensity_unit, self.imt,
						model_name=model_name, filespec=None,
						timespan=self.timespan, return_period=return_period,
						damping=self.damping, vs30s=self.vs30s)

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

		return HazardMap(self.sites, self.period,
						intensities, self.intensity_unit, self.imt,
						model_name=model_name, filespec=None,
						timespan=self.timespan, return_period=return_period,
						damping=self.damping, vs30s=self.vs30s)

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

		return HazardMap(self.sites, self.period,
						intensities, self.intensity_unit, self.imt,
						model_name=model_name, filespec=None,
						timespan=self.timespan, return_period=return_period,
						damping=self.damping, vs30s=self.vs30s)

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

		return HazardMap(self.sites, self.period,
						intensities, self.intensity_unit, self.imt,
						model_name=model_name, filespec=None,
						timespan=self.timespan, return_period=return_period,
						damping=self.damping, vs30s=self.vs30s)

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

		return HazardMap(self.sites, self.period,
						intensities, self.intensity_unit, self.imt,
						model_name=model_name, filespec=None,
						timespan=self.timespan, return_period=return_period,
						damping=self.damping, vs30s=self.vs30s)



if __name__ == "__main__":
	pass
