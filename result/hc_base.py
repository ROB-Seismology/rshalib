# -*- coding: utf-8 -*-
# pylint: disable=W0142, W0312, C0103, R0913
"""
Base classes for hazard-curve results
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
import numpy as np

from ..site import GenericSite
from .base_array import HazardCurveArray



__all__ = ['IntensityResult', 'HazardResult', 'HazardSpectrum',
			'HazardField', 'HazardTree']


class IntensityResult:
	"""
	Generic class providing common methods related to intensities

	:param IMT:
		str, intensity measure type ('PGA', 'PGV', 'PGD', 'SA', 'SV', 'SD')
	:param intensities:
		ndarray, intensities (=ground-motion levels)
	:param intensity_unit:
		str, intensity unit
		(default: "", will take default intensity unit for given IMT
	"""
	def __init__(self, IMT, intensities, intensity_unit=""):
		self.IMT = IMT
		self.intensities = as_array(intensities)
		self.intensity_unit = intensity_unit or self.get_default_intensity_unit(IMT)

	@property
	def num_intensities(self):
		return self.intensities.shape[-1]

	def _convert_intensities(self, intensities, src_intensity_unit,
							target_intensity_unit):
		"""
		Convert intensities from one intensity unit to another

		:param intensities:
			float array, intensities
		:param src_intensity_unit:
			string, intensity unit corresponding to :param:`intensities`
		:param target_intensity_unit:
			string, desired output intensity unit

		:return:
			float array, converted intensities
		"""
		from scipy.constants import g
		conv_factor = None
		if self.IMT in ("PGA", "SA"):
			if src_intensity_unit == "g":
				conv_factor = {"g": 1.0,
								"mg": 1E+3,
								"ms2": g,
								"gal": g*100,
								"cms2": g*100}[target_intensity_unit]
			elif src_intensity_unit == "mg":
				conv_factor = {"g": 1E-3,
								"mg": 1.0,
								"ms2": g*1E-3,
								"gal": g*1E-1,
								"cms2": g*1E-1}[target_intensity_unit]
			elif src_intensity_unit in ("gal", "cms2"):
				conv_factor = {"g": 0.01/g,
								"mg": 10./g,
								"ms2": 1E-2,
								"gal": 1.0,
								"cms2": 1.0}[target_intensity_unit]
			elif src_intensity_unit == "ms2":
				conv_factor = {"g": 1./g,
								"mg": 1E+3/g,
								"ms2": 1.,
								"gal": 100.0,
								"cms2": 100.0}[target_intensity_unit]
		elif self.IMT == "PGV":
			if src_intensity_unit == "ms":
				conv_factor = {"ms": 1., "cms": 1E+2}[target_intensity_unit]
			elif src_intensity_unit == "cms":
				conv_factor = {"ms": 1E-2, "cms": 1.0}[target_intensity_unit]
		elif self.IMT == "PGD":
			if src_intensity_unit == "m":
				conv_factor = {"m": 1., "cm": 1E+2}[target_intensity_unit]
			elif src_intensity_unit == "cm":
				conv_factor = {"m": 1E-2, "cm": 1.0}[target_intensity_unit]
		elif self.IMT == "MMI":
			conv_factor = 1.

		if conv_factor is None:
			raise Exception("Unable to convert intensity unit %s for %s!"
							% (intensity_unit, self.IMT))

		return intensities * conv_factor

	def get_intensities(self, intensity_unit=""):
		"""
		Get intensity array, optionally converted to a different intensity unit

		:param intensity_unit:
			string, intensity unit to scale result,
			either "g", "mg", "ms2", "gal" or "cms2"
			(default: "")
		"""
		if intensity_unit and intensity_unit != self.intensity_unit:
			return self._convert_intensities(self.intensities, self.intensity_unit,
											intensity_unit)
		else:
			return self.intensities

	def set_intensity_unit(self, target_intensity_unit):
		"""
		Convert intensities to a different intensity unit in place

		:param new_intensity_unit:
			string, new intensity unit
		"""
		if target_intensity_unit != self.intensity_unit:
			try:
				intensities = self._convert_intensities(self.intensities,
									self.intensity_unit, target_intensity_unit)
			except:
				raise
			else:
				self.intensities = intensities
				self.intensity_unit = target_intensity_unit

	def get_default_intensity_unit(self, IMT):
		"""
		Return default intensity unit for given IMT

		:param IMT:
			str

		:return:
			str, intensity unit
		"""
		if IMT in ("PGA", "SA"):
			return "g"
		elif IMT in ("PGV", "SV"):
			return "ms"
		elif IMT in ("PGD", "SD"):
			return "m"
		elif IMT == "MMI":
			return ""


class HazardResult(IntensityResult):
	"""
	Generic class providing common methods related to poE's, exceedance rates,
	and return periods

	:param hazard_values:
		instance of subclass of :class:`HazardCurveArray`, either
		exceedance rates or exceedance probabilities
	:param timespan:
		float, time span for exceedance rates or probabilities
		(default: 50)
	:param IMT:
	:param intensities:
	:param intensity_unit:
		see :class:`IntensityResult`
	"""
	def __init__(self, hazard_values, timespan=50,
				IMT="PGA", intensities=None, intensity_unit=""):
		if not isinstance(hazard_values, HazardCurveArray):
			raise Exception("hazard_values should be instance of HazardCurveArray!")
		IntensityResult.__init__(self, IMT, intensities, intensity_unit)
		self._hazard_values = hazard_values
		self.timespan = float(timespan)

	@property
	def exceedance_rates(self):
		"""
		Return property exceedance_rates or compute from poes and timespan
		"""
		return self._hazard_values.to_exceedance_rates(self.timespan)

	@property
	def return_periods(self):
		"""
		Return property return_periods or compute from poes and timespan
		"""
		return self._hazard_values.to_return_periods(self.timespan)

	@property
	def poes(self):
		"""
		Return property poes or compute from exceedance rates and timespan
		"""
		return self._hazard_values.to_probabilities(self.timespan)


class HazardSpectrum():
	"""
	Generic class providing common methods related to periods

	:param periods:
		1-D array, spectral periods (in s)
	:param period_index:
		int, index in multimensional array of hazard values
		corresponding to spectral periods
		(default: None)
	"""
	def __init__(self, periods, period_index=None):
		self.periods = as_array(periods)
		self.period_index = period_index

	def __len__(self):
		return len(self.periods)

	def period_index(self, period_spec):
		"""
		Determine index of a particular period:

		:param period_spec:
			int: period index
			float: spectral period

		:return:
			int, period index
		"""
		if isinstance(period_spec, int):
			period_index = period_spec
		elif isinstance(period_spec, (float, np.floating)):
			if len(self.periods) > 1:
				min_diff = abs(self.periods[1:] - self.periods[:-1]).min()
				min_diff = min(min_diff, 1E-3)
			else:
				min_diff = 1E-3
			period_index = np.where(np.abs(self.periods - period_spec) < min_diff)[0][0]
		else:
			raise Exception("Invalid period specification: %s" % period_spec)
		return period_index

	@property
	def num_periods(self):
		return len(self.periods)

	@property
	def frequencies(self):
		return 1./self.periods

	def reorder_periods(self):
		"""
		Reorder spectral periods from shortest to longest
		"""
		idxs = np.argsort(self.periods)
		self.periods = self.periods[idxs]
		# TODO: check if period_index applies to intensities as well
		self.intensities = self.intensities.take(idxs, axis=0)
		if self.period_axis is not None:
			self._hazard_values = self._hazard_values.take(idxs, axis=self.period_axis)


class HazardField:
	"""
	Generic class providing common methods related to sites

	:param sites:
		list with instances of :class:`rshalib.site.GenericSite`
	"""
	def __init__(self, sites):
		self.sites = sites

	def __len__(self):
		return self.num_sites

	@property
	def num_sites(self):
		return len(self.sites)

	@property
	def site_names(self):
		return [site.name for site in self.sites]

	def set_site_names(self, sites):
		"""
		Set site names from a list of generic (or soil) sites

		:param sites:
			list with instances of :class:`GenericSite`
		"""
		for i in range(self.num_sites):
			lon, lat = (self.sites[i].lon, self.sites[i].lat)
			for site in sites:
				if np.allclose((lon, lat), (site.lon, site.lat), atol=1E-6):
					self.sites[i].name = site.name

	@property
	def longitudes(self):
		"""
		Return array with longitudes of all sites
		"""
		return np.array([site.lon for site in self.sites])

	@property
	def latitudes(self):
		"""
		Return array with latitudes of all sites
		"""
		return np.array([site.lat for site in self.sites])

	def lonmin(self):
		"""
		Return minimum longitude
		"""
		return self.longitudes.min()

	def lonmax(self):
		"""
		Return maximum longitude
		"""
		return self.longitudes.max()

	def latmin(self):
		"""
		Return minimum latitude
		"""
		return self.latitudes.min()

	def latmax(self):
		"""
		Return maximum latitude
		"""
		return self.latitudes.max()

	def get_grid_longitudes(self, lonmin=None, lonmax=None, num_cells=100):
		"""
		Return array of equally spaced longitudes

		:param lonmin:
			Float, minimum longitude
			(default: None, will use minimum longitude of current sites)
		:param lonmax:
			Float, maximum longitude
			(default: None, will use maximum longitude of current sites)
		:param num_cells:
			Integer, number of grid cells
			(default: 100)
		"""
		#unique_lons = list(set(self.longitudes))
		#unique_lons.sort()
		#return unique_lons
		if lonmin is None:
			lonmin = self.lonmin()
		if lonmax is None:
			lonmax = self.lonmax()
		return np.linspace(lonmin, lonmax, num_cells, dtype='f')

	def get_grid_latitudes(self, latmin=None, latmax=None, num_cells=100):
		"""
		Return array of equally spaced latitudes

		:param latmin:
			Float, minimum latitude
			(default: None, will use minimum latitude of current sites)
		:param latmax:
			Float, maximum latitude
			(default: None, will use maximum latitude of current sites)
		:param num_cells:
			Integer, number of grid cells
			(default: 100)
		"""
		#unique_lats = list(set(self.latitudes))
		#unique_lats.sort()
		#return unique_lats
		if latmin is None:
			latmin = self.latmin()
		if latmax is None:
			latmax = self.latmax()
		return np.linspace(latmin, latmax, num_cells, dtype='f')

	def meshgrid(self, extent=(None, None, None, None), num_cells=100):
		"""
		Return longitude and latitude matrices of the grid

		:param extent:
			(lonmin, lonmax, latmin, latmax) tuple of floats
			(default: (None, None, None, None))
		:param num_cells:
			Integer or tuple, number of grid cells in X and Y direction
			(default: 100)

		:return:
			(lons, lats) tuple of 2-D arrays
		"""
		lonmin, lonmax, latmin, latmax = extent
		if isinstance(num_cells, (int, np.integer)):
			num_cells = (num_cells, num_cells)
		return np.meshgrid(self.get_grid_longitudes(lonmin, lonmax, num_cells[0]),
						self.get_grid_latitudes(latmin, latmax, num_cells[1]),
						copy=False)

	def get_grid_intensities(self, extent=(None, None, None, None), num_cells=100,
							method="cubic", intensity_unit="", nodata_value=np.nan):
		"""
		Convert intensities to a spatial grid (2-D array)

		:param extent:
			(lonmin, lonmax, latmin, latmax) tuple of floats
			(default: (None, None, None, None))
		:param num_cells:
			Integer or tuple, number of grid cells in X and Y direction
			(default: 100)
		:param method:
			str interpolation method supported by griddata (either
			"linear", "nearest" or "cubic")
			(default: "cubic")
		:param intensity_unit:
			str, unit for output intensities
			(default: "", will use current intensity unit)
		:param nodata_value:
			float, value to use for cells without data
			(default: np.nan)

		:return:
			2-D array, gridded intensities
		"""
		lonmin, lonmax, latmin, latmax = extent
		if isinstance(num_cells, (int, np.integer)):
			num_cells = (num_cells, num_cells)
		xi, yi = self.meshgrid(extent, num_cells)
		zi = self.get_site_intensities(xi, yi, method, intensity_unit,
										nodata_value=nodata_value)
		return zi

	def get_site_intensities(self, lons, lats, method="cubic", intensity_unit="",
							nodata_value=np.nan):
		"""
		Interpolate intensities for given sites.

		:param lons:
			2-D array, lons of sites
		:param lats:
			2-D array, lats of sites
		:param method:
			Str, interpolation method supported by griddata (either
			"linear", "nearest" or "cubic") (default: "cubic")
		:param intensity_unit:
			str, unit for output intensities
			(default: "", will use current intensity unit)
		:param nodata_value:
			float, value to use for cells without data
			(default: np.nan)

		:return:
			2-D array, gridded intensities
		"""
		from scipy.interpolate import griddata
		x, y = self.longitudes, self.latitudes
		z = self.get_intensities(intensity_unit=intensity_unit)
		zi = griddata((x, y), z, (lons, lats), method=method, fill_value=nodata_value)
		return zi

	def get_grid_properties(self):
		"""
		Return nested tuple with grid extent and spacing in both dimensions:
		((lonmin, lonmax, dlon), (latmin, latmax, dlat))
		"""
		grid_longitudes = self.get_grid_longitudes()
		grid_latitudes = self.get_grid_latitudes()
		dlon = grid_longitudes[1] - grid_longitudes[0]
		dlat = grid_latitudes[1] - grid_latitudes[0]
		lonmin, lonmax = grid_longitudes[0], grid_longitudes[-1]
		latmin, latmax = grid_latitudes[0], grid_latitudes[-1]
		return ((lonmin, lonmax, dlon), (latmin, latmax, dlat))

	def site_index(self, site_spec):
		"""
		Determine index of given site

		:param site_spec:
			int: site index
			str: site name
			instance of :class:`rshalib.site.GenericSite`: site
			(lon, lat) tuple

		:return:
			int
		"""
		if isinstance(site_spec, (int, np.integer)):
			site_index = site_spec
		elif isinstance(site_spec, basestring):
			site_index = self.site_names.index(site_spec)
		elif isinstance(site_spec, GenericSite):
			site_index = self.sites.index(site_spec)
		elif isinstance(site_spec, (list, tuple)) and len(site_spec) >= 2:
			lon, lat = site_spec[:2]
			site = GenericSite(lon, lat)
			site_index = self.sites.index(site)
		else:
			raise Exception("Invalid site specification: %s" % site_spec)
		return site_index

	def get_site_indexes(self, site_specs):
		"""
		Determine index for a list of site specifications.
		Should be much faster than repeatedly calling :meth:`site_index`

		:param site_specs:
			list of site specifications: can be int (site index),
			str (site name), tuple (lon, lat) or instance of :class:`GenericSite`

		:return:
			list of ints, site indexes
		"""
		site_spec0 = site_specs[0]
		if isinstance(site_spec0, (int, np.integer)):
			return site_specs
		elif isinstance(site_spec0, basestring):
			site_spec_index_dict = {self.site_names[i]: i
									for i in range(self.num_sites)}
		elif isinstance(site_spec0, GenericSite):
			site_specs = [site.name for site in site_specs]
			site_spec_index_dict = {self.site_names[i]: i
									for i in range(self.num_sites)}
			#site_spec_index_dict = {self.sites[i]: i for i in range(self.num_sites)}
		elif isinstance(site_spec0, (list, tuple)) and len(site_spec) >= 2:
			site_specs = [GenericSite(*ss).get_name_from_position()
						for ss in site_specs]
			site_spec_index_dict = {site.get_name_from_position(): i
									for i, site in enumerate(self.sites)}
		else:
			raise Exception("Invalid site specification: %s" % site_spec0)
		site_indexes = [site_spec_index_dict.get(site_spec)
						for site_spec in site_specs]

		return site_indexes

	def get_nearest_site_index(self, site_spec):
		"""
		Determine index of nearest site

		:param site_spec:
			instance of :class:`GenericSite` or (lon, lat) tuple

		:return:
			int, index of nearest site
		"""
		from openquake.hazardlib.geo.geodetic import geodetic_distance

		if isinstance(site_spec, GenericSite):
			lon, lat = site_spec.longitude, site_spec.latitude
		elif isinstance(site_spec, (list, tuple)) and len(site_spec) >= 2:
			lon, lat = site_spec[:2]

		distances = geodetic_distance([lon], [lat], self.longitudes, self.latitudes)
		return int(np.argmin(distances))

	## Note: the following functions are no longer used
	def get_grid_index(self, site):
		"""
		Return grid index (row, col) for given site
		"""
		lon, lat = site
		(lonmin, lonmax, dlon), (latmin, latmax, dlat) = self.get_grid_properties()
		col = int(round((lon - lonmin) / dlon))
		row = int(round((lat - latmin) / dlat))
		return row, col

	def get_grid_indexes(self):
		"""
		Return two arrays with row and column indexes of all sites
		"""
		rows = np.zeros(len(self.sites), dtype='i')
		cols = np.zeros(len(self.sites), dtype='i')
		(lonmin, lonmax, dlon), (latmin, latmax, dlat) = self.get_grid_properties()
		for i, site in enumerate(self.sites):
			try:
				lon, lat = site
			except ValueError:
				lon, lat = site.longitude, site.latitude
			cols[i] = int(round((lon - lonmin) / dlon))
			rows[i] = int(round((lat - latmin) / dlat))
		return rows, cols


class HazardTree(HazardResult):
	"""
	Generic class providing common methods related to logic-tree results.
	Inherits from HazardResult
	"""
	def __init__(self, hazard_values, branch_names, weights=None, timespan=50,
				IMT="PGA", intensities=None, intensity_unit="",
				mean=None, percentile_levels=None, percentiles=None):
		HazardResult.__init__(self, hazard_values, timespan=timespan, IMT=IMT,
							intensities=intensities, intensity_unit=intensity_unit)
		self.branch_names = branch_names
		if weights in ([], None):
			weights = np.ones(len(branch_names), 'd') / len(branch_names)
		self.weights = as_array(weights)
		self.set_mean(mean)
		self.set_percentiles(percentiles, percentile_levels)

	def __len__(self):
		return self.num_branches

	@property
	def num_percentiles(self):
		try:
			return self.percentiles.shape[-1]
		except AttributeError:
			return 0

	def branch_index(self, branch_spec):
		"""
		Determine index of a particular branch

		:param branch_spec:
			int: branch index
			str: branch name

		:return:
			int, branch index
		"""
		if isinstance(branch_spec, (int, np.integer)):
			branch_index = branch_spec
		elif isinstance(branch_spec, basestring):
			branch_index =  self.branch_names.index(branch_spec)
		else:
			raise Exception("Invalid branch specification: %s" % branch_spec)
		return branch_index

	def set_mean(self, mean):
		"""
		Set logic-tree mean, performing some checks
		"""
		if mean in ([], None):
			self.mean = None
		else:
			if (isinstance(self, SpectralHazardCurveFieldTree)
				and not isinstance(mean, HazardCurveArray)):
				raise Exception("mean should be instance of HazardCurveArray!")
			else:
				## UHSFieldTree
				self.mean = mean

	def set_percentiles(self, percentiles, percentile_levels):
		"""
		Set logic-tree percentiles, performing some checks
		"""
		self.percentile_levels = as_array(percentile_levels)
		if percentiles in ([], None):
			self.percentiles = None
		else:
			if (isinstance(self, SpectralHazardCurveFieldTree)
				and not isinstance(percentiles, HazardCurveArray)):
				raise Exception("percentiles should be instance of HazardCurveArray!")
			else:
				## UHSFieldTree
				self.percentiles = percentiles

	def weight_sum(self):
		"""
		Return total weight of all branches
		"""
		return np.sum(self.weights)

	def normalize_weights(self):
		"""
		Normalize branch weights
		"""
		self.weights /= self.weight_sum()

	def slice_by_branch_names(self, branch_names, slice_name, normalize_weights=True,
							strict=True, negate=False, verbose=False):
		"""
		Return a subset (slice) of the logic tree based on branch names

		:param branch_names:
			list of strings, branch names to match
		:param slice_name:
			str, name of this slice
		:param normalize_weights:
			bool, indicating whether or not branch weights should be
			renormalized to 1
			(default: True)
		:param strict:
			bool, indicating whether branch names should be matched
			strictly or only partly
			(default: True)
		:param negate:
			bool, indicating whether match should be negated
			(default: False)
		:param verbose:
			bool, indicating whether or not to print extra information
			(default: False)

		:return:
			An object of the same class as the parent object
		"""
		if strict:
			branch_indexes = [self.branch_index(branch_name)
							for branch_name in branch_names]
		else:
			branch_indexes = set()
			for branch_name in branch_names:
				for j, tree_branch_name in enumerate(self.branch_names):
					if branch_name in tree_branch_name:
						branch_indexes.add(j)

		if negate:
			all_branch_indexes = set(range(self.num_branches))
			branch_indexes = all_branch_indexes.difference(set(branch_indexes))

		branch_indexes = sorted(branch_indexes)

		if verbose:
			print("Sliced %d branches" % len(branch_indexes))
		return self.slice_by_branch_indexes(branch_indexes, slice_name,
											normalize_weights=normalize_weights)

	def split_by_branch_name(self, branch_names, normalize_weights=True):
		"""
		Split logic tree in different subsets based on branch names

		:param branch_names:
			list of partial branch names that are unique to each subset
		:param normalize_weights:
			bool, indicating whether or not branch weights should be
			renormalized to 1
			(default: True)

		:return:
			list of objects of the same class as the parent object
		"""
		subsets = []
		for branch_name in branch_names:
			subset = self.slice_by_branch_names([branch_name], branch_name,
								normalize_weights=normalize_weights, strict=False)
			subsets.append(subset)
		return subsets

	def slice_by_branch_indexes(self, branch_indexes, slice_name,
								normalize_weights=True):
		"""
		This method needs to be overriden in descendant classes
		"""
		pass



if __name__ == "__main__":
	pass
