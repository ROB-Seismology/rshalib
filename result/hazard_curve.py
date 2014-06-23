# -*- coding: utf-8 -*-
"""
Blueprint for classes representing hazard results of both OpenQuake and CRISIS
"""

# pylint: disable=W0142, W0312, C0103, R0913

### imports
import os, sys
from decimal import Decimal
import numpy as np

from scipy.stats import mstats, scoreatpercentile
import matplotlib
import pylab

from ..nrml import ns

from ..nrml.common import *
from ..site import SHASite
from plot import plot_hazard_curve, plot_hazard_spectrum, plot_histogram
from ..utils import interpolate, logrange, wquantiles, LevelNorm
from ..pmf import NumericPMF



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
"""


def Poisson(life_time=None, return_period=None, prob=None):
	"""
	Compute return period, life time or probability of exceedance from any
		combination of two of the other parameters for a Poisson distribution
	Parameters:
		life_time: life time (default: None)
		return_period: return period (default: None)
		prob: probability (default: None)
	Two parameters need to be specified, the value will be computed for the
	missing parameter
	"""
	## Return period
	if life_time != None and prob != None:
		return -life_time / np.log(1.0 - prob)
	## Life time
	elif return_period != None and prob != None:
		return -return_period * np.log(1.0 - prob)
	## Probability of exceedance
	elif life_time != None and return_period != None:
		return 1.0 - np.exp(-life_time * 1.0 / return_period)
	else:
		raise TypeError("Need to specify 2 parameters")


def as_array(values):
	"""
	Convert values to array if it is not None or already a numpy array
	"""
	if values in ([], None) or values[0] == None:
		values = None
	else:
		values = {True: values, False: np.array(values, dtype='d')}[isinstance(values, np.ndarray)]
	return values


class HazardCurveArray(np.ndarray):
	"""
	Base class for hazard curve array, subclassed from numpy ndarray

	:param data:
		ndarray, n-d float array containing exceedance rates or
		probabilities of exceedance
	"""
	def __new__(cls, data):
		obj = np.asarray(data).view(cls)
		return obj

	def __array_finalize__(self, obj):
		if obj is None:
			return

	@property
	def array(self):
		return np.asarray(self)


class ExceedanceRateArray(HazardCurveArray):
	def to_exceedance_rates(self, timespan=None):
		return self.array

	def to_exceedance_rate_array(self, timespan=None):
		return ExceedanceRateMatrix(self.array)

	def to_probabilities(self, timespan):
		return Poisson(life_time=timespan, return_period=1./self.array)

	def to_probability_array(self, timespan):
		return ProbabilityArray(self.to_probabilities(timespan))

	def to_return_periods(self, timespan=None):
		return 1./self.array

	def mean(self, axis, weights=None):
		if weights is None:
			return self.__class__(np.mean(self.array, axis=axis))
		else:
			return self.__class__(np.average(self.array, axis=axis, weights=weights))

	def mean_and_variance(self, axis, weights=None):
		"""
		Return the weighted average and variance.
		"""
		mean = self.mean(axis, weights=weights)
		if weights is None:
			variance = np.var(self.array, axis=axis)
		else:
			variance = np.average((self.array - mean)**2, axis=axis, weights=weights)
		variance = self.__class__(variance)
		return (mean, variance)

	def scoreatpercentile(self, axis, percentile_levels, weights=None, interpol=True):
		quantile_levels = np.array(percentile_levels, 'f') / 100.
		if weights is None:
			percentiles = np.apply_along_axis(mstats.mquantiles, axis, self.array, quantile_levels)
		else:
			percentiles = np.apply_along_axis(wquantiles, axis, self.array, weights, quantile_levels, interpol)
		## Rotate percentile axis to last position
		# TODO: consider first position for percentile axis
		ndim = self.array.ndim
		percentiles = np.rollaxis(percentiles, axis, ndim)
		return self.__class__(percentiles)


class ProbabilityArray(HazardCurveArray):
	def __add__(self, other):
		assert isinstance(other, ProbabilityArray)
		return ProbabilityArray(1 - ((1 - self.array) * (1 - other.array)))

	def __sub__(self, other):
		assert isinstance(other, ProbabilityArray)
		return ProbabilityArray(1 - ((1 - self.array) / (1 - other.array)))

	def __mul__(self, number):
		assert isinstance(number, (int, float, Decimal))
		return ProbabilityArray(1 - np.exp(np.log(1 - self.array) * float(number)))

	def __div__(self, number):
		assert isinstance(number, (int, float, Decimal))
		return ProbabilityArray(1 - np.exp(np.log(1 - self.array) / float(number)))

	def to_exceedance_rates(self, timespan):
		return 1. / Poisson(life_time=timespan, prob=self.array)

	def to_exceedance_rate_array(self, timespan):
		return ExceedanceRateArray(self.to_exceedance_rates(timespan))

	def to_probabilities(self, timespan=None):
		return self.array

	def to_probability_array(self, timespan=None):
		return ProbabilityArray(self.array)

	def to_return_periods(self, timespan):
		return Poisson(prob=self.array, life_time=timespan)

	def mean(self, axis, weights=None):
		## exceedance probabilities are not additive, but non-exceedance
		## probabilities are multiplicative, so the logs of non-exceedance
		## probabilities are additive. So, we can take the mean of these logs,
		## take the exponent to obtain the mean non-exceedance probability,
		## then convert back to exceedance probability

		# TODO: this is different from openquake.engine.calculators.post_processing,
		# where the simple mean of the poes is computed.
		# In practice, the differences appear to be minor
		return self.__class__(1 - np.exp(np.average(np.log(1 - self.array),
											axis=axis, weights=weights)))

	def mean_and_variance(self, axis, weights=None):
		"""
		Return the weighted average and variance.
		"""
		log_non_exceedance_probs = np.log(1 - self.array)
		mean = np.average(log_non_exceedance_probs, axis=axis, weights=weights)
		_mean = np.expand_dims(mean, axis)
		variance = np.average((log_non_exceedance_probs - _mean)**2, axis=axis, weights=weights)
		#np.sum(weights * (log_non_exceedance_probs - _mean)**2, axis=axis) / np.sum(weights)
		mean = self.__class__(1 - np.exp(mean))
		# TODO: from Wikipedia, but probably not correct
		variance = (np.exp(variance) - 1.) * np.exp(2 * mean.array + variance)
		#variance = np.exp(variance)
		variance = self.__class__(variance)
		return (mean, variance)

	def scoreatpercentile(self, axis, percentile_levels, weights=None, interpol=True):
		quantile_levels = np.array(percentile_levels, 'f') / 100.
		## Note: for probabilities, we need to take 1 - quantile_levels!
		quantile_levels = 1 - quantile_levels
		if weights is None:
			percentiles = np.apply_along_axis(mstats.mquantiles, axis, np.log(1 - self.array), quantile_levels)
		else:
			percentiles = np.apply_along_axis(wquantiles, axis, np.log(1 - self.array), weights, quantile_levels, interpol)
		## Rotate percentile axis to last position
		ndim = self.array.ndim
		percentiles = np.rollaxis(percentiles, axis, ndim)
		return self.__class__(1 - np.exp(percentiles))


class HazardResult:
	"""
	Generic class providing common methods related to poE's, exceedance rates,
	and return periods
	Arguments:
		timespan: time span for probabilities of exceedance (default: 50)
		poes: list or array with probabilities of exceedance (default: None)
		exceedance_rates: list or array with exceedance rates (default: None)
		return_periods: list or array with return periods (default: None)
	"""
	def __init__(self, hazard_values, timespan=50, IMT="PGA", intensities=None, intensity_unit="g"):
		if not isinstance(hazard_values, HazardCurveArray):
			raise Exception("hazard_values should be instance of HazardCurveArray!")
		self._hazard_values = hazard_values
		self.timespan = float(timespan)
		self.IMT = IMT
		self.intensities = as_array(intensities)
		self.intensity_unit = intensity_unit

	@property
	def num_intensities(self):
		return self.intensities.shape[-1]

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

	def get_intensities(self, intensity_unit="g"):
		"""
		Get intensity array, optionally converted to a different intensity unit

		:param intensity_unit:
			string, intensity unit to scale result,
			either "g", "mg", "ms2", "gal" or "cms2" (default: "g")
		"""
		# TODO: take into account self.intensity_unit
		from scipy.constants import g
		conv_factor = None
		if self.IMT in ("PGA", "SA"):
			if self.intensity_unit == "g":
				conv_factor = {"g": 1.0, "mg": 1E+3, "ms2": g, "gal": g*100, "cms2": g*100}[intensity_unit]
			elif self.intensity_unit == "mg":
				conv_factor = {"g": 1E-3, "mg": 1.0, "ms2": g*1E-3, "gal": g*1E-1, "cms2": g*1E-1}[intensity_unit]
			elif self.intensity_unit in ("gal", "cms2"):
				conv_factor = {"g": 0.01/g, "mg": 10./g, "ms2": 1E-2, "gal": 1.0, "cms2": 1.0}[intensity_unit]
			elif self.intensity_unit == "ms2":
				conv_factor = {"g": 1./g, "mg": 1E+3/g, "ms2": 1., "gal": 100.0, "cms2": 100.0}[intensity_unit]
		elif self.IMT == "PGV":
			if self.intensity_unit == "ms2":
				conv_factor = {"ms": 1., "cms": 1E+2}[intensity_unit]
			elif self.intensity_unit == "cms":
				conv_factor = {"ms": 1E-2, "cms": 1.0}[intensity_unit]
		elif self.IMT == "PGD":
			if self.intensity_unit == "m":
				conv_factor = {"m": 1., "cm": 1E+2}
			elif self.intensity_unit == "cm":
				conv_factor = {"m": 1E-2, "cm": 1.0}
		elif self.IMT == "MMI":
			conv_factor = 1.

		if conv_factor is None:
			raise Exception("Unable to convert intensity units!")

		return self.intensities * conv_factor


class HazardSpectrum:
	"""
	Generic class providing common methods related to periods
	"""
	def __init__(self, periods):
		self.periods = as_array(periods)

	def __len__(self):
		return len(self.periods)

	def period_index(self, period_spec):
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

	def reorder_periods(self):
		"""
		"""
		idxs = np.argsort(self.periods)
		self.periods = self.periods[idxs]
		self.intensities = self.intensities.take(idxs, axis=0)
		if self.period_axis != None:
			self._hazard_values = self._hazard_values.take(idxs, axis=self.period_axis)


class HazardField:
	"""
	Generic class providing common methods related to sites
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
		Set site names from a list of SHA sites

		:param sites:
			list with instances of :class:`SHASite`
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
		return np.array([site[0] for site in self.sites])

	@property
	def latitudes(self):
		"""
		Return array with latitudes of all sites
		"""
		return np.array([site[1] for site in self.sites])

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
			Float, minimum longitude (default: None)
		:param lonmax:
			Float, maximum longitude (default: None)
		:param num_cells:
			Integer, number of grid cells
		"""
		#unique_lons = list(set(self.longitudes))
		#unique_lons.sort()
		#return unique_lons
		if lonmin is None:
			lonmin = self.lonmin()
		if lonmax is None:
			lonmax = self.lonmax()
		return np.linspace(lonmin, lonmax, num_cells)

	def get_grid_latitudes(self, latmin=None, latmax=None, num_cells=100):
		"""
		Return array of equally spaced latitudes

		:param latmin:
			Float, minimum latitude (default: None)
		:param latmax:
			Float, maximum latitude (default: None)
		:param num_cells:
			Integer, number of grid cells
		"""
		#unique_lats = list(set(self.latitudes))
		#unique_lats.sort()
		#return unique_lats
		if latmin is None:
			latmin = self.latmin()
		if latmax is None:
			latmax = self.latmax()
		return np.linspace(latmin, latmax, num_cells)

	def meshgrid(self, extent=(None, None, None, None), num_cells=100):
		"""
		Return longitude and latitude matrices of the grid

		:param extent:
			(lonmin, lonmax, latmin, latmax) tuple of floats
		:param num_cells:
			Integer or tuple, number of grid cells in X and Y direction
		"""
		lonmin, lonmax, latmin, latmax = extent
		if isinstance(num_cells, int):
			num_cells = (num_cells, num_cells)
		return np.meshgrid(self.get_grid_longitudes(lonmin, lonmax, num_cells[0]), self.get_grid_latitudes(latmin, latmax, num_cells[1]))

	def get_grid_intensities(self, extent=(None, None, None, None), num_cells=100, method="cubic", intensity_unit="g"):
		"""
		Convert intensities to a spatial grid (2-D array)

		:param extent:
			(lonmin, lonmax, latmin, latmax) tuple of floats
		:param num_cells:
			Integer or tuple, number of grid cells in X and Y direction
		:param method:
			Str, interpolation method supported by griddata (either
			"linear", "nearest" or "cubic") (default: "cubic")
		"""
#		from scipy.interpolate import griddata
#		x, y = self.longitudes, self.latitudes
		lonmin, lonmax, latmin, latmax = extent
		if isinstance(num_cells, int):
			num_cells = (num_cells, num_cells)
		xi, yi = self.meshgrid(extent, num_cells)
#		z = self.get_intensities(intensity_unit=intensity_unit)
		#zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
#		zi = griddata((x, y), z, (xi, yi), method=method)
		zi = self.get_site_intensities(xi, yi, method, intensity_unit)
		return zi

	def get_site_intensities(self, lons, lats, method="cubic", intensity_unit="g"):
		"""
		Interpolate intensities for given sites.

		:param lons:
			array, lons of sites
		:param lats:
			array, lats of sites
		:param method:
			Str, interpolation method supported by griddata (either
			"linear", "nearest" or "cubic") (default: "cubic")
		"""
		from scipy.interpolate import griddata
		x, y = self.longitudes, self.latitudes
		z = self.get_intensities(intensity_unit=intensity_unit)
		zi = griddata((x, y), z, (lons, lats), method=method)
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
		Return index of given site
		"""
		if isinstance(site_spec, int):
			site_index = site_spec
		elif isinstance(site_spec, (str, unicode)):
			site_index = self.site_names.index(site_spec)
		elif isinstance(site_spec, SHASite):
			site_index = self.sites.index(site_spec)
		elif isinstance(site_spec, (list, tuple)) and len(site_spec) >= 2:
			lon, lat = site_spec[:2]
			site = SHASite(lon, lat)
			site_index = self.sites.index(site)
		else:
			raise Exception("Invalid site specification: %s" % site_spec)
		return site_index

	## Note: the following functions are obsolete
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
	def __init__(self, hazard_values, branch_names, weights=None, timespan=50, IMT="PGA", intensities=None, intensity_unit="g", mean=None, percentile_levels=None, percentiles=None):
		HazardResult.__init__(self, hazard_values, timespan=timespan, IMT=IMT, intensities=intensities, intensity_unit=intensity_unit)
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
		if isinstance(branch_spec, int):
			branch_index = branch_spec
		elif isinstance(branch_spec, (str, unicode)):
			branch_index =  self.branch_names.index(branch_spec)
		else:
			raise Exception("Invalid branch specification: %s" % branch_spec)
		return branch_index

	def set_mean(self, mean):
		"""
		Set logic-tree mean, making sure poes are converted to exceedance rates
		"""
		if mean in ([], None):
			self.mean = None
		else:
			if isinstance(self, SpectralHazardCurveFieldTree) and not isinstance(mean, HazardCurveArray):
				raise Exception("mean should be instance of HazardCurveArray!")
			else:
				self.mean = mean

	def set_percentiles(self, percentiles, percentile_levels):
		"""
		Set logic-tree percentiles, making sure poes are converted to exceedance rates
		"""
		self.percentile_levels = as_array(percentile_levels)
		if percentiles in ([], None):
			self.percentiles = None
		else:
			if isinstance(self, SpectralHazardCurveFieldTree) and not isinstance(percentiles, HazardCurveArray):
				raise Exception("percentiles should be instance of HazardCurveArray!")
			else:
				self.percentiles = percentiles

	def weight_sum(self):
		"""
		Return total weight of all branches
		"""
		return np.add.reduce(self.weights)

	def normalize_weights(self):
		"""
		Normalize branch weights
		"""
		self.weights /= self.weight_sum()

	def slice_by_branch_names(self, branch_names, slice_name, normalize_weights=True, strict=True, negate=False, verbose=False):
		"""
		Return a subset (slice) of the logic tree based on branch names
		Parameters:
			branch_names: list of branch names to match
			slice_name: name of this slice
			normalize_weights: boolean indicating whether or not branch weights
				should be renormalized to 1 (default: True)
			strict: boolean indicating whether branch names should be matched
				strictly or only partly (default: True)
			negate: boolean indicating whether match should be negated
				(default: False)
			verbose: boolean indicating whether or not to print extra information
				(default: False)
		Return value:
			An object of the same class as the parent object
		"""
		if strict:
			branch_indexes = [self.branch_index(branch_name) for branch_name in branch_names]
		else:
			branch_indexes = set()
			for branch_name in branch_names:
				for j, tree_branch_name in enumerate(self.branch_names):
					if branch_name in tree_branch_name:
						branch_indexes.add(j)

		if negate:
			print branch_indexes
			all_branch_indexes = set(range(self.num_branches))
			branch_indexes = all_branch_indexes.difference(set(branch_indexes))
			print branch_indexes

		branch_indexes = sorted(branch_indexes)

		if verbose:
			print("Sliced %d branches" % len(branch_indexes))
		return self.slice_by_branch_indexes(branch_indexes, slice_name, normalize_weights=normalize_weights)

	def split_by_branch_name(self, branch_names, normalize_weights=True):
		"""
		Split logic tree in different subsets based on branch names
		Parameters:
			branch_names: list of partial branch names that are unique to each subset
			normalize_weights: boolean indicating whether or not branch weights
				should be renormalized to 1 (default: True)
		Return value:
			A list of objects of the same class as the parent object
		"""
		subsets = []
		for branch_name in branch_names:
			subset = self.slice_by_branch_names([branch_name], branch_name, normalize_weights=normalize_weights, strict=False)
			subsets.append(subset)
		return subsets

	def slice_by_branch_indexes(self, branch_indexes, slice_name, normalize_weights=True):
		"""
		This method needs to be overriden in descendant classes
		"""
		pass


class SpectralHazardCurveFieldTree(HazardTree, HazardField, HazardSpectrum):
	"""
	Class representing a spectral hazard curve field tree, i.e. a number of
	logic-tree branches, each representing a spectral hazard curve field.
	Corresponds to a set of CRISIS .GRA files defining (part of) a logic tree

	Parameters:
		model_name: name of this logic-tree model
		branch_names: 1-D list [j] of model names of each branch
		filespecs: list with full paths to files containing hazard curves
			(1 file for each branch)
		weights: 1-D list or array [j] with branch weights
		sites: 1-D list [i] with (lon, lat) tuples of site for which hazard curves
			were computed
		periods: 1-D array [k] of spectral periods
		IMT: intensity measure type (PGA, SA, PGV or PGD)
		intensities: 2-D array [k,l] of intensity measure levels (ground-motion values)
			for each spectral period for which exceedance rate or probability of
			exceedance was computed
		intensity_unit: unit in which intensity measure levels are expressed:
			PGA and SA: "g", "mg", "ms2", "gal"
			PGV: "cms"
			PGD: "cm"
			default: "g"
		timespan: period related to the probability of exceedance (aka life time)
			(default: 50)
		poes: 4-D array [i,j,k,l] with probabilities of exceedance computed for each
			intensity measure level [k,l] at each site [i] in each branch [j].
			If None, exceedance_rates must be specified
			(default: None)
		exceedance_rates: 4-D array [i,j,k,l] with exceedance rates computed for each
			intensity measure level [k,l] at each site in each branch [j].
			If None, poes must be specified
			(default: None)
		variances: 4-D array [i,j,k,l] with variance of exceedance rate or probability of exceedance
			(default: None)
		mean: 3-D array [i,k,l] with mean exceedance rate or probability of exceedance
			(default: None)
		percentile_levels: 1-D list or array [p] with percentile levels (default: None)
		percentiles: 4-D array [i,k,l,p] with percentiles of exceedance rate or
			probability of exceedance (default: None)
		site_names: list of site names (default: None)

	Provides iteration and indexing over logic-tree branches
	"""
	def __init__(self, model_name, hazard_values, branch_names, filespecs, weights, sites, periods, IMT, intensities, intensity_unit="g", timespan=50, variances=None, mean=None, percentile_levels=None, percentiles=None):
		HazardTree.__init__(self, hazard_values, branch_names, weights=weights, timespan=timespan, IMT=IMT, intensities=intensities, intensity_unit=intensity_unit, mean=mean, percentile_levels=percentile_levels, percentiles=percentiles)
		HazardField.__init__(self, sites)
		HazardSpectrum.__init__(self, periods)
		self.model_name = model_name
		self.filespecs = filespecs
		self.variances = as_array(variances)
		self.period_axis = 2
		self.validate()

	def __iter__(self):
		self._current_index = 0
		return self

	def next(self):
		"""
		Loop over logic-tree branches
		"""
		try:
			branch_name = self.branch_names[self._current_index]
		except:
			raise StopIteration
		else:
			self._current_index += 1
			return self.getSpectralHazardCurveField(self._current_index-1)

	def __getitem__(self, branch_spec):
		return self.getSpectralHazardCurveField(branch_spec)

	@property
	def num_branches(self):
		return self.exceedance_rates.shape[1]

	def validate(self):
		"""
		Check if arrays have correct dimensions
		"""
		if not (len(self.filespecs) == len(self.branch_names)):
			raise Exception("Number of filespecs not in agreement with number of branch names")
		num_branches, num_sites, num_periods = self.num_branches, self.num_sites, self.num_periods
		if len(self.intensities.shape) != 2:
			raise Exception("intensities array has wrong dimension")
		num_intensities = self.num_intensities
		if self.intensities.shape[0] != num_periods:
			raise Exception("intensities array has wrong shape")
		if len(self._hazard_values.shape) != 4:
			raise Exception("hazard array has wrong dimension")
		if self._hazard_values.shape != (num_sites, num_branches, num_periods, num_intensities):
			raise Exception("hazard array has wrong shape")
		if self.variances != None:
			if len(self.variances.shape) != 4:
				raise Exception("variances array has wrong dimension")
			if self.variances.shape != (num_sites, num_branches, num_periods, num_intensities):
				raise Exception("variances array has wrong shape")

	@classmethod
	def from_branches(self, shcf_list, model_name, branch_names=None, weights=None, mean=None, percentile_levels=None, percentiles=None):
		"""
		Construct spectral hazard curve field tree from spectral hazard curve fields
		for different logic-tree branches.

		:param shcf_list:
			list with instances of :class:`SpectralHazardCurveField`
		:param model_name:
			str, model name
		:param branch_names:
			list of branch names (default: None)
		:param weights:
			1-D list or array [j] with branch weights (default: None)
		:param mean:
			instance of :class:`SpectralHazardCurveField`, representing
			mean shcf (default: None)
		:param percentiles:
			list with instances of :class:`SpectralHazardCurveField`,
			representing shcf's corresponding to percentiles (default: None)
		:param percentile_levels:
			list or array with percentile levels (default: None)

		:return:
			instance of :class:`SpectralHazardCurveFieldTree`
		"""
		shcf0 = shcf_list[0]
		num_branches = len(shcf_list)
		num_sites = shcf0.num_sites
		num_periods = shcf0.num_periods
		num_intensities = shcf0.num_intensities

		all_hazard_values = np.zeros((num_sites, num_branches, num_periods, num_intensities), 'd')
		all_hazard_values = shcf0._hazard_values.__class__(all_hazard_values)
		all_hazard_values[:,0,:,:] = shcf0._hazard_values

		if shcf0.variances != None:
			all_variances = np.zeros((num_sites, num_branches, num_periods, num_intensities), 'd')
			all_variances[:,0,:,:] = shcf0.variances
		else:
			all_variances = None

		filespecs = [shcf.filespecs[0] for shcf in shcf_list]
		if branch_names in (None, []):
			branch_names = [shcf.model_name for shcf in shcf_list]
		if weights in (None, []):
			weights = np.ones(num_branches, 'f') / num_branches

		shcft = SpectralHazardCurveFieldTree(model_name, all_hazard_values, branch_names, filespecs, weights, shcf0.sites, shcf0.periods, shcf0.IMT, shcf0.intensities, shcf0.intensity_unit, shcf0.timespan, variances=all_variances)

		for j, shcf in enumerate(shcf_list[1:]):
			shcft.check_shcf_compatibility(shcf)
			shcft._hazard_values[:,j+1] = shcf._hazard_values
			if shcft.variances != None:
				shcft.variances[:,j+1] = shcf.variances

		if mean != None:
			shcft.check_shcf_compatibility(mean)
			shcft.set_mean(mean._hazard_values)

		if percentiles != None:
			num_percentiles = len(percentiles)
			perc_array = np.zeros((num_sites, num_periods, num_intensities, num_percentiles), 'd')
			for p in range(num_percentiles):
				shcf = percentiles[p]
				shcft.check_shcf_compatibility(shcf)
				perc_array[:,:,:,p] = shcf._hazard_values
				perc_array = shcft._hazard_values.__class__(perc_array)
			shcft.set_percentiles(perc_array, percentile_levels)

		return shcft

	def check_shcf_compatibility(self, shcf):
		"""
		Check the compatibility of a candidate branch.

		:param shcf:
			instance of :class:`SpectralHazardCurveField` or higher
		"""
		if self.sites != shcf.sites:
			raise Exception("Sites do not correspond!")
		if (self.periods != shcf.periods).any():
			raise Exception("Spectral periods do not correspond!")
		if self.IMT != shcf.IMT:
			raise Exception("IMT does not correspond!")
		if (self.intensities != shcf.intensities).any():
			raise Exception("Intensities do not correspond!")
		if self.intensity_unit != shcf.intensity_unit:
			raise Exception("Intensity unit does not correspond!")
		if self.timespan != shcf.timespan:
			raise Exception("Time span does not correspond!")
		if self._hazard_values.__class__ != shcf._hazard_values.__class__:
			raise Exception("Hazard array does not correspond!")

	def append_branch(self, shcf, branch_name="", weight=1.0):
		"""
		Append a new branch
		Parameters:
			shcf: SpectralHazardCurveField object
			branch_name: name of branch. If not specified, shcf.model_name
				will be used as the branch name (default: "")
			weight:
				branch weight (default: 1.0)
		Notes:
			Branch weights are not renormalized to avoid rounding errors.
				This should be done after all branches have been appended.
			Mean and percentiles can be appended with the set_mean() and
				set_percentiles() methods.
		"""
		self.check_shcf_compatibility(shcf)
		if not branch_name:
			branch_name = shcf.model_name
		self.branch_names.append(branch_name)
		self.filespecs.append(shcf.filespecs[0])
		## Do not recompute weights, assume they are correct
		self.weights = np.concatenate([self.weights, [weight]])
		shape = (self.num_sites, 1, self.num_periods, self.num_intensities)
		hazard_values = np.concatenate([self._hazard_values, shcf._hazard_values.reshape(shape)], axis=1)
		self._hazard_values = self._hazard_values.__class__(hazard_values)
		if self.variances != None:
			## Note: this is only correct if both shcft and shcf are of the same type
			## (exceedance rates or probabilities of exceedance)
			self.variances = np.concatenate([self.variances, shcf.variances.reshape(shape)], axis=1)

	def extend(self, shcft):
		"""
		Extend spectral hazard curve field tree in-place with another one

		:param shcft:
			instance of :class:`SpectralHazardCurveFieldTree`
		"""
		self.check_shcf_compatibility(shcft)
		self.branch_names.extend(shcft.branch_names)
		if shcft.filespecs:
			self.filespecs.extend(shcft.filespecs)
		else:
			self.filespecs = []
		self.weights = np.concatenate([self.weights, shcft.weights])
		hazard_values = np.concatenate([self._hazard_values, shcft._hazard_values], axis=1)
		self._hazard_values = self._hazard_values.__class__(hazard_values)
		if self.variances != None:
			variances = np.concatenate([self.variances, shcft.variances])
			self.variances = self.variances.__class__(variances)
		## Remove mean and percentiles
		self.mean = None
		self.percentiles = None
		self.normalize_weights()

	def getSpectralHazardCurveField(self, branch_spec=0):
		"""
		Return spectral hazard curve field for a particular branch
		Parameters:
			branch_spec: branch specification (index or branch name) (default: 0)
		Return value:
			SpectralHazardCurveField object
		"""
		branch_index = self.branch_index(branch_spec)
		try:
			branch_name = self.branch_names[branch_index]
		except:
			raise IndexError("Branch index %s out of range" % branch_index)
		else:
			branch_name = self.branch_names[branch_index]
			filespec = self.filespecs[branch_index]
			hazard_values = self._hazard_values[:,branch_index,:,:]
			if self.variances != None:
				variances = self.variances[:,branch_index,:,:]
			else:
				variances = None
			return SpectralHazardCurveField(branch_name, hazard_values, [filespec]*self.num_periods, self.sites, self.periods, self.IMT, self.intensities, self.intensity_unit, self.timespan, variances=variances)

	def getSpectralHazardCurve(self, branch_spec=0, site_spec=0):
		"""
		Return spectral hazard curve for a particular branch and site
		Parameters:
			branch_spec: branch specification (index or branch name) (default: 0)
			site_spec: site specification (site index, (lon, lat) tuple or site name)
				(default: 0)
		Return value:
			SpectralHazardCurveField object
		"""
		site_index = self.site_index(site_spec)
		try:
			site = self.sites[site_index]
		except:
			raise IndexError("Site index %s out of range" % site_index)
		else:
			site_name = self.site_names[site_index]

		branch_index = self.branch_index(branch_spec)
		try:
			branch_name = self.branch_names[branch_index]
		except:
			raise IndexError("Branch index %s out of range" % branch_index)

		intensities = self.intensities
		hazard_values = self._hazard_values[site_index, branch_index]
		if self.variances != None:
			variances = self.variances[site_index, branch_index]
		else:
			variances = None
		return SpectralHazardCurve(branch_name, hazard_values, self.filespecs[branch_index], site, self.periods, self.IMT, intensities, self.intensity_unit, self.timespan, variances=variances, site_name=site_name)

	def min(self):
		# TODO: does this make sense? Makes more sense with 1 period and 1 site
		return self._hazard_values.min(axis=1)

	def max(self):
		return self._hazard_values.max(axis=1)

	def calc_mean(self, weighted=True):
		"""
		Compute mean exceedance rates
		Parameters:
			weighted: boolean indicating whether or not branch weights should be
				taken into account (default: True)
		Return value:
			mean exceedance rates: 3-D array [i,k,l]
		"""
		if weighted:
			weights = self.weights
		else:
			weights = None
		return self._hazard_values.mean(axis=1, weights=weights)

	def calc_variance_epistemic(self, weighted=True):
		"""
		Compute variance of hazard curves
		"""
		if weighted:
			weights = self.weights
		else:
			weights = np.ones(len(self))
		return self._hazard_values.mean_and_variance(axis=1, weights=weights)[1]

	def calc_variance_of_mean(self, weighted=True):
		"""
		Compute variance of mean exceedance rate
		Parameters:
			weighted: boolean indicating whether or not branch weights should be
				taken into account (default: True)
		Return value:
			variance of mean exceedance rate: 3-D array [i,k,l]
		"""
		if weighted and not self.weights in ([], None):
			# TODO: this needs to be checked
			mean = self.calc_mean(weighted=True)
			weights = np.array(self.weights)
			weights_column = weights.reshape((self.num_branches, 1))
			variance_of_mean = np.zeros((self.num_sites, self.num_periods, self.num_intensities), 'd')
			for i in range(self.num_sites):
				for k in range(self.num_periods):
					variance_of_mean[i,k] = np.add.reduce(weights_column * (self.exceedance_rates[i,:,k] - mean[i,k])**2, axis=0)
		else:
			 variance_of_mean = np.var(self.exceedance_rates, axis=1)
		return variance_of_mean

	def calc_mean_variance(self, weighted=True):
		"""
		Compute mean variance of exceedance rate
		Parameters:
			weighted: boolean indicating whether or not branch weights should be
				taken into account (default: True)
		Return value:
			mean variance of exceedance rate: 3-D array [i,k,l]
		"""
		if self.variances != None:
			if weighted:
				mean_variance = np.average(self.variances, weights=self.weights, axis=1)
			else:
				mean_variance = np.mean(self.variances, axis=1)
		else:
			mean_variance = None
		return mean_variance

	def calc_percentiles_epistemic(self, percentile_levels=[], weighted=True, interpol=True):
		"""
		Compute percentiles of exceedance rate (epistemic uncertainty)

		:param percentile_levels:
			list or array of percentile levels. Percentiles
			may be specified as integers between 0 and 100 or as floats
			between 0 and 1
		:param weighted:
			boolean indicating whether or not branch weights should be
			taken into account (default: True)
		:param interpol:
			bool, whether or not percentile intercept should be
			interpolated. Only applies to weighted percentiles
			(default: True)

		:return:
			percentiles of hazard values: 4-D array [i,k,l,p]
		"""
		if percentile_levels in ([], None):
			percentile_levels = [5, 16, 50, 84, 95]
		num_sites, num_periods, num_intensities = self.num_sites, self.num_periods, self.num_intensities
		num_percentiles = len(percentile_levels)
		#percentiles = np.zeros((num_sites, num_periods, num_intensities, num_percentiles))
		if weighted and self.weights != None and len(set(self.weights)) > 1:
			#for i in range(num_sites):
			#	for k in range(num_periods):
			#		for l in range(num_intensities):
			#			pmf = NumericPMF.from_values_and_weights(self.exceedance_rates[i,:,k,l], self.weights)
			#			percentiles[i,k,l,:] = pmf.get_percentiles(percentile_levels, interpol=interpol)
			weights = self.weights
		else:
			#for i in range(num_sites):
			#	for k in range(num_periods):
			#		for l in range(num_intensities):
			#			for p, per in enumerate(percentile_levels):
			#				percentiles[i,k,l,p] = scoreatpercentile(self.exceedance_rates[i,:,k,l], per)
			weights = None
		percentiles = self._hazard_values.scoreatpercentile(1, percentile_levels, weights=weights)
		#percentiles = ExceedanceRateArray(percentiles)
		#if isinstance(self._hazard_values, ProbabilityArray):
		#	percentiles = percentiles.to_probability_array(self.timespan)
		return percentiles

	def calc_percentiles_combined(self, percentile_levels, weighted=True):
		"""
		Compute percentiles of exceedance rate (combined uncertainty)
		Can only be computed if variances (aleatory uncertainty) is known
		Parameters:
			percentile_levels: list or array of percentile levels. Percentiles
				may be specified as integers between 0 and 100 or as floats
				between 0 and 1
			weighted: boolean indicating whether or not branch weights should be
				taken into account (default: True)
		Return value:
			percentiles of exceedance rate: 4-D array [i,k,l,p]
		"""
		if self.variances is None:
			raise Exception("Combined uncertainties can only be computed if aleatory uncertainties are stored as variances")
		else:
			# TODO !
			return self.calc_percentiles_epistemic(percentile_levels, weighted=weighted)

	def calc_percentiles(self, percentile_levels, weighted=True):
		"""
		Wrapper function to compute percentiles of exceedance rate
		(combined uncertainty if variances is defined, epistemic otherwise)
		Parameters:
			percentile_levels: list or array of percentile levels. Percentiles
				may be specified as integers between 0 and 100 or as floats
				between 0 and 1
			weighted: boolean indicating whether or not branch weights should be
				taken into account (default: True)
		Return value:
			percentiles of exceedance rate: 4-D array [i,k,l,p]
		"""
		if self.variances is None:
			print("Epistemic")
			return self.calc_percentiles_epistemic(percentile_levels, weighted=weighted)
		else:
			print("Combined")
			return self.calc_percentiles_combined(percentile_levels, weighted=weighted)

	def getMeanSpectralHazardCurveField(self, recalc=False, weighted=True):
		"""
		Return mean spectral hazard curve field
		Parameters:
			recalc: boolean indicating whether or not to recompute. If mean is
				None, computation will be performed anyway (default: False)
			weighted: boolean indicating whether or not branch weights should be
				taken into account. Only applies if recomputed (default: True)
		Return value:
			SpectralHazardCurveField object
		"""
		if recalc or self.mean is None:
			mean = self.calc_mean(weighted=weighted)
		else:
			mean = self.mean
		variances = self.calc_variance_of_mean()
		model_name = "Mean(%s)" % self.model_name
		return SpectralHazardCurveField(model_name, mean, [""]*self.num_periods, self.sites, self.periods, self.IMT, self.intensities, self.intensity_unit, self.timespan, variances=variances)

	def getPercentileSpectralHazardCurveField(self, perc, recalc=True, weighted=True):
		if recalc or self.percentiles is None or not perc in self.percentiles:
			hazard_values = self.calc_percentiles([perc], weighted=weighted)[:,:,:,0]
		else:
			print "No recalculaton!"
			perc_index = self.percentile_levels.index(perc)
			hazard_values = self.percentiles[:,:,:,perc_index]

		model_name = "Perc%02d(%s)" % (perc, self.model_name)
		return SpectralHazardCurveField(model_name, hazard_values, [""]*self.num_periods, self.sites, self.periods, self.IMT, self.intensities, self.intensity_unit, self.timespan, variances=None)

	def import_stats_from_AGR(self, agr_filespec, percentile_levels=None):
		"""
		Import logic-tree statistics from a CRISIS .AGR file
		Parameters:
			agr_filespec: full path to .AGR file
			percentile_levels: list or array of percentile levels to import
				(default: None)
		"""
		# TODO: need to take care with intensity_unit
		from ..crisis import IO
		shcft = IO.read_GRA(agr_filespec)
		if shcft.intensities != self.intensities:
			raise Exception("Intensities do not match with those of current object!")
		self.mean = shcft.mean
		if percentile_levels is None:
			self.percentile_levels = shcft.percentile_levels
			self.percentiles = shcft.percentiles
		else:
			perc_indexes = []
			for perc in percentile_levels:
				try:
					perc_index = np.where(shcft.percentile_levels == perc)[0][0]
				except:
					raise Exception("Percentile level %s not found in file %s!" % (perc, agr_filespec))
				perc_indexes.append(perc_index)
			self.percentile_levels = percentiles
			self.percentiles = shcft.percentiles[:,:,:,perc_indexes]

	def export_stats_AGR(self, out_filespec, weighted=True):
		"""
		Export logic-tree statistics to a CRISIS .AGR file
		Parameters:
			out_filespec: full path to output .AGR file
			weighted: boolean indicating whether or not branch weights should be
				taken into account (default: True)
		"""
		if self.mean in (None, []):
			mean = self.calc_mean(weighted=weighted)
		else:
			mean = self.mean
		variance_of_mean = self.calc_variance_of_mean(weighted=weighted)
		if self.percentiles in (None, []):
			if self.percentile_levels in (None, []):
				percentile_levels = [5, 16, 50, 84, 95]
			else:
				percentile_levels = self.percentile_levels
			percentiles = self.calc_percentiles(percentile_levels, weighted=weighted)
		else:
			percentiles = self.percentiles
			percentile_levels = self.percentile_levels

		f = open(out_filespec, "w")
		f.write("************************************************************\n")
		f.write("Logic-tree statistics: mean, variance, percentiles (%s)" % ", ".join(["%d" % p for p in percentile_levels]))
		f.write("\n")
		f.write("Calculated using ROB python routines\n")
		f.write("NumSites, NumPeriods, NumIntensities: %d, %d, %d\n" % (self.num_sites, self.num_periods, self.num_intensities))
		f.write("************************************************************\n")
		f.write("\n\n")
		for i in range(self.num_sites):
			f.write("    %s      %s\n" % self.sites[i])
			for k in range(self.num_periods):
				f.write("INTENSITY %d T=%s\n" % (k+1, self.periods[k]))
				for l in range(self.num_intensities):
					values = [self.intensities[k,l]] + [mean[i,k,l]] + [variance_of_mean[i,k,l]] + list(percentiles[i,k,l,:])
					str = "  ".join(["%.5E" % val for val in values])
					f.write("%s\n" % str)
		f.close()

	def slice_by_branch_indexes(self, branch_indexes, slice_name, normalize_weights=True):
		"""
		Return a subset (slice) of the logic tree based on branch indexes
		Parameters:
			branch_indexes: list or array of branch indexes
			slice_name: name of this slice
			normalize_weights: boolean indicating whether or not branch weights
				should be renormalized to 1 (default: True)
		Return value:
			SpectralHazardCurveFieldTree object
		"""
		model_name = slice_name
		branch_names, filespecs = [], []
		for index in branch_indexes:
			branch_names.append(self.branch_names[index])
			filespecs.append(self.filespecs[index])
		weights = self.weights[branch_indexes]
		## Recompute branch weights
		if normalize_weights:
			weight_sum = np.add.reduce(weights)
			weights /= weight_sum
		sites = self.sites
		periods = self.periods
		IMT = self.IMT
		intensities = self.intensities
		intensity_unit = self.intensity_unit
		timespan = self.timespan
		hazard_values = self._hazard_values[:,branch_indexes,:,:]
		if self.variances != None:
			variances = self.variances[:,branch_indexes,:,:]
		else:
			variances = None
		return SpectralHazardCurveFieldTree(model_name, hazard_values, branch_names, filespecs, weights, sites, periods, IMT, intensities, intensity_unit, timespan, variances=variances)

	def interpolate_return_period(self, return_period):
		"""
		Interpolate intensity measure levels for given return period
		Parameters:
			return_period: return period
		Return value:
			UHSFieldTree object
		"""
		num_sites, num_periods, num_branches = self.num_sites, self.num_periods, self.num_branches
		rp_intensities = np.zeros((num_sites, num_branches, num_periods), dtype='d')
		if self.mean not in (None, []):
			rp_mean = np.zeros((num_sites, num_periods), dtype='d')
		else:
			rp_mean = None
		if self.percentiles not in (None, []):
			rp_percentiles = np.zeros((num_sites, num_periods, self.num_percentiles), dtype='d')
		else:
			rp_percentiles = None
		interpol_exceedance = 1. / return_period
		for i in range(num_sites):
			for k in range(num_periods):
				for j in range(num_branches):
					rp_intensities[i,j,k] = interpolate(self.exceedance_rates[i,j,k], self.intensities[k], [interpol_exceedance])[0]
				if self.mean not in (None, []):
					rp_mean[i,k] = interpolate(self.mean[i,k].to_exceedance_rates(self.timespan), self.intensities[k], [interpol_exceedance])[0]
				if self.percentiles not in (None, []):
					for p in range(self.num_percentiles):
						rp_percentiles[i,k,p] = interpolate(self.percentiles[i,k,:,p].to_exceedance_rates(self.timespan), self.intensities[k], [interpol_exceedance])[0]
		return UHSFieldTree(self.model_name, self.branch_names, self.filespecs, self.weights, self.sites, self.periods, self.IMT, rp_intensities, self.intensity_unit, self.timespan, return_period=return_period, mean=rp_mean, percentile_levels=self.percentile_levels, percentiles=rp_percentiles)

	def interpolate_periods(self, out_periods):
		"""
		Interpolate intensity measure levels at different spectral periods
		Parameters:
			out_periods: list or array of output spectral periods
		Return value:
			SpectralHazardCurveFieldTree object
		"""
		num_sites, num_branches, num_intensities = self.num_sites, self.num_branches, self.num_intensities
		out_hazard_values = np.zeros((num_sites, num_branches, len(out_periods), num_intensities), dtype='d')
		if self.variances != None:
			out_variances = np.zeros((num_sites, num_branches, len(out_periods), num_intensities), dtype='d')
		else:
			out_variances = None
		if self.mean != None:
			out_mean = np.zeros((num_sites, len(out_periods), num_intensities), dtype='d')
		else:
			out_mean = None
		if self.percentiles != None:
			num_percentiles = self.num_percentiles
			out_percentiles = np.zeros((num_sites, len(out_periods), num_intensities, num_percentiles), dtype='d')

		for i in range(num_sites):
			for j in range(num_branches):
				shc = self.getSpectralHazardCurve(site_spec=i, branch_spec=j)
				shc_out = shc.interpolate_periods(out_periods)
				out_hazard_values[i,j] = shc_out._hazard_values
				if self.variances != None:
					out_variances[i,j] = shc_out.variances
			if self.mean != None:
				shc = SpectralHazardCurve("mean", self.mean[i], "", self.periods, self.IMT, self.intensities, self.intensity_unit, self.timespan)
				shc_out = shc.interpolate_periods(out_periods)
				out_mean[i] = shc_out._hazard_values
			if self.percentiles != None:
				for p in range(num_percentiles):
					shc = SpectralHazardCurve("mean", self.percentiles[i,:,:,p], "", self.periods, self.IMT, self.intensities, self.intensity_unit, self.timespan)
					shc_out = shc.interpolate_periods(out_periods)
					out_percentiles[i,:,:,p] = shc_out._hazard_values
		intensities = shc_out.intensities
		return SpectralHazardCurveFieldTree(self.model_name, out_hazard_values, self.branch_names, self.filespecs, self.weights, self.sites, out_periods, self.IMT, intensities, self.intensity_unit, self.timespan, variances=out_variances)

	def plot(self, site_spec=0, period_spec=0, branch_specs=[], fig_filespec=None, title=None, want_recurrence=False, want_poe=False, interpol_rp=None, interpol_prob=None, interpol_rp_range=None, amax=None, rp_max=1E+07, legend_location=0, lang="en"):
		"""
		Plot hazard curves (individual branches, mean, and percentiles) for a
			particular site and spectral period.
		Parameters:
			site_spec: site specification (index, (lon,lat) tuple or site name)
				of site to be plotted (default: 0)
			period_spec: period specification (integer period indexe or float
				spectral period) (default: 0)
			branch_specs: list of branch specifications (indexes or branch names)
				to be plotted (default: [] will plot all branches)
		"""
		site_index = self.site_index(site_spec)
		period_index = self.period_index(period_spec)
		if branch_specs in ([], None):
			branch_indexes = range(self.num_branches)
		else:
			branch_indexes = [self.branch_index(branch_spec) for branch_spec in branch_specs]
		x = self.intensities[period_index]
		datasets, labels, colors, linewidths, linestyles = [], [], [], [], []

		if title is None:
			title = "Hazard Curve Tree"
			title += "\nSite: %s, T: %s s" % (self.site_names[site_index], self.periods[period_index])

		## Plot individual models
		exceedance_rates = self.exceedance_rates
		for branch_index in branch_indexes:
			y = exceedance_rates[site_index, branch_index, period_index]
			datasets.append((x, y))
			labels.append("_nolegend_")
			colors.append((0.5, 0.5, 0.5))
			linewidths.append(1)
			linestyles.append('-')

		## Plot overall mean
		if self.mean is None:
			y = self.calc_mean()[site_index, period_index].to_exceedance_rates(self.timespan)
		else:
			y = self.mean[site_index, period_index].to_exceedance_rates(self.timespan)
		datasets.append((x, y))
		labels.append("_nolegend_")
		colors.append('w')
		linewidths.append(5)
		linestyles.append('-')
		datasets.append((x, y))
		labels.append({"en": "Overall Mean", "nl": "Algemeen gemiddelde"}[lang])
		colors.append('r')
		linewidths.append(3)
		linestyles.append('-')

		## Plot percentiles
		if self.percentiles is None:
			if self.percentile_levels is None:
				percentile_levels = [5, 16, 50, 84, 95]
			else:
				percentile_levels = self.percentile_levels
			percentiles = self.calc_percentiles(percentile_levels, weighted=True)
		else:
			percentiles = self.percentiles
			percentile_levels = self.percentile_levels
		percentiles = percentiles[site_index, period_index]
		## Manage percentile labels and colors
		perc_labels, perc_colors = {}, {}
		p = 0
		for perc in percentile_levels:
			if not perc_labels.has_key(perc):
				if not perc_labels.has_key(100 - perc):
					perc_labels[perc] = "P%02d" % perc
					perc_colors[perc] = ["b", "g", "r", "c", "m", "k"][p%6]
					p += 1
				else:
					perc_labels[100 - perc] += ", P%02d" % perc
					perc_labels[perc] = "_nolegend_"
					perc_colors[perc] = perc_colors[100 - perc]
		for p, perc in enumerate(percentile_levels):
			labels.append(perc_labels[perc])
			colors.append(perc_colors[perc])
			linewidths.append(2)
			linestyles.append('--')
			datasets.append((x, percentiles[:,p].to_exceedance_rates(self.timespan)))
		fixed_life_time = {True: self.timespan, False: None}[want_poe]
		intensity_unit = self.intensity_unit
		plot_hazard_curve(datasets, labels=labels, colors=colors, linestyles=linestyles, linewidths=linewidths, fig_filespec=fig_filespec, title=title, want_recurrence=want_recurrence, fixed_life_time=fixed_life_time, interpol_rp=interpol_rp, interpol_prob=interpol_prob, interpol_rp_range=interpol_rp_range, amax=amax, intensity_unit=intensity_unit, tr_max=rp_max, legend_location=legend_location, lang=lang)

	plot.__doc__ += common_plot_docstring

	def plot_subsets(self, subset_label_patterns, site_spec=0, period_spec=0, labels=[], agr_filespecs=[], percentile_levels=[84], combined_uncertainty=True, fig_filespec=None, title=None, want_recurrence=False, want_poe=False, interpol_rp=None, interpol_prob=None, interpol_rp_range=None, amax=None, rp_max=1E+07, legend_location=0, lang="en"):
		"""
		Plot mean and percentiles of different subsets
		Parameters:
			subset_label_patterns: list of strings that are unique to the branch
				 labels of each subset
			site_spec: site specification (index, (lon,lat) tuple or site name)
				of site to be plotted (default: 0)
			period_spec: period specification (integer period indexe or float
				spectral period) (default: 0)
			labels: subset labels (default: [])
			agr_filespecs: list of .AGR filespecs containing statistics for each
				subset. If empty, mean and percentiles will be computed
			percentile_levels: list of exceedance-rate percentiles to plot in addition
				 to the mean (default: [84])
			combined_uncertainty: boolean. If True, percentiles are calculated for combined
				(epistemic + aleatory) uncertainty. If False, percentiles are calculated for
				epistemic uncertainty only. This setting does not apply if agr_filespec is
				set. Note that this setting does not influence the mean value.
				(default: True)
		"""
		subsets = self.split_by_branch_name(subset_specs)
		site_index = self.site_index(site_spec)
		period_index = self.period_index(period_spec)

		dataset_colors = ["r", "g", "b", "c", "m", "k"]
		if not labels:
			dataset_labels = ["Subset %d" % (i+1) for i in range(len(subsets))]
		else:
			dataset_labels = labels

		## Manage percentile labels and linestyles
		perc_labels, perc_linestyles = {}, {}
		p = 0
		for perc in percentile_levels:
			if not perc_labels.has_key(perc):
				if not perc_labels.has_key(100 - perc):
					perc_labels[perc] = "P%02d" % perc
					perc_linestyles[perc] = ["--", ":", "-:"][p%3]
					p += 1
				else:
					perc_labels[100 - perc] += ", P%02d" % perc
					perc_labels[perc] = "_nolegend_"
					perc_linestyles[perc] = perc_colors[100 - perc]

		x = self.intensities[period_index]

		datasets, labels, colors, linewidths, linestyles = [], [], [], [], []
		for i, subset in enumerate(subsets):
			## Load or calculate statistics
			if not agr_filespecs:
				mean = subset.calc_mean()
				if combined_uncertainty:
					percentiles = subset.calc_percentiles_combined(percentile_levels)
				else:
					percentiles = subset.calc_percentiles_epistemic(percentile_levels)
			else:
				shcft = import_stats_from_AGR(agr_filespecs[i], percentile_levels)
				mean = shcft.mean
				percentiles = shcft.percentiles

			## Plot subset mean
			label = dataset_labels[i]
			label += {"en": " (mean)", "nl": " (gemiddelde)"}[lang]
			labels.append(label)
			colors.append(dataset_colors[i%len(dataset_colors)])
			linewidths.append(2)
			linestyles.append('-')
			datasets.append((x, mean[site_index,period_index]))

			## Plot percentiles
			for p, perc in enumerate(percentile_levels):
				perc_label = perc_labels[perc]
				labels.append(dataset_labels[i] + " (%s)" % perc_label)
				colors.append(dataset_colors[i%len(dataset_colors)])
				linewidths.append(2)
				linestyles.append(perc_linestyles[perc])
				datasets.append((x, percentiles[site_index,period_index,:,p]))

		if amax is None:
			amax = max(self.intensities[period_index])

		## Interpolate
		if interpol_rp:
			interpol_rp = [interpol_rp] + [0] * len(percentile_levels)
			interpol_rp *= len(subsets)
		if interpol_prob:
			interpol_prob = [interpol_prob] + [0] * len(percentile_levels)
			interpol_prob *= len(subsets)

		## Call plot function
		fixed_life_time = {True: self.timespan, False: None}[want_poe]
		intensity_unit = self.intensity_unit
		plot_hazard_curve(datasets, labels=labels, colors=colors, linestyles=linestyles, linewidths=linewidths, fig_filespec=fig_filespec, title=title, want_recurrence=want_recurrence, fixed_life_time=fixed_life_time, interpol_rp=interpol_rp, interpol_prob=interpol_prob, amax=amax, tr_max=rp_max, legend_location=legend_location, lang=lang)

	plot_subsets.__doc__ += common_plot_docstring

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML SpectralHazardCurveField element)
		Arguments:
			encoding: unicode encoding (default: 'latin1')
		"""
		# TODO: add names to nrml namespace
		shcft_elem = etree.Element(ns.SPECTRAL_HAZARD_CURVE_FIELD_TREE)
		shcft_elem.set(ns.NAME, self.model_name)
		shcft_elem.set(ns.IMT, self.IMT)
		for j, branch_name in enumerate(self.branch_names):
			shcf_elem = etree.SubElement(shcft_elem, ns.SPECTRAL_HAZARD_CURVE_FIELD)
			shcf_elem.set(ns.NAME, branch_name)
			for k, period in enumerate(self.periods):
				hcf_elem = etree.SubElement(shcf_elem, ns.HAZARD_CURVE_FIELD)
				hcf_elem.set(ns.PERIOD, str(period))
				imls_elem = etree.SubElement(hcf_elem, ns.IMLS)
				imls_elem.text = " ".join(map(str, self.intensities[k,:]))
				for i, site in enumerate(self.sites):
					hazard_curve_elem = etree.SubElement(hcf_elem, ns.HAZARD_CURVE)
					point_elem = etree.SubElement(hazard_curve_elem, ns.POINT)
					position_elem = etree.SubElement(point_elem, ns.POSITION)
					position_elem.text = "%s %s" % (site[0], site[1])
					poes_elem = etree.SubElement(hazard_curve_elem, ns.POES)
					poes_elem.text = " ".join(map(str, self.poes[i,j,k,:]))
		return shcf_elem

	def write_nrml(self, filespec, encoding='latin1', pretty_print=True):
		"""
		Write spectral hazard curve field tree to XML file
		Arguments:
			filespec: full path to XML output file
			encoding: unicode encoding (default: 'utf-8')
			pretty_print: boolean indicating whether or not to indent each
				element (default: True)
		"""
		tree = create_nrml_root(self, encoding=encoding)
		fd = open(filespec, "w")
		tree.write(fd, xml_declaration=True, encoding=encoding, pretty_print=pretty_print)
		fd.close()


class SpectralHazardCurveField(HazardResult, HazardField, HazardSpectrum):
	"""
	Class representing a hazard curve field for different spectral periods.
	Corresponds to 1 CRISIS .GRA file.
	Parameters:
		model_name: name of this hazard-curve model
		filespecs: list with full paths to files containing hazard curves
			(1 file for each spectral period)
		sites: 1-D list [i] with (lon, lat) tuples of site for which hazard curves
			were computed
		periods: 1-D array [k] of spectral periods
		IMT: intensity measure type (PGA, SA, PGV or PGD)
		intensities: 2-D array [k,l] of intensity measure levels (ground-motion values)
			for each spectral period for which exceedance rate or probability of
			exceedance was computed
		intensity_unit: unit in which intensity measure levels are expressed:
			PGA and SA: "g", "mg", "ms2", "gal"
			PGV: "cms"
			PGD: "cm"
			default: "g"
		timespan: period related to the probability of exceedance (aka life time)
			(default: 50)
		poes: 3-D array [i,k,l] with probabilities of exceedance computed for each
			intensity measure level [k,l] at each site [i].
			If None, exceedance_rates must be specified
			(default: None)
		exceedance_rates: 3-D array [i,k,l] with exceedance rates computed for each
			intensity measure level [k,l] at each site.
			If None, poes must be specified
			(default: None)
		variances: 3-D array [i,k,l] with variance of exceedance rate or probability of exceedance
			(default: None)
		site_names: list of site names (default: None)
	"""
	def __init__(self, model_name, hazard_values, filespecs, sites, periods, IMT, intensities, intensity_unit="g", timespan=50, variances=None):
		HazardResult.__init__(self, hazard_values, timespan=timespan, IMT=IMT, intensities=intensities, intensity_unit=intensity_unit)
		HazardField.__init__(self, sites)
		HazardSpectrum.__init__(self, periods)
		self.model_name = model_name
		self.filespecs = filespecs
		self.variances = as_array(variances)
		self.period_axis = 1
		self.validate()

	def __add__(self, other_shcf):
		"""
		:param other_shc:
			instance of :class:`SpectralHazardCurve`

		:return:
			instance of :class:`SpectralHazardCurve`
		"""
		assert isinstance(other_shcf, SpectralHazardCurveField)
		assert self.sites == other_shcf.sites
		assert self.IMT == other_shcf.IMT
		assert (self.periods == other_shcf.periods).all()
		assert (self.intensities == other_shcf.intensities).all()
		assert self.intensity_unit == other_shcf.intensity_unit
		assert self.timespan == other_shcf.timespan
		hazard_values = self._hazard_values + other_shcf._hazard_values
		return self.__class__(self.model_name, hazard_values, self.filespecs, self.sites, self.periods, self.IMT, self.intensities, self.intensity_unit, self.timespan)

	def __mul__(self, number):
		"""
		:param number:
			int, float or Decimal

		:return:
			instance of :class:`SpectralHazardCurve`
		"""
		assert isinstance(number, (int, float, Decimal))
		hazard_values = self._hazard_values * number
		return self.__class__(self.model_name, hazard_values, self.filespecs, self.sites, self.periods, self.IMT, self.intensities, self.intensity_unit, self.timespan)

	def __rmul__(self, number):
		return self.__mul__(number)

	@classmethod
	def from_hazard_curve_fields(self, hcf_list, model_name):
		"""
		Construct spectral hazard curve field from hazard curve fields
		for different spectral periods.

		:param hcf_list:
			list with instances of :class:`HazardCurveField`
		:param model_name:
			str, model name

		:return:
			instance of :class:`SpectralHazardCurveField`
		"""
		hcf0 = hcf_list[0]
		filespecs = [hcf.filespec for hcf in hcf_list]
		sites = hcf0.sites
		num_sites = hcf0.num_sites
		periods = [hcf.period for hcf in hcf_list]
		num_periods = len(periods)
		IMT = hcf_list[-1].IMT
		num_intensities = hcf0.num_intensities
		intensity_unit = hcf0.intensity_unit
		timespan = hcf0.timespan
		hazard_values = hcf0._hazard_values.__class__(np.zeros((num_sites, num_periods, num_intensities)))
		intensities = np.zeros((num_periods, num_intensities))
		# TODO: variances
		for i, hcf in enumerate(hcf_list):
			hazard_values[:,i,:] = hcf._hazard_values
			intensities[i] = hcf.intensities
		return SpectralHazardCurveField(model_name, hazard_values, filespecs, sites, periods, IMT, intensities, intensity_unit, timespan, variances=None)

	def __iter__(self):
		self._current_index = 0
		return self

	def next(self):
		"""
		Loop over sites
		"""
		try:
			site = self.sites[self._current_index]
		except:
			raise StopIteration
		else:
			self._current_index += 1
			return self.getSpectralHazardCurve(self._current_index-1)

	def __getitem__(self, site_spec):
		return self.getSpectralHazardCurve(site_spec)

	def validate(self):
		"""
		Check if arrays have correct dimensions
		"""
		num_sites, num_periods, num_intensities = self.num_sites, self.num_periods, self.num_intensities
		if len(self.intensities.shape) != 2:
			raise Exception("intensities array has wrong dimension")
		if self.intensities.shape[0] != num_periods:
			raise Exception("intensities array has wrong shape")
		if len(self._hazard_values.shape) != 3:
			raise Exception("exceedance_rates or poes array has wrong dimension")
		if self._hazard_values.shape != (num_sites, num_periods, num_intensities):
			raise Exception("exceedance_rates or poes array has wrong shape")
		if self.variances != None:
			if len(self.variances.shape) != 3:
				raise Exception("variances array has wrong dimension")
			if self.variances.shape != (num_sites, num_periods, num_intensities):
				raise Exception("variances array has wrong shape")

	def getHazardCurveField(self, period_spec=0):
		period_index = self.period_index(period_spec)
		try:
			period = self.periods[period_index]
		except:
			raise IndexError("Period index %s out of range" % period_index)
		intensities = self.intensities[period_index]
		hazard_values = self._hazard_values[:,period_index]
		if self.variances != None:
			variances = self.variances[:,period_index]
		else:
			variances = None
		return HazardCurveField(self.model_name, hazard_values, self.filespecs[period_index], self.sites, period, self.IMT, intensities, self.intensity_unit, self.timespan, variances=variances)

	def appendHazardCurveFields(self, hcf_list):
		pass

	def getSpectralHazardCurve(self, site_spec=0):
		"""
		Return spectral hazard curve for a particular site
		Parameters:
			site_spec: site specification (site index, (lon, lat) tuple or site name)
				(default: 0)
		Return value:
			SpectralHazardCurve object
		"""
		site_index = self.site_index(site_spec)
		try:
			site = self.sites[site_index]
		except:
			raise IndexError("Site index %s out of range" % site_index)

		site_name = self.site_names[site_index]
		filespec = self.filespecs[0]
		intensities = self.intensities
		hazard_values = self._hazard_values[site_index]
		if self.variances != None:
			variances = self.variances[site_index]
		else:
			variances = None
		return SpectralHazardCurve(self.model_name, hazard_values, filespec, site, self.periods, self.IMT, intensities, self.intensity_unit, self.timespan, variances=variances)

	def getHazardCurve(self, site_spec=0, period_spec=0):
		"""
		Return hazard curve for a particular site and a particular period
		Parameters:
			site_spec: site specification (site index, (lon, lat) tuple or site name)
				(default: 0)
			period_spec: period specification (period index if integer, period if float)
				(default: 0)
		Return value:
			HazardCurve object
		"""
		site_index = self.site_index(site_spec)
		try:
			site = self.sites[site_index]
		except:
			raise IndexError("Site index %s out of range" % site_index)
		else:
			site_name = self.site_names[site_index]

		period_index = self.period_index(period_spec)
		try:
			period = self.periods[period_index]
		except:
			raise IndexError("Period index %s out of range" % period_index)

		filespec = self.filespecs[period_index]
		intensities = self.intensities[period_index]
		hazard_values = self._hazard_values[site_index, period_index]
		if self.variances != None:
			variances = self.variances[site_index, period_index]
		else:
			variances = None
		return HazardCurve(self.model_name, hazard_values, filespec, site, period, self.IMT, intensities, self.intensity_unit, self.timespan, variances=variances)

	def toTree(self):
		"""
		Promote to a SpectralazardCurveFieldTree object (multiple spectral periods,
			multiple sites, multiple logic-tree branches)
		"""
		intensities = self.intensities
		hazard_values = self._hazard_values.reshape((self.num_sites, 1, self.num_periods, self.num_intensities))
		if self.variances != None:
			variances = self.variances.reshape((self.num_sites, 1, self.num_periods, self.num_intensities))
		else:
			variances = None
		branch_names = [self.model_name]
		filespecs = [self.filespecs[0]]
		weights = np.array([1.], 'd')
		return SpectralHazardCurveFieldTree(self.model_name, hazard_values, branch_names, filespecs, weights, self.sites, self.periods, self.IMT, intensities, self.intensity_unit, self.timespan, variances=variances)

	def interpolate_return_periods(self, return_periods):
		"""
		Interpolate intensity measure levels for given return periods
		Parameters:
			return_periods: list or array with return periods
		Return value:
			UHSFieldSet object
		"""
		filespecs = [self.filespecs[0]] * len(return_periods)
		return_periods = np.array(return_periods)
		num_sites, num_periods = self.num_sites, self.num_periods
		rp_intensities = np.zeros((len(return_periods), num_sites, num_periods))
		interpol_exceedances = 1. / return_periods
		for i in range(num_sites):
			for k in range(num_periods):
				rp_intensities[:,i,k] = interpolate(self.exceedance_rates[i,k], self.intensities[k], interpol_exceedances)
		return UHSFieldSet(self.model_name, filespecs, self.sites, self.periods, self.IMT, rp_intensities, self.intensity_unit, self.timespan, return_periods=return_periods)

	def interpolate_periods(self, out_periods):
		"""
		Interpolate intensity measure levels at different spectral periods
		Parameters:
			out_periods: list or array of output spectral periods
		Return value:
			SpectralHazardCurveField object
		"""
		num_sites, num_intensities = self.num_sites, self.num_intensities
		out_hazard_values = self._hazard_values.__class__(np.zeros((num_sites, len(out_periods), num_intensities), dtype='d'))
		if self.variances != None:
			out_variances = np.zeros((num_sites, len(out_periods), num_intensities), dtype='d')
		else:
			out_variances = None

		for i in range(num_sites):
			shc = self.getSpectralHazardCurve(site_spec=i)
			shc_out = shc.interpolate_periods(out_periods)
			out_hazard_values[i] = shc_out._hazard_values
			if self.variances != None:
				out_variances[i] = shc_out.variances
		intensities = shc_out.intensities
		return SpectralHazardCurveField(self.model_name, out_hazard_values, self.filespecs, self.sites, out_periods, self.IMT, intensities, self.intensity_unit, self.timespan, variances=out_variances)

	def plot(self, site_specs=[], period_specs=[], labels=None, colors=None, linestyles=None, linewidth=2, fig_filespec=None, title=None, want_recurrence=False, want_poe=False, interpol_rp=None, interpol_prob=None, interpol_rp_range=None, amax=None, rp_max=1E+07, legend_location=0, lang="en"):
		"""
		Plot hazard curves for some sites and/or some spectral periods.
		Parameters:
			site_specs: list with site specs (indexes, (lon,lat) tuples or site names)
				of sites to be plotted (default: [] will plot all sites)
			period_specs: list with period specs (integer period indexes or float
				spectral periods) (default: [] will plot all periods)
			colors: list with curve colors for each site or period (default: None)
			linestyles: list with line styles for each site or period (default: None)
			linewidth: line width (default: 2)
		"""
		## Title
		if title is None:
			title = self.model_name
		## Determine sites and periods
		if site_specs in (None, []):
			site_indexes = range(self.num_sites)
		else:
			site_indexes = [self.site_index(site_spec) for site_spec in site_specs]
		sites = [self.sites[site_index] for site_index in site_indexes]
		if period_specs in (None, []):
			period_indexes = range(self.num_periods)
		else:
			period_indexes = [self.period_index(period_spec) for period_spec in period_specs]
		periods = [self.periods[period_index] for period_index in period_indexes]

		## Labels
		if labels in (None, []):
			if len(sites) == 1:
				labels = ["T = %s s" % period for period in periods]
			elif len(periods) == 1:
				labels = [self.site_names[site_index] for site_index in site_indexes]
			else:
				labels = []
				for i, site in enumerate(sites):
					site_name = self.site_names[site_indexes[i]]
					for period in periods:
						labels.append("Site: %s, T=%s s" % (site_name, period))

		## Colors and linestyles
		if colors in (None, []):
			if len(sites) >= len(periods):
				colors = [["r", "g", "b", "c", "m", "k"][i%6:i%6+1] * len(periods) for i in range(len(sites))]
			else:
				colors = [["r", "g", "b", "c", "m", "k"][i%6:i%6+1] * len(sites) for i in range(len(periods))]
			## Hack to flatten nested list
			colors = sum(colors, [])

		## Linestyles
		if linestyles in (None, []):
			if len(sites) >= len(periods):
				linestyles = [['-', '--', ':', '-.'][i%4:i%4+1] * len(sites) for i in range(len(periods))]
			else:
				linestyles = [['-', '--', ':', '-.'][i%4:i%4+1] * len(periods) for i in range(len(sites))]
			linestyles = sum(linestyles, [])

		linewidths = [linewidth] * len(sites) * len(periods)

		## Data
		datasets = []
		exceedance_rates = self.exceedance_rates
		for site in sites:
			site_index = self.site_index(site)
			for period in periods:
				period_index = self.period_index(period)
				x = self.intensities[period_index]
				y = exceedance_rates[site_index, period_index]
				datasets.append((x, y))

		fixed_life_time = {True: self.timespan, False: None}[want_poe]
		intensity_unit = self.intensity_unit
		plot_hazard_curve(datasets, labels=labels, colors=colors, linestyles=linestyles, linewidths=linewidths, fig_filespec=fig_filespec, title=title, want_recurrence=want_recurrence, fixed_life_time=fixed_life_time, interpol_rp=interpol_rp, interpol_prob=interpol_prob, interpol_rp_range=interpol_rp_range, amax=amax, intensity_unit=intensity_unit, tr_max=rp_max, legend_location=legend_location, lang=lang)

	plot.__doc__ += common_plot_docstring

	def export_GRA(self, out_filespec):
		"""
		Write spectral hazard curve field to CRISIS .GRA format
		Parameters:
			out_filespec: full path to output file
		"""
		f = open(out_filespec, "w")
		f.write("************************************************************\n")
		f.write("Generic exceedance-rate results\n")
		f.write("Calculated outside CRISIS\n")
		f.write("NumSites, NumPeriods, NumIntensities: %d, %d, %d\n" % (self.num_sites, self.num_periods, self.num_intensities))
		f.write("************************************************************\n")
		f.write("\n\n")
		for i in range(self.num_sites):
			f.write("    %s      %s\n" % self.sites[i])
			for k in range(self.num_periods):
				f.write("INTENSITY %d T=%s\n" % (k+1, self.periods[k]))
				for l in range(self.num_intensities):
					f.write("%.5E  %.5E" % (self.intensities[k,l], self.exceedance_rates[i,k,l]))
					if self.variances != None:
						f.write("  %.5E" % self.variances[i,k,l])
					f.write("\n")
		f.close()

	def create_xml_element(self, smlt_path=None, gmpelt_path=None, encoding='latin1'):
		"""
		Create xml element (NRML SpectralHazardCurveField element)
		Arguments:
			encoding: unicode encoding (default: 'latin1')
		"""
		shcf_elem = etree.Element(ns.SPECTRAL_HAZARD_CURVE_FIELD)
		shcf_elem.set(ns.IMT, self.IMT)
		shcf_elem.set(ns.INVESTIGATION_TIME, str(self.timespan))
		if smlt_path:
			shcf_elem.set(ns.SMLT_PATH, smlt_path)
		if gmpelt_path:
			shcf_elem.set(ns.GMPELT_PATH, gmpelt_path)
		shcf_elem.set(ns.NAME, self.model_name)
		for k, period in enumerate(self.periods):
			# TODO: put following in HazardCurveField and HazardCurve !
			hcf_elem = etree.SubElement(shcf_elem, ns.HAZARD_CURVE_FIELD)
			hcf_elem.set(ns.PERIOD, str(period))
			# TODO: add damping for SA ?
			imls_elem = etree.SubElement(hcf_elem, ns.IMLS)
			imls_elem.text = " ".join(map(str, self.intensities[k,:]))
			for i, site in enumerate(self.sites):
				hazard_curve_elem = etree.SubElement(hcf_elem, ns.HAZARD_CURVE)
				point_elem = etree.SubElement(hazard_curve_elem, ns.POINT)
				position_elem = etree.SubElement(point_elem, ns.POSITION)
				position_elem.text = "%s %s" % (site[0], site[1])
				poes_elem = etree.SubElement(hazard_curve_elem, ns.POES)
				poes_elem.text = " ".join(map(str, self.poes[i,k,:]))
		return shcf_elem

	def write_nrml(self, filespec, smlt_path=None, gmpelt_path=None, encoding='latin1', pretty_print=True):
		"""
		Write spectral hazard curve field to XML file
		Arguments:
			filespec: full path to XML output file
			encoding: unicode encoding (default: 'utf-8')
			pretty_print: boolean indicating whether or not to indent each
				element (default: True)
		"""
		tree = create_nrml_root(self, encoding=encoding, smlt_path=smlt_path, gmpelt_path=gmpelt_path)
		fd = open(filespec, 'w')
		tree.write(fd, xml_declaration=True, encoding=encoding, pretty_print=pretty_print)
		fd.close()


class SpectralHazardCurve(HazardResult, HazardSpectrum):
	"""
	Class representing hazard curves at 1 site for different spectral periods
	Parameters:
		model_name: name of this hazard-curve model
		filespec: full path to file containing hazard curves
		site: (lon, lat) tuple of site for which hazard curve was computed
		periods: 1-D array [k] of spectral periods
		IMT: intensity measure type (PGA, SA, PGV or PGD)
		intensities: 2-D array [k,l] of intensity measure levels (ground-motion values)
			for each spectral period for which exceedance rate or probability of
			exceedance was computed
		intensity_unit: unit in which intensity measure levels are expressed:
			PGA and SA: "g", "mg", "ms2", "gal"
			PGV: "cms"
			PGD: "cm"
			default: "g"
		timespan: period related to the probability of exceedance (aka life time)
			(default: 50)
		poes: 2-D array [k,l] with probabilities of exceedance computed for each
			intensity measure level [k,l]. If None, exceedance_rates must be specified
			(default: None)
		exceedance_rates: 2-D array [k,l] with exceedance rates computed for each
			intensity measure level [k,l. If None, poes must be specified
			(default: None)
		variances: 2-D array [k,l] with variance of exceedance rate or probability of exceedance
			(default: None)
		site_name: site name (default: "")
	"""
	# TODO: update docstring
	def __init__(self, model_name, hazard_values, filespec, site, periods, IMT, intensities, intensity_unit="g", timespan=50, variances=None):
		HazardResult.__init__(self, hazard_values, timespan=timespan, IMT=IMT, intensities=intensities, intensity_unit=intensity_unit)
		HazardSpectrum.__init__(self, periods)
		self.model_name = model_name
		self.filespec = filespec
		self.site = site
		self.variances = as_array(variances)
		self.period_axis = 0

	def __iter__(self):
		self._current_index = 0
		return self

	def next(self):
		"""
		Loop over spectral periods
		"""
		try:
			period = self.periods[self._current_index]
		except:
			raise StopIteration
		else:
			self._current_index += 1
			return self.getHazardCurve(self._current_index-1)

	def __getitem__(self, period_spec):
		return self.getHazardCurve(period_spec)

	def __add__(self, other_shc):
		"""
		:param other_shc:
			instance of :class:`SpectralHazardCurve`

		:return:
			instance of :class:`SpectralHazardCurve`
		"""
		assert isinstance(other_shc, SpectralHazardCurve)
		assert self.site == other_shc.site
		assert self.IMT == other_shc.IMT
		assert (self.periods == other_shc.periods).all()
		assert (self.intensities == other_shc.intensities).all()
		assert self.intensity_unit == other_shc.intensity_unit
		assert self.timespan == other_shc.timespan
		hazard_values = self._hazard_values + other_shc._hazard_values
		return self.__class__(self.model_name, hazard_values, self.filespec, self.site, self.periods, self.IMT, self.intensities, self.intensity_unit, self.timespan)

	def __mul__(self, number):
		"""
		:param number:
			int, float or Decimal

		:return:
			instance of :class:`SpectralHazardCurve`
		"""
		assert isinstance(number, (int, float, Decimal))
		hazard_values = self._hazard_values * number
		return self.__class__(self.model_name, hazard_values, self.filespec, self.site, self.periods, self.IMT, self.intensities, self.intensity_unit, self.timespan)

	def __rmul__(self, number):
		return self.__mul__(number)

	def min(self):
		pass

	def max(self):
		pass

	def getHazardCurve(self, period_spec=0):
		"""
		Return hazard curve for a particular spectral period
		Parameters:
			period_spec: period specification (period index if integer, period if float)
				(default: 0)
		"""
		period_index = self.period_index(period_spec)
		try:
			period = self.periods[period_index]
		except:
			raise IndexError("Period index %s out of range" % period_index)
		intensities = self.intensities[period_index]
		hazard_values = self._hazard_values[period_index]
		if self.variances != None:
			variances = self.variances[period_index]
		else:
			variances = None
		return HazardCurve(self.model_name, hazard_values, self.filespec, self.site, period, self.IMT, intensities, self.intensity_unit, self.timespan, variances=variances)

	def interpolate_return_period(self, return_period):
		"""
		Interpolate intensity measure levels for given return period
		Parameters:
			return_period: return period
		Return value:
			UHS object
		"""
		num_periods = self.num_periods
		rp_intensities = np.zeros(num_periods)
		interpol_exceedance_rate = 1. / return_period
		for k in range(num_periods):
			rp_intensities[k] = interpolate(self.exceedance_rates[k], self.intensities[k], [interpol_exceedance_rate])[0]
		return UHS(self.model_name, self.filespec, self.site, self.periods, self.IMT, rp_intensities, self.intensity_unit, self.timespan, return_period=return_period)

	def interpolate_periods(self, out_periods):
		"""
		Interpolate intensity measure levels at different spectral periods
		Parameters:
			out_periods: list or array of output spectral periods
		Return value:
			a SpectralHazardCurve object
		"""
		if out_periods in ([], None):
			return self
		else:
			in_periods = self.periods
			num_intensities = self.num_intensities
			out_hazard_values = self._hazard_values.__class__(np.zeros((len(out_periods), num_intensities), dtype='d'))
			if self.variances != None:
				out_variances = np.zeros((len(out_periods), num_intensities), dtype='d')
			else:
				out_variances = None
			out_period_intensities = np.zeros((len(out_periods), num_intensities), dtype='d')

			for k in range(len(out_periods)):
				## if k is close to an in_period, take over the corresponding intensities,
				## else, define new range of intensities for that period,
				## interpolate corresponding exceedances,
				## and then interpolate at the wanted period
				threshold = 1E-6
				try:
					id = np.where(abs(in_periods - out_periods[k]) < threshold)[0][0]
				except IndexError:
					id1 = np.where(in_periods < out_periods[k])[0][-1]
					id2 = np.where(in_periods > out_periods[k])[0][0]
					Imin = min(self.intensities[id1][0], self.intensities[id2][0])
					Imax = min(self.intensities[id1][-1], self.intensities[id2][-1])
					#Imin, Imax = self.intensities[id1][0], self.intensities[id1][-1]
					out_period_intensities[k] = logrange(Imin, Imax, num_intensities)
					## Interpolate exceedances of adjacent periods to out_period_intensities
					hazard_values1 = interpolate(self.intensities[id1], self._hazard_values[id1], out_period_intensities[k])
					hazard_values2 = interpolate(self.intensities[id2], self._hazard_values[id2], out_period_intensities[k])
					if self.variances != None:
						variances1 = interpolate(self.intensities[id1], self.variances[id1], out_period_intensities[k])
						variances2 = interpolate(self.intensities[id2], self.variances[id2], out_period_intensities[k])
					for l in range(num_intensities):
						out_hazard_values[k,l] = interpolate([in_periods[id1], in_periods[id2]], [hazard_values1[l], hazard_values2[l]], [out_periods[k]])[0]
						if self.variances != None:
							out_variances[k,l] = interpolate([in_periods[id1], in_periods[id2]], [variances1[l], variances2[l]], [out_periods[k]])[0]
				else:
					out_period_intensities[k] = self.intensities[id]
					out_hazard_values[k] = self._hazard_values[id]
					if self.variances != None:
						out_variances[k] = self.variances[id]
			return SpectralHazardCurve(self.model_name, hazard_values, self.filespec, self.site, out_periods, self.IMT, out_period_intensities, self.intensity_unit, self.timespan, variances=out_variances)

	def toField(self):
		"""
		Promote to a SpectralazardCurveField object (multiple spectral periods, multiple sites)
		"""
		intensities = self.intensities
		hazard_values = self._hazard_values.reshape((1, self.num_periods, self.num_intensities))
		if self.variances != None:
			variances = self.variances.reshape((1, self.num_periods, self.num_intensities))
		else:
			variances = None
		return SpectralHazardCurveField(self.model_name, hazard_values, [self.filespec], [self.site], self.period, self.IMT, intensities, self.intensity_unit, self.timespan, variances=variances)

	def plot(self, colors=[], linestyle="-", linewidth=2, fig_filespec=None, title=None, want_recurrence=False, want_poe=False, interpol_rp=None, interpol_prob=None, interpol_rp_range=None, amax=None, rp_max=1E+07, legend_location=0, lang="en"):
		"""
		Plot hazard curves for all spectral periods
		Parameters:
			colors: list with curve colors for each site (default: None)
			linestyle: line style (default: "-")
			linewidth: line width (default: 2)
		"""
		if title is None:
			title = "Spectral Hazard Curve"
			title += "\nSite: %s" % self.site.name
		datasets = [(self.intensities[k], self.exceedance_rates[k]) for k in range(self.num_periods)]
		labels = ["T = %s s" % period for period in self.periods]
		fixed_life_time = {True: self.timespan, False: None}[want_poe]
		intensity_unit = self.intensity_unit
		plot_hazard_curve(datasets, labels=labels, colors=colors, linestyles=[linestyle], linewidths=[linewidth], fig_filespec=fig_filespec, title=title, want_recurrence=want_recurrence, fixed_life_time=fixed_life_time, interpol_rp=interpol_rp, interpol_prob=interpol_prob, interpol_rp_range=interpol_rp_range, amax=amax, intensity_unit=intensity_unit, tr_max=rp_max, legend_location=legend_location, lang=lang)

	plot.__doc__ += common_plot_docstring

	def export_csv(self, csv_filespec=None):
		pass


# TODO: make HazardCurveFieldTree?


class HazardCurveField(HazardResult, HazardField):
	"""
	Class representing a hazard curve field for a single spectral period.
	Corresponds to 1 OpenQuake hazardcurve file.
	Parameters:
		model_name: name of this hazard-curve model
		filespec: full path to file containing hazard curve
		sites: 1-D list [i] of (lon, lat) tuples of sites for which hazard curves
			were computed
		period: spectral period
		IMT: intensity measure type (PGA, SA, PGV or PGD)
		intensities: 1-D array [l] of intensity measure levels (ground-motion values)
			for which exceedance rate or probability of exceedance was computed
		intensity_unit: unit in which intensity measure levels are expressed:
			PGA and SA: "g", "mg", "ms2", "gal"
			PGV: "cms"
			PGD: "cm"
			default: "g"
		timespan: period related to the probability of exceedance (aka life time)
			(default: 50)
		poes: 2-D array [i,l] with probabilities of exceedance computed for each
			intensity measure level [l] at each site [i].
			If None, exceedance_rates must be specified (default: None)
		exceedance_rates: 2-D array [i,l] with exceedance rates computed for each
			intensity measure level [l] at each site [i].
			If None, poes must be specified	(default: None)
		variances: 2-D array [i,l] with variance of exceedance rate or probability of exceedance
			(default: None)
		site_names: list of site names (default: None)
	"""
	def __init__(self, model_name, hazard_values, filespec, sites, period, IMT, intensities, intensity_unit="g", timespan=50, variances=None):
		HazardResult.__init__(self, hazard_values, timespan=timespan, IMT=IMT, intensities=intensities, intensity_unit=intensity_unit)
		HazardField.__init__(self, sites)
		self.model_name = model_name
		self.filespec = filespec
		self.period = period
		self.variances = as_array(variances)

	def __iter__(self):
		self._current_index = 0
		return self

	def next(self):
		"""
		Loop over sites
		"""
		try:
			site = self.sites[self._current_index]
		except:
			raise StopIteration
		else:
			self._current_index += 1
			return self.getHazardCurve(self._current_index-1)

	def __getitem__(self, site_spec):
		return self.getHazardCurve(site_spec)

	def intensity_index(self, intensity):
		"""
		Return index corresponding to a particular intensity measure level
		Parameters:
			return_period: return period
		Return value:
			intensity index (integer)
		"""
		intensity_index = interpolate(self.intensities, range(self.num_intensities), [intensity])[0]
		return int(round(intensity_index))

	def argmin(self, intensity=None, return_period=None):
		"""
		Return index of site with minimum hazard for a particular intensity or
		return period.
		Parameters:
			intensity: intensity measure level. If None, return_period must be specified
				(default: None)
			return_period: return period. If None, intensity must be specified
				(default: None)
		Return value:
			site index (int)
		"""
		if intensity is not None:
			intensity_index = self.intensity_index(intensity)
			site_index = self._hazard_values[:,intensity_index].argmin()
		elif return_period is not None:
			hazardmapset = self.interpolate_return_periods([return_period])
			hazardmap = hazardmapset[0]
			site_index = hazardmap.argmin()
		else:
			raise Exception("Need to specify either intensity or return_period")
		return site_index

	def argmax(self, intensity=None, return_period=None):
		"""
		Return index of site with maximum hazard for a particular intensity or
		return period.
		Parameters:
			intensity: intensity measure level. If None, return_period must be specified
				(default: None)
			return_period: return period. If None, intensity must be specified
				(default: None)
		Return value:
			site index (int)
		"""
		if intensity is not None:
			intensity_index = self.intensity_index(intensity)
			site_index = self._hazard_values[:,intensity_index].argmax()
		elif return_period is not None:
			hazardmapset = self.interpolate_return_periods([return_period])
			hazardmap = hazardmapset[0]
			site_index = hazardmap.argmax()
		else:
			raise Exception("Need to specify either intensity or return_period")
		return site_index

	def min(self, intensity=None, return_period=None):
		"""
		Return minimum hazard for a particular intensity or return period
		Parameters:
			intensity: intensity measure level. If None, return_period must be specified
				(default: None)
			return_period: return period. If None, intensity must be specified
				(default: None)
		Return value:
			minimum exceedance rate (if intensity was specified) or minimum intensity
			(if return period was specified)
		"""
		if intensity is not None:
			intensity_index = self.intensity_index(intensity)
			return self._hazard_values[:,intensity_index].min()
		elif return_period is not None:
			hazardmapset = self.interpolate_return_periods([return_period])
			hazardmap = hazardmapset[0]
			return hazardmap.min()
		else:
			raise Exception("Need to specify either intensity or return_period")

	def max(self, intensity=None, return_period=None):
		"""
		Return maximum hazard curve for a particular intensity or return period
		Parameters:
			intensity: intensity measure level. If None, return_period must be specified
				(default: None)
			return_period: return period. If None, intensity must be specified
				(default: None)
		Return value:
		Return value:
			maximum exceedance rate (if intensity was specified) or maximum intensity
			(if return period was specified)
		"""
		if intensity is not None:
			intensity_index = self.intensity_index(intensity)
			return self._hazard_values[:,intensity_index].max()
		elif return_period is not None:
			hazardmapset = self.interpolate_return_periods([return_period])
			hazardmap = hazardmapset[0]
			return hazardmap.max()
		else:
			raise Exception("Need to specify either intensity or return_period")

	def getHazardCurve(self, site_spec=0):
		"""
		Return hazard curve for a particular site
		Parameters:
			site_spec: site specification (site index, (lon, lat) tuple or site name)
				(default: 0)
		Return value:
			HazardCurve object
		"""
		site_index = self.site_index(site_spec)
		try:
			site = self.sites[site_index]
		except:
			raise IndexError("Site index %s out of range" % site_index)

		site_name = self.site_names[site_index]
		intensities = self.intensities
		hazard_values = self._hazard_values[site_index]
		if self.variances != None:
			variances = self.variances[site_index]
		else:
			variances = None
		return HazardCurve(self.model_name, hazard_values, self.filespec, site, self.period, self.IMT, intensities, self.intensity_unit, self.timespan, variances=variances)

	def interpolate_return_periods(self, return_periods):
		"""
		Interpolate intensity measure levels for given return periods.
		Parameters:
			return_periods: list or array of return periods
		Return value:
			HazardMapSet object
		"""
		filespecs = [self.filespec] * len(return_periods)
		return_periods = np.array(return_periods)
		rp_intensities = np.zeros((len(return_periods), self.num_sites))
		interpol_exceedances = 1. / return_periods
		for i in range(self.num_sites):
				rp_intensities[:,i] = interpolate(self.exceedance_rates[i], self.intensities, interpol_exceedances)
		return HazardMapSet(self.model_name, filespecs, self.sites, self.period, self.IMT, rp_intensities, self.intensity_unit, self.timespan, return_periods=return_periods)

	def toSpectral(self):
		"""
		Promote to a SpectralHazardCurveField object (multiple sites, multiple spectral periods)
		"""
		intensities = self.intensities.reshape((1, self.num_intensities))
		hazard_values = self._hazard_values.reshape((self.num_sites, 1, self.num_intensities))
		if self.variances != None:
			variances = self.variances.reshape((self.num_sites, 1, self.num_intensities))
		else:
			variances = None
		return SpectralHazardCurveField(self.model_name, hazard_values, [self.filespec], self.sites, [self.period], self.IMT, intensities, self.intensity_unit, self.timespan, variances=variances)

	def plot(self, site_specs=[], labels=None, colors=None, linestyles=None, linewidth=2, fig_filespec=None, title=None, want_recurrence=False, want_poe=False, interpol_rp=None, interpol_prob=None, interpol_rp_range=None, amax=None, rp_max=1E+07, legend_location=0, lang="en"):
		"""
		Plot hazard curves for one or more sites.
		Parameters:
			site_specs: list with site specs (indexes, (lon,lat) tuples or site names)
				of sites to be plotted (default: [] will plot all sites)
			colors: list with curve colors for each site (default: None)
			linestyles: list with line styles for each site (default: None)
			linewidth: line width (default: 2)
		"""
		## Determine sites
		if site_specs in (None, []):
			site_indexes = range(self.num_sites)
		else:
			site_indexes = [self.site_index(site_spec) for site_spec in site_specs]
		sites = [self.sites[site_index] for site_index in site_indexes]

		## Labels
		if labels in (None, []):
			labels = [self.site_names[site_index] for site_index in site_indexes]

		## Colors and linestyles
		if colors in (None, []):
			colors = [["r", "g", "b", "c", "m", "k"][i%6] for i in range(len(sites))]

		## Linestyles
		if linestyles in (None, []):
			linestyles = [['-', '--', ':', '-.'][i//len(colors)%4] for i in range(len(sites))]

		linewidths = [linewidth] * len(sites)

		## Data
		datasets = []
		exceedance_rates = self.exceedance_rates
		for site in sites:
			site_index = self.site_index(site)
			x = self.intensities
			y = exceedance_rates[site_index]
			datasets.append((x, y))

		fixed_life_time = {True: self.timespan, False: None}[want_poe]
		intensity_unit = self.intensity_unit
		plot_hazard_curve(datasets, labels=labels, colors=colors, linestyles=linestyles, linewidths=linewidths, fig_filespec=fig_filespec, title=title, want_recurrence=want_recurrence, fixed_life_time=fixed_life_time, interpol_rp=interpol_rp, interpol_prob=interpol_prob, interpol_rp_range=interpol_rp_range, amax=amax, intensity_unit=intensity_unit, tr_max=rp_max, legend_location=legend_location, lang=lang)

	plot.__doc__ += common_plot_docstring

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML HazardCurveField element)
		Arguments:
			encoding: unicode encoding (default: 'latin1')
		"""
		# TODO: use names from nrml namespace
		hcf_elem = etree.Element('hazardCurveField')
		hcf_elem.set('imt', self.IMT)
		hcf_elem.set('period', str(self.period))
		hcf_elem.set('imls', ''.join(map(str, self.intensities)))

		for i, site in enumerate(self.sites):
			site_elem = etree.SubElement(hcf_elem, 'site')
			site_elem.set('lon_lat', ''.join(map(str, site)))
			site_elem.set('exceedance_rates', ''.join(map(str, self.exceedance_rates[i])))

		return hcf_elem

	def write_nrml(self, filespec, encoding='latin1', pretty_print=True):
		"""
		Write hazard curve field to XML file
		Arguments:
			filespec: full path to XML output file
			encoding: unicode encoding (default: 'utf-8')
			pretty_print: boolean indicating whether or not to indent each
				element (default: True)
		"""
		tree = create_nrml_root(self, encoding=encoding)
		fd = open(filespec, 'w')
		tree.write(fd, xml_declaration=True, encoding=encoding, pretty_print=pretty_print)
		fd.close()


class HazardCurve(HazardResult):
	"""
	Class representing a hazard curve for a single site and a single spectral period
	Parameters:
		model_name: name of this hazard-curve model
		filespec: full path to file containing hazard curve
		site: (lon, lat) tuple of site for which hazard curve was computed
		period: spectral period
		IMT: intensity measure type (PGA, SA, PGV or PGD)
		intensities: 1-D array [l] of intensity measure levels (ground-motion values)
			for which exceedance rate or probability of exceedance was computed
		intensity_unit: unit in which intensity measure levels are expressed:
			PGA and SA: "g", "mg", "ms2", "gal"
			PGV: "cms"
			PGD: "cm"
			default: "g"
		timespan: period related to the probability of exceedance (aka life time)
			(default: 50)
		poes: 1-D array [l] with probabilities of exceedance computed for each
			intensity measure level [l]. If None, exceedance_rates must be specified
			(default: None)
		exceedance_rates: 1-D array [l] with exceedance rates computed for each
			intensity measure level [l]. If None, poes must be specified
			(default: None)
		variances: 1-D array [l] with variance of exceedance rate or probability of exceedance
			(default: None)
		site_name: name of site (default: "")
	"""
	def __init__(self, model_name, hazard_values, filespec, site, period, IMT, intensities, intensity_unit="g", timespan=50, variances=None):
		HazardResult.__init__(self, hazard_values, timespan=timespan, IMT=IMT, intensities=intensities, intensity_unit=intensity_unit)
		self.model_name = model_name
		self.filespec = filespec
		self.site = site
		self.period = period
		self.variances = as_array(variances)

	def __len__(self):
		"""
		Return length of hazard curve (= number of intensity measure levels)
		"""
		return self.num_intensities

	def __add__(self, other_hc):
		"""
		:param other_hc:
			instance of :class:`HazardCurve`

		:return:
			instance of :class:`HazardCurve`
		"""
		assert isinstance(other_hc, HazardCurve)
		assert self.site == other_hc.site
		assert self.IMT == other_hc.IMT
		assert self.period == other_hc.period
		assert (self.intensities == other_hc.intensities).all()
		assert self.intensity_unit == other_hc.intensity_unit
		assert self.timespan == other_hc.timespan
		hazard_values = self._hazard_values + other_hc._hazard_values
		return self.__class__(self.model_name, hazard_values, self.filespec, self.site, self.period, self.IMT, self.intensities, self.intensity_unit, self.timespan)

	def __mul__(self, number):
		"""
		:param number:
			int, float or Decimal

		:return:
			instance of :class:`HazardCurve`
		"""
		assert isinstance(number, (int, float, Decimal))
		hazard_values = self._hazard_values * number
		return self.__class__(self.model_name, hazard_values, self.filespec, self.site, self.period, self.IMT, self.intensities, self.intensity_unit, self.timespan)

	def __rmul__(self, number):
		return self.__mul__(number)

	@property
	def site_name(self):
		return self.site.name

	def interpolate_return_periods(self, return_periods):
		"""
		Interpolate intensity measure levels for given return periods.
		Parameters:
			return_periods: list or array of return periods
		Return value:
			1-D array of intensity measure levels
		"""
		return_periods = np.array(return_periods, 'd')
		interpol_exceedance_rates = 1. / return_periods
		rp_intensities = interpolate(self.exceedance_rates, self.intensities, interpol_exceedance_rates)
		return rp_intensities

	def toSpectral(self):
		"""
		Promote to a SpectralHazardCurve object (1 site, multiple spectral periods)
		"""
		intensities = self.intensities.reshape((1, self.num_intensities))
		hazard_values = self._hazard_values.reshape((1, self.num_intensities))
		if self.variances != None:
			variances = self.variances.reshape((1, self.num_intensities))
		else:
			variances = None
		return SpectralHazardCurve(self.model_name, hazard_values, self.filespec, self.site, [self.period], self.IMT, intensities, self.intensity_unit, self.timespan, variances=variances)

	def toField(self):
		"""
		Promote to a HazardCurveField object (1 spectral period, multiple sites)
		"""
		intensities = self.intensities
		hazard_values = self._hazard_values.reshape((1, self.num_intensities))
		if self.variances != None:
			variances = self.variances.reshape((1, self.num_intensities))
		else:
			variances = None
		return HazardCurveField(self.model_name, hazard_values, self.filespec, [self.site], self.period, self.IMT, intensities, self.intensity_unit, self.timespan, variances=variances)

	def plot(self, color="k", linestyle="-", linewidth=2, fig_filespec=None, title=None, want_recurrence=False, want_poe=False, interpol_rp=None, interpol_prob=None, interpol_rp_range=None, amax=None, rp_max=1E+07, legend_location=0, lang="en"):
		"""
		Plot hazard curve
		Parameters:
			color: curve color (default: 'k')
			linestyle: line style (default: "-")
			linewidth: line width (default: 2)
		"""
		if title is None:
			title = "Hazard Curve"
			title += "\nSite: %s, T: %s s" % (self.site_name, self.period)
		datasets = [(self.intensities, self.exceedance_rates)]
		labels = [self.model_name]
		fixed_life_time = {True: self.timespan, False: None}[want_poe]
		intensity_unit = self.intensity_unit
		plot_hazard_curve(datasets, labels=labels, colors=[color], linestyles=[linestyle], linewidths=[linewidth], fig_filespec=fig_filespec, title=title, want_recurrence=want_recurrence, fixed_life_time=fixed_life_time, interpol_rp=interpol_rp, interpol_prob=interpol_prob, interpol_rp_range=interpol_rp_range, amax=amax, intensity_unit=intensity_unit, tr_max=rp_max, legend_location=legend_location, lang=lang)

	plot.__doc__ += common_plot_docstring

	def export_csv(self, csv_filespec=None):
		"""
		Export hazard curve to a csv file
		Parameters:
			csv_filespec: full path to output file. If None, output is written
				to standard output (default: None)
		"""
		if csv_filespec:
			f = open(csv_filespec, "w")
		else:
			f = sys.stdout
		f.write("%s (%s), Exceedance rate (1/yr)\n" % (self.IMT, self.intensity_unit))
		for intensity, exceedance in zip(self.intensities, self.exceedance_rates):
			f.write("%.3E, %.3E\n" % (intensity, exceedance))
		f.close()


class HazardCurveCollection:
	"""
	Container for an arbitrary set of hazard curves.
	Useful for plotting.
	Parameters:
		hazard_curves: list of HazardCurve objects
		colors: list of colors for each hazard curve (default: [])
		linestyles: list of line styles for each hazard curve (default: [])
		linewidths: list of line widhts for each hazard curve (default: [])
		labels: list of labels for each hazard curve (default: [])
	"""
	def __init__(self, hazard_curves, colors=[], linestyles=[], linewidths=[], labels=[]):
		self.hazard_curves = hazard_curves
		self.colors = colors
		self.linestyles = linestyles
		self.linewidths = linewidths
		if not labels:
			labels = [hc.model_name for hc in self.hazard_curves]
		self.labels = labels

	def __len__(self):
		return len(self.hazard_curves)

	def append(self, hc, color=None, linestyle=None, linewidth=None, label=None):
		self.hazard_curves.append(hc)
		if not label:
			label = hc.model_name
		self.labels.append(label)
		self.colors.append(color)
		self.linestyles.append(linestyle)
		self.linewidths.append(linewidth)

	@property
	def intensity_unit(self):
		return self.hazard_curves[0].intensity_unit

	def plot(self, fig_filespec=None, title=None, want_recurrence=False, want_poe=False, interpol_rp=None, interpol_prob=None, interpol_rp_range=None, amax=None, tr_max=1E+07, legend_location=0, lang="en"):
		if title is None:
			title = "Hazard Curve Collection"
		datasets = [(hc.intensities, hc.exceedance_rates) for hc in self.hazard_curves]
		hc0 = self.hazard_curves[0]
		fixed_life_time = {True: hc0.timespan, False: None}[want_poe]
		intensity_unit = hc0.intensity_unit
		plot_hazard_curve(datasets, labels=self.labels, colors=self.colors, linestyles=self.linestyles, linewidths=self.linewidths, fig_filespec=fig_filespec, title=title, want_recurrence=want_recurrence, fixed_life_time=fixed_life_time, interpol_rp=interpol_rp, interpol_prob=interpol_prob, interpol_rp_range=interpol_rp_range, amax=amax, intensity_unit=intensity_unit, tr_max=tr_max, legend_location=legend_location, lang=lang)

	plot.__doc__ = common_plot_docstring


class UHSFieldTree(HazardTree, HazardField, HazardSpectrum):
	def __init__(self, model_name, branch_names, filespecs, weights, sites, periods, IMT, intensities, intensity_unit="g", timespan=50, poe=None, return_period=None, mean=None, percentile_levels=None, percentiles=None, vs30s=None):
		if return_period:
			hazard_values = ExceedanceRateArray([1./return_period])
		elif poe:
			hazard_values = ProbabilityArray([poe])
		HazardTree.__init__(self, hazard_values, branch_names, weights=weights, timespan=timespan, IMT=IMT, intensities=intensities, intensity_unit=intensity_unit, mean=mean, percentile_levels=percentile_levels, percentiles=percentiles)
		HazardField.__init__(self, sites)
		HazardSpectrum.__init__(self, periods)
		self.model_name = model_name
		self.filespecs = filespecs
		self.period_axis = 2
		self.vs30s = vs30s

	def __iter__(self):
		self._current_index = 0
		return self

	def next(self):
		try:
			branch_name = self.branch_names[self._current_index]
		except:
			raise StopIteration
		else:
			self._current_index += 1
			return self.getUHSField(self._current_index-1)

	def __getitem__(self, branch_spec):
		return self.getUHSField(branch_spec)

	@property
	def num_branches(self):
		## Override HazardTree property
		return self.intensities.shape[1]

	@classmethod
	def from_branches(self, uhsf_list, model_name, branch_names=None, weights=None, mean=None, percentile_levels=None, percentiles=None):
		"""
		Construct spectral hazard curve field tree from spectral hazard curve fields
		for different logic-tree branches.

		:param uhsf_list:
			list with instances of :class:`UHSField`
		:param model_name:
			str, model name
		:param branch_names:
			list of branch names (default: None)
		:param weights:
			1-D list or array [j] with branch weights (default: None)
		:param mean:
			instance of :class:`UHSField`, representing
			mean uhsf (default: None)
		:param percentiles:
			list with instances of :class:`UHSField`,
			representing uhsf's corresponding to percentiles (default: None)
		:param percentile_levels:
			list or array with percentile levels (default: None)

		:return:
			instance of :class:`SpectralHazardCurveFieldTree`
		"""
		uhsf0 = uhsf_list[0]
		num_branches = len(uhsf_list)
		num_sites = uhsf0.num_sites
		num_periods = uhsf0.num_periods

		intensities = np.zeros((num_sites, num_branches, num_periods))

		for j, uhsf in enumerate(uhsf_list):
			intensities[:,j,:] = uhsf.intensities

		filespecs = [uhsf.filespec for uhsf in uhsf_list]
		if branch_names in (None, []):
			branch_names = [uhsf.model_name for uhsf in uhsf_list]
		if weights in (None, []):
			weights = np.ones(num_branches, 'f') / num_branches

		uhsft = UHSFieldTree(model_name, branch_names, filespecs, weights, uhsf0.sites, uhsf0.periods, uhsf0.IMT, intensities, uhsf0.intensity_unit, uhsf0.timespan, uhsf0.poe, uhsf0.return_period)

		## Set mean and percentiles
		if mean != None:
			uhsft.set_mean(mean.intensities)

		if percentiles != None:
			num_percentiles = len(percentiles)
			perc_array = np.zeros((num_sites, num_periods, num_percentiles), 'd')
			for p in range(num_percentiles):
				uhsf = percentiles[p]
				perc_array[:,:,p] = uhsf.intensities
			uhsft.set_percentiles(perc_array, percentile_levels)

		return uhsft

	def check_uhsf_compatibility(self, uhsf):
		"""
		Check the compatibility of a candidate branch.

		:param uhsf:
			instance of :class:`UHSField` or higher
		"""
		if self.sites != uhsf.sites:
			raise Exception("Sites do not correspond!")
		if (self.periods != uhsf.periods).any():
			raise Exception("Spectral periods do not correspond!")
		if self.IMT != uhsf.IMT:
			raise Exception("IMT does not correspond!")
		if self.intensity_unit != uhsf.intensity_unit:
			raise Exception("Intensity unit does not correspond!")
		if self.timespan != uhsf.timespan:
			raise Exception("Time span does not correspond!")
		if (self._hazard_values.__class__ != uhsf._hazard_values.__class__
			or (self._hazard_values != uhsf._hazard_values).any()):
			raise Exception("Hazard array does not correspond!")

	def extend(self, uhsft):
		"""
		Extend UHS field tree in-place with another one.

		:param uhsft:
			instance of :class:`UHSFieldTree`
		"""
		self.check_uhsf_compatibility(uhsft)
		self.branch_names.extend(uhsft.branch_names)
		if uhsft.filespecs:
			self.filespecs.extend(uhsft.filespecs)
		else:
			self.filespecs = []
		self.weights = np.concatenate([self.weights, uhsft.weights])
		self.intensities = np.concatenate([self.intensities, uhsft.intensities], axis=1)
		## Remove mean and percentiles
		self.mean = None
		self.percentiles = None
		self.normalize_weights()


	def getUHSField(self, branch_spec):
		branch_index = self.branch_index(branch_spec)
		try:
			branch_name = self.branch_names[branch_index]
		except:
			raise IndexError("Branch index %s out of range" % branch_index)
		else:
			branch_name = self.branch_names[branch_index]
			filespec = self.filespecs[branch_index]

			intensities = self.intensities[:,branch_index,:]
			return UHSField(branch_name, filespec, self.sites, self.periods, self.IMT, intensities, self.intensity_unit, self.timespan, return_period=self.return_period, vs30s=self.vs30s)

	@property
	def poe(self):
		return self.poes[0]

	@property
	def return_period(self):
		return self.return_periods[0]

	@property
	def exceedance_rate(self):
		return self.exceedance_rates[0]

	def min(self):
		# TODO: not really useful
		return self.intensities.min(axis=1)

	def max(self):
		return self.intensities.max(axis=1)

	def calc_mean(self, weighted=True):
		"""
		This is not exactly the same as computing the mean spectral hazard curve,
		and then interpolating at a particular return period
		"""
		if weighted:
			return np.average(self.intensities, weights=self.weights, axis=1)
		else:
			return np.mean(self.intensities, axis=1)

	def getMeanUHSField(self, recalc=False, weighted=True):
		"""
		"""
		if recalc or self.mean is None:
			intensities = self.calc_mean(weighted=weighted)
		else:
			intensities = self.mean

		model_name = "Mean(%s)" % self.model_name

		return UHSField(model_name, "", self.sites, self.periods,
			self.IMT, intensities, self.intensity_unit, self.timespan,
			self.poe, self.return_period, self.vs30s)

	def getPercentileUHSField(self, perc, recalc=False, weighted=True):
		if recalc or self.mean is None or not perc in self.percentiles:
			intensities = self.calc_percentiles([perc], weighted=weighted)[:,:,0]
		else:
			perc_index = self.percentiles.index(perc)
			intensities = self.percentiles[:,:,perc_index]

		model_name = "Perc%02d(%s)" % (perc, self.model_name)

		return UHSField(model_name, "", self.sites, self.periods,
			self.IMT, intensities, self.intensity_unit, self.timespan,
			self.poe, self.return_period, self.vs30s)

	def calc_variance_of_mean(self, weighted=True):
		if weighted and not self.weights in ([], None):
			mean = self.calc_mean(weighted=True)
			weights = np.array(self.weights)
			weights_column = weights.reshape((self.num_branches, 1))
			variance_of_mean = np.zeros((self.num_sites, self.num_periods), 'd')
			for i in range(self.num_sites):
				for k in range(self.num_periods):
					variance_of_mean[i,k] = np.add.reduce(weights_column * (self.intensities[i,:,k] - mean[i,k])**2, axis=0)
		else:
			 variance_of_mean = np.var(self.intensities, axis=1)
		return variance_of_mean

	def calc_percentiles(self, percentile_levels, weighted=True):
		if percentile_levels in ([], None):
			percentile_levels = [5, 16, 50, 84, 95]
		num_sites, num_periods = self.num_sites, self.num_periods
		num_percentiles = len(percentile_levels)
		percentiles = np.zeros((num_sites, num_periods, num_percentiles))
		for i in range(num_sites):
			for k in range(num_periods):
				if weighted and self.weights != None and len(np.unique(self.weights)) > 1:
					pmf = NumericPMF.from_values_and_weights(self.intensities[i,:,k], self.weights)
					percentiles[i,k,:] = pmf.get_percentiles(percentile_levels)
				else:
					for p, per in enumerate(percentile_levels):
						percentiles[i,k,p] = scoreatpercentile(self.intensities[i,:,k], per)
		return percentiles

	def export_stats_csv(self, csv_filespec=None, site_index=0, weighted=True):
		if self.mean in (None, []):
			mean = self.calc_mean(weighted=weighted)
		else:
			mean = self.mean
		if self.percentiles in (None, []):
			if self.percentile_levels in (None, []):
				percentile_levels = [5, 16, 50, 84, 95]
			else:
				percentile_levels = self.percentile_levels
			percentiles = self.calc_percentiles(percentile_levels, weighted=weighted)
		else:
			percentiles = self.percentiles
			percentile_levels = self.percentile_levels

		if csv_filespec:
			f = open(csv_filespec, "w")
		else:
			f = sys.stdout
		f.write("Period (s), Mean")
		f.write(", ".join(["P%02d" % p for p in percentile_levels]))
		f.write("\n")
		i = site_index
		for k, period in enumerate(self.periods):
			f.write("%.3E, %.3E" % (period, mean[i,k]))
			f.write(", ".join(["%.3E" % percentiles[i,k,p] for p in percentile_levels]))
			f.write("\n")
		f.close()

	def slice_by_branch_indexes(self, branch_indexes, slice_name, normalize_weights=True):
		"""
		Return a subset (slice) of the logic tree based on branch indexes
		Parameters:
			branch_indexes: list or array of branch indexes
			slice_name: name of this slice
			normalize_weights: boolean indicating whether or not branch weights
				should be renormalized to 1 (default: True)
		Return value:
			UHSFieldTree object
		"""
		model_name = slice_name
		branch_names, filespecs = [], []
		for index in branch_indexes:
			branch_names.append(self.branch_names[index])
			filespecs.append(self.filespecs[index])
		weights = self.weights[branch_indexes]
		## Recompute branch weights
		if normalize_weights:
			weight_sum = np.add.reduce(weights)
			weights /= weight_sum
		sites = self.sites
		periods = self.periods
		IMT = self.IMT
		intensities = self.intensities[:,branch_indexes,:]
		intensity_unit = self.intensity_unit
		timespan = self.timespan
		poe = self.poe
		return_period = self.return_period
		return UHSFieldTree(model_name, branch_names, filespecs, weights, sites, periods, IMT, intensities, intensity_unit, timespan, poe=poe, return_period=return_period)

	def plot(self, site_spec=0, branch_specs=[], colors=[], linestyles=[], linewidths=[], fig_filespec=None, title=None, plot_freq=False, plot_style="loglin", Tmin=None, Tmax=None, amin=None, amax=None, pgm_period=0.02, legend_location=0, lang="en"):
		site_index = self.site_index(site_spec)
		if branch_specs in ([], None):
			branch_indexes = range(self.num_branches)
		else:
			branch_indexes = [self.branch_index(branch_spec) for branch_spec in branch_specs]
		x = self.periods
		datasets, pgm, labels, colors, linewidths, linestyles = [], [], [], [], [], []

		if title is None:
			title = "UHS Tree"
			title += "\nSite: %s" % self.site_names[site_index]

		## Plot individual models
		for branch_index in branch_indexes:
			y = self.intensities[site_index, branch_index]
			datasets.append((x[self.periods>0], y[self.periods>0]))
			if 0 in self.periods:
				pgm.append(y[self.periods==0])
			labels.append("_nolegend_")
			colors.append((0.5, 0.5, 0.5))
			linewidths.append(1)
			linestyles.append('-')

		## Plot overall mean
		if self.mean is None:
			y = self.calc_mean()[site_index]
		else:
			y = self.mean[site_index]
		datasets.append((x[self.periods>0], y[self.periods>0]))
		if 0 in self.periods:
			pgm.append(y[self.periods==0])
		labels.append("_nolegend_")
		colors.append('w')
		linewidths.append(5)
		linestyles.append('-')
		datasets.append((x[self.periods>0], y[self.periods>0]))
		if 0 in self.periods:
			pgm.append(y[self.periods==0])
		labels.append({"en": "Overall Mean", "nl": "Algemeen gemiddelde"}[lang])
		colors.append('r')
		linewidths.append(3)
		linestyles.append('-')

		## Plot percentiles
		if self.percentiles is None:
			if self.percentile_levels is None:
				percentile_levels = [5, 16, 50, 84, 95]
			else:
				percentile_levels = self.percentile_levels
			percentiles = self.calc_percentiles(percentile_levels)
		else:
			percentiles = self.percentiles
			percentile_levels = self.percentile_levels
		percentiles = percentiles[site_index]
		## Manage percentile labels and colors
		perc_labels, perc_colors = {}, {}
		p = 0
		for perc in percentile_levels:
			if not perc_labels.has_key(perc):
				if not perc_labels.has_key(100 - perc):
					perc_labels[perc] = "P%02d" % perc
					perc_colors[perc] = ["b", "g", "r", "c", "m", "k"][p%6]
					p += 1
				else:
					perc_labels[100 - perc] += ", P%02d" % perc
					perc_labels[perc] = "_nolegend_"
					perc_colors[perc] = perc_colors[100 - perc]
		for p, perc in enumerate(percentile_levels):
			label = perc_labels[perc]
			labels.append(perc_labels[perc])
			colors.append(perc_colors[perc])
			linewidths.append(2)
			linestyles.append('--')
			y = percentiles[:,p]
			datasets.append((x[self.periods>0], y[self.periods>0]))
			if 0 in self.periods:
				pgm.append(y[self.periods==0])
		intensity_unit = self.intensity_unit
		plot_hazard_spectrum(datasets, pgm=pgm, pgm_period=pgm_period, labels=labels, colors=colors, linestyles=linestyles, linewidths=linewidths, fig_filespec=fig_filespec, title=title, plot_freq=plot_freq, plot_style=plot_style, Tmin=Tmin, Tmax=Tmax, amin=amin, amax=amax, intensity_unit=intensity_unit, legend_location=legend_location, lang=lang)

	def plot_histogram(self, site_index=0, period_index=0, fig_filespec=None, title=None, bar_color='g', amax=0, da=0.005, lang="en"):
		if title is None:
			title = "Site: %s / Period: %s s\n" % (self.sites[site_index], self.periods[period_index])
			title += "Return period: %.3G yr" % self.return_period
		intensity_unit = self.intensity_unit
		plot_histogram(self.intensities[site_index,:,period_index], weights=self.weights, fig_filespec=fig_filespec, title=title, bar_color=bar_color, amax=amax, da=da, intensity_unit=intensity_unit, lang=lang)


class UHSField(HazardResult, HazardField, HazardSpectrum):
	"""
	UHS Field (UHS at a number of sites for a single return period)
	sites: 1-D list [i] with (lon, lat) tuples of all sites
	IMT
	periods: 1-D array [k] with structural periods
	intensities: 2-D array [i, k] with interpolated intensity values at one
		return period
	timespan:
	poe: probability of exceedance
	or
	return_period: return period
	"""
	def __init__(self, model_name, filespec, sites, periods, IMT, intensities, intensity_unit="g", timespan=50, poe=None, return_period=None, vs30s=None):
		if return_period:
			hazard_values = ExceedanceRateArray([1./return_period])
		elif poe:
			hazard_values = ProbabilityArray([poe])
		HazardResult.__init__(self, hazard_values, timespan=timespan, IMT=IMT, intensities=intensities, intensity_unit=intensity_unit)
		HazardField.__init__(self, sites)
		HazardSpectrum.__init__(self, periods)
		self.model_name = model_name
		self.filespec = filespec
		self.period_axis = 1
		self.vs30s = vs30s

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
			return self.getUHS(self._current_index-1)

	def __getitem__(self, site_spec):
		return self.getUHS(site_spec)

	def getUHS(self, site_spec=0):
		site_index = self.site_index(site_spec)
		try:
			site = self.sites[site_index]
		except:
			raise IndexError("Site index %s out of range" % site_index)
		else:
			site_name = self.site_names[site_index]
			return UHS(self.model_name + " - " + site_name, self.filespec, site, self.periods, self.IMT, self.intensities[site_index], self.intensity_unit, self.timespan, return_period=self.return_period)

	def getHazardMap(self, period_spec=0):
		period_index = self.period_index(period_spec)
		try:
			period = self.periods[period_index]
		except:
			raise IndexError("Period index %s out of range" % period_index)
		intensities = self.intensities[:, period_index]
		return HazardMap(self.model_name, self.filespec, self.sites, period, self.IMT, intensities, intensity_unit=self.intensity_unit, timespan=self.timespan, poe=self.poe, return_period=self.return_period, vs30s=self.vs30s)

	@property
	def poe(self):
		return self.poes[0]

	@property
	def return_period(self):
		return self.return_periods[0]

	@property
	def exceedance_rate(self):
		return self.exceedance_rates[0]

	def min(self):
		return self.intensities.min(axis=0)

	def max(self):
		return self.intensities.max(axis=0)

	def plot(self, site_specs=None, colors=[], linestyles=[], linewidths=[], fig_filespec=None, title=None, plot_freq=False, plot_style="loglin", Tmin=None, Tmax=None, amin=None, amax=None, pgm_period=0.02, legend_location=0, lang="en"):
		if site_specs in (None, []):
			site_indexes = range(self.num_sites)
		else:
			site_indexes = [self.site_index(site_spec) for site_spec in site_specs]
		sites = [self.sites[site_index] for site_index in site_indexes]

		if title is None:
			title = "Model: %s\n" % self.model_name
			title += "UHS for return period %.3G yr" % self.return_period
		datasets, pgm, labels = [], [], []
		x = self.periods
		for site_index in site_indexes:
			y = self.intensities[site_index]
			datasets.append((x[self.periods>0], y[self.periods>0]))
			if 0 in self.periods:
				pgm.append(y[self.periods==0])
			labels.append(self.site_names[site_index])
		intensity_unit = self.intensity_unit
		plot_hazard_spectrum(datasets, pgm=pgm, pgm_period=pgm_period, labels=labels, colors=colors, linestyles=linestyles, linewidths=linewidths, fig_filespec=fig_filespec, title=title, plot_freq=plot_freq, plot_style=plot_style, Tmin=Tmin, Tmax=Tmax, amin=amin, amax=amax, intensity_unit=intensity_unit, legend_location=legend_location, lang=lang)


class UHS(HazardResult, HazardSpectrum):
	"""
	Uniform Hazard Spectrum
	"""
	def __init__(self, model_name, filespec, site, periods, IMT, intensities, intensity_unit="g", timespan=50, poe=None, return_period=None):
		if return_period:
			hazard_values = ExceedanceRateArray([1./return_period])
		elif poe:
			hazard_values = ProbabilityArray([poe])
		HazardResult.__init__(self, hazard_values, timespan=timespan, IMT=IMT, intensities=intensities, intensity_unit=intensity_unit)
		HazardSpectrum.__init__(self, periods)
		self.model_name = model_name
		self.filespec = filespec
		self.site = site
		self.period_axis = None

	def __getitem__(self, period_spec):
		period_index = self.period_index(period_spec)
		try:
			intensity = self.intensities[period_index]
		except IndexError:
			raise IndexError("Period index %s out of range" % period_index)
		else:
			return intensity

	def __add__(self, other_uhs):
		raise Exception("UHSs cannot gennerally be summed!")

	@property
	def poe(self):
		return self.poes[0]

	@property
	def return_period(self):
		return self.return_periods[0]

	@property
	def exceedance_rate(self):
		return self.exceedance_rates[0]

	@property
	def site_name(self):
		return self.site.name

	def plot(self, color="k", linestyle="-", linewidth=2, fig_filespec=None, title=None, plot_freq=False, plot_style="loglin", Tmin=None, Tmax=None, amin=None, amax=None, pgm_period=0.02, legend_location=0, lang="en"):
		if title is None:
			title = "UHS"
			title += "\nSite: %s, Return period: %d yr" % (self.site_name, self.return_periods[0])
		## Plot PGM separately if present
		if 0 in self.periods:
			idx = list(self.periods).index(0)
			pgm = [self.intensities[idx]]
			datasets = [(self.periods[self.periods>0], self.intensities[self.periods>0])]
		else:
			pgm = None
			datasets = [(self.periods, self.intensities)]
		labels = [self.model_name]
		intensity_unit = self.intensity_unit
		plot_hazard_spectrum(datasets, pgm=pgm, pgm_period=pgm_period, labels=labels, colors=[color], linestyles=[linestyle], linewidths=[linewidth], fig_filespec=fig_filespec, title=title, plot_freq=plot_freq, plot_style=plot_style, Tmin=Tmin, Tmax=Tmax, amin=amin, amax=amax, intensity_unit=intensity_unit, legend_location=legend_location, lang=lang)

	def export_csv(self, csv_filespec=None):
		if csv_filespec:
			f = open(csv_filespec, "w")
		else:
			f = sys.stdout
		f.write("Period (s), %s (%s)\n" % (self.IMT, self.intensity_unit))
		for period, intensity in zip(self.periods, self.intensities):
			f.write("%.3E, %.3E\n" % (period, intensity))
		f.close()

	@classmethod
	def from_csv(self, csv_filespec, site, col_spec=1, intensity_unit="g", model_name="", timespan=50, poe=None, return_period=None):
		"""
		Read UHS from a csv file.
		First line should contain column names
		First column should contain periods or frequencies,
		subsequent column()s should contain intensities, only one of which
		will be read.

		:param csv_filespec:
			str, full path to csv file
		:param site:
			SHASite object, representing site for which UHS was computed
		:param col_spec:
			str or int, name or index of column containing intensities to be read
			(default: 1)
		:param intensity_unit:
			str, unit of intensities in csv file (default: "g")
		:param model_name:
			str, name or description of model
		:param timespan:
			float, time span for UHS (default: 50)
		:param poe:
			float, probability of exceedance for UHS (default: None)
		:param return_period:
			float, return period for UHS (default: None)

		:return:
			instance of :class:`UHS`

		Note: either poe or return_period should be specified
		"""
		periods, intensities = [], []
		csv = open(csv_filespec)
		for i, line in enumerate(csv):
			if i == 0:
				col_names = [s.strip() for s in line.split(',')]
				if col_names[0].lower().split()[0] in ("frequency", "freq"):
					freqs = True
				else:
					freqs = False
				if isinstance(col_spec, str):
					col_index = col_names.index(col_spec)
				else:
					col_index = col_spec
			else:
				col_values = line.split(',')
				T = float(col_values[0])
				a = float(col_values[col_index])
				periods.append(T)
				intensities.append(a)
		csv.close()
		periods = np.array(periods)
		if freqs:
			periods = 1./periods
		intensities = np.array(intensities)

		return UHS(model_name, csv_filespec, site, periods, "SA", intensities, intensity_unit="g", timespan=timespan, poe=poe, return_period=return_period)


class UHSFieldSet(HazardResult, HazardField, HazardSpectrum):
	"""
	UHS fields for different return periods
	sites: 1-D list [i] with (lon, lat) tuples of all sites
	IMT
	periods: 1-D array [k] with structural periods
	intensities: 3-D array [p, i, k] with interpolated intensity values at
		different poE's or return periods [p]
	timespan:
	poes: 1-D array or probabilities of exceedance [p]
	or
	return_periods: 1-D [p] array of return periods
	"""
	def __init__(self, model_name, filespecs, sites, periods, IMT, intensities, intensity_unit="g", timespan=50, poes=None, return_periods=None):
		if not return_periods in (None, []):
			hazard_values = ExceedanceRateArray(1./return_periods)
		elif not poes in (None, []):
			hazard_values = ProbabilityArray(poes)
		HazardResult.__init__(self, hazard_values, timespan=timespan, IMT=IMT, intensities=intensities, intensity_unit=intensity_unit)
		HazardField.__init__(self, sites)
		HazardSpectrum.__init__(self, periods)
		self.model_name = model_name
		if len(filespecs) == 1:
			filespecs *= len(self.return_periods)
		self.filespecs = filespecs

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
			return self.getUHSField(index=self._current_index-1)

	def __getitem__(self, index):
		return self.getUHSField(index=index)

	def __len__(self):
		return len(self.return_periods)

	def getUHSField(self, index=None, poe=None, return_period=None):
		if index is None:
			if poe is not None:
				index = np.where(np.abs(self.poes - poe) < 1E-12)[0]
				if len(index) == 0:
					raise ValueError("No UHS field for poE=%s" % poe)
				else:
					index = index[0]
			elif return_period is not None:
				index = np.where(np.abs(self.return_periods - return_period) < 1E-1)[0]
				if len(index) == 0:
					raise ValueError("No UHS field for return period=%s yr" % return_period)
				else:
					index = index[0]

		try:
			return_period = self.return_periods[index]
		except:
			raise IndexError("Index %s out of range" % index)
		else:
			filespec = self.filespecs[index]
			intensities = self.intensities[index]
			return UHSField(self.model_name, filespec, self.sites, self.periods, self.IMT, intensities, self.intensity_unit, self.timespan, return_period=return_period)

	def interpolate_period(self, period):
		"""
		Should yield a HazardMapSet
		"""
		num_sites, num_rp = self.num_sites, len(self.return_periods)
		period_intensities = np.zeros((num_rp, num_sites))
		for i in range(num_sites):
			for p in range(num_rp):
				period_intensities[p,i] = interpolate(self.periods, self.intensities[p,i], [period])
		return HazardMapSet(self.model_name, self.filespecs, self.sites, period, self.IMT, period_intensities, self.intensity_unit, self.timespan, return_periods=self.return_periods)

	def plot(self, sites=None, return_periods=None):
		"""
		method to plot hazard spectrum
		arguments:
			sites | default="All"
		"""
		if sites is None:
			sites = self.sites
		if return_periods is None:
			return_periods = self.return_periods
		x = self.periods
		datasets, labels = [], []
		for i, site in enumerate(sites):
			for p, rp in enumerate(return_periods):
				y = self.intensities[p,i]
				datasets.append((x, y))
				labels.append("%s, %s - %s" % (site[0], site[1], rp))
		plot_hazard_spectrum(datasets, labels, plot_freq=True)


class UHSCollection:
	"""
	Container for an arbitrary set of Uniform Hazard Spectra
	"""
	def __init__(self, UHSlist, colors=[], linestyles=[], linewidths=[], labels=[]):
		self.UHSlist = UHSlist
		self.colors = colors
		self.linestyles = linestyles
		self.linewidths = linewidths
		if not labels:
			labels = [uhs.model_name for uhs in self.UHSlist]
		self.labels = labels

	def validate(self):
		# intensity unit, IMT should be the same
		pass

	def __len__(self):
		return len(self.UHSlist)

	@property
	def intensity_unit(self):
		return self.UHSlist[0].intensity_unit

	@classmethod
	def from_csv(self, csv_filespec, site, intensity_unit="g", model_name="", timespan=50, poe=None, return_period=None):
		"""
		Read UHSCollection from a csv file.
		First line should contain column names
		First column should contain periods or frequencies,
		subsequent columns should contain intensities
		Each intensity column will represent a UHS in the collection

		:param csv_filespec:
			str, full path to csv file
		:param site:
			SHASite object, representing site for which UHS was computed
		:param intensity_unit:
			str, unit of intensities in csv file (default: "g")
		:param model_name:
			str, name or description of model
		:param timespan:
			float, time span for UHS (default: 50)
		:param poe:
			float, probability of exceedance for UHS (default: None)
		:param return_period:
			float, return period for UHS (default: None)

		:return:
			instance of :class:`UHSCollection`

		Note: either poe or return_period should be specified
		"""
		uhs_list = []
		periods, intensities = [], []
		csv = open(csv_filespec)
		for i, line in enumerate(csv):
			if i == 0:
				col_names = line.split(',')
				if col_names[0].lower() == "frequency":
					freqs = True
				else:
					freqs = False
			else:
				col_values = map(float, line.split(','))
				T = col_values[0]
				a = col_values[1:]
				periods.append(T)
				intensities.append(a)
		csv.close()
		periods = np.array(periods)
		if freqs:
			periods = 1./periods

		intensities = np.array(intensities).transpose()
		for i, uhs_intensities in enumerate(intensities):
			model_name = col_names[i+1]
			uhs = UHS(model_name, csv_filespec, site, periods, "SA", uhs_intensities, intensity_unit="g", timespan=timespan, poe=poe, return_period=return_period)
			uhs_list.append(uhs)

		return UHSCollection(uhs_list)

	def plot(self, fig_filespec=None, title=None, plot_freq=False, plot_style="loglin", Tmin=None, Tmax=None, amin=None, amax=None, pgm_period=0.02, legend_location=0, lang="en"):
		if title is None:
			title = "UHS Collection"
		uhs0 = self.UHSlist[0]
		if 0 in uhs0.periods:
			pgm = [uhs.intensities[uhs.periods==0] for uhs in self.UHSlist]
			datasets = [(uhs.periods[uhs.periods>0], uhs.intensities[uhs.periods>0]) for uhs in self.UHSlist]
		else:
			pgm = None
			datasets = [(uhs.periods, uhs.intensities) for uhs in self.UHSlist]


		intensity_unit = self.intensity_unit
		plot_hazard_spectrum(datasets, pgm=pgm, pgm_period=pgm_period, labels=self.labels, colors=self.colors, linestyles=self.linestyles, linewidths=self.linewidths, fig_filespec=fig_filespec, title=title, plot_freq=plot_freq, plot_style=plot_style, Tmin=Tmin, Tmax=Tmax, amin=amin, amax=amax, intensity_unit=intensity_unit, legend_location=legend_location, lang=lang)


class HazardMap(HazardResult, HazardField):
	"""
	Class representing a hazard map or a ground-motion field
	One hazard map, corresponds to 1 OQ file
	sites: 1-D list [i] with (lon, lat) tuples of all sites
	intensities: 1-D array [i]
	"""
	def __init__(self, model_name, filespec, sites, period, IMT, intensities, intensity_unit="g", timespan=50, poe=None, return_period=None, vs30s=None):
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
				if self.vs30s != None:
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
		grid_intensities = self.get_grid_intensities(grid_extent, num_grid_cells, method)
		intensities = grid_intensities.flatten()

		model_name = self.model_name + " (interpolated)"
		filespec = self.filespec
		sites = zip(grid_lons.flatten(), grid_lats.flatten())
		period = self.period
		IMT = self.IMT
		intensity_unit = self.intensity_unit
		timespan = self.timespan
		poe = self.poe
		return_period = self.return_period
		vs30s = None

		return HazardMap(model_name, filespec, sites, period, IMT, intensities, intensity_unit, timespan, poe, return_period, vs30s)

	def get_residual_map(self, other_map, grid_extent=(None, None, None, None), num_grid_cells=50, interpol_method="linear"):
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

		:return:
			instance of :class:`HazardMap`
		"""
		if self.sites == other_map.sites:
			residuals = self.intensities - other_map.intensities
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
			grid_intensities1 = self.get_grid_intensities(grid_extent, num_grid_cells, interpol_method)
			grid_intensities2 = other_map.get_grid_intensities(grid_extent, num_grid_cells, interpol_method)
			residuals = (grid_intensities1 - grid_intensities2).flatten()
			residuals[np.isnan(residuals)] = 0.
			sites = zip(grid_lons.flatten(), grid_lats.flatten())
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

	def export_kml(self):
		# TODO!
		pass

	def get_plot(self, region=None, projection="merc", resolution="i", grid_interval=(1., 1.), cmap="usgs", norm=None, contour_interval=None, num_grid_cells=100, plot_style="cont", contour_line_style="default", site_style="default", source_model="", source_model_style="default", countries_style="default", coastline_style="default", intensity_unit="g", hide_sea=False, title=None):
		# TODO: update docstring
		"""
		Plot hazard map

		:param cmap:
			String or matplotlib.cm.colors.Colormap instance: Color map
			Some nice color maps are: jet, spectral, gist_rainbow_r, Spectral_r, usgs
			(default: "usgs")
		:param contour_interval:
			Float, ground-motion contour interval (default: None = auto)
		:param intensity_levels:
			List or array of intensity values (in ascending order) that will
			be uniformly spaced in color space. If None or empty list, linear
			normalization will be applied.
			(default: [0., 0.02, 0.06, 0.14, 0.30, 0.90])
		:param num_grid_cells:
			Int, number of grid cells used for interpolating intensity grid
			(default: 100)
		:param plot_style:
			String, either "disc" for discrete or "cont" for continuous
			(default: "cont")
		:param site_symbol:
			Char, matplotlib symbol specification for plotting site points
			(default: ".")
		:param site_color:
			matplotlib color specification for plotting site point symbols
			(default: "w")
		:param site_size:
			Int, size of site point symbols (default: 6)
		:param region:
			(w, e, s, n) tuple specifying rectangular region to plot in
			geographic coordinates (default: None)
		:param projection:
			String, map projection. See Basemap documentation
			(default: "cyl")
		:param resolution:
			String, map resolution (coastlines and country borders):
			'c' (crude), 'l' (low), 'i' (intermediate), 'h' (high), 'f' (full)
			(default: 'i')
		:param dlon:
			Float, meridian interval in degrees (default: 1.)
		:param dlat:
			Float, parallel interval in degrees (default: 1.)
		:param source_model:
			String, name of source model to overlay on the plot
			(default: None)
		:param title:
			String, plot title. If None, title will be generated automatically,
			if empty string, title will not be plotted (default: None)
		:param fig_filespec:
			String, full path of image to be saved.
			If None (default), map is displayed on screen.
		:param fig_width:
			Float, figure width in cm, used to recompute :param:`dpi` with
			respect to default figure width (default: 0)
		:param dpi:
			Int, image resolution in dots per inch (default: 300)
		"""
		import mapping.Basemap as lbm

		## Construct default styles:
		if site_style == "default":
			site_style = lbm.PointStyle(shape="x", line_color="w", size=2.5)
		if source_model_style == "default":
			source_model_style = lbm.PolygonStyle(line_width=2, fill_color="none")
		if countries_style == "default":
			countries_style = lbm.LineStyle(line_width=2, line_color="w")
		if coastline_style == "default":
			coastline_style = lbm.LineStyle(line_width=2, line_color="w")
		if contour_line_style == "default":
			contour_label_style = lbm.TextStyle(font_size=10)
			contour_line_style = lbm.LineStyle(label_style=contour_label_style)

		## Prepare intensity grid and contour levels
		longitudes, latitudes = self.longitudes, self.latitudes
		grid_lons, grid_lats = self.meshgrid(num_cells=num_grid_cells)
		intensity_grid = self.get_grid_intensities(num_cells=num_grid_cells, intensity_unit=intensity_unit)
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

		amin = np.floor(self.min(intensity_unit) / contour_interval) * contour_interval
		amax = np.ceil(self.max(intensity_unit) / contour_interval) * contour_interval

		if contour_interval != None:
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
		if isinstance(cmap, str):
			if cmap.lower() in ("usgs", "share", "gshap"):
				cmap_name = cmap
				cmap = lbm.cm.get_cmap("hazard", cmap_name)
				if norm is None:
					norm = lbm.cm.get_norm("hazard", cmap_name)
			else:
				cmap = matplotlib.cm.get_cmap(cmap)

		if isinstance(norm, str):
			if norm.lower() in ("usgs", "share", "gshap"):
				norm = lbm.cm.get_norm("hazard", norm)

		## Intensity grid
		if self.IMT == "SA":
			imt_label = "%s (%s s)" % (self.IMT, self.period)
		else:
			imt_label = self.IMT
		intensity_unit_label = "%s" % intensity_unit
		if intensity_unit == "ms2":
			intensity_unit_label = "$m/s^2$"
		elif intensity_unit == "cms2":
			intensity_unit_label = "$cm/s^2$"
		cbar_label = imt_label
		if intensity_unit:
			cbar_label += ' (%s)' % intensity_unit_label

		map_layers = []
		colorbar_style = lbm.ColorbarStyle(location="bottom", format="%.2f", ticks=contour_levels, title=cbar_label)
		color_map_theme = lbm.ThematicStyleColormap(color_map=cmap, norm=norm, vmin=amin, vmax=amax, colorbar_style=colorbar_style)
		color_gradient = {"cont": "continuous", "disc": "discontinuous"}[plot_style]
		grid_style = lbm.GridStyle(color_map_theme=color_map_theme, color_gradient=color_gradient, line_style=contour_line_style, contour_levels=contour_levels)
		grid_data = lbm.GridData(grid_lons, grid_lats, intensity_grid)
		layer = lbm.MapLayer(grid_data, grid_style)
		map_layers.append(layer)

		## Intensity data points
		if site_style:
			site_data = lbm.MultiPointData(longitudes, latitudes)
			map_layers.append(lbm.MapLayer(site_data, site_style))

		if hide_sea:
			continent_style = lbm.FocmecStyle(fill_color=(1, 1, 1, 0), bg_color=(1, 1, 1, 1), line_width=0, line_color="none")
			data = lbm.BuiltinData("continents")
			map_layers.append(lbm.MapLayer(data, continent_style))

		## Coastlines and national boundaries
		if coastline_style:
			map_layers.append(lbm.MapLayer(lbm.BuiltinData("coastlines"), coastline_style))
		if countries_style:
			map_layers.append(lbm.MapLayer(lbm.BuiltinData("countries"), countries_style))

		## Source model
		if source_model and source_model_style:
			from eqcatalog.source_models import rob_source_models_dict
			gis_filespec = rob_source_models_dict[source_model].gis_filespec
			gis_data = lbm.GisData(gis_filespec)
			gis_style = lbm.CompositeStyle(polygon_style=source_model_style, line_style=source_model_style.to_line_style())
			map_layers.append(lbm.MapLayer(gis_data, gis_style, legend_label={"polygons": "Source model"}))

		## Title
		if title is None:
			title = "%s\n%.4G yr return period" % (self.model_name, self.return_period)

		legend_style = lbm.LegendStyle(location=0)

		map = lbm.LayeredBasemap(map_layers, title, projection, region=region, grid_interval=grid_interval, resolution=resolution, annot_axes="SE", legend_style=legend_style)
		return map

	def plot(self, cmap="usgs", contour_interval=None, intensity_levels=[0., 0.02, 0.06, 0.14, 0.30, 0.90], num_grid_cells=100, plot_style="cont", site_symbol=".", site_color="w", site_size=6, source_model="", region=None, projection="cyl", resolution="i", dlon=1., dlat=1., hide_sea=False, title=None, fig_filespec=None, fig_width=0, dpi=300):
		"""
		Plot hazard map

		:param cmap:
			String or matplotlib.cm.colors.Colormap instance: Color map
			Some nice color maps are: jet, spectral, gist_rainbow_r, Spectral_r, usgs
			(default: "usgs")
		:param contour_interval:
			Float, ground-motion contour interval (default: None = auto)
		:param intensity_levels:
			List or array of intensity values (in ascending order) that will
			be uniformly spaced in color space. If None or empty list, linear
			normalization will be applied.
			(default: [0., 0.02, 0.06, 0.14, 0.30, 0.90])
		:param num_grid_cells:
			Int, number of grid cells used for interpolating intensity grid
			(default: 100)
		:param plot_style:
			String, either "disc" for discrete or "cont" for continuous
			(default: "cont")
		:param site_symbol:
			Char, matplotlib symbol specification for plotting site points
			(default: ".")
		:param site_color:
			matplotlib color specification for plotting site point symbols
			(default: "w")
		:param site_size:
			Int, size of site point symbols (default: 6)
		:param region:
			(w, e, s, n) tuple specifying rectangular region to plot in
			geographic coordinates (default: None)
		:param projection:
			String, map projection. See Basemap documentation
			(default: "cyl")
		:param resolution:
			String, map resolution (coastlines and country borders):
			'c' (crude), 'l' (low), 'i' (intermediate), 'h' (high), 'f' (full)
			(default: 'i')
		:param dlon:
			Float, meridian interval in degrees (default: 1.)
		:param dlat:
			Float, parallel interval in degrees (default: 1.)
		:param source_model:
			String, name of source model to overlay on the plot
			(default: None)
		:param title:
			String, plot title. If None, title will be generated automatically,
			if empty string, title will not be plotted (default: None)
		:param fig_filespec:
			String, full path of image to be saved.
			If None (default), map is displayed on screen.
		:param fig_width:
			Float, figure width in cm, used to recompute :param:`dpi` with
			respect to default figure width (default: 0)
		:param dpi:
			Int, image resolution in dots per inch (default: 300)
		"""
		from mpl_toolkits.basemap import Basemap
		import matplotlib.cm as cm

		## Prepare intensity grid and contour levels
		longitudes, latitudes = self.longitudes, self.latitudes
		grid_lons, grid_lats = self.meshgrid(num_cells=num_grid_cells)
		intensity_grid = self.get_grid_intensities(num_cells=num_grid_cells)
		if not contour_interval:
			arange = self.max() - self.min()
			candidates = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 1.0])
			try:
				index = np.where(arange / candidates <= 10)[0][0]
			except IndexError:
				index = 0
			contour_interval = candidates[index]

		#if intensity_levels in (None, []):
		#	amin, amax = 0., self.max()
		#else:
		#	amin, amax = intensity_levels[0], intensity_levels[-1]
		amin = np.floor(self.min() / contour_interval) * contour_interval
		amax = np.ceil(self.max() / contour_interval) * contour_interval

		contour_levels = np.arange(amin, amax+contour_interval, contour_interval)
		## Sometimes, there is an empty contour interval at the end
		if len(contour_levels) > 1 and contour_levels[-2] > self.max():
			contour_levels = contour_levels[:-1]

		## Compute map limits
		if not region:
			llcrnrlon, llcrnrlat = min(longitudes), min(latitudes)
			urcrnrlon, urcrnrlat = max(longitudes), max(latitudes)
		else:
			llcrnrlon=region[0]
			llcrnrlat=region[2]
			urcrnrlon=region[1]
			urcrnrlat=region[3]
		lon_0 = (llcrnrlon + urcrnrlon) / 2.
		lat_0 = (llcrnrlat + urcrnrlat) / 2.

		map = Basemap(projection=projection, resolution=resolution, llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, lon_0=lon_0, lat_0=lat_0)
		if hide_sea:
			rgba_land, rgba_ocean = (255, 255, 255, 0), (255, 255, 255, 255)
			map.drawlsmask(rgba_land, rgba_ocean)

		## Color map
		if cmap == "usgs":
			usgs_rgb = [(255, 255, 255),
						(230, 230, 230),
						(200, 200, 200),
						(231, 255, 255),
						(215, 255, 255),
						(198, 255, 255),
						(151, 255, 241),
						(151, 254, 199),
						(200, 255, 153),
						(202, 254, 83),
						(251, 250, 100),
						(255, 238, 0),
						(254, 225, 1),
						(255, 200, 1),
						(255, 94, 0),
						(254, 0, 2),
						(200, 121, 20),
						(151, 74, 20)]
			num_colors = len(usgs_rgb)
			cmap_limits = np.linspace(0, 1, num_colors)
			usgs_cdict = {'red': [], 'green': [], 'blue': []}
			for i in range(num_colors):
				r, g, b = np.array(usgs_rgb[i]) / 255.
				usgs_cdict['red'].append((cmap_limits[i], r, r))
				usgs_cdict['green'].append((cmap_limits[i], g, g))
				usgs_cdict['blue'].append((cmap_limits[i], b, b))
			cmap = matplotlib.colors.LinearSegmentedColormap('usgs', usgs_cdict, 256)
		elif isinstance(cmap, str):
			cmap = cm.get_cmap(cmap)

		## Intensity grid
		if intensity_levels in (None, []):
			norm = matplotlib.colors.Normalize(amin, amax)
		else:
			norm = LevelNorm(intensity_levels)
		x, y = map(grid_lons, grid_lats)
		if plot_style == "disc":
			cs = map.contourf(x, y, intensity_grid, levels=contour_levels, cmap=cmap, norm=norm)
		elif plot_style == "cont":
			cs = map.pcolor(x, y, intensity_grid, vmin=amin, vmax=amax, cmap=cmap, norm=norm)
		cl = map.contour(x, y, intensity_grid, levels=contour_levels, colors='k', linewidths=1)
		pylab.clabel(cl, inline=True, fontsize=10, fmt='%.2f')
		cbar = map.colorbar(cs, location='bottom', pad="10%", format='%.2f', spacing="uniform", ticks=contour_levels)

		## Intensity data points
		if site_symbol:
			x, y = map(longitudes, latitudes)
			map.scatter(x, y, s=site_size, marker=site_symbol, facecolors=site_color, label=None)

		## Coastlines and national boundaries
		map.drawcoastlines(linewidth=2, color="w")
		map.drawcountries(linewidth=2, color="w")

		## Source model
		if source_model:
			from eqcatalog.source_models import read_source_model
			model_data = read_source_model(source_model)
			for i, zone_data in enumerate(model_data.values()):
				geom = zone_data['obj']
				lines = []
				if geom.GetGeometryName() == "LINESTRING":
					lines.append(geom.GetPoints())
				elif geom.GetGeometryName() == "POLYGON":
					for linear_ring in geom:
						lines.append(linear_ring.GetPoints())
				for line in lines:
					lons, lats = zip(*line)
					x, y = map(lons, lats)
					if i == 0:
						label = source_model
					else:
						label = "_nolegend_"
					map.plot(x, y, ls='-', lw=2, color='k')

		## Meridians and parallels
		first_meridian = np.ceil(llcrnrlon / dlon) * dlon
		last_meridian = np.floor(urcrnrlon / dlon) * dlon + dlon
		meridians = np.arange(first_meridian, last_meridian, dlon)
		map.drawmeridians(meridians, labels=[0,1,0,1])
		first_parallel = np.ceil(llcrnrlat / dlat) * dlat
		last_parallel = np.floor(urcrnrlat / dlat) * dlat + dlat
		parallels = np.arange(first_parallel, last_parallel, dlat)
		map.drawparallels(parallels, labels=[0,1,0,1])
		map.drawmapboundary()

		## Title
		if title is None:
			title = "%s\n%.4G yr return period" % (self.model_name, self.return_period)
		pylab.title(title)
		if self.IMT == "SA":
			imt_label = "%s (%s s)" % (self.IMT, self.period)
		else:
			imt_label = self.IMT
		cbar.set_label('%s (%s)' % (imt_label, self.intensity_unit))

		if fig_filespec:
			default_figsize = pylab.rcParams['figure.figsize']
			default_dpi = pylab.rcParams['figure.dpi']
			if fig_width:
				fig_width /= 2.54
				dpi = dpi * (fig_width / default_figsize[0])
			pylab.savefig(fig_filespec, dpi=dpi)
			pylab.clf()
		else:
			pylab.show()


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
HazardCurveFieldTree = SpectralHazardCurveFieldTree



if __name__ == "__main__":
	import rhlib.crisis.IO as IO

	## Convert CRISIS .MAP file to HazardMapSet
	#filespec = r"Test files\CRISIS\VG_Ambr95DD_Leynaud_EC8.MAP"
	filespec = r"D:\PSHA\LNE\CRISIS\VG_Ambr95DD_Leynaud_EC8.MAP"
	hazardmapset = IO.read_MAP(filespec, model_name="VG_Ambr95DD_Leynaud_EC8", convert_to_g=True, IMT="PGA", verbose=True)
	print hazardmapset.longitudes
	for hazardmap in hazardmapset:
		print hazardmap.intensities.shape
		print hazardmap.return_period
	hazardmap = hazardmapset[0]
	intensity_grid = hazardmap.create_grid()
	print intensity_grid.shape
	print hazardmap.min(), hazardmap.max()
	print hazardmap.poe
	#hazardmap.export_VM(r"C:\Temp\hazardmap.grd")
	hazardmap.plot(parallel_interval=0.5, hide_sea=True, want_basemap=False)
	print


	## Convert CRISIS .GRA file to SpectralHazardCurveField
	"""
	#filespec = r"Test files\CRISIS\MC000.GRA"
	#filespec = r"D:\PSHA\NIRAS\LogicTree\PGA\Seismotectonic\BergeThierry2003\3sigma\Mmax+0_00\MC000.GRA"
	#filespec = r"D:\PSHA\NIRAS\LogicTree\Spectral\Seismotectonic\BergeThierry2003\3sigma\Mmax+0_00\MC000.GRA"
	filespec = r"D:\PSHA\NIRAS\LogicTree\Spectral\Spectral_BT2003.AGR"
	#filespec = r"D:\PSHA\LNE\CRISIS\VG_Ambr95DD_Leynaud_EC8.gra"
	shcf = IO.read_GRA(filespec, verbose=True)
	print shcf.intensities.shape, shcf.exceedance_rates.shape
	print shcf.mean.shape
	print shcf.percentiles.shape
	print shcf.percentile_levels
	print shcf.periods
	print shcf.sites
	shc = shcf[0]
	hc = shc[0]
	print hc.interpolate_return_periods([1E+3, 1E+4, 1E+5])
	#hc.plot(want_poe=False)
	UHSfieldset = shcf.interpolate_return_periods([10000])
	UHSfield = UHSfieldset[0]
	#UHSfield.plot()
	uhs = UHSfield[0]
	print uhs.periods, uhs.intensities
	print uhs[1./34]
	uhs.export_csv()
	uhs.plot(linestyle='x', Tmin=1./40, Tmax=1./30, amin=0, color='r')
	print UHSfieldset.return_periods
	print UHSfieldset.intensities
	print UHSfieldset.poes
	hazardmapset = UHSfieldset.interpolate_period(1./34)
	print hazardmapset.return_periods
	print hazardmapset.poes
	hazardmap = hazardmapset[-1]
	print hazardmap.poe
	"""


	## CRISIS --> HazardCurveFieldTree
	"""
	import hazard.psha.BEST_IRE.LogicTree as LogicTree
	GRA_filespecs = LogicTree.slice_logictree()
	hcft = IO.read_GRA_multi(GRA_filespecs, model_name="Test")
	hcft.plot()
	#hcft.write_statistics_AGR(r"C:\Temp\Test.AGR", weighted=False)
	#hcft2 = hcft.slice_by_branch_indexes(range(50), "Subset")
	hcft2 = hcft.slice_by_branch_names(["BergeThierry"], "BergeThierry", strict=False)
	print len(hcft2)
	import np.random
	hcft2.weights = np.random.rand(len(hcft2))
	uhsft = hcft2.interpolate_return_period(10000)
	uhsft.plot_histogram()
	#hcft.plot()
	hcft.plot_subsets(["BergeThierry", "Ambraseys"], labels=["BergeThierry", "Ambraseys"], percentile_levels=[84,95])
	hcf = hcft[0]
	print hcf.periods
	hcf.plot()
	#hcf2 = hcf.interpolate_periods([0.05, 0.275, 1, 2, 10])
	shc = hcf.getSpectralHazardCurve(0)
	#shc.plot()
	uhs = shc.interpolate_return_period(1E4)
	#uhs.plot(plot_freq=True)
	"""