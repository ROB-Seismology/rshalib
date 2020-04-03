"""
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from decimal import Decimal
import numpy as np

import openquake.hazardlib.calc.disagg as disagg

from ..poisson import poisson_conv
from .plot import plot_deaggregation
from .base_array import *
from .deagg_base import DeaggBase
from .hc_base import (IntensityResult, parse_sites)
from .hazard_curve import HazardCurve



__all__ = ['DeaggregationSlice', 'DeaggregationCurve',
			'SpectralDeaggregationCurve', 'get_mean_deaggregation_slice',
			'get_mean_deaggregation_curve']


class DeaggregationSlice(IntensityResult, DeaggBase):
	"""
	Class representing a full deaggregation result for a single
	intensity level, as computed by oqhazlib.

	Deaggregation values represent conditional probability distribution or
	exceedance rate as a function of:
		- rupture magnitude,
		- joyner-boore distance from rupture surface to site,
		- longitude and latitude of surface projection of rupture closest
		  point to site,
		- epsilon: number of standard deviations by which an intensity
		  measure level deviates from the median value predicted by a gsim,
		  given the rupture parameters.
		- rupture tectonic region type
	given the event that an intensity measure type ``imt`` exceeds an
	intensity measure level ``iml`` at a geographical location ``site``.

	:param bin_edges:
		6-tuple, containing:
			- magnitude bin edges
			- distance bin edges
			- longitude bin edges
			- latitude bin edges
			- epsilon bin edges
			- tectonic region types
	:param deagg_matrix:
		instance of :class:`ExceedanceRateMatrix` or :class:`ProbabilityMatrix`,
		6-D array containing deaggregation values, with dimensions
		corresponding to:
			- magnitude bins
			- distance bins
			- longitude bins
			- latitude bins
			- epsilon bins
			- tectonic-region-type bins
	:param site:
		instance of :class:`GenericSite`: site where hazard was computed
	:param iml:
		float, intensity level corresponding to :param:`return_period`
	:param intensity_unit:
		str, unit in which intensity measure levels are expressed:
		PGA and SA: "g", "mg", "m/s2", "gal", "cm/s2"
		PGV: "cm/s"
		PGD: "cm"
	:param imt:
		str, intensity measure type
	:param period:
		float, spectral period
	:param return_period:
		float, return period corresponding to iml
	:param timespan:
		float, time span in Poisson formula.
	:param damping:
		float, damping corresponding to intensities
		(expressed as fraction of critical damping)
		(default: 0.05)
	"""
	def __init__(self, bin_edges, deagg_matrix, site,
				iml, intensity_unit, imt, period,
				return_period, timespan, damping=0.05):
		IntensityResult.__init__(self, [iml], intensity_unit, imt,
								damping=damping)
		DeaggBase.__init__(self, bin_edges, deagg_matrix, timespan)
		self.site = parse_sites([site])[0]
		#self.iml = iml
		self.period = period
		self.return_period = return_period

	@property
	def iml(self):
		return self.intensities[0]

	def to_exceedance_rate(self):
		"""
		Convert deaggregation slice to exceedance rate

		:return:
			instance of :class:`DeaggregationSlice`
		"""
		deagg_matrix = self.deagg_matrix.to_exceedance_rate_matrix(self.timespan)
		return self.__class__(self.bin_edges, deagg_matrix, self.site, self.iml,
							self.intensity_unit, self.imt, self.period,
							self.return_period, self.timespan, self.damping)

	def to_probability(self):
		"""
		Convert deaggregation slice to probability of exceecance

		:return:
			ndarray, 6-D
		"""
		deagg_matrix = self.deagg_matrix.to_probability_matrix(self.timespan)
		return self.__class__(self.bin_edges, deagg_matrix, self.site, self.iml,
							self.intensity_unit, self.imt, self.period,
							self.return_period, self.timespan, self.damping)

	def get_total_probability(self):
		"""
		Return total probability of exceedance
		"""
		return self.deagg_matrix.get_total_probability(self.timespan)

	def get_total_exceedance_rate(self):
		"""
		Return total exceedance rate
		"""
		return self.deagg_matrix.get_total_exceedance_rate(self.timespan)

	def calc_return_period(self):
		"""
		Compute return period corresponding to total probability
		"""
		return 1. / self.get_total_exceedance_rate()

	def plot_mag_dist_pmf(self, title="", fig_filespec=None):
		"""
		Plot magnitude / distance PMF.
		"""
		mag_dist_pmf = self.get_mag_dist_pmf().to_fractional_contribution_matrix()
		if self.neps > 1:
			eps_pmf = self.get_eps_pmf().to_fractional_contribution_matrix()
		else:
			eps_pmf = None
		eps_bin_edges = self.eps_bin_edges[1:]
		if self.ntrts > 1:
			trt_pmf = self.get_trt_pmf().to_fractional_contribution_matrix()
			trt_labels = self.trt_bins
		else:
			trt_pmf = None
			trt_labels = []
		return plot_deaggregation(mag_dist_pmf, self.mag_bin_edges,
								self.dist_bin_edges, self.return_period,
								eps_values=eps_pmf, eps_bin_edges=eps_bin_edges,
								fue_values=trt_pmf, fue_labels=trt_labels,
								mr_style="2D", site_name=self.site.name,
								struc_period=self.period, title_comment=title,
								fig_filespec=fig_filespec)

	def get_modal_eq_scenario(self):
		"""
		Determine modal earthquake scenario (having largest contribution)

		:return:
			(mag, dist) tuple
		"""
		mag_dist_pmf = self.get_mag_dist_pmf()
		mag_index, dist_index = np.unravel_index(mag_dist_pmf.argmax(),
												mag_dist_pmf.shape)
		return (self.mag_bin_centers[mag_index], self.dist_bin_centers[dist_index])

	def get_mean_eq_scenario(self):
		"""
		Determine mean earthquake scenario

		:return:
			(mag, dist) tuple
		"""
		return (self.get_mean_magnitude(), self.get_mean_distance())

	def get_mean_magnitude(self):
		"""
		Return mean magnitude according to formula in DOE-STD-1023-95,
		appendix A, page 5.
		"""
		fcm = self.deagg_matrix.to_fractional_contribution_matrix()
		fcm = fcm.fold_axes([1,2,3,4,5])
		#return float(np.sum(self.mag_bin_centers * fcm))
		return np.average(self.mag_bin_centers, weights=fcm)

	def get_mean_distance(self):
		"""
		Return mean distance according to formula in DOE-STD-1023-95,
		appendix A, page 6.
		"""
		fcm = self.deagg_matrix.to_fractional_contribution_matrix()
		fcm = fcm.fold_axes([0,2,3,4,5])
		#return np.exp(float(np.sum(np.log(self.dist_bin_centers) * fcm)))
		return np.exp(np.average(np.log(self.dist_bin_centers), weights=fcm))

	def get_fractional_contribution_matrix_above(self, threshold, axis,
												renormalize=False):
		"""
		Return deaggregation matrix sliced above a given threshold value
		for a given axis as fractional contribution.
		Note: threshold value will be replaced with bin edge that is
		lower than or equal to threshold.

		:param threshold:
			float, threshold (minimum) value
		:param axis:
			int, axis index
		:param renormalize;
			bool, whether or not resulting matrix should be renormalized
			(default: False)

		:return:
			instance of :class:`FractionalContributionMatrix`
		"""
		axis_bin_edges = self.get_axis_bin_edges(axis)
		start_idx = np.digitize([threshold], axis_bin_edges)[0] - 1
		matrix = self.get_fractional_contribution_matrix()
		matrix = matrix.slice_axis(axis, start=start_idx)
		if renormalize:
			matrix /= np.sum(matrix)
		return matrix

	def get_fractional_contribution_matrix_below(self, threshold, axis,
												renormalize=False):
		"""
		Return deaggregation matrix sliced below a given threshold value
		for a given axis as fractional contribution.
		Note: threshold value will be replaced with bin edge that is
		larger than or equal to threshold.

		:param threshold:
			float, threshold (minimum) value
		:param axis:
			int, axis index
		:param renormalize;
			bool, whether or not resulting matrix should be renormalized
			(default: False)

		:return:
			instance of :class:`FractionalContributionMatrix`
		"""
		axis_bin_edges = self.get_axis_bin_edges(axis)
		stop_idx = np.digitize([threshold], axis_bin_edges)[0]
		matrix = self.get_fractional_contribution_matrix()
		matrix = matrix.slice_axis(axis, stop=stop_idx)
		if renormalize:
			matrix /= np.sum(matrix)
		return matrix

	def get_fractional_contribution_slice_above(self, threshold, axis,
												renormalize=False):
		"""
		Return deaggregation slice sliced above a given threshold value
		for a given axis.
		Note: threshold value will be replaced with bin edge that is
		lower than or equal to threshold.

		:param threshold:
			float, threshold (minimum) value
		:param axis:
			int, axis index
		:param renormalize;
			bool, whether or not resulting matrix should be renormalized
			(default: False)

		:return:
			instance of :class:`DeaggregationSlice`
		"""
		matrix = self.get_fractional_contribution_matrix_above(threshold, axis,
														renormalize=renormalize)
		bin_edges = list(self.bin_edges)
		axis_bin_edges = self.get_axis_bin_edges(axis)
		start_idx = np.digitize([threshold], axis_bin_edges)[0] - 1
		axis_bin_edges = axis_bin_edges[start_idx:]
		bin_edges[axis] = axis_bin_edges
		return self.__class__(tuple(bin_edges), matrix, self.site, self.iml,
								self.intensity_unit, self.imt, self.period,
								self.return_period, self.timespan, self.damping)

	def get_fractional_contribution_slice_below(self, threshold, axis,
												renormalize=False):
		"""
		Return deaggregation slice sliced below a given threshold value
		for a given axis.
		Note: threshold value will be replaced with bin edge that is
		larger than or equal to threshold.

		:param threshold:
			float, threshold (minimum) value
		:param axis:
			int, axis index
		:param renormalize;
			bool, whether or not resulting matrix should be renormalized
			(default: False)

		:return:
			instance of :class:`DeaggregationSlice`
		"""
		matrix = self.get_fractional_contribution_matrix_below(threshold, axis,
														renormalize=renormalize)
		bin_edges = list(self.bin_edges)
		axis_bin_edges = self.get_axis_bin_edges(axis)
		stop_idx = np.digitize([threshold], axis_bin_edges)[0]
		axis_bin_edges = axis_bin_edges[:stop_idx+1]
		bin_edges[axis] = axis_bin_edges
		return self.__class__(tuple(bin_edges), matrix, self.site, self.iml,
							self.intensity_unit, self.imt, self.period,
							self.return_period, self.timespan, self.damping)

	def get_contribution_above(self, threshold, axis):
		"""
		Determine contribution of bins above a threshold value.

		:param threshold:
			Float, threshold value in given axis
		:param axis:
			Int, index of axis

		:return:
			Float, fractional contribution
		"""
		matrix = self.get_fractional_contribution_matrix_above(threshold, axis,
															renormalize=False)
		return np.sum(matrix)

	def get_contribution_below(self, threshold, axis):
		"""
		Determine contribution of bins below a threshold value.

		:param threshold:
			Float, threshold value in given axis
		:param axis:
			Int, index of axis

		:return:
			Float, fractional contribution
		"""
		matrix = self.get_fractional_contribution_matrix_below(threshold, axis,
															renormalize=False)
		return np.sum(matrix)

	def get_contribution_above_magnitude(self, mag):
		"""
		Determine contribution of magnitudes above a threshold magnitude.

		:param mag:
			Float, threshold magnitude

		:return:
			Float, percent contribution
		"""
		return self.get_contribution_above(mag, axis=0)

	def get_contribution_above_distance(self, dist):
		"""
		Determine contribution of distances above a threshold distance.

		:param dist:
			Float, threshold distance

		:return:
			Float, percent contribution
		"""
		return self.get_contribution_above(dist, axis=1)

	def get_trt_slice(self, trt):
		"""
		Obtain deaggregation slice for a given trt

		:param trt:
			str, tectonic region type

		:return:
			instance of :class:`DeaggregationSlice`
		"""
		trt_idx = self.trt_bins.index(trt)
		bin_edges = (self.mag_bin_edges, self.dist_bin_edges, self.lon_bin_edges,
					self.lat_bin_edges, self.eps_bin_edges, [trt])
		deagg_matrix = self.deagg_matrix[:,:,:,:,:,trt_idx:trt_idx+1]
		return self.__class__(bin_edges, deagg_matrix, self.site, self.iml,
							self.intensity_unit, self.imt, self.period,
							self.return_period, self.timespan, self.damping)

	def rebin(self, new_bin_edges, axis=0):
		"""
		Rebin deaggregation slice along a given axis

		:param new_bin_edges:
			array-like, new bin edges
		:param axis:
			Int, axis index

		:return:
			instance of :class:`DeaggregationSlice`
		"""
		axis_bin_edges = self.bin_edges[axis]
		rebinned_deagg_matrix = self.deagg_matrix.rebin_axis(axis, axis_bin_edges,
															new_bin_edges)
		bin_edges = list(self.bin_edges)
		bin_edges[axis] = new_bin_edges
		bin_edges = tuple(bin_edges)
		return self.__class__(bin_edges, rebinned_deagg_matrix, self.site,
							self.iml, self.intensity_unit, self.imt, self.period,
							self.return_period, self.timespan, self.damping)

	def rebin_magnitudes(self, mag_bin_edges):
		"""
		Rebin magnitude bins

		:param mag_bin_edges:
			array-like, new magnitude bin edges
		:param axis:
			Int, axis index

		:return:
			instance of :class:`DeaggregationSlice`
		"""
		return self.rebin(mag_bin_edges, axis=0)

	def rebin_distances(self, dist_bin_edges):
		"""
		Rebin distance bins

		:param dist_bin_edges:
			array-like, new distance bin edges
		:param axis:
			Int, axis index

		:return:
			instance of :class:`DeaggregationSlice`
		"""
		return self.rebin(dist_bin_edges, axis=1)

	def rebin_longitudes(self, lon_bin_edges):
		"""
		Rebin longitude bins

		:param lon_bin_edges:
			array-like, new longitude bin edges
		:param axis:
			Int, axis index

		:return:
			instance of :class:`DeaggregationSlice`
		"""
		return self.rebin(lon_bin_edges, axis=2)

	def rebin_latitudes(self, lat_bin_edges):
		"""
		Rebin latitude bins

		:param lat_bin_edges:
			array-like, new latitude bin edges
		:param axis:
			Int, axis index

		:return:
			instance of :class:`DeaggregationSlice`
		"""
		return self.rebin(lat_bin_edges, axis=3)

	def write_nrml(self, nrml_filespec, smlt_path=None, gmpelt_path=None,
					encoding='latin-1', pretty_print=True):
		"""
		Write deaggregation slice to XML file

		:param filespec:
			str, full path to XML output file
		:param smlt_path:
			str, path to NRML file containing source-model logic tree
			(default: None)
		:param gmpelt_path:
			str, path to NRML file containing ground-motion logic tree
			(default: None)
		:param encoding:
			str, unicode encoding
			(default: 'latin-1')
		pretty_print:
			bool, indicating whether or not to indent each element
			(default: True)
		"""
		from ..openquake.IO import write_disaggregation_slice

		poe = self.deagg_matrix.get_total_probability(timespan=self.timespan)
		matrix = self.deagg_matrix.to_probability_matrix(timespan=self.timespan)
		write_disaggregation_slice(self.site, self.imt, self.period, self.iml,
									poe, self.timespan, self.bin_edges, matrix,
									nrml_filespec, smlt_path, gmpelt_path,
									encoding=encoding, pretty_print=pretty_print)


class DeaggregationCurve(IntensityResult, DeaggBase):
	"""
	Class representing a full deaggregation result for a range of intensities

	:param bin_edges:
		6-tuple, containing:
			- magnitude bin edges
			- distance bin edges
			- longitude bin edges
			- latitude bin edges
			- epsilon bin edges
			- tectonic region types

	:param deagg_matrix:
		instance of :class:`ExceedanceRateMatrix` or :class:`ProbabilityMatrix`,
		7-D array containing deaggregation values, with dimensions
		corresponding to:
			- intensity levels
			- magnitude bins
			- distance bins
			- longitude bins
			- latitude bins
			- epsilon bins
			- tectonic-region-type bins

	:param site:
		instance of :class:`GenericSite`: site where hazard was computed
	:param intensities:
		float array, intensity levels corresponding to
		:param:`return_periods`
	:param intensity_unit:
		str, unit in which intensity measure levels are expressed:
		PGA and SA: "g", "mg", "m/s2", "gal", "cm/s2"
		PGV: "cm/s"
		PGD: "cm"
	:param imt:
		str, intensity measure type
	:param period:
		float, spectral period
	:param return_periods:
		float array, return periods corresponding to intensities
	:param timespan:
		Float, time span in Poisson formula.
	:param damping:
		float, damping corresponding to intensities
		(expressed as fraction of critical damping)
		(default: 0.05)
	"""
	def __init__(self, bin_edges, deagg_matrix, site,
				intensities, intensity_unit, imt,
				period, return_periods, timespan, damping=0.05):
		## Make sure intensities are ordered from small to large
		if intensities[0] > intensities[-1]:
			DeaggBase.__init__(self, bin_edges, deagg_matrix[::-1], timespan)
			self.intensities = intensities[::-1]
		else:
			DeaggBase.__init__(self, bin_edges, deagg_matrix, timespan)
			self.intensities = intensities

		IntensityResult.__init__(self, intensities, intensity_unit, imt,
								damping=damping)
		self.site = parse_sites([site])[0]
		self.period = period
		self.return_periods = np.array(return_periods)

	def __len__(self):
		return len(self.intensities)

	def __iter__(self):
		"""
		Loop over deaggregation slices
		"""
		for iml_index in range(len(self.intensities)):
			yield self.get_slice(iml_index=iml_index)

	def __getitem__(self, iml_index):
		return self.get_slice(iml_index=iml_index)

	def get_intensity_bin_centers(self):
		"""
		Return center values of intensity bins
		Note: in contrast to deaggregation bins, the number of intensity
		bin centers returned is equal to the number of intensities.
		"""
		imls = self.intensities
		log_imls = np.log(imls)
		log_iml_widths = np.diff(log_imls)
		intensity_bin_centers = np.exp(log_imls[:-1] + log_iml_widths / 2.)
		intensity_bin_centers = np.append(intensity_bin_centers,
								np.exp(log_imls[-1] + log_iml_widths[-1] / 2))
		return intensity_bin_centers

	def get_slice(self, iml=None, iml_index=None):
		"""
		Get deaggregation slice for a particular intensity level.

		:param iml:
			Float, intensity level (default: None)
		:param iml_index:
			Int, intensity index (default: None)

		:return:
			instance of :class:`DeaggregationSlice`
		"""
		if iml is not None:
			iml_index = np.argmin(np.abs(self.intensities - iml))
			iml = self.intensities[iml_index]
		else:
			iml = self.intensities[iml_index]
		matrix = self.deagg_matrix[iml_index]

		return DeaggregationSlice(self.bin_edges, matrix, self.site,
								iml, self.intensity_unit, self.imt,
								self.period, self.return_periods[iml_index],
								self.timespan, self.damping)

	def interpolate_curve(self, imls, return_periods=None):
		"""
		Interpolate deaggregation curve at particular intensity levels

		:param imls:
			1-D numpy array
		:param return_periods:
			1-D numpy array with return periods corresponding to imls.
			If None, return periods will be interpolated as well
			(default: None)

		:return:
			instance of :class:`DeaggregationCurve`
		"""
		deagg_matrix = self.deagg_matrix.interpolate_axis(0, self.intensities,
															imls)
		if return_periods is None:
			from ..utils import interpolate
			return_periods = interpolate(self.intensities, self.return_periods,
										imls)

		return DeaggregationCurve(self.bin_edges, deagg_matrix, self.site,
								imls, self.intensity_unit, self.imt, self.period,
								return_periods, self.timespan, self.damping)

	def slice_return_periods(self, return_periods, hc, interpolate_matrix=False):
		"""
		Reduce the spectral deaggregation curve to one where slices
		correspond to return periods. First, intensities are interpolated
		from the given spectral hazard curve, and a new spectral
		deaggregation curve is constructed from slices corresponding
		to these interpolated intensities.

		:param return_periods:
			list of floats, return periods
		:param hc:
			instance of :class:`rshalib.result.HazardCurve`
		:param interpolate_matrix:
			bool, whether or not the deaggregation matrix should be
			interpolated at the intensities interpolated from the
			hazard curve. If False, the nearest slices will be selected
			(default: False)

		:return:
			instance of :class:`DeaggregationCurve`
		"""
		imls = hc.interpolate_return_periods(return_periods)
		if interpolate_matrix is False:
			slices = []
			for iml in imls:
				ds = self.get_slice(iml=iml)
				slices.append(ds)
		else:
			slices = self.interpolate_curve(imls, return_periods=return_periods)
		return DeaggregationCurve.from_deaggregation_slices(slices)

	def get_hazard_curve(self):
		"""
		Get hazard curve corresponding to total exceedance rate or
		probability for each intensity level.

		:return:
			instance of :class:`HazardCurve`
		"""
		from .hc_base import ExceedanceRateArray, ProbabilityArray
		from .hazard_curve import HazardCurve

		hazard_values = self.deagg_matrix.fold_axes([6,5,4,3,2,1])
		if isinstance(self.deagg_matrix, ExceedanceRateMatrix):
			hazard_values = ExceedanceRateArray(hazard_values)
		elif isinstance(self.deagg_matrix, ProbabilityMatrix):
			hazard_values = ProbabilityArray(hazard_values)
		else:
			raise Exception("Not supported for FractionalContributionMatrix!")
		model_name = ""
		filespec = ""

		return HazardCurve(hazard_values, self.site, self.period,
							self.intensities, self.intensity_unit, self.imt,
							model_name=model_name, filespec=filespec,
							timespan=self.timespan, damping=self.damping)

	def get_occurrence_rates(self):
		"""
		Calculate rate of occurrence for each intensity interval (difference
		between exceedance rates of intensity below and above)
		Note: the number of occurrence rates returned is equal to the number
		of exceedance rates (or intensities).

		:return:
			instance of :class:`ExceedanceRateMatrix`
		"""
		exceedance_rates = self.deagg_matrix.to_exceedance_rate_matrix(self.timespan)
		occurrence_rates = exceedance_rates[:-1] - exceedance_rates[1:]
		occurrence_rates = np.append(occurrence_rates, exceedance_rates[-1:], axis=0)
		return ExceedanceRateMatrix(occurrence_rates)

	@classmethod
	def from_deaggregation_slices(cls, deagg_slices):
		"""
		Construct new instance of :class:`DeaggregationCurve` from a number
		of deaggregation slices.

		:param deagg_slices:
			list of instances of :class:`DeaggregationSlice`
		"""
		imls = np.array([ds.iml for ds in deagg_slices])
		iml_indexes = np.argsort(imls)
		imls = imls[iml_indexes]
		deagg_matrixes = [deagg_slices[iml_index].deagg_matrix[np.newaxis]
							for iml_index in iml_indexes]
		deagg_matrix = np.concatenate(deagg_matrixes)
		deagg_matrix = deagg_slices[0].deagg_matrix.__class__(deagg_matrix)
		# TODO: check that bin_edges etc. are identical
		bin_edges = deagg_slices[0].bin_edges
		site = deagg_slices[0].site
		intensity_unit = deagg_slices[0].intensity_unit
		imt = deagg_slices[0].imt
		period = deagg_slices[0].period
		return_periods = [ds.return_period for ds in deagg_slices]
		timespan = deagg_slices[0].timespan
		damping = deagg_slices[0].damping

		return cls(bin_edges, deagg_matrix, site,
					imls, intensity_unit, imt,
					period, return_periods, timespan, damping)

	def filter_cav(self, vs30, cav_min=0.16, gmpe_name=""):
		"""
		Apply CAV filtering to deaggregation curve, according to EPRI
		report by Abrahamson et al. (2006).

		:param vs30:
			Float, shear-wave velocity in the top 30 m (m/s)
		:param cav_min:
			Float, minimum CAV value in g.s (default: 0.16)
		:param gmpe_name:
			Str, name of GMPE (needed when imt is spectral)

		:return:
			instance of :class:`DeaggregationCurve`
		"""
		import scipy.stats
		from ..cav import (calc_ln_sa_given_pga, calc_ln_pga_given_sa,
						calc_cav_exceedance_prob)

		num_intensities = len(self.intensities)

		## Reduce to magnitude-distance pmf, and store in a new DeaggregationCurve object
		deagg_matrix = self.get_mag_dist_pmf()[:,:,:,np.newaxis,np.newaxis,
												np.newaxis,np.newaxis]
		bin_edges = (self.mag_bin_edges, self.dist_bin_edges, np.array([0]),
					np.array([0]), np.array([0]), np.array([0]))
		CAV_deagg_curve = DeaggregationCurve(bin_edges, deagg_matrix, self.site,
										self.intensities, self.intensity_unit,
										self.imt, self.period, self.return_periods,
										self.timespan, self.damping)

		intensities = self.get_intensity_bin_centers()

		if self.imt == "PGA":
			## Calculate CAV exceedance probabilities corresponding to PGA
			num_intensities = len(self.intensities)
			CAV_exceedance_probs = np.zeros((num_intensities, self.nmags), 'd')
			for k in range(num_intensities):
				zk = intensities[k]
				CAV_exceedance_probs[k] = calc_cav_exceedance_prob(zk,
											self.mag_bin_centers, vs30, cav_min)
			CAV_exceedance_probs = CAV_exceedance_probs[:,:,np.newaxis,np.newaxis,
												np.newaxis,np.newaxis,np.newaxis]

		elif self.imt == "SA":
			from ..gsim import gmpe as gmpe_module
			gmpe = getattr(gmpe_module, gmpe_name)()

			## Compute PGA given SA
			"""
			pga_given_sa = np.zeros((num_intensities, self.nmags, self.ndists), 'd')
			prob_pga_given_sa = np.ones((num_intensities, self.nmags, self.ndists), 'd')
			T = self.period
			sa = intensities
			ln_sa = np.log(sa)
			for k in range(num_intensities):
				for r in range(self.ndists):
					R = self.dist_bin_centers[r]
					M = self.mag_bin_centers
					ln_pga_given_sa, sigma_ln_pga_given_sa = calc_ln_pga_given_sa(sa[k], M, R, T, vs30, gmpe_name)
					# Note: ln_sa in line below cannot be correct
					epsilon_pga_given_sa = (ln_sa[k] - ln_pga_given_sa) / sigma_ln_pga_given_sa
					prob_pga_given_sa[k,:,r] = scipy.stats.norm.pdf(epsilon_pga_given_sa)
					#prob_pga_given_sa[k,:,r] = 1.0 - scipy.stats.norm.cdf(epsilon_pga_given_sa)
					pga_given_sa[k,:,r] = np.exp(ln_pga_given_sa)
			"""

			#pga = np.zeros((num_intensities, self.nmags, self.ndists), 'd')
			pga_given_sa = np.zeros((num_intensities, self.nmags, self.ndists), 'd')
			prob_sa_given_pga = np.ones((num_intensities, self.nmags, self.ndists), 'd')
			T = self.period
			sa = intensities
			ln_sa = np.log(sa)
			for k in range(num_intensities):
				for r in range(self.ndists):
					R = self.dist_bin_centers[r]
					M = self.mag_bin_centers
					## Compute epsilon
					#epsilon = gmpe.get_epsilon(sa[k], M, R, imt=self.imt, T=T, vs30=vs30)
					## Compute PGA corresponding to epsilon
					#pga[k,:,r] = gmpe(M, R, imt="PGA", T=0, epsilon=epsilon, vs30=vs30)
					ln_pga_given_sa, sigma_ln_pga_given_sa = calc_ln_pga_given_sa(
												sa[k], M, R, T, vs30, gmpe_name)
					pga_given_sa[k,:,r] = np.exp(ln_pga_given_sa)
					## Compute SA given PGA
					## This is ignored at the moment
					#ln_sa_given_pga, sigma_ln_sa_given_pga = calc_ln_sa_given_pga(pga[k,:,r], M, R, T, vs30, gmpe_name)
					## Compute epsilon of SA given PGA
					#epsilon_sa_given_pga = (ln_sa[k] - ln_sa_given_pga) / sigma_ln_sa_given_pga
					## Compute probability that SA is equal to (or greater than?) sa given PGA
					#prob_sa_given_pga[k,:,r] = scipy.stats.norm.pdf(epsilon_sa_given_pga)
					#prob_sa_given_pga[k,:,r] = 1.0 - scipy.stats.norm.cdf(epsilon_sa_given_pga)

			## Calculate CAV exceedance probabilities corresponding to PGA
			CAV_exceedance_probs = np.zeros((num_intensities, self.nmags, self.ndists), 'd')
			for k in range(num_intensities):
				for r in range(self.ndists):
					zk = pga_given_sa[k,:,r]
					#zk = pga[k,:,r]
					CAV_exceedance_probs[k,:,r] = calc_cav_exceedance_prob(zk,
											self.mag_bin_centers, vs30, cav_min)
			#CAV_exceedance_probs *= prob_sa_given_pga
			CAV_exceedance_probs = CAV_exceedance_probs[:,:,:,np.newaxis,np.newaxis,
														np.newaxis,np.newaxis]

			#prob_sa_given_pga = prob_sa_given_pga[:,:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]

		## Calculate filtered occurrence rates
		deagg_occurrence_rates = CAV_deagg_curve.get_occurrence_rates()
		CAV_deagg_occurrence_rates = deagg_occurrence_rates * CAV_exceedance_probs

		## Convert occurrence rates back to exceedance rates
		CAV_deagg_exceedance_rates = np.cumsum(CAV_deagg_occurrence_rates[::-1],
												axis=0)[::-1]
		#if self.imt == "SA":
		#	CAV_deagg_exceedance_rates *= prob_sa_given_pga
		CAV_deagg_curve.deagg_matrix = ExceedanceRateMatrix(CAV_deagg_exceedance_rates)

		return CAV_deagg_curve


class SpectralDeaggregationCurve(IntensityResult, DeaggBase):
	"""
	Class representing a full deaggregation result for a range of
	intensities and a range of periods

	:param bin_edges:
		6-tuple, containing:
			- magnitude bin edges
			- distance bin edges
			- longitude bin edges
			- latitude bin edges
			- epsilon bin edges
			- tectonic region types

	:param deagg_matrix:
		instance of :class:`ExceedanceRateMatrix` or :class:`ProbabilityMatrix`,
		8-D array containing deaggregation values, with dimensions
		corresponding to:
			- spectral periods
			- intensity levels
			- magnitude bins
			- distance bins
			- longitude bins
			- latitude bins
			- epsilon bins
			- tectonic-region-type bins

	:param site:
		instance of :class:`GenericSite`: site where hazard was computed
	:param intensities:
		2-D float array, intensity levels for each spectral period
		and each return period
	:param intensity_unit:
		str, unit in which intensity measure levels are expressed:
		PGA and SA: "g", "mg", "m/s2", "gal", "cm/s2"
		PGV: "cm/s"
		PGD: "cm"
	:param imt:
		str, intensity measure type
	:param periods:
		float array, spectral periods
	:param return_periods:
		float array, return periods corresponding to intensities
	:param timespan:
		Float, time span in Poisson formula.
	:param damping:
		float, damping corresponding to intensities
		(expressed as fraction of critical damping)
		(default: 0.05)
	"""
	def __init__(self, bin_edges, deagg_matrix, site,
				intensities, intensity_unit, imt,
				periods, return_periods, timespan, damping=0.05):
		## Make sure intensities are ordered from small to large
		if intensities[0,0] > intensities[0,-1]:
			DeaggBase.__init__(self, bin_edges, deagg_matrix[:,::-1], timespan)
			self.intensities = intensities[:,::-1]
		else:
			DeaggBase.__init__(self, bin_edges, deagg_matrix, timespan)
			self.intensities = intensities

		IntensityResult.__init__(self, intensities, intensity_unit, imt,
								damping=damping)

		self.site = parse_sites([site])[0]
		self.periods = np.array(periods)
		# TODO: return periods 2D array like intensities?
		self.return_periods = np.array(return_periods)

	def __iter__(self):
		"""
		Loop over spectral periods
		"""
		for period_index in range(len(self.periods)):
			yield self.get_curve(period_index=period_index)

	def __len__(self):
		return len(self.periods)

	def __getitem__(self, period_index):
		return self.get_curve(period_index=period_index)

	def __add__(self, other_sdc):
		assert isinstance(other_sdc, self.__class__)
		assert [(self.bin_edges[i] == other_sdc.bin_edges[i]).all()
				for i in range(5)]
		assert self.bin_edges[-1] == other_sdc.bin_edges[-1]
		assert self.site == other_sdc.site
		assert self.imt == other_sdc.imt
		assert (self.periods == other_sdc.periods).all()
		assert self.intensities.shape == other_sdc.intensities.shape
		assert ((self.intensities == other_sdc.intensities).all()
				or (self.return_periods == other_sdc.return_periods).all())
		assert self.timespan == other_sdc.timespan
		deagg_matrix = self.deagg_matrix + other_sdc.deagg_matrix

		return self.__class__(self.bin_edges, deagg_matrix, self.site,
							self.intensities, self.intensity_unit, self.imt,
							self.periods, self.return_periods, self.timespan,
							self.damping)

	def __mul__(self, number):
		assert isinstance(number, (int, float, Decimal))
		deagg_matrix = self.deagg_matrix * number

		return self.__class__(self.bin_edges, deagg_matrix, self.site,
							self.intensities, self.intensity_unit, self.imt,
							self.periods, self.return_periods, self.timespan,
							self.damping)

	def __rmul__(self, number):
		return self.__mul__(number)

	@classmethod
	def construct_empty_deagg_matrix(self, num_periods, num_intensities, bin_edges,
									matrix_class=ProbabilityMatrix, dtype='d'):
		"""
		Construct empty deaggregation matrix for a spectral deaggregation
		curve

		:param num_periods:
			int, number of spectral periods
		:param num_intensities:
			int, number of intensities or return periods
		:param bin_edges:
			tuple (mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, trts)
		:param matrix_class:
			matrix class, either :class:`ProbabilityMatrix`,
			:class:`ExceedanceRateMatrix` or
			:class:`FractionalContributionMatrix`
			(default:  :class:`ProbabilityMatrix`)
		:param dtype:
			str, precision of deaggregation matrix
			(default: 'd')

		:return:
			instance of :class:`DeaggMatrix`
		"""
		mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, trts = bin_edges
		nmags = len(mag_bins) - 1
		ndists = len(dist_bins) - 1
		nlons = len(lon_bins) - 1
		nlats = len(lat_bins) - 1
		neps = len(eps_bins) - 1
		ntrts = len(trts)
		shape = (num_periods, num_intensities, nmags, ndists, nlons, nlats, neps,
				ntrts)
		deagg_matrix = matrix_class(np.zeros(shape, dtype))
		return deagg_matrix

	def get_curve(self, period=None, period_index=None):
		"""
		Get deaggregation curve for a particular spectral period.

		:param period:
			Float, spectral period (default: None)
		:param period_index:
			Int, period index (default: None)

		:return:
			instance of :class:`DeaggregationCurve`
		"""
		if period is not None:
			period_index = np.argmin(np.abs(self.periods - period))
		else:
			period = self.periods[period_index]
		matrix = self.deagg_matrix[period_index]
		intensities = self.intensities[period_index]

		return DeaggregationCurve(self.bin_edges, matrix, self.site,
								intensities, self.intensity_unit, self.imt,
								period, self.return_periods, self.timespan,
								self.damping)

	@classmethod
	def from_deaggregation_curves(cls, deagg_curves):
		"""
		Construct new instance of :class:`SpectralDeaggregationCurve`
		from a number of deaggregation curves.

		:param deagg_curves:
			list of instances of :class:`DeaggregationCurve`
		"""
		periods = np.array([dc.period for dc in deagg_curves])
		period_indexes = np.argsort(periods)
		periods = periods[period_indexes]
		deagg_matrixes = [deagg_curves[period_index].deagg_matrix[np.newaxis]
							for period_index in period_indexes]
		deagg_matrix = np.concatenate(deagg_matrixes)
		deagg_matrix = deagg_curves[0].deagg_matrix.__class__(deagg_matrix)
		intensity_arrays = [deagg_curves[period_index].intensities[np.newaxis,:]
							for period_index in period_indexes]
		intensities = np.concatenate(intensity_arrays, axis=0)
		# TODO: check that bin_edges etc. are identical
		bin_edges = deagg_curves[0].bin_edges
		site = deagg_curves[0].site
		intensity_unit = deagg_curves[0].intensity_unit
		imt = deagg_curves[0].imt
		return_periods = deagg_curves[0].return_periods
		timespan = deagg_curves[0].timespan
		damping = deagg_curves[0].damping

		return cls(bin_edges, deagg_matrix, site,
					intensities, intensity_unit, imt,
					periods, return_periods, timespan, damping)

	def filter_cav(self, vs30, cav_min=0.16, gmpe_name=""):
		"""
		Apply CAV filtering to each spectral deaggregation curve,
		and reconstruct into a spectral deaggregation curve

		:param vs30:
			Float, shear-wave velocity in the top 30 m (m/s)
		:param cav_min:
			Float, minimum CAV value in g.s (default: 0.16)
		:param gmpe_name:
			Str, name of GMPE (needed when imt is spectral)

		:return:
			instance of :class:`SpectralDeaggregationCurve`
		"""
		CAV_curves = []
		for curve in self:
			CAV_deagg_curve = curve.filter_cav(vs30, cav_min=cav_min,
												gmpe_name=gmpe_name)
			CAV_curves.append(CAV_deagg_curve)
		return SpectralDeaggregationCurve.from_deaggregation_curves(CAV_curves)

	def get_conditional_mean_spectrum(self):
		"""
		Compute conditional mean spectrum as outlined in e.g., Baker (2011)
		"""
		pass

	def write_nrml(self, nrml_filespec, smlt_path=None, gmpelt_path=None,
					min_poe=1E-8, encoding='latin-1', pretty_print=True):
		"""
		:param nrml_filespec:
			str, full path to output file
		:param smlt_path:
			str, source-model logic-tree path
			(default: None)
		:param gmpelt_path:
			str, ground-motion logic-tree path
			(default: None)
		:param min_poe:
			float, lower probability value below which to clip output
			(default: 1E-8)
		:param encoding:
			str, unicode encoding
			(default: 'latin-1')
		pretty_print:
			bool, indicating whether or not to indent each element
			(default: True)
		"""
		# TODO: use nrml.ns where possible!
		import time
		from lxml import etree
		from ..nrml import ns
		nrml_file = open(nrml_filespec, "w")
		root = etree.Element("nrml", nsmap=ns.NSMAP)
		sdc_elem = etree.SubElement(root, "spectralDeaggregationCurve")
		if sourceModelTreePath:
			sdc_elem.set("sourceModelTreePath", sourceModelTreePath)
		if gsimTreePath:
			sdc_elem.set("gsimTreePath", gsimTreePath)
		lon, lat = self.site[0], self.site[1]
		sdc_elem.set("lon", str(lon))
		sdc_elem.set("lat", str(lat))
		sdc_elem.set("investigationTime", str(self.timespan))
		mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, trts = self.bin_edges
		sdc_elem.set("magBinEdges", ", ".join(map(str, mag_bins)))
		sdc_elem.set("distBinEdges", ", ".join(map(str, dist_bins)))
		sdc_elem.set("lonBinEdges", ", ".join(map(str, lon_bins)))
		sdc_elem.set("latBinEdges", ", ".join(map(str, lat_bins)))
		sdc_elem.set("epsBinEdges", ", ".join(map(str, eps_bins)))
		sdc_elem.set("tectonicRegionTypes", ", ".join(trts))
		dims = ",".join(map(str, self.matrix.shape[2:]))
		sdc_elem.set("dims", dims)
		for dc in self:
			#print(dc, time.time())
			dc_elem = etree.SubElement(sdc_elem, "deaggregationCurve")
			dc_elem.set("IMT", str(dc.imt))
			dc_elem.set("saPeriod", str(dc.period))
			for ds_idx, ds in enumerate(dc):
				#print(ds, time.time())
				ds_elem = etree.SubElement(dc_elem, "deaggregationSlice")
				ds_elem.set("iml", str(ds.iml))
				## Write intended poe, not actual poe
				#poe = ds.deagg_matrix.get_total_probability(timespan=self.timespan)
				poe = poisson_conv(tau=self.return_periods[ds_idx], t=self.timespan)
				ds_elem.set("poE", str(poe))
				matrix = ds.deagg_matrix.to_probability_matrix(timespan=self.timespan)
				for i, nonzero in np.ndenumerate(matrix > min_poe):
					if nonzero:
						index = ",".join(map(str, i))
						value = str(matrix[i])
						prob = etree.SubElement(ds_elem, "prob")
						prob.set("index", index)
						prob.set("value", value)
		nrml_file.write(etree.tostring(root, pretty_print=pretty_print,
										xml_declaration=True, encoding=encoding))
		nrml_file.close()

	def slice_return_periods(self, return_periods, shc, interpolate_matrix=False):
		"""
		Reduce the spectral deaggregation curve to one where slices
		correspond to return periods. First, intensities are interpolated
		from the given spectral hazard curve, and a new spectral
		deaggregation curve is constructed from slices that are nearest
		to these interpolated intensities.

		:param return_periods:
			list of floats, return periods
		:param shc:
			instance of :class:`rshalib.result.SpectralHazardCurve`
		:param interpolate_matrix:
			bool, whether or not the deaggregation matrix should be
			interpolated at the intensities interpolated from the
			hazard curve. If False, the nearest slices will be selected
			(default: False)

		:return:
			instance of :class:`SpectralDeaggregationCurve`
		"""
		curves = []
		for T in self.periods:
			dc = self.get_curve(period=T)
			hc = shc.getHazardCurve(period_spec=float(T))
			rp_dc = dc.slice_return_periods(return_periods, hc,
											interpolate_matrix=interpolate_matrix)
			curves.append(rp_dc)
		rp_sdc = SpectralDeaggregationCurve.from_deaggregation_curves(curves)
		rp_sdc.return_periods = np.array(return_periods)
		return rp_sdc

	def get_mean_low_freq_curve(self):
		"""
		Determine mean low-frequency deaggregation curve.

		:return:
			instance of :class:`DeaggregationCurve`
		"""
		dc_list = []
		periods = np.array([0.4, 1])
		for k, period in enumerate(periods):
			if not period in self.periods:
				period_index = np.argmin(np.abs(self.periods - period))
				msg = "Warning: Period %s s not found, using %s s instead"
				msg %= (period, self.periods[period_index])
				print(msg)
				period = self.periods[period_index]
				periods[k] = period
			dc_list.append(self.get_curve(period=period))
		mean_low_freq_dc = get_mean_deaggregation_curve(dc_list)
		mean_low_freq_dc.period = periods.mean()
		return mean_low_freq_dc

	def get_mean_high_freq_curve(self):
		"""
		Determine mean high-frequency deaggregation curve.

		:return:
			instance of :class:`DeaggregationCurve`
		"""
		dc_list = []
		periods = np.array([0.1, 0.2])
		for k, period in enumerate(periods):
			if not period in self.periods:
				period_index = np.argmin(np.abs(self.periods - period))
				msg = "Warning: Period %s s not found, using %s s instead"
				msg %= (period, self.periods[period_index])
				print(msg)
				period = self.periods[period_index]
				periods[k] = period
			dc_list.append(self.get_curve(period=period))
		mean_high_freq_dc = get_mean_deaggregation_curve(dc_list)
		mean_high_freq_dc.period = periods.mean()
		return mean_high_freq_dc

	def analyze_controlling_earthquakes(self, remote_distance=100,
										truncate_hf_distance=False, filespec=None):
		"""
		Print magnitude and distance of controlling earthquakes.

		:param remote_distance:
			float, threshold distance (in km) considered for remote
			low-frequency earthquakes (default: 100)
		:param truncate_hf_distance:
			bool, whether or not to truncate distance for mean HF eq
			to :param:`remote_distance`
			(default: False)
		:param filespec:
			str, full path to output file
			(default: None)

		:return:
			(mean_high_freq_dc, mean_low_freq_dc) tuple with
			instances of :class:`DeaggregationCurve`
		"""
		mean_high_freq_dc = self.get_mean_high_freq_curve()
		mean_low_freq_dc = self.get_mean_low_freq_curve()

		msg_lines = []
		for i, return_period in enumerate(self.return_periods):
			line = "Return period: %s yr" % return_period
			if not filespec:
				print(line)
			msg_lines.append(line)

			mean_high_freq_ds = mean_high_freq_dc.get_slice(iml_index=i)
			mean_low_freq_ds = mean_low_freq_dc.get_slice(iml_index=i)

			contrib = mean_low_freq_ds.get_contribution_above_distance(remote_distance)
			line = "  Low-freq contribution for d > %s km: %.2f %%"
			line %= (remote_distance, contrib * 100)
			if not filespec:
				print(line)
			msg_lines.append(line)

			if truncate_hf_distance:
				mean_high_freq_ds = mean_high_freq_ds.get_fractional_contribution_slice_below(remote_distance, 1)
			mean_mag, mean_dist = mean_high_freq_ds.get_mean_eq_scenario()
			line = "  High-frequency controlling earthquake: M=%.1f, d=%.0f km"
			line %= (mean_mag, mean_dist)
			if not filespec:
				print(line)
			msg_lines.append(line)

			mean_remote_low_freq_ds = mean_low_freq_ds.get_fractional_contribution_slice_above(
																			remote_distance, 1)
			mean_mag, mean_dist = mean_remote_low_freq_ds.get_mean_eq_scenario()
			line = "  Remote low-frequency controlling earthquake: "
			line += "M=%.1f, d=%.0f km" % (mean_mag, mean_dist)
			if not filespec:
				print(line)
			msg_lines.append(line)

		if filespec:
			with open(filespec, "w") as f:
				for line in msg_lines:
					f.write(line)
					f.write("\n")

		return (mean_high_freq_dc, mean_low_freq_dc)

	def get_spectral_hazard_curve(self):
		"""
		Get spectral hazard curve corresponding to total exceedance rate
		or probability for each intensity level.

		:return:
			instance of :class:`SpectralHazardCurve`
		"""
		from .hc_base import ExceedanceRateArray, ProbabilityArray
		from .hazard_curve import SpectralHazardCurve

		hazard_values = self.deagg_matrix.fold_axes([7,6,5,4,3,2])
		if isinstance(self.deagg_matrix, ExceedanceRateMatrix):
			hazard_values = ExceedanceRateArray(hazard_values)
		elif isinstance(self.deagg_matrix, ProbabilityMatrix):
			hazard_values = ProbabilityArray(hazard_values)
		else:
			raise Exception("Not supported for FractionalContributionMatrix!")
		model_name = ""
		filespec = ""
		return SpectralHazardCurve(hazard_values, self.site, self.periods,
									self.intensities, self.intensity_unit, self.imt,
									model_name=model_name, filespec=filespec,
									timespan=self.timespan, damping=self.damping)


def get_mean_deaggregation_slice(deagg_slices):
	"""
	Compute mean deaggregation slice

	:param deagg_slices:
		list with instances of :class:`DeaggregationSlice`

	:return:
		instance of :class:`DeaggregationSlice`
		Note that matrixes will be converted to instances of
		:class:`FractionalContributionMatrix`
	"""
	ds0 = deagg_slices[0]
	matrix = ds0.get_fractional_contribution_matrix()
	for ds in deagg_slices[1:]:
		assert ds.bin_edges == ds0.bin_edges
		matrix += ds.get_fractional_contribution_matrix()
	matrix /= np.sum(matrix)

	return DeaggregationSlice(ds0.bin_edges, matrix, ds0.site,
							ds0.iml, ds0.intensity_unit, ds0.imt,
							ds0.period, ds0.return_period, ds0.timespan,
							ds0.damping)

def get_mean_deaggregation_curve(deagg_curves):
	"""
	Compute mean deaggregation slice

	:param deagg_slices:
		list with instances of :class:`DeaggregationSlice`

	:return:
		instance of :class:`DeaggregationSlice`
		Note that matrixes will be converted to instances of
		:class:`FractionalContributionMatrix`
	"""
	dc0 = deagg_curves[0]
	matrix = dc0.get_fractional_contribution_matrix()
	for dc in deagg_curves[1:]:
		assert dc.bin_edges == dc0.bin_edges
		matrix += dc.get_fractional_contribution_matrix()
	matrix /= np.sum(matrix)

	return DeaggregationCurve(dc0.bin_edges, matrix, dc0.site,
							dc0.intensities, dc0.intensity_unit, dc0.imt,
							dc0.period, dc0.return_periods, dc0.timespan,
							dc0.damping)
