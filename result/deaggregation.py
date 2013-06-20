import numpy as np

import openquake.hazardlib.calc.disagg as disagg

from plot import plot_deaggregation
from hazard_curve import Poisson, HazardCurve



class DeaggMatrix(np.ndarray):
	def __new__(cls, data):
		obj = np.asarray(data).view(cls)
		return obj

	def __array_finalize__(self, obj):
		if obj is None:
			return

	@property
	def matrix(self):
		return np.asarray(self)


class ExceedanceRateMatrix(DeaggMatrix):
	def get_total_exceedance_rate(self, timespan=None):
		return np.sum(self.matrix)

	def get_total_probability(self, timespan):
		return Poisson(life_time=timespan, return_period=1./self.get_total_exceedance_rate())

	def to_probability_matrix(self, timespan):
		return ProbabilityMatrix(Poisson(life_time=timespan, return_period=1./self.matrix))

	def to_exceedance_rate_matrix(self, timespan):
		return ExceedanceRateMatrix(self.matrix)

	def to_normalized_matrix(self, timespan=None):
		return self.matrix / self.get_total_exceedance_rate()


class ProbabilityMatrix(DeaggMatrix):
	#def __new__(cls, data, timespan):
	#	obj = DeaggMatrix.__new__(cls, data)
	#	obj.timespan = timespan
	#	return obj

	def get_total_exceedance_rate(self, timespan):
		return 1. / Poisson(life_time=timespan, prob=self.get_total_probability())

	def get_total_probability(self, timespan=None):
		return 1 - np.prod(1 - self.matrix)

	def to_exceedance_rate_matrix(self, timespan):
		return ExceedanceRateMatrix(1. / Poisson(life_time=timespan, prob=self.matrix))

	def to_probability_matrix(self, timespan=None):
		return ProbabilityMatrix(self.matrix)

	def to_normalized_matrix(self, timespan=None):
		ln_non_exceedance_probs = np.log(1. - self.matrix)
		return ln_non_exceedance_probs / np.sum(ln_non_exceedance_probs)


class DeaggBase(object):
	def __init__(self, bin_edges, deagg_matrix, timespan):
		self.bin_edges = bin_edges
		self.deagg_matrix = deagg_matrix
		self.timespan = timespan

		# TODO: check shape
		# TODO: check that deagg_matrix is ProbabilityMatrix or ExceedanceRateMatrix

	@property
	def matrix(self):
		return np.asarray(self.deagg_matrix)

	@property
	def nmags(self):
		return self.matrix.shape[-6]

	@property
	def ndists(self):
		return self.matrix.shape[-5]

	@property
	def nlons(self):
		return self.matrix.shape[-4]

	@property
	def nlats(self):
		return self.matrix.shape[-3]

	@property
	def neps(self):
		return self.matrix.shape[-2]

	@property
	def ntrts(self):
		return self.matrix.shape[-1]

	@property
	def mag_bin_edges(self):
		return self.bin_edges[0]

	@property
	def mag_bin_widths(self):
		mag_bin_edges = self.mag_bin_edges
		return mag_bin_edges[1:] - mag_bin_edges[:-1]

	@property
	def mag_bin_centers(self):
		mag_bin_widths = self.mag_bin_widths
		return self.mag_bin_edges[:-1] + mag_bin_widths / 2

	@property
	def dist_bin_edges(self):
		return self.bin_edges[1]

	@property
	def lon_bin_edges(self):
		return self.bin_edges[2]

	@property
	def lat_bin_edges(self):
		return self.bin_edges[3]

	@property
	def eps_bin_edges(self):
		return self.bin_edges[4]

	@property
	def trt_bins(self):
		return self.bin_edges[5]

	def to_percent_contribution(self):
		"""
		Normalize probability matrix.
		"""
		return self.deagg_matrix.to_normalized_matrix(self.timespan)

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


class DeaggregationSlice(DeaggBase):
	"""
	Class representing a full deaggregation result as computed by nhlib.
	6-D array

	Deaggregation values represent conditional probability distribution of:
		- rupture magnitude,
		- joyner-boore distance from rupture surface to site,
		- longitude and latitude of surface projection of rupture closest point
		  to site,
		- epsilon: number of standard deviations by which an intensity measure
		  level deviates from the median value predicted by a gsim, given
		  the rupture parameters.
		- rupture tectonic region type

	given the event that an intensity measure type ``imt`` exceeds an intensity
	measure level ``iml`` at a geographical location ``site``.

	"""
	def __init__(self, bin_edges, deagg_matrix, site, imt, iml, period, timespan):
		DeaggBase.__init__(self, bin_edges, deagg_matrix, timespan)
		self.site = site
		self.imt = imt
		self.iml = iml
		self.period = period

	def get_mag_pmf(self):
		"""
		Fold full deaggregation matrix to magnitude PMF.

		:returns:
			1D array, a histogram representing magnitude PMF.
		"""
		return ProbabilityMatrix(disagg.mag_pmf(self.deagg_matrix.to_probability_matrix()))

	def get_dist_pmf(self):
		"""
		Fold full deaggregation matrix to distance PMF.

		:returns:
			1D array, a histogram representing distance PMF.
		"""
		return ProbabilityMatrix(disagg.dist_pmf(self.deagg_matrix.to_probability_matrix()))

	def get_eps_pmf(self):
		"""
		Fold full deaggregation matrix to epsilon PMF.

		:returns:
			1D array, a histogram representing epsilon PMF.
		"""
		eps_pmf = np.zeros(self.neps)
		for m in xrange(self.neps):
			eps_pmf[m] = 1 - np.prod(1 - self.deagg_matrix.to_probability_matrix()[i][j][k][l][m][n]
							  for i in xrange(self.nmags)
							  for j in xrange(self.ndists)
							  for k in xrange(self.nlons)
							  for l in xrange(self.nlats)
							  for n in xrange(self.ntrts))
		return ProbabilityMatrix(eps_pmf)

	def get_trt_pmf(self):
		"""
		Fold full deaggregation matrix to tectonic region type PMF.

		:returns:
			1D array, a histogram representing tectonic region type PMF.
		"""
		return ProbabilityMatrix(disagg.trt_pmf(self.deagg_matrix.to_probability_matrix()))

	def get_mag_dist_pmf(self):
		"""
		Fold full deaggregation matrix to magnitude / distance PMF.

		:returns:
			2D array, first dimension represents magnitude histogram bins,
			second one -- distance histogram bins.
		"""
		if isinstance(self.deagg_matrix, ProbabilityMatrix):
			return ProbabilityMatrix(disagg.mag_dist_pmf(self.deagg_matrix.to_probability_matrix()))


	def get_mag_dist_eps_pmf(self):
		"""
		Fold full deaggregation matrix to magnitude / distance /epsilon PMF.

		:returns:
			3D array, first dimension represents magnitude histogram bins,
			second one -- distance histogram bins, third one -- epsilon
			histogram bins.
		"""
		return ProbabilityMatrix(disagg.mag_dist_eps_pmf(self.deagg_matrix.to_probability_matrix()))

	def get_lon_lat_pmf(self):
		"""
		Fold full deaggregation matrix to longitude / latitude PMF.

		:returns:
			2D array, first dimension represents longitude histogram bins,
			second one -- latitude histogram bins.
		"""
		return ProbabilityMatrix(disagg.lon_lat_pmf(self.deagg_matrix.to_probability_matrix()))

	def get_mag_lon_lat_pmf(self):
		"""
		Fold full deaggregation matrix to magnitude / longitude / latitude PMF.

		:returns:
			3D array, first dimension represents magnitude histogram bins,
			second one -- longitude histogram bins, third one -- latitude
			histogram bins.
		"""
		return ProbabilityMatrix(disagg.mag_lon_lat_pmf(self.deagg_matrix.to_probability_matrix()))

	def get_lon_lat_trt_pmf(self):
		"""
		Fold full deaggregation matrix to longitude / latitude / trt PMF.

		:returns:
			3D array, first dimension represents longitude histogram bins,
			second one -- latitude histogram bins, third one -- trt histogram bins.
		"""
		return ProbabilityMatrix(disagg.lon_lat_trt_pmf(self.deagg_matrix.to_probability_matrix()))

	def to_exceedance_rate(self):
		"""
		Convert deaggregation matrix to exceedance rate

		:return:
			ndarray with same shape as deaggregation matrix
		"""
		return ExceedanceRateMatrix(self.deagg_matrix.to_exceedance_matrix(self.timespan))

	def plot_mag_dist_pmf(self, return_period=475):
		"""
		Plot magnitude / distance PMF.
		"""
		mag_dist_pmf = self.get_mag_dist_pmf().to_normalized_matrix(self.timespan)
		mag_dist_pmf = mag_dist_pmf.transpose()
		eps_pmf = self.get_eps_pmf().to_normalized_matrix(self.timespan)
		eps_bin_edges = self.eps_bin_edges[1:]
		try:
			imt_period = self.imt.period
		except AttributeError:
			imt_period = None
		plot_deaggregation(mag_dist_pmf, self.mag_bin_edges, self.dist_bin_edges, return_period, eps_values=eps_pmf, eps_bin_edges=eps_bin_edges, mr_style="2D", site_name=self.site.name, struc_period=imt_period, title_comment="", fig_filespec=None)

	def get_mode_eq(self):
		"""

		"""
		mag_dist_pmf = self.get_mag_dist_pmf()
		mag_index, dist_index = np.unravel_index(mag_dist_pmf.argmax(), mag_dist_pmf.shape)
		return (self.mag_bin_centers[mag_index], self.dist_bin_centers[dist_index])


class DeaggregationCurve(DeaggBase):
	"""
	Class representing a full deaggregation result for a range of intensities
	7-D array
	"""
	def __init__(self, bin_edges, deagg_matrix, site, imt, intensities, period, timespan):
		self.site = site
		self.imt = imt
		self.period = period

		## Make sure intensities are ordered from small to large
		if intensities[0] > intensities[-1]:
			DeaggBase.__init__(self, bin_edges, deagg_matrix[::-1], timespan)
			self.intensities = intensities[::-1]
		else:
			DeaggBase.__init__(self, bin_edges, deagg_matrix, timespan)
			self.intensities = intensities

	def __len__(self):
		return len(self.intensities)

	def __iter__(self):
		for iml_index in range(len(self.intensities)):
			yield self.get_slice(iml_index=iml_index)

	def get_slice(self, iml=None, iml_index=None):
		if iml is not None:
			iml_index = np.argmin(np.abs(self.intensities - iml))
		matrix = self.matrix[iml_index]
		iml = self.intensities[iml_index]
		return DeaggregationSlice(self.bin_edges, matrix, self.site, self.imt, iml, self.period, self.timespan)

	def get_hazard_curve(self):
		exceedance_rates = []
		for slice in self:
			exceedance_rates.append(slice.get_total_exceedance_rate())
		model_name = ""
		filespec = ""
		return HazardCurve(model_name, filespec, self.site, self.period, self.imt, self.intensities, "g", self.timespan, poes=None, exceedance_rates=exceedance_rates, variances=None, site_name="")

	def get_occurrence_rates(self):
		"""
		Calculate rate of occurrence for each intensity interval
		"""
		exceedance_rates = self.matrix
		occurrence_rates = exceedance_rates[:-1] - exceedance_rates[1:]
		occurrence_rates = np.append(deagg_occurrences, deagg_exceedances[-1:], axis=0)
		return occurrence_rates


