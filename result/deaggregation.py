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

	@property
	def num_axes(self):
		return len(self.shape)


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

	def fold_axis(self, axis):
		if axis < 0:
			axis = self.num_axes + axis
		return self.sum(axis=axis)

	def fold_axes(self, axes):
		matrix = self
		axes = [{True: self.num_axes + axis, False: axis}[axis < 0] for axis in axes]
		for axis in sorted(axes)[::-1]:
			matrix = matrix.fold_axis(axis)
		return matrix


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

	def fold_axis(self, axis):
		if axis < 0:
			axis = self.num_axes + axis
		return 1 - np.prod(1 - self, axis=axis)

	def fold_axes(self, axes):
		matrix = self
		axes = [{True: self.num_axes + axis, False: axis}[axis < 0] for axis in axes]
		for axis in sorted(axes)[::-1]:
			matrix = matrix.fold_axis(axis)
		return matrix


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
		return self.mag_bin_edges[:-1] + self.mag_bin_widths / 2

	@property
	def dist_bin_edges(self):
		return self.bin_edges[1]

	@property
	def dist_bin_widths(self):
		dist_bin_edges = self.dist_bin_edges
		return dist_bin_edges[1:] - dist_bin_edges[:-1]

	@property
	def dist_bin_centers(self):
		return self.dist_bin_edges[:-1] + self.dist_bin_widths / 2

	@property
	def lon_bin_edges(self):
		return self.bin_edges[2]

	@property
	def lon_bin_widths(self):
		lon_bin_edges = self.lon_bin_edges
		return lon_bin_edges[1:] - lon_bin_edges[:-1]

	@property
	def lon_bin_centers(self):
		return self.lon_bin_edges[:-1] + self.lon_bin_widths / 2

	@property
	def lat_bin_edges(self):
		return self.bin_edges[3]

	@property
	def lat_bin_widths(self):
		lat_bin_edges = self.lat_bin_edges
		return lat_bin_edges[1:] - lat_bin_edges[:-1]

	@property
	def lat_bin_centers(self):
		return self.lat_bin_edges[:-1] + self.lat_bin_widths / 2

	@property
	def eps_bin_edges(self):
		return self.bin_edges[4]

	@property
	def eps_bin_widhts(self):
		eps_bin_edges = self.eps_bin_edges
		return eps_bin_edges[1:] - eps_bin_edges[:-1]

	@property
	def eps_bin_centers(self):
		return self.eps_bin_edges[:-1] + self.eps_bin_widths / 2

	@property
	def trt_bins(self):
		return self.bin_edges[5]

	def get_mag_pmf(self):
		"""
		Fold full deaggregation matrix to magnitude PMF.

		:returns:
			1D array, a histogram representing magnitude PMF.
		"""
		return self.deagg_matrix.fold_axes([-5,-4,-3,-2,-1])

	def get_dist_pmf(self):
		"""
		Fold full deaggregation matrix to distance PMF.

		:returns:
			1D array, a histogram representing distance PMF.
		"""
		return self.deagg_matrix.fold_axes([-6,-4,-3,-2,-1])

	def get_eps_pmf(self):
		"""
		Fold full deaggregation matrix to epsilon PMF.

		:returns:
			1D array, a histogram representing epsilon PMF.
		"""
		return self.deagg_matrix.fold_axes([-6,-5,-3,-2,-1])

	def get_trt_pmf(self):
		"""
		Fold full deaggregation matrix to tectonic region type PMF.

		:returns:
			1D array, a histogram representing tectonic region type PMF.
		"""
		return self.deagg_matrix.fold_axes([-6,-5,-4,-3,-2])

	def get_mag_dist_pmf(self):
		"""
		Fold full deaggregation matrix to magnitude / distance PMF.

		:returns:
			2D array, first dimension represents magnitude histogram bins,
			second one -- distance histogram bins.
		"""
		return self.deagg_matrix.fold_axes([-4,-3,-2,-1])

	def get_mag_dist_eps_pmf(self):
		"""
		Fold full deaggregation matrix to magnitude / distance /epsilon PMF.

		:returns:
			3D array, first dimension represents magnitude histogram bins,
			second one -- distance histogram bins, third one -- epsilon
			histogram bins.
		"""
		return self.deagg_matrix.fold_axes([-3,-2,-1])

	def get_lon_lat_pmf(self):
		"""
		Fold full deaggregation matrix to longitude / latitude PMF.

		:returns:
			2D array, first dimension represents longitude histogram bins,
			second one -- latitude histogram bins.
		"""
		return self.deagg_matrix.fold_axes([-6,-5,-4,-1])

	def get_mag_lon_lat_pmf(self):
		"""
		Fold full deaggregation matrix to magnitude / longitude / latitude PMF.

		:returns:
			3D array, first dimension represents magnitude histogram bins,
			second one -- longitude histogram bins, third one -- latitude
			histogram bins.
		"""
		return self.deagg_matrix.fold_axes([-5,-4,-1])

	def get_lon_lat_trt_pmf(self):
		"""
		Fold full deaggregation matrix to longitude / latitude / trt PMF.

		:returns:
			3D array, first dimension represents longitude histogram bins,
			second one -- latitude histogram bins, third one -- trt histogram bins.
		"""
		return self.deagg_matrix.fold_axes([-6,-5,-4])

	def to_percent_contribution(self):
		"""
		Normalize probability matrix.
		"""
		return self.deagg_matrix.to_normalized_matrix(self.timespan)


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

	def to_exceedance_rate(self):
		"""
		Convert deaggregation slice to exceedance rate

		:return:
			instance of :class:`DeaggregationSlice`
		"""
		deagg_matrix = self.deagg_matrix.to_exceedance_rate_matrix(self.timespan)
		return DeaggregationSlice(self.bin_edges, deagg_matrix, self.site, self.imt, self.iml, self.period, self.timespan)

	def to_probability(self):
		"""
		Convert deaggregation slice to probability of exceecance
		"""
		deagg_matrix = self.deagg_matrix.to_probability_matrix(self.timespan)
		return DeaggregationSlice(self.bin_edges, deagg_matrix, self.site, self.imt, self.iml, self.period, self.timespan)

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

	def plot_mag_dist_pmf(self, return_period=475):
		"""
		Plot magnitude / distance PMF.
		"""
		mag_dist_pmf = self.get_mag_dist_pmf().to_normalized_matrix(self.timespan)
		if self.neps > 1:
			eps_pmf = self.get_eps_pmf().to_normalized_matrix(self.timespan)
		else:
			eps_pmf = None
		eps_bin_edges = self.eps_bin_edges[1:]
		try:
			imt_period = self.imt.period
		except AttributeError:
			imt_period = None
		plot_deaggregation(mag_dist_pmf, self.mag_bin_edges, self.dist_bin_edges, return_period, eps_values=eps_pmf, eps_bin_edges=eps_bin_edges, mr_style="2D", site_name=self.site.name, struc_period=imt_period, title_comment="", fig_filespec=None)

	def get_modal_eq_scenario(self):
		"""
		Determine modal earthquake scenario (having largest contribution)

		:return:
			(mag, dist) tuple
		"""
		mag_dist_pmf = self.get_mag_dist_pmf()
		mag_index, dist_index = np.unravel_index(mag_dist_pmf.argmax(), mag_dist_pmf.shape)
		return (self.mag_bin_centers[mag_index], self.dist_bin_centers[dist_index])

	def rebin_magnitudes(self):
		pass

	def rebin_distances(self):
		pass


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
		else:
			iml = self.intensities[iml_index]
		matrix = self.deagg_matrix[iml_index]
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
		occurrence_rates = np.append(occurrence_rates, exceedance_rates[-1:], axis=0)
		return occurrence_rates

	def filter_cav(self, vs30, CAVmin=0.16, gmpe_name=""):
		from hazard.psha.CAVfiltering import calc_ln_PGA_given_SA, calc_CAV_exceedance_prob

		num_intensities = len(self.intensities)

		## Reduce to magnitude-distance pmf, and store in a new DeaggregationCurve object
		deagg_matrix = self.get_mag_dist_pmf()[:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
		bin_edges = (self.mag_bin_edges, self.dist_bin_edges, np.array([0]), np.array([0]), np.array([0]), np.array([0]))
		CAV_deagg_curve = DeaggregationCurve(bin_edges, deagg_matrix, self.site, self.imt, self.intensities, self.period, self.timespan)

		if self.imt == "PGA":
			## Calculate CAV exceedance probabilities corresponding to PGA
			num_intensities = len(self.intensities)
			CAV_exceedance_probs = np.zeros((num_intensities, self.nmags), 'd')
			for k in range(num_intensities):
				zk = self.intensities[k]
				CAV_exceedance_probs[k] = calc_CAV_exceedance_prob(zk, self.mag_bin_centers, vs30, CAVmin)
			print CAV_exceedance_probs
			CAV_exceedance_probs = CAV_exceedance_probs[:,np.newaxis,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]

		elif self.imt == "SA":
			## Compute PGA given SA
			pga_given_sa = np.zeros((num_intensities, self.ndists, self.nmags), 'd')
			T = self.period
			for k in range(num_intensities):
				sa = intensities[k]
				for r in range(self.ndists):
					R = self.dist_bin_centers[r]
					ln_pga_values = calc_ln_PGA_given_SA(sa, self.mag_bin_centers, R, T, vs30, gmpe_name)[0]
					pga_given_sa[k,r] = np.exp(ln_pga_values)

			## Calculate CAV exceedance probabilities corresponding to PGA
			CAV_exceedance_probs = np.zeros((num_intensities, self.ndists, self.nmags), 'd')
			for k in range(num_intensities):
				for r in range(self.ndists):
					zk = pga_given_sa[T,k,r]
					CAV_exceedance_probs[k,r] = calc_CAV_exceedance_prob(zk, self.mag_bin_centers, vs30, CAVmin)
			CAV_exceedance_probs = CAV_exceedance_probs[:,:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]

		## Calculate filtered occurrence rates
		deagg_occurrence_rates = CAV_deagg_curve.get_occurrence_rates()
		CAV_deagg_occurrence_rates = deagg_occurrence_rates * CAV_exceedance_probs

		## Convert occurrence rates back to exceedance rates
		CAV_deagg_exceedance_rates = np.add.accumulate(CAV_deagg_occurrence_rates[::-1,:,:], axis=0)[::-1,:,:]
		CAV_deagg_curve.deagg_matrix = self.deagg_matrix.__class__(CAV_deagg_exceedance_rates)

		return CAV_deagg_curve


class SpectralDeaggregationCurve(DeaggBase):
	"""
	Class representing a full deaggregation result for a range of intensities
	and a range of periods
	8-D array
	"""
	def __init__(self, bin_edges, deagg_matrix, site, imt, intensities, periods, timespan):
		self.site = site
		self.imt = imt
		self.periods = periods

		## Make sure intensities are ordered from small to large
		if intensities[0,0] > intensities[0,-1]:
			DeaggBase.__init__(self, bin_edges, deagg_matrix[:,::-1], timespan)
			self.intensities = intensities[:,::-1]
		else:
			DeaggBase.__init__(self, bin_edges, deagg_matrix, timespan)
			self.intensities = intensities

	def __iter__(self):
		for period_index in range(len(self.periods)):
			yield self.get_curve(period_index=period_index)

	def __len__(self):
		return len(self.periods)

	def get_curve(self, period=None, period_index=None):
		if period is not None:
			period_index = np.argmin(np.abs(self.periods - period))
		else:
			period = self.periods[period_index]
		matrix = self.deagg_matrix[period_index]
		intensities = self.intensities[period_index]
		return DeaggregationCurve(self.bin_edges, matrix, self.site, self.imt, intensities, period, self.timespan)

	@classmethod
	def from_deaggregation_curves(self, deagg_curves):
		periods = np.array([dc.period for dc in deagg_curves])
		period_indexes = np.argsort(periods)
		deagg_matrixes = [deagg_curves[period_index].deagg_matrix for period_index in period_indexes]
		deagg_matrix = np.concatenate(deagg_matrixes)
		intensity_arrays = [deagg_curves[period_index].intensities for period_index in period_indexes]
		intensities = np.concatenate(intensities)
		# TODO: check that bin_edges etc. are identical
		bin_edges = deagg_curves[0].bin_edges
		site = deagg_curves[0].site
		imt = deagg_curves[0].imt
		timespan = deagg_curves[0].timespan
		return SpectralDeaggregationCurve(bin_edges, deagg_matrix, site, imt, intensities, periods, timespan)

	def filter_cav(self):
		pass
