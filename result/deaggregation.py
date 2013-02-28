import numpy as np

import openquake.hazardlib.calc.disagg as disagg

from plot import plot_deaggregation


class DeaggregationResult(object):
	"""
	Class representing a full deaggregation result as computed by nhlib.

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
	def __init__(self, bin_edges, deagg_matrix, site, imt, iml, time_span):
		self.bin_edges = bin_edges
		self.matrix = deagg_matrix
		self.site = site
		self.imt = imt
		self.iml = iml
		self.time_span = time_span

	@property
	def nmags(self):
		return self.matrix.shape[0]

	@property
	def ndists(self):
		return self.matrix.shape[1]

	@property
	def nlons(self):
		return self.matrix.shape[2]

	@property
	def nlats(self):
		return self.matrix.shape[3]

	@property
	def neps(self):
		return self.matrix.shape[4]

	@property
	def ntrts(self):
		return self.matrix.shape[5]

	@property
	def mag_bin_edges(self):
		return self.bin_edges[0]

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

	def normalize(self):
		"""
		Normalize probability matrix.
		"""
		self.matrix /= self.get_total_probability()

	def get_total_probability(self):
		"""
		Return total probability of exceedance
		"""
		return np.sum(self.matrix)

	def get_mag_pmf(self):
		"""
		Fold full deaggregation matrix to magnitude PMF.

		:returns:
			1D array, a histogram representing magnitude PMF.
		"""
		return disagg.mag_pmf(self.matrix)

	def get_dist_pmf(self):
		"""
		Fold full deaggregation matrix to distance PMF.

		:returns:
			1D array, a histogram representing distance PMF.
		"""
		return disagg.dist_pmf(self.matrix)

	def get_eps_pmf(self):
		"""
		Fold full deaggregation matrix to epsilon PMF.

		:returns:
			1D array, a histogram representing epsilon PMF.
		"""
		eps_pmf = np.zeros(self.neps)
		for m in xrange(self.neps):
			eps_pmf[m] = np.sum(self.matrix[i][j][k][l][m][n]
							  for i in xrange(self.nmags)
							  for j in xrange(self.ndists)
							  for k in xrange(self.nlons)
							  for l in xrange(self.nlats)
							  for n in xrange(self.ntrts))
		return eps_pmf

	def get_trt_pmf(self):
		"""
		Fold full deaggregation matrix to tectonic region type PMF.

		:returns:
			1D array, a histogram representing tectonic region type PMF.
		"""
		return disagg.trt_pmf(self.matrix)

	def get_mag_dist_pmf(self):
		"""
		Fold full deaggregation matrix to magnitude / distance PMF.

		:returns:
			2D array, first dimension represents magnitude histogram bins,
			second one -- distance histogram bins.
		"""
		return disagg.mag_dist_pmf(self.matrix)

	def get_mag_dist_eps_pmf(self):
		"""
		Fold full deaggregation matrix to magnitude / distance /epsilon PMF.

		:returns:
			3D array, first dimension represents magnitude histogram bins,
			second one -- distance histogram bins, third one -- epsilon
			histogram bins.
		"""
		return disagg.mag_dist_eps_pmf(self.matrix)

	def get_lon_lat_pmf(self):
		"""
		Fold full deaggregation matrix to longitude / latitude PMF.

		:returns:
			2D array, first dimension represents longitude histogram bins,
			second one -- latitude histogram bins.
		"""
		return disagg.lon_lat_pmf(self.matrix)

	def get_mag_lon_lat_pmf(self):
		"""
		Fold full deaggregation matrix to magnitude / longitude / latitude PMF.

		:returns:
			3D array, first dimension represents magnitude histogram bins,
			second one -- longitude histogram bins, third one -- latitude
			histogram bins.
		"""
		return disagg.mag_lon_lat_pmf(self.matrix)

	def get_lon_lat_trt_pmf(self):
		"""
		Fold full deaggregation matrix to longitude / latitude / trt PMF.

		:returns:
			3D array, first dimension represents longitude histogram bins,
			second one -- latitude histogram bins, third one -- trt histogram bins.
		"""
		return disagg.lon_lat_trt_pmf(self.matrix)

	def plot_mag_dist_pmf(self, return_period=475):
		"""
		Plot magnitude / distance PMF.
		"""
		total_probability = self.get_total_probability()
		mag_dist_pmf = self.get_mag_dist_pmf() / total_probability
		mag_dist_pmf = mag_dist_pmf.transpose()
		eps_pmf = self.get_eps_pmf() / total_probability
		eps_bin_edges = self.eps_bin_edges[1:]
		try:
			imt_period = self.imt.period
		except AttributeError:
			imt_period = None
		plot_deaggregation(mag_dist_pmf, self.mag_bin_edges, self.dist_bin_edges, return_period, eps_values=eps_pmf, eps_bin_edges=eps_bin_edges, mr_style="2D", site_name=self.site.name, struc_period=imt_period, title_comment="", fig_filespec=None)

