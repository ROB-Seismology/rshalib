"""
PSHAModel class
"""

from __future__ import absolute_import, division, print_function, unicode_literals

try:
	## Python 2
	basestring
except:
	## Python 3
	basestring = str

# TODO: check if documentation is compatibele with Sphinx
# NOTE: damping for spectral periods is fixed at 5.


### imports
import os
from collections import OrderedDict

import numpy as np

from .. import oqhazlib
from openquake.hazardlib.imt import PGA, SA, PGV, PGD, MMI

from ..calc import mp
from ..geo import *
from ..site import *
from ..result import (SpectralHazardCurveField, ProbabilityArray, ProbabilityMatrix,
						DeaggregationSlice, SpectralDeaggregationCurve)
from ..logictree import SeismicSourceSystem
from ..source import SourceModel
from ..gsim import GroundMotionModel
from .pshamodelbase import PSHAModelBase



# TODO: make distinction between imt (PGA, SA) and im (SA(0.5, 5.0), SA(1.0, 5.0))
# (perhaps put all these functions in rshalib.imt)

# im (or imt_name?): intensity measure, e.g. "PGA", "SA"
# imt: IMT object, e.g. PGA(), SA(0.2, 5.0)
# imls: intensity measure levels, e.g. [0.1, 0.2, 0.3]
# im_periods: dict mapping im to spectral periods, e.g. {"PGA": [0], "SA": [0.1, 0.5, 1.]}
# imtls --> imt_imls: dict mapping IMT objects to imls (1-D arrays)
# im_imls: dict mapping im strings to imls (2-D arrays)



__all__ = ['PSHAModel']



class PSHAModel(PSHAModelBase):
	"""
	Class representing a single PSHA model.

	:param source_model:
		instance of :class:`SourceModel`
	:param ground_motion_model:
		instance of :class:`GroundMotionModel`

	See :class:`PSHAModelBase` for other arguments.
	"""

	def __init__(self, name, source_model, ground_motion_model, root_folder,
				site_model, ref_soil_params=REF_SOIL_PARAMS,
				imt_periods={'PGA': [0]}, intensities=None,
				min_intensities=0.001, max_intensities=1., num_intensities=100,
				return_periods=[], time_span=50.,
				truncation_level=3., integration_distance=200.):

		"""
		"""
		# TODO: consider moving 'name' parameter to third position, to be in accordance with order of parameters in docstring.
		PSHAModelBase.__init__(self, name, root_folder, site_model, ref_soil_params,
								imt_periods, intensities, min_intensities,
								max_intensities, num_intensities, return_periods,
								time_span, truncation_level, integration_distance)
		self.source_model = source_model
		self.ground_motion_model = ground_motion_model

	@property
	def smlt_path(self):
		try:
			return self.source_model.description
		except:
			return ""

	@property
	def gmpelt_path(self):
		try:
			return self.ground_motion_model.name
		except:
			return ""

	def calc_shcf(self, cav_min=0., combine_pga_and_sa=True):
		"""
		Run PSHA model with oqhazlib, and store result in one or more
		SpectralHazardCurfeField objects.

		:param cav_min:
			float, CAV threshold in g.s
			(default: 0. = no CAV filtering).
		:param combine_pga_and_sa:
			bool, whether or not to combine PGA and SA, if present
			(default: True)

		:return:
			dict {imt (string) : SpectralHazardCurveField object}
		"""
		hazard_result = self.calc_poes(cav_min=cav_min,
										combine_pga_and_sa=combine_pga_and_sa)
		im_imls = self._get_im_imls(combine_pga_and_sa=combine_pga_and_sa)
		im_shcf_dict = {}
		sites = self.get_generic_sites()
		for imt in hazard_result.keys():
			periods = self.imt_periods[imt]
			if imt == "SA" and combine_pga_and_sa and "PGA" in self.imt_periods.keys():
				periods = [0] + list(periods)
			# TODO: add method to PSHAModelBase to associate oqhazlib/OQ imt's with units
			poes = ProbabilityArray(hazard_result[imt])
			shcf = SpectralHazardCurveField(self.name, poes, [''], sites,
								periods, imt, im_imls[imt], 'g', self.time_span)
			im_shcf_dict[imt] = shcf
		return im_shcf_dict

	def calc_poes(self, cav_min=0., combine_pga_and_sa=True):
		"""
		Run PSHA model with oqhazlib. Output is a dictionary mapping intensity
		measure types to probabilities of exceedance (poes).

		:param cav_min:
			float, CAV threshold in g.s
			(default: 0. = no CAV filtering).
		:param combine_pga_and_sa:
			bool, whether or not to combine PGA and SA, if present
			(default: True)

		:return:
			dict {imt (string) : poes (2-D numpy array of poes)}
		"""
		from openquake.hazardlib.calc import hazard_curves_poissonian

		num_sites = len(self.get_soil_site_model())
		hazard_curves = hazard_curves_poissonian(self.source_model,
								self.get_soil_site_model(),
								self._get_imtls(), self.time_span,
								self._get_trt_gsim_dict(), self.truncation_level,
								self.source_site_filter, self.rupture_site_filter,
								cav_min=cav_min)
		hazard_result = {}
		for imt, periods in self.imt_periods.items():
			if imt == "SA":
				poes = np.zeros((num_sites, len(periods), self.num_intensities))
				for k, period in enumerate(periods):
					poes[:,k,:] = hazard_curves[eval(imt)(period, 5.)]
				hazard_result[imt] = poes
			else:
				poes = hazard_curves[eval(imt)()].reshape(num_sites, 1,
														self.num_intensities)
				hazard_result[imt] = poes
		if (combine_pga_and_sa and "PGA" in self.imt_periods.keys()
			and "SA" in self.imt_periods.keys()):
			hazard_result["SA"] = np.concatenate([hazard_result["PGA"],
												hazard_result["SA"]], axis=1)
			del hazard_result["PGA"]
		return hazard_result

	def calc_shcf_mp(self, cav_min=0, decompose_area_sources=False,
					individual_sources=False, num_cores=None,
					combine_pga_and_sa=True, verbose=True):
		"""
		Parallellized computation of spectral hazard curve field.

		Note: at least in Windows, this method has to be executed in
		a main section (i.e., behind if __name__ == "__main__":)

		:param cav_min:
			float, CAV threshold in g.s
			(default: 0)
		:param decompose_area_sources:
			bool, whether or not area sources should be decomposed into
			point sources for the computation
			(default: False)
		:param individual_sources:
			bool, whether or not hazard curves should be computed for each
			source individually
			(default: False)
		:param num_cores:
			int, number of CPUs to be used. Actual number of cores used
			may be lower depending on available cores and memory
			(default: None, will determine automatically)
		:param combine_pga_and_sa:
			bool, whether or not to combine PGA and SA, if present
			(default: True)
		:param verbose:
			bool, whether or not to print some progress information
			(default: True)

		:return:
			dictionary mapping intensity measure (str) to:
			- instance of :class:`SpectralHazardCurveField`
			(if individual_sources is False) or
			- dict mapping source IDs to instances of
			:class:`SpectralHazardCurveField` (if group_sources is False)
		"""
		if not num_cores:
			num_cores = mp.multiprocessing.cpu_count()

		if decompose_area_sources:
			source_model = self.source_model.decompose_area_sources()
			num_decomposed_sources = self.source_model.get_num_decomposed_sources()
			cum_num_decomposed_sources = np.concatenate([[0],
										np.add.accumulate(num_decomposed_sources)])
		else:
			source_model = self.source_model

		## Create list with arguments for each job
		job_args = []
		for source in source_model:
			job_args.append((self, source, cav_min, verbose))

		## Launch multiprocessing
		curve_list = mp.run_parallel(mp.calc_shcf_by_source, job_args, num_cores,
									verbose=verbose)
		poes = ProbabilityArray(curve_list)

		## Recombine hazard curves computed for each source
		if not individual_sources:
			poes = np.prod(poes, axis=0)
		else:
			total_poes = np.prod(poes, axis=0)
			if decompose_area_sources:
				curve_list = []
				for src_idx, src in enumerate(self.source_model):
					start = cum_num_decomposed_sources[src_idx]
					stop = cum_num_decomposed_sources[src_idx+1]
					curve_list.append(np.prod(poes[start:stop], axis=0))
				poes = ProbabilityArray(curve_list)

		## Convert non-exceedance to exceedance probabilities
		poes -= 1
		poes *= -1
		if individual_sources:
			total_poes -= 1
			total_poes *= -1

		## Construct spectral hazard curve field
		shcf_dict = {}
		sites = self.get_generic_sites()
		imts = self._get_imts()
		im_imls = self._get_im_imls(combine_pga_and_sa=combine_pga_and_sa)
		for im, intensities in im_imls.items():
			periods = self.imt_periods[im]
			## Determine period indexes in poes array
			period_idxs = []
			for T in periods:
				imt = self._construct_imt(im, T)
				period_idxs.append(imts.index(imt))
			if im == "SA" and combine_pga_and_sa and "PGA" in self.imt_periods.keys():
				periods = np.concatenate([[0], periods])
				imt = self._construct_imt("PGA", 0)
				period_idxs.insert(0, imts.index(imt))

			if individual_sources:
				src_shcf_dict = OrderedDict()
				for i, src in enumerate(self.source_model):
					src_shcf_dict[src.source_id] = SpectralHazardCurveField(self.name,
													poes[i][:,period_idxs,:],
													[""]*len(periods), sites,
													periods, im, intensities, 'g',
													self.time_span)
				src_shcf_dict['Total'] = SpectralHazardCurveField(self.name,
											total_poes[:,period_idxs,:],
											[""]*len(periods), sites, periods, im,
											intensities, 'g', self.time_span)
				shcf_dict[im] = src_shcf_dict
			else:
				shcf = SpectralHazardCurveField(self.name, poes[:,period_idxs,:],
								[""]*len(periods), sites, periods, im, intensities,
								'g', self.time_span)
				shcf_dict[im] = shcf

		return shcf_dict

	def deagg_oqhazlib(self, site, imt, iml, return_period,
						mag_bin_width=None, dist_bin_width=10.,
						n_epsilons=None, coord_bin_width=1.0):
		"""
		Run deaggregation with oqhazlib

		:param site:
			instance of :class:`GenericSite` or :class:`SoilSite`
		:param imt:
			Instance of :class:`oqhazlib.imt._IMT`, intensity measure type
		:param iml:
			float, intensity measure level
		:param return_period:
			float, return period corresponding to iml
		:param mag_bin_width:
			float, magnitude bin width (default: None, will take MFD bin width
				of first source)
		:param dist_bin_width:
			float, distance bin width in km (default: 10.)
		:param n_epsilons:
			Int, number of epsilon bins (default: None, will result in bins
				corresponding to integer epsilon values)
		:param coord_bin_width:
			float, lon/lat bin width in decimal degrees (default: 1.)

		:return:
			instance of :class:`DeaggregationSlice`
		"""
		from openquake.hazardlib.calc import disaggregation_poissonian

		if not n_epsilons:
			n_epsilons = 2 * int(np.ceil(self.truncation_level))
		if not mag_bin_width:
			mag_bin_width = self.source_model[0].mfd.bin_width
		if not isinstance(site, SoilSite):
			site = site.to_soil_site(self.ref_soil_params)
		#imt = self._get_imtls()
		ssdf = self.source_site_distance_filter
		rsdf = self.rupture_site_distance_filter

		bin_edges, deagg_matrix = disaggregation_poissonian(self.source_model,
								site, imt, iml,
								self._get_trt_gsim_dict(), self.time_span,
								self.truncation_level, n_epsilons, mag_bin_width,
								dist_bin_width, coord_bin_width, ssdf, rsdf)
		deagg_matrix = ProbabilityMatrix(deagg_matrix)
		imt_name = str(imt).split('(')[0]
		if imt_name == "SA":
			period = imt.period
		else:
			period = 0
		return DeaggregationSlice(bin_edges, deagg_matrix, site, imt_name, iml,
								period, return_period, self.time_span)

	def deagg_oqhazlib_multi(self, site_imtls,
							mag_bin_width=None, dist_bin_width=10.,
							n_epsilons=None, coord_bin_width=1.0):
		"""
		Run deaggregation with oqhazlib for multiple sites, multiple imt's
		per site, and multiple iml's per iml

		:param site_imtls:
			nested dictionary mapping (lon, lat) tuples to dictionaries
			mapping oqhazlib intensity measure type objects to 1-D arrays
			of intensity measure levels
		:param mag_bin_width:
			float, magnitude bin width (default: None, will take MFD bin width
				of first source)
		:param dist_bin_width:
			float, distance bin width in km (default: 10.)
		:param n_epsilons:
			Int, number of epsilon bins (default: None, will result in bins
				corresponding to integer epsilon values)
		:param coord_bin_width:
			float, lon/lat bin width in decimal degrees (default: 1.)

		:return:
			instance of :class:`SpectralDeaggregationCurve` or None
		"""
		from openquake.hazardlib.calc import disaggregation_poissonian_multi

		if not n_epsilons:
			n_epsilons = 2 * int(np.ceil(self.truncation_level))
		if not mag_bin_width:
			mag_bin_width = self.source_model[0].mfd.bin_width

		ssdf = self.source_site_distance_filter
		rsdf = self.rupture_site_distance_filter

		site_model = self.get_soil_site_model()
		all_sites = site_model.get_generic_sites()
		deagg_soil_sites = [site for site in site_model.get_sites()
							if (site.lon, site.lat) in site_imtls.keys()]
		deagg_site_model = SoilSiteModel("", deagg_soil_sites)
		for deagg_result in disaggregation_poissonian_multi(self.source_model,
										deagg_site_model, site_imtls,
										self._get_trt_gsim_dict(),
										self.time_span, self.truncation_level,
										n_epsilons, mag_bin_width, dist_bin_width,
										coord_bin_width, ssdf, rsdf):
			deagg_site, bin_edges, deagg_matrix = deagg_result
			if (bin_edges, deagg_matrix) == (None, None):
				## No deaggregation results for this site
				yield None
			else:
				for site in all_sites:
					if (deagg_site.location.longitude == site.lon
						and deagg_site.location.latitude == site.lat):
						break
				imtls = site_imtls[(site.lon, site.lat)]
				imts = imtls.keys()
				periods = [getattr(imt, "period", 0) for imt in imts]
				intensities = np.array([imtls[imt] for imt in imts])
				deagg_matrix = ProbabilityMatrix(deagg_matrix)
				yield SpectralDeaggregationCurve(bin_edges, deagg_matrix, site,
											"SA", intensities, periods,
											self.return_periods, self.time_span)

	def get_deagg_bin_edges(self, mag_bin_width, dist_bin_width, coord_bin_width,
							n_epsilons):
		"""
		Determine bin edges for deaggregation.
		Note: no default values!

		:param mag_bin_width:
			float, magnitude bin width
		:param dist_bin_width:
			float, distance bin width in km
		:param coord_bin_width:
			float, lon/lat bin width in decimal degrees
		:param n_epsilons:
			Int, number of epsilon bins
			corresponding to integer epsilon values)

		:return:
			(mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, src_bins) tuple
			- mag_bins: magnitude bin edges
			- dist_bins: distance bin edges
			- lon_bins: longitude bin edges
			- lat_bins: latitude bin edges
			- eps_bins: epsilon bin edges
			- src_bins: source bins
		"""
		from openquake.hazardlib.geo.geodetic import npoints_between
		from openquake.hazardlib.geo.utils import get_longitudinal_extent

		min_mag, max_mag = self.source_model.min_mag, self.source_model.max_mag
		dmag = np.ceil((max_mag - min_mag) / mag_bin_width) * mag_bin_width
		max_mag = min_mag + dmag
		nmags = int(round(dmag / mag_bin_width))
		mag_bins = min_mag + mag_bin_width * np.arange(nmags + 1)
		## (copied from oqhazlib)
		#mag_bins = mag_bin_width * np.arange(
		#	int(np.floor(min_mag / mag_bin_width)),
		#	int(np.ceil(max_mag / mag_bin_width) + 1)
		#)

		min_dist, max_dist = 0, self.integration_distance
		dist_bins = dist_bin_width * np.arange(
			int(np.floor(min_dist / dist_bin_width)),
			int(np.ceil(max_dist / dist_bin_width) + 1)
		)

		## Note that ruptures may extend beyond source limits!
		west, east, south, north = self.source_model.get_bounding_box()
		west -= coord_bin_width
		east += coord_bin_width

		lon_bins = coord_bin_width * np.arange(
			int(np.floor(west / coord_bin_width)),
			int(np.ceil(east / coord_bin_width) + 1))

		south -= coord_bin_width
		north += coord_bin_width
		lat_bins = coord_bin_width * np.arange(
			int(np.floor(south / coord_bin_width)),
			int(np.ceil(north / coord_bin_width) + 1))

		eps_bins = np.linspace(-self.truncation_level, self.truncation_level,
								  n_epsilons + 1)

		src_bins = [src.source_id for src in self.source_model]

		return (mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, src_bins)

	def deaggregate(self, site_imtls, mag_bin_width=None, dist_bin_width=10.,
					n_epsilons=None, coord_bin_width=1.0, dtype='d', verbose=False):
		"""
		Hybrid rshalib/oqhazlib deaggregation for multiple sites, multiple
		imt's per site, and multiple iml's per iml, that is more speed- and
		memory-efficient than the standard deaggregation method in oqhazlib.
		Note that deaggregation by tectonic region type is replaced with
		deaggregation by source.

		:param site_imtls:
			nested dictionary mapping (lon, lat) tuples to dictionaries
			mapping oqhazlib IMT objects to 1-D arrays of intensity measure
			levels
		:param mag_bin_width:
			float, magnitude bin width (default: None, will take MFD bin width
			of first source)
		:param dist_bin_width:
			float, distance bin width in km (default: 10.)
		:param n_epsilons:
			Int, number of epsilon bins (default: None, will result in bins
			corresponding to integer epsilon values)
		:param coord_bin_width:
			float, lon/lat bin width in decimal degrees (default: 1.)
		:param dtype:
			str, precision of deaggregation matrix (default: 'd')
		:param verbose:
			Bool, whether or not to print some progress information

		:return:
			dict, mapping site (lon, lat) tuples to instances of
			:class:`SpectralDeaggregationCurve`
		"""
		from openquake.hazardlib.site import SiteCollection

		# TODO: determine site_imtls from self.return_periods (separate method)
		if site_imtls in (None, {}):
			pass

		if not n_epsilons:
			n_epsilons = 2 * int(np.ceil(self.truncation_level))
		if not mag_bin_width:
			mag_bin_width = self.source_model[0].mfd.bin_width

		## Determine bin edges first
		bin_edges = self.get_deagg_bin_edges(mag_bin_width, dist_bin_width,
											coord_bin_width, n_epsilons)
		mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, src_bins = bin_edges

		## Create deaggregation matrices
		deagg_matrix_dict = {}
		for site_key in site_imtls.keys():
			deagg_matrix_dict[site_key] = {}
			imtls = site_imtls[site_key]
			imts = imtls.keys()
			num_imts = len(imts)
			num_imls = len(imtls[imts[0]])

			deagg_matrix_shape = (num_imts, num_imls, len(mag_bins) - 1,
							len(dist_bins) - 1, len(lon_bins) - 1,
							len(lat_bins) - 1, len(eps_bins) - 1, len(src_bins))

			## Initialize array with ones representing NON-exceedance probabilities !
			deagg_matrix = ProbabilityMatrix(np.ones(deagg_matrix_shape, dtype=dtype))
			deagg_matrix_dict[site_key] = deagg_matrix

		## Perform deaggregation
		tom = self.poisson_tom
		gsims = self._get_trt_gsim_dict()
		source_site_filter = self.source_site_filter
		rupture_site_filter = self.rupture_site_filter

		site_model = self.get_soil_site_model()
		deagg_soil_sites = [site for site in site_model.get_sites()
							if (site.lon, site.lat) in site_imtls.keys()]
		deagg_site_model = SoilSiteModel("", deagg_soil_sites)

		sources = self.source_model.sources
		sources_sites = ((source, deagg_site_model) for source in sources)
		for src_idx, (source, s_sites) in \
				enumerate(source_site_filter(sources_sites)):

			if verbose:
				print(source.source_id)

			tect_reg = source.tectonic_region_type
			gsim = gsims[tect_reg]

			ruptures_sites = ((rupture, s_sites)
							  for rupture in source.iter_ruptures(tom))
			for rupture, r_sites in rupture_site_filter(ruptures_sites):
				## Extract rupture parameters of interest
				mag_idx = np.digitize([rupture.mag], mag_bins)[0] - 1

				sitemesh = r_sites.mesh
				sctx, rctx, dctx = gsim.make_contexts(r_sites, rupture)
				if hasattr(dctx, "rjb"):
					jb_dists = getattr(dctx, "rjb")
				else:
					jb_dists = rupture.surface.get_joyner_boore_distance(sitemesh)
				closest_points = rupture.surface.get_closest_points(sitemesh)
				lons = [pt.longitude for pt in closest_points]
				lats = [pt.latitude for pt in closest_points]

				dist_idxs = np.digitize(jb_dists, dist_bins) - 1
				lon_idxs = np.digitize(lons, lon_bins) - 1
				lat_idxs = np.digitize(lats, lat_bins) - 1

				## Compute probability of one or more rupture occurrences
				prob_one_or_more = rupture.get_probability_one_or_more_occurrences()

				## compute conditional probability of exceeding iml given
				## the current rupture, and different epsilon level, that is
				## ``P(IMT >= iml | rup, epsilon_bin)`` for each of epsilon bins
				for site_idx, site in enumerate(r_sites):
					dist_idx = dist_idxs[site_idx]
					lon_idx = lon_idxs[site_idx]
					lat_idx = lat_idxs[site_idx]
					site_key = (site.location.longitude, site.location.latitude)
					imtls = site_imtls[site_key]
					imts = imtls.keys()
					sctx2, rctx2, dctx2 = gsim.make_contexts(SiteCollection([site]),
															rupture)
					for imt_idx, imt in enumerate(imts):
						imls = imtls[imt]
						## In contrast to what is stated in the documentation,
						## disaggregate_poe does handle more than one iml
						poes_given_rup_eps = gsim.disaggregate_poe(sctx2, rctx2,
							dctx2, imt, imls, self.truncation_level, n_epsilons)

						## Probability of non-exceedance
						pone = (1. - prob_one_or_more) ** poes_given_rup_eps

						try:
							deagg_matrix_dict[site_key][imt_idx, :, mag_idx, dist_idx, lon_idx, lat_idx, :, src_idx] *= pone
						except IndexError:
							## May fail if rupture extent is beyond (lon,lat) range
							pass

		## Create SpectralDeaggregationCurve for each site
		deagg_result = {}
		all_sites = site_model.get_generic_sites()
		for deagg_site in deagg_site_model:
			for site in all_sites:
				if (deagg_site.location.longitude == site.lon
					and deagg_site.location.latitude == site.lat):
					break
			site_key = (site.lon, site.lat)
			imtls = site_imtls[site_key]
			imts = imtls.keys()
			periods = [getattr(imt, "period", 0) for imt in imts]
			intensities = np.array([imtls[imt] for imt in imts])
			deagg_matrix = deagg_matrix_dict[site_key]
			## Convert probabilities of non-exceedance back to poes
			#deagg_matrix = 1 - deagg_matrix
			## Modify matrix in-place to save memory
			deagg_matrix -= 1
			deagg_matrix *= -1
			deagg_result[site_key] = SpectralDeaggregationCurve(bin_edges,
										deagg_matrix, site, "SA", intensities,
										periods, self.return_periods, self.time_span)

		return deagg_result

	def deaggregate_mp(self, site_imtls, decompose_area_sources=False,
					mag_bin_width=None, dist_bin_width=10., n_epsilons=None,
					coord_bin_width=1.0, dtype='d', num_cores=None, verbose=False):
		"""
		Hybrid rshalib/oqhazlib deaggregation for multiple sites, multiple
		imt's per site, and multiple iml's per imt, using multiprocessing.
		Note that deaggregation by tectonic region type is replaced with
		deaggregation by source.

		Note: at least in Windows, this method has to be executed in
		a main section (i.e., behind if __name__ == "__main__":)

		:param site_imtls:
			nested dictionary mapping (lon, lat) tuples to dictionaries
			mapping oqhazlib IMT objects to 1-D arrays of intensity measure
			levels
		:param decompose_area_sources:
			bool, whether or not area sources should be decomposed into
			point sources for the computation (default: False)
		:param mag_bin_width:
			float, magnitude bin width (default: None, will take MFD bin width
			of first source)
		:param dist_bin_width:
			float, distance bin width in km (default: 10.)
		:param n_epsilons:
			Int, number of epsilon bins (default: None, will result in bins
			corresponding to integer epsilon values)
		:param coord_bin_width:
			float, lon/lat bin width in decimal degrees (default: 1.)
		:param dtype:
			str, precision of deaggregation matrix (default: 'd')
		:param num_cores:
			int, number of CPUs to be used. Actual number of cores used
			may be lower depending on available cores
			(default: None, will determine automatically)
		:param verbose:
			Bool, whether or not to print some progress information

		:return:
			dict, mapping site (lon, lat) tuples to instances of
			:class:`SpectralDeaggregationCurve`
		"""
		if not n_epsilons:
			n_epsilons = 2 * int(np.ceil(self.truncation_level))
		if not mag_bin_width:
			mag_bin_width = self.source_model[0].mfd.bin_width

		## Determine bin edges first
		bin_edges = self.get_deagg_bin_edges(mag_bin_width, dist_bin_width,
											coord_bin_width, n_epsilons)
		mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, src_bins = bin_edges

		## Create deaggregation matrices
		num_sites = len(site_imtls.keys())
		imtls = site_imtls[site_imtls.keys()[0]]
		num_imts = len(imtls.keys())
		imls = imtls[imtls.keys()[0]]
		num_imls = len(imls)

		deagg_matrix_shape = (num_sites, num_imts, num_imls, len(mag_bins) - 1,
					len(dist_bins) - 1, len(lon_bins) - 1, len(lat_bins) - 1,
					len(eps_bins) - 1, len(src_bins))
		deagg_matrix_len = np.prod(deagg_matrix_shape)

		## Create shared-memory array, and expose it as a numpy array
		shared_deagg_array = mp.multiprocessing.Array(dtype, deagg_matrix_len,
													lock=True)

		## Initialize array with ones representing non-exceedance probabilities !
		deagg_matrix = np.frombuffer(shared_deagg_array.get_obj())
		deagg_matrix = deagg_matrix.reshape(deagg_matrix_shape)
		deagg_matrix += 1

		## Create soil site model for deaggregation sites, making sure
		## order is same as sorted keys of site_imtls
		site_model = self.get_soil_site_model()
		deagg_soil_sites = []
		for (site_lon, site_lat) in sorted(site_imtls.keys()):
			for site in site_model.get_sites():
				if np.allclose((site.lon, site.lat), (site_lon, site_lat), atol=1E-5):
					deagg_soil_sites.append(site)
					break
		deagg_site_model = SoilSiteModel("", deagg_soil_sites)

		## Convert imt's in site_imtls to tuples to avoid mangling up by mp
		copy_of_site_imtls = OrderedDict()
		for site_key in site_imtls.keys():
			copy_of_site_imtls[site_key] = OrderedDict()
			for imt in site_imtls[site_key]:
				copy_of_site_imtls[site_key][tuple(imt)] = site_imtls[site_key][imt]

		if decompose_area_sources:
			source_model = self.source_model.decompose_area_sources()
			num_decomposed_sources = self.source_model.get_num_decomposed_sources()
			cum_num_decomposed_sources = np.add.accumulate(num_decomposed_sources)
		else:
			source_model = self.source_model

		## Create list with arguments for each job
		job_args = []
		for idx, source in enumerate(source_model.sources):
			if decompose_area_sources:
				src_idx = np.where(cum_num_decomposed_sources > idx)[0][0]
			else:
				src_idx = idx
			job_args.append((self, source, src_idx, deagg_matrix_shape,
							copy_of_site_imtls, deagg_site_model, mag_bins,
							dist_bins, eps_bins, lon_bins, lat_bins, dtype,
							verbose))

		## Launch multiprocessing
		if not num_cores:
			num_cores = mp.multiprocessing.cpu_count()

		mp.run_parallel(mp.deaggregate_by_source, job_args, num_cores,
						shared_arr=shared_deagg_array, verbose=verbose)

		## Convert to exceedance probabilities
		deagg_matrix -= 1
		deagg_matrix *= -1

		## Create SpectralDeaggregationCurve for each site
		deagg_result = {}
		all_sites = site_model.get_generic_sites()
		for site_idx, site_key in enumerate(sorted(site_imtls.keys())):
			site_lon, site_lat = site_key
			for site in all_sites:
				if site.lon == site_lon and site.lat == site_lat:
					break
			imtls = site_imtls[site_key]
			imts = imtls.keys()
			periods = [getattr(imt, "period", 0) for imt in imts]
			intensities = np.array([imtls[imt] for imt in imts])
			site_deagg_matrix = ProbabilityMatrix(deagg_matrix[site_idx])
			deagg_result[site_key] = SpectralDeaggregationCurve(bin_edges,
										site_deagg_matrix, site, "SA", intensities,
										periods, self.return_periods, self.time_span)

		return deagg_result

	def _interpolate_oq_site_imtls(self, curve_name, sites, imt_periods,
									curve_path="", calc_id=None):
		"""
		Determine intensity levels corresponding to psha-model return
		periods from saved hazard curves. Mainly useful as helper function
		for deaggregation.

		:param curve_name:
			str, identifying hazard curve
			(e.g., "rlz-01", "mean", "quantile_0.84")
		:param sites:
			list with instances of :class:`GenericSite` or instance of
			:class:`GenericSiteModel`. Note that instances
			of class:`SoilSite` will not work with multiprocessing
		:param imt_periods:
			dictionary mapping intensity measure strings to lists of
			spectral periods.
		:param curve_path:
			str, path to hazard curve relative to main hazard-curve folder
			(default: "")
		:param calc_id:
			str, calculation ID.
			(default: None, will determine from folder structure)

		:return:
			nested dictionary mapping (lon, lat) tuples to dictionaries
			mapping oqhazlib IMT objects to 1-D arrays of intensity measure
			levels
		"""
		site_imtls = OrderedDict()
		for site in sites:
			try:
				lon, lat = site.lon, site.lat
			except AttributeError:
				lon, lat = site.location.longitude, site.location.latitude
			site_imtls[(lon, lat)] = OrderedDict()

		## Read hazard_curve_multi if it exists
		try:
			shcf = self.read_oq_shcf(curve_name, curve_path=curve_path,
									calc_id=calc_id)
		except:
			shcf = None

		for im in sorted(imt_periods.keys()):
			for T in sorted(imt_periods[im]):
				imt = self._construct_imt(im, T)

				if shcf:
					hcf = shcf.getHazardCurveField(period_spec=T)
				else:
					## Read individual hazard curves if there is no shcf
					hcf = self.read_oq_hcf(curve_name, im, T,
										curve_path=curve_path, calc_id=calc_id)
				for i, site in enumerate(sites):
					try:
						site_name = site.name
					except AttributeError:
						site_name = sites.site_names[i]
						lon, lat = site.location.longitude, site.location.latitude
					else:
						lon, lat = site.lon, site.lat
					hc = hcf.getHazardCurve(site_name)
					imls = hc.interpolate_return_periods(self.return_periods)
					site_imtls[(lon, lat)][imt] = imls

		return site_imtls

	def _get_implicit_openquake_params(self):
		"""
		Return a dictionary of implicit openquake parameters that are
		defined in source objects
		(rupture_mesh_spacing, area_source_discretization, mfd_bin_width).
		Warnings will be generated if one or more sources have different
		parameters than the first source.
		"""
		all_sources = self.source_model.sources
		rupture_mesh_spacing = all_sources[0].rupture_mesh_spacing
		mfd_bin_width = all_sources[0].mfd.bin_width
		for src in all_sources[1:]:
			if src.rupture_mesh_spacing != rupture_mesh_spacing:
				print("Warning: rupture mesh spacing of src %s different "
					"from that of 1st source!" % src.source_id)
			if src.mfd.bin_width != mfd_bin_width:
				print("Warning: mfd bin width of src %s different "
					"from that of 1st source!" % src.source_id)

		area_sources = self.source_model.get_area_sources()
		if len(area_sources) > 0:
			area_source_discretization = area_sources[0].area_discretization
			for src in area_sources[1:]:
				if src.area_discretization != area_source_discretization:
					print("Warning: area discretization of src %s different "
						"from that of 1st source!" % src.source_id)
		else:
			area_source_discretization = 5.

		params = {}
		params['rupture_mesh_spacing'] = rupture_mesh_spacing
		params['width_of_mfd_bin'] = mfd_bin_width
		params['area_source_discretization'] = area_source_discretization

		return params

	def write_openquake(self, calculation_mode='classical', user_params=None,
						**kwargs):
		"""
		Write PSHA model input for OpenQuake.

		:param calculation_mode:
			str, calculation mode of OpenQuake (options: "classical" or
			"disaggregation")
			(default: "classical")
		:param user_params:
			{str, val} dict, defining respectively parameters and value
			for OpenQuake
			(default: None).
		"""
		from ..poisson import poisson_conv
		from ..openquake import OQ_Params

		# TODO: depending on how we implement deaggregation, calculation_mode may be dropped in the future
		## set OQ_params object and override with params from user_params
		params = OQ_Params(calculation_mode=calculation_mode, description=self.name)
		implicit_params = self._get_implicit_openquake_params()
		for key in implicit_params:
			setattr(params, key, implicit_params[key])
		if user_params:
			for key in user_params:
				setattr(params, key, user_params[key])

		if calculation_mode == "classical":
			params.mean_hazard_curves = False
			params.quantile_hazard_curves = []
			params.number_of_logic_tree_samples = 1

		## set sites or grid_outline
		if (isinstance(self.site_model, GenericSiteModel)
			and self.site_model.grid_outline):
			grid_spacing_km = self.site_model._get_grid_spacing_km()
			params.set_grid_or_sites(grid_outline=self.site_model.grid_outline,
									grid_spacing=grid_spacing_km)
		else:
			params.set_grid_or_sites(sites=self.get_sites())

		if not os.path.exists(self.oq_root_folder):
			os.mkdir(self.oq_root_folder)

		## write nrml file for source model
		self.source_model.write_xml(os.path.join(self.oq_root_folder,
												self.source_model.name + '.xml'))

		## write nrml file for soil site model if present and set file param
		## or set ref soil params
		self._handle_oq_soil_params(params)

		## validate source model logic tree and write nrml file
		source_model_lt = SeismicSourceSystem(self.source_model.name, self.source_model)
		source_model_lt.validate()
		source_model_lt_file_name = 'source_model_lt.xml'
		source_model_lt.write_xml(os.path.join(self.oq_root_folder,
												source_model_lt_file_name))
		params.source_model_logic_tree_file = source_model_lt_file_name

		## create ground_motion_model logic tree and write nrml file
		optimized_gmm = self.ground_motion_model.get_optimized_model(self.source_model)
		ground_motion_model_lt = optimized_gmm.to_ground_motion_system()
		ground_motion_model_lt_file_name = 'ground_motion_model_lt.xml'
		ground_motion_model_lt.write_xml(os.path.join(self.oq_root_folder,
											ground_motion_model_lt_file_name))
		params.gsim_logic_tree_file = ground_motion_model_lt_file_name

		## convert return periods and time_span to poes
		if not (self.return_periods is None or len(self.return_periods) == 0):
			if calculation_mode == "classical":
				params.poes = poisson_conv(t=self.time_span, tau=self.return_periods)
			elif calculation_mode == "disaggregation":
				params.poes_disagg = poisson_conv(t=self.time_span,
													tau=self.return_periods)

		## set other params
		params.intensity_measure_types_and_levels = self._get_openquake_imts()
		params.investigation_time = self.time_span
		params.truncation_level = self.truncation_level
		params.maximum_distance = self.integration_distance
		params.number_of_logic_tree_samples = 0

		if calculation_mode == "disaggregation":
			params.poes_disagg = kwargs["poes_disagg"]
			params.mag_bin_width = kwargs["mag_bin_width"]
			params.distance_bin_width = kwargs["distance_bin_width"]
			params.coordinate_bin_width =kwargs["coordinate_bin_width"]
			params.num_epsilon_bins = kwargs["num_epsilon_bins"]

		# validate and write oq params to ini file
		params.validate()
		params.write_config(os.path.join(self.oq_root_folder, 'job.ini'))

	def write_crisis(self, filespec="", atn_folder="", site_filespec="",
					atn_Mmax=None, mag_scale_rel="", overwrite=False):
		"""
		Write full PSHA model input for Crisis.

		:param filespec:
			str, full path to CRISIS input .DAT file
			(default: "").
		:param atn_folder:
			str, full path to folder with attenuation tables (.ATN files)
			(default: "").
		:param site_filespec:
			str, full path to .ASC file containing sites where hazard
			will be computed
			(default: "")
		:param atn_Mmax:
			float, maximum magnitude in attenuation table(s)
			(default: None, will determine automatically from source model)
		:param mag_scale_rel:
			str, name of magnitude-area scaling relationship to be used,
			one of "WC1994", "Brune1970" or "Singh1980"
			If empty, the scaling relationships associated with the individual
			source objects will be used.
			(default: "").
		:param overwrite:
			bool, whether or not to overwrite existing input files
			(default: False)

		:return:
			str, full path to CRISIS input .DAT file
		"""
		from ..crisis import write_DAT_2007

		## Raise exception if model contains sites with different
		## vs30 and/or kappa
		if isinstance(self.site_model, SoilSiteModel):
			if (len(set(self.site_model.vs30)) > 1
				or (not np.isnan(self.site_model.kappa).all() and
				len(set(self.site_model.kappa)) > 1)):
				raise Exception("CRISIS2007 does not support sites "
								"with different VS30 and/or kappa!")

		if not os.path.exists(self.crisis_root_folder):
			os.mkdir(self.crisis_root_folder)

		## Construct default filenames and paths if none are specified
		if not filespec:
			filespec = os.path.join(self.crisis_root_folder, self.name + '.DAT')
		if not atn_folder:
			atn_folder = os.path.join(self.crisis_root_folder, 'gsims')
		if not os.path.exists(atn_folder):
			os.mkdir(atn_folder)
		if not site_filespec:
			site_filespec = os.path.join(self.crisis_root_folder, 'sites.ASC')

		## Map gsims to attenuation tables
		gsim_atn_map = {}
		for gsim in self._get_used_gsims():
			gsim_atn_map[gsim] = os.path.join(atn_folder, gsim + '.ATN')

		## Convert grid spacing if necessary
		# TODO: this doesn't work anymore! grid_spacing is now property of site_model
		if isinstance(self.site_model.grid_spacing, basestring):
			grid_spacing = self.site_model._get_grid_spacing_degrees()
		else:
			grid_spacing = self.grid_spacing

		## Write input file. This will also write the site file and attenuation
		## tables if necessary.
		write_DAT_2007(filespec, self.source_model, self.ground_motion_model,
						gsim_atn_map, self.return_periods, self.grid_outline,
						grid_spacing, self.get_sites(), site_filespec,
						self.imt_periods, self.intensities, self.min_intensities,
						self.max_intensities, self.num_intensities, 'g',
						self.name, self.truncation_level, self.integration_distance,
						source_discretization=(1.0, 5.0), vs30=self.ref_soil_params["vs30"],
						kappa=self.ref_soil_params["kappa"],
						mag_scale_rel=mag_scale_rel, atn_Mmax=atn_Mmax,
						output={"gra": True, "map": True, "fue": True,
								"des": True, "smx": True, "eps": True,
								"res_full": False},
						map_filespec="", cities_filespec="", overwrite=overwrite)

		## Return name of output file
		return filespec

	def _get_trt_gsim_dict(self):
		"""
		:return:
			dict, mapping tectonic region types (str) to instances of
			:class:` GroundShakingIntensityModel`
		"""
		return {trt: self._get_gsim(self.ground_motion_model[trt])
				for trt in self._get_used_trts()}

	def _get_used_trts(self):
		"""
		:return:
			list of strings, defining tectonic region types used in
			source model.
		"""
		used_trts = set()
		for source in self.source_model:
			used_trts.add(source.tectonic_region_type)
		return list(used_trts)

	def _get_used_gsims(self):
		"""
		:return:
			list of strings, defining gsims of tectonic region types
			used in source model.
		"""
		used_gsims = set()
		for used_trt in self._get_used_trts():
			used_gsims.add(self.ground_motion_model[used_trt])
		return list(used_gsims)



if __name__ == '__main__':
	"""
	"""
	pass

