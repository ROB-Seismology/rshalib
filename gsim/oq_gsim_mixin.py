"""
Mixin methods to be attached to :class:`oqhazlib.gsim.GroundShakingIntensityModel`
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


## Try importing truncnorm speedup
try:
	from ..c_speedups import (norm, truncnorm)
	from ..c_speedups.norm import sf as _norm_sf
except:
	print("Failed importing truncnorm speedup!")
	from scipy.stats import truncnorm
	from scipy.special import ndtr
	from openquake.hazardlib.gsim.base import _truncnorm_sf
	_norm_sf = lambda zvalues: ndtr(-zvalues)
else:
	_truncnorm_sf = lambda tl, zvalues: truncnorm.sf(zvalues, -tl, tl)
	print("Successfully imported truncnorm speedup!")


__all__ = ['make_contexts', 'get_poes_cav']



def make_contexts(self, sites, rupture):
	"""
	Wrapper for :func:`rshalib.gsim.make_gsim_contexts`
	"""
	return make_gsim_contexts(self, sites, rupture)


def get_poes_cav(self, sctx, rctx, dctx, imt, imls, truncation_level,
				 cav_min=0.16, cav_max_mag=5.5, depsilon=0.02,
				 eps_correlation_model="EUS"):
	"""
	Calculate and return probabilities of exceedance (PoEs) of one or more
	intensity measure levels (IMLs) of one intensity measure type (IMT)
	for one or more pairs "site -- rupture", jointly with the probability
	of exceeding a minimum CAV value.

	:param sctx:
		An instance of :class:`SitesContext` with sites information
		to calculate PoEs on.
	:param rctx:
		An instance of :class:`RuptureContext` with a single rupture
		information.
	:param dctx:
		An instance of :class:`DistancesContext` with information about
		the distances between sites and a rupture.

		All three contexts (``sctx``, ``rctx`` and ``dctx``) must conform
		to each other. The easiest way to get them is to call
		:meth:`make_contexts`.
	:param imt:
		An intensity measure type object (that is, an instance of one
		of classes from :mod:`openquake.hazardlib.imt`).
	:param imls:
		List of interested intensity measure levels (of type ``imt``).
	:param truncation_level:
		Can be ``None``, which means that the distribution of intensity
		is treated as Gaussian distribution with possible values ranging
		from minus infinity to plus infinity.

		When set to zero, the mean intensity is treated as an exact
		value (standard deviation is not even computed for that case)
		and resulting array contains 0 in places where IMT is strictly
		lower than the mean value of intensity and 1.0 where IMT is equal
		or greater.

		If truncation_level is None or zero, no CAV filtering is applied.

		When truncation level is positive number, the intensity
		distribution is processed as symmetric truncated Gaussian with
		range borders being ``mean - truncation_level * stddev`` and
		``mean + truncation_level * stddev``. That is, the truncation
		level expresses how far the range borders are from the mean
		value and is defined in units of sigmas. The resulting PoEs
		for that mode are values of complementary cumulative distribution
		function of that truncated Gaussian applied to IMLs.
	:param cav_min:
		Float, CAV threshold (in g.s)
		(default: 0.16)
	:param cav_max_mag:
		Float, maximum magnitude to consider for CAV filtering
		(default: 5.5)
	:param depsilon:
		Float, bin width used to discretize epsilon pga. Should be
		sufficiently small for PGA
		(default: 0.02).
	:param eps_correlation_model:
		Str, name of model used for correlation of epsilon values
		of PGA and SA, either "WUS" or "EUS"
		(default: "EUS")

	:returns:
		2-D float array, joint probabilities for each site (first
		dimension) and each IML (second dimension)

	:raises ValueError:
		If truncation level is not ``None`` and neither non-negative
		float number, and if ``imts`` dictionary contain wrong or
		unsupported IMTs (see :attr:`DEFINED_FOR_INTENSITY_MEASURE_TYPES`).
	"""
	from openquake.hazardlib import const
	from openquake.hazardlib import imt as imt_module

	from ..cav.cav_oq import (calc_cav_exceedance_prob, B1_FREQS, B1_WUS, B1_EUS)

	if truncation_level is not None and truncation_level < 0:
		raise ValueError('truncation level must be zero, positive number '
						 'or None')
	self._check_imt(imt)

	if truncation_level == 0:
		## zero truncation mode, just compare imls to mean
		## No CAV filtering
		imls = self.to_distribution_values(imls)
		mean, _ = self.get_mean_and_stddevs(sctx, rctx, dctx, imt, [])
		mean = mean.reshape(mean.shape + (1, ))
		return (imls <= mean).astype(float)

	else:
		## use real normal distribution
		assert (const.StdDev.TOTAL
				in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES)
		ln_imls = self.to_distribution_values(imls)
		ln_sa_med, [sigma_ln_sa] = self.get_mean_and_stddevs(sctx, rctx, dctx,
													imt, [const.StdDev.TOTAL])
		nsites = len(ln_sa_med)
		ln_sa_med = ln_sa_med.reshape((nsites, 1))
		sigma_ln_sa = sigma_ln_sa.reshape((nsites, 1))
		zvalues = (ln_imls - ln_sa_med) / sigma_ln_sa
		if truncation_level is None:
			## CAV filtering not supported
			if cav_min > 0:
				raise Exception('CAV filtering not supported '
								'if truncation level is None')
			return _norm_sf(zvalues)

		else:
			sa_exceedance_prob = _truncnorm_sf(truncation_level, zvalues)
			if cav_min == 0 or rctx.mag > cav_max_mag:
				return sa_exceedance_prob
			else:
				imt_pga = imt_module.PGA()
				if imt != imt_pga:
					depsilon = 0.5
				## Normally distributed epsilon values and corresponding probabilities
				neps = int(truncation_level / depsilon) * 2 + 1
				eps_pga_array = np.linspace(-truncation_level, truncation_level,
											neps)
				prob_eps_array = (truncnorm.pdf(eps_pga_array, -truncation_level,
												truncation_level) * depsilon)
				#prob_eps_array /= np.cumsum(prob_eps_array)

				## Pre-calculate CAV exceedance probabilities for epsilon PGA
				ln_pga_eps_pga_array = np.zeros((neps, nsites))
				cav_exceedance_prob = np.zeros((neps, nsites))
				ln_pga_med, [sigma_ln_pga] = self.get_mean_and_stddevs(sctx, rctx,
												dctx, imt_pga, [const.StdDev.TOTAL])

				for e, eps_pga in enumerate(eps_pga_array):
					## Determine PGA corresponding to eps_pga
					ln_pga = ln_pga_med + eps_pga * sigma_ln_pga
					ln_pga_eps_pga_array[e] = ln_pga
					## CAV exceedance probability for PGA
					cav_exceedance_prob[e] = calc_cav_exceedance_prob(ln_pga,
											rctx.mag, sctx.vs30, cav_min=cav_min)

				joint_exceedance_probs = np.zeros((nsites, len(imls)))

				if imt == imt_pga:
					## Integrate explicitly over epsilon
					for e in range(neps):
						for d in range(nsites):
							idxs = ln_pga_eps_pga_array[e][d] > ln_imls
							joint_exceedance_probs[d][idxs] += \
								(prob_eps_array[e] * cav_exceedance_prob[e, d])

				else:
					if eps_correlation_model == "WUS":
						b1 = np.interp([1./imt.period], B1_FREQS, B1_WUS,
										left=B1_WUS[0], right=B1_WUS[-1])[0]
					elif eps_correlation_model == "EUS":
						b1 = np.interp([1./imt.period], B1_FREQS, B1_EUS,
										left=B1_EUS[0], right=B1_EUS[-1])[0]

					## Eq. 3-3
					sigma_ln_sa_given_pga = np.sqrt(1 - b1*b1) * sigma_ln_sa

					## Loop over eps_pga
					for e in range(neps):
						eps_pga = eps_pga_array[e]
						## Determine epsilon value of SA, and sigma
						## Eq. 3-1
						eps_sa = b1 * eps_pga

						## Eq. 3-2
						ln_sa_given_pga = ln_sa_med + eps_sa * sigma_ln_sa

						for d in range(nsites):
							## Determine probability of exceedance of SA given PGA
							## Eq. 4-3
							eps_sa_dot = ((ln_imls - ln_sa_given_pga[d])
											/ sigma_ln_sa_given_pga[d])
							## Eq. 4-2
							prob_sa_given_pga = _truncnorm_sf(truncation_level,
															eps_sa_dot)

							joint_exceedance_probs[d] += \
								(prob_eps_array[e] * cav_exceedance_prob[e, d]
								* prob_sa_given_pga)

					## Workaround to make sure SA values are properly truncated
					joint_exceedance_probs = np.minimum(joint_exceedance_probs,
														sa_exceedance_prob)

				return joint_exceedance_probs
