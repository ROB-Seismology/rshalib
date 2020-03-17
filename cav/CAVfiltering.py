"""
This module implements CAV (cumulative absolute velocity) filtering,
as outlined in Abrahamson et al. (2006) (EPRI technical paper 1014099)
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy.stats



__all__ = ['calc_ln_cav', 'calc_cav_exceedance_prob', 'calc_ln_pga_given_sa',
           'calc_ln_sa_given_pga']


def calc_ln_cav2(pga, M, vs30):
	"""
	Calculate natural logarithm of CAV using a two-step approach including
	duration dependence, based on an empirical model of CAV in the U.S.
	CAV has units of velocity (g.s)

	:param pga:
		float or array, peak horizontal acceleration in g
	:param M:
		float or array, moment magnitude
	:param vs30:
		float or array, shear-wave velocity over the top 30 m in m/s

	If more than one parameter is an array, they must have the same length!

	:return:
		(ln_CAV, sigma_ln_CAV) tuple of floats or arrays
	"""
	## Correlation Coefficients
	C0 = -1.75 #(0.04)
	C1 = 0.0567 #(0.0062)
	C2 = -0.0417 #(0.0043)
	C3 = 0.0737 #(0.10)
	C4 = -0.481 #(0.096)
	C5 = -0.242 #(0.036)
	C6 = -0.0316 #(0.0046)
	C7 = -0.00936 #(0.00833)
	C8 = 0.782 #(0.006)
	C9 = 0.0343 #(0.0013)

	scalar = False
	if np.isscalar(pga) and np.isscalar(M) and np.isscalar(vs30):
		scalar = True
	if np.isscalar(pga):
		pga = np.array([pga])
	if np.isscalar(M):
		M = np.array([M])
	if np.isscalar(vs30):
		vs30 = np.array([vs30])

	## Calculate uniform duration
	ln_dur_uni, sigma_ln_dur_uni = calc_ln_dur_uni(pga, M, vs30)

	## Calculate ln(CAV)
	ln_pga = np.log(pga)
	ln_CAV = np.zeros_like(ln_dur_uni)
	if len(pga) > 1:
		indexes_below = np.where(pga <= 1.0)
		indexes_above = np.where(pga > 1.0)
	else:
		if pga[0] <= 1.0:
			indexes_below = slice(None, None)
			indexes_above = None
		else:
			indexes_below = None
			indexes_above = slice(None, None)

	if indexes_below is not None:
		_ln_pga = ln_pga[indexes_below]
		_M = M[indexes_below] if M.shape == pga.shape else M
		_vs30 = vs30[indexes_below] if vs30.shape == pga.shape else vs30
		_ln_dur_uni = ln_dur_uni[indexes_below] if ln_dur_uni.shape == pga.shape else ln_dur_uni
		ln_CAV[indexes_below] = (C0 + C1 * (_M - 6.5) + C2 * (_M - 6.5)**2
							+ C3 * _ln_pga + C4 * _ln_pga**2 + C5 * _ln_pga**3
							+ C6 * _ln_pga**4 + C7 * (np.log(_vs30) - 6)
							+ C8 * _ln_dur_uni + C9 * _ln_dur_uni**2)
	if indexes_above is not None:
		_ln_pga = ln_pga[indexes_above]
		_M = M[indexes_above] if M.shape == pga.shape else M
		_vs30 = vs30[indexes_above] if vs30.shape == pga.shape else vs30
		_ln_dur_uni = ln_dur_uni[indexes_above] if ln_dur_uni.shape == pga.shape else ln_dur_uni
		ln_CAV[indexes_above] = (C0 + C1 * (_M - 6.5) + C2 * (_M - 6.5)**2
							+ C3 * _ln_pga + C7 * (np.log(_vs30) - 6)
							+ C8 * _ln_dur_uni + C9 * _ln_dur_uni**2)

	## Calculate standard deviation
	dur_uni = np.exp(ln_dur_uni)
	sigma_ln_CAV1 = np.zeros_like(dur_uni)
	sigma_ln_CAV1[:] = 0.1
	sigma_ln_CAV1[dur_uni <= 4] = 0.37 - 0.09 * (ln_dur_uni[dur_uni <= 4] - np.log(0.2))
	sigma_ln_CAV1[dur_uni < 0.2] = 0.37

	sigma_ln_CAV = np.sqrt((C8 + 2 * C9 * ln_dur_uni)**2 * sigma_ln_dur_uni**2
							+ sigma_ln_CAV1**2)

	if scalar:
		ln_CAV = ln_CAV[0]
		sigma_ln_CAV = sigma_ln_CAV[0]

	return (ln_CAV, sigma_ln_CAV)


def calc_ln_dur_uni(pga, M, vs30):
	"""
	Calculate the natural logarithm of the uniform duration (in s)
	above 0.025 g, based on an empirical model for the U.S.
	The uniform duration is the total time during which the absolute
	value of the acceleration time series exceeds a specified threshold
	(0.025 g in the case of CAV)

	:param pga:
		float or array, peak horizontal acceleration in g
	:param M:
		float or array, moment magnitude
	:param vs30:
		float or array, shear-wave velocity over the top 30 m in m/s

	If more than one parameter is an array, they must have the same length!

	:return:
		(ln_dur_uni, sigma_ln_dur_uni) tuple of floats or arrays
	"""
	## Correlation coefficients
	a1 = 3.50 #(0.05)
	a2 = 0.0714 #(0.0421)
	a3 = -4.19 #(0.30)
	a4 = 4.28 #(0.03)
	a5 = 0.733 #(0.010)
	a6 = -0.0871 #(0.0105)
	a7 = -0.355 #(0.020)
	sigma_ln_dur_uni = 0.509

	ln_pga = np.log(pga)
	ln_dur_uni = (a1 + a2 * ln_pga + a3 / (ln_pga + a4) + a5 * (M - 6.5)
				+ a6 * (M - 6.5)**2 + a7 * (np.log(vs30) - 6))

	if not np.isscalar(ln_dur_uni):
		sigma_ln_dur_uni = np.ones_like(ln_dur_uni) * sigma_ln_dur_uni

	return (ln_dur_uni, sigma_ln_dur_uni)


def calc_ln_cav1(pga, M, vs30):
	"""
	Calculate natural logarithm of CAV using a one-step approach
	independent of duration, based on an empirical model of CAV in the U.S.
	CAV has units of velocity (g.s)

	Note: this method does not yield the same results as those shown by
		Abrahamson et al. (2006), so it's better to avoid it

	:param pga:
		float or array, peak horizontal acceleration in g
	:param M:
		float or array, moment magnitude
	:param vs30:
		float or array, shear-wave velocity over the top 30 m in m/s

	If more than one parameter is an array, they must have the same length!

	:return:
		tuple of (ln_CAV, sigma_ln_CAV)
	"""
	## Correlation coefficients
	d1 = -0.405 #(0.11)
	d2 = 0.509 #(0.036)
	d3 = -2.11 #(0.24)
	d4 = 4.25 #(0.05)
	d5 = 0.667 #(0.009)
	d6 = -0.0947 #(0.009)
	d7 = -0.266 #(0.023)
	sigma_ln_CAV = 0.46

	## Calculate ln(CAV)
	ln_pga = np.log(pga)
	ln_CAV = (d1 + d2 * (ln_pga + 2.5) + d3 / (ln_pga + d4) + d5 * (M - 6.5)
			+ d6 * (M - 6.5)**2 + d7 * (np.log(vs30) - 6))

	if not np.isscalar(ln_CAV):
		sigma_ln_CAV = np.ones_like(ln_CAV) * sigma_ln_CAV

	return (ln_CAV, sigma_ln_CAV)


calc_ln_cav = calc_ln_cav2


def calc_cav_exceedance_prob(pga, M, vs30, cav_min=0.16, duration_dependent=True):
	"""
	Calculate probability of exceeding a specified CAV value

	:param pga:
		float or array, peak horizontal acceleration in g
	:param M:
		float or array, moment magnitude
	:param vs30:
		float or array, shear-wave velocity over the top 30 m in m/s
	:param cav_min:
		float, CAV threshold (in g.s)
		(default: 0.16)
	:param duration_dependent:
		bool, indicating whether the two-step, duration-dependent
		or the one-step, duration-independent approach should be used
		(default: True)

	Notes:
	- The one-step approach is not recommended, as it could not
		be validated against the graphs shown in Abrahamson et al. (2006)
	- If more than one parameter is an array, they must have the same length!

	:return:
		float or array, exceedance probability
	"""
	scalar = False
	if np.isscalar(pga) and np.isscalar(M) and np.isscalar(vs30):
		scalar = True
	if np.isscalar(pga):
		pga = np.array([pga])
	if np.isscalar(M):
		M = np.array([M])
	if np.isscalar(vs30):
		vs30 = np.array([vs30])

	prob_shape = (max(len(pga), len(M), len(vs30)),)

	if cav_min > 0:
		if len(pga) > 1:
			non_zero_indexes = np.where(pga >= 0.025)
		else:
			if pga[0] < 0.025:
				non_zero_indexes = np.array([], dtype='i')
			else:
				non_zero_indexes = np.indices((max(len(M), len(vs30)),))

		prob = np.zeros(prob_shape, 'd')

		if duration_dependent:
			ln_CAV, sigma_ln_CAV = calc_ln_cav2(pga, M, vs30)
		else:
			ln_CAV, sigma_ln_CAV = calc_ln_cav1(pga, M, vs30)

		epsilon_CAV = ((np.log(cav_min) - ln_CAV[non_zero_indexes])
						/ sigma_ln_CAV[non_zero_indexes])
		prob[non_zero_indexes] = 1.0 - scipy.stats.norm.cdf(epsilon_CAV)

	else:
		prob = np.ones(prob_shape, 'd')

	if scalar:
		prob = prob[0]

	return prob


def calc_ln_sa_given_pga(pga, M, R, T, vs30, gmpe_name, correlation_model="EUS"):
	"""
	Compute natural logarithm of SA given PGA
	Formulas 3-1 - 3-3 in Abrahamson et al. (2006)

	:param pga:
		float, peak ground acceleration in g
	:param M:
		float, moment magnitude
	:param R:
		float, distance in km
	:param T:
		float, spectral period in s
	:param vs30:
		float, shear-wave velocity over the top 30 m in m/s
	:param gmpe_name:
		str, name of supported GMPE
	:param correlation_model:
		str, name of model used to develop correlation of normalized residuals
		(epsilon values) of PGA and spectral acceleration, either "WUS" or "EUS"
		(default: "EUS")

	:return:
		tuple (ln_SA_given_pga, sigma_ln_SA_given_pga)
	"""
	# Note1: distance should be in same metric as GMPE. This may depend
	# on implementation of deaggregation!
	# Note2: we should include mechanism in addition to vs30
	from ..gsim.gmpe import gmpe as gmpe_module
	import geosurvey.cwp as cwp

	## Determine median spectral acceleration and sigma
	gmpe = getattr(gmpe_module, gmpe_name)()
	sa_med = gmpe(M, R, imt="SA", T=T, vs30=vs30)
	sigma_log_sa = gmpe.log_sigma(M, R, imt="SA", T=T, vs30=vs30)
	sigma_ln_sa = np.log(10) * sigma_log_sa

	## Determine eps_pga from pga
	eps_pga = gmpe.get_epsilon(pga, M, R, imt="PGA", T=0, vs30=vs30)

	## Correlation coefficients between PGA and SA (Table 3-1)
	b1_freqs = np.array([0.5, 1, 2.5, 5, 10, 20, 25, 35])
	b1_WUS = np.array([0.59, 0.59, 0.6, 0.633, 0.787, 0.931, 0.956, 0.976])
	b1_EUS = np.array([0.5, 0.55, 0.6, 0.75, 0.88, 0.9, 0.91, 0.93])
	if correlation_model == "WUS":
		b1 = cwp.intlin(1./b1_freqs, b1_WUS, [T])[0]
	elif correlation_model == "EUS":
		b1 = cwp.intlin(1./b1_freqs, b1_EUS, [T])[0]

	## Determine epsilon value of spectral acceleration at period T, and sigma
	## Eq. 3-1
	eps_sa = b1 * eps_pga
	## Eq. 3-2
	ln_sa_given_pga = np.log(sa_med) + eps_sa * sigma_ln_sa
	## Eq. 3-3
	sigma_ln_sa_given_pga = np.sqrt(1 - b1**2) * sigma_ln_sa

	return (ln_sa_given_pga, sigma_ln_sa_given_pga)


def calc_ln_pga_given_sa(sa, M, R, T, vs30, gmpe_name, correlation_model="EUS"):
	"""
	Compute natural logarithm of PGA given SA
	Formulas 3-1 - 3-3 in Abrahamson et al. (2006)

	:param sa:
		float, spectral acceleration in g
	:param M:
		float, moment magnitude
	:param R:
		float, distance in km
	:param T:
		float, spectral period in s
	:param vs30:
		float, shear-wave velocity over the top 30 m in m/s
	:param gmpe_name:
		str, name of supported GMPE
	:param correlation_model:
		str, name of model used to develop correlation of normalized residuals
		(epsilon values) of PGA and spectral acceleration, either "WUS" or "EUS"
		(default: "EUS")

	:return:
		tuple (ln_pga_given_SA, sigma_ln_pga_given_SA)
	"""
	# Note1: distance should be in same metric as GMPE. This may depend
	# on implementation of deaggregation!
	# Note2: we should include mechanism in addition to vs30
	from ..gsim import gmpe as gmpe_module
	import geosurvey.cwp as cwp

	## Determine median spectral acceleration and sigma
	gmpe = getattr(gmpe_module, gmpe_name)()
	pga_med = gmpe(M, R, imt="PGA", T=0, vs30=vs30)
	sigma_log_pga = gmpe.log_sigma(M, R, imt="PGA", T=0, vs30=vs30)
	sigma_ln_pga = np.log(10) * sigma_log_pga

	## Determine eps_sa from sa
	eps_sa = gmpe.get_epsilon(sa, M, R, imt="SA", T=T, vs30=vs30)

	## Correlation coefficients between PGA and SA (Table 3-1)
	b1_freqs = np.array([0.5, 1, 2.5, 5, 10, 20, 25, 35])
	b1_WUS = np.array([0.59, 0.59, 0.6, 0.633, 0.787, 0.931, 0.956, 0.976])
	b1_EUS = np.array([0.5, 0.55, 0.6, 0.75, 0.88, 0.9, 0.91, 0.93])
	if correlation_model == "WUS":
		b1 = cwp.intlin(1./b1_freqs, b1_WUS, [T])[0]
	elif correlation_model == "EUS":
		b1 = cwp.intlin(1./b1_freqs, b1_EUS, [T])[0]

	## Compute epsilon PGA (Eq. 3-1)
	eps_pga = eps_sa / b1
	## Compute ln PGA given SA (Eq. 3-2)
	ln_pga_given_sa = np.log(pga_med) + eps_pga * sigma_ln_pga
	## Eq. 3-3
	## Not sure if the line below is correct
	#sigma_ln_pga_given_sa = np.sqrt(1 - (1./b1)**2) * sigma_ln_pga
	#sigma_ln_pga_given_sa = np.sqrt(1 - b1**2) * sigma_ln_pga
	## This appears to be more conservative
	sigma_ln_pga_given_sa = sigma_ln_pga

	return (ln_pga_given_sa, sigma_ln_pga_given_sa)


def calc_sa_exceedance_prob_given_pga(z, pga, M, R, T, vs30, gmpe_name,
									correlation_model="EUS"):
	"""
	Compute probability that SA exceeds a target value, given PGA
	Formula 4.2 and 4.3 in Abrahamson et al. (2006)

	:param z:
		float, target spectral acceleration in g
	:param pga:
		float, peak ground acceleration in g
	:param M:
		float, moment magnitude
	:param R:
		float, distance in km
	:param T:
		float, spectral period in s
	:param vs30:
		float, shear-wave velocity over the top 30 m in m/s
	:param gmpe_name:
		str, name of supported GMPE
	:param correlation_model:
		str, name of model used to develop correlation of normalized residuals
		(epsilon values) of PGA and spectral acceleration, either "WUS" or "EUS"
		(default: "EUS")

	:return:
		float, probability of exceedance
	"""
	ln_sa_given_pga, sigma_ln_sa_given_pga = calc_ln_sa_given_pga(pga, M, R, T,
											vs30, gmpe_name, correlation_model)

	## Determine probability of exceedance
	## Eq. 4-3
	eps_sa_prime = (np.log(z) - ln_sa_given_pga) / sigma_ln_sa_given_pga
	## Eq. 4-2
	prob = 1.0 - scipy.stats.norm.cdf(eps_sa_prime)
	return prob


def cav_filtering_from_deagg_pga(deagg_exceedances, magnitudes, intensities,
								vs30, cav_min=0.16, site_amp=1):
	"""
	Perform CAV filtering based on deaggregated mean exceedance rates for PGA

	:param deagg_exceedances:
		3-D array [l,r,m] of M,r-deaggregated exceedance rates
		for a range of intensities (l)
	:param magnitudes:
		1-D array, magnitude values
	:param intensities:
		1-D array,intensity values
	:param vs30:
		float, shear-wave velocity over the top 30 m in m/s
	:param cav_min:
		float, CAV threshold
		(default: 0.16 g.s)
	:param site_amp:
		float, site amplification factor
		(default: 1)

	:return:
		tuple of (CAV_exceedances, CAV_intensities)
		CAV_intensities is identical to the input intensities,
		minus the last element

	Notes:
	- Distance values are not needed, as CAV is distance-independent,
	  at least for PGA
	- The CAV model was developed for surface ground motions (including
	  site effects). When calculating the probability of CAV > 0.16 g.s,
	  it is necessary to first scale zk(rock) to zk(soil) using the
	  site amplification factor !
	- This can only be used for PGA at this moment.
	  For spectral accelerations, a model of the relation between PGA and
	  spectral acceleration needs to be used.
	"""
	## Make sure intensities are ordered from small to large
	if intensities[0] > intensities[-1]:
		intensities = intensities[::-1]
		deagg_exceedances = deagg_exceedances[::-1,:,:]

	## Calculate center values of magnitude bins
	magnitude_bin_widths = np.zeros_like(magnitudes)
	magnitude_bin_widths[:-1] = magnitudes[1:] - magnitudes[:-1]
	magnitude_bin_widths[-1] = magnitude_bin_widths[-2]
	magnitude_bin_centers = magnitudes + magnitude_bin_widths / 2.

	## We can calculate CAV exceedance probabilities before the main loop
	## Note that CAV is distance-independent, so only 2 dimensions are required
	CAV_exceedance_probs = np.zeros((len(intensities), len(magnitudes)), 'd')
	for k in range(len(intensities)):
		zk = intensities[k] * site_amp
		CAV_exceedance_probs[k] = calc_cav_exceedance_prob(zk, magnitude_bin_centers,
															vs30, cav_min)
		#print(zk, CAV_exceedance_probs[k])

	## Calculate rate of occurrence for each intensity interval,
	deagg_occurrences = deagg_exceedances[:-1,:,:] - deagg_exceedances[1:,:,:]
	deagg_occurrences = np.append(deagg_occurrences, deagg_exceedances[-1:,:,:],
								axis=0)
	#print(deagg_occurrences)

	## We can also calculate the filtered occurrence rates before the main loop
	CAV_deagg_occurrences = deagg_occurrences * CAV_exceedance_probs[:,np.newaxis,:]
	CAV_deagg_exceedances = np.cumsum(CAV_deagg_occurrences[::-1,:,:], axis=0)[::-1,:,:]

	## Calculate CAV-filtered exceedance rates
	CAV_total_exceedances = np.zeros(len(intensities), 'd')
	for n in range(len(intensities)):
		CAV_total_exceedances[n] = np.sum(CAV_deagg_exceedances[n])
		#for m in range(len(magnitudes)):
			#M = magnitudes[m]
			#for k in range(n, len(CAV_exceedances)):
				#CAV_exceedances[n] += filtered_occurrence_rates[n,m]
				#CAV_exceedances[n] += filtered_occurrence_rates[k,m]
				#CAV_exceedances[n] += occurrence_rates[k,m] * CAV_exceedance_probs[k,m]
				#zk = intensities[k]
				#CAV_exceedances[n] += occurrence_rates[k,m] * calc_cav_exceedance_prob(zk, M, vs30, cav_min)

	return (CAV_total_exceedances, CAV_deagg_exceedances)


def cav_filtering_from_deagg_sa(deagg_exceedances, magnitudes, distances,
							intensities, periods, gmpe_name, vs30, cav_min=0.16):
	"""
	Perform CAV filtering based on deaggregated mean exceedance rates for PGA

	:param deagg_exceedances:
		4-D array [t,l,r,m] of M,r-deaggregated exceedance rates
		for a range of periods (t) and intensities (l)
	:param magnitudes:
		1-D array, magnitude values
	:param distances:
		1-D array, distance values
	:param intensities:
		2-D array [t,l] of fixed intensity values
	:param periods:
		1-D array, structural periods (s)
	:param gmpe_name:
		str, name of GMPE
	:param vs30:
		float, shear-wave velocity over the top 30 m in m/s
	:param cav_min:
		float, CAV threshold (default: 0.16 g.s)

	:return:
		tuple of (CAV_exceedances, CAV_intensities)
		CAV_intensities is identical to the input intensities,
		minus the last element

	Notes:
	- The CAV model was developed for surface ground motions (including
	  site effects). When calculating the probability of CAV > 0.16 g.s,
	  it is necessary to first scale zk(rock) to zk(soil) using the
	  site amplification factor !
	- This can only be used for PGA at this moment.
	  For spectral accelerations, a model of the relation between PGA and
	  spectral acceleration needs to be used.
	"""
	num_intensities = intensities.shape[1]

	## Make sure intensities are ordered from small to large
	if intensities[0,0] > intensities[0,-1]:
		intensities = intensities[:,::-1]
		deagg_exceedances = deagg_exceedances[:,::-1,:,:]

	## Calculate center values of magnitude and distance bins
	magnitude_bin_widths = np.zeros_like(magnitudes)
	magnitude_bin_widths[:-1] = magnitudes[1:] - magnitudes[:-1]
	magnitude_bin_widths[-1] = magnitude_bin_widths[-2]
	magnitude_bin_centers = magnitudes + magnitude_bin_widths / 2.
	distance_bin_widths = np.zeros_like(distances)
	distance_bin_widths[:-1] = distances[1:] - distances[:-1]
	distance_bin_widths[-1] = distance_bin_widths[-2]
	distance_bin_centers = distances + distance_bin_widths / 2.

	## Compute PGA given SA
	pga_given_sa = np.zeros_like(deagg_exceedances)
	for t in range(len(periods)):
		T = periods[t]
		for k in range(num_intensities):
			sa = intensities[t,k]
			for r in range(len(distance_bin_centers)):
				R = distance_bin_centers[r]
				pga_values = np.exp(calc_ln_pga_given_sa(sa, magnitude_bin_centers,
														R, T, vs30, gmpe_name)[0])
				pga_given_sa[t,k,r] = pga_values

	## Calculate CAV exceedance probabilities corresponding to PGA
	CAV_exceedance_probs = np.zeros((len(periods), num_intensities, len(distances),
									len(magnitudes)), 'd')
	for t in range(len(periods)):
		for k in range(num_intensities):
			for r in range(len(distance_bin_centers)):
				#for m in range(len(magnitude_bin_centers)):
					#M = magnitude_bin_centers[m]
					# Note: no idea how to implement site_amp in this case
					#zk = pga_given_sa[t,k,r,m]
					#CAV_exceedance_probs[t,k,r,m] = calc_cav_exceedance_prob(zk, M, vs30, cav_min)
				zk = pga_given_sa[t,k,r]
				CAV_exceedance_probs[t,k,r] = calc_cav_exceedance_prob(zk,
											magnitude_bin_centers, vs30, cav_min)

	## Calculate rate of occurrence for each intensity interval,
	deagg_occurrences = deagg_exceedances[:,:-1,:,:] - deagg_exceedances[:,1:,:,:]
	deagg_occurrences = np.append(deagg_occurrences, deagg_exceedances[:,-1:,:,:], axis=1)

	## We can calculate the filtered occurrence rates before the main loop
	CAV_deagg_occurrences = deagg_occurrences * CAV_exceedance_probs
	CAV_deagg_exceedances = np.cumsum(CAV_deagg_occurrences[:,::-1,:,:], axis=1)[:,::-1,:,:]

	## Calculate CAV-filtered exceedance rates for each intensity
	#cav_total_occurrences = np.zeros((len(periods), num_intensities - 1), 'd')
	#for t in range(len(periods)):
	#	for n in range(num_intensities - 1):
			#for r in range(len(distances)):
			#	for m in range(len(magnitudes)):
			#		cav_total_occurrences[t,n] += cav_deagg_occurrences[t,n,r,m]
	#		cav_total_occurrences[t,n] = np.sum(cav_deagg_occurrences[t,n,:,:])

	#cav_total_exceedances = np.cumsum(cav_total_occurrences[:,::-1], axis=1)[:,::-1]

	CAV_total_exceedances = np.zeros((len(periods), num_intensities), 'd')
	for t in range(len(periods)):
		for n in range(num_intensities):
			CAV_total_exceedances[t,n] = np.sum(CAV_deagg_exceedances[t,n,:,:])

	#return (CAV_total_exceedances, intensities[:,:-1])
	return (CAV_total_exceedances, CAV_deagg_exceedances)



if __name__ == "__main__":
	import pylab
	import sys
	np.set_printoptions(threshold=sys.maxint)

	## As a test, we reproduce Fig. 2.34 and Fig. 2.39 in Abrahamson et al. (2006)
	## Common parameters
	cav_min = 0.16
	vs30 = 1000
	magnitudes = np.arange(4.0, 7.6, 0.1)

	## Fig. 2.34
	PGA = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
	colors = ["blue", "red", "green", "purple", "orange"]
	for pga, color in zip(PGA, colors):
		CAV_exceedance_probs = calc_cav_exceedance_prob(pga, magnitudes, vs30, cav_min)
		pylab.plot(magnitudes, CAV_exceedance_probs, color, lw=2, label="PGA=%s g" % pga)

	## Fig. 2.39
	## From this plot, it appears that the calculation using the one-step model
	## yields results that are different than in Abrahamson et al. (2006),
	## whereas the two-step model yields similar results
	#PGA = [0.1, 0.2, 0.4]
	#colors = ["b", "r", "g", "m", "y"]
	#duration_dependence = [True, False]
	#method_names = ["two-step", "one-step"]
	#line_styles = ["", "--"]
	#for durdep, method_name, line_style in zip(duration_dependence, method_names, line_styles):
	#	for pga, color in zip(PGA, colors):
	#		CAV_exceedance_probs = calc_cav_exceedance_prob(pga, magnitudes, vs30, cav_min, duration_dependent=durdep)
	#		pylab.plot(magnitudes, CAV_exceedance_probs, color+line_style, lw=2, label="%s g (%s)" % (pga, method_name))

	## Plot decoration
	pylab.xlabel("Moment Magnitude")
	pylab.ylabel("Probability of CAV > %s g.s (for Vs30=%s m/s)" % (cav_min, vs30))
	pylab.grid(True)
	pylab.legend(loc=4)

	fig_filespec = None
	#fig_filespec = r"E:\Home\_kris\Projects\2012 - Electrabel\Progress Report 2\Figures\CAVmodel.png"

	if not fig_filespec:
		pylab.show()
	else:
		pylab.savefig(fig_filespec, dpi=300)
		pylab.cla()


	## Fig. 2-20
	"""
	from hazard.rshalib.utils import logrange
	mags = np.array([4., 5., 6., 7.])
	vs30 = 600
	pga = logrange(0.022, 2., 25)
	for M in mags:
		ln_dur_uni = calc_ln_dur_uni(pga, M, vs30)[0]
		print(ln_dur_uni)
		dur_uni = np.exp(ln_dur_uni)
		pylab.loglog(pga, dur_uni, label="Mag=%d" % M)
	pylab.xlabel("PGA (g)")
	pylab.ylabel("Uniform duration > 0.025 g")
	pylab.grid(True)
	pylab.legend(loc=4)
	pylab.show()
	"""


	## Compare CAV-filtered exceedance rate with original rate
	"""
	import hazard.psha.IO as IO
	cav_min = 0.16
	vs30 = 800
	#psha_filespec = r"D:\PSHA\BEST\LogicTree\PGA\Doel\Seismotectonic\Ambraseys1996\Mmax+0_50\MC000"
	psha_filespec = r"D:\PSHA\BEST\LogicTree\Spectral\Doel\Seismotectonic\Ambraseys1996\Mmax+0_50\MC000"
	psha_result = IO.read_GRA(psha_filespec)
	periods = psha_result.periods
	period_index = 29
	magnitudes, distances, deagg_exceedances, intensities = IO.read_DES_full(psha_filespec)
	#CAV_total_exceedances, CAV_deagg_exceedances = CAV_filtering_from_deagg_PGA(deagg_exceedances[period_index], magnitudes, intensities[period_index], vs30, cav_min)
	CAV_total_exceedances, CAV_deagg_exceedances = CAV_filtering_from_deagg_SA(deagg_exceedances, magnitudes, distances, intensities, periods, "AmbraseysEtAl1996GMPE", vs30, cav_min)

	pylab.semilogy(psha_result.intensities[period_index], psha_result.exceedance_means[0,period_index], 'b', lw=2, label="Original")
	#pylab.semilogy(intensities[period_index], CAV_total_exceedances, 'r', lw=2, label="CAV-filtered")
	pylab.semilogy(intensities[period_index], CAV_total_exceedances[period_index], 'r', lw=2, label="CAV-filtered")

	## Plot decoration
	pylab.xlabel("Acceleration (g)")
	pylab.ylabel("Exceedance rate (1/yr)")
	pylab.title("T = %s s" % psha_result.periods[period_index])
	pylab.grid(True)
	pylab.legend(loc=0)
	pylab.show()

	## Plot deaggregation
	from hazard.psha.plot import plot_deaggregation
	return_period = 1E5
	index = np.argmin(np.abs(psha_result.exceedance_means[period_index] - 1./return_period))
	print(index, intensities[period_index, index])
	plot_deaggregation(deagg_exceedances[period_index,index], magnitudes, distances, return_period)
	index = np.argmin(np.abs(CAV_total_exceedances - 1./return_period))
	print(index, intensities[period_index, index])
	plot_deaggregation(CAV_deagg_exceedances[index], magnitudes, distances, return_period)
	#plot_deaggregation(CAV_deagg_exceedances[period_index,index], magnitudes, distances, return_period)
	"""
