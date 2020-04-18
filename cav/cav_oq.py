"""
This module implements CAV (cumulative absolute velocity) filtering,
as outlined in Abrahamson et al. (2006) (EPRI technical paper 1014099),

Functionality is similar to cav_filtering.py, but optimized for use
in combination with OpenQuake:
- only the 2-step (duration-dependent) approach is implemented;
- all calculations are performed in log space
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np



__all__ = ['calc_cav_exceedance_prob', 'B1_FREQS', 'B1_WUS', 'B1_EUS']


## Calculate these logs only once
LN_1 = 0.
LN_4 = np.log(4)
LN_0_2 = np.log(0.2)
LN_0_025 = np.log(0.025)


## Correlation coefficients between PGA and SA (Table 3-1)
B1_FREQS = np.array([0.5, 1, 2.5, 5, 10, 20, 25, 35])
B1_WUS = np.array([0.59, 0.59, 0.6, 0.633, 0.787, 0.931, 0.956, 0.976])
B1_EUS = np.array([0.5, 0.55, 0.6, 0.75, 0.88, 0.9, 0.91, 0.93])


def calc_ln_cav(ln_pga, mag, vs30):
	"""
	Calculate natural logarithm of CAV using a two-step approach including
	duration dependence, based on an empirical model of CAV in the U.S.
	CAV has units of velocity (g.s)

	:param ln_pga:
		1-D numpy array, natural log of peak horizontal acceleration in g
	:param mag:
		float, moment magnitude
	:param vs30:
		1-D numpy array, shear-wave velocity over the top 30 m in m/s.
		Shape should be same as ln_pga.

	:return:
		tuple of (ln_CAV, sigma_ln_CAV)
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

	ln_vs30 = np.log(vs30)

	## Calculate uniform duration
	ln_dur_uni, sigma_ln_dur_uni = calc_ln_dur_uni(ln_pga, mag, ln_vs30)

	ln_CAV = np.zeros_like(ln_dur_uni)

	indexes_below = np.where(ln_pga <= LN_1)
	indexes_above = np.where(ln_pga > LN_1)

	ln_CAV[indexes_below] = (C0 + C1 * (mag - 6.5) + C2 * (mag - 6.5)**2
							+ C3 * ln_pga + C4 * ln_pga**2 + C5 * ln_pga**3
							+ C6 * ln_pga**4 + C7 * (ln_vs30 - 6)
							+ C8 * ln_dur_uni + C9 * ln_dur_uni**2)
	ln_CAV[indexes_above] = (C0 + C1 * (mag - 6.5) + C2 * (mag - 6.5)**2
							+ C3 * ln_pga + C7 * (ln_vs30 - 6)
							+ C8 * ln_dur_uni + C9 * ln_dur_uni**2)

	## Calculate standard deviation
	sigma_ln_CAV1 = np.zeros_like(ln_dur_uni)
	sigma_ln_CAV1[:] = 0.1
	idxs = np.where(ln_dur_uni <= LN_4)
	sigma_ln_CAV1[idxs] = 0.37 - 0.09 * (ln_dur_uni[idxs] - LN_0_2)
	sigma_ln_CAV1[ln_dur_uni < LN_0_2] = 0.37

	sigma_ln_CAV = np.sqrt((C8 + 2 * C9 * ln_dur_uni)**2 * sigma_ln_dur_uni**2
							+ sigma_ln_CAV1**2)

	return (ln_CAV, sigma_ln_CAV)


def calc_ln_dur_uni(ln_pga, mag, ln_vs30):
	"""
	Calculate the natural logarithm of the uniform duration (in s)
	above 0.025 g, based on an empirical model for the U.S.
	The uniform duration is the total time during which the absolute
	value of the acceleration time series exceeds a specified threshold
	(0.025 g in the case of CAV)

	:param ln_pga:
		1-D numpy array, natural log of peak horizontal acceleration in g
	:param mag:
		float, moment magnitude
	:param ln_vs30:
		1-D numpy array, natural log of shear-wave velocity over the top 30 m
		in m/s. Shape should be same as ln_pga.

	:return:
		tuple of (ln_dur_uni, sigma_ln_dur_uni)
		- ln_dur_uni: 1-D numpy array with same shape as ln_pga and vs30
		- sigma_ln_dur_uni: float
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

	ln_dur_uni = (a1 + a2 * ln_pga + a3 / (ln_pga + a4) + a5 * (mag - 6.5)
				+ a6 * (mag - 6.5)**2 + a7 * (ln_vs30 - 6))

	return (ln_dur_uni, sigma_ln_dur_uni)


def calc_cav_exceedance_prob(ln_pga, mag, vs30, cav_min=0.16):
	"""
	Calculate probability of exceeding a specified CAV value

	:param ln_pga:
		1-D numpy array, natural log of peak horizontal acceleration in g
	:param mag:
		float, moment magnitude
	:param vs30:
		1-D numpy array, shear-wave velocity over the top 30 m in m/s
		Shape should be same as ln_pga.
	:param cav_min:
		float, CAV threshold (default: 0.16 g.s)

	:return:
		1-D array of probability values with same shape as ln_pga and vs30
	"""
	if cav_min > 0:
		try:
			from ..c_speedups import norm
		except:
			#print("Failed importing norm speedup!")
			from scipy.stats import norm
		ln_pga = np.array(ln_pga)
		vs30 = np.array(vs30)
		if len(ln_pga) != len(vs30):
			raise Exception("Length of ln_pga and vs30 arrays should be the same!")

		non_zero_indexes = np.where(ln_pga >= LN_0_025)

		prob = np.zeros_like(ln_pga, 'd')

		ln_CAV, sigma_ln_CAV = calc_ln_cav(ln_pga, mag, vs30)

		epsilon_CAV = ((np.log(cav_min) - ln_CAV[non_zero_indexes])
						/ sigma_ln_CAV[non_zero_indexes])
		prob[non_zero_indexes] = 1.0 - norm.cdf(epsilon_CAV)
	else:
		prob = np.ones_like(ln_pga, 'd')

	return prob
