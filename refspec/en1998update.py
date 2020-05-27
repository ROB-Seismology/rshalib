"""
2019 update of EN1998
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np

from ..result import ResponseSpectrum


def parse_consequence_class(consequence_class):
	"""
	Make sure consequence class is in the right format, e.g.
	- 2 or '2' --> 'CC2'
	- 'CC3-a' --> 'CC3a'
	- 'CC4' --> 'CC3b'

	:param consequenc_class:
		str, consequence class

	:return:
		str, corrected consequence class
	"""
	if isinstance(consequence_class, int) or consequence_class[0] != 'C':
		consequence_class = 'CC%s' % str(consequence_class)
	consequence_class = consequence_class.replace('-', '').upper()
	if consequence_class == 'CC4':
		consequence_class == 'CC3b'
	return consequence_class


CC_DELTA_DICT = {'CC1': 0.6, 'CC2': 1.0, 'CC3a': 1.25, 'CC3b': 1.6}

LS_CC_TR_DICT = {'NC': {'CC1': 800, 'CC2': 1600, 'CC3a': 2500, 'CC3b': 5000},
				'SD': {'CC1': 250, 'CC2': 475, 'CC3a': 800, 'CC3b': 1600},
				'DL': {'CC1': 50, 'CC2': 60, 'CC3a': 60, 'CC3b': 100}}

LS_CC_GAMMA_DICT = {'NC': {'CC1': 1.2, 'CC2': 1.5, 'CC3a': 1.8, 'CC3b': 2.2},
				'SD': {'CC1': 0.8, 'CC2': 1., 'CC3a': 1.2, 'CC3b': 1.5},
				'DL': {'CC1': 0.4, 'CC2': 0.5, 'CC3a': 0.5, 'CC3b': 0.6}}

REF_RETURN_PERIOD = 475


def get_delta(consequence_class='CC2'):
	"""
	Coefficient depending on consequence class

	:param consequence_class:
		str, one of 'CC1', 'CC2', 'CC3a', 'CC3b', ...
		(default: 'CC2')

	:return:
		float, delta value
	"""
	consequence_class = parse_consequence_class(consequence_class)
	delta = CC_DELTA_DICT[consequence_class]
	return delta


def get_return_period(consequence_class='CC2', limit_state='SD'):
	"""
	Return period of seismic action for different limit states
	(the state of damage under a given seismic action)
	and consequence classes

	:param consequence_class:
		str, one of 'CC1', 'CC2', 'CC3a', 'CC3b', ...
		(default: 'CC2')
	:param limit_state:
		str, one of 'NC' (near collapse), 'SD' (significant damage)
		or 'DL' (damage limitation)
		(default: 'SD')

	:return:
		int, return period (in years)
	"""
	consequence_class = parse_consequence_class(consequence_class)
	Tr = LS_CC_TR_DICT[limit_state.upper()][consequence_class]
	return Tr


def get_performance_factor(consequence_class='CC2', limit_state='SD'):
	"""
	Return period of seismic action for different limit states
	and consequence classes

	:param consequence_class:
		str, one of 'CC1', 'CC2', 'CC3a', 'CC3b', ...
		(default: 'CC2')
	:param limit_state:
		str, one of 'NC' (near collapse), 'SD' (significant damage)
		or 'DL' (damage limitation)
		(default: 'SD')

	:return:
		int, return period (in years)
	"""
	consequence_class = parse_consequence_class(consequence_class)
	gamma = LS_CC_GAMMA_DICT[limit_state.upper()][consequence_class]
	return gamma

get_gamma = get_performance_factor


def calc_seismicity_index(Salpha_ref, delta):
	"""
	Compute seismicity index Sdelta from Salpha_ref and delta

	:param Salpha_ref:
		float, maximum response spectral acceleration (5% damping)
		on site category A for the reference return period of 475 yr
		(in m/s2)
	:param delta:
		float, coefficient that depends on the consequence class
		of the considered structure

	:return:
		float, Sdelta (in m/s2)
	"""
	return delta * Salpha_ref


def get_seismicity_class(Sdelta):
	"""
	Determine seismicity class from seismicity index
	- For very low seismicity class, EN 1998 may be neglected;
	- For low seismicity class, performance requirements may be satisfied
	through the application of rules simpler than given in EN 1998

	:param Sdelta:
		float, seismicity index (in m/s2)

	:return:
		str, seismicity class (very low / low / moderate / high)
	"""
	if Sdelta < 1.:
		seismicity_class = 'very low'
	elif 1. <= Sdelta < 2.5:
		seismicity_class = 'low'
	elif 2.5 <= Sdelta < 5.:
		seismicity_class = 'moderate'
	else:
		seismicity_class = 'high'
	return seismicity_class


def get_seismicity_level(Salpha_ref):
	"""
	Determine seismicity level of a territory based on Salpha_ref.
	This corresponds to the seismicity class for consequence class CC2

	:param Salpha_ref:
		float, maximum response spectral acceleration (5% damping)
		on site category A for the reference return period of 475 yr
		(in m/s2)

	:return:
		str, seismicity level (very low / low / moderate / high)
	"""
	return get_seismicity_class(Salpha_ref)


def get_ground_class(VsH):
	"""
	Determine ground class from VsH

	:param VsH:
		float, equivalent value of the shear-wave velocity of the
		deposit above H800 (in m/s)

	:return:
		str, ground class (rock / stiff / medium /soft / very soft) or None
	"""
	if VsH >= 800:
		ground_class = 'rock'
	elif 800 > VsH >= 400:
		ground_class = 'stiff'
	elif 400 > VsH >= 250:
		ground_class = 'medium'
	elif 250 > VsH >= 150:
		ground_class = 'soft'
	elif 150 > VsH >= 100:
		ground_class = 'very soft'
	else:
		ground_class = None
	return ground_class


def get_depth_class(H800):
	"""
	Determine depth class from H800

	:param H800:
		float, depth of bedrock formation (in m) identified by Vs > 800 m/s
		(may be limited to maximum of 30 m)

	:return:
		str, depth class (very shallow / shallow / intermediate / deep)
	"""
	if H800 <= 5:
		depth_class = 'very shallow'
	elif 5 < H800 <= 30:
		depth_class = 'shallow'
	elif 30 < H800 <= 100:
		depth_class = 'intermediate'
	else:
		depth_class = 'deep'
	return depth_class


def get_site_category(H800, VsH):
	"""
	Determine site category based on H800 and VsH

	:param H800:
		float, depth of bedrock formation (in m) identified by Vs > 800 m/s
		(may be limited to maximum of 30 m)
	:param VsH:
		float, equivalent value of the shear-wave velocity of the
		deposit above H800 (in m/s)

	:return:
		str, site category (A / B / C / D / E / F) or None
	"""
	depth_class = get_depth_class(H800)
	ground_class = get_ground_class(VsH)
	if ground_class == 'rock':
		return 'A'
	if depth_class == 'very shallow':
		site_category = {'stiff': 'A', 'medium': 'A', 'soft': 'E'}.get(ground_class, 'F')
	elif depth_class == 'shallow':
		site_category = {'stiff': 'B', 'medium': 'E', 'soft': 'E'}.get(ground_class, 'F')
	elif depth_class == 'intermediate':
		site_category = {'stiff': 'B', 'medium': 'C', 'soft': 'D'}.get(ground_class, 'F')
	elif depth_class == 'deep':
		site_category = {'stiff': 'B', 'medium': 'F', 'soft': 'F'}.get(ground_class, 'F')
	return site_category


def estimate_Sbeta_ref(Salpha_ref, delta):
	"""
	Estimate Sbeta_ref from Salpha_ref and delta.
	Sbeta_ref is the spectral acceleration (5% damping)
	at vibration period T = 1s on site category A
	for the reference return period.

	:param Salpha_ref:
		float, maximum response spectral acceleration (5% damping)
		on site category A for the reference return period (in m/s2)
	:param delta:
		float, coefficient that depends on the consequence class
		of the considered structure

	:return:
		float, Sbeta_ref (in m/s2)
	"""
	Sdelta = calc_seismicity_index(Salpha_ref, delta)
	seismicity_class = get_seismicity_class(Sdelta)
	fh = {'very low': 0.2, 'low': 0.2, 'moderate': 0.3, 'high': 0.4}[seismicity_class]
	Sbeta_ref = fh * Salpha_ref
	return Sbeta_ref


def estimate_Salpha_rp(Salpha_ref, consequence_class='CC2', limit_state='SD'):
	"""
	Estimate maximum spectral acceleration for specific limit state
	and consequence class from Salpha_ref and the performance factor,
	as alternative to the value calculated for the return period
	associated to that limit state and consequence class
	(see :func:`get_return_period`)

	:param Salpha_ref:
		float, maximum response spectral acceleration (5% damping)
		on site category A for the reference return period (in m/s2)
	:param consequence_class:
	:param limit_state:
		see :func:`get_performance_factor`

	:return:
		float, max. spectral acceleration (in m/s2)
	"""
	gamma = get_performance_factor(consequence_class, limit_state)
	Salpha_rp = Salpha_ref * gamma
	return Salpha_rp


def estimate_Sbeta_rp(Sref, ref_is_beta, consequence_class='CC2', limit_state='SD'):
	"""
	Estimate 1-s spectral acceleration for specific limit state
	and consequence class from either Salpha_ref or Sbeta_ref
	and the performance factor,
	as alternative to the value calculated for the return period
	associated to that limit state and consequence class
	(see :func:`get_return_period`)

	:param Sref:
		float, maximum or 1-s response spectral acceleration (5% damping)
		on site category A for the reference return period (in m/s2)
	:param ref_is_beta:
		bool, whether or not :param:`S_ref` corresponds to Salpha_ref
		(False) or Sbeta_ref (False)
	:param consequence_class:
	:param limit_state:
		see :func:`get_performance_factor`

	:return:
		float, 1-s spectral acceleration (in m/s2)
	"""
	if ref_is_beta is False:
		delta = get_delta(consequence_class)
		Sbeta_ref = estimate_Sbeta_ref(Sref, delta)
	elif ref_is_beta is True:
		Sbeta_ref = Sref
	Sbeta_rp = estimate_Salpha_rp(Sbeta_ref, consequence_class, limit_state)
	return Sbeta_rp


def calc_FA(Salpha_rp, S0):
	"""
	Compute FA, the ratio of Salpha with respect to zero-period
	acceleration (PGA)

	:param Salpha_rp:
		float, maximum response spectral acceleration (5% damping)
		on site category A (in m/s2)
	:param S0:
		float, zero-period response spectral acceleration
		on site category A (in m/s2)

	:return:
		float, FA
	"""
	return Salpha_rp / S0


def calc_TB(TC, CHI=4.):
	"""
	Determine the lower corner period of the constant spectral
	acceleration range

	:param TC:
		upper corner period (in s)
	:param CHI:
		float, width of constant acceleration range of spectrum
		(default: 4.)

	:return:
		float, TC (in s)
	"""
	TB = TC / CHI
	TB = np.maximum(TB, 0.05)
	TB = np.minimum(TB, 0.1)
	return TB


def calc_TC(Salpha, Sbeta):
	"""
	Determine the upper corner period of the constant spectral
	acceleration range

	:param Salpha:
		float, maximum response spectral acceleration (5% damping)
		(in m/s2)
	:param Sbeta:
		float, response spectral acceleration (5% damping)
		at vibration period T = 1 s (in m/s2)

	:return:
		float, TC (in s)
	"""
	Tbeta = 1.
	TC = (Sbeta * Tbeta) / Salpha
	return TC


def calc_TD(Sbeta_rp):
	"""
	Determine the corner period at the beginning of the constant
	displacement response range of the spectrum

	:param Sbeta_rp:
		float, response spectral acceleration (5% damping)
		at vibration period T = 1 s on site category A
		for particular return period (in m/s2)

	:return:
		float, TD (in s)
	"""
	if np.isscalar(Sbeta_rp):
		if Sbeta_rp <= 1:
			TD = 2.
		else:
			TD = 1 + Sbeta_rp
	else:
		TD = np.ones_like(Sbeta_rp) * 2
		idxs = Sbeta_rp > 1
		TD[idxs] = 1 + Sbeta_rp[idxs]
	return TD


def estimate_site_amplification_factors(site_category):
	"""
	Estimate site amplification factors from site category.
	If site_category is unspecified, the default values (1.7, 4.)
	will be returned.

	:param site_category:
		str, site category

	:return:
		(Falpha, Fbeta) tuple:
		- Falpha: maximum amplification factor
		- Fbeta: amplification factor at vibration period T = 1s
	"""
	Falpha = {'A': 1., 'B': 1.2, 'C': 1.35, 'D': 1.5, 'E': 1.7, 'F': 1.35}.get(
															site_category, 1.7)
	Fbeta = {'A': 1., 'B': 1.6, 'C': 2.25, 'D': 3.2, 'E': 3., 'F': 4.}.get(
															site_category, 4.)
	return (Falpha, Fbeta)


def calc_site_amplification_factors(site_category, H800=0, VsH=0, Salpha_rp=0, Sbeta_rp=0):
	"""
	Determine site amplification factors from continuous functions
	depending on site category and parameters H800 and VsH,
	as well as Salpha_rp and Sbeta_rp.

	:param site_category:
		str, site category
	:param H800:
		float, depth of bedrock formation (in m) identified by Vs > 800 m/s
		(may be limited to maximum of 30 m)
		(default: 0)
	:param VsH:
		float, equivalent value of the shear-wave velocity of the
		deposit above H800 (in m/s)
		(default: 0)
	:param Salpha_rp:
		float, maximum response spectral acceleration (5% damping)
		on site category A for particular return period (in m/s2)
		(default: 0)
	:param Sbeta_rp:
		float, response spectral acceleration (5% damping)
		at vibration period T = 1 s on site category A
		for particular return period (in m/s2)
		(default: 0)

	:return:
		(Falpha, Fbeta) tuple:
		- Falpha: maximum amplification factor
		- Fbeta: amplification factor at vibration period T = 1s
	"""
	if site_category == 'A':
		Falpha = Fbeta = 1.
	else:
		assert H800 and VsH
		VsH_800 = VsH / 800.
		ralpha = 1. - 2E+3 * (Salpha_rp / VsH**2)
		rbeta = 1. - 2E+3 * (Sbeta_rp / VsH**2)
		if site_category in ('B', 'C', 'D'):
			Falpha = VsH_800**(-0.25*ralpha)
			Fbeta = VsH_800**(-0.7*rbeta)
		elif site_category == 'E':
			Falpha = VsH_800**(-0.25*ralpha*(H800/30.)*(4-H800/10.))
			Fbeta = VsH_800**(-0.7*rbeta*(H800/30.))
		elif site_category == 'F':
			Falpha = 0.9 * VsH_800**(-0.25*ralpha)
			Fbeta = 1.25 * VsH_800**(-0.7*rbeta)

	return (Falpha, Fbeta)


def get_topography_amplification_factor(slope_angle):
	"""
	Determine topography amplification factor based on slope angle

	:param slope_angle:
		float, slope angle (in degrees)

	:return:
		float, amplification factor
	"""
	## Note: this is too simplistic
	Ftopo = 1.
	if slope_angle > 30:
		Ftopo = 1.4
	elif 15 < slope_angle <= 30:
		Ftopo = 1.2
	return Ftopo


def calc_eta(ksi, periods, TC):
	"""
	Determine damping correction factor eta

	:param ksi:
		float, damping ratio of structure, expressed as % of critical
	:param periods:
		float array, vibration periods
	:param TC:
		float, upper corner period of the constant spectral
		acceleration range

	:return:
		float array, eta values
	"""
	eta = np.zeros_like(periods)
	eta[periods > 0.5] = np.sqrt(10.0 / (5 + ksi))
	idxs = (periods <= 0.5)
	eta[idxs] = np.sqrt((10 + (TC * (ksi - 5.)) / (TC + 30*periods[idxs]))
						/ (5 + ksi))
	return eta


def estimate_fvh_alpha(Salpha):
	"""
	Estimate vertical/horizontal ratio for max. spectral acceleration

	:param Salpha:
		float, maximum spectral acceleration (5 % damping) (in m/s2)

	:return:
		float, V/H ratio
	"""
	if Salpha < 2.5:
		fvh_alpha = 0.6
	elif 2.5 <= Salpha < 7.5:
		fvh_alpha = 0.04 * Salpha + 0.5
	else:
		fvh_alpha = 0.8
	return fvh_alpha


def construct_acceleration_rs(Salpha_rp, Sbeta_rp,
						site_category, H800=0, VsH=0, Ftopo = 1.,
						consequence_class='CC2', limit_state='SD', Salpha_ref=None,
						TA=0.02, CHI=4., FA=2.5, S0=None, TD=None,
						periods=None, orientation="horizontal",
						damping=5.):
	"""
	Construct elastic acceleration response spectrum

	:param Salpha_rp:
		float, maximum response spectral acceleration (5% damping)
		on site category A for particular return period (in m/s2)
	:param Sbeta_rp:
		float, response spectral acceleration (5% damping)
		at vibration period T = 1 s on site category A
		for particular return period (in m/s2)
		If None, will be estimated from :param:`Salpha_rp` and :param:`delta`
	:param site_category:
		str, site category
	:param H800:
	:param VsH:
		see :func:`get_site_amplification_factors`
		optional parameters to compute site amplification factors
		more accurately
		(default: 0)
	:param Ftopo:
		float, topography amplification factor
		(default: 1.)
	:param consequence_class:
		str, consequence class of the considered structure.
		If different from 'CC2', it is assumed that :param:`Salpha_rp`
		and :param:`Sbeta_rp` correspond to Salpha_ref and Sbeta_ref!
		(default: 'CC2')
	:param limit_state:
		str, considered limit state of the structure
		If different from 'SD', it is assumed that :param:`Salpha_rp`
		and :param:`Sbeta_rp` correspond to Salpha_ref and Sbeta_ref!
		(default: 'SD')
	:param Salpha_ref:
		float, maximum response spectral acceleration (5% damping)
		on site category A for the reference return period (in m/s2)
		Optional parameter in case :param:`Sbeta_rp` is None AND
		:param:`Salpha_rp` does not correspond to Salpha_ref (i.e.
		consequence class and limit state are not 'CC2' and 'SD').
		If not given in that case, Salpha_ref will be estimated
		from Salpha_rp (inverse of :func:`estimate_Salpha_rp`
		(default: None)
	:param TA:
		float, short-period cut-off associated to the zero-period
		acceleration (in s)
		(default: 0.02)
	:param CHI:
		float, width of constant acceleration range of spectrum
		(default: 4.)
	:param FA:
		float, ratio of Salpha_rp with respect to the zero-period
		acceleration. Will be ignored if :param:`S0` is specified
		(default: 2.5)
	:param S0:
		float, zero-period acceleration on category A (in m/s2) for same
		return period as :param:`Salpha_rp`. Will be used to override
		:param:`FA`
		(default: None)
	:param TD:
		float, corner period at the beginning of the constant
		displacement response range of the spectrum (in s)
		If not specified, will be determined automatically from
		:param:`Sbeta_rp`
		(default: None)
	:param periods:
		float array, spectral periods
		(default: None)
	:param orientation:
		str, either "horizontal" or "vertical", orientation of spectrum
		(default: "horizontal")
	:param damping:
		float, viscous damping, in % (default: 5)

	:return:
		instance of :class:`rshalib.result.ResponseSpectrum`
	"""
	## Estimate Salpha_rp from Salpha_ref if necessary
	if (consequence_class, limit_state) != ('CC2', 'SD'):
		print("Assuming Salpha_rp/Sbeta_rp correspond to Salpha_ref/Sbeta_ref!")
		Salpha_ref = Salpha_rp
		Salpha_rp = estimate_Salpha_rp(Salpha_ref, consequence_class, limit_state)
		Sbeta_ref = Sbeta_rp
		if Sbeta_ref is None:
			Sbeta_rp = estimate_Sbeta_rp(Salpha_ref, False, consequence_class, limit_state)
		else:
			Sbeta_rp = estimate_Sbeta_rp(Sbeta_ref, True, consequence_class, limit_state)

	## Estimate Sbeta_rp if necessary
	if Sbeta_rp is None:
		if Salpha_ref is None:
			print("Estimating Salpha_ref from Salpha_rp!")
			gamma = get_performance_factor(consequence_class, limit_state)
			Salpha_ref = Salpha_rp / gamma
		Sbeta_rp = estimate_Sbeta_rp(Salpha_ref, False, consequence_class, limit_state)

	Tbeta = 1.

	## Determine site amplification factors
	if H800 and VsH:
		Falpha, Fbeta = calc_site_amplification_factors(site_category, H800, VsH,
													Salpha_rp, Sbeta_rp)
	else:
		Falpha, Fbeta = estimate_site_amplification_factors(site_category)
	Salpha = Salpha_rp * Falpha * Ftopo
	Sbeta = Sbeta_rp * Fbeta * Ftopo

	## Vertical / horizontal ratio
	if orientation == "vertical":
		fvh_alpha = estimate_fvh_alpha(Salpha)
		fvh_beta = 0.6
		Salpha *= fvh_alpha
		Sbeta *= fvh_beta

	if not S0 is None:
		FA = calc_FA(Salpha_rp, S0)

	## Determine corner periods
	TC = calc_TC(Salpha, Sbeta)
	if TC >= 1:
		print("Note: TC >= 1: A specific study should be carried out!")
	if orientation == "horizontal":
		TB = calc_TB(TC, CHI)
	else:
		TB = 0.05
	if not TD:
		TD = calc_TD(Sbeta_rp)

	if periods is None:
		periods = [0, TA, TB, TC, TD]
		TE, TF = 6, 10
		if TD < TE:
			periods.append(TE)
		periods.append(TF)
	periods = np.asarray(periods)

	eta = calc_eta(damping, periods, TC)

	## Construct spectrum
	Se = np.zeros_like(periods)
	Se[(0 <= periods) & (periods < TA)] = Salpha / FA
	idxs = (TA <= periods) & (periods < TB)
	Se[idxs] = (Salpha / (TB - TA)) * (eta[idxs] * (periods[idxs] - TA) + (TB - periods[idxs])/FA)
	idxs = (TB <= periods) & (periods < TC)
	Se[idxs] = eta[idxs] * Salpha
	idxs = (TC <= periods) & (periods < TD)
	Se[idxs] = eta[idxs] * Sbeta * Tbeta / periods[idxs]
	idxs = (periods >= TD)
	Se[idxs] = eta[idxs] * TD * Sbeta * Tbeta / periods[idxs]**2

	model_name = "EN1998 horizontal acceleration response spectrum (site category %s)"
	model_name %= site_category
	return ResponseSpectrum(periods, Se, intensity_unit="ms2", imt="SA",
							model_name=model_name)


def calc_long_period_site_amplification_factor(VsH):
	"""
	Calculate long-period site amplification factor from VsH

	:param VsH:
		float, equivalent value of the shear-wave velocity of the
		deposit above H800 (in m/s)

	:return:
		float
	"""
	Flp = VsH / 800.
	return VsH


def estimate_long_period_site_amplification_factor(Fbeta, site_category):
	"""
	Estimate long-period site amplification factor in case VsH
	is not available

	:param Fbeta:
		float, amplification factor at vibration period T = 1s
	:param site_category:
		str, site category

	:return:
		float
	"""
	if site_category == 'A':
		Flp = Fbeta
	elif site_category in ('B', 'E'):
		Flp = 0.9 * Fbeta
	elif site_category in ('C', 'F'):
		Flp = 0.75 * Fbeta
	elif site_category == 'D':
		Flp = 0.6 * Fbeta
	return Flp


def construct_displacement_rs(Salpha_rp, Sbeta_rp, site_category, periods, **kwargs):
	"""
	Construct elastic displacement response spectrum

	See :func:`get_acceleration_rs` for parameters

	:return:
		instance of :class:`rshalib.result.ResponseSpectrum`
	"""
	Se = construct_acceleration_rs(Salpha_rp, Sbeta_rp, site_category,
										periods=periods, **kwargs)
	Se = Se.values
	SDe = np.zeros_like(Se)
	if not TD in kwargs:
		TD = calc_TD(Sbeta_rp)
	TE = max(TD, 6)
	TF = 10

	if kwargs.get(VsH):
		Flp = calc_long_period_site_amplification_factor(VsH)
	else:
		_, Fbeta = estimate_site_amplification_factors(site_category)
		Flp = estimate_long_period_site_amplification_factor(Fbeta, site_category)

	idxs = (periods <= TE)
	SDe[idxs] = Se[idxs] * (periods[idxs] / 2 * np.pi)**2
	if TE in periods:
		[SDe_TE] = SDe[periods == TE]
	else:
		SDe_TE = construct_acceleration_rs(Salpha_rp, Sbeta_rp, site_category,
											periods=[TE], **kwargs)
	idxs = (TE < periods) & (periods <= TF)
	SDe[idxs] = SDe_TE * (1 + ((Flp / Fbeta) - 1) * (periods[idxs] - TE) / (TF - TE))
	SDe[periods > TF] = SDe_TE * Flp / Fbeta

	model_name = "EN1998 displacement response spectrum (site category %s)"
	model_name %= site_category
	return ResponseSpectrum(periods, SDe, intensity_unit="m", imt="SA",
							model_name=model_name)


def calc_design_pga(Salpha, FA=2.5):
	"""
	Calculate design PGA

	:param Salpha:
		float, maximum response spectral acceleration (5% damping)
		(in m/s2)
	:param FA:
		float, ratio of Salpha_rp with respect to the zero-period
		acceleration
		(default: 2.5)

	:return:
		float, design PGA (in m/s2)
	"""
	return Salpha / FA


def calc_design_pgv(Salpha, Sbeta):
	"""
	Calculate design PGV

	:param Salpha:
		float, maximum response spectral acceleration (5% damping)
		(in m/s2)
	:param Sbeta:
		float, response spectral acceleration (5% damping)
		at vibration period T = 1 s (in m/s2)

	:return:
		float, design PGB (in m/s)
	"""
	return 0.06 * (Salpha * Sbeta)**0.55


def calc_design_pgd(Sbeta_rp, Flp, TD=None):
	"""
	Calculate design PGD

	:param Sbeta_rp:
		float, response spectral acceleration (5% damping)
		at vibration period T = 1 s on site category A
		for particular return period (in m/s2)
	:param Flp:
		float, long-period amplification factor
	:param TD:
		float, corner period at the beginning of the constant
		displacement response range of the spectrum (in s)
		If not specified, will be determined automatically from
		:param:`Sbeta_rp`
		(default: None)

	:return:
		float, design PGD (in m)
	"""
	Tbeta = 1.
	if not TD:
		TD = calc_TD(Sbeta_rp)
	return 0.025 * Tbeta * TD * Flp * Sbeta_rp


def estimate_magnitude(Sbeta_rp):
	"""
	Get convential earthquake magnitude (in terms of moment magnitude)
	associated to the elastic response spectrum

	:param Sbeta_rp:
		float, response spectral acceleration (5% damping)
		at vibration period T = 1 s on site category A
		for particular return period (in m/s2)

	:return:
		float, MW
	"""
	if Sbeta_rp < 0.08:
		MW = 4.5
	elif Sbeta_rp <= Sbeta_rp < 0.2:
		MW = 5.0
	elif 0.2 <= Sbeta_rp < 0.5:
		MW = 5.5
	elif 0.5 <= Sbeta_rp < 1.2:
		MW = 6.0
	elif 1.2 <= Sbeta_rp < 2.5:
		MW = 6.5
	elif 2.5 <= Sbeta_rp < 4.0:
		MW = 7.0
	else:
		MW = 7.5
	return MW


def estimate_duration_on_rock(Sbeta_rp, site_category='A'):
	"""
	Get corresponding duration of the strong part of the ground motion
	on rock, associated to the elastic response spectrum

	:param Sbeta_rp:
		float, response spectral acceleration (5% damping)
		at vibration period T = 1 s on site category A
		for particular return period (in m/s2)
	:param site_category:
		str, site category
		(default: 'A')

	:return:
		float, duration (in s)
	"""
	if Sbeta_rp < 0.08:
		DR = 0.5
	elif Sbeta_rp <= Sbeta_rp < 0.2:
		DR = 1.
	elif 0.2 <= Sbeta_rp < 0.5:
		DR = 2.
	elif 0.5 <= Sbeta_rp < 1.2:
		DR = 4.
	elif 1.2 <= Sbeta_rp < 2.5:
		DR = 8.
	elif 2.5 <= Sbeta_rp < 4.0:
		DR = 16.
	else:
		DR = 32.

	if site_category == 'A':
		pass
	elif site_category in ('B', 'C'):
		DR *= 1.2
	else:
		DR *= 1.5

	return DR


def is_site_specific_study_required():
	# TODO: determine which parameters are required
	"""
	seismicity class == 'high', and site category == 'D' or ('E' or 'F'
		with VsH  < 250 m/s
	site_category is None
	TC >= 1
	:return:
		bool
	"""
	raise NotImplementedError



## Ideas:
## - Compare modeled Sbeta_ref with estimated Sbeta_ref
## - Compute FA and CHI, and compare with standard values of 2.5 and 4.0

## TODO:
## - Map Salpha (and optionally Sbeta, FA) for Tref and other return periods
## Plot site amplification factors in function of VsH and/or H800
