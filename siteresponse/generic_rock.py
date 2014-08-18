"""
Host-to-target adjustments based on generic rock profiles
according to Cotton et al. (2006)
"""

import numpy as np



## Anchoring depths in generic rock models of Boore & Joyner (1997)
Za = np.array([1., 30., 190., 4000., 8000.])


## Rock models of Boore & Joyner (1997) (Table 3)
BJ1997_WNA_Vs = np.array([336., 850., 1800., 3300., 3500.])
BJ1997_ENA_Vs = np.array([2768., 2791., 2914., 3570., 3600.])


## Vs0 and p0 values for several Vs30 and depth ranges
## (Table 4)
## Vs0 and p0 are dictionaries:
##   - keys: Vs30
##   - values: arrays with Vs0 and p0 values for depth segments
##             between the anchoring depths Za (starting from the surface)
## Note: This table contains several errors, which have been corrected below
Vs0, p0 = {}, {}
Vs0[600] = np.array([232.48, 322.12, 830.03, 1782.74, 3294.81, 3498.03])
Vs0[900] = np.array([444.54, 560.72, 1134.52, 2023.40, 3363.63, 3524.02])
Vs0[1200] = np.array([705.59, 835.99, 1421.03, 2216.69, 3414.11, 3542.88])
Vs0[1500] = np.array([1011.13, 1143.56, 1695.57, 2381.15, 3454.23, 3557.74])
#Vs0[1800] = np.array([1358, 1480.52, 1961.31, 2525.86, 2487.66, 2815.80])
Vs0[1800] = np.array([1358, 1480.52, 1961.31, 2525.86, 3488.66, 3570.40])
Vs0[2100] = np.array([1744.22, 1844.71, 2220.23, 2656.00, 3516.38, 3580.54])
# Note: Vs0[2400] is identical to Vs0[2700], and is probably incorrect
#Vs0[2400] = np.array([2627.11, 2648.54, 2722.44, 2884.78, 3564.13, 3597.88])
Vs0[2400] = np.array([2185.66, 2231.14, 2471.53, 2773.95, 3541.41, 3589.65])
Vs0[2700] = np.array([2627.11, 2648.54, 2722.44, 2884.78, 3564.13, 3597.88])

p0[600] = np.array([0., 0.278, 0.414, 0.202, 0.086, 0.])
p0[900] = np.array([0., 0.207, 0.313, 0.167, 0.067, 0])
p0[1200] = np.array([0., 0.156, 0.241, 0.142, 0.05, 0.])
p0[1500] = np.array([0., 0.116, 0.184, 0.122, 0.043, 0.])
p0[1800] = np.array([0., 0.082, 0.137, 0.106, 0.034, 0.])
#p0[2100] = np.array([0., 0.054, 0.0970, 0.009, 0.026, 0.])
p0[2100] = np.array([0., 0.054, 0.0970, 0.092, 0.026, 0.])
# Note: p0[2400] is identical to p0[2700]
#p0[2400] = np.array([0., 0.008, 0.003, 0.069, 0.014, 0.])
p0[2400] = np.array([0., 0.031, 0.0635, 0.0805, 0.02, 0.])
#p0[2700] = np.array([0., 0.008, 0.003, 0.069, 0.014, 0.])
p0[2700] = np.array([0., 0.008, 0.03, 0.069, 0.014, 0.])

Vs0_surface = [Vs0[vs30][0] for vs30 in range(600, 2800, 300)]


def calc_Ifrac(vs30):
	"""
	Calculate interpolation fraction for a chosen vs30 rock velocity
	with respect to vs30 of the Californian (WNA) rock model (620 m/s)
	and the ENA rock model (2800 m/s) of  Boore & Joyner (1997)

	:param vs30:
		float, average shear-wave velocity in the upper 30 m (m/s)

	:return:
		float, interpolation fraction
	"""
	## Equation 1 in Cotton et al. (2006)
	vs30_WNA, vs30_ENA = 620., 2800.
	return (np.log10(vs30) - np.log10(vs30_WNA)) / (np.log10(vs30_ENA) - np.log10(vs30_WNA))


def calc_generic_Vs_anchors(vs30):
	"""
	Calculate generic shear-wave velocities for a chosen vs30 rock velocity
	with respect to the WNA and ENA rock models of Boore & Joyner (1997)
	at the anchoring depths Za

	:param vs30:
		float, average shear-wave velocity in the upper 30 m (m/s)

	:return:
		1-D float array
	"""
	## Equation 2 in Cotton et al. (2006)
	Ifrac = calc_Ifrac(vs30)
	generic_Vs = np.zeros_like(Za)
	Vs1 = BJ1997_WNA_Vs
	Vs2 = BJ1997_ENA_Vs
	generic_Vs = 10 ** (Ifrac * (np.log10(Vs2) - np.log10(Vs1)) + np.log10(Vs1))
	return generic_Vs


def calc_generic_Vs_profile(vs30, z_ar, method="powerlaw"):
	"""
	Calculate generic shear-wave velocity profile at depth z according to a
	power-law model which goes through the anchoring depths Za

	:param vs30:
		float, average shear-wave velocity in the upper 30 m (m/s)
		Must be between 600 and 2700 m/s
	:param z_ar:
		1-D float array, depths (m)
	:param method:
		str, calculation method, either "interpolation" or "powerlaw"
		- interpolation: compute VS by cubic-spline interpolation
			between VS at anchoring depths
		- powerlaw_original: compute VS according to the power-law model in
			Cotton et al. (2006)
			Note: this doesn't seem to work, there may be an error
			in the publication
		- powerlaw: modified version of powerlaw_original, where Vs0 and p0
			are computed rather than taken from Table 4 in the paper
		(default: "powerlaw")

	:return:
		1-D float array, shear-wave velocities (m/s)
	"""
	from ..utils import interpolate

	if vs30 < 600 or vs30 > 2700:
		raise Exception("vs30 should be between 600 m/s and 2700 m/s")

	Vs_anchors = calc_generic_Vs_anchors(vs30)
	[Vs00] = interpolate(range(600, 2800, 300), Vs0_surface, [vs30])

	if method == "interpolation":
		## scipy interpolation methods do not perform as well as intcubicspline
		#from scipy.interpolate import interp1d
		#interpolator = interp1d(Za, Vs_anchors, bounds_error=False, kind="slinear")
		#Vs_profile = interpolator(z_ar)

		import geosurvey.cwp as cwp
		Vs_profile = cwp.intcubicspline(Za, Vs_anchors, z_ar, ideriv=0, method="cmonot")

		## Override Vs for z = 0
		Vs_profile[z_ar == 0] = Vs00
		Vs_profile[z_ar > Za[-1]] = Vs_anchors[-1]

	elif method == "powerlaw_original":
		## Equation 3 in Cotton et al.(2006)
		idxs = np.digitize(z_ar, Za, right=True)
		Vs_profile = Vs0[vs30][idxs] * (z_ar / Za[idxs-1]) ** p0[vs30][idxs]

	elif method == "powerlaw":
		## Determine Vs0 and p0 rather than using table of Cotton et al.
		## which contains several errors
		## This also allows to compute profiles for vs30 not appearing in this table
		Vs0_vs30 = np.concatenate([[Vs00], Vs_anchors])
		Vs_profile = np.zeros_like(z_ar, 'd')
		for zi, z in enumerate(z_ar):
			[idx] = np.digitize([z], Za, right=True)
			if idx >= 1:
				z0 = Za[idx-1]
			else:
				z0 = 0

			try:
				z1 = Za[idx]
			except IndexError:
				z1 = None

			if z0 and z1:
				p0_vs30 = np.log(Vs0_vs30[idx+1] / Vs0_vs30[idx]) / np.log(z1/z0)
			else:
				p0_vs30 = 0

			if z0:
				Vs = Vs0_vs30[idx] * (z/z0) ** p0_vs30
			else:
				Vs = Vs00
			Vs_profile[zi] = Vs

	return Vs_profile


def build_generic_rock_profile(vs30, num_depths=100):
	"""
	:param vs30:
		float, average shear-wave velocity in the upper 30 m (m/s)
		Must be between 600 and 2700 m/s
	:param num_depths:
		int, number of depths in profile (default: 100)

	:return:
		instance of :class:`ElasticContinuousModel`
	"""
	from transfer1D import ElasticContinuousModel
	from ..utils import logrange

	## Ignore division warnings
	np.seterr(divide='ignore', invalid='ignore')

	Z = logrange(Za[0], Za[-1], num_depths-1)
	Z = np.insert(Z, 0, [0])

	VS = calc_generic_Vs_profile(vs30, Z, method="powerlaw")
	Rho = np.zeros_like(VS)
	QS = 1./np.zeros_like(VS)

	rock_profile = ElasticContinuousModel(Z, VS, Rho, QS)

	return rock_profile


def get_host_to_target_adjustment(freqs, host_vs30, host_kappa, target_vs30, target_kappa):
	"""
	Compute host-to-target adjustment factor

	:param freqs:
		1-D float array, frequencies for which adjustment factor
		needs to be computed:
	:param host_vs30:
		float, vs30 of host rock (in m/s)
	:param host_kappa:
		float, kappa of host rock (in s)
	:param target_vs30:
		float, vs30 of target rock (in m/s)
	:param target_kappa:
		float, kappa of target rock (in s)

	:return:
		instance of :class:`TransferFunction` representing target/host amplification
	"""
	from tf import TransferFunction

	if host_vs30 != target_vs30:
		vel_source, rho_source = 3570., 2.8
		host_rock_profile = build_generic_rock_profile(host_vs30)
		target_rock_profile = build_generic_rock_profile(target_vs30)
		host_tf, _ = host_rock_profile.site_amp(freqs, vel_source=vel_source, rho_source=rho_source, kappa=host_kappa)
		target_tf, _ = target_rock_profile.site_amp(freqs, vel_source=vel_source, rho_source=rho_source, kappa=target_kappa)
		amps = target_tf.magnitudes / host_tf.magnitudes
	else:
		#host_amps = np.exp(-np.pi * host_kappa * freqs)
		#target_amps = np.exp(-np.pi * target_kappa * freqs)
		#amps = target_amps / host_amps
		amps = np.exp(-np.pi * (target_kappa - host_kappa) * freqs)

#		from ..utils import interpolate

#		freq_anchors = np.array([0.5, 1.0, 2.0, 5.0, 10, 20])
#		C10 = np.array([-1.540, -3.156, -6.056, -13.76, -23.37, -31.83])
#		C10_PGA = -18.64
#		C11 = np.array([-38.64, -24.59, -5.412, 26.11, 98.78, 378.67])
#		C11_PGA = 301.22

#		c10 = interpolate(freq_anchors, C10, freqs)
#		c10[np.isinf(freqs)] = C10_PGA
#		c10[freqs >= 100] = C10_PGA
#		c11 = interpolate(freq_anchors, C11, freqs)
#		c11[np.isinf(freqs)] = C11_PGA
#		c11[freqs >= 100] = C11_PGA

#		amps = np.exp(c10 * (target_kappa - host_kappa) + c11 * ((target_kappa - 0.025)**2 - (host_kappa - 0.025)**2))

	return TransferFunction(freqs, amps)



if __name__ == "__main__":
	import pylab
	import hazard.rshalib as rshalib

	## Plot generic rock profiles
	colors = ["blue", "cyan", "green", "yellow", "orange", "red", "magenta", "purple"]

	for i, vs30 in enumerate(range(600, 2800, 300)):
		print vs30

		generic_Vs_anchors = rshalib.siteresponse.calc_generic_Vs_anchors(vs30)
		print generic_Vs_anchors

		z_ar = np.arange(8001)

		Vs_profile = rshalib.siteresponse.calc_generic_Vs_profile(vs30, z_ar, method="powerlaw")

		pylab.plot(Vs_profile, z_ar, '%s' % colors[i], lw=2, label="Vs30=%d m/s" % vs30)
		pylab.plot(generic_Vs_anchors, Za, 'o', color='%s' % colors[i], lw=2, label="_nolegend_")

	pylab.xlabel("Vs (m/s)")
	pylab.ylabel("Depth (m)")
	ax = pylab.gca()
	ax.set_ylim(ax.get_ylim()[::-1])
	ax.set_ylim((5000,0))

	pylab.legend(loc=3)
	pylab.grid(True)
	pylab.title("Generic rock profiles")
	pylab.show()


	## Plot SITE_AMP response
	vel_source, rho_source = 3500., 2.8
	for i, vs30 in enumerate(range(600, 2800, 300)):
		print vs30

		freqs = np.linspace(0.05, 50, 100)
		rock_profile = rshalib.siteresponse.build_generic_rock_profile(vs30)
		tf, _ = rock_profile.site_amp(freqs)
		pylab.semilogx(tf.freqs, tf.magnitudes, '%s' % colors[i], lw=2, label="Vs30=%d m/s" % vs30)

	pylab.xlabel("Frequency (Hz)")
	pylab.ylabel("Amplification")
	pylab.legend(loc=2)
	pylab.grid(True)
	pylab.title("SITE_AMP response wrt VS=%s m/s, Rho=%s kg/dm3" % (vel_source, rho_source))
	pylab.show()


	## Plot host-to-target adjustment
	freqs = np.linspace(0.05, 50, 100)

	host_vs30, host_kappa = 2700, 0.006
	target1_vs30, target1_kappa = 1700, 0.01
	target2_vs30, target2_kappa = target1_vs30, host_kappa
	target3_vs30, target3_kappa = host_vs30, target1_kappa

	htt1 = rshalib.siteresponse.get_host_to_target_adjustment(freqs, host_vs30, host_kappa, target1_vs30, target1_kappa)
	htt2 = rshalib.siteresponse.get_host_to_target_adjustment(freqs, host_vs30, host_kappa, target2_vs30, target2_kappa)
	htt3 = rshalib.siteresponse.get_host_to_target_adjustment(freqs, host_vs30, host_kappa, target3_vs30, target3_kappa)
	htt4 = rshalib.siteresponse.TransferFunction(freqs, htt2.magnitudes * htt3.magnitudes)

	pylab.semilogx(freqs, htt1.magnitudes, lw=3, label="1) VS30=%s m/s, kappa=%s s" % (target1_vs30, target1_kappa))
	pylab.semilogx(freqs, htt2.magnitudes, lw=2, label="2) VS30=%s m/s, kappa=%s s" % (target2_vs30, target2_kappa))
	pylab.semilogx(freqs, htt3.magnitudes, lw=2, label="3) VS30=%s m/s, kappa=%s s" % (target3_vs30, target3_kappa))
	pylab.semilogx(freqs, htt4.magnitudes, lw=1, label="4) 2 * 3")

	xmin, xmax, ymin, ymax = pylab.axis()
	pylab.axis((xmin, xmax, 0.4, ymax))
	pylab.xlabel("Frequency (Hz)")
	pylab.ylabel("Amplification")
	pylab.legend(loc=3)
	pylab.grid(True)
	pylab.title("Host-to-target adjustments wrt VS30=%s m/s, kappa=%s s" % (host_vs30, host_kappa))
	pylab.show()
