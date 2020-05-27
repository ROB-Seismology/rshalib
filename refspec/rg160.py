"""
RG1.60 reference spectrum
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from ..result import ResponseSpectrum


__all__ = ['get_rg160_rs']


def get_rg160_rs(ag, damping=5):
	"""
	Horizontal design response spectra according to
	US Atomic EnergyCommission Regulatory Guide 1.60

	:param ag:
		float, design ground acceleration (in g)
	:param damping:
		float, percent of critical damping (0.5, 2, 5, 7 or 10)
		Other damping values should be linearly interpolated (TO DO)
		(default: 5)

	:return:
		instance of :class:`rshalib.result.ResponseSpectrum`
	"""
	freqs = np.array([0.1, 0.25, 2.5, 9.0, 33.0, 50.0, np.inf])
	if damping == 0.5:
		values = np.array([1.75E-02,0.09999585,0.853846,0.7,0.14,0.14,0.14])
	elif damping == 2.0:
		values = np.array([1.34E-02,0.07989231,0.6060013,0.5,0.14,0.14,0.14])
	elif damping == 5.0:
		values = np.array([1.08E-02,0.06502016,0.46,0.376,0.14,0.14,0.14])
	elif damping == 7.0:
		values = np.array([0.01,0.05955598,0.3903749,0.3250497,0.14,0.14,0.14])
	elif damping == 10.0:
		values = np.array([0.01,0.0551229,0.324121,0.2726523,0.14,0.14,0.14])
	values *= (ag / 0.14)

	periods = 1.0 / freqs
	return ResponseSpectrum(periods, values, intensity_unit="g", imt="SA",
							model_name="RG1.60")
