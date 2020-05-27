"""
Reference spectra (RG1.60, Eurocode 8)
"""

import numpy as np

from ..result import ResponseSpectrum


def get_refspec_RG160(ag, damping=5):
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
	return ResponseSpectrum("RG1.60", periods, "SA", values, intensity_unit="g")


def get_refspec_EC8(ag, ground_type, resp_type, orientation="horizontal", damping=5):
	"""
	Reference horizontal elastic response spectrum according to Eurocode 8
	(if deep geology is not accounted for)

	:param ag:
		float, design ground acceleration on type A ground (in g)
	:param ground_type:
		str, one of the ground types defined in Eurocode 8: A, B, C, D or E
	:param resp_type:
		int, response spectrum type: 1 or 2
		1: if earthquakes contributing most to seismic hazard for the site
			have MS >= 5.5
		2: if earthquakes contributing most to seismic hazard for the site
			have MS < 5.5
	:param orientation:
		str, either "horizontal" or "vertical", orientation of spectrum
		(default: "horizontal")
	:param damping:
		float, viscous damping, in % (default: 5)

	:return:
		instance of :class:`rshalib.result.ResponseSpectrum`
	"""
	ground_type = ground_type.upper()
	if resp_type == 1:
		if orientation == "horizontal":
			S = {'A': 1.0, 'B': 1.2, 'C': 1.15, 'D': 1.35, 'E': 1.4}[ground_type]
			TB = {'A': 0.15, 'B': 0.15, 'C': 0.2, 'D': 0.2, 'E': 0.15}[ground_type]
			TC = {'A': 0.4, 'B': 0.5, 'C': 0.6, 'D': 0.8, 'E': 0.5}[ground_type]
			TD = {'A': 2.0, 'B': 2.0, 'C': 2.0, 'D': 2.0, 'E': 2.0}[ground_type]
		elif orientation == "vertical":
			vh_ratio = 0.9
			TB, TC, TD = 0.05, 0.15, 1.  ## Is this correct for type 1 ??
	elif resp_type == 2:
		if orientation == "horizontal":
			S = {'A': 1.0, 'B': 1.35, 'C': 1.5, 'D': 1.8, 'E': 1.6}[ground_type]
			TB = {'A': 0.05, 'B': 0.05, 'C': 0.1, 'D': 0.1, 'E': 0.05}[ground_type]
			TC = {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.30, 'E': 0.25}[ground_type]
			TD = {'A': 1.2, 'B': 1.2, 'C': 1.2, 'D': 1.2, 'E': 1.2}[ground_type]
		elif orientation == "vertical":
			vh_ratio = 0.45
			TB, TC, TD = 0.05, 0.15, 1.


	eta = np.sqrt(10.0 / (5 + damping))

	periods = np.concatenate([[0.0], np.arange(0.04, 4.02, 0.02)])
	values = np.zeros(len(periods), 'd')

	if orientation == "horizontal":
		for i, T in enumerate(periods):
			if 0 <= T < TB:
				values[i] = ag * S * (1 + (T/TB)*(eta * 2.5 - 1))
			elif TB <= T < TC:
				values[i] = ag * S * eta * 2.5
			elif TC <= T < TD:
				values[i] = ag * S * eta * 2.5 * (TC/T)
			elif T >= TD:
				values[i] = ag * S * eta * 2.5 * ((TC*TD)/T**2)
	elif orientation == "vertical":
		assert ground_type.upper() == "A"
		avg = ag * vh_ratio
		for i, T in enumerate(periods):
			if 0 <= T < TB:
				values[i] = avg * (1 + (T/TB)*(eta * 3 - 1))
			elif TB <= T < TC:
				values[i] = avg * eta * 3
			elif TC <= T < TD:
				values[i] = avg * eta * 3 * (TC/T)
			elif T >= TD:
				values[i] = avg * eta * 3 * ((TC*TD)/T**2)

	if orientation == "horizontal":
		model_name = "EC8 - class %s, type %d" % (ground_type, resp_type)
	else:
		model_name = "EC8 type %d, vertical" % resp_type
	return ResponseSpectrum(model_name, periods, "SA", values, intensity_unit="g")



if __name__ == "__main__":
	pass
