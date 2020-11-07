"""
Eurocode 8 reference spectra
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from ..result import ResponseSpectrum


__all__ = ['get_ec8_rs']


def get_ec8_rs(ag, ground_type, resp_type, orientation="horizontal",
					damping=5):
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
		float, viscous damping, in %
		(default: 5)

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

	return ResponseSpectrum(periods, values, intensity_unit="g", imt="SA",
							model_name=model_name)



if __name__ == "__main__":
	pass