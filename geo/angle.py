"""
Some functions handling angular calculations
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np



__all__ = ['mean_angle', 'delta_angle']


def mean_angle(angles, weights=None):
	"""
	Compute mean angle

	:param angles:
		List or numpy array with angles in degrees
	:param weights:
		List or numpy array with weights (default: None)
	"""
	rad_angles = np.radians(angles)
	## Mean Y component
	sa = np.average(np.sin(rad_angles), weights=weights)
	## Mean X component
	ca = np.average(np.cos(rad_angles), weights=weights)
	## Take the arctan of the averages, and convert to degrees,
	mean_angle = np.degrees(np.arctan2(sa, ca))
	return mean_angle


def delta_angle(angle1, angle2):
	"""
	Compute difference between two angles

	:param angle1:
		Float, first angle in degrees
	:param angle2:
		Float, second angle in degrees

	:return:
		Float, angle difference in degrees
	"""
	## Convert angles to unit vectors
	rad_angle1 = np.radians(angle1)
	rad_angle2 = np.radians(angle2)
	v1_x, v1_y = np.cos(rad_angle1), np.sin(rad_angle1)
	v2_x, v2_y = np.cos(rad_angle2), np.sin(rad_angle2)
	v1, v2 = np.array([v1_x, v1_y]), np.array([v2_x, v2_y])

	rad_delta = np.arccos(np.dot(v1, v2))
	if np.isnan(rad_delta):
		if (v1 == v2).all():
			rad_delta = 0.0
		else:
			rad_delta = np.pi
	## This is equivalent
	#rad_delta = np.arctan2(v2_y, v2_x) - np.arctan2(v1_y, v1_x)
	return np.degrees(rad_delta)

