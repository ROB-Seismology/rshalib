"""
Module containing various ROB extensions to nhlib
"""


import openquake.hazardlib as nhlib


class GroundMotionModel(object):
	"""
	Class defining a ground motion model, this is a mapping of tectonic region
	types to gsims (one gsim per tectonic region type).

	:param name:
		String, defining name of ground motion model.
	:param trts_gsims_map:
		dict, mapping tectonic region types (string) to gsims (string).
	"""

	def __init__(self, name, trts_gsims_map):
		"""
		"""
		self.name = name
		self.trts_gsims_map = trts_gsims_map

	def __getitem__(self, trt):
		"""
		Return gsim of tectonic region type.

		:param trt:
			String, defining tectinic region type to get gsim from.
		"""
		return self.trts_gsims_map[trt]


	def __str__(self):
		return self.name

	def to_nhlib_soil_site(self, vs30=760., vs30measured=False, z1pt0=100., z2pt5=2.):
		"""
		Convert to a nhlib.site.Site object, representing a site with
		soil characteristics
		"""
		return nhlib.site.Site(self, vs30, vs30measured, z1pt0, z2pt5)


if __name__ == '__main__':
	pass
