"""
Ground-motion model
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import openquake.hazardlib as oqhazlib



__all__ = ['GroundMotionModel']


class GroundMotionModel(object):
	"""
	Class defining a ground motion model, this is a mapping of tectonic
	region types to gsims (one gsim per tectonic region type).
	Usually used together with an instance of :class:`SourceModel`
	in an instance of :class:`PSHAModel`

	:param name:
		str, defining name of ground motion model.
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

	def __repr__(self):
		return '<GroundMotionModel %s>' % self.name

	def get_optimized_model(self, source_model):
		"""
		Return an optimized ground-motion model where unused tectonic
		region types are removed

		:param source_model:
			instance of :class:`SourceModel`

		:return:
			instance of :class:`GroundMotionModel`
		"""
		optimized_trts_gsims_map = {}
		used_trts = set()
		for src in source_model.sources:
			used_trts.add(src.tectonic_region_type)
		for trt in used_trts:
			optimized_trts_gsims_map[trt] = self.trts_gsims_map[trt]
		return GroundMotionModel(self.name, optimized_trts_gsims_map)

	def to_ground_motion_system(self):
		"""
		Convert to a ground-motion logic tree

		:return:
			instance of :class:`GroundMotionSystem`
		"""
		from ..pmf import GMPEPMF
		from ..logictree import GroundMotionSystem
		gmpe_system_def = {}
		for trt, gsim in self.trts_gsims_map.items():
			gmpe_system_def[trt] = GMPEPMF([gsim], [1])
		return GroundMotionSystem(self.name, gmpe_system_def)



if __name__ == '__main__':
	pass
