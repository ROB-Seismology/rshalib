"""
Gridded source model
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import int

import numpy as np
from ..geo import (Point, NodalPlane)
from ..mfd import EvenlyDiscretizedMFD
from ..pmf import (NodalPlaneDistribution, HypocentralDepthDistribution)
from .point import PointSource
from .source_model import SourceModel


# TODO: Needs to be developed further


class SimpleUniformGridSourceModel():
	"""
	Uniformly gridded (in spherical coordinates) source model
	"""
	def __init__(self, grid_outline, grid_spacing,
				min_mag, max_mag, mag_bin_width, depth, strike, dip, rake,
				magnitude_scaling_relationship="WC1994",
				upper_seismogenic_depth=0, lower_seismogenic_depth=None,
				rupture_mesh_spacing=2.5, rupture_aspect_ratio=1,
				tectonic_region_type="TRT"):
		"""
		"""
		self.grid_outline = grid_outline
		if isinstance(grid_spacing, (int, float)):
			grid_spacing = (grid_spacing, grid_spacing)
		self.grid_spacing = grid_spacing
		self.create_grid()
		self.min_mag = min_mag
		self.max_mag = max_mag
		self.mag_bin_width = mag_bin_width
		self.depth = depth
		self.strike = strike
		self.dip = dip
		self.rake = rake
		self.magnitude_scaling_relationship = magnitude_scaling_relationship
		self.upper_seismogenic_depth = upper_seismogenic_depth
		if not lower_seismogenic_depth:
			lower_seismogenic_depth = self.depth + 5.
		self.lower_seismogenic_depth = lower_seismogenic_depth
		self.rupture_mesh_spacing = rupture_mesh_spacing
		self.rupture_aspect_ratio = rupture_aspect_ratio
		self.tectonic_region_type = tectonic_region_type

	@property
	def trt(self):
		return self.tectonic_region_type

	@property
	def rms(self):
		return self.rupture_mesh_spacing

	@property
	def msr(self):
		return self.magnitude_scaling_relationship

	@property
	def rar(self):
		return self.rupture_aspect_ratio

	@property
	def usd(self):
		return self.upper_seismogenic_depth

	@property
	def lsd(self):
		return self.lower_seismogenic_depth

	def create_grid(self):
		lon_min, lon_max, lat_min, lat_max = self.grid_outline
		dlon, dlat = self.grid_spacing
		self.lons = np.arange(lon_min, lon_max + dlon, dlon)
		self.lats = np.arange(lat_min, lat_max + dlat, dlat)
		self.shape = (len(self.lats), len(self.lons))
		self.lon_grid, self.lat_grid = np.meshgrid(self.lons, self.lats)

	def create_mfd(self):
		dM = self.mag_bin_width
		num_mags = int(round((self.max_mag - self.min_mag) / dM))
		occurrence_freqs = np.ones(num_mags)/float(num_mags)
		mfd = EvenlyDiscretizedMFD(self.min_mag + dM/2, dM,
					occurrence_freqs)
		return mfd

	def create_npd(self):
		nopl = NodalPlane(self.strike, self.dip, self.rake)
		npd = NodalPlaneDistribution([nopl], [1])
		return npd

	def create_hdd(self):
		hdd = HypocentralDepthDistribution([self.depth], [1])
		return hdd

	def create_point_source(self, lon, lat, id=""):
		name = "%.2f, %.2f" % (lon, lat)
		point = Point(lon, lat)
		mfd = self.create_mfd()
		npd = self.create_npd()
		hdd = self.create_hdd()

		source = PointSource(id, name, self.trt, mfd, self.rms,
						self.msr, self.rar, self.usd, self.lsd, point, npd, hdd)

		return source

	def __iter__(self):
		i = 0
		for lat in lats:
			for lon in lons:
				source = self.create_point_source(lon, lat, id=i)
				i += 1
				yield source

	def get_source_model(self):
		sources = list(self.__iter__)
		return SourceModel("Grid", sources)

	def get_mesh_index(self, sequential_idx):
		return np.unravel_index(sequential_idx, self.shape)

