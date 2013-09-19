"""
:mod: `rshalib.shamodel.base` defines :class:`rshalib.shamodel.base.SHAModelBase`.
"""


from hazard.rshalib.site import SHASiteModel, REF_SOIL_PARAMS


class SHAModelBase(object):
	"""
	"""
	
	def __init__(self, sites=None, grid_outline=None, grid_spacing=None, soil_site_model=None, ref_soil_params=REF_SOIL_PARAMS):
		"""
		"""
		self.site_model = SHASiteModel(sites=sites, grid_outline=grid_outline, grid_spacing=grid_spacing)
		self.soil_site_model = soil_site_model
		self.ref_soil_params = ref_soil_params
	
	def get_soil_site_model(self):
		"""
		"""
		if self.soil_site_model:
			return self.soil_site_model.filter(self.soil.site_model.mesh._geodetic_min_distance(self.site_model, True))
		else:
			return self.site_model.to_soil_site_model(self.ref_soil_params)


if __name__ == "__main__":
	"""
	"""
	pass

