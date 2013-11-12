"""
:mod:`rshalib.shamodel.base` exports :class:`rshalib.shamodel.base.SHAModelBase`
"""


from hazard.rshalib.site import SHASiteModel, REF_SOIL_PARAMS


class SHAModelBase(object):
	"""
	Base class for SHA models, holding common attributes and methods.
	"""

	def __init__(self, name, sites=None, grid_outline=None, grid_spacing=None, soil_site_model=None, ref_soil_params=REF_SOIL_PARAMS, truncation_level=3):
		"""
		:param name:
			str, name for sha model
		:param sites:
			see :class:`rshalib.site.SHASiteModel` (default: None)
		:param grid_outline:
			see :class:`rshalib.site.SHASiteModel` (default: None)
		:param grid_spacing:
			see :class:`rshalib.site.SHASiteModel` (default: None)
		:param soil_site_model:
			instance of :class:`rshalib.site.SHASiteModel` (default: None)
		:param ref_soil_params:
			dict, value for each soil parameter of :class:`rshalib.site.SoilSite`
			(default: REF_SOIL_PARAMS)
		:param truncation_level:
			float, truncation level of gsims in times standard deviation
			(default: 3.)
		"""
		self.name = name
		if not sites and not grid_outline:
			assert soil_site_model
			self.sha_site_model = None
		else:
			self.sha_site_model = SHASiteModel(
				sites=sites,
				grid_outline=grid_outline,
				grid_spacing=grid_spacing,
				)
		self.soil_site_model = soil_site_model
		self.ref_soil_params = ref_soil_params
		self.truncation_level = truncation_level

	@property
	def sites(self):
		return self.sha_site_model.get_sites()

	@property
	def grid_outline(self):
		return self.sha_site_model.grid_outline

	@property
	def grid_spacing(self):
		return self.sha_site_model.grid_spacing

	def get_soil_site_model(self):
		"""
		If no soil site model is given one is created from sha site model with
		ref_soil_params. If one is given it is used if no sha site model is
		given, else the sites from the sha site model are extracted from it.

		:returns:
			instance of :class:`rshalib.site.SoilSiteModel`
		"""
		if self.soil_site_model:
			if self.sha_site_model:
				return self.soil_site_model.filter(self.soil_site_model.mesh._geodetic_min_distance(self.sha_site_model, True))
			else:
				return self.soil_site_model
		else:
			return self.sha_site_model.to_soil_site_model(self.ref_soil_params)


if __name__ == "__main__":
	"""
	"""
	pass

