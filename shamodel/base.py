"""
:mod:`rshalib.shamodel.base` exports :class:`rshalib.shamodel.base.SHAModelBase`
"""


import openquake.hazardlib as oqhazlib

from ..site import SHASiteModel, REF_SOIL_PARAMS


class SHAModelBase(object):
	"""
	Base class for SHA models, holding common attributes and methods.
	"""

	def __init__(self, name, sites=None, grid_outline=None, grid_spacing=None, soil_site_model=None, ref_soil_params=REF_SOIL_PARAMS, imt_periods={'PGA': [0]}, truncation_level=3, integration_distance=200.):
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
		:param imt_periods:
			{str: list of floats} dict, mapping intensity measure types (e.g. "PGA", "SA", "PGV", "PGD") to periods in seconds (default: {'PGA': [0]}).
			Periods must be monotonically increasing or decreasing.
		:param truncation_level:
			float, truncation level of gsims in times standard deviation
			(default: 3.)
		:param integration_distance:
			Float, defining integration distance in km (default: 200.).
		"""
		self.name = name
		if not sites and not grid_outline:
			assert soil_site_model
			self.sha_site_model = None
		else:
			self._set_sha_sites(sites, grid_outline, grid_spacing)
		self.soil_site_model = soil_site_model
		self.ref_soil_params = ref_soil_params
		self.imt_periods = imt_periods
		self.truncation_level = truncation_level
		self.integration_distance = integration_distance

	@property
	def source_site_filter(self):
		if self.integration_distance:
			return oqhazlib.calc.filters.source_site_distance_filter(self.integration_distance)
		else:
			return oqhazlib.calc.filters.source_site_noop_filter

	@property
	def rupture_site_filter(self):
		if self.integration_distance:
			return oqhazlib.calc.filters.rupture_site_distance_filter(self.integration_distance)
		else:
			return oqhazlib.calc.filters.rupture_site_noop_filter

	def _construct_imt(self, im, period):
		"""
		Construct IMT object from intensity measure and period.

		:param im:
			str, intensity measure, e.g. "PGA", "SA"
		:param period:
			float, spectral period

		:return:
			instance of :class:`IMT`
		"""
		if im == "SA":
			imt = getattr(oqhazlib.imt, im)(period, damping=5.)
		else:
			imt = getattr(oqhazlib.imt, im)()
		return imt

	def _get_imts(self):
		"""
		Construct ordered list of IMT objects
		"""
		imts = []
		for im, periods in sorted(self.imt_periods.items()):
			for period in periods:
				imts.append(self._construct_imt(im, period))
		return imts

	def _get_periods(self):
		"""
		Return list of periods corresponding to ordered list of IMT objects
		"""
		periods = []
		for imt in self._get_imts():
			try:
				periods.append(imt.period)
			except AttributeError:
				periods.append(0)
		return periods

	def _set_sha_sites(self, sites, grid_outline, grid_spacing):
		"""
		Set SHA sites from list of sites or grid outline and grid spacing.
		Note: Use this method only if :param:`soil_site_model` is None.

		:param sites:
			list with instances of class:`SHASite`
		:param grid_outline:
			(lon_min, lon_max, lat_min, lat_max) tuple
		:param grid_spacing:
			float, grid spacing
		"""
		self.sites = sites
		self.grid_outline = grid_outline
		self.grid_spacing = grid_spacing
		self.sha_site_model = SHASiteModel(
				sites=sites,
				grid_outline=grid_outline,
				grid_spacing=grid_spacing,
				)

	def get_sites(self):
		"""
		Get sites.

		:return:
			list with instance of :class:`SoilSite`
		"""
		return self.get_soil_site_model().get_sites()

	def get_sha_sites(self):
		"""
		Get SHA sites.

		:return:
			list with instances of :class:`SHASite`
		"""
		return self.get_soil_site_model().get_sha_sites()

	#@property
	#def grid_outline(self):
	#	if self.sha_site_model:
	#		return self.sha_site_model.grid_outline

	#@property
	#def grid_spacing(self):
	#	if self.sha_site_model:
	#		return self.sha_site_model.grid_spacing

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
			return self.sha_site_model.to_soil_site_model(ref_soil_params=self.ref_soil_params)


if __name__ == "__main__":
	"""
	"""
	pass

