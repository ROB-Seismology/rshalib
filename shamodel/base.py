"""
:mod:`rshalib.shamodel.base` exports :class:`rshalib.shamodel.base.SHAModelBase`
"""

from __future__ import absolute_import, division, print_function, unicode_literals


from .. import oqhazlib

from ..site import GenericSiteModel, SoilSiteModel, REF_SOIL_PARAMS


class SHAModelBase(object):
	"""
	Base class for SHA models, holding common attributes and methods.
	"""

	def __init__(self, name,
				site_model, ref_soil_params=REF_SOIL_PARAMS,
				imt_periods={'PGA': [0]},
				truncation_level=3, integration_distance=200.):
		"""
		:param name:
			str, name for SHA model
		:param site_model:
			instance of :class:`rshalib.site.GenericSiteModel`
			or :class:`rshalib.site.SoilSiteModel`
		:param ref_soil_params:
			dict, value for each soil parameter of :class:`rshalib.site.SoilSite`
			Required if :param:`site_model` is generic, ignored otherwise
			(default: REF_SOIL_PARAMS)
		:param imt_periods:
			{str: list of floats} dict, mapping intensity measure types
			(e.g. "PGA", "SA", "PGV", "PGD") to periods in seconds
			Periods must be monotonically increasing or decreasing.
			(default: {'PGA': [0]}).
		:param truncation_level:
			float >= 0, truncation level of gsims in times standard deviation
			(default: 3.)
		:param integration_distance:
			float, defining integration distance in km
			(default: 200.).
		"""
		self.name = name

		assert isinstance(site_model, (GenericSiteModel, SoilSiteModel))
		assert isinstance(site_model, SoilSiteModel) or ref_soil_params
		self.site_model = site_model
		if isinstance(site_model, GenericSiteModel):
			self.ref_soil_params = ref_soil_params
		else:
			self.ref_soil_params = {}

		self.imt_periods = imt_periods

		self.truncation_level = truncation_level
		self.integration_distance = integration_distance

	@property
	def source_site_filter(self):
		if self.integration_distance:
			return oqhazlib.calc.filters.source_site_distance_filter(
													self.integration_distance)
		else:
			return oqhazlib.calc.filters.source_site_noop_filter

	@property
	def rupture_site_filter(self):
		if self.integration_distance:
			return oqhazlib.calc.filters.rupture_site_distance_filter(
													self.integration_distance)
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

	def get_imt_families(self):
		"""
		Return list of IMT families
		"""
		imts = self.imt_periods.keys()
		if "PGA" in imts and "SA" in imts:
			imts.remove("PGA")
		if "PGV" in imts and "SV" in imts:
			imts.remove("PGV")
		if "PGD" in imts and "SD" in imts:
			imts.remove("PGD")
		return imts

	def get_soil_site_model(self):
		"""
		If no soil site model is given one is created from sha site model with
		ref_soil_params. If one is given it is used if no sha site model is
		given, else the sites from the sha site model are extracted from it.

		:returns:
			instance of :class:`rshalib.site.SoilSiteModel`
		"""
		if isinstance(self.site_model, SoilSiteModel):
			## Extract soil sites corresponding to generic sites.
			## Not sure if this was ever used
			#if self.sha_site_model:
			#	return self.soil_site_model.filter(self.soil_site_model.mesh._geodetic_min_distance(self.sha_site_model, True))
			return self.site_model
		else:
			soil_site_model = self.site_model.to_soil_site_model(name=None,
											ref_soil_params=self.ref_soil_params)

	def get_sites(self):
		"""
		Get sites.

		:return:
			list with instance of :class:`SoilSite`
		"""
		return self.get_soil_site_model().get_sites()

	def get_generic_sites(self):
		"""
		Get generic sites.

		:return:
			list with instances of :class:`GenericSite`
		"""
		return self.get_soil_site_model().get_generic_sites()

	def _get_gsim(self, gsim_name):
		"""
		Fetch gsim

		:param gsim_name:
			str, name of ground shaking intensity model

		:return:
			instance of :class:`openquake.hazardlib.gsim.GroundShakingIntensityModel
		"""
		from openquake.hazardlib.gsim import get_available_gsims

		return get_available_gsims()[gsim_name]()



if __name__ == "__main__":
	"""
	"""
	pass

