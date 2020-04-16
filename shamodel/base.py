"""
:mod:`rshalib.shamodel.base` exports :class:`rshalib.shamodel.base.SHAModelBase`
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from functools import partial

from .. import (oqhazlib, OQ_VERSION)


from ..site import GenericSiteModel, SoilSiteModel, REF_SOIL_PARAMS


class SHAModelBase(object):
	"""
	Base class for SHA models, holding common attributes and methods.
	"""

	def __init__(self, name,
				site_model, ref_soil_params=REF_SOIL_PARAMS,
				imt_periods={'PGA': [0]},
				truncation_level=3, integration_distance=200.,
				damping=0.05, intensity_unit=None):
		"""
		:param name:
			str, name for SHA model
		:param site_model:
			instance of :class:`rshalib.site.GenericSiteModel`
			or :class:`rshalib.site.SoilSiteModel`,
			sites where ground motions will be computed
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
		:param damping:
			damping for spectral ground motions
			(expressed as fraction of critical damping)
			(default: 0.05)
		:param intensity_unit:
			str, unit in which intensities are expressed.
			Set this only if you are not using OpenQuake!
			If None, the default intensity unit in OQ will be used
			(default: None)
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

		self.damping = damping
		if self.damping > 1:
			print('Converting damping from percent to fraction!')
			self.damping /= 100.

		self.intensity_unit = intensity_unit or self.get_default_oq_intensity_unit()

	@property
	def source_site_filter(self):
		if self.integration_distance:
			if OQ_VERSION >= '2.9.0':
				integration_distance = {}
				for trt in self.source_model.get_tectonic_region_types():
					integration_distance[trt] = self.integration_distance
				return partial(oqhazlib.calc.filters.SourceFilter,
								integration_distance=integration_distance)
			else:
				return oqhazlib.calc.filters.source_site_distance_filter(
													self.integration_distance)
		else:
			return oqhazlib.calc.filters.source_site_noop_filter

	@property
	def rupture_site_filter(self):
		if self.integration_distance:
			if OQ_VERSION >= '3.2.0':
				## Not used anymore
				return lambda rupture, sites: sites
			elif OQ_VERSION >= '2.9.0':
				return partial(oqhazlib.calc.filters.filter_sites_by_distance_to_rupture,
								integration_distance=self.integration_distance)
			else:
				return oqhazlib.calc.filters.rupture_site_distance_filter(
													self.integration_distance)
		else:
			return oqhazlib.calc.filters.source_site_noop_filter

	def filter_sites_by_rupture(self, rupture, gsim=None, sites=None):
		"""
		Filter sites that are within integration distance of a rupture
		Note that this works only for OpenQuake versions >= 2.9.0

		:param rupture:
			instance of :class:`oqhazlib.source.[Base]Rupture`
		:param gsim:
			instance of :class:`oqhazlib.gsim.GroundShakingIntensityModel`
			Ignored for OpenQuake versions < 3.2.0, where Joyner-Boore
			distance is always used
			(default: None)
		:param sites:
			instance of :class:`rshalib.site.SoilSiteModel`
			(default: None, will use :prop:`site_model`)

		:return:
			instance of :class:`rshalib.site.SoilSiteModel`
		"""
		if sites is None:
			sites = self.get_soil_site_model()

		if OQ_VERSION >= '3.2.0':
			from openquake.hazardlib.gsim.base import ContextMaker
			from openquake.hazardlib.calc.filters import IntegrationDistance

			trt = 'default'
			maximum_distance = {trt: [(rupture.mag, self.integration_distance)]}
			maximum_distance = IntegrationDistance(maximum_distance)
			if gsim is None:
				gsim_list = []
			else:
				gsim_list = [gsim]
			ctx_maker = ContextMaker(gsim_list, maximum_distance=maximum_distance)
			sites, dctx = ctx_maker.filter(sites, rupture)

		elif OQ_VERSION >= '2.9.0':
			sites = oqhazlib.calc.filters.filter_sites_by_distance_to_rupture(
									rupture, self.integration_distance, sites)

		else:
			pass

		return sites

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
		if im in ("PGA", "PGV", "PGD"):
			imt = getattr(oqhazlib.imt, im)()
		else:
			imt = getattr(oqhazlib.imt, im)(period, damping=self.damping*100)
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
		imts = list(self.imt_periods.keys())
		if "PGA" in imts and "SA" in imts:
			imts.remove("PGA")
		if "PGV" in imts and "SV" in imts:
			imts.remove("PGV")
		if "PGD" in imts and "SD" in imts:
			imts.remove("PGD")
		return imts

	def get_response_type(self):
		"""
		Determine response type from IMTs

		:return:
			str, one of 'acceleration', 'velocity' or 'displacement'
		"""
		imts = self.imt_periods.keys()
		if 'PGA' in imts or 'SA' in imts:
			response_type = 'acceleration'
		elif 'PGV' in imts or 'SV' in imts:
			response_type = 'velocity'
		elif 'PGD' in imts or 'SD' in imts:
			response_type = 'displacement'
		return response_type

	def get_default_oq_intensity_unit(self):
		"""
		Report default intensity used in OpenQuake

		:return:
			str
		"""
		# TODO: not sure if it is cm or m for velocity / displacement
		response_type = self.get_response_type()
		if response_type[:3] == 'acc':
			intensity_unit = 'g'
		elif response_type[:3] == 'vel':
			intensity_unit = 'cm/s'
		elif response_type[:3] == 'dis':
			intensity_unit = 'cm'
		return intensity_unit

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
			soil_site_model = self.site_model
		else:
			soil_site_model = self.site_model.to_soil_site_model(name=None,
											ref_soil_params=self.ref_soil_params)

		return soil_site_model

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
		from ..gsim import get_oq_gsim

		return get_oq_gsim(gsim_name)



if __name__ == "__main__":
	"""
	"""
	pass

