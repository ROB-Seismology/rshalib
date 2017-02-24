
import numpy as np
import openquake.hazardlib as oqhazlib
import hazard.rshalib as rshalib



def calc_rupture_probability_from_ground_motion_thresholds(
	source_model,
	ground_motion_model,
	imt,
	pe_site_models,
	pe_thresholds,
	ne_site_models=[],
	ne_thresholds=[],
	truncation_level=3,
	integration_distance=300,
	strict_intersection=True,
	apply_rupture_probability=False
	):
	"""
	Compute rupture probabilities for potential seismic sources based
	on ground-motion (typically intensity) thresholds

	For each rupture, the probability is computed as the intersection of
	two probabilistic "events":
	- E1: event that ground-motion intensity exceeds a minimum level at sites
	of positive evidence
	- E2: event that ground-motion intensity does not exceed a maximum level
	at sites of negative evidence

	It is assumed that these 2 events are independent, in which case
	the probability P(E1E2) = p(E1).p(E2). If :param:`strict_intersection`
	is set to True, each site is considered separately, so the total number
	of events is equal to the number of sites with positive evidence times
	the number of sites with negative evidence.

	The objective is to infer which rupture has the highest probability
	of generating the observed distribution of sites with positive and
	negative evidence.

	:param source_model:
		instance of :class:`rshalib.source.SourceModel`
		Should contain either point or fault sources
	:param ground_motion_model:
		instance of :class:`rshalib.gsim.GroundMotionModel`
	:param imt:
		instance of :class:`openquake.hazardlib.imt.IMT`
	:param pe_site_models:
		list with instances of :class:`rshalib.site.SoilSiteModel`,
		sites with positive evidence
	:param pe_thresholds:
		float list or array, minimum ground-motion intensity for each
		positive evidence site model
	:param ne_site_models:
		list with instances of :class:`rshalib.site.SoilSiteModel`,
		sites with negative evidence
		(default: [])
	:param ne_thresholds:
		float list or array, maximum ground-motion intensity for each
		negative evidence site model
		(default: [])
	:param truncation_level:
		float, GMPE uncertainty truncation level
		(default: 3)
	:param integration_distance:
		float, integration distance in km
		Not taken into account yet!
		(default: 300)
	:param strict_intersection:
		bool, whether or not strict intersection should be applied for
		all sites in :param:`pe_sites` or :param:`ne_sites`. If False,
		the average probability in each will be used.
		(default: True)
	:param apply_rupture_probability:
		bool, whether or not to multiply probabilities of (non-)exceedance
		with the rupture probability
		(default: False)

	:return:
		dictionary mapping source IDs to lists of probabilities for each
		rupture in the source
	"""
	prob_dict = {}

	for src in source_model:
		prob_dict[src.source_id] = []
		trt = src.tectonic_region_type
		gsim_name = ground_motion_model[trt]
		gsim = oqhazlib.gsim.get_available_gsims()[gsim_name]()
		tom = oqhazlib.tom.PoissonTOM(1)

		for rupture in src.iter_ruptures(tom):
			pe_prob = 1
			for (pe_threshold, pe_site_model) in zip(pe_thresholds, pe_site_models):
				sctx, rctx, dctx = gsim.make_contexts(pe_site_model, rupture)
				pe_poes = gsim.get_poes(sctx, rctx, dctx, imt, pe_threshold, truncation_level)
				pe_poes = pe_poes[:,0]
				if strict_intersection:
					pe_prob *= np.prod(pe_poes)
				else:
					pe_prob *= np.mean(pe_poes)

			ne_prob = 1
			for (ne_threshold, ne_site_model) in zip(ne_thresholds, ne_site_models):
				sctx, rctx, dctx = gsim.make_contexts(ne_site_model, rupture)
				ne_poes = gsim.get_poes(sctx, rctx, dctx, imt, ne_threshold, truncation_level)
				ne_poes = ne_poes[:,0]
				if strict_intersection:
					ne_prob *= np.prod(1 - ne_poes)
				else:
					ne_prob *= np.mean(1 - ne_poes)

			total_prob = pe_prob * ne_prob
			if apply_rupture_probability:
				total_prob *= rupture.get_probability_one_or_more_occurrences()
			prob_dict[src.source_id].append(total_prob)

		prob_dict[src.source_id] = np.array(prob_dict[src.source_id])

	return prob_dict

