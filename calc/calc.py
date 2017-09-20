
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
	integration_distance_dict={},
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
		instance of :class:`rshalib.gsim.GroundMotionModel` or dict
		mapping TRTs (string) to instances of :class:`rshalib.pmf.GMPEPMF`
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
	:param integration_distance_dict:
		dict, mapping GMPE names to (min, max) tuples of the
		integration distance in km, to allow GMPE-dependent distance
		filtering. If not specified, no distance filtering is applied
		(default: {})
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
		rupture in the source. Note that probabilities are reduced to
		single-site probabilities by raising them to a power of 1 over
		the total number of sites
	"""
	# TODO: take into account spatial correlation (but there is no model for MMI) ?

	prob_dict = {}

	for s, src in enumerate(source_model):
		prob_dict[src.source_id] = []
		trt = src.tectonic_region_type
		gsim_model = ground_motion_model[trt]
		if isinstance(gsim_model, rshalib.pmf.GMPEPMF):
			gsim_names, gsim_weights = gsim_model.gmpe_names, gsim_model.weights
		elif isinstance(gsim_model, (str, unicode)):
			gsim_names, gsim_weights = [gsim_model], [1.]
		if s == 0:
			gsims_dict = {gsim_name: oqhazlib.gsim.get_available_gsims()[gsim_name]() for gsim_name in gsim_names}

		tom = oqhazlib.tom.PoissonTOM(1)
		for rupture in src.iter_ruptures(tom):
			## Filter sites by distance
			filtered_pe_site_models, filtered_ne_site_models = {}, {}
			filtered_pe_thresholds, filtered_ne_thresholds = {}, {}
			num_sites = {}
			for gsim_name in gsim_names:
				num_sites[gsim_name] = 0
				if gsim_name in integration_distance_dict:
					min_dist, max_dist = integration_distance_dict[gsim_name]
				else:
					min_dist, max_dist = None, None
				filtered_pe_site_models[gsim_name] = []
				filtered_pe_thresholds[gsim_name] = []
				for p, pe_site_model in enumerate(pe_site_models):
					pe_mask = np.ones(len(pe_site_model))
					if min_dist or max_dist:
						## Filtering required
						jb_dist = rupture.surface.get_joyner_boore_distance(pe_site_model.mesh)
						if min_dist:
							pe_mask *= (min_dist <= jb_dist)
						if max_dist:
							pe_mask *= (jb_dist <= max_dist)
					filtered_pe_site_model = pe_site_model.filter(pe_mask)
					if filtered_pe_site_model:
						filtered_pe_thresholds[gsim_name].append(pe_thresholds[p])
						filtered_pe_site_models[gsim_name].append(filtered_pe_site_model)
						if strict_intersection:
							num_sites[gsim_name] += len(filtered_pe_site_model)
						else:
							num_sites[gsim_name] += 1
				filtered_ne_site_models[gsim_name] = []
				filtered_ne_thresholds[gsim_name] = []
				for n, ne_site_model in enumerate(ne_site_models):
					ne_mask = np.ones(len(ne_site_model))
					if min_dist or max_dist:
						## Filtering required
						jb_dist = rupture.surface.get_joyner_boore_distance(ne_site_model.mesh)
						if min_dist:
							ne_mask *= (min_dist <= jb_dist)
						if max_dist:
							ne_mask *= (jb_dist <= max_dist)
					filtered_ne_site_model = ne_site_model.filter(ne_mask)
					if filtered_ne_site_model:
						filtered_ne_thresholds[gsim_name].append(ne_thresholds[n])
						filtered_ne_site_models[gsim_name].append(filtered_ne_site_model)
						if strict_intersection:
							num_sites[gsim_name] += len(filtered_ne_site_model)
						else:
							num_sites[gsim_name] += 1

			filtered_gsim_names = [gsim_name for gsim_name in gsim_names if num_sites[gsim_name]]
			filtered_gsim_weights = [float(gsim_weights[i]) for i in range(len(gsim_names)) if gsim_names[i] in filtered_gsim_names]
			filtered_gsim_weights = np.array(filtered_gsim_weights)
			filtered_gsim_weights /= np.sum(filtered_gsim_weights)

			rupture_prob = 0
			for gsim_name, gsim_weight in zip(filtered_gsim_names, filtered_gsim_weights):
				gsim = gsims_dict[gsim_name]
				pe_prob = 1

				for (pe_threshold, pe_site_model) in zip(filtered_pe_thresholds[gsim_name], filtered_pe_site_models[gsim_name]):
					sctx, rctx, dctx = gsim.make_contexts(pe_site_model, rupture)
					pe_poes = gsim.get_poes(sctx, rctx, dctx, imt, pe_threshold, truncation_level)
					pe_poes = pe_poes[:,0]
					#print pe_poes
					if strict_intersection:
						pe_prob *= np.prod(pe_poes)
					else:
						pe_prob *= np.mean(pe_poes)

				ne_prob = 1
				for (ne_threshold, ne_site_model) in zip(filtered_ne_thresholds[gsim_name], filtered_ne_site_models[gsim_name]):
					sctx, rctx, dctx = gsim.make_contexts(ne_site_model, rupture)
					ne_poes = gsim.get_poes(sctx, rctx, dctx, imt, ne_threshold, truncation_level)
					ne_poes = ne_poes[:,0]
					if strict_intersection:
						ne_prob *= np.prod(1 - ne_poes)
					else:
						ne_prob *= np.mean(1 - ne_poes)

				## Combine positive and negative evidence probabilities
				#print pe_prob, ne_prob
				total_prob = pe_prob * ne_prob

				## Reduce to single-site probability
				total_prob **= (1./num_sites[gsim_name])
				#total_prob /= (0.5 ** (num_sites[gsim_name]))

				if apply_rupture_probability:
					total_prob *= rupture.get_probability_one_or_more_occurrences()

				rupture_prob += (total_prob * gsim_weight)

			prob_dict[src.source_id].append(rupture_prob)

		prob_dict[src.source_id] = np.array(prob_dict[src.source_id])

	return prob_dict

