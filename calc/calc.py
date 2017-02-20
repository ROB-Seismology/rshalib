
import numpy as np
import openquake.hazardlib as oqhazlib
import hazard.rshalib as rshalib



def calc_rupture_probability_from_ground_motion_thresholds(
	source_model,
	ground_motion_model,
	imt,
	pe_sites,
	pe_threshold,
	ne_sites=None,
	ne_threshold=None,
	truncation_level=3,
	integration_distance=300,
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
	the probability P(E1E2) = p(E1).p(E2). Note that each site is considered
	separately, so the total number of events is equal to the number of
	sites with positive evidence times the number of sites with negative
	evidence.

	The objective is to infer which rupture has the highest probability
	of generating

	:param source_model:
		instance of :class:`rshalib.source.SourceModel`
	:param ground_motion_model:
		instance of :class:`rshalib.gsim.GroundMotionModel`
	:param imt:
		instance of :class:`openquake.hazardlib.imt.IMT`
	:param pe_sites:
		instance of :class:`rshalib.site.SoilSiteModel`, sites with
		positive evidence
	:param pe_threshold:
		float, minimum ground-motion intensity for positive evidence
	:param ne_sites:
		instance of :class:`rshalib.site.SoilSiteModel`, sites with
		negative evidence
		(default: None)
	:param ne_threshold:
		float, maximum ground-motion intensity for negative evidence
		(default: None)
	:param truncation_level:
		float, GMPE uncertainty truncation level
		(default: 3)
	:param integration_distance:
		float, integration distance in km
		Not taken into account yet!
		(default: 300)
	:param apply_rupture_probability:
		bool, whether or not to multiply probabilities of (non-)exceedance
		with the rupture probability
		(default: False)

	:return:
		dictionary mapping source IDs to lists of probabilities for each
		rupture in the source
	"""
	# use area sources or polygons or SiteCollection for pe_area, ne_area
	# or add from_polygon method to SiteCollection?

	prob_dict = {}

	for src in source_model:
		prob_dict[src.source_id] = []
		trt = src.tectonic_region_type
		gsim_name = ground_motion_model[trt]
		gsim = oqhazlib.gsim.get_available_gsims()[gsim_name]()
		tom = oqhazlib.tom.PoissonTOM(1)
		for rupture in src.iter_ruptures(tom):
			sctx, rctx, dctx = gsim.make_contexts(pe_sites, rupture)
			pe_poes = gsim.get_poes(sctx, rctx, dctx, imt, pe_threshold, truncation_level)
			pe_poes = pe_poes[0]
			pe_prob = np.prod(pe_poes)

			if not None in (ne_sites, ne_threshold):
				sctx, rctx, dctx = gsim.make_contexts(ne_sites, rupture)
				ne_poes = gsim.get_poes(sctx, rctx, dctx, imt, ne_threshold, truncation_level)
				ne_poes = ne_poes[0]
				ne_prob = np.prod(1 - ne_poes)
			else:
				ne_prob = 1

			total_prob = pe_prob * ne_prob
			if apply_rupture_probability:
				total_prob *= rupture.get_probability_one_or_more_occurrences()
			prob_dict[src.source_id].append(total_prob)

		prob_dict[src.source_id] = np.array(prob_dict[src.source_id])

	return prob_dict



if __name__ == "__main__":
	grid_outline = (-74, -72, -46, -44.5)
	grid_spacing = 0.25

	mfd = rshalib.mfd.EvenlyDiscretizedMFD(5.05, 0.5, np.ones(6)/6.)
	magnitudes = mfd.get_center_magnitudes()
	trt = "ASC"
	strike, dip, rake = 20, 90, 180
	nopl = rshalib.geo.NodalPlane(strike, dip, rake)
	npd = rshalib.pmf.NodalPlaneDistribution([nopl], [1])
	depth = 5
	hdd = rshalib.pmf.HypocentralDepthDistribution([depth], [1])
	usd = 0
	lsd = 20
	rar = 1
	msr = oqhazlib.scalerel.wc1994.WC1994()
	rms = 2.5

	lons = np.arange(grid_outline[0], grid_outline[1] + grid_spacing, grid_spacing)
	lats = np.arange(grid_outline[2], grid_outline[3] + grid_spacing, grid_spacing)
	sources = []
	i = 0
	for lon in lons:
		for lat in lats:
			point = rshalib.geo.Point(lon, lat)
			name = "%.2f, %.2f" % (lon, lat)
			source = rshalib.source.PointSource(i, name, trt, mfd, rms, msr, rar,
												usd, lsd, point, npd, hdd)
			sources.append(source)
			i += 1
	source_model = rshalib.source.SourceModel("Point source model", sources)

	ipe_name = "AtkinsonWald2007"
	#ipe_name = "BakunWentworth1997"
	trt_gsim_dict = {trt: ipe_name}
	ground_motion_model = rshalib.gsim.GroundMotionModel(ipe_name, trt_gsim_dict)

	imt = oqhazlib.imt.MMI()

	soil_params = rshalib.site.REF_SOIL_PARAMS

	pe_site = rshalib.site.SoilSite(-73, -45.4, soil_params=soil_params)
	pe_sites = rshalib.site.SoilSiteModel("Positive evidence", [pe_site])
	pe_threshold = 7.5

	ne_site = rshalib.site.SoilSite(-73.2, -45.3, soil_params=soil_params)
	ne_sites = rshalib.site.SoilSiteModel("Negative evidence", [ne_site])
	ne_threshold = 7.0

	truncation_level = 0

	prob_dict = calc_rupture_probability_from_ground_motion_thresholds(
						source_model, ground_motion_model, imt, pe_sites,
						pe_threshold, ne_sites, ne_threshold, truncation_level)

	x, y = [], []
	values = {'mag': [], 'prob': []}
	for source_id, probs in prob_dict.items():
		source = sources[source_id]
		idx = probs.argmax()
		prob_max = probs[idx]
		if prob_max > 0:
			values['prob'].append(prob_max)
			values['mag'].append(magnitudes[idx])
			x.append(source.location.longitude)
			y.append(source.location.latitude)

			print x[-1], y[-1], values['mag'][-1], prob_max

	import mapping.Basemap as lbm
	data = lbm.MultiPointData(x, y, values=values)
	thematic_size = lbm.ThematicStyleGradient()

