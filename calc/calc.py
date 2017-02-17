

def calc_rupture_probability_from_ground_motion_thresholds(
	source_model,
	ground_motion_model,
	pe_area,
	ne_area,
	pe_threshold,
	ne_threshold,
	imt,
	truncation_level,
	integration_distance
	):
	"""
	Compute rupture probabilities for potential seismic sources based
	on ground-motion (typically intensity) thresholds
	"""
	# use get_poes method of IPE/GMPE
	# create site, rupture, distance contexts
	# use area sources or polygons or SiteCollection for pe_area, ne_area
	# or add from_polygon method to SiteCollection?
	pass
