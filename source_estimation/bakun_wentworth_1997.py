
import numpy as np

import openquake.hazardlib as oqhazlib

from ..gsim import InverseGSIM
from ..site import SoilSiteModel


def estimate_epicenter_location_and_magnitude_from_intensities(
		ipe_name, imt, grid_source_model, pe_sites, pe_intensities,
		ne_sites=[], ne_intensities=[], method="reverse", mag_bounds=(3, 8.5),
		integration_distance=None):
	"""
	Estimate epicenter and location from a number of observed intensities
	using the grid search method of Bakun & Wentworth (1997),
	extended to take into account negative evidence (see Strasser et al.,
	2006; Kremer et al., 2017)

	:param ipe_name:
		str, name of intensity prediction equation known to OpenQuake
	:param imt:
		instance of :class:`oqhazlib.imt`, intensity measure type
	:param grid_source_model:
		instance of :class:`SimpleUniformGridSourceModel`, uniform gridded
		source model. Note that when :param:`method` = "reverse", the MFD
		at each node should contain only one magnitude, and when
		:param:`method` = forward, the MFD at each node should contain
		a range of candidate magnitudes.
	:param pe_sites:
		list of instances of :class:`rshalib.site.SoilSite`,
		sites with positive evidence
	:param pe_intensities:
		list of floats, intensities at positive evidence sites
	:param ne_sites:
		list of instances of :class:`rshalib.site.SoilSite`,
		sites with negative evidence
		(default: [])
	:param ne_intensities:
		list of floats, intensities at negative evidence sites
		(default: [])
	:param method:
		str, method, one of "reverse" or "forward".
		"reverse" referse to the original method of Bakun & Wentworth,
		which applies IPEs inversely. It is also possible to formulate
		the method using forward IPE application. Both methods give
		slightly different results
		(default: "reverse")
	:param mag_bounds:
		(min_mag, max_mag) tuple of floats, minimum and maximum magnitude
		between which to search for the epicentral magnitude. Only applies
		when :param:`method` = "reverse"
	:param integration_distance:
		float, maximum source-site distance to take into account
		(default: None)

	:return:
		(mag_grid, rms_grid) tuple:
		- mag_grid: 2-D mesh of magnitudes
		- rms_grid: 2-D mesh of RMS errors
	"""
	assert len(pe_sites) == len(pe_intensities)
	assert len(ne_sites) == len(ne_intensities)

	## Construct site models
	num_pe = pe_intensities
	pe_site_model = SoilSiteModel("Positive evicence", pe_sites)
	ind_pe_site_models = [SoilSiteModel("", [site]) for site in pe_sites]
	num_ne = len(ne_intensities)
	if num_ne:
		ne_site_model = SoilSiteModel("Negative evidence", ne_sites)

	## Initialize IPE
	ipe = oqhazlib.gsim.get_available_gsims()[ipe_name]()
	dist_metric = list(ipe.REQUIRES_DISTANCES)[0]
	stddev_type = oqhazlib.const.StdDev.TOTAL
	tom = oqhazlib.tom.PoissonTOM(1)

	if method == "reverse":
		inverse_ipe = InverseGSIM(ipe_name)

	## Compute most likely magnitude at each grid point, and corresponding RMS error
	mag_grid = np.zeros_like(grid_source_model.lon_grid)
	rms_grid = np.ones_like(grid_source_model.lon_grid)

	for lat_idx, lat in enumerate(grid_source_model.lats):
		for lon_idx, lon in enumerate(grid_source_model.lons):
			src = grid_source_model.create_point_source(lon, lat)

			## Standard BW1997 method applies IPEs inversely
			if method == "reverse":
				## There should be only one magnitude in this case
				[rupture] = src.iter_ruptures(tom)
				mag_estimates = []
				for i in range(len(num_pe)):
					site_model = ind_pe_site_models[i]
					mmi = pe_intensities[i]
					sctx, rctx, dctx = ipe.make_contexts(site_model, rupture)
					mag = inverse_ipe.find_mag_from_intensity([mmi], site_model,
															rupture, mag_bounds)
					mag_estimates.append(mag)
				mag_estimates = np.array(mag_estimates)
				sctx, rctx, dctx = ipe.make_contexts(pe_site_model, rupture)
				distances = getattr(dctx, dist_metric)
				weights = np.ones_like(mag_estimates) * 0.1
				idxs = np.where(distances <= 150)
				weights[idxs] += np.cos(distances[idxs]/150 * np.pi/2)
				if integration_distance:
					idxs = np.where(distances > integration_distance)
					weights[idxs] = 0
				#mean_mag_estimate = mag_estimates.mean()
				mean_mag_estimate = np.average(mag_estimates, weights=weights)

				rms = np.sqrt(np.sum(weights * (mean_mag_estimate - mag_estimates)**2)
							/ np.sum(weights**2))
				mag_grid[lat_idx, lon_idx] = mean_mag_estimate
				rms_grid[lat_idx, lon_idx] = rms

				## Negative evidence
				if num_ne:
					sctx, rctx, dctx = ipe.make_contexts(ne_site_model, rupture)
					mmi, [sigma] = ipe.get_mean_and_stddevs(sctx, rctx, dctx, imt,
																[stddev_type])
					if (mmi >= ne_intensities).any():
						mag_grid[lat_idx, lon_idx] = np.nan
						#rms_grid[lat_idx, lon_idx] = np.nan

			elif method == "forward":
				## Forward method
				src_mags = []
				src_rms = []
				## There should be multiple magnitudes in this case
				for rupture in src.iter_ruptures(tom):
					sctx, rctx, dctx = ipe.make_contexts(pe_site_model, rupture)
					mmi, _ = ipe.get_mean_and_stddevs(sctx, rctx, dctx, imt,
														[stddev_type])
					distances = getattr(dctx, dist_metric)

					weights = np.ones_like(mmi) * 0.1
					idxs = np.where(distances <= 150)
					weights[idxs] += np.cos(distances[idxs]/150 * np.pi/2)
					if integration_distance:
						idxs = np.where(distances > integration_distance)
						weights[idxs] = 0

					## RMS with respect to pe_intensities, not mean mmi!
					rms = np.sqrt(np.sum(weights * (pe_intensities - mmi)**2)
								/ np.sum(weights**2))

					## Negative evidence
					if num_ne:
						sctx, rctx, dctx = ipe.make_contexts(ne_site_model, rupture)
						mmi, [sigma] = ipe.get_mean_and_stddevs(sctx, rctx, dctx,
															imt, [stddev_type])
						if (mmi < ne_intensities).all():
							src_mags.append(rupture.mag)
							src_rms.append(rms)

				if len(src_mags):
					idx = np.argmin(src_rms)
					mag_grid[lat_idx, lon_idx] = src_mags[idx]
					rms_grid[lat_idx, lon_idx] = src_rms[idx]
				else:
					mag_grid[lat_idx, lon_idx] = np.nan
					rms_grid[lat_idx, lon_idx] = np.nan

	return (mag_grid, rms_grid)

