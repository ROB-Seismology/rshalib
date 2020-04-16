# -*- coding: iso-Latin-1 -*-
"""
GMPE plotting functions
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pylab
from matplotlib.font_manager import FontProperties

from plotting.generic_mpl import plot_xy

from ...utils import interpolate, logrange


__all__ = ['plot_distance', 'plot_spectrum']


def plot_distance(gmpe_list, mags, dmin=None, dmax=None, distance_metric=None,
				h=0, imt='PGA', T=0, imt_unit='g', epsilon=0, soil_type='rock',
				vs30=None, kappa=None, mechanism='normal', damping=5,
				xscaling='log', yscaling='log', xgrid=1, ygrid=1,
				colors=None, linestyles=None, linewidths=None, legend_location=0,
				lang='en', title='', fig_filespec=None, **kwargs):
	"""
	Function to plot ground motion versus distance for one or more GMPE's.
	Horizontal axis: distances.
	Vertical axis: ground motion values.

	:param gmpe_list:
		list with instances of :class:`GMPE`
	:param mags:
		list of floats, magnitudes to plot
	:param dmin:
		float, lower distance in km. If None, use the lower bound of the
		distance range of each GMPE
		(default: None)
	:param dmax:
		float, upper distance in km. If None, use the lower bound of the
		valid distance range of each GMPE
		(default: None)
	:param distance_metric:
		str, distance_metric to plot (options: "Joyner-Boore", "Rupture")
		(default: None, distance metrics of gmpes are used)
	:param h:
		float, depth in km. Ignored if distance metric of GMPE is
		epicentral or Joyner-Boore
		(default: 0)
	:param imt:
		str, one of the supported intensity measure types.
		(default: 'PGA')
	:param T:
		float, spectral period to plot if imt is not peak-ground motion
		(default: 0).
	:param imt_unit:
		str, unit in which intensities should be expressed,
		depends on IMT
		(default: 'g')
	:param epsilon:
		float, number of standard deviations above or below the mean to
		plot in addition to the mean
		(default: 0)
	:param soil_type:
		str, one of the soil types supported by the particular GMPE
		(default: "rock")
	:param vs30:
		float, shear-wave velocity in the upper 30 m (in m/s).
		If not None, it takes precedence over :param:`soil_type`
		(default: None)
	:param kappa:
		float, kappa value, in seconds
		(default: None)
	:param mechanism:
		str, fault mechanism: either "normal", "reverse" or "strike-slip"
		(default: 'normal')
	:param damping:
		float, damping in percent
		(default: 5)
	:param xscaling:
		str, scaling to use for X axis ('lin' or 'log')
		Prepend '-' to invert orientation of X axis
		(default: 'log')
	:param yscaling:
		cf. :param:`xscaling`, but for Y axis
		(default: 'log')
	:param xgrid:
		int, 0/1/2/3 = draw no/major/minor/major+minor X grid lines
		(default: 1)
	:param ygrid:
		cf. :param:`xgrid`, but for Y axis
		(default: 1)
	:param colors:
		list of matplotlib color specifications for each GMPE
		(default: None, will use default colors)
	:param linestyles:
		list of strings, matplotlib linestyle specifications for each
		magnitude
		(default: None)
	:param linewiths:
		list of 2 floats, line widths for median and +/- epsilon curves
		(default: None)
	:param legend_location:
		int or str, location of legend (matplotlib location code):
			"best" 	0
			"upper right" 	1
			"upper left" 	2
			"lower left" 	3
			"lower right" 	4
			"right" 	5
			"center left" 	6
			"center right" 	7
			"lower center" 	8
			"upper center" 	9
			"center" 	10
		(default: 0)
	:param lang:
		str, shorthand for language of annotations.
		Currently only "en" and "nl" are supported
		(default: "en")
	:param title:
		str, plot title
		If None, a default title will be shown
		(default: "")
	:param fig_filespec:
		str, full path specification of output file
		(default: None)
	:param kwargs:
		additional keyword arguments understood by :func:`generic_mpl.plot_xy`

	:return:
		matplotlib Axes instances
	"""
	from .base import convert_distance_metric

	COLORS = colors or pylab.rcParams['axes.prop_cycle'].by_key()['color']
	LINESTYLES = linestyles or ("-", "--", ":", "-.")
	LINEWIDTHS = linewidths or [3, 1]
	assert len(LINEWIDTHS) == 2

	datasets, colors, linestyles, linewidths, labels = [], [], [], [], []
	for i, gmpe in enumerate(gmpe_list):
		num_dist = 50
		if dmin is None:
			dmin = gmpe.dmin
		if dmax is None:
			dmax = gmpe.dmax
		## Avoid math domain errors with 0
		if xscaling == 'log':
			if dmin == 0:
				dmin = 0.1
			distances = logrange(max(dmin, gmpe.dmin), min(dmax, gmpe.dmax), num_dist)
		else:
			distances = np.linspace(max(dmin, gmpe.dmin), min(dmax, gmpe.dmax), num_dist)

		for j, M in enumerate(mags):
			converted_distances = convert_distance_metric(distances,
									gmpe.distance_metric, distance_metric, M)
			Avalues = gmpe(M, converted_distances, h=h, imt=imt, T=T,
							imt_unit=imt_unit, soil_type=soil_type, vs30=vs30,
							kappa=kappa, mechanism=mechanism, damping=damping)

			datasets.append((converted_distances, Avalues))
			color = COLORS[i%len(COLORS)]
			colors.append(color)
			linestyle = LINESTYLES[j%len(LINESTYLES)]
			linestyles.append(linestyle)
			linewidths.append(LINEWIDTHS[0])
			labels.append(gmpe.name+" (M=%.1f)" % M)

			if epsilon:
				linewidth = LINEWIDTHS[1]
				## Fortunately, log_sigma is independent of scale factor!
				## Thus, the following are equivalent:
				#log_sigma = gmpe.log_sigma(M, imt=imt, T=T, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
				#Asigmavalues = 10**(np.log10(Avalues) + log_sigma)
				Asigmavalues = gmpe(M, converted_distances, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon,
									soil_type=soil_type, vs30=vs30, kappa=kappa,
									mechanism=mechanism, damping=damping)
				datasets.append((converted_distances, Asigmavalues))
				colors.append(color)
				linestyles.append(linestyle)
				linewidths.append(linewidth)
				labels.append(gmpe.name+" (M=%.1f) $\pm %s \sigma$" % (M, epsilon))

				#Asigmavalues = 10**(np.log10(Avalues) - log_sigma)
				Asigmavalues = gmpe(M, converted_distances, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=-epsilon,
									soil_type=soil_type, vs30=vs30, kappa=kappa,
									mechanism=mechanism, damping=damping)
				datasets.append((converted_distances, Asigmavalues))
				colors.append(color)
				linestyles.append(linestyle)
				linewidths.append(linewidth)
				labels.append('_nolegend_')

	## Plot decoration
	xlabel = kwargs.pop('xlabel', None)
	if xlabel is None:
		if distance_metric:
			distance_metrics = [distance_metric]
		else:
			distance_metrics = set()
			for gmpe in gmpe_list:
				distance_metrics.add(gmpe.distance_metric)
			distance_metrics = list(distance_metrics)
		if len(distance_metrics) > 1:
			xlabel = get_distance_label(None, lang)
		else:
			xlabel = get_distance_label(distance_metrics[0], lang)

	ylabel = kwargs.pop('ylabel', None)
	if ylabel is None:
		ylabel = (get_imt_label(imt, lang.lower())
				+ " (%s)" % IMT_UNIT_TO_PLOT_LABEL.get(imt_unit, imt_unit))

	if title is None:
		title = "%s" % imt
		if not imt in ("PGA", "PGV", "PGD"):
			title += " (T=%s s)" % T

	kwargs['xmin'] = kwargs.get('xmin', dmin)
	kwargs['xmax'] = kwargs.get('xmax', dmax)

	return plot_xy(datasets, labels=labels, colors=colors, linestyles=linestyles,
					linewidths=linewidths, xscaling=xscaling, yscaling=yscaling,
					xlabel=xlabel, ylabel=ylabel, xgrid=xgrid, ygrid=xgrid,
					legend_location=legend_location, title=title,
					fig_filespec=fig_filespec, **kwargs)


def plot_spectrum(gmpe_list, mags, d, h=0, imt="SA",
				#Tmin=None, Tmax=None,
				imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None,
				mechanism="normal", damping=5,
				include_pgm=True, pgm_period=1./50, pgm_marker='o',
				plot_freq=False,
				#plot_style="loglog", amin=None, amax=None,
				labels=None, colors=None, linestyles=None, linewidths=None,
				fig_filespec=None, title="",
				#want_minor_grid=False,
				legend_location=0, lang="en", **kwargs):
	"""
	Function to plot ground motion spectrum for one or more GMPE's.
	Horizontal axis: spectral periods or frequencies.
	Vertical axis: ground motion values.

	:param gmpe_list:
		list of GMPE objects.
	:param mags:
		lsit of floats, magnitudes to plot
	:param d:
		float, distance in km.
	:param h:
		float, depth in km. Ignored if distance metric of GMPE is
		epicentral or Joyner-Boore
		(default: 0).
	:param imt:
		str, one of the supported intensity measure types.
		(default: "SA").
	:param imt_unit:
		str, unit in which intensities should be expressed,
		depends on IMT
		(default: "g")
	:param epsilon:
		float, number of standard deviations above or below the mean to
		plot in addition to the mean
		(default: 0).
	:param soil_type:
		str, one of the soil types supported by the particular GMPE
		(default: "rock").
	:param vs30:
		float, shear-wave velocity in the upper 30 m (in m/s).
		If not None, it takes precedence over the soil_type parameter
		(default: None).
	:param kappa:
		float, kappa value, in seconds
		(default: None)
	:param mechanism:
		str, fault mechanism: either "normal", "reverse" or "strike-slip"
		(default: "normal").
	:param damping:
		float, damping in percent
		(default: 5).
	:param include_pgm:
		Boolean, whether or not to include peak ground motion in the plot,
		if possible
		(default: True).
	:param pgm_period:
		float, period (in s) at which to plot PGM if axis is
		logarithmic or corresponds to frequencies
		(default: 0.01)
	:param pgm_marker:
		char, matplotlib marker spec to plot PGM
		If '', PGM will not be plotted; if None, PGM will be connected
		with spectrum
		(default: 'o')
	:param plot_freq:
		Boolean, whether or not to plot frequencies instead of periods
		(default: False).
	:param colors:
		List of matplotlib color specifications
		(default: None).
	:param labels:
		List of labels for each GMPE
		(defaultl: None)
	:param fig_filespec:
		str, full path specification of output file
		(default: None).
	:param title:
		str, plot title
		(default: "")
	:param legend_location:
		Integer, location of legend (matplotlib location code):
		"best" 	0
		"upper right" 	1
		"upper left" 	2
		"lower left" 	3
		"lower right" 	4
		"right" 	5
		"center left" 	6
		"center right" 	7
		"lower center" 	8
		"upper center" 	9
		"center" 	10
		(default: 0)
	:param lang:
		str, shorthand for language of annotations.
		Currently only "en" and "nl" are supported
		(default: "en").
	"""
	from robspy.response import plot_response_spectra

	COLORS = colors or pylab.rcParams['axes.prop_cycle'].by_key()['color']
	LINESTYLES = linestyles or ("-", "--", ":", "-.")
	LINEWIDTHS = linewidths or [3, 1]
	assert len(LINEWIDTHS) == 2
	LABELS = labels or [''] * len(gmpe_list)

	spectra, colors, linestyles, linewidths, labels = [], [], [], [], []
	for i, gmpe in enumerate(gmpe_list):
		periods = gmpe.imt_periods[imt]

		if plot_freq:
			freqs = gmpe.freqs(imt)
			xvalues = freqs
		else:
			xvalues = periods
		for j, M in enumerate(mags):
			rs = gmpe.get_spectrum(M, d, h=h, imt=imt,
									imt_unit=imt_unit, epsilon=0,
									soil_type=soil_type, vs30=vs30,
									kappa=kappa, mechanism=mechanism,
									damping=damping, include_pgm=include_pgm)
			spectra.append(rs)
			color = COLORS[i%len(COLORS)]
			colors.append(color)
			linestyle = LINESTYLES[j%len(LINESTYLES)]
			linestyles.append(linestyle)
			linewidths.append(LINEWIDTHS[0])

			gmpe_label = LABELS[i] or gmpe.name
			if gmpe.is_rake_dependent():
				gmpe_label += " - %s" % mechanism
			label = gmpe_label + " (M=%.1f)" % M
			labels.append(label)

			if epsilon:
				linewidth = LINEWIDTHS[1]
				for sign in (1.0, -1.0):
					num_sigma = epsilon * sign
					rs = gmpe.get_spectrum(M, d, h=h, imt=imt, imt_unit=imt_unit,
										epsilon=num_sigma, soil_type=soil_type,
										mechanism=mechanism, damping=damping,
										include_pgm=include_pgm)
					spectra.append(rs)
					colors.append(color)
					linestyles.append(linestyle)
					linewidths.append(linewidth)
				label = gmpe_label + " (M=%.1f) $\pm %s \sigma$" % (M, epsilon)
				labels.extend([label, '_nolegend_'])

	if title is None:
		title = "\nd=%.1f km, h=%d km" % (d, int(round(h)))
	imt_label = (get_imt_label(imt, lang.lower())
				+ " (%s)" % IMT_UNIT_TO_PLOT_LABEL.get(imt_unit, imt_unit))
	ylabel = kwargs.pop('ylabel', None) or imt_label

	return plot_response_spectra(spectra, labels=labels, colors=colors,
								linestyles=linestyles, linewidths=linewidths,
								pgm_period=pgm_period, pgm_marker=pgm_marker,
								plot_freq=plot_freq, ylabel=ylabel,
								#xmin=xmin, xmax=xmax,
								fig_filespec=fig_filespec, title=title,
								legend_location=legend_location, **kwargs)


## Dictionary to convert IMT units to plot labels
IMT_UNIT_TO_PLOT_LABEL = {}
IMT_UNIT_TO_PLOT_LABEL["g"] = "g"
IMT_UNIT_TO_PLOT_LABEL["gal"] = "gal"
IMT_UNIT_TO_PLOT_LABEL["ms2"] = "$m/s^2$"
IMT_UNIT_TO_PLOT_LABEL["cms2"] = "$cm/s^2$"
IMT_UNIT_TO_PLOT_LABEL["ms"] = "m/s"
IMT_UNIT_TO_PLOT_LABEL["cms"] = "cm/s"
IMT_UNIT_TO_PLOT_LABEL["m"] = "m"
IMT_UNIT_TO_PLOT_LABEL["cm"] = "cm"


def get_imt_label(imt, lang="en"):
	"""
	Return plot label for a particular IMT

	:param imt:
		str, Intensity measure type
	:param lang:
		str, shorthand for language of annotations. Currently only
		"en", "fr" and "nl" are supported (default: "en").

	:return:
		str, plot axis label.
	"""
	imt_label = {}
	imt_label["PGA"] = {"en": "Peak Ground Acceleration",
						"nl": "Piekgrondversnelling",
						"fr": u"Accélération maximale du sol"}
	imt_label["PGV"] = {"en": "Peak Ground Velocity",
						"nl": "Piekgrondsnelheid",
						"fr": "Vitesse maximale"}
	imt_label["PGD"] = {"en": "Peak Ground Displacement",
						"nl": "Piekgrondverplaatsing",
						"fr": u"Déplacement maximal"}
	imt_label["SA"] = {"en": "Spectral Acceleration",
						"nl": "Spectrale versnelling",
						"fr": u"Accélération spectrale"}
	imt_label["PSV"] = {"en": "Spectral Velocity",
						"nl": "Spectrale snelheid",
						"fr": "Vitesse spectrale"}
	imt_label["SD"] = {"en": "Spectral Displacement",
						"nl": "Spectrale verplaatsing",
						"fr": u"Déplacement spectral"}
	return imt_label[imt][lang]


def get_distance_label(distance_metric, lang="en"):
	"""
	Return plot label for a particular distance metric

	:param distance_metric:
		str, distance metric or None
	:param lang:
		str, shorthand for language of annotations. Currently only
		"en", "fr" and "nl" are supported (default: "en").

	:return:
		str, plot axis label.
	"""
	if not distance_metric:
		label = {"en": "Distance", "nl": "Afstand", "fr": "Distance"}[lang]
	else:
		if lang == "en":
			label = "%s distance" % distance_metric
		else:
			if distance_metric.lower() == "hypocentral":
				label = {"nl": "Hypocentrale afstand",
						"fr": "Distance hypocentrale"}[lang]
			elif distance_metric.lower() == "epicentral":
				label = {"nl": "Epicentrale afstand",
						"fr": u"Distance épicentrale"}[lang]
			elif distance_metric.lower() == "rupture":
				label = {"nl": "Ruptuur-afstand", "fr":
						"Distance de rupture"}[lang]
			elif distance_metric.lower() == "joyner-boore":
				label = {"nl": "Joyner-Boore afstand",
						"fr": "Distance Joyner-Boore"}[lang]
	label += ' (km)'
	return label


if __name__ == "__main__":
	pass

