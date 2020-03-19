# -*- coding: iso-Latin-1 -*-
"""
GMPE plotting functions
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pylab
from matplotlib.font_manager import FontProperties

from ...utils import interpolate, logrange


__all__ = ['plot_distance', 'plot_spectrum']


def plot_distance(gmpe_list, mags, dmin=None, dmax=None, distance_metric=None,
				h=0, imt="PGA", T=0, imt_unit="g", epsilon=0, soil_type="rock",
				vs30=None, kappa=None, mechanism="normal", damping=5,
				plot_style="loglog", amin=None, amax=None, colors=None,
				fig_filespec=None, title="", want_minor_grid=False,
				legend_location=0, lang="en"):
	"""
	Function to plot ground motion versus distance for one or more GMPE's.
	Horizontal axis: distances.
	Vertical axis: ground motion values.

	:param gmpe_list:
		list of GMPE objects.
	:param mags:
		list of floats, magnitudes to plot
	:param dmin:
		float, lower distance in km. If None, use the lower bound of the
		distance range of each GMPE
		(default: None).
	:param dmax:
		float, upper distance in km. If None, use the lower bound of the
		valid distance range of each GMPE
		(default: None).
	:param distance_metric:
		str, distance_metric to plot (options: "Joyner-Boore", "Rupture")
		(default: None, distance metrics of gmpes are used)
	:param h:
		float, depth in km. Ignored if distance metric of GMPE is
		epicentral or Joyner-Boore
		(default: 0).
	:param imt:
		str, one of the supported intensity measure types.
		(default: "PGA").
	:param T:
		float, period to plot
		(default: 0).
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
	:param plot_style:
		str, plotting style ("lin", "loglin", "linlog" or "loglog").
		First term refers to horizontal axis, second term to vertical axis.
		(default: "loglog").
	:param amin:
		float, lower ground-motion value to plot
		(default: None).
	:param amax:
		upper ground-motion value to plot
		(default: None).
	:param colors:
		List of matplotlib color specifications
		(default: None)
	:param fig_filespec:
		str, full path specification of output file
		(default: None).
	:param title:
		str, plot title
		(default: "")
	:param want_minor_grid:
		Boolean, whether or not to plot minor gridlines
		(default: False).
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
	from .base import convert_distance_metric

	linestyles = ("", "--", ":", "-.")
	if not colors:
		colors = ("k", "r", "g", "b", "c", "m", "y")

	if plot_style.lower() in ("lin", "linlin"):
		plotfunc = pylab.plot
	elif plot_style.lower() == "linlog":
		plotfunc = pylab.semilogy
	elif plot_style.lower() == "loglin":
		plotfunc = pylab.semilogx
	elif plot_style.lower() == "loglog":
		plotfunc = pylab.loglog

	for i, gmpe in enumerate(gmpe_list):
		if dmin is None:
			dmin = gmpe.dmin
		if dmax is None:
			dmax = gmpe.dmax
		## Avoid math domain errors with 0
		if dmin == 0:
			dmin = 0.1
		distances = logrange(max(dmin, gmpe.dmin), min(dmax, gmpe.dmax), 25)
		for j, M in enumerate(mags):
			converted_distances = convert_distance_metric(distances,
									gmpe.distance_metric, distance_metric, M)
			Avalues = gmpe(M, converted_distances, h=h, imt=imt, T=T,
							imt_unit=imt_unit, soil_type=soil_type, vs30=vs30,
							kappa=kappa, mechanism=mechanism, damping=damping)
			style = colors[i%len(colors)] + linestyles[j%len(linestyles)]
			plotfunc(distances, Avalues, style, linewidth=3,
					label=gmpe.name+" (M=%.1f)" % M)
			if epsilon:
				## Fortunately, log_sigma is independent of scale factor!
				## Thus, the following are equivalent:
				#log_sigma = gmpe.log_sigma(M, imt=imt, T=T, soil_type=soil_type, vs30=vs30, kappa=kappa, mechanism=mechanism, damping=damping)
				#Asigmavalues = 10**(np.log10(Avalues) + log_sigma)
				Asigmavalues = gmpe(M, converted_distances, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon,
									soil_type=soil_type, vs30=vs30, kappa=kappa,
									mechanism=mechanism, damping=damping)
				plotfunc(distances, Asigmavalues, style, linewidth=1,
						label=gmpe.name+" (M=%.1f) $\pm %d \sigma$" % (M, epsilon))
				#Asigmavalues = 10**(np.log10(Avalues) - log_sigma)
				Asigmavalues = gmpe(M, converted_distances, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=-epsilon,
									soil_type=soil_type, vs30=vs30, kappa=kappa,
									mechanism=mechanism, damping=damping)
				plotfunc(distances, Asigmavalues, style, linewidth=1, label='_nolegend_')

	## Plot decoration
	if distance_metric:
		#pylab.xlabel(" ".join([distance_metric, "distance (km)"]), fontsize="x-large")
		distance_metrics = [distance_metric]
	else:
		distance_metrics = set()
		for gmpe in gmpe_list:
			distance_metrics.add(gmpe.distance_metric)
		distance_metrics = list(distance_metrics)
	if len(distance_metrics) > 1:
		pylab.xlabel(get_distance_label(None, lang), fontsize="x-large")
	else:
		pylab.xlabel(get_distance_label(distance_metrics[0], lang), fontsize="x-large")
	imt_label = (get_imt_label(imt, lang.lower())
				+ " (%s)" % IMT_UNIT_TO_PLOT_LABEL.get(imt_unit, imt_unit))
	pylab.ylabel(imt_label, fontsize="x-large")
	pylab.grid(True)
	if want_minor_grid:
		pylab.grid(True, which="minor")
	if title is None:
		title = "%s" % imt
		if not imt in ("PGA", "PGV", "PGD"):
			title += " (T=%s s)" % T
	pylab.title(title)
	font = FontProperties(size='medium')
	pylab.legend(loc=legend_location, prop=font)
	xmin, xmax, ymin, ymax = pylab.axis()
	if amin is None:
		amin = ymin
	if amax is None:
		amax = ymax
	pylab.axis((dmin, dmax, amin, amax))
	ax = pylab.gca()
	for label in ax.get_xticklabels() + ax.get_yticklabels():
		label.set_size('large')
	if fig_filespec:
		pylab.savefig(fig_filespec, dpi=300)
		pylab.clf()
	else:
		pylab.show()


def plot_spectrum(gmpe_list, mags, d, h=0, imt="SA", Tmin=None, Tmax=None,
				imt_unit="g", epsilon=0, soil_type="rock", vs30=None, kappa=None,
				mechanism="normal", damping=5, include_pgm=True, pgm_freq=50,
				plot_freq=False, plot_style="loglog", amin=None, amax=None,
				colors=None, labels=None, fig_filespec=None, title="",
				want_minor_grid=False, legend_location=0, lang="en"):
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
	:param Tmin:
		float, lower period to plot. If None, lower bound of valid period
		range is used
		(default: None).
	:param Tmax:
		float, upper period to plot. If None, upper bound of valid period
		range is used
		(default: None).
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
	:param pgm_freq:
		float, frequency (in Hz) at which to plot PGM if horizontal axis
		is logarithmic or is in frequencies
		(default: 50)
	:param plot_freq:
		Boolean, whether or not to plot frequencies instead of periods
		(default: False).
	:param plot_style:
		str, plotting style ("lin", "loglin", "linlog" or "loglog").
		First term refers to horizontal axis, second term to vertical axis.
		(default: "loglog").
	:param amin:
		float, lower ground-motion value to plot
		(default: None).
	:param amax:
		upper ground-motion value to plot
		(default: None).
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
	:param want_minor_grid:
		Boolean, whether or not to plot minor gridlines
		(default: False).
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
	linestyles = ("", "--", ":", "-.")
	if not colors:
		colors = ("k", "r", "g", "b", "c", "m", "y")

	if plot_style.lower() in ("lin", "linlin"):
		plotfunc = pylab.plot
	elif plot_style.lower() == "linlog":
		plotfunc = pylab.semilogy
	elif plot_style.lower() == "loglin":
		plotfunc = pylab.semilogx
	elif plot_style.lower() == "loglog":
		plotfunc = pylab.loglog

	for i, gmpe in enumerate(gmpe_list):
		periods = gmpe.imt_periods[imt]
		if Tmin is None or gmpe.Tmin(imt) < Tmin:
			Tmin = gmpe.Tmin(imt)
		if Tmax is None or gmpe.Tmax(imt) > Tmax:
			Tmax = gmpe.Tmax(imt)
		if plot_freq:
			freqs = gmpe.freqs(imt)
			xvalues = freqs
		else:
			xvalues = periods
		for j, M in enumerate(mags):
			periods, Avalues = gmpe.get_spectrum(M, d, h=h, imt=imt,
												imt_unit=imt_unit, epsilon=0,
												soil_type=soil_type, vs30=vs30,
												kappa=kappa, mechanism=mechanism,
												damping=damping)
			#Asigma_values = gmpe.get_spectrum(M, d, h=h, imt=imt, imt_unit=imt_unit, epsilon=num_sigma, soil_type=soil_type, mechanism=mechanism, damping=damping)
			Asigma_values = np.array([gmpe.log_sigma(M, d, h=h, imt=imt, T=T,
												soil_type=soil_type, vs30=vs30,
												kappa=kappa, mechanism=mechanism,
												damping=damping)[0] for T in periods])

			#non_zero_Avalues, non_zero_xvalues, non_zero_Asigma_values = [], [], []
			#for a, x, sigma in zip(Avalues, xvalues, Asigma_values):
			#	if a:
			#		non_zero_Avalues.append(a)
			#		non_zero_xvalues.append(x)
			#		non_zero_Asigma_values.append(sigma)

			style = linestyles[j] + colors[i]
			if isinstance(labels, (list, tuple)) and len(labels) > i and labels[i] != None:
				gmpe_label = labels[i]
			else:
				gmpe_label = gmpe.name
			if gmpe.is_rake_dependent():
				gmpe_label += " - %s" % mechanism
			plotfunc(xvalues, Avalues, style, linewidth=3,
					label=gmpe_label+" (M=%.1f)" % M)

			pgm = None
			if include_pgm:
				try:
					pgm = {"SA": "PGA", "PSV": "PGV", "SD": "PGD"}[imt]
				except:
					pass
				else:
					if gmpe.has_imt(pgm):
						[pgm_Avalue] = gmpe.__call__(M, d, h=h, imt=pgm,
													imt_unit=imt_unit, epsilon=0,
													soil_type=soil_type, vs30=vs30,
													kappa=kappa, mechanism=mechanism,
													damping=damping)
						if plot_style in ("loglin", "loglog") or plot_freq == True:
							pgm_T = 1./pgm_freq
						else:
							pgm_T = 0
						pgm_sigma = gmpe.log_sigma(M, d, h=h, imt=pgm,
												soil_type=soil_type, vs30=vs30,
												kappa=kappa, mechanism=mechanism,
												damping=damping)
						Tmin = pgm_T
						# TODO: add outline color and symbol size
						plotfunc(pgm_T, pgm_Avalue, colors[i]+"o", label="_nolegend_")
					else:
						pgm = None

			if epsilon:
				sigma_values = 10**(np.log10(Avalues) + epsilon * Asigma_values)
				plotfunc(xvalues, sigma_values, style, linewidth=1,
						label=gmpe_label+" (M=%.1f) $\pm %d \sigma$" % (M, epsilon))
				sigma_values = 10**(np.log10(Avalues) - epsilon * Asigma_values)
				plotfunc(xvalues, sigma_values, style, linewidth=1, label='_nolegend_')

				if pgm:
					for sign in (1.0, -1.0):
						pgm_sigma_value = (10**(np.log10(pgm_Avalue)
											+ epsilon * sign * pgm_sigma))
						# TODO: add outline color and symbol size
						plotfunc(pgm_T, pgm_sigma_value, "o", mec=colors[i],
								mfc="None", label="_nolegend_")

	## PLot decoration
	pylab.grid(True)
	if want_minor_grid:
		pylab.grid(True, which="minor")
	if not title:
		title = "\nd=%.1f km, h=%d km" % (d, int(round(h)))
	pylab.title(title)
	font = FontProperties(size='medium')
	xmin, xmax, ymin, ymax = pylab.axis()
	if amin is None:
		amin = ymin
	if amax is None:
		amax = ymax
	if plot_freq:
		pylab.xlabel("Frequency (Hz)", fontsize="x-large")
		if legend_location == None:
			legend_location = 4
		pylab.axis((1./Tmax, 1./Tmin, amin, amax))
	else:
		pylab.xlabel("Period (s)", fontsize="x-large")
		if legend_location == None:
			legend_location = 3
		pylab.axis((Tmin, Tmax, amin, amax))
	imt_label = (get_imt_label(imt, lang.lower())
				+ " (%s)" % IMT_UNIT_TO_PLOT_LABEL.get(imt_unit, imt_unit))
	pylab.ylabel(imt_label, fontsize="x-large")
	pylab.legend(loc=legend_location, prop=font)
	ax = pylab.gca()
	for label in ax.get_xticklabels() + ax.get_yticklabels():
		label.set_size('large')
	if fig_filespec:
		pylab.savefig(fig_filespec, dpi=300)
		pylab.clf()
	else:
		pylab.show()


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

