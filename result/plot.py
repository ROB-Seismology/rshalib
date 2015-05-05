# -*- coding: utf-8 -*-
"""
functions for plotting hazard results
"""

# TODO: All functions should have intensity_unit parameter

### imports
## Kludge because matplotlib is broken on seissrv3. Sigh...
import platform
if platform.uname()[1] == "seissrv3":
	import matplotlib
	matplotlib.use('AGG')
import numpy as np
import pylab
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from ..utils import interpolate


def plot_hazard_curve(datasets, labels=[], colors=[], linestyles=[], linewidths=[], fig_filespec=None, title="", want_recurrence=False, fixed_life_time=None, interpol_rp=None, interpol_prob=0, interpol_rp_range=None, amax=None, intensity_unit="g", tr_max=1E+07, legend_location=0, lang="en"):
	"""
	Generic function to plot a hazard curve (exceedance rate or probability of exceedance)
	Parameters:
		datasets: list of datasets. Each dataset is a (intensities, exceedances) tuple
		labels: list of labels for each dataset (default: [])
		colors: list of colors to plot each dataset (default: [])
		linestyles: list of line styles to plot each dataset (default: [])
		linewidths: list of line widths to plot each dataset (default: [])
		fig_filespec: full path to ouptut image. If not set, graph will be plotted on screen
			(default: None)
		title: plot title
		want_recurrence: Y axis is recurrence interval instead of exceedance rate
			(default: False)
		fixed_life_time: plot probability of exceedance for given life time
			instead of exceedance rate (default: None)
		interpol_rp: return period for which to interpolate intensity
			(one value or a list of values for each dataset). Will be plotted
			with a dashed line for each dataset (default: None, i.e. no interpolation)
		interpol_prob: exceedance probability for which to interpolate intensity
			(one value or list of values for each dataset). Will be plotted
			with a dashed line for each dataset  (default: None, i.e. no interpolation)
		interpol_rp_range: return period range for which to interpolate intensity
			([min return period, max return period] list). Will be plotted
			with a grey area for first dataset only (default: None, i.e. no interpolation)
		amax: maximum intensity to plot in X axis (default: 1)
		intensity_unit: intensity unit (default: "g")
		tr_max: maximum return period to plot in Y axis (default: 1E+07)
		legend_location: location of legend (matplotlib location code) (default=0):
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
		lang: language to use for labels: en=English, nl=Dutch (default: en)
	"""
	if not labels:
		labels = ["Set %d" % (i+1) for i in range(len(datasets))]

	if not colors:
		colors = ("r", "g", "b", "c", "m", "k")

	if not linestyles:
		linestyles = ["-"]

	if not linewidths:
		linewidths = [2]

	if interpol_rp != None:
		if not hasattr(interpol_rp, '__iter__'):
			#interpol_rp = [interpol_rp for i in range(len(datasets))]
			interpol_rp = np.array([interpol_rp])
		else:
			interpol_rp = np.array(interpol_rp)

	if interpol_prob != None:
		if not hasattr(interpol_prob, '__iter__'):
			#interpol_prob = [interpol_prob for i in range(len(datasets))]
			interpol_prob = np.array([interpol_prob])
		else:
			interpol_prob = np.array(interpol_prob)

	pylab.clf()
	ax = pylab.subplot(111)

	for i, dataset in enumerate(datasets):
		## Plot
		intensities, exceedances = dataset
		## Ignore zero curves
		if not np.allclose(exceedances, 0):
			if want_recurrence:
				yvalues = 1.0 / exceedances
			elif fixed_life_time:
				yvalues = (1.0 - np.exp(-fixed_life_time*exceedances))
			else:
				yvalues = exceedances

			label = labels[i]
			color = colors[i%len(colors)]
			linestyle = linestyles[i%len(linestyles)]
			linewidth = linewidths[i%len(linewidths)]
			pylab.semilogy(intensities, yvalues, color=color, linestyle=linestyle, linewidth=linewidth, label=label)

			## Interpolate acceleration corresponding to return period or probability of exceedance
			#if interpol_rp[i] or interpol_prob[i]:
			if (interpol_rp, interpol_prob) != (None, None):
				#if interpol_rp[i]:
				if interpol_rp != None:
					#interpol_pga = intcubicspline(exceedances, intensities, [1.0/interpol_rp[i]], ideriv=0)[0]
					#interpol_pga = interpolate(exceedances, intensities, [1.0/interpol_rp[i]])[0]
					interpol_pga = interpolate(exceedances, intensities, 1.0/interpol_rp)
					if want_recurrence:
						#interpol_y = interpol_rp[i]
						interpol_y = interpol_rp
					elif fixed_life_time:
						#interpol_y = (1.0 - np.exp(-fixed_life_time*(1.0/interpol_rp[i])))
						interpol_y = (1.0 - np.exp(-fixed_life_time*(1.0/interpol_rp)))
					else:
						#interpol_y = 1.0/interpol_rp[i]
						interpol_y = 1.0/interpol_rp
				#if interpol_prob[i]:
				if interpol_prob != None:
					if fixed_life_time:
						#interpol_pga = intcubicspline(yvalues, intensities, [interpol_prob[i]], ideriv=0)[0]
						#interpol_pga = interpolate(yvalues, intensities, [interpol_prob[i]])[0]
						#interpol_y = interpol_prob[i]
						interpol_pga = interpolate(yvalues, intensities, interpol_prob)
						interpol_y = interpol_prob
					else:
						interpol_pga = None
						print "fixed_life_time must be set if interpol_prob is used!"
				if interpol_pga != None:
					print "%s" % interpol_pga
					#pylab.semilogy([0.0, interpol_pga, interpol_pga, interpol_pga], [interpol_y, interpol_y, interpol_y, 1E-10], color=color, linestyle=':', linewidth=linewidth, label="_nolegend_")
					for ipga, iy in zip(interpol_pga, interpol_y):
						pylab.semilogy([0.0, ipga, ipga, ipga], [iy, iy, iy, 1E-10], color=color, linestyle=':', linewidth=linewidth, label="_nolegend_")

			## Interpolate acceleration range corresponding to return period range
			if interpol_rp_range and i == 0:
				exc_range = 1.0/np.array(interpol_rp_range)
				shade = '0.75'
				if len(datasets) == 1:
					acc_range = np.interp(exc_range, dataset[1][::-1], dataset[0][::-1])
					pylab.fill([0, 0, max(acc_range), max(acc_range)], [min(exc_range), max(exc_range), max(exc_range), min(exc_range)], shade, edgecolor=shade)
					pylab.fill([min(acc_range), max(acc_range), max(acc_range), min(acc_range)], [1.0/tr_max, 1.0/tr_max, min(exc_range), min(exc_range)], shade, edgecolor=shade)

	if not amax:
		amax = pylab.axis()[1]

	if interpol_rp_range and len(datasets) > 1:
		pylab.fill([0, 0, amax, amax], [min(exc_range), max(exc_range), max(exc_range), min(exc_range)], shade, edgecolor=shade)

	## Plot decoration
	xlabel = {"en": "Acceleration", "nl": "Versnelling", "fr": u"Accélération"}[lang]
	xlabel += " (%s)" % intensity_unit
	pylab.xlabel(xlabel, fontsize='x-large')
	if want_recurrence:
		pylab.ylabel({"en": "Return period (yr)", "nl": "Terugkeerperiode (jaar)"}[lang], fontsize='x-large')
		pylab.axis((0.0, amax, 1, tr_max))
	elif fixed_life_time:
		pylab.ylabel({"en": "Probability of exceedance", "nl": "Overschrijdingskans"}[lang], fontsize='x-large')
		pylab.axis((0.0, amax, 1E-05, 1))
	else:
		pylab.ylabel({"en": "Exceedance rate (1/yr)", "nl": "Overschrijdingssnelheid (1/jaar)", "fr": u"Taux de dépassement (1/a)"}[lang], fontsize='x-large')
		pylab.axis((0, amax, 1.0/tr_max, 1))
	font = FontProperties(size='large')
	pylab.legend(loc=legend_location, prop=font)
	if fixed_life_time:
		title += "\n%s: %d %s" % ({"en": "Fixed life time", "nl": "Vaste levensduur"}[lang], fixed_life_time, {"en": "yr", "nl": "jaar"}[lang])
	majorFormatter = FormatStrFormatter('%.1f')
	if amax <= 1.:
		tick_interval = 0.1
	elif amax <= 2:
		tick_interval = 0.2
	elif amax <= 3:
		tick_interval = 0.25
		majorFormatter = FormatStrFormatter('%.2f')
	else:
		tick_interval = 0.5
	majorLocator = MultipleLocator(tick_interval)
	minorLocator = MultipleLocator(tick_interval / 10.)
	ax.xaxis.set_major_locator(majorLocator)
	ax.xaxis.set_major_formatter(majorFormatter)
	ax.xaxis.set_minor_locator(minorLocator)
	pylab.title(title)
	pylab.grid(True)
	ax = pylab.gca()
	for label in ax.get_xticklabels() + ax.get_yticklabels():
		label.set_size('large')
	if fig_filespec:
		pylab.savefig(fig_filespec, dpi=300)
	else:
		pylab.show()

	#pylab.clf()


def plot_hazard_spectrum(datasets, pgm=None, pgm_period=0.02, labels=[], colors=[], linestyles=[], linewidths=[], fig_filespec=None, title="", plot_freq=False, plot_style="loglin", Tmin=None, Tmax=None, amin=None, amax=None, intensity_unit="g", legend_location=0, lang="en"):
	"""
	Generic function to plot a (usually uniform) hazard spectrum
	Parameters:
		datasets: list of datasets. Each dataset is a (periods, intensities) tuple
		pgm: optional list or array with peak-ground-motions (to be plotted with separate symbol)
		pgm_period: float, period to plot PGM at on a logarithmic axis
		labels: list of labels for each dataset (default: [])
		colors: list of colors to plot each dataset (default: [])
		linestyles: list of line styles to plot each dataset (default: [])
		linewidths: list of line widths to plot each dataset (default: [])
		fig_filespec: full path to ouptut image. If not set, graph will be plotted on screen
			(default: None)
		title: plot title
		plot_freq: if True, horizontal axis are frequencies, if False,
			horizontal axis are periods (default: False)
		plot_style: Plot style, one of lin (or linlin), linlog, loglin, loglog
			(default: "loglin")
		Tmin: minimum period to plot in X axis (default: 0.0294 s)
		Tmax: maximum period to plot in X axis (default: 10 s)
		amin: minimum intensity to plot in Y axis (default: None)
		amax: maximum intensity to plot in Y axis (default: None)
		intensity_unit: intensity unit (default: "g")
		legend_location: location of legend (matplotlib location code) (default=0):
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
		lang: language to use for labels: en=English, nl=Dutch (default: en)
	"""
	if not labels:
		labels = ["Set %d" % (i+1) for i in range(len(datasets))]

	if not colors:
		colors = ("r", "g", "b", "c", "m")

	if not linestyles:
		linestyles = ["-"]

	if not linewidths:
		linewidths = [3]

	if plot_style.lower() in ("lin", "linlin"):
		plotfunc = pylab.plot
	elif plot_style.lower() == "linlog":
		plotfunc = pylab.semilogy
	elif plot_style.lower() == "loglin":
		plotfunc = pylab.semilogx
	elif plot_style.lower() == "loglog":
		plotfunc = pylab.loglog

	pylab.clf()

	for i, dataset in enumerate(datasets):
		periods, intensities = dataset
		if plot_style in ("loglin", "loglog") or plot_freq:
			periods = periods.clip(min=pgm_period)
		else:
			pgm_period = 0.

		if plot_freq:
			freqs = [1.0 / T for T in periods]
			xvalues = freqs
			xval_pgm = 1./pgm_period
		else:
			xvalues = periods
			xval_pgm = pgm_period

		if not Tmin:
			if not pgm in (None, []):
				Tmin = pgm_period
			else:
				Tmin = min(periods)
		if not Tmax:
			Tmax = max(periods)

		## Plot
		label = labels[i]
		color = colors[i%len(colors)]
		if len(xvalues) > 1:
			linestyle = linestyles[i%len(linestyles)]
			linewidth = linewidths[i%len(linewidths)]
			plotfunc(xvalues, intensities, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
		else:
			## Interpret line style as marker symbol
			marker = linestyles[i%len(linestyles)]
			markeredgecolor = color
			markeredgewidth = linewidths[i%len(linewidths)]
			markersize = 8
			plotfunc(xvalues, intensities, color=color, marker=marker, markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth, markersize=markersize, label=label)

		if not pgm in (None, []):
			markeredgecolor = color
			markeredgewidth = linewidths[i%len(linewidths)]
			markersize = 8
			plotfunc([xval_pgm], pgm[i], color=color, marker='o', markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth, markersize=markersize, label="_nolegend_")

	## Plot decoration
	xmin, xmax, ymin, ymax = pylab.axis()
	if amin is None:
		amin = ymin
	if amax is None:
		amax = ymax
	if plot_freq:
		pylab.xlabel({"en": "Frequency (Hz)", "nl": "Frequentie (Hz)", "fr": u"Fréquence (Hz)"}[lang], fontsize='x-large')
		pylab.axis((1.0/Tmax, 1.0/Tmin, amin, amax))
	else:
		pylab.xlabel({"en": "Period (s)", "nl": "Periode (s)", "fr": "Période (s)"}[lang], fontsize='x-large')
		pylab.axis((Tmin, Tmax, amin, amax))
	ylabel = {"en": "Acceleration", "nl": "Versnelling", "fr": u"Accélération"}[lang]
	ylabel += " (%s)" % intensity_unit
	pylab.ylabel(ylabel, fontsize='x-large')
	pylab.grid(True)
	pylab.grid(True, which="minor")
	font = FontProperties(size='large')
	pylab.legend(loc=legend_location, prop=font)
	pylab.title(title)
	ax = pylab.gca()
	for label in ax.get_xticklabels() + ax.get_yticklabels():
		label.set_size('large')

	if fig_filespec:
		pylab.savefig(fig_filespec, dpi=300)
	else:
		pylab.show()

	#pylab.clf()


def plot_histogram(intensities, weights=None, fig_filespec=None, title="", bar_color='g', amax=0, da=0.005, intensity_unit="g", lang="en"):
	"""
	Plot histogram of intensities of a number of GRA files at a single site and a single
	structural period interpolated for the specified return period
	Parameters:
		intensities: 1-D array containing intensity values for different branches of a logic tree
		weights: 1-D array containing branch weights (default: None)
		fig_filespec: full path to ouptut image. If not set, graph will be plotted on screen
			(default: None)
		title: title to appear on top of image (default: "")
		amax: maximum intensity to plot in X axis (default: 0.25)
		da: intensity bin width (default: 0.005)
		intensity_unit: intensity unit (default: "g")
		lang: language to use for labels: en=English, nl=Dutch (default: en)
	"""
	if not amax:
		amax = max(intensities) + da
	bins_acc = np.arange(0.0, amax + da, da)
	bins_N, junk = np.histogram(intensities, bins_acc, normed=False, weights=weights)

	pylab.clf()
	ax1 = pylab.subplot(111)
	plt = pylab.bar(bins_acc[:-1], bins_N, width=da, color=bar_color)

	xlabel = {"en": "Acceleration", "nl": "Versnelling"}[lang]
	xlabel += " (%s)" % intensity_unit
	pylab.xlabel(xlabel, fontsize='x-large')
	pylab.ylabel({"en": "Number of models", "nl": "Aantal modellen"}[lang], fontsize='x-large')

	## Cumulative density function
	cdf = np.add.accumulate(bins_N) * 1.0
	cdf /= cdf.max()
	cdf = np.concatenate([cdf, [1.0]])
	ax2 = pylab.twinx()
	pylab.plot(bins_acc, cdf, 'm', linewidth=3.0)
	pylab.ylabel({"en": "Normalized cumulative number", "nl": "Genormaliseerd cumulatief aantal"}[lang], fontsize='x-large')
	ax2.yaxis.tick_right()

	mean_acc = np.average(intensities, weights=weights)
	print "Mean: %.3f" % mean_acc
	pylab.plot([mean_acc, mean_acc], [0.0, 1.0], 'r--', linewidth=3)

	majorLocator = MultipleLocator(0.1)
	majorFormatter = FormatStrFormatter('%.1f')
	minorLocator = MultipleLocator(0.01)
	ax2.xaxis.set_major_locator(majorLocator)
	ax2.xaxis.set_major_formatter(majorFormatter)
	ax2.xaxis.set_minor_locator(minorLocator)

	#title += "\n%s: %d %s" % ({"en": "Return period", "nl": "Terugkeerperiode"}[lang], return_period, {"en": "yr", "nl": "jaar"}[lang])
	pylab.title(title)
	xmin, xmax, ymin, ymax = pylab.axis()
	pylab.axis((xmin, amax, ymin, ymax))
	pylab.grid(True)
	for label in ax1.get_xticklabels() + ax1.get_yticklabels(): #+ ax2.get_yticklabels():
		label.set_size('large')

	if fig_filespec:
		pylab.savefig(fig_filespec, dpi=300)
	else:
		pylab.show()

	#pylab.clf()


def plot_deaggregation(mr_values, magnitudes, distances, return_period, eps_values=None, eps_bin_edges=[1.0, 1.5, 2.0, 2.5, 3.0], fue_values=None, fue_labels=None, mr_style="2D", site_name="", struc_period=None, title_comment="", fig_filespec=None):
	"""
	Plot deaggregation results
	Parameters:
		mr_values: 2-D array [r, m] of exceedance rates binned by distance and magnitude
		magnitudes: array of magnitude values (magnitude bin edges)
		distances: array of distance values (distance bin edges)
		return_period: return period for which deaggregation results are given
			(mainly required to add information to plot title)
		eps_values: array of exceedance rates by epsilon (default: None)
		eps_bin_edges: list or array with epsilon bin edges
			(default: [1.0, 1.5, 2.0, 2.5, 3.0])
		fue_values: array of exceedance rates by source (default: None)
		fue_labels: list of source names to be used as labels (default: None)
		mr_style: plotting style for M,r deaggregation results, either "2D" or "3D"
			(default: "2D")
			Note: 3D style has some problems in current version (0.99.1.1) of matplotlib
		site_name: name of site to be added to the plot title (default: "")
		struc_period: structural period for which deaggregation results are given,
			to be added to the plot title (default: None, will print "PGA")
		title_comment: additional comment to be added to plot title (default: "")
		fig_filespec: full path to ouptut image. If not set, graph will be plotted on screen
			(default: None)
	"""
	from matplotlib import cm

	Nmag, Ndist = len(magnitudes), len(distances)

	## Normalize M,r deaggregation results
	mr_values *= 100

	## Plot deaggregation by M,r
	fig = pylab.figure()
	pylab.clf()

	if eps_values not in (None, []) or fue_values not in (None, []):
		mr_rect = [0.1, 0.0, 0.5, 0.9]
	else:
		mr_rect = [0.1, 0.0, 0.85, 0.9]

	if mr_style == "3D":
		import mpl_toolkits.mplot3d
		ax1 = mpl_toolkits.mplot3d.Axes3D(fig, rect=mr_rect)

		xpos, ypos = np.meshgrid(distances, magnitudes)
		#print mr_values.shape, xpos.shape
		xpos = xpos.flatten()
		ypos = ypos.flatten()
		zpos = np.zeros_like(xpos)

		dx = 7.5 * np.ones_like(zpos)
		dy = 0.1 * np.ones_like(zpos)
		dz = mr_values.transpose().flatten()

		## Draw all bars at once in a single color
		#ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b')

		## Draw bars using different colors for different distance bins
		cmap = cm.get_cmap('jet')
		colors = []
		for i in range(Ndist):
			colors.append(cmap(i/(Ndist-1.)))
		#colors.reverse()

		j = 0
		for xs, ys, zs, dxs, dys, dzs in zip(xpos, ypos, zpos, dx, dy, dz):
			color = colors[j%len(colors)]
			ax1.bar3d(xs, ys, zs, dxs, dys, dzs, color=color)
			j += 1

		ax1.set_xlabel('Distance (km)')
		ax1.set_ylabel('Magnitude (Ms)')
		#ax1.set_zlabel('Hazard Contribution (1/yr)')
		ax1.set_zlabel('Hazard Contribution (%)')

	else:
		## 2D style
		ax1 = fig.add_axes(mr_rect)
		cmap = cm.Spectral_r
		#cmap = cm.jet
		cmap.set_under('w', 1.0)
		extent = [distances[0], distances[-1], magnitudes[0], magnitudes[-1]]
		img = ax1.imshow(mr_values[::-1,:], vmin=0.1, cmap=cmap, interpolation="nearest", aspect="auto", extent=extent)
		xticks = [0, 50, 100, 150, 200, 250, 300]
		ax1.set_xticks(xticks)
		ax1.set_xticklabels(xticks)
		ax1.set_xlabel("Source-Site Distance (km)")
		yticks = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
		ax1.set_yticks(yticks)
		ax1.set_yticklabels(yticks)
		ax1.set_ylabel("Magnitude ($M_S$)")
		pylab.grid(True)
		cb = pylab.colorbar(img, ax=ax1, orientation="horizontal")
		cb.set_label("Hazard Contribution (%)")

	ax1.set_title('By M,r')

	## Plot deaggregation by epsilon
	if eps_values not in (None, []):
		ax2 = fig.add_axes([0.575, 0.5, 0.4, 0.4], aspect='equal')
		#eps_values = np.concatenate([[0.], eps_values])
		#eps_values = eps_values[1:] - eps_values[:-1]
		#eps_values /= eps_values[-1]
		eps_labels = map(str, eps_bin_edges)
		cmap = cm.Spectral_r
		#cmap = cm.jet
		colors = []
		for i in range(len(eps_values)):
			colors.append(cmap(i/(len(eps_values)-1.)))
		ax2.pie(eps_values, labels=eps_labels, autopct='%1.1f%%', shadow=False, colors=colors)
		ax2.set_title('By $\epsilon$')

	## Plot deaggregation by source
	if fue_values not in (None, []):
		num_sources = len(fue_values)
		if not fue_labels:
			fue_labels = range(1,num_sources+1)
		ax3 = fig.add_axes([0.575, 0.0, 0.4, 0.4], aspect='equal')
		fue_values /= np.add.reduce(fue_values)
		large_slice_indexes = np.where(fue_values >= 0.01)
		small_slice_indexes = np.where(fue_values < 0.01)
		if len(small_slice_indexes[0]):
			small_slice_contribution = np.add.reduce(fue_values[small_slice_indexes])
			fue_values = fue_values[large_slice_indexes]
			fue_values = np.concatenate([fue_values, [small_slice_contribution]])
			#small_slice_label = ",".join(["%s" % lbl for i, lbl in enumerate(fue_labels) if i in small_slice_indexes[0]])
			small_slice_label = "Other"
			fue_labels = [lbl for i, lbl in enumerate(fue_labels) if i in large_slice_indexes[0]]
			fue_labels.append(small_slice_label)
		cmap = cm.Accent
		cdict = cmap._segmentdata
		reds = [item[1] for item in cdict['red']]
		greens = [item[1] for item in cdict['green']]
		blues = [item[1] for item in cdict['blue']]
		## Make sure same zones have same colors
		num_colors = len(reds)
		if num_sources > num_colors:
			color_ar = np.arange(num_colors, dtype='f') / (num_colors - 1)
			source_ar = np.arange(num_sources, dtype='f') / (num_sources -1)
			reds = interpolate(color_ar, reds, source_ar)
			greens = interpolate(color_ar, greens, source_ar)
			blues = interpolate(color_ar, blues, source_ar)
		colors = np.asarray(zip(reds, greens, blues))
		colors = np.concatenate([colors[large_slice_indexes], np.ones((1,3), 'f')])
		ax3.pie(fue_values, labels=fue_labels, autopct='%1.1f%%', shadow=False, colors=colors)
		ax3.set_title('By source')

	## Finish plot
	fig_title = 'Deaggregation results ('
	if site_name:
		fig_title += 'Site: %s, ' % site_name
	if struc_period:
		struc_period_name = "T=%.2f s" % struc_period
	else:
		struc_period_name = "PGA"
	if title_comment:
		fig_title = title_comment
	else:
		fig_title += ' %s, $T_R$=%d yr' % (struc_period_name, return_period)
	fig.canvas.set_window_title('Deaggregation Results')
	pylab.gcf().text(0.5, 0.95, fig_title, horizontalalignment='center', fontproperties=FontProperties(size=15))
	if fig_filespec:
		pylab.savefig(fig_filespec, dpi=300)
	else:
		pylab.show()

	#pylab.clf()
