# -*- coding: iso-Latin-1 -*-

"""
plot MFDs
"""

from __future__ import absolute_import, division, print_function, unicode_literals

try:
	## Python 2
	basestring
except:
	## Python 3
	basestring = str


import datetime

import numpy as np
import pylab
from matplotlib.font_manager import FontProperties



__all__ = ['plot_mfds']

# TODO: reimplement using generic_mpl!

def plot_mfds(mfd_list, colors=[], styles=[], labels=[], discrete=[], cumul_or_inc=[],
			completeness=None, end_year=None, Mrange=(), Freq_range=(), title="",
			lang="en", y_log_labels=True, legend_location=1,
			fig_filespec=None, ax=None, fig_width=0, dpi=300):
	"""
	Plot one or more magnitude-frequency distributions

	:param mfd_list:
		List with instance of :class:`EvenlyDiscretizedMFD`
		or :class:`TruncatedGRMFD`
	:param colors:
		List with matplotlib color specifications, one for each mfd
		(default: [])
	:param styles:
		List with matplotlib symbol styles or line styles, one for each mfd
		(default: [])
	:param labels:
		List with plot labels, one for each mfd
		(default: [])
	:param discrete:
		List of bools, whether or not to plot discrete MFD's
		(default: [])
	:param cumul_or_inc:
		List of strings, either "cumul", "inc" or "both", indicating
		whether to plot cumulative MFD, incremental MFD or both
		(default: [])
	:param completeness:
		instance of :class:`Completeness`, used to plot completeness
		limits
		(default: None)
	:param end_year:
		int, end year of catalog (used when plotting completeness limits)
		(default: None, will use current year)
	:param Mrange:
		(Mmin, Mmax) tuple, minimum and maximum magnitude in X axis
		(default: ())
	:param Freq_range:
		(Freq_min, Freq_max) tuple, minimum and maximum values in
		frequency (Y) axis
		(default: ())
	:param title:
		str, plot title
		(default: "")
	:param lang:
		str, language of plot axis labels
		(default: "en")
	:param legend_location:
		int or str, matplotlib specification for legend location
		(default: 1)
	:param y_log_labels:
		bool, whether or not Y axis labels are plotted as 10 to a power
		(default: True)
	:param fig_filespec:
		str, full path to output image file, if None plot to screen
		(default: None)
	:param ax:
		instance of :class:`~matplotlib.axes.Axes` in which plot will
		be made
		(default: None, will create new figure and axes)
	:param fig_width:
		float, figure width in cm, used to recompute :param:`dpi` with
		respect to default figure width
		(default: 0)
	:param dpi:
		int, image resolution in dots per inch
		(default: 300)

	:return:
		if both :param:`ax` and :param:`fig_filespec` are None, a
		(ax, fig) tuple will be returned
	"""
	if ax is None:
		## Note: clf call seems to create a figure as well
		pylab.clf()
		fig = pylab.gcf()
		#fig = pylab.figure()
		ax = fig.add_subplot(1, 1, 1)
		interactive = True
	else:
		#ax.cla()
		fig = ax.get_figure()
		interactive = False

	if not colors:
		colors = ("r", "g", "b", "c", "m", "k")

	if not labels:
		labels = [""] * len(mfd_list)

	if isinstance(discrete, bool):
		discrete = [discrete] * len(mfd_list)

	if isinstance(cumul_or_inc, basestring):
		cumul_or_inc = [cumul_or_inc] * len(mfd_list)

	## Plot
	## Line below removed because matplotlib crashes if this function is
	## called more than once
	#fig = pylab.figure()

	for i, mfd in enumerate(mfd_list):
		color = colors[i % len(colors)]

		try:
			want_discrete = discrete[i]
		except:
			if isinstance(mfd, TruncatedGRMFD):
				want_discrete = False
			else:
				want_discrete = True

		try:
			cumul_or_inc[i]
		except:
			if isinstance(mfd, TruncatedGRMFD):
				want_cumulative = True
				want_incremental = False
			else:
				want_cumulative = True
				want_incremental = True
		else:
			if cumul_or_inc[i] == "cumul":
				want_cumulative = True
				want_incremental = False
			elif cumul_or_inc[i] == "inc":
				want_cumulative = False
				want_incremental = True
			else:
				want_cumulative = True
				want_incremental = True

		## Discrete MFD
		if want_discrete:
			try:
				symbol = styles[i]
			except:
				symbol = 'o'
			else:
				if symbol in ("", None, "-", "--", ":", ":."):
					symbol = "o"

			## Cumulative
			if want_cumulative:
				label = labels[i]
				if want_incremental:
					label += " (cumul.)"
				obj_list = ax.semilogy(mfd.get_magnitude_bin_edges(),
								mfd.get_cumulative_rates(), symbol, label=label)
				pylab.setp(obj_list, markersize=10.0, markeredgewidth=1.0,
							markeredgecolor='k', markerfacecolor=color)

			## Incremental
			if want_incremental:
				label = labels[i] + " (inc.)"
				obj_list = ax.semilogy(mfd.get_magnitude_bin_centers(),
										mfd.occurrence_rates, symbol, label=label)
				pylab.setp(obj_list, markersize=10.0, markeredgewidth=1.0,
							markeredgecolor=color, markerfacecolor="None")

		## Continuous MFD
		else:
			try:
				linestyle = styles[i]
			except:
				linestyle = "-"
			else:
				if linestyle in ("", None) or not linestyle in ("-", "--", ":", ":."):
					linestyle = "-"

			## Cumulative
			if want_cumulative:
				label = labels[i]
				if want_incremental:
					label += " (cumul.)"
				ax.semilogy(mfd.get_magnitude_bin_edges(), mfd.get_cumulative_rates(),
							color=color, linestyle=linestyle, lw=3, label=label)

			## Incremental
			if want_incremental:
				label = labels[i] + " (inc.)"
				ax.semilogy(mfd.get_magnitude_bin_centers(), mfd.occurrence_rates,
							color=color, linestyle=linestyle, lw=1, label=label)

	if not Mrange:
		Mrange = pylab.axis()[:2]
	if not Freq_range:
		Freq_range = pylab.axis()[2:]

	## Plot limits of completeness
	if completeness:
		annoty = Freq_range[0] * 10**0.5
		bbox_props = dict(boxstyle="round,pad=0.4", fc="w", ec="k", lw=1)
		## Make sure min_mags is not sorted in place,
		## otherwise completeness object may misbehave
		min_mags = np.sort(completeness.min_mags)
		if not end_year:
			end_year = datetime.date.today().year
		for i in range(1, len(min_mags)):
			ax.plot([min_mags[i], min_mags[i]], Freq_range, 'k--', lw=1,
					label="_nolegend_")
			ax.annotate("", xy=(min_mags[i-1], annoty), xycoords='data',
						xytext=(min_mags[i], annoty), textcoords='data',
						arrowprops=dict(arrowstyle="<->"),)
			#label = "%s - %s" % (completeness.get_initial_completeness_year(min_mags[i-1]), end_year)
			label = "%s - " % completeness.get_initial_completeness_year(min_mags[i-1])
			## Add final year of completeness in case of non-monotonically decreasing
			## completeness magnitudes
			final_year = completeness.get_final_completeness_date(min_mags[i-1])
			if final_year:
				label += ("%s" % final_year)
			ax.text(np.mean([min_mags[i-1], min_mags[i]]), annoty*10**-0.25,
					label, ha="center", va="center", size=12, bbox=bbox_props)
		## Uniform completeness has only 1 min_mag
		if len(min_mags) == 1:
			i = 0
		ax.annotate("", xy=(min_mags[i], annoty), xycoords='data',
					xytext=(min(mfd.max_mag, Mrange[1]), annoty), textcoords='data',
					arrowprops=dict(arrowstyle="<->"),)
		label = "%s - %s"
		label %= (completeness.get_initial_completeness_year(min_mags[i]), end_year)
		ax.text(np.mean([min_mags[i], mfd.max_mag]), annoty*10**-0.25,
				label, ha="center", va="center", size=12, bbox=bbox_props)

	## Apply plot limits
	ax.axis((Mrange[0], Mrange[1], Freq_range[0], Freq_range[1]))

	ax.set_xlabel("Magnitude ($M_%s$)" % mfd.Mtype[1].upper(), fontsize="x-large")
	label = {"en": "Annual number of earthquakes",
			"nl": "Aantal aardbevingen per jaar",
			"fr": u"Nombre de séismes par année"}.get(lang.lower())
	if label is not None:
		ax.set_ylabel(label, fontsize="x-large")
	if title is not None:
		ax.set_title(title, fontsize='x-large')
	ax.grid(True)
	if labels:
		font = FontProperties(size='medium')
		ax.legend(loc=legend_location, prop=font)
	for label in ax.get_xticklabels() + ax.get_yticklabels():
		label.set_size('large')

	if not y_log_labels:
		from matplotlib.ticker import FuncFormatter
		ax.yaxis.set_major_formatter(FuncFormatter((lambda y, _: '{:g}'.format(y))))
		for label in ax.get_xticklabels() + ax.get_yticklabels():
			label.set_size('medium')

	if fig_filespec:
		default_figsize = pylab.rcParams['figure.figsize']
		default_dpi = pylab.rcParams['figure.dpi']
		if fig_width:
			fig_width /= 2.54
			dpi = dpi * (fig_width / default_figsize[0])

		fig.savefig(fig_filespec, dpi=dpi)
	elif interactive:
		pylab.show()
	else:
		return ax, fig
