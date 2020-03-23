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


import numpy as np
import pylab
from plotting.generic_mpl import (plot_xy, show_or_save_plot)

from .truncated_gr import TruncatedGRMFD
from .evenly_discretized import EvenlyDiscretizedMFD


__all__ = ['plot_mfds']



def plot_mfds(mfd_list, labels=[], colors=[], styles=[], lw_or_ms=[],
				discrete=[], cumul_or_inc=[],
				completeness=None, end_year=None,
				xgrid=1, ygrid=1,
				title="", lang="en", legend_location=1,
				fig_filespec=None, **kwargs):
	"""
	"""
	COLORS = colors or pylab.rcParams['axes.prop_cycle'].by_key()['color']
	LABELS = labels or [""] * len(mfd_list)

	if not isinstance(styles, list):
		styles = [styles]
	if not isinstance(lw_or_ms, list):
		lw_or_ms = [lw_or_ms]

	if isinstance(discrete, bool):
		discrete = [discrete]
	if len(discrete) == 1:
		discrete *= len(mfd_list)
	if len(discrete) == 0:
		discrete = [not isinstance(mfd, TruncatedGRMFD) for mfd in mfd_list]

	if isinstance(cumul_or_inc, basestring):
		cumul_or_inc = [cumul_or_inc]
	if len(cumul_or_inc) == 1:
		cumul_or_inc *= len(mfd_list)
	if len(cumul_or_inc) == 0:
		cumul_or_inc = ['cumul' if isinstance(mfd, TruncatedGRMFD) else 'both'
						for mfd in mfd_list]

	datasets, labels, colors, marker_edge_colors = [], [], [], []
	linestyles, linewidths, markers, marker_sizes = [], [], [], []
	for i, mfd in enumerate(mfd_list):
		color = COLORS[i % len(COLORS)]

		want_discrete = discrete[i]
		want_cumulative = True if cumul_or_inc[i] in ('cumul', 'both') else False
		want_incremental = True if cumul_or_inc[i] in ('inc', 'both') else False

		## Discrete MFD
		if want_discrete:
			try:
				symbol = styles[i]
			except IndexError:
				symbol = 'o'
			else:
				if symbol in ("", None, "-", "--", ":", ":."):
					symbol = "o"

			try:
				marker_size = lw_or_ms[i] or 8
			except IndexError:
				marker_size = 8

			## Cumulative (filled symbols)
			if want_cumulative:
				label = LABELS[i]
				if want_incremental:
					label += " (cumul.)"
				labels.append(label)
				linestyles.append(None)
				linewidths.append(0)
				markers.append(symbol)
				colors.append(color)
				marker_edge_colors.append('k')
				marker_sizes.append(marker_size)
				datasets.append((mfd.get_magnitude_bin_edges(),
								mfd.get_cumulative_rates()))

			## Incremental (open symbols)
			if want_incremental:
				label = LABELS[i] + " (inc.)"
				labels.append(label)
				linestyles.append(None)
				linewidths.append(0)
				markers.append(symbol)
				colors.append('')
				marker_edge_colors.append(color)
				marker_sizes.append(marker_size)
				datasets.append((mfd.get_magnitude_bin_centers(),
								mfd.occurrence_rates))

		## Continuous MFD
		else:
			try:
				linestyle = styles[i]
			except IndexError:
				linestyle = "-"
			else:
				if linestyle in ("", None) or not linestyle in ("-", "--", ":", ":."):
					linestyle = "-"

			try:
				linewidth = lw_or_ms[i] or 2.5
			except IndexError:
				linewidth = 2.5

			## Cumulative (thick lines)
			if want_cumulative:
				label = LABELS[i]
				if want_incremental:
					label += " (cumul.)"
				labels.append(label)
				linestyles.append(linestyle)
				linewidths.append(linewidth)
				markers.append('')
				colors.append(color)
				marker_edge_colors.append('None')
				marker_sizes.append(0)
				datasets.append((mfd.get_magnitude_bin_edges(),
								mfd.get_cumulative_rates()))

			## Incremental (thin lines)
			if want_incremental:
				label = LABELS[i] + " (inc.)"
				labels.append(label)
				linestyles.append(linestyle)
				linewidths.append(1)
				markers.append('')
				colors.append(color)
				marker_edge_colors.append('None')
				marker_sizes.append(0)
				datasets.append((mfd.get_magnitude_bin_centers(),
								mfd.occurrence_rates))

	## Plot decoration
	xlabel = kwargs.pop('xlabel', "Magnitude ($M_%s$)" % mfd.Mtype[1].upper())
	ylabel = kwargs.pop('ylabel', None)
	if ylabel is None:
		ylabel = {"en": "Annual number of earthquakes",
				"nl": "Aantal aardbevingen per jaar",
				"fr": u"Nombre de séismes par année"}.get(lang.lower())

	xscaling = kwargs.pop('xscaling', 'lin')
	yscaling = kwargs.pop('yscaling', 'log')

	## Determine plot limits
	mfd = mfd_list[0]
	mag_bin_edges = mfd.get_magnitude_bin_edges()
	Mmin, Mmax = mag_bin_edges[0], mag_bin_edges[-1] + mfd.bin_width
	cumul_rates = mfd.get_cumulative_rates()
	fmax, fmin = cumul_rates[0], cumul_rates[-1]
	for mfd in mfd_list[1:]:
		mag_bin_edges = mfd.get_magnitude_bin_edges()
		_Mmin, _Mmax = mag_bin_edges[0], mag_bin_edges[-1] + mfd.bin_width
		if _Mmin < Mmin:
			Mmin = _Mmin
		if _Mmax > Mmax:
			Mmax = _Mmax
		cumul_rates = mfd.get_cumulative_rates()
		_fmax, _fmin = cumul_rates[0], cumul_rates[-1]
		if _fmin < fmin:
			fmin = _fmin
		if _fmax > fmax:
			fmax = _fmax

	xmin = kwargs.pop('xmin', Mmin)
	xmax = kwargs.pop('xmax', Mmax)

	if yscaling == 'log':
		fmin = 10**np.floor(np.log10(fmin))
		fmax = 10**np.ceil(np.log10(fmax))
	ymin = kwargs.pop('ymin', fmin)
	ymax = kwargs.pop('ymax', fmax)

	ax = plot_xy(datasets, labels=labels, colors=colors,
				linestyles=linestyles, linewidths=linewidths,
				markers=markers, marker_sizes=marker_sizes,
				marker_edge_colors=marker_edge_colors,
				xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
				xscaling=xscaling, yscaling=yscaling,
				xlabel=xlabel, ylabel=ylabel, xgrid=xgrid, ygrid=xgrid,
				legend_location=legend_location, title=title,
				fig_filespec='wait', **kwargs)

	## Plot limits of completeness
	if completeness:
		import datetime

		annoty = ymin * 10**0.5
		bbox_props = dict(boxstyle="round,pad=0.4", fc="w", ec="k", lw=1)
		## Make sure min_mags is not sorted in place,
		## otherwise completeness object may misbehave
		min_mags = np.sort(completeness.min_mags)
		if not end_year:
			end_year = datetime.date.today().year
		for i in range(1, len(min_mags)):
			ax.plot([min_mags[i], min_mags[i]], [ymin, ymax], 'k--', lw=1,
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
					xytext=(min(mfd.max_mag, xmax), annoty), textcoords='data',
					arrowprops=dict(arrowstyle="<->"),)
		label = "%s - %s"
		label %= (completeness.get_initial_completeness_year(min_mags[i]), end_year)
		ax.text(np.mean([min_mags[i], mfd.max_mag]), annoty*10**-0.25,
				label, ha="center", va="center", size=12, bbox=bbox_props)

	return show_or_save_plot(ax, fig_filespec=fig_filespec, dpi=kwargs.get('dpi'),
							border_width=kwargs.get('border_width'))
