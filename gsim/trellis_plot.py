"""
GMPE trellis plots
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from plotting.generic_mpl import (create_multi_plot, plot_xy, show_or_save_plot)
from ..utils import logrange
from .plot import (get_imt_label, IMT_UNIT_TO_PLOT_LABEL, plot_spectrum)
from .gmpe import GMPE
from . import (gmpes, oqhazlib_gmpe)


__all__ = ['plot_trellis_distance', 'plot_trellis_spectrum']


def plot_trellis_distance(gmpe_list, mag_list, period_list,
						dmin=1., dmax=200., h=0.,
						pgm_imt="PGA", imt_unit="g", epsilon=0,
						soil_type="rock", vs30=None, kappa=None,
						mechanism="normal", damping=5,
						xscaling='log', yscaling='log', xgrid=1, ygrid=1,
						legend_colrow=(0, 0), **kwargs):
	"""
	Trellis plot comparing ground-motion vs distance curves for a set
	of GMPEs (different colors), for different magnitudes (organized
	in columns) and different spectral periods (organized in rows).

	Note that ground motions are computed using a point source,
	to minimize differences due to different GMPE distance metrics

	:param gmpe_list:
		list with strings (GMPE names) or instances of :class:`rshalib.GMPE`
	:param mag_list:
		list of floats, earthquake magnitudes
	:param period_list:
		list of floats, spectral periods (in s)
	:param dmin:
		float, minimum distance to plot (in km)
		(default: 1.)
	:param dmax:
		float, maximum distance to plot (in km)
		(default: 200.)
	:param h:
		float, focal depth (in km)
		(default: 0, making Rhypo=Repi and Rrup=RJB)
	:param pgm_imt:
		str, IMT for peak ground motion (T=0): 'PGA', 'PGV' or 'PGD':
		(default: 'PGA')
	:param imt_unit:
		str, unit in which intensities should be expressed,
		depends on IMT
		(default: "g")
	:param epsilon:
		float, number of standard deviations above or below the mean to
		compute ground motion for
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
	:param legend_colrow:
		(col, row) tuple of ints: in which column and row the legend
		should be plotted
		(default: (0, 0))
	:param kwargs:
		additional keyword arguments understood by
		:func:`generic_mpl.create_multi_plot`,
		:func:`generic_mpl.plot_xy`
		or :runc:`generic_mpl.show_or_save_plot`

	:return:
		instance of :class:`matplotlib.Figure`
	"""
	num_rows = len(period_list)
	num_cols = len(mag_list)

	col_titles = kwargs.pop('col_titles', ['M = %.1f' % mag for mag in mag_list])
	row_titles = kwargs.pop('row_titles', ['T = %.2f s' % T for T in period_list])

	num_dist = 50
	if xscaling == 'log':
		if dmin == 0:
			dmin = 0.1

	kwargs['xmin'] = kwargs.get('xmin', dmin)
	kwargs['xmax'] = kwargs.get('xmax', dmax)
	kwargs['sharex'] = kwargs.get('sharex', True)
	kwargs['sharey'] = kwargs.get('sharey', True)
	kwargs['xlabel'] = kwargs.get('xlabel', 'Distance (km)')
	ylabel = ('\nSpectral Acceleration (%s)'
				% IMT_UNIT_TO_PLOT_LABEL.get(imt_unit, imt_unit))
	kwargs['ylabel'] = kwargs.get('ylabel', ylabel)
	kwargs['share_xlabel'] = kwargs.get('share_xlabel', True)
	kwargs['share_ylabel'] = kwargs.get('share_ylabel', True)
	kwargs['ylabel_side'] = kwargs.get('ylabel_side', 'right')

	mpl_kwargs = {}
	for key in ('wspace', 'hspace', 'width_ratios', 'height_ratios',
				'labels', 'label_font', 'label_location',
				'xtick_direction', 'xtick_side', 'xlabel_side',
				'ytick_direction', 'ytick_side', 'ylabel_side',
				'sharex', 'share_xlabel', 'sharey', 'share_ylabel',
				'xlabel', 'ylabel', 'ax_label_font', 'col_row_title_font',
				'hide_axes', 'title', 'title_font', 'ax_size'):
		if key in kwargs:
			mpl_kwargs[key] = kwargs.pop(key)
	for key in ('xmin', 'xmax', 'ymin', 'ymax', 'dpi'):
		if key in kwargs:
			mpl_kwargs[key] = kwargs[key]
	save_kwargs = {}
	for key in ('fig_filespec', 'border_width'):
		if key in kwargs:
			save_kwargs[key] = kwargs.pop(key)
	kwargs.pop('ax', None)

	fig = create_multi_plot(num_rows, num_cols, col_titles=col_titles,
							row_titles=row_titles, **mpl_kwargs)

	_gmpe_list = gmpe_list
	gmpe_list = []
	for gmpe in _gmpe_list:
		if not isinstance(gmpe, GMPE):
			try:
				gmpe = getattr(oqhazlib_gmpe, gmpe)()
			except:
				gmpe = getattr(gmpes, gmpe+'GMPE')()
			gmpe_list.append(gmpe)

	gmpe_labels = [gmpe.name for gmpe in gmpe_list]

	for col, M in enumerate(mag_list):
		for row, T in enumerate(period_list):
			if T == 0:
				imt = pgm_imt
			else:
				imt = 'SA'
			ax = fig.axes[col + row*num_cols]
			datasets = []
			if col == legend_colrow[0] and row == legend_colrow[1]:
				labels = gmpe_labels
			else:
				labels = []
			for gmpe in gmpe_list:
				if xscaling == 'log':
					distances = logrange(max(dmin, gmpe.dmin),
										min(dmax, gmpe.dmax), num_dist)
				else:
					distances = np.linspace(max(dmin, gmpe.dmin),
											min(dmax, gmpe.dmax), num_dist)
				Avalues = gmpe(M, distances, h=h, imt=imt, T=T, imt_unit=imt_unit,
								epsilon=epsilon, soil_type=soil_type, vs30=vs30,
								kappa=kappa, mechanism=mechanism, damping=damping)
				datasets.append((distances, Avalues))

			plot_xy(datasets, labels=labels, xscaling=xscaling, yscaling=yscaling,
					xgrid=xgrid, ygrid=ygrid, ax=ax, fig_filespec='wait',
					**kwargs)

	return show_or_save_plot(fig, **save_kwargs)


def plot_trellis_spectrum(gmpe_list, mag_list, distance_list, h=0.,
						imt_unit="g", epsilon=0,
						soil_type="rock", vs30=None, kappa=None,
						mechanism="normal", damping=5,
						include_pgm=True, pgm_period=1./50, pgm_marker='o',
						plot_freq=False, linewidths=[1., 0.5], linestyle='-',
						xscaling='log', yscaling='log', xgrid=1, ygrid=1,
						legend_colrow=(0, 0), **kwargs):
	"""
	Trellis plot comparing ground-motion spectra for a set
	of GMPEs (different colors), for different magnitudes (organized
	in columns) and different distances (organized in rows).

	Note that ground motions are computed using a point source,
	to minimize differences due to different GMPE distance metrics

	:param gmpe_list:
		list with strings (GMPE names) or instances of :class:`rshalib.GMPE`
	:param mag_list:
		list of floats, earthquake magnitudes
	:param distance:
		list of floats, distances (in km)
	:param h:
		float, focal depth (in km)
		(default: 0, making Rhypo=Repi and Rrup=RJB)
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
	:param linewidths:
		list of 2 floats, linewidths for mean and epsilon spectral curves.
		(default: [1., 0.5])
	:param linestyle:
		str, linestyle for spectral curves.
		Note that it is currently not possible to specify different
		line styles for different GMPEs
		(default: '-')
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
	:param legend_colrow:
		(col, row) tuple of ints: in which column and row the legend
		should be plotted
		(default: (0, 0))
	:param kwargs:
		additional keyword arguments understood by
		:func:`generic_mpl.create_multi_plot`,
		:func:`generic_mpl.plot_xy`
		or :runc:`generic_mpl.show_or_save_plot`

	:return:
		instance of :class:`matplotlib.Figure`
	"""
	num_rows = len(mag_list)
	num_cols = len(distance_list)

	col_titles = kwargs.pop('col_titles', ['M = %.1f' % mag for mag in mag_list])
	row_titles = kwargs.pop('row_titles', ['D = %.0f km' % d for d in distance_list])

	kwargs['sharex'] = kwargs.get('sharex', True)
	kwargs['sharey'] = kwargs.get('sharey', True)
	if plot_freq:
		xlabel = 'Frequency (Hz)'
	else:
		xlabel = 'Period (s)'
	kwargs['xlabel'] = kwargs.get('xlabel', xlabel)
	ylabel = ('\nSpectral Acceleration (%s)'
				% IMT_UNIT_TO_PLOT_LABEL.get(imt_unit, imt_unit))
	kwargs['ylabel'] = kwargs.get('ylabel', ylabel)
	kwargs['share_xlabel'] = kwargs.get('share_xlabel', True)
	kwargs['share_ylabel'] = kwargs.get('share_ylabel', True)
	kwargs['ylabel_side'] = kwargs.get('ylabel_side', 'right')

	mpl_kwargs = {}
	for key in ('wspace', 'hspace', 'width_ratios', 'height_ratios',
				'labels', 'label_font', 'label_location',
				'xtick_direction', 'xtick_side', 'xlabel_side',
				'ytick_direction', 'ytick_side', 'ylabel_side',
				'sharex', 'share_xlabel', 'sharey', 'share_ylabel',
				'xlabel', 'ylabel', 'ax_label_font', 'col_row_title_font',
				'hide_axes', 'title', 'title_font', 'ax_size'):
		if key in kwargs:
			mpl_kwargs[key] = kwargs.pop(key)
	for key in ('xmin', 'xmax', 'ymin', 'ymax', 'dpi'):
		if key in kwargs:
			mpl_kwargs[key] = kwargs[key]
	save_kwargs = {}
	for key in ('fig_filespec', 'border_width'):
		if key in kwargs:
			save_kwargs[key] = kwargs.pop(key)
	kwargs.pop('ax', None)
	kwargs['linewidths'] = linewidths
	kwargs['linestyles'] = [linestyle]

	fig = create_multi_plot(num_rows, num_cols, col_titles=col_titles,
							row_titles=row_titles, **mpl_kwargs)

	_gmpe_list = gmpe_list
	gmpe_list = []
	for gmpe in _gmpe_list:
		if not isinstance(gmpe, GMPE):
			try:
				gmpe = getattr(oqhazlib_gmpe, gmpe)()
			except:
				gmpe = getattr(gmpes, gmpe+'GMPE')()
			gmpe_list.append(gmpe)

	gmpe_labels = [gmpe.name for gmpe in gmpe_list]

	imt = 'SA'
	for col, M in enumerate(mag_list):
		for row, D in enumerate(distance_list):
			ax = fig.axes[col + row*num_cols]
			datasets = []
			if col == legend_colrow[0] and row == legend_colrow[1]:
				labels = gmpe_labels
			else:
				labels = ['_nolegend_'] * len(gmpe_list)

			plot_spectrum(gmpe_list, [M], D, h=h, imt=imt, imt_unit=imt_unit,
						epsilon=epsilon, soil_type=soil_type, vs30=vs30,
						kappa=kappa, mechanism=mechanism, damping=damping,
						include_pgm=include_pgm, pgm_period=pgm_period,
						pgm_marker=pgm_marker, plot_freq=plot_freq,
						labels=labels, xscaling=xscaling, yscaling=yscaling,
						xgrid=xgrid, ygrid=ygrid, fig_filespec='wait',
						ax=ax, **kwargs)

	return show_or_save_plot(fig, **save_kwargs)
