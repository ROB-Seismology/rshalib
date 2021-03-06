# -*- coding: utf-8 -*-
# pylint: disable=W0142, W0312, C0103, R0913
"""
HazardCurve and higher classes
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import int

try:
	## Python 2
	basestring
except:
	## Python 3
	basestring = str


### imports
import os, sys
import re
from lxml import etree

import numpy as np
import pylab
from scipy.stats import scoreatpercentile

from ..nrml import ns
from ..nrml.common import create_nrml_root, xmlstr
from ..utils import interpolate, logrange
from ..pmf import NumericPMF

from .plot import plot_hazard_curves, plot_histogram
from .base_array import *
from .hc_base import *
from .uhs import UHS, UHSFieldSet, UHSFieldTree
from .hazard_map import HazardMapSet


# TODO: unit names should be the same as robspy !!!
# TODO: change order IMT, intensities, intensity_unit --> intensities, intensity_unit, IMT


__all__ = ['HazardCurve', 'SpectralHazardCurve',
			'HazardCurveField', 'SpectralHazardCurveField',
			'SpectralHazardCurveFieldTree',
			'HazardCurveCollection']


class HazardCurve(HazardResult):
	"""
	Class representing a hazard curve for a single site and a
	single spectral period

	:param hazard_values:
		1-D array [l], instance of subclass of :class:`HazardCurveArray`,
		either exceedance rates or exceedance probabilities
	:param site:
		instance of :class:`rshalib.result.GenericSite`
		or (lon, lat) tuple of site for which hazard curve was computed
	:param period:
		float, spectral period (in s)
	:param intensities:
		1-D array [l] of intensity measure levels (ground-motion values)
		for which exceedance rate or exceedance probability was computed
	:param intensity_unit:
		unit in which intensity measure levels are expressed:
		PGA and SA: "g", "mg", "m/s2", "gal", "cm/s2"
		PGV: "cm/s"
		PGD: "cm"
		If not specified, default intensity unit for given imt will
		be used
	:param imt:
		str, intensity measure type (PGA, SA, PGV or PGD)
	:param model_name:
		str, name of this hazard-curve model
		(default: "")
	:param filespec:
		str, full path to file containing hazard curve
		(default: None)
	:param timespan:
		float, period related to the probability of exceedance
		(aka life time)
		(default: 50)
	:param damping:
		float, damping corresponding to intensities
		(expressed as fraction of critical damping)
		(default: 0.05)
	:param variances:
		array with same shape as :param:`hazard_values`,
		ALEATORY variance of exceedance rate or exceedance probability
		(default: None)
	"""
	def __init__(self, hazard_values, site, period,
				intensities, intensity_unit, imt,
				model_name='', filespec=None,
				timespan=50, damping=0.05, variances=None):
		HazardResult.__init__(self, hazard_values, timespan=timespan, imt=imt,
							intensities=intensities, intensity_unit=intensity_unit,
							damping=damping)

		self.model_name = model_name
		self.filespec = filespec
		self.site = parse_sites([site])[0]
		self.period = period
		self.variances = as_array(variances)

	def __repr__(self):
		txt = '<HazardCurve "%s" | %s T=%s s | n=%d | %s>'
		txt %= (self.model_name, self.imt, self.period, self.num_intensities,
				self.site_name)
		return txt

	def __len__(self):
		"""
		Return length of hazard curve (= number of intensity measure levels)
		"""
		return self.num_intensities

	def __add__(self, other_hc):
		"""
		:param other_hc:
			instance of :class:`HazardCurve`

		:return:
			instance of :class:`HazardCurve`
		"""
		assert isinstance(other_hc, HazardCurve)
		assert self.site == other_hc.site
		assert self.imt == other_hc.imt
		assert self.period == other_hc.period
		assert (self.intensities == other_hc.intensities).all()
		assert self.intensity_unit == other_hc.intensity_unit
		assert self.timespan == other_hc.timespan
		hazard_values = self._hazard_values + other_hc._hazard_values
		## Note: variances are dropped
		model_name = self.model_name + ' + ' + other_hc.model_name
		return self.__class__(hazard_values, self.site, self.period,
							self.intensities, self.intensity_unit, self.imt,
							model_name=model_name, filespec=None,
							timespan=self.timespan, damping=self.damping)

	def __mul__(self, number):
		"""
		:param number:
			int, float or Decimal

		:return:
			instance of :class:`HazardCurve`
		"""
		assert np.isscalar(number)
		hazard_values = self._hazard_values * number
		model_name = '%s x %s' % (self.model_name, number)
		return self.__class__(hazard_values, self.site, self.period,
							self.intensities, self.intensity_unit, self.imt,
							model_name=model_name, filespec=self.filespec,
							timespan=self.timespan, damping=self.damping)

	def __rmul__(self, number):
		return self.__mul__(number)

	@property
	def site_name(self):
		return self.site.name

	def interpolate_return_periods(self, return_periods, intensity_unit=None):
		"""
		Interpolate intensity measure levels for given return periods.

		:param return_periods:
			list or array of return periods
		:param intensity_unit:
			see :meth:`get_intensities`

		:return:
			1-D array of intensity measure levels
		"""
		return_periods = np.asarray(return_periods, 'd')
		interpol_exceedance_rates = 1. / return_periods
		intensities = self.get_intensities(intensity_unit)
		rp_intensities = interpolate(self.exceedance_rates, intensities,
									interpol_exceedance_rates)
		return rp_intensities

	def get_return_periods(self, intensities):
		"""
		Interpolate return periods for given intensities

		:param intensities:
			1-D float array, intensity in same unit as hazard curve

		:return:
			1-D float array, return period in yr
		"""
		return interpolate(self.intensities, self.return_periods, intensities)

	def to_spectral(self):
		"""
		Promote to a SpectralHazardCurve (1 site, multiple spectral periods)

		:return:
			instance of :class:`SpectralHazardCurve`
		"""
		intensities = self.intensities.reshape((1, self.num_intensities))
		hazard_values = self._hazard_values.reshape((1, self.num_intensities))
		if self.variances is not None:
			variances = self.variances.reshape((1, self.num_intensities))
		else:
			variances = None

		return SpectralHazardCurve(hazard_values, self.site, [self.period],
								intensities, self.intensity_unit, self.imt,
								model_name=self.model_name, filespec=self.filespec,
								timespan=self.timespan, damping=self.damping,
								variances=variances)

	def to_field(self):
		"""
		Promote to a HazardCurveField (1 spectral period, multiple sites)

		:return:
			instance of :class:`HazardCurveField`
		"""
		intensities = self.intensities
		hazard_values = self._hazard_values.reshape((1, self.num_intensities))
		if self.variances is not None:
			variances = self.variances.reshape((1, self.num_intensities))
		else:
			variances = None
		return HazardCurveField(hazard_values, [self.site], self.period,
								intensities, self.intensity_unit, self.imt,
								model_name=self.model_name, filespec=self.filespec,
								timespan=self.timespan, damping=self.damping,
								variances=variances)

	def plot(self, label='', color="k", linestyle="-", linewidth=2,
			intensity_unit=None, yaxis="exceedance_rate",
			timespan=None, interpol_val=None, interpol_range=None,
			interpol_prop='return_period', legend_location=0,
			xscaling='lin', yscaling='log', xgrid=1, ygrid=1,
			title=None, lang="en", fig_filespec=None, **kwargs):
		"""
		Plot hazard curve

		:param label:
			str, legend label
			(default: '')
		:param color:
			matplotlib color spec, curve color
			(default: 'k')
		:param linestyle:
			matplotlib line style spec
			(default: "-")
		:param linewidth:
			float, line width
			(default: 2)

		See :func:`rshalib.result.plot.plot_hazard_curves`
		for remaining arguments

		:return:
			matplotlib Axes instance
		"""
		if title is None:
			title = "Hazard Curve"
			title += "\nSite: %s, T: %s s" % (self.site_name, self.period)

		return plot_hazard_curves([self], labels=[label], colors=[color],
							linestyles=[linestyle], linewidths=[linewidth],
							intensity_unit=intensity_unit, yaxis=yaxis,
							timespan=timespan, interpol_val=interpol_val,
							interpol_range=interpol_range, interpol_prop=interpol_prop,
							legend_location=legend_location, xscaling=xscaling,
							yscaling=yscaling, xgrid=xgrid, ygrid=ygrid,
							title=title, lang=lang, fig_filespec=fig_filespec,
							**kwargs)

	def export_csv(self, csv_filespec=None, format="%.5E"):
		"""
		Export hazard curve to a csv file

		:param csv_filespec:
			str, full path to output file. If None, output is written
			to standard output
			(default: None)
		:param format:
			str, format for float values
			(default: "%.5E")
		"""
		if csv_filespec:
			f = open(csv_filespec, "w")
		else:
			f = sys.stdout

		header = "%s (%s), Exceedance rate (1/yr)\n"
		header %= (self.imt, self.intensity_unit)
		f.write(header)
		for intensity, exceedance in zip(self.intensities, self.exceedance_rates):
			row = ("%s, %s\n" % (format, format)) % (intensity, exceedance)
			f.write(row)
		f.close()

	@classmethod
	def from_csv_file(cls, csv_filespec, site, period, intensity_unit=None,
					model_name='', timespan=50, damping=0.05):
		"""
		Construct from a csv file


		"""
		from .base_array import ExceedanceRateArray

		intensities, exceedance_rates = [], []
		csv = open(csv_filespec)
		for i, line in enumerate(csv):
			if i == 0:
				col_names = [s.strip() for s in line.split(',')]
				try:
					imt, _intensity_unit = col_names[0].split('(')
					imt = imt.strip()
				except:
					pass
				else:
					_intensity_unit = _intensity_unit.rstrip(')')
					intensity_unit = _intensity_unit or intensity_unit
				if not imt:
					imt = cls.infer_imt_from_intensity_unit(intensity_unit)
			else:
				col_values = line.split(',')
				gm = float(col_values[0])
				exc = float(col_values[1])
				if np.isfinite(exc):
					intensities.append(gm)
					exceedance_rates.append(exc)
		csv.close()
		intensities = np.array(intensities)
		exceedance_rates = ExceedanceRateArray(exceedance_rates)
		if intensities[1] < intensities[0]:
			intensities = intensities[::-1]
			exceedance_rates = exceedance_rates[::-1]

		if not model_name:
			model_name = os.path.splitext(os.path.split(csv_filespec)[-1])[0]

		return HazardCurve(exceedance_rates, site, period,
							intensities, intensity_unit, imt,
							model_name=model_name, filespec=csv_filespec,
							timespan=timespan, damping=damping)


class HazardCurveCollection:
	"""
	Container for an arbitrary set of hazard curves,
	mainly useful for plotting.

	:param hazard_curves:
		list with instances of :class:`HazardCurve`
	:param labels:
		list of strings, labels for each hazard curve
		(default: [])
	:param colors:
		list with matplotlib color specifications for each hazard curve
		(default: [])
	:param linestyles:
		list of matplotlib line styles for each hazard curve
		(default: [])
	:param linewidths:
		list of line widhts for each hazard curve
		(default: [])
	"""
	def __init__(self, hazard_curves, labels=[],
				colors=[], linestyles=[], linewidths=[]):
		self.hazard_curves = hazard_curves
		self.colors = colors
		self.linestyles = linestyles
		self.linewidths = linewidths
		if not labels:
			labels = [hc.model_name for hc in self.hazard_curves]
		self.labels = labels

	def __repr__(self):
		return '<HazardCurveCollection (n=%d)>' % len(self)

	def __len__(self):
		return len(self.hazard_curves)

	def append(self, hc, label=None, color=None, linestyle=None, linewidth=None):
		"""
		Append another hazard curve

		:param hc:
			instance of :class:`HazardCurve`
		:param label:
		:param color:
		:param linestyle:
		:param linewidth:
		"""
		self.hazard_curves.append(hc)
		if not label:
			label = hc.model_name
		self.labels.append(label)
		self.colors.append(color)
		self.linestyles.append(linestyle)
		self.linewidths.append(linewidth)

	@property
	def intensity_unit(self):
		return self.hazard_curves[0].intensity_unit

	def plot(self, labels=[], colors=[], linestyles=[], linewidths=[2],
			intensity_unit=None, yaxis="exceedance_rate",
			timespan=None, interpol_val=None, interpol_range=None,
			interpol_prop='return_period', legend_location=0,
			xscaling='lin', yscaling='log', xgrid=1, ygrid=1,
			title="", fig_filespec=None, lang="en", **kwargs):
		"""
		Plot hazard curve collection

		"""
		if title is None:
			title = "Hazard Curve Collection"

		return plot_hazard_curves(self.hazard_curves, labels=self.labels,
								colors=self.colors, linestyles=self.linestyles,
								linewidths=self.linewidths,
								intensity_unit=intensity_unit, yaxis=yaxis,
								timespan=timespan, interpol_val=interpol_val,
								interpol_range=interpol_range,
								interpol_prop=interpol_prop,
								legend_location=legend_location, xscaling=xscaling,
								yscaling=yscaling, xgrid=xgrid, ygrid=ygrid,
								title=title, fig_filespec=fig_filespec,
								lang=lang, **kwargs)

	plot.__doc__ += plot_hazard_curves.__doc__[[s.start() for s in re.finditer(
										'\n', plot_hazard_curves.__doc__)][4]:]



class SpectralHazardCurve(HazardResult, HazardSpectrum):
	"""
	Class representing hazard curves at 1 site for different spectral periods

	:param hazard_values:
		2-D array [k,l], instance of subclass of :class:`HazardCurveArray`,
		either exceedance rates or exceedance probabilities
	:param site:
		see :class:`HazardCurve`
	:param periods:
		1-D array [k] of spectral periods
	:param intensities:
		2-D array [k,l] of intensity measure levels (ground-motion values)
		for each spectral period for which exceedance rate or probability of
		exceedance was computed
	:param intensity_unit:
	:param imt:
	:param model_name:
	:param filespec:
	:param timespan:
	:param damping:
	:param variances:
		see :class:`HazardCurve`
	"""
	def __init__(self, hazard_values, site, periods,
				intensities, intensity_unit, imt,
				model_name='', filespec=None,
				timespan=50, damping=0.05, variances=None):

		HazardResult.__init__(self, hazard_values, timespan=timespan, imt=imt,
							intensities=intensities, intensity_unit=intensity_unit,
							damping=damping)
		HazardSpectrum.__init__(self, periods, period_axis=0)

		self.model_name = model_name
		self.filespec = filespec
		self.site = parse_sites([site])[0]
		self.variances = as_array(variances)

	def __repr__(self):
		txt = '<SpectralHazardCurve "%s" | %s T=%s - %s s | n=%d | %s>'
		txt %= (self.model_name, self.imt, self.Tmin, self.Tmax, self.num_intensities,
				self.site.name)
		return txt

	def __iter__(self):
		"""
		Loop over spectral periods

		:return:
			generator, yielding instances of :class:`HazardCurve`
		"""
		for i in range(self.num_periods):
			yield self.get_hazard_curve(i)

	def __getitem__(self, period_spec):
		return self.get_hazard_curve(period_spec)

	def __add__(self, other_shc):
		"""
		:param other_shc:
			instance of :class:`SpectralHazardCurve`

		:return:
			instance of :class:`SpectralHazardCurve`
		"""
		assert isinstance(other_shc, SpectralHazardCurve)
		assert self.site == other_shc.site
		assert self.imt == other_shc.imt
		assert (self.periods == other_shc.periods).all()
		assert (self.intensities == other_shc.intensities).all()
		assert self.intensity_unit == other_shc.intensity_unit
		assert self.timespan == other_shc.timespan
		hazard_values = self._hazard_values + other_shc._hazard_values
		model_name = self.model_name + ' + ' + other_shc.model_name
		return self.__class__(hazard_values, self.site, self.periods,
							self.intensities, self.intensity_unit, self.imt,
							model_name=model_name, filespec=None,
							timespan=self.timespan)

	def __mul__(self, number):
		"""
		:param number:
			int, float or Decimal

		:return:
			instance of :class:`SpectralHazardCurve`
		"""
		assert np.isscalar(number)
		hazard_values = self._hazard_values * number
		model_name = '%s x %s' % (self.model_name, number)
		return self.__class__(hazard_values, self.site, self.periods,
							self.intensities, self.intensity_unit, self.imt,
							model_name=model_name, filespec=self.filespec,
							timespan=self.timespan)

	def __rmul__(self, number):
		return self.__mul__(number)

	def get_hazard_curve(self, period_spec=0):
		"""
		Return hazard curve for a particular spectral period

		:param period_spec:
			period specification:
			- int: period index
			- float: spectral period
			(default: 0)

		:return:
			instance of :class:`HazardCurve`
		"""
		period_index = self.period_index(period_spec)
		try:
			period = self.periods[period_index]
		except:
			raise IndexError("Period index %s out of range" % period_index)
		intensities = self.intensities[period_index]
		hazard_values = self._hazard_values[period_index]
		if self.variances is not None:
			variances = self.variances[period_index]
		else:
			variances = None

		return HazardCurve(hazard_values, self.site, period,
							intensities, self.intensity_unit, self.imt,
							model_name=self.model_name, filespec=self.filespec,
							timespan=self.timespan, damping=self.damping,
							variances=variances)

	get_hc = get_hazard_curve

	def interpolate_return_period(self, return_period):
		"""
		Interpolate intensity measure levels for given return period

		:param return_period:
			float, return period

		:return:
			instance of :class:`UHS`
		"""
		num_periods = self.num_periods
		rp_intensities = np.zeros(num_periods)
		interpol_exceedance_rate = 1. / return_period
		for k in range(num_periods):
			rp_intensities[k] = interpolate(self.exceedance_rates[k],
											self.intensities[k],
											[interpol_exceedance_rate])[0]

		return UHS(self.periods, rp_intensities, self.intensity_unit, self.imt,
					self.site, model_name=self.model_name, filespec=self.filespec,
					timespan= self.timespan, return_period=return_period,
					damping=self.damping)

	def interpolate_periods(self, out_periods):
		"""
		Interpolate intensity measure levels at different spectral periods

		:param out_periods:
			list or array of output spectral periods

		:return:
			instance of :class:`SpectralHazardCurve`
		"""
		if out_periods in ([], None):
			return self
		else:
			in_periods = self.periods
			num_intensities = self.num_intensities
			out_shape = (len(out_periods), num_intensities)
			out_hazard_values = self._hazard_values.__class__(np.zeros(out_shape,
																	dtype='d'))
			if self.variances is not None:
				out_variances = np.zeros(out_shape, dtype='d')
			else:
				out_variances = None
			out_period_intensities = np.zeros(out_shape, dtype='d')

			for k in range(len(out_periods)):
				## if k is close to an in_period, take over the corresponding intensities,
				## else, define new range of intensities for that period,
				## interpolate corresponding exceedances,
				## and then interpolate at the wanted period
				threshold = 1E-6
				try:
					id = np.where(abs(in_periods - out_periods[k]) < threshold)[0][0]
				except IndexError:
					id1 = np.where(in_periods < out_periods[k])[0][-1]
					id2 = np.where(in_periods > out_periods[k])[0][0]
					Imin = min(self.intensities[id1][0], self.intensities[id2][0])
					Imax = min(self.intensities[id1][-1], self.intensities[id2][-1])
					#Imin, Imax = self.intensities[id1][0], self.intensities[id1][-1]
					out_period_intensities[k] = logrange(Imin, Imax, num_intensities)
					## Interpolate exceedances of adjacent periods to out_period_intensities
					hazard_values1 = interpolate(self.intensities[id1],
												self._hazard_values[id1],
												out_period_intensities[k])
					hazard_values2 = interpolate(self.intensities[id2],
												self._hazard_values[id2],
												out_period_intensities[k])
					if self.variances is not None:
						variances1 = interpolate(self.intensities[id1],
												self.variances[id1],
												out_period_intensities[k])
						variances2 = interpolate(self.intensities[id2],
												self.variances[id2],
												out_period_intensities[k])
					for l in range(num_intensities):
						out_hazard_values[k,l] = interpolate([in_periods[id1],
															in_periods[id2]],
															[hazard_values1[l],
															hazard_values2[l]],
															[out_periods[k]])[0]
						if self.variances is not None:
							out_variances[k,l] = interpolate([in_periods[id1],
															in_periods[id2]],
															[variances1[l],
															variances2[l]],
															[out_periods[k]])[0]
				else:
					out_period_intensities[k] = self.intensities[id]
					out_hazard_values[k] = self._hazard_values[id]
					if self.variances is not None:
						out_variances[k] = self.variances[id]

			return SpectralHazardCurve(hazard_values, self.site, out_periods,
									out_period_intensities, self.intensity_unit,
									self.imt, model_name=self.model_name,
									filespec=self.filespec, timespan=self.timespan,
									damping=self.damping, variances=out_variances)

	def to_field(self):
		"""
		Promote to a SpectralazardCurveField (multiple spectral periods,
		multiple sites)

		:return:
			instance of :class:`SpectralazardCurveField`
		"""
		intensities = self.intensities
		out_shape = (1, self.num_periods, self.num_intensities)
		hazard_values = self._hazard_values.reshape(out_shape)
		if self.variances is not None:
			variances = self.variances.reshape(out_shape)
		else:
			variances = None

		return SpectralHazardCurveField(hazard_values, [self.site], self.period,
										intensities, self.intensity_unit, self.imt,
										model_name=self.model_name,
										filespecs=[self.filespec],
										timespan=self.timespan, damping=self.damping,
										variances=variances)

	def to_hc_collection(self, labels=[], colors=[], linestyles=[], linewidths=[]):
		"""
		Convert to hazard curve collection

		:param labels:
		:param colors:
		:param linestyles:
		:param linewidths:
			see :class:`HazardCurveCollection`

		:return:
			instance of :class:`HazardCurveCollection`
		"""
		hc_list = []
		for k in range(self.num_periods):
			hc = self.get_hazard_curve(k)
			hc_list.append(hc)

		if not labels:
			labels = ['T=%s s' % T for T in self.periods]

		return HazardCurveCollection(hc_list, labels=labels, colors=colors,
									linestyles=linestyles, linewidths=linewidths)

	def plot(self, labels=[], colors=[], linestyles=["-"], linewidths=[2],
			intensity_unit=None, yaxis="exceedance_rate",
			timespan=None, interpol_val=None, interpol_range=None,
			interpol_prop='return_period', legend_location=0,
			xscaling='lin', yscaling='log', xgrid=1, ygrid=1,
			title="", fig_filespec=None, lang="en", **kwargs):
		"""
		Plot hazard curves for all spectral periods

		"""
		if title is None:
			title = "Spectral Hazard Curve"
			title += "\nSite: %s" % self.site.name

		hcc = self.to_hc_collection(labels=labels, colors=colors,
									linestyles=linestyles, linewidths=linewidths)

		return hcc.plot(intensity_unit=intensity_unit, yaxis=yaxis,
						timespan=timespan, interpol_val=interpol_val,
						interpol_range=interpol_range, interpol_prop=interpol_prop,
						legend_location=legend_location, xscaling=xscaling,
						yscaling=yscaling, xgrid=xgrid, ygrid=ygrid,
						title=title, fig_filespec=fig_filespec, lang=lang, **kwargs)

	plot.__doc__ += HazardCurveCollection.plot.__doc__

	# TODO: revise, because intensities may be different for each period!
	def export_csv(self, csv_filespec=None, format="%.5E"):
		"""
		Export spectral hazard curve to a csv file,
		1st column: intensities,
		other columns: exceedance rates for each spectral period

		:param csv_filespec:
			str, full path to output file. If None, output is written
			to standard output
			(default: None)
		:param format:
			str, format for float values
			(default: "%.5E")
		"""
		if csv_filespec:
			f = open(csv_filespec, "w")
		else:
			f = sys.stdout

		period_colnames = ', '.join(['T=%s s' % T for T in self.periods])
		header = "%s (%s), %s\n" % (self.imt, self.intensity_unit, period_colnames)
		f.write(header)
		for l in range(self.num_intensities):
			intensity = self.intensities[l]
			exceedances = self.exceedance_rates[:,l]
			row_values = [intensity] + list(exceedances)
			row = ', '.join([(' %s' % format) % val for val in row_values])
			f.write("%s\n" % row)
		f.close()

	@classmethod
	def from_hazard_curves(cls, hc_list, model_name=''):
		"""
		Construct spectral hazard curve from list of hazard curves
		for different periods

		:param hc_list:
			list with instances of :class:`HazardCurve`
		:param model_name:
			str, name of spectral hazard curve
			(default: '')

		:return:
			instance of :class:`SpectralHazardCurve`
		"""
		hc0 = hc_list[0]

		sites = set([(hc.site.lon, hc.site.lat, hc.site.depth) for hc in hc_list])
		assert len(sites) == 1
		site = sites.pop()
		imt_family = set([hc.imt[-1] for hc in hc_list])
		assert len(imt_family) == 1
		imt = hc0.imt
		dampings = set([hc.damping for hc in hc_list])
		assert len(dampings) == 1
		damping = hc0.damping

		timespan = hc0.timespan
		intensity_unit = hc0.intensity_unit
		model_name = model_name or hc0.model_name

		num_periods = len(hc_list)
		periods = np.array([hc.period for hc in hc_list])
		num_intensities = np.max([hc.num_intensities for hc in hc_list])
		intensities = np.ones((num_periods, num_intensities)) * np.nan
		hazard_values = ExceedanceRateArray(intensities.copy())
		if hc0.variances is not None:
			variances = np.ones_like(intensities) * np.nan
		else:
			variances = None

		for k, hc in enumerate(hc_list):
			intensities[k, :hc.num_intensities] = hc.get_intensities(intensity_unit)
			hazard_values[k, :hc.num_intensities] = \
								hc._hazard_values.to_exceedance_rates(timespan)
			if variances is not None:
				variances[k, :hc.num_intensities] = hc.variances

		return cls(hazard_values, site, periods,
					imt, intensities, intensity_unit,
					model_name=model_name, filespec=None,
					timespan=timespan, damping=damping, variances=variances)


class HazardCurveField(HazardResult, HazardField):
	"""
	Class representing a hazard curve field for a single spectral period.
	Corresponds to 1 OpenQuake hazardcurve file.

	:param hazard_values:
		2-D array [i,l], instance of subclass of :class:`HazardCurveArray`,
		either exceedance rates or exceedance probabilities
	:param sites:
		list [i] with instances of :class:`rshalib.result.GenericSite`
		or (lon, lat, [z]) tuples of sites for which hazard curves
		were computed
	:param period:
	:param intensities:
	:param intensity_unit:
	:param imt:
	:param model_name:
	:param filespec:
	:param timespan:
	:param damping:
	:param variances:
		see :class:`HazardCurve`
	"""
	def __init__(self, hazard_values, sites, period,
				intensities, intensity_unit, imt,
				model_name="", filespec=None,
				timespan=50, damping=0.05, variances=None):
		HazardResult.__init__(self, hazard_values, timespan=timespan, imt=imt,
							intensities=intensities, intensity_unit=intensity_unit,
							damping=damping)
		HazardField.__init__(self, sites)

		self.model_name = model_name
		self.filespec = filespec
		self.period = period
		self.variances = as_array(variances)

	def __repr__(self):
		txt = '<HazardCurveField "%s" | %s T=%s s | n=%d | %d sites>'
		txt %= (self.model_name, self.imt, self.period, self.num_intensities,
				self.num_sites)
		return txt

	def __iter__(self):
		"""
		Loop over sites
		"""
		for i in range(self.num_sites):
			yield self.get_hazard_curve(i)

	def __getitem__(self, site_spec):
		return self.get_hazard_curve(site_spec)

	def intensity_index(self, intensity):
		"""
		Return index corresponding to a particular intensity measure level
		Parameters:
			return_period: return period
		Return value:
			intensity index (integer)
		"""
		intensity_index = interpolate(self.intensities,
										range(self.num_intensities),
										[intensity])[0]
		return int(round(intensity_index))

	def argmin(self, intensity=None, return_period=None):
		"""
		Return index of site with minimum hazard for a particular
		intensity or return period.

		:param intensity:
			float, intensity measure level.
			If None, return_period must be specified
			(default: None)
		:param return_period:
			float, return period.
			If None, intensity must be specified
			(default: None)

		:return:
			int, site index
		"""
		if intensity is not None:
			intensity_index = self.intensity_index(intensity)
			site_index = self._hazard_values[:,intensity_index].argmin()
		elif return_period is not None:
			hazardmapset = self.interpolate_return_periods([return_period])
			hazardmap = hazardmapset[0]
			site_index = hazardmap.argmin()
		else:
			raise Exception("Need to specify either intensity or return_period")
		return site_index

	def argmax(self, intensity=None, return_period=None):
		"""
		Return index of site with maximum hazard for a particular
		intensity or return period.

		:param intensity:
			float, intensity measure level.
			If None, return_period must be specified
			(default: None)
		:param return_period:
			float, return period.
			If None, intensity must be specified
			(default: None)

		:return:
			int, site index
		"""
		if intensity is not None:
			intensity_index = self.intensity_index(intensity)
			site_index = self._hazard_values[:,intensity_index].argmax()
		elif return_period is not None:
			hazardmapset = self.interpolate_return_periods([return_period])
			hazardmap = hazardmapset[0]
			site_index = hazardmap.argmax()
		else:
			raise Exception("Need to specify either intensity or return_period")
		return site_index

	def min(self, intensity=None, return_period=None):
		"""
		Return minimum hazard for a particular intensity or return period

		:param intensity:
		:param return_period:
			see :meth:`argmin`

		:return:
			float, minimum exceedance rate (if intensity was specified)
			or minimum intensity (if return period was specified)
		"""
		if intensity is not None:
			intensity_index = self.intensity_index(intensity)
			return self._hazard_values[:,intensity_index].min()
		elif return_period is not None:
			hazardmapset = self.interpolate_return_periods([return_period])
			hazardmap = hazardmapset[0]
			return hazardmap.min()
		else:
			raise Exception("Need to specify either intensity or return_period")

	def max(self, intensity=None, return_period=None):
		"""
		Return maximum hazard for a particular intensity or return period

		:param intensity:
		:param return_period:
			see :meth:`argmin`

		:return:
			float, maximum exceedance rate (if intensity was specified)
			or maximum intensity (if return period was specified)
		"""
		if intensity is not None:
			intensity_index = self.intensity_index(intensity)
			return self._hazard_values[:,intensity_index].max()
		elif return_period is not None:
			hazardmapset = self.interpolate_return_periods([return_period])
			hazardmap = hazardmapset[0]
			return hazardmap.max()
		else:
			raise Exception("Need to specify either intensity or return_period")

	def get_hazard_curve(self, site_spec=0):
		"""
		Return hazard curve for a particular site

		:param site_spec:
			site specification:
			- int: site index
			- str: site name
			- instance of :class:`rshalib.site.GenericSite`: site
			- (lon, lat) tuple
			(default: 0)

		:return:
			instance of :class:`HazardCurve`
		"""
		site_index = self.site_index(site_spec)
		try:
			site = self.sites[site_index]
		except:
			raise IndexError("Site index %s out of range" % site_index)

		#site_name = self.site_names[site_index]
		intensities = self.intensities
		hazard_values = self._hazard_values[site_index]
		if self.variances is not None:
			variances = self.variances[site_index]
		else:
			variances = None

		return HazardCurve(hazard_values, site, self.period,
							intensities, self.intensity_unit, self.imt,
							model_name=self.model_name, filespec=self.filespec,
							timespan=self.timespan, damping=self.damping,
							variances=variances)

	get_hc = get_hazard_curve

	def interpolate_return_periods(self, return_periods):
		"""
		Interpolate intensity measure levels for given return periods.

		:param return_periods:
			list or array of return periods

		:rturn:
			instance of :class:`HazardMapSet`
		"""
		filespecs = [self.filespec] * len(return_periods)
		return_periods = np.array(return_periods)
		rp_intensities = np.zeros((len(return_periods), self.num_sites))
		interpol_exceedances = 1. / return_periods
		for i in range(self.num_sites):
				rp_intensities[:,i] = interpolate(self.exceedance_rates[i],
													self.intensities,
													interpol_exceedances)

		return HazardMapSet(self.sites, self.period,
							rp_intensities, self.intensity_unit, self.imt,
							model_name=self.model_name, filespecs=filespecs,
							timespan=self.timespan, return_periods=return_periods,
							damping=self.damping)

	def to_spectral(self):
		"""
		Promote to a SpectralHazardCurveField object (multiple sites,
		multiple spectral periods)

		:return:
			instance of :class:`SpectralHazardCurveField`
		"""
		intensities = self.intensities.reshape((1, self.num_intensities))
		out_shape = (self.num_sites, 1, self.num_intensities)
		hazard_values = self._hazard_values.reshape(out_shape)
		if self.variances is not None:
			variances = self.variances.reshape(out_shape)
		else:
			variances = None

		return SpectralHazardCurveField(hazard_values, self.sites, [self.period],
									intensities, self.intensity_unit, self.imt,
									model_name=self.model_name, filespecs=[self.filespec],
									timespan=self.timespan, damping=self.damping,
									variances=variances)

	def to_hc_collection(self, site_specs=[], labels=[],
						colors=[], linestyles=[], linewidths=[2]):
		"""
		Convert to hazard curve collection

		:param site_specs:
			list with site specs (indexes, (lon,lat) tuples or site names)
			of sites to be included in collection
			(default: [] will include all sites)
		:param labels:
		:param colors:
		:param linestyles:
		:param linewidths:
			see :class:`HazardCurveCollection`

		:return:
			instance of :class:`HazardCurveCollection`
		"""
		if labels in ([], None):
			labels = self.site_names

		## Determine sites
		if site_specs in (None, []):
			site_indexes = range(self.num_sites)
		else:
			site_indexes = [self.site_index(site_spec) for site_spec in site_specs]
		sites = [self.sites[site_index] for site_index in site_indexes]

		## Labels
		if labels in (None, []):
			labels = [site.name for site in sites]

		## Linestyles: use different linestyle for each color cycle
		if linestyles in (None, []):
			num_colors = len(colors or
						pylab.rcParams['axes.prop_cycle'].by_key()['color'])
			linestyles = [['-', '--', ':', '-.'][i//num_colors%4]
							for i in range(len(sites))]

		## Hazard curves
		hc_list = []
		for i in site_indexes:
			hc = self.get_hazard_curve(i)
			hc_list.append(hc)

		return HazardCurveCollection(hc_list, labels=labels, colors=colors,
									linestyles=linestyles, linewidths=linewidths)

	def plot(self, site_specs=[], labels=[],
			colors=[], linestyles=["-"], linewidths=[2],
			intensity_unit=None, yaxis="exceedance_rate",
			timespan=None, interpol_val=None, interpol_range=None,
			interpol_prop='return_period', legend_location=0,
			xscaling='lin', yscaling='log', xgrid=1, ygrid=1,
			title="", fig_filespec=None, lang="en", **kwargs):
		"""
		Plot hazard curves for one or more sites.

		:param site_specs:
			see :meth:`to_hc_collection`

		"""
		if title is None:
			title = "Hazard Curve Field"
			title += "\nT = %s s" % self.period

		hcc = self.to_hc_collection(site_specs=site_specs, labels=labels,
					colors=colors, linestyles=linestyles, linewidths=linewidths)

		return hcc.plot(intensity_unit=intensity_unit, yaxis=yaxis,
						timespan=timespan, interpol_val=interpol_val,
						interpol_range=interpol_range, interpol_prop=interpol_prop,
						legend_location=legend_location, xscaling=xscaling,
						yscaling=yscaling, xgrid=xgrid, ygrid=ygrid,
						title=title, fig_filespec=fig_filespec, lang=lang, **kwargs)

	plot.__doc__ += HazardCurveCollection.plot.__doc__

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML HazardCurveField element)

		:param encoding:
			str, unicode encoding
			(default: 'latin1')

		:return:
			instance of :class:`etree.Element`
		"""
		# TODO: use names from nrml namespace
		hcf_elem = etree.Element('hazardCurveField')
		hcf_elem.set('imt', self.imt)
		hcf_elem.set('period', str(self.period))
		hcf_elem.set('imls', ''.join(map(str, self.intensities)))

		for i, site in enumerate(self.sites):
			site_elem = etree.SubElement(hcf_elem, 'site')
			site_elem.set('lon_lat', ''.join(map(str, site)))
			site_elem.set('exceedance_rates', ''.join(map(str,
													self.exceedance_rates[i])))

		return hcf_elem

	def write_nrml(self, filespec, encoding='latin1', pretty_print=True):
		"""
		Write hazard curve field to XML file

		:param filespec:
			str, full path to XML output file
		:param encoding:
			str, unicode encoding
			(default: 'utf-8')
		pretty_print:
			bool, indicating whether or not to indent each element
			(default: True)
		"""
		tree = create_nrml_root(self, encoding=encoding)
		if filespec:
			## Note: open file in binary mode in PY3, as etree.tostring returns bytes
			fd = open(filespec, 'wb')
		else:
			fd = sys.stdout
		tree.write(fd, xml_declaration=True, encoding=encoding,
					pretty_print=pretty_print)
		fd.close()


class SpectralHazardCurveField(HazardResult, HazardField, HazardSpectrum):
	"""
	Class representing a hazard curve field for different spectral periods.
	Corresponds to 1 CRISIS .GRA file.

	:param hazard_values:
		3-D array [i,k,l], instance of subclass of :class:`HazardCurveArray`,
		either exceedance rates or exceedance probabilities
	:param sites:
	:param periods:
	:param intensities:
	:param intensity_unit:
	:param imt:
	:param model_name:
		see :class:`SpectralHazardCurve`
	:param filespecs:
		list with full paths to files containing hazard curves
		(1 file for each spectral period)
	:param timespan:
	:param damping:
	:param variances:
		see :class:`SpectralHazardCurve`
	"""
	def __init__(self, hazard_values, sites, periods,
				intensities, intensity_unit, imt,
				model_name="", filespecs=[],
				timespan=50, damping=0.05, variances=None):

		HazardResult.__init__(self, hazard_values, timespan=timespan, imt=imt,
							intensities=intensities, intensity_unit=intensity_unit,
							damping=damping)
		HazardField.__init__(self, sites)
		HazardSpectrum.__init__(self, periods, period_axis=1)

		self.model_name = model_name
		self.filespecs = filespecs
		self.variances = as_array(variances)
		self.validate()

	def __repr__(self):
		txt = '<SpectralHazardCurveField "%s" | %s T=%s - %s s | n=%d | %d sites>'
		txt %= (self.model_name, self.imt, self.Tmin, self.Tmax, self.num_intensities,
				self.num_sites)
		return txt

	def __add__(self, other_shcf):
		"""
		:param other_shc:
			instance of :class:`SpectralHazardCurve`

		:return:
			instance of :class:`SpectralHazardCurve`
		"""
		assert isinstance(other_shcf, SpectralHazardCurveField)
		assert self.sites == other_shcf.sites
		assert self.imt == other_shcf.imt
		assert (self.periods == other_shcf.periods).all()
		assert (self.intensities == other_shcf.intensities).all()
		assert self.intensity_unit == other_shcf.intensity_unit
		assert self.timespan == other_shcf.timespan
		hazard_values = self._hazard_values + other_shcf._hazard_values

		model_name = self.model_name + ' + ' + other_shcf.model_name
		return self.__class__(hazard_values, self.sites, self.periods,
							self.intensities, self.intensity_unit, self.imt,
							model_name=model_name, filespecs=[],
							timespan=self.timespan, damping=self.damping)

	def __mul__(self, number):
		"""
		:param number:
			int, float or Decimal

		:return:
			instance of :class:`SpectralHazardCurve`
		"""
		assert np.isscalar(number)
		hazard_values = self._hazard_values * number

		model_name = '%s x %s' % (self.model_name, number)
		return self.__class__(hazard_values, self.sites, self.periods,
							self.intensities, self.intensity_unit, self.imt,
							model_name=model_name, filespecs=[],
							timespan=self.timespan, damping=self.damping)

	def __rmul__(self, number):
		return self.__mul__(number)

	@classmethod
	def from_hazard_curve_fields(self, hcf_list, model_name):
		"""
		Construct spectral hazard curve field from hazard curve fields
		for different spectral periods.

		:param hcf_list:
			list with instances of :class:`HazardCurveField`
		:param model_name:
			str, model name

		:return:
			instance of :class:`SpectralHazardCurveField`
		"""
		hcf0 = hcf_list[0]
		filespecs = [hcf.filespec for hcf in hcf_list]
		sites = hcf0.sites
		num_sites = hcf0.num_sites
		periods = [hcf.period for hcf in hcf_list]
		num_periods = len(periods)
		imt = hcf_list[-1].imt
		num_intensities = hcf0.num_intensities
		intensity_unit = hcf0.intensity_unit
		timespan = hcf0.timespan
		damping = hcf0.damping

		out_shape = (num_sites, num_periods, num_intensities)
		hazard_values = hcf0._hazard_values.__class__(np.zeros(out_shape))
		intensities = np.zeros((num_periods, num_intensities))
		# TODO: variances
		for k, hcf in enumerate(hcf_list):
			hazard_values[:,k,:] = hcf._hazard_values
			intensities[k] = hcf.intensities

		return SpectralHazardCurveField(hazard_values, sites, periods,
										intensities, intensity_unit, imt,
										model_name=model_name, filespecs=filespecs,
										timespan=timespan, damping=damping,
										variances=None)

	def __iter__(self):
		"""
		Loop over sites
		"""
		for i in range(self.num_sites):
			yield self.get_spectral_hazard_curve(i)


	def __getitem__(self, site_spec):
		return self.get_spectral_hazard_curve(site_spec)

	def validate(self):
		"""
		Check if arrays have correct dimensions
		"""
		num_sites = self.num_sites
		num_periods = self.num_periods
		num_intensities = self.num_intensities

		if len(self.intensities.shape) != 2:
			raise Exception("intensities array has wrong dimension")
		if self.intensities.shape[0] != num_periods:
			raise Exception("intensities array has wrong shape")
		if len(self._hazard_values.shape) != 3:
			raise Exception("exceedance_rates or poes array has wrong dimension")
		if self._hazard_values.shape != (num_sites, num_periods, num_intensities):
			raise Exception("exceedance_rates or poes array has wrong shape")
		if self.variances is not None:
			if len(self.variances.shape) != 3:
				raise Exception("variances array has wrong dimension")
			if self.variances.shape != (num_sites, num_periods, num_intensities):
				raise Exception("variances array has wrong shape")

	def get_hazard_curve_field(self, period_spec=0):
		"""
		Extract hazard curve field for given period

		:param period_spec:
			period specification:
			- int: period index
			- float: spectral period
			(default: 0)

		:return:
			instance of :class:`HazardCurveField`
		"""
		period_index = self.period_index(period_spec)
		try:
			period = self.periods[period_index]
		except:
			raise IndexError("Period index %s out of range" % period_index)

		intensities = self.intensities[period_index]
		hazard_values = self._hazard_values[:,period_index]
		if self.variances is not None:
			variances = self.variances[:,period_index]
		else:
			variances = None

		return HazardCurveField(hazard_values, self.sites, period,
								intensities, self.intensity_unit, self.imt,
								model_name=self.model_name,
								filespec=self.filespecs[period_index],
								timespan=self.timespan, damping=self.damping,
								variances=variances)

	get_hcf = get_hazard_curve_field

	def append_hazard_curve_fields(self, hcf_list):
		raise NotImplementedError()

	def get_spectral_hazard_curve(self, site_spec=0):
		"""
		Extract spectral hazard curve for a particular site

		:param site_spec:
			site specification:
			- int: site index
			- str: site name
			- instance of :class:`rshalib.site.GenericSite`: site
			- (lon, lat) tuple
			(default: 0)

		:return:
			instance of :class:`SpectralHazardCurve`
		"""
		site_index = self.site_index(site_spec)
		try:
			site = self.sites[site_index]
		except:
			raise IndexError("Site index %s out of range" % site_index)

		#site_name = self.site_names[site_index]
		filespec = self.filespecs[0]
		intensities = self.intensities
		hazard_values = self._hazard_values[site_index]
		if self.variances is not None:
			variances = self.variances[site_index]
		else:
			variances = None

		return SpectralHazardCurve(hazard_values, site, self.periods,
								intensities, self.intensity_unit, self.imt,
								model_name=self.model_name, filespec=filespec,
								timespan=self.timespan, damping=self.damping,
								variances=variances)

	get_shc = get_spectral_hazard_curve

	def get_hazard_curve(self, site_spec=0, period_spec=0):
		"""
		Extract hazard curve for a particular site and a particular period

		:param site_spec:
			see :meth:`get_spectral_hazard_curve`
		:param period_spec:
			period specification:
			- int: period index
			- float: spectral period
			(default: 0)

		:return:
			instance of :class:`HazardCurve`
		"""
		site_index = self.site_index(site_spec)
		try:
			site = self.sites[site_index]
		except:
			raise IndexError("Site index %s out of range" % site_index)
		#else:
		#	site_name = self.site_names[site_index]

		period_index = self.period_index(period_spec)
		try:
			period = self.periods[period_index]
		except:
			raise IndexError("Period index %s out of range" % period_index)

		filespec = self.filespecs[period_index]
		intensities = self.intensities[period_index]
		hazard_values = self._hazard_values[site_index, period_index]
		if self.variances is not None:
			variances = self.variances[site_index, period_index]
		else:
			variances = None

		return HazardCurve(hazard_values, site, period,
							intensities, self.intensity_unit, self.imt,
							model_name=self.model_name, filespec=filespec,
							timespan=self.timespan, damping=self.damping,
							variances=variances)

	get_hc = get_hazard_curve

	def to_tree(self):
		"""
		Promote to a SpectralazardCurveFieldTree object
		(multiple spectral periods, multiple sites, multiple
		logic-tree branches)

		:return:
			instance of :class:`SpectralHazardCurveFieldTree`
		"""
		intensities = self.intensities
		out_shape = (self.num_sites, 1, self.num_periods, self.num_intensities)
		hazard_values = self._hazard_values.reshape(out_shape)
		if self.variances is not None:
			variances = self.variances.reshape(out_shape)
		else:
			variances = None

		branch_names = [self.model_name]
		filespecs = [self.filespecs[0]]
		weights = np.array([1.], 'd')

		return SpectralHazardCurveFieldTree(hazard_values, branch_names, weights,
									self.sites, self.periods,
									intensities, self.intensity_unit, self.imt,
									model_name=self.model_name, filespecs=filespecs,
									timespan=self.timespan, damping=self.damping,
									variances=variances)

	def interpolate_return_periods(self, return_periods):
		"""
		Interpolate intensity measure levels for given return periods

		:param return_periods:
			list or array with return periods

		:return:
			instance of :class:`UHSFieldSet`
		"""
		filespecs = [self.filespecs[0]] * len(return_periods)
		return_periods = np.array(return_periods)
		num_sites, num_periods = self.num_sites, self.num_periods
		rp_intensities = np.zeros((len(return_periods), num_sites, num_periods))
		interpol_exceedances = 1. / return_periods
		for i in range(num_sites):
			for k in range(num_periods):
				rp_intensities[:,i,k] = interpolate(self.exceedance_rates[i,k],
													self.intensities[k],
													interpol_exceedances)

		return UHSFieldSet(self.sites, self.periods,
							rp_intensities, self.intensity_unit, self.imt,
							model_name=self.model_name, filespecs=filespecs,
							timespan=self.timespan, return_periods=return_periods,
							damping=self.damping)

	def interpolate_periods(self, out_periods):
		"""
		Interpolate intensity measure levels at different spectral periods

		:param out_periods:
			list or array of output spectral periods

		:return:
			instance of :class:`SpectralHazardCurveField`
		"""
		num_sites, num_intensities = self.num_sites, self.num_intensities
		out_shape = (num_sites, len(out_periods), num_intensities)
		out_hazard_values = self._hazard_values.__class__(np.zeros(out_shape, dtype='d'))
		if self.variances is not None:
			out_variances = np.zeros(out_shape, dtype='d')
		else:
			out_variances = None

		for i in range(num_sites):
			shc = self.get_spectral_hazard_curve(site_spec=i)
			shc_out = shc.interpolate_periods(out_periods)
			out_hazard_values[i] = shc_out._hazard_values
			if self.variances is not None:
				out_variances[i] = shc_out.variances
		intensities = shc_out.intensities

		return SpectralHazardCurveField(out_hazard_values, self.sites, out_periods,
										intensities, self.intensity_unit, self.imt,
										model_name=self.model_name,
										filespecs=self.filespecs,
										timespan=self.timespan, damping=self.damping,
										variances=out_variances)

	def to_hc_collection(self, site_specs=[], period_specs=[], labels=[],
						colors=[], linestyles=[], linewidths=[2]):
		"""
		Convert to hazard curve collection

		:param site_specs:
			list with site specs (indexes, (lon,lat) tuples or site names)
			of sites to be included in collection
			(default: [] will include all sites)
		:param period_specs:
			list with period specs (integer period indexes or float
			spectral periods)
			(default: [] will include all periods)
		:param labels:
		:param colors:
		:param linestyles:
		:param linewidths:
			see :class:`HazardCurveCollection`

		:return:
			instance of :class:`HazardCurveCollection`
		"""
		## Determine sites and periods
		if site_specs in (None, []):
			site_indexes = range(self.num_sites)
		else:
			site_indexes = [self.site_index(site_spec) for site_spec in site_specs]
		sites = [self.sites[site_index] for site_index in site_indexes]
		if period_specs in (None, []):
			period_indexes = range(self.num_periods)
		else:
			period_indexes = [self.period_index(period_spec) for period_spec in period_specs]
		periods = [self.periods[period_index] for period_index in period_indexes]

		## Labels
		if labels in (None, []):
			if len(sites) == 1:
				labels = ["T = %s s" % period for period in periods]
			elif len(periods) == 1:
				labels = [self.site_names[site_index] for site_index in site_indexes]
			else:
				labels = []
				for i, site in enumerate(sites):
					for period in periods:
						labels.append("Site: %s, T=%s s" % (site.name, period))

		## Colors and linestyles
		if colors in (None, []):
			COLORS = pylab.rcParams['axes.prop_cycle'].by_key()['color']
			nc = len(COLORS)
			if len(sites) >= len(periods):
				colors = [COLORS[i%nc:i%nc+1] * len(periods) for i in range(len(sites))]
			else:
				colors = [COLORS[i%nc:i%nc+1] * len(sites) for i in range(len(periods))]
			## Hack to flatten nested list
			colors = sum(colors, [])

		## Linestyles
		if linestyles in (None, []):
			if len(sites) >= len(periods):
				linestyles = [['-', '--', ':', '-.'][i%4:i%4+1] * len(sites)
								for i in range(len(periods))]
			else:
				linestyles = [['-', '--', ':', '-.'][i%4:i%4+1] * len(periods)
								for i in range(len(sites))]
			linestyles = sum(linestyles, [])

		## Hazard curves
		hc_list = []
		for site in sites:
			site_index = self.site_index(site)
			for period in periods:
				period_index = self.period_index(period)
				hc = self.get_hazard_curve(site_index, period_index)
				hc_list.append(hc)

		return HazardCurveCollection(hc_list, labels=labels, colors=colors,
									linestyles=linestyles, linewidths=linewidths)

	def plot(self, site_specs=[], period_specs=[], labels=[],
			colors=[], linestyles=[], linewidths=[2],
			intensity_unit=None, yaxis="exceedance_rate",
			timespan=None, interpol_val=None, interpol_range=None,
			interpol_prop='return_period', legend_location=0,
			xscaling='lin', yscaling='log', xgrid=1, ygrid=1,
			title="", fig_filespec=None, lang="en", **kwargs):
		"""
		Plot hazard curves for one or more sites.

		:param site_specs:
		:param period_specs:
			see :meth:`to_hc_collection`

		"""
		if title is None:
			title = self.model_name

		hcc = self.to_hc_collection(site_specs, period_specs, labels=labels,
					colors=colors, linestyles=linestyles, linewidths=linewidths)

		return hcc.plot(intensity_unit=intensity_unit, yaxis=yaxis,
						timespan=timespan, interpol_val=interpol_val,
						interpol_range=interpol_range, interpol_prop=interpol_prop,
						legend_location=legend_location, xscaling=xscaling,
						yscaling=yscaling, xgrid=xgrid, ygrid=ygrid,
						title=title, fig_filespec=fig_filespec, lang=lang, **kwargs)

	plot.__doc__ += HazardCurveCollection.plot.__doc__

	def export_GRA(self, out_filespec):
		"""
		Write spectral hazard curve field to CRISIS .GRA format

		:param out_filespec:
			str, full path to output file
		"""
		f = open(out_filespec, "w")
		f.write("************************************************************\n")
		f.write("Generic exceedance-rate results\n")
		f.write("Calculated outside CRISIS\n")
		f.write("NumSites, NumPeriods, NumIntensities: %d, %d, %d\n"
				% (self.num_sites, self.num_periods, self.num_intensities))
		f.write("************************************************************\n")
		f.write("\n\n")
		for i in range(self.num_sites):
			f.write("    %s      %s\n" % self.sites[i])
			for k in range(self.num_periods):
				f.write("INTENSITY %d T=%s\n" % (k+1, self.periods[k]))
				for l in range(self.num_intensities):
					f.write("%.5E  %.5E"
							% (self.intensities[k,l], self.exceedance_rates[i,k,l]))
					if self.variances is not None:
						f.write("  %.5E" % self.variances[i,k,l])
					f.write("\n")
		f.close()

	def create_xml_element(self, smlt_path=None, gmpelt_path=None,
							encoding='latin1'):
		"""
		Create xml element (NRML SpectralHazardCurveField element)

		:param smlt_path:
			str, path to NRML file containing source-model logic tree
			(default: None)
		:param gmpelt_path:
			str, path to NRML file containing ground-motion logic tree
			(default: None)
		:param encoding:
			str, unicode encoding
			(default: 'latin1')

		:return:
			instance of :class:`etree.Element`
		"""
		shcf_elem = etree.Element(ns.SPECTRAL_HAZARD_CURVE_FIELD)
		shcf_elem.set(ns.IMT, xmlstr(self.imt, encoding=encoding))
		shcf_elem.set(ns.INVESTIGATION_TIME, str(self.timespan))
		if smlt_path:
			shcf_elem.set(ns.SMLT_PATH, xmlstr(smlt_path, encoding=encoding))
		if gmpelt_path:
			shcf_elem.set(ns.GMPELT_PATH, xmlstr(gmpelt_path, encoding=encoding))
		shcf_elem.set(ns.NAME, xmlstr(self.model_name, encoding=encoding))
		for k, period in enumerate(self.periods):
			# TODO: put following in HazardCurveField and HazardCurve !
			hcf_elem = etree.SubElement(shcf_elem, ns.HAZARD_CURVE_FIELD)
			hcf_elem.set(ns.PERIOD, str(period))
			# TODO: add damping for SA ?
			if period > 0:
				hcf_elem.set(ns.DAMPING, str(self.damping))
			imls_elem = etree.SubElement(hcf_elem, ns.IMLS)
			imls_elem.text = " ".join(map(str, self.intensities[k,:]))
			for i, site in enumerate(self.sites):
				hazard_curve_elem = etree.SubElement(hcf_elem, ns.HAZARD_CURVE)
				point_elem = etree.SubElement(hazard_curve_elem, ns.POINT)
				position_elem = etree.SubElement(point_elem, ns.POSITION)
				position_elem.text = "%s %s" % (site[0], site[1])
				poes_elem = etree.SubElement(hazard_curve_elem, ns.POES)
				poes_elem.text = " ".join(map(str, self.poes[i,k,:]))

		return shcf_elem

	def write_nrml(self, filespec, smlt_path=None, gmpelt_path=None,
					encoding='latin1', pretty_print=True):
		"""
		Write spectral hazard curve field to XML file

		:param filespec:
			str, full path to XML output file
		:param smlt_path:
		:param gmpelt_path:
		:param encoding:
			see :meth:`create_xml_element`
		:param pretty_print:
			bool, indicating whether or not to indent each element
			(default: True)
		"""
		tree = create_nrml_root(self, encoding=encoding, smlt_path=smlt_path,
								gmpelt_path=gmpelt_path)
		if filespec:
			## Note: open file in binary mode in PY3, as etree.tostring returns bytes
			fd = open(filespec, 'wb')
		else:
			fd = sys.stdout
		tree.write(fd, xml_declaration=True, encoding=encoding,
					pretty_print=pretty_print)
		fd.close()


# TODO: add HazardCurveFieldTree class?
class HazardCurveFieldTree():
	def __init__(self, **kwargs):
		raise NotImplementedError


class SpectralHazardCurveFieldTree(HazardTree, HazardField, HazardSpectrum):
	"""
	Class representing a spectral hazard curve field tree, i.e. a number
	of logic-tree branches, each representing a spectral hazard curve field.
	Corresponds to a set of CRISIS .GRA files defining (part of) a logic tree

	Provides iteration and indexing over logic-tree branches

	:param hazard_values:
		4-D array [j,i,k,l], instance of subclass of :class:`HazardCurveArray`,
		either exceedance rates or exceedance probabilities
	:param branch_names:
		1-D list [j] of model names of each branch
	:param weights:
		1-D list or array [j] with branch weights
	:param sites:
		1-D list [i] with (lon, lat) tuples of site for which hazard
		curves were computed
	:param periods:
		1-D array [k] of spectral periods
	:param intensities:
		2-D array [k,l] of intensity measure levels (ground-motion values)
		for each spectral period for which exceedance rate or probability
		was computed
	:param intensity_unit:
		str, unit in which intensity measure levels are expressed:
		PGA and SA: "g", "mg", "m/s2", "gal", "cm/s2"
		PGV: "cm/s"
		PGD: "cm"
	:param imt:
		str, intensity measure type (PGA, SA, PGV or PGD)
	:param model_name:
		str, name of this logic-tree model
	:param filespecs:
		list with full paths to files containing hazard curves
		(1 file for each branch)
	:param timespan:
		float, period related to the probability of exceedance
		(aka life time)
		(default: 50)
	:param damping:
		float, damping corresponding to intensities
		(expressed as fraction of critical damping)
		(default: 0.05)
	:param variances:
		array with same shape as :param:`hazard_values`,
		ALEATORY variance of exceedance rate or exceedance probability
		(default: None)
	:param mean:
		3-D array [i,k,l] with mean exceedance rate or probability
		(default: None)
	:param percentile_levels:
		1-D list or array [p] with percentile levels
		(default: None)
	:param percentiles:
		4-D array [i,k,l,p] with percentiles of exceedance rate or
		probability
		(default: None)
	"""
	def __init__(self, hazard_values, branch_names, weights,
				sites, periods,
				intensities, intensity_unit, imt,
				model_name='', filespecs=[],
				timespan=50, damping=0.05,
				variances=None, mean=None,
				percentile_levels=None, percentiles=None):
		HazardTree.__init__(self, hazard_values, branch_names, weights=weights,
							timespan=timespan, intensities=intensities,
							intensity_unit=intensity_unit, imt=imt,
							mean=mean, percentiles=percentiles,
							percentile_levels=percentile_levels)
		HazardField.__init__(self, sites)
		HazardSpectrum.__init__(self, periods, period_axis=2)

		self.model_name = model_name
		self.filespecs = filespecs
		self.variances = as_array(variances)
		self.validate()

	def __repr__(self):
		txt = ('<SpectralHazardCurveFieldTree "%s" | T: %s - %s s (n=%d) '
				'| n=%d | %d sites | %d branches>')
		txt %= (self.model_name, self.Tmin, self.Tmax, len(self),
				self.num_intensities, self.num_sites, self.num_branches)
		return txt

	def __iter__(self):
		"""
		Loop over logic-tree branches
		"""
		for i in range(self.num_branches):
			self.get_spectral_hazard_curve_field(i)

	def __getitem__(self, branch_spec):
		return self.get_spectral_hazard_curve_field(branch_spec)

	#@property
	#def num_branches(self):
	#	return self.exceedance_rates.shape[1]

	def validate(self):
		"""
		Check if arrays have correct dimensions
		"""
		if not (len(self.filespecs) == len(self.branch_names)):
			raise Exception("Number of filespecs not in agreement "
							"with number of branch names")
		num_branches = self.num_branches
		num_sites = self.num_sites
		num_periods = self.num_periods
		if len(self.intensities.shape) != 2:
			raise Exception("intensities array has wrong dimension")
		num_intensities = self.num_intensities
		if self.intensities.shape[0] != num_periods:
			raise Exception("intensities array has wrong shape")
		if len(self._hazard_values.shape) != 4:
			raise Exception("hazard array has wrong dimension")
		if self._hazard_values.shape != (num_sites, num_branches, num_periods,
										num_intensities):
			raise Exception("hazard array has wrong shape")
		if self.variances is not None:
			if len(self.variances.shape) != 4:
				raise Exception("variances array has wrong dimension")
			if self.variances.shape != (num_sites, num_branches, num_periods,
										num_intensities):
				raise Exception("variances array has wrong shape")

	@classmethod
	def from_branches(cls, shcf_list, branch_names=None, weights=None,
					model_name="", mean=None, percentile_levels=None,
					percentiles=None):
		"""
		Construct spectral hazard curve field tree from spectral hazard
		curve fields for different logic-tree branches.

		:param shcf_list:
			list with instances of :class:`SpectralHazardCurveField`
		:param branch_names:
			list of branch names
			(default: None)
		:param weights:
			1-D list or array [j] with branch weights
			(default: None)
		:param model_name:
			str, model name
			(default: "")
		:param mean:
			instance of :class:`SpectralHazardCurveField`, representing
			mean shcf
			(default: None)
		:param percentile_levels:
			list or array with percentile levels
			(default: None)
		:param percentiles:
			list with instances of :class:`SpectralHazardCurveField`,
			representing shcf's corresponding to percentiles
			(default: None)

		:return:
			instance of :class:`SpectralHazardCurveFieldTree`
		"""
		shcf0 = shcf_list[0]
		num_branches = len(shcf_list)
		num_sites = shcf0.num_sites
		num_periods = shcf0.num_periods
		num_intensities = shcf0.num_intensities
		#out_shape = (num_sites, num_branches, num_periods, num_intensities)
		out_shape = (num_branches, num_sites, num_periods, num_intensities)

		all_hazard_values = np.zeros(out_shape, 'd')
		all_hazard_values = shcf0._hazard_values.__class__(all_hazard_values)
		#all_hazard_values[:,0,:,:] = shcf0._hazard_values
		all_hazard_values[0,:,:,:] = shcf0._hazard_values

		if shcf0.variances is not None:
			all_variances = np.zeros(out_shape, 'd')
			all_variances[:,0,:,:] = shcf0.variances
		else:
			all_variances = None

		filespecs = [shcf.filespecs[0] for shcf in shcf_list]
		if branch_names in (None, []):
			branch_names = [shcf.model_name for shcf in shcf_list]
		if weights in (None, []):
			weights = np.ones(num_branches, 'f') / num_branches

		shcft = cls(all_hazard_values, branch_names, weights, shcf0.sites,
					shcf0.periods, shcf0.intensities, shcf0.intensity_unit,
					shcf0.imt, model_name=model_name, filespecs=filespecs,
					timespan=shcf0.timespan, damping=shcf0.damping,
					variances=all_variances)

		for j, shcf in enumerate(shcf_list[1:]):
			shcft.check_shcf_compatibility(shcf)
			#shcft._hazard_values[:,j+1] = shcf._hazard_values
			shcft._hazard_values[j+1] = shcf._hazard_values
			if shcft.variances is not None:
				#shcft.variances[:,j+1] = shcf.variances
				shcft.variances[j+1] = shcf.variances

		if mean is not None:
			shcft.check_shcf_compatibility(mean)
			shcft.set_mean(mean._hazard_values)

		if percentiles is not None:
			num_percentiles = len(percentiles)
			perc_shape = (num_sites, num_periods, num_intensities, num_percentiles)
			perc_array = np.zeros(perc_shape, 'd')
			for p in range(num_percentiles):
				shcf = percentiles[p]
				shcft.check_shcf_compatibility(shcf)
				perc_array[:,:,:,p] = shcf._hazard_values
				perc_array = shcft._hazard_values.__class__(perc_array)
			shcft.set_percentiles(perc_array, percentile_levels)

		return shcft

	def check_shcf_compatibility(self, shcf):
		"""
		Check the compatibility of a candidate branch.

		:param shcf:
			instance of :class:`SpectralHazardCurveField` or higher
		"""
		if self.sites != shcf.sites:
			raise Exception("Sites do not correspond!")
		if (self.periods != shcf.periods).any():
			raise Exception("Spectral periods do not correspond!")
		if self.imt != shcf.imt:
			raise Exception("IMT does not correspond!")
		if (self.intensities != shcf.intensities).any():
			raise Exception("Intensities do not correspond!")
		if self.intensity_unit != shcf.intensity_unit:
			raise Exception("Intensity unit does not correspond!")
		if self.timespan != shcf.timespan:
			raise Exception("Time span does not correspond!")
		if self._hazard_values.__class__ != shcf._hazard_values.__class__:
			raise Exception("Hazard array does not correspond!")

	def append_branch(self, shcf, branch_name="", weight=1.0):
		"""
		Append a new branch

		Notes:
			- Branch weights are not renormalized to avoid rounding errors.
			  This should be done after all branches have been appended.
			- Mean and percentiles can be appended with the set_mean() and
			  set_percentiles() methods.

		:param shcf:
			instance of :class:`SpectralHazardCurveField`
		:param branch_name:
			str, name of branch. If not specified, shcf.model_name
			will be used as the branch name
			(default: "")
		:param weight:
			float or Decimal, branch weight
			(default: 1.0)
		"""
		self.check_shcf_compatibility(shcf)
		if not branch_name:
			branch_name = shcf.model_name
		self.branch_names.append(branch_name)
		self.filespecs.append(shcf.filespecs[0])
		## Do not recompute weights, assume they are correct
		self.weights = np.concatenate([self.weights, [weight]])
		shape = (self.num_sites, 1, self.num_periods, self.num_intensities)
		hazard_values = np.concatenate([self._hazard_values,
									#shcf._hazard_values.reshape(shape)], axis=1)
									shcf._hazard_values.reshape(shape)], axis=0)
		self._hazard_values = self._hazard_values.__class__(hazard_values)
		if self.variances is not None:
			## Note: this is only correct if both shcft and shcf are of the same type
			## (exceedance rates or probabilities of exceedance)
			self.variances = np.concatenate([self.variances,
											#shcf.variances.reshape(shape)], axis=1)
											shcf.variances.reshape(shape)], axis=0)

	def extend(self, shcft):
		"""
		Extend spectral hazard curve field tree in-place with another one

		:param shcft:
			instance of :class:`SpectralHazardCurveFieldTree`
		"""
		self.check_shcf_compatibility(shcft)
		self.branch_names.extend(shcft.branch_names)
		if shcft.filespecs:
			self.filespecs.extend(shcft.filespecs)
		else:
			self.filespecs = []
		self.weights = np.concatenate([self.weights, shcft.weights])
		hazard_values = np.concatenate([self._hazard_values, shcft._hazard_values],
										#axis=1)
										axis=0)
		self._hazard_values = self._hazard_values.__class__(hazard_values)
		if self.variances is not None:
			variances = np.concatenate([self.variances, shcft.variances])
			self.variances = self.variances.__class__(variances)
		## Remove mean and percentiles
		self.mean = None
		self.percentiles = None
		self.normalize_weights()

	def get_spectral_hazard_curve_field(self, branch_spec=0):
		"""
		Extract spectral hazard curve field for a particular branch

		:param branch_spec:
			Branch specification:
			- int: branch index
			- str: branch name
			(default: 0)

		:return:
			instance of :class:`SpectralHazardCurveField`
		"""
		branch_index = self.branch_index(branch_spec)
		try:
			branch_name = self.branch_names[branch_index]
		except:
			raise IndexError("Branch index %s out of range" % branch_index)
		else:
			branch_name = self.branch_names[branch_index]
			filespec = self.filespecs[branch_index]
			filespecs = [filespec]*self.num_periods
			hazard_values = self._hazard_values[:,branch_index,:,:]
			if self.variances is not None:
				variances = self.variances[:,branch_index,:,:]
			else:
				variances = None
			return SpectralHazardCurveField(hazard_values, self.sites, self.periods,
									self.intensities, self.intensity_unit, self.imt,
									model_name=branch_name, filespecs=filespecs,
									timespan=self.timespan, damping=self.damping,
									variances=variances)

	get_shcf = get_spectral_hazard_curve_field

	def get_spectral_hazard_curve(self, branch_spec=0, site_spec=0):
		"""
		Extract spectral hazard curve for a particular branch and site

		:param branch_spec:
			see :meth:`get_spectral_hazard_curve_field`
		:param site_spec:
			site specification:
			- int: site index
			- str: site name
			- instance of :class:`rshalib.site.GenericSite`: site
			- (lon, lat) tuple
			(default: 0)

		Return value:
			SpectralHazardCurveField object
		"""
		site_index = self.site_index(site_spec)
		try:
			site = self.sites[site_index]
		except:
			raise IndexError("Site index %s out of range" % site_index)
		#else:
		#	site_name = self.site_names[site_index]

		branch_index = self.branch_index(branch_spec)
		try:
			branch_name = self.branch_names[branch_index]
		except:
			raise IndexError("Branch index %s out of range" % branch_index)

		intensities = self.intensities
		hazard_values = self._hazard_values[site_index, branch_index]
		filespec = self.filespecs[branch_index]
		if self.variances is not None:
			variances = self.variances[site_index, branch_index]
		else:
			variances = None

		return SpectralHazardCurve(hazard_values, site, self.periods,
									intensities, self.intensity_unit, self.imt,
									model_name=branch_name, filespec=filespec,
									timespan=self.timespan, damping=self.damping,
									variances=variances)

	get_shc = get_spectral_hazard_curve

	def min(self):
		# TODO: does this make sense? Makes more sense with 1 period and 1 site
		#return self._hazard_values.min(axis=1)
		return self._hazard_values.min(axis=0)

	def max(self):
		#return self._hazard_values.max(axis=1)
		return self._hazard_values.max(axis=0)

	def calc_mean(self, weighted=True):
		"""
		Compute mean exceedance rates

		:param weighted:
			bool, whether or not to compute weighted mean
			(default: True)

		:return:
			3-D array [i,k,l], mean exceedance rates
		"""
		if weighted:
			weights = self.weights
		else:
			weights = None

		#return self._hazard_values.mean(axis=1, weights=weights)
		return self._hazard_values.mean(axis=0, weights=weights)

	def calc_variance_epistemic(self, weighted=True):
		"""
		Compute variance of hazard curves (= epistemic uncertainty)

		:param weighted:
			see :meth:`calc_mean`

		:return:
			3-D array [i,k,l], variance of exceedance rates
		"""
		if weighted:
			weights = self.weights
		else:
			weights = np.ones(len(self))

		#return self._hazard_values.mean_and_variance(axis=1, weights=weights)[1]
		return self._hazard_values.mean_and_variance(axis=0, weights=weights)[1]

	def calc_variance_of_mean(self, weighted=True):
		"""
		Compute variance (of mean exceedance rate)

		:param weighted:
			see :meth:`calc_mean`

		:return:
			3-D array [i,k,l], variance of mean exceedance rate
		"""
		if weighted and not self.weights in ([], None):
			# TODO: this needs to be checked
			mean = self.calc_mean(weighted=True)
			weights = np.array(self.weights)
			weights_column = weights.reshape((self.num_branches, 1))
			out_shape = (self.num_sites, self.num_periods, self.num_intensities)
			variance_of_mean = np.zeros(out_shape, 'd')
			for i in range(self.num_sites):
				for k in range(self.num_periods):
					#var = (self.exceedance_rates[i,:,k] - mean[i,k])**2
					var = (self.exceedance_rates[:,i,k] - mean[i,k])**2
					variance_of_mean[i,k] = np.sum(weights_column * var, axis=0)
		else:
			#variance_of_mean = np.var(self.exceedance_rates, axis=1)
			variance_of_mean = np.var(self.exceedance_rates, axis=0)
		return variance_of_mean

	def calc_mean_variance(self, weighted=True):
		"""
		Compute mean of aleatory variance
		(i.e., the variance stored in :prop:`variance`)

		:param weighted:
			see :meth:`calc_mean`

		:return:
			3-D array [i,k,l], mean variance of exceedance rate
		"""
		if self.variances is not None:
			if weighted:
				mean_variance = np.average(self.variances, weights=self.weights,
																		#axis=1)
																		axis=0)
			else:
				#mean_variance = np.mean(self.variances, axis=1)
				mean_variance = np.mean(self.variances, axis=0)
		else:
			mean_variance = None
		return mean_variance

	def calc_percentiles_epistemic(self, percentile_levels=[5, 16, 50, 84, 95],
									weighted=True, interpol=True):
		"""
		Compute percentiles of exceedance rate (epistemic uncertainty)

		:param percentile_levels:
			list or array of percentile levels. Percentiles
			may be specified as integers between 0 and 100 or as floats
			between 0 and 1
			(default: [5, 16, 50, 84, 95])
		:param weighted:
			bool indicating whether or not branch weights should be
			taken into account
			(default: True)
		:param interpol:
			bool, whether or not percentile intercept should be
			interpolated. Only applies to weighted percentiles
			(default: True)

		:return:
			4-D array [i,k,l,p], percentiles of hazard values
		"""
		if percentile_levels in ([], None):
			percentile_levels = [5, 16, 50, 84, 95]
		num_sites = self.num_sites
		num_periods = self.num_periods
		num_intensities = self.num_intensities
		num_percentiles = len(percentile_levels)

		if weighted and self.weights is not None and len(set(self.weights)) > 1:
			weights = self.weights
		else:
			weights = None
		#percentiles = self._hazard_values.scoreatpercentile(1, percentile_levels,
		percentiles = self._hazard_values.scoreatpercentile(0, percentile_levels,
												weights=weights, interpol=interpol)
		return percentiles

	def calc_percentiles_combined(self, percentile_levels, weighted=True):
		"""
		Compute percentiles of exceedance rate (combined uncertainty)
		Can only be computed if variances (aleatory uncertainty) is known
		(i.e. if :prop:`variance` is set)

		:param percentile_levels:
		:param weighted:
			see :meth:`calc_percentiles_epistemic`

		:return:
			4-D array [i,k,l,p], percentiles of exceedance rate
		"""
		if self.variances is None:
			raise Exception("Combined uncertainties can only be computed if aleatory uncertainties are stored as variances")
		else:
			# TODO !
			return self.calc_percentiles_epistemic(percentile_levels, weighted=weighted)

	def calc_percentiles(self, percentile_levels, weighted=True):
		"""
		Wrapper function to compute percentiles of exceedance rate
		(combined uncertainty if variances is defined, epistemic otherwise)

		:param percentile_levels:
		:param weighted:
			see :meth:`calc_percentiles_epistemic`

		:return:
			4-D array [i,k,l,p], percentiles of exceedance rate
		"""
		if self.variances is None:
			print("Epistemic")
			return self.calc_percentiles_epistemic(percentile_levels, weighted=weighted)
		else:
			print("Combined")
			return self.calc_percentiles_combined(percentile_levels, weighted=weighted)

	def get_mean_spectral_hazard_curve_field(self, recalc=False, weighted=True):
		"""
		Get mean spectral hazard curve field

		:param recalc:
			bool indicating whether or not to recompute.
			If :prop:`mean` is None, computation will be performed anyway
			(default: False)
		:param weighted:
			bool indicating whether or not branch weights should be
			taken into account. Only applies if :param:`recalc` is True
			(default: True)

		:return:
			instance of :class:`SpectralHazardCurveField`
		"""
		if recalc or self.mean is None:
			mean = self.calc_mean(weighted=weighted)
		else:
			mean = self.mean
		variances = self.calc_variance_of_mean()
		model_name = "Mean(%s)" % self.model_name
		filespecs = [""] * self.num_periods

		return SpectralHazardCurveField(mean, self.sites, self.periods,
								self.intensities, self.intensity_unit, self.imt,
								model_name=model_name, filespecs=filespecs,
								timespan=self.timespan, damping=self.damping,
								variances=variances)

	get_mean_shcf = get_mean_spectral_hazard_curve_field

	def get_percentile_spectral_hazard_curve_field(self, perc, recalc=False,
													weighted=True):
		"""
		Get percentile spectral hazard curve field

		:param perc:
			int, percentile level
		:param recalc:
		:param weighted:
			see :meth:`get_mean_spectral_hazard_curve_field`

		:return:
			instance of :class:`SpectralHazardCurveField`
		"""
		if recalc or self.percentiles is None or not perc in self.percentiles:
			hazard_values = self.calc_percentiles([perc], weighted=weighted)[:,:,:,0]
		else:
			print("No recalculaton!")
			perc_index = self.percentile_levels.index(perc)
			hazard_values = self.percentiles[:,:,:,perc_index]

		model_name = "Perc%02d(%s)" % (perc, self.model_name)
		filespecs = [""]*self.num_periods
		return SpectralHazardCurveField(hazard_values, self.sites, self.periods,
								self.intensities, self.intensity_unit, self.imt,
								model_name=model_name, filespecs=filespecs,
								timespan=self.timespan, damping=self.damping,
								variances=None)

	get_percentile_shcf = get_percentile_spectral_hazard_curve_field

	def import_stats_from_AGR(self, agr_filespec, percentile_levels=None):
		"""
		Import logic-tree statistics from a CRISIS .AGR file

		:param agr_filespec:
			str, full path to .AGR file
		:param percentile_levels:
			list or array of percentile levels to import
			(default: None)
		"""
		# TODO: need to take care with intensity_unit
		from ..crisis import IO

		shcft = IO.read_GRA(agr_filespec)
		if shcft.intensities != self.intensities:
			raise Exception("Intensities do not match with those of "
							"current object!")
		self.mean = shcft.mean
		if percentile_levels is None:
			self.percentile_levels = shcft.percentile_levels
			self.percentiles = shcft.percentiles
		else:
			perc_indexes = []
			for perc in percentile_levels:
				try:
					perc_index = np.where(shcft.percentile_levels == perc)[0][0]
				except:
					raise Exception("Percentile level %s not found in file %s!"
									% (perc, agr_filespec))
				perc_indexes.append(perc_index)
			self.percentile_levels = percentiles
			self.percentiles = shcft.percentiles[:,:,:,perc_indexes]

	def export_stats_AGR(self, out_filespec, weighted=True):
		"""
		Export logic-tree statistics to a CRISIS .AGR file

		:param out_filespec:
			str, full path to output .AGR file
		:param weighted:
			bool indicating whether or not branch weights should be
			taken into account
			(default: True)
		"""
		if is_empty_array(self.mean):
			mean = self.calc_mean(weighted=weighted)
		else:
			mean = self.mean
		variance_of_mean = self.calc_variance_of_mean(weighted=weighted)
		if is_empty_array(self.percentiles):
			if is_empty_array(self.percentile_levels):
				percentile_levels = [5, 16, 50, 84, 95]
			else:
				percentile_levels = self.percentile_levels
			percentiles = self.calc_percentiles(percentile_levels, weighted=weighted)
		else:
			percentiles = self.percentiles
			percentile_levels = self.percentile_levels

		f = open(out_filespec, "w")
		f.write("************************************************************\n")
		f.write("Logic-tree statistics: mean, variance, percentiles (%s)"
				% ", ".join(["%d" % p for p in percentile_levels]))
		f.write("\n")
		f.write("Calculated using ROB python routines\n")
		f.write("NumSites, NumPeriods, NumIntensities: %d, %d, %d\n"
				% (self.num_sites, self.num_periods, self.num_intensities))
		f.write("************************************************************\n")
		f.write("\n\n")
		for i in range(self.num_sites):
			f.write("    %s      %s\n" % self.sites[i])
			for k in range(self.num_periods):
				f.write("INTENSITY %d T=%s\n" % (k+1, self.periods[k]))
				for l in range(self.num_intensities):
					values = ([self.intensities[k,l]] + [mean[i,k,l]]
						+ [variance_of_mean[i,k,l]] + list(percentiles[i,k,l,:]))
					txt = "  ".join(["%.5E" % val for val in values])
					f.write("%s\n" % txt)
		f.close()

	def slice_by_branch_indexes(self, branch_indexes, slice_name,
								normalize_weights=True):
		"""
		Return a subset (slice) of the logic tree based on branch indexes

		:param branch_indexes:
			list or array of branch indexes
		:param slice_name:
			str, name of this slice
		:param normalize_weights:
			bool, indicating whether or not branch weights should be
			renormalized to 1
			(default: True)

		:return:
			instance of :class:`SpectralHazardCurveFieldTree`
		"""
		model_name = slice_name
		branch_names, filespecs = [], []
		for index in branch_indexes:
			branch_names.append(self.branch_names[index])
			filespecs.append(self.filespecs[index])

		weights = self.weights[branch_indexes]
		## Recompute branch weights
		if normalize_weights:
			weight_sum = np.add.reduce(weights)
			weights /= weight_sum

		sites = self.sites
		periods = self.periods
		imt = self.imt
		intensities = self.intensities
		intensity_unit = self.intensity_unit
		timespan = self.timespan
		damping = self.damping
		#hazard_values = self._hazard_values[:,branch_indexes,:,:]
		hazard_values = self._hazard_values[branch_indexes,:,:,:]
		if self.variances is not None:
			#variances = self.variances[:,branch_indexes,:,:]
			variances = self.variances[branch_indexes,:,:,:]
		else:
			variances = None

		return self.__class__(hazard_values, branch_names, weights, sites,
							periods, intensities, intensity_unit, imt,
							model_name=model_name, filespecs=filespecs,
							timespan=timespan, damping=damping, variances=variances)

	def interpolate_return_period(self, return_period):
		"""
		Interpolate intensity measure levels for given return period

		:param return_period:
			float, return period

		:return:
			instance of :class:`UHSFieldTree`
		"""
		# TODO: this is very slow !
		num_sites = self.num_sites,
		num_periods = self.num_periods
		num_branches = self.num_branches

		rp_intensities = np.zeros((num_sites, num_branches, num_periods), dtype='d')
		if not is_empty_array(self.mean):
			rp_mean = np.zeros((num_sites, num_periods), dtype='d')
		else:
			rp_mean = None
		if not is_empty_array(self.percentiles):
			perc_shape = (num_sites, num_periods, self.num_percentiles)
			rp_percentiles = np.zeros(perc_shape, dtype='d')
		else:
			rp_percentiles = None
		interpol_exceedance = 1. / return_period
		for i in range(num_sites):
			for k in range(num_periods):
				for j in range(num_branches):
					rp_intensities[i,j,k] = interpolate(self.exceedance_rates[i,j,k],
														self.intensities[k],
														[interpol_exceedance])[0]
				if not is_empty_array(self.mean):
					mean_exc = self.mean[i,k].to_exceedance_rates(self.timespan)
					rp_mean[i,k] = interpolate(mean_exc, self.intensities[k],
												[interpol_exceedance])[0]
				if not is_empty_array(self.percentiles):
					for p in range(self.num_percentiles):
						perc_exc = self.percentiles[i,k,:,p].to_exceedance_rates(
																	self.timespan)
						rp_percentiles[i,k,p] = interpolate(perc_exc,
															self.intensities[k],
															[interpol_exceedance])[0]

		return UHSFieldTree(self.branch_names, self.weights,
							self.sites, self.periods,
							rp_intensities, self.intensity_unit, self.imt,
							model_name=self.model_name, filespecs=self.filespecs,
							timespan=self.timespan, damping=self.damping,
							return_period=return_period, mean=rp_mean,
							percentile_levels=self.percentile_levels,
							percentiles=rp_percentiles)

	def interpolate_periods(self, out_periods):
		"""
		Interpolate intensity measure levels at different spectral periods

		:param out_periods:
			list or array of output spectral periods

		:return:
			instance of :class:`SpectralHazardCurveFieldTree`
		"""
		num_sites = self.num_sites
		num_branches = self.num_branches
		num_intensities = self.num_intensities

		#out_shape = (num_sites, num_branches, len(out_periods), num_intensities)
		out_shape = (num_branches, num_sites, len(out_periods), num_intensities)
		out_hazard_values = np.zeros(out_shape, dtype='d')
		if self.variances is not None:
			out_variances = np.zeros(out_shape, dtype='d')
		else:
			out_variances = None
		if self.mean is not None:
			out_mean = np.zeros((num_sites, len(out_periods), num_intensities), dtype='d')
		else:
			out_mean = None
		if self.percentiles is not None:
			num_percentiles = self.num_percentiles
			perc_shape = (num_sites, len(out_periods), num_intensities, num_percentiles)
			out_percentiles = np.zeros(perc_shape, dtype='d')

		for i in range(num_sites):
			for j in range(num_branches):
				shc = self.get_spectral_hazard_curve(site_spec=i, branch_spec=j)
				shc_out = shc.interpolate_periods(out_periods)
				out_hazard_values[i,j] = shc_out._hazard_values
				if self.variances is not None:
					out_variances[i,j] = shc_out.variances
			if self.mean is not None:
				shc = SpectralHazardCurve(self.mean[i], self.sites[i], self.periods,
									self.intensities, self.intensity_unit, self.imt,
									model_name="mean", filespec=None,
									timespan=self.timespan, damping=self.damping)
				shc_out = shc.interpolate_periods(out_periods)
				out_mean[i] = shc_out._hazard_values
			if self.percentiles is not None:
				for p in range(num_percentiles):
					model_name = "P%02d" % self.percentile_levels[p]
					shc = SpectralHazardCurve(self.percentiles[i,:,:,p],
									self.sites[i], self.periods,
									self.intensities, self.intensity_unit, self.imt,
									model_name=model_name, filespec=None,
									timespan=self.timespan, damping=self.damping)
					shc_out = shc.interpolate_periods(out_periods)
					out_percentiles[i,:,:,p] = shc_out._hazard_values
		intensities = shc_out.intensities

		return SpectralHazardCurveFieldTree(out_hazard_values, self.branch_names,
									self.weights, self.sites, out_periods,
									intensities, self.intensity_unit, self.imt,
									model_name=self.model_name,
									filespecs=self.filespecs,
									timespan=self.timespan, damping=self.damping,
									variances=out_variances)

	def to_hc_collection(self, site_spec=0, period_spec=0, branch_specs=[],
						labels=[], colors=[], linestyles=[], linewidths=[],
						add_mean=False, add_percentiles=False,
						emphasize_mean=True, lang="en"):
		"""
		Convert hazard curves (individual branches, mean, and percentiles)
		for a particular site and spectral period to hazard curve collection

		:param site_spec:
			site specification (index, (lon,lat) tuple or site name)
			of site to be included in collection
			(default: 0)
		:param period_spec:
			period specification (integer period index or float spectral
			period)
			(default: 0)
		:param branch_specs:
			list of branch specifications (indexes or branch names)
			to be included
			(default: [] will include all branches)
		:param labels:
		:param colors:
		:param linestyles:
		:param linewidths:
			see :class:`HazardCurveCollection`
		:param add_mean:
			bool, whether or not to add the logic-tree mean to the
			collection
			(default: False)
		:param add_percentiles:
			bool, whether or not to add logic-tree precentiles to the
			collection
			(default: False)
		:param emphasize_mean:
			bool, whether or not to emphasize the mean if it is added
			(if True, an additional mean UHS will be added with
			appropriate color and width)
			(default: True)
		:param lang:
			str, language for mean label if it is added: "en", "nl" or "fr"
			(default: "en")

		:return:
			instance of :class:`HazardCurveCollection`
		"""
		## Determine, site, period, branches
		site_index = self.site_index(site_spec)
		period_index = self.period_index(period_spec)
		if branch_specs in ([], None):
			branch_indexes = range(self.num_branches)
		else:
			branch_indexes = [self.branch_index(branch_spec)
							for branch_spec in branch_specs]
		branch_names = [self.branch_names[i] for i in branch_indexes]
		num_branches = len(branch_indexes)

		## Hazard curves
		hc_list = []
		for branch_index in branch_indexes:
			shc = self.get_spectral_hazard_curve(branch_index, site_index)
			hc = shc.get_hazard_curve(period_index)
			hc_list.append(hc)
		if add_mean:
			mean_shcf = self.get_mean_spectral_hazard_curve_field()
			mean_hc = mean_shcf.get_hazard_curve(site_index, period_index)
			hc_list.append(mean_hc)
			if emphasize_mean:
				hc_list.append(mean_hc)
		if add_percentiles:
			if self.percentile_levels is None:
				percentile_levels = [5, 16, 50, 84, 95]
			else:
				percentile_levels = self.percentile_levels
			for perc in percentile_levels:
				perc_shcf = self.get_percentile_shcf(perc)
				perc_hc = perc_shcf.get_hazard_curve(site_index, period_index)
				hc_list.append(perc_hc)

		## Labels
		if labels in (None, []):
			if len(hc_list) <= 6:
				labels = branch_names
			else:
				labels = ["_nolegend_"] * num_branches
			if add_mean:
				if emphasize_mean:
					labels.append("_nolegend_")
				labels.append({"en": "Overall Mean",
								"nl": "Algemeen gemiddelde",
								"fr": "Moyenne globale"}.get(lang, 'Mean'))
			if add_percentiles:
				perc_labels = {}
				p = 0
				for perc in percentile_levels:
					if not perc in perc_labels:
						if not (100 - perc) in perc_labels:
							perc_labels[perc] = "P%02d" % perc
							p += 1
						else:
							perc_labels[100 - perc] += ", P%02d" % perc
							perc_labels[perc] = "_nolegend_"
				labels.extend(perc_labels.values())

		## Colors
		if colors in (None, []):
			if add_mean and add_percentiles:
				colors = [(0.5, 0.5, 0.5)] * num_branches
			if add_mean:
				if emphasize_mean:
					colors.append('w')
				colors.append('r')
			if add_percentiles:
				perc_colors = {}
				p = 0
				for perc in percentile_levels:
					if not perc in perc_colors:
						if not (100 - perc) in perc_colors:
							perc_colors[perc] = ["b", "g", "r", "c", "m", "k"][p%6]
							p += 1
						else:
							perc_colors[perc] = perc_colors[100 - perc]
				colors.extend(perc_colors.values())

		## Linewidths
		if linewidths in (None, []):
			if add_mean and add_percentiles:
				linewidths = [1] * num_branches
				if add_mean:
					if emphasize_mean:
						linewidths.append(5)
					linewidths.append(3)
				if add_percentiles:
					linewidths.extend([2] * len(percentile_levels))

		## Linestyles
		if linestyles in (None, []):
			linestyles = ['-'] * num_branches
			if add_mean:
				if emphasize_mean:
					linestyles.append('-')
				linestyles.append('-')
			if add_percentiles:
				linestyles.extend(['--'] * len(percentile_levels))

		return HazardCurveCollection(hc_list, labels=labels, colors=colors,
									linestyles=linestyles, linewidths=linewidths)


	# TODO
	def plot(self, site_spec=0, period_spec=0, branch_specs=[],
			add_mean=True, add_percentiles=True, emphasize_mean=True,
			labels=[], colors=[], linestyles=[], linewidths=[],
			intensity_unit=None, yaxis="exceedance_rate",
			timespan=None, interpol_val=None, interpol_range=None,
			interpol_prop='return_period', legend_location=0,
			xscaling='lin', yscaling='log', xgrid=1, ygrid=1,
			title="", fig_filespec=None, lang="en", **kwargs):
		"""
		Plot hazard curves (individual branches, mean, and percentiles)
		for a particular site and spectral period.

		:param site_spec:
		:param period_spec:
		:param branch_specs:
		:param add_mean:
		:param add_percentiles:
		:param emphasize_mean:
			see :meth:`to_hc_collection`
		"""
		if title is None:
			title = "Hazard Curve Tree"
			title += "\nSite: %s, T: %s s"
			title %= (self.site_names[site_index], self.periods[period_index])

		hcc = self.to_hc_collection(site_spec, period_spec, branch_specs,
									labels=labels, colors=colors,
									linestyles=linestyles, linewidths=linewidths,
									add_mean=add_mean,
									add_percentiles=add_percentiles,
									emphasize_mean=emphasize_mean, lang=lang)

		return hcc.plot(intensity_unit=intensity_unit, yaxis=yaxis,
						timespan=timespan, interpol_val=interpol_val,
						interpol_range=interpol_range, interpol_prop=interpol_prop,
						legend_location=legend_location, xscaling=xscaling,
						yscaling=yscaling, xgrid=xgrid, ygrid=ygrid,
						title=title, fig_filespec=fig_filespec, lang=lang, **kwargs)

	plot.__doc__ += HazardCurveCollection.plot.__doc__

	def plot_subsets(self, subset_label_patterns, site_spec=0, period_spec=0,
					labels=[], colors=[], linewidths=[2,1],
					agr_filespecs=[], percentile_levels=[84],
					combined_uncertainty=True, **kwargs):
		"""
		Plot mean and percentiles of different subsets

		:param subset_label_patterns:
			list of strings that are unique to the branch names of
			each subset
		:param site_spec:
			site specification (index, (lon,lat) tuple or site name)
			of site to be included in collection
			(default: 0)
		:param period_spec:
			period specification (integer period index or float spectral
			period)
			(default: 0)
		:param labels:
			list of strings, subset labels
			(default: [])
		:param colors:
			list of matplotlib color specifications for each subset
			(default: [])
		:param linewidths:
			list of 2 linewidths, one for mean curves and one for
			percentile curves
			(default: [2,1])
		:param agr_filespecs:
			list of .AGR filespecs containing statistics for each
			subset. If empty, mean and percentiles will be computed
			(default: [])
		:param percentile_levels:
			list of exceedance-rate percentiles to plot in addition
			to the mean
			(default: [84])
		:param combined_uncertainty:
			bool, If True, percentiles are calculated for combined
			(epistemic + aleatory) uncertainty. If False, percentiles are
			calculated for epistemic uncertainty only. This setting does
			not apply if agr_filespec is set. Note that this setting
			does not influence the mean value.
			(default: True)
		:kwargs:
			additional keyword arguments supported by :meth:`plot`

		:return:
			matplotlib Axes instance
		"""
		subsets = self.split_by_branch_name(subset_label_patterns)
		site_index = self.site_index(site_spec)
		period_index = self.period_index(period_spec)

		subset_colors = colors or pylab.rcParams['axes.prop_cycle'].by_key()['color']
		mp_linewidths = linewidths or [2, 1]

		if not labels:
			#subset_labels = ["Subset %d" % (i+1) for i in range(len(subsets))]
			subset_labels = subset_label_patterns
		else:
			subset_labels = labels

		## Manage percentile labels and linestyles
		perc_labels, perc_linestyles = {}, {}
		p = 0
		for perc in percentile_levels:
			if not perc in perc_labels:
				if not (100 - perc) in perc_labels:
					perc_labels[perc] = "P%02d" % perc
					perc_linestyles[perc] = ["--", ":", "-:"][p%3]
					p += 1
				else:
					perc_labels[100 - perc] += ", P%02d" % perc
					perc_labels[perc] = "_nolegend_"
					perc_linestyles[perc] = perc_colors[100 - perc]

		intensities = self.intensities[period_index]
		site = self.sites[site_index]
		period = self.periods[period_index]

		hc_list, labels, colors, linewidths, linestyles = [], [], [], [], []
		for i, subset in enumerate(subsets):
			## Load or calculate statistics
			if not agr_filespecs:
				mean = subset.calc_mean()
				if combined_uncertainty:
					percentiles = subset.calc_percentiles_combined(percentile_levels)
				else:
					percentiles = subset.calc_percentiles_epistemic(percentile_levels)
			else:
				shcft = import_stats_from_AGR(agr_filespecs[i], percentile_levels)
				mean = shcft.mean
				percentiles = shcft.percentiles

			## Plot subset mean
			label = subset_labels[i]
			label += {"en": " (mean)",
					"nl": " (gemiddelde)",
					"fr": " (moyenne)"}.get(lang, " (mean)")
			labels.append(label)
			colors.append(subset_colors[i%len(subset_colors)])
			linewidths.append(mp_linewidths[0])
			linestyles.append('-')
			hc = HazardCurve(mean[site_index,period_index], site, period,
							intensities, self.intensity_unit, self.imt,
							model_name="mean", filespec=None,
							timespan=self.timespan, damping=self.damping)
			hc_list.append(hc)

			## Plot percentiles
			for p, perc in enumerate(percentile_levels):
				perc_label = perc_labels[perc]
				labels.append(dataset_labels[i] + " (%s)" % perc_label)
				colors.append(dataset_colors[i%len(dataset_colors)])
				linewidths.append(mp_linewidths[1])
				linestyles.append(perc_linestyles[perc])
				hc = HazardCurve(percentiles[site_index,period_index,:,p], site,
								period, intensities, self.intensity_unit, self.imt,
								model_name=perc_label, filespec=None,
								timespan=self.timespan, damping=self.damping)
				hc_list.append(hc)

		hc_collection = HazardCurveCollection(hc_list, labels=labels, colors=colors,
										linestyles=linestyles, linewidths=linewidths)

		return hc_collection.plot(**kwargs)

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML SpectralHazardCurveField element)

		:param encoding:
			str, unicode encoding
			(default: 'latin1')

		:return:
			instance of :class:`etree.Element`
		"""
		# TODO: add names to nrml namespace
		shcft_elem = etree.Element(ns.SPECTRAL_HAZARD_CURVE_FIELD_TREE)
		shcft_elem.set(ns.NAME, self.model_name)
		shcft_elem.set(ns.imt, self.imt)
		for j, branch_name in enumerate(self.branch_names):
			shcf_elem = etree.SubElement(shcft_elem, ns.SPECTRAL_HAZARD_CURVE_FIELD)
			shcf_elem.set(ns.NAME, branch_name)
			for k, period in enumerate(self.periods):
				hcf_elem = etree.SubElement(shcf_elem, ns.HAZARD_CURVE_FIELD)
				hcf_elem.set(ns.PERIOD, str(period))
				if period > 0:
					hcf_elem.set(ns.DAMPING, str(self.damping))
				imls_elem = etree.SubElement(hcf_elem, ns.IMLS)
				imls_elem.text = " ".join(map(str, self.intensities[k,:]))
				for i, site in enumerate(self.sites):
					hazard_curve_elem = etree.SubElement(hcf_elem, ns.HAZARD_CURVE)
					point_elem = etree.SubElement(hazard_curve_elem, ns.POINT)
					position_elem = etree.SubElement(point_elem, ns.POSITION)
					position_elem.text = "%s %s" % (site[0], site[1])
					poes_elem = etree.SubElement(hazard_curve_elem, ns.POES)
					poes_elem.text = " ".join(map(str, self.poes[i,j,k,:]))
		return shcf_elem

	def write_nrml(self, filespec, encoding='latin1', pretty_print=True):
		"""
		Write spectral hazard curve field tree to XML file

		:param filespec:
			str, full path to XML output file
		:param encoding:
			str, unicode encoding
			(default: 'utf-8')
		pretty_print:
			bool, indicating whether or not to indent each element
			(default: True)
		"""
		tree = create_nrml_root(self, encoding=encoding)
		if filespec:
			## Note: open file in binary mode in PY3, as etree.tostring returns bytes
			fd = open(filespec, "wb")
		else:
			fd = sys.stdout
		tree.write(fd, xml_declaration=True, encoding=encoding,
					pretty_print=pretty_print)
		fd.close()


if __name__ == "__main__":
	import rhlib.crisis.IO as IO

	## Convert CRISIS .MAP file to HazardMapSet
	#filespec = "Test files\\CRISIS\\VG_Ambr95DD_Leynaud_EC8.MAP"
	filespec = "D:\\PSHA\\LNE\\CRISIS\\VG_Ambr95DD_Leynaud_EC8.MAP"
	hazardmapset = IO.read_MAP(filespec, model_name="VG_Ambr95DD_Leynaud_EC8", convert_to_g=True, imt="PGA", verbose=True)
	print(hazardmapset.longitudes)
	for hazardmap in hazardmapset:
		print(hazardmap.intensities.shape)
		print(hazardmap.return_period)
	hazardmap = hazardmapset[0]
	intensity_grid = hazardmap.create_grid()
	print(intensity_grid.shape)
	print(hazardmap.min(), hazardmap.max())
	print(hazardmap.poe)
	#hazardmap.export_VM("C:\\Temp\\hazardmap.grd")
	hazardmap.plot(parallel_interval=0.5, hide_sea=True, want_basemap=False)
	print


	## Convert CRISIS .GRA file to SpectralHazardCurveField
	"""
	#filespec = "Test files\\CRISIS\\MC000.GRA"
	#filespec = "D:\\PSHA\\NIRAS\\LogicTree\\PGA\\Seismotectonic\\BergeThierry2003\\3sigma\\Mmax+0_00\\MC000.GRA"
	#filespec = "D:\\PSHA\\NIRAS\\LogicTree\\Spectral\\Seismotectonic\\BergeThierry2003\\3sigma\\Mmax+0_00\\MC000.GRA"
	filespec = "D:\\PSHA\\NIRAS\\LogicTree\\Spectral\\Spectral_BT2003.AGR"
	#filespec = "D:\\PSHA\\LNE\\CRISIS\\VG_Ambr95DD_Leynaud_EC8.gra"
	shcf = IO.read_GRA(filespec, verbose=True)
	print(shcf.intensities.shape, shcf.exceedance_rates.shape)
	print(shcf.mean.shape)
	print(shcf.percentiles.shape)
	print(shcf.percentile_levels)
	print(shcf.periods)
	print(shcf.sites)
	shc = shcf[0]
	hc = shc[0]
	print(hc.interpolate_return_periods([1E+3, 1E+4, 1E+5]))
	#hc.plot(want_poe=False)
	UHSfieldset = shcf.interpolate_return_periods([10000])
	UHSfield = UHSfieldset[0]
	#UHSfield.plot()
	uhs = UHSfield[0]
	print(uhs.periods, uhs.intensities)
	print(uhs[1./34])
	uhs.export_csv()
	uhs.plot(linestyle='x', Tmin=1./40, Tmax=1./30, amin=0, color='r')
	print(UHSfieldset.return_periods)
	print(UHSfieldset.intensities)
	print(UHSfieldset.poes)
	hazardmapset = UHSfieldset.interpolate_period(1./34)
	print(hazardmapset.return_periods)
	print(hazardmapset.poes)
	hazardmap = hazardmapset[-1]
	print(hazardmap.poe)
	"""


	## CRISIS --> HazardCurveFieldTree
	"""
	import hazard.psha.BEST_IRE.LogicTree as LogicTree
	GRA_filespecs = LogicTree.slice_logictree()
	hcft = IO.read_GRA_multi(GRA_filespecs, model_name="Test")
	hcft.plot()
	#hcft.write_statistics_AGR("C:\\Temp\\Test.AGR", weighted=False)
	#hcft2 = hcft.slice_by_branch_indexes(range(50), "Subset")
	hcft2 = hcft.slice_by_branch_names(["BergeThierry"], "BergeThierry", strict=False)
	print(len(hcft2))
	import np.random
	hcft2.weights = np.random.rand(len(hcft2))
	uhsft = hcft2.interpolate_return_period(10000)
	uhsft.plot_histogram()
	#hcft.plot()
	hcft.plot_subsets(["BergeThierry", "Ambraseys"], labels=["BergeThierry", "Ambraseys"], percentile_levels=[84,95])
	hcf = hcft[0]
	print(hcf.periods)
	hcf.plot()
	#hcf2 = hcf.interpolate_periods([0.05, 0.275, 1, 2, 10])
	shc = hcf.get_spectral_hazard_curve(0)
	#shc.plot()
	uhs = shc.interpolate_return_period(1E4)
	#uhs.plot(plot_freq=True)
	"""