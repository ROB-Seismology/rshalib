# -*- coding: utf-8 -*-
# pylint: disable=W0142, W0312, C0103, R0913
"""
Blueprint for classes representing hazard results of both OpenQuake and CRISIS
"""

from __future__ import absolute_import, division, print_function, unicode_literals


### imports
import os, sys

import numpy as np

from scipy.stats import scoreatpercentile

from ..site import GenericSite
from ..utils import interpolate
from ..pmf import NumericPMF

from .plot import plot_hazard_spectra, plot_histogram
from .base_array import *
from .hc_base import *

from .response_spectrum import ResponseSpectrum
from .hazard_map import (HazardMap, HazardMapSet)



__all__ = ['UHS', 'UHSField', 'UHSFieldSet', 'UHSFieldTree',
			'UHSCollection']


class UHS(HazardResult, ResponseSpectrum):
	"""
	Uniform Hazard Spectrum, this is a response spectrum corresponding
	to a particular return period or probability of exceedance

	:param periods:
	:param intensities:
	:param intensity_unit:
	:param imt:
		see :class:`ResponseSpectrum`
	:param site:
		instance of :class:`rshalib.site.GenericSite`
	:param model_name:
		str, name of hazard model
		(default: "")
	:param filespec:
		str, full path to file corresponding to this UHS
		(default: None)
	:param timespan:
		float, time span corresponding to probability of exceedance
		(default: 50)
	:param poe:
		float, probability of exceedance
		(default: None)
	:param return_period:
		float, return period
		(default: None)
		Note: either return period or poe must be specified!
	:param damping:
		see :class:`ResponseSpectrum`
	"""
	## Ensure compatibility with ResponseSpectrum class
	_opt_kwargs = ['imt', 'model_name', 'site', 'filespec',
						'timespan', 'poe', 'return_period']

	def __init__(self, periods, intensities, intensity_unit, imt,
				site, model_name="", filespec=None,
				timespan=50, poe=None, return_period=None,
				damping=0.05):
		assert (poe and timespan) or return_period
		if return_period:
			hazard_values = ExceedanceRateArray([1./return_period])
		elif poe:
			hazard_values = ProbabilityArray([poe])

		HazardResult.__init__(self, hazard_values, timespan=timespan, imt=imt,
							intensities=intensities, intensity_unit=intensity_unit,
							damping=damping)
		ResponseSpectrum.__init__(self, periods, intensities, intensity_unit, imt,
								damping=damping, model_name=model_name)

		self.site = site
		self.filespec = filespec

	def __repr__(self):
		txt = '<UHS %s "%s" | T: %s - %s s (n=%d) | %d%% damping>'
		txt %= (self.imt, self.model_name, self.Tmin, self.Tmax, len(self),
				self.damping*100)
		return txt

	def __getitem__(self, period_spec):
		"""
		:param period_spec:
			int: period index
			float: spectral period

		:return:
			float, intensity corresponding to given period
		"""
		period_index = self.period_index(period_spec)
		try:
			intensity = self.intensities[period_index]
		except IndexError:
			raise IndexError("Period index %s out of range" % period_index)
		else:
			return intensity

	def __add__(self, other_uhs):
		raise Exception("UHSs cannot gennerally be summed!")

	@property
	def poe(self):
		return self.poes[0]

	@property
	def return_period(self):
		return self.return_periods[0]

	@property
	def exceedance_rate(self):
		return self.exceedance_rates[0]

	@property
	def site_name(self):
		return self.site.name

	def plot(self, label=None, color="k", linestyle="-", linewidth=2,
			pgm_period=0.02, plot_freq=False, intensity_unit="g",
			title=None,  legend_location=0, lang="en",
			fig_filespec=None, **kwargs):
		"""
		Plot UHS

		:param color:
			matplotlib color specification
			(default: 'k')
		:param linestyle:
			matplotlib linestyle specification
			(default: "-")
		:param linewidth:
			float, line width
			(default: 2)
		:param pgm_period:
		:param plot_freq:
		:param intensity_unit:
		:param title:
		:param legend_location:
		:param lang:
		:param fig_filespec:
		:kwargs:
			see :func:`rshalib.result.plot.plot_hazard_spectra`

		:return:
			matplotlib Axes instance
		"""
		from .plot import plot_hazard_spectra

		if title is None:
			title = "UHS"
			title += "\nSite: %s, Return period: %d yr"
			title %= (self.site_name, self.return_periods[0])
		kwargs['title'] = title

		if label is None:
			label = self.model_name
		kwargs['labels'] = [label]

		kwargs['colors'] = [color]
		kwargs['linestyles'] = [linestyle]
		kwargs['linewidths'] = [linewidth]
		kwargs['pgm_period'] = pgm_period
		kwargs['plot_freq'] = plot_freq
		kwargs['intensity_unit'] = intensity_unit
		kwargs['legend_location'] = legend_location
		kwargs['lang'] = lang
		kwargs['fig_filespec'] = fig_filespec

		return plot_hazard_spectra([self], **kwargs)

	@classmethod
	def from_csv_file(self, csv_filespec, site, col_spec=1, intensity_unit="g",
						damping=0.05, model_name="",
						timespan=50, poe=None, return_period=None):
		"""
		Read UHS from a csv file.
		First line should contain column names
		First column should contain periods or frequencies,
		subsequent column()s should contain intensities, only one of which
		will be read.

		:param csv_filespec:
			str, full path to csv file
		:param site:
			GenericSite object, representing site for which UHS was computed
		:param col_spec:
			str or int, name or index of column containing intensities
			to be read
			(default: 1)
		:param intensity_unit:
			str, unit of intensities in csv file, will be overridden
			if specified between parentheses in the column name
			(default: "g")
		:param damping:
			float, damping corresponding to UHS
			(default: 0.05)
		:param model_name:
			str, name or description of model
			(default: "")
		:param timespan:
			float, time span for UHS
			(default: 50)
		:param poe:
			float, probability of exceedance for UHS
			(default: None)
		:param return_period:
			float, return period for UHS
			(default: None)

		:return:
			instance of :class:`UHS`

		Note: either poe or return_period should be specified
		"""
		return super(UHS, self).from_csv_file(csv_filespec, col_spec=col_spec,
					unit=intensity_unit, damping=damping,
					site=site, model_name=model_name,
					timespan=timespan, poe=poe, return_period=return_period)

	def to_response_spectrum(self):
		"""
		Convert UHS to generic response spectrum

		:return:
			instance of :class:`ResponseSpectrum`
		"""
		return ResponseSpectrum(self.periods, self.intensities, self.intensity_unit,
								self.imt, self.model_name)


class UHSField(HazardSpectrum, HazardResult, HazardField):
	"""
	UHS Field, i.e. UHS at a number of sites for the same return period

	:param sites:
		1-D list [i] with:
		- (lon, lat) tuples of all sites or
		- instances of :class:`rshalib.site.GenericSite`
	:param periods:
		1-D array [k] with spectral periods
	:param intensities:
		2-D array [i, k] with intensity values for 1 return period
	:param intensity_unit:
	:param imt:
	:param model_name:
	:param filespec:
	:param timespan:
	:param poe:
	:param return_period:
	:param damping:
		see :class:`UHS`
	:param vs30s:
		1-D array [i] of VS30 values for each site
		(default: None)
	"""
	def __init__(self, sites, periods, intensities, intensity_unit, imt,
				model_name="", filespec=None,
				timespan=50, poe=None, return_period=None,
				damping=0.05, vs30s=None):
		assert (poe and timespan) or return_period
		if return_period:
			hazard_values = ExceedanceRateArray([1./return_period])
		elif poe:
			hazard_values = ProbabilityArray([poe])

		HazardSpectrum.__init__(self, periods, period_axis=1)
		HazardResult.__init__(self, hazard_values, timespan=timespan, imt=imt,
							intensities=intensities, intensity_unit=intensity_unit,
							damping=damping)
		HazardField.__init__(self, sites)

		self.model_name = model_name
		self.filespec = filespec
		self.vs30s = vs30s

	def __repr__(self):
		txt = '<UHSField %s "%s" | T: %s - %s s (n=%d) | %d sites>'
		txt %= (self.imt, self.model_name, self.Tmin, self.Tmax, len(self),
				self.num_sites)
		return txt

	def __iter__(self):
		for i in range(self.num_sites):
			yield self.get_uhs(i)

	def __getitem__(self, site_spec):
		return self.get_uhs(site_spec)

	@property
	def poe(self):
		return self.poes[0]

	@property
	def return_period(self):
		return self.return_periods[0]

	@property
	def exceedance_rate(self):
		return self.exceedance_rates[0]

	def min(self):
		return self.intensities.min(axis=0)

	def max(self):
		return self.intensities.max(axis=0)

	def get_uhs(self, site_spec=0):
		"""
		Extract UHS for a particular site

		:param site_spec:
			site specification:
			- int: site index
			- str: site name
			- instance of :class:`rshalib.site.GenericSite`: site
			- (lon, lat) tuple
			(default: 0)

		:return:
			instance of :class:`UHS`
		"""
		site_index = self.site_index(site_spec)
		try:
			site = self.sites[site_index]
		except:
			raise IndexError("Site index %s out of range" % site_index)
		else:
			site_name = self.site_names[site_index]
			model_name = self.model_name + " - " + site_name
			intensities = self.intensities[site_index]
			return UHS(self.periods, intensities, self.intensity_unit, self.imt,
					site, model_name=model_name, filespec=self.filespec,
					timespan=self.timespan, return_period=self.return_period,
					damping=self.damping)

	def get_hazard_map(self, period_spec=0):
		"""
		Extract hazard map for a particular spectral period

		:param period_spec:
			period specification:
			- int: period index
			- float: spectral period
			(default: 0)

		:return:
			instance of class:`rshalib.result.HazardMap`
		"""
		period_index = self.period_index(period_spec)
		try:
			period = self.periods[period_index]
		except:
			raise IndexError("Period index %s out of range" % period_index)
		intensities = self.intensities[:, period_index]
		return HazardMap(self.sites,
						intensities, self.intensity_unit, self.imt,
						period,
						model_name=self.model_name, filespec=self.filespec,
						timespan=self.timespan, return_period=self.return_period,
						damping=damping, vs30s=self.vs30s)

	def interpolate_period(self, period):
		"""
		Interpolate UHS field at given period

		:param period:
			float, spectral period

		:return:
			instance of :class:`HazardMap`
		"""
		period_intensities = np.zeros(self.num_sites)
		for i in range(num_sites):
			period_intensities[i] = interpolate(self.periods,
												self.intensities[i], [period])

		return HazardMap(self.sites,
						period_intensities, self.intensity_unit, self.imt,
						period,
						model_name=self.model_name, filespec=self.filespec,
						timespan=self.timespan, return_period=self.return_period,
						damping=damping, vs30s=self.vs30s)

	def to_uhs_collection(self, labels=[], colors=[], linestyles=[], linewidths=[]):
		"""
		Convert UHS field to UHS collection

		:param labels:
		:param colors:
		:param linestyles:
		:param linewidths:
			see :class:`UHSCollection`

		:return:
			instance of :class:`UHSCollection`
		"""
		uhs_list = [self.get_uhs(i) for i in range(self.num_sites)]
		if labels in ([], None):
			labels = self.site_names

		return UHSCollection(uhs_list, labels=labels, colors=colors,
							linestyles=linestyles, linewidths=linewidths)

	# TODO: previous plot function supported site_specs argument,
	# but it would be better to implement this as a separate method (slice_sites)
	def plot(self, **kwargs):
		"""
		Convert to UHS collection and plot

		:kwargs:
			keyword arguments understood by
			:func:`rshalib.result.plot.plot_hazard_spectra`

		:return:
			matplotlib Axes instance
		"""
		labels = kwargs.pop('labels', [])
		colors = kwargs.pop('colors', [])
		linestyles = kwargs.pop('linestyles', [])
		linewidths = kwargs.pop('linewidths', [])
		title = kwargs.pop('title')
		if title is None:
			title = "Model: %s\n" % self.model_name
			title += "UHS for return period %.3G yr" % self.return_period

		uhs_collection = self.to_uhs_collection(labels=labels, colors=colors,
									linestyles=linestyles, linewdiths=linewidths)
		return uhs_collection.plot(title=title, **kwargs)

#	def plot(self, site_specs=None, colors=[], linestyles=[], linewidths=[],
#			fig_filespec=None, title=None, plot_freq=False, plot_style="loglin",
#			Tmin=None, Tmax=None, amin=None, amax=None, pgm_period=0.02,
#			axis_label_size='x-large', tick_label_size='large',
#			legend_label_size='large', legend_location=0, lang="en"):
#		if site_specs in (None, []):
#			site_indexes = range(self.num_sites)
#		else:
#			site_indexes = [self.site_index(site_spec) for site_spec in site_specs]
#		sites = [self.sites[site_index] for site_index in site_indexes]

#		if title is None:
#			title = "Model: %s\n" % self.model_name
#			title += "UHS for return period %.3G yr" % self.return_period
#		datasets, pgm, labels = [], [], []
#		x = self.periods
#		for site_index in site_indexes:
#			y = self.intensities[site_index]
#			datasets.append((x[self.periods>0], y[self.periods>0]))
#			if 0 in self.periods:
#				pgm.append(y[self.periods==0])
#			labels.append(self.site_names[site_index])
#		intensity_unit = self.intensity_unit
#		plot_hazard_spectrum(datasets, pgm=pgm, pgm_period=pgm_period, labels=labels,
#						colors=colors, linestyles=linestyles, linewidths=linewidths,
#						fig_filespec=fig_filespec, title=title, plot_freq=plot_freq,
#						plot_style=plot_style, Tmin=Tmin, Tmax=Tmax, amin=amin,
#						amax=amax, intensity_unit=intensity_unit, axis_label_size=axis_label_size,
#						tick_label_size=tick_label_size, legend_label_size=legend_label_size,
#						legend_location=legend_location, lang=lang)


class UHSFieldSet(HazardSpectrum, HazardResult, HazardField):
	"""
	Set of UHS fields for different return periods

	:param sites:
		1-D list [i] with (lon, lat) tuples of all sites
	:param periods:
		1-D array [k] with spectral periods
	:param intensities:
		3-D array [p, i, k] with interpolated intensity values at
		different poE's or return periods [p]
	:param intensity_unit:
	:param imt
	:param model_name:
		see :class:`UHSField`
	:param filespecs:
		list of strings [p], paths to files corresponding to UHS fields
		(default: [])
	:param timespan:
		float, time span corresponding to exceedance probabilities
		(default: 50)
	:param poes:
		1-D array or probabilities of exceedance [p]
		(default: None)
	:param return_periods:
		1-D [p] array of return periods
		(default: None)
	:param damping:
		see :class:`UHSField`
	"""
	def __init__(self, sites, periods, intensities, intensity_unit, imt,
				model_name="", filespecs=[],
				timespan=50, poes=None, return_periods=None,
				damping=0.05):
		if not is_empty_array(return_periods):
			hazard_values = ExceedanceRateArray(1./return_periods)
		elif not is_empty_array(poes):
			hazard_values = ProbabilityArray(poes)
		else:
			raise Exception("Either return periods or exceedance probabilities "
							" should be specified!")

		HazardSpectrum.__init__(self, periods)
		HazardResult.__init__(self, hazard_values, timespan=timespan, imt=imt,
							intensities=intensities, intensity_unit=intensity_unit,
							damping=damping)
		HazardField.__init__(self, sites)

		self.model_name = model_name
		if len(filespecs) == 1:
			filespecs *= len(self.return_periods)
		self.filespecs = filespecs

	def __repr__(self):
		txt = '<UHSFieldSet %s "%s" | T: %s - %s s (n=%d) | %d sites | nTr=%d>'
		txt %= (self.imt, self.model_name, self.Tmin, self.Tmax, len(self),
				self.num_sites, len(self))
		return txt

	def __iter__(self):
		for i in range(len(self)):
			yield self.get_uhs_field(index=i)

	def __getitem__(self, index):
		return self.get_uhs_field(index=index)

	def __len__(self):
		return len(self.return_periods)

	def get_uhs_field(self, index=None, poe=None, return_period=None):
		"""
		Extract UHS field

		:param index:
			int, index of UHS field in field set
			(default: None)
		:param poe:
			float, probability of exceedance to extract
			(default: None)
		:param return_period:
			float, return period to extract
			(default: None)

		:return:
			instance of :class:`UHSField`
		"""
		if index is None:
			if poe is not None:
				index = np.where(np.abs(self.poes - poe) < 1E-12)[0]
				if len(index) == 0:
					raise ValueError("No UHS field for poE=%s" % poe)
				else:
					index = index[0]
			elif return_period is not None:
				index = np.where(np.abs(self.return_periods - return_period) < 1E-1)[0]
				if len(index) == 0:
					raise ValueError("No UHS field for return period=%s yr"
									% return_period)
				else:
					index = index[0]

		try:
			return_period = self.return_periods[index]
		except:
			raise IndexError("Index %s out of range" % index)
		else:
			filespec = self.filespecs[index] if len(filespecs) > 0 else ""
			intensities = self.intensities[index]
			return UHSField(self.sites, self.periods, intensities,
							self.intensity_unit, self.imt, model_name=model_name,
							filespec=self.filespec, timespan=self.timespan,
							return_period=self.return_period, damping=self.damping)

	def interpolate_period(self, period):
		"""
		Interpolate UHS field set at given period

		:param period:
			float, spectral period

		:return:
			instance of :class:`HazardMapSet`
		"""
		num_sites, num_rp = self.num_sites, len(self.return_periods)
		period_intensities = np.zeros((num_rp, num_sites))
		for p in range(num_rp):
			for i in range(num_sites):
				period_intensities[p,i] = interpolate(self.periods,
														self.intensities[p,i],
														[period])
		return HazardMapSet(self.sites,
							period_intensities, self.intensity_unit, self.imt,
							period,
							model_name=self.model_name, filespecs=self.filespecs,
							timespan=self.timespan, return_period=self.return_period,
							damping=damping)

	def interpolate_return_period(self, return_period):
		"""
		Interpolate UHS field at given return period

		:param return_period:
			float, return period

		:return:
			instance of :class:`UHSField`
		"""
		num_sites, num_periods = self.num_sites, self.num_periods
		rp_intensities = np.zeros((num_sites, num_periods))
		for i in range(num_sites):
			for k in range(num_periods):
				rp_intensities[i, k] = interpolate(self.return_periods,
													self.intensities[:,i,k],
													[return_period])
		return UHSField(self.sites, self.periods, rp_intensities,
						self.intensity_unit, self.imt, model_name=model_name,
						filespec=self.filespec, timespan=self.timespan,
						return_period=return_period, damping=self.damping)

	def to_uhs_collection(self, site_spec, labels=[], colors=[], linestyles=[],
						linewidths=[]):
		"""
		Extract UHS collection for a single site

		:param site_spec:
			site specification:
			- int: site index
			- str: site name
			- instance of :class:`rshalib.site.GenericSite`: site
			- (lon, lat) tuple

		:param labels:
		:param colors:
		:param linestyles:
		:param linewidths:
			see :class:`UHSCollection`

		:return:
			instance of :class:`UHSCollection`
		"""
		uhs_list = []
		for uhs_field in self:
			uhs = uhs_field.get_uhs(site_spec)
			uhs_list.append(uhs)

		if labels in (None, []):
			labels = ['Tr = %s yr' % rp for rp in self.return_periods]

		return UHSCollection(uhs_list, labels=labels, colors=colors,
							linestyles=linestyles, linewidths=linewidths)

#	def plot(self, sites=None, return_periods=None):
#		"""
#		method to plot hazard spectrum
#		arguments:
#			sites | default="All"
#		"""
#		if sites is None:
#			sites = self.sites
#		if return_periods is None:
#			return_periods = self.return_periods
#		x = self.periods
#		datasets, labels = [], []
#		for i, site in enumerate(sites):
#			for p, rp in enumerate(return_periods):
#				y = self.intensities[p,i]
#				datasets.append((x, y))
#				labels.append("%s, %s - %s" % (site[0], site[1], rp))
#		plot_hazard_spectrum(datasets, labels, plot_freq=True)


class UHSFieldTree(HazardSpectrum, HazardField, HazardTree):
	"""
	UHS Field Tree, this is a set of UHS fields for 1 return period
	and corresponding to different branches of a logic tree

	:param branch_names:
		1-D list [j] of model names of each logic-tree branch
	:param weights:
		1-D list or array [j] with branch weights
	:param sites:
	:param periods:
		see :class:`UHSField`
	:param intensities:
		3-D array [j, i, k] with intensity values for each branch, site
		and spectral period
	:param intensity_unit:
	:param imt
	:param model_name:
		see :class:`UHSField`
	:param filespecs:
		list of strings [j], paths to files corresponding to each branch
		(default: [])
	:param timespan:
	:param poe:
	:param return_period:
	:param damping:
		see :class:`UHSField`
	:param mean:
		2-D array [i,k] containing mean UHS field of all branches
		(default: None)
	:param percentile_levels:
		1-D list or array [p] with percentile levels
		(default: None)
	:param percentiles:
		3-D array [i,k,p] with percentile UHS fields of all branches
		(default: None)
	:param vs30s:
		see :class:`UHSField`
	"""
	def __init__(self, branch_names, weights,
				sites, periods, intensities, intensity_unit, imt,
				model_name="", filespecs=[],
				timespan=50, poe=None, return_period=None, damping=0.05,
				mean=None, percentile_levels=None, percentiles=None,
				vs30s=None):
		if return_period:
			hazard_values = ExceedanceRateArray([1./return_period])
		elif poe:
			hazard_values = ProbabilityArray([poe])

		HazardSpectrum.__init__(self, periods, period_axis=2)
		HazardField.__init__(self, sites)
		HazardTree.__init__(self, hazard_values, branch_names, weights=weights,
							timespan=timespan, intensities=intensities,
							intensity_unit=intensity_unit, imt=imt, damping=damping,
							mean=mean, percentile_levels=percentile_levels,
							percentiles=percentiles)

		self.model_name = model_name
		self.filespecs = filespecs
		self.vs30s = vs30s

	def __repr__(self):
		txt = '<UHSFieldTree %s "%s" | T: %s - %s s (n=%d) | %d sites | %d branches>'
		txt %= (self.imt, self.model_name, self.Tmin, self.Tmax, len(self),
				self.num_sites, self.num_branches)
		return txt

	def __iter__(self):
		for i in range(self.num_branches):
			yield self.get_uhs_field(i)

	def __getitem__(self, branch_spec):
		return self.get_uhs_field(branch_spec)

	def get_uhs_field(self, branch_spec):
		"""
		Extract UHS field corresponding to a particular branch

		:param branch_spec:
			Branch specification:
			- int: branch index
			- str: branch name

		:return:
			instance of :class:`UHSField`
		"""
		branch_index = self.branch_index(branch_spec)
		try:
			branch_name = self.branch_names[branch_index]
		except:
			raise IndexError("Branch index %s out of range" % branch_index)
		else:
			branch_name = self.branch_names[branch_index]
			filespec = self.filespecs[branch_index]

			#intensities = self.intensities[:,branch_index,:]
			intensities = self.intensities[branch_index]

			return UHSField(self.sites, self.periods, intensities,
						self.intensity_unit, self.imt, model_name=branch_name,
						filespec=filespec, timespan=self.timespan,
						return_period=self.return_period, damping=self.damping,
						vs30s=vs30)

	#@property
	#def num_branches(self):
	#	return self.intensities.shape[1]

	@property
	def poe(self):
		return self.poes[0]

	@property
	def return_period(self):
		return self.return_periods[0]

	@property
	def exceedance_rate(self):
		return self.exceedance_rates[0]

	#def min(self):
		# TODO: not really useful
	#	return self.intensities.min(axis=0)

	#def max(self):
	#	return self.intensities.max(axis=0)

	@classmethod
	def from_branches(cls, uhsf_list, model_name, branch_names=None, weights=None,
					mean=None, percentile_levels=None, percentiles=None):
		"""
		Construct spectral hazard curve field tree from list of UHS
		fields for different logic-tree branches.

		:param uhsf_list:
			list with instances of :class:`UHSField`
		:param model_name:
			str, model name
		:param branch_names:
			list of branch names
			(default: None)
		:param weights:
			1-D list or array [j] with branch weights
			(default: None)
		:param mean:
			instance of :class:`UHSField`, representing mean uhsf
			(default: None)
		:param percentile_levels:
			list or array with percentile levels
			(default: None)
		:param percentiles:
			list with instances of :class:`UHSField`,
			representing uhsf's corresponding to percentiles
			(default: None)

		:return:
			instance of :class:`UHSFieldTree`
		"""
		num_branches = len(uhsf_list)
		uhsf0 = uhsf_list[0]
		num_sites = uhsf0.num_sites
		num_periods = uhsf0.num_periods

		intensities = np.zeros((num_branches, num_sites, num_periods))

		for j, uhsf in enumerate(uhsf_list):
			intensities[j] = uhsf.intensities

		filespecs = [uhsf.filespec for uhsf in uhsf_list]
		if branch_names in (None, []):
			branch_names = [uhsf.model_name for uhsf in uhsf_list]
		if weights in (None, []):
			weights = np.ones(num_branches, 'f') / num_branches

		uhsft = cls(branch_names, weights, uhsf0.sites, uhsf0.periods,
					intensities, uhsf0.intensity_unit, uhsf0.imt,
					model_name=model_name, filespecs=filespecs,
					timespan=uhsf0.timespan, return_period=uhsf0.return_period,
					damping=uhsf0.damping, vs30s=uhsf0.vs30s)

		## Set mean and percentiles
		if mean is not None:
			uhsft.set_mean(mean.intensities)

		if percentiles is not None:
			num_percentiles = len(percentiles)
			perc_array = np.zeros((num_sites, num_periods, num_percentiles), 'd')
			for p in range(num_percentiles):
				uhsf = percentiles[p]
				perc_array[:,:,p] = uhsf.intensities
			uhsft.set_percentiles(perc_array, percentile_levels)

		return uhsft

	def check_uhsf_compatibility(self, uhsf):
		"""
		Check the compatibility of a candidate branch.

		:param uhsf:
			instance of :class:`UHSField` or higher
		"""
		if self.sites != uhsf.sites:
			raise Exception("Sites do not correspond!")
		if (self.periods != uhsf.periods).any():
			raise Exception("Spectral periods do not correspond!")
		if self.imt != uhsf.imt:
			raise Exception("IMT does not correspond!")
		if self.intensity_unit != uhsf.intensity_unit:
			raise Exception("Intensity unit does not correspond!")
		if self.timespan != uhsf.timespan:
			raise Exception("Time span does not correspond!")
		if (self._hazard_values.__class__ != uhsf._hazard_values.__class__
			or (self._hazard_values != uhsf._hazard_values).any()):
			raise Exception("Hazard array does not correspond!")

	def extend(self, uhsft, renormalize_weights=False):
		"""
		Extend UHS field tree in-place with another one.

		:param uhsft:
			instance of :class:`UHSFieldTree`
		:param normalize_weights:
			bool, whether or not to renormalize weights
			(default: False)
		"""
		self.check_uhsf_compatibility(uhsft)
		self.branch_names.extend(uhsft.branch_names)
		if uhsft.filespecs:
			self.filespecs.extend(uhsft.filespecs)
		else:
			self.filespecs = []
		self.weights = np.concatenate([self.weights, uhsft.weights])
		#self.intensities = np.concatenate([self.intensities, uhsft.intensities], axis=1)
		self.intensities = np.concatenate([self.intensities, uhsft.intensities], axis=0)
		## Remove mean and percentiles
		self.mean = None
		self.percentiles = None
		if renormalize_weights:
			self.normalize_weights()

	def calc_mean(self, weighted=True):
		"""
		Compute mean UHS field.
		Note that it is more correct to compute the mean spectral hazard
		curve, and then interpolating at a particular return period!

		:param weighted:
			bool, whether or not to compute weighted mean
			(default: True)

		:return:
			2-D array [i, k] with intensity values for each site and
			spectral period
		"""
		if weighted:
			return np.average(self.intensities, weights=self.weights, axis=1)
		else:
			return np.mean(self.intensities, axis=1)

	#TODO: rename to calc_variance?
	def calc_variance_of_mean(self, weighted=True):
		"""
		Compute variance

		:param weighted:
			bool, whether or not to take into account branch weights
			(default: True)

		:return:
			2-D array [i,k]
		"""
		if weighted and not self.weights in ([], None):
			mean = self.calc_mean(weighted=True)
			weights = np.array(self.weights)
			weights_column = weights.reshape((self.num_branches, 1))
			variance_of_mean = np.zeros((self.num_sites, self.num_periods), 'd')
			for i in range(self.num_sites):
				for k in range(self.num_periods):
					#var = (self.intensities[i,:,k] - mean[i,k])**2
					var = (self.intensities[:,i,k] - mean[i,k])**2
					variance_of_mean[i,k] = np.sum(weights_column * var, axis=0)
		else:
			 #variance_of_mean = np.var(self.intensities, axis=1)
			 variance_of_mean = np.var(self.intensities, axis=0)
		return variance_of_mean

	def calc_percentiles(self, percentile_levels, weighted=True):
		"""
		Compute percentiles

		:param percentile_levels:
			list of ints [p], percentile levels
		:param weighted:
			see :meth:`calc_mean`

		:return:
			3-D array [i,k,p]
		"""
		if percentile_levels in ([], None):
			percentile_levels = [5, 16, 50, 84, 95]
		num_sites, num_periods = self.num_sites, self.num_periods
		num_percentiles = len(percentile_levels)
		percentiles = np.zeros((num_sites, num_periods, num_percentiles))
		for i in range(num_sites):
			for k in range(num_periods):
				if (weighted and self.weights is not None
					and len(np.unique(self.weights)) > 1):
					pmf = NumericPMF.from_values_and_weights(self.intensities[:,i,k],
															self.weights)
					percentiles[i,k,:] = pmf.get_percentiles(percentile_levels)
				else:
					for p, perc in enumerate(percentile_levels):
						percentiles[i,k,p] = scoreatpercentile(self.intensities[:,i,k],
																perc)
		return percentiles

	def get_mean_uhs_field(self, recalc=False, weighted=True):
		"""
		Get mean UHS field, recomputing it if necessary

		:param recalc:
			bool, whether or not to recompute the mean
			(default: False)
		:param weighted:
			bool, whether or not to compute weighted mean
			if :param:`recalc` is True
			(default: True)

		:return:
			instance of :class:`UHSField`
		"""
		if recalc or self.mean is None:
			intensities = self.calc_mean(weighted=weighted)
		else:
			intensities = self.mean

		model_name = "Mean(%s)" % self.model_name

		return UHSField(self.sites, self.periods, intensities,
						self.intensity_unit, self.imt, model_name=model_name,
						filespec="", timespan=self.timespan,
						return_period=self.return_period, damping=self.damping,
						vs30s=self.vs30)

	def get_percentile_uhs_field(self, perc, recalc=False, weighted=True):
		"""
		Get percentile UHS field, recomputing it if necessary

		:param perc:
			int, percentile level
		:param recalc:
		:param weighted:
			see :meth:`get_mean_uhs_field`

		:return:
			instance of :class:`UHSField`
		"""
		if recalc or self.percentiles is None or not perc in self.percentiles:
			intensities = self.calc_percentiles([perc], weighted=weighted)[:,:,0]
		else:
			perc_index = self.percentiles.index(perc)
			intensities = self.percentiles[:,:,perc_index]

		model_name = "Perc%02d(%s)" % (perc, self.model_name)

		return UHSField(self.sites, self.periods, intensities,
						self.intensity_unit, self.imt, model_name=model_name,
						filespec="", timespan=self.timespan,
						return_period=self.return_period, damping=self.damping,
						vs30s=self.vs30)

	def export_stats_csv(self, csv_filespec=None, site_index=0, weighted=True):
		"""
		Export mean and percentiles to CSV file

		:param csv_filespec:
			str, full path to output CSV file,
			(default: None, will print on screen)
		:param site_index:
			int, index of site for which to output results
			(default: 0)
		:param weighted:
			see :meth:`calc_mean`
		"""
		if is_empty_array(self.mean):
			mean = self.calc_mean(weighted=weighted)
		else:
			mean = self.mean
		if is_empty_array(self.percentiles):
			if is_empty_array(self.percentile_levels):
				percentile_levels = [5, 16, 50, 84, 95]
			else:
				percentile_levels = self.percentile_levels
			percentiles = self.calc_percentiles(percentile_levels, weighted=weighted)
		else:
			percentiles = self.percentiles
			percentile_levels = self.percentile_levels

		if csv_filespec:
			f = open(csv_filespec, "w")
		else:
			f = sys.stdout
		f.write("Period (s), Mean")
		f.write(", ".join(["P%02d" % p for p in percentile_levels]))
		f.write("\n")
		i = site_index
		for k, period in enumerate(self.periods):
			f.write("%.3E, %.3E" % (period, mean[i,k]))
			f.write(", ".join(["%.3E" % percentiles[i,k,p] for p in percentile_levels]))
			f.write("\n")
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
			instance of :class:`UHSFieldTree`
		"""
		branch_names, filespecs = [], []
		for index in branch_indexes:
			branch_names.append(self.branch_names[index])
			filespecs.append(self.filespecs[index])
		weights = self.weights[branch_indexes]
		## Recompute branch weights
		if normalize_weights:
			weight_sum = np.add.reduce(weights)
			weights /= weight_sum

		#intensities = self.intensities[:,branch_indexes,:]
		intensities = self.intensities[branch_indexes]

		return self.__class__(branch_names, weights, self.sites, self.periods,
							intensities, self.intensity_unit, self.imt,
							model_name=slice_name, filespecs=filespecs,
							timespan=self.timespan, return_period=self.return_period,
							damping=self.damping)

	def to_uhs_collection(self, site_spec=0, labels=[], colors=[], linestyles=[],
						linewidths=[], add_mean=False, add_percentiles=False,
						emphasize_mean=True, lang="en"):
		"""
		Extract UHS collection for a single site

		:param site_spec:
			site specification:
			- int: site index
			- str: site name
			- instance of :class:`rshalib.site.GenericSite`: site
			- (lon, lat) tuple

		:param labels:
		:param colors:
		:param linestyles:
		:param linewidths:
			see :class:`UHSCollection`
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
			instance of :class:`UHSCollection`
		"""
		uhs_list = []
		for i in range(self.num_branches):
			uhs_field = self.get_uhs_field(i)
			uhs = uhs_field.get_uhs(site_spec)
			uhs_list.append(uhs)
		if add_mean:
			mean_uhs_field = self.get_mean_uhs_field()
			mean_uhs = mean_uhs_field.get_uhs(site_spec)
			uhs_list.append(mean_uhs)
			if emphasize_mean:
				uhs_list.append(mean_uhs)
		if add_percentiles:
			if self.percentile_levels is None:
				percentile_levels = [5, 16, 50, 84, 95]
			else:
				percentile_levels = self.percentile_levels
			for perc in percentile_levels:
				perc_uhs_field = self.get_percentile_uhs_field(perc)
				perc_uhs = perc_uhs_field.get_uhs(site_spec)
				uhs_list.append(perc_uhs)

		if labels in (None, []):
			if len(uhs_list) <= 6:
				labels = self.branch_names
			else:
				labels = ["_nolegend_"] * self.num_branches
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

		if colors in (None, []):
			if add_mean and add_percentiles:
				colors = [(0.5, 0.5, 0.5)] * self.num_branches
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
							#perc_labels[perc] = "P%02d" % perc
							perc_colors[perc] = ["b", "g", "r", "c", "m", "k"][p%6]
							p += 1
						else:
							#perc_labels[100 - perc] += ", P%02d" % perc
							#perc_labels[perc] = "_nolegend_"
							perc_colors[perc] = perc_colors[100 - perc]
				colors.extend(perc_colors.values())

		if linewidths in (None, []):
			if add_mean and add_percentiles:
				linewidths = [1] * self.num_branches
				if add_mean:
					if emphasize_mean:
						linewidths.append(5)
					linewidths.append(3)
				if add_percentiles:
					linewidths.extend([2] * len(percentile_levels))

		if linestyles in (None, []):
			linestyles = ['-'] * self.num_branches
			if add_mean:
				if emphasize_mean:
					linestyles.append('-')
				linestyles.append('-')
			if add_percentiles:
				linestyles.extend(['--'] * len(percentile_levels))

		return UHSCollection(uhs_list, labels=labels, colors=colors,
							linestyles=linestyles, linewidths=linewidths)

	def plot(self, site_spec=0, add_mean=True, add_percentiles=True,
			emphasize_mean=True, lang="en", **kwargs):
		"""
		Convert to UHS collection and plot

		:param site_spec:
		:param add_mean:
		:param add_percentiles:
		:param emphasize_mean:
		:param lang:
			see :meth:`to_uhs_collection`
		:kwargs:
			keyword arguments understood by
			:func:`rshalib.result.plot.plot_hazard_spectra`

		:return:
			matplotlib Axes instance
		"""
		site_index = self.site_index(site_spec)

		labels = kwargs.pop('labels', [])
		colors = kwargs.pop('colors', [])
		linestyles = kwargs.pop('linestyles', [])
		linewidths = kwargs.pop('linewidths', [])

		title = kwargs.pop('title')
		if title is None:
			title = "UHS Tree"
			title += "\nSite: %s" % self.site_names[site_index]

		uhs_collection = self.to_uhs_collection(site_index, labels=labels,
										colors=colors, linestyles=linestyles,
										linewdiths=linewidths, add_mean=add_mean,
										add_percentiles=add_percentiles,
										emphasize_mean=emphasize_mean,
										lang=lang)

		return uhs_collection.plot(title=title, **kwargs)

	# TODO
#	def plot(self, site_spec=0, branch_specs=[], colors=[], linestyles=[],
#			linewidths=[], fig_filespec=None, title=None, plot_freq=False,
#			plot_style="loglin", Tmin=None, Tmax=None, amin=None, amax=None,
#			pgm_period=0.02, axis_label_size='x-large', tick_label_size='large',
#			legend_label_size='large', legend_location=0, lang="en"):
#		site_index = self.site_index(site_spec)
#		if branch_specs in ([], None):
#			branch_indexes = range(self.num_branches)
#		else:
#			branch_indexes = [self.branch_index(branch_spec) for branch_spec in branch_specs]
#		x = self.periods
#		datasets, pgm, labels, colors, linewidths, linestyles = [], [], [], [], [], []

#		if title is None:
#			title = "UHS Tree"
#			title += "\nSite: %s" % self.site_names[site_index]

#		## Plot individual models
#		for branch_index in branch_indexes:
#			y = self.intensities[site_index, branch_index]
#			datasets.append((x[self.periods>0], y[self.periods>0]))
#			if 0 in self.periods:
#				pgm.append(y[self.periods==0])
#			labels.append("_nolegend_")
#			colors.append((0.5, 0.5, 0.5))
#			linewidths.append(1)
#			linestyles.append('-')

#		## Plot overall mean
#		if self.mean is None:
#			y = self.calc_mean()[site_index]
#		else:
#			y = self.mean[site_index]
#		datasets.append((x[self.periods>0], y[self.periods>0]))
#		if 0 in self.periods:
#			pgm.append(y[self.periods==0])
#		labels.append("_nolegend_")
#		colors.append('w')
#		linewidths.append(5)
#		linestyles.append('-')
#		datasets.append((x[self.periods>0], y[self.periods>0]))
#		if 0 in self.periods:
#			pgm.append(y[self.periods==0])
#		labels.append({"en": "Overall Mean", "nl": "Algemeen gemiddelde"}[lang])
#		colors.append('r')
#		linewidths.append(3)
#		linestyles.append('-')

#		## Plot percentiles
#		if self.percentiles is None:
#			if self.percentile_levels is None:
#				percentile_levels = [5, 16, 50, 84, 95]
#			else:
#				percentile_levels = self.percentile_levels
#			percentiles = self.calc_percentiles(percentile_levels)
#		else:
#			percentiles = self.percentiles
#			percentile_levels = self.percentile_levels
#		percentiles = percentiles[site_index]
#		## Manage percentile labels and colors
#		perc_labels, perc_colors = {}, {}
#		p = 0
#		for perc in percentile_levels:
#			if not perc in perc_labels:
#				if not (100 - perc) in perc_labels:
#					perc_labels[perc] = "P%02d" % perc
#					perc_colors[perc] = ["b", "g", "r", "c", "m", "k"][p%6]
#					p += 1
#				else:
#					perc_labels[100 - perc] += ", P%02d" % perc
#					perc_labels[perc] = "_nolegend_"
#					perc_colors[perc] = perc_colors[100 - perc]
#		for p, perc in enumerate(percentile_levels):
#			label = perc_labels[perc]
#			labels.append(perc_labels[perc])
#			colors.append(perc_colors[perc])
#			linewidths.append(2)
#			linestyles.append('--')
#			y = percentiles[:,p]
#			datasets.append((x[self.periods>0], y[self.periods>0]))
#			if 0 in self.periods:
#				pgm.append(y[self.periods==0])
#		intensity_unit = self.intensity_unit
#		plot_hazard_spectrum(datasets, pgm=pgm, pgm_period=pgm_period, labels=labels,
#					colors=colors, linestyles=linestyles, linewidths=linewidths,
#					fig_filespec=fig_filespec, title=title, plot_freq=plot_freq,
#					plot_style=plot_style, Tmin=Tmin, Tmax=Tmax, amin=amin, amax=amax,
#					intensity_unit=intensity_unit, axis_label_size=axis_label_size,
#					tick_label_size=tick_label_size, legend_label_size=legend_label_size,
#					legend_location=legend_location, lang=lang)

	# TODO:
	def plot_histogram(self, site_index=0, period_index=0, fig_filespec=None, title=None, bar_color='g', amax=0, da=0.005, lang="en"):
		if title is None:
			title = "Site: %s / Period: %s s\n" % (self.sites[site_index], self.periods[period_index])
			title += "Return period: %.3G yr" % self.return_period
		intensity_unit = self.intensity_unit
		plot_histogram(self.intensities[site_index,:,period_index], weights=self.weights, fig_filespec=fig_filespec, title=title, bar_color=bar_color, amax=amax, da=da, intensity_unit=intensity_unit, lang=lang)


class UHSCollection:
	"""
	Container for an arbitrary set of Uniform Hazard Spectra,
	mainly used for plotting

	:param uhs_list:
		list with instances of :class:`UHS`
	:param labels:
		list of strings, labels for each UHS
		(default: [])
	:param colors:
		list with matplotlib color specifications for each UHS
		(default: [])
	:param linestyles:
		list with matplotlib line styles for each UHS
		(default: [])
	:param linewidths:
		list with line widths for each UHS
		(default: [])
	:param validate:
		bool, whether or not to check if all UHS have same unit and IMT
		(default: True)
	"""
	def __init__(self, uhs_list, labels=[],
				colors=[], linestyles=[], linewidths=[],
				validate=True):
		self.uhs_list = uhs_list
		self.colors = colors
		self.linestyles = linestyles
		self.linewidths = linewidths
		if not labels:
			labels = [uhs.model_name for uhs in self.uhs_list]
		self.labels = labels
		if validate:
			self.validate()

	def __repr__(self):
		return '<USSCollection (n=%d)>' % len(self)

	def validate(self):
		"""
		Check if all UHS have same intensity unit and IMT
		"""
		imts = set([uhs.imt for uhs in self.uhs_list])
		if len(imts) > 1:
			raise Exception("UHS have different IMT!")
		intensity_units = set([uhs.intensity_unit for uhs in self.uhs_list])
		if len(intensity_units) > 1:
			raise Exception("UHS have different intensity unit!")

	def __len__(self):
		return len(self.uhs_list)

	@property
	def intensity_unit(self):
		return self.uhs_list[0].intensity_unit

	@classmethod
	def from_csv_file(self, csv_filespec, site, intensity_unit="g", model_name="",
					timespan=50, poe=None, return_period=None, damping=0.05):
		"""
		Read UHSCollection from a csv file.
		First line should contain column names
		First column should contain periods or frequencies,
		subsequent columns should contain intensities
		Each intensity column will represent a UHS in the collection

		:param csv_filespec:
			str, full path to csv file
		:param site:
			GenericSite object, representing site for which UHS was computed
		:param intensity_unit:
			str, unit of intensities in csv file
			(default: "g")
		:param model_name:
			str, name or description of model
			(default: "")
		:param timespan:
			float, time span for UHS
			(default: 50)
		:param poe:
			float, probability of exceedance for UHS
			(default: None)
		:param return_period:
			float, return period for UHS
			(default: None)
		:param damping:
			float, damping corresponding to UHS collection
			(expressed as fraction of critical damping)
			(default: 0.05)

		:return:
			instance of :class:`UHSCollection`

		Note: either poe or return_period should be specified
		"""
		assert (poe and timespan) or return_period

		uhs_list = []
		periods, intensities = [], []
		csv = open(csv_filespec)
		for i, line in enumerate(csv):
			if i == 0:
				col_names = line.split(',')
				if col_names[0].lower() == "frequency":
					freqs = True
				else:
					freqs = False
			else:
				col_values = list(map(float, line.split(',')))
				T = col_values[0]
				a = col_values[1:]
				periods.append(T)
				intensities.append(a)
		csv.close()
		periods = np.array(periods)
		if freqs:
			periods = 1./periods

		if intensity_unit in ("g", "mg", "m/s2", "cm/s2", "gal"):
			imt = "SA"
		elif intensity_unit in ("m/s", "cm/s"):
			imt = "SV"
		elif intensity_unit in ("m", "cm"):
			imt = "SD"
		else:
			imt = ""

		intensities = np.array(intensities).transpose()
		for i, uhs_intensities in enumerate(intensities):
			model_name = col_names[i+1]
			uhs = UHS(periods, uhs_intensities, intensity_unit, imt,
					site, model_name=model_name, filespec=csv_filespec,
					timespan=timespan, return_period=return_period,
					damping=damping)
			uhs_list.append(uhs)

		return UHSCollection(uhs_list)

	def plot(self, intensity_unit="g",
			pgm_period=0.01, pgm_marker='o', plot_freq=False,
			xscaling='log', yscaling='lin', xgrid=1, ygrid=1,
			title=None, fig_filespec=None, lang="en", **kwargs):
		"""
		Plot UHS collection

		See :func:`rshalib.result.plot.plot_hazard_spectra`

		:return:
			matplotlib Axes instance
		"""
		if title is None:
			title = "UHS Collection"

		#pgm, datasets = [], []
		#for uhs in self.uhs_list:
		#	intensities = uhs.get_intensities(intensity_unit)
		#	if 0 in uhs.periods:
		#		pgm.append(intensities[uhs.periods == 0])
		#	else:
		#		pgm.append(np.nan)
		#	datasets.append((uhs.periods[uhs.periods>0], intensities[uhs.periods>0]))

		return plot_hazard_spectra(self.uhs_list,
							labels=self.labels, colors=self.colors,
							linestyles=self.linestyles, linewidths=self.linewidths,
							intensity_unit=intensity_unit,
							pgm_period=pgm_period, pgm_marker=pgm_marker,
							plot_freq=plot_freq, xgrid=xgrid, ygrid=ygrid,
							xscaling=xscaling, yscaling=yscaling,
							title=title, fig_filespec=fig_filespec, lang=lang,
							**kwargs)

		#plot_hazard_spectrum(datasets, pgm=pgm, pgm_period=pgm_period,
		#		labels=self.labels, colors=self.colors, linestyles=self.linestyles,
		#		linewidths=self.linewidths, fig_filespec=fig_filespec, title=title,
		#		plot_freq=plot_freq, plot_style=plot_style, Tmin=Tmin, Tmax=Tmax,
		#		amin=amin, amax=amax, intensity_unit=intensity_unit,
		#		axis_label_size=axis_label_size, tick_label_size=tick_label_size,
		#		legend_label_size=legend_label_size, legend_location=legend_location,
		#		lang=lang, dpi=dpi, ax=ax)



if __name__ == "__main__":
	pass
