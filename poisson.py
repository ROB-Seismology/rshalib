# coding: utf-8
"""
Poisson model
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.misc import factorial
import pylab


__all__ = ['PoissonTau', 'PoissonT']


class PoissonTau:
	"""
	Poisson model with fixed mean return period

	:param tau:
		float, mean return period
	"""
	def __init__(self, tau):
		self.tau = float(tau)

	def get_prob_n(self, n, t):
		"""
		Compute probability of n events in time t

		:param n:
			int, number of events
		:param t:
			float or array, time interval(s)

		:return:
			float or array, probability
		"""
		return (t / self.tau)**n * np.exp(-t / self.tau) / factorial(n)

	def get_prob_one_or_more(self, t):
		"""
		Compute probability of one or more events in time t
		This is the exceedance probability.

		:param t:
			float or array, time interval(s)

		:return:
			float or array, probability
		"""
		return 1 - np.exp(-t / self.tau)

	def get_prob_iet(self, iet):
		"""
		Compute probability of given inter-event time
		This is the probability mass function of the exponential
		distribution with rate equal to 1 over the return period
		This is equivalent to get_prob_n(1, iet) / iet
		The CDF of this pmf corresponds to the exceedance probability
		distribution of :meth:`get_prob_one_or_more`

		:param iet:
			float or array, inter-event time(s)

		:return:
			float or array, probability
		"""
		#return self.get_prob_n(1, iet) / iet
		return (1. / self.tau) * np.exp(-iet / self.tau)

	@property
	def lamda(self):
		"""
		average occurrence rate
		"""
		return 1. / self.tau

	@classmethod
	def from_conditional_probability(cls, cond_prob, t):
		"""
		Create Poisson model with return period that is equivalent to
		the conditional (time-dependent) rupture probability for given
		timespan

		:param cond_prob:
			float or array of floats, conditional rupture probability(ies)
		:param t:
			float, time interval (number of years) over which probability
			is computed

		:return:
			instance of :class:`PoissonTau`
		"""
		effective_rate = -np.log(1 - cond_prob) / float(t)
		return cls(1./effective_rate)

	def plot(self, tmax, t_test=None, n="one_or_more", normalize_t=False,
			color=None, linestyle='-', linewidth=2, label=None,
			**kwargs):
		"""
		Plot Poisson probabilities for range of time intervals

		:param tmax:
			float, maximum time interval to plot
		:param t_test:
			float, 'testing' time interval for which dash lines
			will be drawn
			(default: None)
		:param n:
			int: number of events
			or str: "one_or_more"
			(default: "one_or_more")
		:param normalize_t:
			bool, whether or not to normalize t by the return period
			(default: False)
		:param color:
			matplotlib color specification
			(default: None)
		:param linestyle:
			str, matplotlib line style
			(default: "-")
		:param linewidth:
			float, line width
			(default: 2)
		:param label:
			str, plot label
			(default: "")
		:param kwargs:
			keyword-arguments understood by :func:`generic_mpl.plot_xy`

		:return:
			matplotlib Axes instance
		"""
		from plotting.generic_mpl import plot_xy

		kwargs['colors'] = [color] if color is not None else None
		kwargs['linestyles'] = [linestyle] if linestyle is not None else None
		kwargs['linewidths'] = [linewidth] if linewidth is not None else None
		label = "Fixed tau: %s yr" % self.tau
		kwargs['labels'] = [label] if label is not None else None

		t = np.linspace(0, tmax, 51)
		if n == "one_or_more":
			prob = self.get_prob_one_or_more(t)
		else:
			n = int(n)
			prob = self.get_prob_n(n, t)
		if t_test:
			if n == "one_or_more":
				prob_test = self.get_prob_one_or_more(t_test)
			else:
				prob_test = self.get_prob_n(n, t_test)
			kwargs['vline_args'] = kwargs.get('vline_args', {'linestyle': '--'})
			kwargs['hline_args'] = kwargs.get('hline_args', {'linestyle': '--'})

		if normalize_t:
			t_over_tau = t / self.tau
			datasets = [(t_over_tau, prob)]
			t_over_tau_test = t_test / self.tau
			if t_test:
				kwargs['vlines'] = [t_over_tau_test, 0, 1]
				kwargs['hlines'] = [prob_test, t_over_tau[0], t_over_tau[-1]]
				print("t/tau=%.1f --> %.1f%% prob." % (t_over_tau_test, prob_test*100))
			xlabel = kwargs.pop('xlabel', r"$t/\tau$")
		else:
			datasets = [(t, prob)]
			if t_test:
				kwargs['vlines'] = [t_test, None, None]
				kwargs['hlines'] = [prob_test, t[0], t[-1]]
				print("t=%.1f --> %.1f%% prob." % (t_test, prob_test*100))
			xlabel = kwargs.pop('xlabel', r"Timespan (yr)")
		ylabel = kwargs.pop('ylabel', "Poisson probability")
		xgrid = kwargs.pop('xgrid', 1)
		ygrid = kwargs.pop('ygrid', 1)
		legend_location = kwargs.pop('legend_location', 2)

		return plot_xy(datasets, xlabel=xlabel, ylabel=ylabel, xgrid=xgrid,
						ygrid=ygrid, legend_location=legend_location, **kwargs)


class PoissonT:
	"""
	Poisson model with fixed time interval

	:param t:
		float, time interval
	"""
	def __init__(self, t):
		self.t = float(t)

	def get_prob_n(self, n, tau):
		"""
		Compute probability of n events with given return period

		:param n:
			int, number of events
		:param tau:
			float or array, return period(s)

		:return:
			float or array, probability
		"""
		return (self.t / tau)**n * np.exp(-self.t / tau) / factorial(n)

	def get_prob_one_or_more(self, tau):
		"""
		Compute probability of one or more events with given return period

		:param tau:
			float or array, return period(s)

		:return:
			float or array, probability
		"""
		return 1 - np.exp(-self.t / tau)

	def get_tau(self, prob_one_or_more):
		"""
		Compute return period for one or more events corresponding to
		given probability

		:param prob_one_or_more:
			float or array, probability of one or more events in fixed
			time interval

		:return:
			float or array, return period(s)
		"""
		return -self.t / np.log(1 - prob_one_or_more)

	def plot(self, tau_max, tau_test=None, n="one_or_more",
			color=None, linestyle='-', linewidth=2, label=None,
			**kwargs):
		"""
		Plot Poisson probabilities for range of return periods

		:param tau_max:
			float, maximum return period to plot
		:param tau_test:
			float, 'testing' return period for which dash lines
			will be drawn
			(default: None)
		:param n:
			int: number of events
			or str: "one_or_more"
			(default: "one_or_more")
		:param color:
			matplotlib color specification
			(default: None)
		:param linestyle:
			str, matplotlib line style
			(default: "-")
		:param linewidth:
			float, line width
			(default: 2)
		:param label:
			str, plot label
			(default: "")
		:param kwargs:
			keyword-arguments understood by :func:`generic_mpl.plot_xy`

		:return:
			matplotlib Axes instance
		"""
		from plotting.generic_mpl import plot_xy

		kwargs['colors'] = [color] if color is not None else None
		kwargs['linestyles'] = [linestyle] if linestyle is not None else None
		kwargs['linewidths'] = [linewidth] if linewidth is not None else None
		label = "Fixed timespan: %s yr" % self.t
		kwargs['labels'] = [label] if label is not None else None

		tau_min = 1
		tau = np.linspace(tau_min, tau_max, 1000)

		if n == "one_or_more":
			prob = self.get_prob_one_or_more(tau)
		else:
			n = int(n)
			prob = self.get_prob_n(n, tau)

		if tau_test:
			if n == "one_or_more":
				prob_test = self.get_prob_one_or_more(tau_test)
			else:
				prob_test = self.get_prob_n(n, tau_test)
			kwargs['vline_args'] = kwargs.get('vline_args', {'linestyle': '--'})
			kwargs['hline_args'] = kwargs.get('hline_args', {'linestyle': '--'})
			kwargs['vlines'] = [tau_test, None, None]
			kwargs['hlines'] = [prob_test, tau_min, tau_max]
			msg = "tau=%.1f yr --> %.1f%% prob. in %d yr"
			msg %= (tau_test, prob_test*100, self.t)
			print(msg)

		datasets = [(tau, prob)]
		xlabel = kwargs.pop('xlabel', "Return period (yr)")
		ylabel = kwargs.pop('ylabel', "Poisson probability")
		xgrid = kwargs.pop('xgrid', 1)
		ygrid = kwargs.pop('ygrid', 1)
		legend_location = kwargs.pop('legend_location', 0)

		return plot_xy(datasets, xlabel=xlabel, ylabel=ylabel, xgrid=xgrid,
						ygrid=ygrid, legend_location=legend_location, **kwargs)
