# coding: utf-8
import numpy as np
from scipy.misc import factorial
import pylab


class PoissonTau:
	def __init__(self, tau):
		self.tau = float(tau)

	def get_prob_n(self, n, t):
		return (t / self.tau)**n * np.exp(-t / self.tau) / factorial(n)

	def get_prob_one_or_more(self, t):
		return 1 - np.exp(-t / self.tau)

	def get_pmf(self, t):
		#return self.get_prob_n(1, t) / t
		return (1. / self.tau) * np.exp(-t / self.tau)

	@property
	def lamda(self):
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

	def plot(self, tmax, t_test=None, n="one_or_more", t_over_tau=False):
		t = np.linspace(0, tmax, 51)
		if n == "one_or_more":
			prob = self.get_prob_one_or_more(t)
		else:
			n = int(n)
			prob = self.get_prob_n(n, t)
		label = "Fixed tau: %s yr" % self.tau
		if t_test:
			if n == "one_or_more":
				prob_test = self.get_prob_one_or_more(t_test)
			else:
				prob_test = self.get_prob_n(n, t_test)
		if t_over_tau:
			t_over_tau = t / self.tau
			pylab.plot(t_over_tau, prob)
			t_over_tau_test = t_test / self.tau
			if t_test:
				pylab.vlines(t_over_tau_test, 0, 1, linestyle='--')
				pylab.hlines(prob_test, t_over_tau[0], t_over_tau[-1], linestyle='--')
				print "t/tau=%.1f --> %.1f%% prob." % (t_over_tau_test, prob_test*100)
			pylab.xlabel(r"$t/\tau$", fontsize="x-large")
		else:
			pylab.plot(t, prob, label=label)
			if t_test:
				pylab.vlines(t_test, 0, 1, linestyle='--')
				pylab.hlines(prob_test, t[0], t[-1], linestyle='--')
				print "t=%.1f --> %.1f%% prob." % (t_test, prob_test*100)
			pylab.xlabel(r"Timespan (yr)")
		pylab.ylabel("Poisson probability")
		pylab.grid(True)
		pylab.legend(loc=2)
		pylab.show()


class PoissonT:
	def __init__(self, t):
		self.t = float(t)

	def get_prob_n(self, n, tau):
		return (self.t / tau)**n * np.exp(-self.t / tau) / factorial(n)

	def get_prob_one_or_more(self, tau):
		return 1 - np.exp(-self.t / tau)

	def get_tau(self, prob_one_or_more):
		return -self.t / np.log(1 - prob_one_or_more)

	def plot(self, tau_max, tau_test=None, n="one_or_more"):
		tau_min = 1
		tau = np.linspace(tau_min, tau_max, 1000)

		if n == "one_or_more":
			prob = self.get_prob_one_or_more(tau)
		else:
			n = int(n)
			prob = self.get_prob_n(n, tau)
		label = "Fixed timespan: %s yr" % self.t
		if tau_test:
			if n == "one_or_more":
				prob_test = self.get_prob_one_or_more(tau_test)
			else:
				prob_test = self.get_prob_n(n, tau_test)

		pylab.plot(tau, prob, label=label)
		pylab.vlines(tau_test, 0, 1, linestyle='--')
		pylab.hlines(prob_test, tau_min, tau_max, linestyle='--')
		pylab.xlabel("Return period (yr)")
		pylab.ylabel("Poisson probability")
		pylab.grid(True)
		pylab.legend()
		pylab.show()
		print "tau=%.1f yr --> %.1f%% prob. in %d yr" % (tau_test, prob_test*100, self.t)
