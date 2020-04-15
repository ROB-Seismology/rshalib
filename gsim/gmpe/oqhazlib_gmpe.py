# -*- coding: iso-Latin-1 -*-
"""
Interface with GMPEs defined in OpenQuake
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.constants import g
import pylab

from ... import (oqhazlib, OQ_VERSION)
from openquake.hazardlib.imt import PGD, PGV, PGA, SA, MMI

from .base import *



__all__ = ['get_oq_gsim', 'make_gsim_contexts', 'OqhazlibGMPE',
			'AbrahamsonSilva2008', 'AkkarBommer2010',
			'AkkarEtAl2013', 'Anbazhagan2013', 'Atkinson2015',
			'AtkinsonBoore2006', 'AtkinsonBoore2006Prime',
			'BindiEtAl2011', 'BooreAtkinson2008', 'BooreAtkinson2008Prime',
			'Campbell2003', 'Campbell2003adjusted', 'Campbell2003HTT',
			'Campbell2003SHARE', 'CampbellBozorgnia2008',
			'CauzziFaccioli2008', 'ChiouYoungs2008',
			'FaccioliEtAl2010', 'FaccioliEtAl2010Ext', 'PezeshkEtAl2011',
			'RietbrockEtAl2013MD', 'RietbrockEtAl2013MDHTT',
			'RietbrockEtAl2013SS', 'ToroEtAl2002', 'ToroEtAl2002adjusted',
			'ToroEtAl2002HTT', 'ToroEtAl2002SHARE',
			'ZhaoEtAl2006Asc', 'ZhaoEtAl2006AscMOD']


IMT_DICT = {"PGD": PGD, "PGV": PGV, "PGA": PGA, "SA": SA, "MMI": MMI}


def get_oq_gsim(gsim_name):
	"""
	Get OpenQuake ground-shaking intensity model

	:param gsim_name:
		str, name of gsim

	:return:
		instance of :class:`oqhazlib.gsim.GMPE`
		or :class:`oqhazlib.gsim.IPE`
	"""
	gsim = oqhazlib.gsim.get_available_gsims().get(gsim_name)
	if gsim:
		return gsim()
	else:
		return None


def make_gsim_contexts(gsim, sites, rupture, max_distance=None):
	"""
	Wrapper function to make contexts for a gsim,
	which keeps changing all the time in OpenQuake...

	:param gsim:
		instance of :class:`oqhazlib.gsim.GroundShakingIntensityModel`
	:param sites:
		instance of :class:`oqhazlib.site.SiteCollection`
	:param rupture:
		instance of :class:`oqhazlib.source.[Base]Rupture`
	:param max_distance:
		float, maximum distance
		Only taken into account in more recent versions of OpenQuake
		(default: None)

	:return:
		(sctx, rctx, dctx) tuple
		- sctx: site context
		- rctx: rupture context (or rupture)
		- dctx distance context
	"""
	if OQ_VERSION >= '2.9.0':
		from openquake.hazardlib.calc.filters import IntegrationDistance

		maximum_distance = None
		if max_distance:
			trt = 'default'
			maximum_distance = {trt: [(rupture.mag, max_distance)]}
			maximum_distance = IntegrationDistance(maximum_distance)

		ctx_maker = oqhazlib.gsim.base.ContextMaker([gsim],
												maximum_distance=maximum_distance)

		if OQ_VERSION >= '3.2.0':
			sctx, dctx = ctx_maker.make_contexts(sites, rupture)
			rctx = rupture
		else:
			sctx, rctx, dctx = ctx_maker.make_contexts(sites, rupture)
	else:
		sctx, rctx, dctx = gsim.make_contexts(sites, rupture)

	return (sctx, rctx, dctx)


class OqhazlibGMPE(GMPE):
	"""
	Class to implement GMPEs from OpenQuake in rshalib.

	:param name:
		str, defines name of GMPE. The names of all available GMPEs in
		Nhlib can be retrieved by oqhazlib.gsim.get_available_gsims().keys().
	:param imt_periods:
		dict, String: List of Floats, mapping name of imt (PGD, PGV, PGA or
		SA) to periods (Default: None). Must only be provided if spectral
		periods cannot be retrieved from COEFFS attribute of GMPE.

	See :class:`GMPE` for other params.
	"""
	def __init__(self, name, short_name, distance_metric, Mmin, Mmax, dmin, dmax,
				Mtype, dampings=[5], imt_periods=None):
		"""
		"""
		## get dict mapping gmpe names to classes
		self.gsim = get_oq_gsim(name)

		## get imt periods
		if not imt_periods:
			imt_periods = self._get_imt_periods_from_oqhazlib_coeffs(name)

		if name in ("ToroEtAl2002", "ToroEtAl2002SHARE", "ToroEtAl2002HTT"):
			## Remove T=3 and T=4 seconds, which are not supported by the
			## original publication
			imt_periods['SA'] = imt_periods['SA'][:-2]

		GMPE.__init__(self, imt_periods, distance_metric, Mmin, Mmax, dmin, dmax,
					Mtype, dampings, name, short_name)

		## Unit conversion
		self.imt_scaling = {}
		self.imt_scaling["PGA"] = {"g": 1.0, "mg": 1E+3, "ms2": g,
									"gal": g*100, "cms2": g*100}
		self.imt_scaling["SA"] = self.imt_scaling["PGA"]
		self.imt_scaling["PGV"] = {"ms": 1E-2, "cms": 1.0}
		self.imt_scaling["PGD"] = {"m": 1E-2, "cm": 1.0}

	def __repr__(self):
		return '<OqhazlibGMPE %s>' % self.name

	def _get_imt_periods_from_oqhazlib_coeffs(self, name=""):
		imt_periods = {}
		for imt in self.gsim.DEFINED_FOR_INTENSITY_MEASURE_TYPES:
			if imt == SA:
				if name in ("AtkinsonBoore2006", "AtkinsonBoore2006Prime"):
					imt_periods['SA'] = [sa.period for sa in
										sorted(self.gsim.COEFFS_BC.sa_coeffs.keys())]
				elif "adjusted" in name:
					(vs30, kappa) = list(self.gsim.COEFFS.keys())[0]
					imt_periods['SA'] = [sa.period for sa in
						sorted(self.gsim.COEFFS[(vs30, kappa)].sa_coeffs.keys())]
				else:
					imt_periods['SA'] = [sa.period for sa in
										sorted(self.gsim.COEFFS.sa_coeffs.keys())]
			else:
				imt_periods[imt.__name__] = [0]
		return imt_periods

	def _get_oqhazlib_mean_and_stddevs(self, M, d, h=0., imt="PGA", T=0, vs30=None,
										vs30measured=None, z1pt0=None, z2pt5=None,
										kappa=None, mechanism="normal", damping=5):
		## Convert arguments to correct shape
		if isinstance(M, (int, float)):
			magnitude_is_array = False
			Mags = [M]
		else:
			magnitude_is_array = True
			Mags = M

		if isinstance(d, (int, float)):
			distance_is_array = False
			d = np.array([d], dtype='d')
		else:
			if isinstance(d, list):
				d = np.array(d, dtype='d')
			else:
				d = d.astype('d')
			distance_is_array = True

		ln_means, ln_stddevs = [], []
		for M in Mags:
			## get sctx, rctx, dctx and imt
			sctx, rctx, dctx, imt_object = self._get_contexts_and_imt(M, d, h,
												vs30, vs30measured, z1pt0, z2pt5,
												kappa, mechanism, imt, T, damping)

			## get mean and sigma
			ln_mean, [ln_sigma] = self.gsim.get_mean_and_stddevs(sctx, rctx,
													dctx, imt_object, ['Total'])

			#if not distance_is_array:
			#	ln_mean = ln_mean[0]
			#	ln_sigma = ln_sigma[0]

			ln_means.append(ln_mean)
			ln_stddevs.append(ln_sigma)

		if not magnitude_is_array:
			ln_means = ln_means[0]
			ln_stddevs = ln_stddevs[0]
		else:
			ln_means = np.array(ln_means)
			ln_stddevs = np.array(ln_stddevs)

		return (ln_means, ln_stddevs)

	def _get_contexts_and_imt(self, M, d, h, vs30, vs30measured, z1pt0, z2pt5,
							kappa, mechanism, imt, T, damping):
		"""
		Return site, rupture and distance context, and imt objects.

		See :meth:`__call__` for params.
		"""

		from openquake.hazardlib.scalerel import PointMSR
		from hazard.rshalib.source.rupture import Rupture

		## Create rupture
		lon, lat = 0., 0.
		depth = max(0.1, h)
		mag = M
		strike, dip = 0., 45.
		rake = {'normal': -90., 'reverse': 90., 'strike-slip': 0.}[mechanism]
		trt = ""
		rms, rar = 1., 1.
		usd, lsd = 0., depth*2
		msr = PointMSR()

		# TODO: include slip_direction??
		rupture = Rupture.from_hypocenter(lon, lat, depth, mag, strike, dip, rake,
											trt, rms, rar, usd, lsd, msr)

		## Create site collection
		from ...site import GenericSiteModel

		azimuth = 90
		if self.distance_metric in ("Rupture", "rrup", "Hypocentral", "rhypo"):
			## Convert hypocentral/rupture distance to epicentral/joyner-boore distance
			d = np.sqrt(np.asarray(d)**2 - depth**2)
			d[np.isnan(d)] = 0
		else:
			d = np.asarray(d)
		lons, lats = oqhazlib.geo.geodetic.point_at(lon, lat, azimuth, d)
		depths = np.zeros_like(lons)
		sha_site_model = GenericSiteModel(lons=lons, lats=lats, depths=depths)

		if not vs30:
			vs30 = 800.
		if not z1pt0:
			z1pt0 = 1.
		if not z2pt5:
			z2pt5 = 1.
		ref_soil_params = {"vs30": vs30, "vs30measured": vs30measured,
							"z1pt0": z1pt0, "z2pt5": z2pt5, "kappa": kappa}
		soil_site_model = sha_site_model.to_soil_site_model(ref_soil_params)

		## Create contexts
		#if OQ_VERSION >= '2.9.0':
		#	ctx_maker = oqhazlib.gsim.base.ContextMaker([self.gsim])
		#	id_dict = {'default': self.dmax}
		#	integration_distance = oqhazlib.calc.filters.IntegrationDistance(id_dict)
		#	sctx, rctx, dctx = ctx_maker.make_contexts(soil_site_model, rupture,
		#												integration_distance)
		#else:
		#	sctx, rctx, dctx = self.gsim.make_contexts(soil_site_model, rupture)

		sctx, rctx, dctx = make_contexts(gsim, max_distance=self.dmax)

		## Set imt
		if imt == "SA":
			imt_object = IMT_DICT[imt](T, damping)
		else:
			imt_object = IMT_DICT[imt]()

		return sctx, rctx, dctx, imt_object

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type=None, vs30=None, vs30measured=None, z1pt0=None,
				z2pt5=None, kappa=None, mechanism="normal", damping=5):
		"""
		See class GMPE for params.

		Note: soil_type not supported. Should be implemented for each subclass.
		"""
		if soil_type:
			print("Warning: soil_type parameter is ignored!")
		ln_means, ln_stddevs = self._get_oqhazlib_mean_and_stddevs(M, d, h=h,
								imt=imt, T=T, vs30=vs30, vs30measured=vs30measured,
								z1pt0=z1pt0, z2pt5=z2pt5, kappa=kappa,
								mechanism=mechanism, damping=damping)

		## number of standard deviations
		if epsilon:
			unc = ln_stddevs * epsilon
		else:
			unc = 0.

		## convert to accelerations
		imls = np.exp(ln_means + unc)

		## apply scale factor
		scale_factor = self.imt_scaling[imt.upper()][imt_unit.lower()]
		imls *= scale_factor

		return imls

	def log_sigma(self, M, d, h=0., imt="PGA", T=0, soil_type="rock", vs30=None,
					vs30measured=None, z1pt0=None, z2pt5=None, kappa=None,
					mechanism="normal", damping=5):
		"""
		Return sigma as log10.

		See method self.__call__ for params.

		Note: :param:`soil_type` is ignored!
		"""
		#TODO: check whether we should support soil_type !
		_, ln_stddevs = self._get_oqhazlib_mean_and_stddevs(M, d, h=h, imt=imt,
									T=T, vs30=vs30, vs30measured=vs30measured,
									z1pt0=z1pt0, z2pt5=z2pt5, kappa=kappa,
									mechanism=mechanism, damping=damping)

		#log_sigmas = np.log10(np.exp(ln_stddevs))
		log_sigmas = ln_stddevs * np.log10(np.e)

		return log_sigmas

	def is_rake_dependent(self):
		"""
		Indicate whether or not GMPE depends on rake of the source
		"""
		M, d = 6.0, 15.0
		# TODO: may fail if vs30, kappa values are not supported
		val1 = self.__call__(M, d, vs30=800., kappa=0.03, mechanism="normal")
		val2 = self.__call__(M, d, vs30=800., kappa=0.03, mechanism="reverse")
		if np.allclose(val1, val2):
			return False
		else:
			return True


class AbrahamsonSilva2008(OqhazlibGMPE):
	"""
	"""
	def __init__(self):
		name, short_name = "AbrahamsonSilva2008", "AS_2008"
		distance_metric = "Rupture"
		Mmin, Mmax = 5.0, 8.5
		dmin, dmax = 0., 200
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="rock", vs30=None, kappa=None, mechanism="normal",
				damping=5):
		if vs30 is None:
			if soil_type == ("rock"):
				vs30 = 800
			else:
				raise SoilTypeNotSupportedError(soil_type)
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									mechanism=mechanism, damping=damping)

	def plot_figure1_Abrahamson_2008(self, imt="PGA", T=0):
		"""
		Plot Figure 1 in the paper of Abrahamson et al. (2008)
		Note that the figure does not match, because they use RJB distance,
		and we ignore hanging-wall effects (and possibly other reasons...)

		:param imt:
			str, intensity measure type, either "PGA" or "SA"
			(default: "PGA")
		:param T:
			float, spectral period, either 0. or 1.
			(default: 1.)
		"""
		self.plot_distance(mags=[5., 6., 7., 8.], plot_style="loglog",
							ymin=0.001, ymax=1, dmin=1, dmax=200,
							mechanism="strike-slip", vs30=760.)

	def plot_figure7_Abrahamson_2008(self):
		"""
		Plot Figure 7 in the paper of Abrahamson et al. (2008)
		Note that the figure does not match, because they use RJB distance,
		and we ignore hanging-wall effects (and possibly other reasons...)
		"""
		self.plot_spectrum(mags=[5., 6., 7., 8.], d=10., Tmin=0.01, Tmax=10,
						amin=1E-3, amax=1, mechanism="strike-slip", vs30=760.)


class AkkarBommer2010(OqhazlibGMPE):
	"""
	"""
	def __init__(self):
		name, short_name = "AkkarBommer2010", "AB_2010"
		distance_metric = "Joyner-Boore"
		Mmin, Mmax = 5.0, 7.6
		dmin, dmax = 0., 100.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="rock", vs30=None, kappa=None, mechanism="normal",
				damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type == "rock":
				vs30 = 800
			elif soil_type == "stiff":
				vs30 = 550
			elif soil_type == "soft":
				vs30 = 300
			else:
				raise SoilTypeNotSupportedError(soil_type)
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									mechanism=mechanism, damping=damping)


class AkkarEtAl2013(OqhazlibGMPE):
	"""
	"""
	def __init__(self):
		name, short_name = "AkkarEtAl2013", "A_2013"
		distance_metric = "Joyner-Boore"
		Mmin, Mmax = 4., 7.6
		dmin, dmax = 0., 200.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="rock", vs30=None, kappa=None, mechanism="normal",
				damping=5):
		if vs30 is None:
			if soil_type == "rock":
				vs30 = 750.
			else:
				raise SoilTypeNotSupportedError(soil_type)
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									mechanism=mechanism, damping=damping)

	def plot_figure_10(self, period):
		"""
		Reproduce Figure 10 of Akkart et al. (2013).

		:param period:
			float, period (options: 0 (=PGA), 0.2, 1. or 4.)
		"""
		self.plot_distance(mags=[6.0, 6.5, 7.0, 7.5, 8.0], dmin=1., dmax=200.,
							distance_metric="Joyner-Boore", h=0,
							imt={0: "PGA", 0.2: "SA", 1.: "SA", 4.: "SA"}[period],
							T=period, imt_unit="g", mechanism="strike-slip",
							vs30=750., damping=5, xscaling="log", yscaling="log",
							ymin={0: 0.001, 0.2: 0.001, 1.: 0.0001, 4.: 0.0001}[period],
							ymax=2., xgrid=2, ygrid=2)


class AtkinsonBoore2006(OqhazlibGMPE):
	"""
	"""
	def __init__(self):
		name, short_name = "AtkinsonBoore2006", "AB_2006"
		distance_metric = "Rupture"
		Mmin, Mmax = 3.5, 8.0
		dmin, dmax = 1., 1000.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="rock", vs30=None, kappa=None, mechanism="normal",
				damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type == "hard rock":
				vs30 = 2000.
			elif soil_type == "rock":
				vs30 = 760.
			else:
				raise SoilTypeNotSupportedError(soil_type)
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									mechanism=mechanism, damping=damping)

	def plot_figure4(self, imt="PGA", T=0):
		"""
		Reproduce Figure 4 in the original publication of Atkinson & Boore (2006)

		Not sure why, but I can't reproduce these plots...

		:param imt:
			str, intensity measure type, either "PGA" or "SA"
		:param T:
			float, period, either 0.2, 1 or 2 s
		"""
		if imt == "SA" and T in (1, 2):
			amin, amax = 1E-2, 1E+3
		else:
			amin, amax = 1E-1, 1E+4
		self.plot_distance(mags=[5., 6., 7., 8.], imt=imt, T=T, xscaling="log",
						yscaling="log", 	vs30=1999., imt_unit="cms2",
						ymin=amin, ymax=amax)

	def plot_figure_Boore_notes(self, T=0.2):
		"""
		Reproduce figures in the notes by Boore
		http://www.daveboore.com/pubs_online/ab06_gmpes_programs_and_tables.pdf

		:param T:
			float, period, either, 0.2, 1 or 5 s
		"""
		VS30 = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 1999, 2000, 2500]
		M5values, M6values, M7values = [], [], []
		for vs30 in VS30:
			values = self.__call__([5.,6.,7.], 15., vs30=vs30, imt="SA", T=T,
									imt_unit="cms2")
			M5values.append(values[0])
			M6values.append(values[1])
			M7values.append(values[2])
		pylab.loglog(VS30, M5values, 'b', label="M=5")
		pylab.loglog(VS30, M6values, 'g', label="M=6")
		pylab.loglog(VS30, M7values, 'r', label="M=7")
		pylab.grid(True)
		pylab.grid(True, which="minor")
		pylab.legend()
		pylab.title("T = %.1f Hz" % (1./T))
		if T == 0.2:
			amin, amax = 10, 5000
		elif T == 1:
			amin, amax = 1, 500
		elif T == 5:
			amin, amax = 0.1, 50
		pylab.axis((150, 2500, amin, amax))
		pylab.xlabel("V30")
		pylab.ylabel("y (cm/s2)")
		pylab.show()


class AtkinsonBoore2006Prime(OqhazlibGMPE):
	def __init__(self):
		name, short_name = "AtkinsonBoore2006Prime", "AB_2006'"
		distance_metric = "Rupture"
		Mmin, Mmax = 3.5, 8.0
		dmin, dmax = 1., 1000.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="rock", vs30=None, kappa=None, mechanism="normal",
				damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type == "hard rock":
				vs30 = 2000.
			elif soil_type == "rock":
				vs30 = 760.
			else:
				raise SoilTypeNotSupportedError(soil_type)
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									mechanism=mechanism, damping=damping)

	def plot_figure10(self, period=0.1):
		"""
		Plot figure 10 in Atkinson and Boore (2011).

		:param period:
			float, 0.1 or 1.
			(default: 0.1)
		"""
		self.plot_distance(mags=[5., 6., 7.5], dmin=5., dmax=500.,
							distance_metric="Joyner-Boore", h=0, imt="SA",
							T=period, imt_unit="cms2", mechanism="normal",
							damping=5, xscaling="log", yscaling="log",
							ymin={0.1: 0.5, 1.: 0.2}[period],
							ymax={0.1: 5000, 1.: 2000}[period], xgrid=2, ygrid=2)


class BindiEtAl2011(OqhazlibGMPE):
	"""
	"""
	def __init__(self):
		name, short_name = "BindiEtAl2011", "Bi_2011"
		distance_metric = "Joyner-Boore"
		Mmin, Mmax = 4., 6.9
		dmin, dmax = 0., 200.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="rock", vs30=None, kappa=None, mechanism="normal",
				damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type in ["A", "rock"]:
				vs30 = 800.
			elif soil_type == "B":
				vs30 = 360.
			elif soil_type == "C":
				vs30 = 180.
			elif soil_type == "D":
				vs30 = 179.
			else:
				raise SoilTypeNotSupportedError(soil_type)
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									mechanism=mechanism, damping=damping)

	def plot_figure11(self, imt="PGA", soil_type="A"):
		"""
		Plot figure 11 in Bindi et al 2011, p. 33.
		Note that we use strike-slip mechanism instead of unspecified
		(does not seem to be possible with oqhazlib).

		:param imt:
			str, intensity measure type, either "PGA" or "SA"
			(default: "PGA")
		:param soil_type:
			str, one of the soil types A (=rock), B and C
			(default: "A")
		"""
		if imt == "PGA":
			T = 0.
		if imt == "SA":
			T = 1.
		title = "Bindi et al. (2011) - Figure 11 - soil type %s" % soil_type
		self.plot_distance(mags=[6.3], dmin=0, dmax=300, imt=imt, T=T,
							imt_unit="ms2", epsilon=1, soil_type=soil_type,
							ymin=0.01, ymax=50., color="r", xgrid=2, ygrid=2,
							title=title)

	def plot_figure12(self, imt="PGA", soil_type="A"):
		"""
		Plot figure 12 in Bindi et al 2011, p. 34.
		Note that we use strike-slip mechanism instead of unspecified
		(does not seem to be possible with oqhazlib).

		:param imt:
			str, intensity measure type, either "PGA" or "SA"
			(default: "PGA")
		:param soil_type:
			str, one of the soil types A (=rock), B and C
			(default: "A")
		"""
		if imt == "PGA":
			T = 0.
			amax = 5
		if imt == "SA":
			T = 1.
			amax = 2
		title = "Bindi et al. (2011) - Figure 12 - soil type %s" % soil_type
		self.plot_distance(mags=[4.6], dmin=0, dmax=300, imt=imt, T=T,
							imt_unit="ms2", epsilon=1, soil_type=soil_type,
							ymin=0.001, ymax=amax, color="r", xgrid=2, ygrid=2,
							title=title)


class BooreAtkinson2008(OqhazlibGMPE):
	"""
	Valid VS30 range: 180 - 1300 m/s
	"""
	def __init__(self):
		name, short_name = "BooreAtkinson2008", "BA_2008"
		distance_metric = "Joyner-Boore"
		#Mmin, Mmax = 4.27, 7.9
		#dmin, dmax = 0., 280.
		Mmin, Mmax = 5.0, 8.0
		dmin, dmax = 0., 200.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="rock", vs30=None, kappa=None, mechanism="normal",
				damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type == "rock":
				vs30 = 760.
			else:
				raise SoilTypeNotSupportedError(soil_type)
		elif vs30 > 1500:
			raise VS30OutOfRangeError(vs30)
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									mechanism=mechanism, damping=damping)

	def plot_figure10(self, r=0.):
		"""
		Plot Figure 10 in the paper of Boore & Atkinson (2008)
		Note that we use strike-slip mechanism instead of unspecified
		(does not seem to be possible with oqhazlib).

		:param r:
			float, distance, either 0., 30. or 200.
		"""
		self.plot_spectrum(mags=[5., 6., 7., 8.], dmin=1., dmax=r, soil_type="rock",
							Tmin=1E-2, Tmax=10, amin=1E-2, amax=2000,
							imt_unit="cms2", mechanism="strike-slip")

	def plot_figure11(self, T=0.2):
		"""
		Plot Figure 10 in the paper of Boore & Atkinson (2008)
		Note that we use strike-slip mechanism instead of unspecified
		(does not seem to be possible with oqhazlib).

		:param T:
			float, period, either 0.2 or 3.0
		"""
		if T == 0.2:
			amin, amax = 1, 1E4
		elif T == 3.0:
			amin, amax = 0.05, 500
		self.plot_distance(mags=[5., 6., 7., 8.], imt="SA", T=T, soil_type="rock",
							mechanism="strike-slip", imt_unit="cms2",
							ymin=amin, ymax=amax, xscaling="log", yscaling="log",
							dmin=0.05, dmax=400)

	def plot_figure12(self, T=0.2):
		"""
		Plot Figure 10 in the paper of Boore & Atkinson (2008)
		Note that we use strike-slip mechanism instead of unspecified
		(does not seem to be possible with oqhazlib).

		:param T:
			float, period, either 0.2 or 3.0
		"""
		if T == 0.2:
			amin, amax = 20, 2000
		elif T == 3.0:
			amin, amax = 5, 500
		for vs30 in (180, 250, 360):
			self.plot_distance(mags=[7.], imt="SA", T=T, vs30=vs30,
								mechanism="strike-slip", imt_unit="cms2",
								ymin=amin, ymax=amax, dmin=0.5, dmax=200,
								xscaling="log", yscaling="log")


class BooreAtkinson2008Prime(OqhazlibGMPE):
	"""
	"""
	def __init__(self):
		name, short_name = "BooreAtkinson2008Prime", "BA_2008'"
		distance_metric = "Joyner-Boore"
		#Mmin, Mmax = 4.27, 7.9
		#dmin, dmax = 0., 280.
		Mmin, Mmax = 4.0, 8.0
		dmin, dmax = 0.0, 200.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="rock", vs30=None, kappa=None, mechanism="normal",
				damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type == "rock":
				vs30 = 760.
			else:
				raise SoilTypeNotSupportedError(soil_type)
		elif vs30 > 1500:
			raise VS30OutOfRangeError(vs30)
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									mechanism=mechanism, damping=damping)

	def plot_figure7(self, period=0.3):
		"""
		Plot figure 7 in Atkinson and Boore (2011).

		:param period:
			float, 0.3 or 1.
			(default: 0.3)

		NOTE: hack is required: unspecified rake must be used (see commented
		line in hazardlib files of BooreAtkinson2008 and BooreAtkinson2008Prime)
		"""
		plot_distance([self, BooreAtkinson2008()], mags=[4.], dmin=1., dmax=200.,
						h=0, imt="SA", T=period, imt_unit="cms2",
						mechanism="normal", damping=5, xscaling="log",
						yscaling="log", ymin={0.3: 0.01, 1.: 0.001}[period],
						ymax={0.3: 1000, 1.: 100}[period], xgrid=2, ygrid=2)


class Campbell2003(OqhazlibGMPE):
	"""
	"""
	def __init__(self):
		name, short_name = "Campbell2003", "C_2003"
		distance_metric = "Rupture"
		Mmin, Mmax = 5.0, 8.2
		dmin, dmax = 0., 1000.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="hard rock", vs30=None, kappa=None, mechanism="normal",
				damping=5):
		"""
		"""
		if vs30 is None:
			if not soil_type in ("generic rock", "hard rock"):
				raise SoilTypeNotSupportedError(soil_type)
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									mechanism=mechanism, damping=damping)

	def plot_figure2(self, imt="PGA", T=0):
		"""
		Plot Figure 2 in the paper of Campbell (2003)

		:param imt:
			str, intensity measure type, either "PGA" or "SA"
			(default: "PGA")
		:param T:
			float, period, either 0 (PGA), 0.2 (SA), 1.0 (SA) or 3.0 (SA)
		"""
		if (imt, T) == ("SA", 3.0):
			amin, amax = 1E-5, 1E0
		else:
			amin, amax = 1E-4, 1E1
		self.plot_distance(mags=[5., 6., 7., 8.], imt=imt, T=T,
							soil_type="hard rock", dmin=1, ymin=amin, ymax=amax)

	def plot_figure3(self, r=3.):
		"""
		Plot Figure 3 in the paper of Campbell (2003)
		Note that PGA is not included in the plot.

		:param r:
			float, either 3. or 30.
		"""
		self.plot_spectrum(mags=[5., 6., 7., 8.], d=r, soil_type="hard rock",
							Tmin=1E-3, Tmax=1E1, amin=1E-3, amax=1E+1)

	def plot_Figure13b_Drouet(self):
		"""
		Plot Figure 13b in the SHARE report by Drouet et al.
		"""
		self.plot_spectrum(mags=[6.], d=20., soil_type="hard rock",
							Tmin=1E-2, Tmax=8, imt_unit="ms2", amin=0.1, amax=10,
							plot_style="loglog")


class Campbell2003SHARE(OqhazlibGMPE):
	"""
	"""
	def __init__(self):
		name, short_name = "Campbell2003SHARE", "C_2003_SHARE"
		distance_metric = "Rupture"
		Mmin, Mmax = 5.0, 8.2
		dmin, dmax = 0., 1000.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="rock", vs30=None, kappa=None, mechanism="normal",
				damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type != "rock":
				raise SoilTypeNotSupportedError(soil_type)
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									mechanism=mechanism, damping=damping)

	def plot_figure13b_Drouet(self):
		"""
		Plot Figure 13b in the SHARE report by Drouet et al.
		Seems to correspond more or less with kappa = 0.03
		"""
		self.plot_spectrum(mags=[6.], d=20., soil_type="rock", Tmin=1E-2, Tmax=8,
						imt_unit="ms2", amin=0.1, amax=10, plot_style="loglog")


class Campbell2003adjusted(OqhazlibGMPE):
	"""
	"""
	def __init__(self):
		name, short_name = "Campbell2003adjusted", "C_2003adj"
		distance_metric = "Rupture"
		Mmin, Mmax = 5.0, 8.2
		dmin, dmax = 0., 1000.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="rock", vs30=None, kappa=0.03, mechanism="normal",
				damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type != "rock":
				raise SoilTypeNotSupportedError(soil_type)
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									kappa=kappa, mechanism=mechanism, damping=damping)


class Campbell2003HTT(Campbell2003):
	"""
	"""
	def __init__(self):
		name, short_name = "Campbell2003HTT", "C_2003HTT"
		distance_metric = "Rupture"
		Mmin, Mmax = 5.0, 8.2
		dmin, dmax = 0., 1000.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type=None, vs30=None, kappa=None, mechanism="normal",
				damping=5):
		"""
		"""
		if vs30 is None:
			raise SoilTypeNotSupportedError(soil_type)
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									kappa=kappa, mechanism=mechanism, damping=damping)


class CauzziFaccioli2008(OqhazlibGMPE):
	"""
	"""
	def __init__(self):
		name, short_name = "CauzziFaccioli2008", "CF_2008"
		distance_metric = "Hypocentral"
		Mmin, Mmax = 5.0, 7.2
		dmin, dmax = 6., 150.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="rock", vs30=None, kappa=None, mechanism="normal",
				damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type in ("rock", "typeA"):
				vs30 = 801
			elif soil_type == "typeB":
				vs30 = 600
			elif soil_type == "typeC":
				vs30 = 270
			elif soil_type == "typeD":
				vs30 = 150
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									mechanism=mechanism, damping=damping)


class ChiouYoungs2008(OqhazlibGMPE):
	"""
	"""
	def __init__(self):
		name, short_name = "ChiouYoungs2008", "CY_2008"
		## Note: this does not implement the hanging-wall effect, which also
		## depends on Rx and RJB
		distance_metric = "Rupture"
		Mmin, Mmax = 4.3, 7.9
		dmin, dmax = 0.2, 70.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="rock", vs30=None, vs30measured=None, z1pt0=None,
				z2pt5=None, kappa=None, mechanism="normal", damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type == ("rock"):
				vs30 = 800
			else:
				raise SoilTypeNotSupportedError(soil_type)
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									vs30measured=vs30measured, z1pt0=z1pt0,
									z2pt5=z2pt5, kappa=kappa, mechanism=mechanism,
									damping=damping)


class FaccioliEtAl2010(OqhazlibGMPE):
	"""
	"""
	def __init__(self):
		name, short_name = "FaccioliEtAl2010", "F_2010"
		distance_metric = "Rupture"
		Mmin, Mmax = 4.5, 7.6
		# TODO: check valid distance range!
		dmin, dmax = 1., 150.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="rock", vs30=None, kappa=None, mechanism="normal",
				damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type in ("rock", "typeA"):
				vs30 = 801
			elif soil_type == "typeB":
				vs30 = 600
			elif soil_type == "typeC":
				vs30 = 270
			elif soil_type == "typeD":
				vs30 = 150
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									mechanism=mechanism, damping=damping)


class FaccioliEtAl2010Ext(OqhazlibGMPE):
	"""
	"""
	def __init__(self):
		name, short_name = "FaccioliEtAl2010Ext", "F_2010X"
		distance_metric = "Rupture"
		Mmin, Mmax = 4.5, 7.6
		# TODO: check valid distance range!
		dmin, dmax = 1., 150.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="rock", vs30=None, kappa=None, mechanism="normal",
				damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type in ("rock", "typeA"):
				vs30 = 801
			elif soil_type == "typeB":
				vs30 = 600
			elif soil_type == "typeC":
				vs30 = 270
			elif soil_type == "typeD":
				vs30 = 150
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									mechanism=mechanism, damping=damping)


class ToroEtAl2002(OqhazlibGMPE):
	"""
	"""
	def __init__(self):
		name, short_name = "ToroEtAl2002", "T_2002"
		distance_metric = "Joyner-Boore"
		Mmin, Mmax = 5.0, 8.0
		dmin, dmax = 1., 1000.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="hard rock", vs30=None, kappa=None, mechanism="normal",
				damping=5):
		"""
		"""
		if vs30 is None:
			if not soil_type in ("generic rock", "hard rock"):
				raise SoilTypeNotSupportedError(soil_type)
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									mechanism=mechanism, damping=damping)

	def plot_figure13a_Drouet(self):
		"""
		Plot Figure 13a in the SHARE report by Drouet et al.
		"""
		self.plot_spectrum(mags=[6.], d=20., Tmin=1E-2, Tmax=8, imt_unit="ms2",
							amin=0.1, amax=10, plot_style="loglog",
							soil_type="hard rock")


class ToroEtAl2002SHARE(OqhazlibGMPE):
	"""
	"""
	def __init__(self):
		name, short_name = "ToroEtAl2002SHARE", "T_2002_SHARE"
		distance_metric = "Joyner-Boore"
		Mmin, Mmax = 5.0, 8.0
		dmin, dmax = 1., 100.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="rock", vs30=None, kappa=None, mechanism="normal",
				damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type != "rock":
				raise SoilTypeNotSupportedError(soil_type)
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									mechanism=mechanism, damping=damping)

	def plot_figure13a_Drouet(self):
		"""
		Plot Figure 13a in the SHARE report by Drouet et al.
		Not sure which kappa value this should correspond to (it should be 0.03)
		"""
		self.plot_spectrum(mags=[6.], d=20., Tmin=1E-2, Tmax=8, imt_unit="ms2",
							amin=0.1, amax=10, plot_style="loglog",
							soil_type="rock", mechanism="reverse")


class ToroEtAl2002adjusted(OqhazlibGMPE):
	"""
	"""
	def __init__(self):
		name, short_name = "ToroEtAl2002adjusted", "T_2002adj"
		distance_metric = "Joyner-Boore"
		Mmin, Mmax = 5.0, 8.0
		dmin, dmax = 1., 100.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="hard rock", vs30=None, kappa=0.03, mechanism="normal",
				damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type == "rock":
				vs30, kappa = 800, 0.03
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									mechanism=mechanism, damping=damping, kappa=kappa)


class ToroEtAl2002HTT(OqhazlibGMPE):
	"""
	"""
	def __init__(self):
		name, short_name = "ToroEtAl2002HTT", "T_2002HTT"
		distance_metric = "Joyner-Boore"
		Mmin, Mmax = 5.0, 8.0
		dmin, dmax = 1., 1000.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type=None, vs30=None, kappa=None, mechanism="normal",
				damping=5):
		"""
		"""
		if vs30 is None:
			raise SoilTypeNotSupportedError(soil_type)
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									kappa=kappa, mechanism=mechanism, damping=damping)


class ZhaoEtAl2006Asc(OqhazlibGMPE):
	"""
	"""
	def __init__(self):
		name, short_name = "ZhaoEtAl2006Asc", "Z_2006_Asc"
		distance_metric = "Rupture"
		Mmin, Mmax = 5.0, 8.3
		dmin, dmax = 0., 400.
		Mtype = "MW"
		dampings = [5.]
		imt_periods = {}
		## Note: attribute name for periods in ZhaoEtAl2006Asc is different, therefore they are provided here
		imt_periods["PGA"] = [0]
		imt_periods["SA"] = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
							0.60, 0.70, 0.80, 0.90, 1.00, 1.25, 1.50, 2.00,
							2.50, 3.00, 4.00, 5.00]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings, imt_periods)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="rock", vs30=None, vs30measured=None, z1pt0=None,
				z2pt5=None, kappa=None, mechanism="normal", damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type == "hard rock":
				vs30 = 1101
			elif soil_type in ("rock", "SC I"):
				vs30 = 850
			elif soil_type in ("hard soil", "SC II"):
				vs30 = 450
			elif soil_type in ("medium soil", "SC III"):
				vs30 = 250
			elif soil_type in ("soft soil", "SC IV"):
				vs30 = 200
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									mechanism=mechanism, damping=damping)

	def plot_figure2a(self):
		"""
		Plot Figure 2a in the paper of Zhao et al. (2006)
		"""
		self.plot_distance(mags=[7.], mechanism="strike-slip", soil_type="SC II",
							epsilon=1, dmin=0.3, dmax=400, ymin=2E-3, ymax=3.0,
							xscaling="log", yscaling="log")

	def plot_figure3a(self):
		"""
		Plot Figure 3a in the paper of Zhao et al. (2006)
		Note that it is not possible to exactly reproduce the figure, which uses
		"mean site conditions"
		"""
		self.plot_distance(mags=[5., 6., 7., 8.], mechanism="strike-slip",
							soil_type="SC II", dmin=1, dmax=400,
							ymin=5E-4, ymax=3.0, xscaling="log", yscaling="log")


class ZhaoEtAl2006AscMOD(ZhaoEtAl2006Asc):
	"""
	"""
	def __init__(self):
		name, short_name = "ZhaoEtAl2006AscMOD", "Z_2006_AscMOD"
		distance_metric = "Rupture"
		Mmin, Mmax = 5.0, 8.3
		dmin, dmax = 0., 400.
		Mtype = "MW"
		dampings = [5.]
		imt_periods = {}
		## Note: attribute name for periods in ZhaoEtAl2006Asc is different, therefore they are provided here
		imt_periods["PGA"] = [0]
		imt_periods["SA"] = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
							0.60, 0.70, 0.80, 0.90, 1.00, 1.25, 1.50, 2.00,
							2.50, 3.00, 4.00, 5.00]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings, imt_periods)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="rock", vs30=None, vs30measured=None, z1pt0=None,
				z2pt5=None, kappa=None, mechanism="normal", damping=5):
		"""
		"""
		if vs30 is None:
			if soil_type == "hard rock":
				vs30 = 1101
			elif soil_type in ("rock", "SC I"):
				vs30 = 850
			elif soil_type in ("hard soil", "SC II"):
				vs30 = 450
			elif soil_type in ("medium soil", "SC III"):
				vs30 = 250
			elif soil_type in ("soft soil", "SC IV"):
				vs30 = 220
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									mechanism=mechanism, damping=damping)


class RietbrockEtAl2013SS(OqhazlibGMPE):
	"""
	VS30 ~2300 m/s
	"""
	def __init__(self):
		name, short_name = "RietbrockEtAl2013SS", "R_2013SS"
		distance_metric = "Joyner-Boore"
		Mmin, Mmax = 3.0, 7.0
		dmin, dmax = 1., 300.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="hard_rock", vs30=None, kappa=None, mechanism="normal",
				damping=5):
		"""
		"""
		if vs30 is None:
			if not soil_type in ("generic rock", "hard rock"):
				raise SoilTypeNotSupportedError(soil_type)
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									mechanism=mechanism, damping=damping)

	def plot_figure4(self, T=0):
		"""
		Plot Figure 4 on p. 69.

		:param T:
			float, period, either 0 (PGA), 0.1 (SA), 0.2 (SA) or 1.0 (SA)
		"""
		if T==0:
			imt = "PGA"
		else:
			imt = "SA"
		self.plot_distance(mags=[4., 6.], imt=imt, imt_unit="cms2", T=T,
							soil_type="hard rock", dmin=1, dmax=300,
							ymin=0.01, ymax=2000)


class RietbrockEtAl2013MD(RietbrockEtAl2013SS):
	"""
	VS30 ~2300 m/s
	"""
	def __init__(self):
		name, short_name = "RietbrockEtAl2013MD", "R_2013MD"
		distance_metric = "Joyner-Boore"
		Mmin, Mmax = 3.0, 7.0
		dmin, dmax = 1., 300.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)


class RietbrockEtAl2013MDHTT(RietbrockEtAl2013SS):
	"""
	VS30 ~2300 m/s
	"""
	def __init__(self):
		name, short_name = "RietbrockEtAl2013MDHTT", "R_2013MDHTT"
		distance_metric = "Joyner-Boore"
		Mmin, Mmax = 3.0, 7.0
		dmin, dmax = 1., 300.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type=None, vs30=None, kappa=None, mechanism="normal",
				damping=5):
		"""
		"""
		if vs30 is None:
			raise SoilTypeNotSupportedError(soil_type)
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									kappa=kappa, mechanism=mechanism, damping=damping)


class PezeshkEtAl2011(OqhazlibGMPE):
	"""
	VS30 >= 2000 m/s
	"""
	def __init__(self):
		name, short_name = "PezeshkEtAl2011", "Pz_2011"
		distance_metric = "Rupture"
		Mmin, Mmax = 5.0, 8.0
		dmin, dmax = 1., 1000.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="hard_rock", vs30=None, kappa=None, mechanism="normal",
				damping=5):
		"""
		"""
		if vs30 is None:
			if not soil_type in ("generic rock", "hard rock"):
				raise SoilTypeNotSupportedError(soil_type)
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									mechanism=mechanism, damping=damping)


class CampbellBozorgnia2008(OqhazlibGMPE):
	"""
	"""
	def __init__(self):
		name, short_name = "CampbellBozorgnia2008", "CB_2008"
		# TODO: parameters below need to be looked up
		distance_metric = "Rupture"
		Mmin, Mmax = 5.0, 8.0
		dmin, dmax = 1., 1000.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type=None, vs30=None, vs30measured=None, z1pt0=None,
				z2pt5=None, kappa=None, mechanism="normal", damping=5):
		if vs30 is None:
			raise SoilTypeNotSupportedError(soil_type)
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									vs30measured=vs30measured, z1pt0=z1pt0,
									z2pt5=z2pt5, kappa=kappa, mechanism=mechanism,
									damping=damping)


class Anbazhagan2013(OqhazlibGMPE):
	"""
	"""
	def __init__(self):
		name, short_name = "Anbazhagan2013", "An_2013"
		distance_metric = "Rupture"
		Mmin, Mmax = 4.3, 8.7
		dmin, dmax = 1., 300.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="rock", vs30=None, vs30measured=None, z1pt0=None,
				z2pt5=None, kappa=None, mechanism="normal", damping=5):
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									vs30measured=vs30measured, z1pt0=z1pt0,
									z2pt5=z2pt5, kappa=kappa, mechanism=mechanism,
									damping=damping)


class Atkinson2015(OqhazlibGMPE):
	"""
	"""
	def __init__(self):
		name, short_name = "Atkinson2015", "Atk2015"
		distance_metric = "Hypocentral"
		Mmin, Mmax = 3.0, 6.0
		dmin, dmax = 1., 40.
		Mtype = "MW"
		dampings = [5.]

		OqhazlibGMPE.__init__(self, name, short_name, distance_metric, Mmin, Mmax,
							dmin, dmax, Mtype, dampings)

	def __call__(self, M, d, h=0., imt="PGA", T=0, imt_unit="g", epsilon=0,
				soil_type="rock", vs30=None, vs30measured=None, z1pt0=None,
				z2pt5=None, kappa=None, mechanism="normal", damping=5):
		return OqhazlibGMPE.__call__(self, M, d, h=h, imt=imt, T=T,
									imt_unit=imt_unit, epsilon=epsilon, vs30=vs30,
									vs30measured=vs30measured, z1pt0=z1pt0,
									z2pt5=z2pt5, kappa=kappa, mechanism=mechanism,
									damping=damping)



if __name__ == "__main__":
	pass

