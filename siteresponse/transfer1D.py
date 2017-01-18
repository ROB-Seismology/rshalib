"""
1-D linear site response using Kennet reflectivity theory
"""

import os
import numpy as np
import pylab

from tf import TransferFunction, ComplexTransferFunction, ComplexTransferFunctionSet


class ElasticLayer:
	"""
	Class representing a sediment layer with elastic properties

	:param Th:
		float, thickness (m)
	:param VS:
		float, shear-wave velocity (m/s)
	:param Rho:
		float, density (kg / cubic dm)
	:param QS:
		float, quality factor for shear waves
	:param VP:
		float, P-wave velocity (m/s)
		(default: 0.)
	:param QP:
		float, quality factor for P-waves
		(default: 0.)
	"""
	def __init__(self, Th, VS, Rho, QS, VP=0., QP=0.):
		self.Th = Th
		self.VS = VS
		self.Rho = Rho
		self.QS = QS
		self.VP = VP
		self.QP = QP

	def initial_critical_damping_ratio(self):
		"""
		Return initical critical damping ratio
		"""
		return 1. / (2 * self.QS)

	@property
	def kappa(self):
		"""
		Return kappa corresponding to sediment layer
		"""
		return self.Th / (self.QS * self.VS)

	@property
	def impedance(self):
		"""
		Return impedance
		"""
		# TODO: watch out for units (N s m-1) !
		return self.Rho * 1000. * self.VS

	@property
	def unit_weight(self):
		"""
		Unit weight (kN / cubic m)
		"""
		from scipy.constants import g

		return self.Rho * g


class ElasticLayerModel:
	"""
	Class representing model of elastic sediment layers (= piecewise
	constant profile), described from top to bottom.
	Bottom layer is assumed to be bedrock (half space), its thickness
	is not important.

	:param Th:
		list or array of floats, thickness (m)
	:param VS:
		list or array of floats, shear-wave velocity (m/s)
	:param Rho:
		list or array of floats, density (kg / cubic dm)
	:param QS:
		list or array of floats, quality factor for shear waves
	:param VP:
		list or array of floats, P-wave velocity (m/s)
		(default: [])
	:param QP:
		list or array of floats, quality factor for P-waves
		(default: [])
	"""
	def __init__(self, Th, VS, Rho, QS, VP=[], QP=[]):
		if len(Th) != len(VS) != len(Rho) != len(QS):
			raise Exception("Length of input arrays should be the same!")
		self.Th = np.asarray(Th)
		self.VS = np.asarray(VS)
		self.Rho = np.asarray(Rho)
		self.QS = np.asarray(QS)
		if len(VP) == 0:
			self.VP = np.zeros_like(self.VS)
		else:
			self.VP = np.asarray(VP)
		if len(QP) == 0:
			self.QP = np.zeros_like(self.QS)
		else:
			self.QP = np.asarray(QP)

	def __iter__(self):
		for i in range(self.num_layers):
			layer = ElasticLayer(self.Th[i], self.VS[i], self.Rho[i], self.QS[i])
			yield layer

	def __getitem__(self, key):
		if isinstance(key, (slice, list)):
			return ElasticLayerModel(self.Th[key], self.VS[key], self.Rho[key], self.QS[key])
		elif isinstance(key, int):
			return ElasticLayer(self.Th[key], self.VS[key], self.Rho[key], self.QS[key])
		else:
			raise TypeError, "Invalid argument type."

	@property
	def num_layers(self):
		return len(self.VS)

	@property
	def total_thickness(self):
		"""
		Total thickness excluding bedrock halfspace
		"""
		return np.sum(self.Th[:-1])

	def initial_critical_damping_ratio(self):
		"""
		Return initical critical damping ratio
		"""
		return 1. / (2 * self.QS)

	def get_layer_tops(self):
		"""
		Return layer tops
		"""
		return np.cumsum(np.concatenate([[0], self.Th[:-1]]))

	def get_layer_bases(self):
		"""
		Return layer bases
		"""
		return np.cumsum(self.Th)

	def get_all_depths(self):
		"""
		Return list with top and base of each layer
		"""
		layer_tops = self.get_layer_tops()
		layer_bases = self.get_layer_bases()
		return np.array(zip(layer_tops, layer_bases)).flatten()

	def plot(self):
		"""
		Plot layer model
		"""
		depths = self.get_all_depths()
		depths = -depths
		VS = np.array(zip(self.VS, self.VS)).flatten()
		Rho = np.array(zip(self.Rho, self.Rho)).flatten()
		QS = np.array(zip(self.QS, self.QS)).flatten()
		pylab.semilogx(VS, depths, 'r', linewidth=2, label="VS (m/s)")
		pylab.semilogx(Rho, depths, 'b', linewidth=2, label="Density (kg/m^3)")
		pylab.semilogx(QS, depths, 'g', linewidth=2, label="QS")
		pylab.grid(True)
		pylab.ylabel("Depth (m)")
		pylab.legend()
		pylab.show()

	def sample(self, num_samples, CVarVS=0.08, CVarRho=0.05, CVarTh=0.03, CVarQS=0.3, num_sigma=2, random_seed=None):
		"""
		:param num_samples:
			int, number of random samples
		:param CVarVS:
			float, coefficient of variation of input shear-wave velocity
			(default: 0.08)
		:param CVarRho:
			float, coefficient of variation of input density
			(default: 0.05)
		:param CVarTh:
			float, coefficient of variation of input layer thickness
			(default: 0.03)
		:param CVarQS:
			float, coefficient of variation of input shear-wave quality factor
			(default: 0.3)
		:paam num_sigma:
			float, number of standard deviations to consider for sampling
			(default: 2)
		:param random_seed:
			int or array-like, random seed (default: None)
		"""
		import scipy.stats
		np.random.seed(random_seed)

		for i in range(num_samples):
			## Note: coefficient of variation = sigma / mu
			#samples = scipy.stats.norm.rvs(0, 1, size=self.num_layers * 4)
			samples = scipy.stats.truncnorm.rvs(-num_sigma, num_sigma, 0, 1, size=self.num_layers * 4)
			#samples = np.random.normal(size=self.num_layers * 4)
			VSi = (self.VS + CVarVS * self.VS * samples[0::4]).clip(min=1)
			DEi = (self.Rho + CVarRho * self.Rho * samples[1::4]).clip(min=0.1)
			THi = (self.Th + CVarTh * self.Th * samples[2::4]).clip(min=0.1)
			QSi = (self.QS + CVarQS * self.QS * samples[3::4]).clip(min=0.1)
			yield ElasticLayerModel(THi, VSi, DEi, QSi)

	def get_mean_VS(self, depth=30):
		"""
		Compute average VS down to a certain depth

		:param depth:
			total depth over which to compute average VS

		:return:
			float: average VS (m/s)
		"""
		cum_thickness = 0
		Th, VS = [], []
		for layer in self:
			if cum_thickness <= depth:
				Th.append(layer.Th)
				VS.append(layer.VS)
				cum_thickness += layer.Th
		if cum_thickness < depth:
			print("Warning: Cumulative thickness of model less than %s m!" % depth)
		elif cum_thickness > depth:
			Th[-1] -= (cum_thickness - depth)

		return depth / np.sum(np.array(Th, 'f') / np.array(VS, 'f'))
		#weights = np.array(Th, 'f') / np.sum(Th)
		#return np.average(VS, weights=weights)

	def get_VS30(self):
		"""
		Compute VS30

		:return:
			float, VS30 (m/s)
		"""
		# TODO: Formula in Eurocode 8 is vs,30 = 30/Sum(hi/vi) !
		return self.get_mean_VS(30)

	def get_average_density(self, exclude_bottom_layer=True):
		"""
		Compute thickness-weighted average density

		:param exclude_bottom_layer:
			bool, whether or not to exclude the bottom layer
			(usually halfspace) from the average
			(default: True)
		"""
		if exclude_bottom_layer:
			Rho, Th = self.Rho[:-1], self.Th[:-1]
		else:
			Rho, Th = self.Rho, self.Th
		return np.average(Rho, weights=Th)

	def get_cumulative_kappa(self, kappa0, kappa0_top):
		"""
		Compute cumulative kappa through the sediment column

		:param kappa0:
			float, kappa of top or bottom layer (in seconds)
		:param kappa0_top:
			bool, whether or not kappa0 corresponds to kappa at surface

		:return:
			array of kappa values, corresponding to layer tops
		"""
		layer_kappas = np.array([layer.kappa for layer in self])
		if kappa0_top:
			cum_kappa = kappa0 - np.cumsum(layer_kappas)[:-1]
			cum_kappa = np.concatenate([[kappa0], cum_kappa])
		else:
			cum_kappa = (kappa0 + np.cumsum(layer_kappas[:-1][::-1]))[::-1]
			cum_kappa = np.concatenate([cum_kappa, [kappa0]])
		return cum_kappa

	def to_continuous_model(self):
		"""
		Convert layered model into a piecewise continuous model
		"""
		depths = self.get_all_depths()
		VS = np.array(zip(self.VS, self.VS)).flatten()
		Rho = np.array(zip(self.Rho, self.Rho)).flatten()
		QS = np.array(zip(self.QS, self.QS)).flatten()
		return ElasticContinuousModel(depths, VS, Rho, QS)


def reflectivity(layer_model, theta=0., FrMax=50, NFr=2048):
	"""
	Compute 1-D linear transfer function between lower layer in layer
	model (assumed to be outcropping) and the surface.
	This is a literal transcription of Philippe Rosset's reflectivity.m

	:param layer_model:
		instance of :class:`ElasticLayerModel`
	:param theta:
		float, angle of incidence of SH waves (default: 0.)
	:param FrMax:
		float, max. frequency of spectrum (default: 50)
	:param NFr:
		int, number of frequencies (default: 2048)

	:return:
		instance of :class:`ComplexTransferFunction`
	"""
	VS, QS = layer_model.VS, layer_model.QS
	DE, TH = layer_model.Rho, layer_model.Th

	## Considering only SH (Horizontal Shear wave), the parameters
	## ALO and QP will not be used
	NLa = len(VS)
	AS = (1+1j/2./QS) / (1+0.25/QS**2) * VS
	DFr = float(FrMax) / NFr
	AW = 0.
	TETH = np.radians(theta)
	STH = np.sin(TETH)
	PHI = -0.05/2
	FrArray = np.linspace(DFr, FrMax, NFr)

	W = np.zeros(NFr, dtype=np.complex)

	for NFri in range(NFr):
		RW = 2 * np.pi * FrArray[NFri]
		OMEGA = np.complex(RW, AW)
		ZOM = np.abs(OMEGA) / 2. / np.pi
		AIOM = -1j * OMEGA
		XLNF = (1j * PHI + np.log(ZOM)) / np.pi
		PHI = np.arctan2(AW,RW)
		BETA = AS / (1-XLNF / QS)
		WX0 = STH * OMEGA / BETA[NLa-1]
		AK = WX0
		AK2 = AK**2
		wb = OMEGA / BETA
		EMU = BETA**2 * DE
		wzb = np.sqrt(AK2 - wb**2)
		wzb[np.real(wzb<0)] = -wzb[np.real(wzb<0)]

		## Calculation of the coefficients of reflection and transmission on the interface (L-1/L)
		for L in range(NLa-1, 0, -1):
			CF1, CF2 = 0, 0
			CF1 = CF1 + 2 * AK2 * EMU[L-1]
			CF2 = CF2 + 2 * AK2 * EMU[L]
			CC1 = EMU[L] * wzb[L] - EMU[L-1] * wzb[L-1]
			CC2 = EMU[L-1] * wzb[L-1] + EMU[L] * wzb[L]
			RDIH = -CC1 / CC2
			RUIH = -RDIH
			TDIH = 2 * EMU[L-1] * wzb[L-1] / CC2
			TUIH = 2 * EMU[L] * wzb[L] / CC2

			## Application of iterative relations for obtaining the characteristics
			## of reflection and trasmission in the lower medium to the interface(L-1/L)
			if L < (NLa - 1):
				QQ1 = 1 - RUIH * RDH
				RDIH = RDIH + TUIH *RDH * (TDIH / QQ1)
				TDIH = TDH * TDIH / QQ1
				TUIH = TUH * TUIH / QQ1
				RUIH = RUH + TDH * RUIH * (TUH / QQ1)

			PHB = -wzb[L-1] * TH[L-1]
			Q2 = np.exp(PHB)
			RDH = RDIH * Q2**2
			RUH = RUIH
			TDH = TDIH * Q2
			TUH = TUIH * Q2

		QQ1 = 1 - RDH
		DD = TUH / QQ1

		## Calculation of the incident field
		## Ignore halfspace thickness!
		A1 = np.exp(wzb[NLa-1] * np.sum(TH[:-1]))
		## The obtained transfer function W of one iteration (complex data)
		#W[NFri] = 2 * DD * A1 / 2
		W[NFri] = DD * A1
		## If necessary the absolute data can be taken but this is inappropriate
		## if convolutions will be done
		#%V[NFi] = np.abs(V[NFi])

	return ComplexTransferFunction(FrArray, W)


def nrattle(layer_model, zdepth=0, theta=0., FrMax=50, NFr=2048):
	"""
	Compute 1-D linear transfer function between lower layer in layer
	model (assumed to be outcropping) and the surface.
	Haskell-Thomson transfer function (SH) for horizontally stratified medium
	This is a transcription of the Fortran program NRATTLE by Chuck Mueller,
	which is part of Dave Boore's SITE_AMP package

	:param layer_model:
		instance of :class:`ElasticLayerModel`
	:param zdepth:
		float, depth at which response has to be computed (in m)
		(default: 0)
	:param theta:
		float, angle of incidence of SH waves (default: 0.)
	:param FrMax:
		float, max. frequency of spectrum (default: 50)
	:param NFr:
		int, number of frequencies (default: 2048)

	:return:
		instance of :class:`ComplexTransferFunction`
	"""
	## Constant
	CI = 0.0 + 1.0j

	## Frequencies
	INYQ = NFr
	#SAMP = 2 * FrMax
	#T = 2 * (INYQ - 1) / SAMP
	T = float(NFr) / FrMax
	DELF = 1.0 / T

	zdepth /= 1000.

	## Layer properties
	VS = layer_model.VS / 1000.
	QiS = 1.0 / layer_model.QS[:-1]
	RHO = layer_model.Rho
	H = layer_model.Th / 1000.
	Z = layer_model.get_layer_bases()[:-1] / 1000.
	NLAY = layer_model.num_layers - 1

	## Halfspace = bottom layer
	NLAY_SRC = n = NLAY
	BETAN = VS[n]
	RHON = RHO[n]

	THETA = np.radians(theta)
	HSLOW = np.sin(THETA) / BETAN
	HS2 = HSLOW ** 2

	## Complex rigidity of layer
	XMU = np.zeros(NLAY+1, dtype=np.complex128)
	## Complex vertical slowness of layer for given angle of incidence
	XXX = np.zeros(NLAY+1, dtype=np.complex128)

	XMU[n] = RHON * BETAN * BETAN
	XXX[n] = np.sqrt((RHON / XMU[n] - HS2) * np.complex(1, 0))

	for i in range(NLAY):
		XMU[i] = (RHO[i] * VS[i] * VS[i] *
					(1.0 + 0.5 * CI * QiS[i]) * (1.0 + 0.5 * CI * QiS[i]))
		XXX[i] = np.sqrt(RHO[i] / XMU[i] - HS2)

	## Frequency array (omitting zero frequency)
	F = np.linspace(DELF, FrMax, NFr)

	## Loop over frequency
	TSTAR = 0.0
	VV = np.zeros(NFr, dtype=np.complex128)
	#F = np.zeros(NFr)
	for IFRQ in range(INYQ):
		#F[IFRQ] = IFRQ * DELF
		W = 2 * np.pi * F[IFRQ]

		## Loop over the number of layers (propagate)
		## Complex wave propagation coefficients for the layer
		A = np.zeros(NLAY+1, dtype=np.complex128)
		B = np.zeros(NLAY+1, dtype=np.complex128)
		## Exponent for extended floating point
		EXE = np.zeros(NLAY+1, dtype=np.complex128)

		A[0] = 1.0 + 0.0j
		B[0] = 1.0 + 0.0j
		for i in range(NLAY):
			j = i + 1
			CY = (XMU[i] / XMU[j]) * (XXX[i] / XXX[j])
			CZ = CI * W * H[i]
			FACC = CZ * XXX[i]
			FACR = np.real(FACC)
			FACI = np.imag(FACC)
			FACZ = np.complex(np.cos(FACI), np.sin(FACI))

			EXPP = 1.0
			EXPM = -FACR - FACR

			Q11 = 0.5 * (1.0 + CY) * FACZ
			Q21 = 0.5 * (1.0 - CY) * FACZ
			if EXPM < -50:
				Q12 = 0.0
				Q22 = 0.0
			else:
				Q12 = 0.5 * (1.0 - CY) * np.conj(FACZ) * np.exp(EXPM)
				Q22 = 0.5 * (1.0 + CY) * np.conj(FACZ) * np.exp(EXPM)

			A[j] = Q11 * A[i] + Q12 * B[i]
			B[j] = Q21 * A[i] + Q22 * B[i]
			EXE[j] = EXE[i] + FACR

			if IFRQ == 0 and i < NLAY:
				TSTAR = TSTAR - 2.0 * np.imag(XXX[i]) * H[i]

		i = n

		## Compute layer stack response at desired depth
		if zdepth < 0:
			raise Exception("Depth must be positive!")

		elif zdepth >= Z[NLAY-1]:
			## Desire motion in half space
			ii = NLAY
			DS = zdepth - Z[NLAY-1]
			CZ = CI * W * DS
			DEXPP = EXE[ii] - EXE[n]
			if DEXPP < -50:
				VV[IFRQ] = np.complex(0.0, 0.0)
			else:
				VV[IFRQ] = ((0.5 * (A[ii] * np.exp(CZ * XXX[ii])
						+ B[ii] * np.exp(-CZ * XXX[ii])) / A[n])
						* np.exp(DEXPP))

		elif zdepth == 0.0:
			## Desire motion at the surface
			ii = 0
			VV[IFRQ] = A[0] / A[n]
			DEXPP = EXE[ii] - EXE[n]
			if DEXPP < -50:
				VV[IFRQ] = np.complex(0.0, 0.0)
			else:
				VV[IFRQ] = (A[0] / A[n]) * np.exp(DEXPP)

		else:
			## Desire motion within a layer
			for i in range(NLAY):
				DS = zdepth - Z[i]
				if DS < 0:
					DS = -DS
					CZ = CI * W * DS
					ii = i
					DEXPP = EXE[ii] - EXE[n]
					if DEXPP < -50:
						VV[IFRQ] = np.complex(0.0, 0.0)
					else:
						VV[IFRQ] = ((0.5 * (A[ii] * np.exp(CZ * XXX[ii])
								+ B[ii] * np.exp(-CZ * XXX[ii])) / A[n])
								* np.exp(DEXPP))
					break

	return ComplexTransferFunction(F, VV)


def roll(layer_model, zdepth=0, theta=0., PorS="S", HorV="H", FrMax=50, NFr=2048):
	"""
	Requires both VP and VS
	"""
	## Constant
	CI = 0.0 + 1.0j

	## Frequencies
	INYQ = NFr
	SAMP = 2 * FrMax
	T = 2 * (INYQ - 1) / SAMP
	DELF = 1.0 / T

	#zdepth /= 1000.

	## Layer properties
	VP = layer_model.VP
	QP = layer_model.QP
	VS = layer_model.VS
	QS = layer_model.QS
	RHO = layer_model.Rho
	Z = layer_model.get_layer_bases()
	NLAY = layer_model.num_layers - 1

	## Halfspace = bottom layer
	n = NLAY
	BETAN = VS[n]
	VPN = VP[n]
	RHON = RHO[n]

	if PorS == "P":
		RAT = np.sin(np.radians(theta)) / VPN
	else:
		RAT = np.sin(np.radians(theta)) / BETAN
	RAT2 = RAT * RAT

	A2 = VP * VP * (1.0 + CI / QP)
	A = np.sqrt(A2)
	XLA = np.sqrt(1.0 / A2 - RAT2)
	B2 = VS * VS * (1.0 + CI / QS)
	B = np.sqrt(B2)
	XLB = np.sqrt(1.0 / B2 - RAT2)

	A[n] = VPN
	A2[n] = A[n] * A[n]
	XLA[n] = np.sqrt(1.0 / A2[n] - RAT2)
	B[n] = BETAN
	B2[n] = B[n] * B[n]
	XLB[n] = np.sqrt(1.0 / B2[n] - RAT2)

	P11 = np.ones(NLAY+1, dtype=np.complex128)
	P12 = np.zeros(NLAY+1, dtype=np.complex128)
	P13 = np.zeros(NLAY+1, dtype=np.complex128)
	P14 = np.zeros(NLAY+1, dtype=np.complex128)
	P21 = np.zeros(NLAY+1, dtype=np.complex128)
	P22 = np.ones(NLAY+1, dtype=np.complex128)
	P23 = np.zeros(NLAY+1, dtype=np.complex128)
	P24 = np.zeros(NLAY+1, dtype=np.complex128)
	P31 = np.zeros(NLAY+1, dtype=np.complex128)
	P32 = np.zeros(NLAY+1, dtype=np.complex128)
	P33 = np.ones(NLAY+1, dtype=np.complex128)
	P34 = np.zeros(NLAY+1, dtype=np.complex128)
	P41 = np.zeros(NLAY+1, dtype=np.complex128)
	P42 = np.zeros(NLAY+1, dtype=np.complex128)
	P43 = np.zeros(NLAY+1, dtype=np.complex128)
	P44 = np.ones(NLAY+1, dtype=np.complex128)

	R31 = RHO[0] * A2[0] * (1.0 - 2.0 * RAT2 * B2[0])
	R34 = 4.0 * RHO[0] * RAT * XLB[0] * B2[0] * B2[0]
	R42 = -2.0 * RHO[0] * RAT * XLA[0] * A2[0] * B2[0]
	R43 = 2.0 * RHO[0] * B2[0] * B2[0] * (XLB[0] * XLB[0] - RAT2)
	RN31 = RHO[n] * A2[n] * (1.0 - 2.0 * RAT2 * B2[n])
	RN34 = 4.0 * RHO[n] * RAT * XLB[n] * B2[n] * B2[n]
	RN42 = -2.0 * RHO[n] * RAT * XLA[n] * A2[n] * B2[n]
	RN43 = 2.0 * RHO[n] * B2[n] * B2[n] * (XLB[n] * XLB[n] - RAT2)

	if PorS == "P":
		BNR = (RN43 * RN31 + RN34 * RN42) / (RN34 * RN42 - RN43 * RN31)
		DNR = 2.0 * RN42 * RN31 / (RN42 * RN34 - RN31 * RN43)
	else:
		BNR = 2.0 * RN43 * RN34 / (RN34 * RN42 - RN43 * RN31)
		DNR = (RN42 * RN34 + RN31 * RN43) / (RN42 * RN34 - RN31 * RN43)

	## Frequency array (omitting zero frequency)
	F = np.linspace(DELF, FrMax, NFr)

	## Loop over frequency
	VV = np.zeros(NFr, dtype=np.complex128)
	#F = np.zeros(NFr)
	for IFRQ in range(INYQ):
		if IFRQ == 0:
			#F[IFRQ] = 0.0
			#PER = 1.0E+6
			VV[IFRQ] = np.complex(1.0, 0.0)
			continue

		#F[IFRQ] = (IFRQ - 1) * DELF
		P = 2.0 * np.pi * F[IFRQ]
		CIP = CI * P

		## Loop over the number of layers (propagate)
		for i in range(NLAY-1, -1, -1):
			k = i + 1
			CX = RHO[k] / RHO[i]
			CY = (RHO[k] * B2[k] - RHO[i] * B2[i]) / RHO[i]
			CZ = 2.0 * RAT2 * CY

			T11 = (A2[k] / A2[i]) * (CX - CZ)
			T14 = (4.0 * RAT * XLB[k] * B2[k] / A2[i]) * CY
			T41 = ((RAT * A2[k]) / (2.0 * XLB[i] * B2[i])) * (CX - 1.0 - CZ)
			T44 = ((XLB[k] * B2[k]) / (XLB[i] * B2[i])) * (1.0 + CZ)
			T22 = ((XLA[k] * A2[k]) / (XLA[i] * A2[i])) * (1.0 + CZ)
			T23 = ((2.0 * RAT * B2[k]) / (XLA[i] * A2[i])) * (1.0 - CX + CZ)
			T32 = ((-RAT * XLA[k] * A2[k]) / B2[i]) * CY
			T33 = (B2[k] / B2[i]) * (CX - CZ)

			CX = CIP * Z[i]
			Q11 = 0.5 * (T11 + T22) * np.exp(CX * (XLA[k] - XLA[i]))
			Q12 = 0.5 * (T11 - T22) * np.exp(-CX * (XLA[k] + XLA[i]))
			Q13 = 0.5 * (T14 + T23) * np.exp(CX * (XLB[k] - XLA[i]))
			Q14 = 0.5 * (T23 - T14) * np.exp(-CX * (XLB[k] + XLA[i]))
			Q21 = 0.5 * (T11 - T22) * np.exp(CX * (XLA[i] + XLA[k]))
			Q22 = 0.5 * (T11 + T22) * np.exp(CX * (XLA[i] - XLA[k]))
			Q23 = 0.5 * (T14 - T23) * np.exp(CX * (XLA[i] + XLB[k]))
			Q24 = 0.5 * (-T14 - T23) * np.exp(CX * (XLA[i] - XLB[k]))
			Q31 = 0.5 * (T32 + T41) * np.exp(CX * (XLA[k] - XLB[i]))
			Q32 = 0.5 * (T41 - T32) * np.exp(-CX * (XLA[k] + XLB[i]))
			Q33 = 0.5 * (T33 + T44) * np.exp(CX * (XLB[k] - XLB[i]))
			Q34 = 0.5 * (T33 - T44) * np.exp(-CX * (XLB[k] + XLB[i]))
			Q41 = 0.5 * (T32 - T41) * np.exp(CX * (XLB[i] + XLA[k]))
			Q42 = 0.5 * (-T32 - T41) * np.exp(CX * (XLB[i] - XLA[k]))
			Q43 = 0.5 * (T33 - T44) * np.exp(CX * (XLB[i] + XLB[k]))
			Q44 = 0.5 * (T33 + T44) * np.exp(CX * (XLB[i] - XLB[k]))
			P11[i] = Q11 * P11[k] + Q12 * P21[k] + Q13 * P31[k] + Q14 * P41[k]
			P12[i] = Q11 * P12[k] + Q12 * P22[k] + Q13 * P32[k] + Q14 * P42[k]
			P13[i] = Q11 * P13[k] + Q12 * P23[k] + Q13 * P33[k] + Q14 * P43[k]
			P14[i] = Q11 * P14[k] + Q12 * P24[k] + Q13 * P34[k] + Q14 * P44[k]
			P21[i] = Q21 * P11[k] + Q22 * P21[k] + Q23 * P31[k] + Q24 * P41[k]
			P22[i] = Q21 * P12[k] + Q22 * P22[k] + Q23 * P32[k] + Q24 * P42[k]
			P23[i] = Q21 * P13[k] + Q22 * P23[k] + Q23 * P33[k] + Q24 * P43[k]
			P24[i] = Q21 * P14[k] + Q22 * P24[k] + Q23 * P34[k] + Q24 * P44[k]
			P31[i] = Q31 * P11[k] + Q32 * P21[k] + Q33 * P31[k] + Q34 * P41[k]
			P32[i] = Q31 * P12[k] + Q32 * P22[k] + Q33 * P32[k] + Q34 * P42[k]
			P33[i] = Q31 * P13[k] + Q32 * P23[k] + Q33 * P33[k] + Q34 * P43[k]
			P34[i] = Q31 * P14[k] + Q32 * P24[k] + Q33 * P34[k] + Q34 * P44[k]
			P41[i] = Q41 * P11[k] + Q42 * P21[k] + Q43 * P31[k] + Q44 * P41[k]
			P42[i] = Q41 * P12[k] + Q42 * P22[k] + Q43 * P32[k] + Q44 * P42[k]
			P43[i] = Q41 * P13[k] + Q42 * P23[k] + Q43 * P33[k] + Q44 * P43[k]
			P44[i] = Q41 * P14[k] + Q42 * P24[k] + Q43 * P34[k] + Q44 * P44[k]

		G11 = R31 *(P12[0] + P22[0]) + R34 *(P32[0] - P42[0])
		G21 = R42 *(P12[0] - P22[0]) + R43 *(P32[0] + P42[0])
		G12 = R31 *(P14[0] + P24[0]) + R34 *(P34[0] - P44[0])
		G22 = R42 *(P14[0] - P24[0]) + R43 *(P34[0] + P44[0])

		if np.abs(G22 * G11 - G12 * G21) == 0.0:
			## Matrix singular
			VV[IFRQ] = np.nan
			break

		if PorS == "P":
			F1 = R31 * (P11[0] + P21[0]) + R34 *(P31[0] - P41[0])
			F2 = R42 * (P11[0] - P21[0]) + R43 *(P31[0] + P41[0])
		else:
			F1 = R31 * (P13[0] + P23[0]) + R34 * (P33[0] - P43[0])
			F2 = R42 * (P13[0] - P23[0]) + R43 * (P33[0] + P43[0])

		BN = (G12 * F2 - G22 * F1) / (G22 * G11 - G12 * G21)
		DN = (G11 * F2 - G21 * F1) / (G21 * G12 - G11 * G22)

		found_zdepth = False
		for i4loop in range(NLAY):
			if zdepth <= Z[i4loop]:
				found_zdepth = True
				break

		if found_zdepth:
			i = i4loop
		else:
			i = n

		R11 = -RAT * A2[i]
		R14 = 2.0 * XLB[i] * B2[i]
		R22 = XLA[i] * A2[i]
		R23 = 2.0 * RAT * B2[i]
		CA = CIP * XLA[i] * zdepth
		CB = CIP * XLB[i] * zdepth

		if PorS == "P":
			if HorV == "V":
				VV[IFRQ] = ((R22 * ((P11[i] + P12[i] * BN + P14[i] * DN) * np.exp(CA)
					- (P21[i] + P22[i] * BN + P24[i] * DN) * np.exp(-CA))
					+ R23 *((P31[i] + P32[i] * BN + P34[i] * DN) * np.exp(CB)
					+ (P41[i] + P42[i] * BN + P44[i] * DN) * np.exp(-CB)))
					/ (XLA[n] * A2[n] * (1.0 - BNR) +2.0 * RAT * B2[n] * DNR))
			elif HorV == "H":
				if theta == 0.0:
					VV[IFRQ] = 0.0
				else:
					VV[IFRQ] = ((R11 * ((P11[i] + P12[i] * BN + P14[i] * DN) * np.exp(CA)
						+ (P21[i] + P22[i] * BN + P24[i] * DN) * np.exp(-CA))
						+ R14 * ((P31[i] + P32[i] * BN + P34[i] * DN) * np.exp(CB)
						- (P41[i] + P42[i] * BN + P44[i] * DN) * np.exp(-CB)))
						/ (-RAT * A2[n] * (1.0 + BNR) -2.0 * XLB[n] * B2[n] * DNR))
		elif PorS == "S":
			if HorV == "V":
				if theta == 0.0:
					VV[IFRQ] = 0.0
				else:
					VV[IFRQ] = ((R22 * ((P12[i] * BN + P13[i] + P14[i] * DN) * np.exp(CA)
						- (P22[i] * BN + P23[i] + P24[i] * DN) * np.exp(-CA))
						+ R23 * ((P32[i] * BN + P33[i] + P34[i] * DN) * np.exp(CB)
						+ (P42[i] * BN + P43[i] + P44[i] * DN) * np.exp(-CB)))
						/ (-XLA[n] * A2[n] * BNR + 2.0 * RAT * B2[n] * (1.0 + DNR)))
			elif HorV == "H":
					VV[IFRQ] = ((R11 * ((P12[i] * BN + P13[i] + P14[i] * DN) * np.exp(CA)
						+ (P22[i] * BN + P23[i] + P24[i] * DN) * np.exp(-CA))
						+ R14 * ((P32[i] * BN + P33[i] + P34[i] * DN) * np.exp(CB)
						- (P42[i] * BN + P43[i] + P44[i] * DN) * np.exp(-CB)))
						/ (-RAT * A2[n] * BNR + 2.0 * XLB[n] * B2[n] * (1.0 - DNR)))

	return ComplexTransferFunction(F, VV)


def randomized_reflectivity(layer_model, zdepth=0, theta=0., FrMax=50, NFr=2048, num_samples=10, method="nrattle", CVarVS=0.08, CVarRho=0.05, CVarTh=0.03, CVarQS=0.3, num_sigma=2, random_seed=None):
	"""
	:param layer_model:
		instance of :class:`ElasticLayerModel`
	:param zdepth:
		float, depth at which response has to be computed (in m)
		Only applies wheh :param:`method` is set to "nrattle"
		(default: 0)
	:param theta:
		float, angle of incidence of SH waves (default: 0.)
	:param FrMax:
		float, max. frequency of spectrum (default: 50)
	:param NFr:
		int, number of frequencies (default: 2048)
	:param num_samples:
		int, number of random samples (default: 10)
	:param method:
		str, name of function to compute transfer function, one of:
		- "nrattle": using NRATTLE code by Chuck Mueller from Dave Boore's
			SITE_AMP package
		- "reflectivity": using Philip Rosset's implementation of Kennet's
			reflectivity method
		(default: "nrattle")
	:param CVarVS:
		float, coefficient of variation of input shear-wave velocity
		(default: 0.08)
	:param CVarRho:
		float, coefficient of variation of input density
		(default: 0.05)
	:param CVarTh:
		float, coefficient of variation of input layer thickness
		(default: 0.03)
	:param CVarQS:
		float, coefficient of variation of input shear-wave quality factor
		(default: 0.3)
	:paam num_sigma:
		float, number of standard deviations to consider for sampling
		(default: 2)
	:param random_seed:
		int or array-like, random seed (default: None)

	:return:
		instance of :class:`ComplexTransferFunctionSet`
	"""
	# TODO: add CVarVP and CVarQP for roll
	TRco = np.zeros((num_samples, NFr), dtype=np.complex)
	i = 0
	for lm in layer_model.sample(num_samples, CVarVS=CVarVS, CVarRho=CVarRho, CVarTh=CVarTh, CVarQS=CVarQS, num_sigma=num_sigma, random_seed=random_seed):
		#lm.plot()
		if method.lower() == "reflectivity":
			TFi = reflectivity(lm, theta=theta, FrMax=FrMax, NFr=NFr)
		elif method.lower() == "nrattle":
			TFi = nrattle(lm, zdepth=zdepth, theta=theta, FrMax=FrMax, NFr=NFr)
		TRco[i] = TFi.data
		i += 1

	return ComplexTransferFunctionSet(TFi.freqs, TRco)


def randomized_reflectivity_mp(layer_model, zdepth=0., theta=0., FrMax=50, NFr=2048, num_samples=10, method="nrattle", CVarVS=0.08, CVarRho=0.05, CVarTh=0.03, CVarQS=0.3, num_sigma=2, random_seed=None):
	"""
	Multiprocessing version of :func:`randomized_reflectivity`.
	"""
	import multiprocessing
	from ..calc.mp import mp_func_wrapper

	num_processes = min(multiprocessing.cpu_count(), num_samples)
	print("Starting %d parallel processes" % num_processes)

	TRco = np.zeros((num_samples, NFr), dtype=np.complex)

	job_arg_list = []
	for i, lm in enumerate(layer_model.sample(num_samples, CVarVS=CVarVS, CVarRho=CVarRho, CVarTh=CVarTh, CVarQS=CVarQS, num_sigma=num_sigma, random_seed=random_seed)):
		if method == "reflectivity":
			tf_func = reflectivity
			params = (lm, theta, FrMax, NFr)
		elif method == "nrattle":
			tf_func = nrattle
			params = (lm, zdepth, theta, FrMax, NFr)
		job_arg_list.append((tf_func, params))

	pool = multiprocessing.Pool(processes=num_processes)
	TFi_list = pool.map(mp_func_wrapper, job_arg_list, chunksize=1)

	freqs = TFi_list[0].freqs
	for i in range(num_samples):
		TFi = TFi_list[i]
		TRco[i] = TFi.data

	return ComplexTransferFunctionSet(TFi.freqs, TRco)


def transfer1D(input_filespec, out_path=None, want_plot=True, theta=0., FrMax=50, NFr=2048, num_samples=10, CVarVS=0.08, CVarRho=0.05, CVarTh=0.03, CVarQS=0.3, num_sigma=2, random_seed=None, mp=True):
	"""
	Python implementation of matlab program transfer1D by Philippe Rosset
	1-D modelling of site effects using the Kennett reflectivity method
	combined with a Monte Carlo approach.

	The program reads a layer model stored in a .txt file (shear-wave velocity,
	density, thickness, and shear-wave quality factor) to calculate the ground
	spectrum (also called transfer function) of the soil. A Monte-Carlo approach
	is used to calculate the uncertainties on the parameters. The results of the
	transfer function are written to csv files (mean and std deviation in absolute
	and complex numbers), and are plotted.

	:param input_filespec:
		str, full path to input file containing layer model
	:param out_path:
		str, path where output files should be stored
		(default: None, will write to folder of input file)
	:param want_plot:
		bool, whether or not a plot should be shown
	:param theta:
		float, angle of incidence of SH waves (default: 0.)
	:param FrMax:
		float, max. frequency of spectrum (default: 50)
	:param NFr:
		int, number of frequencies (default: 2048)
	:param num_samples:
		int, number of random samples (default: 10)
	:param CVarVS:
		float, coefficient of variation of input shear-wave velocity
		(default: 0.08)
	:param CVarRho:
		float, coefficient of variation of input density
		(default: 0.05)
	:param CVarTh:
		float, coefficient of variation of input layer thickness
		(default: 0.03)
	:param CVarQS:
		float, coefficient of variation of input shear-wave quality factor
		(default: 0.3)
	:param random_seed:
		int or array-like, random seed (default: None)
	:param mp:
		bool, whether or not to use multiprocessing (default: True)
	"""
	layer_model = parse_layer_model(input_filespec)
	#layer_model.plot()
	TF = reflectivity(layer_model, theta=theta, FrMax=FrMax, NFr=NFr)
	if num_samples > 0:
		if mp == False:
			TFsim = randomized_reflectivity(layer_model, theta=theta, FrMax=FrMax, NFr=NFr, num_samples=num_samples, CVarVS=CVarVS, CVarRho=CVarRho, CVarTh=CVarTh, CVarQS=CVarQS, num_sigma=num_sigma, random_seed=random_seed)
		else:
			TFsim = randomized_reflectivity_mp(layer_model, theta=theta, FrMax=FrMax, NFr=NFr, num_samples=num_samples, CVarVS=CVarVS, CVarRho=CVarRho, CVarTh=CVarTh, CVarQS=CVarQS, num_sigma=num_sigma, random_seed=random_seed)

	if not out_path:
		out_path = os.path.split(input_filespec)[0]
	base_filename = os.path.splitext(os.path.split(input_filespec)[1])[0]

	## Write TF calculated from absolute input values
	out_filespec = os.path.join(out_path, base_filename + "_TF.csv")
	TF.export_csv(out_filespec)

	## Write TF's from randomly varied input values
	if num_samples > 0:
		out_filespec = os.path.join(out_path, base_filename + "_sim.csv")
		TFsim.export_csv(out_filespec)

	## Plot
	if want_plot:
		if num_samples > 0:
			TF_list = list(TFsim)
			colors = [(0.5, 0.5, 0.5)] * num_samples
			line_widths = [1] * num_samples
			line_styles = ['-'] * num_samples
			labels = ["Individual models"] + ["_nolegend_"] * (num_samples - 1)

			TF_list.append(TFsim.median())
			colors.append('r')
			line_widths.append(2)
			line_styles.append('-')
			labels.append("Median")

			TF_list.append(TFsim.percentile(84))
			colors.append('r')
			line_widths.append(2)
			line_styles.append('--')
			labels.append("84th percentile")
		else:
			TF_list, colors, line_widths, line_styles, labels = [], [], [], [], []

		TF_list.append(TF)
		colors.append('b')
		line_widths.append(2)
		line_styles.append('-')
		labels.append("TF (no MC)")

		plot_TF_magnitude(TF_list, colors=colors, line_widths=line_widths, line_styles=line_styles, labels=labels)


def parse_layer_model(filespec):
	"""
	Parse Transfer1D input file

	:param filespec:
		str, full path to input file

	:return:
		instance of :class:`ElasticLayerModel`
	"""
	VS, DE, TH, QS = [], [], [], []
	fd = open(filespec)
	for line in fd:
		vs, de, th, qs = map(float, line.split())
		VS.append(vs)
		DE.append(de)
		TH.append(th)
		QS.append(qs)
	fd.close()
	return ElasticLayerModel(TH, VS, DE, QS)


class ElasticContinuousModel:
	"""
	Class representing a piecewise continuous model of elastic sediment
	profile, described from top to bottom.
	Bottom layer is assumed to be bedrock (half space).

	:param Z:
		list or array of floats, depths (m)
	:param VS:
		list or array of floats, shear-wave velocity (m/s)
	:param Rho:
		list or array of floats, density (kg / cubic dm = g / cubic cm)
	:param QS:
		list or array of floats, quality factor for shear waves
	"""
	def __init__(self, Z, VS, Rho, QS):
		if len(Z) != len(VS) != len(Rho) != len(QS):
			raise Exception("Length of input arrays should be the same!")
		self.Z = np.asarray(Z)
		self.VS = np.asarray(VS)
		self.Rho = np.asarray(Rho)
		self.QS = np.asarray(QS)

	def __len__(self):
		return len(self.VS)

	@classmethod
	def from_mdl_file(self, mdl_filespec, units="kps"):
		"""
		Read from SITE_AMP MDL file

		:param mdl_filespec:
			str, full path to MDL file
		:param units:
			str, velocity units used in file, either "kps" (km/s)
			or "mps" (m/s). Depth units are assumed to be km and m,
			respectively.

		:return:
			instance of :class:`ElasticContinuousModel`
		"""
		Z, VS, Rho, QS = [], [], [], []
		fp = open(mdl_filespec)
		for line in fp:
			columns = line.split()
			try:
				z, vs, rho, qs = map(float, columns[:4])
			except:
				pass
			else:
				Z.append(z)
				VS.append(vs)
				Rho.append(rho)
				QS.append(qs)
		Z = np.array(Z)
		VS = np.array(VS)
		if units == "kps":
			Z *= 1000
			VS *= 1000
		Rho = np.array(Rho)
		QS = np.array(QS)
		if (QS < 1.).all():
			## 1/Q in file
			QS = 1./QS
		return ElasticContinuousModel(Z, VS, Rho, QS)

	def write_mdl(self, mdl_filespec):
		"""
		Write profile to SITE_AMP mdl input file.

		:param mdl_filespec:
			str, full path to MDL file
		"""
		if not os.path.splitext(mdl_filespec)[1]:
			mdl_filespec += ".mdl"
		f = open(mdl_filespec, "w")
		f.write("depth            vel       dens        1/Q\n")
		for i in range(len(self.Z)):
			z, vs, rho, qi = self.Z[i], self.VS[i], self.Rho[i], 1./self.QS[i]
			f.write("  %E  %E  %E  %E\n" % (z/1000., vs/1000., rho, qi))
		f.close()

	def get_densities_from_velocities(self):
		"""
		Compute densities based on shear-wave velocities
		Adapted from SITE_AMP by David Boore

		:return:
			1-D float array
		"""
		## Convert VS to km/s
		VS = self.VS / 1000.

		Rho = np.zeros_like(VS)

		Rho[VS < 0.3] = 1.93

		VS2 = VS[VS >= 0.3]
		Rho2 = Rho[VS >= 0.3]
		VP2 = 0.9409 + 2.0947*VS2 -0.8206*VS2**2 + 0.2683*VS2**3 - 0.0251*VS2**4
		Rho2[VS2 < 3.55] = 1.74 * VP2[VS2 < 3.55]**0.25
		VP3 = VP2[VS2 >= 3.55]
		Rho2[VS2 >= 3.55] = 1.6612*VP3 - 0.4721*VP3**2 + 0.0671*VP3**3 - 0.0043*VP3**4 + 0.000106*VP3**5
		Rho[VS >= 0.3] = Rho2

		return Rho

	def get_densities_from_gradient(self, rho_min, rho_max, vs_min=None, vs_max=None):
		"""
		Compute densities from density/velocity gradient

		:param rho_min:
			float, minimum density
		:param rho_max:
			float, maximum density
		:param vs_min:
			float, VS corresponding to rho_min
			(default: None, will use velocity at top of profile)
		:param vs_max:
			float, VS corresponding to rhox_max
			(default: None, will use velocity at bottom of profile)

		:return:
			1-D float array
		"""
		if vs_min is None:
			vs_min = self.VS[0]
		if vs_max is None:
			vs_max = self.VS[-1]
		drho_dvs = (rho_max - rho_min)/(vs_max - vs_min)
		Rho = rho_min + (self.VS - vs_min) * drho_dvs
		Rho[self.VS <= vs_min] = rho_min
		Rho[self.VS >= vs_max] = rho_max
		return Rho

	def fill_zero_densities(self, rho_min=None, rho_max=None, vs_min=None, vs_max=None):
		"""
		Replace zero densities with density inferred from velocity
		or gradient

		:param rho_min:
			float, minimum density (default: None)
		:param rho_max:
			float, maximum density (default: None)
		:param vs_min:
			float, VS corresponding to rho_min
			(default: None, will use velocity at top of profile)
		:param vs_max:
			float, VS corresponding to rhox_max
			(default: None, will use velocity at top of profile)

		If one of (rho_min, rho_max) is None, densities are computed
		using :meth:`get_densities_from_velocities`, else
		using :meth:`get_densities_from_gradient`
		"""
		if None in (rho_min, rho_max):
			inferred_densities = self.get_densities_from_velocities()
		else:
			inferred_densities = self.get_densities_from_gradient(rho_min, rho_max, vs_min, vs_max)
		idxs = np.where(self.Rho == 0)
		self.Rho[idxs] = inferred_densities[idxs]

	def get_VS_gradient(self):
		"""
		Compute velocity gradient (in downward direction)

		:return:
			1-D float array with length of profile minue one
		"""
		return np.diff(self.VS) / np.diff(self.Z)

	def get_Rho_gradient(self):
		"""
		Compute density gradient (in downward direction)

		:return:
			1-D float array with length of profile minue one
		"""
		return np.diff(self.Rho) / np.diff(self.Z)

	def get_QS_gradient(self):
		"""
		Compute QS gradient (in downward direction)

		:return:
			1-D float array with length of profile minue one
		"""
		return np.diff(self.QS) / np.diff(self.Z)

	def get_Qi_gradient(self):
		"""
		Compute gradient of 1/QS (in downward direction)
		This is not the same as 1/get_QS_gradient !

		:return:
			1-D float array with length of profile minue one
		"""
		return np.diff(1./self.QS) / np.diff(self.Z)

	def get_travel_times(self):
		"""
		Compute travel times for downgoing waves

		:return:
			1-D float array
		"""
		tt = np.zeros_like(self.Z)
		delta_Z = np.diff(self.Z)
		delta_VS = np.diff(self.VS)
		VS_gradient = delta_VS / delta_Z
		for i in range(1, len(self.Z)):
			grad = VS_gradient[i-1]
			if grad == 0:
				## Constant-velocity layer
				tt[i] = tt[i-1] + delta_Z[i-1] / self.VS[i-1]
			elif np.isinf(grad) or np.isnan(grad):
				## Velocity step
				tt[i] = tt[i-1]
			else:
				## Velocity gradient
				tt[i] = tt[i-1] + (1.0 / grad) * np.log(self.VS[i] / self.VS[i-1])
		return tt

	def get_min_freq(self):
		"""
		Compute minimum frequency for the model, this is the frequency
		below which the halfspace affects the average velocity

		:return:
			float, minimum frequency (in seconds)
		"""
		tt = self.get_travel_times()
		return 1.0 / (4.0 * tt[-1])

	def get_downgoing_average_velocity(self):
		"""
		Compute cumulative average velocity (in downward direction)

		:return:
			1-D float array
		"""
		tt = self.get_travel_times()
		return self.Z / tt

	def get_downgoing_average_density(self):
		"""
		Compute cumulative average density (in downward direction)

		:return:
			1-D float array
		"""
		avg_rho = np.zeros_like(self.Z)
		avg_rho[0] = self.Rho[0]
		delta_Z = np.diff(self.Z)
		delta_rho = np.diff(self.Rho)
		rho_gradient = delta_rho / delta_Z
		delta_VS = np.diff(self.VS)
		VS_gradient = delta_VS / delta_Z
		for i in range(1, len(self.Z)):
			grad = rho_gradient[i-1]
			if grad == 0 or VS_gradient[i-1] == 0:
				## Constant density or constant VS (for compatibility with SITE_AMP)
				avg_rho[i] = (self.Z[i-1] * avg_rho[i-1] + delta_Z[i-1] * self.Rho[i-1]) / self.Z[i]
			elif np.isinf(grad) or np.isnan(grad):
				## Density step
				avg_rho[i] = avg_rho[i-1]
			else:
				## Density gradient
				avg_rho[i] = (self.Z[i-1] * avg_rho[i-1] + delta_Z[i-1] * self.Rho[i-1] + 0.5 * grad * delta_Z[i-1]**2) / self.Z[i]
		return avg_rho

	def get_downgoing_cumulative_kappa(self):
		"""
		Compute cumulative kappa (in downward direction)

		:return:
			1-D float array
		"""
		cum_kappa = np.zeros_like(self.Z)
		Qi = 1./self.QS	## Convert to 1/Q
		delta_Z = np.diff(self.Z)
		delta_Qi = np.diff(Qi)
		Qi_gradient = delta_Qi / delta_Z
		delta_VS = np.diff(self.VS)
		VS_gradient = delta_VS / delta_Z
		for i in range(1, len(self.Z)):
			## Note: consider different cases based on VS_gradient instead of Qi_gradient
			## for compatibility with SITE_AMP. Not sure if this is entirely correct, however
			grad = Qi_gradient[i-1]
			#if grad == 0 or VS_gradient[i-1] == 0:
			if VS_gradient[i-1] == 0:
				## Constant QS/VS
				cum_kappa[i] = cum_kappa[i-1] + delta_Z[i-1] * Qi[i-1] / self.VS[i-1]
			#elif np.isinf(grad) or np.isnan(grad):
			elif np.isinf(VS_gradient[i-1]):
				## QS step
				cum_kappa[i] = cum_kappa[i-1]
			else:
				## QS/VS gradient
				## Note: fails if VS_gradient[i-1] is zero
				cum_kappa[i] = (cum_kappa[i-1]
					+ delta_Z[i-1] * grad / VS_gradient[i-1]
					+ (1.0 / VS_gradient[i-1]**2) * (Qi[i-1] * VS_gradient[i-1] - self.VS[i-1] * grad) * np.log(self.VS[i]/self.VS[i-1]))
		return cum_kappa

	def add_halfspace(self, vel_source, rho_source):
		"""
		Add interface corresponding to top of halfspace

		:param vel_source:
			float, source shear-wave velocity (in m/s)
		:param rho_source:
			float, density near source (in kg / cubic dm)
		"""
		self.Z = np.concatenate([self.Z, self.Z[-1:]])
		self.VS = np.concatenate([self.VS, [vel_source]])
		self.Rho = np.concatenate([self.Rho, [rho_source]])
		self.QS = np.concatenate([self.QS, [np.inf]])

	def remove_halfspace(self):
		"""
		Remove bottom interface corresponding to halfspace
		"""
		self.Z = self.Z[:-1]
		self.VS = self.VS[:-1]
		self.Rho = self.Rho[:-1]
		self.QS = self.QS[:-1]

	def site_amp(self, freqs, vel_source=3500., rho_source=2.8, kappa=0., density_coeffs=[], aoi=0, out_filespec=""):
		"""
		Compute site amplification using quarter-wavelength method.
		This is an implementation of David Boore's SITE_AMP program.

		:param freqs:
			1-D float array, frequencies for which to compute amplification
			If None, frequencies will be used corresponding to breakpoints
			in the velocity profile.
		:param vel_source:
			float, source shear-wave velocity (in m/s)
			(default: 3500 m/s)
		:param rho_source:
			float, density near source (in kg / cubic dm)
			(default: 2.8 kg / cubic dm)
		:param density_coeffs:
			(rho_min, rho_max) or (rho_min, rho_max, vs_min, vs_max) tuple
			defining density gradient that will be used to fill zero
			density values. If empty, zero density values will be
			computed from the shear_wave velocity
			(default: [])
		:param kappa:
			float, kappa for amplitude computation (in seconds),
			will be added to the cumulative kappa through the sediment
			column determined from QS
			(default: 0.)
		:param aoi:
			float, angle of incidence (in degrees)
			(default: 0)
		:param out_filespec:
			str, full path to output file
			(default: "", will not write output)

		:return:
			tuple containing
			- instance of :class:`TransferFunction`
			- instance of :class:`ElasticLayerModel`
		"""
		## Ignore division warnings
		np.seterr(divide='ignore', invalid='ignore')

		## Fill zero densities first
		rho_min, rho_max, vs_min, vs_max = None, None, None, None
		density_coeffs_specified = False
		if density_coeffs != None and len(density_coeffs) >= 2:
			density_coeffs_specified = True
			rho_min, rho_max = density_coeffs[:2]
			if len(density_coeffs) == 4:
				vs_min, vs_max = map(float, density_coeffs[2:])
		self.fill_zero_densities(rho_min, rho_max, vs_min, vs_max)

		## Add interface above halfspace
		vel_source, rho_source = float(vel_source), float(rho_source)
		self.add_halfspace(vel_source, rho_source)

		## Issue warning when density or QS change while VS remains constant
		delta_VS = np.diff(self.VS)
		delta_Rho = np.diff(self.Rho)
		delta_Qi = np.diff(1./self.QS)
		zero_idxs = np.where(delta_VS == 0)
		non_zero_rho_idxs = np.where(delta_Rho[zero_idxs] != 0)[0]
		if len(non_zero_rho_idxs):
			print("Warning: model contains density gradients where VS remains constant!")
		non_zero_qi_idxs = np.where(delta_Qi[zero_idxs] != 0)[0]
		if len(non_zero_qi_idxs):
			print("Warning: model contains QS gradients where VS remains constant!")

		tt = self.get_travel_times()
		fmin4model = 1.0 / (4.0 * tt[-1])
		avg_rho = self.get_downgoing_average_density()
		cum_kappa = self.get_downgoing_cumulative_kappa()

		## Ray parameter
		rp = np.sin(np.radians(aoi)) / vel_source
		imp_source = vel_source * rho_source

		Z, VS, Rho, Qi = self.Z, self.VS, self.Rho, 1./self.QS

		if freqs != None and len(freqs) != 0:
			## Frequencies specified
			delta_Z = np.diff(Z)
			VS_gradient = delta_VS / delta_Z
			Rho_gradient = delta_Rho / delta_Z
			Qi_gradient = delta_Qi / delta_Z
			ndepths = len(Z)

			tt_spcfy = np.zeros(len(freqs)+1, dtype='d')
			z_spcfy = np.zeros(len(freqs)+1, dtype='d')
			avg_vs_spcfy = np.zeros(len(freqs)+1, dtype='d')
			avg_rho_spcfy = np.zeros(len(freqs)+1, dtype='d')
			cum_kappa_spcfy = np.zeros(len(freqs)+1, dtype='d')
			avg_aoi_spcfy = np.zeros(len(freqs)+1, dtype='d')

			ampli_spcfy = np.zeros(len(freqs), dtype='d')
			amplik_spcfy = np.zeros(len(freqs), dtype='d')

			## Frequencies must be in descending order
			freqs = np.array(freqs, 'd')
			reversed_freqs = False
			if freqs[1] > freqs[0]:
				freqs = freqs[::-1]
				reversed_freqs = True
			for k in range(1, len(freqs) + 1):
				## Note extra zero element at k=0 for *_spcfy arrays (except ampli(k)_spcfy)
				freq = freqs[k-1]
				## Compute tt corresponding to freq_spcfy
				tt_spcfy[k] = 1.0 / (4.0 * freq)
				## Find location of layer above the one in which tt_spcfy occurs
				[j] = np.digitize([tt_spcfy[k]], tt) - 1

				## Compute the depth corresponding to tt_spcfy
				delta_tt = tt_spcfy[k] - tt[j]
				if j == (ndepths - 1):
					## Into the halfspace
					delta_z = vel_source * delta_tt
				elif VS_gradient[j] == 0:
					## Constant velocity
					delta_z = VS[j] * delta_tt
				else:
					delta_z = VS[j] * (np.exp(VS_gradient[j] * delta_tt) - 1.0) / VS_gradient[j]
				z_spcfy[k] = Z[j] + delta_z

				## Compute average velocity
				avg_vs_spcfy[k] = z_spcfy[k] / tt_spcfy[k]

				## Compute the average density and kappa
				if j == (ndepths - 1) or VS_gradient[j] == 0:
					## Constant-velocity layer
					avg_rho_spcfy[k] = (Z[j] * avg_rho[j] + delta_z * Rho[j]) / z_spcfy[k]
					cum_kappa_spcfy[k] = cum_kappa[j] + delta_z * Qi[j] / VS[j]
				else:
					## Velocity gradient
					avg_rho_spcfy[k] = (Z[j] * avg_rho[j] + delta_z * Rho[j] + 0.5 * Rho_gradient[j] * delta_z**2) / z_spcfy[k]
					cum_kappa_spcfy[k] = (cum_kappa[j]
										+ delta_z * Qi_gradient[j] / VS_gradient[j]
										+ (1. / VS_gradient[j]**2) * (Qi[j] * VS_gradient[j] - VS[j] * Qi_gradient[j])
										* np.log((VS[j] + VS_gradient[j] * delta_z) / VS[j]))


				avg_aoi_spcfy[k] = np.arccos(np.sqrt(1.0 - (rp * avg_vs_spcfy[k])**2))
				ampli_spcfy[k-1] = np.sqrt(imp_source / (avg_vs_spcfy[k] * avg_rho_spcfy[k]))
				ampli_spcfy[k-1] *= np.sqrt(np.cos(np.radians(aoi))) / np.cos(avg_aoi_spcfy[k])
				avg_aoi_spcfy[k] = np.degrees(avg_aoi_spcfy[k])

			## Construct layer model
			Th_layer = np.diff(z_spcfy)
			VS_layer = Th_layer / np.diff(tt_spcfy)
			Rho_layer = np.diff(z_spcfy * avg_rho_spcfy) / Th_layer
			Qi_layer = np.diff(cum_kappa_spcfy) / np.diff(tt_spcfy)

			## Correct amplifcation for kappa
			akappa4amp = cum_kappa_spcfy[-1] + kappa
			amplik_spcfy = ampli_spcfy * (np.exp(-np.pi * akappa4amp * freqs))

			if out_filespec:
				of = open(out_filespec, "w")
				of.write("   model file:\n")
				of.write(" source vel & dens =  %10.3E %10.3E angle-of-incidence at source level = %5.1f\n" % (vel_source/1000, rho_source, aoi))
				if density_coeffs_specified:
					of.write("  Density coefficients specified; dens, vel, low, high =  %10.3E %10.3E %10.3E %10.3E\n" % (rho_min, vs_min/1000, rho_max, vs_max/1000))
				else:
					of.write("  Density coeffs not specified\n")
				of.write(" akappa_fixed = %10.3E\n" % kappa)
				of.write("   ndepths = %4d  nfreq_spcfy = %4d   frequency below which halfspace affects avgvel = %10.3E\n" % (ndepths, len(freqs), fmin4model))
				of.write("  ofr_amp-d  ofr_amp-v  ofr_amp-r  ofr_ampqi  depth_out  travltime   thck_lyr  vel_layer  dens_layr    qi_layr     avgvel     avgaoi  ofr_amp-f  ofr_amp-a  ofr_amp-k  ofr_ampak\n")
				for i in range(max(ndepths-1, len(freqs))):
					if i < (ndepths - 1):
						of.write(" %10.3E %10.3E %10.3E %10.3E" % (Z[i]/1000, VS[i]/1000, Rho[i], Qi[i]))
					else:
						of.write("          .          .          .          .")
					of.write(" %10.3E %10.3E %10.3E %10.3E %10.3E %10.3E %10.3E %10.3E %10.3E %10.3E %10.3E %10.3E\n" % (z_spcfy[i+1]/1000, tt_spcfy[i+1], Th_layer[i]/1000, VS_layer[i]/1000, Rho_layer[i], Qi_layer[i], avg_vs_spcfy[i+1]/1000, avg_aoi_spcfy[i+1], freqs[i], ampli_spcfy[i], cum_kappa_spcfy[i+1], amplik_spcfy[i]))
				of.close()

			## Make sure frequencies are in same order as specified
			if reversed_freqs:
				freqs = freqs[::-1]
				amplik_spcfy = amplik_spcfy[::-1]

		else:
			## No frequencies specified
			avg_vs = Z / tt
			avg_aoi = np.arccos(np.sqrt(1.0 - (rp * avg_vs)**2)) # in radians

			## Skip repeated depths
			idxs = np.where(np.diff(Z) != 0)[0] + 1

			Th_layer = np.diff(Z[idxs])
			VS_layer = Th_layer / np.diff(tt[idxs])
			Rho_layer = np.diff(Z[idxs] * avg_rho[idxs]) / Th_layer
			Qi_layer = np.diff(cum_kappa[idxs]) / np.diff(tt[idxs])

			freqs = 1.0 / (4.0 * tt[idxs])
			ampli_spcfy = np.sqrt(imp_source / (avg_rho[idxs] * avg_vs[idxs]))
			ampli_spcfy *= np.sqrt(np.cos(np.radians(aoi))) / np.cos(avg_aoi[idxs])
			avg_aoi = np.degrees(avg_aoi)

			akappa4amp = cum_kappa[idxs[-1]] + kappa
			amplik_spcfy = ampli_spcfy * (np.exp(-np.pi * akappa4amp * freqs))

			if out_filespec:
				# TODO
				pass

		## Remove halfspace again so that site_amp does not cause side effects
		self.remove_halfspace()

		tf = TransferFunction(freqs, amplik_spcfy)
		layer_model = ElasticLayerModel(Th_layer, VS_layer, Rho_layer, 1./Qi_layer)

		return (tf, layer_model)


if __name__ == "__main__":
	import hazard.rshalib as rshalib

	## Compare site_amp with SITE_AMP
	site_amp_folder = r"C:\Geo\SITE_AMP"
	asc_filespec = os.path.join(site_amp_folder, "OFR_AMP.ASC")
	tf_boore = rshalib.siteresponse.read_TF_SITE_AMP(asc_filespec)
	mdl_filespec = os.path.join(site_amp_folder, "OFR_AMP.DAT")
	mdl = ElasticContinuousModel.from_mdl_file(mdl_filespec)
	freqs = tf_boore.freqs
	vel_source, rho_source =3000., 2.8
	kappa, aoi = 0., 0.
	density_coeffs = [2.5, 2.8, 300, 3500]
	tf, _ = mdl.site_amp(freqs, vel_source=vel_source, rho_source=rho_source, kappa=kappa, aoi=aoi, density_coeffs=density_coeffs)

	pylab.semilogx(freqs, tf_boore.magnitudes, "r", lw=3, label="SITE_AMP")
	pylab.semilogx(freqs, tf.magnitudes, "b", lw=1, label="site_amp")

	pylab.xlabel("Frequency (Hz)")
	pylab.ylabel("Amplification")
	pylab.legend(loc=2)
	pylab.grid(True)
	pylab.title("SITE_AMP benchmark: OFR_AMP example")
	pylab.show()
