"""
ResponseSpectrum class
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import robspy

from ..utils import interpolate
from .hc_base import (HazardSpectrum, IntensityResult)



__all__ = ['ResponseSpectrum']


class ResponseSpectrum(HazardSpectrum, IntensityResult, robspy.ResponseSpectrum):
	"""
	Hazard response spectrum

	:param periods:
		1-D array, spectral periods (in s)
	:param intensities:
		ndarray, intensities (=ground-motion levels)
	:param intensity_unit:
		str, intensity unit
		If not specified, default intensity unit for given IMT will be used
	:param IMT:
		str, intensity measure type ('PGA', 'PGV', 'PGD', 'SA', 'SV', 'SD')
	:param damping:
		float, damping corresponding to response spectrum
		(expressed as fraction of critical damping)
		(default: 0.05)
	:param model_name:
		str, name of this model
		(default: "")
	"""
	def __init__(self, periods, intensities, intensity_unit, IMT,
				damping=0.05, model_name=""):
		## Fix position of PGA with respect to spectrum if necessary
		if periods[0] == 0 and periods[1] > periods[2]:
			print("Moving PGA to end of array")
			periods = np.roll(periods, -1)
			intensities = np.roll(intensities, -1)
		elif periods[-1] == 0 and periods[0] < periods[1]:
			print("Moving PGA to beginning of array")
			periods = np.roll(periods, 1)
			intensities = np.roll(intensities, 1)

		HazardSpectrum.__init__(self, periods)
		IntensityResult.__init__(self, intensities, intensity_unit, IMT)
		response_type = self._get_response_type(IMT)
		robspy.ResponseSpectrum.__init__(self, periods, intensities, intensity_unit,
										damping, response_type)

		self.model_name = model_name

		self._opt_kwargs = ['IMT', 'model_name']

	def __repr__(self):
		txt = '<ResponseSpectrum %s | T: %s - %s s (n=%d) | %d%% damping>'
		txt %= (self.IMT, self.Tmin, self.Tmax, len(self), self.damping*100)
		return txt

	def __add__(self, other):
		"""
		Sum response spectrum with another spectrum or a fixed value

		:param other:
			int, float or instance of :class:`ResponseSpectrum`
			If other response spectrum has different periods, interpolation
			will occur.

		:return:
			instance of :class:`ResponseSpectrum`
		"""
		if isinstance(other, ResponseSpectrum):
			if not np.array_equal(self.periods, other.periods):
				print("Warning: Periods incompatible, need to interpolate")
				other = other.interpolate_periods(self.periods)
			intensities = self.intensities + other.intensities
			model_name = "%s + %s" % (self.model_name, other.model_name)
		elif isinstance(other, (int, float)):
			intensities = self.intensities + other
			model_name = "%s + %s" % (self.model_name, other)
		else:
			raise TypeError

		return ResponseSpectrum(self.periods, intensities, self.intensity_unit,
								IMT=self.IMT, damping=self.damping,
								model_name=model_name)

	def __sub__(self, other):
		"""
		Subtract response spectrum or fixed value from current spectrum

		:param other:
			int, float or instance of :class:`ResponseSpectrum`
			If other response spectrum has different periods, interpolation
			will occur.

		:return:
			instance of :class:`ResponseSpectrum`
		"""
		if isinstance(other, ResponseSpectrum):
			if not np.array_equal(self.periods, other.periods):
				print("Warning: Periods incompatible, need to interpolate")
				other = other.interpolate_periods(self.periods)
			intensities = self.intensities - other.intensities
			model_name = "%s - %s" % (self.model_name, other.model_name)
		elif isinstance(other, (int, float)):
			intensities = self.intensities - other
			model_name = "%s - %s" % (self.model_name, other)
		else:
			raise TypeError

		return ResponseSpectrum(self.periods, intensities, self.intensity_unit,
								IMT=self.IMT, damping=self.damping,
								model_name=model_name)

	def __mul__(self, other):
		"""
		Multiply response spectrum with another spectrum or a fixed value

		:param other:
			int, float or instance of :class:`ResponseSpectrum`
			If other response spectrum has different periods, interpolation
			will occur.

		:return:
			instance of :class:`ResponseSpectrum`
		"""
		if isinstance(other, ResponseSpectrum):
			if not np.array_equal(self.periods, other.periods):
				print("Warning: Periods incompatible, need to interpolate")
				other = other.interpolate_periods(self.periods)
			intensities = self.intensities * other.intensities
			model_name = "%s * %s" % (self.model_name, other.model_name)
		elif isinstance(other, (int, float)):
			intensities = self.intensities * other
			model_name = "%s * %s" % (self.model_name, other)
		else:
			raise TypeError

		return ResponseSpectrum(self.periods, intensities, self.intensity_unit,
								IMT=self.IMT, damping=self.damping,
								model_name=model_name)

	def __div__(self, other):
		"""
		Divide response spectrum by another spectrum or a fixed value

		:param other:
			int, float or instance of :class:`ResponseSpectrum`
			If other response spectrum has different periods, interpolation
			will occur.

		:return:
			instance of :class:`ResponseSpectrum`
		"""
		if isinstance(other, ResponseSpectrum):
			if not np.array_equal(self.periods, other.periods):
				print("Warning: Periods incompatible, need to interpolate")
				other = other.interpolate_periods(self.periods)
			intensities = self.intensities / other.intensities
			model_name = "%s / %s" % (self.model_name, other.model_name)
		elif isinstance(other, (int, float)):
			intensities = self.intensities / other
			model_name = "%s / %s" % (self.model_name, other)
		else:
			raise TypeError

		return ResponseSpectrum(self.periods, intensities, self.intensity_unit,
								IMT=self.IMT, damping=self.damping,
								model_name=model_name)

	@staticmethod
	def _get_response_type(IMT):
		"""
		Determine response type (property of base class in robspy)
		from IMT

		:param IMT:
			str, intensity measure type ("SA", "SV", "SD")

		:return:
			str, response type
		"""
		response_type = {'SA': 'psa', 'SV': 'psd', 'SD': 'sd'}.get(IMT, IMT.lower())
		return response_type

	interpolate_periods = robspy.ResponseSpectrum.interpolate

	to_srs = robspy.ResponseSpectrum.apply_tf_irvt

#	@property
#	def pgm(self):
#		"""
#		Peak ground motion
#		"""
#		if 0 in self.periods:
#			idx = list(self.periods).index(0)
#			return self.intensities[idx]

#	def interpolate_periods(self, out_periods):
#		"""
#		Interpolate response spectrum at different periods

#		:param out_periods:
#			list or 1-D array: periods of output response spectrum

#		:return:
#			instance of :class:`ResponseSpectrum`
#		"""
#		intensities = interpolate(self.periods, self.intensities, out_periods)
#		model_name = "%s (interpolated)" % self.model_name
#		return ResponseSpectrum(model_name, out_periods, self.IMT, intensities,
#								self.intensity_unit)

#	def plot(self, color="k", linestyle="-", linewidth=2, fig_filespec=None,
#			title=None, plot_freq=False, plot_style="loglin", Tmin=None, Tmax=None,
#			amin=None, amax=None, intensity_unit="g", pgm_period=0.02,
#			axis_label_size='x-large', tick_label_size='large',
#			legend_label_size='large', legend_location=0, lang="en", ax=None):
#		if title is None:
#			title = "Response Spectrum"
#		intensities = self.get_intensities(intensity_unit)
#		## Plot PGM separately if present
#		if 0 in self.periods:
#			pgm = [intensities[self.periods==0]]
#			datasets = [(self.periods[self.periods>0], intensities[self.periods>0])]
#		else:
#			pgm = None
#			datasets = [(self.periods, intensities)]
#		labels = [self.model_name]
#		plot_hazard_spectrum(datasets, pgm=pgm, pgm_period=pgm_period, labels=labels,
#						colors=[color], linestyles=[linestyle], linewidths=[linewidth],
#						fig_filespec=fig_filespec, title=title, plot_freq=plot_freq,
#						plot_style=plot_style, Tmin=Tmin, Tmax=Tmax, amin=amin,
#						amax=amax, intensity_unit=intensity_unit,
#						axis_label_size=axis_label_size, tick_label_size=tick_label_size,
#						legend_label_size=legend_label_size, legend_location=legend_location,
#						lang=lang, ax=ax)

#	def get_fas_irvt(self, pgm_freq=50. , mag=6.0, distance=10, region="ENA"):
#		"""
#		Obtain "matching" Fourier Amplitude Spectrum using Inverse Random Vibration Theory

#		:param pgm_freq:
#			float, frequency (in Hz) at which to consider PGM (zero period)
#			(default: 50.)
#		:param mag:
#			float, earthquake magnitude (default: 6.0)
#		:param distance:
#			float, distance in km (default: 10)
#		:param region:
#			str, region, either "ENA" or "WNA"

#		:return:
#			instance of :class:`pyrvt.motions.CompatibleRvtMotion`
#		"""
#		import pyrvt

#		freqs = 1./self.periods
#		freqs[self.periods == 0] = pgm_freq
#		irvt = pyrvt.motions.CompatibleRvtMotion(freqs, self.intensities, magnitude=mag, distance=distance, region=region)
#		return irvt

#	def to_fas(self, pgm_freq=50. , mag=6.0, distance=10, region="ENA"):
#		"""
#		Convert to Fourier Amplitude Spectrum based on inverse RVT

#		:param pgm_freq:
#		:param mag:
#		:param distance:
#		:param region:
#			see :meth:`get_fas_irvt`

#		:return:
#			instance of :class:`ResponseSpectrum`
#		"""
#		irvt = self.get_fas_irvt(pgm_freq=pgm_freq, mag=mag, distance=distance, region=region)

#		model_name = self.model_name + " (FAS)"
#		periods = 1./irvt.freqs
#		periods[irvt.freqs == pgm_freq] = 0
#		amps = irvt.fourier_amps
#		return ResponseSpectrum(model_name, periods, self.IMT, amps,
#								self.intensity_unit)

#	def to_srs(self, tf, pgm_freq=50., mag=6.0, distance=10, region="ENA"):
#		"""
#		Convert UHS to surface response spectrum with a transfer function
#		and using Inverse Random Vibration Theory.
#		There is a slight dependency on magnitude and distance considered
#		for the conversion of response spectrum to Fourier amplitude spectrum
#		using IRVT.
#		The resulting response spectrum will be clipped to the frequency
#		range of the transfer function.

#		:param tf:
#			instance of :class:`rshalib.siteresponse.TransferFunction` or
#			:class:`rshalib.siteresponse.ComplexTransferFunction`
#		:param pgm_freq:
#			float, frequency (in Hz) at which to consider PGM (zero period)
#			(default: 50.)
#		:param mag:
#			float, earthquake magnitude (default: 6.0)
#		:param distance:
#			float, distance in km (default: 10)
#		:param region:
#			str, region, either "ENA" or "WNA"
#			(default: "ENA")

#		:return:
#			instance of :class:`UHS`
#		"""
#		import pyrvt
#		from ..siteresponse import ComplexTransferFunction

#		irvt = self.get_fas_irvt(pgm_freq=pgm_freq, mag=mag, distance=distance, region=region)
#		#print(irvt.freqs.min(), irvt.freqs.max())
#		#print(tf.freqs.min(), tf.freqs.max())
#		if isinstance(tf, ComplexTransferFunction):
#			tf = tf.to_transfer_function()
#		tf2 = tf.interpolate(irvt.freqs)
#		irvt.fourier_amps *= tf2.magnitudes
#		rvt = pyrvt.motions.RvtMotion(irvt.freqs, irvt.fourier_amps, irvt.duration)

#		sa_periods = self.periods[self.periods > 0]
#		sa_freqs = 1./ sa_periods
#		out_freqs = tf.freqs[(tf.freqs >= sa_freqs.min()) & (tf.freqs <= sa_freqs.max())]
#		out_freqs = np.concatenate([out_freqs, [pgm_freq]])
#		out_periods = 1./out_freqs
#		out_periods[out_freqs == pgm_freq] = 0

#		srs_motion = rvt.compute_osc_resp(out_freqs)

#		model_name = self.model_name + " (SRS)"
#		#return UHS(model_name, "", self.site, out_periods, self.IMT, srs_motion, self.intensity_unit, self.timespan, poe=self.poe, return_period=self.return_period)
#		return ResponseSpectrum(model_name, out_periods, self.IMT, srs_motion,
#								self.intensity_unit)

#	def export_csv(self, csv_filespec=None, format="%.5E"):
#		"""
#		Export to csv (comma-separated values) file

#		:param csv_filespec:
#			str, full path to output file
#			(default: None, will output to screen)
#		"""
#		if csv_filespec:
#			f = open(csv_filespec, "w")
#		else:
#			f = sys.stdout
#		f.write("Period (s), %s (%s)\n" % (self.IMT, self.intensity_unit))
#		for period, intensity in zip(self.periods, self.intensities):
#			f.write(("%s, %s\n" % (format, format)) % (period, intensity))
#		f.close()

#	@classmethod
#	def from_csv(cls, csv_filespec, col_spec=1, intensity_unit="g", model_name=""):
#		"""
#		Read response spectrum from a csv file.
#		First line should contain column names
#		First column should contain periods or frequencies,
#		subsequent column()s should contain intensities, only one of which
#		will be read.

#		:param csv_filespec:
#			str, full path to csv file
#		:param col_spec:
#			str or int, name or index of column containing intensities to be read
#			(default: 1)
#		:param intensity_unit:
#			str, unit of intensities in csv file (default: "g")
#		:param model_name:
#			str, name or description of model

#		:return:
#			instance of :class:`ResponseSpectrum`
#		"""
#		periods, intensities = [], []
#		csv = open(csv_filespec)
#		for i, line in enumerate(csv):
#			if i == 0:
#				col_names = [s.strip() for s in line.split(',')]
#				if col_names[0].lower().split()[0] in ("frequency", "freq"):
#					freqs = True
#				else:
#					freqs = False
#				if isinstance(col_spec, basestring):
#					col_index = col_names.index(col_spec)
#				else:
#					col_index = col_spec
#			else:
#				col_values = line.split(',')
#				T = float(col_values[0])
#				a = float(col_values[col_index])
#				periods.append(T)
#				intensities.append(a)
#		csv.close()
#		periods = np.array(periods)
#		if freqs:
#			periods = 1./periods
#		intensities = np.array(intensities)

#		if not model_name:
#			model_name = os.path.splitext(os.path.split(csv_filespec)[-1])[0]

#		if intensity_unit in ("g", "mg", "ms2", "cms2", "gal"):
#			imt = "SA"
#		elif intensity_unit in ("ms", "cms"):
#			imt = "SV"
#		elif intensity_unit in ("m", "cm"):
#			imt = "SD"
#		else:
#			imt = ""

#		return cls(model_name, periods, imt, intensities, intensity_unit)

	def get_vertical_spectrum(self, guidance="RG1.60"):
		"""
		Derive vertical response spectrum, assuming rs is horizontal
		acceleration

		:param guidance:
			str, guidance to follow
			(one of "RG1.60", "ASCE4-98", "EC8_TYPE1", "EC8_TYPE2")
			(default: "RG1.60")

		:return:
			instance of :class:`ResponseSpectrum`
		"""
		if guidance.upper() == "RG1.60":
			f1, f2 = 0.25, 3.5
			T1, T2 = 1./f1, 1./f2
			cnv_factor = np.ones_like(self.intensities)
			cnv_factor[self.periods > T1] = 2./3
			idxs = np.where((self.periods >= T2) & (self.periods <= T1))
			freqs = 1./self.periods[idxs]
			cnv_factor[idxs] = interpolate(np.log([f1, f2]), [2./3, 1], np.log(freqs))
		elif guidance.upper() == "ASCE4-98":
			cnv_factor = np.ones_like(self.intensities) * 2./3
		elif guidance.upper()[:-1] == "EC8_TYPE":
			from ..siteresponse import get_refspec_EC8
			resp_type = int(guidance[-1])
			hrefspec = get_refspec_EC8(1., "A", resp_type=resp_type,
										orientation="horizontal")
			vrefspec = get_refspec_EC8(1., "A", resp_type=resp_type,
										orientation="vertical")
			vh_ratio_ec8 = vrefspec / hrefspec
			cnv_factor = interpolate(np.log(hrefspec.frequencies),
									vh_ratio_ec8.intensities,
									np.log(self.frequencies))
		else:
			raise NotImplementedError("Guidance %s not implemented" % guidance)

		intensities = self.intensities * cnv_factor
		model_name = self.model_name + " (V)"
		return self.__class__(self.periods, intensities, self.intensity_unit,
							IMT=self.IMT, damping=self.damping, model_name=model_name)

	def get_piecewise_linear_envelope(self, corner_freqs=[0.25, 2.5, 9, 33],
										num_iterations=100):
		"""
		Compute best-fitting piecewise linear envelope in the frequency
		domain

		:param corner_freqs:
			list or array of floats, corner frequencies (in Hz)
			(default: [0.25, 2.5, 9, 33], frequencies specified in RG1.60)
		:param num_iterations:
			int, number of iterations for fitting algorithm
			If set to 1, result will be best piecewise linear fit rather
			than envelope
			(default: 100)

		:return:
			instance of :class:`ResponseSpectrum`
		"""
		from scipy.optimize import curve_fit

		assert num_iterations > 0

		## First compute interpolated spectrum that includes corner freqs
		corner_freqs = np.asarray(corner_freqs)
		x = np.sort(np.unique((list(self.frequencies) + list(corner_freqs))))
		idxs = np.where((x >= min(corner_freqs)) & (x <= max(corner_freqs)))
		x = x[idxs]
		int_spec = self.interpolate_periods(1./x)
		freqs = int_spec.frequencies
		intensities = int_spec.intensities

		## Use interpolated values at corner frequencies as initial guess
		## for curve-fitting algorithm ...
		corner_spec = self.interpolate_periods(1./corner_freqs)
		## ... and convert to logarithmic domain
		Yc0 = np.log(corner_spec.intensities)
		Xc = np.log(corner_freqs)

		def piecewise_linear(x, *y):
			"""
			Piecewise linear function for fixed x values (corner frequencies)

			:param x:
				numpy array, X values
			:param y:
				numpy array, Y values at corner frequencies

			:return:
				numpy array, predicted values corresponding to :param:`x`
			"""
			Yc = np.array(y)
			yfit = np.zeros_like(x)
			## Note: Explicit loop because np.piecewise didn't seem to work
			for i in range(len(Xc) - 1):
				## Note: Xc defined outside of this function
				xmin, xmax = Xc[i], Xc[i+1]
				condition = (xmin <= x) & (x <= xmax)

				## Best-fitting line
				bfit = (Yc[i+1] - Yc[i]) / (Xc[i+1] - Xc[i])
				yfit[condition] = Yc[i] + bfit * (x[condition] - Xc[i])

				## Envelope, doesn't work inside fitting function
				"""
				yreal = np.log(intensities[condition])
				afit = Yc[i] - bfit * Xc[i]
				aenv = np.max(yreal - bfit * x[condition])
				da = aenv - afit
				yfit[condition] = afit - da + bfit * x[condition]
				"""
			return yfit

		x, y = np.log(freqs), np.log(intensities)

		for l in range(num_iterations):
			## Find best-fitting piecewise linear function
			popt, pcov = curve_fit(piecewise_linear, x, y, p0=Yc0)
			envelope = piecewise_linear(x, *popt)

			# Set Y values which are lower than envelope to predicted Y
			idxs = y < envelope
			y[idxs] = envelope[idxs]

		#corner_envelope = piecewise_linear(Xc, *popt)
		corner_envelope = popt

		model_name = self.model_name + " (envelope)"
		periods = 1. / np.array(corner_freqs)
		intensities = np.exp(corner_envelope)
		return self.__class__(periods, intensities, self.intensity_unit,
							IMT=self.IMT, damping=self.damping, model_name=model_name)

	def get_damped_spectrum(self, damping_ratio):
		"""
		Compute response spectrum for different damping ratio following
		RG1.60, and assuming current spectrum corresponds to 5% damping

		:param damping_ratio:
			float, damping ratio in percent, one of 0.5, 2, 7 or 10

		:return:
			instance of :class:`ResponseSpectrum`
		"""
		assert self.damping == 0.05

		corner_freqs = [33, 9, 2.5, 0.25]

		if damping_ratio == 0.5:
			conv_coeffs = np.array([1.00, 1.90, 1.90, 1.56])
		elif damping_ratio == 2:
			conv_coeffs = np.array([1.00, 1.36, 1.36, 1.22])
		elif damping_ratio == 7:
			conv_coeffs = np.array([1.00, 0.87, 0.87, 0.92])
		elif damping_ratio == 10:
			conv_coeffs = np.array([1.00, 0.73, 0.73, 0.83])
		## Note: PGA implicitly taken into account

		conv_factor = interpolate(np.log(corner_freqs), conv_coeffs,
									np.log(self.frequencies))

		model_name = self.model_name + " (%.1f %% damping)" % damping_ratio
		intensities = self.intensities * conv_factor
		damping = damping_ratio / 100.
		return self.__class__(self.periods, intensities, self.intensity_unit,
							IMT=self.IMT, damping=damping, model_name=model_name)

	def scale_to_pgm(self, target_pgm):
		"""
		Scale response spectrum to different target peak ground motion

		:param target_pgm:
			float, target PGM (in same units as response spectrum)

		:return:
			instance of :class:`ResponseSpectrum`
		"""
		current_pgm = self.pgm
		if current_pgm:
			#target_pga = self._convert_intensities(target_pga, "g", self.intensity_unit)
			intensities = self.intensities * target_pgm / current_pgm
			model_name = self.model_name + " (scaled to PGM=%s)" % target_pgm
			return self.__class__(self.periods, intensities, self.intensity_unit,
							IMT=self.IMT, damping=self.damping, model_name=model_name)
		else:
			print("Warning: response spectrum doesn't contain PGM!")
