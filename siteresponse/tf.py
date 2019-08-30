"""
1-D linear site response using Kennet reflectivity theory
"""

import numpy as np
import pylab
from matplotlib.font_manager import FontProperties


# TODO: implement convolution with accelerogram
# TODO: determine frequency of max peak(s)

class ComplexTransferFunction:
	"""
	Class representing complex transfer function

	:param freqs:
		list or array of frequencies
	:param data:
		list or array of complex data representing spectral transfer function
	"""
	def __init__(self, freqs, data):
		if len(freqs) != len(data):
			raise Exception("freqs and data should have same length!")
		self.freqs = np.asarray(freqs)
		self.data = np.asarray(data)

	def __len__(self):
		return len(self.freqs)

	@property
	def num_freqs(self):
		return len(self.freqs)

	@property
	def periods(self):
		return 1. / self.freqs

	@property
	def magnitudes(self):
		return np.abs(self.data)

	@property
	def phases(self):
		return np.angle(self.data)

	def real(self):
		return np.real(self.data)

	def imag(self):
		return np.imag(self.data)

	def to_transfer_function(self):
		"""
		Convert to simple transfer function

		:return:
			instance of :class:`TransferFunction`
		"""
		return TransferFunction(self.freqs, self.magnitudes)

	def export_csv(self, out_filespec):
		"""
		Export to csv file

		:param out_filespec:
			str, full path to output file
		"""
		of = open(out_filespec, "w")
		of.write("Freq, Real, Imag, Mag, Phase\n")
		TRr = self.real()
		TRi = self.imag()
		TR_amp = self.magnitudes
		TR_phase = self.phases
		for k in range(len(self)):
			of.write("%E, %E, %E, %E, %E\n" % (self.freqs[k], TRr[k], TRi[k], TR_amp[k], TR_phase[k]))
		of.close()

	def plot_magnitude(self, color='b', line_style='-', line_width=2, label="",
						want_freq=True):
		"""
		Plot magnitude

		:param color:
			matplotlib color specification (default: 'b')
		:param line_style:
			str, matplotlib line style (default: "-")
		:param line_width:
			float, line width (default: 2)
		:param label:
			str, plot label
		:param want_freq:
			bool, whether X axis should show frequencies (True) or periods
			(False)
			(default: True)
		"""
		plot_TF_magnitude([self], colors=[color], line_styles=[line_style],
				line_widths=[line_width], labels=[label], want_freq=want_freq)

	def get_response_spectrum(self, accelerogram):
		"""
		Compute response spectrum from an accelerogram
		"""
		# TODO
		## FT of accelerogram; multiply; iFT
		pass


class ComplexTransferFunctionSet(ComplexTransferFunction):
	"""
	Class representing set of complex transfer functions

	:param freqs:
		list or array of frequencies
	:param data:
		2-D [i,k] array of complex data representing spectral transfer functions
		i: represents index in set
		k: represents frequency index
	"""
	def __init__(self, freqs, data):
		self.freqs = np.asarray(freqs)
		self.data = np.asarray(data)

	def __len__(self):
		return int(self.data.shape[0])

	def __iter__(self):
		for i in range(len(self)):
			TF = ComplexTransferFunction(self.freqs, self.data[i])
			yield TF

	def __getitem__(self, key):
		if isinstance(key, (slice, list)):
			return ComplexTransferFunctionSet(self.freqs, self.data[key])
		elif isinstance(key, int):
			return ComplexTransferFunction(self.freqs, self.data[key])
		else:
			raise TypeError, "Invalid argument type."

	def mean(self):
		"""
		Return mean transfer function (amplitude only)

		:return:
			instance of :class:`TransferFunction`
		"""
		mean_amp = np.mean(self.magnitudes, axis=0)
		return TransferFunction(self.freqs, mean_amp)

	def median(self):
		"""
		Return median transfer function (amplitude only)

		:return:
			instance of :class:`TransferFunction`
		"""
		median_amp = np.median(self.magnitudes, axis=0)
		return TransferFunction(self.freqs, median_amp)

	def amp_std(self):
		"""
		Return standard deviation of amplitude
		"""
		return np.std(self.magnitudes, axis=0)

	def percentile(self, perc):
		"""
		Return percentile of transfer function

		:param perc:
			percentile in the range [0, 100]

		:return:
			instance of :class:`TransferFunction`
		"""
		import scipy.stats as stats
		perc_amp = stats.scoreatpercentile(self.magnitudes, per=perc, axis=0)
		return TransferFunction(self.freqs, perc_amp)

	def export_csv(self, out_filespec):
		"""
		Export to csv file

		:param out_filespec:
			str, full path to output file
		"""
		of = open(out_filespec, "w")
		of.write("Freq")
		for i in range(len(self)):
			of.write(", Real%03d, Imag%03d, Mag%03d, Phase%03d" % (i+1, i+1, i+1, i+1))
		of.write("\n")
		TRr = self.real()
		TRi = self.imag()
		TR_amp = self.magnitudes
		TR_phase = self.phases
		for k in range(self.num_freqs):
			of.write("%E" % self.freqs[k])
			for i in range(len(self)):
				of.write(", %E, %E, %E, %E" % (TRr[i,k], TRi[i,k], TR_amp[i,k], TR_phase[i,k]))
			of.write("\n")
		of.close()


class TransferFunction:
	def __init__(self, freqs, magnitudes):
		"""
		Class representing simple transfer function (amplitude spectrum only)

		:param freqs:
			list or array of frequencies
		:param data:
			list or array of magnitudes of spectral transfer function
		"""
		if len(freqs) != len(magnitudes):
			raise Exception("freqs and magnitudes should have same length!")
		self.freqs = np.asarray(freqs)
		self.magnitudes = np.asarray(magnitudes)

	def __len__(self):
		return len(self.freqs)

	@property
	def num_freqs(self):
		return len(self.freqs)

	@property
	def periods(self):
		return 1. / self.freqs

	def reconstruct_phase(self, wrap=True):
		"""
		Reconstruct phase assuming a minimum-phase system

		:param wrap:
			bool, indicating whether or not phases should be wrapped
			between -pi and +pi (default: True)

		:return:
			float array of phases in radians
			wrapped between -pi and pi
		"""
		import scipy.fftpack as fftpack
		phases = fftpack.hilbert(np.log(self.magnitudes))
		if wrap:
			phases = (phases + np.pi) % (2 * np.pi ) - np.pi
		return phases

	def to_complex_transfer_function(self):
		"""
		Convert to complex transfer function, assuming a minimum-phase system

		:return:
			instance of :class:`ComplexTransferFunction`
		"""
		phases = self.reconstruct_phase()
		reals = self.magnitudes * np.cos(phases)
		imags = self.magnitudes * np.sin(phases)
		#data = np.complex(reals, imags)
		data = reals + 1j * imags
		return ComplexTransferFunction(self.freqs, data)

	def interpolate(self, freqs):
		"""
		Interpolate transfer function at different frequencies

		:param freqs:
			float array, frequencies at which to interpolate tf

		:return:
			instance of :class:`TransferFunction`
		"""
		from ..utils import interpolate

		magnitudes = interpolate(self.freqs, self.magnitudes, freqs)
		return TransferFunction(freqs, magnitudes)

	def plot_magnitude(self, color='b', line_style='-', line_width=2, label=""):
		"""
		Plot magnitudes

		:param color:
			matplotlib color specification (default: 'b')
		:param line_style:
			str, matplotlib line style (default: "-")
		:param line_width:
			float, line width (default: 2)
		:param label:
			str, plot label
		"""
		plot_TF_magnitude([self], colors=[color], line_styles=[line_style], line_widths=[line_width], labels=[label])

	def export_csv(self, out_filespec):
		"""
		Export to csv file

		:param out_filespec:
			str, full path to output file
		"""
		of = open(out_filespec, "w")
		of.write("Freq, Mag\n")
		for k in range(self.num_freqs):
			of.write("%E, %E\n" % (self.freqs[k], self.magnitudes[k]))
		of.close()



def plot_TF_magnitude(TF_list, colors=[], line_styles=[], line_widths=[],
					labels=[], want_freq=True, ymax=None,
					title="", fig_filespec=None, dpi=300):
	"""
	Magnitude plot of transfer functions

	:param TF_list:
		list of instances of :class:`TransferFunction` or
		:class:`ComplexTransferFunction`
	:param colors:
		list of matplotlib line color specifications
	:param line_styles:
		list of matplotlib line style specifications
	:param line_widths:
		list of line widths
	:param labels:
		list of plot labels
	:param want_freq:
		bool, whether X axis should show frequencies (True) or periods
		(False)
		(default: True)
	:param title:
		str, figure title
		(default: "")
	:param fig_filespec:
		str, full path to output file. If None, figure will be plotted
		on screen
		(default: None)
	:param dpi:
		int, figure resolution in dots per inch
		(default: 300)
	"""
	if not labels:
		labels = ["Set %d" % (i+1) for i in range(len(TF_list))]

	if not colors:
		colors = ("r", "g", "b", "c", "m", "k")

	if not line_styles:
		line_styles = ["-"]

	if not line_widths:
		line_widths = [2]

	pylab.clf()
	# TODO: add more options, and implement empty lists
	for i in range(len(TF_list)):
		TF = TF_list[i]
		if want_freq:
			x_values = TF.freqs
		else:
			x_values = TF.periods
		label = labels[i]
		color = colors[i%len(colors)]
		linestyle = line_styles[i%len(line_styles)]
		linewidth = line_widths[i%len(line_widths)]
		pylab.semilogx(x_values, TF.magnitudes, color=color, linestyle=linestyle, linewidth=linewidth, label=label)

	if ymax is not None:
		pylab.ylim(ymax=ymax)

	if want_freq:
		pylab.xlabel("Frequency (Hz)", fontsize="x-large")
	else:
		pylab.xlabel("Period (s)", fontsize="x-large")
	pylab.ylabel("Spectral Amplification", fontsize="x-large")
	font = FontProperties(size='large')
	pylab.legend(loc="upper right", prop=font)
	pylab.grid(True, which="both")
	pylab.title(title)
	ax = pylab.gca()
	for label in ax.get_xticklabels() + ax.get_yticklabels():
		label.set_size('large')

	if fig_filespec:
		pylab.savefig(fig_filespec, dpi=dpi)
	else:
		pylab.show()


def read_TF_transfer1D(csv_filespec):
	"""
	Read transfer function(s) from csv file written by transfer1D.

	:param csv_filespec:
		str, full path to csv file containing tansfer function(s)

	:return:
		instance of :class:`ComplexTransferFunction` or
		:class:`ComplexTransferFunctionSet`
	"""
	freqs, complex_data = [], []
	fd = open(csv_filespec)
	for i, line in enumerate(fd):
		if i == 0:
			num_columns = len(line.split(','))
			num_TFs = (num_columns - 1) / 4
			for i in range(num_TFs):
				complex_data.append([])
		else:
			fields = line.split(',')
			freqs.append(float(fields[0]))
			for k in range(num_TFs):
				val = complex(float(fields[k*4+1]), float(fields[k*4+2]))
				complex_data[k].append(val)
	fd.close()

	freqs = np.array(freqs)
	complex_data = np.array(complex_data)

	if num_TFs == 1:
		return ComplexTransferFunction(freqs, complex_data[0])
	else:
		return ComplexTransferFunctionSet(freqs, complex_data)


def read_TF_EERA_csv(csv_filespec, column=1):
	"""
	Read transfer function from csv file written by EERA

	:param csv_filespec:
		str, full path to csv file containing tansfer function(s)
	:param column:
		int, index of column containing TF magnitude (default: 1)

	:return:
		instance of :class:`TransferFunction`
	"""
	fd = open(csv_filespec)
	freqs, data = [], []
	for i, line in enumerate(fd):
		if line[0] != '#' and i >= 1:
			fields = line.split(',')
			freqs.append(float(fields[0]))
			data.append(float(fields[column]))
	fd.close()
	return TransferFunction(freqs, data)


def read_TF_SITE_AMP(filespec):
	"""
	Read transfer function from ASC file written by SITE_AMP
	"""
	f = open(filespec)
	start_line = 6
	freqs, amps = [], []
	for i, line in enumerate(f):
		columns = line.split()
		if columns[0][-2:] == "-d":
			start_line = i + 1
		elif i >= start_line:
			freq, amp = float(columns[12]), float(columns[-1])
			freqs.append(freq)
			amps.append(amp)
	freqs = np.array(freqs)
	amps = np.array(amps)
	return TransferFunction(freqs, amps)



if __name__ == "__main__":
	pass