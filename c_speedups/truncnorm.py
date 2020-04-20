"""
Wrapper functions mimicking behavior of scipy.stats.truncnorm methods
"""

from .truncated_normal import truncated_normal as truncnorm
#import ctypes
import numpy as np



def pdf(x, a, b, loc=0., scale=1.):
	a = loc + a * scale
	b = loc + b * scale
	if isinstance(x, (int, float)):
		return truncnorm.truncated_normal_ab_pdf(x, loc, scale, a, b)
	elif isinstance(x, np.ndarray):
		len = int(x.nbytes / x.itemsize)
		pdf = np.zeros(len)
		for i, xi in enumerate(x.astype('d').flat):
			pdf[i] = truncnorm.truncated_normal_ab_pdf(xi, loc, scale, a, b)
		#pdf = np.frombuffer(truncnorm.truncated_normal_ab_pdf_array(x.flatten().astype('d').data, len, loc, scale, a, b))
		pdf = pdf.reshape(x.shape)
		return pdf
	else:
		raise Exception("%s format not supported for x argument!" % type(x))

def cdf(x, a, b, loc=0., scale=1.):
	a = loc + a * scale
	b = loc + b * scale
	if isinstance(x, (int, float)):
		return truncnorm.truncated_normal_ab_cdf(x, loc, scale, a, b)
	elif isinstance(x, np.ndarray):
		len = int(x.nbytes / x.itemsize)
		cdf = np.zeros(len)
		for i, xi in enumerate(x.astype('d').flat):
			cdf[i] = truncnorm.truncated_normal_ab_cdf(xi, loc, scale, a, b)
		#cdf = np.frombuffer(truncnorm.truncated_normal_ab_cdf_array(x.flatten().astype('d').ctypes.data, len, loc, scale, a, b))
		cdf = cdf.reshape(x.shape)
		return cdf
	else:
		raise Exception("%s format not supported for x argument!" % type(x))

def sf(x, a, b, loc=0., scale=1.):
	return 1 - cdf(x, a, b, loc=loc, scale=scale)

def ppf(cdf, a, b, loc=0., scale=1.):
	a = loc + a * scale
	b = loc + b * scale
	if isinstance(cdf, (int, float)):
		return truncnorm.truncated_normal_ab_cdf_inv(cdf, loc, scale, a, b)
	elif isinstance(cdf, np.ndarray):
		len = int(cdf.nbytes / cdf.itemsize)
		pdf = np.zeros(len)
		for i, cdfi in enumerate(cdf.astype('d').flat):
			ppf[i] = truncnorm.truncated_normal_ab_cdf_inv(cdfi, loc, scale, a, b)
		ppf = ppf.reshape(cdf.shape)
		return ppf
	else:
		raise Exception("%s format not supported for cdf argument!" % type(x))
