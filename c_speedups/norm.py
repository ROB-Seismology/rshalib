"""
Wrapper functions mimicking behavior of scipy.stats.norm methods
"""

#import truncated_normal.truncated_normal as truncnorm
from .truncated_normal import truncated_normal as truncnorm
import numpy as np



def pdf(x, loc=0., scale=1.):
	if isinstance(x, (int, float)):
		return truncnorm.normal_pdf(x, loc, scale)
	elif isinstance(x, np.ndarray):
		len = int(x.nbytes / x.itemsize)
		pdf = np.zeros(len)
		for i, xi in enumerate(x.astype('d').flat):
			pdf[i] = truncnorm.normal_pdf(xi, loc, scale)
		pdf = pdf.reshape(x.shape)
		return pdf
	else:
		raise Exception("%s format not supported for x argument!" % type(x))

def cdf(x, loc=0., scale=1.):
	if isinstance(x, (int, float)):
		return truncnorm.normal_cdf(x, loc, scale)
	elif isinstance(x, np.ndarray):
		len = int(x.nbytes / x.itemsize)
		cdf = np.zeros(len)
		for i, xi in enumerate(x.astype('d').flat):
			cdf[i] = truncnorm.normal_cdf(xi, loc, scale)
		cdf = cdf.reshape(x.shape)
		return cdf
	else:
		raise Exception("%s format not supported for x argument!" % type(x))

def sf(x, loc=0., scale=1.):
	return 1 - cdf(x, loc=loc, scale=scale)

def ppf(cdf, loc=0., scale=1.):
	if isinstance(cdf, (int, float)):
		return truncnorm.normal_cdf_inv(cdf, loc, scale)
	elif isinstance(cdf, np.ndarray):
		len = int(cdf.nbytes / cdf.itemsize)
		ppf = np.zeros(len)
		for i, cdfi in enumerate(cdf.astype('d').flat):
			ppf[i] = truncnorm.normal_cdf_inv(cdfi, loc, scale)
		ppf = ppf.reshape(cdf.shape)
		return ppf
	else:
		raise Exception("%s format not supported for cdf argument!" % type(x))
