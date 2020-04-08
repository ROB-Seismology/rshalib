"""
Magnitude scaling relations
"""

from __future__ import absolute_import, division, print_function, unicode_literals

try:
	## Python 2
	basestring
except:
	## Python 3
	basestring = str



__all__ = ['get_oq_msr']


def get_oq_msr(msr_or_name):
	"""
	Get OpenQuake magnitude scaling relationship object

	:param msr_or_name:
		str or instance of :class:`oqhazlib.scalerel.BaseMSR`

	:return:
		instance of :class:`oqhazlib.scalerel.BaseMSR`
	"""
	from . import oqhazlib

	if isinstance(msr_or_name, oqhazlib.scalerel.BaseMSR):
		msr = msr_or_name
	elif isinstance(msr_or_name, basestring):
		if msr_or_name[-3:] != 'MSR':
			msr_or_name += 'MSR'
		msr = getattr(oqhazlib.scalerel, msr_or_name)()

	return msr
