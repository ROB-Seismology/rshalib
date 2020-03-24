"""
Common constants and functions used to generate NRML
"""

# Note: do not import unicode_literals here!
from __future__ import absolute_import, division, print_function

import sys
from lxml import etree


if sys.version_info.major == 2:
	PY2 = True
else:
	PY2 = False
	basestring = str
	unicode = str



__all__ = ['ENUM_OQ_UNCERTAINTYTYPES', 'ENUM_OQ_TRT', 'INVALID_XML_CHARS',
			'xmlstr', 'create_nrml_root', 'NRMLError']


ENUM_OQ_UNCERTAINTYTYPES = set(("gmpeModel",
								"sourceModel",
								"maxMagGRAbsolute",
								"maxMagGRRelative",
								"abGRAbsolute",
								"bGRRelative",
								"incrementalMFDRates"))

ENUM_OQ_TRT = set(('Active Shallow Crust',
					'Stable Shallow Crust',
					'Subduction Interface',
					'Subduction IntraSlab',
					'Volcanic'))

INVALID_XML_CHARS = ['&', '<', '>', '"', "'"]


def xmlstr(s, encoding='latin1'):
	"""
	Generate XML-readable string, making sure unicode characters
	or codes are replaced with xml characters

	:param s:
		any object that can be represented as a string
	:param encoding:
		unicode encoding if s is not unicode
		(default: 'latin1')

	:return:
		str
	"""
	if isinstance(s, basestring):
		for char in INVALID_XML_CHARS:
			s = s.replace(char, '')
		if not isinstance(s, unicode):
			s = s.decode(encoding)
	if isinstance(s, unicode):
		return s.encode('ascii', 'xmlcharrefreplace')
	else:
		return bytes(s)


def create_nrml_root(xml_elem, encoding='latin1', **kwargs):
	"""
	Create nrml root

	:param xml_elem:
		instance of class having a `create_xml_element` method
	:param encoding:
		unicode encoding
		(default: 'latin1')

	:return:
		instance of :class:`etree.ElementTree`
	"""
	from . import ns

	nrml_elem = etree.Element(ns.ROOT, nsmap=ns.NSMAP)
	nrml_elem.append(xml_elem.create_xml_element(encoding=encoding, **kwargs))

	root = etree.ElementTree(nrml_elem)

	return root


class NRMLError(Exception):
	pass

