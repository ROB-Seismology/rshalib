"""
Common constants and functions used to generate NRML
"""

from lxml import etree
import ns


ENUM_OQ_UNCERTAINTYTYPES = set(("gmpeModel", "sourceModel", "maxMagGRAbsolute", "maxMagGRRelative", "abGRAbsolute", "bGRRelative", "incrementalMFDRates"))
ENUM_OQ_TRT = set(('Active Shallow Crust', 'Stable Shallow Crust', 'Subduction Interface', 'Subduction IntraSlab', 'Volcanic'))

INVALID_XML_CHARS = ['&', '<', '>', '"', "'"]


def xmlstr(s, encoding='latin1'):
	"""
	Generate XML-readable string, making sure unicode characters or codes are
	replaced with xml characters
	Arguments:
		s: any object that can be represented as a string
		encoding: unicode encoding if s is a string (default: 'latin1')
	"""
	for char in INVALID_XML_CHARS:
		s = s.replace(char, '')
	if isinstance(s, str):
		return s.decode(encoding).encode('ascii', 'xmlcharrefreplace')
	elif isinstance(s, unicode):
		return s.encode('ascii', 'xmlcharrefreplace')
	else:
		return str(s)


def create_nrml_root(xml_elem, encoding='latin1'):
	"""
	Create nrml root
	Arguments:
		encoding: unicode encoding (default: 'latin1')
	"""
	nrml_elem = etree.Element(ns.ROOT, nsmap=ns.NSMAP)
	nrml_elem.append(xml_elem.create_xml_element(encoding=encoding))

	root = etree.ElementTree(nrml_elem)

	return root


class NRMLError(Exception):
	pass


