"""
Read EC8 information from GIS files
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import platform
if platform.uname()[0] == "Windows":
	GIS_root = "D:\\GIS-data"
else:
	GIS_root = os.path.join(os.environ["HOME"], "gis-data")

from mapping.geotools.read_gis import read_gis_file



__all__ = ['read_ec8_information']


def read_ec8_information():
	"""
	:return:
		village_ec8_PGA_dict: dict, mapping village names to tuple of
		EC8 zone (int) and reference PGA (float)
	"""
	village_filespec = os.path.join(GIS_root, 'Belgium',
									'Bel_main_village_points.TAB')
	village_records = read_gis_file(village_filespec)
	ec8_filespec = os.path.join(GIS_root, 'KSB-ORB', 'EC8_2002-2011.TAB')
	ec8_records = read_gis_file(ec8_filespec, encoding=None)

	village_ID_dict = {}
	for rec in village_records:
		village_ID_dict[rec['Name']] = rec['Village_number']
	ID_ec8_dict = {}
	for rec in ec8_records:
		ID_ec8_dict[rec['CommuneID']] = rec['EC8_2011_Zone']

	ec8_zone_PGA_dict = {0: 0.0, 1: 0.04, 2: 0.06, 3: 0.08, 4: 0.1}

	village_ec8_PGA_dict = {}
	for village in village_ID_dict.keys():
		ID = village_ID_dict[village]
		EC8_zone = ID_ec8_dict[ID]
		PGA = ec8_zone_PGA_dict[EC8_zone]
		village_ec8_PGA_dict[village] = (EC8_zone, PGA)

	return village_ec8_PGA_dict



if __name__ == "__main__":
	commune = "Dessel"
	print(commune, read_ec8_information()[commune])
