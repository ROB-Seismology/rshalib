"""
"""

# https://github.com/gem/oq-engine/pull/3874

import os
import numpy as np
import openquake.hazardlib as oqhazlib
from prettytable import PrettyTable


gmpe_name = "Atkinson2015"
gmpe = oqhazlib.gsim.get_available_gsims()[gmpe_name]()

rup_mags = range(2, 7)
hypo_dists = np.array([1., 5., 10., 20., 50.])

stddev_type_names = ['TOTAL', 'INTER_EVENT', 'INTRA_EVENT']
stddev_types = [getattr(oqhazlib.const.StdDev, stddev_type)
				for stddev_type in stddev_type_names]

damping = 5
imts = [oqhazlib.imt.PGV(), oqhazlib.imt.PGA()]
for period in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
	imts.append(oqhazlib.imt.SA(period, damping))

col_names = ['rup_mag', 'dist_rhypo', 'result_type', 'damping', 'pgv', 'pga',
			'0.050', '0.100', '0.200', '0.500', '1.000', '2.000']
mean_table = PrettyTable(col_names)
std_total_table = PrettyTable(col_names)
std_inter_table = PrettyTable(col_names)
std_intra_table = PrettyTable(col_names)

for mag in rup_mags:
	sctx = oqhazlib.gsim.base.SitesContext()
	rctx = oqhazlib.gsim.base.RuptureContext()
	rctx.mag = mag
	for rhypo in hypo_dists:
		dctx = oqhazlib.gsim.base.DistancesContext()
		dctx.rhypo = np.array([rhypo])
		mean_row = ["%.2f" % mag, "%.2f" % rhypo, "MEAN", "%.1f" % damping]
		std_total_row = ["%.2f" % mag, "%.2f" % rhypo, "TOTAL_STDDEV", "%.1f" % damping]
		std_inter_row = ["%.2f" % mag, "%.2f" % rhypo, "INTER_EVENT_STDDEV", "%.1f" % damping]
		std_intra_row = ["%.2f" % mag, "%.2f" % rhypo, "INTRA_EVENT_STDDEV", "%.1f" % damping]
		for imt in imts:
			mean, stddevs = gmpe.get_mean_and_stddevs(sctx, rctx, dctx, imt, stddev_types)
			mean_row.append("%E" % np.exp(mean))
			std_total_row.append("%f" % stddevs[0])
			std_inter_row.append("%f" % stddevs[1])
			std_intra_row.append("%f" % stddevs[2])
		mean_table.add_row(mean_row)
		std_total_table.add_row(std_total_row)
		std_inter_table.add_row(std_inter_row)
		std_intra_table.add_row(std_intra_row)

namespace = __import__(__name__)
for result_type in ["mean", "std_total", "std_inter", "std_intra"]:
	tab_name = "%s_table" % result_type
	tab = getattr(namespace, tab_name)
	print(tab)
	csv_filename = "ATKINSON2015_%s.csv" % tab_name.upper()
	csv_filespec = os.path.join(r"C:\Temp", csv_filename)
	with open(csv_filespec, 'w') as csv:
		csv.write(', '.join(tab.field_names))
		csv.write('\n')
		for row in tab._rows:
			csv.write(', '.join(row))
			csv.write('\n')
