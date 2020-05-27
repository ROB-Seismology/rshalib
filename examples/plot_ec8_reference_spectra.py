"""
Compare Eurocode-8 response spectra for different ground types
"""

import hazard.rshalib as rshalib

ag = 1.0
resp_type = 2
rs_list = []
ground_types = ['A', 'B', 'C', 'D', 'E']
for ground_type in ground_types:
	rs = rshalib.refspec.get_ec8_rs(ag, ground_type, resp_type)
	rs_list.append(rs)

rshalib.result.plot_hazard_spectra(rs_list, labels=ground_types, plot_freq=False,
									ymax=5, xmin=1./34, xmax=4.0,
									xscaling='lin', yscaling='lin',
									linewidths=[2])
