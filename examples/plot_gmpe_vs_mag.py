"""
Compare GMPE for different magnitudes
"""

import os
import hazard.rshalib as rshalib

fig_folder = r"C:\Temp"


gmpe_name = "AkkarEtAl2013"
gmpe = getattr(rshalib.gsim, gmpe_name)()

mags = range(6, 2, -1)

fig_filespec = os.path.join(fig_folder, "%s_PGA_vs_distance.PNG" % gmpe_name)
#fig_filespec = None
gmpe.plot_distance(mags, fig_filespec=fig_filespec)

#fig_filespec = os.path.join(fig_folder, "%s_spectra.PNG" % gmpe_name)
fig_filespec = None
d = 10
gmpe.plot_spectrum(mags, d, plot_freq=True, include_pgm=True, pgm_freq=50,
					fig_filespec=fig_filespec)
