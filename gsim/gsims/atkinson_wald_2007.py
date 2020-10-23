# coding: utf-8
# The Hazard Library
# Copyright (C) 2012 GEM Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Module exports :class:`AtkinsonWald2007`.
"""
from __future__ import division

import numpy as np

from openquake.hazardlib import const
from openquake.hazardlib.imt import MMI
from . import IPE


class AtkinsonWald2007(IPE):
    """
    Implements IPE developed by Atkinson and Wald (2007)
    California, USA
    MS!
    """

    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        MMI
    ])

    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.AVERAGE_HORIZONTAL

    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL
    ])

    # TODO !
    REQUIRES_SITES_PARAMETERS = set(('vs30', ))

    REQUIRES_RUPTURE_PARAMETERS = set(('mag',))

    REQUIRES_DISTANCES = set(('rrup', ))


    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        h = 14.0
        R = np.sqrt(dists.rrup**2 + h**2)
        B = np.zeros_like(dists.rrup)
        B[R > 30.] = np.log10(R / 30.)[R > 30.]
        mean_mmi = 12.27 + 2.270 * (rup.mag - 6) + 0.1304 * (rup.mag - 6)**2 - 1.30 * np.log10(R) - 0.0007070 * R + 1.95 * B - 0.577 * rup.mag * np.log10(R)
        mean_mmi += self.compute_site_term(sites)
        mean_mmi = mean_mmi.clip(min=1, max=12)

        stddevs = np.zeros_like(dists.rrup)
        stddevs.fill(0.4)
        stddevs = stddevs.reshape(1, len(stddevs))
        return mean_mmi, stddevs

    def compute_site_term(self, sites):
        # TODO !
        return 0
