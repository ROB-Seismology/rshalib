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
Module exports :class:`AllenEtAl2012Rrup`.
"""
from __future__ import division

import numpy as np

from openquake.hazardlib import const
from openquake.hazardlib.imt import MMI
from . import IPE


class AllenEtAl2012Rrup(IPE):
    """
    Implements IPE developed by Allen et al. (2012)
    for active regions
    Rupture distance
    """

    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        MMI
    ])

    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.AVERAGE_HORIZONTAL

    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL
    ])

    # TODO
    REQUIRES_SITES_PARAMETERS = set(('vs30',))

    REQUIRES_RUPTURE_PARAMETERS = set(('mag',))

    REQUIRES_DISTANCES = set(('rrup',))


    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        ## Table 2
        c0 = 3.95
        c1 = 0.913
        c2 = -1.107
        c3 = 0.813

        ## Eqn. 1
        mean_mmi = (c0 + c1 * rup.mag + c2 *
                np.log(np.sqrt(dists.rrup**2 + (1 + c3 * np.exp(rup.mag-5))**2)))

        mean_mmi += self.compute_site_term(sites)
        mean_mmi = mean_mmi.clip(min=1, max=12)

        ## Table 3
        s1 = 0.72
        s2 = 0.23
        s3 = 44.7

        ## Eqn. XX
        stddevs = s1 + (s2 / (1 + (dists.rrup / s3)**2))
        stddevs = stddevs.reshape(1, len(stddevs))
        return mean_mmi, stddevs

    def compute_site_term(self, sites):
        # TODO !
        return 0

# TODO: AllenEtAl2012Rhyp
