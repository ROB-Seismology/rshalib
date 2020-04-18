# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2015-2017 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

"""
Module exports :class:'GhoshMahajan2013'
"""
from __future__ import division
import numpy as np

from openquake.hazardlib.gsim.base import IPE, CoeffsTable
from openquake.hazardlib import const
from openquake.hazardlib.imt import MMI


class GhoshMahajan2013(IPE):
    """
    Implements the Intensity Prediction Equation of Ghosh & Mahajan (2013)
    for MSK Intensity in the NW Himalaya
    Ghosh, G. K. and Mahajan, A. K. (2013) Intensity attenuation relation
    at Chamba–Garhwal area in north-west Himalaya with epicentral distance
    and magnitude, J. Earth Syst. Sci. 122(1): 107 - 122
    """
    #:
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Should be MSK !
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        MMI,
    ])

    #: Supported intensity measure component is not considered for IPEs, so
    #: we assume equivalent to 'average horizontal'
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.AVERAGE_HORIZONTAL

    #: Supported standard deviation types is total.
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL
    ])

    #: No required site parameters
    REQUIRES_SITES_PARAMETERS = set()

    #: Required rupture parameters are magnitude (MS in the paper)
    REQUIRES_RUPTURE_PARAMETERS = set(('mag', ))

    #: Required distance measure is epicentral distance
    REQUIRES_DISTANCES = set(('repi',))

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        C = self.COEFFS[imt]
        mean = (self._compute_magnitude_term(C, rup.mag) +
                self._compute_distance_term(C, dists.repi, rup.mag))
        stddevs = self._get_stddevs(C, dists.repi, stddev_types)
        return mean, stddevs

    def _compute_magnitude_term(self, C, mag):
        """
        Returns the magnitude scaling term
        """
        return C['a'] + (C['b'] * mag)

    def _compute_distance_term(self, C, repi, mag):
        """
        Returns the distance scaling term
        """
        return C['c'] * repi + C['d'] * np.log(repi)

    def _get_stddevs(self, C, distance, stddev_types):
        """
        Returns the total standard deviation, which is a function of distance
        """
        # Note: standard deviation is given as 2.459146e-10 on p. 113, we assume it is e-1
        stddevs = []
        for stddev_type in stddev_types:
            assert stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
            if stddev_type == const.StdDev.TOTAL:
                sigma = np.ones_like(distance) * 0.2459146
                stddevs.append(sigma)
        return stddevs

    # Note: In Eq. 6 on p. 114, b is given as -1.1313, but this should most
    # likely be positive (compare with Eq. 7)
    COEFFS = CoeffsTable(sa_damping=5, table="""
    IMT       a        b         c         d
    mmi  1.4838  1.1313  -0.00017  -0.80248
    """)
