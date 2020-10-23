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
Module exports :class:'SzeligaEtAl2010'
"""
from __future__ import division
import numpy as np

from openquake.hazardlib.gsim.base import CoeffsTable
from openquake.hazardlib import const
from openquake.hazardlib.imt import MMI
from . import IPE


class SzeligaEtAl2010(IPE):
    """
    Implements the Intensity Prediction Equation of Szeliga et al. (2010)
    MSK Intensity in the Himalaya
    Szeliga, W., Hough, S., Martin, S. and Bilham, R. (2010) Intensity,
    Magnitude, Location, and Attenuation in India for Felt Earthquakes
    since 1762, BSSA, 100(2): 570 – 584
    """
    #:
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Should be EMS-98 !
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

    #: Required rupture parameters are magnitude
    REQUIRES_RUPTURE_PARAMETERS = set(('mag', ))

    #: Required distance measure is hypocentral distance
    REQUIRES_DISTANCES = set(('rhypo',))

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        C = self.COEFFS[imt]
        mean = (self._compute_magnitude_term(C, rup.mag) +
                self._compute_distance_term(C, dists.rhypo, rup.mag))
        stddevs = self._get_stddevs(C, dists.rhypo, stddev_types)
        return mean, stddevs

    def _compute_magnitude_term(self, C, mag):
        """
        Returns the magnitude scaling term
        """
        return C['a'] + (C['b'] * mag)

    def _compute_distance_term(self, C, rhypo, mag):
        """
        Returns the distance scaling term
        """
        #: Note: Eq. 1 on p. 571 says log, but Fig. 3 can only be reproduced it it is log10 !!!
        return C['c'] * rhypo + C['d'] * np.log10(rhypo)

    def _get_stddevs(self, C, distance, stddev_types):
        """
        Returns the total standard deviation, which is a function of distance
        """
        stddevs = []
        for stddev_type in stddev_types:
            assert stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
            if stddev_type == const.StdDev.TOTAL:
                sigma = np.zeros_like(distance)
                stddevs.append(sigma)
        return stddevs

    COEFFS = CoeffsTable(sa_damping=5, table="""
    IMT      a      b        c      d
    mmi   6.05   1.11  -0.0006  -3.91
    """)
