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
Module exports :class:`BakunWentworth1997`.
"""
from __future__ import division

import numpy as np

from openquake.hazardlib.gsim.base import IPE
from openquake.hazardlib import const
from openquake.hazardlib.imt import MMI


class BakunWentworth1997(IPE):
    """
    Implements IPE developed by Bakun & Wentworth (1997)
    California
    Epicentral distance range 0 - 150 km
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

    REQUIRES_DISTANCES = set(('repi', ))


    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """

        # Eq. 4, page 1508
        mean_mmi = 3.67 + 1.17 * rup.mag - 3.19 * np.log10(dists.repi)
        mean_mmi = mean_mmi.clip(min=1, max=12)

        stddevs = np.zeros_like(dists.repi)
        stddevs = stddevs.reshape(1, len(stddevs))

        return mean_mmi, stddevs

    def compute_site_term(self, sites):
        # TODO !
        pass


class BakunWentworth1997WithSigma(BakunWentworth1997):
    """
    Implements IPE developed by Bakun & Wentworth (1997)
    with added sigma = 0.4, similar to AtkinsonWald2007
    """
    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """

        mean, stddevs = super(BakunWentworth1997WithSigma, self).get_mean_and_stddevs(
                                            sites, rup, dists, imt, stddev_types)

        stddevs.fill(0.4)

        return mean, stddevs

