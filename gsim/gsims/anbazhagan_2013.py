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
Module exports :class:`Anbazhagan2013`.
"""
from __future__ import division

import numpy as np
# standard acceleration of gravity in m/s**2
from scipy.constants import g

# Compute this log only once
ln10 = np.log(10)

from openquake.hazardlib.gsim.base import CoeffsTable, GMPE
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, PGV, SA


class Anbazhagan2013(GMPE):
    """
    Implements GMPE developed by Anbazhagan et al. (2013),  and published
    as "Ground motion prediction equation considering combined dataset
    of recorded and simulated ground motions" (Soil Dynamics and Earthquake
    Engineering, 2013, volume 53, pages 92-108).

    This GMPE was developed for the Himalayan region.
    """

    #: Supported tectonic region type is active shallow crust.
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST

    #: Supported intensity measure types are spectral acceleration
    #: and peak ground acceleration.
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        PGA,
        SA
    ])

    #: Supported intensity measure component is the geometric mean of two
    #: horizontal components.
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.AVERAGE_HORIZONTAL

    #: Supported standard deviation type is only total.
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL
    ])

    #: There ar no required site parameters
    REQUIRES_SITES_PARAMETERS = set()

    #: Required rupture parameter is only magnitude.
    REQUIRES_RUPTURE_PARAMETERS = set(('mag',))

    #: Required distance measure is Rhypo.
    REQUIRES_DISTANCES = set(('rhypo', ))

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        # extract dictionaries of coefficients specific to required
        # intensity measure type
        C = self.COEFFS[imt]

        # mean value as given by equation (4), p. 103.
        # log y = c1 + c2 * M - b * log10(rhypo + exp(c3 * M))
        c1, c2, c3, b = C['c1'], C['c2'], C['c3'], C['b']
        log10_mean = c1 + c2 * rup.mag - b * np.log10(dists.rhypo + np.exp(c3 * rup.mag))

        # From base-10 to natural logarithm
        mean = log10_mean * ln10

        stddevs = self._get_stddevs(C, stddev_types, num_sites=len(mean))
        # From base-10 to natural logarithm
        stddevs *= ln10

        return mean, stddevs

    def _get_stddevs(self, C, stddev_types, num_sites):
        """
        Return total standard deviation.
        """
        stddevs = []
        for stddev_type in stddev_types:
            assert stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
            stddevs.append(C['sigma'] + np.zeros(num_sites))

        return np.array(stddevs)

    #: Coefficient table constructed from Table 3 of the original paper.
    COEFFS = CoeffsTable(sa_damping=5, table="""\
    IMT      c1         c2          b           c3          sigma
    pga     -1.283      0.544       1.792       0.381       0.283
    0.1     -1.475      0.544       1.585       0.322       0.307
    0.2     -1.366      0.546       1.641       0.410       0.318
    0.3     -1.982      0.542       1.385       0.367       0.298
    0.4     -2.602      0.555       1.178       0.329       0.298
    0.5     -2.980      0.606       1.206       0.350       0.292
    0.6     -3.00       0.623       1.258       0.387       0.299
    0.8     -3.812      0.670       1.080       0.365       0.296
    1.0     -4.357      0.731       1.114       0.383       0.300
    1.2     -4.750      0.766       1.082       0.390       0.298
    1.4     -5.018      0.779       1.032       0.375       0.303
    1.6     -5.219      0.824       1.123       0.399       0.306
    1.8     -5.327      0.840       1.139       0.412       0.313
    2.0     -4.920      0.953       1.617       0.581       0.310
    """)
