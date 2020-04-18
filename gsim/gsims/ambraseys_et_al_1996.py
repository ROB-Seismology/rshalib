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
# along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
Module exports :class:`AmbraseysEtAl1996`.
"""
from __future__ import division

import numpy as np

from openquake.hazardlib.gsim.base import GMPE, CoeffsTable
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, PGV, SA


class AmbraseysEtAl1996(GMPE):
    """
    Implements GMPE developed by Ambraseys et al. and published as "Prediction
    of horizontal response spectra in Europe" (1996, Earthquake Engineering and
    Structural Dynamics, Volume 25, pages 371-400).
    """

    #: Supported tectonic region type is stable shallow crust.
    #: See paragraph 'Introduction', pag 371.
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.STABLE_CONTINENTAL

    #: Supported intensity measure types are spectral acceleration and
    #: peak ground acceleration.
    #: See equation 15, pag 378, and table 1, pag 383.
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        PGA,
        SA
    ])

    #: Supported intensity measure component is greater of two horizontal
    #: :attr:`~openquake.hazardlib.const.IMC.BOTH_HORIZONTAL`.
    #: See Douglas J, 2001. Ground-motion prediction equations 1964–2010. BRGM/RP-59356-FR, 444 pages.
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.GREATER_OF_TWO_HORIZONTAL

    #: Supported standard deviation type is total.
    #: See table 1, pag 383.
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL
    ])

    #: Required site parameter is vs30 (used to distinguish rock (vs30 >= 750 m/s),
    #: stiff soil (360 m/s <= vs30 < 750 m/s)) and soft soil (vs30 < 360 m/s).
    #: See paragraph 'Local soil conditions', pag 373, and
    #: 'Inclusion of the site geology in the attenuation model', pag 378.
    REQUIRES_SITES_PARAMETERS = set(('vs30',))

    #: Required rupture parameter is magnitude.
    #: See equation 11, pag 378.
    REQUIRES_RUPTURE_PARAMETERS = set(('mag',))

    #: Required distance measure is rjb.
    #: See paragraph 'Source distance', pag 372.
    REQUIRES_DISTANCES = set(('rjb',))

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        # extracting dictionary of coefficients specific to required
        # intensity measure type.
        C = self.COEFFS[imt]

        # Equation 11, pag 378, with intercept, magnitude, distance and site amplification term
        log10_mean = C['c1'] + \
            self._compute_magnitude_scaling(rup, C) + \
            self._compute_distance_scaling(dists, C) + \
            self._get_site_amplification(sites, C)

        # From common to natural logarithm
        mean = np.log(10**log10_mean)

        stddevs = self._get_stddevs(C, stddev_types, num_sites=len(sites.vs30))
        # From common to natural logarithm
        stddevs = np.log(stddevs)

        return mean, stddevs

    def _get_stddevs(self, C, stddev_types, num_sites):
        """
        Return standard deviations as defined in table 1, pag 383.
        """
        stddevs = []
        for stddev_type in stddev_types:
            assert stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
            if stddev_type == const.StdDev.TOTAL:
                stddevs.append(C['std'] + np.zeros(num_sites))
        stddevs = 10**np.array(stddevs)
        return stddevs

    def _compute_magnitude_scaling(self, rup, C):
        """
        Compute magnitude-scaling term, equation 11, pag 378.
        """
        val = C['c2'] * rup.mag
        return val

    def _compute_distance_scaling(self, dists, C):
        """
        Compute distance-scaling term, equations 11, pag 378, and 4, pag 376.
        """
        r = np.sqrt(dists.rjb * dists.rjb + C['h0'] * C['h0'])
        val = C['c4'] * np.log10(r)
        return val

    def _get_site_amplification(self, sites, C):
        """
        Compute site amplification term, equation 11, pag 378.
        Distinguishes between rock (vs30 >= 750 m/s),
        stiff soil (360 m/s <= vs30 < 750 m/s)) and soft soil (vs30 < 360 m/s).
        See paragraph 'Local soil conditions', pag 373, and
        'Inclusion of the site geology in the attenuation model', pag 378.
        """
        Sa, Ss = self._get_site_type_dummy_variables(sites)
        val = (C['ca']*Sa) + (C['cs']*Ss)
        return val

    def _get_site_type_dummy_variables(self, sites):
        """
        Compute site amplification coefficients.
        See paragraph 'Inclusion of the site geology in the attenuation model', pag 378.
        """
        Sa = np.zeros((len(sites.vs30),))
        Ss = np.zeros((len(sites.vs30),))
        idx_Sa = (sites.vs30 >= 360.0)&(sites.vs30 < 750.0)
        idx_Ss = (sites.vs30 < 360.0)
        Sa[idx_Sa] = 1
        Ss[idx_Ss] = 1
        return Sa, Ss

    #: Coefficient table is constructed from values in table 1, pag 383.
    #: Spectral acceleration is defined for damping of 5%.
    #: See paragraph 'Attenuation of spectral ordinates', pag 382.
    #: c1 is the intercept coefficient.
    #: c2 is the magnitude scaling coefficient.
    #: h0 is the depth projection coefficient.
    #: c4 is the distance scaling coefficient.
    #: ca is the site amplification coefficient for stiff soil (360 m/s <= vs30 < 750 m/s).
    #: cs is the site amplification coefficient for soft soil (vs30 < 360 m/s).
    #: std is the total standard deviation.

    COEFFS = CoeffsTable(sa_damping=5, table="""\
    IMT    c1    c2    h0    c4    ca    cs    std
    pga    -1.48    0.266    3.5    -0.922    0.117    0.124    0.25
    0.10    -0.84    0.219    4.5    -0.954    0.078    0.027    0.27
    0.11    -0.86    0.221    4.5    -0.945    0.098    0.036    0.27
    0.12    -0.87    0.231    4.7    -0.960    0.111    0.052    0.27
    0.13    -0.87    0.238    5.3    -0.981    0.131    0.068    0.27
    0.14    -0.94    0.244    4.9    -0.955    0.136    0.077    0.27
    0.15    -0.98    0.247    4.7    -0.938    0.143    0.085    0.27
    0.16    -1.05    0.252    4.4    -0.907    0.152    0.101    0.27
    0.17    -1.08    0.258    4.3    -0.896    0.140    0.102    0.27
    0.18    -1.13    0.268    4.0    -0.901    0.129    0.107    0.27
    0.19    -1.19    0.278    3.9    -0.907    0.133    0.130    0.28
    0.20    -1.21    0.284    4.2    -0.922    0.135    0.142    0.27
    0.22    -1.28    0.295    4.1    -0.911    0.120    0.143    0.28
    0.24    -1.37    0.308    3.9    -0.916    0.124    0.155    0.28
    0.26    -1.40    0.318    4.3    -0.942    0.134    0.163    0.28
    0.28    -1.46    0.326    4.4    -0.946    0.134    0.158    0.29
    0.30    -1.55    0.338    4.2    -0.933    0.133    0.148    0.30
    0.32    -1.63    0.349    4.2    -0.932    0.125    0.161    0.31
    0.34    -1.65    0.351    4.4    -0.939    0.118    0.163    0.31
    0.36    -1.69    0.354    4.5    -0.936    0.124    0.160    0.31
    0.38    -1.82    0.364    3.9    -0.900    0.132    0.164    0.31
    0.40    -1.94    0.377    3.6    -0.888    0.139    0.172    0.31
    0.42    -1.99    0.384    3.7    -0.897    0.147    0.180    0.32
    0.44    -2.05    0.393    3.9    -0.908    0.153    0.187    0.32
    0.46    -2.11    0.401    3.7    -0.911    0.149    0.191    0.32
    0.48    -2.17    0.410    3.5    -0.920    0.150    0.197    0.32
    0.50    -2.25    0.420    3.3    -0.913    0.147    0.201    0.32
    0.55    -2.38    0.434    3.1    -0.911    0.134    0.203    0.32
    0.60    -2.49    0.438    2.5    -0.881    0.124    0.212    0.32
    0.65    -2.58    0.451    2.8    -0.901    0.122    0.215    0.32
    0.70    -2.67    0.463    3.1    -0.914    0.116    0.214    0.33
    0.75    -2.75    0.477    3.5    -0.942    0.113    0.212    0.32
    0.80    -2.86    0.485    3.7    -0.925    0.127    0.218    0.32
    0.85    -2.93    0.492    3.9    -0.920    0.124    0.218    0.32
    0.90    -3.03    0.502    4.0    -0.920    0.124    0.225    0.32
    0.95    -3.10    0.503    4.0    -0.892    0.121    0.217    0.32
    1.00    -3.17    0.508    4.3    -0.885    0.128    0.219    0.32
    1.10    -3.30    0.513    4.0    -0.857    0.123    0.206    0.32
    1.20    -3.38    0.513    3.6    -0.851    0.128    0.214    0.31
    1.30    -3.43    0.514    3.6    -0.848    0.115    0.200    0.31
    1.40    -3.52    0.522    3.4    -0.839    0.109    0.197    0.31
    1.50    -3.61    0.524    3.0    -0.817    0.109    0.204    0.31
    1.60    -3.68    0.520    2.5    -0.781    0.108    0.206    0.31
    1.70    -3.74    0.517    2.5    -0.759    0.105    0.206    0.31
    1.80    -3.79    0.514    2.4    -0.730    0.104    0.204    0.32
    1.90    -3.80    0.508    2.8    -0.724    0.103    0.194    0.32
    2.00    -3.79    0.503    3.2    -0.728    0.101    0.182    0.32
    """)