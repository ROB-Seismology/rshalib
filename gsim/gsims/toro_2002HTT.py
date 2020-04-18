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
Module exports :class:`ToroEtAl2002HTT`
"""
from __future__ import division

import numpy as np

from openquake.hazardlib.gsim.toro_2002 import ToroEtAl2002
from openquake.hazardlib.gsim.base import CoeffsTable
from openquake.hazardlib.imt import PGA, SA


class ToroEtAl2002HTT(ToroEtAl2002):
    """
    Implements vs30 and kappa host to target adjusments for ToroEtAl2002 GMPE.
    """

    #: HTT requires vs30 and kappa
    REQUIRES_SITES_PARAMETERS = set(('vs30', 'kappa'))
    DEFINED_FOR_VS30 = np.arange(600, 2700, 100)
    DEFINED_FOR_KAPPA = np.array([0.01, 0.015, 0.02, 0.025, 0.0275, 0.03, 0.04, 0.05])

    host_vs30 = 2800
    host_kappa = 0.007

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        Apply HTT
        """
        mean, stddevs = super(ToroEtAl2002HTT, self).get_mean_and_stddevs(sites, rup, dists, imt, stddev_types)

        mean += np.log(self.get_kappa_cf(sites.kappa, imt))
        mean += np.log(self.get_vs30_cf(sites.vs30, imt))

        return mean, stddevs

    def get_kappa_cf(self, target_kappas, imt):
        """
        """
        kappa_coeffs = self.KAPPA_COEFFS[imt]
        if (target_kappas == target_kappas[0]).all():
            nearest_kappa = self.find_nearest(self.DEFINED_FOR_KAPPA, target_kappas[0])
            return kappa_coeffs['%s' % nearest_kappa]
        else:
            cfs = []
            nearest_kappas = self.find_nearest(self.DEFINED_FOR_KAPPA, target_kappas)
            for nearest_kappa in nearest_kappas:
               cfs.append(kappa_coeffs['%s' % nearest_kappa])
            return np.array(cfs)

    def get_vs30_cf(self, target_vs30s, imt):
        """
        """
        vs30_coeffs = self.VS30_COEFFS[imt]
        if (target_vs30s == target_vs30s[0]).all():
            nearest_vs30 = self.find_nearest(self.DEFINED_FOR_VS30, target_vs30s[0])
            return vs30_coeffs['%.f' % nearest_vs30]
        else:
            cfs = []
            nearest_vs30s = self.find_nearest(self.DEFINED_FOR_VS30, target_vs30s)
            for nearest_vs30 in nearest_vs30s:
               cfs.append(vs30_coeffs['%.f' % nearest_vs30])
            return np.array(cfs)

    @staticmethod
    def find_nearest(array, values):
        """
        Find nearest values in a particular array

        :param array:
            float array, array containing predefined values
        :param values:
            float or float array, value(s) to be looked up in :param:`array`

        :return:
            float or float array selected from :param:`array`
        """
        indices = np.abs(np.subtract.outer(array, values)).argmin(0)
        return array[indices]


    VS30_COEFFS = CoeffsTable(sa_damping=5, table="""\
imt 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400 2500 2600 2700
pga  2.9680 2.5647 2.2814 2.0708 1.8993 1.7656 1.6574 1.5654 1.4884 1.4225 1.3645 1.3138 1.2690 1.2286 1.1923 1.1594 1.1292 1.1016 1.0762 1.0528 1.0311 1.0108
0.03 3.3039 2.8093 2.4693 2.2215 2.0215 1.8663 1.7413 1.6350 1.5462 1.4706 1.4040 1.3460 1.2947 1.2484 1.2070 1.1695 1.1349 1.1034 1.0744 1.0476 1.0228 0.9998
0.04 3.1602 2.7026 2.3865 2.1545 1.9683 1.8228 1.7050 1.6051 1.5213 1.4496 1.3865 1.3313 1.2824 1.2383 1.1987 1.1628 1.1298 1.0996 1.0718 1.0461 1.0222 1.0000
0.10 2.6809 2.3432 2.1048 1.9264 1.7843 1.6710 1.5779 1.4986 1.4311 1.3727 1.3212 1.2757 1.2353 1.1988 1.1658 1.1359 1.1084 1.0832 1.0600 1.0385 1.0186 1.0000
0.20 2.3705 2.1037 1.9105 1.7638 1.6468 1.5528 1.4753 1.4097 1.3539 1.3057 1.2634 1.2261 1.1928 1.1629 1.1359 1.1114 1.0889 1.0682 1.0492 1.0316 1.0152 1.0000
0.40 2.0531 1.8453 1.6963 1.5839 1.4949 1.4235 1.3646 1.3149 1.2725 1.2359 1.2037 1.1752 1.1498 1.1269 1.1061 1.0872 1.0698 1.0537 1.0388 1.0250 1.0121 1.0000
1.00 1.6630 1.5441 1.4571 1.3900 1.3358 1.2912 1.2537 1.2215 1.1935 1.1688 1.1469 1.1272 1.1093 1.0931 1.0782 1.0645 1.0518 1.0400 1.0290 1.0187 1.0091 1.0000
2.00 1.4711 1.3908 1.3304 1.2830 1.2443 1.2122 1.1851 1.1616 1.1412 1.1231 1.1071 1.0926 1.0796 1.0677 1.0568 1.0468 1.0375 1.0289 1.0210 1.0135 1.0065 1.0000
    """)

    KAPPA_COEFFS = CoeffsTable(sa_damping=5, table="""\
imt 0.01 0.015 0.02 0.025 0.0275 0.03 0.04 0.05
pga  0.9745 0.7627 0.6361 0.5489 0.5145 0.4846 0.3951 0.3350
0.03 0.9495 0.6319 0.4434 0.3320 0.2943 0.2646 0.1928 0.1557
0.04 0.9601 0.6909 0.5090 0.3865 0.3413 0.3042 0.2099 0.1621
0.10 0.9826 0.8507 0.7391 0.6442 0.6020 0.5630 0.4341 0.3393
0.20 0.9905 0.9160 0.8489 0.7879 0.7594 0.7321 0.6340 0.5508
0.40 0.9945 0.9513 0.9112 0.8739 0.8560 0.8387 0.7742 0.7161
1.00 0.9972 0.9748 0.9543 0.9354 0.9264 0.9178 0.8869 0.8623
2.00 0.9989 0.9927 0.9923 1.0037 1.0196 1.0550 0.9384 0.9289
    """)
