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
Module exports :class:`Campbell2003HTT`
"""
from __future__ import division

import numpy as np

from openquake.hazardlib.gsim.campbell_2003 import Campbell2003
from openquake.hazardlib.gsim.base import CoeffsTable
from openquake.hazardlib.imt import PGA, SA


class Campbell2003HTT(Campbell2003):
    """
    Implements vs30 and kappa host to target adjusments for Campbell2003 GMPE.
    """

    #: HTT requires vs30 and kappa
    REQUIRES_SITES_PARAMETERS = set(('vs30', 'kappa'))
    DEFINED_FOR_VS30 = np.arange(600, 2700, 100)
    DEFINED_FOR_KAPPA = np.array([0.01, 0.015, 0.02, 0.025, 0.0275, 0.03, 0.04, 0.05])

    host_vs30 = 2800.
    host_kappa = 0.0069

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        Apply HTT
        """
        mean, stddevs = super(Campbell2003HTT, self).get_mean_and_stddevs(sites, rup, dists, imt, stddev_types)

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
pga  3.2775 2.8121 2.4810 2.2343 2.0282 1.8702 1.7438 1.6358 1.5462 1.4702 1.4032 1.3451 1.2939 1.2477 1.2065 1.1692 1.1349 1.1036 1.0750 1.0486 1.0242 1.0015
0.02 3.5924 3.0258 2.6386 2.3595 2.1309 1.9560 1.8164 1.6971 1.5984 1.5148 1.4410 1.3771 1.3209 1.2702 1.2249 1.1840 1.1463 1.1120 1.0806 1.0516 1.0248 1.0000
0.03 3.3759 2.8641 2.5129 2.2573 2.0507 1.8906 1.7619 1.6524 1.5611 1.4834 1.4149 1.3553 1.3026 1.2551 1.2126 1.1741 1.1386 1.1062 1.0765 1.0491 1.0237 1.0000
0.05 3.0722 2.6374 2.3357 2.1135 1.9357 1.7961 1.6828 1.5867 1.5059 1.4366 1.3756 1.3222 1.2748 1.2320 1.1936 1.1587 1.1266 1.0971 1.0700 1.0450 1.0217 1.0000
0.07 2.8455 2.4675 2.2025 2.0057 1.8487 1.7244 1.6227 1.5366 1.4636 1.4008 1.3455 1.2967 1.2533 1.2140 1.1786 1.1464 1.1168 1.0896 1.0646 1.0415 1.0200 1.0000
0.10 2.6993 2.3573 2.1160 1.9356 1.7919 1.6773 1.5832 1.5031 1.4349 1.3759 1.3239 1.2781 1.2372 1.2004 1.1672 1.1370 1.1093 1.0839 1.0605 1.0388 1.0187 1.0000
0.15 2.5136 2.2164 2.0043 1.8438 1.7149 1.6113 1.5258 1.4532 1.3915 1.3382 1.2913 1.2499 1.2132 1.1801 1.1502 1.1230 1.0982 1.0753 1.0543 1.0349 1.0168 1.0000
0.20 2.3903 2.1193 1.9231 1.7741 1.6554 1.5600 1.4814 1.4150 1.3584 1.3096 1.2667 1.2288 1.1952 1.1649 1.1375 1.1127 1.0899 1.0690 1.0498 1.0320 1.0154 1.0000
0.30 2.2102 1.9713 1.7997 1.6700 1.5673 1.4848 1.4169 1.3595 1.3107 1.2684 1.2314 1.1987 1.1695 1.1433 1.1196 1.0981 1.0783 1.0602 1.0434 1.0279 1.0135 1.0000
0.50 1.9677 1.7772 1.6409 1.5381 1.4569 1.3918 1.3382 1.2928 1.2541 1.2205 1.1908 1.1645 1.1409 1.1195 1.1001 1.0823 1.0659 1.0508 1.0368 1.0237 1.0115 1.0000
0.75 1.7841 1.6360 1.5299 1.4494 1.3851 1.3327 1.2890 1.2516 1.2193 1.1910 1.1659 1.1434 1.1232 1.1047 1.0879 1.0724 1.0581 1.0449 1.0325 1.0210 1.0102 1.0000
1.00 1.6813 1.5592 1.4698 1.4009 1.3452 1.2994 1.2609 1.2278 1.1990 1.1737 1.1511 1.1308 1.1125 1.0957 1.0804 1.0664 1.0533 1.0412 1.0299 1.0193 1.0093 1.0000
1.50 1.5705 1.4732 1.4006 1.3437 1.2972 1.2587 1.2260 1.1977 1.1731 1.1513 1.1318 1.1143 1.0984 1.0838 1.0705 1.0582 1.0468 1.0361 1.0262 1.0169 1.0082 1.0000
2.00 1.5054 1.4207 1.3569 1.3066 1.2654 1.2311 1.2020 1.1768 1.1548 1.1353 1.1179 1.1022 1.0879 1.0749 1.0630 1.0520 1.0418 1.0323 1.0234 1.0151 1.0073 1.0000
3.00 1.4175 1.3483 1.2952 1.2529 1.2182 1.1893 1.1649 1.1437 1.1253 1.1091 1.0946 1.0817 1.0700 1.0594 1.0497 1.0409 1.0327 1.0251 1.0182 1.0117 1.0056 1.0000
4.00 1.3601 1.2986 1.2522 1.2156 1.1858 1.1611 1.1403 1.1222 1.1063 1.0924 1.0798 1.0686 1.0586 1.0494 1.0411 1.0335 1.0265 1.0201 1.0143 1.0091 1.0043 1.0000
    """)

    KAPPA_COEFFS = CoeffsTable(sa_damping=5, table="""\
imt 0.01 0.015 0.02 0.025 0.0275 0.03 0.04 0.05
pga  0.9409 0.6376 0.4951 0.4109 0.3801 0.3541 0.2807 0.2339
0.02 0.9252 0.5106 0.3211 0.2326 0.2061 0.1863 0.1393 0.1139
0.03 0.9457 0.6038 0.4009 0.2816 0.2420 0.2117 0.1426 0.1107
0.05 0.9659 0.7289 0.5571 0.4318 0.3826 0.3405 0.2254 0.1634
0.07 0.9767 0.8061 0.6698 0.5600 0.5132 0.4710 0.3400 0.2532
0.10 0.9821 0.8475 0.7349 0.6396 0.5973 0.5584 0.4295 0.3347
0.15 0.9873 0.8903 0.8056 0.7307 0.6964 0.6639 0.5503 0.4583
0.20 0.9900 0.9127 0.8440 0.7820 0.7531 0.7256 0.6267 0.5430
0.30 0.9927 0.9358 0.8845 0.8373 0.8151 0.7936 0.7149 0.6457
0.50 0.9950 0.9558 0.9199 0.8865 0.8706 0.8552 0.7973 0.7450
0.75 0.9961 0.9660 0.9384 0.9126 0.9003 0.8882 0.8430 0.8016
1.00 0.9967 0.9714 0.9482 0.9267 0.9164 0.9064 0.8689 0.8349
1.50 0.9975 0.9780 0.9608 0.9453 0.9381 0.9313 0.9082 0.8952
2.00 0.9980 0.9835 0.9719 0.9634 0.9606 0.9591 0.9838 0.8946
3.00 0.9992 0.9965 1.0046 1.0455 1.1341 0.9631 0.9588 0.9732
4.00 0.9999 1.0057 1.0356 1.6432 0.9844 0.9860 1.0120 1.2118
    """)
