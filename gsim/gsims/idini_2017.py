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
Module exports :class:`IdiniEtAl2017SInter`, class:`IdiniEtAl2017SSlab`
"""
from __future__ import division

import numpy as np
# standard acceleration of gravity in m/s**2
from scipy.constants import g

from openquake.hazardlib.gsim.base import GMPE, CoeffsTable
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, SA


class IdiniEtAl2017SInter(GMPE):
    """
    Implements GMPE developed by Idini et al. and published as
    "Ground motion prediction equations for the Chilean subduction zone"
    (Bull. Earthquake Eng.,  Volume 15, Issue 5, pp 1853-1880, 2017,
    DOI 10.1007/s10518-016-0050-1).
    The class implements the equations for 'Subduction Interface'
    (that's why the class name ends with 'SInter').
    """
    #: Supported tectonic region type is subduction interface
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTERFACE

    #: Supported intensity measure types are spectral acceleration,
    #: and peak ground acceleration
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        PGA,
        SA
    ])

    #: Supported intensity measure component is not mentioned in the paper
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.RANDOM_HORIZONTAL

    #: Supported standard deviation types are inter-event, intra-event
    #: and total
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL,
        const.StdDev.INTER_EVENT,
        const.StdDev.INTRA_EVENT
    ])

    #: Required site parameters is Vs30, used to distinguish between NEHRP
    #: soil classes
    REQUIRES_SITES_PARAMETERS = set(('vs30', ))

    #: Required rupture parameters are magnitude and focal depth
    REQUIRES_RUPTURE_PARAMETERS = set(('mag', 'hypo_depth'))

    #: Required distance measure is closest distance to rupture or hypocentral
    #: distance
    REQUIRES_DISTANCES = set(('rrup', 'rhypo'))

    #: Dummy variable to distinguish between interface (0) and intraslab (1)
    Feve = 0

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        # extracting dictionary of coefficients specific to required
        # intensity measure type.
        C = self.COEFFS[imt]

        if self.Feve == 1 or rup.mag < 7.7:
            R = dists.rhypo
        else:
            R = dists.rrup
        mean = self._compute_mean(C, rup.mag, rup.hypo_depth, R, sites.vs30)

        # convert from log10 to ln
        mean = np.log(10 ** mean)

        stddevs = self._get_stddevs(C, stddev_types, sites.vs30.shape[0])

        return mean, stddevs

    def _compute_mean(self, C, mag, hypo_depth, R, vs30):
        """
        Compute (log10 of) mean according to Eqn. 2
        """
        Ff = self._compute_source_contribution(C, mag, hypo_depth)
        Fd = self._compute_path_contribution(C, mag, R)
        Fs = self._compute_site_contribution(C, vs30)
        mean = (Ff + Fd + Fs)

        return mean

    def _compute_source_contribution(self, C, mag, hypo_depth):
        # Eqn. 3
        c1, c2, c8 = C['c1'], C['c2'], C['c8']
        h0 = 50.
        Ff = c1 + c2 * mag + c8 * (hypo_depth - h0) * self.Feve
        Ff += self._compute_delta_fM(C, mag)
        return Ff

    def _compute_delta_fM(self, C, mag):
        # Eqn. 4
        c9 = C['c9']
        delta_fM = c9 * mag**2
        return delta_fM

    def _compute_path_contribution(self, C, mag, R):
        c3, c5 = C['c3'], C['c5']
        c4 = 0.1
        c6 = 5.
        c7 = 0.35
        Dc3 = C['Dc3']
        Mr = 5.
        # Eqn. 7
        g = c3 + c4 * (mag - Mr) + Dc3 * self.Feve
        # Eqn. 6
        R0 = (1 - self.Feve) * c6 * 10 ** (c7 * (mag - Mr))
        # Eqn. 5
        Fd = g * np.log10(R + R0) + c5 * R
        return Fd

    def _compute_site_contribution(self, C, vs30):
        # Not currently possible due to site classification based on predominant
        # period (see Table 2)
        # Eqn. 18
        #Vref = 1530
        #Fs = ST * np.log10(vs30/Vref)
        # rock
        Fs = 0
        return Fs

    def _get_stddevs(self, C, stddev_types, num_sites):
        """
        Return standard deviations.
        """
        stddevs = []
        for stddev_type in stddev_types:
            assert stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
            if stddev_type == const.StdDev.TOTAL:
                stddevs.append(np.log(10 ** C['st']) + np.zeros(num_sites))
            elif stddev_type == const.StdDev.INTRA_EVENT:
                stddevs.append(np.log(10 ** C['sr']) + np.zeros(num_sites))
            elif stddev_type == const.StdDev.INTER_EVENT:
                stddevs.append(np.log(10 ** C['se']) + np.zeros(num_sites))
        return stddevs

    COEFFS = CoeffsTable(sa_damping=5, table="""\
    IMT   c3        c5        Dc3       sr     c1       c2      c9        c8       Dc1     Dc2      se     st     sII     sIII    sIV     sV      sVI
    pga   -0.97558  -0.00174  -0.52745  0.232  -2.8548  0.7741  -0.03958  0.00586  2.5699  -0.4761  0.172  0.289  -0.584  -0.322  -0.109  -0.095  -0.212
    0.01  -1.02993  -0.00175  -0.50466  0.231  -2.8424  0.8052  -0.04135  0.00584  2.7370  -0.5191  0.173  0.288  -0.523  -0.262  -0.100  -0.092  -0.193
    0.02  -1.08567  -0.00176  -0.48043  0.233  -2.8337  0.8383  -0.04325  0.00583  2.9087  -0.5640  0.176  0.292  -0.459  -0.208  -0.092  -0.089  -0.177
    0.03  -1.15951  -0.00176  -0.42490  0.235  -2.8235  0.8838  -0.04595  0.00586  3.0735  -0.6227  0.178  0.295  -0.390  -0.160  -0.085  -0.088  -0.164
    0.05  -1.28640  -0.00178  -0.31239  0.241  -2.7358  0.9539  -0.05033  0.00621  3.2147  -0.7079  0.190  0.307  -0.306  -0.088  -0.075  -0.090  -0.146
    0.07  -1.34644  -0.00181  -0.17995  0.251  -2.6004  0.9808  -0.05225  0.00603  3.0851  -0.7425  0.213  0.329  -0.351  -0.056  -0.069  -0.096  -0.141
    0.10  -1.32353  -0.00182  -0.13208  0.255  -2.4891  0.9544  -0.05060  0.00571  2.8091  -0.7055  0.195  0.321  -0.524  -0.087  -0.070  -0.113  -0.156
    0.15  -1.17687  -0.00183  -0.26451  0.255  -2.6505  0.9232  -0.04879  0.00560  2.6260  -0.6270  0.160  0.302  -0.691  -0.336  -0.095  -0.166  -0.245
    0.20  -1.04508  -0.00182  -0.39105  0.268  -3.0096  0.9426  -0.05034  0.00573  2.6063  -0.5976  0.157  0.310  -0.671  -0.547  -0.127  -0.209  -0.359
    0.25  -0.94363  -0.00178  -0.34348  0.264  -3.3321  0.9578  -0.05143  0.00507  2.3654  -0.5820  0.142  0.299  -0.584  -0.674  -0.178  -0.235  -0.444
    0.30  -0.84814  -0.00173  -0.36695  0.260  -3.5422  0.9441  -0.05052  0.00428  2.2017  -0.5412  0.141  0.296  -0.506  -0.730  -0.258  -0.234  -0.491
    0.40  -0.69278  -0.00166  -0.46301  0.263  -3.3985  0.7773  -0.03885  0.00308  1.6367  -0.3448  0.157  0.306  -0.386  -0.718  -0.423  -0.164  -0.535
    0.50  -0.57899  -0.00161  -0.54098  0.261  -2.8041  0.5069  -0.01973  0.00257  0.7621  -0.0617  0.152  0.302  -0.300  -0.635  -0.537  -0.110  -0.557
    0.75  -0.56887  -0.00158  -0.46266  0.252  -4.4588  0.8691  -0.04179  0.00135  2.1003  -0.4349  0.146  0.291  -0.276  -0.395  -0.575  -0.358  -0.599
    1.00  -0.53282  -0.00154  -0.42314  0.247  -5.3391  1.0167  -0.04999  0.00045  2.5610  -0.5678  0.153  0.290  -0.275  -0.254  -0.462  -0.670  -0.584
    1.50  -0.46263  -0.00145  -0.58519  0.246  -6.1204  1.1005  -0.05426  0.00068  2.8923  -0.5898  0.152  0.289  -0.249  -0.238  -0.300  -0.801  -0.522
    2.00  -0.40594  -0.00139  -0.65999  0.245  -7.0334  1.2501  -0.06356  0.00051  3.3941  -0.7009  0.157  0.291  -0.218  -0.231  -0.220  -0.746  -0.479
    3.00  -0.33957  -0.00137  -0.79004  0.231  -8.2507  1.4652  -0.07797  0.00066  4.0033  -0.8465  0.155  0.279  -0.180  -0.219  -0.210  -0.628  -0.461
    4.00  -0.26479  -0.00137  -0.86545  0.228  -8.7433  1.4827  -0.07863  0.00063  3.9337  -0.8134  0.160  0.279  -0.171  -0.218  -0.212  -0.531  -0.448
    5.00  -0.22333  -0.00137  -0.88735  0.232  -8.9927  1.4630  -0.07638  0.00067  3.7576  -0.7642  0.167  0.286  -0.168  -0.218  -0.203  -0.438  -0.439
    7.50  -0.30346  -0.00131  -0.91259  0.231  -9.8245  1.6383  -0.08620  0.00108  4.3948  -0.9313  0.164  0.283  -0.168  -0.218  -0.153  -0.256  -0.435
    10.00 -0.33771  -0.00117  -0.96363  0.204  -9.8671  1.5877  -0.08168  0.00014  4.3875  -0.8892  0.176  0.270  -0.168  -0.218  -0.125  -0.231  -0.435
    """)


class IdiniEtAl2017SSlab(IdiniEtAl2017SInter):
    """
    Implements GMPE developed by Idini et al. and published as
    "Ground motion prediction equations for the Chilean subduction zone"
    (Bull. Earthquake Eng.,  Volume 15, Issue 5, pp 1853-1880, 2017,
    DOI 10.1007/s10518-016-0050-1).
    The class implements the equations for 'Subduction
    IntraSlab' (that's why the class name ends with 'SSlab').
    """
    #: Supported tectonic region type is subduction interface
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTRASLAB

    Feve = 1

    def _compute_delta_fM(self, C, mag):
        # Eqn. 4
        Dc1, Dc2 = C['Dc1'], C['Dc2']
        delta_fM = Dc1 + Dc2 * mag
        return delta_fM
