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
Module exports :class:`ToroEtAl2002`, class:`ToroEtAl2002SHARE`.
"""
from __future__ import division

import numpy as np
# standard acceleration of gravity in m/s**2
from scipy.constants import g

from openquake.hazardlib.gsim.campbell_2003 import _compute_faulting_style_term
from openquake.hazardlib.gsim.base import CoeffsTable, GMPE
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, SA


## Compute these logs only once
ln_g = np.log(g)


class ToroEtAl2002adjusted(GMPE):
    """
    Implements GMPE developed by G. R. Toro, N. A. Abrahamson, J. F. Sneider
    and published in "Model of Strong Ground Motions from Earthquakes in
    Central and Eastern North America: Best Estimates and Uncertainties"
    (Seismological Research Letters, Volume 68, Number 1, 1997) and
    "Modification of the Toro et al. 1997 Attenuation Equations for Large
    Magnitudes and Short Distances" (available at:
    http://www.riskeng.com/downloads/attenuation_equations)
    The class implements equations for Midcontinent, based on moment magnitude.
    SA at 3 and 4 s (not supported by the original equations) have been added
    in the context of the SHARE project and they are obtained from SA at 2 s
    scaled by specific factors for 3 and 4 s.
    """

    #: Supported tectonic region type is stable continental crust,
    #: given that the equations have been derived for central and eastern
    #: north America
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.STABLE_CONTINENTAL

    #: Supported intensity measure types are spectral acceleration,
    #: and peak ground acceleration, see table 2 page 47.
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        PGA,
        SA
    ])

    #: Supported intensity measure component is the geometric mean of two
    #: horizontal components :attr:`~openquake.hazardlib.const.IMC.AVERAGE_HORIZONTAL`,
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.AVERAGE_HORIZONTAL

    #: Supported standard deviation type is only total.
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL
    ])

    #: Required site parameters is Vs30.
    #: See paragraph 'Equations for soil sites', p. 2200
    REQUIRES_SITES_PARAMETERS = set(('vs30', 'kappa'))

    #: Required rupture parameter is only magnitude.
    REQUIRES_RUPTURE_PARAMETERS = set(('mag', ))

    #: Required distance measure is rjb, see equation 4, page 46.
    REQUIRES_DISTANCES = set(('rjb', ))

    DEFINED_FOR_VS30 = np.array([800, 2000, 2600, 2800])
    DEFINED_FOR_KAPPA = {800: np.array([0.02, 0.03, 0.05]),
                        2000: np.array([0.002, 0.005, 0.01]),
                        2600: np.array([0.002, 0.005, 0.01]),
                        2800: np.array([0.002, 0.005, 0.01])
                        }

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        assert all(stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
                   for stddev_type in stddev_types)

        ## Allow different (vs30, kappa) for different sites
        mean = np.zeros_like(sites.vs30)
        vs30_kappa = set(zip(sites.vs30, sites.kappa))
        for (vs30, kappa) in vs30_kappa:
            ## Determine nearest vs30 and kappa that is defined
            nearest_vs30 = self.DEFINED_FOR_VS30[np.abs(self.DEFINED_FOR_VS30 - vs30).argmin()]
            nearest_kappa = self.DEFINED_FOR_KAPPA[nearest_vs30][np.abs(self.DEFINED_FOR_KAPPA[nearest_vs30] - kappa).argmin()]
            C = self.COEFFS[(nearest_vs30, nearest_kappa)][imt]
            idxs = (sites.vs30 == vs30) * (sites.kappa == kappa)
            self._compute_mean(C, rup.mag, dists.rjb, idxs, mean)
        ## Coefficients for standard deviations are independent of (vs30, kappa)
        stddevs = self._compute_stddevs(C, rup.mag, dists.rjb, imt,
                                        stddev_types)

        # apply decay factor for 3 and 4 seconds (not originally supported
        # by the equations)
        if isinstance(imt, SA):
            if imt.period == 3.0:
                mean /= 0.612
            if imt.period == 4.0:
                mean /= 0.559

        # convert mean in m/s2 to mean in g
        mean = mean - ln_g

        return mean, stddevs

    def _compute_term1(self, C, mag):
        """
        Compute magnitude dependent terms (2nd and 3rd) in equation 3
        page 46.
        """
        mag_diff = mag - 6

        #return C['c2'] * mag_diff + C['c3'] * mag_diff ** 2
        return C['c2'] * mag_diff + C['c3'] * mag_diff * mag_diff

    def _compute_term2(self, C, mag, rjb):
        """
        Compute distance dependent terms (4th, 5th and 6th) in equation 3
        page 46. The factor 'RM' is computed according to the 2002 model
        (equation 4-3).
        """
        x = np.exp(-1.25 + 0.227 * mag)
        RM = np.sqrt(rjb * rjb + (C['c7'] * C['c7']) * x * x)
        #RM = np.sqrt(rjb ** 2 + (C['c7'] ** 2) *
        #             np.exp(-1.25 + 0.227 * mag) ** 2)

        return (-C['c4'] * np.log(RM) -
                (C['c5'] - C['c4']) *
                np.maximum(np.log(RM / 100), 0) - C['c6'] * RM)

    def _compute_mean(self, C, mag, rjb, idxs, mean):
        """
        Compute mean value according to equation 3, page 46.
        """
        mean[idxs] = (C['c1'] +
                self._compute_term1(C, mag) +
                self._compute_term2(C, mag, rjb[idxs].clip(min=1)))
        #return mean

    def _compute_stddevs(self, C, mag, rjb, imt, stddev_types):
        """
        Compute total standard deviation, equations 5 and 6, page 48.
        """
        # aleatory uncertainty
        sigma_ale_m = np.interp(mag, [5.0, 5.5, 8.0],
                                [C['m50'], C['m55'], C['m80']])
        sigma_ale_rjb = np.interp(rjb, [5.0, 20.0], [C['r5'], C['r20']])
        sigma_ale = np.sqrt(sigma_ale_m ** 2 + sigma_ale_rjb ** 2)

        # epistemic uncertainty
        if isinstance(imt, PGA) or (isinstance(imt, SA) and imt.period < 1):
            sigma_epi = 0.36 + 0.07 * (mag - 6)
        else:
            sigma_epi = 0.34 + 0.06 * (mag - 6)

        sigma_total = np.sqrt(sigma_ale ** 2 + sigma_epi ** 2)

        stddevs = []
        for _ in stddev_types:
            stddevs.append(sigma_total)

        return stddevs

    #: Coefficient table: factors c1 to c7 replaced with the values
    #: in Table 20 in the report of Drouet et al. (2010), p. 35
    COEFFS = {}
    COEFFS[(2800, 0.002)] = CoeffsTable(sa_damping=5, table="""\
    IMT      c1    c2    c3    c4    c5    c6      c7   m50   m55   m80   r5    r20
    pga      5.96  0.74  0.03  1.49  1.05  0.0023  9.0  0.55  0.59  0.50  0.54  0.20
    0.028    6.65  0.76  0.02  1.45  1.82  0.0014  9.4  0.62  0.63  0.50  0.62  0.35
    0.040    6.15  0.76  0.02  1.36  1.52  0.0021  8.9  0.62  0.63  0.50  0.57  0.29
    0.100    4.62  0.78  0.01  1.02  0.42  0.0052  7.1  0.59  0.61  0.50  0.50  0.17
    0.200    3.96  0.83  0.00  0.93  0.13  0.0049  6.6  0.60  0.64  0.56  0.45  0.12
    0.400    3.27  1.05  -0.10  0.89  0.16  0.0036  6.4  0.63  0.68  0.64  0.45  0.12
    1.000    2.29  1.41  -0.19  0.87  0.20  0.0025  6.3  0.63  0.64  0.67  0.45  0.12
    2.000    1.52  1.83  -0.29  0.91  0.22  0.0018  6.7  0.61  0.62  0.66  0.45  0.12
    """)

    COEFFS[(2800, 0.005)] = CoeffsTable(sa_damping=5, table="""\
    IMT      c1    c2    c3    c4    c5    c6      c7   m50   m55   m80   r5    r20
    pga      4.99  0.76  0.02  1.32  0.85  0.0025  8.5  0.55  0.59  0.50  0.54  0.20
    0.028    6.29  0.76  0.02  1.43  1.63  0.0015  9.3  0.62  0.63  0.50  0.62  0.35
    0.040    5.93  0.77  0.01  1.36  1.43  0.0021  8.9  0.62  0.63  0.50  0.57  0.29
    0.100    4.54  0.79  0.01  1.02  0.43  0.0051  7.1  0.59  0.61  0.50  0.50  0.17
    0.200    3.92  0.83  0.00  0.93  0.14  0.0049  6.6  0.60  0.64  0.56  0.45  0.12
    0.400    3.25  1.05  -0.10  0.89  0.16  0.0036  6.4  0.63  0.68  0.64  0.45  0.12
    1.000    2.28  1.41  -0.19  0.87  0.20  0.0025  6.3  0.63  0.64  0.67  0.45  0.12
    2.000    1.52  1.84  -0.29  0.91  0.22  0.0018  6.7  0.61  0.62  0.66  0.45  0.12
    """)

    COEFFS[(2800, 0.010)] = CoeffsTable(sa_damping=5, table="""\
    IMT      c1    c2    c3    c4    c5    c6      c7   m50   m55   m80   r5    r20
    pga      4.28  0.78  0.01  1.21  0.67  0.0028  8.1  0.55  0.59  0.50  0.54  0.20
    0.028    5.64  0.77  0.01  1.38  1.31  0.0019  9.1  0.62  0.63  0.50  0.62  0.35
    0.040    5.54  0.78  0.01  1.34  1.25  0.0021  8.9  0.62  0.63  0.50  0.57  0.29
    0.100    4.41  0.79  0.01  1.03  0.43  0.0049  7.2  0.59  0.61  0.50  0.50  0.17
    0.200    3.85  0.83  0.00  0.93  0.15  0.0048  6.7  0.60  0.64  0.56  0.45  0.12
    0.400    3.22  1.05  -0.10  0.89  0.17  0.0036  6.4  0.63  0.68  0.64  0.45  0.12
    1.000    2.28  1.41  -0.19  0.88  0.19  0.0025  6.3  0.63  0.64  0.67  0.45  0.12
    2.000    1.52  1.84  -0.30  0.91  0.22  0.0018  6.7  0.61  0.62  0.66  0.45  0.12
    """)

    COEFFS[(2600, 0.002)] = CoeffsTable(sa_damping=5, table="""\
    IMT      c1    c2    c3    c4    c5    c6      c7   m50   m55   m80   r5    r20
    pga      5.96  0.74  0.03  1.49  1.05  0.0023  9.0  0.55  0.59  0.50  0.54  0.20
    0.028    6.65  0.76  0.02  1.45  1.82  0.0014  9.4  0.62  0.63  0.50  0.62  0.35
    0.040    6.15  0.76  0.02  1.36  1.52  0.0021  8.9  0.62  0.63  0.50  0.57  0.29
    0.100    4.62  0.78  0.01  1.02  0.42  0.0052  7.1  0.59  0.61  0.50  0.50  0.17
    0.200    3.96  0.83  0.00  0.93  0.13  0.0049  6.6  0.60  0.64  0.56  0.45  0.12
    0.400    3.27  1.05  -0.10  0.89  0.16  0.0036  6.4  0.63  0.68  0.64  0.45  0.12
    1.000    2.29  1.41  -0.19  0.87  0.20  0.0025  6.3  0.63  0.64  0.67  0.45  0.12
    2.000    1.52  1.83  -0.29  0.91  0.22  0.0018  6.7  0.61  0.62  0.66  0.45  0.12
    """)

    COEFFS[(2600, 0.005)] = CoeffsTable(sa_damping=5, table="""\
    IMT      c1    c2    c3    c4    c5    c6      c7   m50   m55   m80   r5    r20
    pga      4.99  0.76  0.02  1.32  0.85  0.0025  8.5  0.55  0.59  0.50  0.54  0.20
    0.028    6.29  0.76  0.02  1.43  1.63  0.0015  9.3  0.62  0.63  0.50  0.62  0.35
    0.040    5.93  0.77  0.01  1.36  1.43  0.0021  8.9  0.62  0.63  0.50  0.57  0.29
    0.100    4.54  0.79  0.01  1.02  0.43  0.0051  7.1  0.59  0.61  0.50  0.50  0.17
    0.200    3.92  0.83  0.00  0.93  0.14  0.0049  6.6  0.60  0.64  0.56  0.45  0.12
    0.400    3.25  1.05  -0.10  0.89  0.16  0.0036  6.4  0.63  0.68  0.64  0.45  0.12
    1.000    2.28  1.41  -0.19  0.87  0.20  0.0025  6.3  0.63  0.64  0.67  0.45  0.12
    2.000    1.52  1.84  -0.29  0.91  0.22  0.0018  6.7  0.61  0.62  0.66  0.45  0.12
    """)

    COEFFS[(2600, 0.010)] = CoeffsTable(sa_damping=5, table="""\
    IMT      c1    c2    c3    c4    c5    c6      c7   m50   m55   m80   r5    r20
    pga      4.28  0.78  0.01  1.21  0.67  0.0028  8.1  0.55  0.59  0.50  0.54  0.20
    0.028    5.64  0.77  0.01  1.38  1.31  0.0019  9.1  0.62  0.63  0.50  0.62  0.35
    0.040    5.54  0.78  0.01  1.34  1.25  0.0021  8.9  0.62  0.63  0.50  0.57  0.29
    0.100    4.41  0.79  0.01  1.03  0.43  0.0049  7.2  0.59  0.61  0.50  0.50  0.17
    0.200    3.85  0.83  0.00  0.93  0.15  0.0048  6.7  0.60  0.64  0.56  0.45  0.12
    0.400    3.22  1.05  -0.10  0.89  0.17  0.0036  6.4  0.63  0.68  0.64  0.45  0.12
    1.000    2.28  1.41  -0.19  0.88  0.19  0.0025  6.3  0.63  0.64  0.67  0.45  0.12
    2.000    1.52  1.84  -0.30  0.91  0.22  0.0018  6.7  0.61  0.62  0.66  0.45  0.12
    """)

    COEFFS[(2000, 0.002)] = CoeffsTable(sa_damping=5, table="""\
    IMT      c1    c2    c3    c4    c5    c6      c7   m50   m55   m80   r5    r20
    pga      6.24  0.73  0.03  1.52  1.11  0.0022  9.0  0.55  0.59  0.50  0.54  0.20
    0.028    6.84  0.75  0.02  1.44  1.87  0.0014  9.3  0.62  0.63  0.50  0.62  0.35
    0.040    6.33  0.76  0.02  1.35  1.55  0.0022  8.8  0.62  0.63  0.50  0.57  0.29
    0.100    4.77  0.78  0.01  1.01  0.41  0.0053  7.1  0.59  0.61  0.50  0.50  0.17
    0.200    4.09  0.82  0.00  0.93  0.11  0.0050  6.6  0.60  0.64  0.56  0.45  0.12
    0.400    3.37  1.05  -0.11  0.89  0.16  0.0036  6.4  0.63  0.68  0.64  0.45  0.12
    1.000    2.36  1.41  -0.18  0.87  0.20  0.0025  6.3  0.63  0.64  0.67  0.45  0.12
    2.000    1.58  1.83  -0.28  0.91  0.22  0.0018  6.7  0.61  0.62  0.66  0.45  0.12
    """)

    COEFFS[(2000, 0.005)] = CoeffsTable(sa_damping=5, table="""\
    IMT      c1    c2    c3    c4    c5    c6      c7   m50   m55   m80   r5    r20
    pga      5.22  0.75  0.02  1.34  0.89  0.0026  8.6  0.55  0.59  0.50  0.54  0.20
    0.028    6.49  0.76  0.02  1.43  1.69  0.0015  9.3  0.62  0.63  0.50  0.62  0.35
    0.040    6.11  0.76  0.01  1.35  1.46  0.0021  8.9  0.62  0.63  0.50  0.57  0.29
    0.100    4.69  0.78  0.01  1.02  0.41  0.0052  7.1  0.59  0.61  0.50  0.50  0.17
    0.200    4.04  0.82  0.00  0.93  0.12  0.0050  6.6  0.60  0.64  0.56  0.45  0.12
    0.400    3.35  1.05  -0.10  0.89  0.16  0.0036  6.4  0.63  0.68  0.64  0.45  0.12
    1.000    2.35  1.41  -0.19  0.87  0.20  0.0025  6.3  0.63  0.64  0.67  0.45  0.12
    2.000    1.58  1.83  -0.29  0.91  0.22  0.0018  6.7  0.61  0.62  0.66  0.45  0.12
    """)

    COEFFS[(2000, 0.010)] = CoeffsTable(sa_damping=5, table="""\
    IMT      c1    c2    c3    c4    c5    c6      c7   m50   m55   m80   r5    r20
    pga      4.48  0.78  0.02  1.23  0.71  0.0028  8.2  0.55  0.59  0.50  0.54  0.20
    0.028    5.86  0.77  0.01  1.38  1.37  0.0019  9.1  0.62  0.63  0.50  0.62  0.35
    0.040    5.73  0.77  0.01  1.34  1.30  0.0021  8.9  0.62  0.63  0.50  0.57  0.29
    0.100    4.56  0.79  0.01  1.02  0.42  0.0051  7.1  0.59  0.61  0.50  0.50  0.17
    0.200    3.97  0.83  0.00  0.93  0.13  0.0049  6.6  0.60  0.64  0.56  0.45  0.12
    0.400    3.31  1.05  -0.10  0.89  0.16  0.0036  6.4  0.63  0.68  0.64  0.45  0.12
    1.000    2.35  1.41  -0.19  0.88  0.19  0.0025  6.3  0.63  0.64  0.67  0.45  0.12
    2.000    1.57  1.84  -0.30  0.91  0.22  0.0018  6.7  0.61  0.62  0.66  0.45  0.12
    """)

    COEFFS[(800, 0.020)] = CoeffsTable(sa_damping=5, table="""\
    IMT      c1    c2    c3    c4    c5    c6      c7   m50   m55   m80   r5    r20
    pga      4.34  0.78  0.01  1.15  0.58  0.0031  7.9  0.55  0.59  0.50  0.54  0.20
    0.028    5.29  0.77  0.01  1.26  1.00  0.0025  8.5  0.62  0.63  0.50  0.62  0.35
    0.040    5.56  0.77  0.01  1.30  1.06  0.0025  8.7  0.62  0.63  0.50  0.57  0.29
    0.100    4.79  0.78  0.01  1.02  0.39  0.0052  7.1  0.59  0.61  0.50  0.50  0.17
    0.200    4.28  0.82  0.00  0.92  0.10  0.0051  6.6  0.60  0.64  0.56  0.45  0.12
    0.400    3.62  1.04  -0.10  0.89  0.14  0.0037  6.3  0.63  0.68  0.64  0.45  0.12
    1.000    2.57  1.40  -0.18  0.88  0.19  0.0025  6.3  0.63  0.64  0.67  0.45  0.12
    2.000    1.76  1.82  -0.29  0.92  0.22  0.0018  6.7  0.61  0.62  0.66  0.45  0.12
    """)

    COEFFS[(800, 0.030)] = CoeffsTable(sa_damping=5, table="""\
    IMT      c1    c2    c3    c4    c5    c6      c7   m50   m55   m80   r5    r20
    pga      3.56  0.83  0.00  1.02  0.58  0.0029  6.4  0.55  0.59  0.50  0.54  0.20
    0.028    4.02  0.85  0.00  1.06  0.84  0.0026  6.7  0.62  0.63  0.50  0.62  0.35
    0.040    4.39  0.83  0.00  1.13  0.87  0.0026  7.1  0.62  0.63  0.50  0.57  0.29
    0.100    4.20  0.81  0.01  0.95  0.47  0.0047  6.1  0.59  0.61  0.50  0.50  0.17
    0.200    3.89  0.83  0.01  0.87  0.19  0.0048  5.7  0.60  0.64  0.56  0.45  0.12
    0.400    3.36  1.05  -0.10  0.85  0.21  0.0035  5.7  0.63  0.68  0.64  0.45  0.12
    1.000    2.40  1.42  -0.20  0.84  0.24  0.0024  5.8  0.63  0.64  0.67  0.45  0.12
    2.000    1.56  1.84  -0.30  0.87  0.27  0.0017  6.1  0.61  0.62  0.66  0.45  0.12
    """)

    COEFFS[(800, 0.050)] = CoeffsTable(sa_damping=5, table="""\
    IMT      c1    c2    c3    c4    c5    c6      c7   m50   m55   m80   r5    r20
    pga      3.03  0.89  -0.02  0.97  0.45  0.0029  6.2  0.55  0.59  0.50  0.54  0.20
    0.028    3.20  0.92  -0.02  0.97  0.62  0.0027  6.2  0.62  0.63  0.50  0.62  0.35
    0.040    3.33  0.91  -0.02  1.01  0.55  0.0028  6.4  0.62  0.63  0.50  0.57  0.29
    0.100    3.67  0.84  0.00  0.95  0.40  0.0044  6.1  0.59  0.61  0.50  0.50  0.17
    0.200    3.61  0.85  0.00  0.88  0.23  0.0045  5.7  0.60  0.64  0.56  0.45  0.12
    0.400    3.21  1.05  -0.10  0.85  0.22  0.0035  5.7  0.63  0.68  0.64  0.45  0.12
    1.000    2.34  1.42  -0.20  0.84  0.24  0.0024  5.8  0.63  0.64  0.67  0.45  0.12
    2.000    1.52  1.86  -0.31  0.87  0.27  0.0017  6.0  0.61  0.62  0.66  0.45  0.12
    """)


