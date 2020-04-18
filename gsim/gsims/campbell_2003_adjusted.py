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
Module exports :class:`Campbell2003`, :class:`Campbell2003SHARE`
"""
from __future__ import division

import numpy as np
# standard acceleration of gravity in m/s**2
from scipy.constants import g

from openquake.hazardlib.gsim.base import CoeffsTable, GMPE
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, SA


## Compute these logs only once
ln70, ln130 = np.log(70), np.log(130)
ln_g = np.log(g)

class Campbell2003adjusted(GMPE):
    """
    Implements GMPE developed by K.W Campbell and published as "Prediction of
    Strong Ground Motion Using the Hybrid Empirical Method and Its Use in the
    Development of Ground Motion (Attenuation) Relations in Eastern North
    America" (Bulletting of the Seismological Society of America, Volume 93,
    Number 3, pages 1012-1033, 2003). The class implements also the corrections
    given in the erratum (2004).
    """

    #: Supported tectonic region type is stable continental crust given that
    #: the equations have been derived for Eastern North America.
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.STABLE_CONTINENTAL

    #: Supported intensity measure types are spectral acceleration,
    #: and peak ground acceleration, see table 6, page 1022 (PGA is assumed
    #: to be equal to SA at 0.01 s)
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        PGA,
        SA
    ])

    #: Supported intensity measure component is the geometric mean of two
    #: horizontal components :attr:`~openquake.hazardlib.const.IMC.AVERAGE_HORIZONTAL`,
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.AVERAGE_HORIZONTAL

    #: Supported standard deviation type is only total, see equation 35, page
    #: 1021
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL
    ])

    #: Required site parameters is Vs30.
    #: See paragraph 'Equations for soil sites', p. 2200
    REQUIRES_SITES_PARAMETERS = set(('vs30', 'kappa'))

    #: Required rupture parameter is only magnitude, see equation 30 page
    #: 1021.
    REQUIRES_RUPTURE_PARAMETERS = set(('mag', ))

    #: Required distance measure is closest distance to rupture, see equation
    #: 30 page 1021.
    REQUIRES_DISTANCES = set(('rrup', ))

    DEFINED_FOR_VS30 = np.array([800, 2000, 2600, 2800])
    DEFINED_FOR_KAPPA = {800: np.array([0.02, 0.03, 0.05]),
                        2000: np.array([0.002, 0.006, 0.01]),
                        2600: np.array([0.002, 0.006, 0.01]),
                        2800: np.array([0.002, 0.006, 0.01])
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
            self._compute_mean(C, rup.mag, dists.rrup, idxs, mean)
        ## Coefficients for standard deviations are independent of (vs30, kappa)
        stddevs = self._get_stddevs(C, stddev_types, rup.mag,
                                    dists.rrup.shape[0])

        # convert mean in m/s2 to mean in g
        mean = mean - ln_g

        return mean, stddevs

    def _compute_mean(self, C, mag, rrup, idxs, mean):
        """
        Compute mean value according to equation 30, page 1021.
        """
        mean[idxs] = (C['c1'] +
                self._compute_term1(C, mag) +
                self._compute_term2(C, mag, rrup[idxs]) +
                self._compute_term3(C, rrup[idxs]))
        #return mean

    def _get_stddevs(self, C, stddev_types, mag, num_sites):
        """
        Return total standard deviation as for equation 35, page 1021.
        """
        stddevs = []
        for stddev_type in stddev_types:
            if mag < 7.16:
                sigma = C['c11'] + C['c12'] * mag
            elif mag >= 7.16:
                sigma = C['c13']
            stddevs.append(np.zeros(num_sites) + sigma)

        return stddevs

    def _compute_term1(self, C, mag):
        """
        This computes the term f1 in equation 31, page 1021
        """
        x = 8.5 - mag
        #return (C['c2'] * mag) + C['c3'] * (8.5 - mag) ** 2
        return (C['c2'] * mag) + C['c3'] * x * x

    def _compute_term2(self, C, mag, rrup):
        """
        This computes the term f2 in equation 32, page 1021
        """
        x = (C['c7'] * np.exp(C['c8'] * mag))
        c78_factor = x * x
        #c78_factor = (C['c7'] * np.exp(C['c8'] * mag)) ** 2
        #R = np.sqrt(rrup ** 2 + c78_factor)
        R = np.sqrt(rrup * rrup + c78_factor)

        return C['c4'] * np.log(R) + (C['c5'] + C['c6'] * mag) * rrup

    def _compute_term3(self, C, rrup):
        """
        This computes the term f3 in equation 34, page 1021 but corrected
        according to the erratum.
        """
        f3 = np.zeros_like(rrup)

        idx_between_70_130 = (rrup > 70) & (rrup <= 130)
        idx_greater_130 = rrup > 130

        f3[idx_between_70_130] = (
            #C['c9'] * (np.log(rrup[idx_between_70_130]) - np.log(70))
            C['c9'] * (np.log(rrup[idx_between_70_130]) - ln70)
        )

        f3[idx_greater_130] = (
            #C['c9'] * (np.log(rrup[idx_greater_130]) - np.log(70)) +
            #C['c10'] * (np.log(rrup[idx_greater_130]) - np.log(130))
            C['c9'] * (np.log(rrup[idx_greater_130]) - ln70) +
            C['c10'] * (np.log(rrup[idx_greater_130]) - ln130)
        )

        return f3

    #: Coefficient tables are constructed from the electronic suplements of
    #: the original paper.
    COEFFS = {}
    COEFFS[(2800, 0.002)] = CoeffsTable(sa_damping=5, table="""\
    IMT       c1        c2        c3        c4        c5          c6          c7       c8       c9       c10       c11       c12       c13
    pga       3.1852    0.819     -0.0211   -2.122    -0.00227    0.000297    0.996    0.378    1.940    -1.564    1.030    -0.0860    0.414
    0.028     3.8205    0.791     -0.0208   -2.048    -0.00144    0.000294    1.112    0.366    1.540    -1.913    1.030    -0.0860    0.414
    0.040     3.1677    0.782     -0.0209   -1.879    -0.00189    0.000289    0.913    0.393    1.603    -2.005    1.036    -0.0849    0.429
    0.100     2.1332    0.759     -0.0243   -1.648    -0.00287    0.000233    0.628    0.448    1.829    -1.852    1.059    -0.0838    0.460
    0.200     2.0433    0.715     -0.0544   -1.579    -0.00286    0.000175    0.572    0.456    1.834    -1.485    1.077    -0.0838    0.478
    0.400     1.9062    0.622     -0.1113   -1.448    -0.00234    0.000135    0.510    0.452    1.614    -1.168    1.089    -0.0831    0.495
    1.000     1.6656    0.513     -0.1983   -1.307    -0.00167    0.000127    0.501    0.440    1.284    -0.865    1.110    -0.0793    0.543
    2.000     1.0483    0.478     -0.2496   -1.232    -0.00115    0.000115    0.472    0.450    1.094    -0.736    1.093    -0.0758    0.551
    """)

    COEFFS[(2800, 0.006)] = CoeffsTable(sa_damping=5, table="""\
    IMT       c1        c2        c3        c4        c5          c6          c7       c8       c9       c10       c11       c12       c13
    pga       2.2542    0.805     -0.0276   -1.941    -0.00237    0.000284    0.844    0.405    1.967    -1.622    1.030    -0.0860    0.414
    0.028     3.3606    0.788     -0.0231   -2.018    -0.00182    0.000301    1.048    0.375    1.622    -1.724    1.030    -0.0860    0.414
    0.040     2.8863    0.777     -0.0228   -1.869    -0.00201    0.000300    0.889    0.398    1.616    -1.874    1.036    -0.0849    0.429
    0.100     2.0642    0.753     -0.0256   -1.651    -0.00285    0.000251    0.635    0.446    1.821    -1.855    1.059    -0.0838    0.460
    0.200     1.9855    0.715     -0.0544   -1.581    -0.00287    0.000182    0.572    0.456    1.834    -1.489    1.077    -0.0838    0.478
    0.400     1.8338    0.630     -0.1099   -1.452    -0.00234    0.000137    0.503    0.455    1.621    -1.173    1.089    -0.0831    0.495
    1.000     1.7140    0.508     -0.1997   -1.311    -0.00166    0.000126    0.493    0.443    1.291    -0.869    1.110    -0.0793    0.543
    2.000     1.2271    0.452     -0.2554   -1.230    -0.00112    0.000110    0.475    0.448    1.090    -0.733    1.093    -0.0758    0.551
    """)

    COEFFS[(2800, 0.010)] = CoeffsTable(sa_damping=5, table="""\
    IMT       c1        c2        c3        c4        c5          c6          c7       c8       c9       c10       c11       c12       c13
    pga       1.8864    0.791     -0.0335   -1.868    -0.00242    0.000275    0.783    0.417    2.000    -1.630    1.030    -0.0860    0.414
    0.028     2.9087    0.785     -0.0260   -1.981    -0.00215    0.000302    0.973    0.387    1.732    -1.607    1.030    -0.0860    0.414
    0.040     2.6055    0.773     -0.0251   -1.856    -0.00217    0.000308    0.858    0.403    1.647    -1.755    1.036    -0.0849    0.429
    0.100     1.9983    0.747     -0.0270   -1.654    -0.00283    0.000270    0.641    0.445    1.814    -1.855    1.059    -0.0838    0.460
    0.200     1.9374    0.713     -0.0548   -1.581    -0.00288    0.000190    0.574    0.455    1.832    -1.492    1.077    -0.0838    0.478
    0.400     1.7849    0.633     -0.1093   -1.454    -0.00235    0.000138    0.500    0.456    1.624    -1.175    1.089    -0.0831    0.495
    1.000     1.7366    0.504     -0.2005   -1.314    -0.00166    0.000125    0.487    0.445    1.296    -0.871    1.110    -0.0793    0.543
    2.000     1.3595    0.433     -0.2599   -1.228    -0.00110    0.000106    0.478    0.446    1.087    -0.731    1.093    -0.0758    0.551
    """)

    COEFFS[(2600, 0.002)] = CoeffsTable(sa_damping=5, table="""\
    IMT       c1        c2        c3        c4        c5          c6          c7       c8       c9       c10       c11       c12       c13
    pga       3.1852    0.819     -0.0211   -2.122    -0.00227    0.000297    0.996    0.378    1.940    -1.564    1.030    -0.0860    0.414
    0.028     3.8205    0.791     -0.0208   -2.048    -0.00144    0.000294    1.112    0.366    1.540    -1.913    1.030    -0.0860    0.414
    0.040     3.1677    0.782     -0.0209   -1.879    -0.00189    0.000289    0.913    0.393    1.603    -2.005    1.036    -0.0849    0.429
    0.100     2.1332    0.759     -0.0243   -1.648    -0.00287    0.000233    0.628    0.448    1.829    -1.852    1.059    -0.0838    0.460
    0.200     2.0433    0.715     -0.0544   -1.579    -0.00286    0.000175    0.572    0.456    1.834    -1.485    1.077    -0.0838    0.478
    0.400     1.9062    0.622     -0.1113   -1.448    -0.00234    0.000135    0.510    0.452    1.614    -1.168    1.089    -0.0831    0.495
    1.000     1.6656    0.513     -0.1983   -1.307    -0.00167    0.000127    0.501    0.440    1.284    -0.865    1.110    -0.0793    0.543
    2.000     1.0483    0.478     -0.2496   -1.232    -0.00115    0.000115    0.472    0.450    1.094    -0.736    1.093    -0.0758    0.551
    """)

    COEFFS[(2600, 0.006)] = CoeffsTable(sa_damping=5, table="""\
    IMT       c1        c2        c3        c4        c5          c6          c7       c8       c9       c10       c11       c12       c13
    pga       2.2542    0.805     -0.0276   -1.941    -0.00237    0.000284    0.844    0.405    1.967    -1.622    1.030    -0.0860    0.414
    0.028     3.3606    0.788     -0.0231   -2.018    -0.00182    0.000301    1.048    0.375    1.622    -1.724    1.030    -0.0860    0.414
    0.040     2.8863    0.777     -0.0228   -1.869    -0.00201    0.000300    0.889    0.398    1.616    -1.874    1.036    -0.0849    0.429
    0.100     2.0642    0.753     -0.0256   -1.651    -0.00285    0.000251    0.635    0.446    1.821    -1.855    1.059    -0.0838    0.460
    0.200     1.9855    0.715     -0.0544   -1.581    -0.00287    0.000182    0.572    0.456    1.834    -1.489    1.077    -0.0838    0.478
    0.400     1.8338    0.630     -0.1099   -1.452    -0.00234    0.000137    0.503    0.455    1.621    -1.173    1.089    -0.0831    0.495
    1.000     1.7140    0.508     -0.1997   -1.311    -0.00166    0.000126    0.493    0.443    1.291    -0.869    1.110    -0.0793    0.543
    2.000     1.2271    0.452     -0.2554   -1.230    -0.00112    0.000110    0.475    0.448    1.090    -0.733    1.093    -0.0758    0.551
    """)

    COEFFS[(2600, 0.010)] = CoeffsTable(sa_damping=5, table="""\
    IMT       c1        c2        c3        c4        c5          c6          c7       c8       c9       c10       c11       c12       c13
    pga       1.8864    0.791     -0.0335   -1.868    -0.00242    0.000275    0.783    0.417    2.000    -1.630    1.030    -0.0860    0.414
    0.028     2.9087    0.785     -0.0260   -1.981    -0.00215    0.000302    0.973    0.387    1.732    -1.607    1.030    -0.0860    0.414
    0.040     2.6055    0.773     -0.0251   -1.856    -0.00217    0.000308    0.858    0.403    1.647    -1.755    1.036    -0.0849    0.429
    0.100     1.9983    0.747     -0.0270   -1.654    -0.00283    0.000270    0.641    0.445    1.814    -1.855    1.059    -0.0838    0.460
    0.200     1.9374    0.713     -0.0548   -1.581    -0.00288    0.000190    0.574    0.455    1.832    -1.492    1.077    -0.0838    0.478
    0.400     1.7849    0.633     -0.1093   -1.454    -0.00235    0.000138    0.500    0.456    1.624    -1.175    1.089    -0.0831    0.495
    1.000     1.7366    0.504     -0.2005   -1.314    -0.00166    0.000125    0.487    0.445    1.296    -0.871    1.110    -0.0793    0.543
    2.000     1.3595    0.433     -0.2599   -1.228    -0.00110    0.000106    0.478    0.446    1.087    -0.731    1.093    -0.0758    0.551
    """)

    COEFFS[(2000, 0.002)] = CoeffsTable(sa_damping=5, table="""\
    IMT       c1        c2        c3        c4        c5          c6          c7       c8       c9       c10       c11       c12       c13
    pga       3.4195    0.821     -0.0198   -2.140    -0.00218    0.000283    1.025    0.373    1.920    -1.567    1.030    -0.0860    0.414
    0.028     4.0055    0.795     -0.0199   -2.050    -0.00131    0.000276    1.118    0.365    1.535    -1.970    1.030    -0.0860    0.414
    0.040     3.3297    0.786     -0.0200   -1.879    -0.00182    0.000270    0.913    0.394    1.608    -2.046    1.036    -0.0849    0.429
    0.100     2.2503    0.765     -0.0235   -1.645    -0.00287    0.000211    0.620    0.450    1.837    -1.845    1.059    -0.0838    0.460
    0.200     2.1615    0.716     -0.0541   -1.578    -0.00284    0.000164    0.569    0.457    1.836    -1.480    1.077    -0.0838    0.478
    0.400     2.0057    0.622     -0.1112   -1.447    -0.00233    0.000132    0.510    0.452    1.614    -1.167    1.089    -0.0831    0.495
    1.000     1.6773    0.522     -0.1961   -1.308    -0.00168    0.000128    0.499    0.441    1.286    -0.867    1.110    -0.0793    0.543
    2.000     0.9599    0.500     -0.2440   -1.240    -0.00117    0.000120    0.463    0.455    1.105    -0.743    1.093    -0.0758    0.551
    """)

    COEFFS[(2000, 0.006)] = CoeffsTable(sa_damping=5, table="""\
    IMT       c1        c2        c3        c4        c5          c6          c7       c8       c9       c10       c11       c12       c13
    pga       2.4359    0.809     -0.0258   -1.952    -0.00231    0.000272    0.861    0.402    1.954    -1.629    1.030    -0.0860    0.414
    0.028     3.5486    0.791     -0.0221   -2.022    -0.00168    0.000284    1.059    0.373    1.606    -1.770    1.030    -0.0860    0.414
    0.040     3.0519    0.781     -0.0218   -1.870    -0.00193    0.000282    0.892    0.397    1.616    -1.916    1.036    -0.0849    0.429
    0.100     2.1810    0.758     -0.0247   -1.648    -0.00284    0.000230    0.627    0.448    1.829    -1.850    1.059    -0.0838    0.460
    0.200     2.1018    0.717     -0.0541   -1.580    -0.00285    0.000171    0.569    0.457    1.836    -1.484    1.077    -0.0838    0.478
    0.400     1.9349    0.629     -0.1098   -1.451    -0.00234    0.000134    0.503    0.455    1.621    -1.172    1.089    -0.0831    0.495
    1.000     1.7373    0.514     -0.1980   -1.312    -0.00167    0.000127    0.492    0.444    1.293    -0.870    1.110    -0.0793    0.543
    2.000     1.1442    0.473     -0.2502   -1.236    -0.00115    0.000115    0.468    0.452    1.100    -0.739    1.093    -0.0758    0.551
    """)

    COEFFS[(2000, 0.010)] = CoeffsTable(sa_damping=5, table="""\
    IMT       c1        c2        c3        c4        c5          c6          c7       c8       c9       c10       c11       c12       c13
    pga       2.0418    0.795     -0.0315   -1.877    -0.00237    0.000265    0.795    0.414    1.989    -1.638    1.030    -0.0860    0.414
    0.028     3.0990    0.787     -0.0248   -1.987    -0.00202    0.000287    0.989    0.384    1.709    -1.637    1.030    -0.0860    0.414
    0.040     2.7734    0.777     -0.0240   -1.858    -0.00207    0.000291    0.864    0.402    1.641    -1.794    1.036    -0.0849    0.429
    0.100     2.1146    0.752     -0.0260   -1.651    -0.00281    0.000248    0.634    0.446    1.821    -1.852    1.059    -0.0838    0.460
    0.200     2.0527    0.715     -0.0544   -1.580    -0.00286    0.000178    0.571    0.456    1.835    -1.487    1.077    -0.0838    0.478
    0.400     1.8855    0.633     -0.1092   -1.453    -0.00234    0.000135    0.500    0.456    1.624    -1.174    1.089    -0.0831    0.495
    1.000     1.7684    0.509     -0.1991   -1.314    -0.00166    0.000126    0.488    0.445    1.296    -0.872    1.110    -0.0793    0.543
    2.000     1.2829    0.452     -0.2550   -1.233    -0.00113    0.000111    0.472    0.450    1.096    -0.736    1.093    -0.0758    0.551
    """)

    COEFFS[(800, 0.020)] = CoeffsTable(sa_damping=5, table="""\
    IMT       c1        c2        c3        c4        c5          c6          c7       c8       c9       c10       c11       c12       c13
    pga       2.0736    0.776     -0.0376   -1.805    -0.00225    0.000210    0.744    0.423    2.022    -1.640    1.030    -0.0860    0.414
    0.028     2.6986    0.783     -0.0297   -1.899    -0.00221    0.000223    0.844    0.408    1.904    -1.576    1.030    -0.0860    0.414
    0.040     2.6558    0.777     -0.0267   -1.824    -0.00211    0.000240    0.796    0.415    1.722    -1.655    1.036    -0.0849    0.429
    0.100     2.3586    0.753     -0.0265   -1.649    -0.00268    0.000216    0.624    0.449    1.828    -1.838    1.059    -0.0838    0.460
    0.200     2.3632    0.717     -0.0542   -1.579    -0.00280    0.000156    0.564    0.458    1.840    -1.478    1.077    -0.0838    0.478
    0.400     2.1661    0.637     -0.1080   -1.455    -0.00231    0.000127    0.496    0.457    1.629    -1.174    1.089    -0.0831    0.495
    1.000     1.9029    0.523     -0.1954   -1.321    -0.00167    0.000128    0.479    0.449    1.307    -0.878    1.110    -0.0793    0.543
    2.000     1.2437    0.486     -0.2458   -1.250    -0.00118    0.000121    0.456    0.457    1.120    -0.751    1.093    -0.0758    0.551
    """)

    COEFFS[(800, 0.030)] = CoeffsTable(sa_damping=5, table="""\
    IMT       c1        c2        c3        c4        c5          c6          c7       c8       c9       c10       c11       c12       c13
    pga       1.8740    0.746     -0.0485   -1.760    -0.00225    0.000204    0.710    0.430    2.048    -1.614    1.030    -0.0860    0.414
    0.028     2.0672    0.752     -0.0427   -1.793    -0.00236    0.000206    0.715    0.433    2.036    -1.565    1.030    -0.0860    0.414
    0.040     2.0876    0.758     -0.0357   -1.769    -0.00237    0.000237    0.703    0.435    1.846    -1.536    1.036    -0.0849    0.429
    0.100     2.2006    0.737     -0.0302   -1.655    -0.00263    0.000255    0.636    0.446    1.812    -1.832    1.059    -0.0838    0.460
    0.200     2.2518    0.712     -0.0554   -1.580    -0.00282    0.000173    0.569    0.457    1.836    -1.485    1.077    -0.0838    0.478
    0.400     2.0873    0.638     -0.1081   -1.456    -0.00232    0.000130    0.495    0.458    1.630    -1.175    1.089    -0.0831    0.495
    1.000     1.9377    0.515     -0.1975   -1.323    -0.00166    0.000126    0.475    0.450    1.310    -0.879    1.110    -0.0793    0.543
    2.000     1.4741    0.450     -0.2545   -1.242    -0.00114    0.000113    0.463    0.453    1.108    -0.743    1.093    -0.0758    0.551
    """)

    COEFFS[(800, 0.050)] = CoeffsTable(sa_damping=5, table="""\
    IMT       c1        c2        c3        c4        c5          c6          c7       c8       c9       c10       c11       c12       c13
    pga       1.7367    0.696     -0.0677   -1.712    -0.00219    0.000195    0.680    0.435    2.067    -1.572    1.030    -0.0860    0.414
    0.028     1.7176    0.684     -0.0677   -1.687    -0.00233    0.000181    0.647    0.445    2.084    -1.529    1.030    -0.0860    0.414
    0.040     1.5379    0.699     -0.0608   -1.660    -0.00240    0.000203    0.586    0.461    1.969    -1.471    1.036    -0.0849    0.429
    0.100     1.8933    0.707     -0.0391   -1.660    -0.00259    0.000317    0.640    0.446    1.799    -1.779    1.059    -0.0838    0.460
    0.200     2.0447    0.700     -0.0580   -1.583    -0.00288    0.000213    0.581    0.453    1.827    -1.500    1.077    -0.0838    0.478
    0.400     1.9447    0.636     -0.1089   -1.457    -0.00233    0.000136    0.494    0.458    1.630    -1.177    1.089    -0.0831    0.495
    1.000     1.9418    0.508     -0.1997   -1.326    -0.00164    0.000123    0.469    0.453    1.316    -0.882    1.110    -0.0793    0.543
    2.000     1.8235    0.394     -0.2679   -1.230    -0.00107    0.000100    0.473    0.447    1.091    -0.732    1.093    -0.0758    0.551
    """)

