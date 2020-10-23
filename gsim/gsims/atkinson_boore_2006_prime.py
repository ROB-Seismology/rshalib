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
Module exports :class:`AtkinsonBoore2006Prime`.
"""
from __future__ import division

import numpy as np
# standard acceleration of gravity in m/s**2
from scipy.constants import g

from openquake.hazardlib.gsim.boore_atkinson_2008 import BooreAtkinson2008
from openquake.hazardlib.gsim.base import CoeffsTable, SitesContext
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, PGV, SA


## Compute these logs only once
log_e = np.log10(np.e)
ln10, ln100 = np.log(10), np.log(100)
ln_g = np.log(g)


class AtkinsonBoore2006Prime(BooreAtkinson2008):
    """
    Implements GMPE developed by Gail M. Atkinson and David M. Boore and
    published as "Earthquake Ground-Motion Prediction Equations for Eastern
    North America" (2006, Bulletin of the Seismological Society of America,
    Volume 96, No. 6, pages 2181-2205), including a variable stress drop.
    Implements modifications developed by Gail M. Atkinson and David M. Boore
    and published as "Modifications to Existing Ground-Motion Prediction
    Equations in Light of New Data" (2011, Bulleting of the Seismological
    Society of America, Volume 101, pages 1121-1135).
    This class extends the BooreAtkinson2008 because it uses the same soil
    amplification function. Note that in the paper, the reported soil
    amplification function is the one used in a preliminary version of the
    Boore and Atkinson 2008 GMPE, while the one that should be used is the
    one described in the final paper. See comment in:
    http://www.daveboore.com/pubs_online/ab06_gmpes_programs_and_tables.pdf
    """
    #: Supported tectonic region type is stable continental, given
    #: that the equations have been derived for Eastern North America
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.STABLE_CONTINENTAL

    #: Supported intensity measure types are spectral acceleration,
    #: peak ground velocity and peak ground acceleration, see paragraph
    #: 'Methodology and Model Parameters', p. 2182
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
        PGA,
        PGV,
        SA
    ])

    #: Supported intensity measure component is horizontal
    #: :attr:`~openquake.hazardlib.const.IMC.HORIZONTAL`,
    #: see paragraph 'Results', pag 2190, and caption to table 6, p. 2192
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.HORIZONTAL

    #: Supported standard deviation type is total, see table 6
    #: and 9, p. 2192 and 2202, respectively.
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
        const.StdDev.TOTAL
    ])

    #: Required site parameters is Vs30.
    #: See paragraph 'Equations for soil sites', p. 2200
    REQUIRES_SITES_PARAMETERS = set(('vs30', ))

    #: Required rupture parameter is magnitude (see
    #: paragraph 'Methodology and Model Parameters', p. 2182)
    REQUIRES_RUPTURE_PARAMETERS = set(('mag', ))

    #: Required distance measure is Rrup.
    #: See paragraph 'Methodology and Model Parameters', p. 2182
    REQUIRES_DISTANCES = set(('rrup', ))

    def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
        for spec of input and result values.
        """
        # extract dictionaries of coefficients specific to required
        # intensity measure type.
        C_HR = self.COEFFS_HARD_ROCK[imt]
        C_BC = self.COEFFS_BC[imt]
        C_SR = self.COEFFS_SOIL_RESPONSE[imt]
        C_SD = self.COEFFS_STRESS_DROP[imt]

        # clip distances to avoid singularity at 0
        rrup = self._clip_distances(dists)

        # compute factors required for mean value calculation
        f0 = self._compute_f0_factor(rrup)
        f1 = self._compute_f1_factor(rrup)
        f2 = self._compute_f2_factor(rrup)

        # compute pga for BC boundary (required for soil amplification
        # calculation on non-hard-rock sites)
        pga_bc = np.zeros_like(sites.vs30)
        self._compute_mean(self.COEFFS_BC[PGA()], f0, f1, f2, rup.mag,
                           rrup, sites, sites.vs30 < 760.0, pga_bc)
        pga_bc = (10 ** pga_bc) * 1e-2 / g

        # compute mean values for hard-rock sites (vs30 >= 2000),
        # and non-hard-rock sites (vs30 < 2000) and add soil amplification
        # term
        mean_hr = np.zeros_like(sites.vs30)
        mean_bc = np.zeros_like(sites.vs30)
        ## Compute mean hard-rock for sites with VS30 > Vref
        self._compute_mean(C_HR, f0, f1, f2, rup.mag, rrup, sites,
                           sites.vs30 > 760.0, mean_hr)
        ## Compute mean B/C for sites with VS30 < 2000
        non_hr_idxs = sites.vs30 < 2000.0
        self._compute_mean(C_BC, f0, f1, f2, rup.mag, rrup, sites,
                           non_hr_idxs, mean_bc)
        self._compute_soil_amplification(C_SR, sites, pga_bc, mean_bc, mean_hr)
        mean = mean_hr
        mean[non_hr_idxs] = mean_bc[non_hr_idxs]
        # compute adjustment for magnitude dependend stress
        self._compute_stress_drop_adjustment(C_SD, rup.mag, mean)

        # convert from base 10 to base e
        if imt == PGV():
            # convert from cm/s to m/s
            mean = mean * ln10 - ln100
        else:
            # convert from cm/s**2 to g
            #mean = np.log((10 ** mean) * 1e-2 / g)
            mean = mean * ln10 - ln_g - ln100

        # stddevs are already in base e
        stddevs = self._get_stddevs(stddev_types, num_sites=len(sites.vs30))

        return mean, stddevs

    def _clip_distances(self, dists):
        """
        Return array of distances with values clipped to 1. See end of
        paragraph 'Methodology and Model Parameters', p. 2182. The equations
        have a singularity for distance = 0, so that's why distances are
        clipped to 1.
        """
        rrup = dists.rrup
        rrup[rrup < 1] = 1

        return rrup

    def _compute_f0_factor(self, rrup):
        """
        Compute and return factor f0 - see equation (5), 6th term, p. 2191.
        """
        # f0 = max(log10(R0/rrup),0)
        f0 = np.log10(self.COEFFS_IMT_INDEPENDENT['R0'] / rrup)
        f0[f0 < 0] = 0.0

        return f0

    def _compute_f1_factor(self, rrup):
        """
        Compute and return factor f1 - see equation (5), 4th term, p. 2191
        """
        # f1 = min(log10(rrup),log10(R1))
        f1 = np.log10(rrup)
        logR1 = np.log10(self.COEFFS_IMT_INDEPENDENT['R1'])
        f1[f1 > logR1] = logR1

        return f1

    def _compute_f2_factor(self, rrup):
        """
        Compute and return factor f2, see equation (5), 5th term, pag 2191
        """
        # f2 = max(log10(rrup/R2),0)
        f2 = np.log10(rrup / self.COEFFS_IMT_INDEPENDENT['R2'])
        f2[f2 < 0] = 0.0

        return f2

    def _compute_mean(self, C, f0, f1, f2, mag, rrup, _sites, idxs, mean):
        """
        Compute mean value (for a set of indexes) without site amplification
        terms. This is equation (5), p. 2191, without S term.
        """
        mean[idxs] = (C['c1'] +
                      C['c2'] * mag +
                      C['c3'] * (mag ** 2) +
                      (C['c4'] + C['c5'] * mag) * f1[idxs] +
                      (C['c6'] + C['c7'] * mag) * f2[idxs] +
                      (C['c8'] + C['c9'] * mag) * f0[idxs] +
                      C['c10'] * rrup[idxs])

    def _compute_soil_amplification(self, C, sites, pga_bc, mean_bc, mean_hr):
        """
        Compute soil amplification, that is S term in equation (5), p. 2191,
        and add to mean values for non hard rock sites.

        :param mean_bc:
            array containing mean values for B/C boundary,
            will be modified in place
        :param mean_hr:
            array containing mean values for hard rock,
            is only used for sites wtih VS30 between 760 (B/C) and 2000 m/s
        """
        # convert from base e (as defined in BA2008) to base 10 (as used in
        # AB2006)
        soft_soil_idxs = sites.vs30 <= 760.0
        soft_soil_sites = SitesContext()
        setattr(soft_soil_sites, 'vs30', sites.vs30[soft_soil_idxs])
        sal = log_e * self._get_site_amplification_linear(soft_soil_sites.vs30, C)
        sanl = log_e * self._get_site_amplification_non_linear(soft_soil_sites.vs30,
                                                            pga_bc[soft_soil_idxs], C)

        mean_bc[soft_soil_idxs] += (sal + sanl)

        ## According to the notes accompanying the program by David Boore,
        ## values of ground motion for VS30 between 800 m/s and 2000 m/s
        ## can be obtained by interpolating between the B/C value of 760 m/s
        ## and the hard rock value, assuming it to represent motions for
        ## VS30=2000 m/s
        idxs = ~soft_soil_idxs  * (sites.vs30 < 2000.0)
        idxs = np.arange(len(mean_bc))[idxs]
        for idx in idxs:
            mean_bc[idx] = np.interp(sites.vs30[idx], [760.,2000.], [mean_bc[idx], mean_hr[idx]])

    def _compute_stress_drop_adjustment(self, C, mag, mean):
        """
        Compute magnitude dependend stress drop adjustment. See equation (6),
        p. 2200 in Atkinson and Boore (2006) + erratum and equation (6), p. 1128
        in Atkinson and Boore (2011).
        """
        if mag >= 5.:
            stress = 10 ** (3.45 - 0.2 * mag)
        else:
            stress = 10 ** (3.45 - 0.2 * 5.)
        log_sf2 = min([C['d'] + 0.05, 0.05 + C['d'] * max([mag - C['M1'], 0.])
            / (C['Mh'] - C['M1'])])
##         scale_factor = np.log10(stress / (10 ** (3.45 - 0.2 * 6.5))) / np.log10(2.)
        scale_factor = np.log10(stress / 140.) / np.log10(2.)
        adjustment = log_sf2 * scale_factor
        mean[:] += adjustment

    def _get_stddevs(self, stddev_types, num_sites):
        """
        Return total standard deviation (see table 6, p. 2192).
        """
        assert all(stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
                   for stddev_type in stddev_types)
        stddevs = [np.zeros(num_sites) +
                   self.COEFFS_IMT_INDEPENDENT['std_total']
                   for _ in stddev_types]
        return stddevs

    #: Hard rock coefficents, table 6, pag 2192,
    #: coefficient values taken from Fortran implementation of Dave Boore
    #: (higher precision than in the paper)
    COEFFS_HARD_ROCK = CoeffsTable(sa_damping=5, table="""\
    IMT     c1          c2          c3          c4          c5          c6          c7          c8          c9          c10
    5.000  -5.408E+00   1.714E+00  -9.012E-02  -2.537E+00   2.267E-01  -1.268E+00   1.162E-01   9.792E-01  -1.767E-01  -1.757E-04
    4.000  -5.791E+00   1.916E+00  -1.071E-01  -2.441E+00   2.113E-01  -1.162E+00   1.018E-01   1.012E+00  -1.824E-01  -2.010E-04
    3.125  -6.038E+00   2.080E+00  -1.221E-01  -2.367E+00   2.002E-01  -1.073E+00   8.950E-02   1.002E+00  -1.803E-01  -2.306E-04
    2.500  -6.169E+00   2.211E+00  -1.348E-01  -2.299E+00   1.898E-01  -9.860E-01   7.860E-02   9.683E-01  -1.765E-01  -2.823E-04
    2.000  -6.183E+00   2.302E+00  -1.442E-01  -2.223E+00   1.770E-01  -9.370E-01   7.067E-02   9.518E-01  -1.768E-01  -3.220E-04
    1.587  -6.043E+00   2.342E+00  -1.496E-01  -2.157E+00   1.662E-01  -8.704E-01   6.047E-02   9.207E-01  -1.734E-01  -3.748E-04
    1.250  -5.724E+00   2.324E+00  -1.505E-01  -2.104E+00   1.565E-01  -8.202E-01   5.186E-02   8.563E-01  -1.661E-01  -4.329E-04
    1.000  -5.272E+00   2.264E+00  -1.483E-01  -2.069E+00   1.497E-01  -8.132E-01   4.666E-02   8.262E-01  -1.622E-01  -4.862E-04
    0.794  -4.604E+00   2.132E+00  -1.406E-01  -2.062E+00   1.468E-01  -7.974E-01   4.345E-02   7.748E-01  -1.558E-01  -5.790E-04
    0.629  -3.917E+00   1.987E+00  -1.314E-01  -2.045E+00   1.419E-01  -7.818E-01   4.297E-02   7.878E-01  -1.590E-01  -6.948E-04
    0.500  -3.216E+00   1.826E+00  -1.201E-01  -2.018E+00   1.344E-01  -8.134E-01   4.437E-02   8.839E-01  -1.751E-01  -7.704E-04
    0.397  -2.437E+00   1.649E+00  -1.084E-01  -2.051E+00   1.363E-01  -8.426E-01   4.483E-02   7.386E-01  -1.557E-01  -8.509E-04
    0.315  -1.721E+00   1.483E+00  -9.739E-02  -2.080E+00   1.382E-01  -8.893E-01   4.869E-02   6.101E-01  -1.389E-01  -9.538E-04
    0.251  -1.121E+00   1.342E+00  -8.722E-02  -2.082E+00   1.349E-01  -9.714E-01   5.628E-02   6.140E-01  -1.432E-01  -1.055E-03
    0.199  -6.153E-01   1.227E+00  -7.886E-02  -2.087E+00   1.312E-01  -1.120E+00   6.788E-02   6.055E-01  -1.459E-01  -1.125E-03
    0.158  -1.455E-01   1.123E+00  -7.143E-02  -2.116E+00   1.302E-01  -1.303E+00   8.311E-02   5.617E-01  -1.438E-01  -1.182E-03
    0.125   2.144E-01   1.054E+00  -6.664E-02  -2.154E+00   1.295E-01  -1.608E+00   1.046E-01   4.273E-01  -1.303E-01  -1.153E-03
    0.100   4.797E-01   1.017E+00  -6.404E-02  -2.201E+00   1.270E-01  -2.007E+00   1.326E-01   3.371E-01  -1.266E-01  -1.047E-03
    0.079   6.906E-01   9.974E-01  -6.276E-02  -2.262E+00   1.246E-01  -2.487E+00   1.636E-01   2.139E-01  -1.207E-01  -8.469E-04
    0.063   9.109E-01   9.802E-01  -6.208E-02  -2.360E+00   1.263E-01  -2.972E+00   1.910E-01   1.069E-01  -1.173E-01  -5.786E-04
    0.050   1.105E+00   9.719E-01  -6.197E-02  -2.466E+00   1.276E-01  -3.390E+00   2.144E-01  -1.391E-01  -9.839E-02  -3.167E-04
    0.040   1.264E+00   9.680E-01  -6.232E-02  -2.581E+00   1.317E-01  -3.644E+00   2.276E-01  -3.506E-01  -8.126E-02  -1.225E-04
    0.031   1.436E+00   9.592E-01  -6.276E-02  -2.714E+00   1.400E-01  -3.728E+00   2.343E-01  -5.430E-01  -6.448E-02  -3.230E-05
    0.025   1.522E+00   9.597E-01  -6.351E-02  -2.813E+00   1.458E-01  -3.654E+00   2.362E-01  -6.544E-01  -5.500E-02  -4.848E-05
    pga     9.069E-01   9.830E-01  -6.595E-02  -2.698E+00   1.594E-01  -2.795E+00   2.120E-01  -3.011E-01  -6.532E-02  -4.484E-04
    pgv    -1.442E+00   9.909E-01  -5.848E-02  -2.701E+00   2.155E-01  -2.436E+00   2.659E-01   8.479E-02  -6.927E-02  -3.734E-04
    """)

    #: Coefficients for NEHRP BC boundary (Vs30 = 760 m/s), table 9, pag 2202
    #: coefficient values taken from Fortran implementation of Dave Boore
    #: (higher precision than in the paper)
    COEFFS_BC = CoeffsTable(sa_damping=5, table="""\
    IMT     c1          c2          c3          c4          c5          c6          c7          c8          c9         c10
    5.000  -4.852E+00   1.580E+00  -8.066E-02  -2.530E+00   2.216E-01  -1.426E+00   1.361E-01   6.340E-01  -1.413E-01  -1.608E-04
    4.000  -5.256E+00   1.787E+00  -9.785E-02  -2.435E+00   2.068E-01  -1.307E+00   1.210E-01   7.340E-01  -1.560E-01  -1.959E-04
    3.125  -5.590E+00   1.972E+00  -1.136E-01  -2.331E+00   1.908E-01  -1.204E+00   1.099E-01   8.449E-01  -1.723E-01  -2.452E-04
    2.500  -5.800E+00   2.126E+00  -1.278E-01  -2.257E+00   1.790E-01  -1.123E+00   9.539E-02   8.911E-01  -1.797E-01  -2.601E-04
    2.000  -5.853E+00   2.233E+00  -1.385E-01  -2.195E+00   1.688E-01  -1.037E+00   8.002E-02   8.666E-01  -1.790E-01  -2.860E-04
    1.587  -5.754E+00   2.287E+00  -1.450E-01  -2.131E+00   1.582E-01  -9.568E-01   6.762E-02   8.670E-01  -1.789E-01  -3.429E-04
    1.250  -5.489E+00   2.289E+00  -1.476E-01  -2.081E+00   1.501E-01  -9.000E-01   5.794E-02   8.208E-01  -1.719E-01  -4.070E-04
    1.000  -5.058E+00   2.233E+00  -1.454E-01  -2.030E+00   1.408E-01  -8.744E-01   5.412E-02   7.922E-01  -1.697E-01  -4.886E-04
    0.794  -4.446E+00   2.119E+00  -1.387E-01  -2.009E+00   1.356E-01  -8.576E-01   4.976E-02   7.084E-01  -1.589E-01  -5.751E-04
    0.629  -3.748E+00   1.973E+00  -1.294E-01  -1.997E+00   1.313E-01  -8.417E-01   4.820E-02   6.772E-01  -1.557E-01  -6.763E-04
    0.500  -3.007E+00   1.803E+00  -1.178E-01  -1.982E+00   1.274E-01  -8.466E-01   4.698E-02   6.670E-01  -1.546E-01  -7.676E-04
    0.397  -2.281E+00   1.629E+00  -1.054E-01  -1.967E+00   1.227E-01  -8.880E-01   5.033E-02   6.839E-01  -1.582E-01  -8.587E-04
    0.315  -1.560E+00   1.455E+00  -9.312E-02  -1.977E+00   1.209E-01  -9.466E-01   5.576E-02   6.499E-01  -1.558E-01  -9.552E-04
    0.251  -8.756E-01   1.293E+00  -8.193E-02  -2.014E+00   1.226E-01  -1.027E+00   6.341E-02   5.808E-01  -1.491E-01  -1.053E-03
    0.199  -3.056E-01   1.156E+00  -7.211E-02  -2.038E+00   1.220E-01  -1.147E+00   7.375E-02   5.082E-01  -1.430E-01  -1.140E-03
    0.158   1.194E-01   1.057E+00  -6.473E-02  -2.054E+00   1.190E-01  -1.355E+00   9.160E-02   5.164E-01  -1.503E-01  -1.178E-03
    0.125   5.356E-01   9.647E-01  -5.835E-02  -2.110E+00   1.205E-01  -1.672E+00   1.156E-01   3.433E-01  -1.322E-01  -1.130E-03
    0.100   7.818E-01   9.235E-01  -5.555E-02  -2.165E+00   1.191E-01  -2.097E+00   1.483E-01   2.847E-01  -1.319E-01  -9.897E-04
    0.079   9.667E-01   9.033E-01  -5.476E-02  -2.249E+00   1.215E-01  -2.530E+00   1.775E-01   1.001E-01  -1.147E-01  -7.724E-04
    0.063   1.109E+00   8.875E-01  -5.386E-02  -2.334E+00   1.229E-01  -2.881E+00   2.007E-01  -3.189E-02  -1.069E-01  -5.483E-04
    0.050   1.209E+00   8.830E-01  -5.441E-02  -2.440E+00   1.295E-01  -3.035E+00   2.133E-01  -2.098E-01  -8.997E-02  -4.145E-04
    0.040   1.261E+00   8.789E-01  -5.515E-02  -2.536E+00   1.388E-01  -2.994E+00   2.158E-01  -3.908E-01  -6.746E-02  -3.881E-04
    0.031   1.191E+00   8.884E-01  -5.642E-02  -2.577E+00   1.451E-01  -2.840E+00   2.121E-01  -4.370E-01  -5.866E-02  -4.329E-04
    0.025   1.052E+00   9.030E-01  -5.768E-02  -2.571E+00   1.483E-01  -2.652E+00   2.065E-01  -4.084E-01  -5.769E-02  -5.122E-04
    pga     5.233E-01   9.686E-01  -6.196E-02  -2.439E+00   1.465E-01  -2.335E+00   1.912E-01  -8.695E-02  -8.285E-02  -6.304E-04
    pgv    -1.662E+00   1.050E+00  -6.035E-02  -2.496E+00   1.840E-01  -2.301E+00   2.500E-01   1.268E-01  -8.704E-02  -4.266E-04
    """)

    #: IMT-independent coefficients. std_total is the total standard deviation,
    #: see Table 6, pag 2192 and Table 9, pag 2202. R0, R1, R2 are coefficients
    #: required for mean calculation - see equation (5) pag 2191. v1, v2, Vref
    #: are coefficients required for soil response calculation, see table 8,
    #: p. 2201
    COEFFS_IMT_INDEPENDENT = {
        # the std is converted from base 10 to base e
        'std_total': np.log(10 ** 0.30),
        'R0': 10.0,
        'R1': 70.0,
        'R2': 140.0,
        'v1': 180.0,
        'v2': 300.0,
        'Vref': 760.0
    }

    #: Coefficients for stress drop adjustment, table 7, pag 2201
    COEFFS_STRESS_DROP = CoeffsTable(sa_damping=5, table="""\
    IMT   d    M1   Mh
    5.000 0.15 6.00 8.50
    4.000 0.15 5.75 8.37
    3.125 0.15 5.50 8.25
    2.500 0.15 5.25 8.12
    2.000 0.15 5.00 8.00
    1.587 0.15 4.84 7.70
    1.250 0.15 4.67 7.45
    1.000 0.15 4.50 7.20
    0.794 0.15 4.34 6.95
    0.629 0.15 4.17 6.70
    0.500 0.15 4.00 6.50
    0.397 0.15 3.65 6.37
    0.315 0.15 3.30 6.25
    0.251 0.15 2.90 6.12
    0.199 0.15 2.50 6.00
    0.158 0.15 1.85 5.84
    0.125 0.15 1.15 5.67
    0.100 0.15 0.50 5.50
    0.079 0.15 0.34 5.34
    0.063 0.15 0.17 5.17
    0.050 0.15 0.00 5.00
    0.040 0.15 0.00 5.00
    0.031 0.15 0.00 5.00
    0.025 0.15 0.00 5.00
    PGA   0.15 0.50 5.50
    PGV   0.11 2.00 5.50
    """)
