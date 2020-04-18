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
Module exports :class:`ZhaoEtAl2006AscMOD`
"""
from __future__ import division

import numpy as np

from openquake.hazardlib.gsim.zhao_2006 import ZhaoEtAl2006Asc


class ZhaoEtAl2006AscMOD(ZhaoEtAl2006Asc):
    """
	Modification of ZhaoEtAl2006Asc GMPE. For the site classes the vs30 boundary
	of 200 m/s between medium soil and soft soil is changed to 220 m/s.
    """

    def _compute_site_class_term(self, C, vs30):
        """
        Compute nine-th term in equation 1, p. 901.
        """
        # map vs30 value to site class, see table 2, p. 901.
        site_term = np.zeros(len(vs30))

        # hard rock
        site_term[vs30 > 1100.0] = C['CH']

        # rock
        site_term[(vs30 > 600) & (vs30 <= 1100)] = C['C1']

        # hard soil
        site_term[(vs30 > 300) & (vs30 <= 600)] = C['C2']

        # medium soil
        site_term[(vs30 > 220) & (vs30 <= 300)] = C['C3']

        # soft soil
        site_term[vs30 <= 220] = C['C4']

        return site_term

