## The Hazard Library
## Copyright (C) 2012 GEM Foundation
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU Affero General Public License as
## published by the Free Software Foundation, either version 3 of the
## License, or (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Affero General Public License for more details.
##
## You should have received a copy of the GNU Affero General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Module exports :class:`RietbrockEtAl2013SS` and :class:`RietbrockEtAl2013MD`.
"""
from __future__ import division

import numpy as np

from scipy.constants import g

from openquake.hazardlib.gsim.base import GMPE, CoeffsTable
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA, PGV, SA


class RietbrockEtAl2013SS(GMPE):
	"""
	Implements GMPE developed by Rietbrock et al. (2013). Based on weak-motion
	events (2 < Ml < 4.7). Model 1 (self-similar) for the stress parameter is used.
	"""
	
	#: Supported tectonic region type is 'Stable Shallow Crust' because the
	#: equations have been derived from data from the UK.
	DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.STABLE_CONTINENTAL

	#: Set of :mod:`intensity measure types <openquake.hazardlib.imt>`
	#: this GSIM can calculate. A set should contain classes from module
	#: :mod:`openquake.hazardlib.imt`.
	DEFINED_FOR_INTENSITY_MEASURE_TYPES = set([
		PGA,
		PGV,
		SA
	])
	
	#: Supported intensity measure component is the geometric mean of two
	#: horizontal components
	DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.AVERAGE_HORIZONTAL

	#: Supported standard deviation types are inter-event, intra-event
	#: and total.
	DEFINED_FOR_STANDARD_DEVIATION_TYPES = set([
		const.StdDev.TOTAL,
		const.StdDev.INTER_EVENT,
		const.StdDev.INTRA_EVENT
	])
	
	REQUIRES_SITES_PARAMETERS = set()

	#: Required rupture parameters are magnitude (eq. 10, page 63).
	REQUIRES_RUPTURE_PARAMETERS = set(('mag',))

	#: Required distance measure is Rjb (eq. 11, page 63).
	REQUIRES_DISTANCES = set(('rjb', ))

	def get_mean_and_stddevs(self, sites, rup, dists, imt, stddev_types):
		"""
		See :meth:`superclass method
		<.base.GroundShakingIntensityModel.get_mean_and_stddevs>`
		for spec of input and result values.
		"""
		# extracting dictionary of coefficients specific to required
		# intensity measure type.
		C = self.COEFFS[imt]
		
		R = self._get_r(C, dists)
		
		log10_mean = (
			self._compute_1(C, rup) +
			self._compute_2(C, rup, R) +
			self._compute_3(C, R)
			)

		# Convert units to g,
		# but only for PGA and SA (not PGV):
		if isinstance(imt, (PGA, SA)):
			mean = np.log((10.0 ** (log10_mean - 2.0)) / g)
		else:
			# PGV:
			mean = np.log(10.0 ** log10_mean)

		istddevs = self._get_stddevs(C, stddev_types,
									 num_sites=len(dists.rjb))

		stddevs = np.log(10 ** np.array(istddevs))

		return mean, stddevs
	
	def _get_stddevs(self, C, stddev_types, num_sites):
		"""
		Return standard deviations.
		"""
		stddevs = []
		for stddev_type in stddev_types:
			assert stddev_type in self.DEFINED_FOR_STANDARD_DEVIATION_TYPES
			if stddev_type == const.StdDev.TOTAL:
				stddevs.append(C['st'] + np.zeros(num_sites))
			elif stddev_type == const.StdDev.INTRA_EVENT:
				stddevs.append(C['sw'] + np.zeros(num_sites))
			elif stddev_type == const.StdDev.INTER_EVENT:
				stddevs.append(C['sb'] + np.zeros(num_sites))
		return stddevs
	
	def _compute_1(self, C, rup):
		"""
		Compute part 1 of equation 10 described on p. 63.
		"""
		return C['c1'] + (C['c2'] * rup.mag) + (C['c3'] * (rup.mag ** 2))
	
	def _compute_2(self, C, rup, R):
		"""
		Compute part 2 of equation 10 described on p. 63:
		"""
		return (
			(C['c4'] + C['c5'] * rup.mag) * self._get_f0(R) +
			(C['c6'] + C['c7'] * rup.mag) * self._get_f1(R) +
			(C['c8'] + C['c9'] * rup.mag) * self._get_f2(R)
			)
	
	def _compute_3(self, C, R):
		"""
		Compute part 3 of equation 10 described on p. 63.
		"""
		return C['c10'] * R
	
	def _get_f0(self, R):
		"""
		Compute f0 from equation 12a described on p. 63.
		"""
		r0 = 10
		f0 = np.where(R <= r0, np.log10(r0 / R), 0)
		return f0
	
	def _get_f1(self, R):
		"""
		Compute f2 from equation 12b described on p. 63.
		"""
		r1 = 50
		f1 = np.where(R <= r1, np.log10(R), np.log10(r1))
		return f1
	
	def _get_f2(self, R):
		"""
		Compute f2 from equation 12c described on p. 63.
		"""
		r2 = 100
		f2 = np.where(R <= r2, 0, np.log10(R / r2))
		return f2
	
	def _get_r(self, C, dists):
		"""
		Compute r from equation 11 described on p. 63.
		"""
		return np.sqrt(dists.rjb**2 + C['c11']**2)
	
	## table for model 1 (self-similar) for stress parameter, p. 64.
	COEFFS = CoeffsTable(sa_damping=5, table="""\
	IMT	 c1     c2     c3      c4      c5     c6      c7     c8      c9     c10       c11    st    sb    sw
	PGV -2.9598 0.9039 -0.0434 -1.6243 0.1987 -1.6511 0.1654 -2.4308 0.0851 -0.001472 1.7736 0.347 0.311 0.153
	PGA -0.0135 0.6889 -0.0488 -1.8987 0.2151 -1.9063 0.1740 -2.0131 0.0887 -0.002747 1.5473 0.436 0.409 0.153
	0.03 0.8282 0.5976 -0.0418 -2.1321 0.2159 -2.0530 0.1676 -1.5148 0.1163 -0.004463 1.1096 0.449 0.417 0.167
	0.04 0.4622 0.6273 -0.0391 -1.7242 0.1644 -1.6849 0.1270 -1.4513 0.0910 -0.004355 1.1344 0.445 0.417 0.155
	0.05 0.2734 0.6531 -0.0397 -1.5932 0.1501 -1.5698 0.1161 -1.5350 0.0766 -0.003939 1.1493 0.442 0.416 0.149
	0.06 0.0488 0.6945 -0.0420 -1.4913 0.1405 -1.4807 0.1084 -1.6563 0.0657 -0.003449 1.2154 0.438 0.414 0.143
	0.08 -0.2112 0.7517 -0.0460 -1.4151 0.1340 -1.4130 0.1027 -1.7821 0.0582 -0.002987 1.2858 0.433 0.410 0.140
	0.10 -0.5363 0.8319 -0.0521 -1.3558 0.1296 -1.3579 0.0985 -1.8953 0.0520 -0.002569 1.3574 0.428 0.405 0.138
	0.12 -0.9086 0.9300 -0.0597 -1.3090 0.1264 -1.3120 0.0948 -1.9863 0.0475 -0.002234 1.4260 0.422 0.399 0.138
	0.16 -1.3733 1.0572 -0.0698 -1.2677 0.1237 -1.2684 0.0910 -2.0621 0.0434 -0.001944 1.4925 0.416 0.392 0.139
	0.20 -1.9180 1.2094 -0.0819 -1.2315 0.1213 -1.2270 0.0872 -2.1196 0.0396 -0.001708 1.5582 0.409 0.384 0.141
	0.25 -2.5107 1.3755 -0.0949 -1.1992 0.1189 -1.1881 0.0833 -2.1598 0.0361 -0.001522 1.6049 0.402 0.376 0.144
	0.31 -3.1571 1.5549 -0.1087 -1.1677 0.1160 -1.1494 0.0791 -2.1879 0.0328 -0.001369 1.6232 0.395 0.366 0.148
	0.40 -3.8516 1.7429 -0.1228 -1.1354 0.1126 -1.1099 0.0746 -2.2064 0.0294 -0.001240 1.6320 0.387 0.356 0.152
	0.50 -4.5556 1.9258 -0.1360 -1.1015 0.1084 -1.0708 0.0700 -2.2171 0.0261 -0.001129 1.6109 0.378 0.345 0.156
	0.63 -5.2405 2.0926 -0.1471 -1.0659 0.1035 -1.0328 0.0655 -2.2220 0.0229 -0.001033 1.5735 0.369 0.333 0.160
	0.79 -5.8909 2.2357 -0.1557 -1.0279 0.0981 -0.9969 0.0612 -2.2229 0.0197 -0.000945 1.5262 0.360 0.320 0.164
	1.00 -6.4633 2.3419 -0.1605 -0.9895 0.0925 -0.9665 0.0577 -2.2211 0.0167 -0.000863 1.4809 0.350 0.307 0.168
	1.25 -6.9250 2.4037 -0.1612 -0.9545 0.0879 -0.9462 0.0558 -2.2178 0.0139 -0.000785 1.4710 0.341 0.294 0.172
	1.59 -7.2960 2.4189 -0.1573 -0.9247 0.0848 -0.9421 0.0567 -2.2137 0.0111 -0.000701 1.5183 0.331 0.280 0.177
	2.00 -7.5053 2.3805 -0.1492 -0.9128 0.0855 -0.9658 0.0619 -2.2110 0.0086 -0.000618 1.6365 0.323 0.267 0.181
	2.50 -7.5569 2.2933 -0.1376 -0.9285 0.0915 -1.0264 0.0729 -2.2108 0.0067 -0.000535 1.8421 0.315 0.254 0.186
	3.13 -7.4510 2.1598 -0.1228 -0.9872 0.1050 -1.1349 0.0914 -2.2141 0.0060 -0.000458 2.1028 0.308 0.242 0.190
	4.00 -7.1688 1.9738 -0.1048 -1.1274 0.1325 -1.3132 0.1207 -2.2224 0.0079 -0.000397 2.4336 0.299 0.227 0.195
	5.00 -6.8063 1.7848 -0.0879 -1.3324 0.1691 -1.5158 0.1533 -2.2374 0.0142 -0.000387 2.6686 0.291 0.214 0.198
	""")


class RietbrockEtAl2013MD(RietbrockEtAl2013SS):
	"""
	Implements GMPE developed by Rietbrock et al. (2013). Based on weak-motion
	events (2 < Ml < 4.7). Model 2 (magnitude-dependent) for the stress parameter is used.
	"""
	
	## table for model 2 (magnitude-dependent) for stress parameter, p. 65.
	COEFFS = CoeffsTable(sa_damping=5, table="""\
	IMT	 c1     c2     c3      c4      c5     c6      c7     c8      c9     c10       c11    st    sb    sw
	PGV -4.9398 1.7051 -0.1081 -1.6063 0.2084 -1.6040 0.1527 -2.2932 0.0659 -0.001643 3.1138 0.276 0.229 0.155
	PGA -2.6934 1.7682 -0.1366 -1.8544 0.2123 -1.8467 0.1590 -1.8809 0.0681 -0.002888 2.1589 0.335 0.298 0.154
	0.03 -1.9654 1.7265 -0.1346 -2.1011 0.2101 -2.0063 0.1555 -1.3684 0.0946 -0.004626 1.3437 0.348 0.304 0.169
	0.04 -2.3216 1.7592 -0.1328 -1.7198 0.1635 -1.6582 0.1192 -1.3348 0.0730 -0.004488 1.3363 0.343 0.305 0.157
	0.05 -2.4879 1.7771 -0.1328 -1.5910 0.1502 -1.5447 0.1087 -1.4304 0.0599 -0.004056 1.3942 0.339 0.304 0.150
	0.06 -2.6647 1.8009 -0.1336 -1.4916 0.1409 -1.4576 0.1013 -1.5603 0.0494 -0.003549 1.4587 0.335 0.302 0.144
	0.08 -2.8579 1.8325 -0.1353 -1.4169 0.1349 -1.3909 0.0959 -1.6918 0.0423 -0.003077 1.5466 0.330 0.299 0.140
	0.10 -3.0920 1.8771 -0.1381 -1.3587 0.1312 -1.3363 0.0919 -1.8091 0.0363 -0.002651 1.6583 0.325 0.295 0.138
	0.12 -3.3595 1.9336 -0.1419 -1.3133 0.1289 -1.2908 0.0884 -1.9038 0.0322 -0.002311 1.7807 0.320 0.289 0.137
	0.16 -3.6974 2.0096 -0.1472 -1.2717 0.1262 -1.2472 0.0847 -1.9844 0.0286 -0.002015 1.8540 0.314 0.282 0.138
	0.20 -4.1004 2.1032 -0.1537 -1.2341 0.1234 -1.2055 0.0809 -2.0474 0.0254 -0.001772 1.9055 0.307 0.273 0.141
	0.25 -4.5462 2.2068 -0.1607 -1.1985 0.1198 -1.1658 0.0770 -2.0935 0.0227 -0.001579 1.9052 0.301 0.263 0.145
	0.31 -5.0372 2.3180 -0.1679 -1.1625 0.1155 -1.1258 0.0728 -2.1276 0.0201 -0.001419 1.8732 0.294 0.253 0.150
	0.40 -5.5650 2.4307 -0.1745 -1.1242 0.1104 -1.0847 0.0682 -2.1519 0.0176 -0.001282 1.8142 0.288 0.242 0.155
	0.50 -6.0933 2.5325 -0.1795 -1.0837 0.1043 -1.0436 0.0635 -2.1680 0.0152 -0.001166 1.7238 0.282 0.231 0.161
	0.63 -6.5914 2.6123 -0.1820 -1.0432 0.0985 -1.0044 0.0591 -2.1775 0.0129 -0.001066 1.6524 0.276 0.220 0.167
	0.79 -7.0402 2.6616 -0.1813 -1.0023 0.0927 -0.9683 0.0552 -2.1820 0.0109 -0.000977 1.5900 0.272 0.210 0.172
	1.00 -7.4028 2.6715 -0.1767 -0.9634 0.0877 -0.9394 0.0526 -2.1827 0.0092 -0.000898 1.5549 0.268 0.201 0.177
	1.25 -7.6577 2.6402 -0.1686 -0.9299 0.0842 -0.9226 0.0519 -2.1811 0.0079 -0.000827 1.5712 0.265 0.193 0.181
	1.59 -7.8128 2.5609 -0.1559 -0.9021 0.0829 -0.9242 0.0544 -2.1782 0.0069 -0.000756 1.6701 0.262 0.185 0.186
	2.00 -7.8368 2.4440 -0.1409 -0.8896 0.0857 -0.9546 0.0613 -2.1751 0.0064 -0.000691 1.9205 0.260 0.177 0.191
	2.50 -7.7341 2.2967 -0.1244 -0.9012 0.0975 -1.0270 0.0747 -2.1717 0.0066 -0.000634 2.6233 0.258 0.169 0.195
	3.13 -7.4991 2.1232 -0.1072 -0.9638 0.1202 -1.1604 0.0965 -2.1763 0.0077 -0.000571 3.5221 0.256 0.161 0.199
	4.00 -7.1376 1.9232 -0.0893 -1.1238 0.1549 -1.3647 0.1284 -2.1901 0.0114 -0.000515 4.0984 0.253 0.152 0.203
	5.00 -6.7757 1.7502 -0.0753 -1.3603 0.1960 -1.5804 0.1613 -2.2070 0.0191 -0.000518 4.3313 0.251 0.144 0.205
	""")

