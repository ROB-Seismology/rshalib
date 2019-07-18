# -*- coding: iso-Latin-1 -*-

"""
Construct 'official' ROB GMPE logic tree
"""

from __future__ import absolute_import, division, print_function, unicode_literals



def construct_gmpe_lt(version=2015, site_conditions="rock"):
	"""
	Construct GMPE logic tree corresponding to different versions
	used at ROB

	:param version:
		int, version year, one of 2009, 2011, 2014 or 2015
		- 2009: used in Eurocode 8 study
		- 2011: used in cAt and BEST studies
		- 2014: used in cAt_rev study (hardrock conditions only)
		- 2015: used in SHRE_NPP (with different weights), Région Wallonne
			and Belgoprocess (hybrid site conditions only) studies
		(default: 2015)

	:param site_conditions:
		str, site conditions, either "rock", "hardrock" or "hybrid"
		only applies if :param:`version` is 2014 or 2015
		"hybrid" refers to a mix of rock and hardrock GMPEs, and was
		used in cAt_rev and Belgoprocess studies
		(default: "")

	:return:
		instance of :class:`rshalib.logictree.GroundMotionSystem`
	"""
	from decimal import Decimal
	from ..pmf import GMPEPMF
	from ..logictree import GroundMotionSystem

	if version in (2009, 2011):
		## Override site_conditions
		site_conditions = "rock"

	msg = "Constructing GMPE logic tree v. %s for %s conditions..."
	msg %= (version, site_conditions)
	print(msg)

	if version == 2009:
		## Used in Eurocode 8
		ssc_gmpes = ["Ambraseys1995DDGMPE"]
		ssc_weights = [Decimal('1')]
		ssc_pmf = GMPEPMF(ssc_gmpes, ssc_weights)
		asc_pmf = ssc_pmf

	elif version == 2011:
		## Used in BEST, cAt
		site_conditions = "rock"
		ssc_gmpes = ["AmbraseysEtAl1996",
					"BergeThierry2003"]
		ssc_weights = [Decimal('0.5'),
						Decimal('0.5')]
		ssc_pmf = GMPEPMF(ssc_gmpes, ssc_weights)
		asc_pmf = ssc_pmf

	elif version == 2014:
		## Used in cAt_rev (hardrock conditions only!)
		if site_conditions == "rock":
			## GMPE's for a target vs30 of 800 m/s
			ssc_gmpes = ["AtkinsonBoore2006Prime",
						"AkkarBommer2010",
						"Campbell2003SHARE",
						"FaccioliEtAl2010",
						"ToroEtAl2002SHARE"]
			ssc_weights = [Decimal('0.25'),
							Decimal('0.2'),
							Decimal('0.175'),
							Decimal('0.2'),
							Decimal('0.175')]
			ssc_pmf = GMPEPMF(ssc_gmpes, ssc_weights)

			asc_gmpes = ["AkkarBommer2010",
						"BindiEtAl2011",
						"BooreAtkinson2008Prime",
						"FaccioliEtAl2010",
						"ZhaoEtAl2006Asc"]
			asc_weights = [Decimal('0.225'),
							Decimal('0.1625'),
							Decimal('0.225'),
							Decimal('0.225'),
							Decimal('0.1625')]
			asc_pmf = GMPEPMF(asc_gmpes, asc_weights)

		elif site_conditions in ("hardrock", "hybrid"):
			## GMPE's for a target vs30 of 2000 m/s
			ssc_gmpes = ["AtkinsonBoore2006Prime",
						"Campbell2003HTT",
						"ToroEtAl2002HTT",
						"AkkarEtAl2013",
						"FaccioliEtAl2010"]
			ssc_weights = [Decimal('0.21'),
							Decimal('0.25'),
							Decimal('0.14'),
							Decimal('0.26'),
							Decimal('0.14')]
			ssc_pmf = GMPEPMF(ssc_gmpes, ssc_weights)

			asc_pmf = ssc_pmf

	elif version == 2015:
		## Used in SHRE_NPP (with different weights) and Région Wallonne studies
		if site_conditions == "rock":
			## GMPE's for a target vs30 of 800 m/s
			ssc_gmpes = ["AtkinsonBoore2006Prime",
						"AkkarEtAl2013",
						"Campbell2003SHARE",
						"FaccioliEtAl2010",
						"ToroEtAl2002SHARE"]
			ssc_weights = [Decimal('0.168'),
							Decimal('0.308'),
							Decimal('0.288'),
							Decimal('0.168'),
							Decimal('0.068')]
			ssc_pmf = GMPEPMF(ssc_gmpes, ssc_weights)

			asc_gmpes = ["AkkarEtAl2013",
						"BindiEtAl2011",
						"BooreAtkinson2008Prime",
						"FaccioliEtAl2010",
						"ZhaoEtAl2006Asc"]
			asc_weights = [Decimal('0.288'),
							Decimal('0.108'),
							Decimal('0.288'),
							Decimal('0.188'),
							Decimal('0.128')]
			asc_pmf = GMPEPMF(asc_gmpes, asc_weights)

		elif site_conditions == "hardrock":
			## GMPE's for a target vs30 of 2000 m/s
			ssc_gmpes = ["AtkinsonBoore2006Prime",
						"Campbell2003HTT",
						"RietbrockEtAl2013MDHTT",
						"ToroEtAl2002HTT"]
			ssc_weights = [Decimal('0.225'),
							Decimal('0.275'),
							Decimal('0.35'),
							Decimal('0.15')]
			ssc_pmf = GMPEPMF(ssc_gmpes, ssc_weights)

			asc_pmf = ssc_pmf

		elif site_conditions == "hybrid":
			## Mix of hardrock/rock used in Belgoprocess study
			ssc_gmpes = ["AtkinsonBoore2006Prime",
						"Campbell2003HTT",
						"RietbrockEtAl2013MDHTT",
						"AkkarEtAl2013",
						"FaccioliEtAl2010Ext"]
			ssc_weights = [Decimal('0.15'),
							Decimal('0.19'),
							Decimal('0.26'),
							Decimal('0.26'),
							Decimal('0.14')]
			ssc_pmf = GMPEPMF(ssc_gmpes, ssc_weights)

			asc_pmf = ssc_pmf

		else:
			raise Exception("%s site conditions not yet supported!" % site_conditions)

	gmpe_system_def = {}
	gmpe_system_def["Stable Shallow Crust"] = ssc_pmf
	gmpe_system_def["Active Shallow Crust"] = asc_pmf

	gms_name = "ROB GMS v. %s (%s)" % (version, site_conditions)
	gmpe_lt = GroundMotionSystem(gms_name, gmpe_system_def, use_short_names=False)
	return gmpe_lt

