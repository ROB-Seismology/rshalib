# -*- coding: utf-8 -*-

"""
SimpleFaultSource class
"""

from __future__ import absolute_import, division, print_function, unicode_literals


## Note: Don't use np as abbreviation for nodalplane!!
import numpy as np

from .. import oqhazlib

from ..msr import get_oq_msr
from ..mfd import *
from ..geo import Point, Line, Polygon, NodalPlane
from ..pmf import HypocentralDepthDistribution, NodalPlaneDistribution
from .rupture_source import RuptureSource
from .point import PointSource



__all__ = ['SimpleFaultSource']


class SimpleFaultSource(RuptureSource, oqhazlib.source.SimpleFaultSource):
	"""
	Class representing a simple fault source.

	:param source_id:
		source identifier
	:param name:
		full name of source
	:param tectonic_region_type:
		tectonic region type known to oqhazlib/OQ.
		See :class:`oqhazlib.const.TRT`
	:param mfd:
		magnitude-frequency distribution (instance of :class:`TruncatedGRMFD`
		or :class:`EvenlyDiscretizedMFD`)
	:param rupture_mesh_spacing:
		The desired distance between two adjacent points in source's
		ruptures' mesh, in km. Mainly this parameter allows to balance
		the trade-off between time needed to compute the :meth:`distance
		<oqhazlib.geo.surface.base.BaseSurface.get_min_distance>` between
		the rupture surface and a site and the precision of that computation.
	:param magnitude_scaling_relationship:
		Instance of subclass of :class:`oqhazlib.scalerel.base.BaseMSR` to
		describe how does the area of the rupture depend on magnitude and rake.
	:param rupture_aspect_ratio:
		Float number representing how much source's ruptures are more wide
		than tall. Aspect ratio of 1 means ruptures have square shape,
		value below 1 means ruptures stretch vertically more than horizontally
		and vice versa.
	:param upper_seismogenic_depth:
		Minimum depth an earthquake rupture can reach, in km.
	:param lower_seismogenic_depth:
		Maximum depth an earthquake rupture can reach, in km.
	:param fault_trace:
		An instance of :class:`Line` representing the line of intersection
		between the fault plane and the Earth's surface.
	:param dip:
		Angle between earth surface and fault plane in decimal degrees.
	:param rake:
		Angle describing fault rake in decimal degrees.
	:param slip_rate:
		Slip rate in mm/yr (default: NaN)
	:param bg_zone:
		String, ID of background zone (default: None)
	:param timespan:
		float, timespan for Poisson temporal occurrence model.
		Introduced in more recent versions of OpenQuake
		(default: 1)
	"""
	# TODO: SlipratePMF
	# TODO: add aseismic_coef and strain_drop parameters (see get_Anderson_Luco_mfd method)
	# TODO: add char_mag property, to make distinction with max_mag !!!
	def __init__(self, source_id, name, tectonic_region_type, mfd,
				rupture_mesh_spacing, magnitude_scaling_relationship,
				rupture_aspect_ratio,
				upper_seismogenic_depth, lower_seismogenic_depth,
				fault_trace, dip, rake, slip_rate=np.NaN, bg_zone=None,
				timespan=1):
		"""
		"""
		self.timespan = timespan
		magnitude_scaling_relationship = get_oq_msr(magnitude_scaling_relationship)

		## OQ version dependent keyword arguments
		oqver_kwargs = {}
		if OQ_VERSION >= '2.9.0':
			oqver_kwargs['temporal_occurrence_model'] = self.tom
		super(SimpleFaultSource, self).__init__(source_id=source_id,
				name=name,
				tectonic_region_type=tectonic_region_type,
				mfd=mfd,
				rupture_mesh_spacing=rupture_mesh_spacing,
				magnitude_scaling_relationship=magnitude_scaling_relationship,
				rupture_aspect_ratio=rupture_aspect_ratio,
				upper_seismogenic_depth=upper_seismogenic_depth,
				lower_seismogenic_depth=lower_seismogenic_depth,
				fault_trace=fault_trace,
				dip=dip,
				rake=rake,
				**oqver_args)
		self.slip_rate = slip_rate
		self.bg_zone = bg_zone

	def __repr__(self):
		return '<SimpleFaultSource #%s>' % self.source_id

	@property
	def tom(self):
		"""
		Temporal occurrence model
		"""
		return oqhazlib.tom.PoissonTOM(self.timespan)

	def create_xml_element(self, encoding='latin1'):
		"""
		Create xml element (NRML simpleFaultSource element)

		:param encoding:
			unicode encoding
			(default: 'latin1')
		"""
		from lxml import etree
		from ..nrml import ns
		from ..nrml.common import xmlstr

		sfs_elem = etree.Element(ns.SIMPLE_FAULT_SOURCE)
		sfs_elem.set(ns.ID, xmlstr(self.source_id, encoding=encoding))
		sfs_elem.set(ns.NAME, xmlstr(self.name, encoding=encoding))
		sfs_elem.set(ns.TECTONIC_REGION_TYPE, xmlstr(self.tectonic_region_type,
													encoding=encoding))

		sfg_elem = etree.SubElement(sfs_elem, ns.SIMPLE_FAULT_GEOMETRY)
		sfg_elem.append(self.fault_trace.create_xml_element(encoding=encoding))
		dip_elem = etree.SubElement(sfg_elem, ns.DIP)
		dip_elem.text = str(self.dip)
		usd_elem = etree.SubElement(sfg_elem, ns.UPPER_SEISMOGENIC_DEPTH)
		usd_elem.text = str(self.upper_seismogenic_depth)
		lsd_elem = etree.SubElement(sfg_elem, ns.LOWER_SEISMOGENIC_DEPTH)
		lsd_elem.text = str(self.lower_seismogenic_depth)

		magScaleRel_elem = etree.SubElement(sfs_elem, ns.MAGNITUDE_SCALING_RELATIONSHIP)
		magScaleRel_elem.text = self.magnitude_scaling_relationship.__class__.__name__

		ruptAspectRatio_elem = etree.SubElement(sfs_elem, ns.RUPTURE_ASPECT_RATIO)
		ruptAspectRatio_elem.text = str(self.rupture_aspect_ratio)

		sfs_elem.append(self.mfd.create_xml_element(encoding=encoding))

		rake_elem = etree.SubElement(sfs_elem, ns.RAKE)
		rake_elem.text = str(self.rake)

		return sfs_elem

	def print_report(self):
		print("Fault source %s - %s" % (self.source_id, self.name))
		print("Strike/dip/rake: %.1f/%.1f/%.1f"
				% (self.get_mean_strike(), self.dip, self.rake))
		print("Upper/Lower depth: %.1f/%.1f km"
				% (self.upper_seismogenic_depth, self.lower_seismogenic_depth))
		print("Length/width: %.1f/%.1f km"
				% (self.get_length(), self.get_width()))
		print("Area: %.1f km2" % (self.get_area()))
		if isinstance(self.mfd, CharacteristicMFD):
			print("Mchar: %.1f" % self.mfd.char_mag)
		else:
			print("Mmax: %.1f" % self.max_mag)

	@property
	def min_mag(self):
		"""
		Return Mmin specified in the source's MFD
		"""
		if isinstance(self.mfd, TruncatedGRMFD):
			return self.mfd.min_mag
		elif isinstance(self.mfd, EvenlyDiscretizedMFD):
			return self.mfd.min_mag - self.mfd.bin_width / 2.

	@property
	def max_mag(self):
		"""
		Return Mmax specified in the source's MFD
		"""
		return self.mfd.max_mag

	@property
	def longitudes(self):
		"""
		Return list of longitudes in the source's SimpleFaultGeometry object
		"""
		return self.fault_trace.lons

	@property
	def latitudes(self):
		"""
		Return list of latitudes in the source's SimpleFaultGeometry object
		"""
		return self.fault_trace.lats

	def get_bounding_box(self):
		"""
		Determine rectangular bounding box

		:return:
			(west, east, south, north) tuple
		"""
		polygon = self.get_polygon()
		w = polygon.lons.min()
		e = polygon.lons.max()
		s = polygon.lats.min()
		n = polygon.lats.max()
		return (w, e, s, n)

	def get_depth_range(self):
		"""
		Compute depth range in km

		:return:
			Float, depth range in km
		"""
		return self.lower_seismogenic_depth - self.upper_seismogenic_depth

	def get_width(self):
		"""
		Compute fault width in km

		:return:
			Float, fault width in km

		Note: don't change to m, in order to keep compatibility with mtoolkit!
		"""
		return self.get_depth_range() / np.sin(np.radians(self.dip))

	def get_projected_width(self):
		"""
		Compute width of fault projected at the surface in km

		:return:
			Float, projected fault width in km
		"""
		return self.get_width() * np.cos(np.radians(self.dip))

	def get_length(self):
		"""
		Compute length of fault in km

		:return:
			Float, fault length in km
		"""
		lons, lats = self.fault_trace.lons, self.fault_trace.lats
		lons1, lats1 = lons[:-1], lats[:-1]
		lons2, lats2 = lons[1:], lats[1:]
		distances = oqhazlib.geo.geodetic.geodetic_distance(lons1, lats1, lons2, lats2)
		return np.add.reduce(distances)

	def get_area(self):
		"""
		Compute area of fault in square km

		:return:
			Float, fault area in square km
		"""
		return self.get_length() * self.get_width()

	def get_aspect_ratio(self):
		"""
		Compute fault aspect ratio (length / width)

		:return:
			Float, aspect ratio
		"""
		return self.get_length() / self.get_width()

	def _get_trace_azimuths_and_distances(self):
		"""
		Compute point-to-point azimuth and distance along fault trace

		:return:
			Tuple (azimuths, distances):
			azimuths: instance of :class:`mapping.geotools.Azimuth`,
				point-to-point azimuths
			distances: numpy array with point-to-point distances in km
		"""
		from mapping.geotools.angle import Azimuth

		lons, lats = self.fault_trace.lons, self.fault_trace.lats
		lons1, lats1 = lons[:-1], lats[:-1]
		lons2, lats2 = lons[1:], lats[1:]
		distances = oqhazlib.geo.geodetic.geodetic_distance(lons1, lats1, lons2, lats2)
		azimuths = oqhazlib.geo.geodetic.azimuth(lons1, lats1, lons2, lats2)
		azimuths = Azimuth(azimuths, "deg")
		return (azimuths, distances)

	def get_mean_strike(self):
		"""
		Compute mean strike in degrees, supposing fault trace points are ordered
		in the opposite direction (left-hand rule).

		:return:
			Float, mean strike in degrees
		"""
		azimuths, distances = self._get_trace_azimuths_and_distances()
		weights = distances / np.add.reduce(distances)
		mean_strike = azimuths.mean(weights).deg()
		return mean_strike

	def get_mesh_mean_strike(self):
		"""
		Compute mean strike of fault mesh in degrees

		:return:
			Float, mean strike in degrees
		"""
		return self.get_surface().get_strike()

	def get_std_strike(self):
		"""
		Compute standard deviation of strike from surface trace

		:return:
			Float, standard deviation of strike in degrees
		"""
		from mapping.geotools.angle import Azimuth

		mean_strike = Azimuth(self.get_mean_strike(), "deg")
		#mean_strike_angle = 90 - mean_strike
		azimuths, distances = self._get_trace_azimuths_and_distances()
		weights = distances / np.add.reduce(distances)
		azimuth_deltas = azimuths.get_enclosed_angle(mean_strike).deg()

		variance = np.dot(weights, np.array(azimuth_deltas)**2) / weights.sum()
		return np.sqrt(variance)

	def get_min_max_strike(self):
		"""
		Compute minimum and maximum strike from surface trace

		:return:
			Tuple (min_strike, max_strike) in degrees
		"""
		azimuths, _ = self._get_trace_azimuths_and_distances()
		az_min, az_max = azimuths.min().deg(), azimuths.max().deg()
		if az_max - az_min > 180:
			azimuths[azimuths < np.pi] += 360
			az_min, az_max = azimuths.min().deg(), azimuths.max.deg()
			az_max -= 360
		return (az_min, az_max)

	def get_strike_deltas(self, other_faults):
		"""
		Compute difference in strike compared to other faults

		:param other_faults:
			list with instances of :class:`SimpleFault`

		:return:
			float array: strike differences (in degrees, between -180 and 180)
			- negative values: other faults are in anticlockwise direction
			- positive values: other faults are in clockwise direction
		"""
		from mapping.geotools.angle import Azimuth

		strike0 = Azimuth(self.get_mean_strike(), "deg")
		other_strikes = [flt.get_mean_strike() for flt in other_faults]
		other_strikes = Azimuth(other_strikes, "deg")
		strike_deltas = (other_strikes - strike0).deg()
		strike_deltas[strike_deltas > 180] = 360 - strike_deltas[strike_deltas > 180]
		strike_deltas[strike_deltas < -180] += 360
		return strike_deltas

	def get_abs_strike_deltas(self, other_faults):
		"""
		Compute absolute difference in strike compared to other faults

		:param other_faults:
			list with instances of :class:`SimpleFault`

		:return:
			float array: absolute strike differences
			(in degrees, between 0 and 180)
		"""
		return np.abs(self.get_strike_deltas(other_faults))

	def reverse_trace(self):
		"""
		Reverse direction of fault trace
		"""
		self.fault_trace.reverse_direction()

	def get_dip_direction(self):
		"""
		Return dip direction (azimuth in degrees)
		"""
		return (self.get_mean_strike() + 90) % 360

	def get_top_edge(self):
		"""
		Construct top edge of fault (downdip projection of surface trace
		down to upper seismogenic depth)

		:return:
			instance of :class:`Line`
		"""
		usd = self.upper_seismogenic_depth
		top_edge = self.fault_trace.project(usd, self.dip)
		top_edge = [Point(pt.longitude, pt.latitude, usd) for pt in top_edge]
		return top_edge

	def get_bottom_edge(self):
		"""
		Construct bottom edge of fault (downdip projection of surface trace
		down to lower seismogenic depth)

		:return:
			instance of :class:`Line`
		"""
		lsd = self.lower_seismogenic_depth
		bottom_edge = self.fault_trace.project(lsd, self.dip)
		bottom_edge = [Point(pt.longitude, pt.latitude, lsd) for pt in bottom_edge]
		return bottom_edge

	def get_polygon(self):
		"""
		Construct polygonal outline of fault

		:return:
			instance of :class:`Polygon`
		"""
		top_edge = self.get_top_edge()
		bottom_edge = self.get_bottom_edge()
		bottom_edge.reverse()
		return Polygon(top_edge + bottom_edge + [top_edge[0]])

	def calc_Mmax_Wells_Coppersmith(self):
		"""
		Compute Mmax from Wells & Coppersmith (1994) scaling relation

		:return:
			Float, maximum magnitude
		"""
		from openquake.hazardlib.scalerel.wc1994 import WC1994
		wc = WC1994()
		max_mag = wc.get_median_mag(self.get_area(), self.rake)
		return max_mag

	def get_characteristic_mfd(self, bin_width=None, M_sigma=0.3, num_sigma=1,
								force_bin_alignment=True):
		"""
		Construct MFD corresponding to a characteristic Mmax
		:param bin_width:
			float, Magnitude interval for evenly discretized magnitude
			frequency distribution
			(default: None, take bin_width from current MFD)
		:param M_sigma:
			float, standard deviation on magnitude
			(default: 0.3)
		:param num_sigma:
			float, number of standard deviations to spread occurrence rates over
			(default: 1)
		:param force_bin_alignment:
			bool, whether or not to enforce bin edges to aligh with bin width
			If True, characteristic magnitude may be raised by up to half a
			bin width.
			(default: True)

		:return:
			instance of :class:`CharacteristicMFD`
		"""
		if bin_width is None:
			bin_width = self.mfd.bin_width

		try:
			#char_mag = self.mfd.char_mag + bin_width/2.
			char_mag = self.mfd.char_mag
		except AttributeError:
			#char_mag = self.max_mag - bin_width
			#char_mag = self.max_mag + bin_width/2.
			char_mag = self.max_mag
			return_period = self.get_Mmax_return_period()
		else:
			return_period = self.mfd.char_return_period
		MFD = CharacteristicMFD(char_mag, return_period, bin_width, M_sigma=M_sigma,
						num_sigma=num_sigma, force_bin_alignment=force_bin_alignment)
		return MFD

	def get_Anderson_Luco_mfd(self, min_mag=None, max_mag=None, bin_width=None,
								b_val=None, aseismic_coef=0., strain_drop=None,
								mu=3E+10, arbitrary_surface=True):
		"""
		Compute MFD according to Anderson & Luco (1983), based on slip rate

		:param min_mag:
			float, Minimum magnitude
			(default: None, take min_mag from current MFD).
		:param max_mag:
			Maximum magnitude
			(default: None, take max_mag from current MFD).
		:param bin_width:
			float, Magnitude interval for evenly discretized magnitude frequency
			distribution
			(default: None, take bin_width from current MFD.
		:param b_val:
			float, Parameter of the truncated Gutenberg-Richter model.
			(default: None, take b_val from current MFD)
		:param aseismic_coef:
			float, The proportion of the fault slip that is released aseismically.
			(default: 0.)
		:param strain_drop:
			float, strain drop (dimensionless), this is the ratio between
			rupture displacement and rupture length. This parameter is only
			required if :param:`arbitrary_surface` is set to True.
			If not provided, it will be computed from Mmax
			(default: None)
		:param mu:
			rigidity or shear modulus in N/m**2 or Pascal
			(default: 3E+10)
		:param arbitrary_surface:
			bool, indicating whether rupture surface is arbitrary or
			corresponds to max_mag
			(default: True)

		:return:
			instance of :class:`TruncatedGRMFD`
		"""
		if b_val is None:
			b_val = self.mfd.b_val
		if bin_width is None:
			bin_width = self.mfd.bin_width
		if min_mag is None:
			min_mag = self.mfd.get_min_mag_edge()
		if max_mag is None:
			try:
				max_mag = self.mfd.char_mag
			except AttributeError:
				#max_mag = self.mfd.max_mag - bin_width / 2.
				max_mag = self.mfd.max_mag
		if not strain_drop and not arbitrary_surface:
			strain_drop = self.get_Mmax_strain_drop()
		## For seismic moment in units of dyn-cm
		c, d = 16.05, 1.5

		## Using mtoolkit, returns EvenlyDiscretizedMFD
		#from mtoolkit.scientific.fault_calculator import get_mfd, MOMENT_SCALING
		#from mtoolkit.geo.tectonic_region import TectonicRegionBuilder
		#moment_scaling = (c, d)
		#min_mag = min_mag + bin_width / 2.
		#trt = TectonicRegionBuilder.create_tect_region_by_name(self.tectonic_region_type)
		## Override displacement-length ratio with actual value computed from Mmax
		#trt._dlr = {'value': [strain_drop], 'weight': [1.0]}
		#trt._smod = {'value': [mu/1E+9], 'weight': [1.0]}
		#occurrence_rates = get_mfd(self.slip_rate, aseismic_coef, trt, self, b_val, min_mag, bin_width, max_mag, self.rake, moment_scaling, arbitrary_surface)
		#return EvenlyDiscretizedMFD(min_mag, bin_width, occurrence_rates)

		dbar = d * np.log(10.0)
		bbar = b_val * np.log(10.0)
		seismic_slip = self.slip_rate * (1.0 - aseismic_coef)

		## Compute cumulative value for M=0
		mag_value = 0
		if arbitrary_surface:
			max_mag_moment = 10 ** ((max_mag + 10.73) * 1.5)
			arbitrary_surface_param = (max_mag_moment
										/ ((mu * 10) * (self.get_area() * 1E10)))
			N0 = (((dbar - bbar) / (dbar))
					* ((seismic_slip / 10.) / arbitrary_surface_param)
					* np.exp(bbar * (max_mag - mag_value)))
		else:
			beta = np.sqrt(strain_drop * (10.0 ** c)
							/ ((mu * 10) * (self.get_width() * 1E5)))
			N0 = (((dbar - bbar) / (dbar)) * ((seismic_slip / 10.) / beta)
						* np.exp(bbar * (max_mag - mag_value))
						* np.exp(-(dbar / 2.) * max_mag))

		a_val = np.log10(N0)
		return TruncatedGRMFD(min_mag, max_mag, bin_width, a_val, b_val)

	def get_Youngs_Coppersmith_mfd(self, min_mag=None, bin_width=None, b_val=None):
		"""
		Compute MFD according to Youngs & Coppersmith (1985), based on
		frequency of Mmax

		:param min_mag:
			float, Minimum magnitude
			(default: None, take min_mag from current MFD).
		:param bin_width:
			float, Magnitude interval for evenly discretized magnitude
			frequency distribution
			(default: None, take bin_width from current MFD).
		:param b_val:
			float, Parameter of the truncated Gutenberg-Richter model.
			(default: None, take b_val from current MFD)

		:return:
			instance of :class:`YoungsCoppersmith1985MFD`
		"""
		if b_val is None:
			b_val = self.mfd.b_val
		if bin_width is None:
			bin_width = self.mfd.bin_width
		if min_mag is None:
			min_mag = self.mfd.get_min_mag_center()
		try:
			char_mag = self.mfd.char_mag
		except AttributeError:
			char_mag = self.max_mag
			char_rate = 1. / self.get_Mmax_return_period()
		else:
			char_rate = 1. / self.mfd.char_return_period
		MFD = YoungsCoppersmith1985MFD.from_characteristic_rate(min_mag, b_val,
												char_mag, char_rate, bin_width)
		return MFD

	def get_moment_rate(self, mu=3E+10):
		"""
		Compute moment rate from slip rate and fault dimensions

		:param mu:
			rigidity or shear modulus in N/m**2 or Pascal
			(default: 3E+10)

		:return:
			float, moment rate in N.m/yr
		"""
		return mu * self.get_area() * 1E+6 * self.slip_rate * 1E-3

	def get_Mmax_moment(self):
		"""
		Compute seismic moment corresponding to maximum magnitude,
		assuming Mmax corresponds to max_mag of MFD minus one bin width

		:return:
			Float, seismic moment in N.m
		"""
		try:
			max_mag = self.mfd.char_mag
		except AttributeError:
			max_mag = self.max_mag - self.mfd.bin_width / 2.
		return 10 ** (1.5 * (max_mag + 6.06))

	def get_Mmax_return_period(self, mu=3E+10):
		"""
		Compute predicted return period for Mmax based on slip rate
		assuming Mmax corresponds to max_mag of MFD minus one bin width

		:param mu:
			rigidity or shear modulus in N/m**2 or Pascal
			(default: 3E+10)

		:return:
			float, return period in yr
		"""
		return self.get_Mmax_moment() / self.get_moment_rate(mu=mu)

	def get_Mmax_moment_rate(self, mu=3E+10):
		"""
		Compute moment rate corresponding to maximum magnitude
		assuming Mmax corresponds to max_mag of MFD minus one bin width

		:param mu:
			rigidity or shear modulus in N/m**2 or Pascal
			(default: 3E+10)

		:return:
			float, moment rate in N.m/yr
		"""
		return self.get_Mmax_moment() / self.get_Mmax_return_period(mu=mu)

	def get_Mmax_slip(self, mu=3E+10):
		"""
		Compute slip corresponding to maximum magnitude, assuming it would
		occur over the entire fault plane, and assuming Mmax corresponds to
		max_mag of MFD minus one bin width

		:param mu:
			rigidity or shear modulus in N/m**2 or Pascal
			(default: 3E+10)

		:return:
			float, slip in m
		"""
		return self.get_Mmax_moment() / (mu * self.get_area() * 1E+6)

	def get_Mmax_strain_drop(self, mu=3E+10):
		"""
		Compute strain drop corresponding to maximum magnitude, assuming
		it would occur over the entire fault plane, and assuming Mmax
		corresponds to max_mag of MFD minus one bin width

		:param mu:
			rigidity or shear modulus in N/m**2 or Pascal
			(default: 3E+10)

		:return:
			float, strain drop (dimensionless)
		"""
		return self.get_Mmax_slip(mu=mu) / (self.get_length() * 1E+3)

	def get_strike_slip_rate(self):
		"""
		Compute strike-slip component of slip rate.

		:return:
			float, strike-slip component of slip rate (+ = left-lateral,
				- = right-lateral)
		"""
		return self.slip_rate * np.cos(np.radians(self.rake))

	def get_dip_slip_rate(self):
		"""
		Compute dip-slip component of slip rate.

		:return:
			float, dip-slip component of slip rate (+ = reverse,
				- = normal)
		"""
		return self.slip_rate * np.sin(np.radians(self.rake))

	def get_vertical_dip_slip_rate(self):
		"""
		Compute vertical component of dip-slip component of slip rate.

		:return:
			float, vertical component of dip-slip rate (+ = reverse,
				- = normal)
		"""
		return self.get_dip_slip_rate() * np.sin(np.radians(self.dip))

	def get_horizontal_dip_slip_rate(self):
		"""
		Compute horizontal component of dip-slip component of slip rate.

		:return:
			float, vertical component of dip-slip rate (+ = shortening,
				- = extension)
		"""
		return self.get_dip_slip_rate() * np.cos(np.radians(self.dip))

	def to_ogr_geometry(self):
		"""
		Create OGR Geometry object
		"""
		# TODO: complete
		import osr, ogr

		## Construct WGS84 projection system corresponding to earthquake coordinates
		wgs84 = osr.SpatialReference()
		## Note: str for PY2/3 compatibility
		wgs84.SetWellKnownGeogCS(str("WGS84"))

		## Create line
		line = ogr.Geometry(type=ogr.wkbLineString)

		## Add points
		for lon, lat in zip(self.longitudes, self.latitudes):
			line.AddPoint_2D(lon, lat)

		line.AssignSpatialReference(wgs84)

		return line

	def get_centroid(self):
		"""
		Compute centroid of area source

		:return:
			(float, float) tuple: longitude, latitude of centroid
		"""
		centroid = self.to_ogr_geometry().Centroid()
		return (centroid.GetX(), centroid.GetY())

	def to_characteristic_source(self, convert_mfd=False):
		"""
		Convert to a characteristic fault source

		:param convert_mfd:
			bool, whether or not to convert fault MFD to
			instance of :class:`CharacteristicMFD`
			(default: False)

		:return:
			instance of :class:`CharacteristicSource`
		"""
		surface = oqhazlib.geo.surface.SimpleFaultSurface.from_fault_data(
			self.fault_trace, self.upper_seismogenic_depth,
			self.lower_seismogenic_depth, self.dip, self.rupture_mesh_spacing)

		if convert_mfd and not isinstance(self.mfd, CharacteristicMFD):
			mfd = self.get_characteristic_mfd()
		else:
			mfd = self.mfd

		return CharacteristicFaultSource(self.source_id, self.name,
			self.tectonic_region_type, mfd, surface, self.rake,
			timespan=self.timespan)

	def get_subfaults(self, as_num=1, ad_num=1, rigidity=3E+10):
		"""
		Divide fault into subfaults (to be used for dislocation modeling).

		:param as_num:
			int, number of subfaults along strike
			(default: 1)
		:param ad_num:
			int, number of subfaults along dip
			(default: 1)
		:param rigidity:
			rigidity or shear modulus in N/m**2 or Pascal
			(default: 3E+10)

		:return:
			2-D array with instances of :class:`eqgeology.faultlib.okada.ElasticSubFault`
			First dimension is along strike, second dimension is down dip.
		"""
		from ..utils import interpolate
		from eqgeology.faultlib.okada import ElasticSubFault

		## It is not possible to have different rupture_mesh_spacing
		## along-strike and down-dip, so the strategy is to subdivide
		## using the resolution required to obtain the specified
		## number of along_strike sections; for each along-strike
		## section, we interpolate the positions of the downdip
		## sections along the left and right edges
		rupture_mesh_spacing = self.get_length() / as_num
		fault_mesh = self.get_mesh(rupture_mesh_spacing)
		#print fault_mesh.shape

		## In- and output "positions" for interpolation
		dd_mesh_pos = np.linspace(0, 1, fault_mesh.shape[0])
		subflt_edge_pos = np.linspace(0, 1, ad_num + 1)

		subfaults = []
		slip = self.get_Mmax_slip(mu=rigidity)
		for i in range(as_num):
			subfaults.append([])

			left_lons = fault_mesh.lons[:,i]
			left_lats = fault_mesh.lats[:,i]
			left_depths = fault_mesh.depths[:,i] * 1E+3

			right_lons = fault_mesh.lons[:,i+1]
			right_lats = fault_mesh.lats[:,i+1]
			right_depths = fault_mesh.depths[:,i+1] * 1E+3

			## Interpolate downdip edge positions
			left_lons = interpolate(dd_mesh_pos, left_lons, subflt_edge_pos)
			left_lats = interpolate(dd_mesh_pos, left_lats, subflt_edge_pos)
			left_depths = interpolate(dd_mesh_pos, left_depths, subflt_edge_pos)

			right_lons = interpolate(dd_mesh_pos, right_lons, subflt_edge_pos)
			right_lats = interpolate(dd_mesh_pos, right_lats, subflt_edge_pos)
			right_depths = interpolate(dd_mesh_pos, right_depths, subflt_edge_pos)

			for j in range(ad_num):
				A = [right_lons[j], right_lats[j], right_depths[j]]
				B = [right_lons[j+1], right_lats[j+1], right_depths[j+1]]
				C = [left_lons[j+1], left_lats[j+1], left_depths[j+1]]
				D = [left_lons[j], left_lats[j], left_depths[j]]

				subfault = ElasticSubFault.from_corner_points([A, B, C, D], slip,
														self.rake, rigidity)
				subfaults[i].append(subfault)

		"""
		fault_mesh = self.get_mesh(rupture_mesh_spacing / 2.)
		#print fault_mesh.shape

		## In- and output "positions" for interpolation
		dd_mesh_pos = np.linspace(0, 1, fault_mesh.shape[0])
		subflt_centroid_pos = (np.arange(ad_num) + 0.5) / ad_num

		subfaults = []
		for i in range(as_num):
			subfaults.append([])

			## Center line
			center_lons = fault_mesh.lons[:,i*2+1]
			center_lats = fault_mesh.lats[:,i*2+1]
			center_depths = fault_mesh.depths[:,i*2+1]

			## Compute projected width and height before interpolation
			projected_width = oqhazlib.geo.geodetic.geodetic_distance(center_lons[0], center_lats[0], center_lons[-1], center_lats[-1])
			total_height = center_depths[-1] - center_depths[0]

			## Interpolate downdip centroid positions
			center_lons = interpolate(dd_mesh_pos, center_lons, subflt_centroid_pos)
			center_lats = interpolate(dd_mesh_pos, center_lats, subflt_centroid_pos)
			center_depths = interpolate(dd_mesh_pos, center_depths, subflt_centroid_pos)

			## Left and right edges at the surface
			left_lon = fault_mesh.lons[0,i*2]
			left_lat = fault_mesh.lats[0,i*2]
			right_lon = fault_mesh.lons[0,i*2+2]
			right_lat = fault_mesh.lats[0,i*2+2]

			## Compute subfault length, strike, dip and width
			subfault_length = oqhazlib.geo.geodetic.geodetic_distance(left_lon, left_lat, right_lon, right_lat)
			subfault_strike = oqhazlib.geo.geodetic.azimuth(left_lon, left_lat, right_lon, right_lat)
			subfault_dip = np.degrees(np.arctan2(total_height, projected_width))
			subfault_width = (np.sqrt(projected_width**2 + total_height**2) / ad_num)

			for j in range(ad_num):
				subfault = ElasticSubFault()
				subfault.strike = subfault_strike
				subfault.dip = subfault_dip
				subfault.length = subfault_length * 1E+3
				subfault.width = subfault_width * 1E+3
				subfault.longitude = center_lons[j]
				subfault.latitude = center_lats[j]
				subfault.depth = center_depths[j] * 1E+3
				subfault.coordinate_specification = "centroid"
				subfault.rake = self.rake
				subfault.slip = self.get_Mmax_slip()
				subfaults[i].append(subfault)
		"""

		return np.array(subfaults)

	def get_surface(self, rupture_mesh_spacing=None):
		"""
		Get fault surface object

		:param rupture_mesh_spacing:
			float, alternative rupture mesh spacing (in km)
			(default: None, will use fault's current rupture mesh spacing)

		:return:
			instance of :class:`openquake.hazardlib.geo.surface.simple_fault.SimpleFaultSurface`
		"""
		from openquake.hazardlib.geo.surface.simple_fault import SimpleFaultSurface

		rupture_mesh_spacing = rupture_mesh_spacing or self.rupture_mesh_spacing
		return SimpleFaultSurface.from_fault_data(self.fault_trace,
					self.upper_seismogenic_depth, self.lower_seismogenic_depth,
					self.dip, rupture_mesh_spacing)

	def get_mesh(self, rupture_mesh_spacing=None):
		"""
		Get fault mesh

		:param rupture_mesh_spacing:
			float, alternative rupture mesh spacing (in km)
			(default: None, will use fault's current rupture mesh spacing)

		:return:
			instance of :class:`openquake.hazardlib.geo.mesh.RectangularMesh`
		"""
		return self.get_surface(rupture_mesh_spacing=rupture_mesh_spacing).get_mesh()

	def calc_distance(self, lons, lats, depths):
		"""
		Compute distance to a set of points

		:param lons:
			list or array, longitudes (in degrees)
		:param lats:
			list or array, latitudes (in degrees)
		:param depths:
			list or array, depths (in km)

		:return:
			array, distances (in km)
		"""
		lons = np.asarray(lons)
		lats = np.asarray(lats)
		depths = np.asarray(depths)
		mesh = oqhazlib.geo.mesh.Mesh(lons, lats, depths)
		return self.get_surface().get_min_distance(mesh)

	def get_top_edge_rupture_faults(self, mag):
		"""
		Get list of floating top-edge ruptures as fault objects
		with dimensions corresponding to given magnitude and
		fault's rupture mesh spacing and rupture aspect ratio

		:param mag:
			float, magnitude

		:return:
			list with instances of :class:`CharacteristicFaultSource`
		"""
		rms = self.rupture_mesh_spacing
		usd, lsd = self.upper_seismogenic_depth, self.lower_seismogenic_depth
		trt = self.tectonic_region_type
		msr = self.magnitude_scaling_relationship
		rar = self.rupture_aspect_ratio

		fault_mesh = self.get_mesh()
		surface_locations = fault_mesh[0:1]
		mesh_rows, mesh_cols = fault_mesh.shape
		#print fault_mesh[:,:1].depths

		subfaults = []
		for i in range(1, mesh_cols-1):
			submesh = fault_mesh[:, i-1:i+1]
			dip, strike = submesh.get_mean_inclination_and_azimuth()
			hypocenter = submesh.get_middle_point()
			nodal_plane = NodalPlane(strike, dip, self.rake)
			npd = NodalPlaneDistribution([nodal_plane], [1])
			hdd = HypocentralDepthDistribution([hypocenter.depth], [1])
			subfault_name = "%s #%02d" % (self.name, i+1)
			subfault_mfd = EvenlyDiscretizedMFD(mag, self.mfd.bin_width, [1])
			subfault_id = self.source_id + "#%02d_M=%s" % (i+1, mag)
			point_source = PointSource(subfault_id, subfault_name, trt,
										subfault_mfd, rms, msr, rar,
										usd, lsd, hypocenter, npd, hdd)
			## Only select ruptures staying within fault limits
			distance_to_start = i * rms - rms/2.
			distance_to_end = (mesh_cols - i) * rms - rms/2.
			rup_length, rup_width = point_source._get_rupture_dimensions(mag,
																	nodal_plane)
			if rup_length / 2 <= min(distance_to_start, distance_to_end):
				rup_num_cols = int(round(rup_length / rms))
				start_col = min(i, int(i - (rup_num_cols / 2.)))
				end_col = max(i, int(i + (rup_num_cols / 2.)))
				#rup_row_num = int(round(rup_width / rms))
				subfault_trace = list(surface_locations)[start_col:end_col+1]
				subfault_trace = Line(subfault_trace)
				subfault_lsd =  rup_width * np.cos(np.radians(90 - self.dip))
				subfault = SimpleFaultSource(subfault_id, subfault_name,
								trt, subfault_mfd, rms, msr, rar, usd,
								subfault_lsd, subfault_trace, self.dip, self.rake)

				subfaults.append(subfault.to_characteristic_source())

		return subfaults

	def discretize_along_strike(self):
		"""
		Divide fault in subfaults along strike with length equal
		to rupture mesh spacing.

		:return:
			list with instances of :class:`SimpleFaultSource`
		"""
		rms = self.rupture_mesh_spacing
		usd, lsd = self.upper_seismogenic_depth, self.lower_seismogenic_depth
		trt = self.tectonic_region_type
		msr = self.magnitude_scaling_relationship
		rar = self.rupture_aspect_ratio

		fault_mesh = self.get_mesh()
		surface_locations = fault_mesh[0:1]
		subfaults = []
		for i in range(1, len(surface_locations)):
			subfault_id = self.source_id + "#%02d" % i
			subfault_name = "%s #%02d" % (self.name, i)
			subfault_trace = list(surface_locations)[i-1:i+1]
			## Make sure start of first subfault and end of last subfault
			## correspond exactly to fault trace endpoints
			if i == 0:
				subfault_trace[0] = self.fault_trace[0]
			elif i == len(surface_locations) - 1:
				subfault_trace[-1] = self.fault_trace[-1]
			subfault_trace = Line(subfault_trace)
			subfault = SimpleFaultSource(subfault_id, subfault_name,
							trt, self.mfd, rms, msr, rar, usd, lsd,
							subfault_trace, self.dip, self.rake)

			subfaults.append(subfault)

		return subfaults

	def to_lbm_data(self, geom_type="line"):
		"""
		Convert to layeredbasemap line or polygon

		:param geom_type:
			str, "line" or "polygon"
			(default: "line")

		:return:
			layeredbasemap LineData or PolygonData object
		"""
		import mapping.layeredbasemap as lbm

		if geom_type == "line":
			ft = self.fault_trace
			geom = lbm.LineData(ft.lons, ft.lats, label=self.source_id)
		elif geom_type == "polygon":
			pg = self.get_polygon()
			geom = lbm.PolygonData(pg.lons, pg.lats, label=self.source_id)

		geom.value = {}
		attrs = ("source_id", "name", "tectonic_region_type",
			"rupture_mesh_spacing", "magnitude_scaling_relationship",
			"rupture_aspect_ratio", "upper_seismogenic_depth",
			"lower_seismogenic_depth", "dip", "rake", "slip_rate", "bg_zone")
		for attr in attrs:
			setattr(geom, attr, getattr(self, attr))

		return geom

# TODO: need common base class for SimpleFaultSource and CharacteristicFaultSource



if __name__ == '__main__':
	pass
