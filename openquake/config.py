import os
import pprint
import copy
import numpy as np
import json
from configobj import Section, ConfigObj
from validate import Validator


#import sys
#f = sys._current_frames().values()[0]
#print f.f_back.f_globals['__file__']
#print f.f_back.f_globals['__name__']


class ConfigError(StandardError):
	pass


class OQ_Params(ConfigObj):
	def __init__(self, ini_filespec=None, calculation_mode="classical", description="", site_type="grid"):
		configspec = os.path.join(os.path.dirname(os.path.realpath( __file__ )), "configspec.gem")
		super(OQ_Params, self).__init__(ini_filespec, configspec=configspec, write_empty_values=True, list_values=False)
		if not ini_filespec:
			## Construct dictionary with sensible default values
			General_params = Section(self, 1, self, name="general")
			General_params["description"] = description
			General_params["calculation_mode"] = calculation_mode
			General_params.comments["calculation_mode"] = ["classical, event_based, ..."]
			General_params["random_seed"] = 42

			Geometry_params = Section(self, 1, self, name="geometry")
			if site_type == "grid":
				Geometry_params["region"] = []
				# TODO: check order: lon, lat or lat, lon?
				Geometry_params.comments["region"] = ["lon, lat of polygon vertices (in clock or counter-clock wise order)"]
				Geometry_params["region_grid_spacing"] = 10
				Geometry_params.comments["region_grid_spacing"] = ["km"]
			elif site_type in ("site", "sites"):
				# TODO: check if this still exists
				Geometry_params["sites"] = []

			LogicTree_params = Section(self, 1, self, name="logic_tree")
			LogicTree_params["number_of_logic_tree_samples"] = 1
			LogicTree_params.comments["number_of_logic_tree_samples"] = ["Monte Carlo sampling of logic tree"]

			Erf_params = Section(self, 1, self, name="erf")
			Erf_params["rupture_mesh_spacing"] = 5
			Erf_params.comments["rupture_mesh_spacing"] = ["km"]
			Erf_params["area_source_discretization"] = 5
			Erf_params.comments["area_source_discretization"] = ["km"]
			Erf_params["width_of_mfd_bin"] = 0.2
			Erf_params.comments["width_of_mfd_bin"] = ["bin width of the magnitude frequency distribution"]

			Site_params = Section(self, 1, self, name="site_params")
			Site_params["reference_vs30_type"] = "inferred"
			Site_params.comments["reference_vs30_type"] = ["measured or inferred"]
			# TODO: what other options are there?
			Site_params["reference_vs30_value"] = 760.
			Site_params.comments["reference_vs30_value"] = ["(m/s)"]
			Site_params["reference_depth_to_2pt5km_per_sec"] = 2
			Site_params.comments["reference_depth_to_2pt5km_per_sec"] = ["The depth to where shear-wave velocity = 2.5 km/sec.", "Cambpell basin depth. Measure is (km)"]
			Site_params["reference_depth_to_1pt0km_per_sec"] = 100.
			# TODO: in m?

			Calculation_params = Section(self, 1, self, name="calculation")
			Calculation_params["source_model_logic_tree_file"] = "source_model_logic_tree.xml"
			Calculation_params.comments["source_model_logic_tree_file"] = ["file containing erf logic tree structure"]
			Calculation_params["gsim_logic_tree_file"] = "gmpe_logic_tree.xml"
			Calculation_params.comments["gsim_logic_tree_file"] = ["file containing gmpe logic tree structure"]
			Calculation_params["investigation_time"] = 50
			Calculation_params.comments["investigation_time"] = ["years"]
			Calculation_params["intensity_measure_types_and_levels"] = '{"PGA": []}'
			# TODO: in event-based example, it's only intensity_measure_types!
			Calculation_params["truncation_level"] = 3
			Calculation_params.comments["truncation_level"] = ["(1,2,3,...)"]
			Calculation_params["maximum_distance"] = 200
			Calculation_params.comments["maximum_distance"] = ["maximum integration distance (km)"]

			Output_params = Section(self, 1, self, name="output")
			# TODO: is export_dir the same as output_dir in previous versions of OQ?
			Output_params["export_dir"] = "computed_output/%s" % calculation_mode
			Output_params.comments["export_dir"] = ["output directory - relative to this file"]
			Output_params["mean_hazard_curves"] = True
			Output_params.comments["mean_hazard_curves"] = ["Compute mean hazard curve"]
			#Output_params["quantile_hazard_curves"] = [0.05, 0.16, 0.50, 0.84, 0.95]
			Output_params["quantile_hazard_curves"] = "0.05, 0.16, 0.50, 0.84, 0.95"
			Output_params.comments["quantile_hazard_curves"] = ["List of quantiles to compute"]
			Output_params["hazard_maps"] = True
			Output_params["uniform_hazard_spectra"] = True
			Output_params["poes"] = "0.1"
			# TODO: 0.1 = 10 percent?
			Output_params.comments["poes"] = ["List of POEs to use for computing hazard maps"]

			# TODO: the following output params are probably for event-based only
			EventBasedOutput_params = ConfigObj()
			EventBasedOutput_params["complete_logic_tree_ses"] = True
			EventBasedOutput_params["complete_logic_tree_gmf"] = True
			EventBasedOutput_params["ground_motion_fields"] = True

			EventBased_params = Section(self, 1, self, name="event_based_params")
			EventBased_params["ses_per_logic_tree_path"] = 5
			EventBased_params["ground_motion_correlation_model"] = "JB2009"
			EventBased_params["ground_motion_correlation_params"] = '{"vs30_clustering": True}'


			self["general"] = General_params
			self["geometry"] = Geometry_params
			self.comments["geometry"] = [""]
			self["logic_tree"] = LogicTree_params
			self.comments["logic_tree"] = [""]
			self["erf"] = Erf_params
			self.comments["erf"] = [""]
			self["site_params"] = Site_params
			self.comments["site_params"] = [""]
			self["calculation"] = Calculation_params
			self.comments["calculation"] = [""]
			if calculation_mode == "event_based":
				self["event_based_params"] = EventBased_params
				self.comments["event_based_params"] = [""]
				Output_params.merge(EventBasedOutput_params)
				for key in EventBasedOutput_params.keys():
					Output_params.comments[key] = EventBasedOutput_params.comments[key]
			self["output"] = Output_params
			self.comments["output"] = [""]

		self.validator = Validator()
		self.__initialized = True

	def __getattr__(self, name):
		key = name.lower()
		for section_name in self.sections:
			if key in self[section_name].keys():
				return self[section_name][key]
		raise AttributeError(name)

	def __setattr__(self, name, value):
		if not self.__dict__.has_key('_OQ_Params__initialized'):
			## This test allows attributes to be set in the __init__ method
			return super(OQ_Params, self).__setattr__(name, value)
		elif self.__dict__.has_key(name):
			## Any normal attributes are handled normally
			super(OQ_Params, self).__setattr__(name, value)
		else:
			## Allow dictionary elements to be set as if they were attributes
			key = name.lower()
			if key == "calculation_mode":
				raise ConfigError("Calculation mode cannot be changed after initialization")
			## POES and QUANTILE_LEVELS don't follow the rules...
			elif key == "poes":
				self["output"]["poes"] = " ".join(map(str, value))
			elif key == "quantile_hazard_curves" and self["general"]["calculation_mode"] == "classical":
				self["output"]["quantile_hazard_curves"] = ", ".join(map(str, value))
			elif key == "percentiles" and self["general"]["calculation_mode"] == "classical":
				self["output"]["quantile_hazard_curves"] = ", ".join(map(str, [val/100. for val in value]))
			elif key == "intensity_measure_types_and_levels":
				self.set_imts(value)
			# TODO: we will need to do something similar for ground_motion_correlation_params
			else:
				for section_name in self.sections:
					if key in self[section_name].keys():
						self[section_name][key] = self.validator.check(self.configspec[section_name][key], value)
						break
				else:
					raise AttributeError(name)

	def validate(self):
		"""
		Validate parameters according to configspec
		"""
		for section_name in self.sections:
			for key, value in self[section_name].items():
				## Workaround to prevent validation failure when there is only 1 site
				if section_name == "geometry" and key == "sites":
					if not isinstance(value, list):
						value = [value]
				self.validator.check(self.configspec[section_name][key], value)
		## The following doesn't work:
		#return super(OQ_Params, self).validate(self.validator, preserve_errors=True)

	def compare(self, other_params):
		"""
		Compare with another OQ_Params object

		:param other_params:
			instance of :class:`OQ_Params`
		"""
		for section_name in self.sections:
			for key in self[section_name]:
				if key in other_params[section_name]:
					value1 = self[section_name][key]
					value2 = other_params[section_name][key]
					if self.validator.check(self.configspec[section_name][key], value1) != other_params.validator.check(self.configspec[section_name][key], value2):
						print "%s: values differ" % key
						print "  this: %s" % value1
						print "  other: %s" % value2
				else:
					print "%s: missing in other" % key
			for key in other_params[section_name]:
				if not key in self[section_name]:
					print "%s: defined in other" % key

#	def set_imts(self, imts):
#		"""
#		Set intensity measure types and levels

#		:param imts:
#			Dictionary mapping intensity measure type objects (see
#			:mod:`nhlib.imt`) to lists of intensity measure levels.
#		"""
#		if isinstance(imts, dict):
#			#self["calculation"]["intensity_measure_types_and_levels"] = repr(imts)
#			## Copy to a new dictionary with seriazable keys
#			imts2 = {}
#			for key in imts.keys():
#				imts2[repr(key)] = imts[key]
#			self["calculation"]["intensity_measure_types_and_levels"] = json.dumps(imts2, default=list)
#		elif isinstance(imts, str):
#			self["calculation"]["intensity_measure_types_and_levels"] = imts
#		else:
#			raise TypeError("IMTS must be dict or string")

	def set_imts(self, imts):
		"""
		Set intensity measure types and levels

		:param imts:
			Dictionary mapping string for intensity measure type to lists of intensity measure levels.
		"""
		if isinstance(imts, dict):
			self["calculation"]["intensity_measure_types_and_levels"] = json.dumps(imts, default=list)
		else:
			raise TypeError("IMTS must be dict or string")

	def set_grid_or_sites(self, sites=[], grid_outline=[], grid_spacing=10.):
		"""
		Set locations (grid or sites) where to compute hazard

		:param sites:
			list of (lon, lat) tuples or nhlib Site objects (default: [])
		:param grid_outline:
			list of points ((lon, lat) tuples) defining the outline of a regular
			grid. If grid_outline contains only 2 points, these are considered
			as the lower left and upper right corners (default: [])
		:param grid_spacing:
			Float, grid spacing in km (default: 10)
		"""
		if sites and len(sites) > 0:
			self["geometry"]["sites"] = []
			for point in sites:
				if hasattr(point, "location"):
					lon, lat = point.location.longitude, point.location.latitude
				else:
					lon, lat = point[:2]
				self["geometry"]["sites"].append(" ".join(map(str, (lon, lat))))
			## Workaround to prevent trailing comma if there is only 1 site
			if len(self["geometry"]["sites"]) == 1:
				self["geometry"]["sites"] = self["geometry"]["sites"][0]
			## Remove grid parameters if necessary
			if self["geometry"].has_key("region"):
				del self["geometry"]["region"]
				del self["geometry"]["region_grid_spacing"]

		elif grid_outline and len(grid_outline) > 0:
			self["geometry"]["region"] = []
			if len(grid_outline) == 1:
				raise ConfigError("grid_outline must contain 2 points at least")
			elif len(grid_outline) == 2:
				ll, ur = grid_outline
				lr = (ur[0], ll[1])
				ul = (ll[0], ur[1])
				grid_outline = [ll, lr, ur, ul]
			for lon, lat in grid_outline:
				self["geometry"]["region"].append(" ".join(map(str, (lon, lat))))
			self["geometry"]["region_grid_spacing"] = grid_spacing
			## Remove sites key if present
			if self["geometry"].has_key("sites"):
				del self["geometry"]["sites"]

	def set_soil_site_model_or_reference_params(self, soil_site_model_file=None, reference_vs30_value=760.0, reference_vs30_type="inferred", reference_depth_to_1pt0km_per_sec=100., reference_depth_to_2pt5km_per_sec=2.):
		"""
		Set site parameters, either as site model or as reference parameters

		:param soil_site_model_file:
			String, full path specification of file containing site model.
			If specified, soil_site_model_file takes precedence over other parameters
			(default: None)
		:param reference_vs30_value:
			Float, reference vs30 value in m/s (default: 760.)
		:param reference_vs30_type:
			String, reference vs30 type ("measured" or "inferred")
			(default: "inferred")
		:param reference_depth_to_1pt0km_per_sec:
			Float, reference depth to vs=1.0 km/s, specified in m
			(default: 100.)
		:param reference_depth_to_2pt5km_per_sec:
			Float, reference depth to vs=2.5 km/s, specified in km
			(default: 2.)
		"""
		if soil_site_model_file:
			self["site_params"]["site_model_file"] = soil_site_model_file
			## Remove reference_params keys if present
			if self["site_params"].has_key("reference_vs30_value"):
				del self["site_params"]["reference_vs30_value"]
				del self["site_params"]["reference_vs30_type"]
				del self["site_params"]["reference_depth_to_1pt0km_per_sec"]
				del self["site_params"]["reference_depth_to_2pt5km_per_sec"]
		else:
			self["site_params"]["reference_vs30_value"] = reference_vs30_value
			self["site_params"]["reference_vs30_type"] = reference_vs30_type
			self["site_params"]["reference_depth_to_1pt0km_per_sec"] = reference_depth_to_1pt0km_per_sec
			self["site_params"]["reference_depth_to_2pt5km_per_sec"] = reference_depth_to_2pt5km_per_sec
			## Remove soil_site_model_file key if present
			if self["site_params"].has_key("site_model_file"):
				del self["site_params"]["site_model_file"]

	# TODO: the following methods will need to be checked when we know how deaggregation parameters are specified in job.ini

	def set_epsilon_bin_limits(self, bin_width=0.5):
		if self.calculation_mode == "Disaggregation":
			num_bins, remainder = divmod(self.truncation_level, bin_width)
			self.epsilon_bin_limits = np.linspace(0, self.truncation_level + remainder, num_bins+1)

	def set_distance_bin_limits(self, bin_width=15):
		if self.calculation_mode == "Disaggregation":
			num_bins, remainder = divmod(self.maximum_distance, bin_width)
			self.distance_bin_limits = np.linspace(0, self.maximum_distance + remainder, num_bins+1)

	def set_magnitude_bin_limits(self, Mmax, bin_width=None):
		if self.calculation_mode == "Disaggregation":
			if not bin_width:
				bin_width = self.width_of_mfd_bin
			Mrange = Mmax - self.minimum_magnitude
			num_bins, remainder = divmod(Mrange, bin_width)
			self.magnitude_bin_limits = np.linspace(self.minimum_magnitude, Mmax + remainder, num_bins+1)

	def set_latitude_bin_limits(self, lat_min, lat_max, bin_width=0.1):
		if self.calculation_mode == "Disaggregation":
			lat_range = lat_max - lat_min
			num_bins, remainder = divmod(lat_range, bin_width)
			self.latitude_bin_limits = np.linspace(lat_min, lat_max + remainder, num_bins+1)

	def set_longitude_bin_limits(self, lon_min, lon_max, bin_width=0.1):
		if self.calculation_mode == "Disaggregation":
			lon_range = lon_max - lon_min
			num_bins, remainder = divmod(lon_range, bin_width)
			self.longitude_bin_limits = np.linspace(lon_min, lon_max + remainder, num_bins+1)

	def clear(self):
		"""
		Clear all configuration parameters
		"""
		configspec = self.configspec
		filename = self.filename
		super(OQ_Params, self).clear()
		self.filename = filename
		self.configspec = configspec

	def pretty_print(self):
		"""
		Print configuration parameters as a dictionary
		"""
		pp = pprint.PrettyPrinter()
		for section_name in self.sections:
			section = self[section_name]
			print "[%s]" % section.name
			pp.pprint(section.dict())

	def print_screen(self):
		"""
		Print contents of INI file to screen
		"""
		filename = self.filename
		self.filename = None
		for line in self.write():
			print line
		self.filename = filename

	def write_config(self, ini_filespec):
		"""
		Write INI file

		:param ini_filespec:
			full path specification to output INI file
		"""
		of = open(ini_filespec, "w")
		self.write(of)
		of.close()



if __name__ == "__main__":
	import openquake.hazardlib as nhlib
	ini_filespec = r"C:\Temp\job.ini"
	calculation_mode = "classical"
	site_type = "sites"
	params = OQ_Params(calculation_mode=calculation_mode, site_type=site_type)

	## Get and set some parameters
	params.random_seed = 30
	print params.random_seed
	params.percentiles = [16, 50, 84]
	params.intensity_measure_types_and_levels = {"PGA": np.arange(0.1, 1.1, 0.1)}
	params.set_grid_or_sites(grid_outline=[(4.0, 50.0), [5.0, 51.0]], grid_spacing=10.)
#	params.validate()

	print params.intensity_measure_types_and_levels

	## Print or write to file
	params.print_screen()
	#params.write_config(ini_filespec)

	#params2 = OQ_Params(ini_filespec)
	#params.compare(params2)
