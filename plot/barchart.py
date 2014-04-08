"""
"""

import numpy as np
import pylab


def plot_variation_barchart(mean_value, category_value_dict, ylabel, title):
	"""
	Plot simple barchart showing variations between different categories

	:param mean_value:
		float, mean value
	:param category_value_dict:
		dict, mapping category names to category values (lists)
	:param ylabel:
		str, label for Y axis
	:param title
		str, plot title
	"""
	category_names = category_value_dict.keys()
	category_values = category_value_dict.values()
	category_diffs = np.asarray(category_values) - mean_value
	num_categories = len(category_names)
	xvalues = np.arange(1, num_categories + 1)
	bar_width = 0.35

	rects = pylab.bar(xvalues, category_diffs, bar_width, color='r')

	def autolabel(rects, labels, values):
		# attach some text labels
		for rect, label, value in zip(rects, labels, values):
			x = rect.get_x() + rect.get_width() / 2.
			if value < 0:
				y = -rect.get_height() * 1.05
				va = 'top'
			else:
				y = rect.get_height() * 1.05
				va = 'bottom'

			pylab.text(x, y, label, ha='center', va=va)

	autolabel(rects, category_names, category_diffs)
	xmin, xmax = 0.5, num_categories + 1
	ymin = category_diffs.min() + 0.2 * category_diffs.min()
	ymax = category_diffs.max() + 0.2 * category_diffs.max()
	pylab.hlines(0, xmin, xmax, lw=2, color='k')
	pylab.text(xmax, 0, "%s" % mean_value, ha='left', va='center')
	pylab.ylabel("$\Delta$ " + ylabel)
	pylab.title(title)
	pylab.xticks([])
	pylab.axis((xmin, xmax, ymin, ymax))
	pylab.grid(True)

	pylab.show()


def plot_nested_variation_barchart(mean_value, category_value_dict, ylabel, title):
	"""
	Plot barchart showing variations between different nested categories

	:param mean_value:
		float, mean value
	:param category_value_dict:
		dict, mapping category names to dictionaries mapping subcategory names
		to subcategory values (lists)
	:param ylabel:
		str, label for Y axis
	:param title
		str, plot title
	"""
	category_names = category_value_dict.keys()
	num_categories = len(category_names)
	category_lengths = [len(category_value_dict[cat_name]) for cat_name in category_names]
	xvalues, x0_values, yvalues, labels = [], [], [], []
	for i, subcat_value_dict in enumerate(category_value_dict.values()):
		try:
			x0 = xvalues[-1] + 1
		except:
			x0 = 1
		x0values.append(x0)
		for j, (subcat_names, subcat_values) in enumerate(subcat_value_dict.items()):
			subcat_len = len(subcat_names)
			xvalues = np.concatenate([xvalues, np.add.accumulate(np.ones(subcat_len)) + x0])
			yvalues.extend(subcat_values)
			labels.extend(subcat.keys())

	yvalues = np.array(yvalues)
	diffs = yvalues - mean_value
	bar_width = 0.5

	rects = pylab.bar(xvalues, diffs, bar_width, color='r')

	def autolabel(rects, labels, values):
		# attach some text labels
		for rect, label, value in zip(rects, labels, values):
			x = rect.get_x() + rect.get_width() / 2.
			if value < 0:
				y = -rect.get_height() * 1.05
				va = 'top'
			else:
				y = rect.get_height() * 1.05
				va = 'bottom'

			pylab.text(x, y, label, ha='center', va=va)

	autolabel(rects, labels, diffs)
	xmin, xmax = 0.5, xvalues[-1] + 1
	ymin = diffs.min() + 0.2 * diffs.min()
	ymax = diffs.max() + 0.2 * diffs.max()
	pylab.hlines(0, xmin, xmax, lw=2, color='k')
	pylab.vlines(x0values[1:], ymin, ymax, lw=2, ls='--', color='k')
	pylab.text(xmax, 0, "%s" % mean_value, ha='left', va='center')
	pylab.ylabel("$\Delta$ " + ylabel)
	pylab.title(title)
	pylab.xticks([])
	pylab.axis((xmin, xmax, ymin, ymax))
	pylab.grid(True)

	pylab.show()





if __name__ == "__main__":
	from collections import OrderedDict

	mean_value = 0.5
	gmpe_names = ["GMPE1", "GMPE2", "GMPE3", "GMPE4"]
	gmpe_values = [0.7, 0.3, 0.45, 0.63]
	category_value_dict = OrderedDict()
	for gmpe_name, value in zip(gmpe_names, gmpe_values):
		category_value_dict[gmpe_name] = value
	ylabel = "PGA (g)"
	title = "Doel, ASC sources, Tr=1E+4 yr"
	plot_variation_barchart(mean_value, category_value_dict, ylabel, title)
