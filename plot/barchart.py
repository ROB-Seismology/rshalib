"""
"""

import numpy as np
import pylab


def plot_variation_barchart(mean_value, category_value_dict, ylabel, title, ymin=None, ymax=None, bar_width=0.35, color='r', fig_filespec=None, fig_width=0, dpi=300):
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

	rects = pylab.bar(xvalues, category_diffs, bar_width, color=color)

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
	min_diff, max_diff = category_diffs.min(), category_diffs.max()
	if ymin is None:
		if min_diff < 0:
			ymin = min_diff + 0.2 * min_diff
		else:
			ymin = mean_value
	if ymax is None:
		if max_diff > 0:
			ymax = max_diff + 0.4 * max_diff
		else:
			ymax = mean_value
	pylab.hlines(0, xmin, xmax, lw=2, color='k')
	pylab.text(xmax, 0, "%s" % mean_value, ha='left', va='center')
	pylab.ylabel("$\Delta$ " + ylabel)
	pylab.title(title)
	pylab.xticks([])
	pylab.axis((xmin, xmax, ymin, ymax))
	pylab.grid(True)

	if fig_filespec:
		default_figsize = pylab.rcParams['figure.figsize']
		default_dpi = pylab.rcParams['figure.dpi']
		if fig_width:
			fig_width /= 2.54
			dpi = dpi * (fig_width / default_figsize[0])

		pylab.savefig(fig_filespec, dpi=dpi)
		pylab.clf()
	else:
		pylab.show()


def plot_nested_variation_barchart(mean_value, category_value_dict, ylabel, title, ymin=None, ymax=None, bar_width=0.5, color='r', fig_filespec=None, fig_width=0, dpi=300):
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
	def autolabel(ax, rects, labels, values):
		# attach some text labels
		for rect, label, value in zip(rects, labels, values):
			x = rect.get_x() + rect.get_width() / 2.
			if value < 0:
				y = -rect.get_height()
				offset = (0, -6)
				va = 'top'
			else:
				y = rect.get_height()
				offset = (0, 6)
				va = 'bottom'

			#ax.text(x, y, label, ha='center', va=va)
			ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", ha='center', va=va)

	def update_ax2(ax1):
	   y1, y2 = ax1.get_ylim()
	   ax2.set_ylim(y1 + mean_value, y2 + mean_value)
	   ax2.figure.canvas.draw()

	fig, ax1 = pylab.subplots()
	ax2 = ax1.twinx()

	category_names = category_value_dict.keys()
	num_categories = len(category_names)
	category_lengths = [len(category_value_dict[cat_name]) for cat_name in category_names]
	xvalues, x0values, yvalues, labels, colors = [], [], [], [], []
	for i, (cat_name, subcat_value_dict) in enumerate(category_value_dict.items()):
		try:
			x0 = xvalues[-1] + 1
		except:
			x0 = 0
		x0values.append(x0)
		subcat_len = len(subcat_value_dict)
		xvalues = np.concatenate([xvalues, np.add.accumulate(np.ones(subcat_len)) + x0])
		for j, (subcat_name, subcat_value) in enumerate(subcat_value_dict.items()):
			yvalues.append(subcat_value)
			labels.append(subcat_name)
			try:
				c = color[cat_name][subcat_name]
			except:
				try:
					c = color[subcat_name]
				except:
					try:
						c = color[cat_name]
					except:
						c = color
			colors.append(c)
	x0values.append(xvalues[-1] + 1)
	x0values = np.array(x0values)

	## Set up double Y axes
	ax1.callbacks.connect("ylim_changed", update_ax2)

	## Plot bars
	yvalues = np.array(yvalues)
	diffs = yvalues - mean_value
	rects = ax1.bar(xvalues, diffs, bar_width, color=colors)

	## Draw horizontal line at mean value and vertical lines to separate categories
	xmin, xmax = 0, xvalues[-1] + 1 + bar_width
	min_diff, max_diff = diffs.min(), diffs.max()
	if ymin is None:
		if min_diff < 0:
			ymin = min_diff + 0.2 * min_diff
		else:
			ymin = mean_value
	if ymax is None:
		if max_diff > 0:
			ymax = max_diff + 0.4 * max_diff
		else:
			ymax = mean_value
	ax1.hlines(0, xmin, xmax, lw=2, color='k')
	ax1.vlines(x0values[1:-1]+bar_width/2., ymin, ymax, lw=2, linestyle='--', color='k')
	ax1.axis((xmin, xmax, ymin, ymax))

	## Label bars
	autolabel(ax1, rects, labels, diffs)

	## Label main categories
	for i, cat_name in enumerate(category_names):
		x = (x0values[i:i+2] + bar_width/2).mean()
		ax1.annotate(cat_name, (x, ymax), xytext=(0, -6), textcoords="offset points", ha='center', va='top')

	#ax1.text(xmax, 0, "%s" % mean_value, ha='left', va='center')

	ax1.set_ylabel("$\Delta$ " + ylabel)
	ax2.set_ylabel(ylabel)
	ax1.set_title(title)
	ax1.set_xticks([])
	ax1.grid(True)

	if fig_filespec:
		default_figsize = pylab.rcParams['figure.figsize']
		default_dpi = pylab.rcParams['figure.dpi']
		if fig_width:
			fig_width /= 2.54
			dpi = dpi * (fig_width / default_figsize[0])

		pylab.savefig(fig_filespec, dpi=dpi)
		pylab.clf()
	else:
		pylab.show()




if __name__ == "__main__":
	from collections import OrderedDict

	"""
	mean_value = 0.5
	gmpe_names = ["GMPE1", "GMPE2", "GMPE3", "GMPE4"]
	gmpe_values = [0.7, 0.3, 0.45, 0.63]
	category_value_dict = OrderedDict()
	for gmpe_name, value in zip(gmpe_names, gmpe_values):
		category_value_dict[gmpe_name] = value
	ylabel = "PGA (g)"
	title = "Doel, ASC sources, Tr=1E+4 yr"
	plot_variation_barchart(mean_value, category_value_dict, ylabel, title)
	"""

	mean_value = 0.040329
	category_value_dict = {'F_2010': {'TZ': 0.046959064687517017, 'LE': 0.034461389039388036, 'SH': 0.057164967504535466}, 'Bi_2011': {'TZ': 0.022108010941297277, 'LE': 0.016370916981100894, 'SH': 0.063303583715201187}, 'A_2013': {'TZ': 0.035813658675486783, 'LE': 0.026141876429232672, 'SH': 0.042522319984264326}, 'Z_2006': {'TZ': 0.028407355714886597, 'LE': 0.021415252280295456, 'SH': 0.036726634123728154}, "BA_2008'": {'TZ': 0.045361042694182202, 'LE': 0.034428230784055755, 'SH': 0.047580601898182501}}
	ylabel = "PGA (g)"
	title = "Doel, ASC, T=0s, Tr=10000 yr"
	color = {'LE': "green", "SH": "red", "TZ": "blue"}
	plot_nested_variation_barchart(mean_value, category_value_dict, ylabel, title, color=color, bar_width=0.75)
