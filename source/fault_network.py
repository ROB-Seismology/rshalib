

# Perhaps see also:
# https://www.python.org/doc/essays/graphs/


class FaultNetwork:
	"""
	Simple implementation of a fault network

	:param fault_links:
		dictionary, mapping fault IDs to tuples of two lists, containing
		the fault IDs linked on either side of each fault (i.e., 'start'
		links and 'end' links)
	"""
	def __init__(self, fault_links):
		self.fault_links = fault_links

	def __len__(self):
		return len(self.fault_links)

	@property
	def fault_ids(self):
		return self.fault_links.keys()

	def check_consistency(self):
		"""
		Check consistency of the network. Faults that are linked
		should be linked in both directions.
		Note: we do NOT impose that if flt_id2 is in end_links of flt_id1,
		flt_id1 should be in start_links of flt_id2

		:return:
			bool, indicating whether or not network is consistent
		"""
		missing_links = 0
		for flt_id1 in self.fault_ids:
			start_links1, end_links1 = self.fault_links[flt_id1]
			for flt_id2 in start_links1 + end_links1:
				start_links2, end_links2 = self.fault_links[flt_id2]
				if not flt_id1 in start_links2 + end_links2:
					missing_links += 1
					msg = "Fault ID %s not in start/end links of fault ID %s"
					msg %= (flt_id1, flt_id2)
					print(msg)
		if missing_links:
			return False
		else:
			return True

	def get_opposite_links(self, flt_id1, flt_id2):
		"""
		Return faults linked with fault 1 on the opposite side of fault 2

		:return:
			list of fault IDs
		"""
		start_links, end_links = self.fault_links[flt_id1]
		if flt_id2 in start_links:
			return end_links
		elif flt_id2 in end_links:
			return start_links

	def get_neighbours(self, flt_id):
		"""
		Return all faults linked with a particular fault

		:return:
			list of fault IDs
		"""
		start_links, end_links = self.fault_links[flt_id]
		return start_links + end_links

	def get_connections(self, flt_id, max_len, parents=[], allow_triple_junctions=True):
		"""
		Construct all fault connections that include a particular fault

		:param flt_id:
			ID of fault that should be included
		:param max_len:
			int, maximum number of linked fault sections
		:param parent:
			ID of 'parent' fault, connections through that fault should
			not be included (avoids going back to fault where we started
			when the function is applied recursively)
		:param allow_triple_junctions:
			bool, whether or not triple junctions are allowed
			(only plays a role when :param:`parent` is not None)
			(default: True)

		:return:
			list of tuples containing fault IDs that are connected
		"""
		if max_len < 1:
			return []
		elif max_len == 1:
			return [[flt_id]]
		else:
			connections = [[flt_id]]

			neighbours = self.get_neighbours(flt_id)
			if parents:
				if allow_triple_junctions:
					for parent in parents:
						if parent in neighbours:
							neighbours.remove(parent)
						"""
						try:
							idx = neighbours.index(parent)
						except:
							pass
						else:
							neighbours.pop(idx)
						"""
				else:
					neighbours = self.get_opposite_links(flt_id, parents[0])

			for flt_id2 in neighbours:
				sorted_connections = [sorted(conn) for conn in connections]
				parents2 = [flt_id] + parents
				connections2 = self.get_connections(flt_id2, max_len-1, parents=parents2,
									allow_triple_junctions=allow_triple_junctions)
				for conn2 in connections2:
					if not sorted([flt_id] + conn2) in sorted_connections:
						connections.append([flt_id] + conn2)

			return connections

	def get_all_connections(self, max_len, allow_triple_junctions=True):
		"""
		Construct all possible fault connections in the network

		:param max_len:
			int, maximum number of linked fault sections
		:param allow_triple_junctions:
			bool, whether or not triple junctions are allowed
			(default: True)

		:return:
			list of tuples containing fault IDs that are connected
		"""
		connections = []
		for flt_id in sorted(self.fault_ids):
			sorted_connections = [sorted(conn) for conn in connections]
			connections2 = self.get_connections(flt_id, max_len, parents=[],
								allow_triple_junctions=allow_triple_junctions)
			for conn2 in connections2:
				if not sorted(conn2) in sorted_connections:
					connections.append(tuple(conn2))
		#return sorted(connections)
		return connections




if __name__ == "__main__":
	fault_links = {'F01': ([], ['F02']),
					'F02': (['F01'], ['F03','F05']),
					'F03': (['F02'], ['F04']),
					'F04': (['F03'], []),
					'F05': (['F02'], ['F06']),
					'F06': ([], ['F05'])}
	fn = FaultNetwork(fault_links)
	print fn.check_consistency()
	#print fn.get_opposite_links(3, 2)
	#print fn.get_neighbours(2)
	#for conn in fn.get_connections(1, 2):
	for conn in fn.get_all_connections(3):
		print conn

