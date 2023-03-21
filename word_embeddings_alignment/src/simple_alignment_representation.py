class SimpleAlignmentRepresentation(object):

	def __init__(self, score, seq1_start, seq2_start):
		super().__init__()
		self._seq1 = []
		self._seq2 = []
		self._seq1_start_position = seq1_start
		self._seq2_start_position = seq2_start
		self._seq1_end_position = None
		self._seq2_end_position = None
		self._score = score

	@property
	def seq1(self):
		return ''.join(reversed(self._seq1))

	@property
	def seq2(self):
		return ''.join(reversed(self._seq2))

	@property
	def seq1_start_position(self):
		return self._seq1_start_position

	@property
	def seq2_start_position(self):
		return self._seq2_start_position

	@property
	def seq1_end_position(self):
		if self._seq1_end_position:
			return self._seq1_end_position
		raise AttributeNotSet('Not set yet')

	@property
	def seq2_end_position(self):
		if self._seq2_end_position:
			return self._seq2_end_position
		raise AttributeNotSet('Not set yet')

	@property
	def score(self):
		return self._score

	def add_data(self, char_a: str, char_b: str):
		self._seq1.append(char_a)
		self._seq2.append(char_b)

	def __str__(self):
		return f"{self.seq1}\n{self.seq2}\nScore:{self.score:.2f}"

	def __eq__(self, other):
		if isinstance(other, SimpleAlignmentRepresentation):
			return self.__dict__ == other.__dict__
		return False


class AttributeNotSet(RuntimeError):
	pass
