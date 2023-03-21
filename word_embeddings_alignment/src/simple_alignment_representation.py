class SimpleAlignmentRepresentation(object):

	def __init__(self, score, seq1_end, seq2_end):
		super().__init__()
		self._seq1 = []
		self._seq2 = []
		self._seq1_start_position = None
		self._seq2_start_position = None
		self._seq1_end_position = seq1_end
		self._seq2_end_position = seq2_end
		self._score = score

	@property
	def seq1(self):
		return ''.join(reversed(self._seq1))

	@property
	def seq2(self):
		return ''.join(reversed(self._seq2))

	@property
	def seq1_end_position(self):
		return self._seq1_end_position

	@property
	def seq2_end_position(self):
		return self._seq2_end_position

	@property
	def seq1_start_position(self):
		if self._seq1_start_position:
			return self._seq1_start_position
		raise AttributeNotSet('Not set yet')

	@property
	def seq2_start_position(self):
		if self._seq2_start_position:
			return self._seq2_start_position
		raise AttributeNotSet('Not set yet')

	@property
	def score(self):
		return self._score

	def add_data(self, char_a: str, char_b: str):
		self._seq1.append(char_a)
		self._seq2.append(char_b)

	def set_start_position(self, seq1_start: int, seq2_start: int):
		self._seq1_start_position = seq1_start
		self._seq2_start_position = seq2_start

	def __str__(self):
		return f"{self.seq1}\n{self.seq2}\nScore:{self.score:.2f}"

	def __eq__(self, other):
		if isinstance(other, SimpleAlignmentRepresentation):
			return self.__dict__ == other.__dict__
		return False


class AttributeNotSet(RuntimeError):
	pass
