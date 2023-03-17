class SimpleAlignmentRepresentation(object):

	def __init__(self, score):
		super().__init__()
		self._seq1 = []
		self._seq2 = []
		self._score = score

	@property
	def seq1(self):
		return ''.join(reversed(self._seq1))

	@property
	def seq2(self):
		return ''.join(reversed(self._seq2))

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
