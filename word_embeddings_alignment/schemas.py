from enum import Enum
from typing import Optional
from pydantic import BaseModel


class SequenceType(str, Enum):
    nucleic: str = 'nucleic'
    protein: str = 'protein'


class InputData(BaseModel):
    seq_1: str
    seq_2: str
    gap_open: float
    gap_extend: float
    sequence_type: SequenceType
    return_multiple: Optional[bool]
