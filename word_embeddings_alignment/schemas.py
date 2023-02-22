from pydantic import BaseModel


class InputData(BaseModel):
    seq_1: str
    seq_2: str
    gap_open: float
    gap_extend: float