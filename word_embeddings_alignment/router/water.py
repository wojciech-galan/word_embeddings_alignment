from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from regular_water import my_water
from regular_water.matrix import EDNAFULL_matrix
from schemas import InputData

router = APIRouter(prefix='/water', tags=['water'])


@router.post('/regular_water', response_class=PlainTextResponse)
def align(request: InputData):
	res = my_water.align(request.seq_1, request.seq_2, EDNAFULL_matrix, request.gap_open, request.gap_extend)
	return PlainTextResponse(
		content=str(res),
		media_type='text/plain'
	)
