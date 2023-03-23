from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import status
from fastapi import Form
from fastapi import File
from fastapi import UploadFile
from fastapi.responses import PlainTextResponse
from word_embeddings_alignment.src.fasta_alignment import FastaAlignment
from word_embeddings_alignment.src.utils import align
from word_embeddings_alignment.src.utils import read_fasta_buffer
from word_embeddings_alignment.src.regular_water.matrices.edna_full import EDNAFULL_matrix
from word_embeddings_alignment.data_transformation import read_prot_vec
from word_embeddings_alignment.schemas import InputData
from word_embeddings_alignment.schemas import SequenceType
import blosum as bl

PROTEIN_EMBEDDINGS = read_prot_vec.read()
router = APIRouter(prefix='/water', tags=['water'])


@router.post('/regular_water', response_class=PlainTextResponse)
def calculate_alignment(request: InputData):
	if request.sequence_type == SequenceType.nucleic:
		res = align(request.seq_1, request.seq_2, EDNAFULL_matrix, request.gap_open, request.gap_extend, 'classic',
		            request.return_multiple)
	else:
		res = align(request.seq_1, request.seq_2, bl.BLOSUM(45), request.gap_open, request.gap_extend, 'classic',
		            request.return_multiple)
	return PlainTextResponse(
		content=str('\n'.join(str(x) for x in res)),
		media_type='text/plain'
	)


@router.post('/word_embeddings_water', response_class=PlainTextResponse)
def calculate_alignment(request: InputData):
	if request.sequence_type == SequenceType.protein:
		res = align(request.seq_1, request.seq_2, PROTEIN_EMBEDDINGS, request.gap_open, request.gap_extend,
		            'word_embeddings', request.return_multiple)
	else:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Not implemented yet')
	return PlainTextResponse(
		content=str('\n'.join(str(x) for x in res)),
		media_type='text/plain'
	)

@router.post('/word_embeddings_water_fasta', response_class=PlainTextResponse)
def calculate_alignment(
		seq_1_file: UploadFile = File(...),
	    seq_2_file: UploadFile = File(...),
	    gap_open: float = Form(...),
	    gap_extend: float = Form(...),
		sequence_type: SequenceType = Form(...)
):
	if sequence_type == SequenceType.protein:
		id_1, seq_1 = next(read_fasta_buffer(seq_1_file.file))
		id_2, seq_2 = next(read_fasta_buffer(seq_2_file.file))
		alignment = next(align(seq_1, seq_2, PROTEIN_EMBEDDINGS, gap_open, gap_extend, 'word_embeddings', False))
		fasta_alignment = FastaAlignment(alignment, seq_1, seq_2, id_1, id_2)
	else:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Not implemented yet')
	return PlainTextResponse(
		content=str(fasta_alignment),
		media_type='text/plain'
	)