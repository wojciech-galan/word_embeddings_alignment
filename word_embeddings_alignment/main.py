from fastapi import FastAPI
from word_embeddings_alignment.src.router import water

app = FastAPI()
app.include_router(water.router)
