from fastapi import FastAPI
from router import water

app = FastAPI()
app.include_router(water.router)