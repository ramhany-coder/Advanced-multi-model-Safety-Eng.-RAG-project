from contextlib import asynccontextmanager
from fastapi import FastAPI
from config import settings
from Embedding_Model.model import Embedding_Model
from loggers import setup_logger

logger = setup_logger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pass local directory path and specify local_files_only=True
    client_embedding_model = Embedding_Model(settings.EMBEDDING_MODEL_PATH)
    yield
    

app = FastAPI(lifespan=lifespan)
