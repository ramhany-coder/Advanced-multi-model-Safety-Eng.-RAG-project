import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# Importing api.endpoints pulls in every agent module, which is what actually
# constructs the embedding and Whisper models today (module-level singletons)
# -- so by the time `lifespan` below runs, those two have already gone
# through their own "is it in the local model dir yet?" check on import. The
# PII engines are lazy, though, so `warm_up_pii_engines` is what makes their
# check-and-download-if-missing step happen at startup instead of on the
# first live request.
from api.endpoints import router
from agents.PII.helpers import warm_up_pii_engines
from config import settings

logger = logging.getLogger("api.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.WARM_UP_PII_ON_STARTUP:
        logger.info("Startup: verifying local PII models (downloads only if missing)...")
        warm_up_pii_engines()
        logger.info("Model check complete.")
    else:
        logger.info(
            "Skipping PII engine warm-up at startup (WARM_UP_PII_ON_STARTUP=False); "
            "engines will load lazily on first use instead."
        )
    yield


app = FastAPI(title="OSHA Multimodal RAG Pipeline API", lifespan=lifespan)

app.include_router(router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}


# Run from the project root with:
#   uvicorn api.app:app --reload
