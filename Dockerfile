FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# gcc/build-essential are only needed to compile wheel-less deps (e.g.
# sentencepiece, chromadb's hnswlib) during `pip install`; they never make it
# into the final image below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# --index-url pytorch's CPU wheel repo (checked first) + --extra-index-url
# PyPI (fallback for everything else): without this, `pip install torch`
# resolves to the default CUDA build on linux/amd64, pulling in ~1.5GB of
# unused nvidia-cu*/cuDNN/NCCL packages that a GPU-less Cloud Run instance
# never touches.
RUN pip install --prefix=/install \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt

FROM python:3.12-slim AS final

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV PYTHONPATH=/app/packages

WORKDIR /app

# curl/ca-certificates: container healthcheck.
# libgl1/tesseract-ocr: runtime deps of presidio-image-redactor + pytesseract
# (OCR on redacted images) -- see packages.txt, which lists the same pair for
# the Streamlit Cloud deployment of this app.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgl1 \
    tesseract-ocr \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install/lib/python3.12/site-packages /app/packages

COPY . .

# GPT_API, GEMINI_API, GROQ_API, PARENT_PATH have no defaults in config.py and
# must be supplied as Cloud Run env vars / secrets at deploy time -- they are
# deliberately not baked into the image.
ENV PORT=8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "python -m uvicorn api.app:app --host 0.0.0.0 --port ${PORT}"]
