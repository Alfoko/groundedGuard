# CPU image for the groundguard HTTP service (NLI verifier by default).
#
#   docker build -t groundguard .
#   docker run -p 8000:8000 groundguard                      # /verify only
#   docker run -p 8000:8000 -v "$PWD/data/index:/app/data/index:ro" \
#       -e LLM_BASE_URL -e LLM_MODEL -e LLM_API_KEY groundguard  # /ask too
#
# The index is mounted rather than baked in: build it on the host with
# `groundguard ingest`. Pass --build-arg PRELOAD_MODEL=0 to skip downloading
# the NLI weights at build time; they are then fetched on the first request.

FROM python:3.11-slim

ARG TORCH_VERSION=2.15.0
ARG NLI_MODEL=MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
ARG PRELOAD_MODEL=1

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# CPU-only torch first, from the PyTorch index. The PyPI wheel bundles CUDA
# and is several times larger; with torch already installed, the pip
# install below leaves it alone.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[service,nli]"

RUN if [ "$PRELOAD_MODEL" = "1" ]; then \
        python -c "from transformers import AutoModelForSequenceClassification as M, AutoTokenizer as T; T.from_pretrained('${NLI_MODEL}'); M.from_pretrained('${NLI_MODEL}')"; \
    fi

RUN useradd --create-home --uid 1000 app \
    && mkdir -p data/index "$HF_HOME" \
    && chown -R app /app "$HF_HOME"
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["groundguard", "serve", "--host", "0.0.0.0", "--port", "8000"]
