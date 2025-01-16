# Base image
FROM python:3.12-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src/ src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY configs/ configs/
COPY data/ data/

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install -r requirements_dev.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

ENTRYPOINT ["python", "-u", "src/signal_classification/evaluate.py"]
CMD []