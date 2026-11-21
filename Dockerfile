# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv

COPY requirements.txt .
# sympy comes with onnxruntime but is only used by its model-conversion tools
RUN pip install --no-cache-dir -r requirements.txt && pip uninstall -y -q sympy mpmath

# Sticker cut-out model (ISNet anime, as published by rembg), checked against its hash.
# --chmod also sets the new /models folder, so it must allow entering it (755).
ADD --chmod=755 --checksum=sha256:f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99 \
    https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-anime.onnx /models/isnet-anime.onnx

COPY app ./app

RUN useradd --create-home --uid 1000 app && mkdir /data && chown app /data
ENV DATA_DIR=/data
USER app

EXPOSE 8000
# the app listens on 8000, the GPU helper (python -m app.broker) on 9000
HEALTHCHECK --interval=30s --timeout=5s CMD ["python", "-m", "app.healthcheck"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
