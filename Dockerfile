# Gateway-only image. Does NOT contain mlx-lm or the model itself —
# mlx_lm.server keeps running natively on the Mac host (Docker on macOS
# runs inside a Linux VM with no access to Apple's Metal GPU, so the
# actual model server can never live in this container).
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (separate layer) so code edits below don't
# force a slow dependency reinstall on every rebuild.
COPY requirements-gateway.txt .
RUN pip install --no-cache-dir -r requirements-gateway.txt

# Only the two packages the gateway actually imports — not training/,
# data/, or mcp/, which have no business being in this image.
COPY src/modeltune/__init__.py ./modeltune/__init__.py
COPY src/modeltune/serving ./modeltune/serving
COPY src/modeltune/search ./modeltune/search

ENV PYTHONPATH=/app
EXPOSE 8100

CMD ["uvicorn", "modeltune.serving.gateway:app", "--host", "0.0.0.0", "--port", "8100"]
