# GPU torch is retained even when running a CPU-only device fallback.
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_ANALYTICS_ENABLED=False GRADIO_TEMP_DIR=/app/runs/web-cache \
    YOLO_CONFIG_DIR=/app/runs/ultralytics MPLCONFIGDIR=/app/runs/matplotlib \
    DESKVISION_QUEUE_MODE=celery DESKVISION_DATABASE=/app/runs/stage3.sqlite3 \
    DESKVISION_FINETUNED_WEIGHTS=/app/models/best.pt

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates libgl1 libglib2.0-0 libgomp1 fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*
COPY environment/requirements.txt environment/requirements-s5.txt environment/constraints.txt /tmp/deps/
RUN python -m pip install --no-cache-dir pip==25.2 setuptools==75.8.0 wheel==0.45.1 \
    && python -m pip install --no-cache-dir torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
       --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install --no-cache-dir -r /tmp/deps/requirements.txt \
       -r /tmp/deps/requirements-s5.txt -c /tmp/deps/constraints.txt \
       pandas==2.3.3 matplotlib==3.9.2 \
    && SAM2_BUILD_CUDA=0 python -m pip install --no-cache-dir --no-build-isolation --no-deps \
       git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4 \
    && python -m pip check

RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app \
    && mkdir -p /app/runs /app/models /app/.local \
    && chown -R app:app /app
COPY --chown=10001:10001 deskvision /app/deskvision
COPY --chown=10001:10001 configs /app/configs
COPY --chown=10001:10001 environment /app/environment
USER 10001:10001
ENTRYPOINT ["python", "/app/environment/container_entrypoint.py"]
CMD ["web"]
