FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Slim image, no baked deps — everything comes from venv in volume
CMD ["/app/start.sh"]
