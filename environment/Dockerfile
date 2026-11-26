FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv python3-dev build-essential ffmpeg libsndfile1 libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY environment/requirements-inference.txt /tmp/requirements-inference.txt
RUN python3 -m pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-inference.txt
COPY . /app
RUN python3 -m pip install --no-cache-dir --no-deps .
ENTRYPOINT ["sva"]
