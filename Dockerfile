ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# avoid selecting 'Geographic area' during installation
ARG DEBIAN_FRONTEND=noninteractive

# apt install required packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    ninja-build \
    libglib2.0-0 \
    libxrender-dev \
    libjpeg-dev \
    libpng-dev \
    git \
    wget \
    sudo \
    htop \
    tmux \
    openssh-server \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/Baijiong-Lin/LoRA-Torch
RUN python -m ipykernel install --name docker_env --display-name "Python (Docker)"

EXPOSE 8888

WORKDIR /workspace