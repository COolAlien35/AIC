FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/workspace/.hf_home \
    TRANSFORMERS_CACHE=/workspace/.hf_home \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    NCCL_IB_DISABLE=1 \
    NCCL_P2P_DISABLE=0 \
    NCCL_DEBUG=WARN \
    TORCH_NCCL_BLOCKING_WAIT=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4

# System packages: Python 3.11, git, build tools for bitsandbytes/wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        python3-pip git curl ca-certificates build-essential \
        libgomp1 unzip zip \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# HF Spaces requires UID 1000 named "user"
RUN useradd -m -u 1000 user
USER user
WORKDIR /workspace

# Install JupyterLab and core CLI tools as the user
RUN python -m pip install --user --no-cache-dir --upgrade pip \
    && python -m pip install --user --no-cache-dir \
        jupyterlab==4.2.5 ipywidgets==8.1.5 hf_transfer

ENV PATH="/home/user/.local/bin:${PATH}"

COPY --chown=user:user start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

EXPOSE 7860

CMD ["/workspace/start.sh"]
