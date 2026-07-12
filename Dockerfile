# =============================================================================
# 多阶段构建: vLLM for Pascal GPU (uaysk/vllm-pascal)
# 目标运行时环境: CUDA 12.4
# =============================================================================

# ---------------------------------------------------------------------------
# 阶段一: 构建阶段 — 编译 vLLM CUDA 内核
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 限制并行编译任务数，防止 OOM（GitHub Actions 标准 runner 为 2 vCPU）
ARG MAX_JOBS=2
ENV MAX_JOBS=${MAX_JOBS}

# 仅编译 Pascal 架构 (sm_60, sm_61) 的 CUDA 内核，大幅缩短构建时间
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1"
ENV VLLM_TARGET_DEVICE="cuda"

# 安装系统依赖:
#   - Ubuntu 22.04 默认 Python 为 3.10，通过 deadsnakes PPA 安装 Python 3.12
#   - gcc-12 兼容 CUDA 12.4 的要求
#   - ccache 加速重复构建
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        gcc-12 g++-12 \
        cmake ninja-build \
        git curl ccache && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

# 创建并激活虚拟环境
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 克隆 uaysk/vllm-pascal 仓库
WORKDIR /workspace
RUN git clone https://github.com/uaysk/vllm-pascal.git
WORKDIR /workspace/vllm-pascal

# 安装 Python 构建工具链
RUN pip install --upgrade pip && \
    pip install "setuptools>=77,<81" wheel packaging cmake ninja jinja2 regex protobuf setuptools-scm numpy

# 安装 CUDA 12.4 适配的 PyTorch 2.5.1（vLLM 仅需要 torch）
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.5.1

# 从源码编译并安装 vLLM（--no-build-isolation 确保使用已安装的 PyTorch）
RUN pip install -e . --no-build-isolation

# 清理构建缓存，减小最终构建阶段体积
RUN rm -rf /root/.cache/pip /root/.cache/ccache /tmp/*

# ---------------------------------------------------------------------------
# 阶段二: 运行阶段 — 仅包含运行时所需的最小依赖
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1"

# 安装 Python 3.12 运行时（仅所需的最小包）
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev && \
    rm -rf /var/lib/apt/lists/*

# 从构建阶段复制虚拟环境和源码（保持路径一致，可编辑安装仍有效）
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /workspace/vllm-pascal /workspace/vllm-pascal

ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /workspace/vllm-pascal

# 健康检查（vLLM API 服务就绪探测）
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 启动 OpenAI 兼容的 API 服务器
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
