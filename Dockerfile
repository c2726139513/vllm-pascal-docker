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
RUN git clone --depth 1 https://github.com/uaysk/vllm-pascal.git
WORKDIR /workspace/vllm-pascal

# 安装 Python 构建工具链
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "setuptools>=77,<81" wheel packaging cmake ninja jinja2 regex protobuf setuptools-scm numpy

# 安装 CUDA 12.4 适配的 PyTorch 2.5.1（vLLM 仅需要 torch）
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.5.1

# 从源码编译并安装 vLLM（非可编辑模式，所有产物进 site-packages）
RUN pip install . --no-build-isolation

# 验证 vLLM 编译产物可正确加载（提前暴露链接问题）
RUN python3 -c "import vllm._C; print('vLLM _C module loaded OK')"

# 清理构建中间产物
RUN rm -rf /root/.cache/pip /root/.cache/ccache /tmp/* \
    /workspace/vllm-pascal/.git \
    /workspace/vllm-pascal/build

# ---------------------------------------------------------------------------
# 阶段二: 运行阶段 — 仅包含运行时所需的最小依赖
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装 Python 3.12 运行时（仅 python3.12 本体 + venv，不要 -dev 头文件包）
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

# 从构建阶段复制虚拟环境（仅 site-packages，不含源码和构建缓存）
COPY --from=builder /opt/venv /opt/venv

# torch/lib 必须加入 LD_LIBRARY_PATH，否则 vLLM 编译的 _C 扩展
# 在运行时找不到 libtorch.so 中的符号（undefined symbol）
ENV PATH="/opt/venv/bin:$PATH" \
    LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}"
WORKDIR /workspace

# 健康检查（vLLM API 服务就绪探测）
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 启动 OpenAI 兼容的 API 服务器
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
