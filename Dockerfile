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

ARG MAX_JOBS=2
ENV MAX_JOBS=${MAX_JOBS}

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1"
ENV VLLM_TARGET_DEVICE="cuda"

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

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace
RUN git clone --depth 1 https://github.com/uaysk/vllm-pascal.git
WORKDIR /workspace/vllm-pascal

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "setuptools>=77,<81" wheel packaging cmake ninja jinja2 regex protobuf setuptools-scm numpy

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# 使用可编辑模式构建（这是 uaysk/vllm-pascal 官方推荐的安装方式）
# CMake 会将 _C.abi3.so 等扩展文件直接输出到源码目录 vllm/ 下
RUN pip install -e . --no-build-isolation

# 列出源码 vllm/ 目录中的 .so 文件（确认 CMake 产生了哪些）
RUN python3 <<'PYEOF'
import os, glob
vllm_dir = '/workspace/vllm-pascal/vllm'
so_files = sorted(glob.glob(os.path.join(vllm_dir, '*.so*')))
print('SO files in source vllm/:')
for f in so_files:
    sz = os.path.getsize(f)
    print(f'  {os.path.basename(f):40s} {sz//1024:>6d} KB')
if not so_files:
    print('  (none found!)')
    import sys; sys.exit(1)
PYEOF

# 将源码 vllm/ 目录（含 .so 文件）复制到 site-packages，转为普通安装
RUN rm -rf /opt/venv/lib/python3.12/site-packages/vllm* && \
    cp -r /workspace/vllm-pascal/vllm /opt/venv/lib/python3.12/site-packages/vllm && \
    rm -f /opt/venv/lib/python3.12/site-packages/vllm.egg-link
# 注意: 此处不验证 import vllm._C，因为 builder 阶段没有 NVIDIA 驱动（libcuda.so.1）
# 运行时验证将在容器启动后由 NVIDIA 容器运行时提供

# 清理
RUN rm -rf /root/.cache/pip /root/.cache/ccache /tmp/* \
    /workspace/vllm-pascal/.git \
    /workspace/vllm-pascal/build

# ---------------------------------------------------------------------------
# 阶段二: 运行阶段
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}"

WORKDIR /workspace

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
