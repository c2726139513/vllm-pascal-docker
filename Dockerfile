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

# 安装构建依赖（镜像 pyproject.toml build-system.requires）
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "setuptools>=77.0.3,<81.0.0" wheel packaging cmake ninja \
        jinja2 regex protobuf setuptools-scm numpy grpcio-tools==1.78.0

# 安装 PyTorch 2.10.0（项目实际要求的版本，从 PyPI 获取 2.10.0+cu128）
# 不再经过 cu121 占位再升级的中间步骤
RUN pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

# 使用可编辑模式构建（官方推荐方式）
# pip 先解析依赖：cuda.txt 要求 torch==2.10.0（已装好，跳过）→ CMake 编译 .so
RUN pip install -e . --no-build-isolation

# 确认构建产物
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

# 将源码 vllm/ 目录（含 .so 文件）复制到 site-packages，替换掉 editable 链接
# 注意: 只能用精确路径删除，不能用 vllm* 通配符（会误删 vllm-*.dist-info/）。
#       .dist-info 是 importlib.metadata.version("vllm") 的查询来源，
#       删掉会导致 vllm.platforms.__init__ 中 CUDA 平台检测失败 → "Failed to infer device type"
RUN rm -rf /opt/venv/lib/python3.12/site-packages/vllm \
           /opt/venv/lib/python3.12/site-packages/vllm.egg-link \
           /opt/venv/lib/python3.12/site-packages/__editable__.*vllm* \
    && cp -r /workspace/vllm-pascal/vllm /opt/venv/lib/python3.12/site-packages/vllm
# 注意: 此处不验证 import vllm._C，因为 builder 无 GPU 驱动（libcuda.so.1 需要运行时提供）

# 清理构建依赖和源码
RUN rm -rf /root/.cache/pip /root/.cache/ccache /tmp/* \
    /workspace/vllm-pascal

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
