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

# 安装系统依赖
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

# 安装 PyTorch（uaysk/vllm-pascal 测试版本: 2.5.1+cu121，CUDA 12.1 兼容宿主机 12.4）
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# 以 wheel 方式构建 vLLM（bdist_wheel 确保 .so 扩展被打包进 wheel，
# 避免 pip install . 的 PEP 517 temp copy 导致 cmake install 路径失效）
RUN python3 setup.py bdist_wheel --dist-dir=/workspace/dist

# 验证 wheel 中包含 _C 扩展
RUN python3 -c "
import zipfile, os
wheels = [f for f in os.listdir('/workspace/dist') if f.endswith('.whl')]
print('Built wheels:', wheels)
with zipfile.ZipFile(os.path.join('/workspace/dist', wheels[0]), 'r') as z:
    so_files = [f for f in z.namelist() if f.endswith('.so')]
    print('SO files in wheel:')
    for f in sorted(so_files):
        print(' ', f)
    py_files = [f for f in z.namelist() if f.endswith('.py')]
    print(f'Total .py files: {len(py_files)}')
    print(f'Total .so files: {len(so_files)}')
"

# 在构建阶段临时安装验证链接没问题
RUN pip install /workspace/dist/vllm-*.whl --no-deps && \
    python3 -c "import vllm._C; print('vLLM _C module loaded OK')" && \
    pip uninstall -y vllm

# 清理构建中间产物（保留 /workspace/dist 下的 wheel）
RUN rm -rf /root/.cache/pip /root/.cache/ccache /tmp/* \
    /workspace/vllm-pascal/.git \
    /workspace/vllm-pascal/build

# ---------------------------------------------------------------------------
# 阶段二: 运行阶段 — 仅包含运行时所需的最小依赖
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装 Python 3.12 运行时
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

# 从构建阶段复制虚拟环境和 wheel
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /workspace/dist /workspace/dist

ENV PATH="/opt/venv/bin:$PATH" \
    LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}"

# 安装 vLLM wheel（包含编译好的 _C.abi3.so）
RUN pip install /workspace/dist/vllm-*.whl --no-deps --no-cache-dir && \
    rm -rf /workspace/dist

WORKDIR /workspace

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 启动 OpenAI 兼容的 API 服务器
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
