# SAM 3D Objects 설치 메모 - Ubuntu 24.04 + RTX 5060 Ti

이 문서는 `demo.py`를 RTX 5060 Ti 환경에서 실행하기 위해, 우선 **호환성 문제를 줄이는 버전 조합**을 정리한 메모입니다.

현재 공식 README의 기본 설치 흐름은 `torch 2.5.1 + cu121` 계열에 가깝습니다. RTX 5060 Ti는 Blackwell 계열 GPU라서 이 조합에서는 `sm_120 is not compatible`, `no kernel image is available for execution on the device` 같은 오류가 날 수 있습니다. 따라서 Ubuntu 24.04는 유지하되, Python 환경 안의 PyTorch/CUDA stack은 `cu128` 이상으로 맞추는 방향을 권장합니다.

## 목표 버전

우선 아래 조합을 1차 목표로 둡니다.

```text
OS: Ubuntu 24.04
GPU: RTX 5060 Ti
NVIDIA Driver: 570 이상 권장
Python: 3.11
CUDA Toolkit / nvcc: 12.8
PyTorch: 2.8.0 + cu128
torchvision: 0.23.0 + cu128
torchaudio: 2.8.0 + cu128
Kaolin: 0.18.0, torch-2.8.0_cu128 wheel
```

중요한 점:

- `nvidia-smi`에 보이는 CUDA 버전은 드라이버가 지원하는 최대 런타임 버전에 가깝습니다.
- PyTorch 호환성에서는 `torch.__version__`이 `+cu128`인지가 더 중요합니다.
- 시스템 CUDA 12.8은 `pytorch3d`, `gsplat`, `spconv` 같은 CUDA extension을 빌드할 때 필요합니다.
- OOM은 별도 문제입니다. 이 문서는 우선 호환성 문제를 줄이는 설치 순서를 다룹니다.

## 먼저 다운로드할 것

아래 파일/페이지를 먼저 준비합니다.

1. NVIDIA Driver

   Ubuntu 24.04에서 RTX 5060 Ti를 쓰려면 Blackwell을 지원하는 최신 드라이버가 필요합니다. 최소 570 계열 이상을 권장합니다. 가능하면 Ubuntu의 `ubuntu-drivers` 또는 NVIDIA 공식 드라이버 페이지에서 최신 production/studio 계열을 설치합니다.

   확인:

   ```bash
   nvidia-smi
   ```

2. CUDA Toolkit 12.8

   NVIDIA CUDA 12.8 archive에서 Ubuntu 24.04 / x86_64 / deb network 또는 deb local 방식으로 설치합니다.

   공식 페이지:

   ```text
   https://developer.nvidia.com/cuda-12-8-0-download-archive
   ```

   deb network 방식을 쓰면 보통 아래 흐름입니다.

   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
   sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600

   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb

   sudo apt update
   sudo apt install cuda-toolkit-12-8
   ```

   설치 후 shell 설정:

   ```bash
   export CUDA_HOME=/usr/local/cuda-12.8
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}
   ```

   확인:

   ```bash
   nvcc --version
   ```

3. SAM 3D Objects checkpoint

   Hugging Face 접근 권한 승인 후 다운로드합니다.

   ```bash
   pip install 'huggingface-hub[cli]<1.0'
   hf auth login

   TAG=hf
   hf download \
     --repo-type model \
     --local-dir checkpoints/${TAG}-download \
     --max-workers 1 \
     facebook/sam-3d-objects

   mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
   rm -rf checkpoints/${TAG}-download
   ```

   다운로드가 끝나면 아래 파일이 있어야 합니다.

   ```text
   checkpoints/hf/pipeline.yaml
   ```

## mamba 환경 생성

이미 새 환경을 만들었다면 이 단계는 건너뜁니다. 새로 판다면 Python 3.11을 권장합니다.

```bash
mamba create -n sam3d-bw python=3.11 -y
mamba activate sam3d-bw

python -m pip install -U pip setuptools wheel ninja cmake packaging
```

## PyTorch cu128 설치

기존에 설치된 torch 계열이 있으면 먼저 지웁니다.

```bash
pip uninstall -y torch torchvision torchaudio xformers flash-attn flash_attn
```

그 다음 cu128 wheel을 설치합니다.

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

확인:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device="cuda")
print("matmul ok:", (x @ x).shape)
PY
```

기대값:

```text
torch: 2.8.0+cu128
cuda runtime: 12.8
capability: (12, 0)
```

## repo 설치는 --no-deps로 시작

공식 README처럼 바로 `pip install -e '.[dev]'`를 실행하면 `requirements.txt`의 구버전 CUDA 패키지들이 다시 들어올 수 있습니다. 우선 repo 자체만 설치합니다.

```bash
pip install -e . --no-deps
```

그 다음 일반 Python 의존성을 필요한 것부터 설치합니다. 아래는 시작점입니다.

```bash
pip install \
  hydra-core==1.3.2 \
  hydra-submitit-launcher==1.2.0 \
  omegaconf \
  loguru==0.7.2 \
  lightning==2.3.3 \
  safetensors \
  tqdm \
  pillow \
  numpy \
  scipy \
  einops \
  trimesh \
  open3d==0.18.0 \
  opencv-python==4.9.0.80 \
  scikit-image==0.23.1 \
  timm==0.9.16 \
  roma==1.5.1 \
  point-cloud-utils==0.29.5 \
  pymeshfix==0.17.0 \
  xatlas==0.0.9 \
  gradio==5.49.0 \
  seaborn==0.13.2
```

## Kaolin 설치

`requirements.inference.txt`의 `kaolin==0.17.0` 대신 `torch 2.8.0 + cu128`에 맞는 Kaolin wheel을 설치합니다.

```bash
pip install kaolin==0.18.0 \
  -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
```

확인:

```bash
python - <<'PY'
import kaolin
print("kaolin:", kaolin.__version__)
PY
```

## PyTorch3D 설치

이 repo는 PyTorch3D를 많이 import합니다. cu128 조합에서는 prebuilt wheel이 없을 수 있으므로 source build를 시도합니다.

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST="12.0"

pip install "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
```

빌드가 실패하면 먼저 아래 시스템 패키지를 확인합니다.

```bash
sudo apt install -y build-essential git cmake ninja-build
```

## gsplat 설치

Gaussian rendering 쪽에서 `gsplat`을 사용합니다.

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST="12.0"

pip install "git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"
```

## MoGe 설치

pointmap/depth model 쪽에서 MoGe가 필요합니다.

```bash
pip install "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"
```

## attention backend는 일단 sdpa 사용

처음 호환성 맞출 때는 `flash_attn`, `xformers`를 끄고 PyTorch 기본 SDPA로 시작합니다.

```bash
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=sdpa
```

`flash_attn`은 나중에 별도로 `torch 2.8.0 + cu128 + sm_120`에서 빌드 가능한지 확인한 뒤 붙이는 편이 낫습니다.

## spconv 주의

가장 까다로운 부분입니다. repo의 sparse backend 기본값은 `spconv`입니다.

```text
sam3d_objects/model/backbone/tdfy_dit/modules/sparse/__init__.py
BACKEND = "spconv"
```

공식 requirements는 `spconv-cu121==2.3.8`을 요구하지만, RTX 5060 Ti에서는 cu121 wheel을 그대로 쓰면 호환성 문제가 날 가능성이 큽니다.

우선 시도 순서:

```bash
pip uninstall -y spconv spconv-cu121 spconv-cu124 spconv-cu126 cumm cumm-cu121 cumm-cu124 cumm-cu126
```

그 다음 선택지는 둘입니다.

1. `spconv`를 CUDA 12.8 / arch 12.0 대상으로 source 또는 JIT build
2. repo의 sparse backend를 `torchsparse`로 바꾸고 torchsparse를 별도 빌드

1번을 먼저 시도하는 것을 권장합니다.

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST="12.0"
export CUMM_CUDA_ARCH_LIST="12.0"
```

spconv 쪽은 환경에 따라 빌드 절차가 달라질 수 있으므로, 여기서 실패하면 에러 로그를 기준으로 따로 맞추는 것이 좋습니다. `demo.py` 호환성 문제의 마지막 큰 산이 보통 이 지점입니다.

## 설치 후 점검

아래 import가 통과하는지 먼저 봅니다.

```bash
python - <<'PY'
import torch
import torchvision
import kaolin
import pytorch3d
import gsplat
import moge
import sam3d_objects

print("torch:", torch.__version__, torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("kaolin:", kaolin.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY
```

그 다음 checkpoint가 있는지 확인합니다.

```bash
ls checkpoints/hf/pipeline.yaml
```

마지막으로 demo를 실행합니다.

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=sdpa
export TORCH_CUDA_ARCH_LIST="12.0"
export CUMM_CUDA_ARCH_LIST="12.0"

python demo.py
```

## 피해야 할 설치 명령

RTX 5060 Ti 환경에서는 아래 명령을 처음부터 그대로 실행하지 않는 편이 좋습니다.

```bash
pip install -e '.[dev]'
pip install -e '.[p3d]'
pip install -e '.[inference]'
```

이 명령들은 repo의 pinned requirements를 따라가면서 `torch 2.5.1 + cu121`, `spconv-cu121`, `kaolin 0.17.0`, `xformers 0.0.28.post3` 같은 조합을 다시 끌어올 수 있습니다.

## 오류 구분

호환성 문제:

```text
sm_120 is not compatible
no kernel image is available for execution on the device
undefined symbol
cannot import xformers / flash_attn / spconv
```

VRAM 부족 문제:

```text
CUDA out of memory
```

OOM은 이후 inference 단계를 쪼개거나 decode 옵션을 줄여서 별도로 대응합니다. 이 문서의 1차 목표는 `demo.py`가 최소한 import와 CUDA kernel 호환성에서 막히지 않게 만드는 것입니다.

## 참고 링크

- PyTorch install matrix: https://pytorch.org/get-started
- PyTorch previous versions: https://pytorch.org/get-started/previous-versions
- CUDA 12.8 archive: https://developer.nvidia.com/cuda-12-8-0-download-archive
- CUDA 12.8 Linux install guide: https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/
- Kaolin install guide: https://kaolin.readthedocs.io/en/stable/notes/installation.html
- SAM 3D Objects checkpoints: https://huggingface.co/facebook/sam-3d-objects
