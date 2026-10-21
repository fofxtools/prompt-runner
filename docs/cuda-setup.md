# CUDA setup (Ubuntu)

Image generation needs the CUDA toolkit to build stable-diffusion.cpp. These steps follow NVIDIA's
[CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) (network repo install).

## NVIDIA driver and CUDA toolkit

`nvidia-smi` must already work. Since CUDA 13.4 the `cuda-toolkit` package no longer installs the driver, so install a compatible
driver first (see NVIDIA's Driver Installation Guide). Replace `ubuntu2404` and `x86_64` for your release and architecture:

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit
```

Reboot if the driver was installed or updated. Add the toolkit to `PATH` in `~/.bashrc` (`/usr/local/cuda` is a symlink to the
installed version), then reload it and check:

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

```bash
source ~/.bashrc
nvcc --version
```

If CMake picks the wrong CUDA version, set `CUDACXX=/usr/local/cuda/bin/nvcc` before the first `cmake` configure.

## Set up prompt-runner

```bash
git clone https://github.com/fofxtools/prompt-runner.git
cd prompt-runner
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## stable-diffusion.cpp (image generation)

Image generation runs through `sd-server` from [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), which
prompt-runner starts once per model. Build it once and set its absolute path as `sd_server` in `config/config.yaml`. Build steps
are from the project's [build docs](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/build.md);
`CMAKE_CUDA_ARCHITECTURES` is a standard CMake option (your GPU's compute capability, e.g. `120` for RTX 50xx, or `native` on
CMake 3.24+):

```bash
git clone --recursive https://github.com/leejet/stable-diffusion.cpp
cd stable-diffusion.cpp
cmake -S . -B build -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --config Release -j"$(nproc)"
build/bin/sd-server --version
```

To update, pull and rebuild:

```bash
git pull origin master
git submodule update --init --recursive
cmake --build build --config Release -j"$(nproc)"
```

New model families land upstream often, so a model that "isn't supported" usually needs a newer build (if the build misbehaves
after an update, `rm -rf build` and configure again).

`config/config.yaml` sets `--sage-attn` for every model. SageAttention needs an NVIDIA GPU with compute capability 8.0 or newer
(RTX 30xx and up) and CUDA 12.0+, with kernels compiled for that GPU ([docs](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/sage_attention.md)).
On older GPUs sd-server reports an error: remove `--sage-attn` from `image_server_args`.

To check the GPU is used, watch `nvidia-smi` during a run.

## llama-cpp-python (optional)

Not used by prompt-runner itself (the LLM runner uses a llama.cpp `llama serve`); only for local experiments. From the
[llama-cpp-python README](https://github.com/abetlen/llama-cpp-python), build with CUDA:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

(`--force-reinstall --no-cache-dir` make pip rebuild instead of reusing a cached CPU build.) Check it:

```bash
python3 -c "from llama_cpp import llama_cpp; print('CUDA supported:', llama_cpp.llama_supports_gpu_offload())"
```
