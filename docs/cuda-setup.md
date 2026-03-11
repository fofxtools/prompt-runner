Install CUDA:

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit

# Must reboot
reboot
```

Verify CUDA installed:

```bash
nvcc --version
```

Add to `~/.bashrc`. CUDACXX is to avoid `ERROR: Failed building wheel` issues when trying to pip install with CUDA support:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Then source the new environment variables:

```bash
source ~/.bashrc
```

Set up prompt-runner and install packages:

```bash
mkdir ~/prompt-runner
cd ~/prompt-runner
python3 -m venv .venv

echo -e "ollama" >> requirements.txt

source .venv/bin/activate
pip install -r requirements.txt

# To install llama-cpp-python and stable-diffusion-cpp-python with CUDA support
# --no-binary forces source build
# --no-cache-dir prevents pip from reusing cached CPU wheel
CMAKE_ARGS="-DGGML_CUDA=ON" pip install llama-cpp-python --force-reinstall --no-binary llama-cpp-python --no-cache-dir
CMAKE_ARGS="-DSD_CUDA=ON" pip install stable-diffusion-cpp-python --force-reinstall --no-binary stable-diffusion-cpp-python --no-cache-dir
```

To check they compiled with CUDA:

```bash
ldd .venv/lib/python3.12/site-packages/llama_cpp/lib/libllama.so | grep cuda

ldd .venv/lib/python3.12/site-packages/stable_diffusion_cpp/lib/libstable-diffusion.so | grep cuda
```

Test if llama-cpp-python is using CUDA or CPU backend:

```bash
python3 - << 'EOF'
from llama_cpp import llama_cpp
print("CUDA supported:", llama_cpp.llama_supports_gpu_offload())
EOF
```

To check with unlimited GPU layers:

```bash
python3 - << 'EOF'
from llama_cpp import Llama
llm = Llama(
    model_path="/home/saqib/ai/llms/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
    n_gpu_layers=-1,
    verbose=True,
)
EOF
```

Test if stable-diffusion-cpp-python is using CUDA or CPU backend:

```bash
python3 - << 'EOF'
from stable_diffusion_cpp import StableDiffusion
sd = StableDiffusion(
    model_path="/home/saqib/ai/diffusion/checkpoints/v1-5-pruned-emaonly.safetensors",
    verbose=True,
)
EOF
```

To check GPU memory usage:

```bash
nvidia-smi --query-gpu=memory.used --format=noheader,nounits
```
