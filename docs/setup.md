Add to `~/.bashrc`. CUDACXX is to avoid `ERROR: Failed building wheel` issues when trying to pip install with CUDA support:

```
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Set up evals and install packages:

```
mkdir ~/evals
cd ~/evals
python3 -m venv .venv

echo -e "ollama" >> requirements.txt

source .venv/bin/activate
pip install -r requirements.txt

# To install llama-cpp-python and stable-diffusion-cpp-python with CUDA support
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
CMAKE_ARGS="-DSD_CUDA=ON" pip install stable-diffusion-cpp-python
```

Then this to install CUDA:

```
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit

# Must reboot
reboot
```

Test if llama-cpp-python is using CUDA or CPU backend:

```
python3 - << 'EOF'
from llama_cpp import llama_cpp
print("CUDA supported:", llama_cpp.llama_supports_gpu_offload())
EOF
```

To check with unlimited GPU layers:

```
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

```
python3 - << 'EOF'
from stable_diffusion_cpp import StableDiffusion
sd = StableDiffusion(
    model_path="/home/saqib/ai/diffusion/checkpoints/v1-5-pruned-emaonly.safetensors",
    verbose=True,
)
EOF
```

To check GPU memory usage.

```
nvidia-smi --query-gpu=memory.used --format=noheader,nounits
```
