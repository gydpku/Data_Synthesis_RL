#!/bin/bash

# Create and activate a virtual environment
#conda create --name myenv python=3.9
#pip install --upgrade pip

# Install PyTorch (optional, vLLM can install the correct version)


# Install vLLM
pip install vllm==0.6.3  # Change version if needed
pip install ray

pip install tensordict

# Install verl
pip install omegaconf
pip install -e .
pip install -r requirements.txt

# Install FlashAttention 2

# Install quality-of-life tools
pip install wandb IPython matplotlib
pip install openai anthropic tree_sitter
pip install tenacity==8.2.2 pydantic==1.10.7 rank-bm25==0.2.2
pip install -U "ray[default]"
pip install "pydantic>=2"
pip install huggingface_hub
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn==2.7.3
