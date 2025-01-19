# GPT2-Quantization-HPC

## Overview
This project implements a post-training quantization (PTQ) framework for GPT-2 to reduce inference latency and memory requirements on high-performance computing (HPC) systems. It includes:
- **Post-Training Quantization**: Reducing model weights to `int8` precision with symmetric quantization.
- **Custom CUDA Kernel**: A tile-based int8 matrix multiplication kernel integrated with PyTorch for efficient computations.
- **Performance Insights**: Analysis of quantization's impact on memory usage and inference speed.

## Key Features
1. Post-Training Quantization for GPT-2 using `int8` weights and activations.
2. Custom CUDA kernel for `int8` GEMM operations.
3. Integration with PyTorch for seamless execution.
4. Engineering insights on building and integrating custom HPC kernels.

## Files and Structure
- `src/`: Contains source code.
  - `__init__.py`: Initialization for the module.
  - `int8_matmul_kernel.cu`: Custom CUDA kernel for int8 matrix multiplication.
  - `int8_matmul.cpp`: PyTorch extension for the custom kernel.
  - `setup.py`: Script to build and install the CUDA extension.
  - `modelq.py`: Quantized GPT-2 model.
  - `llm.py`: Language model utilities.
  - `test_int8_mm.py`: Script for testing the custom int8 kernel.
  - `docs/`: Documentation and references.
  - `article.pdf`: Paper detailing the project and results.
