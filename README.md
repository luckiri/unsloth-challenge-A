# Unsloth Challenge A - NF4 to Triton Conversion

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/unsloth-challenge-A/blob/main/notebooks/Unsloth_Challenge_A.ipynb)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/yourusername/unsloth-challenge-a)

## Implementation
- Single Triton kernel with PTX assembly (`extract.u32` + direct codebook access)
- Cache eviction policies (`evict_first`/`evict_last`)
- Tesla T4-optimized block size (1024 threads)
- BF16/FP16 support

## Key Optimizations
1. **4.2x faster** than baseline (5.38s → 1.28s)
2. **Zero intermediate buffers**
3. `torch.compile` compatible

## Setup
```bash
git clone https://github.com/yourusername/unsloth-challenge-A
pip install triton torch transformers
