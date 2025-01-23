# Hasegawa-Wakatani Prediction

## Procedure

### 1. Baseline Models
- **Single Variable FNOs:** Train and evaluate using three separate single-variable Fourier Neural Operators (FNOs).
- **Single Combined FNO:** Train a single FNO handling all variables collectively.

### 2. Multi-Variable FNO
- Implement and evaluate a multi-variable FNO. Reference: [IOP Science Paper](https://iopscience.iop.org/article/10.1088/1741-4326/ad313a/pdf).

### 3. Push-Forward Trick
- Current Implementation: [Google Colab Notebook](https://colab.research.google.com/drive/1BxM3sRk-1SS8E6h49D7s6-krdo0AJEok?authuser=2#scrollTo=7QHRHGo7CRoK&uniqifier=1).
- Alternative Implementations:
  - [Pretraining PDEs Repository](https://github.com/anthonyzhou-1/pretraining_pdes/tree/main)
  - [Arxiv Paper](https://arxiv.org/html/2406.08473v1#bib.bib17)
  - [PhiFlow Framework](https://tum-pbs.github.io/PhiFlow/), based on [this paper](https://openreview.net/pdf?id=vSix3HPYKSU).

### 4. Physics-Based Loss for Multi-Variable FNO
- Incorporate physics-informed loss functions for training the multi-variable FNO.
- References:
  - [Arxiv Paper](https://arxiv.org/pdf/2308.07051)
  - [NVIDIA Modulus Documentation](https://docs.nvidia.com/deeplearning/modulus/modulus-v2209/user_guide/neural_operators/darcy_pino.html)
  - [ACM Digital Library](https://dl.acm.org/doi/10.1145/3648506)
