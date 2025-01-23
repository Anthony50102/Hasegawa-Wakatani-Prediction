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


## Misc

### Key Directions for Training the Surrogate Model

1. **Loss Function Design**:
   - Investigate **multi-objective loss functions**. Combine the current MSE loss for the image output with an additional loss term that evaluates the discrepancy in derived quantities. For example:
     ```
     Loss = λ₁ * MSE_image + λ₂ * MSE_derived
     ```
     This allows the model to learn to predict the next "image" while also improving predictions of the derived quantities.

2. **Differentiability of Derived Quantities**:
   - Ensure that the calculation of derived quantities is differentiable so that gradients can propagate back through this part of the pipeline during training. If it's not differentiable, approximate it with a differentiable proxy or explore gradient-free methods for that component.

3. **Autoregressive Consistency**:
   - Validate that the loss on derived quantities aligns with the autoregressive use case. Perform short rollouts during training to ensure that predictions remain consistent over time when used autoregressively.

4. **Derived Quantity-Based Training Objectives**:
   - Explore training with **custom loss functions** that measure discrepancies in statistical properties of the derived quantities (e.g., mean, variance, higher-order moments) across an autoregressive sequence.

5. **Curriculum Learning**:
   - Gradually shift the focus of training from purely image-based MSE to derived-quantity-based objectives. For example, start with a high weight on `MSE_image` and then transition to giving more importance to the derived quantity loss.

6. **Domain-Specific Regularization**:
   - Add regularization terms that enforce physical constraints or statistical properties specific to the domain. For example, penalize invalid or unphysical predictions of the derived quantities to improve alignment with the problem’s requirements.
