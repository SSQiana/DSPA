# Mitigating Dynamic Graph Distribution Shifts via Spectral Augmentation

This repository contains the official implementation of the paper:

> **Mitigating Dynamic Graph Distribution Shifts via Spectral Augmentation (DSPY)**

---
## 0. Abstract

Dynamic Graph Neural Networks (DyGNNs) face significant challenges under distribution shifts between training and test data. Existing OOD methods primarily focus on spatial invariances, overlooking the impact of distribution shifts on graph spectral properties.  In this work, we propose a spectral-based graph augmentation framework to enhance generalization in dynamic graphs. Specifically, we augment input graph spectra into a mixture of spectral shift components by maximizing spectral distance variance, and introduce an efficient approximation strategy to reduce the computational cost of eigen-decomposition. Building upon this augmentation strategy, we design a multi-encoder architecture, where each encoder specializes in a specific spectral shift component to generate referential representations. Furthermore, we introduce a novel learning objective that encourages the model to rely on robust spectral properties for better generalization.  Extensive experiments on both real-world and synthetic datasets demonstrate that DSPY significantly outperforms state-of-the-art methods on node classification and link prediction tasks under distribution shifts.

## 1. Requirements

Main package requirements:

- `CUDA == 10.1`
- `Python == 3.8.12`
- `PyTorch == 1.9.1`
- `PyTorch-Geometric == 2.0.1`

To install the complete requiring packages, use following command at the root directory of the repository:

```setup
pip install -r requirements.txt
```

## 2. Quick Start

### Training

To train the DSPY, run the following command in the directory `./scripts`:

```train
python main.py --mode=train --use_cfg=1 --dataset=<dataset_name>
```

## 3. Acknowledgements

We sincerely appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/haonan-yuan/EAGLE

https://github.com/yule-BUAA/DyGLib

https://github.com/wondergo2017/DIDA

https://github.com/Louise-LuLin/GCL-SPAN
