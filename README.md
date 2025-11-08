# Official Code for "Beyond Parallelism: Synergistic Computational Graph Effects in Multi-Head Attention"

**Beyond Parallelism: Synergistic Computational Graph Effects in Multi-Head Attention**

**Author:** Haitz Sáez de Ocáriz Borde  
**Workshop:** NeurIPS 2025 Workshop on Symmetry and Geometry in Neural Representations, Proceedings Track 1–16, 2025 

## 📖 Abstract

Multi-head attention powers Transformer networks, the primary deep learning architecture behind the success of large language models~(LLMs). Yet, the theoretical advantages of multi-head versus single-head attention, beyond mere parallel processing, remain underexplored. In this paper, we reframe multi-head attention as a system of potentially synergistic computational graphs, where each head functions as a feedforward directed acyclic graph (DAG) with a common sink state. We provide intuition and preliminary theoretical analysis of mixing time and minimax fidelity in this framework. Our results show that multi-head attention can synergistically enhance information propagation, yielding faster mixing times and minimax fidelity amplification under specific head-diversity conditions. Finally, we train single-head and multi-head Transformers, each with the same total number of parameters, on sequence manipulation tasks and empirically verify the predicted effects.

**Keywords:** Attention, Transformer, Graph Theory, Directed Acyclic Graph, Computational Graph, Markov Chain, Mixing Time, Minimax Fidelity, Signal Propagation

## 🎯 Research Motivation

### Core Research Questions

• **How do multi-head attention mechanisms provide computational advantages beyond parallel processing?**  
Traditional understanding focuses on parallelization, but we investigate whether multiple heads create synergistic effects that fundamentally improve information propagation.

• **Can we model attention heads as computational graphs with analyzable diffusion properties?**  
We reframe each attention head as a directed acyclic graph (DAG) with specific mixing time and fidelity characteristics, enabling theoretical analysis of signal propagation.

• **What conditions lead to synergistic amplification in multi-head systems?**  
We identify head-diversity conditions where combining multiple attention heads yields improvements in information flow and signal fidelity.

• **Does mixing time improve when combining multiple parallel feedforward computational graphs?**  
We theoretically show that multi-head attention can achieve faster mixing times than individual heads under optimal weighting conditions.

• **Do theoretical predictions about mixing times and fidelity match empirical performance?**  
We validate our theoretical framework through controlled experiments on sequence manipulation tasks (copy and cycle) with parameter-matched single and multi-head Transformers.

### Theoretical Framework

**Multi-Head Attention as Graph Systems:** We model each attention head as a feedforward DAG where:
- Nodes represent tokens in the input sequence
- Edges represent attention weights between tokens
- The sink state (τ) aggregates information from all source nodes
- Signal propagation follows diffusion dynamics on the graph

**Key Concepts:**
- **Mixing Time:** How quickly information propagates from source nodes to the sink
- **Minimax Fidelity:** The worst-case signal preservation across all source nodes
- **Head Diversity:** Structural differences between attention graphs that enable synergistic effects
- **Synergistic Amplification:** Performance improvements that exceed simple averaging of individual heads

## 📁 Project Structure

```
beyondparallelism/
├── example2.py              # Figure 3 generation: Signal diffusion visualization
├── main.py                  # Run main experiments in Section 4 of the paper
├── transformer.py           # Transformer implementation
├── train.py                 # Training pipeline
├── sequence_datasets.py     # Sequence manipulation task datasets
├── test_amplification.py    # Tables 5-6: Individual vs combined fidelity comparison
├── proxies.py               # Proxy metrics for attention analysis based on Appendix E
└── checkpoints/             # Trained model checkpoints
```

### Key Files Description

**`example2.py`:** Generates Figure 3 plots showing diffusion of signal from nodes u, v, w to the sink τ under single-head and multi-head diffusion kernels. Solid lines show signal arrival percentages over diffusion steps, while the dashed line φ_min indicates the cumulative fidelity.

**`transformer.py`:** Core Transformer implementation, enabling extraction of attention graphs.

**`train.py`:** Training pipeline for controlled experiments comparing single-head and multi-head Transformers with matched parameter counts on sequence manipulation tasks.

**`test_amplification.py`:** Empirical validation of synergistic amplification effects predicted by the theoretical framework. Generates Tables 5 and 6 comparing best individual vs combined multi-head fidelity for copy and cycle sequences.

**`sequence_datasets.py`:** Generates synthetic datasets for sequence manipulation tasks (copy and cycle) used in empirical validation.

**`proxies.py`:** Implements algorithms for computing mixing time and minimax fidelity proxies from trained attention matrices as described in Appendix E.

## 🚀 Quick Start

### Dependencies

Create a conda environment for reproducible results:

```bash
# Create environment
conda create -n bparallelism python=3.9
conda activate bparallelism
```

```bash
# Core dependencies (required)
pip install torch==2.8.0
pip install numpy==2.0.2
pip install pandas==2.3.3
pip install matplotlib==3.9.4
```

### Reproduce Figure 3: Signal Diffusion Analysis

Figure 3 for Example 2 in the paper demonstrating synergistic effects in multi-head attention can be reproduced using:

```bash
python example2.py
```

### Run Full Experimental Suite

Run main experiments in Section 4:

```bash
python main.py
```

### Run Full Experimental Suite

Generate Tables 5-6: Individual vs combined multi-head fidelity comparison:

```bash
python test_amplification.py
```

**Disclaimer:**  
Results may vary slightly depending on hardware (GPU vs CPU), floating-point precision, and random seeds.  
Exact reproduction of the published paper numbers may not be possible on different machines, but the qualitative trends (combined > individual fidelity) should remain consistent.  
Original experiments were run on an H100 GPU with float32 precision.

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{
borde2025beyond,
title={Beyond Parallelism: Synergistic Computational Graph Effects in Multi-Head Attention},
author={Haitz S{\'a}ez de Oc{\'a}riz Borde},
booktitle={NeurIPS 2025 Workshop on Symmetry and Geometry in Neural Representations},
year={2025},
url={https://openreview.net/forum?id=NFvdUjBGNb}
}
```


