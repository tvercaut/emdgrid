# emdgrid
Fast and exact optimal transport solvers for discrete histograms on regular grids.

## Features
- **1D Optimal Transport**: Exact $O(N)$ solvers for 1D histograms under both **$L_1$** (Manhattan distance) and **squared Euclidean** ($W_2^2$) ground metrics (`emd_1d` / `emd_l1` and `emd_sqeuclidean_1d`).
- **Multidimensional EMD-L1**: Efficient exact tree-based network simplex solver (`emd_l1`) for 2D and 3D grid histograms based on Ling & Okada (2007).
- **Multidimensional dpartion MCF Solver**: Exact $N$-D optimal transport solver (`mcf_dpartion` / `dpartion`) using $(d+1)$-partite DAG layered graphs (Auricchio et al. 2018) supporting $L_1$ and squared Euclidean ground metrics via LEMON NetworkSimplex or CostScaling.
- **Multidimensional Knothe-Rosenblatt Heuristic**: Fast $N$-D heuristic solver (`knothe_rosenblatt`) supporting both $L_1$ and squared Euclidean ground metrics, custom dimension traversal permutations, and parallelization via OpenMP.
- **Python Bindings**: Pybind11 Python bindings (`pyemdgrid`) with support for retrieving sparse transport plans (`scipy.sparse.coo_matrix`).

## Installation
`pyemdgrid` can be installed directly with `pip`:
```bash
pip install git+https://github.com/tvercaut/emdgrid.git
```
Or from a local clone:
```bash
pip install .
```

## Examples & Notebooks
- **Google Colab Interactive Demo**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tvercaut/emdgrid/blob/main/example/colab_demo.ipynb)
  Demonstrates installing `pyemdgrid` in Google Colab via `pip`, computing EMD-L1 and Knothe-Rosenblatt transport plans for 10x10 grid histograms, and visualizing transport plan sparsity patterns.

## References
Ling H, Okada K. An efficient earth mover's distance algorithm for robust histogram comparison. IEEE transactions on pattern analysis and machine intelligence. 2007 May 31;29(5):840-53.
https://doi.org/10.1109/TPAMI.2007.1058

Auricchio G, Codotti M, Scarselli F, Lodi A, Yoshiyasu Y. Computing Kantorovich-Wasserstein Distances on d-dimensional histograms using (d+1)-partite graphs. In Advances in Neural Information Processing Systems (NeurIPS). 2018;31.
https://arxiv.org/abs/1805.07416

Auricchio, G., Lin, M., Zhou, L., Guo, Z. and Cai, Z., 2026, May. Scalable Knothe--Rosenblatt-like Heuristic Transportation Plans for Imaging Problems. In Proc. of the 25th International Conference on Autonomous Agents and Multiagent Systems (pp. 977-985).
https://doi.org/10.65109/CEMK9641
