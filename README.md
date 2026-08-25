# emdgrid
Fast and exact optimal transport solvers for discrete histograms on regular grids.

## Features
- **1D Optimal Transport**: Exact $O(N)$ solvers for 1D histograms under both **$L_1$** (Manhattan distance) and **squared Euclidean** ($W_2^2$) ground metrics (`emd_1d` / `emd_l1` and `emd_sqeuclidean_1d`).
- **Multidimensional EMD-L1**: Efficient exact tree-based network simplex solver (`emd_l1`) for 2D and 3D grid histograms based on Ling & Okada (2007).
- **Python Bindings**: Pybind11 Python bindings (`pyemdgrid`) with support for retrieving sparse transport plans (`scipy.sparse.coo_matrix`).

## References
Ling H, Okada K. An efficient earth mover's distance algorithm for robust histogram comparison. IEEE transactions on pattern analysis and machine intelligence. 2007 May 31;29(5):840-53.
https://doi.org/10.1109/TPAMI.2007.1058
