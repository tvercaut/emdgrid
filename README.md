# emdgrid
Fast and exact optimal transport solvers for discrete histograms on regular grids.

## Features
- **1D Optimal Transport**: Exact $O(N)$ solvers for 1D histograms under both **$L_1$** (Manhattan distance) and **squared Euclidean** ($W_2^2$) ground metrics (`emd_1d` / `emd_l1` and `emd_sqeuclidean_1d`).
- **Multidimensional EMD-L1**: Efficient exact tree-based network simplex solver (`emd_l1`) for 2D and 3D grid histograms based on Ling & Okada (2007).
- **Multidimensional dpartion MCF Solver**: Exact $N$-D optimal transport solver (`mcf_dpartion` / `dpartion`) using $(d+1)$-partite DAG layered graphs (Auricchio et al. 2018) supporting $L_1$ and squared Euclidean ground metrics via LEMON NetworkSimplex or CostScaling.
- **Bipartite Reference Solver**: Textbook transportation formulation (`emd_lemon`) over the occupied bins, solved with LEMON NetworkSimplex or CostScaling. Arcs are implicit and the $n \times m$ ground costs are computed on demand rather than stored, following POT's `EMD_wrap_lazy`. It ignores the grid's adjacency structure by design, which makes it the cross-check the structure-exploiting solvers are validated against, and it accepts arbitrary separable ground metrics.
- **Multidimensional Knothe-Rosenblatt Heuristic**: Fast $N$-D heuristic solver (`knothe_rosenblatt`) supporting both $L_1$ and squared Euclidean ground metrics, custom dimension traversal permutations, and parallelization via OpenMP.
- **Selectable Computation Precision**: Every solver is templated on a `CompScalar` computation type (defaulting to `double`) that governs the internal arithmetic, the returned cost and the flows of the returned `SparseTransportPlan<CompScalar>`. Pass `--comp-scalar float` to `emdgrid_example` to compare a single-precision run against the double-precision reference.
- **Python Bindings**: Pybind11 Python bindings (`pyemdgrid`) with support for retrieving sparse transport plans (`scipy.sparse.coo_matrix`).

## Logging
Every solver reports its progress through [spdlog](https://github.com/gabime/spdlog) using the same set of stages, so runs can be compared across solvers:

```
[info] mcf_dpartion: starting (Dim=2, algo=NetworkSimplex, scale=1000000)
[info] mcf_dpartion: supply setup took 0.766 ms (bins=4, layered nodes=12)
[info] mcf_dpartion: graph construction took 0.031 ms (nodes=12, arcs=16)
[info] mcf_dpartion: LEMON solve took 0.031 ms
[info] mcf_dpartion: solver finished with status OPTIMAL
[info] mcf_dpartion: flow decomposition took 0.003 ms (entries=2)
[info] mcf_dpartion: done in 0.865 ms, cost 1
```

Each stage line reports the time since the previous stage; the final line reports the whole run. The status line carries the backend's own exit code (`OPTIMAL`, `MAX_ITER_REACHED`, `INFEASIBLE`, ...).

**A solver that delivered less than it promised logs its status at `warn` level** — an exhausted iteration cap, or a backend reporting anything but optimality — because such a run still returns a number that looks exactly like a converged one. The heuristics (`greedy_emd_l1_approx`, `knothe_rosenblatt`) stay at `info`: returning an upper bound is their contract, not a failure, and warning on every call would only teach you to ignore the channel.

Set the verbosity with spdlog in the usual way, for example `spdlog::set_level(spdlog::level::warn)` to keep only the status warnings, or `spdlog::level::off` to silence the library.

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
