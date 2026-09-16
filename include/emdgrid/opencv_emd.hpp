#pragma once

// Adapted from OpenCV's Earth Mover's Distance implementation.
//
// Original file: opencv/modules/imgproc/src/emd_new.cpp (branch 5.x)
// Original copyright:
//   This file is part of OpenCV project.
//   It is subject to the license terms in the LICENSE file found in the
//   top-level directory of this distribution and at
//   http://opencv.org/license.html
//
// Partially based on Yossi Rubner's code:
//   emd.c, last update: 3/14/98
//   An implementation of the Earth Movers Distance.
//   Based on the solution for the Transportation problem as described in
//   "Introduction to Mathematical Programming" by F. S. Hillier and
//   G. J. Lieberman, McGraw-Hill, 1990.
//   Copyright (C) 1998 Yossi Rubner
//   Computer Science Department, Stanford University
//   E-Mail: rubner@cs.stanford.edu  URL: http://vision.stanford.edu/~rubner
//
// Adaptations for emdgrid:
//   - Removed all OpenCV dependencies (cv::Mat, InputArray, AutoBuffer,
//     utils::BufferArea, precomp.hpp).
//   - Replaced with standard C++23 (std::vector, std::span, concepts).
//   - Integrated with emdgrid GridDataView / SparseTransportPlan API.
//   - Supports L1, squared-Euclidean, and custom cost functions.
//   - Histograms are treated as sparse sets of (weight, coordinate) pairs;
//     zero-weight bins are skipped automatically.

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <vector>

#include "emdgrid/emdgrid.hpp"
#include "emdgrid/utils.hpp"

namespace emdgrid {

namespace detail {

// ============================================================================
// Primal transportation simplex (Rubner 1998 / OpenCV emd_new.cpp)
// ============================================================================

struct EmdNode1D {
  float val{0.F};
  EmdNode1D* next{nullptr};
};

// A basic variable: cell (i, j) with flow val.  Linked into per-row and
// per-column singly-linked lists for O(1) neighbourhood traversal.
struct EmdNode2D {
  float val{0.F};
  int i{0};
  int j{0};
  EmdNode2D* next[2]{nullptr, nullptr};  // [0]=next-in-row, [1]=next-in-col
};

struct EmdSolver {
  static constexpr int kMaxIter = 500;
  static constexpr float kInf = 1e20F;
  static constexpr float kEps = 1e-5F;

  int ssize{0};
  int dsize{0};

  // Flat cost matrix (ssize × dsize, row-major).
  std::vector<float> cost_buf;
  float& cost(int i, int j) {
    return cost_buf[(static_cast<std::size_t>(i) * dsize) + j];
  }
  [[nodiscard]] const float& cost(int i, int j) const {
    return cost_buf[(static_cast<std::size_t>(i) * dsize) + j];
  }

  // Flat bitmap: non-zero if (i,j) is a basic variable.
  std::vector<uint8_t> is_x_buf;
  uint8_t& is_x(int i, int j) {
    return is_x_buf[(static_cast<std::size_t>(i) * dsize) + j];
  }

  // Active basis nodes (at most ssize+dsize–1 entries).
  // x_nodes[end_x] is reserved as scratch for the entering variable.
  std::vector<EmdNode2D> x_nodes;
  EmdNode2D* end_x{nullptr};
  EmdNode2D* enter_x{nullptr};

  // Per-row / per-column linked lists into the basis.
  std::vector<EmdNode2D*> rows_x;
  std::vector<EmdNode2D*> cols_x;

  // Dual variables; kInf means "not yet assigned".
  std::vector<float> u;
  std::vector<float> v;

  // Original flat bin indices for SparseTransportPlan extraction (-1=dummy).
  std::vector<int> orig_idx1;
  std::vector<int> orig_idx2;

  // Scratch for the find_loop_dfs recursion.
  std::vector<EmdNode2D*> loop_buf;
  std::vector<uint8_t> is_used;

  // -----------------------------------------------------------------------
  // Solve the (possibly unbalanced) transportation problem.
  //   sig1, sig2  — supply / demand vectors
  //   c_flat      — ssize0 × dsize0 cost matrix (row-major)
  //   idx1, idx2  — original bin indices (-1 for a dummy row/col)
  // Returns the minimised weighted cost (EMD).
  // -----------------------------------------------------------------------
  float solve(std::span<const float> sig1, std::span<const float> sig2,
              std::span<const float> c_flat,
              std::span<const int> idx1, std::span<const int> idx2) {
    const int ssize0 = static_cast<int>(sig1.size());
    const int dsize0 = static_cast<int>(sig2.size());

    float s_sum = 0.F;
    float d_sum = 0.F;
    for (float w : sig1) {
      s_sum += w;
    }
    for (float w : sig2) {
      d_sum += w;
    }

    std::vector<float> s(sig1.begin(), sig1.end());
    std::vector<float> d(sig2.begin(), sig2.end());
    orig_idx1.assign(idx1.begin(), idx1.end());
    orig_idx2.assign(idx2.begin(), idx2.end());

    ssize = ssize0;
    dsize = dsize0;

    // Balance with a dummy row or column (zero cost).
    const bool add_row = (s_sum < d_sum - kEps);
    const bool add_col = (d_sum < s_sum - kEps);
    if (add_row) {
      s.push_back(d_sum - s_sum);
      orig_idx1.push_back(-1);
      ssize++;
    } else if (add_col) {
      d.push_back(s_sum - d_sum);
      orig_idx2.push_back(-1);
      dsize++;
    }

    // Build the cost matrix; dummy row/column entries remain 0.
    cost_buf.assign(static_cast<std::size_t>(ssize) * dsize, 0.F);
    for (int i = 0; i < ssize0; ++i) {
      for (int j = 0; j < dsize0; ++j) {
        cost(i, j) =
            c_flat[(static_cast<std::size_t>(i) * dsize0) + j];
      }
    }

    const int max_basic = ssize + dsize;
    x_nodes.resize(static_cast<std::size_t>(max_basic) + 1);
    end_x = x_nodes.data();

    is_x_buf.assign(static_cast<std::size_t>(ssize) * dsize, 0);
    rows_x.assign(ssize, nullptr);
    cols_x.assign(dsize, nullptr);

    u.assign(ssize, kInf);
    v.assign(dsize, kInf);

    loop_buf.resize(static_cast<std::size_t>(max_basic) + 1);
    is_used.assign(static_cast<std::size_t>(max_basic) + 1, 0);

    call_russel(s, d);
    run_simplex();
    return calc_flow();
  }

  // -----------------------------------------------------------------------
  // Russell's method: greedy initial basic feasible solution.
  // delta[i][j] = cost(i,j) - row_max[i] - col_max[j].
  // -----------------------------------------------------------------------
  void call_russel(std::vector<float>& s, std::vector<float>& d) {
    std::vector<float> row_max(ssize, -kInf);
    std::vector<float> col_max(dsize, -kInf);
    for (int i = 0; i < ssize; ++i) {
      for (int j = 0; j < dsize; ++j) {
        row_max[i] = std::max(row_max[i], cost(i, j));
        col_max[j] = std::max(col_max[j], cost(i, j));
      }
    }

    for (;;) {
      float min_d = kInf;
      int mi = -1;
      int mj = -1;
      for (int i = 0; i < ssize; ++i) {
        if (s[i] <= 0.F) {
          continue;
        }
        for (int j = 0; j < dsize; ++j) {
          if (d[j] <= 0.F) {
            continue;
          }
          const float delta = cost(i, j) - row_max[i] - col_max[j];
          if (delta < min_d) {
            min_d = delta;
            mi = i;
            mj = j;
          }
        }
      }
      if (mi < 0) {
        break;
      }

      const float flow = std::min(s[mi], d[mj]);
      s[mi] -= flow;
      d[mj] -= flow;

      EmdNode2D* node = end_x++;
      node->i = mi;
      node->j = mj;
      node->val = flow;
      is_x(mi, mj) = 1;
      node->next[0] = rows_x[mi];
      rows_x[mi] = node;
      node->next[1] = cols_x[mj];
      cols_x[mj] = node;
    }
  }

  // -----------------------------------------------------------------------
  // Main simplex loop.
  // -----------------------------------------------------------------------
  void run_simplex() {
    for (int iter = 0; iter < kMaxIter; ++iter) {
      find_basic_vars();
      if (!check_optimal()) {
        break;
      }
      check_new_solution();
    }
  }

  // BFS to compute dual variables u[i], v[j].
  // Maintains: cost(i,j) = u[i] + v[j] for every basic (i,j).
  // NOLINTNEXTLINE(readability-make-member-function-const)
  void find_basic_vars() {
    std::fill(u.begin(), u.end(), kInf);
    std::fill(v.begin(), v.end(), kInf);
    u[0] = 0.F;

    bool changed = true;
    while (changed) {
      changed = false;
      for (EmdNode2D* node = x_nodes.data(); node != end_x; ++node) {
        const int i = node->i;
        const int j = node->j;
        // NOLINTNEXTLINE(bugprone-branch-clone)
        if (u[i] < kInf && v[j] >= kInf) {
          v[j] = cost(i, j) - u[i];
          changed = true;
        } else if (v[j] < kInf && u[i] >= kInf) {
          u[i] = cost(i, j) - v[j];
          changed = true;
        }
      }
    }
  }

  // Find the most-negative reduced cost.  If found, sets enter_x.
  // Returns true iff a pivot is needed.
  bool check_optimal() {
    float min_rc = -kEps;
    enter_x = nullptr;

    for (int i = 0; i < ssize; ++i) {
      if (u[i] >= kInf) {
        continue;
      }
      for (int j = 0; j < dsize; ++j) {
        if (v[j] >= kInf) {
          continue;
        }
        if (is_x(i, j) != 0) {
          continue;
        }
        const float rc = cost(i, j) - u[i] - v[j];
        if (rc < min_rc) {
          min_rc = rc;
          enter_x = end_x;  // scratch slot after current basis
          enter_x->i = i;
          enter_x->j = j;
        }
      }
    }
    return enter_x != nullptr;
  }

  // Bring enter_x into the basis; remove the leaving arc.
  void check_new_solution() {
    loop_buf[0] = enter_x;
    std::fill(is_used.begin(), is_used.end(), 0);

    const int loop_size = find_loop_dfs(1);
    if (loop_size < 4) {
      return;
    }

    // Minimum flow on odd-indexed arcs (they will lose flow).
    float theta = kInf;
    for (int k = 1; k < loop_size; k += 2) {
      theta = std::min(theta, loop_buf[k]->val);
    }

    // Update flows.
    EmdNode2D* leaving = nullptr;
    for (int k = 0; k < loop_size; ++k) {
      EmdNode2D* node = loop_buf[k];
      if (k % 2 == 0) {
        node->val += theta;
      } else {
        node->val -= theta;
        if (leaving == nullptr && std::abs(node->val) < kEps) {
          leaving = node;
        }
      }
    }

    if (leaving == nullptr) {
      for (int k = 1; k < loop_size; k += 2) {
        if (loop_buf[k] != enter_x) {
          leaving = loop_buf[k];
          break;
        }
      }
    }
    if (leaving == nullptr) {
      return;
    }

    is_x(enter_x->i, enter_x->j) = 1;
    is_x(leaving->i, leaving->j) = 0;

    // Remove leaving from its row list.
    for (EmdNode2D** pp = &rows_x[leaving->i];
         (*pp) != nullptr; pp = &(*pp)->next[0]) {
      if (*pp == leaving) {
        *pp = leaving->next[0];
        break;
      }
    }
    // Remove leaving from its column list.
    for (EmdNode2D** pp = &cols_x[leaving->j];
         (*pp) != nullptr; pp = &(*pp)->next[1]) {
      if (*pp == leaving) {
        *pp = leaving->next[1];
        break;
      }
    }

    // Reuse the leaving node's slot for the entering variable.
    leaving->i = enter_x->i;
    leaving->j = enter_x->j;
    leaving->val = enter_x->val;
    leaving->next[0] = rows_x[leaving->i];
    rows_x[leaving->i] = leaving;
    leaving->next[1] = cols_x[leaving->j];
    cols_x[leaving->j] = leaving;

    enter_x->val = 0.F;
  }

  // -----------------------------------------------------------------------
  // DFS loop-finder for the transportation simplex.
  //
  // loop_buf[0] = enter_x (the entering nonbasic cell).
  // At odd depth  (1,3,...): look for a basic cell in the same ROW as prev.
  // At even depth (2,4,...): look for a basic cell in the same COL as prev.
  // Closing condition: at odd depth >= 3, a cell whose j == enter_x->j.
  //
  // Returns loop length (>= 4) on success, 0 on failure.
  // -----------------------------------------------------------------------
  int find_loop_dfs(int depth) {
    const bool by_row = (depth % 2 == 1);
    EmdNode2D* prev = loop_buf[depth - 1];
    EmdNode2D* cand = by_row ? rows_x[prev->i] : cols_x[prev->j];

    while (cand != nullptr) {
      const std::size_t idx =
          static_cast<std::size_t>(cand - x_nodes.data());

      if (is_used[idx] == 0) {
        if (by_row && depth >= 3 && cand->j == loop_buf[0]->j) {
          loop_buf[depth] = cand;
          return depth + 1;
        }

        if (depth < ssize + dsize) {
          is_used[idx] = 1;
          loop_buf[depth] = cand;
          const int result = find_loop_dfs(depth + 1);
          if (result > 0) {
            return result;
          }
          is_used[idx] = 0;
        }
      }

      cand = by_row ? cand->next[0] : cand->next[1];
    }
    return 0;
  }

  // Total weighted cost from the current basis. The accumulator is double
  // regardless of the caller's CompScalar: the basis values are float, and
  // summing them in double simply avoids losing digits in the reduction.
  [[nodiscard]] float calc_flow() const {
    double total = 0.0;
    for (const EmdNode2D* node = x_nodes.data(); node != end_x; ++node) {
      if (orig_idx1[node->i] >= 0 && orig_idx2[node->j] >= 0) {
        total += (static_cast<double>(node->val) *
                  static_cast<double>(cost(node->i, node->j)));
      }
    }
    return static_cast<float>(total);
  }

  // Fill a SparseTransportPlan from the current basis.
  //
  // The simplex itself runs in float (faithful port of OpenCV's emd_new.cpp);
  // CompScalar only controls the type the flows are reported in.
  template <std::floating_point CompScalar>
  void extract_plan(SparseTransportPlan<CompScalar>& plan) const {
    for (const EmdNode2D* node = x_nodes.data(); node != end_x; ++node) {
      if (node->val <= 0.F) {
        continue;
      }
      const int oi = orig_idx1[node->i];
      const int oj = orig_idx2[node->j];
      if (oi < 0 || oj < 0) {
        continue;
      }
      plan.source.push_back(static_cast<uint32_t>(oi));
      plan.target.push_back(static_cast<uint32_t>(oj));
      plan.flow.push_back(static_cast<CompScalar>(node->val));
    }
  }
};

}  // namespace detail

// ============================================================================
// Public API
// ============================================================================

/// Earth Mover's Distance via the primal transportation simplex
/// (Rubner 1998, adapted from OpenCV's emd_new.cpp).
///
/// Constructs the full pairwise cost matrix between all non-zero bins and
/// solves the resulting transportation problem with a primal simplex.
///
/// **Complexity warning**: time and memory are O(n²) in the number of
/// non-zero bins n (one entry per source-destination pair). Each simplex
/// iteration scans all ~n² non-basic arcs, and the number of iterations
/// can grow with n. In practice this limits practical use to histograms
/// with at most a few hundred non-zero bins (e.g. grids up to ~8×8×8).
/// For larger grids prefer the grid-aware solvers (emd_l1, mcf_dpartion,
/// mcf_potlemon), which exploit the regular structure and run in
/// sub-quadratic time.
///
/// **Why this solver cannot be adapted for grid-graph EMD-L1**: the
/// transportation simplex is a bipartite-only algorithm — its internals
/// (the ssize×dsize cost matrix, the per-row/column linked lists) are
/// hard-wired to a source-set versus sink-set structure. The efficient
/// grid-graph formulation for L1 is a general min-cost flow problem where
/// every bin can be a source or a sink and flow routes through intermediate
/// nodes along axis-aligned edges (O(n·d) arcs total). Feeding that graph
/// into a transportation simplex would require pre-computing the L1 cost
/// between every source–sink pair, collapsing back to an O(n²) dense
/// matrix and erasing all benefit of the sparse structure. General network
/// simplex solvers (LEMON, OR-Tools, POT) handle arbitrary graphs and are
/// therefore the right tool for grid-based EMD-L1.
///
/// @tparam Dim        Grid dimensionality (>= 1).
/// @tparam Scalar     Input histogram scalar type.
/// @tparam CostFn     Callable: (Coordinates, Coordinates) -> float.
/// @tparam CompScalar Scalar type used to report the cost and plan flows
///                    (default: double). The transportation simplex itself is
///                    a faithful port of OpenCV's single-precision code and
///                    always computes in float.
template <std::size_t Dim, std::floating_point Scalar, typename CostFn,
          std::floating_point CompScalar = double>
  requires(Dim >= 1 && detail::ValidCostFn<CostFn>)
[[nodiscard]] CompScalar opencv_emd(
    const GridDataView<Dim, Scalar>& h1, const GridDataView<Dim, Scalar>& h2,
    CostFn&& cost_fn, SparseTransportPlanPtr<CompScalar> plan = nullptr,
    CompScalar mass_tol = default_mass_tolerance<CompScalar>) {
  using Coords = GridLayout<Dim>::Coordinates;

  if (h1.layout().shape() != h2.layout().shape()) {
    throw std::invalid_argument("histogram shapes do not match");
  }

  const auto& layout = h1.layout();
  const std::size_t n_nodes = layout.node_count();

  CompScalar t1{0};
  CompScalar t2{0};
  for (std::size_t i = 0; i < n_nodes; ++i) {
    const CompScalar v1 = static_cast<CompScalar>(h1.data()[i]);
    const CompScalar v2 = static_cast<CompScalar>(h2.data()[i]);
    if (v1 < CompScalar{0} || v2 < CompScalar{0}) {
      throw std::invalid_argument("histograms must be nonnegative");
    }
    t1 += v1;
    t2 += v2;
  }
  if (std::abs(t1 - CompScalar{1}) > mass_tol ||
      std::abs(t2 - CompScalar{1}) > mass_tol) {
    throw std::invalid_argument("expected unit-mass histograms");
  }

  if (plan != nullptr) {
    plan->source.clear();
    plan->target.clear();
    plan->flow.clear();
  }

  // Build sparse signatures: skip zero-weight bins.
  std::vector<float> sig1;
  std::vector<float> sig2;
  std::vector<int> idx1;
  std::vector<int> idx2;
  std::vector<Coords> coords1;
  std::vector<Coords> coords2;

  for (std::size_t i = 0; i < n_nodes; ++i) {
    const float w = static_cast<float>(h1.data()[i]);
    if (w > 0.F) {
      sig1.push_back(w);
      idx1.push_back(static_cast<int>(i));
      coords1.push_back(layout.coordinates(static_cast<std::ptrdiff_t>(i)));
    }
  }
  for (std::size_t i = 0; i < n_nodes; ++i) {
    const float w = static_cast<float>(h2.data()[i]);
    if (w > 0.F) {
      sig2.push_back(w);
      idx2.push_back(static_cast<int>(i));
      coords2.push_back(layout.coordinates(static_cast<std::ptrdiff_t>(i)));
    }
  }

  if (sig1.empty() || sig2.empty()) {
    return static_cast<CompScalar>(0.0);
  }

  const int n = static_cast<int>(sig1.size());
  const int m = static_cast<int>(sig2.size());

  // Build the pairwise cost matrix (n × m).
  std::vector<float> cost_mat(static_cast<std::size_t>(n) * m);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      cost_mat[(static_cast<std::size_t>(i) * m) + j] =
          cost_fn(coords1[i], coords2[j]);
    }
  }

  detail::EmdSolver solver;
  const float raw = solver.solve(sig1, sig2, cost_mat, idx1, idx2);

  if (plan != nullptr) {
    solver.template extract_plan<CompScalar>(*plan);
  }

  return static_cast<CompScalar>(raw);
}

/// Overload with `GroundMetric` enum (L1 or SqEuclidean). See the primary
/// overload for complexity constraints.
template <std::size_t Dim, std::floating_point Scalar,
          std::floating_point CompScalar = double>
  requires(Dim >= 1)
[[nodiscard]] CompScalar opencv_emd(
    const GridDataView<Dim, Scalar>& h1, const GridDataView<Dim, Scalar>& h2,
    GroundMetric metric = GroundMetric::L1,
    SparseTransportPlanPtr<CompScalar> plan = nullptr,
    CompScalar mass_tol = default_mass_tolerance<CompScalar>) {
  using Coords = GridLayout<Dim>::Coordinates;
  using CostFn = std::function<float(const Coords&, const Coords&)>;

  if (metric == GroundMetric::L1) {
    return opencv_emd<Dim, Scalar, CostFn, CompScalar>(
        h1, h2,
        CostFn{[](const Coords& a, const Coords& b) -> float {
          float d = 0.F;
          for (std::size_t k = 0; k < Dim; ++k) {
            d += static_cast<float>(std::abs(
                static_cast<std::ptrdiff_t>(a[k]) -
                static_cast<std::ptrdiff_t>(b[k])));
          }
          return d;
        }},
        plan, mass_tol);
  }
  return opencv_emd<Dim, Scalar, CostFn, CompScalar>(
      h1, h2,
      CostFn{[](const Coords& a, const Coords& b) -> float {
        float d = 0.F;
        for (std::size_t k = 0; k < Dim; ++k) {
          const float diff = static_cast<float>(
              static_cast<std::ptrdiff_t>(a[k]) -
              static_cast<std::ptrdiff_t>(b[k]));
          d += diff * diff;
        }
        return d;
      }},
      plan, mass_tol);
}

}  // namespace emdgrid
