#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "emdgrid/emdgrid.hpp"

namespace emdgrid {

/// Sparse transport plan represented in Coordinate (COO) format.
struct SparseTransportPlan {
  std::vector<uint32_t> source;
  std::vector<uint32_t> target;
  std::vector<double> flow;
};

}  // namespace emdgrid

#include "emdgrid/emd_l1_detail.hpp"

namespace emdgrid {

// ---------------------------------------------------------------------------
//  Public API: emd_l1
// ---------------------------------------------------------------------------

/// EMD-L1 for 1-D histograms — exact O(N) prefix-sum formula.
///
/// The Earth Mover's Distance under the L1 ground metric for 1-D discrete
/// histograms equals the sum of absolute values of the cumulative-sum
/// differences: EMD = Σ_k |Σ_{i≤k} (H1[i] − H2[i])|.
///
/// @tparam Scalar     Input histogram scalar type.
/// @tparam CompScalar Scalar type used for computation (default: double).
template <std::floating_point Scalar, std::floating_point CompScalar = double>
[[nodiscard]] CompScalar emd_l1(const GridDataView<1, Scalar>& h1,
                                const GridDataView<1, Scalar>& h2,
                                SparseTransportPlan* plan = nullptr) {
  if (h1.layout().shape() != h2.layout().shape()) {
    throw std::invalid_argument("histogram shapes do not match");
  }
  const std::size_t n = h1.layout().node_count();
  CompScalar total{0};
  CompScalar cumsum{0};
  for (std::size_t i = 0; i < n; ++i) {
    cumsum += static_cast<CompScalar>(h1.data()[i]) -
              static_cast<CompScalar>(h2.data()[i]);
    using std::abs;
    total += abs(cumsum);
  }

  if (plan) {
    plan->source.clear();
    plan->target.clear();
    plan->flow.clear();

    std::vector<double> s(n);
    std::vector<double> d(n);
    for (std::size_t i = 0; i < n; ++i) {
      double h1_val = static_cast<double>(h1.data()[i]);
      double h2_val = static_cast<double>(h2.data()[i]);
      using std::min;
      double self_mass = min(h1_val, h2_val);
      if (self_mass > 0.0) {
        plan->source.push_back(static_cast<uint32_t>(i));
        plan->target.push_back(static_cast<uint32_t>(i));
        plan->flow.push_back(self_mass);
      }
      s[i] = h1_val - self_mass;
      d[i] = h2_val - self_mass;
    }

    std::size_t src_idx = 0;
    std::size_t tgt_idx = 0;
    while (src_idx < n && tgt_idx < n) {
      if (s[src_idx] <= 1e-12) {
        ++src_idx;
        continue;
      }
      if (d[tgt_idx] <= 1e-12) {
        ++tgt_idx;
        continue;
      }
      using std::min;
      double transfer = min(s[src_idx], d[tgt_idx]);
      plan->source.push_back(static_cast<uint32_t>(src_idx));
      plan->target.push_back(static_cast<uint32_t>(tgt_idx));
      plan->flow.push_back(transfer);

      s[src_idx] -= transfer;
      d[tgt_idx] -= transfer;
    }
  }

  return total;
}

/// EMD-L1 for multi-dimensional grid histograms using network simplex.
///
/// Implements the algorithm of Ling & Okada (PAMI 2007):
///   H. Ling and K. Okada,
///   "An Efficient Earth Mover's Distance Algorithm for Robust Histogram
///   Comparison," IEEE TPAMI 29(5):840-853, 2007.
///
/// The ground distance is the L1 (Manhattan) distance between bin indices.
/// Both histograms must share the same grid layout and have equal total mass.
///
/// @tparam Dim        Grid dimensionality (>= 2).
/// @tparam Scalar     Input histogram scalar type.
/// @tparam CompScalar Scalar type used for computation (default: double).
template <std::size_t Dim, std::floating_point Scalar,
          std::floating_point CompScalar = double>  // NOLINT(*)
  requires(Dim >= 2)  // NOLINT(whitespace/indent_namespace)
[[nodiscard]] CompScalar emd_l1(const GridDataView<Dim, Scalar>& h1,
                                const GridDataView<Dim, Scalar>& h2,
                                SparseTransportPlan* plan = nullptr) {
  if (h1.layout().shape() != h2.layout().shape()) {
    throw std::invalid_argument("histogram shapes do not match");
  }
  const auto& layout = h1.layout();
  const auto& shape = layout.shape();
  const std::size_t n_nodes = layout.node_count();
  const std::size_t n_edges = layout.edge_count();

  detail::LingOkadaSolver solver(n_nodes, n_edges);

  // ---- Greedy initial Basic Feasible Solution ----------------------------
  // Set node demands d[i] = H1[i] - H2[i] and initialise working copy.
  std::vector<CompScalar> demand(n_nodes);
  for (std::size_t i = 0; i < n_nodes; ++i) {
    CompScalar d = static_cast<CompScalar>(h1.data()[i]) -
                   static_cast<CompScalar>(h2.data()[i]);
    demand[i] = d;
    solver.node(static_cast<std::ptrdiff_t>(i)).d =
        static_cast<double>(d);
  }

  // Strides: stride[a] = product of shape[a+1..Dim-1]
  std::array<std::ptrdiff_t, Dim> stride{};
  stride[Dim - 1] = 1;
  for (std::size_t a = Dim - 1; a-- > 0;) {
    stride[a] =
        stride[a + 1] * static_cast<std::ptrdiff_t>(shape[a + 1]);
  }

  // Cache coordinates during initial sweep to avoid calling
  // layout.coordinates() repeatedly.
  std::vector<typename GridLayout<Dim>::Coordinates> coords(n_nodes);
  for (std::size_t i = 0; i < n_nodes; ++i) {
    coords[i] = layout.coordinates(static_cast<std::ptrdiff_t>(i));
  }

  // prefix[a][k] = -(sum of demand[i] for all i with coord[a] < k),
  // maintained during the sweep.  Initialised from the original demands.
  std::vector<std::vector<CompScalar>> prefix(Dim);
  for (std::size_t a = 0; a < Dim; ++a) {
    prefix[a].assign(shape[a], CompScalar{0});
    // Compute per-slice sums first
    std::vector<CompScalar> slice(shape[a], CompScalar{0});
    for (std::size_t i = 0; i < n_nodes; ++i) {
      slice[static_cast<std::size_t>(coords[i][a])] += demand[i];
    }
    // Build prefix sums: prefix[a][k] = -(sum of slice[0..k-1])
    for (std::size_t k = 0; k + 1 < shape[a]; ++k) {
      prefix[a][k + 1] = prefix[a][k] - slice[k];
    }
  }

  // Sweep nodes 0 .. N-2 in lexicographic (flat) order
  for (std::ptrdiff_t i = 0;
       i < static_cast<std::ptrdiff_t>(n_nodes) - 1; ++i) {
    const auto& coord = coords[static_cast<std::size_t>(i)];
    const CompScalar d_i = demand[static_cast<std::size_t>(i)];

    // Choose the axis that minimises |d_i + prefix[a][coord[a]+1]|
    std::size_t best_axis = Dim;  // sentinel
    CompScalar best_cost = std::numeric_limits<CompScalar>::max();
    for (std::size_t a = 0; a < Dim; ++a) {
      const std::size_t k = static_cast<std::size_t>(coord[a]);
      if (k + 1 < shape[a]) {
        using std::abs;
        const CompScalar cost = abs(d_i + prefix[a][k + 1]);
        if (cost < best_cost) {
          best_cost = cost;
          best_axis = a;
        }
      }
    }

    const std::size_t ka = static_cast<std::size_t>(coord[best_axis]);
    const std::ptrdiff_t neighbour = i + stride[best_axis];

    // BV edge: i → neighbour
    const int bv_dir = (d_i > CompScalar{0}) ? 1 : 0;
    using std::abs;
    solver.register_bv(i, neighbour,
                       static_cast<double>(abs(d_i)), bv_dir);

    // Update working arrays
    demand[static_cast<std::size_t>(neighbour)] += d_i;
    prefix[best_axis][ka + 1] += d_i;

    // All other forward edges from i are NBV
    for (std::size_t a = 0; a < Dim; ++a) {
      if (a == best_axis) {
        continue;
      }
      const std::size_t k = static_cast<std::size_t>(coord[a]);
      if (k + 1 < shape[a]) {
        solver.register_nbv(i, i + stride[a]);
      }
    }
  }

  // Choose the grid centre as the spanning-tree root
  std::ptrdiff_t root = 0;
  {
    typename GridLayout<Dim>::Coordinates rc{};
    for (std::size_t a = 0; a < Dim; ++a) {
      rc[a] = static_cast<std::ptrdiff_t>(shape[a] / 2);
      if (shape[a] > 1 && rc[a] > 0) {
        --rc[a];
      }
    }
    root = layout.node(rc);
  }

  const double cost = solver.solve(root);

  if (plan) {
    std::vector<double> h1_d(n_nodes);
    std::vector<double> h2_d(n_nodes);
    for (std::size_t i = 0; i < n_nodes; ++i) {
      h1_d[i] = static_cast<double>(h1.data()[i]);
      h2_d[i] = static_cast<double>(h2.data()[i]);
    }
    detail::extract_transport_plan(n_nodes, h1_d, h2_d,
                                   solver.get_directed_edge_flows(), plan);
  }

  return static_cast<CompScalar>(cost);
}

}  // namespace emdgrid
