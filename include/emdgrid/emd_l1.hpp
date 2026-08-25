#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "emdgrid/emd_1d.hpp"
#include "emdgrid/emd_l1_detail.hpp"
#include "emdgrid/emdgrid.hpp"
#include "emdgrid/greedy_emd_l1.hpp"

namespace emdgrid {

// ---------------------------------------------------------------------------
//  Public API: emd_l1
// ---------------------------------------------------------------------------

/// EMD-L1 for 1-D histograms — thin wrapper around emd_1d.
template <std::floating_point Scalar, std::floating_point CompScalar = double>
[[nodiscard]] CompScalar emd_l1(const GridDataView<1, Scalar>& h1,
                                const GridDataView<1, Scalar>& h2,
                                SparseTransportPlan* plan = nullptr) {
  return emd_1d(h1, h2, plan);
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
  detail::greedy_init<Dim, Scalar, CompScalar>(h1, h2, solver);

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
