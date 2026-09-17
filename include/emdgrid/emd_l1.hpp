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
#include "emdgrid/log_detail.hpp"
#include "emdgrid/greedy_emd_l1.hpp"

namespace emdgrid {

// ---------------------------------------------------------------------------
//  Public API: emd_l1
// ---------------------------------------------------------------------------

/// EMD-L1 for 1-D histograms — thin wrapper around emd_1d.
template <std::floating_point Scalar, std::floating_point CompScalar = double>
[[nodiscard]] CompScalar emd_l1(
    const GridDataView<1, Scalar>& h1, const GridDataView<1, Scalar>& h2,
    SparseTransportPlanPtr<CompScalar> plan = nullptr) {
  return emd_1d<Scalar, CompScalar>(h1, h2, plan);
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
[[nodiscard]] CompScalar emd_l1(
    const GridDataView<Dim, Scalar>& h1, const GridDataView<Dim, Scalar>& h2,
    SparseTransportPlanPtr<CompScalar> plan = nullptr,
    int max_iter = 500000) {
  detail::SolverLog log("emd_l1",
                        fmt::format("Dim={}, max_iter={}", Dim, max_iter));

  if (h1.layout().shape() != h2.layout().shape()) {
    throw std::invalid_argument("histogram shapes do not match");
  }
  const auto& layout = h1.layout();
  const auto& shape = layout.shape();
  const std::size_t n_nodes = layout.node_count();
  const std::size_t n_edges = layout.edge_count();

  detail::LingOkadaSolver<CompScalar> solver(n_nodes, n_edges);
  detail::greedy_init<Dim, Scalar, CompScalar>(h1, h2, solver);
  log.phase("greedy initialisation",
            fmt::format("nodes={}, edges={}", n_nodes, n_edges));

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

  const CompScalar cost = solver.solve(root, max_iter);
  log.phase("network simplex", fmt::format("pivots={}", solver.pivots()));
  using Outcome = detail::SolverLog::Outcome;
  log.status(solver.converged() ? "OPTIMAL" : "MAX_ITER_REACHED",
             solver.converged() ? Outcome::Optimal : Outcome::Degraded);

  if (plan) {
    std::vector<CompScalar> h1_c(n_nodes);
    std::vector<CompScalar> h2_c(n_nodes);
    for (std::size_t i = 0; i < n_nodes; ++i) {
      h1_c[i] = static_cast<CompScalar>(h1.data()[i]);
      h2_c[i] = static_cast<CompScalar>(h2.data()[i]);
    }
    detail::extract_transport_plan<CompScalar>(
        n_nodes, h1_c, h2_c, solver.get_directed_edge_flows(), plan);
    log.phase("plan extraction", fmt::format("entries={}", plan->flow.size()));
  }

  log.finish(cost);
  return cost;
}

}  // namespace emdgrid
