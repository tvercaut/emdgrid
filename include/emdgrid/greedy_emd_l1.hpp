#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <spdlog/spdlog.h>  // NOLINT(build/include_order)

#include "emdgrid/emd_1d.hpp"
#include "emdgrid/emd_l1_detail.hpp"
#include "emdgrid/emdgrid.hpp"

namespace emdgrid {

// ---------------------------------------------------------------------------
// Public API: greedy_emd_l1_approx
// ---------------------------------------------------------------------------

/// EMD-L1 greedy approximation for 1-D histograms — thin wrapper around emd_1d.
template <std::floating_point Scalar, std::floating_point CompScalar = double>
[[nodiscard]] CompScalar greedy_emd_l1_approx(
    const GridDataView<1, Scalar>& h1,
    const GridDataView<1, Scalar>& h2,
    SparseTransportPlanPtr<CompScalar> plan = nullptr) {
  return emd_1d<Scalar, CompScalar>(h1, h2, plan);
}

/// EMD-L1 greedy approximation for multi-dimensional grid histograms.
///
/// Computes an approximate transport cost and optional transport plan using
/// Ling & Okada's greedy basic feasible solution heuristic without network
/// simplex pivoting.
///
/// @tparam Dim Grid dimensionality (>= 2).
/// @tparam Scalar Input histogram scalar type.
/// @tparam CompScalar Scalar type used for computation (default: double).
template <std::size_t Dim, std::floating_point Scalar,
          std::floating_point CompScalar = double>  // NOLINT(*)
  requires(Dim >= 2)  // NOLINT(whitespace/indent_namespace)
[[nodiscard]] CompScalar greedy_emd_l1_approx(
    const GridDataView<Dim, Scalar>& h1,
    const GridDataView<Dim, Scalar>& h2,
    SparseTransportPlanPtr<CompScalar> plan = nullptr) {
  if (h1.layout().shape() != h2.layout().shape()) {
    throw std::invalid_argument("histogram shapes do not match");
  }
  const auto& layout = h1.layout();
  const std::size_t n_nodes = layout.node_count();
  const std::size_t n_edges = layout.edge_count();

  spdlog::info("Computing greedy EMD-L1 upper bound approximation...");

  detail::LingOkadaSolver<CompScalar> solver(n_nodes, n_edges);
  detail::greedy_init<Dim, Scalar, CompScalar>(h1, h2, solver);

  SparseTransportPlan<CompScalar> local_plan;
  SparseTransportPlan<CompScalar>* target_plan = plan ? plan : &local_plan;

  std::vector<CompScalar> h1_c(n_nodes);
  std::vector<CompScalar> h2_c(n_nodes);
  for (std::size_t i = 0; i < n_nodes; ++i) {
    h1_c[i] = static_cast<CompScalar>(h1.data()[i]);
    h2_c[i] = static_cast<CompScalar>(h2.data()[i]);
  }
  detail::extract_transport_plan<CompScalar>(
      n_nodes, h1_c, h2_c, solver.get_directed_edge_flows(), target_plan);

  CompScalar cost{0};
  for (std::size_t k = 0; k < target_plan->flow.size(); ++k) {
    const auto c_src =
        layout.coordinates(static_cast<std::ptrdiff_t>(target_plan->source[k]));
    const auto c_tgt =
        layout.coordinates(static_cast<std::ptrdiff_t>(target_plan->target[k]));
    CompScalar l1_dist{0};
    for (std::size_t a = 0; a < Dim; ++a) {
      l1_dist += static_cast<CompScalar>(std::abs(c_src[a] - c_tgt[a]));
    }
    cost += target_plan->flow[k] * l1_dist;
  }

  return cost;
}

}  // namespace emdgrid
