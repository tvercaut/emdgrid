#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "emdgrid/emd_l1_detail.hpp"
#include "emdgrid/emdgrid.hpp"

namespace emdgrid {

// ---------------------------------------------------------------------------
// Public API: greedy_emd_l1_approx
// ---------------------------------------------------------------------------

/// EMD-L1 greedy approximation for 1-D histograms.
///
/// For 1-D histograms, the prefix-sum solution is exact.
///
/// @tparam Scalar Input histogram scalar type.
/// @tparam CompScalar Scalar type used for computation (default: double).
template <std::floating_point Scalar, std::floating_point CompScalar = double>
[[nodiscard]] CompScalar greedy_emd_l1_approx(
    const GridDataView<1, Scalar>& h1,
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
    SparseTransportPlan* plan = nullptr) {
  if (h1.layout().shape() != h2.layout().shape()) {
    throw std::invalid_argument("histogram shapes do not match");
  }
  const auto& layout = h1.layout();
  const std::size_t n_nodes = layout.node_count();
  const std::size_t n_edges = layout.edge_count();

  detail::LingOkadaSolver solver(n_nodes, n_edges);
  detail::greedy_init<Dim, Scalar, CompScalar>(h1, h2, solver);

  const double cost = solver.total_flow();

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
