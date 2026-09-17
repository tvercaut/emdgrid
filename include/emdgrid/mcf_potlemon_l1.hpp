#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "emdgrid/detail/potlemon/network_simplex_simple.h"
#include "emdgrid/detail/potlemon/sparse_bipartitegraph.h"
#include "emdgrid/emdgrid.hpp"
#include "emdgrid/grid_detail.hpp"
#include "emdgrid/log_detail.hpp"
#include "emdgrid/mcf_detail.hpp"
#include "emdgrid/utils.hpp"

namespace emdgrid {

namespace detail {

/// Human-readable name for a potlemon network-simplex exit status.
template <typename Simplex>
[[nodiscard]] std::string_view potlemon_status_name(
    typename Simplex::ProblemType status) {
  switch (status) {
    case Simplex::INFEASIBLE:
      return "INFEASIBLE";
    case Simplex::OPTIMAL:
      return "OPTIMAL";
    case Simplex::UNBOUNDED:
      return "UNBOUNDED";
    case Simplex::MAX_ITER_REACHED:
      return "MAX_ITER_REACHED";
  }
  return "UNKNOWN";
}

}  // namespace detail

/// EMD-L1 for multi-dimensional grid histograms solved via the potlemon
/// Network Simplex solver.
///
/// The underlying solver traces its lineage from:
///   - LEMON's network_simplex.h (Egervary Research Group, 2003–2010)
///   - Adapted by Nicolas Bonneel (2013–2018) for mass transport:
///     https://github.com/nbonneel/network_simplex
///   - Included in the POT library:
///     https://github.com/PythonOT/POT/blob/master/ot/lp/network_simplex_simple.h
///   - Further adapted for emdgrid (namespace potlemon, C++20, optional OpenMP)
///
/// Solves a min-cost flow on the grid graph (adjacent cells, unit costs).
/// This formulation is optimal for L1 EMD and uses O(Dim × n_cells) arcs.
/// When a transport plan is requested, arc flows are decomposed into a direct
/// (source_bin, target_bin, mass) coupling using the same path-tracing
/// strategy as mcf_lemon_l1.
///
/// @tparam Dim        Grid dimensionality (>= 1).
/// @tparam Scalar     Input histogram scalar type.
/// @tparam CompScalar Scalar type used for computation (default: double).
template <std::size_t Dim, std::floating_point Scalar,
          std::floating_point CompScalar = double>  // NOLINT(*)
  requires(Dim >= 1)  // NOLINT(whitespace/indent_namespace)
[[nodiscard]] CompScalar mcf_potlemon_l1(
    const GridDataView<Dim, Scalar>& h1, const GridDataView<Dim, Scalar>& h2,
    SparseTransportPlanPtr<CompScalar> plan = nullptr,
    CompScalar scale = static_cast<CompScalar>(1e6),
    CompScalar mass_tol = default_mass_tolerance<CompScalar>,
    uint64_t max_iter = 500000) {
  detail::SolverLog log(
      "mcf_potlemon_l1",
      fmt::format("Dim={}, scale={}, max_iter={}", Dim, scale, max_iter));

  detail::validate_unit_mass_pair(h1, h2, mass_tol);

  const auto& layout = h1.layout();
  const std::size_t n_nodes = layout.node_count();

  if (plan) {
    plan->source.clear();
    plan->target.clear();
    plan->flow.clear();
    detail::emit_self_mass(h1, h2, plan);
  }

  const std::vector<int64_t> node_supply =
      detail::quantize_net_supply(h1, h2, scale);

  bool any_nonzero = false;
  for (std::size_t i = 0; i < n_nodes; ++i) {
    if (node_supply[i] != 0) {
      any_nonzero = true;
      break;
    }
  }
  log.phase("supply setup", fmt::format("nodes={}", n_nodes));

  if (!any_nonzero) {
    spdlog::info(
        "mcf_potlemon_l1: histograms are identical after quantization, "
        "nothing to transport");
    log.finish(CompScalar{0});
    return static_cast<CompScalar>(0.0);
  }

  // Grid-adjacent arcs: forward and backward for each adjacent cell pair.
  using Digraph = potlemon::SparseDigraph;
  std::vector<std::pair<int, int>> edges;

  detail::for_each_grid_arc(layout, [&](std::size_t u, std::size_t v) {
    edges.emplace_back(static_cast<int>(u), static_cast<int>(v));
    edges.emplace_back(static_cast<int>(v), static_cast<int>(u));
  });

  const int64_t total_arcs = static_cast<int64_t>(edges.size());

  Digraph di(static_cast<int>(n_nodes));
  di.buildFromEdges(edges);

  using Simplex = potlemon::NetworkSimplexSimple<Digraph, int64_t, int64_t>;
  Simplex::SimplexOptions options(true);
  Simplex net(di, options, static_cast<int>(n_nodes), total_arcs, max_iter);
  net.supplyMap(node_supply);

  for (int64_t k = 0; k < total_arcs; ++k) {
    net.setCost(Digraph::arcFromId(k), 1);
  }

  log.phase("graph construction",
            fmt::format("nodes={}, arcs={}", n_nodes, total_arcs));

  const auto status = net.run();
  log.phase("network simplex solve");
  using Outcome = detail::SolverLog::Outcome;
  log.status(detail::potlemon_status_name<Simplex>(status),
             status == Simplex::OPTIMAL ? Outcome::Optimal
                                        : Outcome::Degraded);
  if (status != Simplex::OPTIMAL && status != Simplex::MAX_ITER_REACHED) {
    throw std::runtime_error(
        "potlemon network simplex solve failed, status=" +
        std::string(detail::potlemon_status_name<Simplex>(status)));
  }

  const int64_t raw_cost = net.totalCost();
  const CompScalar total_cost = static_cast<CompScalar>(raw_cost) / scale;

  if (!plan) {
    log.finish(total_cost);
    return total_cost;
  }

  // Decompose arc flows into (source_bin, target_bin, mass) pairs using the
  // same path-tracing strategy as mcf_lemon_l1: repeatedly trace from each
  // source along edges with remaining flow to a sink, record the bottleneck,
  // and subtract it from the path.
  std::vector<std::vector<detail::FlowEdge>> flow_adj(n_nodes);
  for (int64_t k = 0; k < total_arcs; ++k) {
    const Digraph::Arc a = Digraph::arcFromId(k);
    const int64_t f = net.flow(a);
    if (f > 0) {
      const auto u = static_cast<std::size_t>(di.source(a));
      const auto v = static_cast<uint32_t>(di.target(a));
      flow_adj[u].push_back({v, f});
    }
  }

  detail::decompose_flows(&flow_adj, node_supply, n_nodes, scale,
                          std::identity{}, plan);
  log.phase("flow decomposition",
            fmt::format("entries={}", plan->flow.size()));

  log.finish(total_cost);
  return total_cost;
}

}  // namespace emdgrid
