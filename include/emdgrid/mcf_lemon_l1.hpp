#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#pragma push_macro("MAX")
#pragma push_macro("MIN")
#undef MAX
#undef MIN

#include <lemon/cost_scaling.h>     // NOLINT(build/include_order)
#include <lemon/network_simplex.h>  // NOLINT(build/include_order)
#include <lemon/smart_graph.h>      // NOLINT(build/include_order)

#pragma pop_macro("MIN")
#pragma pop_macro("MAX")


#include "emdgrid/emdgrid.hpp"
#include "emdgrid/grid_detail.hpp"
#include "emdgrid/log_detail.hpp"
#include "emdgrid/mcf_detail.hpp"
#include "emdgrid/utils.hpp"

namespace emdgrid {

/// Algorithm variant for LEMON Min-Cost Flow solver.
enum class McfLemonAlgorithm : std::uint8_t { NetworkSimplex, CostScaling };

namespace detail {

/// A cost functor usable by the LEMON-backed solvers.
///
/// Rules out the two overload-selector enums so that the cost-functor
/// overload of a solver never competes with its `GroundMetric` overload nor
/// with the one that only picks a `McfLemonAlgorithm`.
template <typename T>
concept ValidLemonCostFn =
    ValidCostFn<T> &&                                        // NOLINT(*)
    !std::is_same_v<std::decay_t<T>, McfLemonAlgorithm>;     // NOLINT(*)

template <typename T>
concept HasExtractSelfMassMember = requires {
  { std::decay_t<T>::extract_self_mass } -> std::convertible_to<bool>;
};

template <typename T>
constexpr bool should_extract_self_mass_v = []() {
  using CleanT = std::decay_t<T>;
  if constexpr (HasExtractSelfMassMember<CleanT>) {
    return CleanT::extract_self_mass;
  } else {
    return false;
  }
}();

/// Human-readable name for a LEMON min-cost-flow exit status.
///
/// NetworkSimplex and CostScaling each declare their own ProblemType, with the
/// same three values, so this is templated on the solver rather than the enum.
template <typename Solver>
[[nodiscard]] std::string_view lemon_status_name(
    typename Solver::ProblemType status) {
  switch (status) {
    case Solver::INFEASIBLE:
      return "INFEASIBLE";
    case Solver::OPTIMAL:
      return "OPTIMAL";
    case Solver::UNBOUNDED:
      return "UNBOUNDED";
  }
  return "UNKNOWN";
}

/// Shared helper to solve LEMON Min-Cost Flow and collect flow edges.
///
/// Reports the backend's exit status through `log` before throwing on
/// anything but an optimum, so a failed solve is visible in the log and not
/// only in the exception.
template <typename Solver, typename Graph>
int64_t run_lemon_mcf(
    Graph& graph,
    typename Graph::template ArcMap<int64_t>& capacity,
    typename Graph::template ArcMap<int64_t>& cost,
    typename Graph::template NodeMap<int64_t>& supply,
    SolverLog* log,
    // NOLINTNEXTLINE(readability-non-const-parameter)
    std::vector<std::vector<FlowEdge>>* flow_adj = nullptr) {
  Solver mcf(graph);
  mcf.upperMap(capacity).costMap(cost).supplyMap(supply);
  const typename Solver::ProblemType status = mcf.run();
  if (log) {
    log->phase("LEMON solve");
    log->status(lemon_status_name<Solver>(status),
                 status == Solver::OPTIMAL ? SolverLog::Outcome::Optimal
                                           : SolverLog::Outcome::Degraded);
  }
  if (status != Solver::OPTIMAL) {
    throw std::runtime_error(
        "min-cost flow solve failed, status=" +
        std::string(lemon_status_name<Solver>(status)));
  }
  if (flow_adj) {
    for (typename Graph::ArcIt a(graph); a != lemon::INVALID; ++a) {
      const int64_t f = mcf.flow(a);
      if (f > 0) {
        const auto u = static_cast<std::size_t>(Graph::id(graph.source(a)));
        const auto v = static_cast<uint32_t>(Graph::id(graph.target(a)));
        (*flow_adj)[u].push_back({v, f});
      }
    }
  }
  return mcf.totalCost();
}

}  // namespace detail

/// EMD-L1 for multi-dimensional grid histograms solved via Min-Cost Flow
/// using LEMON (NetworkSimplex or CostScaling).
///
/// Converts the grid histogram distance into a min-cost flow problem on
/// a 2d-connected spatial grid graph with unit edge costs and quantized
/// node supplies.
///
/// @tparam Dim        Grid dimensionality (>= 1).
/// @tparam Scalar     Input histogram scalar type.
/// @tparam CompScalar Scalar type used for computation (default: double).
template <std::size_t Dim, std::floating_point Scalar,
          std::floating_point CompScalar = double>  // NOLINT(*)
  requires(Dim >= 1)  // NOLINT(whitespace/indent_namespace)
[[nodiscard]] CompScalar mcf_lemon_l1(
    const GridDataView<Dim, Scalar>& h1, const GridDataView<Dim, Scalar>& h2,
    McfLemonAlgorithm algo = McfLemonAlgorithm::NetworkSimplex,
    SparseTransportPlanPtr<CompScalar> plan = nullptr,
    CompScalar scale = static_cast<CompScalar>(1e6),
    CompScalar mass_tol = default_mass_tolerance<CompScalar>) {
  detail::SolverLog log(
      "mcf_lemon_l1",
      fmt::format("Dim={}, algo={}, scale={}", Dim,
                  algo == McfLemonAlgorithm::NetworkSimplex ? "NetworkSimplex"
                                                            : "CostScaling",
                  scale));

  detail::validate_unit_mass_pair(h1, h2, mass_tol);

  const auto& layout = h1.layout();
  const std::size_t n_nodes = layout.node_count();

  const std::vector<int64_t> supply =
      detail::quantize_net_supply(h1, h2, scale);
  const int64_t cap_val = detail::total_positive_supply(supply);
  log.phase("supply setup", fmt::format("nodes={}", n_nodes));

  using Graph = lemon::SmartDigraph;
  using Node = Graph::Node;
  using Arc = Graph::Arc;

  Graph graph;
  graph.reserveNode(static_cast<int>(n_nodes));

  std::vector<Node> nodes;
  nodes.reserve(n_nodes);
  for (std::size_t i = 0; i < n_nodes; ++i) {
    nodes.push_back(graph.addNode());
  }

  Graph::ArcMap<int64_t> capacity(graph);
  Graph::ArcMap<int64_t> cost(graph);
  Graph::NodeMap<int64_t> supply_map(graph);

  detail::for_each_grid_arc(layout, [&](std::size_t u, std::size_t v) {
    const Arc a1 = graph.addArc(nodes[u], nodes[v]);
    capacity[a1] = cap_val;
    cost[a1] = 1;

    const Arc a2 = graph.addArc(nodes[v], nodes[u]);
    capacity[a2] = cap_val;
    cost[a2] = 1;
  });

  for (std::size_t i = 0; i < n_nodes; ++i) {
    supply_map[nodes[i]] = supply[i];
  }

  log.phase("graph construction",
            fmt::format("nodes={}, arcs={}", n_nodes, graph.arcNum()));

  int64_t raw_optimal_cost = 0;
  std::vector<std::vector<detail::FlowEdge>> flow_adj(n_nodes);
  auto* flow_adj_ptr = plan ? &flow_adj : nullptr;

  if (algo == McfLemonAlgorithm::NetworkSimplex) {
    using Solver = lemon::NetworkSimplex<Graph, int64_t, int64_t>;
    raw_optimal_cost = detail::run_lemon_mcf<Solver>(
        graph, capacity, cost, supply_map, &log, flow_adj_ptr);
  } else {
    using Solver = lemon::CostScaling<Graph, int64_t, int64_t>;
    raw_optimal_cost = detail::run_lemon_mcf<Solver>(
        graph, capacity, cost, supply_map, &log, flow_adj_ptr);
  }

  const CompScalar total_cost =
      static_cast<CompScalar>(raw_optimal_cost) / scale;

  if (plan) {
    plan->source.clear();
    plan->target.clear();
    plan->flow.clear();

    detail::emit_self_mass(h1, h2, plan);

    detail::decompose_flows(&flow_adj, supply, n_nodes, scale,
                            std::identity{}, plan);
    log.phase("flow decomposition",
              fmt::format("entries={}", plan->flow.size()));
  }

  log.finish(total_cost);
  return total_cost;
}

/// N-D d-partite Min-Cost Flow solver for grid histograms (Auricchio et al.
/// 2018).
///
/// Embeds the grid transport problem with a separable ground metric onto a
/// (Dim+1)-partite DAG layered graph, solved via LEMON (NetworkSimplex or
/// CostScaling).
///
/// @tparam Dim        Grid dimensionality (>= 1).
/// @tparam Scalar     Input histogram scalar type.
/// @tparam CompScalar Scalar type used for computation (default: double).
/// @tparam CostFn     Separable axis cost functor.
template <std::size_t Dim, std::floating_point Scalar,
          std::floating_point CompScalar = double, typename CostFn>  // NOLINT(*)
  requires(Dim >= 1 && detail::ValidLemonCostFn<CostFn>)                // NOLINT(*)
[[nodiscard]] CompScalar mcf_dpartion(
    const GridDataView<Dim, Scalar>& h1, const GridDataView<Dim, Scalar>& h2,
    CostFn&& cost_fn,
    McfLemonAlgorithm algo = McfLemonAlgorithm::NetworkSimplex,
    SparseTransportPlanPtr<CompScalar> plan = nullptr,
    CompScalar scale = static_cast<CompScalar>(1e6),
    CompScalar mass_tol = default_mass_tolerance<CompScalar>) {
  detail::SolverLog log(
      "mcf_dpartion",
      fmt::format("Dim={}, algo={}, scale={}", Dim,
                  algo == McfLemonAlgorithm::NetworkSimplex ? "NetworkSimplex"
                                                            : "CostScaling",
                  scale));

  detail::validate_unit_mass_pair(h1, h2, mass_tol);

  const auto& layout = h1.layout();
  const auto& shape = layout.shape();
  const std::size_t n_nodes = layout.node_count();

  const std::size_t total_nodes = (Dim + 1) * n_nodes;
  const std::size_t layer_dim_offset = Dim * n_nodes;

  std::vector<int64_t> supply(total_nodes, 0);
  std::size_t max_abs_idx = 0;
  int64_t max_abs_val = -1;

  constexpr bool do_extract_self_mass =
      detail::should_extract_self_mass_v<CostFn>;

  detail::CumulativeQuantizer<CompScalar> quantizer1(scale);
  detail::CumulativeQuantizer<CompScalar> quantizer2(scale);

  for (std::size_t i = 0; i < n_nodes; ++i) {
    const CompScalar v1 = static_cast<CompScalar>(h1.data()[i]);
    const CompScalar v2 = static_cast<CompScalar>(h2.data()[i]);
    const CompScalar self_mass =
        do_extract_self_mass ? std::min(v1, v2) : CompScalar{0};

    const int64_t s1 = quantizer1.push(v1 - self_mass);
    const int64_t s2 = -quantizer2.push(v2 - self_mass);

    supply[i] = s1;
    const int64_t abs_s1 = std::abs(s1);
    if (abs_s1 > max_abs_val) {
      max_abs_val = abs_s1;
      max_abs_idx = i;
    }

    supply[layer_dim_offset + i] = s2;
    const int64_t abs_s2 = std::abs(s2);
    if (abs_s2 > max_abs_val) {
      max_abs_val = abs_s2;
      max_abs_idx = layer_dim_offset + i;
    }
  }

  // Drift is charged to the largest-magnitude supply, tracked above across
  // both layers; detail::absorb_quantization_drift is not used here because a
  // linear scan of `supply` would visit the two layers in a different order
  // and so break ties differently.
  const int64_t total_supply_sum =
      quantizer1.scaled_total() - quantizer2.scaled_total();
  if (total_supply_sum != 0) {
    supply[max_abs_idx] -= total_supply_sum;
  }

  const int64_t cap_val = detail::total_positive_supply(supply);

  log.phase("supply setup",
            fmt::format("bins={}, layered nodes={}", n_nodes, total_nodes));

  using Graph = lemon::SmartDigraph;
  using Node = Graph::Node;
  using Arc = Graph::Arc;

  Graph graph;
  graph.reserveNode(static_cast<int>(total_nodes));

  std::size_t expected_arcs = 0;
  for (std::size_t k = 0; k < Dim; ++k) {
    expected_arcs += (n_nodes / shape[k]) * shape[k] * shape[k];
  }
  graph.reserveArc(static_cast<int>(expected_arcs));

  std::vector<Node> nodes;
  nodes.reserve(total_nodes);
  for (std::size_t i = 0; i < total_nodes; ++i) {
    nodes.push_back(graph.addNode());
  }

  Graph::ArcMap<int64_t> capacity(graph);
  Graph::ArcMap<int64_t> cost(graph);
  Graph::NodeMap<int64_t> supply_map(graph);

  const auto stride = detail::compute_grid_strides<Dim>(shape);

  for (std::size_t k = 0; k < Dim; ++k) {
    const std::size_t extent_k = shape[k];
    const std::size_t st_k = stride[k];
    const std::size_t layer_src_offset = k * n_nodes;
    const std::size_t layer_dst_offset = (k + 1) * n_nodes;

    for (std::size_t base_u = 0; base_u < n_nodes; ++base_u) {
      if ((base_u / st_k) % extent_k != 0) {
        continue;
      }
      for (std::size_t a_k = 0; a_k < extent_k; ++a_k) {
        const std::size_t u = base_u + (a_k * st_k);
        const Node src = nodes[layer_src_offset + u];
        for (std::size_t b_k = 0; b_k < extent_k; ++b_k) {
          const std::size_t v = base_u + (b_k * st_k);
          const Node dst = nodes[layer_dst_offset + v];
          const Arc arc = graph.addArc(src, dst);
          capacity[arc] = cap_val;
          const auto c_val = cost_fn(k, a_k, b_k);
          cost[arc] =
              static_cast<int64_t>(std::llround(static_cast<double>(c_val)));
        }
      }
    }
  }

  for (std::size_t i = 0; i < total_nodes; ++i) {
    supply_map[nodes[i]] = supply[i];
  }

  log.phase("graph construction",
            fmt::format("nodes={}, arcs={}", total_nodes, expected_arcs));

  int64_t raw_optimal_cost = 0;
  std::vector<std::vector<detail::FlowEdge>> flow_adj(total_nodes);
  auto* flow_adj_ptr = plan ? &flow_adj : nullptr;

  if (algo == McfLemonAlgorithm::NetworkSimplex) {
    using Solver = lemon::NetworkSimplex<Graph, int64_t, int64_t>;
    raw_optimal_cost = detail::run_lemon_mcf<Solver>(
        graph, capacity, cost, supply_map, &log, flow_adj_ptr);
  } else {
    using Solver = lemon::CostScaling<Graph, int64_t, int64_t>;
    raw_optimal_cost = detail::run_lemon_mcf<Solver>(
        graph, capacity, cost, supply_map, &log, flow_adj_ptr);
  }

  const CompScalar total_cost =
      static_cast<CompScalar>(raw_optimal_cost) / scale;

  if (plan) {
    plan->source.clear();
    plan->target.clear();
    plan->flow.clear();

    if constexpr (do_extract_self_mass) {
      detail::emit_self_mass(h1, h2, plan);
    }

    // On the layered DAG a path leaves layer 0, crosses Dim arcs and lands on
    // a sink-layer node; the intermediate layers carry zero supply, so the
    // generic "walk until the current node has a deficit" rule stops in
    // exactly the same places the hand-rolled layer test did.
    detail::decompose_flows(
        &flow_adj, supply, n_nodes, scale,
        [layer_dim_offset](std::size_t node) {
          return node - layer_dim_offset;
        },
        plan);
    log.phase("flow decomposition",
              fmt::format("entries={}", plan->flow.size()));
  }

  log.finish(total_cost);
  return total_cost;
}

/// Overload of mcf_dpartion accepting GroundMetric enum.
template <std::size_t Dim, std::floating_point Scalar,
          std::floating_point CompScalar = double>  // NOLINT(*)
  requires(Dim >= 1)                                 // NOLINT(*)
[[nodiscard]] CompScalar mcf_dpartion(
    const GridDataView<Dim, Scalar>& h1, const GridDataView<Dim, Scalar>& h2,
    GroundMetric metric,
    McfLemonAlgorithm algo = McfLemonAlgorithm::NetworkSimplex,
    SparseTransportPlanPtr<CompScalar> plan = nullptr,
    CompScalar scale = static_cast<CompScalar>(1e6),
    CompScalar mass_tol = default_mass_tolerance<CompScalar>) {
  if (metric == GroundMetric::SqEuclidean) {
    return mcf_dpartion<Dim, Scalar, CompScalar>(
        h1, h2, SqEuclideanCost{}, algo, plan, scale, mass_tol);
  }
  return mcf_dpartion<Dim, Scalar, CompScalar>(
      h1, h2, L1Cost{}, algo, plan, scale, mass_tol);
}

/// Overload of mcf_dpartion defaulting to GroundMetric::L1.
template <std::size_t Dim, std::floating_point Scalar,
          std::floating_point CompScalar = double>  // NOLINT(*)
  requires(Dim >= 1)                                 // NOLINT(*)
[[nodiscard]] CompScalar mcf_dpartion(
    const GridDataView<Dim, Scalar>& h1, const GridDataView<Dim, Scalar>& h2,
    McfLemonAlgorithm algo = McfLemonAlgorithm::NetworkSimplex,
    SparseTransportPlanPtr<CompScalar> plan = nullptr,
    CompScalar scale = static_cast<CompScalar>(1e6),
    CompScalar mass_tol = default_mass_tolerance<CompScalar>) {
  return mcf_dpartion<Dim, Scalar, CompScalar>(
      h1, h2, GroundMetric::L1, algo, plan, scale, mass_tol);
}

}  // namespace emdgrid
