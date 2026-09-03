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
#include "emdgrid/knothe_rosenblatt_detail.hpp"

namespace emdgrid {

/// Default cost functor for L1 (Manhattan) ground metric.
struct L1Cost {
  [[nodiscard]] int64_t operator()(std::size_t /*axis*/, std::size_t a,
                                  std::size_t b) const noexcept {
    const auto diff =
        static_cast<std::ptrdiff_t>(a) - static_cast<std::ptrdiff_t>(b);
    return std::abs(diff);
  }
};

/// Default cost functor for SqEuclidean (squared Euclidean) ground metric.
struct SqEuclideanCost {
  [[nodiscard]] int64_t operator()(std::size_t /*axis*/, std::size_t a,
                                  std::size_t b) const noexcept {
    const auto diff =
        static_cast<std::ptrdiff_t>(a) - static_cast<std::ptrdiff_t>(b);
    return diff * diff;
  }
};

/// Algorithm variant for LEMON Min-Cost Flow solver.
enum class McfLemonAlgorithm : std::uint8_t { NetworkSimplex, CostScaling };

namespace detail {

template <typename T>
concept ValidDpartCost =
    !std::is_same_v<std::decay_t<T>, McfLemonAlgorithm> &&  // NOLINT(*)
    !std::is_same_v<std::decay_t<T>, GroundMetric>;         // NOLINT(*)

/// Edge with target node and positive flow value.
struct FlowEdge {
  uint32_t head;
  int64_t flow;
};

/// Shared helper to solve LEMON Min-Cost Flow and collect flow edges.
template <typename Solver, typename Graph>
int64_t run_lemon_mcf(
    Graph& graph,
    typename Graph::template ArcMap<int64_t>& capacity,
    typename Graph::template ArcMap<int64_t>& cost,
    typename Graph::template NodeMap<int64_t>& supply,
    std::vector<std::vector<FlowEdge>>* flow_adj = nullptr) {
  Solver mcf(graph);
  mcf.upperMap(capacity).costMap(cost).supplyMap(supply);
  if (mcf.run() != Solver::OPTIMAL) {
    throw std::runtime_error("min-cost flow solve failed");
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
    SparseTransportPlan* plan = nullptr, double scale = 1e6,
    double mass_tol = 1e-6) {
  if (h1.layout().shape() != h2.layout().shape()) {
    throw std::invalid_argument("histogram shapes do not match");
  }

  const auto& layout = h1.layout();
  const auto& shape = layout.shape();
  const std::size_t n_nodes = layout.node_count();

  double t1 = 0.0;
  double t2 = 0.0;
  for (std::size_t i = 0; i < n_nodes; ++i) {
    const double v1 = static_cast<double>(h1.data()[i]);
    const double v2 = static_cast<double>(h2.data()[i]);
    if (v1 < 0.0 || v2 < 0.0) {
      throw std::invalid_argument("histograms must be nonnegative");
    }
    t1 += v1;
    t2 += v2;
  }

  if (std::abs(t1 - 1.0) > mass_tol || std::abs(t2 - 1.0) > mass_tol) {
    throw std::invalid_argument("expected unit-mass histograms");
  }

  std::vector<int64_t> supply(n_nodes);
  int64_t total_supply_sum = 0;
  std::size_t max_abs_idx = 0;
  int64_t max_abs_val = -1;

  for (std::size_t i = 0; i < n_nodes; ++i) {
    const double diff = static_cast<double>(h1.data()[i]) -
                        static_cast<double>(h2.data()[i]);
    const int64_t s = std::llround(diff * scale);
    supply[i] = s;
    total_supply_sum += s;

    const int64_t abs_s = std::abs(s);
    if (abs_s > max_abs_val) {
      max_abs_val = abs_s;
      max_abs_idx = i;
    }
  }

  // Fix rounding drift so total supply sums to exactly 0
  if (total_supply_sum != 0) {
    supply[max_abs_idx] -= total_supply_sum;
  }

  int64_t total_pos_supply = 0;
  for (const int64_t s : supply) {
    if (s > 0) {
      total_pos_supply += s;
    }
  }
  const int64_t cap_val = std::max<int64_t>(total_pos_supply, 1);

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

  std::array<std::ptrdiff_t, Dim> stride{};
  stride[Dim - 1] = 1;
  for (std::size_t a = Dim - 1; a-- > 0;) {
    stride[a] = stride[a + 1] * static_cast<std::ptrdiff_t>(shape[a + 1]);
  }

  for (std::size_t a = 0; a < Dim; ++a) {
    const std::size_t extent = shape[a];
    if (extent < 2) {
      continue;
    }
    const std::ptrdiff_t st = stride[a];
    const auto extent_ptrdiff = static_cast<std::ptrdiff_t>(extent);

    for (std::size_t u = 0; u < n_nodes; ++u) {
      const std::ptrdiff_t u_idx = static_cast<std::ptrdiff_t>(u);
      if ((u_idx / st) % extent_ptrdiff < extent_ptrdiff - 1) {
        const std::size_t v = u + static_cast<std::size_t>(st);

        const Arc a1 = graph.addArc(nodes[u], nodes[v]);
        capacity[a1] = cap_val;
        cost[a1] = 1;

        const Arc a2 = graph.addArc(nodes[v], nodes[u]);
        capacity[a2] = cap_val;
        cost[a2] = 1;
      }
    }
  }

  for (std::size_t i = 0; i < n_nodes; ++i) {
    supply_map[nodes[i]] = supply[i];
  }

  int64_t raw_optimal_cost = 0;
  std::vector<std::vector<detail::FlowEdge>> flow_adj(n_nodes);
  auto* flow_adj_ptr = plan ? &flow_adj : nullptr;

  if (algo == McfLemonAlgorithm::NetworkSimplex) {
    using Solver = lemon::NetworkSimplex<Graph, int64_t, int64_t>;
    raw_optimal_cost = detail::run_lemon_mcf<Solver>(
        graph, capacity, cost, supply_map, flow_adj_ptr);
  } else {
    using Solver = lemon::CostScaling<Graph, int64_t, int64_t>;
    raw_optimal_cost = detail::run_lemon_mcf<Solver>(
        graph, capacity, cost, supply_map, flow_adj_ptr);
  }

  const CompScalar total_cost = static_cast<CompScalar>(raw_optimal_cost) /
                                static_cast<CompScalar>(scale);

  if (plan) {
    plan->source.clear();
    plan->target.clear();
    plan->flow.clear();

    for (std::size_t i = 0; i < n_nodes; ++i) {
      const double self_mass =
          std::min(static_cast<double>(h1.data()[i]),
                   static_cast<double>(h2.data()[i]));
      if (self_mass > 0.0) {
        plan->source.push_back(static_cast<uint32_t>(i));
        plan->target.push_back(static_cast<uint32_t>(i));
        plan->flow.push_back(self_mass);
      }
    }

    std::vector<int64_t> rem_supply = supply;
    std::vector<std::size_t> ptr(n_nodes, 0);

    for (std::size_t src = 0; src < n_nodes; ++src) {
      while (rem_supply[src] > 0) {
        std::vector<std::pair<std::size_t, std::size_t>> path_edges;
        std::size_t cur = src;

        while (true) {
          if (rem_supply[cur] < 0 && cur != src) {
            break;
          }
          auto& list = flow_adj[cur];
          std::size_t p = ptr[cur];
          while (p < list.size() && list[p].flow <= 0) {
            ++p;
          }
          ptr[cur] = p;
          if (p >= list.size()) {
            break;
          }
          path_edges.emplace_back(cur, p);
          cur = static_cast<std::size_t>(list[p].head);
        }

        if (path_edges.empty()) {
          break;
        }

        const std::size_t target = cur;
        if (rem_supply[target] >= 0) {
          break;
        }

        int64_t bottleneck = rem_supply[src];
        bottleneck = std::min(bottleneck, -rem_supply[target]);
        for (const auto& [u, p] : path_edges) {
          bottleneck = std::min(bottleneck, flow_adj[u][p].flow);
        }

        if (bottleneck <= 0) {
          break;
        }

        for (const auto& [u, p] : path_edges) {
          flow_adj[u][p].flow -= bottleneck;
        }
        rem_supply[src] -= bottleneck;
        rem_supply[target] += bottleneck;

        plan->source.push_back(static_cast<uint32_t>(src));
        plan->target.push_back(static_cast<uint32_t>(target));
        plan->flow.push_back(static_cast<double>(bottleneck) / scale);
      }
    }
  }

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
  requires(Dim >= 1 && detail::ValidDpartCost<CostFn>)                // NOLINT(*)
[[nodiscard]] CompScalar mcf_dpartition(
    const GridDataView<Dim, Scalar>& h1, const GridDataView<Dim, Scalar>& h2,
    CostFn&& cost_fn,
    McfLemonAlgorithm algo = McfLemonAlgorithm::NetworkSimplex,
    SparseTransportPlan* plan = nullptr, double scale = 1e6,
    double mass_tol = 1e-6) {
  if (h1.layout().shape() != h2.layout().shape()) {
    throw std::invalid_argument("histogram shapes do not match");
  }

  const auto& layout = h1.layout();
  const auto& shape = layout.shape();
  const std::size_t n_nodes = layout.node_count();

  double t1 = 0.0;
  double t2 = 0.0;
  for (std::size_t i = 0; i < n_nodes; ++i) {
    const double v1 = static_cast<double>(h1.data()[i]);
    const double v2 = static_cast<double>(h2.data()[i]);
    if (v1 < 0.0 || v2 < 0.0) {
      throw std::invalid_argument("histograms must be nonnegative");
    }
    t1 += v1;
    t2 += v2;
  }

  if (std::abs(t1 - 1.0) > mass_tol || std::abs(t2 - 1.0) > mass_tol) {
    throw std::invalid_argument("expected unit-mass histograms");
  }

  const std::size_t total_nodes = (Dim + 1) * n_nodes;
  const std::size_t layer_dim_offset = Dim * n_nodes;

  std::vector<int64_t> supply(total_nodes, 0);
  int64_t total_supply_sum = 0;
  std::size_t max_abs_idx = 0;
  int64_t max_abs_val = -1;

  for (std::size_t i = 0; i < n_nodes; ++i) {
    const double v1 = static_cast<double>(h1.data()[i]);
    const int64_t s1 = std::llround(v1 * scale);
    supply[i] = s1;
    total_supply_sum += s1;

    const int64_t abs_s1 = std::abs(s1);
    if (abs_s1 > max_abs_val) {
      max_abs_val = abs_s1;
      max_abs_idx = i;
    }
  }

  for (std::size_t i = 0; i < n_nodes; ++i) {
    const double v2 = static_cast<double>(h2.data()[i]);
    const int64_t s2 = -std::llround(v2 * scale);
    supply[layer_dim_offset + i] = s2;
    total_supply_sum += s2;

    const int64_t abs_s2 = std::abs(s2);
    if (abs_s2 > max_abs_val) {
      max_abs_val = abs_s2;
      max_abs_idx = layer_dim_offset + i;
    }
  }

  if (total_supply_sum != 0) {
    supply[max_abs_idx] -= total_supply_sum;
  }

  int64_t total_pos_supply = 0;
  for (const int64_t s : supply) {
    if (s > 0) {
      total_pos_supply += s;
    }
  }
  const int64_t cap_val = std::max<int64_t>(total_pos_supply, 1);

  using Graph = lemon::SmartDigraph;
  using Node = Graph::Node;
  using Arc = Graph::Arc;

  Graph graph;
  graph.reserveNode(static_cast<int>(total_nodes));

  std::vector<Node> nodes;
  nodes.reserve(total_nodes);
  for (std::size_t i = 0; i < total_nodes; ++i) {
    nodes.push_back(graph.addNode());
  }

  Graph::ArcMap<int64_t> capacity(graph);
  Graph::ArcMap<int64_t> cost(graph);
  Graph::NodeMap<int64_t> supply_map(graph);

  std::array<std::ptrdiff_t, Dim> stride{};
  stride[Dim - 1] = 1;
  for (std::size_t a = Dim - 1; a-- > 0;) {
    stride[a] = stride[a + 1] * static_cast<std::ptrdiff_t>(shape[a + 1]);
  }

  for (std::size_t k = 0; k < Dim; ++k) {
    const std::size_t extent_k = shape[k];
    const std::ptrdiff_t st_k = stride[k];
    const std::size_t layer_src_offset = k * n_nodes;
    const std::size_t layer_dst_offset = (k + 1) * n_nodes;

    for (std::size_t base_u = 0; base_u < n_nodes; ++base_u) {
      if ((static_cast<std::ptrdiff_t>(base_u) / st_k) %
              static_cast<std::ptrdiff_t>(extent_k) !=
          0) {
        continue;
      }
      for (std::size_t a_k = 0; a_k < extent_k; ++a_k) {
        const std::size_t u = base_u + a_k * static_cast<std::size_t>(st_k);
        const Node src = nodes[layer_src_offset + u];
        for (std::size_t b_k = 0; b_k < extent_k; ++b_k) {
          const std::size_t v = base_u + b_k * static_cast<std::size_t>(st_k);
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

  int64_t raw_optimal_cost = 0;
  std::vector<std::vector<detail::FlowEdge>> flow_adj(total_nodes);
  auto* flow_adj_ptr = plan ? &flow_adj : nullptr;

  if (algo == McfLemonAlgorithm::NetworkSimplex) {
    using Solver = lemon::NetworkSimplex<Graph, int64_t, int64_t>;
    raw_optimal_cost = detail::run_lemon_mcf<Solver>(
        graph, capacity, cost, supply_map, flow_adj_ptr);
  } else {
    using Solver = lemon::CostScaling<Graph, int64_t, int64_t>;
    raw_optimal_cost = detail::run_lemon_mcf<Solver>(
        graph, capacity, cost, supply_map, flow_adj_ptr);
  }

  const CompScalar total_cost = static_cast<CompScalar>(raw_optimal_cost) /
                                static_cast<CompScalar>(scale);

  if (plan) {
    plan->source.clear();
    plan->target.clear();
    plan->flow.clear();

    std::vector<int64_t> rem_supply = supply;
    std::vector<std::size_t> ptr(total_nodes, 0);

    for (std::size_t src = 0; src < n_nodes; ++src) {
      while (rem_supply[src] > 0) {
        std::vector<std::pair<std::size_t, std::size_t>> path_edges;
        std::size_t cur = src;

        while (cur < layer_dim_offset) {
          auto& list = flow_adj[cur];
          std::size_t p = ptr[cur];
          while (p < list.size() && list[p].flow <= 0) {
            ++p;
          }
          ptr[cur] = p;
          if (p >= list.size()) {
            break;
          }
          path_edges.emplace_back(cur, p);
          cur = static_cast<std::size_t>(list[p].head);
        }

        if (path_edges.size() != Dim || cur < layer_dim_offset) {
          break;
        }

        const std::size_t target_node = cur;
        if (rem_supply[target_node] >= 0) {
          break;
        }

        int64_t bottleneck = rem_supply[src];
        bottleneck = std::min(bottleneck, -rem_supply[target_node]);
        for (const auto& [u, p] : path_edges) {
          bottleneck = std::min(bottleneck, flow_adj[u][p].flow);
        }

        if (bottleneck <= 0) {
          break;
        }

        for (const auto& [u, p] : path_edges) {
          flow_adj[u][p].flow -= bottleneck;
        }
        rem_supply[src] -= bottleneck;
        rem_supply[target_node] += bottleneck;

        const std::size_t dst_bin = target_node - layer_dim_offset;
        plan->source.push_back(static_cast<uint32_t>(src));
        plan->target.push_back(static_cast<uint32_t>(dst_bin));
        plan->flow.push_back(static_cast<double>(bottleneck) / scale);
      }
    }
  }

  return total_cost;
}

/// Overload of mcf_dpartition accepting GroundMetric enum.
template <std::size_t Dim, std::floating_point Scalar,
          std::floating_point CompScalar = double>  // NOLINT(*)
  requires(Dim >= 1)                                 // NOLINT(*)
[[nodiscard]] CompScalar mcf_dpartition(
    const GridDataView<Dim, Scalar>& h1, const GridDataView<Dim, Scalar>& h2,
    GroundMetric metric,
    McfLemonAlgorithm algo = McfLemonAlgorithm::NetworkSimplex,
    SparseTransportPlan* plan = nullptr, double scale = 1e6,
    double mass_tol = 1e-6) {
  if (metric == GroundMetric::SqEuclidean) {
    return mcf_dpartition<Dim, Scalar, CompScalar>(
        h1, h2, SqEuclideanCost{}, algo, plan, scale, mass_tol);
  }
  return mcf_dpartition<Dim, Scalar, CompScalar>(
      h1, h2, L1Cost{}, algo, plan, scale, mass_tol);
}

/// Overload of mcf_dpartition defaulting to GroundMetric::L1.
template <std::size_t Dim, std::floating_point Scalar,
          std::floating_point CompScalar = double>  // NOLINT(*)
  requires(Dim >= 1)                                 // NOLINT(*)
[[nodiscard]] CompScalar mcf_dpartition(
    const GridDataView<Dim, Scalar>& h1, const GridDataView<Dim, Scalar>& h2,
    McfLemonAlgorithm algo = McfLemonAlgorithm::NetworkSimplex,
    SparseTransportPlan* plan = nullptr, double scale = 1e6,
    double mass_tol = 1e-6) {
  return mcf_dpartition<Dim, Scalar, CompScalar>(
      h1, h2, GroundMetric::L1, algo, plan, scale, mass_tol);
}

}  // namespace emdgrid
