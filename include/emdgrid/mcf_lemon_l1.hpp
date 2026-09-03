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

namespace emdgrid {

/// Algorithm variant for LEMON Min-Cost Flow solver.
enum class McfLemonAlgorithm : std::uint8_t { NetworkSimplex, CostScaling };

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

  struct FlowEdge {
    uint32_t head;
    int64_t flow;
  };
  std::vector<std::vector<FlowEdge>> flow_adj(n_nodes);

  if (algo == McfLemonAlgorithm::NetworkSimplex) {
    lemon::NetworkSimplex<Graph, int64_t, int64_t> mcf(graph);
    mcf.upperMap(capacity).costMap(cost).supplyMap(supply_map);
    const auto status = mcf.run();
    if (status != decltype(mcf)::OPTIMAL) {
      throw std::runtime_error("min-cost flow solve failed (NetworkSimplex)");
    }
    raw_optimal_cost = mcf.totalCost();

    if (plan) {
      for (Graph::ArcIt a(graph); a != lemon::INVALID; ++a) {
        const int64_t f = mcf.flow(a);
        if (f > 0) {
          const auto u = static_cast<std::size_t>(Graph::id(graph.source(a)));
          const auto v = static_cast<uint32_t>(Graph::id(graph.target(a)));
          flow_adj[u].push_back({v, f});
        }
      }
    }
  } else {
    lemon::CostScaling<Graph, int64_t, int64_t> mcf(graph);
    mcf.upperMap(capacity).costMap(cost).supplyMap(supply_map);
    const auto status = mcf.run();
    if (status != decltype(mcf)::OPTIMAL) {
      throw std::runtime_error("min-cost flow solve failed (CostScaling)");
    }
    raw_optimal_cost = mcf.totalCost();

    if (plan) {
      for (Graph::ArcIt a(graph); a != lemon::INVALID; ++a) {
        const int64_t f = mcf.flow(a);
        if (f > 0) {
          const auto u = static_cast<std::size_t>(Graph::id(graph.source(a)));
          const auto v = static_cast<uint32_t>(Graph::id(graph.target(a)));
          flow_adj[u].push_back({v, f});
        }
      }
    }
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

}  // namespace emdgrid
