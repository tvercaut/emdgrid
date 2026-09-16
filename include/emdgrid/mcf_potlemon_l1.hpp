#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "emdgrid/detail/potlemon/network_simplex_simple.h"
#include "emdgrid/detail/potlemon/sparse_bipartitegraph.h"
#include "emdgrid/emdgrid.hpp"
#include "emdgrid/utils.hpp"

namespace emdgrid {

namespace detail {

/// Edge in the grid-flow adjacency list used for plan decomposition.
struct PotlemonFlowEdge {
  uint32_t head;
  int64_t flow;
};

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
    SparseTransportPlan* plan = nullptr, double scale = 1e6,
    double mass_tol = 1e-6, uint64_t max_iter = 500000) {
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

  if (plan) {
    plan->source.clear();
    plan->target.clear();
    plan->flow.clear();
  }

  // Integer net supply per node via cumulative rounding (keeps total == 0).
  std::vector<int64_t> node_supply(n_nodes, 0);
  double cum = 0.0;
  int64_t cum_scaled_prev = 0;
  int64_t max_abs_supply = -1;
  std::size_t max_abs_idx = 0;

  for (std::size_t i = 0; i < n_nodes; ++i) {
    const double v1 = static_cast<double>(h1.data()[i]);
    const double v2 = static_cast<double>(h2.data()[i]);

    if (plan) {
      const double self_mass = std::min(v1, v2);
      if (self_mass > 0.0) {
        plan->source.push_back(static_cast<uint32_t>(i));
        plan->target.push_back(static_cast<uint32_t>(i));
        plan->flow.push_back(self_mass);
      }
    }

    cum += v1 - v2;
    const int64_t cum_scaled = std::llround(cum * scale);
    node_supply[i] = cum_scaled - cum_scaled_prev;
    cum_scaled_prev = cum_scaled;
    if (std::abs(node_supply[i]) > max_abs_supply) {
      max_abs_supply = std::abs(node_supply[i]);
      max_abs_idx = i;
    }
  }

  // Correct any residual rounding error.
  if (cum_scaled_prev != 0) {
    node_supply[max_abs_idx] -= cum_scaled_prev;
  }

  bool any_nonzero = false;
  for (std::size_t i = 0; i < n_nodes; ++i) {
    if (node_supply[i] != 0) {
      any_nonzero = true;
      break;
    }
  }
  if (!any_nonzero) {
    return static_cast<CompScalar>(0.0);
  }

  // Grid-adjacent arcs: forward and backward for each adjacent cell pair.
  using Digraph = potlemon::SparseDigraph;
  std::vector<std::pair<int, int>> edges;

  for (std::size_t u = 0; u < n_nodes; ++u) {
    const auto coords =
        layout.coordinates(static_cast<std::ptrdiff_t>(u));
    for (std::size_t axis = 0; axis < Dim; ++axis) {
      if (coords[axis] + 1 < static_cast<std::ptrdiff_t>(shape[axis])) {
        auto next_coords = coords;
        ++next_coords[axis];
        const std::size_t v =
            static_cast<std::size_t>(layout.node(next_coords));
        edges.emplace_back(static_cast<int>(u), static_cast<int>(v));
        edges.emplace_back(static_cast<int>(v), static_cast<int>(u));
      }
    }
  }

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

  const auto status = net.run();
  if (status != Simplex::OPTIMAL && status != Simplex::MAX_ITER_REACHED) {
    throw std::runtime_error("potlemon network simplex solve failed");
  }

  const int64_t raw_cost = net.totalCost();
  const CompScalar total_cost =
      static_cast<CompScalar>(raw_cost) / static_cast<CompScalar>(scale);

  if (!plan) {
    return total_cost;
  }

  // Decompose arc flows into (source_bin, target_bin, mass) pairs using the
  // same path-tracing strategy as mcf_lemon_l1: repeatedly trace from each
  // source along edges with remaining flow to a sink, record the bottleneck,
  // and subtract it from the path.
  std::vector<std::vector<detail::PotlemonFlowEdge>> flow_adj(n_nodes);
  for (int64_t k = 0; k < total_arcs; ++k) {
    const Digraph::Arc a = Digraph::arcFromId(k);
    const int64_t f = net.flow(a);
    if (f > 0) {
      const auto u = static_cast<std::size_t>(di.source(a));
      const auto v = static_cast<uint32_t>(di.target(a));
      flow_adj[u].push_back({v, f});
    }
  }

  std::vector<int64_t> rem_supply = node_supply;
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

  return total_cost;
}

}  // namespace emdgrid
