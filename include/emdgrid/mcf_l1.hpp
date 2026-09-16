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

#pragma push_macro("CHECK")
#pragma push_macro("CHECK_EQ")
#pragma push_macro("CHECK_NE")
#pragma push_macro("CHECK_LE")
#pragma push_macro("CHECK_LT")
#pragma push_macro("CHECK_GE")
#pragma push_macro("CHECK_GT")
#pragma push_macro("DCHECK")
#pragma push_macro("DCHECK_EQ")
#pragma push_macro("DCHECK_NE")
#pragma push_macro("DCHECK_LE")
#pragma push_macro("DCHECK_LT")
#pragma push_macro("DCHECK_GE")
#pragma push_macro("DCHECK_GT")

#undef CHECK
#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_LE
#undef CHECK_LT
#undef CHECK_GE
#undef CHECK_GT
#undef DCHECK
#undef DCHECK_EQ
#undef DCHECK_NE
#undef DCHECK_LE
#undef DCHECK_LT
#undef DCHECK_GE
#undef DCHECK_GT

#include <ortools/graph/min_cost_flow.h>  // NOLINT(build/include_order)

#pragma pop_macro("DCHECK_GT")
#pragma pop_macro("DCHECK_GE")
#pragma pop_macro("DCHECK_LT")
#pragma pop_macro("DCHECK_LE")
#pragma pop_macro("DCHECK_NE")
#pragma pop_macro("DCHECK_EQ")
#pragma pop_macro("DCHECK")
#pragma pop_macro("CHECK_GT")
#pragma pop_macro("CHECK_GE")
#pragma pop_macro("CHECK_LT")
#pragma pop_macro("CHECK_LE")
#pragma pop_macro("CHECK_NE")
#pragma pop_macro("CHECK_EQ")
#pragma pop_macro("CHECK")

#include "emdgrid/emdgrid.hpp"
#include "emdgrid/grid_detail.hpp"
#include "emdgrid/mcf_detail.hpp"
#include "emdgrid/utils.hpp"

namespace emdgrid {

/// EMD-L1 for multi-dimensional grid histograms solved via Min-Cost Flow
/// using OR-Tools SimpleMinCostFlow.
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
[[nodiscard]] CompScalar mcf_l1(
    const GridDataView<Dim, Scalar>& h1, const GridDataView<Dim, Scalar>& h2,
    SparseTransportPlanPtr<CompScalar> plan = nullptr,
    CompScalar scale = static_cast<CompScalar>(1e6),
    CompScalar mass_tol = default_mass_tolerance<CompScalar>) {
  detail::validate_unit_mass_pair(h1, h2, mass_tol);

  const auto& layout = h1.layout();
  const auto& shape = layout.shape();
  const std::size_t n_nodes = layout.node_count();

  const std::vector<int64_t> supply =
      detail::quantize_net_supply(h1, h2, scale);
  const int64_t cap_val = detail::total_positive_supply(supply);

  operations_research::SimpleMinCostFlow mcf;

  const auto stride = detail::compute_grid_strides<Dim>(shape);

  for (std::size_t a = 0; a < Dim; ++a) {
    const std::size_t extent = shape[a];
    if (extent < 2) {
      continue;
    }
    const std::size_t st = stride[a];

    for (std::size_t u = 0; u < n_nodes; ++u) {
      if ((u / st) % extent < extent - 1) {
        using NodeIdx = operations_research::SimpleMinCostFlow::NodeIndex;
        const auto u_node = static_cast<NodeIdx>(u);
        const auto v = static_cast<NodeIdx>(u + st);
        mcf.AddArcWithCapacityAndUnitCost(u_node, v, cap_val, 1);
        mcf.AddArcWithCapacityAndUnitCost(v, u_node, cap_val, 1);
      }
    }
  }

  for (std::size_t i = 0; i < n_nodes; ++i) {
    if (supply[i] != 0) {
      using NodeIdx = operations_research::SimpleMinCostFlow::NodeIndex;
      mcf.SetNodeSupply(static_cast<NodeIdx>(i), supply[i]);
    }
  }

  const auto status = mcf.Solve();
  if (status != operations_research::SimpleMinCostFlow::OPTIMAL) {
    throw std::runtime_error("min-cost flow solve failed, status=" +
                             std::to_string(static_cast<int>(status)));
  }

  const CompScalar total_cost =
      static_cast<CompScalar>(mcf.OptimalCost()) / scale;

  if (plan) {
    plan->source.clear();
    plan->target.clear();
    plan->flow.clear();

    detail::emit_self_mass(h1, h2, plan);

    std::vector<std::vector<detail::FlowEdge>> flow_adj(n_nodes);
    for (int a = 0; a < mcf.NumArcs(); ++a) {
      const int64_t f = mcf.Flow(a);
      if (f > 0) {
        const auto u = static_cast<std::size_t>(mcf.Tail(a));
        const auto v = static_cast<uint32_t>(mcf.Head(a));
        flow_adj[u].push_back({v, f});
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
        plan->flow.push_back(static_cast<CompScalar>(bottleneck) / scale);
      }
    }
  }

  return total_cost;
}

}  // namespace emdgrid
