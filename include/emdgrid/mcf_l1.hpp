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
[[nodiscard]] CompScalar mcf_l1(const GridDataView<Dim, Scalar>& h1,
                                const GridDataView<Dim, Scalar>& h2,
                                SparseTransportPlan* plan = nullptr,
                                double scale = 1e6,
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
  std::size_t max_abs_idx = 0;
  int64_t max_abs_val = -1;

  double cum_target = 0.0;
  int64_t cum_scaled_prev = 0;

  for (std::size_t i = 0; i < n_nodes; ++i) {
    const double diff = static_cast<double>(h1.data()[i]) -
                        static_cast<double>(h2.data()[i]);
    cum_target += diff;
    const int64_t cum_scaled = std::llround(cum_target * scale);
    const int64_t s = cum_scaled - cum_scaled_prev;
    supply[i] = s;
    cum_scaled_prev = cum_scaled;

    const int64_t abs_s = std::abs(s);
    if (abs_s > max_abs_val) {
      max_abs_val = abs_s;
      max_abs_idx = i;
    }
  }

  // Fix rounding drift so total supply sums to exactly 0
  if (cum_scaled_prev != 0) {
    supply[max_abs_idx] -= cum_scaled_prev;
  }

  int64_t total_pos_supply = 0;
  for (const int64_t s : supply) {
    if (s > 0) {
      total_pos_supply += s;
    }
  }
  const int64_t cap_val = std::max<int64_t>(total_pos_supply, 1);

  operations_research::SimpleMinCostFlow mcf;

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
      static_cast<CompScalar>(mcf.OptimalCost()) /
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

    struct FlowEdge {
      uint32_t head;
      int64_t flow;
    };

    std::vector<std::vector<FlowEdge>> flow_adj(n_nodes);
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
        plan->flow.push_back(static_cast<double>(bottleneck) / scale);
      }
    }
  }

  return total_cost;
}

}  // namespace emdgrid
