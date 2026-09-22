#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <utility>
#include <vector>

#include "emdgrid/emdgrid.hpp"
#include "emdgrid/grid_detail.hpp"
#include "emdgrid/utils.hpp"

namespace emdgrid {

namespace detail {

/// An arc carrying positive flow, keyed by its head node.
///
/// Every min-cost-flow backend reports its solution as arc flows; the solvers
/// bucket those into a per-tail adjacency list of these edges before
/// decomposing them into a transport plan.
struct FlowEdge {
  uint32_t head;
  int64_t flow;
};

/// Turns a running real-valued total into integer increments.
///
/// Rounding each bin independently lets the per-bin errors accumulate, so the
/// quantized supplies drift away from summing to zero. Rounding the
/// *cumulative* total instead and emitting differences bounds the error at
/// half a quantum for every prefix, and makes the increments sum exactly to
/// `llround(total * scale)`.
template <std::floating_point CompScalar>
class CumulativeQuantizer {
 public:
  explicit CumulativeQuantizer(CompScalar scale) : m_scale(scale) {}

  /// Adds `value` to the running total and returns this bin's increment.
  [[nodiscard]] int64_t push(CompScalar value) {
    m_total += value;
    const int64_t scaled = std::llround(m_total * m_scale);
    const int64_t increment = scaled - m_scaled;
    m_scaled = scaled;
    return increment;
  }

  /// Scaled running total, i.e. the sum of every increment emitted so far.
  [[nodiscard]] int64_t scaled_total() const noexcept { return m_scaled; }

 private:
  CompScalar m_scale;
  CompScalar m_total{0};
  int64_t m_scaled{0};
};

/// Removes `residual` from the largest-magnitude supply so the total is zero.
///
/// A min-cost flow is infeasible unless supplies balance exactly. Charging the
/// leftover to the largest entry keeps the relative perturbation smallest.
/// Does nothing when the supplies already balance. Used both to fix up
/// cumulative rounding drift in an integer-quantized supply (`Value=int64_t`)
/// and, in emd_potlemon, to force an exact zero total on a real-valued one
/// (`Value=double`) before handing it to a solver whose own `Value` template
/// parameter need not be an integer type — there it also absorbs any
/// imbalance left by the caller's mass tolerance being looser than the
/// solver's own feasibility epsilon.
template <typename Value>
void absorb_residual(std::span<Value> supply, Value residual) {
  if (residual == Value{0} || supply.empty()) {
    return;
  }

  std::size_t max_abs_idx = 0;
  Value max_abs_val{-1};
  for (std::size_t i = 0; i < supply.size(); ++i) {
    const Value abs_s = std::abs(supply[i]);
    if (abs_s > max_abs_val) {
      max_abs_val = abs_s;
      max_abs_idx = i;
    }
  }
  supply[max_abs_idx] -= residual;
}

/// Quantizes the per-bin net supply `h1 - h2` onto an integer `scale`.
///
/// The returned supplies sum to exactly zero, as the min-cost flow backends
/// require.
template <std::size_t Dim, std::floating_point Scalar,
          std::floating_point CompScalar>
[[nodiscard]] std::vector<int64_t> quantize_net_supply(
    const GridDataView<Dim, Scalar>& h1, const GridDataView<Dim, Scalar>& h2,
    CompScalar scale) {
  const std::size_t n_nodes = h1.layout().node_count();

  std::vector<int64_t> supply(n_nodes);
  CumulativeQuantizer<CompScalar> quantizer(scale);
  for (std::size_t i = 0; i < n_nodes; ++i) {
    supply[i] = quantizer.push(static_cast<CompScalar>(h1.data()[i]) -
                               static_cast<CompScalar>(h2.data()[i]));
  }
  absorb_residual<int64_t>(supply, quantizer.scaled_total());

  return supply;
}

/// Sum of the positive supplies, clamped to at least one.
///
/// Usable as a uniform per-arc capacity: no arc can ever carry more than the
/// total mass being moved, so this never constrains the optimum. The clamp
/// keeps the capacity valid when there is nothing to move.
[[nodiscard]] inline int64_t total_positive_supply(
    std::span<const int64_t> supply) {
  int64_t total = 0;
  for (const int64_t s : supply) {
    if (s > 0) {
      total += s;
    }
  }
  return std::max<int64_t>(total, 1);
}

/// Decomposes arc flows into direct (source bin, target bin, mass) entries.
///
/// A min-cost flow backend reports how much mass crosses each arc, but a
/// transport plan has to say which bin each unit came from and where it ended
/// up. This walks from a node with surplus along arcs that still carry flow
/// until it reaches one with a deficit, appends that path's bottleneck to the
/// plan, and subtracts it from every arc on the path, until no surplus is
/// left. The per-node `ptr` cursor never rewinds, so an exhausted arc is
/// skipped once and never revisited.
///
/// `flow_adj` is consumed in place and `rem_supply` is taken by value, both as
/// scratch. `n_sources` bounds the nodes that can start a path. `target_bin`
/// maps the node a path ended on to the bin reported in the plan: solvers on
/// the grid graph pass std::identity, layered solvers subtract their sink
/// layer's offset.
template <std::floating_point CompScalar, class BinOf>
void decompose_flows(std::vector<std::vector<FlowEdge>>* flow_adj,
                     std::vector<int64_t> rem_supply, std::size_t n_sources,
                     CompScalar scale, BinOf&& target_bin,
                     SparseTransportPlan<CompScalar>* plan) {
  std::vector<std::size_t> ptr(flow_adj->size(), 0);

  for (std::size_t src = 0; src < n_sources; ++src) {
    while (rem_supply[src] > 0) {
      std::vector<std::pair<std::size_t, std::size_t>> path_edges;
      std::size_t cur = src;

      while (true) {
        if (rem_supply[cur] < 0 && cur != src) {
          break;
        }
        auto& list = (*flow_adj)[cur];
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
        bottleneck = std::min(bottleneck, (*flow_adj)[u][p].flow);
      }

      if (bottleneck <= 0) {
        break;
      }

      for (const auto& [u, p] : path_edges) {
        (*flow_adj)[u][p].flow -= bottleneck;
      }
      rem_supply[src] -= bottleneck;
      rem_supply[target] += bottleneck;

      plan->source.push_back(static_cast<uint32_t>(src));
      plan->target.push_back(static_cast<uint32_t>(target_bin(target)));
      plan->flow.push_back(static_cast<CompScalar>(bottleneck) / scale);
    }
  }
}

}  // namespace detail

}  // namespace emdgrid
