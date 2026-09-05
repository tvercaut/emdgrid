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

struct BinSupply {
  std::size_t bin_idx;
  int64_t supply;
};

}  // namespace detail

/// EMD-L1 for multi-dimensional grid histograms solved via POT Network
/// Simplex.
///
/// Converts the grid histogram distance into a bipartite min-cost flow
/// problem with L1 ground distance costs between bins with non-zero mass
/// differences.
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

  std::vector<detail::BinSupply> sources;
  std::vector<detail::BinSupply> targets;

  int64_t sum_pos = 0;
  int64_t sum_neg = 0;

  std::size_t max_src_idx = 0;
  int64_t max_src_val = -1;
  std::size_t max_tgt_idx = 0;
  int64_t max_tgt_val = -1;

  double cum_target = 0.0;
  int64_t cum_scaled_prev = 0;

  for (std::size_t i = 0; i < n_nodes; ++i) {
    const double v1 = static_cast<double>(h1.data()[i]);
    const double v2 = static_cast<double>(h2.data()[i]);
    const double self_mass = std::min(v1, v2);

    if (plan && self_mass > 0.0) {
      plan->source.push_back(static_cast<uint32_t>(i));
      plan->target.push_back(static_cast<uint32_t>(i));
      plan->flow.push_back(self_mass);
    }

    const double diff = (v1 - self_mass) - (v2 - self_mass);
    cum_target += diff;
    const int64_t cum_scaled = std::llround(cum_target * scale);
    const int64_t s = cum_scaled - cum_scaled_prev;
    cum_scaled_prev = cum_scaled;

    if (s > 0) {
      sources.push_back({i, s});
      sum_pos += s;
      if (s > max_src_val) {
        max_src_val = s;
        max_src_idx = sources.size() - 1;
      }
    } else if (s < 0) {
      targets.push_back({i, s});
      sum_neg += (-s);
      if (-s > max_tgt_val) {
        max_tgt_val = -s;
        max_tgt_idx = targets.size() - 1;
      }
    }
  }

  const std::size_t n = sources.size();
  const std::size_t m = targets.size();

  if (n == 0 || m == 0) {
    return static_cast<CompScalar>(0.0);
  }

  const int64_t diff = sum_pos - sum_neg;
  if (diff != 0) {
    if (diff > 0) {
      targets[max_tgt_idx].supply -= diff;
    } else {
      sources[max_src_idx].supply -= diff;
    }
  }

  using Digraph = potlemon::SparseBipartiteDigraph;
  Digraph di(static_cast<int>(n), static_cast<int>(m));

  const std::size_t total_edges = n * m;
  std::vector<std::pair<int, int>> edges;
  edges.reserve(total_edges);
  std::vector<int64_t> arc_costs;
  arc_costs.reserve(total_edges);

  for (std::size_t i = 0; i < n; ++i) {
    const auto src_bin = sources[i].bin_idx;
    const auto c_src =
        layout.coordinates(static_cast<std::ptrdiff_t>(src_bin));
    for (std::size_t j = 0; j < m; ++j) {
      const auto tgt_bin = targets[j].bin_idx;
      const auto c_tgt =
          layout.coordinates(static_cast<std::ptrdiff_t>(tgt_bin));
      int64_t dist = 0;
      for (std::size_t axis = 0; axis < Dim; ++axis) {
        dist += std::abs(c_src[axis] - c_tgt[axis]);
      }
      edges.emplace_back(static_cast<int>(i), static_cast<int>(j + n));
      arc_costs.push_back(dist);
    }
  }

  di.buildFromEdges(edges);

  using Simplex = potlemon::NetworkSimplexSimple<Digraph, int64_t, int64_t>;
  Simplex::SimplexOptions options(true);
  Simplex net(di, options, static_cast<int>(n + m),
              static_cast<int64_t>(total_edges), max_iter);

  std::vector<int64_t> src_supplies(n);
  for (std::size_t i = 0; i < n; ++i) {
    src_supplies[i] = sources[i].supply;
  }

  std::vector<int64_t> tgt_demands(m);
  for (std::size_t j = 0; j < m; ++j) {
    tgt_demands[j] = targets[j].supply;
  }

  net.supplyMap(src_supplies.data(), static_cast<int>(n), tgt_demands.data(),
                static_cast<int>(m));

  for (int64_t k = 0; k < static_cast<int64_t>(total_edges); ++k) {
    net.setCost(Digraph::arcFromId(k), arc_costs[k]);
  }

  const auto status = net.run();
  if (status != Simplex::OPTIMAL && status != Simplex::MAX_ITER_REACHED) {
    throw std::runtime_error("potlemon network simplex solve failed");
  }

  const int64_t raw_optimal_cost = net.totalCost();
  const CompScalar total_cost = static_cast<CompScalar>(raw_optimal_cost) /
                                static_cast<CompScalar>(scale);

  if (plan) {
    for (int64_t k = 0; k < static_cast<int64_t>(total_edges); ++k) {
      const Digraph::Arc a = Digraph::arcFromId(k);
      const int64_t f = net.flow(a);
      if (f > 0) {
        const int src_idx = di.source(a);
        const int tgt_idx = di.target(a) - static_cast<int>(n);
        plan->source.push_back(static_cast<uint32_t>(
            sources[static_cast<std::size_t>(src_idx)].bin_idx));
        plan->target.push_back(static_cast<uint32_t>(
            targets[static_cast<std::size_t>(tgt_idx)].bin_idx));
        plan->flow.push_back(static_cast<double>(f) / scale);
      }
    }
  }

  return total_cost;
}

}  // namespace emdgrid
