#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "emdgrid/detail/potlemon/full_bipartitegraph.h"
#include "emdgrid/detail/potlemon/network_simplex_simple.h"
#include "emdgrid/emdgrid.hpp"
#include "emdgrid/grid_detail.hpp"
#include "emdgrid/log_detail.hpp"
#include "emdgrid/mcf_detail.hpp"
#include "emdgrid/mcf_potlemon_l1.hpp"
#include "emdgrid/utils.hpp"

namespace emdgrid {

namespace detail {

/// Metric selector understood by potlemon's `setLazyCost`.
///
/// The lazy cost is computed inside the solver, from raw coordinate arrays, so
/// the metric is an integer code rather than a functor. Only the two codes
/// emdgrid exposes are named here; code 1 (Euclidean) is deliberately left out
/// because the solver evaluates it as `sqrt` of an integer accumulator, which
/// truncates.
constexpr int lazy_metric_sqeuclidean = 0;
constexpr int lazy_metric_cityblock = 2;

}  // namespace detail

/// Bipartite Earth Mover's Distance with genuinely lazy costs.
///
/// Same formulation as emd_lemon — every bin with leftover supply is joined to
/// every bin with leftover demand — but solved by the vendored potlemon
/// network simplex rather than by LEMON, which is what makes the laziness
/// reach all the way down. This is the direct counterpart of POT's
/// `EMD_wrap_lazy`
/// (https://github.com/PythonOT/POT/blob/master/ot/lp/EMD_wrapper.cpp).
///
/// **Nothing is stored per arc.** The graph's arcs are implicit
/// (potlemon::FullBipartiteDigraph) and four storage modes keep the solver
/// from ever materialising them:
///   - `CostStorageMode::ArtificialArcCosts` — real arc costs are recomputed
///     from the two bins' coordinates on demand; only the artificial root arcs
///     carry stored costs, because the simplex assigns those itself.
///   - `EndpointStorageMode::ArcEndpoints` — endpoints come from the arc id by
///     a division and a remainder, not from `_source` / `_target` arrays.
///   - `FlowStorageMode::SparseArcFlows` — only the nonzero flows are kept, of
///     which a basic solution has at most `n + m - 1`.
///   - `StateStorageMode::PackedArcStates` — two bits per arc instead of a
///     byte.
///
/// Peak memory is therefore proportional to `n + m`, not to `n · m`. Measured
/// on a 20^3 grid under a squared Euclidean metric (8000 x 8000 arcs), this
/// solver peaks around 0.12 GB where emd_lemon peaks at 3.4 GB, because
/// LEMON's NetworkSimplex copies the whole problem into arc-indexed arrays and
/// offers no way to opt out. See emd_lemon.hpp for that accounting.
///
/// The trade is speed: recomputing a cost at every pricing step is slower than
/// reading a packed array, so expect this to run several times slower than
/// emd_lemon on a problem that fits in memory. Reach for it when the dense
/// problem does not fit, and for a separable ground metric prefer mcf_dpartion
/// to either — it is smaller *and* faster (again, see emd_lemon.hpp).
///
/// **Metric.** The cost lives inside the vendored solver and is selected by an
/// integer code, so unlike emd_lemon this entry point takes a `GroundMetric`
/// and cannot accept a custom cost functor. `GroundMetric::L1` extracts the
/// self-mass `min(h1[i], h2[i])`, which a metric never has to move;
/// `GroundMetric::SqEuclidean` does not, since it fails the triangle
/// inequality.
///
/// @tparam Dim        Grid dimensionality (>= 1).
/// @tparam Scalar     Input histogram scalar type.
/// @tparam CompScalar Scalar type used for computation (default: double).
///
/// @param h1       Source histogram, unit mass.
/// @param h2       Target histogram, unit mass.
/// @param metric   Ground metric, L1 or squared Euclidean.
/// @param plan     Optional sparse transport plan output.
/// @param scale    Quantization scale for the supplies.
/// @param mass_tol Tolerance on each histogram's deviation from unit mass.
/// @param max_iter Iteration cap; reaching it logs a warning and returns a
///                 feasible but unproven cost.
template <std::size_t Dim, std::floating_point Scalar,
          std::floating_point CompScalar = double>  // NOLINT(*)
  requires(Dim >= 1)                                 // NOLINT(*)
[[nodiscard]] CompScalar emd_potlemon(
    const GridDataView<Dim, Scalar>& h1, const GridDataView<Dim, Scalar>& h2,
    GroundMetric metric = GroundMetric::L1,
    SparseTransportPlanPtr<CompScalar> plan = nullptr,
    CompScalar scale = static_cast<CompScalar>(1e6),
    CompScalar mass_tol = default_mass_tolerance<CompScalar>,
    uint64_t max_iter = 500000) {
  const bool is_l1 = (metric == GroundMetric::L1);

  detail::SolverLog log(
      "emd_potlemon",
      fmt::format("Dim={}, metric={}, scale={}, max_iter={}", Dim,
                  is_l1 ? "L1" : "SqEuclidean", scale, max_iter));

  detail::validate_unit_mass_pair(h1, h2, mass_tol);

  const auto& layout = h1.layout();
  const std::size_t n_nodes = layout.node_count();

  if (plan != nullptr) {
    plan->source.clear();
    plan->target.clear();
    plan->flow.clear();
  }

  // Quantize each side's residual separately, exactly as emd_lemon does, so
  // that the source supplies stay nonnegative and the target demands too.
  std::vector<int64_t> supply1(n_nodes);
  std::vector<int64_t> supply2(n_nodes);
  detail::CumulativeQuantizer<CompScalar> quantizer1(scale);
  detail::CumulativeQuantizer<CompScalar> quantizer2(scale);

  for (std::size_t i = 0; i < n_nodes; ++i) {
    const CompScalar v1 = static_cast<CompScalar>(h1.data()[i]);
    const CompScalar v2 = static_cast<CompScalar>(h2.data()[i]);
    const CompScalar self_mass = is_l1 ? std::min(v1, v2) : CompScalar{0};
    supply1[i] = quantizer1.push(v1 - self_mass);
    supply2[i] = quantizer2.push(v2 - self_mass);
  }

  detail::absorb_quantization_drift(
      supply1, quantizer1.scaled_total() - quantizer2.scaled_total());

  if (plan != nullptr && is_l1) {
    detail::emit_self_mass(h1, h2, plan);
  }

  // Keep the bins that still have something to move, and flatten their grid
  // coordinates into the layout setLazyCost expects: row-major, Dim doubles
  // per point. This is the only per-bin buffer the solve needs.
  std::vector<uint32_t> bin_a;
  std::vector<uint32_t> bin_b;
  std::vector<double> coords_a;
  std::vector<double> coords_b;
  std::vector<int64_t> supply;

  for (std::size_t i = 0; i < n_nodes; ++i) {
    if (supply1[i] == 0) {
      continue;
    }
    bin_a.push_back(static_cast<uint32_t>(i));
    const auto coord = layout.coordinates(static_cast<std::ptrdiff_t>(i));
    for (std::size_t k = 0; k < Dim; ++k) {
      coords_a.push_back(static_cast<double>(coord[k]));
    }
  }
  for (std::size_t i = 0; i < n_nodes; ++i) {
    if (supply2[i] == 0) {
      continue;
    }
    bin_b.push_back(static_cast<uint32_t>(i));
    const auto coord = layout.coordinates(static_cast<std::ptrdiff_t>(i));
    for (std::size_t k = 0; k < Dim; ++k) {
      coords_b.push_back(static_cast<double>(coord[k]));
    }
  }

  const auto n_src = static_cast<int>(bin_a.size());
  const auto n_tgt = static_cast<int>(bin_b.size());

  supply.reserve(bin_a.size() + bin_b.size());
  for (const uint32_t i : bin_a) {
    supply.push_back(supply1[i]);
  }
  for (const uint32_t j : bin_b) {
    supply.push_back(-supply2[j]);
  }

  log.phase("supply setup",
            fmt::format("bins={}, sources={}, sinks={}", n_nodes, n_src,
                        n_tgt));

  if (n_src == 0 || n_tgt == 0) {
    spdlog::info(
        "emd_potlemon: histograms agree after quantization, nothing to "
        "transport");
    log.finish(CompScalar{0});
    return static_cast<CompScalar>(0.0);
  }

  using Digraph = potlemon::FullBipartiteDigraph;
  using Simplex = potlemon::NetworkSimplexSimple<Digraph, int64_t, int64_t>;

  const int64_t total_arcs =
      static_cast<int64_t>(n_src) * static_cast<int64_t>(n_tgt);

  const Digraph di(n_src, n_tgt);

  Simplex::SimplexOptions options(false);
  options.cost_storage_mode = Simplex::CostStorageMode::ArtificialArcCosts;
  options.flow_storage_mode = Simplex::FlowStorageMode::SparseArcFlows;
  options.endpoint_storage_mode = Simplex::EndpointStorageMode::ArcEndpoints;
  options.state_storage_mode = Simplex::StateStorageMode::PackedArcStates;

  Simplex net(di, options, n_src + n_tgt, total_arcs, max_iter);
  net.supplyMap(supply);
  net.setLazyCost(coords_a.data(), coords_b.data(), static_cast<int>(Dim),
                  is_l1 ? detail::lazy_metric_cityblock
                        : detail::lazy_metric_sqeuclidean,
                  n_src, n_tgt);

  log.phase("graph construction",
            fmt::format("nodes={}, implicit arcs={}", n_src + n_tgt,
                        total_arcs));

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

  // Read the answer straight out of the sparse flow map. `totalCost()` would
  // work too, but under SparseArcFlows it walks all n·m arc ids and hashes
  // each one; the map holds only the basis, so this is O(n + m) instead.
  // POT's own extraction takes the O(n·m) route.
  int64_t raw_cost = 0;
  for (const auto& [arc_id, arc_flow] : net._real_flow) {
    if (arc_flow <= 0) {
      continue;
    }
    // The solver numbers its arcs in reverse, so undo that to recover the
    // graph arc id, then split it into the two sides.
    const int64_t graph_arc = total_arcs - arc_id - 1;
    const auto i = static_cast<int>(graph_arc / n_tgt);
    const auto j = static_cast<int>(graph_arc % n_tgt);

    raw_cost += arc_flow * net.computeLazyCost(i, j);

    if (plan != nullptr) {
      plan->source.push_back(bin_a[static_cast<std::size_t>(i)]);
      plan->target.push_back(bin_b[static_cast<std::size_t>(j)]);
      plan->flow.push_back(static_cast<CompScalar>(arc_flow) / scale);
    }
  }

  const CompScalar total_cost = static_cast<CompScalar>(raw_cost) / scale;

  if (plan != nullptr) {
    log.phase("plan extraction",
              fmt::format("entries={}", plan->flow.size()));
  }

  log.finish(total_cost);
  return total_cost;
}

}  // namespace emdgrid
