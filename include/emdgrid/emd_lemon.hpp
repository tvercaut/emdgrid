#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#pragma push_macro("MAX")
#pragma push_macro("MIN")
#undef MAX
#undef MIN

#include <lemon/cost_scaling.h>     // NOLINT(build/include_order)
#include <lemon/maps.h>             // NOLINT(build/include_order)
#include <lemon/network_simplex.h>  // NOLINT(build/include_order)

#pragma pop_macro("MIN")
#pragma pop_macro("MAX")

#include "emdgrid/emd_lemon_detail.hpp"
#include "emdgrid/emdgrid.hpp"
#include "emdgrid/grid_detail.hpp"
#include "emdgrid/log_detail.hpp"
#include "emdgrid/mcf_detail.hpp"
#include "emdgrid/mcf_lemon_l1.hpp"
#include "emdgrid/utils.hpp"

namespace emdgrid {

namespace detail {

/// Runs a LEMON min-cost flow on a complete bipartite graph.
///
/// Every map is taken as a template parameter so that the caller can pass
/// maps that compute their values rather than store them; nothing here
/// materialises an arc-indexed container. `on_flow(arc, flow)` is invoked for
/// each arc carrying positive flow, and only when `collect_flows` is set —
/// enumerating the arcs is O(n·m), which is wasted work when the caller only
/// wants the cost.
///
/// Reports the backend's exit status through `log` before throwing on
/// anything but an optimum, matching mcf_lemon_l1's behaviour.
template <typename Solver, typename Graph, typename CapMap, typename CostMap,
          typename SupplyMap, typename OnFlow>
int64_t run_lemon_bipartite_mcf(const Graph& graph, const CapMap& capacity,
                                const CostMap& cost, const SupplyMap& supply,
                                SolverLog* log, bool collect_flows,
                                OnFlow&& on_flow) {
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
  if (collect_flows) {
    for (typename Graph::ArcIt a(graph); a != lemon::INVALID; ++a) {
      const int64_t f = mcf.flow(a);
      if (f > 0) {
        on_flow(a, f);
      }
    }
  }
  return mcf.totalCost();
}

}  // namespace detail

/// Bipartite Earth Mover's Distance for grid histograms, solved with LEMON.
///
/// This is the textbook transportation formulation: one node per bin with
/// leftover supply, one per bin with leftover demand, and an arc between
/// every such pair carrying the ground-metric cost. It makes no use of the
/// grid's adjacency structure, which is exactly the point — it is the
/// reference any structure-exploiting solver (emd_l1, mcf_lemon_l1,
/// mcf_dpartion) should agree with, and it accepts separable ground metrics
/// the grid-graph formulation cannot express.
///
/// The formulation follows POT's `EMD_wrap_lazy`
/// (https://github.com/PythonOT/POT/blob/master/ot/lp/EMD_wrapper.cpp), with
/// LEMON's NetworkSimplex or CostScaling in place of POT's bundled simplex,
/// and with the point coordinates coming from the grid layout instead of from
/// a caller-supplied array.
///
/// **What is lazy.** The `n·m` arc costs are never assembled into a matrix on
/// this side of the call. The arcs are implicit (detail::FullBipartiteDigraph),
/// the cost map (detail::LazyArcMap) evaluates `cost_fn` from the two bins'
/// grid coordinates on demand, and the capacity is a lemon::ConstMap. Stating
/// the problem costs O(n + m), and building the graph is O(1).
///
/// **What is not lazy, and it dominates.** LEMON's NetworkSimplex copies
/// everything into its own arc-indexed arrays as soon as `reset()` runs, and
/// offers no way to opt out: `_source` and `_target` (int), `_lower`,
/// `_upper`, `_cap`, `_cost` and `_flow` (Value/Cost) and `_state` (char), plus
/// the `ArcMap<int> _arc_id` it indexes them through. With Value = Cost =
/// int64_t that is 53 bytes for every one of the `n·m` arcs no matter what the
/// cost map does. Measured on a 20^3 grid under a squared Euclidean metric —
/// 8000 x 8000 arcs, since that metric gets no self-mass extraction — peak RSS
/// is 3.4 GB, i.e. those 53 bytes/arc plus ~20 MB of everything else. So
/// laziness here saves the caller's cost matrix and nothing more; it does not
/// make the solve sub-quadratic in memory.
///
/// POT's `EMD_wrap_lazy` does not pay this. Its forked simplex has storage
/// modes (ArtificialArcCosts, SparseArcFlows, PackedArcStates) that keep
/// essentially nothing per arc, and the same problem fits in 0.12 GB there
/// against 3.4 GB here — while running ~4-5x slower, because it recomputes a
/// distance at every pricing step where LEMON reads a packed array. This port
/// buys LEMON's speed and its CostScaling option; it does not buy POT's memory
/// profile, and no amount of work on the graph representation will, because
/// the cost is proportional to the arc count LEMON is handed.
///
/// **So do not reach for this solver when memory-bound.** Every cost it
/// accepts is separable, and for a separable cost mcf_dpartion solves the
/// identical problem on a layered DAG with `n · sum_k shape[k]` arcs rather
/// than `n^2`: 480k instead of 64M on that same 20^3 grid, reaching a
/// bit-identical optimum in 0.62 s and 0.078 GB against 5.0 s and 3.4 GB here.
/// The saving is a factor `s^(Dim-1)/Dim` for a side-`s` cube, so it widens as
/// the grid grows. This solver exists to be the independent check on that one.
///
/// **Self-mass.** When `CostFn::extract_self_mass` is true — as for L1Cost,
/// and for any ground metric obeying the triangle inequality — the mass
/// `min(h1[i], h2[i])` provably never has to move, so only the residual is
/// routed and the diagonal is re-attached to the reported plan. This shrinks
/// both sides of the bipartite graph considerably on similar histograms.
///
/// @tparam Dim        Grid dimensionality (>= 1).
/// @tparam Scalar     Input histogram scalar type.
/// @tparam CompScalar Scalar type used for computation (default: double).
/// @tparam CostFn     Separable axis cost functor, `(axis, a, b) -> int64_t`.
///
/// @param h1       Source histogram, unit mass.
/// @param h2       Target histogram, unit mass.
/// @param cost_fn  Per-axis ground cost; the bin-to-bin cost is its sum over
///                 the axes.
/// @param algo     LEMON backend to use.
/// @param plan     Optional sparse transport plan output.
/// @param scale    Quantization scale for the supplies.
/// @param mass_tol Tolerance on each histogram's deviation from unit mass.
template <std::size_t Dim, std::floating_point Scalar,
          std::floating_point CompScalar = double, typename CostFn>  // NOLINT(*)
  requires(Dim >= 1 && detail::ValidLemonCostFn<CostFn>)              // NOLINT(*)
[[nodiscard]] CompScalar emd_lemon(
    const GridDataView<Dim, Scalar>& h1, const GridDataView<Dim, Scalar>& h2,
    CostFn&& cost_fn,
    McfLemonAlgorithm algo = McfLemonAlgorithm::NetworkSimplex,
    SparseTransportPlanPtr<CompScalar> plan = nullptr,
    CompScalar scale = static_cast<CompScalar>(1e6),
    CompScalar mass_tol = default_mass_tolerance<CompScalar>) {
  detail::SolverLog log(
      "emd_lemon",
      fmt::format("Dim={}, algo={}, scale={}", Dim,
                  algo == McfLemonAlgorithm::NetworkSimplex ? "NetworkSimplex"
                                                            : "CostScaling",
                  scale));

  detail::validate_unit_mass_pair(h1, h2, mass_tol);

  const auto& layout = h1.layout();
  const std::size_t n_nodes = layout.node_count();

  if (plan != nullptr) {
    plan->source.clear();
    plan->target.clear();
    plan->flow.clear();
  }

  constexpr bool do_extract_self_mass =
      detail::should_extract_self_mass_v<CostFn>;

  // Quantize each side's residual mass separately so that the supplies stay
  // nonnegative on the source side and the demands on the target side, as the
  // bipartite formulation requires. quantize_net_supply cannot be reused here
  // because it collapses the two histograms into a single signed net supply.
  std::vector<int64_t> supply1(n_nodes);
  std::vector<int64_t> supply2(n_nodes);
  detail::CumulativeQuantizer<CompScalar> quantizer1(scale);
  detail::CumulativeQuantizer<CompScalar> quantizer2(scale);

  for (std::size_t i = 0; i < n_nodes; ++i) {
    const CompScalar v1 = static_cast<CompScalar>(h1.data()[i]);
    const CompScalar v2 = static_cast<CompScalar>(h2.data()[i]);
    const CompScalar self_mass =
        do_extract_self_mass ? std::min(v1, v2) : CompScalar{0};
    supply1[i] = quantizer1.push(v1 - self_mass);
    supply2[i] = quantizer2.push(v2 - self_mass);
  }

  // The two sides round independently and so may end up a quantum apart.
  // Charging the difference to the largest source supply keeps it
  // nonnegative: that entry is at least the average, and the drift is at most
  // one quantum. When the totals are both zero the drift is zero too, so the
  // all-zero case is left untouched.
  detail::absorb_quantization_drift(
      supply1, quantizer1.scaled_total() - quantizer2.scaled_total());

  if (plan != nullptr && do_extract_self_mass) {
    detail::emit_self_mass(h1, h2, plan);
  }

  // Keep only the bins that still have something to send or receive, and
  // cache their grid coordinates for the lazy cost map.
  using Coords = GridLayout<Dim>::Coordinates;
  std::vector<uint32_t> bin_a;
  std::vector<uint32_t> bin_b;
  std::vector<int64_t> sup_a;
  std::vector<int64_t> sup_b;
  std::vector<Coords> coords_a;
  std::vector<Coords> coords_b;

  for (std::size_t i = 0; i < n_nodes; ++i) {
    const auto node = static_cast<std::ptrdiff_t>(i);
    if (supply1[i] > 0) {
      bin_a.push_back(static_cast<uint32_t>(i));
      sup_a.push_back(supply1[i]);
      coords_a.push_back(layout.coordinates(node));
    }
    if (supply2[i] > 0) {
      bin_b.push_back(static_cast<uint32_t>(i));
      sup_b.push_back(supply2[i]);
      coords_b.push_back(layout.coordinates(node));
    }
  }

  const auto n_src = static_cast<int>(bin_a.size());
  const auto n_tgt = static_cast<int>(bin_b.size());
  log.phase("supply setup",
            fmt::format("bins={}, sources={}, sinks={}", n_nodes, n_src,
                        n_tgt));

  if (n_src == 0 || n_tgt == 0) {
    spdlog::info(
        "emd_lemon: histograms agree after quantization, nothing to "
        "transport");
    log.finish(CompScalar{0});
    return static_cast<CompScalar>(0.0);
  }

  using Graph = detail::FullBipartiteDigraph;
  using Arc = Graph::Arc;

  const Graph graph(n_src, n_tgt);

  Graph::NodeMap<int64_t> supply_map(graph);
  for (int i = 0; i < n_src; ++i) {
    supply_map[Graph::source_node(i)] = sup_a[static_cast<std::size_t>(i)];
  }
  for (int j = 0; j < n_tgt; ++j) {
    supply_map[graph.target_node(j)] = -sup_b[static_cast<std::size_t>(j)];
  }

  // A constant capacity map: no arc can carry more than the total mass, so
  // this never binds, and it costs nothing to represent.
  const lemon::ConstMap<Arc, int64_t> capacity(
      detail::total_positive_supply(sup_a));

  const auto cost_map = detail::make_lazy_arc_map<Graph, int64_t>(
      [&graph, &coords_a, &coords_b, &cost_fn](const Arc& arc) -> int64_t {
        const auto& ca =
            coords_a[static_cast<std::size_t>(graph.source_index(arc))];
        const auto& cb =
            coords_b[static_cast<std::size_t>(graph.target_index(arc))];
        int64_t total = 0;
        for (std::size_t k = 0; k < Dim; ++k) {
          total += cost_fn(k, static_cast<std::size_t>(ca[k]),
                           static_cast<std::size_t>(cb[k]));
        }
        return total;
      });

  log.phase("graph construction",
            fmt::format("nodes={}, implicit arcs={}", graph.nodeNum(),
                        graph.arcNum()));

  const auto emit = [&](const Arc& arc, int64_t flow) {
    plan->source.push_back(
        bin_a[static_cast<std::size_t>(graph.source_index(arc))]);
    plan->target.push_back(
        bin_b[static_cast<std::size_t>(graph.target_index(arc))]);
    plan->flow.push_back(static_cast<CompScalar>(flow) / scale);
  };

  int64_t raw_optimal_cost = 0;
  if (algo == McfLemonAlgorithm::NetworkSimplex) {
    using Solver = lemon::NetworkSimplex<Graph, int64_t, int64_t>;
    raw_optimal_cost = detail::run_lemon_bipartite_mcf<Solver>(
        graph, capacity, cost_map, supply_map, &log, plan != nullptr, emit);
  } else {
    using Solver = lemon::CostScaling<Graph, int64_t, int64_t>;
    raw_optimal_cost = detail::run_lemon_bipartite_mcf<Solver>(
        graph, capacity, cost_map, supply_map, &log, plan != nullptr, emit);
  }

  const CompScalar total_cost =
      static_cast<CompScalar>(raw_optimal_cost) / scale;

  if (plan != nullptr) {
    log.phase("plan extraction",
              fmt::format("entries={}", plan->flow.size()));
  }

  log.finish(total_cost);
  return total_cost;
}

/// Overload of emd_lemon accepting a GroundMetric enum.
template <std::size_t Dim, std::floating_point Scalar,
          std::floating_point CompScalar = double>  // NOLINT(*)
  requires(Dim >= 1)                                 // NOLINT(*)
[[nodiscard]] CompScalar emd_lemon(
    const GridDataView<Dim, Scalar>& h1, const GridDataView<Dim, Scalar>& h2,
    GroundMetric metric,
    McfLemonAlgorithm algo = McfLemonAlgorithm::NetworkSimplex,
    SparseTransportPlanPtr<CompScalar> plan = nullptr,
    CompScalar scale = static_cast<CompScalar>(1e6),
    CompScalar mass_tol = default_mass_tolerance<CompScalar>) {
  if (metric == GroundMetric::SqEuclidean) {
    return emd_lemon<Dim, Scalar, CompScalar>(h1, h2, SqEuclideanCost{}, algo,
                                              plan, scale, mass_tol);
  }
  return emd_lemon<Dim, Scalar, CompScalar>(h1, h2, L1Cost{}, algo, plan,
                                            scale, mass_tol);
}

/// Overload of emd_lemon defaulting to GroundMetric::L1.
template <std::size_t Dim, std::floating_point Scalar,
          std::floating_point CompScalar = double>  // NOLINT(*)
  requires(Dim >= 1)                                 // NOLINT(*)
[[nodiscard]] CompScalar emd_lemon(
    const GridDataView<Dim, Scalar>& h1, const GridDataView<Dim, Scalar>& h2,
    McfLemonAlgorithm algo = McfLemonAlgorithm::NetworkSimplex,
    SparseTransportPlanPtr<CompScalar> plan = nullptr,
    CompScalar scale = static_cast<CompScalar>(1e6),
    CompScalar mass_tol = default_mass_tolerance<CompScalar>) {
  return emd_lemon<Dim, Scalar, CompScalar>(h1, h2, GroundMetric::L1, algo,
                                            plan, scale, mass_tol);
}

}  // namespace emdgrid
