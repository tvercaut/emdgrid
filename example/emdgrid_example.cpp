#include <concepts>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <CLI/CLI.hpp>  // NOLINT(build/include_order)
#include <spdlog/sinks/stdout_color_sinks.h>  // NOLINT(build/include_order)
#include <spdlog/spdlog.h>                    // NOLINT(build/include_order)

#include "emdgrid/emd_l1.hpp"
#include "emdgrid/emd_lemon.hpp"
#include "emdgrid/emd_potlemon.hpp"
#include "emdgrid/emdgrid.hpp"
#include "emdgrid/greedy_emd_l1.hpp"
#include "emdgrid/knothe_rosenblatt.hpp"
#include "emdgrid/mcf_l1.hpp"
#include "emdgrid/mcf_lemon_l1.hpp"
#include "emdgrid/mcf_potlemon_l1.hpp"
#include "emdgrid/opencv_emd.hpp"
#include "emdgrid/utils.hpp"

namespace {

/// Options shared by every solver run, parsed once from the command line.
struct RunOptions {
  std::size_t dim = 10;
  bool need_plan = false;
  bool diagnostics = false;
  std::string solver = "all";
  emdgrid::GroundMetric kr_metric = emdgrid::GroundMetric::L1;
  std::vector<std::size_t> dimension_order;
  unsigned int seed1 = 42;
  unsigned int seed2 = 1337;
  int max_iter = 500000;
};

/// Tolerance on the plan total mass, loosened for low-precision computations.
template <std::floating_point CompScalar>
constexpr CompScalar plan_sum_tolerance =
    std::is_same_v<CompScalar, float> ? static_cast<CompScalar>(1e-3)
                                      : static_cast<CompScalar>(1e-5);

template <std::size_t Dim, std::floating_point Scalar,
          std::floating_point CompScalar>
void run_diagnostics(
    const emdgrid::GridLayout<Dim>& layout,
    const emdgrid::GridDataView<Dim, Scalar>& h1,
    const emdgrid::GridDataView<Dim, Scalar>& h2,
    const emdgrid::SparseTransportPlan<CompScalar>& plan,
    emdgrid::GroundMetric metric = emdgrid::GroundMetric::L1,
    CompScalar reported_cost = static_cast<CompScalar>(-1)) {
  const std::size_t n_nodes = layout.node_count();
  std::vector<CompScalar> src_marginal(n_nodes, CompScalar{0});
  std::vector<CompScalar> tgt_marginal(n_nodes, CompScalar{0});

  CompScalar dot_product_cost{0};
  CompScalar plan_total_sum{0};
  bool all_positive = true;

  for (std::size_t k = 0; k < plan.flow.size(); ++k) {
    const std::size_t src = plan.source[k];
    const std::size_t tgt = plan.target[k];
    const CompScalar f = plan.flow[k];

    if (f <= CompScalar{0}) {
      all_positive = false;
    }
    plan_total_sum += f;

    src_marginal[src] += f;
    tgt_marginal[tgt] += f;

    const auto c_src = layout.coordinates(static_cast<std::ptrdiff_t>(src));
    const auto c_tgt = layout.coordinates(static_cast<std::ptrdiff_t>(tgt));

    CompScalar dist_unit{0};
    for (std::size_t a = 0; a < Dim; ++a) {
      const CompScalar diff = static_cast<CompScalar>(c_src[a] - c_tgt[a]);
      dist_unit += (metric == emdgrid::GroundMetric::L1) ? std::abs(diff)
                                                         : (diff * diff);
    }
    dot_product_cost += f * dist_unit;
  }

  CompScalar max_src_residual{0};
  CompScalar max_tgt_residual{0};
  for (std::size_t i = 0; i < n_nodes; ++i) {
    max_src_residual = std::max(
        max_src_residual,
        std::abs(src_marginal[i] - static_cast<CompScalar>(h1.data()[i])));
    max_tgt_residual = std::max(
        max_tgt_residual,
        std::abs(tgt_marginal[i] - static_cast<CompScalar>(h2.data()[i])));
  }

  std::cout << "  [Diagnostics]\n";
  if (reported_cost >= CompScalar{0}) {
    std::cout << "    Cost difference (|reported - dot product|): "
              << std::abs(reported_cost - dot_product_cost) << '\n';
  } else {
    std::cout << "    Recomputed cost (dot product with C): "
              << dot_product_cost << '\n';
  }
  const bool sum_is_valid = std::abs(plan_total_sum - CompScalar{1}) <
                            plan_sum_tolerance<CompScalar>;
  std::cout << "    Plan total sum: " << plan_total_sum
            << " (valid = " << (sum_is_valid ? "yes" : "NO") << ")\n";
  std::cout << "    All flows positive: " << (all_positive ? "yes" : "NO")
            << '\n';
  std::cout << "    Max src marginal residual: " << max_src_residual << '\n';
  std::cout << "    Max tgt marginal residual: " << max_tgt_residual << '\n';
}

/// Reports one solver result and, when requested, its plan diagnostics.
template <std::size_t Dim, std::floating_point Scalar,
          std::floating_point CompScalar>
void report(const char* title, CompScalar dist, double elapsed_ms,
            const RunOptions& opts, const emdgrid::GridLayout<Dim>& layout,
            const emdgrid::GridDataView<Dim, Scalar>& h1,
            const emdgrid::GridDataView<Dim, Scalar>& h2,
            const emdgrid::SparseTransportPlan<CompScalar>& plan,
            emdgrid::GroundMetric metric = emdgrid::GroundMetric::L1) {
  std::cout << "\n--- " << title << " ---\n";
  std::cout << "Distance: " << dist << '\n';
  std::cout << "Computation time: " << elapsed_ms << " ms\n";
  if (opts.need_plan) {
    std::cout << "Transport plan flow entries: " << plan.source.size() << '\n';
    if (opts.diagnostics) {
      run_diagnostics(layout, h1, h2, plan, metric, dist);
    }
  }
}

/// Runs the selected solvers on a 3-D grid with computations in CompScalar.
template <std::floating_point CompScalar>
void run_benchmarks(const RunOptions& opts) {
  using Plan = emdgrid::SparseTransportPlan<CompScalar>;

  const emdgrid::GridLayout<3> layout({opts.dim, opts.dim, opts.dim});
  const std::size_t n_bins = layout.node_count();

  std::cout << "Grid shape: " << opts.dim << 'x' << opts.dim << 'x' << opts.dim
            << " (" << n_bins << " bins)\n";
  std::cout << "Compute transport plan: " << (opts.need_plan ? "yes" : "no")
            << '\n';

  const std::vector<double> h1_data =
      emdgrid::generate_random_histogram<double>(n_bins, opts.seed1);
  const std::vector<double> h2_data =
      emdgrid::generate_random_histogram<double>(n_bins, opts.seed2);

  const emdgrid::GridDataView<3, double> h1(layout, std::span(h1_data));
  const emdgrid::GridDataView<3, double> h2(layout, std::span(h2_data));

  std::cout << std::fixed << std::setprecision(6);

  const std::string& solver = opts.solver;
  const bool run_emd_l1 = (solver == "all" || solver == "emd_l1" ||
                           solver == "exact");
  const bool run_mcf_l1 = (solver == "all" || solver == "mcf_l1" ||
                           solver == "mcf");
  const bool run_mcf_lemon_ns = (solver == "all" || solver == "mcf_lemon" ||
                                 solver == "mcf_lemon_ns");
  const bool run_mcf_lemon_cs = (solver == "all" || solver == "mcf_lemon" ||
                                 solver == "mcf_lemon_cs");
  const bool run_mcf_potlemon = (solver == "all" || solver == "mcf_potlemon" ||
                                  solver == "potlemon");
  const bool run_dpartion = (solver == "all" || solver == "dpartion" ||
                             solver == "mcf_dpartion");
  const bool run_opencv_emd = (solver == "all_extended" ||
                               solver == "opencv_emd" || solver == "opencv");
  const bool run_emd_lemon = (solver == "all_extended" ||
                              solver == "emd_lemon" ||
                              solver == "lemon_bipartite");
  const bool run_emd_potlemon = (solver == "all_extended" ||
                                 solver == "emd_potlemon" ||
                                 solver == "potlemon_bipartite");
  const bool run_greedy = (solver == "all" || solver == "greedy");
  const bool run_kr = (solver == "all" || solver == "kr" ||
                       solver == "knothe_rosenblatt");

  if (run_emd_l1) {
    Plan plan;
    const emdgrid::Timer timer;
    const CompScalar dist = emdgrid::emd_l1<3, double, CompScalar>(
        h1, h2, opts.need_plan ? &plan : nullptr, opts.max_iter);
    report("Exact EMD-L1 (Ling & Okada)", dist, timer.elapsed_milliseconds(),
           opts, layout, h1, h2, plan);
  }

  if (run_mcf_l1) {
    Plan plan;
    const emdgrid::Timer timer;
    const CompScalar dist = emdgrid::mcf_l1<3, double, CompScalar>(
        h1, h2, opts.need_plan ? &plan : nullptr);
    report("Min-Cost Flow EMD-L1 (OR-Tools)", dist,
           timer.elapsed_milliseconds(), opts, layout, h1, h2, plan);
  }

  if (run_mcf_lemon_ns) {
    Plan plan;
    const emdgrid::Timer timer;
    const CompScalar dist = emdgrid::mcf_lemon_l1<3, double, CompScalar>(
        h1, h2, emdgrid::McfLemonAlgorithm::NetworkSimplex,
        opts.need_plan ? &plan : nullptr);
    report("Min-Cost Flow EMD-L1 (LEMON NetworkSimplex)", dist,
           timer.elapsed_milliseconds(), opts, layout, h1, h2, plan);
  }

  if (run_mcf_lemon_cs) {
    Plan plan;
    const emdgrid::Timer timer;
    const CompScalar dist = emdgrid::mcf_lemon_l1<3, double, CompScalar>(
        h1, h2, emdgrid::McfLemonAlgorithm::CostScaling,
        opts.need_plan ? &plan : nullptr);
    report("Min-Cost Flow EMD-L1 (LEMON CostScaling)", dist,
           timer.elapsed_milliseconds(), opts, layout, h1, h2, plan);
  }

  if (run_mcf_potlemon) {
    Plan plan;
    const emdgrid::Timer timer;
    const CompScalar dist = emdgrid::mcf_potlemon_l1<3, double, CompScalar>(
        h1, h2, opts.need_plan ? &plan : nullptr);
    report("Min-Cost Flow EMD-L1 (POT NetworkSimplex)", dist,
           timer.elapsed_milliseconds(), opts, layout, h1, h2, plan);
  }

  if (run_dpartion) {
    std::cout << "\n--- dpartion (Auricchio et al. 2018) ---\n";

    const struct Variant {
      const char* name;
      emdgrid::GroundMetric metric;
      emdgrid::McfLemonAlgorithm algo;
    } variants[] = {
        {"NetworkSimplex + L1", emdgrid::GroundMetric::L1,
         emdgrid::McfLemonAlgorithm::NetworkSimplex},
        {"NetworkSimplex + SqEuclidean", emdgrid::GroundMetric::SqEuclidean,
         emdgrid::McfLemonAlgorithm::NetworkSimplex},
        {"CostScaling + L1", emdgrid::GroundMetric::L1,
         emdgrid::McfLemonAlgorithm::CostScaling},
        {"CostScaling + SqEuclidean", emdgrid::GroundMetric::SqEuclidean,
         emdgrid::McfLemonAlgorithm::CostScaling},
    };

    for (const auto& [name, metric, algo] : variants) {
      Plan plan;
      const emdgrid::Timer timer;
      const CompScalar dist = emdgrid::mcf_dpartion<3, double, CompScalar>(
          h1, h2, metric, algo, opts.need_plan ? &plan : nullptr);
      const double elapsed_ms = timer.elapsed_milliseconds();

      std::cout << "Variant [" << name << "]:\n";
      std::cout << "  Distance: " << dist << '\n';
      std::cout << "  Computation time: " << elapsed_ms << " ms\n";
      if (opts.need_plan) {
        std::cout << "  Transport plan flow entries: " << plan.source.size()
                  << '\n';
        if (opts.diagnostics) {
          run_diagnostics(layout, h1, h2, plan, metric, dist);
        }
      }
    }
  }

  if (run_opencv_emd) {
    Plan plan;
    const emdgrid::Timer timer;
    const CompScalar dist = emdgrid::opencv_emd<3, double, CompScalar>(
        h1, h2, emdgrid::GroundMetric::L1, opts.need_plan ? &plan : nullptr);
    report("OpenCV EMD (Rubner transportation simplex, L1)", dist,
           timer.elapsed_milliseconds(), opts, layout, h1, h2, plan);
  }

  if (run_emd_lemon) {
    std::cout << "\n--- emd_lemon (LEMON bipartite, lazy costs) ---\n";

    const struct LemonVariant {
      const char* name;
      emdgrid::GroundMetric metric;
      emdgrid::McfLemonAlgorithm algo;
    } lemon_variants[] = {
        {"NetworkSimplex + L1", emdgrid::GroundMetric::L1,
         emdgrid::McfLemonAlgorithm::NetworkSimplex},
        {"NetworkSimplex + SqEuclidean", emdgrid::GroundMetric::SqEuclidean,
         emdgrid::McfLemonAlgorithm::NetworkSimplex},
        {"CostScaling + L1", emdgrid::GroundMetric::L1,
         emdgrid::McfLemonAlgorithm::CostScaling},
        {"CostScaling + SqEuclidean", emdgrid::GroundMetric::SqEuclidean,
         emdgrid::McfLemonAlgorithm::CostScaling},
    };

    for (const auto& [name, metric, algo] : lemon_variants) {
      Plan plan;
      const emdgrid::Timer timer;
      const CompScalar dist = emdgrid::emd_lemon<3, double, CompScalar>(
          h1, h2, metric, algo, opts.need_plan ? &plan : nullptr);
      const double elapsed_ms = timer.elapsed_milliseconds();

      std::cout << "Variant [" << name << "]:\n";
      std::cout << "  Distance: " << dist << '\n';
      std::cout << "  Computation time: " << elapsed_ms << " ms\n";
      if (opts.need_plan) {
        std::cout << "  Transport plan flow entries: " << plan.source.size()
                  << '\n';
        if (opts.diagnostics) {
          run_diagnostics(layout, h1, h2, plan, metric, dist);
        }
      }
    }
  }

  if (run_emd_potlemon) {
    std::cout << "\n--- emd_potlemon (POT bipartite, fully lazy) ---\n";

    const struct PotVariant {
      const char* name;
      emdgrid::GroundMetric metric;
    } pot_variants[] = {
        {"L1", emdgrid::GroundMetric::L1},
        {"SqEuclidean", emdgrid::GroundMetric::SqEuclidean},
    };

    for (const auto& [name, metric] : pot_variants) {
      Plan plan;
      const emdgrid::Timer timer;
      const CompScalar dist = emdgrid::emd_potlemon<3, double, CompScalar>(
          h1, h2, metric, opts.need_plan ? &plan : nullptr);
      const double elapsed_ms = timer.elapsed_milliseconds();

      std::cout << "Variant [" << name << "]:\n";
      std::cout << "  Distance: " << dist << '\n';
      std::cout << "  Computation time: " << elapsed_ms << " ms\n";
      if (opts.need_plan) {
        std::cout << "  Transport plan flow entries: " << plan.source.size()
                  << '\n';
        if (opts.diagnostics) {
          run_diagnostics(layout, h1, h2, plan, metric, dist);
        }
      }
    }
  }

  if (run_greedy) {
    Plan plan;
    const emdgrid::Timer timer;
    const CompScalar dist =
        emdgrid::greedy_emd_l1_approx<3, double, CompScalar>(
            h1, h2, opts.need_plan ? &plan : nullptr);
    report("Greedy Basic Feasible EMD-L1", dist,
           timer.elapsed_milliseconds(), opts, layout, h1, h2, plan);
  }

  if (run_kr) {
    Plan plan;
    const emdgrid::Timer timer;
    const CompScalar dist = emdgrid::knothe_rosenblatt<3, double, CompScalar>(
        h1, h2, opts.kr_metric, opts.dimension_order,
        opts.need_plan ? &plan : nullptr);
    const char* title =
        (opts.kr_metric == emdgrid::GroundMetric::L1)
            ? "Knothe-Rosenblatt (L1)"
            : "Knothe-Rosenblatt (SqEuclidean)";
    report(title, dist, timer.elapsed_milliseconds(), opts, layout, h1, h2,
           plan, opts.kr_metric);
  }
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"emdgrid CLI example benchmark tool"};

  RunOptions opts;
  bool compute_plan = false;
  std::string kr_metric_str = "l1";
  std::string comp_scalar = "double";
  bool verbose = false;

  app.add_option("-d,--dim", opts.dim,
                 "Grid extent along each 3D axis (default: 10)");
  app.add_flag("-p,--plan", compute_plan, "Compute sparse transport plan");
  app.add_flag("--diagnostics", opts.diagnostics,
               "Run plan diagnostics (forces transport plan calculation)");
  app.add_option("-s,--solver", opts.solver,
                 "Solver to run: 'emd_l1', 'mcf_l1', 'mcf_lemon_ns', "
                 "'mcf_lemon_cs', 'mcf_potlemon', 'dpartion', 'opencv_emd', "
                 "'emd_lemon', 'emd_potlemon', 'greedy', 'kr', 'all' "
                 "(default, the grid-structure solvers), or 'all_extended' "
                 "(the dense bipartite solvers 'opencv_emd', 'emd_lemon' and "
                 "'emd_potlemon', which are slow on large grids)");
  app.add_option("-m,--metric", kr_metric_str,
                 "Knothe-Rosenblatt metric: 'l1' or 'sqeuclidean' "
                 "(default: 'l1')");
  app.add_option("-c,--comp-scalar", comp_scalar,
                 "Computation scalar type: 'double' (default) or 'float'. "
                 "Histograms are always generated in double; this selects "
                 "the CompScalar the solvers compute and report in")
      ->check(CLI::IsMember({"double", "float"}));
  app.add_option("-o,--order", opts.dimension_order,
                 "Knothe-Rosenblatt dimension order permutation (e.g., 0 1 2)");
  app.add_option("--seed1", opts.seed1,
                 "Random seed for first histogram (default: 42)");
  app.add_option("--seed2", opts.seed2,
                 "Random seed for second histogram (default: 1337)");
  app.add_option("--max-iter", opts.max_iter,
                 "Maximum network-simplex iterations for EMD-L1 "
                 "(default: 500000)");
  app.add_flag("-v,--verbose", verbose, "Enable verbose logging");

  CLI11_PARSE(app, argc, argv);

  opts.need_plan = compute_plan || opts.diagnostics;
  if (kr_metric_str == "sqeuclidean" || kr_metric_str == "squared_euclidean") {
    opts.kr_metric = emdgrid::GroundMetric::SqEuclidean;
  }

  // Configure spdlog stdout logger
  auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  auto logger = std::make_shared<spdlog::logger>("emdgrid", stdout_sink);
  spdlog::set_default_logger(logger);
  spdlog::set_level(verbose ? spdlog::level::info : spdlog::level::off);

  std::cout << "emdgrid version " << emdgrid::version() << '\n';
  std::cout << "Computation scalar type: " << comp_scalar << '\n';

  if (comp_scalar == "float") {
    run_benchmarks<float>(opts);
  } else {
    run_benchmarks<double>(opts);
  }

  return 0;
}
