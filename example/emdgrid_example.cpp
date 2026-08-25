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
#include "emdgrid/emdgrid.hpp"
#include "emdgrid/greedy_emd_l1.hpp"
#include "emdgrid/knothe_rosenblatt.hpp"
#include "emdgrid/utils.hpp"

int main(int argc, char** argv) {
  CLI::App app{"emdgrid CLI example benchmark tool"};

  std::size_t dim = 10;
  bool compute_plan = false;
  std::string solver = "all";
  std::string kr_metric_str = "l1";
  std::vector<std::size_t> dimension_order;
  unsigned int seed1 = 42;
  unsigned int seed2 = 1337;
  int max_iter = 500000;
  bool verbose = false;

  app.add_option("-d,--dim", dim,
                 "Grid extent along each 3D axis (default: 10)");
  app.add_flag("-p,--plan", compute_plan, "Compute sparse transport plan");
  app.add_option("-s,--solver", solver,
                 "Solver to run: 'emd_l1', 'greedy', 'kr', or 'all' "
                 "(default: 'all')");
  app.add_option("-m,--metric", kr_metric_str,
                 "Knothe-Rosenblatt metric: 'l1' or 'sqeuclidean' "
                 "(default: 'l1')");
  app.add_option("-o,--order", dimension_order,
                 "Knothe-Rosenblatt dimension order permutation (e.g., 0 1 2)");
  app.add_option("--seed1", seed1,
                 "Random seed for first histogram (default: 42)");
  app.add_option("--seed2", seed2,
                 "Random seed for second histogram (default: 1337)");
  app.add_option("--max-iter", max_iter,
                 "Maximum network-simplex iterations for EMD-L1 "
                 "(default: 500000)");
  app.add_flag("-v,--verbose", verbose, "Enable verbose logging");

  CLI11_PARSE(app, argc, argv);

  // Configure spdlog stdout logger
  auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  auto logger = std::make_shared<spdlog::logger>("emdgrid", stdout_sink);
  spdlog::set_default_logger(logger);
  spdlog::set_level(verbose ? spdlog::level::info : spdlog::level::off);

  std::cout << "emdgrid version " << emdgrid::version() << '\n';

  const emdgrid::GridLayout<3> layout({dim, dim, dim});
  const std::size_t n_bins = layout.node_count();

  std::cout << "Grid shape: " << dim << 'x' << dim << 'x' << dim << " ("
            << n_bins << " bins)\n";
  std::cout << "Compute transport plan: " << (compute_plan ? "yes" : "no")
            << '\n';

  const std::vector<double> h1_data =
      emdgrid::generate_random_histogram<double>(n_bins, seed1);
  const std::vector<double> h2_data =
      emdgrid::generate_random_histogram<double>(n_bins, seed2);

  const emdgrid::GridDataView<3, double> h1(layout, std::span(h1_data));
  const emdgrid::GridDataView<3, double> h2(layout, std::span(h2_data));

  std::cout << std::fixed << std::setprecision(6);

  const bool run_emd_l1 = (solver == "all" || solver == "emd_l1" ||
                           solver == "exact");
  const bool run_greedy = (solver == "all" || solver == "greedy");
  const bool run_kr = (solver == "all" || solver == "kr" ||
                       solver == "knothe_rosenblatt");

  if (run_emd_l1) {
    emdgrid::SparseTransportPlan plan;
    const emdgrid::Timer timer;
    const double dist = emdgrid::emd_l1(h1, h2, compute_plan ? &plan : nullptr,
                                         max_iter);
    const double elapsed_ms = timer.elapsed_milliseconds();

    std::cout << "\n--- Exact EMD-L1 ---\n";
    std::cout << "Distance: " << dist << '\n';
    std::cout << "Computation time: " << elapsed_ms << " ms\n";
    if (compute_plan) {
      std::cout << "Transport plan flow entries: " << plan.source.size()
                << '\n';
    }
  }

  if (run_greedy) {
    emdgrid::SparseTransportPlan plan;
    const emdgrid::Timer timer;
    const double dist =
        emdgrid::greedy_emd_l1_approx(h1, h2, compute_plan ? &plan : nullptr);
    const double elapsed_ms = timer.elapsed_milliseconds();

    std::cout << "\n--- Greedy Basic Feasible EMD-L1 ---\n";
    std::cout << "Approx distance: " << dist << '\n';
    std::cout << "Computation time: " << elapsed_ms << " ms\n";
    if (compute_plan) {
      std::cout << "Transport plan flow entries: " << plan.source.size()
                << '\n';
    }
  }

  if (run_kr) {
    emdgrid::GroundMetric metric = emdgrid::GroundMetric::L1;
    if (kr_metric_str == "sqeuclidean" ||
        kr_metric_str == "squared_euclidean") {
      metric = emdgrid::GroundMetric::SqEuclidean;
    }

    emdgrid::SparseTransportPlan plan;
    const emdgrid::Timer timer;
    const double dist = emdgrid::knothe_rosenblatt(
        h1, h2, metric, dimension_order, compute_plan ? &plan : nullptr);
    const double elapsed_ms = timer.elapsed_milliseconds();

    std::cout << "\n--- Knothe-Rosenblatt ("
              << (metric == emdgrid::GroundMetric::L1 ? "L1" : "SqEuclidean")
              << ") ---\n";
    std::cout << "Approx distance: " << dist << '\n';
    std::cout << "Computation time: " << elapsed_ms << " ms\n";
    if (compute_plan) {
      std::cout << "Transport plan flow entries: " << plan.source.size()
                << '\n';
    }
  }

  return 0;
}
