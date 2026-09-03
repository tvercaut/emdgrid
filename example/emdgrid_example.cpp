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
#include "emdgrid/mcf_l1.hpp"
#include "emdgrid/mcf_lemon_l1.hpp"
#include "emdgrid/utils.hpp"

template <std::size_t Dim>
void run_diagnostics(
    const emdgrid::GridLayout<Dim>& layout,
    const emdgrid::GridDataView<Dim, double>& h1,
    const emdgrid::GridDataView<Dim, double>& h2,
    const emdgrid::SparseTransportPlan& plan,
    emdgrid::GroundMetric metric = emdgrid::GroundMetric::L1) {
  const std::size_t n_nodes = layout.node_count();
  std::vector<double> src_marginal(n_nodes, 0.0);
  std::vector<double> tgt_marginal(n_nodes, 0.0);

  double dot_product_cost = 0.0;
  double plan_total_sum = 0.0;
  bool all_positive = true;

  for (std::size_t k = 0; k < plan.flow.size(); ++k) {
    const std::size_t src = plan.source[k];
    const std::size_t tgt = plan.target[k];
    const double f = plan.flow[k];

    if (f <= 0.0) {
      all_positive = false;
    }
    plan_total_sum += f;

    src_marginal[src] += f;
    tgt_marginal[tgt] += f;

    const auto c_src = layout.coordinates(static_cast<std::ptrdiff_t>(src));
    const auto c_tgt = layout.coordinates(static_cast<std::ptrdiff_t>(tgt));

    double dist_unit = 0.0;
    for (std::size_t a = 0; a < Dim; ++a) {
      const double diff = static_cast<double>(c_src[a] - c_tgt[a]);
      dist_unit += (metric == emdgrid::GroundMetric::L1) ? std::abs(diff)
                                                         : (diff * diff);
    }
    dot_product_cost += f * dist_unit;
  }

  double max_src_residual = 0.0;
  double max_tgt_residual = 0.0;
  for (std::size_t i = 0; i < n_nodes; ++i) {
    max_src_residual = std::max(
        max_src_residual,
        std::abs(src_marginal[i] - static_cast<double>(h1.data()[i])));
    max_tgt_residual = std::max(
        max_tgt_residual,
        std::abs(tgt_marginal[i] - static_cast<double>(h2.data()[i])));
  }

  std::cout << "  [Diagnostics]\n";
  std::cout << "    Recomputed cost (dot product with C): " << dot_product_cost
            << '\n';
  std::cout << "    Plan total sum: " << plan_total_sum
            << " (valid = "
            << (std::abs(plan_total_sum - 1.0) < 1e-5 ? "yes" : "NO") << ")\n";
  std::cout << "    All flows positive: " << (all_positive ? "yes" : "NO")
            << '\n';
  std::cout << "    Max src marginal residual: " << max_src_residual << '\n';
  std::cout << "    Max tgt marginal residual: " << max_tgt_residual << '\n';
}

int main(int argc, char** argv) {
  CLI::App app{"emdgrid CLI example benchmark tool"};

  std::size_t dim = 10;
  bool compute_plan = false;
  bool diagnostics = false;
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
  app.add_flag("--diagnostics", diagnostics,
               "Run plan diagnostics (forces transport plan calculation)");
  app.add_option("-s,--solver", solver,
                 "Solver to run: 'emd_l1', 'mcf_l1', 'mcf_lemon_ns', "
                 "'mcf_lemon_cs', 'dpartion', 'greedy', 'kr', or 'all' "
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
  const bool run_mcf_l1 = (solver == "all" || solver == "mcf_l1" ||
                           solver == "mcf");
  const bool run_mcf_lemon_ns = (solver == "all" || solver == "mcf_lemon" ||
                                 solver == "mcf_lemon_ns");
  const bool run_mcf_lemon_cs = (solver == "all" || solver == "mcf_lemon" ||
                                 solver == "mcf_lemon_cs");
  const bool run_dpartion = (solver == "all" || solver == "dpartion" ||
                             solver == "mcf_dpartion");
  const bool run_greedy = (solver == "all" || solver == "greedy");
  const bool run_kr = (solver == "all" || solver == "kr" ||
                       solver == "knothe_rosenblatt");

  const bool need_plan = compute_plan || diagnostics;

  if (run_emd_l1) {
    emdgrid::SparseTransportPlan plan;
    const emdgrid::Timer timer;
    const double dist =
        emdgrid::emd_l1(h1, h2, need_plan ? &plan : nullptr, max_iter);
    const double elapsed_ms = timer.elapsed_milliseconds();

    std::cout << "\n--- Exact EMD-L1 (Ling & Okada) ---\n";
    std::cout << "Distance: " << dist << '\n';
    std::cout << "Computation time: " << elapsed_ms << " ms\n";
    if (need_plan) {
      std::cout << "Transport plan flow entries: " << plan.source.size()
                << '\n';
      if (diagnostics) {
        run_diagnostics(layout, h1, h2, plan, emdgrid::GroundMetric::L1);
      }
    }
  }

  if (run_mcf_l1) {
    emdgrid::SparseTransportPlan plan;
    const emdgrid::Timer timer;
    const double dist =
        emdgrid::mcf_l1(h1, h2, need_plan ? &plan : nullptr);
    const double elapsed_ms = timer.elapsed_milliseconds();

    std::cout << "\n--- Min-Cost Flow EMD-L1 (OR-Tools) ---\n";
    std::cout << "Distance: " << dist << '\n';
    std::cout << "Computation time: " << elapsed_ms << " ms\n";
    if (need_plan) {
      std::cout << "Transport plan flow entries: " << plan.source.size()
                << '\n';
      if (diagnostics) {
        run_diagnostics(layout, h1, h2, plan, emdgrid::GroundMetric::L1);
      }
    }
  }

  if (run_mcf_lemon_ns) {
    emdgrid::SparseTransportPlan plan;
    const emdgrid::Timer timer;
    const double dist = emdgrid::mcf_lemon_l1(
        h1, h2, emdgrid::McfLemonAlgorithm::NetworkSimplex,
        need_plan ? &plan : nullptr);
    const double elapsed_ms = timer.elapsed_milliseconds();

    std::cout << "\n--- Min-Cost Flow EMD-L1 (LEMON NetworkSimplex) ---\n";
    std::cout << "Distance: " << dist << '\n';
    std::cout << "Computation time: " << elapsed_ms << " ms\n";
    if (need_plan) {
      std::cout << "Transport plan flow entries: " << plan.source.size()
                << '\n';
      if (diagnostics) {
        run_diagnostics(layout, h1, h2, plan, emdgrid::GroundMetric::L1);
      }
    }
  }

  if (run_mcf_lemon_cs) {
    emdgrid::SparseTransportPlan plan;
    const emdgrid::Timer timer;
    const double dist = emdgrid::mcf_lemon_l1(
        h1, h2, emdgrid::McfLemonAlgorithm::CostScaling,
        need_plan ? &plan : nullptr);
    const double elapsed_ms = timer.elapsed_milliseconds();

    std::cout << "\n--- Min-Cost Flow EMD-L1 (LEMON CostScaling) ---\n";
    std::cout << "Distance: " << dist << '\n';
    std::cout << "Computation time: " << elapsed_ms << " ms\n";
    if (need_plan) {
      std::cout << "Transport plan flow entries: " << plan.source.size()
                << '\n';
      if (diagnostics) {
        run_diagnostics(layout, h1, h2, plan, emdgrid::GroundMetric::L1);
      }
    }
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
      emdgrid::SparseTransportPlan plan;
      const emdgrid::Timer timer;
      const double dist = emdgrid::mcf_dpartion(
          h1, h2, metric, algo, need_plan ? &plan : nullptr);
      const double elapsed_ms = timer.elapsed_milliseconds();

      std::cout << "Variant [" << name << "]:\n";
      std::cout << "  Distance: " << dist << '\n';
      std::cout << "  Computation time: " << elapsed_ms << " ms\n";
      if (need_plan) {
        std::cout << "  Transport plan flow entries: " << plan.source.size()
                  << '\n';
        if (diagnostics) {
          run_diagnostics(layout, h1, h2, plan, metric);
        }
      }
    }
  }

  if (run_greedy) {
    emdgrid::SparseTransportPlan plan;
    const emdgrid::Timer timer;
    const double dist =
        emdgrid::greedy_emd_l1_approx(h1, h2, need_plan ? &plan : nullptr);
    const double elapsed_ms = timer.elapsed_milliseconds();

    std::cout << "\n--- Greedy Basic Feasible EMD-L1 ---\n";
    std::cout << "Approx distance: " << dist << '\n';
    std::cout << "Computation time: " << elapsed_ms << " ms\n";
    if (need_plan) {
      std::cout << "Transport plan flow entries: " << plan.source.size()
                << '\n';
      if (diagnostics) {
        run_diagnostics(layout, h1, h2, plan, emdgrid::GroundMetric::L1);
      }
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
        h1, h2, metric, dimension_order, need_plan ? &plan : nullptr);
    const double elapsed_ms = timer.elapsed_milliseconds();

    std::cout << "\n--- Knothe-Rosenblatt ("
              << (metric == emdgrid::GroundMetric::L1 ? "L1" : "SqEuclidean")
              << ") ---\n";
    std::cout << "Approx distance: " << dist << '\n';
    std::cout << "Computation time: " << elapsed_ms << " ms\n";
    if (need_plan) {
      std::cout << "Transport plan flow entries: " << plan.source.size()
                << '\n';
      if (diagnostics) {
        run_diagnostics(layout, h1, h2, plan, metric);
      }
    }
  }

  return 0;
}
