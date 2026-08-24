#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "emdgrid/emd_l1.hpp"
#include "emdgrid/emdgrid.hpp"
#include "emdgrid/utils.hpp"

int main(int argc, char** argv) {
  std::cout << "emdgrid version " << emdgrid::version() << '\n';

  std::size_t dim = 10;
  bool compute_plan = false;

  if (argc > 1) {
    try {
      dim = std::stoull(argv[1]);
    } catch (...) {
      std::cerr << "Invalid dimension argument: " << argv[1] << '\n';
      return 1;
    }
  }

  if (argc > 2) {
    const std::string arg = argv[2];
    compute_plan = (arg == "1" || arg == "true" || arg == "TRUE" ||
                    arg == "yes" || arg == "YES" || arg == "y" || arg == "Y");
  }

  const emdgrid::GridLayout<3> layout({dim, dim, dim});
  const std::size_t n_bins = layout.node_count();

  std::cout << "Grid shape: " << dim << 'x' << dim << 'x' << dim << " ("
            << n_bins << " bins)\n";
  std::cout << "Compute transport plan: " << (compute_plan ? "yes" : "no")
            << '\n';

  // Generate a pair of random normalized histograms
  const std::vector<double> h1_data =
      emdgrid::generate_random_histogram<double>(n_bins, 42);
  const std::vector<double> h2_data =
      emdgrid::generate_random_histogram<double>(n_bins, 1337);

  const emdgrid::GridDataView<3, double> h1(layout, std::span(h1_data));
  const emdgrid::GridDataView<3, double> h2(layout, std::span(h2_data));

  // Time the computation of emd_l1
  emdgrid::SparseTransportPlan plan;
  const emdgrid::Timer timer;
  const double distance =
      emdgrid::emd_l1(h1, h2, compute_plan ? &plan : nullptr);
  const double elapsed_ms = timer.elapsed_milliseconds();

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Computed EMD-L1 distance: " << distance << '\n';
  std::cout << "Computation time: " << elapsed_ms << " ms\n";
  if (compute_plan) {
    std::cout << "Transport plan flow entries: " << plan.source.size() << '\n';
  }

  return 0;
}
