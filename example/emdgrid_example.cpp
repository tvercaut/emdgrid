#include <iomanip>
#include <iostream>
#include <span>
#include <vector>

#include "emdgrid/emd_l1.hpp"
#include "emdgrid/emdgrid.hpp"
#include "emdgrid/utils.hpp"

int main() {
  std::cout << "emdgrid version " << emdgrid::version() << '\n';

  constexpr std::size_t dim1 = 10;
  constexpr std::size_t dim2 = 10;
  constexpr std::size_t dim3 = 10;

  const emdgrid::GridLayout<3> layout({dim1, dim2, dim3});
  const std::size_t n_bins = layout.node_count();

  std::cout << "Grid shape: 10x10x10 (" << n_bins << " bins)\n";

  // Generate a pair of random 10x10x10 normalized histograms
  const std::vector<double> h1_data =
      emdgrid::generate_random_histogram<double>(n_bins, 42);
  const std::vector<double> h2_data =
      emdgrid::generate_random_histogram<double>(n_bins, 1337);

  const emdgrid::GridDataView<3, double> h1(layout, std::span(h1_data));
  const emdgrid::GridDataView<3, double> h2(layout, std::span(h2_data));

  // Time the computation of emd_l1
  const emdgrid::Timer timer;
  const double distance = emdgrid::emd_l1(h1, h2);
  const double elapsed_ms = timer.elapsed_milliseconds();

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Computed EMD-L1 distance: " << distance << '\n';
  std::cout << "Computation time: " << elapsed_ms << " ms\n";

  return 0;
}
