#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

#include <doctest/doctest.h>  // NOLINT(build/include_order)

#include "emdgrid/emd_l1.hpp"
#include "emdgrid/emdgrid.hpp"
#include "emdgrid/mcf_lemon_l1.hpp"
#include "emdgrid/opencv_emd.hpp"
#include "emdgrid/utils.hpp"

TEST_SUITE_BEGIN("opencv_emd");

// ============================================================================
//  L1 metric correctness
// ============================================================================

TEST_CASE("opencv_emd L1 1D: identical histograms give zero") {
  const emdgrid::GridLayout<1> layout({5});
  const std::vector<double> v = {0.1, 0.2, 0.4, 0.2, 0.1};
  const emdgrid::GridDataView<1, double> h(layout, std::span(v));
  CHECK(emdgrid::opencv_emd(h, h) == doctest::Approx(0.0).epsilon(1e-5));
}

TEST_CASE("opencv_emd L1 1D: unit shift by one bin costs 1") {
  const emdgrid::GridLayout<1> layout({3});
  const std::vector<double> h1v = {1.0, 0.0, 0.0};
  const std::vector<double> h2v = {0.0, 1.0, 0.0};
  const emdgrid::GridDataView<1, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<1, double> h2(layout, std::span(h2v));
  CHECK(emdgrid::opencv_emd(h1, h2) == doctest::Approx(1.0).epsilon(1e-5));
}

TEST_CASE("opencv_emd L1 1D: unit shift by two bins costs 2") {
  const emdgrid::GridLayout<1> layout({3});
  const std::vector<double> h1v = {1.0, 0.0, 0.0};
  const std::vector<double> h2v = {0.0, 0.0, 1.0};
  const emdgrid::GridDataView<1, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<1, double> h2(layout, std::span(h2v));
  CHECK(emdgrid::opencv_emd(h1, h2) == doctest::Approx(2.0).epsilon(1e-5));
}

TEST_CASE("opencv_emd L1 1D: symmetry") {
  const emdgrid::GridLayout<1> layout({4});
  const std::vector<double> av = {0.5, 0.5, 0.0, 0.0};
  const std::vector<double> bv = {0.0, 0.0, 0.5, 0.5};
  const emdgrid::GridDataView<1, double> ha(layout, std::span(av));
  const emdgrid::GridDataView<1, double> hb(layout, std::span(bv));
  CHECK(emdgrid::opencv_emd(ha, hb) ==
        doctest::Approx(emdgrid::opencv_emd(hb, ha)).epsilon(1e-5));
}

TEST_CASE("opencv_emd L1 2D: unit shift along one axis costs 1") {
  const emdgrid::GridLayout<2> layout({2, 2});
  const std::vector<double> h1v = {1.0, 0.0, 0.0, 0.0};
  const std::vector<double> h2v = {0.0, 1.0, 0.0, 0.0};
  const emdgrid::GridDataView<2, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(h2v));
  CHECK(emdgrid::opencv_emd(h1, h2) == doctest::Approx(1.0).epsilon(1e-5));
}

TEST_CASE("opencv_emd L1 2D: diagonal shift costs 2") {
  const emdgrid::GridLayout<2> layout({2, 2});
  const std::vector<double> h1v = {1.0, 0.0, 0.0, 0.0};
  const std::vector<double> h2v = {0.0, 0.0, 0.0, 1.0};
  const emdgrid::GridDataView<2, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(h2v));
  CHECK(emdgrid::opencv_emd(h1, h2) == doctest::Approx(2.0).epsilon(1e-5));
}

TEST_CASE("opencv_emd L1 3D: diagonal shift costs 3") {
  const emdgrid::GridLayout<3> layout({2, 2, 2});
  std::vector<double> h1v(8, 0.0);
  std::vector<double> h2v(8, 0.0);
  h1v[0] = 1.0;
  h2v[7] = 1.0;
  const emdgrid::GridDataView<3, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<3, double> h2(layout, std::span(h2v));
  CHECK(emdgrid::opencv_emd(h1, h2) == doctest::Approx(3.0).epsilon(1e-5));
}

// ============================================================================
//  SqEuclidean metric
// ============================================================================

TEST_CASE("opencv_emd SqEuclidean 1D: unit shift by two bins costs 4") {
  const emdgrid::GridLayout<1> layout({3});
  const std::vector<double> h1v = {1.0, 0.0, 0.0};
  const std::vector<double> h2v = {0.0, 0.0, 1.0};
  const emdgrid::GridDataView<1, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<1, double> h2(layout, std::span(h2v));
  CHECK(emdgrid::opencv_emd(h1, h2, emdgrid::GroundMetric::SqEuclidean) ==
        doctest::Approx(4.0).epsilon(1e-5));
}

TEST_CASE("opencv_emd SqEuclidean 2D: shift (0,0)->(2,2) costs 8") {
  const emdgrid::GridLayout<2> layout({3, 3});
  std::vector<double> h1v(9, 0.0);
  std::vector<double> h2v(9, 0.0);
  h1v[0] = 1.0;
  h2v[8] = 1.0;
  const emdgrid::GridDataView<2, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(h2v));
  CHECK(emdgrid::opencv_emd(h1, h2, emdgrid::GroundMetric::SqEuclidean) ==
        doctest::Approx(8.0).epsilon(1e-5));
}

// ============================================================================
//  Custom cost functor
// ============================================================================

TEST_CASE("opencv_emd custom cost: 3*L1 on 2D grid") {
  using Layout2 = emdgrid::GridLayout<2>;
  const Layout2 layout({3, 3});
  std::vector<double> h1v(9, 0.0);
  std::vector<double> h2v(9, 0.0);
  h1v[0] = 1.0;  // (0,0)
  h2v[8] = 1.0;  // (2,2)
  const emdgrid::GridDataView<2, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(h2v));

  auto custom = [](const Layout2::Coordinates& a,
                   const Layout2::Coordinates& b) -> float {
    return 3.F * (static_cast<float>(std::abs(
                      static_cast<std::ptrdiff_t>(a[0]) -
                      static_cast<std::ptrdiff_t>(b[0]))) +
                  static_cast<float>(std::abs(
                      static_cast<std::ptrdiff_t>(a[1]) -
                      static_cast<std::ptrdiff_t>(b[1]))));
  };

  // L1 distance (0,0)->(2,2) = 4, scaled by 3 = 12.
  CHECK(emdgrid::opencv_emd(h1, h2, custom) ==
        doctest::Approx(12.0).epsilon(1e-4));
}

// ============================================================================
//  Cross-solver agreement with emd_l1
// ============================================================================

TEST_CASE("opencv_emd L1 matches emd_l1 on 3D random histograms") {
  constexpr std::size_t dim = 4;
  const emdgrid::GridLayout<3> layout({dim, dim, dim});
  const std::size_t n_bins = layout.node_count();

  const std::vector<double> h1_data =
      emdgrid::generate_random_histogram<double>(n_bins, 42);
  const std::vector<double> h2_data =
      emdgrid::generate_random_histogram<double>(n_bins, 1337);

  const emdgrid::GridDataView<3, double> h1(layout, std::span(h1_data));
  const emdgrid::GridDataView<3, double> h2(layout, std::span(h2_data));

  const double dist_emd = emdgrid::emd_l1(h1, h2);
  const double dist_ocv = emdgrid::opencv_emd(h1, h2);

  CHECK(dist_ocv == doctest::Approx(dist_emd).epsilon(1e-3));
}

TEST_CASE("opencv_emd SqEuclidean matches mcf_dpartion SqEuclidean") {
  constexpr std::size_t dim = 4;
  const emdgrid::GridLayout<2> layout({dim, dim});
  const std::size_t n_bins = layout.node_count();

  const std::vector<double> h1_data =
      emdgrid::generate_random_histogram<double>(n_bins, 42);
  const std::vector<double> h2_data =
      emdgrid::generate_random_histogram<double>(n_bins, 1337);

  const emdgrid::GridDataView<2, double> h1(layout, std::span(h1_data));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(h2_data));

  const double dist_ocv =
      emdgrid::opencv_emd(h1, h2, emdgrid::GroundMetric::SqEuclidean);
  const double dist_dpart = emdgrid::mcf_dpartion(
      h1, h2, emdgrid::GroundMetric::SqEuclidean,
      emdgrid::McfLemonAlgorithm::NetworkSimplex);

  CHECK(dist_ocv == doctest::Approx(dist_dpart).epsilon(1e-3));
}

// ============================================================================
//  Transport plan
// ============================================================================

TEST_CASE("opencv_emd 2D: transport plan cost reconstruction") {
  const emdgrid::GridLayout<2> layout({2, 2});
  const std::vector<double> h1_norm = {0.5, 0.0, 0.0, 0.5};
  const std::vector<double> h2_norm = {0.0, 0.5, 0.5, 0.0};
  const emdgrid::GridDataView<2, double> h1(layout, std::span(h1_norm));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(h2_norm));

  emdgrid::SparseTransportPlan plan;
  const double cost = emdgrid::opencv_emd(h1, h2, emdgrid::GroundMetric::L1,
                                          &plan);
  CHECK(cost == doctest::Approx(1.0).epsilon(1e-5));
  REQUIRE(!plan.flow.empty());

  double reconstructed = 0.0;
  double total_flow = 0.0;
  for (std::size_t k = 0; k < plan.flow.size(); ++k) {
    const auto cs = layout.coordinates(
        static_cast<std::ptrdiff_t>(plan.source[k]));
    const auto ct = layout.coordinates(
        static_cast<std::ptrdiff_t>(plan.target[k]));
    const double d = static_cast<double>(std::abs(cs[0] - ct[0]) +
                                         std::abs(cs[1] - ct[1]));
    reconstructed += plan.flow[k] * d;
    total_flow += plan.flow[k];
  }
  CHECK(total_flow == doctest::Approx(1.0).epsilon(1e-5));
  CHECK(reconstructed == doctest::Approx(cost).epsilon(1e-5));
}

// ============================================================================
//  Input validation
// ============================================================================

TEST_CASE("opencv_emd: shape mismatch throws") {
  const emdgrid::GridLayout<2> layout2({2, 2});
  const emdgrid::GridLayout<2> layout3({3, 3});
  const std::vector<double> v4(4, 0.25);
  const std::vector<double> v9(9, 1.0 / 9);
  const emdgrid::GridDataView<2, double> h4(layout2, std::span(v4));
  const emdgrid::GridDataView<2, double> h9(layout3, std::span(v9));
  CHECK_THROWS_AS(static_cast<void>(emdgrid::opencv_emd(h4, h9)),
                  std::invalid_argument);
}

TEST_CASE("opencv_emd: unnormalized mass throws") {
  const emdgrid::GridLayout<2> layout2({2, 2});
  const std::vector<double> v1(4, 0.25);
  const std::vector<double> v2(4, 0.50);
  const emdgrid::GridDataView<2, double> h1(layout2, std::span(v1));
  const emdgrid::GridDataView<2, double> h2(layout2, std::span(v2));
  CHECK_THROWS_AS(static_cast<void>(emdgrid::opencv_emd(h1, h2)),
                  std::invalid_argument);
}

TEST_SUITE_END();
