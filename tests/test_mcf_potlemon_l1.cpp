#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

#include <doctest/doctest.h>  // NOLINT(build/include_order)

#include "emdgrid/emd_l1.hpp"
#include "emdgrid/emdgrid.hpp"
#include "emdgrid/mcf_l1.hpp"
#include "emdgrid/mcf_lemon_l1.hpp"
#include "emdgrid/mcf_potlemon_l1.hpp"
#include "emdgrid/utils.hpp"

TEST_SUITE_BEGIN("mcf_potlemon_l1");

TEST_CASE("mcf_potlemon_l1 1D: identical histograms give zero") {
  const emdgrid::GridLayout<1> layout({5});
  const std::vector<double> h_data = {0.1, 0.2, 0.4, 0.2, 0.1};

  const emdgrid::GridDataView<1, double> h1(layout, std::span(h_data));
  const emdgrid::GridDataView<1, double> h2(layout, std::span(h_data));

  const double dist = emdgrid::mcf_potlemon_l1(h1, h2);
  CHECK(dist == doctest::Approx(0.0).epsilon(1e-9));
}

TEST_CASE("mcf_potlemon_l1 2D: unit shift along one axis costs 1") {
  const emdgrid::GridLayout<2> layout({3, 3});
  std::vector<double> h1_data(9, 0.0);
  std::vector<double> h2_data(9, 0.0);

  h1_data[layout.node({1, 1})] = 1.0;
  h2_data[layout.node({2, 1})] = 1.0;

  const emdgrid::GridDataView<2, double> h1(layout, std::span(h1_data));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(h2_data));

  const double dist = emdgrid::mcf_potlemon_l1(h1, h2);
  CHECK(dist == doctest::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("mcf_potlemon_l1 3D: diagonal shift costs 3") {
  const emdgrid::GridLayout<3> layout({3, 3, 3});
  std::vector<double> h1_data(27, 0.0);
  std::vector<double> h2_data(27, 0.0);

  h1_data[layout.node({0, 0, 0})] = 1.0;
  h2_data[layout.node({1, 1, 1})] = 1.0;

  const emdgrid::GridDataView<3, double> h1(layout, std::span(h1_data));
  const emdgrid::GridDataView<3, double> h2(layout, std::span(h2_data));

  const double dist = emdgrid::mcf_potlemon_l1(h1, h2);
  CHECK(dist == doctest::Approx(3.0).epsilon(1e-6));
}

TEST_CASE("mcf_potlemon_l1 3D: matches emd_l1, mcf_l1, and mcf_lemon_l1 on random histograms") {
  const emdgrid::GridLayout<3> layout({4, 4, 4});
  const std::size_t n = layout.node_count();

  const std::vector<double> h1_data =
      emdgrid::generate_random_histogram<double>(n, 12345);
  const std::vector<double> h2_data =
      emdgrid::generate_random_histogram<double>(n, 67890);

  const emdgrid::GridDataView<3, double> h1(layout, std::span(h1_data));
  const emdgrid::GridDataView<3, double> h2(layout, std::span(h2_data));

  const double emd_dist = emdgrid::emd_l1(h1, h2);
  const double mcf_dist = emdgrid::mcf_l1(h1, h2);
  const double lemon_dist = emdgrid::mcf_lemon_l1(h1, h2);
  const double potlemon_dist = emdgrid::mcf_potlemon_l1(h1, h2);

  CHECK(potlemon_dist == doctest::Approx(emd_dist).epsilon(1e-4));
  CHECK(potlemon_dist == doctest::Approx(mcf_dist).epsilon(1e-4));
  CHECK(potlemon_dist == doctest::Approx(lemon_dist).epsilon(1e-4));
}

TEST_CASE("mcf_potlemon_l1 2D: transport plan computation and cost reconstruction") {
  const emdgrid::GridLayout<2> layout({3, 3});
  const std::size_t n = layout.node_count();

  const std::vector<double> h1_data =
      emdgrid::generate_random_histogram<double>(n, 111);
  const std::vector<double> h2_data =
      emdgrid::generate_random_histogram<double>(n, 222);

  const emdgrid::GridDataView<2, double> h1(layout, std::span(h1_data));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(h2_data));

  emdgrid::SparseTransportPlan plan;
  const double dist = emdgrid::mcf_potlemon_l1(h1, h2, &plan);

  double recomputed_cost = 0.0;
  double flow_sum = 0.0;
  std::vector<double> src_margin(n, 0.0);
  std::vector<double> tgt_margin(n, 0.0);

  for (std::size_t k = 0; k < plan.flow.size(); ++k) {
    const uint32_t u = plan.source[k];
    const uint32_t v = plan.target[k];
    const double f = plan.flow[k];

    CHECK(f > 0.0);
    flow_sum += f;
    src_margin[u] += f;
    tgt_margin[v] += f;

    const auto cu = layout.coordinates(static_cast<std::ptrdiff_t>(u));
    const auto cv = layout.coordinates(static_cast<std::ptrdiff_t>(v));
    const double l1_dist =
        std::abs(static_cast<double>(cu[0] - cv[0])) +
        std::abs(static_cast<double>(cu[1] - cv[1]));

    recomputed_cost += f * l1_dist;
  }

  CHECK(flow_sum == doctest::Approx(1.0).epsilon(1e-5));
  CHECK(recomputed_cost == doctest::Approx(dist).epsilon(1e-4));

  for (std::size_t i = 0; i < n; ++i) {
    CHECK(src_margin[i] == doctest::Approx(h1_data[i]).epsilon(1e-4));
    CHECK(tgt_margin[i] == doctest::Approx(h2_data[i]).epsilon(1e-4));
  }
}

TEST_CASE("mcf_potlemon_l1 2D: shape mismatch throws") {
  const emdgrid::GridLayout<2> l1({3, 3});
  const emdgrid::GridLayout<2> l2({3, 4});

  const std::vector<double> d1(9, 1.0 / 9.0);
  const std::vector<double> d2(12, 1.0 / 12.0);

  const emdgrid::GridDataView<2, double> h1(l1, std::span(d1));
  const emdgrid::GridDataView<2, double> h2(l2, std::span(d2));

  CHECK_THROWS_AS(void(emdgrid::mcf_potlemon_l1(h1, h2)),
                  std::invalid_argument);
}

TEST_CASE("mcf_potlemon_l1 2D: unnormalized mass throws") {
  const emdgrid::GridLayout<2> l({2, 2});
  const std::vector<double> d1 = {0.5, 0.5, 0.0, 0.0};
  const std::vector<double> d2 = {0.2, 0.2, 0.2, 0.2};

  const emdgrid::GridDataView<2, double> h1(l, std::span(d1));
  const emdgrid::GridDataView<2, double> h2(l, std::span(d2));

  CHECK_THROWS_AS(void(emdgrid::mcf_potlemon_l1(h1, h2)),
                  std::invalid_argument);
}

TEST_SUITE_END();
