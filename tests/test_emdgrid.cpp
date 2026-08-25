#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>  // NOLINT(build/include_order)

#include "emdgrid/emd_l1.hpp"
#include "emdgrid/emdgrid.hpp"
#include "emdgrid/greedy_emd_l1.hpp"
#include "emdgrid/knothe_rosenblatt.hpp"
#include "emdgrid/utils.hpp"
#include "emdgrid/version.hpp"

TEST_CASE("library version matches generated version") {
  CHECK_EQ(emdgrid::version(), emdgrid::detail::version);
}

TEST_CASE("2D grid layout operations") {
  constexpr emdgrid::GridLayout<2>::NodeId t2d_last_node = 5;
  constexpr std::size_t t2d_node_count = 6;
  constexpr std::size_t t2d_edge_count = 7;

  const emdgrid::GridLayout<2> layout2d({2, 3});
  CHECK_EQ(layout2d.shape(), emdgrid::GridLayout<2>::Shape{2, 3});
  CHECK_EQ(layout2d.node_count(), t2d_node_count);
  CHECK_EQ(layout2d.edge_count(), t2d_edge_count);
  CHECK_EQ(layout2d.node({1, 2}), t2d_last_node);
  CHECK_EQ(layout2d.coordinates(4), emdgrid::GridLayout<2>::Coordinates{1, 1});

  const std::vector<emdgrid::GridLayout<2>::NodeId>
      expected_interior_neighbours = {1, 3, t2d_last_node};
  CHECK_EQ(layout2d.neighbours(layout2d.node({1, 1})),
           expected_interior_neighbours);

  const std::vector<emdgrid::GridLayout<2>::NodeId> expected_corner_neighbours =
      {3, 1};
  CHECK_EQ(layout2d.neighbours(layout2d.node({0, 0})),
           expected_corner_neighbours);
}

TEST_CASE("3D grid layout operations") {
  constexpr emdgrid::GridLayout<3>::NodeId t3d_last_node = 23;
  constexpr emdgrid::GridLayout<3>::NodeId t3d_node = 17;
  constexpr std::size_t t3d_node_count = 24;
  constexpr std::size_t t3d_edge_count = 46;

  const emdgrid::GridLayout<3> layout3d({2, 3, 4});
  CHECK_EQ(layout3d.node_count(), t3d_node_count);
  CHECK_EQ(layout3d.edge_count(), t3d_edge_count);
  CHECK_EQ(layout3d.node({1, 2, 3}), t3d_last_node);
  CHECK_EQ(layout3d.coordinates(t3d_node),
           emdgrid::GridLayout<3>::Coordinates{1, 1, 1});
}

TEST_CASE("grid data view exposes layout and values") {
  constexpr int t2d_last_value = 5;

  const emdgrid::GridLayout<2> layout2d({2, 3});
  const std::vector<int> values = {0, 1, 2, 3, 4, 5};
  const emdgrid::GridDataView<2, int> view(layout2d, std::span(values));

  CHECK_EQ(&view.layout(), &layout2d);
  CHECK_EQ(view.data().data(), values.data());
  CHECK_EQ(view.data().size(), values.size());
  CHECK_EQ(view(1, 2), t2d_last_value);
  CHECK_EQ(view(emdgrid::GridLayout<2>::Coordinates{0, 1}), 1);
}

TEST_CASE("layout throws for out of range access") {
  constexpr emdgrid::GridLayout<2>::NodeId t2d_out_of_range_node = 6;
  const emdgrid::GridLayout<2> layout2d({2, 3});

  CHECK_THROWS_AS(static_cast<void>(layout2d.node({-1, 0})),
                  std::out_of_range);
  CHECK_THROWS_AS(static_cast<void>(layout2d.node({2, 0})),
                  std::out_of_range);
  CHECK_THROWS_AS(static_cast<void>(layout2d.coordinates(-1)),
                  std::out_of_range);
  CHECK_THROWS_AS(
      static_cast<void>(layout2d.coordinates(t2d_out_of_range_node)),
      std::out_of_range);
}

TEST_CASE("grid data view validates data size") {
  constexpr int t2d_last_value = 5;
  const emdgrid::GridLayout<2> layout2d({2, 3});
  const std::vector<int> values = {0, 1, 2, 3, 4, 5};
  CHECK_THROWS_AS(
      static_cast<void>(emdgrid::GridDataView<2, int>(
          layout2d, std::span(values).first(t2d_last_value))),
      std::invalid_argument);
}

// ============================================================================
//  EMD-L1 tests
// ============================================================================

TEST_CASE("emd_l1 1D: identical histograms give zero") {
  const emdgrid::GridLayout<1> layout({5});
  const std::vector<double> v = {0.1, 0.2, 0.4, 0.2, 0.1};
  const emdgrid::GridDataView<1, double> h(layout, std::span(v));
  CHECK(emdgrid::emd_l1(h, h) == doctest::Approx(0.0));
}

TEST_CASE("emd_l1 1D: unit shift by one bin") {
  // Moving one unit from bin 0 to bin 1 costs 1.
  const emdgrid::GridLayout<1> layout({3});
  const std::vector<double> h1v = {1.0, 0.0, 0.0};
  const std::vector<double> h2v = {0.0, 1.0, 0.0};
  const emdgrid::GridDataView<1, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<1, double> h2(layout, std::span(h2v));
  CHECK(emdgrid::emd_l1(h1, h2) == doctest::Approx(1.0));
}

TEST_CASE("emd_l1 1D: unit shift by two bins") {
  // Moving one unit from bin 0 to bin 2 costs 2.
  const emdgrid::GridLayout<1> layout({3});
  const std::vector<double> h1v = {1.0, 0.0, 0.0};
  const std::vector<double> h2v = {0.0, 0.0, 1.0};
  const emdgrid::GridDataView<1, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<1, double> h2(layout, std::span(h2v));
  CHECK(emdgrid::emd_l1(h1, h2) == doctest::Approx(2.0));
}

TEST_CASE("emd_l1 1D: symmetry") {
  const emdgrid::GridLayout<1> layout({4});
  const std::vector<double> av = {0.5, 0.5, 0.0, 0.0};
  const std::vector<double> bv = {0.0, 0.0, 0.5, 0.5};
  const emdgrid::GridDataView<1, double> ha(layout, std::span(av));
  const emdgrid::GridDataView<1, double> hb(layout, std::span(bv));
  CHECK(emdgrid::emd_l1(ha, hb) == doctest::Approx(emdgrid::emd_l1(hb, ha)));
}

TEST_CASE("emd_l1 1D: shape mismatch throws") {
  const emdgrid::GridLayout<1> layout3({3});
  const emdgrid::GridLayout<1> layout4({4});
  const std::vector<double> v3 = {1.0, 0.0, 0.0};
  const std::vector<double> v4 = {0.0, 0.0, 0.0, 1.0};
  const emdgrid::GridDataView<1, double> h3(layout3, std::span(v3));
  const emdgrid::GridDataView<1, double> h4(layout4, std::span(v4));
  CHECK_THROWS_AS(static_cast<void>(emdgrid::emd_l1(h3, h4)),
                  std::invalid_argument);
}

// ============================================================================
//  emd_sqeuclidean_1d tests
// ============================================================================

TEST_CASE("emd_sqeuclidean_1d: identical histograms give zero") {
  const emdgrid::GridLayout<1> layout({5});
  const std::vector<double> v = {0.1, 0.2, 0.4, 0.2, 0.1};
  const emdgrid::GridDataView<1, double> h(layout, std::span(v));
  CHECK(emdgrid::emd_sqeuclidean_1d(h, h) == doctest::Approx(0.0));
}

TEST_CASE("emd_sqeuclidean_1d: unit shift by one bin costs 1") {
  // Moving 1 unit from bin 0 to bin 1: (1-0)^2 = 1.
  const emdgrid::GridLayout<1> layout({3});
  const std::vector<double> h1v = {1.0, 0.0, 0.0};
  const std::vector<double> h2v = {0.0, 1.0, 0.0};
  const emdgrid::GridDataView<1, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<1, double> h2(layout, std::span(h2v));
  CHECK(emdgrid::emd_sqeuclidean_1d(h1, h2) == doctest::Approx(1.0));
}

TEST_CASE("emd_sqeuclidean_1d: unit shift by two bins costs 4") {
  // Moving 1 unit from bin 0 to bin 2: (2-0)^2 = 4.
  const emdgrid::GridLayout<1> layout({3});
  const std::vector<double> h1v = {1.0, 0.0, 0.0};
  const std::vector<double> h2v = {0.0, 0.0, 1.0};
  const emdgrid::GridDataView<1, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<1, double> h2(layout, std::span(h2v));
  CHECK(emdgrid::emd_sqeuclidean_1d(h1, h2) == doctest::Approx(4.0));
}

TEST_CASE("emd_sqeuclidean_1d: split mass transfer") {
  // h1 = [0.5, 0.5, 0.0], h2 = [0.0, 0.5, 0.5]
  // 0.5 shifted 0->1 (cost 0.5), 0.5 shifted 1->2 (cost 0.5). Total cost = 1.0.
  const emdgrid::GridLayout<1> layout({3});
  const std::vector<double> h1v = {0.5, 0.5, 0.0};
  const std::vector<double> h2v = {0.0, 0.5, 0.5};
  const emdgrid::GridDataView<1, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<1, double> h2(layout, std::span(h2v));
  CHECK(emdgrid::emd_sqeuclidean_1d(h1, h2) == doctest::Approx(1.0));
}

TEST_CASE("emd_sqeuclidean_1d: symmetry") {
  const emdgrid::GridLayout<1> layout({4});
  const std::vector<double> av = {0.5, 0.5, 0.0, 0.0};
  const std::vector<double> bv = {0.0, 0.0, 0.5, 0.5};
  const emdgrid::GridDataView<1, double> ha(layout, std::span(av));
  const emdgrid::GridDataView<1, double> hb(layout, std::span(bv));
  CHECK(emdgrid::emd_sqeuclidean_1d(ha, hb) ==
        doctest::Approx(emdgrid::emd_sqeuclidean_1d(hb, ha)));
}

TEST_CASE("emd_sqeuclidean_1d: shape mismatch throws") {
  const emdgrid::GridLayout<1> layout3({3});
  const emdgrid::GridLayout<1> layout4({4});
  const std::vector<double> v3 = {1.0, 0.0, 0.0};
  const std::vector<double> v4 = {0.0, 0.0, 0.0, 1.0};
  const emdgrid::GridDataView<1, double> h3(layout3, std::span(v3));
  const emdgrid::GridDataView<1, double> h4(layout4, std::span(v4));
  CHECK_THROWS_AS(static_cast<void>(emdgrid::emd_sqeuclidean_1d(h3, h4)),
                  std::invalid_argument);
}

TEST_CASE(
    "emd_sqeuclidean_1d: transport plan computation and cost reconstruction") {
  const emdgrid::GridLayout<1> layout({3});
  const std::vector<double> h1v = {1.0, 0.0, 0.0};
  const std::vector<double> h2v = {0.0, 0.0, 1.0};
  const emdgrid::GridDataView<1, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<1, double> h2(layout, std::span(h2v));

  emdgrid::SparseTransportPlan plan;
  double cost = emdgrid::emd_sqeuclidean_1d(h1, h2, &plan);

  CHECK(cost == doctest::Approx(4.0));
  REQUIRE_EQ(plan.source.size(), 1);
  REQUIRE_EQ(plan.target.size(), 1);
  REQUIRE_EQ(plan.flow.size(), 1);
  CHECK_EQ(plan.source[0], 0);
  CHECK_EQ(plan.target[0], 2);
  CHECK(plan.flow[0] == doctest::Approx(1.0));
}

TEST_CASE("emd_l1 2D: identical histograms give zero") {
  const emdgrid::GridLayout<2> layout({3, 3});
  const std::vector<double> v(9, 1.0 / 9);
  const emdgrid::GridDataView<2, double> h(layout, std::span(v));
  CHECK(emdgrid::emd_l1(h, h) == doctest::Approx(0.0));
}

TEST_CASE("emd_l1 2D: unit shift along one axis costs 1") {
  // Move one unit from (0,0) to (0,1): L1 distance = 1.
  const emdgrid::GridLayout<2> layout({2, 2});
  const std::vector<double> h1v = {1.0, 0.0, 0.0, 0.0};
  const std::vector<double> h2v = {0.0, 1.0, 0.0, 0.0};
  const emdgrid::GridDataView<2, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(h2v));
  CHECK(emdgrid::emd_l1(h1, h2) == doctest::Approx(1.0));
}

TEST_CASE("emd_l1 3D: max_iter parameter exposure and convergence") {
  constexpr std::size_t dim = 16;
  const emdgrid::GridLayout<3> layout({dim, dim, dim});
  const std::size_t n_bins = layout.node_count();

  const std::vector<double> h1_data =
      emdgrid::generate_random_histogram<double>(n_bins, 42);
  const std::vector<double> h2_data =
      emdgrid::generate_random_histogram<double>(n_bins, 1337);

  const emdgrid::GridDataView<3, double> h1(layout, std::span(h1_data));
  const emdgrid::GridDataView<3, double> h2(layout, std::span(h2_data));

  const double dist_kr = emdgrid::knothe_rosenblatt(
      h1, h2, emdgrid::GroundMetric::L1, {}, nullptr);

  // With sufficient max_iter, network simplex converges to exact optimal EMD-L1,
  // which is strictly <= Knothe-Rosenblatt heuristic upper bound.
  const double dist_emd = emdgrid::emd_l1(h1, h2, nullptr, 50000);

  CHECK(dist_emd <= dist_kr);
}

TEST_CASE("emd_l1 2D: respects max_iter limit") {
  const emdgrid::GridLayout<2> layout({4, 4});
  const std::vector<double> h1_data =
      emdgrid::generate_random_histogram<double>(16, 42);
  const std::vector<double> h2_data =
      emdgrid::generate_random_histogram<double>(16, 1337);

  const emdgrid::GridDataView<2, double> h1(layout, std::span(h1_data));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(h2_data));

  const double dist_0_iter = emdgrid::emd_l1(h1, h2, nullptr, 0);
  const double dist_full = emdgrid::emd_l1(h1, h2, nullptr, 10000);

  // max_iter = 0 performs 0 pivots, staying at initial greedy cost (or > optimal)
  CHECK(dist_0_iter >= dist_full);
}

// ============================================================================
//  knothe_rosenblatt tests
// ============================================================================

TEST_CASE("knothe_rosenblatt 1D: identical histograms give zero") {
  const emdgrid::GridLayout<1> layout({5});
  const std::vector<double> v = {0.1, 0.2, 0.4, 0.2, 0.1};
  const emdgrid::GridDataView<1, double> h(layout, std::span(v));
  CHECK(emdgrid::knothe_rosenblatt(h, h, emdgrid::GroundMetric::L1) ==
        doctest::Approx(0.0));
  CHECK(emdgrid::knothe_rosenblatt(h, h, emdgrid::GroundMetric::SqEuclidean) ==
        doctest::Approx(0.0));
}

TEST_CASE("knothe_rosenblatt 1D: matches 1D exact solvers") {
  const emdgrid::GridLayout<1> layout({3});
  const std::vector<double> h1v = {1.0, 0.0, 0.0};
  const std::vector<double> h2v = {0.0, 0.0, 1.0};
  const emdgrid::GridDataView<1, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<1, double> h2(layout, std::span(h2v));

  double l1_cost =
      emdgrid::knothe_rosenblatt(h1, h2, emdgrid::GroundMetric::L1);
  double sq_cost =
      emdgrid::knothe_rosenblatt(h1, h2, emdgrid::GroundMetric::SqEuclidean);

  CHECK(l1_cost == doctest::Approx(emdgrid::emd_1d(h1, h2)));
  CHECK(sq_cost == doctest::Approx(emdgrid::emd_sqeuclidean_1d(h1, h2)));
}

TEST_CASE("knothe_rosenblatt 2D: metric selection L1 vs SqEuclidean") {
  const emdgrid::GridLayout<2> layout({3, 3});
  std::vector<double> h1v(9, 0.0);
  std::vector<double> h2v(9, 0.0);
  h1v[0] = 1.0;  // bin (0,0)
  h2v[8] = 1.0;  // bin (2,2)
  const emdgrid::GridDataView<2, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(h2v));

  // Distance along axis 0: |0-2| = 2, sq = 4.
  // Distance along axis 1: |0-2| = 2, sq = 4.
  // L1 total cost = 4.0. SqEuclidean total cost = 8.0.
  double cost_l1 =
      emdgrid::knothe_rosenblatt(h1, h2, emdgrid::GroundMetric::L1);
  double cost_sq =
      emdgrid::knothe_rosenblatt(h1, h2, emdgrid::GroundMetric::SqEuclidean);

  CHECK(cost_l1 == doctest::Approx(4.0));
  CHECK(cost_sq == doctest::Approx(8.0));
}

TEST_CASE("knothe_rosenblatt 2D: custom dimension order permutation") {
  const emdgrid::GridLayout<2> layout({2, 2});
  // H1 = [[0.5, 0.0], [0.0, 0.5]], H2 = [[0.0, 0.5], [0.0, 0.5]]
  const std::vector<double> h1v = {0.5, 0.0, 0.0, 0.5};
  const std::vector<double> h2v = {0.0, 0.5, 0.0, 0.5};
  const emdgrid::GridDataView<2, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(h2v));

  const std::array<std::size_t, 2> order_01 = {0, 1};
  const std::array<std::size_t, 2> order_10 = {1, 0};

  double cost_01 = emdgrid::knothe_rosenblatt(
      h1, h2, emdgrid::GroundMetric::L1, std::span(order_01));
  double cost_10 = emdgrid::knothe_rosenblatt(
      h1, h2, emdgrid::GroundMetric::L1, std::span(order_10));

  CHECK(cost_01 >= 0.0);
  CHECK(cost_10 >= 0.0);
}

TEST_CASE("knothe_rosenblatt 3D: diagonal shift") {
  const emdgrid::GridLayout<3> layout({2, 2, 2});
  std::vector<double> h1v(8, 0.0);
  std::vector<double> h2v(8, 0.0);
  h1v[0] = 1.0;
  h2v[7] = 1.0;
  const emdgrid::GridDataView<3, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<3, double> h2(layout, std::span(h2v));

  double cost_l1 =
      emdgrid::knothe_rosenblatt(h1, h2, emdgrid::GroundMetric::L1);
  double cost_sq =
      emdgrid::knothe_rosenblatt(h1, h2, emdgrid::GroundMetric::SqEuclidean);

  // From (0,0,0) to (1,1,1):
  // L1: 1 + 1 + 1 = 3.0
  // Sq: 1^2 + 1^2 + 1^2 = 3.0
  CHECK(cost_l1 == doctest::Approx(3.0));
  CHECK(cost_sq == doctest::Approx(3.0));
}

TEST_CASE("knothe_rosenblatt: invalid dimension order throws") {
  const emdgrid::GridLayout<2> layout({2, 2});
  const std::vector<double> h1v = {0.5, 0.5, 0.0, 0.0};
  const emdgrid::GridDataView<2, double> h1(layout, std::span(h1v));

  const std::vector<std::size_t> wrong_length = {0};
  const std::vector<std::size_t> duplicate_dims = {1, 1};
  const std::vector<std::size_t> out_of_bounds = {0, 5};

  CHECK_THROWS_AS(
      static_cast<void>(emdgrid::knothe_rosenblatt(
          h1, h1, emdgrid::GroundMetric::L1, std::span(wrong_length))),
      std::invalid_argument);
  CHECK_THROWS_AS(
      static_cast<void>(emdgrid::knothe_rosenblatt(
          h1, h1, emdgrid::GroundMetric::L1, std::span(duplicate_dims))),
      std::invalid_argument);
  CHECK_THROWS_AS(
      static_cast<void>(emdgrid::knothe_rosenblatt(
          h1, h1, emdgrid::GroundMetric::L1, std::span(out_of_bounds))),
      std::invalid_argument);
}

TEST_CASE("knothe_rosenblatt: shape mismatch throws") {
  const emdgrid::GridLayout<2> layout2({2, 2});
  const emdgrid::GridLayout<2> layout3({3, 3});
  const std::vector<double> v4(4, 0.25);
  const std::vector<double> v9(9, 1.0 / 9);
  const emdgrid::GridDataView<2, double> h4(layout2, std::span(v4));
  const emdgrid::GridDataView<2, double> h9(layout3, std::span(v9));
  CHECK_THROWS_AS(static_cast<void>(emdgrid::knothe_rosenblatt(h4, h9)),
                  std::invalid_argument);
}

TEST_CASE(
    "knothe_rosenblatt 2D: transport plan computation and cost "
    "reconstruction") {
  const emdgrid::GridLayout<2> layout({2, 2});
  const std::vector<double> h1v = {0.5, 0.5, 0.0, 0.0};
  const std::vector<double> h2v = {0.0, 0.0, 0.5, 0.5};
  const emdgrid::GridDataView<2, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(h2v));

  emdgrid::SparseTransportPlan plan_l1;
  double cost_l1 = emdgrid::knothe_rosenblatt(
      h1, h2, emdgrid::GroundMetric::L1, {}, &plan_l1);

  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  REQUIRE(!plan_l1.flow.empty());
  double reconstructed_l1 = 0.0;
  for (std::size_t k = 0; k < plan_l1.flow.size(); ++k) {
    auto c_src =
        layout.coordinates(static_cast<std::ptrdiff_t>(plan_l1.source[k]));
    auto c_tgt =
        layout.coordinates(static_cast<std::ptrdiff_t>(plan_l1.target[k]));
    double dist = static_cast<double>(std::abs(c_src[0] - c_tgt[0]) +
                                      std::abs(c_src[1] - c_tgt[1]));
    reconstructed_l1 += plan_l1.flow[k] * dist;
  }
  CHECK(reconstructed_l1 == doctest::Approx(cost_l1));

  emdgrid::SparseTransportPlan plan_sq;
  double cost_sq = emdgrid::knothe_rosenblatt(
      h1, h2, emdgrid::GroundMetric::SqEuclidean, {}, &plan_sq);

  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  REQUIRE(!plan_sq.flow.empty());
  double reconstructed_sq = 0.0;
  for (std::size_t k = 0; k < plan_sq.flow.size(); ++k) {
    auto c_src =
        layout.coordinates(static_cast<std::ptrdiff_t>(plan_sq.source[k]));
    auto c_tgt =
        layout.coordinates(static_cast<std::ptrdiff_t>(plan_sq.target[k]));
    double d0 = static_cast<double>(c_src[0] - c_tgt[0]);
    double d1 = static_cast<double>(c_src[1] - c_tgt[1]);
    reconstructed_sq += plan_sq.flow[k] * (d0 * d0 + d1 * d1);
  }
  CHECK(reconstructed_sq == doctest::Approx(cost_sq));
}

// ============================================================================
//  greedy_emd_l1_approx tests
// ============================================================================

TEST_CASE("greedy_emd_l1_approx 1D: identical histograms give zero") {
  const emdgrid::GridLayout<1> layout({5});
  const std::vector<double> v = {0.1, 0.2, 0.4, 0.2, 0.1};
  const emdgrid::GridDataView<1, double> h(layout, std::span(v));
  CHECK(emdgrid::greedy_emd_l1_approx(h, h) == doctest::Approx(0.0));
}

TEST_CASE("greedy_emd_l1_approx 2D: identical histograms give zero") {
  const emdgrid::GridLayout<2> layout({3, 3});
  const std::vector<double> v(9, 1.0 / 9);
  const emdgrid::GridDataView<2, double> h(layout, std::span(v));
  CHECK(emdgrid::greedy_emd_l1_approx(h, h) == doctest::Approx(0.0));
}

TEST_CASE("greedy_emd_l1_approx 2D: produces valid upper bound") {
  const emdgrid::GridLayout<2> layout({2, 2});
  const std::vector<double> h1v = {0.5, 0.5, 0.0, 0.0};
  const std::vector<double> h2v = {0.0, 0.0, 0.5, 0.5};
  const emdgrid::GridDataView<2, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(h2v));

  double exact_cost = emdgrid::emd_l1(h1, h2);
  double approx_cost = emdgrid::greedy_emd_l1_approx(h1, h2);

  CHECK(exact_cost == doctest::Approx(1.0));
  CHECK(approx_cost >= exact_cost);
}

TEST_CASE(
    "greedy_emd_l1_approx 2D: transport plan computation and cost "
    "reconstruction") {
  const emdgrid::GridLayout<2> layout({2, 2});
  const std::vector<double> h1v = {1.0, 0.0, 0.0, 1.0};
  const std::vector<double> h2v = {0.0, 1.0, 1.0, 0.0};
  const emdgrid::GridDataView<2, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(h2v));

  emdgrid::SparseTransportPlan plan;
  double approx_cost = emdgrid::greedy_emd_l1_approx(h1, h2, &plan);

  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  REQUIRE(!plan.flow.empty());

  double reconstructed_cost = 0.0;
  double total_flow = 0.0;
  bool all_positive = true;

  for (std::size_t k = 0; k < plan.flow.size(); ++k) {
    uint32_t src = plan.source[k];
    uint32_t tgt = plan.target[k];
    double f = plan.flow[k];
    if (f <= 0.0) {
      all_positive = false;
    }
    total_flow += f;

    auto c_src = layout.coordinates(static_cast<std::ptrdiff_t>(src));
    auto c_tgt = layout.coordinates(static_cast<std::ptrdiff_t>(tgt));
    double l1_dist = static_cast<double>(std::abs(c_src[0] - c_tgt[0]) +
                                         std::abs(c_src[1] - c_tgt[1]));
    reconstructed_cost += f * l1_dist;
  }

  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  CHECK(all_positive);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  CHECK(total_flow == doctest::Approx(2.0));
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  CHECK(reconstructed_cost == doctest::Approx(approx_cost));
}

TEST_CASE("greedy_emd_l1_approx 2D: shape mismatch throws") {
  const emdgrid::GridLayout<2> layout2({2, 2});
  const emdgrid::GridLayout<2> layout3({3, 3});
  const std::vector<double> v4(4, 0.25);
  const std::vector<double> v9(9, 1.0 / 9);
  const emdgrid::GridDataView<2, double> h4(layout2, std::span(v4));
  const emdgrid::GridDataView<2, double> h9(layout3, std::span(v9));
  CHECK_THROWS_AS(static_cast<void>(emdgrid::greedy_emd_l1_approx(h4, h9)),
                  std::invalid_argument);
}

TEST_CASE("emd_l1 1D: transport plan computation") {
  const emdgrid::GridLayout<1> layout({3});
  const std::vector<double> h1v = {1.0, 0.0, 0.0};
  const std::vector<double> h2v = {0.0, 0.0, 1.0};
  const emdgrid::GridDataView<1, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<1, double> h2(layout, std::span(h2v));

  emdgrid::SparseTransportPlan plan;
  double cost = emdgrid::emd_l1(h1, h2, &plan);

  CHECK(cost == doctest::Approx(2.0));
  REQUIRE_EQ(plan.source.size(), 1);
  REQUIRE_EQ(plan.target.size(), 1);
  REQUIRE_EQ(plan.flow.size(), 1);
  CHECK_EQ(plan.source[0], 0);
  CHECK_EQ(plan.target[0], 2);
  CHECK(plan.flow[0] == doctest::Approx(1.0));
}

TEST_CASE("emd_l1 2D: transport plan computation and cost reconstruction") {
  const emdgrid::GridLayout<2> layout({2, 2});
  const std::vector<double> h1v = {1.0, 0.0, 0.0, 1.0};
  const std::vector<double> h2v = {0.0, 1.0, 1.0, 0.0};
  const emdgrid::GridDataView<2, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(h2v));

  emdgrid::SparseTransportPlan plan;
  double cost = emdgrid::emd_l1(h1, h2, &plan);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  CHECK(cost == doctest::Approx(2.0));

  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  REQUIRE(!plan.flow.empty());

  double reconstructed_cost = 0.0;
  double total_flow = 0.0;
  bool all_positive = true;

  for (std::size_t k = 0; k < plan.flow.size(); ++k) {
    uint32_t src = plan.source[k];
    uint32_t tgt = plan.target[k];
    double f = plan.flow[k];
    if (f <= 0.0) {
      all_positive = false;
    }
    total_flow += f;

    auto c_src = layout.coordinates(static_cast<std::ptrdiff_t>(src));
    auto c_tgt = layout.coordinates(static_cast<std::ptrdiff_t>(tgt));
    double l1_dist = static_cast<double>(std::abs(c_src[0] - c_tgt[0]) +
                                         std::abs(c_src[1] - c_tgt[1]));
    reconstructed_cost += f * l1_dist;
  }

  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  CHECK(all_positive);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  CHECK(total_flow == doctest::Approx(2.0));
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  CHECK(reconstructed_cost == doctest::Approx(cost));
}

TEST_CASE("emd_l1 2D: diagonal shift costs 2") {
  // Move one unit from (0,0) to (1,1): L1 distance = 2.
  const emdgrid::GridLayout<2> layout({2, 2});
  const std::vector<double> h1v = {1.0, 0.0, 0.0, 0.0};
  const std::vector<double> h2v = {0.0, 0.0, 0.0, 1.0};
  const emdgrid::GridDataView<2, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(h2v));
  CHECK(emdgrid::emd_l1(h1, h2) == doctest::Approx(2.0));
}

TEST_CASE("emd_l1 2D: antisymmetric 2x2 histograms") {
  // H1 = [[1,0],[0,1]], H2 = [[0,1],[1,0]].
  // Optimal: move 1 unit (0,0)→(0,1) and 1 unit (1,1)→(1,0). Total cost = 2.
  const emdgrid::GridLayout<2> layout({2, 2});
  const std::vector<double> h1v = {1.0, 0.0, 0.0, 1.0};
  const std::vector<double> h2v = {0.0, 1.0, 1.0, 0.0};
  const emdgrid::GridDataView<2, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(h2v));
  CHECK(emdgrid::emd_l1(h1, h2) == doctest::Approx(2.0));
}

TEST_CASE("emd_l1 2D: symmetry") {
  const emdgrid::GridLayout<2> layout({3, 3});
  const std::vector<double> av = {0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.5};
  const std::vector<double> bv = {0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0,
                                  0.0};
  const emdgrid::GridDataView<2, double> ha(layout, std::span(av));
  const emdgrid::GridDataView<2, double> hb(layout, std::span(bv));
  CHECK(emdgrid::emd_l1(ha, hb) == doctest::Approx(emdgrid::emd_l1(hb, ha)));
}

TEST_CASE("emd_l1 2D: shape mismatch throws") {
  const emdgrid::GridLayout<2> layout2({2, 2});
  const emdgrid::GridLayout<2> layout3({3, 3});
  const std::vector<double> v4(4, 0.25);
  const std::vector<double> v9(9, 1.0 / 9);
  const emdgrid::GridDataView<2, double> h4(layout2, std::span(v4));
  const emdgrid::GridDataView<2, double> h9(layout3, std::span(v9));
  CHECK_THROWS_AS(static_cast<void>(emdgrid::emd_l1(h4, h9)),
                  std::invalid_argument);
}

TEST_CASE("emd_l1 3D: diagonal shift costs 3") {
  // Move one unit from (0,0,0) to (1,1,1): L1 distance = 3.
  const emdgrid::GridLayout<3> layout({2, 2, 2});
  std::vector<double> h1v(8, 0.0);
  std::vector<double> h2v(8, 0.0);
  h1v[0] = 1.0;
  h2v[7] = 1.0;
  const emdgrid::GridDataView<3, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<3, double> h2(layout, std::span(h2v));
  CHECK(emdgrid::emd_l1(h1, h2) == doctest::Approx(3.0));
}

TEST_CASE("emd_l1 2D: consistent with 1D projection for separable transport") {
  // If transport is purely along axis 0, EMD-L1 should equal the 1D EMD-L1
  // of the marginal distributions along axis 0.
  const emdgrid::GridLayout<2> layout({3, 1});
  const std::vector<double> h1v = {1.0, 0.0, 0.0};
  const std::vector<double> h2v = {0.0, 0.0, 1.0};
  const emdgrid::GridDataView<2, double> h1_2d(layout, std::span(h1v));
  const emdgrid::GridDataView<2, double> h2_2d(layout, std::span(h2v));

  const emdgrid::GridLayout<1> layout1d({3});
  const emdgrid::GridDataView<1, double> h1_1d(layout1d, std::span(h1v));
  const emdgrid::GridDataView<1, double> h2_1d(layout1d, std::span(h2v));

  CHECK(emdgrid::emd_l1(h1_2d, h2_2d) ==
        doctest::Approx(emdgrid::emd_l1(h1_1d, h2_1d)));
}

TEST_CASE("emd_l1 2D: pivot required — greedy is suboptimal") {
  // H1 has mass at top row, H2 at bottom row.
  // Greedy: diagonal (cost 2); optimal: straight down (cost 1).
  const emdgrid::GridLayout<2> layout({2, 2});
  const std::vector<double> h1v = {0.5, 0.5, 0.0, 0.0};
  const std::vector<double> h2v = {0.0, 0.0, 0.5, 0.5};
  const emdgrid::GridDataView<2, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(h2v));
  CHECK(emdgrid::emd_l1(h1, h2) == doctest::Approx(1.0));
}
