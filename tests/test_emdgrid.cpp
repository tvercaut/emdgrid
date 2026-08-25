#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>  // NOLINT(build/include_order)

#include "emdgrid/emd_l1.hpp"
#include "emdgrid/emdgrid.hpp"
#include "emdgrid/greedy_emd_l1.hpp"
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

  CHECK(all_positive);
  CHECK(total_flow == doctest::Approx(2.0));
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
  CHECK(cost == doctest::Approx(2.0));

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

  CHECK(all_positive);
  CHECK(total_flow == doctest::Approx(2.0));
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
