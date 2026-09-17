#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include <doctest/doctest.h>  // NOLINT(build/include_order)

#include "emdgrid/emd_l1.hpp"
#include "emdgrid/emd_lemon.hpp"
#include "emdgrid/emdgrid.hpp"
#include "emdgrid/knothe_rosenblatt.hpp"
#include "emdgrid/mcf_lemon_l1.hpp"
#include "emdgrid/opencv_emd.hpp"
#include "emdgrid/utils.hpp"

TEST_SUITE_BEGIN("emd_lemon");

namespace {

constexpr auto kNs = emdgrid::McfLemonAlgorithm::NetworkSimplex;
constexpr auto kCs = emdgrid::McfLemonAlgorithm::CostScaling;

/// Ground cost between two bins, summed over the axes.
template <std::size_t Dim>
double ground_cost(const emdgrid::GridLayout<Dim>& layout, uint32_t u,
                   uint32_t v, emdgrid::GroundMetric metric) {
  const auto cu = layout.coordinates(static_cast<std::ptrdiff_t>(u));
  const auto cv = layout.coordinates(static_cast<std::ptrdiff_t>(v));
  double total = 0.0;
  for (std::size_t k = 0; k < Dim; ++k) {
    const auto diff = static_cast<double>(cu[k] - cv[k]);
    total += (metric == emdgrid::GroundMetric::L1) ? std::abs(diff)
                                                   : (diff * diff);
  }
  return total;
}

/// Checks that a plan is a valid coupling of h1 and h2 and re-prices it.
template <std::size_t Dim>
void check_plan(const emdgrid::GridLayout<Dim>& layout,
                const std::vector<double>& h1, const std::vector<double>& h2,
                const emdgrid::SparseTransportPlan<>& plan, double dist,
                emdgrid::GroundMetric metric) {
  const std::size_t n = layout.node_count();
  std::vector<double> src_margin(n, 0.0);
  std::vector<double> tgt_margin(n, 0.0);
  double recomputed = 0.0;
  double flow_sum = 0.0;

  REQUIRE(plan.source.size() == plan.target.size());
  REQUIRE(plan.source.size() == plan.flow.size());

  for (std::size_t k = 0; k < plan.flow.size(); ++k) {
    const double f = plan.flow[k];
    CHECK(f > 0.0);
    flow_sum += f;
    src_margin[plan.source[k]] += f;
    tgt_margin[plan.target[k]] += f;
    recomputed += f * ground_cost(layout, plan.source[k], plan.target[k],
                                  metric);
  }

  CHECK(flow_sum == doctest::Approx(1.0).epsilon(1e-5));
  CHECK(recomputed == doctest::Approx(dist).epsilon(1e-4));
  for (std::size_t i = 0; i < n; ++i) {
    CHECK(src_margin[i] == doctest::Approx(h1[i]).epsilon(1e-4));
    CHECK(tgt_margin[i] == doctest::Approx(h2[i]).epsilon(1e-4));
  }
}

/// L1 ground metric with the cost along axis 0 doubled.
struct WeightedL1 {
  static constexpr bool extract_self_mass = true;

  [[nodiscard]] int64_t operator()(std::size_t axis, std::size_t p,
                                   std::size_t q) const noexcept {
    const auto diff = std::abs(static_cast<std::ptrdiff_t>(p) -
                               static_cast<std::ptrdiff_t>(q));
    return (axis == 0) ? 2 * diff : diff;
  }
};

}  // namespace

// ============================================================================
//  Closed-form cases
// ============================================================================

TEST_CASE("emd_lemon 1D: identical histograms give zero") {
  const emdgrid::GridLayout<1> layout({5});
  const std::vector<double> v = {0.1, 0.2, 0.4, 0.2, 0.1};
  const emdgrid::GridDataView<1, double> h(layout, std::span(v));

  CHECK(emdgrid::emd_lemon(h, h, kNs) == doctest::Approx(0.0).epsilon(1e-9));
  CHECK(emdgrid::emd_lemon(h, h, kCs) == doctest::Approx(0.0).epsilon(1e-9));
}

TEST_CASE("emd_lemon 1D: unit shift by two bins costs 2") {
  const emdgrid::GridLayout<1> layout({3});
  const std::vector<double> a = {1.0, 0.0, 0.0};
  const std::vector<double> b = {0.0, 0.0, 1.0};
  const emdgrid::GridDataView<1, double> h1(layout, std::span(a));
  const emdgrid::GridDataView<1, double> h2(layout, std::span(b));

  CHECK(emdgrid::emd_lemon(h1, h2, kNs) ==
        doctest::Approx(2.0).epsilon(1e-6));
  CHECK(emdgrid::emd_lemon(h1, h2, kCs) ==
        doctest::Approx(2.0).epsilon(1e-6));
}

TEST_CASE("emd_lemon 3D: diagonal shift costs 3 under L1, 3 under W2^2") {
  const emdgrid::GridLayout<3> layout({3, 3, 3});
  std::vector<double> a(27, 0.0);
  std::vector<double> b(27, 0.0);
  a[layout.node({0, 0, 0})] = 1.0;
  b[layout.node({1, 1, 1})] = 1.0;

  const emdgrid::GridDataView<3, double> h1(layout, std::span(a));
  const emdgrid::GridDataView<3, double> h2(layout, std::span(b));

  CHECK(emdgrid::emd_lemon(h1, h2, emdgrid::GroundMetric::L1, kNs) ==
        doctest::Approx(3.0).epsilon(1e-6));
  CHECK(emdgrid::emd_lemon(h1, h2, emdgrid::GroundMetric::SqEuclidean, kNs) ==
        doctest::Approx(3.0).epsilon(1e-6));
}

TEST_CASE("emd_lemon 2D: squared Euclidean splits differ from L1") {
  // Two unit masses at (0,0) and (2,0) move to (1,0) twice. Under L1 the cost
  // is 1/2 + 1/2 = 1; under squared Euclidean it is also 1, so use an
  // asymmetric case where the two metrics genuinely disagree: move all mass
  // from (0,0) to (2,0). L1 gives 2, squared Euclidean gives 4.
  const emdgrid::GridLayout<2> layout({3, 1});
  const std::vector<double> a = {1.0, 0.0, 0.0};
  const std::vector<double> b = {0.0, 0.0, 1.0};
  const emdgrid::GridDataView<2, double> h1(layout, std::span(a));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(b));

  CHECK(emdgrid::emd_lemon(h1, h2, emdgrid::GroundMetric::L1, kNs) ==
        doctest::Approx(2.0).epsilon(1e-6));
  CHECK(emdgrid::emd_lemon(h1, h2, emdgrid::GroundMetric::SqEuclidean, kNs) ==
        doctest::Approx(4.0).epsilon(1e-6));
}

TEST_CASE("emd_lemon 1D: symmetric in its arguments") {
  const emdgrid::GridLayout<1> layout({4});
  const std::vector<double> a = {0.5, 0.5, 0.0, 0.0};
  const std::vector<double> b = {0.0, 0.0, 0.5, 0.5};
  const emdgrid::GridDataView<1, double> ha(layout, std::span(a));
  const emdgrid::GridDataView<1, double> hb(layout, std::span(b));

  CHECK(emdgrid::emd_lemon(ha, hb, kNs) ==
        doctest::Approx(emdgrid::emd_lemon(hb, ha, kNs)).epsilon(1e-6));
}

// ============================================================================
//  Agreement with the other solvers
// ============================================================================

TEST_CASE("emd_lemon 3D L1: matches emd_l1, mcf_lemon_l1 and opencv_emd") {
  const emdgrid::GridLayout<3> layout({4, 4, 4});
  const std::size_t n = layout.node_count();

  const std::vector<double> a =
      emdgrid::generate_random_histogram<double>(n, 12345);
  const std::vector<double> b =
      emdgrid::generate_random_histogram<double>(n, 67890);

  const emdgrid::GridDataView<3, double> h1(layout, std::span(a));
  const emdgrid::GridDataView<3, double> h2(layout, std::span(b));

  const double reference = emdgrid::emd_l1(h1, h2);

  CHECK(emdgrid::emd_lemon(h1, h2, kNs) ==
        doctest::Approx(reference).epsilon(1e-4));
  CHECK(emdgrid::emd_lemon(h1, h2, kCs) ==
        doctest::Approx(reference).epsilon(1e-4));
  CHECK(emdgrid::mcf_lemon_l1(h1, h2) ==
        doctest::Approx(reference).epsilon(1e-4));
  CHECK(emdgrid::opencv_emd(h1, h2, emdgrid::GroundMetric::L1) ==
        doctest::Approx(reference).epsilon(1e-3));
}

TEST_CASE("emd_lemon 2D W2^2: matches mcf_dpartion's SqEuclidean optimum") {
  const emdgrid::GridLayout<2> layout({5, 5});
  const std::size_t n = layout.node_count();

  const std::vector<double> a =
      emdgrid::generate_random_histogram<double>(n, 7);
  const std::vector<double> b =
      emdgrid::generate_random_histogram<double>(n, 9);

  const emdgrid::GridDataView<2, double> h1(layout, std::span(a));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(b));

  const double dpart = emdgrid::mcf_dpartion(
      h1, h2, emdgrid::GroundMetric::SqEuclidean, kNs);

  CHECK(emdgrid::emd_lemon(h1, h2, emdgrid::GroundMetric::SqEuclidean, kNs) ==
        doctest::Approx(dpart).epsilon(1e-4));
  CHECK(emdgrid::emd_lemon(h1, h2, emdgrid::GroundMetric::SqEuclidean, kCs) ==
        doctest::Approx(dpart).epsilon(1e-4));
}

TEST_CASE("emd_lemon 2D: no worse than the Knothe-Rosenblatt upper bound") {
  const emdgrid::GridLayout<2> layout({6, 6});
  const std::size_t n = layout.node_count();

  const std::vector<double> a =
      emdgrid::generate_random_histogram<double>(n, 31);
  const std::vector<double> b =
      emdgrid::generate_random_histogram<double>(n, 97);

  const emdgrid::GridDataView<2, double> h1(layout, std::span(a));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(b));

  const double exact = emdgrid::emd_lemon(h1, h2, kNs);
  const double bound =
      emdgrid::knothe_rosenblatt(h1, h2, emdgrid::GroundMetric::L1);

  CHECK(exact <= bound + 1e-6);
}

// ============================================================================
//  Transport plans
// ============================================================================

TEST_CASE("emd_lemon 2D L1: plan is a valid coupling and reprices to cost") {
  const emdgrid::GridLayout<2> layout({4, 4});
  const std::size_t n = layout.node_count();

  const std::vector<double> a =
      emdgrid::generate_random_histogram<double>(n, 111);
  const std::vector<double> b =
      emdgrid::generate_random_histogram<double>(n, 222);

  const emdgrid::GridDataView<2, double> h1(layout, std::span(a));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(b));

  emdgrid::SparseTransportPlan<> plan;
  const double dist = emdgrid::emd_lemon(h1, h2, kNs, &plan);
  check_plan(layout, a, b, plan, dist, emdgrid::GroundMetric::L1);
}

TEST_CASE("emd_lemon 2D W2^2: plan is a valid coupling") {
  const emdgrid::GridLayout<2> layout({4, 4});
  const std::size_t n = layout.node_count();

  const std::vector<double> a =
      emdgrid::generate_random_histogram<double>(n, 4242);
  const std::vector<double> b =
      emdgrid::generate_random_histogram<double>(n, 2424);

  const emdgrid::GridDataView<2, double> h1(layout, std::span(a));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(b));

  emdgrid::SparseTransportPlan<> plan;
  const double dist = emdgrid::emd_lemon(
      h1, h2, emdgrid::GroundMetric::SqEuclidean, kCs, &plan);
  check_plan(layout, a, b, plan, dist, emdgrid::GroundMetric::SqEuclidean);
}

TEST_CASE("emd_lemon 1D: identical histograms yield a diagonal plan") {
  const emdgrid::GridLayout<1> layout({4});
  const std::vector<double> v = {0.25, 0.25, 0.25, 0.25};
  const emdgrid::GridDataView<1, double> h(layout, std::span(v));

  emdgrid::SparseTransportPlan<> plan;
  const double dist = emdgrid::emd_lemon(h, h, kNs, &plan);

  CHECK(dist == doctest::Approx(0.0).epsilon(1e-9));
  REQUIRE(plan.flow.size() == 4);
  for (std::size_t k = 0; k < plan.flow.size(); ++k) {
    CHECK(plan.source[k] == plan.target[k]);
    CHECK(plan.flow[k] == doctest::Approx(0.25).epsilon(1e-9));
  }
}

TEST_CASE("emd_lemon: a reused plan is cleared before being refilled") {
  const emdgrid::GridLayout<1> layout({3});
  const std::vector<double> a = {1.0, 0.0, 0.0};
  const std::vector<double> b = {0.0, 0.0, 1.0};
  const emdgrid::GridDataView<1, double> h1(layout, std::span(a));
  const emdgrid::GridDataView<1, double> h2(layout, std::span(b));

  emdgrid::SparseTransportPlan<> plan;
  const double first = emdgrid::emd_lemon(h1, h2, kNs, &plan);
  const std::size_t first_size = plan.flow.size();
  const double second = emdgrid::emd_lemon(h1, h2, kNs, &plan);

  CHECK(first == doctest::Approx(second).epsilon(1e-9));
  CHECK(plan.flow.size() == first_size);
}

// ============================================================================
//  Custom cost functors and precision
// ============================================================================

TEST_CASE("emd_lemon: custom separable cost functor is honoured") {
  const emdgrid::GridLayout<2> layout({3, 3});
  std::vector<double> a(9, 0.0);
  std::vector<double> b(9, 0.0);
  a[layout.node({0, 0})] = 1.0;
  b[layout.node({2, 1})] = 1.0;

  const emdgrid::GridDataView<2, double> h1(layout, std::span(a));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(b));

  // 2 steps along axis 0 at cost 2 each, plus 1 step along axis 1.
  CHECK(emdgrid::emd_lemon(h1, h2, WeightedL1{}, kNs) ==
        doctest::Approx(5.0).epsilon(1e-6));
}

TEST_CASE("emd_lemon: float CompScalar tracks the double result") {
  const emdgrid::GridLayout<2> layout({4, 4});
  const std::size_t n = layout.node_count();

  const std::vector<double> a =
      emdgrid::generate_random_histogram<double>(n, 555);
  const std::vector<double> b =
      emdgrid::generate_random_histogram<double>(n, 666);

  const emdgrid::GridDataView<2, double> h1(layout, std::span(a));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(b));

  const double as_double = emdgrid::emd_lemon<2, double, double>(h1, h2, kNs);
  const float as_float = emdgrid::emd_lemon<2, double, float>(h1, h2, kNs);

  CHECK(static_cast<double>(as_float) ==
        doctest::Approx(as_double).epsilon(1e-4));
}

// ============================================================================
//  Input validation
// ============================================================================

TEST_CASE("emd_lemon 2D: shape mismatch throws") {
  const emdgrid::GridLayout<2> la({3, 3});
  const emdgrid::GridLayout<2> lb({3, 4});
  const std::vector<double> a(9, 1.0 / 9.0);
  const std::vector<double> b(12, 1.0 / 12.0);
  const emdgrid::GridDataView<2, double> h1(la, std::span(a));
  const emdgrid::GridDataView<2, double> h2(lb, std::span(b));

  CHECK_THROWS_AS(void(emdgrid::emd_lemon(h1, h2, kNs)),
                  std::invalid_argument);
}

TEST_CASE("emd_lemon 2D: unnormalized mass throws") {
  const emdgrid::GridLayout<2> layout({2, 2});
  const std::vector<double> a = {0.5, 0.5, 0.0, 0.0};
  const std::vector<double> b = {0.2, 0.2, 0.2, 0.2};
  const emdgrid::GridDataView<2, double> h1(layout, std::span(a));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(b));

  CHECK_THROWS_AS(void(emdgrid::emd_lemon(h1, h2, kNs)),
                  std::invalid_argument);
}

TEST_CASE("emd_lemon 1D: a negative bin throws") {
  const emdgrid::GridLayout<1> layout({3});
  const std::vector<double> a = {1.2, -0.2, 0.0};
  const std::vector<double> b = {0.0, 0.0, 1.0};
  const emdgrid::GridDataView<1, double> h1(layout, std::span(a));
  const emdgrid::GridDataView<1, double> h2(layout, std::span(b));

  CHECK_THROWS_AS(void(emdgrid::emd_lemon(h1, h2, kNs)),
                  std::invalid_argument);
}

// ============================================================================
//  The implicit-arc bipartite digraph itself
// ============================================================================

TEST_CASE("FullBipartiteDigraph: arcs and endpoints are consistent") {
  const emdgrid::detail::FullBipartiteDigraph graph(3, 4);
  using Graph = emdgrid::detail::FullBipartiteDigraph;

  CHECK_EQ(graph.nodeNum(), 7);
  CHECK_EQ(graph.arcNum(), 12);

  int seen = 0;
  std::vector<std::vector<int>> hits(3, std::vector<int>(4, 0));
  for (Graph::ArcIt a(graph); a != lemon::INVALID; ++a) {
    const int i = graph.source_index(a);
    const int j = graph.target_index(a);
    REQUIRE(i >= 0);
    REQUIRE(i < 3);
    REQUIRE(j >= 0);
    REQUIRE(j < 4);
    hits[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]++;
    CHECK(Graph::id(graph.source(a)) == i);
    CHECK(Graph::id(graph.target(a)) == 3 + j);
    ++seen;
  }
  CHECK_EQ(seen, 12);
  for (const auto& row : hits) {
    for (const int count : row) {
      CHECK_EQ(count, 1);
    }
  }
}

TEST_CASE("FullBipartiteDigraph: out- and in-arc iteration") {
  const emdgrid::detail::FullBipartiteDigraph graph(3, 4);
  using Graph = emdgrid::detail::FullBipartiteDigraph;

  for (int i = 0; i < 3; ++i) {
    int out_count = 0;
    for (Graph::OutArcIt a(graph, Graph::source_node(i)); a != lemon::INVALID;
         ++a) {
      CHECK(graph.source_index(a) == i);
      ++out_count;
    }
    CHECK_EQ(out_count, 4);

    // Source-side nodes have no incoming arcs.
    int in_count = 0;
    for (Graph::InArcIt a(graph, Graph::source_node(i)); a != lemon::INVALID;
         ++a) {
      ++in_count;
    }
    CHECK_EQ(in_count, 0);
  }

  for (int j = 0; j < 4; ++j) {
    int in_count = 0;
    for (Graph::InArcIt a(graph, graph.target_node(j)); a != lemon::INVALID;
         ++a) {
      CHECK(graph.target_index(a) == j);
      ++in_count;
    }
    CHECK_EQ(in_count, 3);

    // Target-side nodes have no outgoing arcs.
    int out_count = 0;
    for (Graph::OutArcIt a(graph, graph.target_node(j)); a != lemon::INVALID;
         ++a) {
      ++out_count;
    }
    CHECK_EQ(out_count, 0);
  }
}

TEST_CASE("FullBipartiteDigraph: findArc round-trips arc endpoints") {
  const emdgrid::detail::FullBipartiteDigraph graph(2, 3);
  using Graph = emdgrid::detail::FullBipartiteDigraph;

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      const Graph::Arc a =
          graph.findArc(Graph::source_node(i), graph.target_node(j));
      REQUIRE(a != lemon::INVALID);
      CHECK(graph.source_index(a) == i);
      CHECK(graph.target_index(a) == j);
      // A complete bipartite graph has exactly one arc per pair.
      CHECK(graph.findArc(Graph::source_node(i), graph.target_node(j), a) ==
            lemon::INVALID);
    }
  }

  // No arc runs backwards, from the target side to the source side.
  CHECK(graph.findArc(graph.target_node(0), Graph::source_node(0)) ==
        lemon::INVALID);
}

TEST_SUITE_END();
