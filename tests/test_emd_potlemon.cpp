#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include <doctest/doctest.h>  // NOLINT(build/include_order)

#include "emdgrid/detail/potlemon/full_bipartitegraph.h"
#include "emdgrid/emd_l1.hpp"
#include "emdgrid/emd_lemon.hpp"
#include "emdgrid/emd_potlemon.hpp"
#include "emdgrid/emdgrid.hpp"
#include "emdgrid/mcf_lemon_l1.hpp"
#include "emdgrid/utils.hpp"

TEST_SUITE_BEGIN("emd_potlemon");

namespace {

constexpr auto kL1 = emdgrid::GroundMetric::L1;
constexpr auto kSq = emdgrid::GroundMetric::SqEuclidean;

/// Ground cost between two bins, summed over the axes.
template <std::size_t Dim>
double ground_cost(const emdgrid::GridLayout<Dim>& layout, uint32_t u,
                   uint32_t v, emdgrid::GroundMetric metric) {
  const auto cu = layout.coordinates(static_cast<std::ptrdiff_t>(u));
  const auto cv = layout.coordinates(static_cast<std::ptrdiff_t>(v));
  double total = 0.0;
  for (std::size_t k = 0; k < Dim; ++k) {
    const auto diff = static_cast<double>(cu[k] - cv[k]);
    total += (metric == kL1) ? std::abs(diff) : (diff * diff);
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
    recomputed +=
        f * ground_cost(layout, plan.source[k], plan.target[k], metric);
  }

  CHECK(flow_sum == doctest::Approx(1.0).epsilon(1e-5));
  CHECK(recomputed == doctest::Approx(dist).epsilon(1e-4));
  for (std::size_t i = 0; i < n; ++i) {
    CHECK(src_margin[i] == doctest::Approx(h1[i]).epsilon(1e-4));
    CHECK(tgt_margin[i] == doctest::Approx(h2[i]).epsilon(1e-4));
  }
}

}  // namespace

// ============================================================================
//  Closed-form cases
// ============================================================================

TEST_CASE("emd_potlemon 1D: identical histograms give zero") {
  const emdgrid::GridLayout<1> layout({5});
  const std::vector<double> v = {0.1, 0.2, 0.4, 0.2, 0.1};
  const emdgrid::GridDataView<1, double> h(layout, std::span(v));

  CHECK(emdgrid::emd_potlemon(h, h, kL1) ==
        doctest::Approx(0.0).epsilon(1e-9));
  CHECK(emdgrid::emd_potlemon(h, h, kSq) ==
        doctest::Approx(0.0).epsilon(1e-9));
}

TEST_CASE("emd_potlemon 1D: unit shift by two bins costs 2 under L1") {
  const emdgrid::GridLayout<1> layout({3});
  const std::vector<double> a = {1.0, 0.0, 0.0};
  const std::vector<double> b = {0.0, 0.0, 1.0};
  const emdgrid::GridDataView<1, double> h1(layout, std::span(a));
  const emdgrid::GridDataView<1, double> h2(layout, std::span(b));

  CHECK(emdgrid::emd_potlemon(h1, h2, kL1) ==
        doctest::Approx(2.0).epsilon(1e-6));
  CHECK(emdgrid::emd_potlemon(h1, h2, kSq) ==
        doctest::Approx(4.0).epsilon(1e-6));
}

TEST_CASE("emd_potlemon 3D: diagonal shift costs 3 under both metrics") {
  const emdgrid::GridLayout<3> layout({3, 3, 3});
  std::vector<double> a(27, 0.0);
  std::vector<double> b(27, 0.0);
  a[layout.node({0, 0, 0})] = 1.0;
  b[layout.node({1, 1, 1})] = 1.0;

  const emdgrid::GridDataView<3, double> h1(layout, std::span(a));
  const emdgrid::GridDataView<3, double> h2(layout, std::span(b));

  CHECK(emdgrid::emd_potlemon(h1, h2, kL1) ==
        doctest::Approx(3.0).epsilon(1e-6));
  CHECK(emdgrid::emd_potlemon(h1, h2, kSq) ==
        doctest::Approx(3.0).epsilon(1e-6));
}

TEST_CASE("emd_potlemon 1D: symmetric in its arguments") {
  const emdgrid::GridLayout<1> layout({4});
  const std::vector<double> a = {0.5, 0.5, 0.0, 0.0};
  const std::vector<double> b = {0.0, 0.0, 0.5, 0.5};
  const emdgrid::GridDataView<1, double> ha(layout, std::span(a));
  const emdgrid::GridDataView<1, double> hb(layout, std::span(b));

  CHECK(emdgrid::emd_potlemon(ha, hb, kL1) ==
        doctest::Approx(emdgrid::emd_potlemon(hb, ha, kL1)).epsilon(1e-6));
}

// ============================================================================
//  Agreement with the other solvers
//
//  emd_potlemon runs on potlemon's Value=double network simplex directly
//  (matching POT's own EMD_wrap_lazy, see emd_potlemon.hpp), while emd_lemon
//  still quantizes onto an integer lattice. They are independent solves of
//  the same real-valued problem, not the identical quantized LP, so they are
//  only expected to agree to ordinary cross-algorithm numerical precision,
//  not bit-for-bit.
// ============================================================================

TEST_CASE("emd_potlemon 3D L1: matches emd_lemon and emd_l1") {
  const emdgrid::GridLayout<3> layout({4, 4, 4});
  const std::size_t n = layout.node_count();

  const std::vector<double> a =
      emdgrid::generate_random_histogram<double>(n, 12345);
  const std::vector<double> b =
      emdgrid::generate_random_histogram<double>(n, 67890);

  const emdgrid::GridDataView<3, double> h1(layout, std::span(a));
  const emdgrid::GridDataView<3, double> h2(layout, std::span(b));

  const double potlemon = emdgrid::emd_potlemon(h1, h2, kL1);

  CHECK(potlemon ==
        doctest::Approx(emdgrid::emd_lemon(h1, h2, kL1)).epsilon(1e-4));
  CHECK(potlemon == doctest::Approx(emdgrid::emd_l1(h1, h2)).epsilon(1e-4));
}

TEST_CASE("emd_potlemon 2D W2^2: matches emd_lemon and mcf_dpartion") {
  const emdgrid::GridLayout<2> layout({5, 5});
  const std::size_t n = layout.node_count();

  const std::vector<double> a =
      emdgrid::generate_random_histogram<double>(n, 7);
  const std::vector<double> b =
      emdgrid::generate_random_histogram<double>(n, 9);

  const emdgrid::GridDataView<2, double> h1(layout, std::span(a));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(b));

  const double potlemon = emdgrid::emd_potlemon(h1, h2, kSq);

  CHECK(potlemon ==
        doctest::Approx(emdgrid::emd_lemon(h1, h2, kSq)).epsilon(1e-4));
  CHECK(potlemon == doctest::Approx(emdgrid::mcf_dpartion(
                                        h1, h2, kSq,
                                        emdgrid::McfLemonAlgorithm::
                                            NetworkSimplex))
                        .epsilon(1e-4));
}

TEST_CASE("emd_potlemon 3D: agrees with emd_lemon across several seeds") {
  const emdgrid::GridLayout<3> layout({4, 4, 4});
  const std::size_t n = layout.node_count();

  for (unsigned int seed = 1; seed <= 5; ++seed) {
    const std::vector<double> a =
        emdgrid::generate_random_histogram<double>(n, seed);
    const std::vector<double> b =
        emdgrid::generate_random_histogram<double>(n, seed + 1000);

    const emdgrid::GridDataView<3, double> h1(layout, std::span(a));
    const emdgrid::GridDataView<3, double> h2(layout, std::span(b));

    CAPTURE(seed);
    CHECK(emdgrid::emd_potlemon(h1, h2, kL1) ==
          doctest::Approx(emdgrid::emd_lemon(h1, h2, kL1)).epsilon(1e-4));
    CHECK(emdgrid::emd_potlemon(h1, h2, kSq) ==
          doctest::Approx(emdgrid::emd_lemon(h1, h2, kSq)).epsilon(1e-4));
  }
}

// ============================================================================
//  Transport plans
// ============================================================================

TEST_CASE("emd_potlemon 2D L1: plan is a valid coupling and reprices") {
  const emdgrid::GridLayout<2> layout({4, 4});
  const std::size_t n = layout.node_count();

  const std::vector<double> a =
      emdgrid::generate_random_histogram<double>(n, 111);
  const std::vector<double> b =
      emdgrid::generate_random_histogram<double>(n, 222);

  const emdgrid::GridDataView<2, double> h1(layout, std::span(a));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(b));

  emdgrid::SparseTransportPlan<> plan;
  const double dist = emdgrid::emd_potlemon(h1, h2, kL1, &plan);
  check_plan(layout, a, b, plan, dist, kL1);
}

TEST_CASE("emd_potlemon 2D W2^2: plan is a valid coupling and reprices") {
  const emdgrid::GridLayout<2> layout({4, 4});
  const std::size_t n = layout.node_count();

  const std::vector<double> a =
      emdgrid::generate_random_histogram<double>(n, 4242);
  const std::vector<double> b =
      emdgrid::generate_random_histogram<double>(n, 2424);

  const emdgrid::GridDataView<2, double> h1(layout, std::span(a));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(b));

  emdgrid::SparseTransportPlan<> plan;
  const double dist = emdgrid::emd_potlemon(h1, h2, kSq, &plan);
  check_plan(layout, a, b, plan, dist, kSq);
}

TEST_CASE("emd_potlemon 1D: identical histograms yield a diagonal plan") {
  const emdgrid::GridLayout<1> layout({4});
  const std::vector<double> v = {0.25, 0.25, 0.25, 0.25};
  const emdgrid::GridDataView<1, double> h(layout, std::span(v));

  emdgrid::SparseTransportPlan<> plan;
  const double dist = emdgrid::emd_potlemon(h, h, kL1, &plan);

  CHECK(dist == doctest::Approx(0.0).epsilon(1e-9));
  REQUIRE(plan.flow.size() == 4);
  for (std::size_t k = 0; k < plan.flow.size(); ++k) {
    CHECK_EQ(plan.source[k], plan.target[k]);
    CHECK(plan.flow[k] == doctest::Approx(0.25).epsilon(1e-9));
  }
}

TEST_CASE("emd_potlemon: a reused plan is cleared before being refilled") {
  const emdgrid::GridLayout<1> layout({3});
  const std::vector<double> a = {1.0, 0.0, 0.0};
  const std::vector<double> b = {0.0, 0.0, 1.0};
  const emdgrid::GridDataView<1, double> h1(layout, std::span(a));
  const emdgrid::GridDataView<1, double> h2(layout, std::span(b));

  emdgrid::SparseTransportPlan<> plan;
  const double first = emdgrid::emd_potlemon(h1, h2, kL1, &plan);
  const std::size_t first_size = plan.flow.size();
  const double second = emdgrid::emd_potlemon(h1, h2, kL1, &plan);

  CHECK(first == doctest::Approx(second).epsilon(1e-9));
  CHECK(plan.flow.size() == first_size);
}

TEST_CASE("emd_potlemon: float CompScalar tracks the double result") {
  const emdgrid::GridLayout<2> layout({4, 4});
  const std::size_t n = layout.node_count();

  const std::vector<double> a =
      emdgrid::generate_random_histogram<double>(n, 555);
  const std::vector<double> b =
      emdgrid::generate_random_histogram<double>(n, 666);

  const emdgrid::GridDataView<2, double> h1(layout, std::span(a));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(b));

  const double as_double =
      emdgrid::emd_potlemon<2, double, double>(h1, h2, kL1);
  const float as_float = emdgrid::emd_potlemon<2, double, float>(h1, h2, kL1);

  CHECK(static_cast<double>(as_float) ==
        doctest::Approx(as_double).epsilon(1e-4));
}

// ============================================================================
//  Input validation
// ============================================================================

TEST_CASE("emd_potlemon 2D: shape mismatch throws") {
  const emdgrid::GridLayout<2> la({3, 3});
  const emdgrid::GridLayout<2> lb({3, 4});
  const std::vector<double> a(9, 1.0 / 9.0);
  const std::vector<double> b(12, 1.0 / 12.0);
  const emdgrid::GridDataView<2, double> h1(la, std::span(a));
  const emdgrid::GridDataView<2, double> h2(lb, std::span(b));

  CHECK_THROWS_AS(void(emdgrid::emd_potlemon(h1, h2, kL1)),
                  std::invalid_argument);
}

TEST_CASE("emd_potlemon 2D: unnormalized mass throws") {
  const emdgrid::GridLayout<2> layout({2, 2});
  const std::vector<double> a = {0.5, 0.5, 0.0, 0.0};
  const std::vector<double> b = {0.2, 0.2, 0.2, 0.2};
  const emdgrid::GridDataView<2, double> h1(layout, std::span(a));
  const emdgrid::GridDataView<2, double> h2(layout, std::span(b));

  CHECK_THROWS_AS(void(emdgrid::emd_potlemon(h1, h2, kL1)),
                  std::invalid_argument);
}

TEST_CASE("emd_potlemon 1D: a negative bin throws") {
  const emdgrid::GridLayout<1> layout({3});
  const std::vector<double> a = {1.2, -0.2, 0.0};
  const std::vector<double> b = {0.0, 0.0, 1.0};
  const emdgrid::GridDataView<1, double> h1(layout, std::span(a));
  const emdgrid::GridDataView<1, double> h2(layout, std::span(b));

  CHECK_THROWS_AS(void(emdgrid::emd_potlemon(h1, h2, kL1)),
                  std::invalid_argument);
}

// ============================================================================
//  The implicit-arc bipartite digraph
//
//  NetworkSimplexSimple reconstructs arc endpoints from the id arithmetic
//  below when EndpointStorageMode::ArcEndpoints is in force, so these
//  invariants are load-bearing rather than cosmetic.
// ============================================================================

TEST_CASE("potlemon FullBipartiteDigraph: arcs and endpoints are consistent") {
  const potlemon::FullBipartiteDigraph graph(3, 4);
  using Graph = potlemon::FullBipartiteDigraph;

  CHECK_EQ(graph.nodeNum(), 7);
  CHECK_EQ(graph.arcNum(), 12);

  int seen = 0;
  std::vector<std::vector<int>> hits(3, std::vector<int>(4, 0));
  Graph::Arc a = 0;
  graph.first(a);
  for (; a != -1; Graph::next(a)) {
    const int i = graph.source(a);
    const int j = graph.target(a) - 3;
    REQUIRE(i >= 0);
    REQUIRE(i < 3);
    REQUIRE(j >= 0);
    REQUIRE(j < 4);
    // The id layout the solver assumes: arc = i * n2 + j.
    CHECK_EQ(a, (i * 4) + j);
    hits[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]++;
    ++seen;
  }
  CHECK_EQ(seen, 12);
  for (const auto& row : hits) {
    for (const int count : row) {
      CHECK_EQ(count, 1);
    }
  }
}

TEST_CASE("potlemon FullBipartiteDigraph: out- and in-arc iteration") {
  const potlemon::FullBipartiteDigraph graph(3, 4);
  using Graph = potlemon::FullBipartiteDigraph;

  for (int i = 0; i < 3; ++i) {
    int out_count = 0;
    Graph::Arc a = 0;
    graph.firstOut(a, i);
    for (; a != -1; graph.nextOut(a)) {
      CHECK_EQ(graph.source(a), i);
      ++out_count;
    }
    CHECK_EQ(out_count, 4);

    // Source-side nodes have no incoming arcs.
    int in_count = 0;
    graph.firstIn(a, i);
    for (; a != -1; graph.nextIn(a)) {
      ++in_count;
    }
    CHECK_EQ(in_count, 0);
  }

  for (int j = 0; j < 4; ++j) {
    int in_count = 0;
    Graph::Arc a = 0;
    graph.firstIn(a, 3 + j);
    for (; a != -1; graph.nextIn(a)) {
      CHECK_EQ(graph.target(a), 3 + j);
      ++in_count;
    }
    CHECK_EQ(in_count, 3);

    // Target-side nodes have no outgoing arcs.
    int out_count = 0;
    graph.firstOut(a, 3 + j);
    for (; a != -1; graph.nextOut(a)) {
      ++out_count;
    }
    CHECK_EQ(out_count, 0);
  }
}

TEST_CASE("potlemon FullBipartiteDigraph: findArc round-trips endpoints") {
  const potlemon::FullBipartiteDigraph graph(2, 3);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      const auto a = graph.findArc(i, 2 + j);
      REQUIRE(a != -1);
      CHECK_EQ(graph.source(a), i);
      CHECK_EQ(graph.target(a), 2 + j);
      // A complete bipartite graph has exactly one arc per pair.
      CHECK_EQ(graph.findArc(i, 2 + j, a), -1);
    }
  }

  // No arc runs backwards, from the target side to the source side.
  CHECK_EQ(graph.findArc(2, 0), -1);
}

TEST_SUITE_END();

TEST_CASE("emd_potlemon 3D: agrees with emd_lemon on a larger grid") {
  const emdgrid::GridLayout<3> layout({8, 8, 8});
  const std::size_t n = layout.node_count();

  double max_rel_err_l1 = 0.0;
  double max_rel_err_sq = 0.0;
  for (unsigned int seed = 1; seed <= 20; ++seed) {
    const std::vector<double> a =
        emdgrid::generate_random_histogram<double>(n, (seed * 13) + 1);
    const std::vector<double> b =
        emdgrid::generate_random_histogram<double>(n, (seed * 17) + 7000);

    const emdgrid::GridDataView<3, double> h1(layout, std::span(a));
    const emdgrid::GridDataView<3, double> h2(layout, std::span(b));

    const double potlemon_l1 = emdgrid::emd_potlemon(h1, h2, kL1);
    const double ref_l1 = emdgrid::emd_lemon(h1, h2, kL1);
    const double potlemon_sq = emdgrid::emd_potlemon(h1, h2, kSq);
    const double ref_sq = emdgrid::emd_lemon(h1, h2, kSq);

    max_rel_err_l1 =
        std::max(max_rel_err_l1, std::abs(potlemon_l1 - ref_l1) / ref_l1);
    max_rel_err_sq =
        std::max(max_rel_err_sq, std::abs(potlemon_sq - ref_sq) / ref_sq);
  }
  MESSAGE("max_rel_err_l1=", max_rel_err_l1, " max_rel_err_sq=",
          max_rel_err_sq);
  CHECK(max_rel_err_l1 < 1e-4);
  CHECK(max_rel_err_sq < 1e-4);
}

TEST_CASE("emd_potlemon 3D: plan mass is conserved on a larger grid") {
  const emdgrid::GridLayout<3> layout({8, 8, 8});
  const std::size_t n = layout.node_count();

  double max_flow_err = 0.0;
  double max_margin_err = 0.0;
  for (unsigned int seed = 1; seed <= 10; ++seed) {
    const std::vector<double> a =
        emdgrid::generate_random_histogram<double>(n, (seed * 29) + 3);
    const std::vector<double> b =
        emdgrid::generate_random_histogram<double>(n, (seed * 31) + 9000);

    const emdgrid::GridDataView<3, double> h1(layout, std::span(a));
    const emdgrid::GridDataView<3, double> h2(layout, std::span(b));

    emdgrid::SparseTransportPlan<> plan;
    (void)emdgrid::emd_potlemon(h1, h2, kSq, &plan);

    std::vector<double> row_sum(n, 0.0);
    std::vector<double> col_sum(n, 0.0);
    double total_flow = 0.0;
    for (std::size_t k = 0; k < plan.flow.size(); ++k) {
      total_flow += plan.flow[k];
      row_sum[plan.source[k]] += plan.flow[k];
      col_sum[plan.target[k]] += plan.flow[k];
    }
    max_flow_err = std::max(max_flow_err, std::abs(total_flow - 1.0));
    for (std::size_t i = 0; i < n; ++i) {
      max_margin_err =
          std::max(max_margin_err, std::abs(row_sum[i] - a[i]));
      max_margin_err =
          std::max(max_margin_err, std::abs(col_sum[i] - b[i]));
    }
  }
  MESSAGE("max_flow_err=", max_flow_err, " max_margin_err=", max_margin_err);
  CHECK(max_flow_err < 1e-6);
  CHECK(max_margin_err < 1e-6);
}
