// Tests that every solver honours its CompScalar template argument: the
// reported cost and the transport-plan flows must carry that type, and a
// float computation must agree with the double reference to within float
// resolution.

#include <cmath>
#include <concepts>
#include <span>
#include <type_traits>
#include <vector>

#include <doctest/doctest.h>  // NOLINT(build/include_order)

#include "emdgrid/emd_1d.hpp"
#include "emdgrid/emd_l1.hpp"
#include "emdgrid/emdgrid.hpp"
#include "emdgrid/greedy_emd_l1.hpp"
#include "emdgrid/knothe_rosenblatt.hpp"
#include "emdgrid/mcf_l1.hpp"
#include "emdgrid/mcf_lemon_l1.hpp"
#include "emdgrid/mcf_potlemon_l1.hpp"
#include "emdgrid/opencv_emd.hpp"
#include "emdgrid/utils.hpp"

TEST_SUITE_BEGIN("comp_scalar");

namespace {

// Relative slack between a float and a double run. Single precision carries
// ~7 significant digits and the supply quantization amplifies that, so 1e-4
// relative is the useful agreement target.
constexpr double float_vs_double_rel_tol = 1e-4;

// A 3-D grid small enough to keep opencv_emd's O(n^2) simplex tractable.
constexpr std::size_t grid_extent = 4;

struct Fixture {
  emdgrid::GridLayout<3> layout{{grid_extent, grid_extent, grid_extent}};
  std::vector<double> h1_data;
  std::vector<double> h2_data;

  Fixture()
      : h1_data(emdgrid::generate_random_histogram<double>(
            layout.node_count(), 42)),
        h2_data(emdgrid::generate_random_histogram<double>(
            layout.node_count(), 1337)) {}

  [[nodiscard]] emdgrid::GridDataView<3, double> h1() const {
    return {layout, std::span(h1_data)};
  }
  [[nodiscard]] emdgrid::GridDataView<3, double> h2() const {
    return {layout, std::span(h2_data)};
  }
};

/// Checks a float result against the double reference and returns the plan
/// total mass so callers can assert the plan is still a valid coupling.
void check_close(double float_cost, double double_cost) {
  CHECK(float_cost ==
        doctest::Approx(double_cost).epsilon(float_vs_double_rel_tol));
}

template <std::floating_point CompScalar>
CompScalar plan_mass(const emdgrid::SparseTransportPlan<CompScalar>& plan) {
  CompScalar total{0};
  for (const CompScalar f : plan.flow) {
    total += f;
  }
  return total;
}

}  // namespace

// ============================================================================
//  The plan flow type follows CompScalar
// ============================================================================

TEST_CASE("SparseTransportPlan flow type follows CompScalar") {
  static_assert(
      std::is_same_v<decltype(emdgrid::SparseTransportPlan<float>::flow),
                     std::vector<float>>);
  static_assert(
      std::is_same_v<decltype(emdgrid::SparseTransportPlan<double>::flow),
                     std::vector<double>>);
  static_assert(std::is_same_v<emdgrid::SparseTransportPlan<>,
                               emdgrid::SparseTransportPlan<double>>);
}

// ============================================================================
//  1-D solvers
// ============================================================================

TEST_CASE("emd_1d in float: exact analytic value and float plan") {
  const emdgrid::GridLayout<1> layout({3});
  const std::vector<double> h1v = {1.0, 0.0, 0.0};
  const std::vector<double> h2v = {0.0, 0.0, 1.0};
  const emdgrid::GridDataView<1, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<1, double> h2(layout, std::span(h2v));

  emdgrid::SparseTransportPlan<float> plan;
  const auto cost = emdgrid::emd_1d<double, float>(h1, h2, &plan);
  static_assert(std::is_same_v<decltype(cost), const float>);

  CHECK(cost == doctest::Approx(2.0F));
  CHECK(plan_mass(plan) == doctest::Approx(1.0F));
}

TEST_CASE("emd_sqeuclidean_1d in float matches the double reference") {
  const emdgrid::GridLayout<1> layout({6});
  const std::vector<double> h1v = {0.4, 0.1, 0.2, 0.1, 0.1, 0.1};
  const std::vector<double> h2v = {0.1, 0.1, 0.1, 0.2, 0.1, 0.4};
  const emdgrid::GridDataView<1, double> h1(layout, std::span(h1v));
  const emdgrid::GridDataView<1, double> h2(layout, std::span(h2v));

  const float cost_f = emdgrid::emd_sqeuclidean_1d<double, float>(h1, h2);
  const double cost_d = emdgrid::emd_sqeuclidean_1d(h1, h2);
  check_close(cost_f, cost_d);
}

// ============================================================================
//  Grid solvers: float agrees with double
// ============================================================================

TEST_CASE("emd_l1 3D in float matches the double reference") {
  const Fixture fx;
  emdgrid::SparseTransportPlan<float> plan_f;
  const float cost_f =
      emdgrid::emd_l1<3, double, float>(fx.h1(), fx.h2(), &plan_f);
  const double cost_d = emdgrid::emd_l1(fx.h1(), fx.h2());

  check_close(cost_f, cost_d);
  CHECK(plan_mass(plan_f) == doctest::Approx(1.0F).epsilon(1e-4));
}

TEST_CASE("greedy_emd_l1_approx 3D in float matches the double reference") {
  const Fixture fx;
  emdgrid::SparseTransportPlan<float> plan_f;
  const float cost_f = emdgrid::greedy_emd_l1_approx<3, double, float>(
      fx.h1(), fx.h2(), &plan_f);
  const double cost_d = emdgrid::greedy_emd_l1_approx(fx.h1(), fx.h2());

  check_close(cost_f, cost_d);
  CHECK(plan_mass(plan_f) == doctest::Approx(1.0F).epsilon(1e-4));
}

TEST_CASE("mcf_l1 3D in float matches the double reference") {
  const Fixture fx;
  emdgrid::SparseTransportPlan<float> plan_f;
  const float cost_f =
      emdgrid::mcf_l1<3, double, float>(fx.h1(), fx.h2(), &plan_f);
  const double cost_d = emdgrid::mcf_l1(fx.h1(), fx.h2());

  check_close(cost_f, cost_d);
  CHECK(plan_mass(plan_f) == doctest::Approx(1.0F).epsilon(1e-4));
}

TEST_CASE("mcf_lemon_l1 3D in float matches the double reference") {
  const Fixture fx;
  for (const auto algo : {emdgrid::McfLemonAlgorithm::NetworkSimplex,
                          emdgrid::McfLemonAlgorithm::CostScaling}) {
    emdgrid::SparseTransportPlan<float> plan_f;
    const float cost_f = emdgrid::mcf_lemon_l1<3, double, float>(
        fx.h1(), fx.h2(), algo, &plan_f);
    const double cost_d = emdgrid::mcf_lemon_l1(fx.h1(), fx.h2(), algo);

    check_close(cost_f, cost_d);
    CHECK(plan_mass(plan_f) == doctest::Approx(1.0F).epsilon(1e-4));
  }
}

TEST_CASE("mcf_potlemon_l1 3D in float matches the double reference") {
  const Fixture fx;
  emdgrid::SparseTransportPlan<float> plan_f;
  const float cost_f =
      emdgrid::mcf_potlemon_l1<3, double, float>(fx.h1(), fx.h2(), &plan_f);
  const double cost_d = emdgrid::mcf_potlemon_l1(fx.h1(), fx.h2());

  check_close(cost_f, cost_d);
  CHECK(plan_mass(plan_f) == doctest::Approx(1.0F).epsilon(1e-4));
}

TEST_CASE("mcf_dpartion 3D in float matches the double reference") {
  const Fixture fx;
  for (const auto metric :
       {emdgrid::GroundMetric::L1, emdgrid::GroundMetric::SqEuclidean}) {
    emdgrid::SparseTransportPlan<float> plan_f;
    const float cost_f = emdgrid::mcf_dpartion<3, double, float>(
        fx.h1(), fx.h2(), metric,
        emdgrid::McfLemonAlgorithm::NetworkSimplex, &plan_f);
    const double cost_d = emdgrid::mcf_dpartion(
        fx.h1(), fx.h2(), metric,
        emdgrid::McfLemonAlgorithm::NetworkSimplex);

    check_close(cost_f, cost_d);
    CHECK(plan_mass(plan_f) == doctest::Approx(1.0F).epsilon(1e-4));
  }
}

TEST_CASE("knothe_rosenblatt 3D in float matches the double reference") {
  const Fixture fx;
  for (const auto metric :
       {emdgrid::GroundMetric::L1, emdgrid::GroundMetric::SqEuclidean}) {
    emdgrid::SparseTransportPlan<float> plan_f;
    const float cost_f = emdgrid::knothe_rosenblatt<3, double, float>(
        fx.h1(), fx.h2(), metric, {}, &plan_f);
    const double cost_d =
        emdgrid::knothe_rosenblatt(fx.h1(), fx.h2(), metric);

    check_close(cost_f, cost_d);
    CHECK(plan_mass(plan_f) == doctest::Approx(1.0F).epsilon(1e-4));
  }
}

TEST_CASE("opencv_emd 3D in float matches the double reference") {
  const Fixture fx;
  emdgrid::SparseTransportPlan<float> plan_f;
  const float cost_f = emdgrid::opencv_emd<3, double, float>(
      fx.h1(), fx.h2(), emdgrid::GroundMetric::L1, &plan_f);
  const double cost_d =
      emdgrid::opencv_emd(fx.h1(), fx.h2(), emdgrid::GroundMetric::L1);

  check_close(cost_f, cost_d);
  CHECK(plan_mass(plan_f) == doctest::Approx(1.0F).epsilon(1e-4));
}

// ============================================================================
//  Float inputs with a double computation type
// ============================================================================

TEST_CASE("float histograms computed in double match double histograms") {
  const Fixture fx;
  std::vector<float> h1_f(fx.h1_data.begin(), fx.h1_data.end());
  std::vector<float> h2_f(fx.h2_data.begin(), fx.h2_data.end());
  const emdgrid::GridDataView<3, float> h1(fx.layout, std::span(h1_f));
  const emdgrid::GridDataView<3, float> h2(fx.layout, std::span(h2_f));

  const double cost_from_float = emdgrid::emd_l1(h1, h2);
  const double cost_from_double = emdgrid::emd_l1(fx.h1(), fx.h2());
  check_close(cost_from_float, cost_from_double);
}

TEST_SUITE_END();
