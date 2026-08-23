#include <cmath>
#include <numeric>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>  // NOLINT(build/include_order)

#include "emdgrid/utils.hpp"

TEST_CASE("utils: softmax empty input") {
  const std::vector<double> input = {};
  const auto result = emdgrid::softmax<double>(input);
  CHECK(result.empty());
}

TEST_CASE("utils: softmax equal inputs produce uniform distribution") {
  const std::vector<double> input = {2.0, 2.0, 2.0, 2.0};
  const auto result = emdgrid::softmax<double>(input);
  CHECK_EQ(result.size(), 4);
  for (const double val : result) {
    CHECK(val == doctest::Approx(0.25));
  }
}

TEST_CASE("utils: softmax output sums to 1.0 and is positive") {
  const std::vector<double> input = {-1.0, 0.0, 2.5, 5.0};
  const auto result = emdgrid::softmax<double>(input);
  CHECK_EQ(result.size(), 4);
  double sum = 0.0;
  for (const double val : result) {
    CHECK_GT(val, 0.0);
    sum += val;
  }
  CHECK(sum == doctest::Approx(1.0));
}

TEST_CASE("utils: generate_random_histogram size and normalization") {
  constexpr std::size_t n = 100;
  const auto h = emdgrid::generate_random_histogram<double>(n, 42);
  CHECK_EQ(h.size(), n);
  double sum = 0.0;
  for (const double val : h) {
    CHECK_GT(val, 0.0);
    sum += val;
  }
  CHECK(sum == doctest::Approx(1.0));
}

TEST_CASE("utils: generate_random_histogram reproducible with same seed") {
  constexpr std::size_t n = 50;
  const auto h1 = emdgrid::generate_random_histogram<double>(n, 123);
  const auto h2 = emdgrid::generate_random_histogram<double>(n, 123);
  const auto h3 = emdgrid::generate_random_histogram<double>(n, 456);
  CHECK_EQ(h1, h2);
  CHECK_NE(h1, h3);
}

TEST_CASE("utils: Timer measures elapsed time") {
  const emdgrid::Timer timer;
  const double ms = timer.elapsed_milliseconds();
  const double sec = timer.elapsed_seconds();
  CHECK_GE(ms, 0.0);
  CHECK_GE(sec, 0.0);
  CHECK(sec == doctest::Approx(ms / 1000.0));
}
