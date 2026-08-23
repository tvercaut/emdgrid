#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <numeric>
#include <random>
#include <span>
#include <vector>

namespace emdgrid {

/// Computes the softmax of a given sequence of numbers.
template <std::floating_point Scalar = double>
[[nodiscard]] std::vector<Scalar> softmax(std::span<const Scalar> input) {
  if (input.empty()) {
    return {};
  }
  Scalar max_val = input[0];
  for (const Scalar x : input) {
    if (x > max_val) {
      max_val = x;
    }
  }

  std::vector<Scalar> result(input.size());
  Scalar sum{0};
  for (std::size_t i = 0; i < input.size(); ++i) {
    const Scalar exp_val = std::exp(input[i] - max_val);
    result[i] = exp_val;
    sum += exp_val;
  }

  if (sum > Scalar{0}) {
    for (Scalar& x : result) {
      x /= sum;
    }
  }

  return result;
}

/// Generates a normalized histogram of given size using random numbers
/// transformed via softmax.
template <std::floating_point Scalar = double, class Generator>
[[nodiscard]] std::vector<Scalar> generate_random_histogram(
    std::size_t size, Generator& g) {
  std::uniform_real_distribution<Scalar> dist(Scalar{-1}, Scalar{1});
  std::vector<Scalar> raw(size);
  for (std::size_t i = 0; i < size; ++i) {
    raw[i] = dist(g);
  }
  return softmax<Scalar>(raw);
}

/// Generates a normalized histogram with a default pseudo-random engine.
template <std::floating_point Scalar = double>
[[nodiscard]] std::vector<Scalar> generate_random_histogram(
    std::size_t size, unsigned int seed = 42) {
  std::mt19937 g(seed);
  return generate_random_histogram<Scalar>(size, g);
}

/// Execution timer utility using std::chrono.
class Timer {
 public:
  Timer() : m_start(std::chrono::high_resolution_clock::now()) {}

  void reset() noexcept {
    m_start = std::chrono::high_resolution_clock::now();
  }

  [[nodiscard]] double elapsed_milliseconds() const noexcept {
    const auto now = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = now - m_start;
    return elapsed.count();
  }

  [[nodiscard]] double elapsed_seconds() const noexcept {
    return elapsed_milliseconds() / 1000.0;
  }

 private:
  std::chrono::high_resolution_clock::time_point m_start;
};

}  // namespace emdgrid
