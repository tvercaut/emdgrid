#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <span>
#include <type_traits>
#include <vector>

namespace emdgrid {

/// Ground metric choice for optimal transport problems.
enum class GroundMetric : std::uint8_t { L1, SqEuclidean };

/// Tolerance below which a residual mass is considered exhausted.
///
/// The value tracks the resolution of the computation type: 1e-12 keeps the
/// historical behaviour in double precision, while a float computation needs
/// a threshold above its own rounding noise (~1e-7 relative) so that greedy
/// mass-splitting loops terminate instead of emitting denormal leftovers.
template <std::floating_point CompScalar>
inline constexpr CompScalar residual_mass_epsilon =
    std::is_same_v<CompScalar, float> ? static_cast<CompScalar>(1e-6)
                                      : static_cast<CompScalar>(1e-12);

/// Default tolerance on the deviation of a histogram total mass from one.
///
/// Summing n bins accumulates O(n) rounding errors, so the unit-mass check
/// must be looser than the computation type's epsilon. 1e-6 is comfortable
/// for double; float needs 1e-4 to accept histograms of a few thousand bins.
template <std::floating_point CompScalar>
inline constexpr CompScalar default_mass_tolerance =
    std::is_same_v<CompScalar, float> ? static_cast<CompScalar>(1e-4)
                                      : static_cast<CompScalar>(1e-6);

namespace detail {

/// Rejects the overload-selector enums so that a cost-functor overload never
/// competes with the `GroundMetric` overload of the same solver.
template <typename T>
concept ValidCostFn = !std::is_same_v<std::decay_t<T>, GroundMetric>;

}  // namespace detail

/// Default cost functor for L1 (Manhattan) ground metric.
struct L1Cost {
  static constexpr bool extract_self_mass = true;

  [[nodiscard]] int64_t operator()(std::size_t /*axis*/, std::size_t a,
                                  std::size_t b) const noexcept {
    const auto diff =
        static_cast<std::ptrdiff_t>(a) - static_cast<std::ptrdiff_t>(b);
    return std::abs(diff);
  }
};

/// Default cost functor for SqEuclidean (squared Euclidean) ground metric.
struct SqEuclideanCost {
  static constexpr bool extract_self_mass = false;

  [[nodiscard]] int64_t operator()(std::size_t /*axis*/, std::size_t a,
                                  std::size_t b) const noexcept {
    const auto diff =
        static_cast<std::ptrdiff_t>(a) - static_cast<std::ptrdiff_t>(b);
    return diff * diff;
  }
};

/// Computes the softmax of a given sequence of numbers using an online,
/// numerically safe streaming formulation.
template <std::floating_point Scalar = double>
[[nodiscard]] std::vector<Scalar> softmax(std::span<const Scalar> input) {
  if (input.empty()) {
    return {};
  }

  Scalar max_val = input.front();
  Scalar normalizer{1};

  for (const Scalar x : input.subspan(1)) {
    if (x > max_val) {
      normalizer = (normalizer * std::exp(max_val - x)) + Scalar{1};
      max_val = x;
    } else {
      normalizer += std::exp(x - max_val);
    }
  }

  std::vector<Scalar> result;
  result.reserve(input.size());
  for (const Scalar x : input) {
    result.push_back(std::exp(x - max_val) / normalizer);
  }

  return result;
}

/// Generates a normalized histogram of given size using random numbers
/// transformed via online softmax.
template <std::floating_point Scalar = double, class Generator>
[[nodiscard]] std::vector<Scalar> generate_random_histogram(
    std::size_t size, Generator& g) {
  std::uniform_real_distribution<Scalar> dist(Scalar{-1}, Scalar{1});
  std::vector<Scalar> raw(size);
  std::generate(raw.begin(), raw.end(), [&] { return dist(g); });
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
