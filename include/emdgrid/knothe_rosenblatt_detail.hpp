#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "emdgrid/emdgrid.hpp"

namespace emdgrid {

/// Ground metric choice for 1-D marginal transport matching in
/// Knothe-Rosenblatt transportation plan.
enum class GroundMetric : std::uint8_t { L1, SqEuclidean };

namespace detail {

/// Represents a single active subproblem task in Knothe-Rosenblatt.
struct KrTask {
  std::size_t src_base_offset{0};
  std::size_t tgt_base_offset{0};
  double mass{0.0};
};

/// Active 1-D monotone matching flow triple.
struct MonotoneFlow {
  std::size_t src_idx{0};
  std::size_t tgt_idx{0};
  double flow{0.0};
};

/// Computes 1-D monotone transport matching between two 1-D probability
/// distributions u and v (both assumed non-negative and normalized to sum 1).
inline void compute_monotone_matching(
    const std::vector<double>& u, const std::vector<double>& v,
    std::vector<MonotoneFlow>* matching) {
  matching->clear();
  std::size_t i = 0;
  std::size_t j = 0;
  const std::size_t n_u = u.size();
  const std::size_t n_v = v.size();
  if (n_u == 0 || n_v == 0) {
    return;
  }

  double rem_u = u[0];
  double rem_v = v[0];
  constexpr double kEps = 1e-12;

  while (i < n_u && j < n_v) {
    if (rem_u <= kEps) {
      ++i;
      if (i < n_u) {
        rem_u = u[i];
      }
      continue;
    }
    if (rem_v <= kEps) {
      ++j;
      if (j < n_v) {
        rem_v = v[j];
      }
      continue;
    }
    const double transfer = std::min(rem_u, rem_v);
    matching->push_back({i, j, transfer});
    rem_u -= transfer;
    rem_v -= transfer;
  }
}

/// Compute C-order linear strides for a grid shape.
template <std::size_t Dim>
[[nodiscard]] std::array<std::size_t, Dim> compute_grid_strides(
    const typename GridLayout<Dim>::Shape& shape) {
  std::array<std::size_t, Dim> strides{};
  strides[Dim - 1] = 1;
  for (std::size_t a = Dim - 1; a-- > 0;) {
    strides[a] = strides[a + 1] * shape[a + 1];
  }
  return strides;
}

/// Precompute linear offset deltas for free dimensions R = {order[k+1] ...}.
template <std::size_t Dim>
[[nodiscard]] std::vector<std::size_t> precompute_free_offsets(
    const typename GridLayout<Dim>::Shape& shape,
    const std::array<std::size_t, Dim>& strides,
    const std::vector<std::size_t>& free_dims) {
  std::vector<std::size_t> offsets;
  if (free_dims.empty()) {
    offsets.push_back(0);
    return offsets;
  }

  offsets.push_back(0);
  for (const std::size_t dim : free_dims) {
    const std::size_t extent = shape[dim];
    const std::size_t stride = strides[dim];
    const std::size_t current_size = offsets.size();
    offsets.reserve(current_size * extent);
    for (std::size_t e = 1; e < extent; ++e) {
      const std::size_t delta = e * stride;
      for (std::size_t i = 0; i < current_size; ++i) {
        offsets.push_back(offsets[i] + delta);
      }
    }
  }
  return offsets;
}

/// Validates that dimension_order is a permutation of {0, ..., Dim-1}.
template <std::size_t Dim>
[[nodiscard]] std::vector<std::size_t> validate_and_get_dimension_order(
    std::span<const std::size_t> dimension_order) {
  if (dimension_order.empty()) {
    std::vector<std::size_t> order(Dim);
    for (std::size_t i = 0; i < Dim; ++i) {
      order[i] = i;
    }
    return order;
  }

  if (dimension_order.size() != Dim) {
    throw std::invalid_argument("dimension_order length must match Dim");
  }

  std::vector<bool> seen(Dim, false);
  std::vector<std::size_t> order(Dim);
  for (std::size_t i = 0; i < Dim; ++i) {
    const std::size_t d = dimension_order[i];
    if (d >= Dim || seen[d]) {
      throw std::invalid_argument(
          "dimension_order must be a valid permutation of dimensions");
    }
    seen[d] = true;
    order[i] = d;
  }
  return order;
}

}  // namespace detail
}  // namespace emdgrid
