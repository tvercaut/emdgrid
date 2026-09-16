#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "emdgrid/emdgrid.hpp"
#include "emdgrid/utils.hpp"

namespace emdgrid {

namespace detail {

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

/// Appends the mass each bin already shares with itself to `plan`.
///
/// For a subadditive ground metric the mass `min(h1[i], h2[i])` never needs to
/// move, so solvers route only the residual and re-attach these diagonal
/// entries when reporting the plan. Entries are appended, not assigned, so the
/// caller controls when the plan is cleared.
template <std::size_t Dim, std::floating_point Scalar,
          std::floating_point CompScalar>
void emit_self_mass(const GridDataView<Dim, Scalar>& h1,
                    const GridDataView<Dim, Scalar>& h2,
                    SparseTransportPlan<CompScalar>* plan) {
  const std::size_t n_nodes = h1.layout().node_count();
  for (std::size_t i = 0; i < n_nodes; ++i) {
    const CompScalar self_mass =
        std::min(static_cast<CompScalar>(h1.data()[i]),
                 static_cast<CompScalar>(h2.data()[i]));
    if (self_mass > CompScalar{0}) {
      plan->source.push_back(static_cast<uint32_t>(i));
      plan->target.push_back(static_cast<uint32_t>(i));
      plan->flow.push_back(self_mass);
    }
  }
}

/// Validates a pair of grid histograms shared by every grid solver.
///
/// Checks that both views share a layout, that no bin is negative, and that
/// each histogram carries unit total mass within `mass_tol`. The accumulation
/// runs in `CompScalar` so that the tolerance matches the precision the caller
/// actually computes in.
///
/// @throws std::invalid_argument on shape mismatch, a negative bin, or a
///         total mass that deviates from one by more than `mass_tol`.
template <std::size_t Dim, std::floating_point Scalar,
          std::floating_point CompScalar>
void validate_unit_mass_pair(const GridDataView<Dim, Scalar>& h1,
                             const GridDataView<Dim, Scalar>& h2,
                             CompScalar mass_tol) {
  if (h1.layout().shape() != h2.layout().shape()) {
    throw std::invalid_argument("histogram shapes do not match");
  }

  const std::size_t n_nodes = h1.layout().node_count();

  CompScalar t1{0};
  CompScalar t2{0};
  for (std::size_t i = 0; i < n_nodes; ++i) {
    const CompScalar v1 = static_cast<CompScalar>(h1.data()[i]);
    const CompScalar v2 = static_cast<CompScalar>(h2.data()[i]);
    if (v1 < CompScalar{0} || v2 < CompScalar{0}) {
      throw std::invalid_argument("histograms must be nonnegative");
    }
    t1 += v1;
    t2 += v2;
  }

  if (std::abs(t1 - CompScalar{1}) > mass_tol ||
      std::abs(t2 - CompScalar{1}) > mass_tol) {
    throw std::invalid_argument("expected unit-mass histograms");
  }
}

}  // namespace detail

}  // namespace emdgrid
