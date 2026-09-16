#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "emdgrid/emdgrid.hpp"
#include "emdgrid/utils.hpp"

namespace emdgrid {

namespace detail {

/// Active 1-D monotone matching flow triple.
///
/// @tparam CompScalar Scalar type used for computation (default: double).
template <std::floating_point CompScalar = double>
struct MonotoneFlow {
  std::size_t src_idx{0};
  std::size_t tgt_idx{0};
  CompScalar flow{0};
};

/// Computes 1-D monotone transport matching between two 1-D mass distributions.
template <std::floating_point CompScalar>
void compute_1d_monotone_matching(
    std::span<const CompScalar> u, std::span<const CompScalar> v,
    std::vector<MonotoneFlow<CompScalar>>* matching) {
  matching->clear();
  std::size_t i = 0;
  std::size_t j = 0;
  const std::size_t n_u = u.size();
  const std::size_t n_v = v.size();
  if (n_u == 0 || n_v == 0) {
    return;
  }

  CompScalar rem_u = u[0];
  CompScalar rem_v = v[0];
  constexpr CompScalar eps = residual_mass_epsilon<CompScalar>;

  while (i < n_u && j < n_v) {
    if (rem_u <= eps) {
      ++i;
      if (i < n_u) {
        rem_u = u[i];
      }
      continue;
    }
    if (rem_v <= eps) {
      ++j;
      if (j < n_v) {
        rem_v = v[j];
      }
      continue;
    }
    const CompScalar transfer = std::min(rem_u, rem_v);
    matching->push_back({i, j, transfer});
    rem_u -= transfer;
    rem_v -= transfer;
  }
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Public API: emd_1d
// ---------------------------------------------------------------------------

/// Exact EMD-L1 solver for 1-D histograms using prefix sums.
///
/// The Earth Mover's Distance under the L1 ground metric for 1-D discrete
/// histograms equals the sum of absolute values of cumulative-sum
/// differences: EMD = Σ_k |Σ_{i≤k} (H1[i] − H2[i])|.
///
/// @tparam Scalar Input histogram scalar type.
/// @tparam CompScalar Scalar type used for computation (default: double).
template <std::floating_point Scalar, std::floating_point CompScalar = double>
[[nodiscard]] CompScalar emd_1d(
    const GridDataView<1, Scalar>& h1, const GridDataView<1, Scalar>& h2,
    SparseTransportPlanPtr<CompScalar> plan = nullptr) {
  if (h1.layout().shape() != h2.layout().shape()) {
    throw std::invalid_argument("histogram shapes do not match");
  }
  const std::size_t n = h1.layout().node_count();
  CompScalar total{0};
  CompScalar cumsum{0};
  for (std::size_t i = 0; i < n; ++i) {
    cumsum += static_cast<CompScalar>(h1.data()[i]) -
              static_cast<CompScalar>(h2.data()[i]);
    total += std::abs(cumsum);
  }

  if (plan) {
    plan->source.clear();
    plan->target.clear();
    plan->flow.clear();

    std::vector<CompScalar> s(n);
    std::vector<CompScalar> d(n);
    for (std::size_t i = 0; i < n; ++i) {
      CompScalar h1_val = static_cast<CompScalar>(h1.data()[i]);
      CompScalar h2_val = static_cast<CompScalar>(h2.data()[i]);
      CompScalar self_mass = std::min(h1_val, h2_val);
      if (self_mass > CompScalar{0}) {
        plan->source.push_back(static_cast<uint32_t>(i));
        plan->target.push_back(static_cast<uint32_t>(i));
        plan->flow.push_back(self_mass);
      }
      s[i] = h1_val - self_mass;
      d[i] = h2_val - self_mass;
    }

    std::vector<detail::MonotoneFlow<CompScalar>> matching;
    detail::compute_1d_monotone_matching<CompScalar>(s, d, &matching);
    for (const auto& flow_pair : matching) {
      plan->source.push_back(static_cast<uint32_t>(flow_pair.src_idx));
      plan->target.push_back(static_cast<uint32_t>(flow_pair.tgt_idx));
      plan->flow.push_back(flow_pair.flow);
    }
  }

  return total;
}

/// Exact 1-D Optimal Transport solver under squared Euclidean ground metric.
///
/// For 1-D histograms on an integer grid, the squared Euclidean cost is
/// defined as C_{i,j} = (i − j)². The optimal transport plan is uniquely
/// determined by monotonic matching of cumulative mass from left to right.
///
/// @tparam Scalar Input histogram scalar type.
/// @tparam CompScalar Scalar type used for computation (default: double).
template <std::floating_point Scalar, std::floating_point CompScalar = double>
[[nodiscard]] CompScalar emd_sqeuclidean_1d(
    const GridDataView<1, Scalar>& h1,
    const GridDataView<1, Scalar>& h2,
    SparseTransportPlanPtr<CompScalar> plan = nullptr) {
  if (h1.layout().shape() != h2.layout().shape()) {
    throw std::invalid_argument("histogram shapes do not match");
  }
  const std::size_t n = h1.layout().node_count();

  if (plan) {
    plan->source.clear();
    plan->target.clear();
    plan->flow.clear();
  }

  std::vector<CompScalar> s(n);
  std::vector<CompScalar> d(n);
  for (std::size_t i = 0; i < n; ++i) {
    s[i] = static_cast<CompScalar>(h1.data()[i]);
    d[i] = static_cast<CompScalar>(h2.data()[i]);
  }

  std::vector<detail::MonotoneFlow<CompScalar>> matching;
  detail::compute_1d_monotone_matching<CompScalar>(s, d, &matching);

  CompScalar total{0};
  for (const auto& flow_pair : matching) {
    const CompScalar dist = static_cast<CompScalar>(flow_pair.src_idx) -
                            static_cast<CompScalar>(flow_pair.tgt_idx);
    total += flow_pair.flow * dist * dist;

    if (plan) {
      plan->source.push_back(static_cast<uint32_t>(flow_pair.src_idx));
      plan->target.push_back(static_cast<uint32_t>(flow_pair.tgt_idx));
      plan->flow.push_back(flow_pair.flow);
    }
  }

  return total;
}

}  // namespace emdgrid
