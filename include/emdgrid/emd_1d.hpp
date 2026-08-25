#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "emdgrid/emdgrid.hpp"

namespace emdgrid {

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
[[nodiscard]] CompScalar emd_1d(const GridDataView<1, Scalar>& h1,
                                const GridDataView<1, Scalar>& h2,
                                SparseTransportPlan* plan = nullptr) {
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

    std::vector<double> s(n);
    std::vector<double> d(n);
    for (std::size_t i = 0; i < n; ++i) {
      double h1_val = static_cast<double>(h1.data()[i]);
      double h2_val = static_cast<double>(h2.data()[i]);
      double self_mass = std::min(h1_val, h2_val);
      if (self_mass > 0.0) {
        plan->source.push_back(static_cast<uint32_t>(i));
        plan->target.push_back(static_cast<uint32_t>(i));
        plan->flow.push_back(self_mass);
      }
      s[i] = h1_val - self_mass;
      d[i] = h2_val - self_mass;
    }

    std::size_t src_idx = 0;
    std::size_t tgt_idx = 0;
    while (src_idx < n && tgt_idx < n) {
      if (s[src_idx] <= 1e-12) {
        ++src_idx;
        continue;
      }
      if (d[tgt_idx] <= 1e-12) {
        ++tgt_idx;
        continue;
      }
      double transfer = std::min(s[src_idx], d[tgt_idx]);
      plan->source.push_back(static_cast<uint32_t>(src_idx));
      plan->target.push_back(static_cast<uint32_t>(tgt_idx));
      plan->flow.push_back(transfer);

      s[src_idx] -= transfer;
      d[tgt_idx] -= transfer;
    }
  }

  return total;
}

}  // namespace emdgrid
