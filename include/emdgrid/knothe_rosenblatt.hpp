#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "emdgrid/emdgrid.hpp"
#include "emdgrid/knothe_rosenblatt_detail.hpp"

namespace emdgrid {

/// N-D Knothe-Rosenblatt transport heuristic solver for grid histograms.
///
/// Computes an approximate transport plan and ground cost by sequentially
/// solving 1-D Optimal Transport problems along the axes specified by
/// dimension_order.
///
/// @tparam Dim Grid dimensionality.
/// @tparam Scalar Input histogram scalar type.
/// @tparam CompScalar Scalar type used for computation (default: double).
template <std::size_t Dim, std::floating_point Scalar,
          std::floating_point CompScalar = double>  // NOLINT(*)
  requires(Dim >= 1)  // NOLINT(whitespace/indent_namespace)
[[nodiscard]] CompScalar knothe_rosenblatt(
    const GridDataView<Dim, Scalar>& h1, const GridDataView<Dim, Scalar>& h2,
    GroundMetric metric = GroundMetric::L1,
    std::span<const std::size_t> dimension_order = {},
    SparseTransportPlan* plan = nullptr) {
  if (h1.layout().shape() != h2.layout().shape()) {
    throw std::invalid_argument("histogram shapes do not match");
  }

  const auto order =
      detail::validate_and_get_dimension_order<Dim>(dimension_order);
  const auto& shape = h1.layout().shape();
  const auto strides = detail::compute_grid_strides<Dim>(shape);
  const std::size_t node_count = h1.layout().node_count();

  if (plan) {
    plan->source.clear();
    plan->target.clear();
    plan->flow.clear();
  }

  double total_mass_h1 = 0.0;
  for (std::size_t i = 0; i < node_count; ++i) {
    total_mass_h1 += static_cast<double>(h1.data()[i]);
  }
  if (total_mass_h1 <= 1e-12) {
    return CompScalar{0};
  }

  std::vector<detail::KrTask> current_tasks;
  current_tasks.push_back({0, 0, total_mass_h1});

  double total_cost = 0.0;

  for (std::size_t k = 0; k < Dim; ++k) {
    const std::size_t current_dim = order[k];
    const std::size_t extent = shape[current_dim];
    const std::size_t dim_stride = strides[current_dim];

    std::vector<std::size_t> free_dims;
    for (std::size_t r = k + 1; r < Dim; ++r) {
      free_dims.push_back(order[r]);
    }

    const auto free_offsets =
        detail::precompute_free_offsets<Dim>(shape, strides, free_dims);

    std::vector<detail::KrTask> next_tasks;
    const std::size_t num_tasks = current_tasks.size();

#ifdef _OPENMP
#pragma omp parallel
    {
      std::vector<detail::KrTask> local_next_tasks;
      double local_cost = 0.0;
      SparseTransportPlan local_plan;
      std::vector<double> u(extent, 0.0);
      std::vector<double> v(extent, 0.0);
      std::vector<detail::MonotoneFlow> matching;

#pragma omp for nowait
      for (std::ptrdiff_t t_idx = 0;
           t_idx < static_cast<std::ptrdiff_t>(num_tasks); ++t_idx) {
        const auto& task = current_tasks[static_cast<std::size_t>(t_idx)];
        if (task.mass <= 1e-12) {
          continue;
        }

        std::fill(u.begin(), u.end(), 0.0);
        std::fill(v.begin(), v.end(), 0.0);

        for (std::size_t x = 0; x < extent; ++x) {
          const std::size_t src_off = task.src_base_offset + x * dim_stride;
          const std::size_t tgt_off = task.tgt_base_offset + x * dim_stride;
          double u_val = 0.0;
          double v_val = 0.0;
          for (const std::size_t delta : free_offsets) {
            u_val += static_cast<double>(h1.data()[src_off + delta]);
            v_val += static_cast<double>(h2.data()[tgt_off + delta]);
          }
          u[x] = u_val;
          v[x] = v_val;
        }

        double sum_u = 0.0;
        double sum_v = 0.0;
        for (std::size_t x = 0; x < extent; ++x) {
          sum_u += u[x];
          sum_v += v[x];
        }

        if (sum_u <= 1e-12 || sum_v <= 1e-12) {
          continue;
        }

        for (std::size_t x = 0; x < extent; ++x) {
          u[x] /= sum_u;
          v[x] /= sum_v;
        }

        // 1D monotone matching is optimal for both L1 and squared Euclidean
        // metrics on 1D grids (Monge property), yielding identical couplings.
        detail::compute_1d_monotone_matching(u, v, &matching);

        for (const auto& flow_pair : matching) {
          const double branch_mass = task.mass * flow_pair.flow;
          const double diff = static_cast<double>(flow_pair.src_idx) -
                              static_cast<double>(flow_pair.tgt_idx);
          const double dist_unit =
              (metric == GroundMetric::L1) ? std::abs(diff) : (diff * diff);

          local_cost += branch_mass * dist_unit;

          const std::size_t child_src_off =
              task.src_base_offset + flow_pair.src_idx * dim_stride;
          const std::size_t child_tgt_off =
              task.tgt_base_offset + flow_pair.tgt_idx * dim_stride;

          if (k + 1 < Dim) {
            local_next_tasks.push_back(
                {child_src_off, child_tgt_off, branch_mass});
          } else if (plan) {
            local_plan.source.push_back(static_cast<uint32_t>(child_src_off));
            local_plan.target.push_back(static_cast<uint32_t>(child_tgt_off));
            local_plan.flow.push_back(branch_mass);
          }
        }
      }

#pragma omp critical
      {
        total_cost += local_cost;
        if (k + 1 < Dim) {
          next_tasks.insert(next_tasks.end(), local_next_tasks.begin(),
                            local_next_tasks.end());
        }
        if (plan) {
          plan->source.insert(plan->source.end(), local_plan.source.begin(),
                              local_plan.source.end());
          plan->target.insert(plan->target.end(), local_plan.target.begin(),
                              local_plan.target.end());
          plan->flow.insert(plan->flow.end(), local_plan.flow.begin(),
                            local_plan.flow.end());
        }
      }
    }
#else
    std::vector<double> u(extent, 0.0);
    std::vector<double> v(extent, 0.0);
    std::vector<detail::MonotoneFlow> matching;

    for (std::size_t t_idx = 0; t_idx < num_tasks; ++t_idx) {
      const auto& task = current_tasks[t_idx];
      if (task.mass <= 1e-12) {
        continue;
      }

      std::fill(u.begin(), u.end(), 0.0);
      std::fill(v.begin(), v.end(), 0.0);

      for (std::size_t x = 0; x < extent; ++x) {
        const std::size_t src_off = task.src_base_offset + x * dim_stride;
        const std::size_t tgt_off = task.tgt_base_offset + x * dim_stride;
        double u_val = 0.0;
        double v_val = 0.0;
        for (const std::size_t delta : free_offsets) {
          u_val += static_cast<double>(h1.data()[src_off + delta]);
          v_val += static_cast<double>(h2.data()[tgt_off + delta]);
        }
        u[x] = u_val;
        v[x] = v_val;
      }

      double sum_u = 0.0;
      double sum_v = 0.0;
      for (std::size_t x = 0; x < extent; ++x) {
        sum_u += u[x];
        sum_v += v[x];
      }

      if (sum_u <= 1e-12 || sum_v <= 1e-12) {
        continue;
      }

      for (std::size_t x = 0; x < extent; ++x) {
        u[x] /= sum_u;
        v[x] /= sum_v;
      }

      // 1D monotone matching is optimal for both L1 and squared Euclidean
      // metrics on 1D grids (Monge property), yielding identical couplings.
      detail::compute_1d_monotone_matching(u, v, &matching);

      for (const auto& flow_pair : matching) {
        const double branch_mass = task.mass * flow_pair.flow;
        const double diff = static_cast<double>(flow_pair.src_idx) -
                            static_cast<double>(flow_pair.tgt_idx);
        const double dist_unit =
            (metric == GroundMetric::L1) ? std::abs(diff) : (diff * diff);

        total_cost += branch_mass * dist_unit;

        const std::size_t child_src_off =
            task.src_base_offset + flow_pair.src_idx * dim_stride;
        const std::size_t child_tgt_off =
            task.tgt_base_offset + flow_pair.tgt_idx * dim_stride;

        if (k + 1 < Dim) {
          next_tasks.push_back({child_src_off, child_tgt_off, branch_mass});
        } else if (plan) {
          plan->source.push_back(static_cast<uint32_t>(child_src_off));
          plan->target.push_back(static_cast<uint32_t>(child_tgt_off));
          plan->flow.push_back(branch_mass);
        }
      }
    }
#endif

    current_tasks = std::move(next_tasks);
  }

  return static_cast<CompScalar>(total_cost);
}

}  // namespace emdgrid
