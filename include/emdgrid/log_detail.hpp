#pragma once

#include <concepts>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include <spdlog/fmt/fmt.h>  // NOLINT(build/include_order)
#include <spdlog/spdlog.h>   // NOLINT(build/include_order)

#include "emdgrid/utils.hpp"

namespace emdgrid {

namespace detail {

/// Uniform stage logging for a solver run.
///
/// The solvers differ in their internals but share a shape: validate and
/// quantize the inputs, build a graph, hand it to a backend, decode the answer
/// into a plan. Reporting those stages identically everywhere means a log from
/// one solver can be read, and compared against another's, without knowing
/// which produced it.
///
/// Timings are cumulative in the way you would want: each `phase` reports the
/// time since the previous stage, and `finish` reports the whole run.
class SolverLog {
 public:
  /// @param solver Solver name, used as the prefix on every line.
  /// @param config Optional one-line summary of the options in force.
  explicit SolverLog(std::string_view solver, const std::string& config = {})
      : m_solver(solver) {
    if (config.empty()) {
      spdlog::info("{}: starting", m_solver);
    } else {
      spdlog::info("{}: starting ({})", m_solver, config);
    }
  }

  /// Reports a finished stage and how long it took.
  ///
  /// @param name   What the stage did, as a verb phrase ("graph construction").
  /// @param detail Optional stage-specific figures ("nodes=405, arcs=972").
  void phase(std::string_view name, const std::string& detail = {}) {
    const double elapsed = m_phase.elapsed_milliseconds();
    if (detail.empty()) {
      spdlog::info("{}: {} took {:.3f} ms", m_solver, name, elapsed);
    } else {
      spdlog::info("{}: {} took {:.3f} ms ({})", m_solver, name, elapsed,
                   detail);
    }
    m_phase.reset();
  }

  /// What a finished solver actually delivered.
  enum class Outcome : std::uint8_t {
    /// A proven optimum, which is what the solver promised.
    Optimal,
    /// An upper bound, which is also what the solver promised — the
    /// heuristics are not trying to be exact.
    Heuristic,
    /// Less than the solver promised: it stopped on an iteration cap, or the
    /// backend reported something other than optimality.
    Degraded,
  };

  /// Reports the backend's exit status.
  ///
  /// Only `Degraded` warns. A solver that stops early still returns a number,
  /// and the whole hazard is that the number looks exactly like a converged
  /// one, so that case has to be loud. Warning on every heuristic call would
  /// just teach the reader to ignore the channel.
  void status(std::string_view status_name, Outcome outcome) {
    if (outcome == Outcome::Degraded) {
      spdlog::warn("{}: solver finished with status {} — the reported cost "
                   "is not a proven optimum",
                   m_solver, status_name);
    } else {
      spdlog::info("{}: solver finished with status {}", m_solver,
                   status_name);
    }
  }

  /// Reports the final cost and the wall time for the whole run.
  template <std::floating_point CompScalar>
  void finish(CompScalar cost) {
    spdlog::info("{}: done in {:.3f} ms, cost {}", m_solver,
                 m_total.elapsed_milliseconds(), cost);
  }

 private:
  std::string m_solver;
  Timer m_total;
  Timer m_phase;
};

}  // namespace detail

}  // namespace emdgrid
