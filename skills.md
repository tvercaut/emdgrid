# Project Coding Conventions

This project follows the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/).

## Naming

- **Types** (classes, structs, enums): `CamelCase` (e.g., `LingOkadaSolver`).
- **Functions and local variables**: `snake_case` (e.g., `register_bv`, `find_loop`).
- **Private member variables**: `m_` prefix with `snake_case` suffix
  (e.g., `m_nodes`, `m_n_edges`, `m_enter_nbv_pos`).
- **No `k` prefix** for constants; prefer `static constexpr` private members
  named in `snake_case` (e.g., `static constexpr int default_max_iter = 500;`).
- **No Hungarian notation** or other type-encoding prefixes.

## Style

- Lines must be ≤ 80 characters.
- At least two spaces between code and inline comments.
- Do not indent within a namespace.
- Follow `cpplint` rules enforced by CI (`.github/workflows/cpplint.yml`).

## Detail namespaces

Implementation helpers that are not part of the public API live in a
`detail` namespace and in a separate `_detail.hpp` header
(e.g., `emd_l1_detail.hpp`).

## Computation types

When a function template accepts an input scalar type `Scalar`,
computations should be performed in a separate `CompScalar` type
(defaulting to `double`) to avoid precision loss from low-precision inputs.

`CompScalar` must be threaded all the way through: intermediate buffers,
accumulators, tolerances and the `SparseTransportPlan<CompScalar>` output all
use it rather than a hardcoded `double`. Numerical thresholds that used to be
literals belong in `utils.hpp` as `CompScalar`-dependent constants
(`residual_mass_epsilon`, `default_mass_tolerance`) so a `float` computation
is not blocked by a threshold below its own resolution.

Solvers take the plan as `SparseTransportPlanPtr<CompScalar>`, an alias that
puts the parameter in a non-deduced context. `CompScalar` is therefore chosen
by the solver's explicit template arguments, and callers can still pass a
plain `nullptr` when they do not want a plan.
