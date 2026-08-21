# Project Coding Conventions

This project follows the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/).

## Naming

- **Types** (classes, structs, enums): `CamelCase` (e.g., `LingOkadaSolver`).
- **Functions and local variables**: `camelCase` (e.g., `totalFlow`, `numNodes`).
- **Private member variables**: `m_` prefix, no suffix (e.g., `m_nodes`, `m_nEdges`).
- **No `k` prefix** for constants; prefer `static constexpr` private members
  named in `camelCase` (e.g., `static constexpr int defaultMaxIter = 500;`).
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
