# Feature Request: Compute and expose Optimal Transport Plan for EMD-L1

## Overview

Requesting a new feature to compute and expose the Optimal Transport Plan associated with the EMD-L1 calculation.

## Requirements

1. **Optional Computation**: The transport plan should only be computed if explicitly requested by the caller (e.g. via an optional output parameter or overloaded `emd_l1` function), ensuring no performance overhead when only the distance scalar is needed.
2. **Data Structure**: The transport plan should be represented using a simple sparse Coordinate (COO) format:

```cpp
struct SparseTransportPlan
{
    std::vector<uint32_t> source;
    std::vector<uint32_t> target;
    std::vector<double> flow;
};
```

3. **API & Bindings**:
   - Expose `SparseTransportPlan` in C++ (`emdgrid` namespace).
   - Expose Python bindings for `SparseTransportPlan` when Python bindings are enabled.

## Related Issues

Relates to #8.
