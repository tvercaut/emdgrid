# emdgrid

Basic scaffold for a modern C++ codebase with Python bindings.

## Layout

```text
.
├── .github/workflows
├── bindings/python
├── example
├── include/emdgrid
├── src
└── tests
```

## Build

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Python bindings

The CMake build also produces a `pyemdgrid` extension module through
`bindings/python`.

```bash
PYTHONPATH=build/bindings/python python3 -c "import pyemdgrid; print(pyemdgrid.version())"
```
