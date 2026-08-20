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

Python bindings are available through `bindings/python` and can be enabled
explicitly during configuration.

```bash
cmake -S . -B build -DEMDGRID_BUILD_PYTHON_BINDINGS=ON
cmake --build build
PYTHONPATH=build/bindings/python python3 -c "import pyemdgrid; print(pyemdgrid.version())"
```
