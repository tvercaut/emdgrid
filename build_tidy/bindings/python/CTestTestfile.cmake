# CMake generated Testfile for
# Source directory: /app/bindings/python
# Build directory: /app/build_tidy/bindings/python
#
# This file includes the relevant testing commands required for
# testing this directory and lists subdirectories to be tested as well.
add_test([=[pyemdgrid_import]=] "/usr/bin/cmake" "-E" "env" "PYTHONPATH=/app/build_tidy/bindings/python" "ASAN_OPTIONS=detect_leaks=0" "LD_PRELOAD=/usr/lib/gcc/x86_64-linux-gnu/13/libasan.so /usr/lib/gcc/x86_64-linux-gnu/13/../../../x86_64-linux-gnu/libstdc++.so.6" "/home/jules/.pyenv/shims/python3" "-c" "import pyemdgrid; assert pyemdgrid.version() == 'ae2b6fe-dirty'")
set_tests_properties([=[pyemdgrid_import]=] PROPERTIES  _BACKTRACE_TRIPLES "/app/bindings/python/CMakeLists.txt;26;add_test;/app/bindings/python/CMakeLists.txt;0;")
add_test([=[pyemdgrid_pytest]=] "/usr/bin/cmake" "-E" "env" "PYTHONPATH=/app/build_tidy/bindings/python" "ASAN_OPTIONS=detect_leaks=0" "LD_PRELOAD=/usr/lib/gcc/x86_64-linux-gnu/13/libasan.so /usr/lib/gcc/x86_64-linux-gnu/13/../../../x86_64-linux-gnu/libstdc++.so.6" "/home/jules/.pyenv/shims/python3" "-m" "pytest" "/app/tests/test_emdgrid.py" "-v")
set_tests_properties([=[pyemdgrid_pytest]=] PROPERTIES  _BACKTRACE_TRIPLES "/app/bindings/python/CMakeLists.txt;35;add_test;/app/bindings/python/CMakeLists.txt;0;")
subdirs("../../_deps/pybind11-build")
