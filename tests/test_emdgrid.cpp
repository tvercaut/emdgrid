#include <cstdlib>
#include <string_view>

#include "emdgrid/emdgrid.hpp"
#include "emdgrid/version.hpp"

int main() {
  constexpr int kExpectedSum = 5;

  if (emdgrid::add(2, 3) != kExpectedSum) {
    return EXIT_FAILURE;
  }

  if (emdgrid::version() != emdgrid::kVersion) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
