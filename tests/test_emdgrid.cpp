#include <cstdlib>
#include <string_view>

#include "emdgrid/emdgrid.hpp"

int main() {
  constexpr int kExpectedSum = 5;

  if (emdgrid::add(2, 3) != kExpectedSum) {
    return EXIT_FAILURE;
  }

  if (emdgrid::version() != std::string_view(EMDGRID_VERSION)) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
