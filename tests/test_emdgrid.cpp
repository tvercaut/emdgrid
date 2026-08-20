#include <cstdlib>
#include <string_view>

#include "emdgrid/emdgrid.hpp"

int main() {
  if (emdgrid::add(2, 3) != 5) {
    return EXIT_FAILURE;
  }

  if (emdgrid::version() != std::string_view("0.1.0")) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
