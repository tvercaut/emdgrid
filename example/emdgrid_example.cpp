#include <iostream>

#include "emdgrid/emdgrid.hpp"

int main() {
  constexpr int kExpectedSum = 5;
  std::cout << "emdgrid " << emdgrid::version() << '\n';
  return emdgrid::add(2, 3) == kExpectedSum ? 0 : 1;
}
