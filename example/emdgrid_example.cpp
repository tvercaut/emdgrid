#include <iostream>

#include "emdgrid/emdgrid.hpp"

int main() {
  std::cout << "emdgrid " << emdgrid::version() << '\n';
  return emdgrid::add(2, 3) == 5 ? 0 : 1;
}
