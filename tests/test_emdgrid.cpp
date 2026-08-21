#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "emdgrid/emdgrid.hpp"
#include "emdgrid/version.hpp"

int main() {
  try {
    constexpr int kExpectedSum = 5;
    constexpr std::size_t k2dNodeCount = 6;
    constexpr std::size_t k2dEdgeCount = 7;
    constexpr std::size_t k2dLastNode = 5;
    constexpr std::size_t k2dOutOfRangeNode = 6;
    constexpr std::size_t k3dNodeCount = 24;
    constexpr std::size_t k3dEdgeCount = 46;
    constexpr std::size_t k3dLastNode = 23;
    constexpr std::size_t k3dNode = 17;

    if (emdgrid::add(2, 3) != kExpectedSum) {
      return EXIT_FAILURE;
    }

    if (emdgrid::version() != emdgrid::detail::version) {
      return EXIT_FAILURE;
    }

    const emdgrid::GridLayout<2> layout2d({2, 3});
    if (layout2d.shape() != emdgrid::GridLayout<2>::Shape{2, 3}) {
      return EXIT_FAILURE;
    }

    if (layout2d.node_count() != k2dNodeCount ||
        layout2d.edge_count() != k2dEdgeCount) {
      return EXIT_FAILURE;
    }

    if (layout2d.node({1, 2}) != k2dLastNode) {
      return EXIT_FAILURE;
    }

    if (layout2d.coordinates(4) !=
        emdgrid::GridLayout<2>::Coordinates{1, 1}) {
      return EXIT_FAILURE;
    }

    const std::vector<std::size_t> expected_interior_neighbours = {
        1, 3, k2dLastNode};
    if (layout2d.neighbours(layout2d.node({1, 1})) !=
        expected_interior_neighbours) {
      return EXIT_FAILURE;
    }

    const std::vector<std::size_t> expected_corner_neighbours = {3, 1};
    if (layout2d.neighbours(layout2d.node({0, 0})) !=
        expected_corner_neighbours) {
      return EXIT_FAILURE;
    }

    const emdgrid::GridLayout<3> layout3d({2, 3, 4});
    if (layout3d.node_count() != k3dNodeCount ||
        layout3d.edge_count() != k3dEdgeCount) {
      return EXIT_FAILURE;
    }

    if (layout3d.node({1, 2, 3}) != k3dLastNode) {
      return EXIT_FAILURE;
    }

    if (layout3d.coordinates(k3dNode) !=
        emdgrid::GridLayout<3>::Coordinates{1, 1, 1}) {
      return EXIT_FAILURE;
    }

    const std::vector<int> values = {0, 1, 2, 3, 4, 5};
    const emdgrid::GridDataView<2, int> view(layout2d,
                                             std::span<const int>(values));
    if (&view.layout() != &layout2d) {
      return EXIT_FAILURE;
    }

    if (view.data().data() != values.data() ||
        view.data().size() != values.size()) {
      return EXIT_FAILURE;
    }

    if (view(1, 2) != kExpectedSum ||
        view(emdgrid::GridLayout<2>::Coordinates{0, 1}) != 1) {
      return EXIT_FAILURE;
    }

    bool caught_out_of_range = false;
    try {
      static_cast<void>(layout2d.node({2, 0}));
    } catch (const std::out_of_range&) {
      caught_out_of_range = true;
    }
    if (!caught_out_of_range) {
      return EXIT_FAILURE;
    }

    caught_out_of_range = false;
    try {
      static_cast<void>(layout2d.coordinates(k2dOutOfRangeNode));
    } catch (const std::out_of_range&) {
      caught_out_of_range = true;
    }
    if (!caught_out_of_range) {
      return EXIT_FAILURE;
    }

    bool caught_invalid_argument = false;
    try {
      static_cast<void>(
          emdgrid::GridDataView<2, int>(
              layout2d, std::span<const int>(values).first(kExpectedSum)));
    } catch (const std::invalid_argument&) {
      caught_invalid_argument = true;
    }
    if (!caught_invalid_argument) {
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
  } catch (...) {
    return EXIT_FAILURE;
  }
}
