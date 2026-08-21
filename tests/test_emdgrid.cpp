#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "emdgrid/emdgrid.hpp"
#include "emdgrid/version.hpp"

int main() {
  try {
    constexpr emdgrid::GridLayout<2>::NodeId t2d_last_node = 5;
    constexpr emdgrid::GridLayout<2>::NodeId t2d_out_of_range_node = 6;
    constexpr std::size_t t2d_node_count = 6;
    constexpr std::size_t t2d_edge_count = 7;
    constexpr int t2d_last_value = 5;
    constexpr emdgrid::GridLayout<3>::NodeId t3d_last_node = 23;
    constexpr emdgrid::GridLayout<3>::NodeId t3d_node = 17;
    constexpr std::size_t t3d_node_count = 24;
    constexpr std::size_t t3d_edge_count = 46;

    if (emdgrid::version() != emdgrid::detail::version) {
      return EXIT_FAILURE;
    }

    const emdgrid::GridLayout<2> layout2d({2, 3});
    if (layout2d.shape() != emdgrid::GridLayout<2>::Shape{2, 3}) {
      return EXIT_FAILURE;
    }

    if (layout2d.node_count() != t2d_node_count ||
        layout2d.edge_count() != t2d_edge_count) {
      return EXIT_FAILURE;
    }

    if (layout2d.node({1, 2}) != t2d_last_node) {
      return EXIT_FAILURE;
    }

    if (layout2d.coordinates(4) !=
        emdgrid::GridLayout<2>::Coordinates{1, 1}) {
      return EXIT_FAILURE;
    }

    const std::vector<emdgrid::GridLayout<2>::NodeId>
        expected_interior_neighbours = {1, 3, t2d_last_node};
    if (layout2d.neighbours(layout2d.node({1, 1})) !=
        expected_interior_neighbours) {
      return EXIT_FAILURE;
    }

    const std::vector<emdgrid::GridLayout<2>::NodeId>
        expected_corner_neighbours = {3, 1};
    if (layout2d.neighbours(layout2d.node({0, 0})) !=
        expected_corner_neighbours) {
      return EXIT_FAILURE;
    }

    const emdgrid::GridLayout<3> layout3d({2, 3, 4});
    if (layout3d.node_count() != t3d_node_count ||
        layout3d.edge_count() != t3d_edge_count) {
      return EXIT_FAILURE;
    }

    if (layout3d.node({1, 2, 3}) != t3d_last_node) {
      return EXIT_FAILURE;
    }

    if (layout3d.coordinates(t3d_node) !=
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

    if (view(1, 2) != t2d_last_value ||
        view(emdgrid::GridLayout<2>::Coordinates{0, 1}) != 1) {
      return EXIT_FAILURE;
    }

    bool caught_out_of_range = false;
    try {
      static_cast<void>(layout2d.node({-1, 0}));
    } catch (const std::out_of_range&) {
      caught_out_of_range = true;
    }
    if (!caught_out_of_range) {
      return EXIT_FAILURE;
    }

    caught_out_of_range = false;
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
      static_cast<void>(layout2d.coordinates(-1));
    } catch (const std::out_of_range&) {
      caught_out_of_range = true;
    }
    if (!caught_out_of_range) {
      return EXIT_FAILURE;
    }

    caught_out_of_range = false;
    try {
      static_cast<void>(layout2d.coordinates(t2d_out_of_range_node));
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
              layout2d, std::span<const int>(values).first(t2d_last_value)));
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
