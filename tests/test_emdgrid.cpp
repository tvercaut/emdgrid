#include <span>
#include <stdexcept>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>  // NOLINT(build/include_order)

#include "emdgrid/emdgrid.hpp"
#include "emdgrid/version.hpp"

TEST_CASE("library version matches generated version") {
  CHECK_EQ(emdgrid::version(), emdgrid::detail::version);
}

TEST_CASE("2D grid layout operations") {
  constexpr emdgrid::GridLayout<2>::NodeId t2d_last_node = 5;
  constexpr std::size_t t2d_node_count = 6;
  constexpr std::size_t t2d_edge_count = 7;

  const emdgrid::GridLayout<2> layout2d({2, 3});
  CHECK_EQ(layout2d.shape(), emdgrid::GridLayout<2>::Shape{2, 3});
  CHECK_EQ(layout2d.node_count(), t2d_node_count);
  CHECK_EQ(layout2d.edge_count(), t2d_edge_count);
  CHECK_EQ(layout2d.node({1, 2}), t2d_last_node);
  CHECK_EQ(layout2d.coordinates(4), emdgrid::GridLayout<2>::Coordinates{1, 1});

  const std::vector<emdgrid::GridLayout<2>::NodeId>
      expected_interior_neighbours = {1, 3, t2d_last_node};
  CHECK_EQ(layout2d.neighbours(layout2d.node({1, 1})),
           expected_interior_neighbours);

  const std::vector<emdgrid::GridLayout<2>::NodeId> expected_corner_neighbours =
      {3, 1};
  CHECK_EQ(layout2d.neighbours(layout2d.node({0, 0})),
           expected_corner_neighbours);
}

TEST_CASE("3D grid layout operations") {
  constexpr emdgrid::GridLayout<3>::NodeId t3d_last_node = 23;
  constexpr emdgrid::GridLayout<3>::NodeId t3d_node = 17;
  constexpr std::size_t t3d_node_count = 24;
  constexpr std::size_t t3d_edge_count = 46;

  const emdgrid::GridLayout<3> layout3d({2, 3, 4});
  CHECK_EQ(layout3d.node_count(), t3d_node_count);
  CHECK_EQ(layout3d.edge_count(), t3d_edge_count);
  CHECK_EQ(layout3d.node({1, 2, 3}), t3d_last_node);
  CHECK_EQ(layout3d.coordinates(t3d_node),
           emdgrid::GridLayout<3>::Coordinates{1, 1, 1});
}

TEST_CASE("grid data view exposes layout and values") {
  constexpr int t2d_last_value = 5;

  const emdgrid::GridLayout<2> layout2d({2, 3});
  const std::vector<int> values = {0, 1, 2, 3, 4, 5};
  const emdgrid::GridDataView<2, int> view(layout2d, std::span(values));

  CHECK_EQ(&view.layout(), &layout2d);
  CHECK_EQ(view.data().data(), values.data());
  CHECK_EQ(view.data().size(), values.size());
  CHECK_EQ(view(1, 2), t2d_last_value);
  CHECK_EQ(view(emdgrid::GridLayout<2>::Coordinates{0, 1}), 1);
}

TEST_CASE("layout throws for out of range access") {
  constexpr emdgrid::GridLayout<2>::NodeId t2d_out_of_range_node = 6;
  const emdgrid::GridLayout<2> layout2d({2, 3});

  CHECK_THROWS_AS(static_cast<void>(layout2d.node({-1, 0})),
                  std::out_of_range);
  CHECK_THROWS_AS(static_cast<void>(layout2d.node({2, 0})),
                  std::out_of_range);
  CHECK_THROWS_AS(static_cast<void>(layout2d.coordinates(-1)),
                  std::out_of_range);
  CHECK_THROWS_AS(
      static_cast<void>(layout2d.coordinates(t2d_out_of_range_node)),
      std::out_of_range);
}

TEST_CASE("grid data view validates data size") {
  constexpr int t2d_last_value = 5;
  const emdgrid::GridLayout<2> layout2d({2, 3});
  const std::vector<int> values = {0, 1, 2, 3, 4, 5};
  CHECK_THROWS_AS(
      static_cast<void>(emdgrid::GridDataView<2, int>(
          layout2d, std::span(values).first(t2d_last_value))),
      std::invalid_argument);
}
