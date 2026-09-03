#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace emdgrid {

template <std::size_t Dim>
requires(Dim > 0)
class GridLayout {
 public:
  using NodeId = std::ptrdiff_t;
  using Shape = std::array<std::size_t, Dim>;
  using Coordinates = std::array<std::ptrdiff_t, Dim>;

  explicit GridLayout(Shape shape) : m_shape(shape) {}

  [[nodiscard]] const Shape& shape() const noexcept { return m_shape; }

  [[nodiscard]] std::size_t node_count() const noexcept {
    std::size_t count = 1;
    for (const std::size_t extent : m_shape) {
      count *= extent;
    }
    return count;
  }

  [[nodiscard]] std::size_t edge_count() const noexcept {
    const std::size_t count = node_count();
    if (count == 0) {
      return 0;
    }

    std::size_t total = 0;
    for (const std::size_t extent : m_shape) {
      total += (count / extent) * (extent - 1);
    }
    return total;
  }

  [[nodiscard]] NodeId node(Coordinates coord) const {
    NodeId index = 0;
    for (std::size_t axis = 0; axis < Dim; ++axis) {
      const NodeId extent = static_cast<NodeId>(m_shape[axis]);
      if (coord[axis] < 0 || coord[axis] >= extent) {
        throw std::out_of_range("grid coordinate out of range");
      }
      index *= extent;
      index += coord[axis];
    }
    return index;
  }

  [[nodiscard]] Coordinates coordinates(NodeId node) const {
    if (node < 0 || static_cast<std::size_t>(node) >= node_count()) {
      throw std::out_of_range("grid node out of range");
    }

    Coordinates coord{};
    for (std::size_t axis = Dim; axis-- > 0;) {
      const NodeId extent = static_cast<NodeId>(m_shape[axis]);
      coord[axis] = node % extent;
      node /= extent;
    }
    return coord;
  }

  [[nodiscard]] std::vector<NodeId> neighbours(NodeId node) const {
    const Coordinates coord = coordinates(node);
    std::vector<NodeId> result;
    result.reserve(2 * Dim);
    for (std::size_t axis = 0; axis < Dim; ++axis) {
      if (coord[axis] > 0) {
        Coordinates previous = coord;
        --previous[axis];
        result.push_back(this->node(previous));
      }
      if (coord[axis] + 1 < static_cast<NodeId>(m_shape[axis])) {
        Coordinates next = coord;
        ++next[axis];
        result.push_back(this->node(next));
      }
    }
    return result;
  }

 private:
  Shape m_shape;
};

template <std::size_t Dim, class Scalar>
class GridDataView {
 public:
  using value_type = Scalar;
  using GridLayout = emdgrid::GridLayout<Dim>;
  using Coordinates = GridLayout::Coordinates;

  GridDataView(const GridLayout& layout, std::span<const Scalar> data)
      : m_layout(layout), m_data(data) {
    if (m_data.size() != layout.node_count()) {
      throw std::invalid_argument("grid data size does not match layout");
    }
  }
  GridDataView(GridLayout&&, std::span<const Scalar>) = delete;

  [[nodiscard]] const GridLayout& layout() const noexcept {
    return m_layout.get();
  }

  [[nodiscard]] const Scalar& operator()(Coordinates coord) const {
    return m_data[static_cast<std::size_t>(m_layout.get().node(coord))];
  }

  template <class... Indices>
    requires(sizeof...(Indices) == Dim &&
             (std::convertible_to<Indices, std::ptrdiff_t> && ...))
  [[nodiscard]] const Scalar& operator()(Indices... indices) const {
    return (*this)(Coordinates{static_cast<std::ptrdiff_t>(indices)...});
  }

  [[nodiscard]] std::span<const Scalar> data() const noexcept { return m_data; }

 private:
  const std::reference_wrapper<const GridLayout> m_layout;
  const std::span<const Scalar> m_data;
};

[[nodiscard]] std::string_view version() noexcept;

/// Sparse transport plan represented in Coordinate (COO) format.
struct SparseTransportPlan {
  std::vector<uint32_t> source;
  std::vector<uint32_t> target;
  std::vector<double> flow;
};

}  // namespace emdgrid
