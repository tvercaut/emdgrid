#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace emdgrid {

[[nodiscard]] constexpr int add(int lhs, int rhs) noexcept { return lhs + rhs; }

template <std::size_t Dim>
class GridLayout {
 public:
  static_assert(Dim > 0, "GridLayout requires at least one dimension.");

  using NodeId = std::size_t;
  using Shape = std::array<std::size_t, Dim>;
  using Coordinates = Shape;

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
    std::size_t index = 0;
    for (std::size_t axis = 0; axis < Dim; ++axis) {
      if (coord[axis] >= m_shape[axis]) {
        throw std::out_of_range("grid coordinate out of range");
      }
      index *= m_shape[axis];
      index += coord[axis];
    }
    return index;
  }

  [[nodiscard]] Coordinates coordinates(NodeId node) const {
    if (node >= node_count()) {
      throw std::out_of_range("grid node out of range");
    }

    Coordinates coord{};
    for (std::size_t axis = Dim; axis-- > 0;) {
      coord[axis] = node % m_shape[axis];
      node /= m_shape[axis];
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
      if (coord[axis] + 1 < m_shape[axis]) {
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
  static_assert(Dim > 0, "GridDataView requires at least one dimension.");

  using value_type = Scalar;
  using Layout = GridLayout<Dim>;
  using Coordinates = typename Layout::Coordinates;

  GridDataView(const Layout& layout, std::span<const Scalar> data)
      : m_layout(&layout), m_data(data) {
    if (m_data.size() != layout.node_count()) {
      throw std::invalid_argument("grid data size does not match layout");
    }
  }

  [[nodiscard]] const Layout& layout() const noexcept { return *m_layout; }

  [[nodiscard]] const Scalar& operator()(Coordinates coord) const {
    return m_data[m_layout->node(coord)];
  }

  template <class... Indices>
    requires(sizeof...(Indices) == Dim &&
             (std::convertible_to<Indices, std::size_t> && ...))
  [[nodiscard]] const Scalar& operator()(Indices... indices) const {
    return (*this)(Coordinates{static_cast<std::size_t>(indices)...});
  }

  [[nodiscard]] std::span<const Scalar> data() const noexcept { return m_data; }

 private:
  const Layout* m_layout;
  std::span<const Scalar> m_data;
};

[[nodiscard]] std::string_view version() noexcept;

}  // namespace emdgrid
