#pragma once

#include <string_view>

namespace emdgrid {

[[nodiscard]] constexpr int add(int lhs, int rhs) noexcept { return lhs + rhs; }

[[nodiscard]] std::string_view version() noexcept;

}  // namespace emdgrid
