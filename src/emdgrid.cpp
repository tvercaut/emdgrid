#include "emdgrid/emdgrid.hpp"
#include "emdgrid/version.hpp"

namespace emdgrid {

int add(int lhs, int rhs) noexcept { return lhs + rhs; }

std::string_view version() noexcept { return detail::version; }

}  // namespace emdgrid
