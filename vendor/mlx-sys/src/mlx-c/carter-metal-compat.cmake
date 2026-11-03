# Minimal compatibility fix for upstream MLX 0.32.2, which uses a Metal 4.1-only
# type trait even when targeting older Metal/macOS versions. See CARTER-PATCH.md.
if(NOT DEFINED MLX_SOURCE_DIR)
  message(FATAL_ERROR "MLX_SOURCE_DIR must identify the pinned MLX source tree")
endif()
set(header "${MLX_SOURCE_DIR}/mlx/backend/metal/kernels/steel/utils/integral_constant.h")
file(READ "${header}" source)
string(FIND "${source}" "using carter_integral_result_t" already_patched)
if(NOT already_patched EQUAL -1)
  return()
endif()
# Fail loudly if the pinned upstream source changes instead of silently modifying
# an unrelated declaration or applying only part of the patch.
file(SHA256 "${header}" source_sha)
if(NOT source_sha STREQUAL "886eccaac4f8f00eab5a10551fdb5315df7ab7462aba13d69bb95765c1a4735e")
  message(FATAL_ERROR "Unexpected MLX integral_constant.h; review the Metal compatibility patch")
endif()
set(compat [=[// Carter compatibility: preserve the pre-Metal-4.1 type on older compilers.
#if __METAL_VERSION__ >= 410
template <typename T>
using carter_integral_result_t = metal::remove_addrspace_t<T>;
#else
template <typename T>
using carter_integral_result_t = T;
#endif

]=])
string(REPLACE "#define integral_const_binop(" "${compat}#define integral_const_binop(" source "${source}")
string(REPLACE "using res_t = metal::remove_addrspace_t<decltype(res)>;" "using res_t = carter_integral_result_t<decltype(res)>;" source "${source}")
file(WRITE "${header}" "${source}")
message(STATUS "Applied MLX integral-constant compatibility for Metal before 4.1")
