# Local compatibility patch

This is the published `mlx-sys` 0.7.0 crate, from mlx-rs commit
`01595fc4e05648175e4b52a5c209955ecdfb7638`, under its original MIT or Apache-2.0
license. The included mlx-c sources retain their MIT license.

Two changes are made to the mlx-c CMake dependency declaration:

1. Pin its existing MLX 0.32.2 dependency to that tag's exact commit,
   `1f8e74e3f12f31365464a6867c6579f0e9b29d85`.
2. Apply `src/mlx-c/carter-metal-compat.cmake` after fetching that source.

MLX 0.32.2 uses `metal::remove_addrspace_t` unconditionally in the integral
constant binary-operation macro. That trait is unavailable in the older Metal
compiler used with the app's supported macOS 14 deployment target. The upstream
change was introduced for the new Metal 4.1 address-space rules:

- https://github.com/ml-explore/mlx/pull/3963
- https://github.com/ml-explore/mlx-swift/issues/497

The patch uses the upstream trait for `__METAL_VERSION__ >= 410` and the original
`decltype(res)` type otherwise. It changes no mathematical operations, constants,
layouts, or kernel dispatch. The exact upstream header SHA-256 is checked before
patching; a repeated application is harmless. Native-accelerator kernels remain
subject to upstream deployment/SDK checks. This preserves the API and native MLX
version while retaining compatibility with the declared minimum OS.

The build script also watches and hashes the compatibility script alongside its
CMake input so native build/cache inputs include this patch.

Registry download bookkeeping and its nested Cargo.lock were omitted. Upstream
license texts are included as LICENSE-MIT, LICENSE-APACHE, and src/mlx-c/LICENSE.
Remove this patch when the upstream release supplies the same compatibility fix.
