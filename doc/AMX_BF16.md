# AMX BF16 bridge on `tg/amx`

This branch starts from master `31d577ac735fa2b3b785140bf91e6324d1345988`.
The earlier Tahoma pin was 2,365 master commits behind and is not the base of
these changes.

## Reuse upstream kernels

Current master already provides AMX INT8 GEMM/iGEMM in core XNNPACK and AMX
BF16/FP16/INT8 dot kernels in YNNPACK. The core BF16-to-FP32 fully-connected
operator only selected AVX-512BF16. This branch adds the missing integration:

- Compile the unchanged upstream `ynnpack/kernels/dot/x86_amxbf16.cc` into the
  production microkernel archive. No YNNPACK subgraph or Slinky runtime is linked.
- Adapt the existing x32c2 fully-connected weight format to upstream 32x32 AMX
  dot kernels. The FP32 bias, clamping and byte strides preserve the operator
  contract. One unpaired K element is handled without allocation or overread.
- Keep a small internal 32x48 dot bridge for the F32 fully-connected operator's
  explicit BF16 and compensated BF16x3 compute policies. BF16x3 evaluates
  high/high and both high/residual products with FP32 accumulation; it is
  approximate, not bitwise FP32. Consumers never depend on its packed format.
- Detect AMX BF16 separately from AMX INT8 and request Linux XTILEDATA permission.
  Existing AVX-512 BF16 fallback remains available. Enable supported GCC 14+
  AMX builds, while preserving the older-compiler guard.

The upstream dot code manages tile state; each bridge invocation owns its state
so intervening operators cannot leave stale cached tile configurations. The
bridge uses the caller's scheduling; it does not create any worker threads.

## F32 operator compute policies

`xnn_create_fully_connected_nc_f32` accepts the mutually exclusive
`XNN_FLAG_F32_COMPUTE_BF16` and `XNN_FLAG_F32_COMPUTE_BF16X3` flags. Its normal
reshape/setup/run/delete lifecycle owns conversion, weight packing, ISA dispatch,
tile scheduling, reusable execution buffers and FP32 fallback. Input/output and
source checkpoint weights remain FP32. Non-finite/BF16-overflowing inputs or
weights fall back to strict F32; bias and clamp follow accumulation. Reduced
operators need no readable input tail, even on fallback. Normal F32 API padding
requirements are unchanged.

The native operator shares its strict-F32 and reduced packed weights through
the normal weights cache. Operator instances remain nonconcurrent: applications
can use one instance per worker sharing a finalized cache. The non-AMX fallback
keeps its rounded F32 child weights private so temporary conversions never become
external-cache identity keys. `XNN_FLAG_SLOW_CONSISTENT_ARITHMETIC` forces this
portable fallback for regression comparisons. No new runtime dependency or
thread pool is introduced. The public flags do not imply AMX on unsupported CPUs.

The extension is currently enabled by CMake (`XNN_ENABLE_F32_REDUCED=1`). Other
build manifests retain upstream operator behavior and explicitly reject the new
F32 flags until they integrate the new operator and upstream dot sources. Only
the Linux CPU CMake products have been validated; this is not a Bazel/GN or
Windows validation claim.

## Focused validation

`XNNPACK_BUILD_AMX_TESTS=ON` enables `xnnpack_amx_bf16_test` without enabling the
entire GoogleTest/microkernel suite. It exercises fully-connected row/column/K
tails, nontrivial input/output strides, bias and clamping at 1 and 4 threads.
Unsupported BF16 hardware is reported rather than claiming an AMX pass.
The test rejects NaNs, asserts native dispatch when OS/build capability permits
it, and returns CTest's explicit skip code on unsupported hosts.

The same option enables `xnnpack_f32_reduced_test`. It tests both compute modes,
native and forced fallback, both weight layouts, 1/4/16 threads, odd 193-column
panel boundaries, output guards, zero/growing/shrinking batches, shared-cache
deduplication and non-finite strict fallback against an independent double oracle.

The Tahoma integration retains small FP32/BF16/BF16x3 adapter checks,
broadcasting, per-worker operator/weight sharing, attention masks and odd query
blocks, and output lifetime. Kernel shape sweeps live here instead of being
duplicated in that adapter. The full three-page DocMT benchmark remains the quality gate
for any reduced-precision policy.

The arithmetic sources remain the upstream BSD-licensed implementation. The
new code is dispatch, adapter, build integration and regression tests, not a
second implementation of the upstream AMX tile loops.
