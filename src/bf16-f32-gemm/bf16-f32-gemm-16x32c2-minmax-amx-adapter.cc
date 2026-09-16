// Copyright 2026 XNNPACK contributors
// This source code is licensed under the BSD-style license found in LICENSE.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "src/xnnpack/gemm.h"
#include "ynnpack/kernels/dot/dot.h"

extern "C" void xnn_bf16_f32_gemm_32x48__amx(
    size_t m, size_t n, size_t k, const uint16_t* a_high,
    const uint16_t* a_low, size_t a_stride, const uint16_t* b_high,
    const uint16_t* b_low, size_t b_stride_k, float* c, size_t c_stride) {
  assert((a_low == nullptr) == (b_low == nullptr));
  ynn::dot_kernel_state state;
  const auto run = [&](const uint16_t* a, const uint16_t* b, const float* previous) {
    ynn::dot_bf16_bf16_fp32_32x48x32_16x16x2_amxbf16(
        m, n, 1, 1, k, a_stride, 0, 0, a, 0, 0, b_stride_k, b,
        c_stride, previous, c_stride, c, &state);
  };
  if (a_low) {
    run(a_low, b_high, nullptr);
    run(a_high, b_low, c);
    run(a_high, b_high, c);
  } else {
    run(a_high, b_high, nullptr);
  }
}

// Bridge the existing XNNPACK x32c2 pack to the upstream YNNPACK dot kernel.
// Only the dot source is linked: no Slinky scheduler or YNNPACK graph runtime.
extern "C" void xnn_bf16_f32_gemm_minmax_ukernel_16x32c2__avx512amx(
    size_t mr, size_t nc, size_t kc, const uint16_t* a, size_t a_stride,
    const void* w, float* c, size_t cm_stride, size_t cn_stride,
    const xnn_f32_minmax_params* params) {
  assert(mr > 0 && mr <= 16 && nc > 0 && kc > 0 && kc % 2 == 0);
  const size_t k = kc / sizeof(uint16_t), padded_k = (k + 1) & ~size_t{1};
  ynn::dot_kernel_state state;
  while (nc) {
    const size_t n = std::min(nc, size_t{32});
    const auto* bias = static_cast<const float*>(w);
    const auto* weights = reinterpret_cast<const uint16_t*>(bias + 32);
    if (k >= 2) {
      ynn::dot_bf16_bf16_fp32_32x32x32_16x16x2_amxbf16(
          mr, n, 1, 1, k & ~size_t{1}, a_stride, 0, 0, a, 0, 0, 32*sizeof(uint16_t),
          weights, 0, bias, cm_stride, c, &state);
    }
    for (size_t row = 0; row < mr; ++row) {
      auto* out = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(c) + row*cm_stride);
      const auto* input = reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(a) + row*a_stride);
      for (size_t col = 0; col < n; ++col) {
        if (k == 1) out[col] = bias[col];
        // Handle one unpaired K element without allocation or an input overread.
        if (k & 1) out[col] = std::fma(math_cvt_fp32_bf16(input[k-1]),
            math_cvt_fp32_bf16(weights[(k/2)*64+col*2]), out[col]);
        out[col] = std::min(std::max(out[col], params->scalar.min), params->scalar.max);
      }
    }
    w = weights + padded_k * 32;
    c = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(c) + cn_stride);
    nc -= n;
  }
}
