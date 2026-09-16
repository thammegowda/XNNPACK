// Copyright 2026 XNNPACK contributors
// This source code is licensed under the BSD-style license found in LICENSE.
#include <xnnpack.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "src/xnnpack/cache.h"
#include "src/xnnpack/fully-connected-reduced.h"
#include "src/xnnpack/hardware-config.h"

namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void check(xnn_status status) { require(status == xnn_status_success, "F32 compute operator failed"); }
using Op = std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>;

void matrix(size_t m, size_t k, size_t n, uint32_t flags, bool cached, pthreadpool_t pool) {
  const size_t input_stride = k + 3, output_stride = n + 7;
  std::vector<float> input((m + 1) * input_stride + XNN_EXTRA_BYTES / 4);
  std::vector<float> weights(k * n + XNN_EXTRA_BYTES / 4), bias(n + XNN_EXTRA_BYTES / 4);
  for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(float(i) * .13f) * .7f;
  for (size_t i = 0; i < weights.size(); ++i) weights[i] = std::cos(float(i) * .07f) * .3f;
  for (size_t i = 0; i < n; ++i) bias[i] = float(i) * .01f;
  const auto original_weights = weights, original_input = input;
  xnn_weights_cache_t raw_cache = nullptr;
  if (cached) check(xnn_create_weights_cache(&raw_cache));
  std::unique_ptr<xnn_weights_cache_provider, decltype(&xnn_delete_weights_cache)> cache(raw_cache, xnn_delete_weights_cache);
  const auto create = [&] {
    xnn_operator_t raw{};
    check(xnn_create_fully_connected_nc_f32(k, n, input_stride, output_stride,
        weights.data(), bias.data(), -.75f, 1.25f, flags, cache.get(), &raw));
    return Op(raw, xnn_delete_operator);
  };
  auto first = create();
  const auto cache_bytes = [&] { return static_cast<xnn_internal_weights_cache*>(raw_cache->context)->cache.weights.size; };
  const auto size = cached ? cache_bytes() : 0;
  auto second = create();
  if (cached) {
    require(cache_bytes() == size, "identical operators duplicated immutable packed weights");
    check(xnn_finalize_weights_cache(cache.get(), xnn_weights_cache_finalization_kind_soft));
  }
#if XNN_ARCH_X86_64
  const bool expected_amx = (xnn_init_hardware_config()->arch_flags & xnn_arch_x86_amx_bf16) &&
      !(flags & XNN_FLAG_SLOW_CONSISTENT_ARITHMETIC);
  require(xnn_fully_connected_nc_f32_uses_amx(first.get()) == expected_amx, "F32 compute dispatch differs");
#endif
  for (auto* op : {first.get(), second.get()}) {
    require(xnn_run_operator(op, pool) == xnn_status_invalid_state, "unshaped operator ran");
    for (size_t rows : {size_t{0}, size_t{1}, m, m + 1, m}) {
      check(xnn_reshape_fully_connected_nc_f32(op, rows, pool));
      std::vector<float> output((rows + 1) * output_stride, 12345.0f);
      check(xnn_setup_fully_connected_nc_f32(op, rows ? input.data() : nullptr, rows ? output.data() : nullptr));
      check(xnn_run_operator(op, pool));
      const double tolerance = flags & XNN_FLAG_F32_COMPUTE_BF16 ? .015 : .00012;
      for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < n; ++col) {
          double expected = bias[col];
          for (size_t inner = 0; inner < k; ++inner) {
            expected += double(input[row * input_stride + inner]) *
                weights[flags & XNN_FLAG_TRANSPOSE_WEIGHTS ? inner * n + col : col * k + inner];
          }
          expected = std::clamp(expected, -.75, 1.25);
          const auto value = output[row * output_stride + col];
          require(std::isfinite(value) && std::abs(value - expected) <= tolerance, "double-reference mismatch");
        }
        for (size_t col = n; col < output_stride; ++col) require(output[row * output_stride + col] == 12345.0f, "row guard overwritten");
      }
      for (size_t i = rows * output_stride; i < output.size(); ++i) require(output[i] == 12345.0f, "output guard overwritten");
    }
  }
  require(input == original_input && weights == original_weights, "caller storage changed");
}

void nonfinite_fallback(uint32_t flags) {
  for (float special : {INFINITY, -INFINITY, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::max()}) {
    for (bool weight_special : {false, true}) {
      std::vector<float> a(1 + XNN_EXTRA_BYTES / 4, 0), w(a.size(), 0);
      a[0] = weight_special ? 2 : special;
      w[0] = weight_special ? special : 2;
      float values[2]{};
      for (size_t i = 0; i < 2; ++i) {
        xnn_operator_t raw{};
        check(xnn_create_fully_connected_nc_f32(1, 1, 1, 1, w.data(), nullptr,
            -INFINITY, INFINITY, i ? flags : 0, nullptr, &raw));
        Op op(raw, xnn_delete_operator);
        check(xnn_reshape_fully_connected_nc_f32(op.get(), 1, nullptr));
        check(xnn_setup_fully_connected_nc_f32(op.get(), a.data(), values + i));
        check(xnn_run_operator(op.get(), nullptr));
      }
      require(std::isnan(values[0]) ? std::isnan(values[1]) : values[0] == values[1], "nonfinite fallback differs from strict F32");
    }
  }
}
}  // namespace

int main() {
  try {
    check(xnn_initialize(nullptr));
    for (size_t threads : {size_t{1}, size_t{4}, size_t{16}}) {
      auto pool = std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)>(pthreadpool_create(threads), pthreadpool_destroy);
      require(bool(pool), "thread pool allocation failed");
      for (uint32_t mode : {XNN_FLAG_F32_COMPUTE_BF16, XNN_FLAG_F32_COMPUTE_BF16X3}) {
        for (uint32_t fallback : {0, XNN_FLAG_SLOW_CONSISTENT_ARITHMETIC}) {
          for (uint32_t transpose : {0, XNN_FLAG_TRANSPOSE_WEIGHTS}) {
            for (bool cached : {false, true}) {
              for (auto shape : {std::array<size_t, 3>{5, 17, 9}, {32, 64, 48}, {128, 128, 64}, {33, 65, 193}}) {
                matrix(shape[0], shape[1], shape[2], mode | fallback | transpose, cached, pool.get());
              }
            }
          }
          nonfinite_fallback(mode | fallback);
        }
      }
    }
    std::cout << "F32 BF16/BF16x3 dispatch, fallback, stride, cache and reshape regressions passed\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
