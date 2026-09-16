// Copyright 2026 XNNPACK contributors
// This source code is licensed under the BSD-style license found in LICENSE.

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "src/operators/fingerprint_id.h"
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/cache.h"
#include "src/xnnpack/fully-connected-reduced.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/operator-utils.h"
#include "src/xnnpack/operator.h"

#if XNN_ARCH_X86_64 && XNN_ENABLE_AVX512BF16 && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

// The panel layout is private to this operator and the upstream YNNPACK dot.
enum { panel_columns = 192, tile_rows = 32 };
typedef bool (*split_fn)(const float*, uint16_t*, uint16_t*, size_t);

struct xnn_f32_reduced_context {
  size_t k, n, padded_k, panels, plane_size, bias_bytes;
  bool correction, native, finite_weights, has_bias;
  float output_min, output_max;
  split_fn split;
  xnn_operator_t strict, high, low;
  uint16_t* split_input;
  float* fallback;
  uint8_t* finite_rows;
  size_t split_capacity, fallback_capacity, row_capacity;
  const float* input;
  float* output;
};

static uint16_t bf16(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return (uint16_t)((bits + UINT32_C(0x7fff) + ((bits >> 16) & 1)) >> 16);
}

static float f32(uint16_t value) {
  const uint32_t bits = (uint32_t)value << 16;
  float result;
  memcpy(&result, &bits, sizeof(result));
  return result;
}

static bool split_scalar(const float* input, uint16_t* high, uint16_t* low, size_t count) {
  bool finite = true;
  for (size_t i = 0; i < count; ++i) {
    finite = finite && isfinite(input[i]) && fabsf(input[i]) <= 0x1.fep127f;
    high[i] = bf16(input[i]);
    if (low) low[i] = bf16(input[i] - f32(high[i]));
  }
  return finite;
}

#if XNN_ARCH_X86_64 && XNN_ENABLE_AVX512BF16 && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx512f,avx512bw,avx512bf16")))
static bool split_avx512(const float* input, uint16_t* high, uint16_t* low, size_t count) {
  size_t i = 0;
  bool finite = true;
  for (; i + 16 <= count; i += 16) {
    const __m512 x = _mm512_loadu_ps(input + i);
    finite = finite && _mm512_cmp_ps_mask(_mm512_abs_ps(x),
        _mm512_set1_ps(0x1.fep127f), _CMP_LE_OQ) == 0xffff;
    const __m256bh h = _mm512_cvtneps_pbh(x);
    _mm256_storeu_si256((__m256i*)(high + i), (__m256i)h);
    if (low) _mm256_storeu_si256((__m256i*)(low + i),
        (__m256i)_mm512_cvtneps_pbh(_mm512_sub_ps(x, _mm512_cvtpbh_ps(h))));
  }
  return split_scalar(input + i, high + i, low ? low + i : NULL, count - i) && finite;
}
#endif

static bool reserve(void** pointer, size_t* capacity, size_t bytes) {
  if (bytes <= *capacity) return true;
  void* next = xnn_allocate_zero_simd_memory(bytes);
  if (!next) return false;
  xnn_release_simd_memory(*pointer);
  *pointer = next;
  *capacity = bytes;
  return true;
}

enum xnn_status xnn_create_fully_connected_nc_f32_reduced(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* op_out) {
  if (!(xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK)) return xnn_status_uninitialized;
  if (!op_out) return xnn_status_invalid_parameter;
  *op_out = NULL;
  size_t elements, bytes, bias_bytes, packed_bytes, plane_size;
  const bool correction = (flags & XNN_FLAG_F32_COMPUTE_BF16X3) != 0;
  if (!kernel || !input_channels || !output_channels || input_stride < input_channels ||
      output_stride < output_channels || isnan(output_min) || isnan(output_max) ||
      output_min >= output_max || (flags & XNN_F32_REDUCED_FLAGS) == XNN_F32_REDUCED_FLAGS ||
      input_channels > SIZE_MAX - 31 || output_channels > SIZE_MAX - (panel_columns - 1) ||
      !xnn_safe_mul(input_channels, output_channels, &elements) ||
      !xnn_safe_mul(elements, sizeof(float), &bytes) ||
      !xnn_safe_add(bytes, XNN_EXTRA_BYTES, &bytes) ||
      !xnn_safe_mul(output_channels, sizeof(float), &bias_bytes) ||
      !xnn_safe_add(bias_bytes, XNN_ALLOCATION_ALIGNMENT - 1, &bias_bytes)) {
    return xnn_status_invalid_parameter;
  }
  bias_bytes &= ~(size_t)(XNN_ALLOCATION_ALIGNMENT - 1);
  const size_t padded_k = round_up_po2(input_channels, 32);
  const size_t panels = divide_round_up(output_channels, panel_columns);
  if (!xnn_safe_mul(panels, panel_columns, &plane_size) ||
      !xnn_safe_mul(plane_size, padded_k, &plane_size) ||
      !xnn_safe_mul(plane_size, sizeof(uint16_t) * (correction ? 2 : 1), &packed_bytes) ||
      !xnn_safe_add(packed_bytes, bias_bytes, &packed_bytes)) {
    return xnn_status_invalid_parameter;
  }
  xnn_operator_t op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (!op) return xnn_status_out_of_memory;
  op->type = xnn_operator_type_fully_connected_nc_f32;
  op->flags = flags;
  op->input_pixel_stride = input_stride;
  op->output_pixel_stride = output_stride;
  op->weights_cache = weights_cache;
  op->f32_reduced = xnn_allocate_zero_memory(sizeof(struct xnn_f32_reduced_context));
  enum xnn_status status = xnn_status_out_of_memory;
  float* weight = NULL;
  uint16_t* parts = NULL;
  if (!op->f32_reduced) goto error;
  struct xnn_f32_reduced_context* c = op->f32_reduced;
  c->k = input_channels;
  c->n = output_channels;
  c->padded_k = padded_k;
  c->panels = panels;
  c->plane_size = plane_size;
  c->bias_bytes = bias_bytes;
  c->correction = correction;
  c->finite_weights = true;
  c->has_bias = bias != NULL;
  c->output_min = output_min;
  c->output_max = output_max;
  c->split = split_scalar;
  const struct xnn_hardware_config* hardware = xnn_init_hardware_config();
#if XNN_ARCH_X86_64 && XNN_ENABLE_AVX512AMX
  c->native = hardware && (hardware->arch_flags & xnn_arch_x86_amx_bf16) &&
      !(flags & XNN_FLAG_SLOW_CONSISTENT_ARITHMETIC);
#endif
#if XNN_ARCH_X86_64 && XNN_ENABLE_AVX512BF16 && (defined(__GNUC__) || defined(__clang__))
  if (hardware && (hardware->arch_flags & xnn_arch_x86_avx512bf16)) c->split = split_avx512;
#endif
  (void)hardware;
  status = xnn_create_fully_connected_nc_f32(input_channels, output_channels,
      input_stride, output_stride, kernel, NULL, -INFINITY, INFINITY,
      flags & (XNN_FLAG_TRANSPOSE_WEIGHTS | XNN_FLAG_DONT_SPIN_WORKERS), weights_cache, &c->strict);
  if (status != xnn_status_success) goto error;
  status = xnn_status_out_of_memory;
  weight = xnn_allocate_zero_simd_memory(bytes);
  size_t part_count, part_bytes;
  if (!xnn_safe_mul(output_channels, padded_k, &part_count) ||
      !xnn_safe_mul(part_count, sizeof(uint16_t) * (correction ? 2 : 1), &part_bytes)) {
    status = xnn_status_invalid_parameter;
    goto error;
  }
  parts = xnn_allocate_zero_simd_memory(part_bytes);
  if (!weight || !parts) goto error;
  if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    for (size_t n = 0; n < output_channels; ++n) {
      for (size_t k = 0; k < input_channels; ++k) weight[n * input_channels + k] = kernel[k * output_channels + n];
    }
  } else memcpy(weight, kernel, elements * sizeof(float));
  for (size_t n = 0; n < output_channels; ++n) {
    const bool finite = c->split(weight + n * input_channels, parts + n * padded_k,
        correction ? parts + part_count + n * padded_k : NULL, input_channels);
    c->finite_weights = c->finite_weights && finite;
  }
  if (!c->native && c->finite_weights) {
    // Child conversions are private snapshots, not aliases of caller weights.
    // Never key an external cache by the address of this temporary conversion.
    for (size_t n = 0; n < output_channels; ++n) {
      for (size_t k = 0; k < input_channels; ++k) weight[n * input_channels + k] = f32(parts[n * padded_k + k]);
    }
    status = xnn_create_fully_connected_nc_f32(input_channels, output_channels,
        input_channels, output_stride, weight, NULL, -INFINITY, INFINITY, 0, NULL, &c->high);
    if (status != xnn_status_success) goto error;
    if (correction) {
      for (size_t n = 0; n < output_channels; ++n) {
        for (size_t k = 0; k < input_channels; ++k) weight[n * input_channels + k] = f32(parts[part_count + n * padded_k + k]);
      }
      status = xnn_create_fully_connected_nc_f32(input_channels, output_channels,
          input_channels, output_stride, weight, NULL, -INFINITY, INFINITY, 0, NULL, &c->low);
      if (status != xnn_status_success) goto error;
    }
  }
  if (!c->native || !c->finite_weights) packed_bytes = bias_bytes;
  const struct xnn_weights_cache_look_up_key key = {
      .seed = (uint32_t)(input_channels ^ output_channels ^ flags ^ (c->native ? 0x80000000 : 0)),
      .kernel = kernel, .bias = bias,
      .fingerprint_id = XNN_FINGERPRINT_ID_VALUE(fully_connected_nc, f32, f32, bf16),
  };
  size_t offset = weights_cache ? xnn_weights_cache_look_up(weights_cache, &key) : XNN_CACHE_NOT_FOUND;
  if (offset == XNN_CACHE_NOT_FOUND) {
    void* packed = xnn_get_pointer_to_write_weights(op, packed_bytes);
    if (!packed) { status = xnn_status_out_of_memory; goto error; }
    memset(packed, 0, packed_bytes);
    if (bias) memcpy(packed, bias, output_channels * sizeof(float));
    if (c->native && c->finite_weights) {
      uint16_t* high = (uint16_t*)((uint8_t*)packed + bias_bytes);
      for (size_t n = 0; n < output_channels; ++n) {
        for (size_t k = 0; k < padded_k; ++k) {
          const size_t index = n / panel_columns * padded_k * panel_columns +
              k / 2 * (2 * panel_columns) + n % panel_columns * 2 + k % 2;
          high[index] = parts[n * padded_k + k];
          if (correction) high[plane_size + index] = parts[part_count + n * padded_k + k];
        }
      }
    }
    if (weights_cache) offset = xnn_look_up_or_insert_weights_cache(weights_cache, &key, packed, packed_bytes);
  }
  if (weights_cache) {
    if (offset == XNN_CACHE_NOT_FOUND) { status = xnn_status_out_of_memory; goto error; }
    op->packed_weights.offset = offset;
  }
  xnn_release_simd_memory(weight);
  xnn_release_simd_memory(parts);
  *op_out = op;
  return xnn_status_success;
error:
  xnn_release_simd_memory(weight);
  xnn_release_simd_memory(parts);
  xnn_delete_operator(op);
  return status;
}

enum xnn_status xnn_reshape_fully_connected_nc_f32_reduced(
    xnn_operator_t op, size_t batch_size, pthreadpool_t threadpool) {
  op->state = xnn_run_state_invalid;
  if (op->weights_cache && !op->weights_cache->is_finalized(op->weights_cache->context)) return xnn_status_invalid_state;
  op->batch_size = batch_size;
  if (!batch_size) { op->state = xnn_run_state_skip; return xnn_status_success; }
  struct xnn_f32_reduced_context* c = op->f32_reduced;
  size_t count, split_bytes, input_elements, output_elements, fallback_bytes;
  if (!xnn_safe_mul(batch_size, c->padded_k, &count) ||
      !xnn_safe_mul(count, sizeof(uint16_t) * (c->correction ? 2 : 1), &split_bytes) ||
      !xnn_safe_mul(batch_size, op->input_pixel_stride, &input_elements) ||
      !xnn_safe_mul(input_elements, sizeof(float), &input_elements) ||
      !xnn_safe_add(input_elements, XNN_EXTRA_BYTES, &input_elements) ||
      !xnn_safe_mul(batch_size, op->output_pixel_stride, &output_elements) ||
      !xnn_safe_mul(output_elements, sizeof(float), &fallback_bytes)) return xnn_status_invalid_parameter;
  if (c->finite_weights &&
      (!reserve((void**)&c->split_input, &c->split_capacity, split_bytes) ||
       !reserve((void**)&c->finite_rows, &c->row_capacity, batch_size))) return xnn_status_out_of_memory;
  if (!c->native && c->finite_weights) {
    if (!xnn_safe_mul(batch_size, c->k, &input_elements) ||
        !xnn_safe_add(input_elements, XNN_EXTRA_BYTES / sizeof(float), &input_elements) ||
        !xnn_safe_mul(input_elements, 2, &count) ||
        !xnn_safe_add(count, output_elements, &count) ||
        !xnn_safe_mul(count, sizeof(float), &fallback_bytes)) return xnn_status_invalid_parameter;
    if (!reserve((void**)&c->fallback, &c->fallback_capacity, fallback_bytes)) return xnn_status_out_of_memory;
  }
  const xnn_operator_t children[] = {c->strict, c->high, c->low};
  for (size_t i = 0; i < 3; ++i) {
    if (!children[i]) continue;
    const enum xnn_status status = xnn_reshape_fully_connected_nc_f32(children[i], batch_size, threadpool);
    if (status != xnn_status_success) return status;
  }
  op->state = xnn_run_state_needs_setup;
  return xnn_status_success;
}

enum xnn_status xnn_setup_fully_connected_nc_f32_reduced(
    xnn_operator_t op, const float* input, float* output) {
  if (op->state == xnn_run_state_skip) return xnn_status_success;
  if (op->state == xnn_run_state_invalid) return xnn_status_invalid_state;
  if (!input || !output) return xnn_status_invalid_parameter;
  op->f32_reduced->input = input;
  op->f32_reduced->output = output;
  op->state = xnn_run_state_ready;
  return xnn_status_success;
}

static void convert_row(void* context, size_t row) {
  xnn_operator_t op = context;
  struct xnn_f32_reduced_context* c = op->f32_reduced;
  uint16_t* high = c->split_input + row * c->padded_k;
  uint16_t* low = c->correction ? high + op->batch_size * c->padded_k : NULL;
  c->finite_rows[row] = c->split(c->input + row * op->input_pixel_stride, high, low, c->k);
  memset(high + c->k, 0, (c->padded_k - c->k) * sizeof(uint16_t));
  if (low) memset(low + c->k, 0, (c->padded_k - c->k) * sizeof(uint16_t));
}

#if XNN_ARCH_X86_64 && XNN_ENABLE_AVX512AMX
static void compute_tile(void* context, size_t tile) {
  xnn_operator_t op = context;
  const struct xnn_f32_reduced_context* c = op->f32_reduced;
  const size_t row = tile / c->panels * tile_rows, col = tile % c->panels * panel_columns;
  const uint16_t* a = c->split_input + row * c->padded_k;
  const uint16_t* b = (const uint16_t*)((const uint8_t*)packed_weights(op) + c->bias_bytes) + col * c->padded_k;
  xnn_bf16_f32_gemm_32x48__amx(min(tile_rows, op->batch_size - row), min(panel_columns, c->n - col), c->padded_k,
      a, c->correction ? a + op->batch_size * c->padded_k : NULL, c->padded_k * sizeof(uint16_t),
      b, c->correction ? b + c->plane_size : NULL, panel_columns * sizeof(uint16_t),
      c->output + row * op->output_pixel_stride + col, op->output_pixel_stride * sizeof(float));
}
#endif

static void epilogue_row(void* context, size_t row) {
  xnn_operator_t op = context;
  const struct xnn_f32_reduced_context* c = op->f32_reduced;
  const float* bias = packed_weights(op);
  float* output = c->output + row * op->output_pixel_stride;
  for (size_t n = 0; n < c->n; ++n) {
    float value = output[n];
    if (c->has_bias) value += bias[n];
    if (value < c->output_min) value = c->output_min;
    if (value > c->output_max) value = c->output_max;
    output[n] = value;
  }
}

static enum xnn_status run_child(xnn_operator_t op, const float* input, float* output, pthreadpool_t pool) {
  enum xnn_status status = xnn_setup_fully_connected_nc_f32(op, input, output);
  return status == xnn_status_success ? xnn_run_operator(op, pool) : status;
}

enum xnn_status xnn_run_fully_connected_nc_f32_reduced(xnn_operator_t op, pthreadpool_t threadpool) {
  struct xnn_f32_reduced_context* c = op->f32_reduced;
  const uint32_t flags = op->flags & XNN_FLAG_DONT_SPIN_WORKERS ? PTHREADPOOL_FLAG_YIELD_WORKERS : 0;
  bool finite = c->finite_weights;
  if (finite) {
    pthreadpool_parallelize_1d(threadpool, convert_row, op, op->batch_size, flags);
    for (size_t row = 0; row < op->batch_size; ++row) finite = finite && c->finite_rows[row];
  }
  enum xnn_status status = xnn_status_success;
  if (!finite) {
    // Only exceptional inputs need the ordinary F32 kernel's readable tail.
    const size_t bytes = op->batch_size * op->input_pixel_stride * sizeof(float) + XNN_EXTRA_BYTES;
    float* staged = xnn_allocate_zero_simd_memory(bytes);
    if (!staged) return xnn_status_out_of_memory;
    for (size_t row = 0; row < op->batch_size; ++row) {
      memcpy(staged + row * op->input_pixel_stride, c->input + row * op->input_pixel_stride, c->k * sizeof(float));
    }
    status = run_child(c->strict, staged, c->output, threadpool);
    xnn_release_simd_memory(staged);
  } else if (c->native) {
#if XNN_ARCH_X86_64 && XNN_ENABLE_AVX512AMX
    pthreadpool_parallelize_1d(threadpool, compute_tile, op,
        divide_round_up(op->batch_size, tile_rows) * c->panels, flags);
#endif
  } else {
    const size_t input_count = op->batch_size * c->k + XNN_EXTRA_BYTES / sizeof(float);
    float* high = c->fallback;
    float* low = high + input_count;
    float* temporary = low + input_count;
    for (size_t row = 0; row < op->batch_size; ++row) {
      for (size_t k = 0; k < c->k; ++k) {
        high[row * c->k + k] = f32(c->split_input[row * c->padded_k + k]);
        if (c->correction) low[row * c->k + k] = f32(c->split_input[(row + op->batch_size) * c->padded_k + k]);
      }
    }
    status = run_child(c->high, c->correction ? low : high, c->output, threadpool);
    if (status != xnn_status_success) return status;
    if (c->correction) {
      const xnn_operator_t products[] = {c->low, c->high};
      for (size_t product = 0; product < 2; ++product) {
        status = run_child(products[product], high, temporary, threadpool);
        if (status != xnn_status_success) return status;
        for (size_t row = 0; row < op->batch_size; ++row) {
          for (size_t n = 0; n < c->n; ++n) c->output[row * op->output_pixel_stride + n] += temporary[row * op->output_pixel_stride + n];
        }
      }
    }
  }
  if (status != xnn_status_success) return status;
  if (c->has_bias || c->output_min != -INFINITY || c->output_max != INFINITY) {
    pthreadpool_parallelize_1d(threadpool, epilogue_row, op, op->batch_size, flags);
  }
  return xnn_status_success;
}

void xnn_destroy_fully_connected_nc_f32_reduced(xnn_operator_t op) {
  struct xnn_f32_reduced_context* c = op->f32_reduced;
  if (c->strict) xnn_delete_operator(c->strict);
  if (c->high) xnn_delete_operator(c->high);
  if (c->low) xnn_delete_operator(c->low);
  xnn_release_simd_memory(c->split_input);
  xnn_release_simd_memory(c->fallback);
  xnn_release_simd_memory(c->finite_rows);
  xnn_release_memory(c);
  op->f32_reduced = NULL;
}

bool xnn_fully_connected_nc_f32_uses_amx(xnn_operator_t op) {
  return op && op->f32_reduced && op->f32_reduced->native;
}
