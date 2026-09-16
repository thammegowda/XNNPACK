// Copyright 2026 XNNPACK contributors
// This source code is licensed under the BSD-style license found in LICENSE.
#ifndef XNNPACK_SRC_XNNPACK_FULLY_CONNECTED_REDUCED_H_
#define XNNPACK_SRC_XNNPACK_FULLY_CONNECTED_REDUCED_H_

#include "include/xnnpack.h"
#include "src/xnnpack/common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define XNN_F32_REDUCED_FLAGS (XNN_FLAG_F32_COMPUTE_BF16 | XNN_FLAG_F32_COMPUTE_BF16X3)

XNN_INTERNAL enum xnn_status xnn_create_fully_connected_nc_f32_reduced(
    size_t input_channels, size_t output_channels, size_t input_stride,
    size_t output_stride, const float* kernel, const float* bias,
    float output_min, float output_max, uint32_t flags,
    xnn_weights_cache_t weights_cache, xnn_operator_t* op_out);
XNN_INTERNAL enum xnn_status xnn_reshape_fully_connected_nc_f32_reduced(
    xnn_operator_t op, size_t batch_size, pthreadpool_t threadpool);
XNN_INTERNAL enum xnn_status xnn_setup_fully_connected_nc_f32_reduced(
    xnn_operator_t op, const float* input, float* output);
XNN_INTERNAL enum xnn_status xnn_run_fully_connected_nc_f32_reduced(
    xnn_operator_t op, pthreadpool_t threadpool);
XNN_INTERNAL void xnn_destroy_fully_connected_nc_f32_reduced(xnn_operator_t op);
XNN_INTERNAL bool xnn_fully_connected_nc_f32_uses_amx(xnn_operator_t op);

#ifdef __cplusplus
}
#endif
#endif
