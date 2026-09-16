// Copyright 2026 XNNPACK contributors
// This source code is licensed under the BSD-style license found in LICENSE.
#include <xnnpack.h>
#include <pthreadpool.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "src/xnnpack/config.h"
#include "src/xnnpack/hardware-config.h"

static void check(xnn_status s) { if(s!=xnn_status_success) throw std::runtime_error("XNNPACK operator failed"); }
static uint16_t bf16(float x) { uint32_t b;std::memcpy(&b,&x,4);return uint16_t((b+0x7fff+((b>>16)&1))>>16); }
static float f32(uint16_t x) { uint32_t b=uint32_t(x)<<16;float f;std::memcpy(&f,&b,4);return f; }

int main() {
  try {
    check(xnn_initialize(nullptr));
    const auto* cfg=xnn_init_bf16_f32_gemm_config();
#if XNN_ARCH_X86_64
    if ((xnn_init_hardware_config()->arch_flags & xnn_arch_x86_amx_bf16) &&
        (!cfg || cfg->arch != xnn_arch_x86_amx_bf16)) {
      throw std::runtime_error("AMX-capable build/host did not select AMX BF16");
    }
#endif
    if(!cfg) {std::cout<<"BF16 unsupported on this host\n";return 77;}
    for(size_t threads : {size_t{1},size_t{4}}) {
      auto pool=std::unique_ptr<pthreadpool,decltype(&pthreadpool_destroy)>(pthreadpool_create(threads),pthreadpool_destroy);
      for(size_t m : {size_t{1},size_t{7},size_t{16},size_t{33}})
      for(size_t k : {size_t{1},size_t{3},size_t{32},size_t{33},size_t{65}})
      for(size_t n : {size_t{1},size_t{17},size_t{31},size_t{32},size_t{65}}) {
        const size_t stride_a=k+3,stride_c=n+7;
        std::vector<uint16_t>a(m*stride_a+XNN_EXTRA_BYTES/2),w(n*k+XNN_EXTRA_BYTES/2);
        std::vector<float>bias(n+XNN_EXTRA_BYTES/4),out(m*stride_c,12345.f);
        for(size_t i=0;i<a.size();++i)a[i]=bf16(std::sin(float(i)*.31f));
        for(size_t i=0;i<w.size();++i)w[i]=bf16(std::cos(float(i)*.13f)*.2f);
        for(size_t i=0;i<n;++i)bias[i]=float(i%5)*.1f;
        xnn_operator_t raw{};
        check(xnn_create_fully_connected_nc_bf16_f32(k,n,stride_a,stride_c,w.data(),bias.data(),-2,2,0,nullptr,&raw));
        auto op=std::unique_ptr<xnn_operator,decltype(&xnn_delete_operator)>(raw,xnn_delete_operator);
        check(xnn_reshape_fully_connected_nc_bf16_f32(op.get(),m,pool.get()));
        check(xnn_setup_fully_connected_nc_bf16_f32(op.get(),a.data(),out.data()));
        check(xnn_run_operator(op.get(),pool.get()));
        for(size_t row=0;row<m;++row) {
          for(size_t col=0;col<n;++col) {
            double ref=bias[col];
            for(size_t j=0;j<k;++j)ref+=double(f32(a[row*stride_a+j]))*f32(w[col*k+j]);
            ref=std::clamp(ref,-2.,2.);
            if(!std::isfinite(out[row*stride_c+col]) || std::abs(out[row*stride_c+col]-ref)>1e-5)
              throw std::runtime_error("BF16 row/column/reduction tail mismatch");
          }
          for(size_t col=n;col<stride_c;++col) if(out[row*stride_c+col]!=12345.f)throw std::runtime_error("output stride guard overwritten");
        }
      }
    }
    std::cout<<"BF16 operator/tail tests passed; selected arch="<<uint64_t(cfg->arch)<<'\n';
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
