// nvcc -DMODE=... -std=c++20 -Xcompiler=-mbmi2 -arch=sm_90a -O3 -o check_exp check_exp.cu
#include "ptx_math_exp2f.h"

#include <cstdlib>
#include <cstdio>
#include <cstdint>

enum class Mode {
    Exp2,
    FastExp,
    Exp,
};

#ifndef MODE
#error "Use -DMODE=..."
#endif
constexpr Mode kMode = (Mode)MODE;

constexpr size_t kElems = 1ULL << 32;
constexpr size_t kSms = 132;

template <auto>
inline constexpr bool AlwaysFalse = false;

template <Mode mode>
__global__ void Check(float* out) {
    for (
        size_t xx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        xx < kElems;
        xx += (size_t)blockDim.x * gridDim.x
    ) {
        unsigned xx_int = xx;
        float xx_float;
        memcpy(&xx_float, &xx_int, sizeof(float));

        float xx_out;
        if constexpr (mode == Mode::Exp2) {
            asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(xx_out) : "f"(xx_float));
        } else if constexpr (mode == Mode::FastExp) {
            xx_out = __expf(xx_float);
        } else if constexpr (mode == Mode::Exp) {
            xx_out = expf(xx_float);
        } else {
            static_assert(AlwaysFalse<mode>);
        }
        out[xx] = xx_out;
    }
}

bool AreSame(float x, float y) {
    uint32_t xb, yb;
    memcpy(&xb, &x, sizeof xb);
    memcpy(&yb, &y, sizeof y);
    return xb == yb;
}

template <Mode mode>
float GetCpuRes(float x) {
    constexpr float log2e = 0x1.715476p+0f;

    if constexpr (mode == Mode::Exp2) {
        return ptxm_ex2_sm5x(x);
    } else if constexpr (mode == Mode::FastExp) {
        x *= log2e;
        const bool subnormal_res = x < -126.0f;
        if (subnormal_res) {
            x *= 0.5f;
        }
        float res = ptxm_ex2_sm5x(x);
        if (subnormal_res) {
            res *= res;
        }
        return res;
    } else if constexpr (mode == Mode::Exp) {
        // TODO: implement
    } else {
        static_assert(AlwaysFalse<mode>);
    }
}

int main() {
    printf("Checking %d\n", (int)kMode);

    float* out;
    auto status = cudaMalloc(&out, kElems * sizeof(float));
    if (status != cudaSuccess) {
        fprintf(stderr, "failed to malloc: %d\n", (int)status);
        exit(1);
    }

    Check<kMode><<<kSms, 1024>>>(out);

    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        fprintf(stderr, "failed to launch: %d\n", (int)status);
        exit(1);
    }

    float* host_out = (float*)malloc(kElems * sizeof(float));
    status = cudaMemcpy(host_out, out, kElems * sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        fprintf(stderr, "failed to memcpy: %d\n", (int)status);
        exit(1);
    }

    size_t diff_count = 0;
    for (size_t ii = 0; ii < kElems; ++ii) {
        if (ii % 1000000 == 0) {
            fprintf(stderr, "%zu/%zu\n", ii, kElems);
        }

        float xx;
        memcpy(&xx, &ii, sizeof(float));
        const auto cpu_res = GetCpuRes<kMode>(xx);
        if (!AreSame(cpu_res, host_out[ii])) {
            printf("%g (%a): cpu=%a gpu=%a\n", xx, xx, cpu_res, host_out[ii]);
            ++diff_count;
        }
    }

    printf("DIFF COUNT: %zu\n", diff_count);
}
