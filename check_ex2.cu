#include "ptx_math_exp2f.h"

#include <cstdlib>
#include <cstdio>
#include <cstdint>

constexpr size_t kElems = 1ULL << 32;
constexpr size_t kSms = 132;

__global__ void CheckEx2(float* out) {
    for (
        size_t xx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        xx < kElems;
        xx += (size_t)blockDim.x * gridDim.x
    ) {
        unsigned xx_int = xx;
        float xx_float;
        memcpy(&xx_float, &xx_int, sizeof(float));

        float xx_out;
        asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(xx_out) : "f"(xx_float));
        out[xx] = xx_out;
    }
}

bool are_same(float x, float y) {
    uint32_t xb, yb;
    memcpy(&xb, &x, sizeof xb);
    memcpy(&yb, &y, sizeof y);
    return xb == yb;
}

int main() {
    float* out;
    auto status = cudaMalloc(&out, kElems * sizeof(float));
    if (status != cudaSuccess) {
        fprintf(stderr, "failed to malloc: %d\n", (int)status);
        exit(1);
    }

    CheckEx2<<<kSms, 1024>>>(out);

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
        const auto ref_res = ptxm_ex2_sm5x(xx);
        if (!are_same(ref_res, host_out[ii])) {
            printf("%g (%a): ref=%a hardware=%a\n", xx, xx, ref_res, host_out[ii]);
            ++diff_count;
        }
    }

    printf("DIFF COUNT: %zu\n", diff_count);
}
