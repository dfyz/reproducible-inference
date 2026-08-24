#include <climits>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <vector>

constexpr float H = 6656.0f;

// # Need to patch
// # @!P1 BRA 0x880 /* 0x0000000400289947 */
// # to
// # @P1  BRA 0x880 /* 0x0000000400281947 */
//
// from pathlib import Path
// import struct

// src_bs = struct.pack('<Q', 0x0000000400289947)
// dst_bs = struct.pack('<Q', 0x0000000400281947)
// binary = Path('check_fchk')
// data = bytearray(binary.read_bytes())
// start = data.find(src_bs)
// assert start >= 0
// data[start:start+len(src_bs)] = dst_bs
// binary.write_bytes(data)
__global__ void check_fchk_kernel(const float* ins, float* outs, size_t n_vals) {
    for (
        size_t ii = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        ii < n_vals;
        ii += (size_t)blockDim.x * gridDim.x
    ) {
        outs[ii] = ins[ii] / H;
    }
}

int main() {
    size_t n_vals = 0;
    std::vector<float> vals;
    vals.reserve(UINT_MAX);
    for (uint64_t ii = 0; ii <= UINT_MAX; ++ii) {
        float x;
        memcpy(&x, &ii, sizeof(float));

        if (isfinite(x)) {
            ++n_vals;
            vals.push_back(x);
        }
    }
    printf("got %zu vals\n", n_vals);

    float* device_vals;
    float* device_outs;
    if (cudaMalloc(&device_vals, n_vals * sizeof(vals[0])) != cudaSuccess) {
        fprintf(stderr, "failed to malloc vals\n");
        exit(1);
    }
    if (cudaMemcpy(device_vals, vals.data() , n_vals * sizeof(vals[0]), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "failed to copy vals\n");
        exit(1);
    }
    if (cudaMalloc(&device_outs, n_vals * sizeof(vals[0])) != cudaSuccess) {
        fprintf(stderr, "failed to malloc outs\n");
        exit(1);
    }

    printf("running the kernel\n");

    check_fchk_kernel<<<132, 1024>>>(device_vals, device_outs, n_vals);
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "something went wrong when running the kernel");
        exit(1);
    }

    std::vector<float> outs(n_vals);
    printf("checking the results\n");
    if (cudaMemcpy(outs.data(), device_outs, n_vals * sizeof(vals[0]), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "failed to copy outs\n");
        exit(1);
    }

    for (size_t ii = 0; ii < n_vals; ++ii) {
        const float ref = vals[ii] / H;
        if (ref != outs[ii]) {
            fprintf(stderr, "mismatch at %.13a: %.13a vs. %.13a\n", vals[ii], ref, outs[ii]);
        }
    }
}