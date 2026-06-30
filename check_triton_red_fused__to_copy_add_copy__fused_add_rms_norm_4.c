#pragma STDC FP_CONTRACT OFF

#include "ptx_math_recip.h"
#include "ptx_math_rsqrt.h"

#include <err.h>
#include <fcntl.h>
#include <sys/mman.h>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define ROWS 8192
#define COLS 2048
#define EPS 0x1.0c6f7ap-20f
#define ELEMS_PER_THREAD 8
#define WARP_THREADS 32

typedef uint16_t bf16;

struct InOuts {
    bf16 a[ROWS][COLS];
    bf16 b[ROWS][COLS];
    bf16 c[ROWS][COLS];
    bf16 d[ROWS][COLS];
    bf16 norm_weight[COLS];
    bf16 ref_out[ROWS][COLS];
};

union flint {
    float f;
    uint32_t i;
};

float to_float(bf16 x) {
    union flint res = {.i = ((uint32_t)x) << 16};
    return res.f;
}

float to_bf16(float x) {
    union flint res = {.f = x};
    unsigned bias = (res.i >> 16) & 1;
    res.i += 0x7FFFU + bias;
    res.i &= 0xFFFF0000U;
    return res.f;
}

float get_sum(struct InOuts* in_outs, size_t rr, size_t cc) {
    return (to_float(in_outs->a[rr][cc]) + to_float(in_outs->b[rr][cc])) +
           (to_float(in_outs->c[rr][cc]) + to_float(in_outs->d[rr][cc]));
}

int main(int argc, char** argv) {
    if (argc != 2) {
        errx(EXIT_FAILURE, "Provide a path to the file with inputs/outputs");
    }

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) {
        err(EXIT_FAILURE, "Failed to open %s", argv[1]);
    }

    struct InOuts* in_outs = mmap(NULL, sizeof(struct InOuts), PROT_READ, MAP_SHARED, fd, 0);
    if (in_outs == MAP_FAILED) {
        err(EXIT_FAILURE, "Failed to mmap");
    }

    for (size_t rr = 0; rr < ROWS; ++rr) {
        float thread_acc[COLS/ELEMS_PER_THREAD/2] = {0};
        for (size_t cc = COLS/2; cc < COLS; cc += ELEMS_PER_THREAD) {
            float local_sq_sum = 0.0f;
            for (size_t off = 0; off < ELEMS_PER_THREAD; ++off) {
                float prev_val = get_sum(in_outs, rr, cc - COLS/2 + off);
                float val =      get_sum(in_outs, rr, cc - 0      + off);
                local_sq_sum += fmaf(prev_val, prev_val, val * val);
            }
            thread_acc[(cc - COLS/2)/ELEMS_PER_THREAD] = local_sq_sum;
        }

        float warp_acc[COLS/ELEMS_PER_THREAD/2/WARP_THREADS] = {0};
        for (size_t start = 0; start < COLS/ELEMS_PER_THREAD/2; start += WARP_THREADS) {
            for (size_t to_xor = WARP_THREADS/2; to_xor > 0; to_xor >>= 1) {
                for (size_t cc = 0; cc < to_xor; ++cc) {
                    thread_acc[start + cc] += thread_acc[start + (cc ^ to_xor)];
                }
            }
            warp_acc[start/WARP_THREADS] = thread_acc[start];
        }

        for (size_t to_xor = 2; to_xor > 0; to_xor >>= 1) {
            for (size_t cc = 0; cc < to_xor; ++cc) {
                warp_acc[cc] += warp_acc[cc ^ to_xor];
            }
        }

        float rms = ptxm_rsqrt_sm5x(fmaf(warp_acc[0], 1.0f/COLS, EPS));

        for (size_t cc = 0; cc < COLS; ++cc) {
            float val = get_sum(in_outs, rr, cc);
            float weight = to_float(in_outs->norm_weight[cc]) + 1.0f;

            float ours = to_bf16(val * rms * weight);
            float ref = to_float(in_outs->ref_out[rr][cc]);

            if (ours != ref) {
                printf("(%zu, %zu): %1.8e (%a) vs. %1.8e (%a)\n", rr, cc, ours, ours, ref, ref);
            }
        }
    }
}
