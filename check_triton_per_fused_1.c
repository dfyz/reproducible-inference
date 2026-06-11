#include "ptx_math_exp2f.h"
#include "ptx_math_recip.h"
#include "ptx_math_rsqrt.h"

#include <err.h>
#include <fcntl.h>
#include <sys/mman.h>

#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROWS 32
#define COLS 128
// 1e-6
#define EPS 0x1.0c6f7ap-20f

typedef uint16_t bf16;

struct InOuts {
    bf16 norm_input[ROWS][COLS];
    bf16 silu_input[ROWS][COLS];
    bf16 norm_weight[COLS];
    float ref_out[ROWS][COLS];
};

union flint {
    float f;
    uint32_t i;
};

float to_float(bf16 x) {
    union flint res = {.i = ((uint32_t)x) << 16};
    return res.f;
}

struct InOuts* load(const char* file_name) {
    // We rely on the OS unmapping the file and closing the FD.
    int fd = open(file_name, O_RDONLY);
    if (fd < 0) {
        err(EXIT_FAILURE, "Failed to open %s", file_name);
    }

    void* res = mmap(NULL, sizeof(struct InOuts), PROT_READ, MAP_SHARED, fd, 0);
    if (res == MAP_FAILED) {
        err(EXIT_FAILURE, "Failed to mmap");
    }

    return (struct InOuts*)res;
}

// Identical to `ptx_expf()`, except that the final multiplication
// is fused with adding one.
float ptx_expf_plus_1(float x) {
    const float log2e = 0x1.715476p+0f; // 1.442695e+00
    float y = fmaf(x, 0x1.77313ap-8 /*5.724980e-03*/, 0.5f);
    y = fminf(1.0f, fmaxf(0.0f, y));
    fesetround(FE_DOWNWARD);
    y = fmaf(y, 252.0f, 12582913.0f);
    fesetround(FE_TONEAREST);

    float z = y - 12583039.0f;
    z = fmaf(x, log2e, -z);
    z = fmaf(x, 0x1.4ae0cp-26 /*1.925963e-08*/, z);

    unsigned y_int;
    memcpy(&y_int, &y, sizeof(unsigned));
    y_int <<= 23;
    memcpy(&y, &y_int, sizeof(float));

    return fmaf(y, ptxm_ex2_sm5x(z), 1.0f);
}


float silu(float x) {
    return x * ptxm_rcp_sm5x(ptx_expf_plus_1(-x));
}

int main(int argc, char** argv) {
    if (argc != 2) {
        errx(EXIT_FAILURE, "Provide a path to the file with inputs/outputs");
    }

    struct InOuts* in_outs = load(argv[1]);

    for (size_t rr = 0; rr < ROWS; ++rr) {
        float acc[COLS/2] = {};
        for (size_t cc = 0; cc < COLS; cc += 2) {
            float val1 = to_float(in_outs->norm_input[rr][cc + 0]);
            float val2 = to_float(in_outs->norm_input[rr][cc + 1]);
            acc[cc/2] = fmaf(val1, val1, val2 * val2);
        }
        for (size_t warp_start = 0; warp_start < COLS/2; warp_start += COLS/4) {
            for (size_t to_xor = 16; to_xor > 0; to_xor >>= 1) {
                for (size_t cc = 0; cc < to_xor; ++cc) {
                    acc[warp_start + cc] += acc[warp_start + (cc ^ to_xor)];
                }
            }
        }
        float sq_sum = acc[0] + acc[COLS/4];
        float rms = ptxm_rsqrt_sm5x(fmaf(sq_sum, 1.0f/COLS, EPS));

        for (size_t cc = 0; cc < COLS; ++cc) {
            float val = to_float(in_outs->norm_input[rr][cc]);
            float weight = to_float(in_outs->norm_weight[cc]);
            float silu_res = silu(to_float(in_outs->silu_input[rr][cc]));
            float ours = val * rms * weight * silu_res;
            float ref = in_outs->ref_out[rr][cc];

            if (ours != ref) {
                printf("(%zu, %zu): %1.8e (%a) vs. %1.8e (%a)\n", rr, cc, ours, ours, ref, ref);
            }
        }
    }
}