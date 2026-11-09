#include "../muse_glimmer/utils/load_st.h"
#include "../muse_glimmer/utils/rcp.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint64_t ulp_diff(float x, float y) {
    int64_t x_int, y_int;
    memcpy(&x_int, &x, sizeof(x));
    memcpy(&y_int, &y, sizeof(y));
    return llabs(x_int - y_int);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        errx(1, "Usage: PATH_TO_TPU_FILE");
    }

    float* tpu = load_st(argv[1], NULL);
    uint64_t mismatches = 0;

    uint64_t max_gpu_diff = 0;
    uint64_t max_tpu_diff = 0;

    uint64_t gpu_inexact = 0;
    uint64_t tpu_inexact = 0;

    for (size_t ii = 0; ii <= UINT_MAX; ++ii) {
        if (ii % 100'000'000 == 0) {
            fprintf(stderr, "%zu/%u (%lu mismatches)\n", ii, UINT_MAX + 1, mismatches);
            fprintf(stderr, "max GPU diff = %lu ULPs, max TPU diff = %lu ULPs\n", max_gpu_diff, max_tpu_diff);
            fprintf(stderr, "GPU inexact = %lu, TPU inexact = %lu\n", gpu_inexact, tpu_inexact);
        }

        float x;
        memcpy(&x, &ii, sizeof(float));

        if (isnan(x)) {
            continue;
        }

        float gpu_res = rcp(x);
        float tpu_res = tpu[ii];
        float ref_res = 1.0f/x;

        if (fpclassify(x) != FP_SUBNORMAL && fpclassify(ref_res) != FP_SUBNORMAL) {
            if (ref_res != gpu_res) {
                uint64_t diff = ulp_diff(ref_res, gpu_res);
                if (diff > max_gpu_diff) {
                    max_gpu_diff = diff;
                }
                ++gpu_inexact;
            }

            if (ref_res != tpu_res) {
                uint64_t diff = ulp_diff(ref_res, tpu_res);
                if (diff > max_tpu_diff) {
                    max_tpu_diff = diff;
                }
                ++tpu_inexact;
            }
        }

        if (tpu_res != gpu_res) {
            ++mismatches;
            if (mismatches % 1'000'000 == 0) {
                fprintf(
                    stderr,
                    "mismatch at %a (%1.8e): %a (%1.8e) (GPU) vs. %a (%1.8e) (TPU) vs. %a (%1.8e) (IEEE)\n",
                    x, x,
                    gpu_res, gpu_res,
                    tpu_res, tpu_res,
                    ref_res, ref_res
                );
            }
        }
    }

    fprintf(stderr, "%lu mismatches in total\n", mismatches);
}
