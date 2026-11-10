// clang -O2 -std=c23 -fopenmp -march=native -lm -o check_mfu check_mfu.c
#include "utils/ex.h"
#include "utils/rcp.h"
#include "utils/rsq.h"

#include "../ptx_math_exp2f.h"
#include "../ptx_math_recip.h"
#include "../ptx_math_rsqrt.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

union fp32_int {
    float    f;
    uint32_t i;
};

typedef float (*mfu_func)(float);

float ptxm_rcp_sm5x_no_denorms(float x) {
    auto x_class = fpclassify(x);
    if (x_class == FP_SUBNORMAL || x_class == FP_ZERO) {
        return copysign(INFINITY, x);
    }
    float res = ptxm_rcp_sm5x(x);
    if (fpclassify(res) == FP_SUBNORMAL) {
        return copysign(0.0f, res);
    }
    return res;
}

float ptxm_rsqrt_sm5x_no_denorms(float x) {
    if (x < -FLT_MIN) {
        return NAN;
    }
    if (x < FLT_MIN) {
        return copysign(INFINITY, x);
    }
    if (x == INFINITY) {
        return 0.0f;
    }
    float res = ptxm_rsqrt_sm5x(x);
    if (fpclassify(res) == FP_SUBNORMAL) {
        return copysign(0.0f, res);
    }
    return res;
}

bool are_equal(float x, float y) {
    if (isnan(x) || isnan(y)) {
        return isnan(x) && isnan(y);
    }
    return x == y;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "specify the function to test: ex2|rcp\n");
        exit(2);
    }

    mfu_func ref_func, our_func;
    const char* func = argv[1];
    if (strcmp(func, "ex2") == 0) {
        ref_func = ptxm_ex2_sm5x;
        our_func = ex2;
    } else if (strcmp(func, "rcp") == 0) {
        ref_func = ptxm_rcp_sm5x_no_denorms;
        our_func = rcp;
    } else if (strcmp(func, "rsq") == 0) {
        ref_func = ptxm_rsqrt_sm5x_no_denorms;
        our_func = rsq;
    } else {
        fprintf(stderr, "unknown function %s\n", func);
        exit(3);
    }

    #pragma omp parallel for
    for (uint64_t ii = 0; ii <= UINT_MAX; ++ii) {
        if (ii % 100'000'000 == 0) {
            fprintf(stderr, "%lu/%u\n", ii, UINT_MAX + 1);
        }

        union fp32_int tmp = {.i = ii};

        if (!isnan(tmp.f)) {
            float ref = ref_func(tmp.f);
            float our = our_func(tmp.f);

            if (!are_equal(ref, our)) {
                #pragma omp critical
                {
                    printf("x   = %.13a\n", tmp.f);
                    printf("ref = %.13a\n", ref);
                    printf("our = %.13a\n", our);
                }
                exit(1);
            }
        }
    }

    puts("ALL GOOD");
}
