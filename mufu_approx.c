// gcc -O2 -lm -march=x86-64-v3 -o mufu_approx mufu_approx.c
#include "ptx_math_exp2f.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <limits.h>

typedef union {
    float f;
    uint32_t i;
} fi;

const float MIN_X = -126.0f;
const float MAX_X = 0x1.fffffep+6; // nextafterf(128.0f, -INFINITY)

int32_t ulp_diff(float a, float b) {
    fi a_fi = {.f = a};
    fi b_fi = {.f = b};
    return (int32_t)a_fi.i - (int32_t)b_fi.i;
}

int main() {
    uint32_t max_ulp_diff = 0;

    for (uint32_t ii = 0; ; ++ii) {
        if (ii % 100000000 == 0) {
            fprintf(stderr, "%u/%u\n", ii, UINT_MAX);
        }

        fi xx = {.i = ii};

        if (isfinite(xx.f) && xx.f >= MIN_X && xx.f <= MAX_X) {
            float int_part;
            float frac_part = modff(xx.f, &int_part);
            float red_res = scalbnf(ptxm_ex2_sm5x(frac_part), (int)int_part);

            float hw_res = ptxm_ex2_sm5x(xx.f);

            if (red_res != hw_res) {
                int32_t diff = ulp_diff(red_res, hw_res);
                printf(
                    "%g (%a): %d ULPs (%g (%a) vs. %g (%a))\n",
                    xx.f, xx.f, diff, red_res, red_res, hw_res, hw_res
                );
                uint32_t abs_diff = abs(diff);
                if (abs_diff > max_ulp_diff) {
                    max_ulp_diff = abs_diff;
                }
            }
        }

        if (ii == UINT_MAX) {
            break;
        }
    }

    printf("Max ULP diff: %u\n", max_ulp_diff);
}