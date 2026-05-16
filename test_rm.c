// clang -O2 -lm -march=x86-64-v3 -o test_rm test_rm.c
#include <fenv.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#pragma STDC FENV_ACCESS ON

const float B = 252.0f;
const float C = 12582913.0f;

float f_rn(float x) {
    return fmaf(x, B, C);
}

float f_rm(float x) {
    fesetround(FE_DOWNWARD);
    const float res = fmaf(x, B, C);
    fesetround(FE_TONEAREST);
    return res;
}

int main() {
    for (uint32_t ii = 0; ; ++ii) {
        float x;
        memcpy(&x, &ii, sizeof(float));

        float res_rn = f_rn(x);
        float res_rm = f_rm(x);

        if (!isnan(res_rn) && !isnan(res_rm) && res_rn != res_rm) {
            printf("%g (%a): %g (%a) vs. %g (%a)\n", x, x, res_rn, res_rn, res_rm, res_rm);
        }

        if (ii % 100000000 == 0) {
            printf("Progress: %u\n", ii);
        }

        if (ii == UINT_MAX) {
            break;
        }
    }
}