// clang -O2 -std=c23 -fopenmp -march=native -lm -o check_ex2 check_ex2.c
#include "utils/ex.h"
#include "../ptx_math_exp2f.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

union fp32_int {
    float    f;
    uint32_t i;
};

int main() {
    #pragma omp parallel for
    for (uint64_t ii = 0; ii <= UINT_MAX; ++ii) {
        if (ii % 100'000'000 == 0) {
            fprintf(stderr, "%lu/%u\n", ii, UINT_MAX + 1);
        }

        union fp32_int tmp = {.i = ii};

        if (!isnan(tmp.f)) {
            float ref = ptxm_ex2_sm5x(tmp.f);
            float our = ex2(tmp.f);

            if (ref != our) {
                printf("x   = %.13a\n", tmp.f);
                printf("ref = %.13a\n", ref);
                printf("our = %.13a\n", our);
                exit(1);
            }
        }
    }

    puts("ALL GOOD");
}
